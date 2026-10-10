"""
sticker_generator.py - turns ONE image into a pack of 100+ distinct stickers.

Pipeline per sticker
    extract subject (rembg -> GrabCut -> full frame fallbacks)
      -> [optional] AI img2img restyle (Replicate)  OR  local PIL/OpenCV style
      -> geometry (tilt / squash / shake / flip)
      -> compose: crop, auto-pad, white die-cut border, drop shadow, overlays/text
      -> encode 512x512 transparent WEBP < 500 KB
      -> auto_tester.heal_index() validates and retries with escalating fixes

Escalation ladder used by StickerFactory.render(spec, attempt):
    0  normal (AI if enabled, else local)
    1  retry (AI backoff already applied inside the backend)
    2  re-run background removal with a LOWER alpha threshold, local styling
    3  re-run with a HIGHER threshold + GrabCut, local styling
    4  last resort: minimal sticker (plain subject + border + label)

CLI:  python sticker_generator.py photo.png --out output/run1
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

import auto_tester as at

log = logging.getLogger("stickers")

SIZE = 512
MARGIN = 36          # keeps border + shadow inside the canvas
WORK_MAX = 420       # subject is downscaled to this before styling (speed)
MAX_PASSES = 3
LAST_RESORT = 4


# =========================================================================== #
# 1. Subject extraction
# =========================================================================== #
_REMBG_SESSION = None


def _rembg_mask(rgb: Image.Image) -> Image.Image:
    global _REMBG_SESSION
    from rembg import new_session, remove          # optional heavy dependency
    if _REMBG_SESSION is None:
        _REMBG_SESSION = new_session("u2net")
    return remove(rgb, session=_REMBG_SESSION, only_mask=True)


def _grabcut_mask(rgb: Image.Image) -> Image.Image:
    arr = np.array(rgb.convert("RGB"))
    h, w = arr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (int(w * .05), int(h * .05), int(w * .9), int(h * .9))
    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    cv2.grabCut(arr, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    return Image.fromarray(np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8))


def _clean_mask(mask: Image.Image, threshold: int) -> np.ndarray:
    """Binarise, remove specks/holes, keep main components, soften the edge."""
    a = np.array(mask.convert("L"))
    b = (a >= threshold).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    b = cv2.morphologyEx(cv2.morphologyEx(b, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = [i + 1 for i, ar in enumerate(areas) if ar >= 0.1 * areas.max()]
        b = np.where(np.isin(labels, keep), 255, 0).astype(np.uint8)
    return cv2.GaussianBlur(b, (0, 0), 1.2)


def extract_subject(img: Image.Image, threshold: int = 128, method: str = "auto") -> Image.Image:
    """Return an RGBA image whose alpha is the foreground mask (<=1024px)."""
    img = ImageOps.exif_transpose(img).copy()
    img.thumbnail((1024, 1024))
    rgba = img.convert("RGBA")
    alpha = np.array(rgba.getchannel("A"))

    mask: Optional[Image.Image] = None
    if method == "auto" and alpha.min() < 200 and 0.02 < (alpha > 128).mean() < 0.98:
        mask = Image.fromarray(alpha)                       # already a cut-out
    else:
        if method != "grabcut":
            try:
                mask = _rembg_mask(rgba.convert("RGB"))
            except Exception as exc:                        # rembg missing / model download failed
                log.warning("rembg unavailable (%s) -> GrabCut fallback", exc)
        if mask is None:
            mask = _grabcut_mask(rgba.convert("RGB"))

    a = _clean_mask(mask, threshold)
    if (a > 128).mean() < 0.01:                             # nothing found: try GrabCut
        a = _clean_mask(_grabcut_mask(rgba.convert("RGB")), threshold)
    if (a > 128).mean() < 0.01:                             # still nothing: keep whole frame, rounded
        m = Image.new("L", rgba.size, 0)
        ImageDraw.Draw(m).rounded_rectangle((4, 4, rgba.width - 4, rgba.height - 4), 40, fill=255)
        a = np.array(m)
    rgba.putalpha(Image.fromarray(a))
    return rgba


def _crop_alpha(img: Image.Image, pad: int = 0) -> Image.Image:
    box = img.getchannel("A").point(lambda v: 255 if v > 16 else 0).getbbox()
    if not box:
        raise ValueError("empty subject")
    l, t, r, b = box
    return img.crop((max(0, l - pad), max(0, t - pad), min(img.width, r + pad), min(img.height, b + pad)))


def prep_subject(img: Image.Image) -> Image.Image:
    img = _crop_alpha(img)
    img.thumbnail((WORK_MAX, WORK_MAX), Image.LANCZOS)
    return img


# =========================================================================== #
# 2. Local style effects (all take/return uint8 RGB ndarrays)
# =========================================================================== #
def _sat(rgb, k):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * k, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def _hue(rgb, deg):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hsv[..., 0] = (hsv[..., 0].astype(np.int16) + int(deg / 2)) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def _tint(rgb, color, a):
    return np.clip(rgb * (1 - a) + np.array(color, np.float32) * a, 0, 255).astype(np.uint8)


def _grad_tint(rgb, c1, c2, a):
    w = rgb.shape[1]
    t = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
    col = np.array(c1, np.float32) * (1 - t) + np.array(c2, np.float32) * t
    return np.clip(rgb * (1 - a) + col * a, 0, 255).astype(np.uint8)


def _poster(rgb, levels):
    step = 256 // levels
    return ((rgb.astype(np.int16) // step) * step + step // 2).clip(0, 255).astype(np.uint8)


def _duotone(rgb, dark, light):
    g = (cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255)[..., None]
    return np.clip(np.array(dark, np.float32) * (1 - g) + np.array(light, np.float32) * g, 0, 255).astype(np.uint8)


def _lines(rgb, block=9, c=7):
    gray = cv2.medianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 5)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, c)


def _scan(rgb):
    out = rgb.copy()
    out[::4] = (out[::4] * 0.7).astype(np.uint8)
    return out


def _duo(dark, light):
    return lambda rgb: _duotone(rgb, dark, light)


def _st_pixel(rgb):
    h, w = rgb.shape[:2]
    small = cv2.resize(rgb, (max(8, w // 9), max(8, h // 9)), interpolation=cv2.INTER_AREA)
    return cv2.resize(_poster(small, 6), (w, h), interpolation=cv2.INTER_NEAREST)


def _st_mosaic(rgb):
    h, w = rgb.shape[:2]
    small = cv2.resize(rgb, (max(6, w // 18), max(6, h // 18)), interpolation=cv2.INTER_AREA)
    return cv2.resize(_poster(_sat(small, 1.3), 8), (w, h), interpolation=cv2.INTER_NEAREST)


def _st_line_art(rgb):
    return np.repeat(_lines(rgb)[..., None], 3, axis=2)


def _st_sketch(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(255 - gray, (0, 0), 10)
    den = np.clip(256 - blur.astype(np.int16), 1, 255).astype(np.uint8)
    pencil = cv2.divide(gray, den, scale=256)
    return np.repeat(pencil[..., None], 3, axis=2)


def _st_comic(rgb):
    smooth = cv2.bilateralFilter(rgb, 9, 60, 60)
    return (_sat(_poster(smooth, 4), 1.5) * (_lines(rgb, 9, 8)[..., None] / 255.0)).astype(np.uint8)


def _st_doodle(rgb):
    return (_sat(_poster(cv2.bilateralFilter(rgb, 9, 70, 70), 4), 1.4) * (_lines(rgb, 11, 6)[..., None] / 255.0)).astype(np.uint8)


def _st_anime(rgb):
    s = cv2.bilateralFilter(cv2.bilateralFilter(rgb, 9, 50, 50), 9, 50, 50)
    return (_sat(_poster(s, 8), 1.5) * (_lines(rgb, 9, 9)[..., None] / 255.0 * 0.6 + 0.4)).astype(np.uint8)


def _st_watercolor(rgb):
    return cv2.stylization(rgb, sigma_s=60, sigma_r=0.45)


def _st_clay(rgb):
    for _ in range(3):
        rgb = cv2.bilateralFilter(rgb, 9, 60, 60)
    return _sat(rgb, 1.25)


def _st_pop(rgb):
    return _hue(_sat(_poster(rgb, 3), 1.9), 20)


def _st_neon(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.dilate(cv2.Canny(g, 60, 140), np.ones((3, 3), np.uint8)).astype(np.float32) / 255
    glow = cv2.GaussianBlur(edges, (0, 0), 6)
    out = rgb * 0.25 + edges[..., None] * np.array([255, 60, 220]) + glow[..., None] * np.array([0, 220, 255]) * 0.9
    return np.clip(out, 0, 255).astype(np.uint8)


def _st_holo(rgb):
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    hue = ((xx + yy) / (w + h) * 270) % 180
    hsv = np.stack([hue, np.full_like(hue, 170.0), np.full_like(hue, 255.0)], -1).astype(np.uint8)
    rain = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    streak = np.exp(-(((xx - yy * 0.6) - w * 0.45) / (w * 0.08)) ** 2)[..., None] * 90
    return np.clip(rgb * 0.62 + rain * 0.38 + streak, 0, 255).astype(np.uint8)


def _st_emboss(rgb):
    k = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], np.float32)
    emb = cv2.filter2D(rgb, -1, k) + 20
    return np.clip(rgb * 0.5 + emb * 0.5, 0, 255).astype(np.uint8)


def _st_halftone(rgb):
    h, w = rgb.shape[:2]
    c = 8
    sm = _sat(cv2.resize(rgb, (max(1, w // c), max(1, h // c)), interpolation=cv2.INTER_AREA), 1.5)
    gs = cv2.cvtColor(sm, cv2.COLOR_RGB2GRAY)
    out = np.full_like(rgb, 255)
    for y in range(sm.shape[0]):
        for x in range(sm.shape[1]):
            dark = 1 - gs[y, x] / 255.0
            r = max(1, int(c * 0.75 * (0.35 + 0.65 * dark)))
            cv2.circle(out, (x * c + c // 2, y * c + c // 2), r, tuple(int(v) for v in sm[y, x]), -1)
    return out


def _st_glossy(rgb):
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    hl = np.clip(1 - (xx / w + yy / h) * 1.1, 0, 1)[..., None] ** 2 * 90
    return np.clip(_sat(rgb, 1.3) + hl, 0, 255).astype(np.uint8)


def _st_dream(rgb):
    blur = cv2.GaussianBlur(rgb, (0, 0), 8)
    return _tint(np.clip(rgb * 0.5 + blur * 0.6, 0, 255).astype(np.uint8), (255, 170, 220), 0.12)


def _st_glitch(rgb):
    out = rgb.copy()
    out[..., 0] = np.roll(rgb[..., 0], 8, axis=1)
    out[..., 2] = np.roll(rgb[..., 2], -8, axis=1)
    rng = random.Random(7)
    for _ in range(6):                                   # horizontal slice shifts
        y = rng.randrange(0, max(1, rgb.shape[0] - 20))
        hh = rng.randrange(6, 20)
        out[y:y + hh] = np.roll(out[y:y + hh], rng.randrange(-25, 25), axis=1)
    return out


def _st_blueprint(rgb):
    base = _duotone(rgb, (10, 40, 110), (200, 225, 255))
    line = (_lines(rgb) == 0)[..., None]
    return np.where(line, np.array([255, 255, 255], np.uint8), base)


def _st_noir(rgb):
    g = cv2.equalizeHist(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    return np.repeat(g[..., None], 3, axis=2)


STYLE_FN: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    # emotion tones
    "warm": lambda r: _tint(_sat(r, 1.3), (255, 210, 80), .08),
    "blue": lambda r: _tint(r, (70, 120, 255), .28),
    "green": lambda r: _tint(r, (90, 200, 90), .32),
    "pink": lambda r: _tint(r, (255, 120, 170), .26),
    "red": lambda r: _tint(_sat(r, 1.3), (255, 30, 30), .30),
    "pale": lambda r: _tint(r, (255, 255, 255), .35),
    # themes
    "superhero": lambda r: _grad_tint(_sat(_poster(r, 6), 1.6), (230, 30, 40), (30, 70, 230), .25),
    "cyberpunk": lambda r: _scan(_sat(_grad_tint(r, (0, 255, 255), (255, 0, 200), .35), 1.4)),
    "pixel_art": _st_pixel,
    "retro_80s": lambda r: _scan(_tint(_hue(_sat(r, 1.5), 30), (255, 40, 200), .18)),
    "anime": _st_anime,
    "chibi": lambda r: _sat(np.clip(cv2.bilateralFilter(r, 9, 60, 60).astype(np.int16) + 18, 0, 255).astype(np.uint8), 1.3),
    "fantasy": lambda r: np.clip(_tint(_sat(r, 1.3), (150, 90, 255), .18) + cv2.GaussianBlur(r, (0, 0), 10) * 0.35, 0, 255).astype(np.uint8),
    "neon": _st_neon,
    "line_art": _st_line_art,
    "watercolor": _st_watercolor,
    "halloween": lambda r: _tint(_duotone(r, (20, 10, 30), (255, 140, 20)), (255, 120, 0), .1),
    "christmas": lambda r: _sat(_grad_tint(r, (220, 30, 40), (30, 160, 70), .28), 1.2),
    "vaporwave": _duo((90, 20, 140), (120, 255, 240)),
    "noir": _st_noir,
    "comic": _st_comic,
    "sketch": _st_sketch,
    "gold": _duo((60, 35, 5), (255, 220, 100)),
    "ice": _duo((10, 30, 90), (220, 250, 255)),
    "lava": _duo((60, 0, 0), (255, 220, 60)),
    "sepia": _duo((40, 25, 10), (240, 215, 170)),
    "glitch": _st_glitch,
    "sakura": lambda r: _sat(_tint(r, (255, 170, 200), .25), 1.2),
    "space": _duo((10, 5, 40), (120, 200, 255)),
    "summer": lambda r: _tint(_sat(r, 1.6), (255, 200, 80), .15),
    # explicit "styles"
    "holographic": _st_holo,
    "die_cut": lambda r: _sat(r, 1.15),
    "claymation": _st_clay,
    "pop_art": _st_pop,
    "doodle": _st_doodle,
    "posterize": lambda r: _sat(_poster(r, 4), 1.3),
    "emboss": _st_emboss,
    "halftone": _st_halftone,
    "glossy": _st_glossy,
    "dream": _st_dream,
    "mosaic": _st_mosaic,
    "sunset": _duo((60, 10, 90), (255, 170, 60)),
    "blueprint": _st_blueprint,
}


def _extrude(img: Image.Image, depth: int = 10) -> Image.Image:
    """'3D pop': stack darkened copies diagonally behind the subject."""
    w, h = img.size
    out = Image.new("RGBA", (w + depth, h + depth), (0, 0, 0, 0))
    dark = np.array(img)
    dark[..., :3] = (dark[..., :3] * 0.35).astype(np.uint8)
    dark = Image.fromarray(dark)
    for i in range(depth, 0, -1):
        out.alpha_composite(dark, (i, i))
    arr = np.array(img)
    arr[..., :3] = _sat(arr[..., :3], 1.4)
    out.alpha_composite(Image.fromarray(arr), (0, 0))
    return out


def apply_style(img: Image.Image, name: str) -> Image.Image:
    if name == "none":
        return img
    if name == "3d_pop":
        return _extrude(img)
    arr = np.array(img)
    arr[..., :3] = STYLE_FN[name](np.ascontiguousarray(arr[..., :3]))
    return Image.fromarray(arr)


def apply_geo(img: Image.Image, geo) -> Image.Image:
    for g in geo:
        kind = g[0]
        if kind == "flip":
            img = ImageOps.mirror(img)
        elif kind == "squash":
            img = img.resize((max(1, int(img.width * g[1])), max(1, int(img.height * g[2]))), Image.LANCZOS)
        elif kind == "tilt":
            img = img.rotate(g[1], resample=Image.BICUBIC, expand=True)
        elif kind == "shake":
            d = max(6, img.width // 30)
            out = Image.new("RGBA", (img.width + 4 * d, img.height), (0, 0, 0, 0))
            for dx in (-d, d):
                ghost = img.copy()
                ghost.putalpha(ghost.getchannel("A").point(lambda v: int(v * 0.25)))
                out.alpha_composite(ghost, (2 * d + dx, 0))
            out.alpha_composite(img, (2 * d, 0))
            img = out
    return img


# =========================================================================== #
# 3. Catalog: 100+ distinct sticker specs
# =========================================================================== #
@dataclass(frozen=True)
class Spec:
    idx: int
    category: str
    name: str
    prompt: str
    styles: tuple = ()
    geo: tuple = ()
    overlays: tuple = ()
    text: str = ""
    text_color: str = "#111827"
    emoji: str = "😀"
    border: int = 10

    def simplified(self) -> "Spec":
        """Last-resort variant: no effects, just the subject + a label."""
        return replace(self, styles=(), geo=(), overlays=(), text=self.name.replace("_", " ").upper()[:14],
                       text_color="#111827")


# (name, emoji, expression prompt, styles, geo, overlays)
EMOTIONS = [
    ("happy", "😀", "big joyful smile", ("warm",), (("tilt", -6),), (("stars", "tr"),)),
    ("angry", "😡", "furious angry scowl", ("red",), (("shake",),), (("text", "#@!", "tr", "#dc2626"),)),
    ("crying", "😭", "crying with streaming tears", ("blue",), (("squash", 1.0, .94),), (("drops", "face"),)),
    ("mind_blown", "🤯", "mind blown, head exploding", ("warm",), (("tilt", 8),), (("burst", "top"), ("text", "!!!", "tr", "#f59e0b"))),
    ("facepalm", "🤦", "facepalm, exasperated", (), (("tilt", -14), ("squash", 1.0, .92)), (("text", "sigh", "tr", "#6b7280"),)),
    ("laughing", "😂", "laughing out loud", ("warm",), (("tilt", 10), ("shake",)), (("text", "HA HA", "tl", "#f59e0b"),)),
    ("shocked", "😱", "shocked wide-eyed scream", ("pale",), (("squash", .92, 1.06),), (("text", "!?", "tr", "#ef4444"),)),
    ("thinking", "🤔", "thinking, hand on chin", (), (("tilt", 6),), (("text", "hmm...", "tr", "#6b7280"),)),
    ("sleeping", "😴", "fast asleep", ("blue",), (("tilt", -12),), (("text", "Zzz", "tr", "#4f7cff"),)),
    ("confused", "😕", "confused, puzzled", (), (("tilt", -8),), (("text", "???", "tr", "#8b5cf6"),)),
    ("love", "😍", "heart eyes in love", ("pink",), (), (("hearts", "tl"), ("hearts", "tr"))),
    ("cool", "😎", "cool with sunglasses", (), (("tilt", 5),), (("stars", "tr"), ("text", "cool", "tl", "#0ea5e9"))),
    ("embarrassed", "😳", "blushing embarrassed", ("pink",), (("squash", .95, 1.0),), (("drops", "tr"),)),
    ("scared", "😨", "terrified", ("pale",), (("shake",),), (("text", "AAA!", "tr", "#6366f1"),)),
    ("sick", "🤢", "nauseated sick", ("green",), (("tilt", -6),), (("text", "blegh", "tr", "#16a34a"),)),
    ("excited", "🤩", "starstruck excited", ("warm",), (("tilt", 4),), (("stars", "tl"), ("stars", "tr"), ("confetti",))),
    ("bored", "😑", "bored deadpan", (), (("squash", 1.05, .95),), (("text", "meh", "tr", "#6b7280"),)),
    ("sad", "😢", "sad teary", ("blue",), (("squash", 1.0, .93),), (("drops", "tr"),)),
    ("wink", "😉", "playful wink", ("warm",), (("tilt", 8),), (("stars", "tr"),)),
    ("surprised", "😮", "surprised open mouth", (), (("squash", .95, 1.05),), (("text", "!", "tr", "#ef4444"),)),
    ("dizzy", "😵", "dizzy, spiral eyes", (), (("tilt", 18),), (("text", "@_@", "tr", "#8b5cf6"),)),
    ("celebrate", "🥳", "party celebration", ("warm",), (("tilt", -6),), (("confetti",),)),
    ("sweating", "😅", "nervous sweat", (), (("tilt", 5),), (("drops", "tr"),)),
    ("eye_roll", "🙄", "rolling eyes", (), (("tilt", -10),), (("text", "...", "tr", "#6b7280"),)),
]

# (name, emoji, style, geo, overlays, prompt)
THEMES = [
    ("superhero", "🦸", "superhero", (("tilt", -5),), (("burst", "tr"), ("text", "POW!", "tl", "#e11d48")), "comic book superhero with cape"),
    ("cyberpunk", "🤖", "cyberpunk", (), (), "cyberpunk neon city style"),
    ("pixel_art", "👾", "pixel_art", (), (), "16-bit pixel art"),
    ("retro_80s", "📼", "retro_80s", (("tilt", 4),), (("text", "80s", "tr", "#ff2bd6"),), "retro 1980s synthwave"),
    ("anime", "🎌", "anime", (), (("stars", "tr"),), "anime cel-shaded"),
    ("chibi", "🍡", "chibi", (("squash", 1.1, .8),), (("hearts", "tr"),), "cute chibi big head"),
    ("fantasy", "🧙", "fantasy", (), (("stars", "tl"), ("stars", "tr")), "high fantasy magical"),
    ("neon", "💡", "neon", (), (), "glowing neon outline"),
    ("line_art", "✏️", "line_art", (), (), "clean black line art"),
    ("watercolor", "🎨", "watercolor", (), (), "soft watercolor painting"),
    ("halloween", "🎃", "halloween", (), (("text", "BOO!", "tr", "#f97316"),), "spooky halloween"),
    ("christmas", "🎄", "christmas", (), (("stars", "tr"), ("text", "Ho Ho", "tl", "#dc2626")), "christmas festive"),
    ("vaporwave", "🌴", "vaporwave", (("tilt", -4),), (), "vaporwave aesthetic"),
    ("noir", "🕵️", "noir", (), (), "black and white film noir"),
    ("comic", "💥", "comic", (), (("burst", "tr"),), "american comic book"),
    ("sketch", "📝", "sketch", (), (), "pencil sketch"),
    ("gold", "🥇", "gold", (), (("stars", "tr"),), "solid gold statue"),
    ("ice", "🧊", "ice", (), (), "frozen ice crystal"),
    ("lava", "🌋", "lava", (), (), "molten lava"),
    ("sepia", "📜", "sepia", (), (), "vintage sepia photo"),
    ("glitch", "📺", "glitch", (), (), "digital glitch art"),
    ("sakura", "🌸", "sakura", (), (("hearts", "tr"),), "cherry blossom pink"),
    ("space", "🚀", "space", (), (("stars", "tl"), ("stars", "tr")), "deep space cosmic"),
    ("summer", "🏖️", "summer", (), (("stars", "tr"),), "sunny summer vibes"),
]

# (text, emoji, text colour)
TEXTS = [
    ("LOL", "😂", "#f59e0b"), ("AFK", "⌨️", "#0ea5e9"), ("BRB", "⏳", "#8b5cf6"), ("GG", "🎮", "#16a34a"),
    ("No Way", "🙅", "#dc2626"), ("Yes!", "✅", "#16a34a"), ("Please", "🙏", "#ec4899"), ("Mood", "🫠", "#6366f1"),
    ("100", "💯", "#dc2626"), ("FIRE", "🔥", "#ea580c"), ("OMG", "😱", "#e11d48"), ("WTF", "🤨", "#7c3aed"),
    ("Thanks", "🙌", "#0d9488"), ("Sorry", "😔", "#64748b"), ("Hi!", "👋", "#0ea5e9"), ("Bye", "🚪", "#64748b"),
    ("Love U", "❤️", "#e11d48"), ("Good Night", "🌙", "#4f46e5"), ("Good Morning", "☀️", "#f59e0b"),
    ("Congrats", "🎉", "#d946ef"), ("Wow", "🤩", "#f59e0b"), ("Nope", "🚫", "#dc2626"), ("OK", "👌", "#16a34a"),
    ("Big Mood", "😶‍🌫️", "#6366f1"), ("Vibes", "✨", "#a855f7"), ("Boss", "😎", "#0f172a"),
    ("GOAT", "🐐", "#b45309"), ("Cheers", "🍻", "#ca8a04"),
]

# (name, emoji)  - pure rendering styles
STYLES = [
    ("holographic", "🌈"), ("3d_pop", "🧊"), ("die_cut", "✂️"), ("claymation", "🧸"), ("pop_art", "🖼️"),
    ("doodle", "🖍️"), ("posterize", "🎚️"), ("emboss", "🪙"), ("halftone", "📰"), ("glossy", "💎"),
    ("dream", "☁️"), ("mosaic", "🧩"), ("sunset", "🌇"), ("blueprint", "📐"),
]

# emotion x style mash-ups
COMBOS = [
    ("happy", "neon"), ("angry", "comic"), ("crying", "watercolor"), ("mind_blown", "glitch"), ("laughing", "pop_art"),
    ("shocked", "halftone"), ("thinking", "sketch"), ("love", "holographic"), ("cool", "cyberpunk"), ("sleeping", "dream"),
    ("excited", "3d_pop"), ("confused", "doodle"), ("celebrate", "vaporwave"), ("scared", "noir"), ("wink", "anime"),
]


def build_catalog() -> list[Spec]:
    """Deterministic list of 100+ Spec objects, one per sticker."""
    specs: list[Spec] = []
    emo = {}
    for name, emoji, prompt, styles, geo, ov in EMOTIONS:
        s = Spec(0, "Emotions/Reactions", name, f"the same character with a {prompt} expression",
                 styles, geo, ov, emoji=emoji)
        emo[name] = s
        specs.append(s)
    for name, emoji, style, geo, ov, prompt in THEMES:
        specs.append(Spec(0, "Themes & Costumes", name, f"the same character, {prompt}", (style,), geo, ov, emoji=emoji))
    for i, (text, emoji, color) in enumerate(TEXTS):
        specs.append(Spec(0, "Texts/Memes", text.lower().replace(" ", "_").replace("!", ""),
                          f'sticker with big bold text "{text}"', (), (("tilt", 4 if i % 2 else -4),),
                          (), text=text, text_color=color, emoji=emoji))
    for name, emoji in STYLES:
        specs.append(Spec(0, "Styles", name, f"the same character in {name.replace('_', ' ')} style",
                          (name,), (), (), emoji=emoji, border=18 if name == "die_cut" else 10))
    for e, style in COMBOS:
        base = emo[e]
        specs.append(replace(base, category="Mash-ups", name=f"{e}+{style}",
                             prompt=f"{base.prompt}, {style.replace('_', ' ')} style", styles=base.styles + (style,)))
    return [replace(s, idx=i + 1) for i, s in enumerate(specs)]


# =========================================================================== #
# 4. Post-processing: overlays, border, shadow, encoding
# =========================================================================== #
def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


ANCHORS = {"tl": (95, 85), "tr": (420, 85), "bl": (95, 400), "br": (420, 400),
           "top": (256, 55), "center": (256, 256), "bottom": (256, 440)}


def _draw_text(canvas, text, xy, color, size=64, max_w=440):
    d = ImageDraw.Draw(canvas)
    font = _font(size)
    while size > 16 and d.textlength(text, font=font) > max_w:
        size -= 4
        font = _font(size)
    d.text(xy, text, font=font, fill=color, anchor="mm", stroke_width=max(4, size // 8), stroke_fill="white")


def _outlined(d, shape_fn, color):
    shape_fn(d, "white", 5)
    shape_fn(d, color, 0)


def _star(d, x, y, r, color):
    def draw(dr, fill, grow):
        pts = []
        for k in range(10):
            rad = (r + grow) * (1 if k % 2 == 0 else 0.45)
            ang = -np.pi / 2 + k * np.pi / 5
            pts.append((x + rad * np.cos(ang), y + rad * np.sin(ang)))
        dr.polygon(pts, fill=fill)
    _outlined(d, draw, color)


def _heart(d, x, y, s, color):
    def draw(dr, fill, grow):
        s2 = s + grow
        dr.ellipse((x - s2 * 1.05, y - s2 * .85, x + .02 * s2, y + s2 * .25), fill=fill)
        dr.ellipse((x - .02 * s2, y - s2 * .85, x + s2 * 1.05, y + s2 * .25), fill=fill)
        dr.polygon([(x - s2 * 1.02, y - s2 * .1), (x + s2 * 1.02, y - s2 * .1), (x, y + s2 * 1.1)], fill=fill)
    _outlined(d, draw, color)


def _drop(d, x, y, s):
    def draw(dr, fill, grow):
        s2 = s + grow
        dr.polygon([(x, y - s2 * 1.5), (x - s2 * .8, y), (x + s2 * .8, y)], fill=fill)
        dr.ellipse((x - s2 * .8, y - s2 * .6, x + s2 * .8, y + s2), fill=fill)
    _outlined(d, draw, (60, 150, 255))


def draw_overlay(canvas: Image.Image, ov: tuple, seed: int):
    d = ImageDraw.Draw(canvas)
    kind = ov[0]
    if kind == "text":
        _draw_text(canvas, ov[1], ANCHORS[ov[2]], ov[3], size=58, max_w=200)
    elif kind == "stars":
        x, y = ANCHORS[ov[1]]
        for dx, dy, r in ((0, 0, 26), (34, 30, 16), (-30, 34, 12)):
            _star(d, x + dx, y + dy, r, (255, 200, 30))
    elif kind == "hearts":
        x, y = ANCHORS[ov[1]]
        for dx, dy, s in ((0, 0, 20), (30, 34, 12), (-26, 30, 10)):
            _heart(d, x + dx, y + dy, s, (240, 50, 90))
    elif kind == "drops":
        pts = [(200, 235), (312, 235)] if ov[1] == "face" else [ANCHORS[ov[1]]]
        for x, y in pts:
            _drop(d, x, y, 16)
    elif kind == "confetti":
        rng = random.Random(seed)
        for _ in range(38):
            x, y = rng.choice([(rng.randint(10, 500), rng.randint(10, 90)),
                               (rng.randint(10, 60), rng.randint(10, 500)),
                               (rng.randint(450, 500), rng.randint(10, 500))])
            c = tuple(rng.randint(80, 255) for _ in range(3))
            d.rectangle((x, y, x + rng.randint(6, 12), y + rng.randint(4, 8)), fill=c + (255,))
    elif kind == "burst":
        x, y = ANCHORS[ov[1]]
        for k in range(12):
            a = k * np.pi / 6
            d.line((x + 34 * np.cos(a), y + 34 * np.sin(a), x + 62 * np.cos(a), y + 62 * np.sin(a)),
                   fill=(255, 190, 0, 255), width=7)


def compose(subject: Image.Image, spec: Spec) -> Image.Image:
    """Crop -> auto-pad -> border -> shadow -> overlays -> 512x512 RGBA."""
    subj = _crop_alpha(subject, pad=4)
    band = 100 if spec.text else 0
    box_w, box_h = SIZE - 2 * MARGIN, SIZE - 2 * MARGIN - band
    scale = min(box_w / subj.width, box_h / subj.height)
    subj = subj.resize((max(1, int(subj.width * scale)), max(1, int(subj.height * scale))), Image.LANCZOS)

    layer = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    layer.alpha_composite(subj, ((SIZE - subj.width) // 2, MARGIN + (box_h - subj.height) // 2))

    alpha = np.array(layer.getchannel("A"))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * spec.border + 1,) * 2)
    dil = cv2.GaussianBlur(cv2.dilate(alpha, k), (0, 0), 1.0)

    shadow_a = cv2.GaussianBlur(cv2.dilate(alpha, k), (0, 0), 5)
    shadow_a = np.roll(np.roll(shadow_a, 6, axis=0), 3, axis=1) * 0.35
    shadow = np.zeros((SIZE, SIZE, 4), np.uint8)
    shadow[..., 3] = shadow_a.astype(np.uint8)
    border = np.full((SIZE, SIZE, 4), 255, np.uint8)
    border[..., 3] = dil

    out = Image.fromarray(shadow)
    out.alpha_composite(Image.fromarray(border))
    out.alpha_composite(layer)

    for ov in spec.overlays:
        draw_overlay(out, ov, spec.idx)
    if spec.text:
        _draw_text(out, spec.text, (SIZE // 2, SIZE - MARGIN - 44), spec.text_color, size=96, max_w=430)
    return out


def encode_webp(img: Image.Image, limit: int = at.MAX_BYTES) -> bytes:
    buf = io.BytesIO()
    for q in (92, 85, 78, 70, 60, 50, 40, 30, 20):
        buf = io.BytesIO()
        img.save(buf, "WEBP", quality=q, method=4)
        if buf.tell() <= limit:
            break
    return buf.getvalue()


# =========================================================================== #
# 5. Optional AI backend (Replicate img2img)
# =========================================================================== #
class ReplicateBackend:
    """
    Minimal Replicate REST client for an img2img model (e.g. an SDXL img2img
    version). Model input names differ between versions, so they are configurable:

        REPLICATE_API_TOKEN   your token
        REPLICATE_VERSION     model version hash (required)
        REPLICATE_IMAGE_KEY   default "image"
        REPLICATE_PROMPT_KEY  default "prompt"

    NOTE: written against Replicate's documented REST API but not exercised in
    CI (needs a live token). If the backend fails, StickerFactory falls back to
    local styling automatically.
    """
    API = "https://api.replicate.com/v1"

    def __init__(self, token: Optional[str] = None, version: Optional[str] = None, timeout: int = 180):
        self.token = token or os.getenv("REPLICATE_API_TOKEN", "")
        self.version = version or os.getenv("REPLICATE_VERSION", "")
        self.image_key = os.getenv("REPLICATE_IMAGE_KEY", "image")
        self.prompt_key = os.getenv("REPLICATE_PROMPT_KEY", "prompt")
        self.timeout = timeout
        if not (self.token and self.version):
            raise ValueError("Replicate needs REPLICATE_API_TOKEN and REPLICATE_VERSION")

    def stylize(self, img: Image.Image, prompt: str) -> Image.Image:
        import requests
        headers = {"Authorization": f"Bearer {self.token}"}

        def call() -> Image.Image:
            buf = io.BytesIO()
            img.convert("RGB").save(buf, "PNG")
            uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
            r = requests.post(f"{self.API}/predictions", headers=headers, timeout=120, json={
                "version": self.version,
                "input": {self.image_key: uri, self.prompt_key: prompt + ", sticker art, plain white background"}})
            if r.status_code == 429:
                raise at.RateLimitError(float(r.headers.get("retry-after", 5)))
            r.raise_for_status()
            pred, t0 = r.json(), time.time()
            while pred["status"] not in ("succeeded", "failed", "canceled"):
                if time.time() - t0 > self.timeout:
                    raise TimeoutError("replicate prediction timed out")
                time.sleep(1.5)
                pred = requests.get(pred["urls"]["get"], headers=headers, timeout=60).json()
            if pred["status"] != "succeeded":
                raise RuntimeError(f"replicate: {pred.get('error')}")
            out = pred["output"]
            url = out[0] if isinstance(out, list) else out
            return Image.open(io.BytesIO(requests.get(url, timeout=60).content)).convert("RGB")

        return at.retry_with_backoff(call, retries=3)


# =========================================================================== #
# 6. Factory + pack generation loop
# =========================================================================== #
class StickerFactory:
    def __init__(self, source: Image.Image, threshold: int = 128, ai: Optional[ReplicateBackend] = None):
        self.source, self.threshold, self.ai = source, threshold, ai
        self.subject = prep_subject(extract_subject(source, threshold))

    def _local(self, subject: Image.Image, spec: Spec) -> Image.Image:
        img = subject
        for style in spec.styles:
            img = apply_style(img, style)
        return compose(apply_geo(img, spec.geo), spec)

    def _ai(self, spec: Spec) -> Image.Image:
        flat = Image.new("RGB", self.subject.size, (255, 255, 255))
        flat.paste(self.subject, mask=self.subject.getchannel("A"))
        styled = self.ai.stylize(flat, spec.prompt)
        cut = prep_subject(extract_subject(styled, self.threshold))
        return compose(apply_geo(cut, spec.geo), spec)

    def render(self, spec: Spec, attempt: int = 0) -> bytes:
        attempt = min(attempt, LAST_RESORT)
        subject = self.subject
        if attempt >= 2:                                   # self-heal: redo background removal
            thr = 100 if attempt == 2 else 160
            method = "auto" if attempt == 2 else "grabcut"
            subject = prep_subject(extract_subject(self.source, thr, method))
        if attempt == LAST_RESORT:
            spec = spec.simplified()
        if self.ai and attempt < 2:
            try:
                return encode_webp(self._ai(spec))
            except at.RateLimitError:
                raise                                       # let heal_index back off
            except Exception as exc:                        # AI down -> local fallback right away
                log.warning("[#%03d] AI backend failed (%s); using local pipeline", spec.idx, exc)
        return encode_webp(self._local(subject, spec))


@dataclass
class StickerResult:
    index: int
    name: str
    category: str
    emoji: str
    prompt: str
    ok: bool
    attempts: int
    path: str = ""
    problems: list = field(default_factory=list)


def setup_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not any(isinstance(h, logging.FileHandler) for h in log.handlers):
        fh = logging.FileHandler(out_dir / "errors.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(fh)
    log.setLevel(logging.INFO)


def generate_pack(image: Image.Image, out_dir, count: Optional[int] = None, use_ai: bool = False,
                  threshold: int = 128, progress: Optional[Callable[[int, int, str], None]] = None,
                  backend: Optional[ReplicateBackend] = None) -> list[StickerResult]:
    """
    Generate the whole pack. Failed indices are re-run (with escalating fixes)
    for up to MAX_PASSES passes. Writes stickers/*.webp + manifest.json to out_dir.
    """
    out = Path(out_dir)
    sticker_dir = out / "stickers"
    sticker_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out)

    specs = build_catalog()[: count or None]
    ai = None
    if use_ai:
        try:
            ai = backend or ReplicateBackend()
        except Exception as exc:
            log.warning("AI disabled: %s", exc)
    factory = StickerFactory(image, threshold, ai)

    seen: set = set()
    results: dict[int, StickerResult] = {}
    pending = list(specs)
    done = 0
    for pass_no in range(MAX_PASSES):
        still: list[Spec] = []
        for spec in pending:
            heal = at.heal_index(spec.idx, lambda a, s=spec: factory.render(s, a), seen,
                                 start_attempt=min(pass_no * 2, LAST_RESORT), max_attempts=5,
                                 last_resort_attempt=LAST_RESORT)
            prev = results.get(spec.idx)
            attempts = (prev.attempts if prev else 0) + heal.attempts
            if heal.ok:
                path = sticker_dir / f"{spec.idx:03d}_{spec.name.replace('+', '_')}.webp"
                path.write_bytes(heal.data)
                results[spec.idx] = StickerResult(spec.idx, spec.name, spec.category, spec.emoji, spec.prompt,
                                                  True, attempts, str(path))
                done += 1
                if progress:
                    progress(done, len(specs), f"{spec.category}: {spec.name}")
            else:
                results[spec.idx] = StickerResult(spec.idx, spec.name, spec.category, spec.emoji, spec.prompt,
                                                  False, attempts, "", heal.problems)
                still.append(spec)
        pending = still
        if not pending:
            break
        log.warning("pass %d: %d stickers still failing, re-running", pass_no + 1, len(pending))

    ordered = [results[s.idx] for s in specs]
    (out / "manifest.json").write_text(json.dumps([r.__dict__ for r in ordered], indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    failed = [r.index for r in ordered if not r.ok]
    log.info("generated %d/%d stickers; failed: %s", len(ordered) - len(failed), len(ordered), failed or "none")
    return ordered


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Generate a 100+ sticker pack from one image")
    ap.add_argument("image")
    ap.add_argument("--out", default="output/run")
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--ai", action="store_true", help="use Replicate img2img (needs env vars)")
    a = ap.parse_args()
    res = generate_pack(Image.open(a.image), a.out, a.count, a.ai,
                        progress=lambda d, t, m: print(f"\r{d}/{t} {m:40s}", end=""))
    print(f"\nOK: {sum(r.ok for r in res)}/{len(res)}  ->  {a.out}/stickers")
