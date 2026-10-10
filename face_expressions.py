"""
face_expressions.py - real facial-expression changes with pure OpenCV + PIL.

    face = find_face(subject_rgba)                    # detect (Haar) or estimate the face box
    out  = apply_expression(subject_rgba, face, "angry", intensity=1.0)

How it works
  1. LOCATE   Haar-cascade face (+ eye) detection; if none (cartoon, pet, mascot, emoji...)
              the face box is estimated from the subject silhouette.
  2. ERASE    the original eyes / brows / mouth zones are inpainted away (cv2.inpaint).
  3. REDRAW   new eyes, brows, mouth, blush and tears are drawn (4x supersampled, antialiased)
              for the requested emotion, clipped to the subject's silhouette.
  4. VERIFY   face_change_score() measures how much the face region changed; the generator
              rejects stickers below MIN_FACE_CHANGE so the healing loop retries with a
              higher `intensity`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

S = 4                                   # supersampling factor
INK = (38, 24, 24, 255)
WHITE = (255, 255, 255, 255)
MIN_FACE_CHANGE = 3.5                   # mean abs RGB diff (0-255) inside the face box


class NoFaceChange(Exception):
    """Raised when an expression did not visibly change the face (triggers a retry)."""


# --------------------------------------------------------------------------- #
# 1. Locate the face
# --------------------------------------------------------------------------- #
@dataclass
class Face:
    x: float; y: float; w: float; h: float          # face box in subject pixels
    eye_y: float; eye_dx: float; eye_r: float       # eye line, half-separation, radius
    mouth_y: float; mouth_w: float
    detected: bool = False

    @property
    def cx(self): return self.x + self.w / 2

    @property
    def box(self): return (int(self.x), int(self.y), int(self.x + self.w), int(self.y + self.h))


def _layout(x, y, w, h, detected, eyes=None) -> Face:
    eye_y, eye_dx = y + 0.40 * h, 0.21 * w
    if eyes is not None and len(eyes) == 2:
        (x1, y1, r1), (x2, y2, r2) = sorted(eyes)
        eye_y, eye_dx = (y1 + y2) / 2, max(0.14 * w, min(0.30 * w, abs(x2 - x1) / 2))
    return Face(x, y, w, h, eye_y, eye_dx, 0.085 * w, y + 0.74 * h, 0.34 * w, detected)


def _cascade(name: str):
    """Load a Haar cascade, or None when this OpenCV build has none (e.g. OpenCV 5.x)."""
    try:
        if not hasattr(cv2, "CascadeClassifier"):
            return None
        casc = cv2.CascadeClassifier(cv2.data.haarcascades + name)
        return None if casc.empty() else casc
    except Exception:                                   # noqa: BLE001 - detection is optional
        return None


def _detect_eyes(gray, x, y, w, h):
    casc = _cascade("haarcascade_eye.xml")
    if casc is None:
        return None
    roi = gray[y:y + int(h * 0.6), x:x + w]
    found = casc.detectMultiScale(roi, 1.1, 5, minSize=(max(8, w // 10), max(8, w // 10)))
    if len(found) < 2:
        return None
    found = sorted(found, key=lambda e: -e[2])[:2]
    pts = [(x + ex + ew / 2, y + ey + eh / 2, ew / 2) for ex, ey, ew, eh in found]
    return pts if abs(pts[0][1] - pts[1][1]) < 0.12 * h and abs(pts[0][0] - pts[1][0]) > 0.2 * w else None


def find_face(subject: Image.Image) -> Face:
    """Detect a frontal face; otherwise estimate the face box from the silhouette."""
    rgba = np.array(subject.convert("RGBA"))
    alpha = rgba[..., 3]
    gray = cv2.equalizeHist(cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2GRAY))
    H, W = gray.shape
    for name in ("haarcascade_frontalface_default.xml", "haarcascade_frontalface_alt2.xml"):
        casc = _cascade(name)
        if casc is None:                                        # no Haar support -> silhouette estimate below
            break
        faces = casc.detectMultiScale(gray, 1.08, 4, minSize=(max(30, W // 8), max(30, H // 8)))
        faces = [f for f in faces if int(alpha[int(f[1] + f[3] // 2), int(f[0] + f[2] // 2)]) > 128]
        if len(faces):
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            y2 = max(0, y - int(0.04 * h))                     # Haar boxes start at the brow line
            return _layout(x, y2, w, h + (y - y2), True, _detect_eyes(gray, x, y, w, h))

    ys, xs = np.where(alpha > 128)                              # estimate from silhouette
    if len(xs) == 0:
        return _layout(0, 0, W, H, False)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    if bh / bw < 1.3:                                           # avatar / emoji / close-up: whole thing is the face
        return _layout(x0, y0, bw, bh, False)
    fw = min(bw * 0.7, bh * 0.45)                               # head-and-shoulders / full body: head on top
    return _layout((x0 + x1) / 2 - fw / 2, y0, fw, fw * 1.15, False)


# --------------------------------------------------------------------------- #
# 2. Expression table
# --------------------------------------------------------------------------- #
# eyes: open wide happy sleep squint hearts stars spiral shades wink half roll side up
# mouth: smile big_smile grin laugh frown wail o big_o flat wavy grit smirk tongue
# brow: (inner_angle_deg, raise, mode)  angle>0 = angry (inner ends down), <0 = worried
EXPR: dict[str, dict] = {
    "neutral":    dict(eyes="open", mouth="flat", brow=(0, 0, "both")),
    "happy":      dict(eyes="happy", mouth="big_smile", brow=(-4, .05, "both"), blush=.35),
    "angry":      dict(eyes="angry", mouth="grit", brow=(28, -.02, "both")),
    "crying":     dict(eyes="sleep", mouth="wail", brow=(-24, .02, "both"), tears=2),
    "mind_blown": dict(eyes="wide", mouth="big_o", brow=(-4, .14, "both")),
    "facepalm":   dict(eyes="half", mouth="flat", brow=(-14, 0, "both")),
    "laughing":   dict(eyes="squint", mouth="laugh", brow=(-4, .05, "both"), blush=.35, tears=1),
    "shocked":    dict(eyes="wide", mouth="big_o", brow=(-8, .16, "both")),
    "thinking":   dict(eyes="up", mouth="smirk", brow=(0, .05, "one_up")),
    "sleeping":   dict(eyes="sleep", mouth="o", brow=(-6, .02, "both")),
    "confused":   dict(eyes="side", mouth="wavy", brow=(0, .05, "one_up")),
    "love":       dict(eyes="hearts", mouth="big_smile", brow=(-4, .06, "both"), blush=.5),
    "cool":       dict(eyes="shades", mouth="smirk", brow=(0, .0, "none")),
    "embarrassed": dict(eyes="side", mouth="wavy", brow=(-16, .03, "both"), blush=.75),
    "scared":     dict(eyes="wide", mouth="wavy", brow=(-22, .14, "both")),
    "sick":       dict(eyes="half", mouth="wavy", brow=(-18, 0, "both")),
    "excited":    dict(eyes="stars", mouth="big_smile", brow=(-4, .12, "both"), blush=.3),
    "bored":      dict(eyes="half", mouth="flat", brow=(0, -.01, "both")),
    "sad":        dict(eyes="open", mouth="frown", brow=(-26, .0, "both"), tears=1),
    "wink":       dict(eyes="wink", mouth="tongue", brow=(-4, .05, "one_up")),
    "surprised":  dict(eyes="wide", mouth="o", brow=(-6, .13, "both")),
    "dizzy":      dict(eyes="spiral", mouth="wavy", brow=(-8, .02, "both")),
    "celebrate":  dict(eyes="happy", mouth="laugh", brow=(-4, .1, "both"), blush=.3),
    "sweating":   dict(eyes="side", mouth="wavy", brow=(-18, .04, "both")),
    "eye_roll":   dict(eyes="roll", mouth="flat", brow=(0, .06, "one_up")),
}
# extra expressions used by text / theme / style stickers
EXPR["smug"] = EXPR["cool"]

# --------------------------------------------------------------------------- #
# 3. Drawing primitives (all coordinates are in subject pixels, scaled by S internally)
# --------------------------------------------------------------------------- #
def _bez(p0, p1, p2, n=40):
    return [((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0],
             (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1]) for t in np.linspace(0, 1, n)]


class Painter:
    def __init__(self, size):
        self.size = size
        self.img = Image.new("RGBA", (size[0] * S, size[1] * S), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.img)

    def pt(self, p): return (p[0] * S, p[1] * S)

    def line(self, pts, w, color=INK):
        pts = [self.pt(p) for p in pts]
        ww = max(2, w * S)
        self.d.line(pts, fill=color, width=int(ww), joint="curve")
        for p in (pts[0], pts[-1]):
            self.d.ellipse((p[0] - ww / 2, p[1] - ww / 2, p[0] + ww / 2, p[1] + ww / 2), fill=color)

    def ellipse(self, cx, cy, rx, ry, fill=None, outline=None, w=0):
        self.d.ellipse(((cx - rx) * S, (cy - ry) * S, (cx + rx) * S, (cy + ry) * S), fill=fill,
                       outline=outline, width=int(w * S))

    def poly(self, pts, fill):
        self.d.polygon([self.pt(p) for p in pts], fill=fill)

    def finish(self, clip_alpha: np.ndarray) -> Image.Image:
        out = self.img.resize(self.size, Image.LANCZOS)
        a = np.array(out.getchannel("A")).astype(np.float32) * (clip_alpha.astype(np.float32) / 255)
        out.putalpha(Image.fromarray(a.astype(np.uint8)))
        return out


def _star_pts(x, y, r):
    return [(x + r * (1 if k % 2 == 0 else .45) * math.cos(-math.pi / 2 + k * math.pi / 5),
             y + r * (1 if k % 2 == 0 else .45) * math.sin(-math.pi / 2 + k * math.pi / 5)) for k in range(10)]


def _heart_pts(x, y, s):
    pts = []
    for t in np.linspace(0, 2 * math.pi, 60):
        pts.append((x + s * 0.062 * 16 * math.sin(t) ** 3,
                    y - s * 0.062 * (13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)) + s * .1))
    return pts


def _draw_eye(p: Painter, kind: str, x, y, r, side: int, skin, k):
    """side: -1 left eye, +1 right eye."""
    lw = max(2.0, r * 0.32)
    def open_eye(rx, ry, pupil=(0, 0), pr=.5):
        p.ellipse(x, y, rx, ry, fill=WHITE, outline=INK, w=lw * .8)
        p.ellipse(x + pupil[0] * rx, y + pupil[1] * ry, pr * r, pr * r, fill=INK)
        p.ellipse(x + pupil[0] * rx - pr * r * .3, y + pupil[1] * ry - pr * r * .3, pr * r * .28, pr * r * .28, fill=WHITE)

    if kind == "open":
        open_eye(r * 1.15, r * 1.05, (0, .05), .62)
    elif kind == "wide":
        open_eye(r * 1.45 * k ** .3, r * 1.45 * k ** .3, (0, 0), .34)
    elif kind == "side":
        open_eye(r * 1.15, r * 1.05, (side * -.42 + .42 * 0, .05), .58)
    elif kind == "up":
        open_eye(r * 1.15, r * 1.05, (.35, -.5), .55)
    elif kind == "roll":
        open_eye(r * 1.2, r * 1.1, (.1 * side, -.62), .55)
    elif kind == "angry":
        open_eye(r * 1.1, r * .85, (side * -.15, .1), .6)
        ox, ix = x + side * r * 1.5, x - side * r * 1.5           # outer / inner end of the lid
        oy, iy = y - r * .95, y - r * .95 + r * .9 * min(1.3, k)  # inner end sits lower -> scowl
        p.poly([(ox, oy - r * 1.3), (ix, iy - r * 1.3), (ix, iy), (ox, oy)], skin + (255,))
        p.line([(ox, oy), (ix, iy)], lw)
    elif kind == "half":
        open_eye(r * 1.15, r * 1.0, (0, .3), .6)
        p.poly([(x - r * 1.5, y - r * 1.3), (x + r * 1.5, y - r * 1.3), (x + r * 1.5, y - r * .05), (x - r * 1.5, y - r * .05)],
               skin + (255,))
        p.line([(x - r * 1.2, y - r * .05), (x + r * 1.2, y - r * .05)], lw)
    elif kind == "happy":
        p.line(_bez((x - r * 1.1, y + r * .5), (x, y - r * 1.9), (x + r * 1.1, y + r * .5)), lw * 1.15)
    elif kind == "sleep":
        p.line(_bez((x - r * 1.1, y - r * .2), (x, y + r * 1.5), (x + r * 1.1, y - r * .2)), lw * 1.15)
    elif kind == "squint":
        s = -side
        p.line([(x + s * r * 1.1, y - r * .9), (x - s * r * 1.0, y), (x + s * r * 1.1, y + r * .9)], lw * 1.2)
    elif kind == "hearts":
        pts = _heart_pts(x, y, r * 3.0 * min(1.25, k))
        p.poly(pts, (235, 40, 85, 255))
        p.ellipse(x - r * .5, y - r * .6, r * .28, r * .28, fill=WHITE)
    elif kind == "stars":
        p.poly(_star_pts(x, y, r * 1.9 * min(1.25, k)), (255, 196, 20, 255))
        p.ellipse(x, y, r * .4, r * .4, fill=(255, 250, 220, 255))
    elif kind == "spiral":
        pts = [(x + (0.15 + t / 22.0) * r * 1.35 * math.cos(t), y + (0.15 + t / 22.0) * r * 1.35 * math.sin(t))
               for t in np.linspace(0, 6 * math.pi, 120)]
        p.line(pts, lw * .7)
    elif kind == "wink":
        if side < 0:
            open_eye(r * 1.15, r * 1.05, (0, .05), .62)
        else:
            p.line(_bez((x - r * 1.1, y + r * .4), (x, y - r * 1.6), (x + r * 1.1, y + r * .4)), lw * 1.2)


def _draw_brows(p: Painter, f: Face, brow, k, skin):
    ang, raise_, mode = brow
    if mode == "none":
        return
    L = f.eye_r * 3.0
    base_y = f.eye_y - f.eye_r * 2.2 - raise_ * f.h * (1 + .3 * k)
    dy = math.tan(math.radians(ang * min(1.4, k))) * L / 2           # +: inner end lower (angry)
    for side in (-1, 1):
        cx = f.cx + side * f.eye_dx
        yy = base_y
        if mode == "one_up":
            yy += -f.eye_r * 1.4 if side > 0 else f.eye_r * .5
        inner = (cx - side * L / 2, yy + dy)
        outer = (cx + side * L / 2, yy - dy)
        mid = ((inner[0] + outer[0]) / 2, (inner[1] + outer[1]) / 2 - f.eye_r * .25)
        p.line(_bez(inner, mid, outer, 20), max(2.5, f.eye_r * .42), (48, 30, 24, 255))


def _open_mouth(p: Painter, cx, my, mw, top_c, bot_c, teeth=True, tongue=True, ink_w=2.0):
    """Filled mouth between two bezier curves, with optional teeth band and tongue."""
    L, R = (cx - mw / 2, my), (cx + mw / 2, my)
    top = _bez(L, (cx, my + top_c), R)
    bot = _bez(R, (cx, my + bot_c), L)
    shape = top + bot
    size = p.img.size
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).polygon([p.pt(q) for q in shape], fill=255)
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    ld.polygon([p.pt(q) for q in shape], fill=(70, 12, 24, 255))
    mid_top = my + top_c / 2
    if tongue:
        ld.ellipse(((cx - mw * .28) * S, (my + bot_c / 2 - abs(bot_c) * .28) * S, (cx + mw * .28) * S, (my + bot_c / 2 + abs(bot_c) * .25) * S),
                   fill=(240, 100, 120, 255))
    if teeth:
        ld.polygon([p.pt((q[0], q[1] - 0)) for q in top] + [p.pt((q[0], q[1] + abs(bot_c) * .28 + 3)) for q in reversed(top)],
                   fill=WHITE)
    a = np.minimum(np.array(layer.getchannel("A")), np.array(mask))
    layer.putalpha(Image.fromarray(a))
    p.img.alpha_composite(layer)
    p.line(top + bot[1:], ink_w, INK)


def _draw_mouth(p: Painter, f: Face, kind: str, k):
    cx, my, mw = f.cx, f.mouth_y, f.mouth_w
    h = f.h
    lw = max(2.5, f.eye_r * .38)
    if kind == "smile":
        p.line(_bez((cx - mw / 2, my), (cx, my + .16 * h * k), (cx + mw / 2, my)), lw)
    elif kind == "frown":
        p.line(_bez((cx - mw / 2, my + .04 * h), (cx, my - .10 * h * k), (cx + mw / 2, my + .04 * h)), lw)
    elif kind == "flat":
        p.line([(cx - mw * .38, my), (cx + mw * .38, my + .003 * h)], lw)
    elif kind == "big_smile":
        _open_mouth(p, cx, my - .03 * h, mw * 1.05, 0.0, .30 * h * k, True, True, lw * .7)
    elif kind == "laugh":
        _open_mouth(p, cx, my - .04 * h, mw * 1.2, 0.02 * h, .36 * h * k, True, True, lw * .7)
    elif kind == "wail":
        _open_mouth(p, cx, my + .04 * h, mw * 1.0, -.30 * h * k, .06 * h, False, True, lw * .7)
    elif kind == "o":
        p.ellipse(cx, my + .02 * h, mw * .16 * k ** .3, .075 * h * k ** .3, fill=(70, 12, 24, 255), outline=INK, w=lw * .6)
    elif kind == "big_o":
        p.ellipse(cx, my + .04 * h, mw * .24 * k ** .3, .13 * h * k ** .3, fill=(70, 12, 24, 255), outline=INK, w=lw * .7)
        p.ellipse(cx, my + .09 * h, mw * .13, .05 * h, fill=(230, 100, 120, 255))
    elif kind == "wavy":
        xs = np.linspace(cx - mw * .45, cx + mw * .45, 40)
        p.line([(x, my + math.sin((x - xs[0]) / (xs[-1] - xs[0]) * 3 * math.pi) * .022 * h * k) for x in xs], lw)
    elif kind == "grit":
        x0, x1, y0, y1 = cx - mw * .45, cx + mw * .45, my - .045 * h, my + .05 * h
        p.d.rounded_rectangle(tuple(v * S for v in (x0, y0, x1, y1)), radius=int(.02 * h * S), fill=WHITE, outline=INK, width=int(lw * .8 * S))
        for i in range(1, 5):
            xx = x0 + (x1 - x0) * i / 5
            p.line([(xx, y0), (xx, y1)], lw * .5)
        p.line([(x0, my), (x1, my)], lw * .5)
    elif kind == "smirk":
        p.line(_bez((cx - mw * .42, my + .02 * h), (cx + mw * .05, my + .10 * h * k), (cx + mw * .48, my - .045 * h)), lw)
    elif kind == "tongue":
        p.line(_bez((cx - mw / 2, my), (cx, my + .14 * h * k), (cx + mw / 2, my)), lw)
        p.d.rounded_rectangle(tuple(v * S for v in (cx - mw * .13, my + .035 * h, cx + mw * .17, my + .15 * h * k)),
                              radius=int(mw * .14 * S), fill=(240, 100, 125, 255), outline=INK, width=int(lw * .5 * S))
        p.line([(cx + mw * .02, my + .06 * h), (cx + mw * .02, my + .12 * h * k)], lw * .35, (190, 60, 90, 255))


def _blush(p: Painter, f: Face, amount: float):
    layer = Image.new("RGBA", p.img.size, (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    for side in (-1, 1):
        cx, cy = f.cx + side * f.w * .27, (f.eye_y + f.mouth_y) / 2 + f.h * .02
        ld.ellipse(((cx - f.w * .09) * S, (cy - f.h * .05) * S, (cx + f.w * .09) * S, (cy + f.h * .05) * S),
                   fill=(255, 90, 130, int(190 * min(1, amount))))
    p.img.alpha_composite(layer.filter(ImageFilter.GaussianBlur(float(f.w) * .02 * S)))


def _tears(p: Painter, f: Face, n: int):
    sides = (-1, 1) if n >= 2 else (1,)
    for side in sides:
        x = f.cx + side * f.eye_dx * 1.05
        y0, y1 = f.eye_y + f.eye_r * 1.3, f.mouth_y - f.h * .02
        p.line(_bez((x, y0), (x + side * f.w * .03, (y0 + y1) / 2), (x + side * f.w * .015, y1)), f.eye_r * .55, (110, 185, 255, 235))
        p.ellipse(x + side * f.w * .015, y1 + f.eye_r * .3, f.eye_r * .42, f.eye_r * .55, fill=(110, 185, 255, 245))
        p.ellipse(x - f.eye_r * .1, y0 + f.eye_r * .5, f.eye_r * .13, f.eye_r * .3, fill=(255, 255, 255, 190))


def _draw_shades(p: Painter, f: Face, k):
    r = f.eye_r
    y = f.eye_y + r * .1
    lw, lh = r * 2.7, r * 2.0
    for side in (-1, 1):
        x = f.cx + side * f.eye_dx
        p.d.rounded_rectangle(tuple(v * S for v in (x - lw / 2, y - lh / 2, x + lw / 2, y + lh / 2)),
                              radius=int(r * .8 * S), fill=(18, 18, 28, 255), outline=(0, 0, 0, 255), width=int(r * .3 * S))
        p.line([(x - lw * .3, y - lh * .3), (x - lw * .1, y - lh * .05)], r * .18, (255, 255, 255, 170))
    p.line([(f.cx - f.eye_dx + lw / 2, y - r * .3), (f.cx + f.eye_dx - lw / 2, y - r * .3)], r * .3, (0, 0, 0, 255))


# --------------------------------------------------------------------------- #
# 4. Public API
# --------------------------------------------------------------------------- #
def _skin_color(rgb: np.ndarray, f: Face):
    pts = [(f.cx - f.w * .3, (f.eye_y + f.mouth_y) / 2), (f.cx + f.w * .3, (f.eye_y + f.mouth_y) / 2),
           (f.cx, f.y + f.h * .18), (f.cx, (f.eye_y + f.mouth_y) / 2)]
    H, W = rgb.shape[:2]
    vals = [rgb[int(np.clip(y, 0, H - 1)), int(np.clip(x, 0, W - 1))] for x, y in pts]
    return tuple(int(v) for v in np.median(vals, axis=0))


def _erase_features(rgb: np.ndarray, alpha: np.ndarray, f: Face) -> np.ndarray:
    """Inpaint away the original eyes / brows / mouth so the new ones read cleanly."""
    mask = np.zeros(rgb.shape[:2], np.uint8)
    r = f.eye_r
    for side in (-1, 1):
        x = f.cx + side * f.eye_dx
        cv2.ellipse(mask, (int(x), int(f.eye_y)), (int(r * 1.9), int(r * 1.5)), 0, 0, 360, 255, -1)
        cv2.ellipse(mask, (int(x), int(f.eye_y - r * 2.3)), (int(r * 2.1), int(r * .9)), 0, 0, 360, 255, -1)
    cv2.ellipse(mask, (int(f.cx), int(f.mouth_y + f.h * .01)), (int(f.mouth_w * .62), int(f.h * .085)), 0, 0, 360, 255, -1)
    mask[alpha < 128] = 0
    return cv2.inpaint(np.ascontiguousarray(rgb), mask, 6, cv2.INPAINT_TELEA)


def apply_expression(subject: Image.Image, face: Face, name: str, intensity: float = 1.0) -> Image.Image:
    """Return a copy of `subject` (RGBA) whose face shows the emotion `name`."""
    ex = EXPR.get(name)
    if ex is None:
        return subject
    arr = np.array(subject.convert("RGBA"))
    alpha = arr[..., 3]
    skin = _skin_color(arr[..., :3], face)
    arr[..., :3] = _erase_features(arr[..., :3], alpha, face)

    p = Painter(subject.size)
    k = float(intensity)
    if ex["eyes"] == "shades":
        _draw_shades(p, face, k)
    else:
        for side in (-1, 1):
            _draw_eye(p, ex["eyes"], face.cx + side * face.eye_dx, face.eye_y, face.eye_r * (1.0 + .1 * (k - 1)), side, skin, k)
    _draw_brows(p, face, ex["brow"], k, skin)
    _draw_mouth(p, face, ex["mouth"], k)
    if ex.get("blush"):
        _blush(p, face, ex["blush"] * k)
    if ex.get("tears"):
        _tears(p, face, ex["tears"])
    layer = p.finish(alpha)
    base = Image.fromarray(arr)
    base.alpha_composite(layer)
    return base


def face_change_score(before: Image.Image, after: Image.Image, face: Face) -> float:
    """Mean absolute RGB difference inside the face box (0-255)."""
    l, t, r, b = face.box
    a = np.asarray(before.convert("RGB"))[t:b, l:r].astype(np.int16)
    c = np.asarray(after.convert("RGB"))[t:b, l:r].astype(np.int16)
    return float(np.abs(a - c).mean()) if a.size else 0.0


def expression_pair_distance(a: Image.Image, b: Image.Image, face: Face) -> float:
    """How different two expressions look (used to make sure emotions are distinguishable)."""
    return face_change_score(a, b, face)
