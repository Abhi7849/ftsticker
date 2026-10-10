"""
face_filters.py - Snapchat-style funny face filters, pure PIL + OpenCV.

    face = fe.find_face(subject_rgba)
    out  = apply_filter(subject_rgba, face, "dog")

Every filter is drawn 4x supersampled on a transparent layer, anchored to the
face geometry (eye line, eye separation, mouth position, face box). Filters
that stick out past the subject (ears, crowns, hats) grow the canvas, so the
result is always an RGBA image that can go straight into compose().

Warp filters (big_eyes, big_head, fat_face, pinch) distort the real pixels.
"""
from __future__ import annotations

import math
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import face_expressions as fe

S = 4
INK = (38, 24, 24, 255)
WHITE = (255, 255, 255, 255)
MIN_FILTER_CHANGE = 0.004      # fraction of visibly changed pixels required (else healing loop retries)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
class _Layer:
    """Supersampled drawing layer with an origin offset (so we can draw outside the subject)."""

    def __init__(self, w, h, pad):
        self.pad = pad
        self.w, self.h = w + 2 * pad, h + 2 * pad
        self.img = Image.new("RGBA", (self.w * S, self.h * S), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.img)

    def P(self, x, y):
        return ((x + self.pad) * S, (y + self.pad) * S)

    def poly(self, pts, fill, outline=None, w=0):
        pp = [self.P(*p) for p in pts]
        self.d.polygon(pp, fill=fill)
        if outline and w:
            self.d.line(pp + [pp[0]], fill=outline, width=int(w * S), joint="curve")

    def ell(self, cx, cy, rx, ry, fill=None, outline=None, w=0):
        x0, y0 = self.P(cx - rx, cy - ry)
        x1, y1 = self.P(cx + rx, cy + ry)
        self.d.ellipse((x0, y0, x1, y1), fill=fill, outline=outline, width=int(w * S))

    def line(self, pts, w, color=INK):
        pp = [self.P(*p) for p in pts]
        ww = max(2, int(w * S))
        self.d.line(pp, fill=color, width=ww, joint="curve")
        for p in (pp[0], pp[-1]):
            self.d.ellipse((p[0] - ww / 2, p[1] - ww / 2, p[0] + ww / 2, p[1] + ww / 2), fill=color)

    def rrect(self, x0, y0, x1, y1, r, fill=None, outline=None, w=0):
        a, b = self.P(x0, y0), self.P(x1, y1)
        self.d.rounded_rectangle((a[0], a[1], b[0], b[1]), radius=int(r * S), fill=fill,
                                 outline=outline, width=int(w * S))

    def result(self) -> Image.Image:
        return self.img.resize((self.w, self.h), Image.LANCZOS)


def _bez(p0, p1, p2, n=30):
    return [((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0],
             (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1]) for t in np.linspace(0, 1, n)]


def _star(cx, cy, r):
    return [(cx + r * (1 if k % 2 == 0 else .45) * math.cos(-math.pi / 2 + k * math.pi / 5),
             cy + r * (1 if k % 2 == 0 else .45) * math.sin(-math.pi / 2 + k * math.pi / 5)) for k in range(10)]


def _heart(cx, cy, s):
    pts = []
    for t in np.linspace(0, 2 * math.pi, 50):
        pts.append((cx + s * 0.062 * 16 * math.sin(t) ** 3,
                    cy - s * 0.062 * (13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))))
    return pts


def _flower(L: _Layer, x, y, r, petal, centre=(255, 210, 40, 255)):
    for k in range(6):
        a = k * math.pi / 3
        L.ell(x + r * .75 * math.cos(a), y + r * .75 * math.sin(a), r * .55, r * .55, fill=petal, outline=INK, w=1.2)
    L.ell(x, y, r * .45, r * .45, fill=centre, outline=INK, w=1.2)


# --------------------------------------------------------------------------- #
# filters: each fn(L, f) draws on a layer. `under` filters are drawn BEHIND the subject.
# --------------------------------------------------------------------------- #
def _f_dog(L, f):
    ex, ey, ew = f.cx, f.y, f.w
    for side in (-1, 1):                                    # floppy ears
        cx = ex + side * ew * .40
        L.ell(cx + side * ew * .05, ey + ew * .02, ew * .14, ew * .30, fill=(120, 76, 40, 255), outline=INK, w=2.5)
        L.ell(cx + side * ew * .05, ey + ew * .06, ew * .08, ew * .20, fill=(176, 120, 78, 255))
    # snout / nose
    ny = f.mouth_y - f.h * .17
    L.ell(f.cx, ny, f.w * .09, f.w * .065, fill=(30, 20, 20, 255), outline=INK, w=2)
    L.ell(f.cx - f.w * .03, ny - f.w * .02, f.w * .025, f.w * .015, fill=(255, 255, 255, 190))
    L.line([(f.cx, ny + f.w * .06), (f.cx, f.mouth_y - f.h * .03)], 2.5)
    # tongue
    L.rrect(f.cx - f.w * .06, f.mouth_y + f.h * .01, f.cx + f.w * .06, f.mouth_y + f.h * .16, f.w * .06,
            fill=(240, 100, 125, 255), outline=INK, w=2)
    L.line([(f.cx, f.mouth_y + f.h * .03), (f.cx, f.mouth_y + f.h * .13)], 1.6, (190, 60, 90, 255))


def _f_cat(L, f):
    for side in (-1, 1):
        bx = f.cx + side * f.w * .30
        pts = [(bx - f.w * .15, f.y + f.h * .10), (bx + side * f.w * .06, f.y - f.h * .20), (bx + f.w * .15, f.y + f.h * .10)]
        L.poly(pts, (90, 90, 100, 255), INK, 2.5)
        L.poly([(bx - f.w * .08, f.y + f.h * .08), (bx + side * f.w * .05, f.y - f.h * .10), (bx + f.w * .08, f.y + f.h * .08)],
               (255, 160, 190, 255))
    ny = f.mouth_y - f.h * .13
    L.poly([(f.cx - f.w * .05, ny), (f.cx + f.w * .05, ny), (f.cx, ny + f.w * .045)], (255, 120, 150, 255), INK, 1.8)
    for side in (-1, 1):                                    # whiskers
        for dy in (-.025, .0, .025):
            L.line([(f.cx + side * f.w * .12, ny + f.h * dy + 4),
                    (f.cx + side * f.w * .42, ny + f.h * (dy * 3.2) + 4)], 1.8, (40, 40, 40, 255))


def _f_bunny(L, f):
    for side in (-1, 1):
        cx = f.cx + side * f.w * .17
        L.ell(cx + side * f.w * .02, f.y - f.h * .12, f.w * .09, f.h * .30, fill=(250, 245, 245, 255), outline=INK, w=2.5)
        L.ell(cx + side * f.w * .02, f.y - f.h * .10, f.w * .045, f.h * .22, fill=(255, 175, 195, 255))
    ny = f.mouth_y - f.h * .12
    L.ell(f.cx, ny, f.w * .035, f.w * .027, fill=(255, 130, 160, 255), outline=INK, w=1.5)
    L.rrect(f.cx - f.w * .05, f.mouth_y + f.h * .005, f.cx + f.w * .005, f.mouth_y + f.h * .09, 3,
            fill=WHITE, outline=INK, w=1.8)
    L.rrect(f.cx, f.mouth_y + f.h * .005, f.cx + f.w * .055, f.mouth_y + f.h * .09, 3, fill=WHITE, outline=INK, w=1.8)


def _f_flower_crown(L, f):
    n = 7
    cols = [(255, 120, 160, 255), (255, 255, 255, 255), (255, 170, 60, 255), (190, 130, 255, 255)]
    cy0 = f.y + f.h * .03
    for i in range(n):
        t = i / (n - 1)
        x = f.x + f.w * (.06 + .88 * t)
        y = cy0 - math.sin(t * math.pi) * f.h * .07 + (1 - math.sin(t * math.pi)) * f.h * .10
        L.ell(x, y + 3, f.w * .05, f.w * .03, fill=(60, 160, 70, 255))
        _flower(L, x, y, f.w * (.06 + .025 * math.sin(t * math.pi)), cols[i % len(cols)])


def _f_glasses_funny(L, f):                                 # big groucho-style: glasses + nose + moustache
    r = f.eye_r * 2.0
    y = f.eye_y
    for side in (-1, 1):
        x = f.cx + side * f.eye_dx
        L.ell(x, y, r * 1.25, r * 1.25, fill=(255, 255, 255, 140), outline=(20, 20, 20, 255), w=f.eye_r * .45)
    L.line([(f.cx - f.eye_dx + r * 1.25, y - r * .1), (f.cx + f.eye_dx - r * 1.25, y - r * .1)], f.eye_r * .4, (20, 20, 20, 255))
    ny = f.mouth_y - f.h * .12                                # fake bulb nose
    L.ell(f.cx, ny, f.w * .075, f.w * .07, fill=(235, 90, 80, 255), outline=INK, w=2)
    L.ell(f.cx - f.w * .02, ny - f.w * .02, f.w * .02, f.w * .016, fill=(255, 255, 255, 170))
    my = f.mouth_y - f.h * .045                               # moustache
    for side in (-1, 1):
        L.poly(_bez((f.cx, my), (f.cx + side * f.w * .15, my - f.h * .07), (f.cx + side * f.w * .30, my + f.h * .03), 14)
               + _bez((f.cx + side * f.w * .30, my + f.h * .03), (f.cx + side * f.w * .15, my + f.h * .02), (f.cx, my + f.h * .04), 14),
               (60, 38, 24, 255), INK, 1.5)


def _f_shades_party(L, f):                                  # heart-shaped sunglasses
    for side in (-1, 1):
        x = f.cx + side * f.eye_dx
        L.poly(_heart(x, f.eye_y + f.eye_r * 1.2, f.eye_r * 3.6), (235, 40, 90, 255), INK, 2.5)
        L.line([(x - f.eye_r * 1.5, f.eye_y - f.eye_r * .4), (x - f.eye_r * .5, f.eye_y)], f.eye_r * .3, (255, 255, 255, 190))
    L.line([(f.cx - f.eye_dx + f.eye_r * 1.6, f.eye_y - f.eye_r * .2), (f.cx + f.eye_dx - f.eye_r * 1.6, f.eye_y - f.eye_r * .2)],
           f.eye_r * .35, INK)


def _f_crown(L, f):
    y1, y0 = f.y + f.h * .06, f.y - f.h * .22
    x0, x1 = f.cx - f.w * .30, f.cx + f.w * .30
    pts = [(x0, y1), (x0, y0 + f.h * .06), (x0 + f.w * .15, y0 + f.h * .13), (f.cx, y0),
           (x1 - f.w * .15, y0 + f.h * .13), (x1, y0 + f.h * .06), (x1, y1)]
    L.poly(pts, (255, 206, 40, 255), INK, 3)
    for px, py, c in ((x0, y0 + f.h * .06, (235, 50, 70, 255)), (f.cx, y0, (60, 120, 240, 255)), (x1, y0 + f.h * .06, (235, 50, 70, 255))):
        L.ell(px, py, f.w * .028, f.w * .028, fill=c, outline=INK, w=1.5)
    L.rrect(x0, y1 - f.h * .045, x1, y1, 3, fill=(240, 175, 20, 255), outline=INK, w=2)


def _f_party_hat(L, f):
    base = f.y + f.h * .04
    pts = [(f.cx - f.w * .24, base), (f.cx + f.w * .03, f.y - f.h * .45), (f.cx + f.w * .24, base)]
    L.poly(pts, (255, 90, 170, 255), INK, 3)
    for k in range(4):
        t = (k + 1) / 5
        L.ell(f.cx - f.w * .18 + f.w * .36 * t * .8 + f.w * .02 * k, base - f.h * .08 - k * f.h * .09,
              f.w * .022, f.w * .022, fill=(255, 230, 60, 255))
    L.ell(f.cx + f.w * .03, f.y - f.h * .45, f.w * .05, f.w * .05, fill=(80, 200, 255, 255), outline=INK, w=2)


def _f_halo_devil(L, f):                                    # halo + devil horns combo (angel/devil)
    L.ell(f.cx, f.y - f.h * .12, f.w * .24, f.h * .06, outline=(255, 215, 40, 255), w=f.w * .035)
    for side in (-1, 1):
        bx = f.cx + side * f.w * .36
        L.poly([(bx - f.w * .07, f.y + f.h * .08), (bx + side * f.w * .07, f.y - f.h * .20), (bx + f.w * .07, f.y + f.h * .05)],
               (220, 30, 40, 255), INK, 2.5)


def _f_clown(L, f):
    L.ell(f.cx, f.mouth_y - f.h * .12, f.w * .085, f.w * .085, fill=(235, 40, 50, 255), outline=INK, w=2)
    L.ell(f.cx - f.w * .03, f.mouth_y - f.h * .15, f.w * .022, f.w * .018, fill=(255, 255, 255, 190))
    for side in (-1, 1):                                    # rainbow wig puffs
        for i, c in enumerate(((255, 90, 40, 255), (255, 210, 40, 255), (70, 190, 90, 255))):
            L.ell(f.cx + side * f.w * (.44 - .01 * i), f.y + f.h * (.18 + .16 * i), f.w * .13, f.w * .13, fill=c, outline=INK, w=2)
        L.ell(f.cx + side * f.w * .30, f.eye_y + f.eye_r * 2.4, f.w * .07, f.w * .05, fill=(255, 60, 120, 150))


def _f_pirate(L, f):
    x = f.cx + f.eye_dx
    L.line([(f.x + f.w * .02, f.y + f.h * .12), (f.x + f.w * .98, f.y + f.h * .30)], f.eye_r * .35, (30, 30, 30, 255))
    L.ell(x, f.eye_y, f.eye_r * 2.1, f.eye_r * 1.8, fill=(25, 25, 25, 255), outline=INK, w=2)
    L.poly([(f.x + f.w * .02, f.y + f.h * .16), (f.x + f.w * .98, f.y + f.h * .16), (f.x + f.w * .90, f.y - f.h * .04), (f.x + f.w * .10, f.y - f.h * .04)],
           (200, 30, 40, 255), INK, 2.5)
    for i in range(5):
        L.ell(f.x + f.w * (.18 + .16 * i), f.y + f.h * .06, f.w * .02, f.w * .02, fill=WHITE)


def _f_rainbow_vomit(L, f):
    cols = [(255, 60, 60, 255), (255, 160, 40, 255), (255, 235, 60, 255), (80, 200, 90, 255), (70, 140, 255, 255), (150, 90, 255, 255)]
    x0, y0 = f.cx, f.mouth_y + f.h * .02
    for i, c in enumerate(cols):
        off = (i - 2.5) * f.w * .035
        pts = _bez((x0 + off * .3, y0), (x0 + off + f.w * .02, y0 + f.h * .35), (x0 + off * 1.8, y0 + f.h * .75), 30)
        L.line(pts, f.w * .045, c)
    L.ell(f.cx, f.mouth_y + f.h * .03, f.w * .13, f.h * .06, fill=(70, 12, 24, 255), outline=INK, w=2)


def _f_tears_fountain(L, f):
    for side in (-1, 1):
        x0, y0 = f.cx + side * f.eye_dx, f.eye_y + f.eye_r * 1.2
        for off in (0, .5, 1):
            pts = _bez((x0, y0), (x0 + side * f.w * (.20 + .08 * off), y0 - f.h * .05), (x0 + side * f.w * (.38 + .1 * off), y0 + f.h * .45), 30)
            L.line(pts, f.eye_r * .55, (100, 185, 255, 230))
        L.ell(x0 + side * f.w * .02, y0, f.eye_r * .9, f.eye_r * .6, fill=(110, 190, 255, 230))


def _f_hearts_orbit(L, f):
    rng = np.random.default_rng(3)
    for i in range(7):
        a = -math.pi * (.1 + .8 * i / 6)
        r = f.w * (.62 + .08 * (i % 2))
        x, y = f.cx + r * math.cos(a), f.y + f.h * .35 + r * .85 * math.sin(a)
        L.poly(_heart(x, y, f.w * (.07 + .02 * (i % 3))), (240, 50, 100, 255), WHITE, 3)


def _f_star_eyes_sparkle(L, f):
    for i, (dx, dy, r) in enumerate(((-.50, -.05, .09), (.52, .10, .07), (-.42, .55, .06), (.50, .62, .08), (0, -.18, .07))):
        L.poly(_star(f.cx + dx * f.w, f.y + f.h * .35 + dy * f.h, f.w * r), (255, 205, 40, 255), INK, 2)


def _f_mustache_only(L, f):
    my = f.mouth_y - f.h * .06
    for side in (-1, 1):
        pts = _bez((f.cx, my), (f.cx + side * f.w * .20, my - f.h * .10), (f.cx + side * f.w * .36, my + f.h * .04), 16)
        pts2 = _bez((f.cx + side * f.w * .36, my + f.h * .04), (f.cx + side * f.w * .18, my + f.h * .035), (f.cx, my + f.h * .05), 16)
        L.poly(pts + pts2, (50, 30, 20, 255), INK, 1.5)


def _f_unicorn(L, f):
    hx, hy = f.cx, f.y + f.h * .02
    L.poly([(hx - f.w * .07, hy), (hx + f.w * .005, hy - f.h * .45), (hx + f.w * .07, hy)], (255, 225, 90, 255), INK, 2.5)
    for k in (.2, .4, .6):
        L.line([(hx - f.w * .06 * (1 - k), hy - f.h * .45 * k), (hx + f.w * .06 * (1 - k), hy - f.h * .45 * k + 6)], 2.2, (230, 160, 30, 255))
    for side in (-1, 1):
        L.poly([(hx + side * f.w * .16, hy + f.h * .05), (hx + side * f.w * .22, hy - f.h * .18), (hx + side * f.w * .29, hy + f.h * .06)],
               (255, 190, 220, 255), INK, 2.5)


def _f_alien(L, f):
    for side in (-1, 1):
        x = f.cx + side * f.eye_dx
        L.ell(x, f.eye_y + f.eye_r * .3, f.eye_r * 2.1, f.eye_r * 2.7, fill=(10, 10, 14, 255), outline=(0, 120, 60, 255), w=2.5)
        L.ell(x - f.eye_r * .6, f.eye_y - f.eye_r * .8, f.eye_r * .5, f.eye_r * .7, fill=(255, 255, 255, 190))
    for side in (-1, 1):
        L.line([(f.cx + side * f.w * .12, f.y), (f.cx + side * f.w * .22, f.y - f.h * .30)], 4, (60, 160, 80, 255))
        L.ell(f.cx + side * f.w * .22, f.y - f.h * .32, f.w * .035, f.w * .035, fill=(120, 230, 120, 255), outline=INK, w=2)


FILTERS: dict[str, dict] = {
    "dog":          dict(fn=_f_dog, pad=.45, tint=(255, 200, 120)),
    "cat":          dict(fn=_f_cat, pad=.30),
    "bunny":        dict(fn=_f_bunny, pad=.45),
    "flower_crown": dict(fn=_f_flower_crown, pad=.20),
    "funny_glasses": dict(fn=_f_glasses_funny, pad=.05),
    "heart_shades": dict(fn=_f_shades_party, pad=.05),
    "king_crown":   dict(fn=_f_crown, pad=.30),
    "party_hat":    dict(fn=_f_party_hat, pad=.45),
    "angel_devil":  dict(fn=_f_halo_devil, pad=.30),
    "clown":        dict(fn=_f_clown, pad=.25),
    "pirate":       dict(fn=_f_pirate, pad=.25),
    "rainbow_barf": dict(fn=_f_rainbow_vomit, pad=.55),
    "tear_fountain": dict(fn=_f_tears_fountain, pad=.25),
    "hearts_orbit": dict(fn=_f_hearts_orbit, pad=.45),
    "sparkle":      dict(fn=_f_star_eyes_sparkle, pad=.40),
    "moustache":    dict(fn=_f_mustache_only, pad=.10),
    "unicorn":      dict(fn=_f_unicorn, pad=.50),
    "alien":        dict(fn=_f_alien, pad=.40),
}
# warps (distort real pixels)
WARPS = ("big_eyes", "big_head", "fat_face", "pinch", "long_face", "tiny_face")


# --------------------------------------------------------------------------- #
# warps
# --------------------------------------------------------------------------- #
def _bulge(rgba: np.ndarray, cx, cy, radius, strength) -> np.ndarray:
    """strength>0 magnify (bulge), <0 shrink (pinch). Smooth falloff, works on all 4 channels."""
    h, w = rgba.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xx - np.float32(cx), yy - np.float32(cy)
    d = np.sqrt(dx * dx + dy * dy) / max(1.0, radius)
    inside = d < 1
    f = np.ones_like(d)
    f[inside] = 1 - strength * (1 - d[inside] ** 2)
    f = np.clip(f, 0.15, 3.0)
    mx = (cx + dx * f).astype(np.float32)
    my = (cy + dy * f).astype(np.float32)
    return cv2.remap(rgba, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def _apply_warp(subject: Image.Image, f: fe.Face, name: str) -> Image.Image:
    arr = np.array(subject.convert("RGBA"))
    pad = int(max(arr.shape[:2]) * .25)                      # room for bulges to grow into
    arr = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)))
    f = _shift_face(f, pad, pad)
    if name == "big_eyes":
        for side in (-1, 1):
            arr = _bulge(arr, f.cx + side * f.eye_dx, f.eye_y, f.eye_r * 4.2, .75)
    elif name == "big_head":
        arr = _bulge(arr, f.cx, f.y + f.h * .5, max(f.w, f.h) * .85, .55)
    elif name == "fat_face":
        arr = _bulge(arr, f.cx, f.y + f.h * .6, f.w * .85, .65)
    elif name == "pinch":
        arr = _bulge(arr, f.cx, f.y + f.h * .5, f.w * .8, -.9)
    elif name == "long_face":
        h, w = arr.shape[:2]
        y0 = int(f.y)
        top = arr[:y0]
        face = cv2.resize(arr[y0:], (w, int((h - y0) * 1.45)), interpolation=cv2.INTER_LINEAR)
        arr = np.concatenate([top, face], axis=0)
    elif name == "tiny_face":
        arr = _bulge(arr, f.cx, f.y + f.h * .45, f.w * .55, -.85)
        arr = _bulge(arr, f.cx, f.mouth_y, f.w * .3, 1.0)       # big mouth
    return Image.fromarray(arr)


def _shift_face(f: fe.Face, dx: float, dy: float) -> fe.Face:
    from dataclasses import replace
    return replace(f, x=f.x + dx, y=f.y + dy, eye_y=f.eye_y + dy, mouth_y=f.mouth_y + dy)


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
ALL_FILTERS = tuple(FILTERS) + WARPS


def apply_filter(subject: Image.Image, face: fe.Face, name: str) -> Image.Image:
    """Return RGBA subject with the funny filter `name` applied (canvas may grow)."""
    subject = subject.convert("RGBA")
    if name in WARPS:
        return _apply_warp(subject, face, name)
    spec = FILTERS[name]
    W, H = subject.size
    pad = int(max(W, H) * spec.get("pad", .3))
    L = _Layer(W, H, pad)
    spec["fn"](L, face)
    layer = L.result()
    canvas = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    base = subject
    if spec.get("tint"):
        arr = np.array(base).astype(np.float32)
        arr[..., :3] = arr[..., :3] * .88 + np.array(spec["tint"], np.float32) * .12
        base = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    canvas.alpha_composite(base, (pad, pad))
    canvas.alpha_composite(layer)
    return canvas


def filter_change_score(before: Image.Image, after: Image.Image) -> float:
    """Fraction of pixels that differ visibly after centre-aligning sizes (detects no-op filters)."""
    a = np.asarray(before.convert("RGBA").resize((128, 128)), dtype=np.int16)
    b = np.asarray(after.convert("RGBA").resize((128, 128)), dtype=np.int16)
    return float((np.abs(a - b).max(axis=2) > 25).mean())
