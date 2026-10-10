"""
auto_tester.py - sanity checks + self-healing loop for generated stickers.

Public API
----------
validate_bytes(data)        -> list[str]   problems (empty list == sticker is valid)
validate_image(img, nbytes) -> list[str]
retry_with_backoff(fn)      -> result of fn(), retrying transient failures
heal_index(...)             -> HealResult  runs render(attempt) until output validates
run_selftest()              -> bool        `python auto_tester.py --selftest`

This module deliberately does NOT import sticker_generator at module level
(the generator imports us), so there is no circular dependency.
"""
from __future__ import annotations

import hashlib
import io
import logging
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from PIL import Image, ImageDraw

SIZE = 512
MAX_BYTES = 500 * 1024
MIN_COVERAGE = 0.03      # < 3% opaque pixels  -> blank
MAX_COVERAGE = 0.98      # > 98% opaque pixels -> background not removed
MIN_BRIGHTNESS = 8.0     # mean RGB of non-white content below this -> all-black

log = logging.getLogger("stickers")


class RateLimitError(Exception):
    """Raised by AI backends on HTTP 429 / quota errors."""

    def __init__(self, retry_after: float = 5.0):
        super().__init__(f"rate limited, retry in {retry_after:.1f}s")
        self.retry_after = retry_after


# --------------------------------------------------------------------------- #
# Sanity checks
# --------------------------------------------------------------------------- #
def validate_image(img: Image.Image, size_bytes: Optional[int] = None) -> list[str]:
    """Return a list of problems found in a decoded sticker image."""
    problems: list[str] = []
    if img.size != (SIZE, SIZE):
        problems.append(f"wrong dimensions {img.size}, expected {(SIZE, SIZE)}")
    if size_bytes is not None and size_bytes > MAX_BYTES:
        problems.append(f"file too large: {size_bytes / 1024:.0f} KB > {MAX_BYTES // 1024} KB")
    if "A" not in img.getbands():
        problems.append("no alpha channel")
        return problems

    rgba = np.asarray(img.convert("RGBA"))
    alpha = rgba[..., 3]
    if alpha.min() >= 250:
        problems.append("background is not transparent")

    coverage = float((alpha > 16).mean())
    if coverage < MIN_COVERAGE:
        problems.append(f"blank image (only {coverage:.1%} opaque)")
        return problems
    if coverage > MAX_COVERAGE:
        problems.append(f"subject fills the canvas ({coverage:.1%}); background likely not removed")

    opaque = rgba[alpha > 16][:, :3]
    content = opaque[opaque.min(axis=1) < 235]          # ignore the white border
    if len(content) < 0.01 * SIZE * SIZE:
        problems.append("no visible content inside the border")
    elif content.mean() < MIN_BRIGHTNESS:
        problems.append("all-black output")
    return problems


def validate_bytes(data: Optional[bytes]) -> list[str]:
    """Decode + validate raw WEBP/PNG bytes."""
    if not data:
        return ["empty output"]
    try:
        Image.open(io.BytesIO(data)).verify()
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:  # corrupted / truncated
        return [f"corrupted image: {exc}"]
    return validate_image(img, len(data))


# --------------------------------------------------------------------------- #
# Retry + healing
# --------------------------------------------------------------------------- #
def retry_with_backoff(fn: Callable, retries: int = 3, base: float = 1.5,
                       retry_on=(RateLimitError, OSError, TimeoutError),
                       sleep: Callable[[float], None] = time.sleep):
    """Call fn(); on transient errors wait base**n (or Retry-After) and retry."""
    for n in range(retries + 1):
        try:
            return fn()
        except retry_on as exc:
            if n == retries:
                raise
            wait = getattr(exc, "retry_after", None) or base ** (n + 1)
            log.warning("transient error (%s); retry %d/%d in %.1fs", exc, n + 1, retries, wait)
            sleep(min(wait, 60))


@dataclass
class HealResult:
    ok: bool
    data: Optional[bytes] = None
    attempts: int = 0
    problems: list[str] = field(default_factory=list)


def heal_index(index: int, render: Callable[[int], bytes], seen_hashes: set,
               *, start_attempt: int = 0, max_attempts: int = 5,
               last_resort_attempt: int = 4,
               sleep: Callable[[float], None] = time.sleep) -> HealResult:
    """
    Run render(attempt) with escalating attempt numbers until the output passes
    validate_bytes(). The *renderer* decides what each attempt number means
    (retry -> lower threshold -> other threshold + GrabCut -> minimal sticker);
    this function owns the test / retry / logging policy.

    Exact-duplicate outputs count as failures except on the last-resort attempt.
    """
    problems: list[str] = []
    made = 0
    for attempt in range(start_attempt, start_attempt + max_attempts):
        made += 1
        try:
            data = render(attempt)
        except RateLimitError as exc:
            problems = [f"rate limited: {exc}"]
            log.warning("[#%03d] attempt %d: %s", index, attempt, problems[0])
            sleep(min(exc.retry_after, 30))
            continue
        except Exception as exc:                       # noqa: BLE001 - self-healing on purpose
            problems = [f"exception: {exc!r}"]
            log.warning("[#%03d] attempt %d: %s", index, attempt, problems[0])
            continue

        problems = validate_bytes(data)
        digest = hashlib.sha1(data).hexdigest()
        if digest in seen_hashes and attempt < last_resort_attempt:
            problems.append("exact duplicate of an earlier sticker")
        if not problems:
            seen_hashes.add(digest)
            if made > 1:
                log.info("[#%03d] healed after %d attempts", index, made)
            return HealResult(True, data, made, [])
        log.warning("[#%03d] attempt %d failed sanity check: %s", index, attempt, "; ".join(problems))
    return HealResult(False, None, made, problems)


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _synthetic_photo() -> Image.Image:
    """A colourful 'character' on a busy gradient background."""
    w = h = 640
    xs = np.linspace(0, 255, w, dtype=np.uint8)
    bg = np.stack([np.tile(xs, (h, 1)), np.tile(xs[::-1], (h, 1)), np.full((h, w), 90, np.uint8)], -1)
    img = Image.fromarray(bg)
    d = ImageDraw.Draw(img)
    d.ellipse((170, 130, 470, 500), fill=(250, 200, 60), outline=(120, 60, 0), width=6)
    d.ellipse((240, 250, 285, 300), fill=(20, 20, 20))
    d.ellipse((355, 250, 400, 300), fill=(20, 20, 20))
    d.arc((240, 320, 400, 430), 20, 160, fill=(150, 0, 0), width=8)
    return img


def run_selftest() -> bool:
    import sticker_generator as sg
    failures = 0

    def check(name: str, cond: bool):
        nonlocal failures
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        failures += (not cond)

    def enc(img, fmt="WEBP"):
        b = io.BytesIO()
        img.save(b, fmt)
        return b.getvalue()

    print("Validator unit tests")
    good = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    ImageDraw.Draw(good).ellipse((100, 100, 400, 400), fill=(200, 60, 60, 255),
                                 outline=(255, 255, 255, 255), width=12)
    check("valid sticker passes", validate_bytes(enc(good)) == [])
    check("wrong size caught", any("dimensions" in p for p in validate_bytes(enc(good.resize((256, 256))))))
    check("missing alpha caught", any("alpha" in p for p in validate_bytes(enc(good.convert("RGB")))))
    check("opaque background caught",
          any(("transparent" in p or "alpha" in p) for p in validate_bytes(enc(Image.new("RGBA", (SIZE, SIZE), (9, 9, 9, 255))))))
    check("blank image caught",
          any("blank" in p for p in validate_bytes(enc(Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))))))
    black = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    ImageDraw.Draw(black).ellipse((100, 100, 400, 400), fill=(0, 0, 0, 255))
    check("all-black caught", any("black" in p for p in validate_bytes(enc(black))))
    check("corrupted data caught", any("corrupted" in p for p in validate_bytes(b"RIFFxxxxWEBPgarbage")))
    big = Image.fromarray(np.random.randint(0, 255, (SIZE, SIZE, 4), dtype=np.uint8))
    check("oversize file caught", any("too large" in p for p in validate_bytes(enc(big, "PNG"))))

    print("Healing loop tests")

    def flaky(attempt):
        if attempt == 0:
            raise RuntimeError("boom")
        if attempt == 1:
            return b"not an image"
        return enc(good)

    res = heal_index(0, flaky, set(), sleep=lambda s: None)
    check("heals after exception + corrupted output", res.ok and res.attempts == 3)

    def limited(attempt):
        if attempt < 2:
            raise RateLimitError(0.01)
        return enc(good)

    check("survives rate limits", heal_index(1, limited, set(), sleep=lambda s: None).ok)
    check("gives up cleanly when hopeless",
          not heal_index(2, lambda a: b"x", set(), max_attempts=2, sleep=lambda s: None).ok)

    print("End-to-end pipeline (first 8 catalog entries, local mode)")
    with tempfile.TemporaryDirectory() as tmp:
        results = sg.generate_pack(_synthetic_photo(), tmp, count=8, use_ai=False)
        check("all 8 stickers generated", len(results) == 8 and all(r.ok for r in results))
        check("all files re-validate from disk",
              all(validate_bytes(open(r.path, "rb").read()) == [] for r in results if r.ok))
    cat = sg.build_catalog()
    check("catalog has 100+ unique entries", len(cat) >= 100 and len({(s.category, s.name) for s in cat}) == len(cat))
    print("\nSELF-TEST", "PASSED" if not failures else f"FAILED ({failures})")
    return failures == 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    if "--selftest" in sys.argv:
        sys.exit(0 if run_selftest() else 1)
    print(__doc__)
