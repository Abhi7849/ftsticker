"""
exporter.py - packages a generated run directory into shareable formats.

  export_zip       -> <title>.zip            (stickers/*.webp + png/*.png (iMessage/Xcode) + manifest)
  export_whatsapp  -> <title>_N.wastickers   (30 stickers per pack, WhatsApp's hard limit)
                      + contents_N.json      (official WhatsApp sample-app format)
  export_telegram  -> telegram/ folder       (telegram_payload.json + upload_to_telegram.py + stickers)

Platform limits that shape this module
  WhatsApp : static sticker WEBP 512x512, <= 100 KB each, tray icon 96x96 PNG <= 50 KB, <= 30 stickers/pack
  Telegram : static sticker WEBP/PNG 512px on one side, <= 512 KB, <= 120 stickers per set
So WhatsApp copies are re-encoded from the master files to fit 100 KB.

NOTE: `.wastickers` is the community archive format read by apps such as "Sticker Maker";
WhatsApp itself has no file-import path. Test in your target app.
"""
from __future__ import annotations

import io
import json
import re
import shutil
import zipfile
from pathlib import Path

from PIL import Image

WA_MAX_PACK = 30
WA_MAX_BYTES = 100 * 1024
TG_MAX_SET = 120


def load_manifest(out_dir) -> list[dict]:
    items = json.loads((Path(out_dir) / "manifest.json").read_text(encoding="utf-8"))
    return [m for m in items if m["ok"] and Path(m["path"]).exists()]


def _slug(title: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", title).strip("_").lower()
    return s if s and s[0].isalpha() else f"pack_{s}"


# --------------------------------------------------------------------------- #
def export_zip(out_dir, title: str = "stickers") -> Path:
    out_dir = Path(out_dir)
    items = load_manifest(out_dir)
    dest = out_dir / f"{_slug(title)}.zip"
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for m in items:
            p = Path(m["path"])
            z.write(p, f"stickers/{p.name}")
            png = io.BytesIO()
            Image.open(p).convert("RGBA").save(png, "PNG", optimize=True)
            z.writestr(f"png/{p.stem}.png", png.getvalue())
        z.write(out_dir / "manifest.json", "manifest.json")
    return dest


# --------------------------------------------------------------------------- #
def _wa_webp(path: Path) -> bytes:
    """Re-encode to <= 100 KB, keeping 512x512 and transparency."""
    img = Image.open(path).convert("RGBA")
    data = b""
    for q in (85, 75, 65, 55, 45, 35, 25, 15, 8):
        buf = io.BytesIO()
        img.save(buf, "WEBP", quality=q, method=4)
        data = buf.getvalue()
        if len(data) <= WA_MAX_BYTES:
            return data
    return data


def _tray_png(path: Path) -> bytes:
    buf = io.BytesIO()
    Image.open(path).convert("RGBA").resize((96, 96), Image.LANCZOS).save(buf, "PNG", optimize=True)
    return buf.getvalue()


def export_whatsapp(out_dir, title: str = "My Stickers", author: str = "Sticker Studio") -> list[Path]:
    out_dir = Path(out_dir)
    items = load_manifest(out_dir)
    wa_dir = out_dir / "whatsapp"
    wa_dir.mkdir(exist_ok=True)
    chunks = [items[i:i + WA_MAX_PACK] for i in range(0, len(items), WA_MAX_PACK)]
    files: list[Path] = []
    for n, chunk in enumerate(chunks, 1):
        pack_title = f"{title} {n}/{len(chunks)}" if len(chunks) > 1 else title
        dest = wa_dir / f"{_slug(title)}_{n}.wastickers"
        contents = {
            "android_play_store_link": "", "ios_app_store_link": "",
            "sticker_packs": [{
                "identifier": f"{_slug(title)}_{n}", "name": pack_title, "publisher": author,
                "tray_image_file": "tray.png", "image_data_version": "1", "avoid_cache": False,
                "publisher_email": "", "publisher_website": "", "privacy_policy_website": "",
                "license_agreement_website": "",
                "stickers": [{"image_file": Path(m["path"]).name, "emojis": [m["emoji"]]} for m in chunk],
            }]}
        with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("title.txt", pack_title)
            z.writestr("author.txt", author)
            z.writestr("tray.png", _tray_png(Path(chunk[0]["path"])))
            for m in chunk:
                z.writestr(Path(m["path"]).name, _wa_webp(Path(m["path"])))
        (wa_dir / f"contents_{n}.json").write_text(json.dumps(contents, indent=2, ensure_ascii=False), encoding="utf-8")
        files.append(dest)
    return files


# --------------------------------------------------------------------------- #
TELEGRAM_SCRIPT = r'''#!/usr/bin/env python3
"""
Uploads telegram_payload.json as Telegram sticker set(s) via the Bot API.

    pip install requests
    python upload_to_telegram.py --token 123456:ABC... --user-id 111222333

Create a bot with @BotFather, then message it once. --user-id is YOUR numeric
Telegram id (e.g. from @userinfobot); the sets will be owned by that user.
Set names must end in _by_<botusername>; that suffix is added automatically.
"""
import argparse, json, sys, time
from pathlib import Path
import requests

HERE = Path(__file__).parent
CREATE_BATCH = 50   # Telegram accepts at most 50 stickers in createNewStickerSet


def call(api, method, data, paths=()):
    for _ in range(8):
        files = {f"s{i}": open(p, "rb") for i, p in enumerate(paths)}
        try:
            r = requests.post(f"{api}/{method}", data=data, files=files or None, timeout=90)
        finally:
            for f in files.values():
                f.close()
        j = r.json()
        if j.get("ok"):
            return j["result"]
        if j.get("error_code") == 429:
            wait = j.get("parameters", {}).get("retry_after", 5) + 1
            print(f"  rate limited, sleeping {wait}s"); time.sleep(wait); continue
        sys.exit(f"{method} failed: {j}")
    sys.exit(f"{method}: too many retries")


def item(i, st):
    return {"sticker": f"attach://s{i}", "format": "static", "emoji_list": st["emoji_list"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", required=True)
    ap.add_argument("--user-id", type=int, required=True)
    a = ap.parse_args()
    api = f"https://api.telegram.org/bot{a.token}"
    bot = call(api, "getMe", {})["username"]
    payload = json.loads((HERE / "telegram_payload.json").read_text(encoding="utf-8"))
    for s in payload["sets"]:
        name = f'{s["short_name"]}_by_{bot}'
        files = [HERE / st["file"] for st in s["stickers"]]
        first = s["stickers"][:CREATE_BATCH]
        print(f"Creating {name} ({len(s['stickers'])} stickers)")
        call(api, "createNewStickerSet",
             {"user_id": a.user_id, "name": name, "title": s["title"],
              "stickers": json.dumps([item(i, st) for i, st in enumerate(first)])}, files[:len(first)])
        for k, st in enumerate(s["stickers"][CREATE_BATCH:], CREATE_BATCH):
            call(api, "addStickerToSet",
                 {"user_id": a.user_id, "name": name, "sticker": json.dumps(item(0, st))}, [files[k]])
            print(f"  added {k + 1}/{len(s['stickers'])}"); time.sleep(0.4)
        print(f"  -> https://t.me/addstickers/{name}")


if __name__ == "__main__":
    main()
'''


def export_telegram(out_dir, title: str = "My Stickers") -> Path:
    out_dir = Path(out_dir)
    items = load_manifest(out_dir)
    tg = out_dir / "telegram"
    (tg / "stickers").mkdir(parents=True, exist_ok=True)
    sets = []
    chunks = [items[i:i + TG_MAX_SET] for i in range(0, len(items), TG_MAX_SET)]
    for n, chunk in enumerate(chunks, 1):
        entries = []
        for m in chunk:
            src = Path(m["path"])
            shutil.copy(src, tg / "stickers" / src.name)
            entries.append({"file": f"stickers/{src.name}", "emoji_list": [m["emoji"]], "name": m["name"]})
        sets.append({"short_name": f"{_slug(title)}_{n}",
                     "title": f"{title} {n}" if len(chunks) > 1 else title, "stickers": entries})
    (tg / "telegram_payload.json").write_text(
        json.dumps({"sticker_format": "static", "sets": sets}, indent=2, ensure_ascii=False), encoding="utf-8")
    (tg / "upload_to_telegram.py").write_text(TELEGRAM_SCRIPT, encoding="utf-8")
    return tg


def export_telegram_zip(out_dir, title: str = "My Stickers") -> Path:
    tg = export_telegram(out_dir, title)
    dest = Path(out_dir) / f"{_slug(title)}_telegram.zip"
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for p in tg.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(tg))
    return dest


def export_all(out_dir, title: str = "My Stickers", author: str = "Sticker Studio") -> dict:
    return {"zip": export_zip(out_dir, title),
            "whatsapp": export_whatsapp(out_dir, title, author),
            "telegram": export_telegram_zip(out_dir, title)}
