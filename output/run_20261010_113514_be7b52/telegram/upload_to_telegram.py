#!/usr/bin/env python3
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
