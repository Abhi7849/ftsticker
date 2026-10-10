# Sticker Pack Studio

One image in -> 137 stickers out -> ZIP / WhatsApp `.wastickers` / Telegram bundle.

    pip install -r requirements.txt
    streamlit run app.py                          # web UI
    python sticker_generator.py photo.png --out output/run1   # CLI
    python auto_tester.py --selftest              # validator + healing-loop tests

| File | Role |
|---|---|
| `app.py` | Streamlit UI (upload, progress, preview, downloads) |
| `sticker_generator.py` | Subject extraction, 137-entry catalog (incl. 32 Snapchat-style filter stickers), local styles, optional Replicate img2img, compose/encode, pack loop |
| `auto_tester.py` | Sanity checks, retry/backoff, `heal_index` self-healing loop, self-test |
| `face_expressions.py` | Face detection (Haar, or silhouette estimate) + redrawn emotions |
| `face_filters.py` | Snapchat-style funny filters: dog/cat/bunny ears, flower crown, big eyes/head, rainbow barf, ... |
| `exporter.py` | ZIP, `.wastickers` (30/pack, <=100 KB), Telegram payload + upload script |

## Optional AI restyling
    export REPLICATE_API_TOKEN=r8_...
    export REPLICATE_VERSION=<version hash of an SDXL img2img model>
    # if the model's input names differ: REPLICATE_IMAGE_KEY / REPLICATE_PROMPT_KEY
If the API fails or is rate-limited, each sticker falls back to the local pipeline.

## Telegram upload
    cd output/<run>/telegram && pip install requests
    python upload_to_telegram.py --token <BOT_TOKEN> --user-id <YOUR_NUMERIC_ID>

## Snap filters
Categories `Snap Filters` (24) and `Snap Mash-ups` (8) put funny face filters on your image. They are anchored to the detected face
(or an estimate from the silhouette) and verified by the self-healing loop (`MIN_FILTER_CHANGE`).
Use OpenCV 4.x for Haar face detection; on OpenCV 5.x the code falls back to the silhouette estimate automatically.
