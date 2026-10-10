"""
app.py - Streamlit UI.   Run:  streamlit run app.py

Upload one image -> 100+ stickers -> download ZIP / WhatsApp / Telegram bundles.
Optional AI restyling: set REPLICATE_API_TOKEN and REPLICATE_VERSION (see README).
"""
import time
import uuid
from pathlib import Path

import streamlit as st
from PIL import Image

import exporter
import sticker_generator as sg

st.set_page_config(page_title="Sticker Pack Studio", page_icon="🎟️", layout="wide")
st.title("🎟️ Sticker Pack Studio")
st.caption("One image in, 130+ stickers out - now with Snapchat-style funny face filters. Self-testing pipeline, WhatsApp / Telegram / iMessage exports.")

catalog = sg.build_catalog()
catalog_size = len(catalog)
all_categories = list(dict.fromkeys(s.category for s in catalog))

with st.sidebar:
    st.header("Settings")
    chosen = st.multiselect("Categories", all_categories, default=all_categories,
                            help="'Snap Filters' / 'Snap Mash-ups' = Snapchat-style funny face filters "
                                 "(dog ears, flower crown, big eyes, rainbow barf ...).")
    pool = len([s for s in catalog if s.category in chosen]) if chosen else catalog_size
    count = st.slider("Stickers to generate", 1, max(2, pool), max(2, pool))
    threshold = st.slider("Background-removal threshold", 40, 220, 128,
                          help="Lower keeps more of the subject; higher trims halo. "
                               "The self-healing loop also retries with 100 / 160 automatically.")
    use_ai = st.checkbox("AI restyle via Replicate (needs env vars)", value=False)
    title = st.text_input("Pack title", "My Stickers")
    author = st.text_input("Author", "Sticker Studio")
    st.caption("Local mode (default) runs fully offline with PIL + OpenCV.")

upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
if upload:
    src = Image.open(upload)
    st.image(src, caption="Source", width=220)

    if st.button("Generate sticker pack", type="primary"):
        run_dir = Path("output") / f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        bar, status = st.progress(0.0), st.empty()

        def on_progress(done, total, msg):
            bar.progress(done / total)
            status.text(f"{done}/{total}  ·  {msg}")

        try:
            results = sg.generate_pack(src, run_dir, count=count, use_ai=use_ai,
                                       threshold=threshold, progress=on_progress,
                                       categories=set(chosen) or None)
            bundles = exporter.export_all(run_dir, title, author)
            st.session_state["run"] = {"dir": str(run_dir), "results": results, "bundles": bundles}
        except Exception as exc:   # last-line defence: show, don't crash
            st.error(f"Generation failed: {exc}")
            st.stop()
        status.empty()

run = st.session_state.get("run")
if run:
    results, bundles = run["results"], run["bundles"]
    ok = [r for r in results if r.ok]
    healed = sum(1 for r in ok if r.attempts > 1)
    c1, c2, c3 = st.columns(3)
    c1.metric("Stickers generated", f"{len(ok)}/{len(results)}")
    c2.metric("Self-healed", healed)
    c3.metric("Failed", len(results) - len(ok))
    if len(results) - len(ok):
        st.warning("Some stickers failed all attempts; see errors.log in " + run["dir"])

    st.subheader("Download")
    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇️ ZIP (WEBP + PNG)", bundles["zip"].read_bytes(), bundles["zip"].name, "application/zip")
    d3.download_button("⬇️ Telegram bundle", bundles["telegram"].read_bytes(), bundles["telegram"].name, "application/zip")
    with d2:
        for p in bundles["whatsapp"]:
            st.download_button(f"⬇️ {p.name}", p.read_bytes(), p.name, "application/octet-stream", key=p.name)

    st.subheader("Preview")
    for cat in dict.fromkeys(r.category for r in ok):
        with st.expander(cat, expanded=True):
            group = [r for r in ok if r.category == cat]
            st.image([r.path for r in group], width=110, caption=[r.name for r in group])
