"""
Member 2 (ML/VLM Engineer) — embedding / semantic search pipeline.

** LEGACY / NOT USED AT RUNTIME **
This script is left over from an earlier lunar/Chandrayaan-2 prototype
(note the Moon-specific caption prompt below). SatQuery is now an
Earth-observation console, and `api/data/embeddings.json` has already been
regenerated directly from the real `api/data/metadata.json` captions using
the same TF-IDF approach this script implements. The paths below
(`backend/data/`, `backend/images/dataset/`) also don't match the current
`api/data/` layout. Kept for reference only — rewrite before reuse.

WHAT THIS DOES (historical)
--------------
1. Reads backend/data/metadata.json.
2. If ANTHROPIC_API_KEY is set, calls Claude's vision API once per real
   image file present in backend/images/dataset/ to generate a feature-
   focused caption (this is the "Vision-LLM" step). If no key is set, it
   just uses the captions already in metadata.json (which are already
   real, human-written descriptions — this keeps the pipeline runnable
   on Day 1 before anyone has an API key configured).
3. Builds a TF-IDF index over all captions and saves it to
   backend/data/embeddings.json. This is what main.py loads at query time
   to do real text-similarity search (cosine similarity), instead of the
   naive "count matching words" scoring that was there before.

USAGE
-----
    cd backend
    python embed_images.py            # normal run
    python embed_images.py --force    # re-caption every image with the VLM,
                                       # even ones that already look real

WHY TF-IDF AND NOT A LIVE VLM CALL PER QUERY
---------------------------------------------
Calling a Vision-LLM on every user query would be slow and depends on a
live API being reachable during your demo — a bad idea in front of judges.
Captioning happens once, offline, in this script. Query-time search is a
fast, local, fully offline cosine-similarity lookup, so the live demo
never depends on network/API latency for its core search step. main.py
can still optionally call an LLM at query time just to phrase the final
answer text (see ANSWER_STYLE in main.py), with a template fallback.
"""
import argparse
import base64
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
METADATA_FILE = BASE_DIR / "data" / "metadata.json"
IMAGES_DIR = BASE_DIR / "images" / "dataset"
EMBEDDINGS_FILE = BASE_DIR / "data" / "embeddings.json"

import os

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
VISION_MODEL = os.getenv("VISION_LLM_MODEL", "claude-sonnet-4-6")

CAPTION_PROMPT = (
    "Describe this lunar/satellite image in one or two sentences. Focus on "
    "concrete visible terrain features (craters, mare/plains, highlands, "
    "rilles, ejecta rays, ridges, shadows) so the description works well as "
    "a search index entry. Do not mention that this is an AI description."
)


def load_metadata() -> dict:
    if not METADATA_FILE.exists():
        print(f"ERROR: {METADATA_FILE} not found.")
        sys.exit(1)
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_metadata(data: dict) -> None:
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def caption_image_with_vlm(image_path: Path) -> str | None:
    """Call Claude's vision API to caption one image. Returns None on any
    failure so the caller can fall back to the existing metadata caption."""
    if not ANTHROPIC_API_KEY:
        return None
    try:
        import anthropic
    except ImportError:
        print("  ! 'anthropic' package not installed (pip install anthropic) — using existing caption instead")
        return None

    suffix = image_path.suffix.lower()
    media_type = "image/png" if suffix == ".png" else "image/jpeg"
    try:
        img_b64 = base64.standard_b64encode(image_path.read_bytes()).decode("utf-8")
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        resp = client.messages.create(
            model=VISION_MODEL,
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": img_b64}},
                    {"type": "text", "text": CAPTION_PROMPT},
                ],
            }],
        )
        text = "".join(block.text for block in resp.content if block.type == "text").strip()
        return text or None
    except Exception as e:
        print(f"  ! VLM captioning failed for {image_path.name}: {e}")
        return None


def build_tfidf_index(captions: list[str]) -> dict:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        print("ERROR: scikit-learn not installed. Run: pip install scikit-learn")
        sys.exit(1)

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(captions)
    return {
        "vocabulary": vectorizer.vocabulary_,
        "idf": vectorizer.idf_.tolist(),
        "stop_words": "english",
        "vectors": matrix.toarray().tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-caption every image with the VLM, even if it already has a caption")
    args = parser.parse_args()

    data = load_metadata()
    images = data.get("images", [])
    if not images:
        print("No entries in metadata.json — run download_dataset.py first, or add entries manually.")
        sys.exit(1)

    if not ANTHROPIC_API_KEY:
        print("NOTE: ANTHROPIC_API_KEY is not set, so this run will use the human-written")
        print("captions already in metadata.json rather than generating new ones with a")
        print("Vision-LLM. That's fine for Day 1 — set the key later and re-run with --force")
        print("to upgrade every caption to a VLM-generated one.\n")

    changed = False
    for item in images:
        filename = item.get("filename", "")
        image_path = IMAGES_DIR / filename
        if not image_path.exists():
            print(f"  (skipping VLM captioning for {filename} — file not found; run download_dataset.py)")
            continue
        if args.force or ANTHROPIC_API_KEY:
            print(f"  -> captioning {filename} with {VISION_MODEL} ...")
            new_caption = caption_image_with_vlm(image_path)
            if new_caption:
                item["caption"] = new_caption
                changed = True
                print(f"     {new_caption[:90]}")

    if changed:
        save_metadata(data)
        print("\nUpdated metadata.json with VLM-generated captions.")

    captions = [item.get("caption", "") for item in images]
    print(f"\nBuilding TF-IDF index over {len(captions)} captions ...")
    index = build_tfidf_index(captions)
    index["ids"] = [item.get("id") for item in images]
    with open(EMBEDDINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f)
    print(f"Saved index to {EMBEDDINGS_FILE}")
    print("\nNext step: restart the backend (uvicorn main:app --reload) — it will pick up")
    print("the new embeddings.json automatically and use real semantic search.")


if __name__ == "__main__":
    main()
