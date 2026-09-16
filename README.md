# SatQuery — Vercel deployment

FastAPI backend + static frontend, restructured to run on Vercel's Python
serverless runtime.

## Layout

```
api/index.py              FastAPI app (the serverless function)
api/config.py             Credential defaults — move these to env vars
api/data/                 metadata.json, embeddings.json, bundled_images.json
public/index.html         Frontend, served from the CDN at /
public/dataset-images/    27 dataset images, served from the CDN
public/images/            Fallback demo frame
tools/                    Offline scripts (embed_images.py, download_dataset.py)
vercel.json               Routing + function config
requirements.txt          Runtime dependencies
```

Static files under `public/` are served by Vercel's CDN. Everything that
doesn't match a static file is rewritten to `api/index.py`, so `/query`,
`/health`, `/metadata`, `/earth-search`, `/agent-query` and
`/isro/download/...` all reach FastAPI while images and the HTML never
invoke the function at all.

## Environment variables

`api/config.py` ships with working defaults so the first deploy runs
without any dashboard setup. Set these in Vercel (Settings → Environment
Variables) and then blank the file out — anything set in the environment
wins:

| Variable | Purpose |
|---|---|
| `ISRO_USER_ID` | Bhoonidhi account |
| `ISRO_PASSWORD` | Bhoonidhi account |
| `ISRO_API_URL` | Defaults to `https://bhoonidhi-api.nrsc.gov.in` |
| `DISABLE_BHOONIDHI` | `true` forces the local-dataset path |
| `GEMINI_API_KEY` | Read by `google-genai` when it is wired up |

## Changes made for serverless

Serverless functions have a read-only filesystem apart from `/tmp`, no
persistent local disk, and a per-request time budget. The original
`backend/main.py` assumed all three, so:

- **Module-level `genai.Client()` removed.** It raised at import time when
  `GEMINI_API_KEY` was unset, which takes down the whole function. Nothing
  calls it yet, so it is now created lazily in `get_genai_client()`.
- **`os.makedirs(BASE_DIR / "isro_cache")` moved to `/tmp/isro_cache`.**
  Writing next to the code fails on a read-only filesystem, again at
  import time. The download proxy streams through `/tmp` instead.
- **`image_url_for()` no longer stats the filesystem.** Dataset images
  live on the CDN, not in the function bundle, so the old `.exists()`
  check would have failed for every image and silently returned the
  placeholder. It now checks `api/data/bundled_images.json`.
- **TF-IDF vectors stored sparsely** (`{column: value}`) rather than as
  dense arrays. `_cosine()` accepts both, so an older dense
  `embeddings.json` still works.
- **Upstream timeouts reduced** (auth 8s, search 12s, geocode 6s) to stay
  inside the 60s function budget, which `vercel.json` sets explicitly.
- **Duplicate `/history` route removed** — it was defined twice.
- **`requirements.txt` trimmed** to what the request path actually
  imports. `scikit-learn`, `anthropic`, `Pillow` and `sqlalchemy` were
  unused at runtime; `scikit-learn` is only needed by
  `tools/embed_images.py`, which runs offline.

## Known data issues

`aristarchus_crater.jpg`, `tycho_crater.jpg` and `rima_marius_rille.jpg`
are byte-identical — all three are the LROC site logo, not lunar imagery.
`download_dataset.py` fetches those three by scraping a `source_page` for
its `og:image`, and it picked up the site header instead. The three
`direct_url` downloads (Chandrayaan-2) and `copernicus_crater.png` are
genuine imagery. To fix, find the real image URLs and switch those entries
to `direct_url`, then re-run `tools/download_dataset.py`.

Twenty `lunar_frame_*.jpg` entries carry `latitude: null` and
`longitude: null`, so they plot at 0°, 0° on the map. Real coordinates
need to come from whoever supplied the files.

## Local development

```bash
pip install -r requirements.txt uvicorn
cd api && uvicorn index:app --reload --port 8000
```

The frontend uses relative URLs (`API_BASE = ""`), so it works unchanged
on both localhost and the deployed domain.

## Rebuilding the search index

After editing captions in `api/data/metadata.json`:

```bash
pip install scikit-learn
python tools/embed_images.py
```

Note that `tools/embed_images.py` writes dense vectors. `_cosine()` reads
both formats, so that is fine, but the file will be larger.
