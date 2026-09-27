# 🛰️ SatQuery AI

**An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries**

Built for **Smart India Hackathon 2026** — Problem Statement **SIH26167** · Theme: **Space Technology** · Category: **Software**

🔗 **Live App:** [satquery-nu.vercel.app](https://satquery-nu.vercel.app/)

---

## 🚀 What is SatQuery AI?

Analysing satellite imagery today usually means opening a GIS tool, manually tiling scenes, and knowing which model or workflow to apply for a given question — a process that can take **~45 minutes per query** and requires specialist expertise.

SatQuery AI removes that barrier. A user simply **asks a question in plain English** — *"What changed between these 2023 and 2024 images?"*, *"Highlight the mining expansion in this scene"*, *"Compare this optical and SAR image"* — and the system automatically:

1. Figures out what kind of task is being asked (search, single-image Q&A, region grounding, bi-temporal change, or optical–SAR fusion),
2. Routes the request to the right specialist model instead of forcing one generic vision-language model to do everything,
3. Returns an answer grounded in the imagery, along with an **execution trace** showing exactly how it got there.

This turns satellite image analysis from a GIS expert's task into a conversation anyone can have.

---

## 🧠 Why "Specialist Models" Instead of One Big VLM?

The core design bet behind SatQuery AI is that **one generic vision-language model is not the best tool for every remote-sensing task**. Instead, the system maintains a **registry of task-specific specialist models**, each domain-adapted on a benchmark suited to it:

| Task | Specialist | Trained / Referenced On |
|---|---|---|
| Visual Question Answering | Lightweight VQA head | RSVQA |
| Captioning / Region Grounding | Phrase-grounding model | VRSBench |
| Change Detection / Change-VQA | Siamese change-detection network | CDVQA, ChangeFormer / BIT |
| Optical–SAR Fusion | Cross-attention dual encoder | BigEarthNet (590K Sentinel-1/2 pairs) |

A **thin LLM layer** sits on top purely to turn each specialist's structured output (bounding box, label, confidence) into a natural-language sentence — **the LLM never does the visual reasoning itself**. Every step (task classification, model selected, confidence, coordinates, latency) is logged into a structured **execution trace**, so results are auditable rather than a black box.

---

## ✨ Key Features

- **Natural-language query interface** — no GIS software or query syntax required
- **Multimodal support** — works across optical (Sentinel-2) and SAR (Sentinel-1) imagery
- **Agentic orchestration** — automatically classifies the query and dispatches it to the right specialist
- **Multi-image analysis** — supports single-image, paired/bi-temporal, and fusion queries
- **Evidence-grounded outputs** with GeoJSON bounding boxes / spatial grounding
- **Auditable execution summaries** — every answer comes with an inspector-panel trace of task, model, and confidence
- **Interactive web GUI** with live query console and sample "quick action" prompts (detect mining expansion, flag encroachment, track mangrove health, etc.)
- **Cost-efficient data pipeline** — COG-based tiling and cached imagery avoid downloading full scenes

---

## 🏗️ Technical Approach

**Methodology / Pipeline:**

1. **Query Decomposition** — classify the request as VQA, Grounding, Change-VQA, or Optical–SAR Fusion
2. **Geospatial Verification** — GDAL/Rasterio checks CRS, area of interest, and matches imagery to the requested location/time
3. **Specialist Model Dispatch** — route to the appropriate model rather than a single general-purpose VLM
4. **Cross-Modal Fusion** — dual-encoder + cross-attention combine Sentinel-2 optical and Sentinel-1 SAR information
5. **Spatial Grounding** — decoder links the answer to specific image regions (GeoJSON bounding boxes / masks)
6. **Execution Trace** — every step logged as `{task, model_id, bbox, confidence, latency_ms}` for transparency

**System Flow:**

```
Frontend (Web GUI)
      │
FastAPI backend
      │
Agentic Controller  →  classifies query → routes to specialist
      │
┌─────┴─────┬─────────────┬──────────────┬───────────────────┐
│  VQA head │  Grounding  │  Change / VQA │ Optical–SAR Fusion │
│  (RSVQA)  │ (VRSBench)  │   (CDVQA)     │   (BigEarthNet)    │
└─────┬─────┴─────────────┴──────────────┴───────────────────┘
      │
Thin LLM Formatter → Evidence-Grounded Response
      │
Data Layer: GDAL / Rasterio COG tiling + PostGIS
```

**Tech Stack:** React (frontend concepts) · FastAPI · GDAL / Rasterio · Docker · PostGIS · Google Gemini (response formatting)

---

## 💻 This Repository (Implementation)

This repo contains the deployed reference implementation of SatQuery AI — a FastAPI backend + static frontend, structured to run on **Vercel's Python serverless runtime**.

```
api/index.py              FastAPI app (the serverless function) — search, query & agent-query endpoints
api/config.py             Credential defaults (move to env vars for production)
api/data/                 metadata.json, embeddings.json, bundled_images.json — local fallback dataset + TF-IDF index
public/index.html         Frontend, served from the CDN at /
public/dataset-images/    90 sample Earth-observation dataset images (India-focused: agriculture, disaster, urban, water, mining, forest, coastal, infrastructure)
public/AI Agent images/   Sample outputs for the agent's analysis tasks (crop health, coastal erosion, deforestation, urban sprawl, etc.)
tools/                    Offline scripts (embed_images.py, download_dataset.py)
vercel.json               Routing + serverless function config
requirements.txt          Runtime dependencies
```

### API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/query` | POST | Natural-language search across live ISRO Bhoonidhi data or the local fallback dataset (semantic / TF-IDF search) |
| `/agent-query` | POST | Upload 1–2 images with a text query; the agent classifies the task (VQA / grounding / change detection / fusion) and returns a grounded answer + execution trace |
| `/isro/download/{collection}/{image_id}` | GET | Proxies authenticated downloads from ISRO Bhoonidhi |
| `/metadata` | GET | Returns the local dataset catalog |
| `/health` | GET | Reports data-source and configuration status |

### Data Sources

- **Live:** ISRO **Bhoonidhi** — Sentinel-1/2, Landsat, ResourceSat-2, EOS-04 (free, <5 m GSD baseline)
- **Demo-only (licensed):** Cartosat-2S/3, CartoDEM, under NSIL government-entity access
- **Training references:** BigEarthNet (Optical–SAR fusion), RSVQA (VQA), VRSBench (grounding), CDVQA (change-VQA)

> ⚠️ **Note:** The bundled local dataset (`api/data/metadata.json`) is sample imagery used only when live Bhoonidhi search is unreachable/unconfigured — it is not yet wired to real Sentinel/Landsat/Cartosat tiles. This is called out honestly in the UI via a source-note banner, and is the next milestone for the project.

### Running Locally

```bash
git clone https://github.com/<your-org>/SatQuery.git
cd SatQuery
pip install -r requirements.txt uvicorn
cd api && uvicorn index:app --reload --port 8000
```

Then open `http://localhost:8000` — the frontend uses relative URLs, so it works unchanged locally and in production.

**Environment variables** (set in Vercel → Settings → Environment Variables, or a local `.env`):

| Variable | Purpose |
|---|---|
| `ISRO_USER_ID` / `ISRO_PASSWORD` | Bhoonidhi account credentials |
| `ISRO_API_URL` | Defaults to `https://bhoonidhi-api.nrsc.gov.in` |
| `DISABLE_BHOONIDHI` | Set `true` to force the local-dataset path |
| `GEMINI_API_KEY` | Required for `/agent-query` response generation |

### Rebuilding the Search Index

After editing captions in `api/data/metadata.json`:

```bash
pip install scikit-learn
python tools/embed_images.py
```

---

## 📈 Feasibility, Risks & Mitigation

**Risks:**
- High-resolution imagery (Sentinel-2 ~10,970×10,970 px, Cartosat 100+ MP) can exceed model input limits
- Maintaining multiple specialist models (VQA, grounding, change detection, fusion) adds operational complexity
- Cartosat-2S/3 and some DEM data are commercially licensed, not freely available for all use cases

**Mitigation:**
- Large scenes are split into overlapping tiles, processed independently, and stitched back using geo-referencing
- Specialist models share a common feature encoder where possible; a single model registry tracks versions with automated pre-deployment tests
- Licensed data is scoped to a small set of demo scenes under NSIL access; the core system runs entirely on free sources (Sentinel-1/2, Landsat, ResourceSat-2, EOS-04)

---

## 💡 Impact & Market Opportunity

- Cuts image analysis time from **~45 minutes of manual GIS work down to a few minutes per query**
- Detects urban and agricultural changes with **78–92% confidence** by fusing optical + SAR data from the start
- Keeps the core pipeline on **free public data**, minimizing acquisition costs
- Targets a **₹76,000 Cr+ Indian Earth Observation market by 2033 (~28% CAGR)**
- **Sectors:** Agriculture, disaster response, urban planning, infrastructure
- **Customers:** Government agencies, enterprises, research institutions

---

## 📚 Research References

- **GeoRSCLIP / RS5M** — remote-sensing vision-language backbone & multimodal fusion reference
- **GeoPixel** — pixel-level grounding and tile-based processing for large satellite images
- **ChangeFormer / BIT** — bi-temporal change detection reference for the Change-VQA component
- **RSVQA & VRSBench** — benchmarks for satellite VQA and spatial/region grounding
- **ISRO / NSIL Earth Observation Data Policy** — governs use of open Bhoonidhi data and licensed Cartosat products

---

## 👥 Team — Spotlight

**Team ID:** 174259 · **Problem Statement:** SIH26167

| Member | GitHub / Profile |
|---|---|
| Team Leader | [ThanuShree99](https://github.com/ThanuShree99) |
| Team Member | [Samdcruzzz](https://github.com/Samdcruzzz) |
| Team Member | [sudhamanikandan206](https://github.com/sudhamanikandan206) |
| Team Member | [archanas126002-bit](https://github.com/archanas126002-bit) |
| Team Member | [Aishukv13-nebula](https://github.com/Aishukv13-nebula) |
| Team Member | [advikamannan](https://github.com/advikamannan) | 

---

## 🔗 Links

- **Live Demo:** [satquery-nu.vercel.app](https://satquery-nu.vercel.app/)
- **Problem Statement:** SIH26167 — SatQuery AI, Smart India Hackathon 2026

---

*Built with FastAPI, GDAL/Rasterio, and domain-adapted remote-sensing models — because satellite image analysis shouldn't require a GIS degree.*
