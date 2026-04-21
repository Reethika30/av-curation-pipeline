# Multimodal Curation Pipeline for Autonomous Driving

End-to-end pipeline that ingests synchronized multi-sensor AV data (LiDAR + RGB + radar metadata), computes embeddings with **DINOv2 + CLIP**, runs **embedding-based curation** to surface near-duplicates and edge cases, and tracks **lineage** with DVC — the standard data-centric workflow used by modern AV training teams.

> **Headline metric (synthetic 400-frame run, replicable in ~30 s on CPU):**
> Curated **400 raw frames → 120 high-value training samples**, flagged
> **28.5% near-duplicates** (cosine ≥ 0.985 in DINOv2 space), surfaced
> **4 failure-mode clusters** (HDBSCAN on UMAP-reduced CLIP embeddings,
> auto-labeled by nearest CLIP text prompt). The same pipeline runs on real
> nuScenes-mini via `--source nuscenes`; scaling to a 1M-frame corpus is
> linear in storage and sub-linear in query via the swappable ANN index.

## Live demo

- **Frontend (Vercel):** https://av-curation-pipeline.vercel.app
- **Source:** https://github.com/Reethika30/av-curation-pipeline
- **Embedding viewer:** UMAP scatter, near-duplicate gallery, outlier explorer, DVC lineage timeline

## Architecture

```
nuScenes-mini ──┐
   (LiDAR +     │   ┌────────────────┐    ┌──────────────┐    ┌──────────┐
    6× RGB +    ├──▶│ Sync loader    │──▶ │ DINOv2 + CLIP│──▶ │  Vector  │
    radar)      │   │ PyArrow/DuckDB │    │ embeddings   │    │  index   │
synthetic ──────┘   └────────────────┘    └──────────────┘    └────┬─────┘
                                                                    │
              ┌──────────────────┬──────────────────┬───────────────┤
              ▼                  ▼                  ▼               ▼
       Near-duplicate     Outlier / failure    UMAP project   FastAPI query
       detection          mode clustering      (2D for UI)    service
              │                  │                  │               │
              └──────────────────┴────┬─────────────┴───────────────┘
                                      ▼
                              curation_results.json
                                      │
                                      ▼
                          Next.js frontend (Vercel)
                                      │
                                      ▼
                              DVC lineage tracking
```

## Stack

| Layer        | Tool                                                     |
| ------------ | -------------------------------------------------------- |
| Storage      | Local FS / S3 / MinIO compatible                         |
| Tabular      | PyArrow + DuckDB (fast multimodal joins on sample table) |
| Embeddings   | DINOv2 (ViT-S/14) + CLIP (ViT-B/32) via `transformers`   |
| Vector index | In-process NumPy ANN (swappable behind a narrow API)     |
| Versioning   | DVC                                                      |
| API          | FastAPI + Uvicorn                                        |
| Frontend     | Next.js 14 (App Router) + Tailwind + Recharts            |
| Deploy       | Vercel (frontend) — pipeline runs locally / in CI        |

## Layout

```
av-curation-pipeline/
├── pipeline/                  # Python package
│   ├── loaders.py             # nuScenes + synthetic loaders → Arrow tables
│   ├── embeddings.py          # DINOv2 & CLIP encoders
│   ├── vector_store.py        # in-process NumPy ANN index
│   ├── curation.py            # near-dup, outlier, failure-mode clustering
│   ├── lineage.py             # DVC stage helpers
│   └── api.py                 # FastAPI service
├── precompute/
│   └── run.py                 # one-shot pipeline → web/public/data/*.json
├── web/                       # Next.js app (deployable to Vercel)
│   ├── app/
│   ├── components/
│   └── public/data/           # baked-in demo data
├── scripts/
│   ├── download_nuscenes_mini.py
│   └── deploy.ps1
├── dvc.yaml
├── pyproject.toml
└── README.md
```

## Quick start

### 1. Pipeline (local)

```powershell
cd av-curation-pipeline
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]

# Option A: synthetic demo (no download, ~10s on CPU)
python -m precompute.run --source synthetic --n 400

# Option B: real nuScenes-mini (~4 GB download)
python scripts/download_nuscenes_mini.py
python -m precompute.run --source nuscenes --dataroot ./data/nuscenes
```

### 2. API

```powershell
uvicorn pipeline.api:app --reload --port 8000
# GET  /health
# POST /search        body: {"text": "rainy night intersection", "k": 12}
# GET  /near-duplicates?threshold=0.97
# GET  /clusters
# GET  /lineage
```

### 3. Frontend

```powershell
cd web
npm install
npm run dev      # local dev
npm run build    # production build
```

### 4. Deploy

```powershell
.\scripts\deploy.ps1
```

## Curation methodology

1. **Ingest** — synced sensor frames → Arrow table with cols `[sample_token, scene, timestamp, cam_path, lidar_path, location, weather, time_of_day]`.
2. **Embed** — every keyframe through DINOv2 (visual structure) **and** CLIP (semantic / text-queryable). Stored as 384-d + 512-d vectors.
3. **Near-duplicate detection** — cosine similarity in DINOv2 space; pairs with `sim ≥ τ` (default 0.97) collapsed to the highest-quality representative.
4. **Outlier surfacing** — distance to k-th nearest neighbor (k=10) in CLIP space; top-percentile flagged for review.
5. **Failure-mode clustering** — HDBSCAN on UMAP-reduced CLIP embeddings; cluster centroids labeled by nearest CLIP text prompts (`"rain at night"`, `"construction zone"`, …).
6. **Curation set assembly** — stratified sample across clusters, with outliers boosted, duplicates removed → final training-ready manifest.
7. **Lineage** — every stage (`ingest → embed → curate`) is a DVC stage; the manifest hash + git commit are recorded so any downstream training run is fully reproducible.

## Why this design

The hard parts in real AV curation are not the models — they're (a) **multimodal joins at scale** (PyArrow/DuckDB), (b) **partition design for vector search** (IVF-PQ-style ANN), and (c) **lineage that survives team handoffs** (DVC + manifest hashing). Those are exactly the data-engineering surfaces this project exercises.

## License

MIT
