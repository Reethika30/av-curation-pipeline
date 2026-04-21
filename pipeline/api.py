"""FastAPI service exposing curation queries.

Loads precomputed artifacts from ``web/public/data/`` (the same JSON the
frontend reads), so the API and the static demo always agree.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pipeline.embeddings import get_encoder

DATA_DIR = Path(__file__).resolve().parent.parent / "web" / "public" / "data"

app = FastAPI(title="AV Curation Pipeline", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class _State:
    summary: dict[str, Any] | None = None
    samples: list[dict] | None = None
    duplicates: list[dict] | None = None
    outliers: list[dict] | None = None
    clusters: list[dict] | None = None
    lineage: list[dict] | None = None
    clip_matrix: np.ndarray | None = None
    encoder = None


state = _State()


def _load_json(name: str) -> Any:
    path = DATA_DIR / name
    if not path.exists():
        raise HTTPException(503, f"missing artifact: {name} — run `python -m precompute.run` first")
    return json.loads(path.read_text(encoding="utf-8"))


@app.on_event("startup")
def _load() -> None:
    if not DATA_DIR.exists():
        return
    try:
        state.summary = _load_json("summary.json")
        state.samples = _load_json("samples.json")
        state.duplicates = _load_json("duplicates.json")
        state.outliers = _load_json("outliers.json")
        state.clusters = _load_json("clusters.json")
        state.lineage = _load_json("lineage.json")
        emb_path = DATA_DIR / "clip_embeddings.npz"
        if emb_path.exists():
            state.clip_matrix = np.load(emb_path)["clip"]
    except HTTPException:
        pass  # API still answers /health


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "loaded": state.summary is not None,
        "summary": state.summary,
    }


@app.get("/summary")
def summary() -> dict:
    if state.summary is None:
        raise HTTPException(503, "no precomputed data; run precompute.run first")
    return state.summary


@app.get("/near-duplicates")
def near_duplicates(min_size: int = 2, limit: int = 50) -> list[dict]:
    if state.duplicates is None:
        raise HTTPException(503, "no precomputed data")
    out = [d for d in state.duplicates if len(d["members"]) >= min_size]
    return out[:limit]


@app.get("/outliers")
def outliers(limit: int = 50) -> list[dict]:
    if state.outliers is None:
        raise HTTPException(503, "no precomputed data")
    return state.outliers[:limit]


@app.get("/clusters")
def clusters() -> list[dict]:
    if state.clusters is None:
        raise HTTPException(503, "no precomputed data")
    return state.clusters


@app.get("/lineage")
def lineage() -> list[dict]:
    if state.lineage is None:
        raise HTTPException(503, "no precomputed data")
    return state.lineage


class SearchBody(BaseModel):
    text: str = Field(..., description="Natural-language query (CLIP).")
    k: int = Field(12, ge=1, le=100)


@app.post("/search")
def search(body: SearchBody) -> list[dict]:
    if state.clip_matrix is None or state.samples is None:
        raise HTTPException(503, "no precomputed data")
    if state.encoder is None:
        state.encoder = get_encoder()
    q = state.encoder.encode_text([body.text])[0]
    sims = state.clip_matrix @ q / (
        np.linalg.norm(state.clip_matrix, axis=1) * (np.linalg.norm(q) + 1e-8) + 1e-8
    )
    order = np.argsort(-sims)[: body.k]
    out = []
    for i in order:
        s = state.samples[int(i)]
        out.append({**s, "score": float(sims[int(i)])})
    return out
