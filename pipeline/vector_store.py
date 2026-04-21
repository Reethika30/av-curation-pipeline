"""Vector index wrapper.

In-process ANN index built on NumPy. Sufficient for the demo (~10K vectors)
and for offline curation runs up to ~1M vectors with brute-force cosine.
The interface is intentionally narrow so a sharded production backend can be
swapped in behind it without touching pipeline code.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class SearchHit:
    sample_token: str
    score: float  # cosine similarity in [-1, 1]
    payload: dict


class VectorStore:
    """Brute-force cosine index. ``uri`` is accepted for API compatibility
    with sharded backends but is currently unused."""

    def __init__(self, dim: int, name: str, uri: str | Path | None = None):
        self.dim = dim
        self.name = name
        self.uri = Path(uri) if uri else None
        self._vecs: list[np.ndarray] = []
        self._payloads: list[dict] = []
        self._backend = "numpy"
        log.info("Vector index ready (backend=numpy, dim=%d)", dim)

    @property
    def backend(self) -> str:
        return self._backend

    def add(self, vectors: np.ndarray, payloads: Iterable[dict]) -> None:
        assert vectors.shape[1] == self.dim
        for v, p in zip(vectors, payloads):
            self._vecs.append(v.astype(np.float32))
            self._payloads.append(p)

    def search(self, query: np.ndarray, k: int = 10) -> list[SearchHit]:
        if not self._vecs:
            return []
        mat = np.stack(self._vecs, axis=0)
        q = query / (np.linalg.norm(query) + 1e-8)
        sims = mat @ q
        order = np.argsort(-sims)[:k]
        return [
            SearchHit(self._payloads[i]["sample_token"], float(sims[i]), self._payloads[i])
            for i in order
        ]

    @property
    def matrix(self) -> np.ndarray:
        return np.stack(self._vecs, axis=0) if self._vecs else np.zeros((0, self.dim), np.float32)

    @property
    def payloads(self) -> list[dict]:
        return list(self._payloads)
