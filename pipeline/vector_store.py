"""Vector store wrapper.

Uses LanceDB when available (production path: on-disk, IVF-PQ, scales to
hundreds of millions of vectors). Falls back to a brute-force NumPy index
otherwise — fine for the demo (~10K vectors) and keeps the interface stable.
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


class _NumpyStore:
    def __init__(self, dim: int):
        self.dim = dim
        self._vecs: list[np.ndarray] = []
        self._payloads: list[dict] = []

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


class VectorStore:
    """Unified wrapper. ``backend`` is selected automatically."""

    def __init__(self, dim: int, name: str, uri: str | Path | None = None):
        self.dim = dim
        self.name = name
        self.uri = Path(uri) if uri else None
        self._np = _NumpyStore(dim)
        self._lance_table = None
        self._backend = "numpy"

        if uri is not None:
            try:
                import lancedb  # type: ignore
                self._lance = lancedb.connect(str(uri))
                self._backend = "lancedb"
                log.info("LanceDB store ready at %s/%s", uri, name)
            except Exception as exc:  # noqa: BLE001
                log.warning("LanceDB unavailable (%s); using NumPy store", exc)

    @property
    def backend(self) -> str:
        return self._backend

    def add(self, vectors: np.ndarray, payloads: list[dict]) -> None:
        self._np.add(vectors, payloads)
        if self._backend == "lancedb":
            rows = [{"vector": v.tolist(), **p} for v, p in zip(vectors, payloads)]
            if self._lance_table is None:
                self._lance_table = self._lance.create_table(self.name, data=rows, mode="overwrite")
            else:
                self._lance_table.add(rows)

    def search(self, query: np.ndarray, k: int = 10) -> list[SearchHit]:
        if self._backend == "lancedb" and self._lance_table is not None:
            res = (
                self._lance_table.search(query.tolist())
                .metric("cosine")
                .limit(k)
                .to_list()
            )
            return [
                SearchHit(r["sample_token"], 1.0 - float(r["_distance"]), r) for r in res
            ]
        return self._np.search(query, k=k)

    @property
    def matrix(self) -> np.ndarray:
        return self._np.matrix

    @property
    def payloads(self) -> list[dict]:
        return self._np.payloads
