"""Vector index wrappers.

Two backends, same narrow interface:

* :class:`NumpyVectorStore` — brute-force cosine. Exact, but O(N) per query.
  Fine up to ~50K vectors; the original demo backend.

* :class:`FaissVectorStore` — IVF index from FAISS. Sub-linear search,
  ~10-50x faster on 10K+ vectors. Defaults to ``IndexIVFFlat`` (exact
  per-cell scoring, no quantization loss) and switches to ``IndexIVFPQ``
  only when the corpus is large enough that memory matters
  (configurable via ``flavor=``). Falls back to ``IndexFlatIP`` for
  corpora too small to train an IVF index; promote with
  :meth:`FaissVectorStore.finalize`.

Use :func:`make_store` to pick a backend. The ``faiss`` extra
(``pip install av-curation-pipeline[ann]``) is required for FAISS.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class SearchHit:
    sample_token: str
    score: float  # cosine similarity in [-1, 1]
    payload: dict


class VectorStore(Protocol):
    backend: str
    dim: int
    name: str

    def add(self, vectors: np.ndarray, payloads: Iterable[dict]) -> None: ...
    def search(self, query: np.ndarray, k: int = 10) -> list[SearchHit]: ...
    def search_batch(self, queries: np.ndarray, k: int = 10) -> list[list[SearchHit]]: ...
    @property
    def matrix(self) -> np.ndarray: ...
    @property
    def payloads(self) -> list[dict]: ...
    def __len__(self) -> int: ...


# ---------------------------------------------------------------------------
# NumPy brute force
# ---------------------------------------------------------------------------
class NumpyVectorStore:
    """Brute-force cosine index. Exact, simple, no deps."""

    backend = "numpy"

    def __init__(self, dim: int, name: str, uri: str | Path | None = None):
        self.dim = dim
        self.name = name
        self.uri = Path(uri) if uri else None
        self._vecs: list[np.ndarray] = []
        self._payloads: list[dict] = []
        log.info("Vector index ready (backend=numpy, dim=%d)", dim)

    def __len__(self) -> int:
        return len(self._vecs)

    def add(self, vectors: np.ndarray, payloads: Iterable[dict]) -> None:
        assert vectors.shape[1] == self.dim
        for v, p in zip(vectors, payloads):
            self._vecs.append(v.astype(np.float32))
            self._payloads.append(p)

    def _normalized_matrix(self) -> np.ndarray:
        if not self._vecs:
            return np.zeros((0, self.dim), np.float32)
        m = np.stack(self._vecs, axis=0)
        return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-8)

    def search(self, query: np.ndarray, k: int = 10) -> list[SearchHit]:
        return self.search_batch(query[None, :], k)[0]

    def search_batch(self, queries: np.ndarray, k: int = 10) -> list[list[SearchHit]]:
        if not self._vecs:
            return [[] for _ in queries]
        mat = self._normalized_matrix()
        qn = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
        sims = qn @ mat.T  # (Q, N)
        k = min(k, sims.shape[1])
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        out: list[list[SearchHit]] = []
        for qi, idx in enumerate(part):
            order = idx[np.argsort(-sims[qi, idx])]
            out.append([
                SearchHit(self._payloads[i]["sample_token"],
                          float(sims[qi, i]), self._payloads[i])
                for i in order
            ])
        return out

    def finalize(self) -> dict:
        return {"index_type": "numpy-bruteforce", "n": len(self)}

    @property
    def matrix(self) -> np.ndarray:
        return np.stack(self._vecs, axis=0) if self._vecs else np.zeros((0, self.dim), np.float32)

    @property
    def payloads(self) -> list[dict]:
        return list(self._payloads)


# ---------------------------------------------------------------------------
# FAISS IVF-PQ
# ---------------------------------------------------------------------------
def _suggest_ivf_params(n: int, dim: int) -> tuple[int, int, int]:
    """Heuristic (nlist, m, nbits) for IVF-PQ.

    * ``nlist`` ~ sqrt(n), clamped so each cell has ~30+ training samples.
    * ``m`` divides ``dim`` and gives ~8-dim sub-vectors when possible.
    * ``nbits`` is 8 — the standard PQ choice; 256 centroids per sub-quantizer.
    """
    nlist = max(8, min(256, int(np.sqrt(max(n, 16)))))
    while nlist > 8 and n // nlist < 30:
        nlist //= 2
    candidates = [m for m in (16, 12, 8, 6, 4, 2) if dim % m == 0]
    m = candidates[0] if candidates else 1
    return nlist, m, 8


class FaissVectorStore:
    """FAISS IVF index over inner-product (vectors are L2-normalised on
    insert, so IP == cosine similarity).

    Vectors are appended to a flat staging buffer; call :meth:`finalize` once
    all data has been added to train the IVF clustering (and, in the PQ
    flavor, the PQ codebooks). Searches before finalize use ``IndexFlatIP``
    (exact).

    ``flavor``:
      * ``"auto"``   — IVFFlat if raw memory < ``pq_threshold_mb``, else IVFPQ.
      * ``"ivfflat"``— always IVFFlat (exact per-cell, no quantization loss).
      * ``"ivfpq"``  — always IVFPQ (lossy, small memory footprint).
    """

    backend = "faiss"

    def __init__(self, dim: int, name: str, nlist: int | None = None,
                 m: int | None = None, nbits: int = 8, nprobe: int = 16,
                 flavor: str = "auto", pq_threshold_mb: int = 200):
        try:
            import faiss  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "faiss is required for FaissVectorStore. "
                "Install with: pip install av-curation-pipeline[ann]"
            ) from exc
        if flavor not in ("auto", "ivfflat", "ivfpq"):
            raise ValueError(f"unknown flavor: {flavor}")
        self._faiss = faiss
        self.dim = dim
        self.name = name
        self._nlist_override = nlist
        self._m_override = m
        self._nbits = nbits
        self._nprobe = nprobe
        self._flavor = flavor
        self._pq_threshold_mb = pq_threshold_mb
        self._payloads: list[dict] = []
        self._raw: list[np.ndarray] = []
        self._index = faiss.IndexFlatIP(dim)
        self._trained = False
        self.build_info: dict | None = None
        log.info("Vector index ready (backend=faiss, dim=%d, flavor=%s, untrained)", dim, flavor)

    def __len__(self) -> int:
        return len(self._payloads)

    def _norm(self, v: np.ndarray) -> np.ndarray:
        v = np.ascontiguousarray(v, dtype=np.float32)
        n = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8
        return v / n

    def add(self, vectors: np.ndarray, payloads: Iterable[dict]) -> None:
        assert vectors.shape[1] == self.dim
        v = self._norm(vectors)
        payloads = list(payloads)
        self._raw.append(v.copy())
        self._payloads.extend(payloads)
        if not self._trained:
            self._index.add(v)

    def _pick_flavor(self, n: int) -> str:
        if self._flavor != "auto":
            return self._flavor
        raw_mb = (n * self.dim * 4) / (1024 * 1024)
        return "ivfpq" if raw_mb >= self._pq_threshold_mb else "ivfflat"

    def finalize(self) -> dict:
        """Promote to a trained IVF index if we have enough vectors."""
        n = len(self._payloads)
        nlist, m, nbits = _suggest_ivf_params(n, self.dim)
        if self._nlist_override is not None:
            nlist = self._nlist_override
        if self._m_override is not None:
            m = self._m_override
        if self._nbits is not None:
            nbits = self._nbits

        if n < max(256, 4 * nlist):
            log.info("FAISS: keeping IndexFlatIP (n=%d < 4*nlist=%d)", n, 4 * nlist)
            self.build_info = {"index_type": "IndexFlatIP", "n": n}
            return self.build_info

        faiss = self._faiss
        all_vecs = np.concatenate(self._raw, axis=0)
        quantizer = faiss.IndexFlatIP(self.dim)
        flavor = self._pick_flavor(n)
        if flavor == "ivfflat":
            ivf = faiss.IndexIVFFlat(quantizer, self.dim, nlist,
                                      faiss.METRIC_INNER_PRODUCT)
            ivf.train(all_vecs)
            ivf.add(all_vecs)
            ivf.nprobe = self._nprobe
            self._index = ivf
            self._trained = True
            log.info("FAISS: trained IVFFlat (nlist=%d, nprobe=%d) on n=%d",
                     nlist, self._nprobe, n)
            self.build_info = {"index_type": "IndexIVFFlat", "n": n,
                               "nlist": nlist, "nprobe": self._nprobe}
            return self.build_info

        ivfpq = faiss.IndexIVFPQ(quantizer, self.dim, nlist, m, nbits,
                                  faiss.METRIC_INNER_PRODUCT)
        ivfpq.train(all_vecs)
        ivfpq.add(all_vecs)
        ivfpq.nprobe = self._nprobe
        self._index = ivfpq
        self._trained = True
        log.info("FAISS: trained IVF-PQ (nlist=%d, m=%d, nbits=%d, nprobe=%d) on n=%d",
                 nlist, m, nbits, self._nprobe, n)
        self.build_info = {"index_type": "IndexIVFPQ", "n": n,
                           "nlist": nlist, "m": m, "nbits": nbits,
                           "nprobe": self._nprobe}
        return self.build_info

    def search(self, query: np.ndarray, k: int = 10) -> list[SearchHit]:
        return self.search_batch(query[None, :], k)[0]

    def search_batch(self, queries: np.ndarray, k: int = 10) -> list[list[SearchHit]]:
        if len(self._payloads) == 0:
            return [[] for _ in queries]
        q = self._norm(queries)
        k = min(k, len(self._payloads))
        sims, idx = self._index.search(q, k)
        out: list[list[SearchHit]] = []
        for qi in range(q.shape[0]):
            hits: list[SearchHit] = []
            for j, i in enumerate(idx[qi]):
                if i < 0:
                    continue
                p = self._payloads[int(i)]
                hits.append(SearchHit(p["sample_token"], float(sims[qi, j]), p))
            out.append(hits)
        return out

    @property
    def matrix(self) -> np.ndarray:
        return np.concatenate(self._raw, axis=0) if self._raw else np.zeros((0, self.dim), np.float32)

    @property
    def payloads(self) -> list[dict]:
        return list(self._payloads)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def make_store(dim: int, name: str, backend: str = "auto",
               n_hint: int | None = None):
    """Pick a backend.

    * ``"numpy"`` — always brute force.
    * ``"faiss"`` — always IVF-PQ (requires faiss).
    * ``"auto"`` — FAISS if installed and ``n_hint >= 2000``, else NumPy.
    """
    if backend == "numpy":
        return NumpyVectorStore(dim=dim, name=name)
    if backend == "faiss":
        return FaissVectorStore(dim=dim, name=name)
    if backend == "auto":
        try:
            import faiss  # type: ignore  # noqa: F401
        except ImportError:
            log.info("faiss not available; using numpy backend")
            return NumpyVectorStore(dim=dim, name=name)
        if n_hint is not None and n_hint < 2000:
            log.info("n_hint=%d below FAISS threshold; using numpy", n_hint)
            return NumpyVectorStore(dim=dim, name=name)
        return FaissVectorStore(dim=dim, name=name)
    raise ValueError(f"unknown backend: {backend}")
