"""Benchmark FAISS IVF-PQ against brute-force NumPy.

Reports recall@k and search latency at multiple corpus sizes. Used by
``precompute.run --benchmark`` to bake a real ANN comparison into the
demo, so the dashboard shows measured numbers rather than claims.
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass

import numpy as np

from pipeline.vector_store import FaissVectorStore, NumpyVectorStore

log = logging.getLogger(__name__)


@dataclass
class BenchPoint:
    n: int
    dim: int
    k: int
    n_queries: int
    brute_ms_per_query: float
    faiss_ms_per_query: float
    speedup: float
    recall_at_k: float
    faiss_index: str
    faiss_params: dict


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def _recall_at_k(brute: list[list], approx: list[list]) -> float:
    if not brute:
        return 1.0
    hits = 0
    total = 0
    for b, a in zip(brute, approx):
        bset = {h.sample_token for h in b}
        aset = {h.sample_token for h in a}
        hits += len(bset & aset)
        total += len(bset)
    return hits / max(1, total)


def _time_search(store, queries: np.ndarray, k: int, repeats: int = 3):
    # warmup
    store.search_batch(queries[:1], k)
    best = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        results = store.search_batch(queries, k)
        elapsed = time.perf_counter() - t0
        if best is None or elapsed < best[0]:
            best = (elapsed, results)
    elapsed, results = best
    return elapsed * 1000.0 / max(1, len(queries)), results


def benchmark_one(corpus: np.ndarray, queries: np.ndarray, k: int = 10) -> BenchPoint:
    n, dim = corpus.shape
    payloads = [{"sample_token": f"s{i:07d}"} for i in range(n)]

    np_store = NumpyVectorStore(dim=dim, name="bench-numpy")
    np_store.add(corpus, payloads)
    fa_store = FaissVectorStore(dim=dim, name="bench-faiss")
    fa_store.add(corpus, payloads)
    info = fa_store.finalize()

    brute_ms, brute_results = _time_search(np_store, queries, k)
    faiss_ms, faiss_results = _time_search(fa_store, queries, k)
    recall = _recall_at_k(brute_results, faiss_results)
    return BenchPoint(
        n=n,
        dim=dim,
        k=k,
        n_queries=len(queries),
        brute_ms_per_query=round(brute_ms, 3),
        faiss_ms_per_query=round(faiss_ms, 3),
        speedup=round(brute_ms / max(1e-6, faiss_ms), 2),
        recall_at_k=round(recall, 4),
        faiss_index=info["index_type"],
        faiss_params={k_: info[k_] for k_ in ("nlist", "m", "nbits", "nprobe") if k_ in info},
    )


def run_benchmark(
    sizes: tuple[int, ...] = (1_000, 5_000, 10_000),
    dim: int = 384,
    n_queries: int = 200,
    k: int = 10,
    seed: int = 0,
) -> list[dict]:
    """Run the benchmark on synthetic Gaussian vectors at several sizes.

    Synthetic vectors are *anisotropic clusters* (10 random centers + noise),
    which is closer to real embedding distributions than uniform noise and
    gives IVF something meaningful to cluster on.
    """
    rng = np.random.default_rng(seed)
    points = []
    for n in sizes:
        n_centers = max(8, n // 200)
        centers = rng.standard_normal((n_centers, dim)).astype(np.float32) * 2.0
        labels = rng.integers(0, n_centers, size=n)
        corpus = centers[labels] + rng.standard_normal((n, dim)).astype(np.float32) * 0.3
        corpus = _normalize(corpus)
        # queries: pick random corpus points + perturb (realistic "find similar")
        q_idx = rng.integers(0, n, size=n_queries)
        queries = corpus[q_idx] + rng.standard_normal((n_queries, dim)).astype(np.float32) * 0.05
        queries = _normalize(queries)

        log.info("Benchmark n=%d dim=%d ...", n, dim)
        try:
            point = benchmark_one(corpus, queries, k=k)
            log.info("  brute=%.2f ms/q  faiss=%.2f ms/q  speedup=%.1fx  recall@%d=%.3f  (%s)",
                     point.brute_ms_per_query, point.faiss_ms_per_query,
                     point.speedup, k, point.recall_at_k, point.faiss_index)
            points.append(asdict(point))
        except Exception as exc:  # noqa: BLE001
            log.warning("Benchmark at n=%d failed: %s", n, exc)
    return points
