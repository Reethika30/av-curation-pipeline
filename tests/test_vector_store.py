"""Unit tests for vector_store backends and the ANN benchmark."""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.vector_store import (
    FaissVectorStore,
    NumpyVectorStore,
    _suggest_ivf_params,
    make_store,
)


def _gaussian_clusters(n: int, dim: int, n_centers: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_centers, dim)).astype(np.float32) * 2.0
    labels = rng.integers(0, n_centers, size=n)
    x = centers[labels] + rng.standard_normal((n, dim)).astype(np.float32) * 0.3
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x


def test_numpy_store_returns_topk_in_order():
    x = _gaussian_clusters(500, 64)
    store = NumpyVectorStore(dim=64, name="t")
    store.add(x, [{"sample_token": f"s{i}"} for i in range(len(x))])
    hits = store.search(x[42], k=5)
    assert len(hits) == 5
    # query identical to a stored vector → first hit should be that vector
    assert hits[0].sample_token == "s42"
    # scores must be monotonically non-increasing
    assert all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1))


def test_numpy_store_search_batch_matches_search():
    x = _gaussian_clusters(200, 32, seed=1)
    store = NumpyVectorStore(dim=32, name="t")
    store.add(x, [{"sample_token": f"s{i}"} for i in range(len(x))])
    queries = x[:10]
    batch = store.search_batch(queries, k=5)
    one = [store.search(q, k=5) for q in queries]
    for a, b in zip(batch, one):
        assert [h.sample_token for h in a] == [h.sample_token for h in b]


def test_suggest_ivf_params_reasonable():
    nlist, m, nbits = _suggest_ivf_params(10_000, 384)
    assert nlist >= 8
    assert 384 % m == 0
    assert nbits == 8
    # tiny corpus → nlist clamped low
    nlist_small, _, _ = _suggest_ivf_params(100, 64)
    assert nlist_small <= 16


def test_make_store_falls_back_to_numpy_for_small_n():
    s = make_store(dim=64, name="t", backend="auto", n_hint=500)
    assert s.backend == "numpy"


# FAISS-only tests below — skip cleanly if faiss isn't installed.
faiss = pytest.importorskip("faiss")


def test_faiss_store_recall_high_on_clustered_data():
    n, dim = 5000, 96
    x = _gaussian_clusters(n, dim, n_centers=20, seed=7)
    queries = x[:100]
    fa = FaissVectorStore(dim=dim, name="t", nprobe=16)
    fa.add(x, [{"sample_token": f"s{i}"} for i in range(n)])
    info = fa.finalize()
    assert info["index_type"] == "IndexIVFFlat"
    np_store = NumpyVectorStore(dim=dim, name="t")
    np_store.add(x, [{"sample_token": f"s{i}"} for i in range(n)])

    brute = np_store.search_batch(queries, k=10)
    approx = fa.search_batch(queries, k=10)
    hits = sum(
        len({h.sample_token for h in b} & {h.sample_token for h in a})
        for b, a in zip(brute, approx)
    )
    recall = hits / (10 * len(queries))
    assert recall >= 0.9, f"IVFFlat recall@10 too low: {recall:.3f}"


def test_faiss_ivfpq_flavor_trains():
    n, dim = 5000, 96
    x = _gaussian_clusters(n, dim, n_centers=20, seed=11)
    fa = FaissVectorStore(dim=dim, name="t", flavor="ivfpq", nprobe=16)
    fa.add(x, [{"sample_token": f"s{i}"} for i in range(n)])
    info = fa.finalize()
    assert info["index_type"] == "IndexIVFPQ"
    # PQ is lossy; just verify search returns plausible results
    hits = fa.search(x[10], k=10)
    assert len(hits) == 10
    # the query is a stored vector — at least one hit should be a near neighbor
    assert hits[0].score > 0.5


def test_faiss_store_falls_back_to_flat_for_tiny_corpus():
    fa = FaissVectorStore(dim=32, name="t")
    fa.add(np.random.rand(50, 32).astype(np.float32),
           [{"sample_token": f"s{i}"} for i in range(50)])
    info = fa.finalize()
    assert info["index_type"] == "IndexFlatIP"
    # search still works
    hits = fa.search(np.random.rand(32).astype(np.float32), k=5)
    assert len(hits) == 5
