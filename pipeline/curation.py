"""Curation algorithms: near-duplicate detection, outlier surfacing,
failure-mode clustering, and stratified curation-set assembly."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Near-duplicate detection
# ---------------------------------------------------------------------------
@dataclass
class DuplicateGroup:
    representative: str
    members: list[str]
    mean_similarity: float


def find_near_duplicates(
    embeddings: np.ndarray,
    tokens: Sequence[str],
    threshold: float = 0.97,
) -> list[DuplicateGroup]:
    """Greedy union-find on cosine similarity. O(N^2) — fine for ≤ 50K
    samples; production code would shard via LanceDB ANN + verify."""
    if len(embeddings) == 0:
        return []
    norms = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    sims = norms @ norms.T
    np.fill_diagonal(sims, -1.0)
    n = len(tokens)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    pairs = np.argwhere(sims >= threshold)
    for i, j in pairs:
        if i < j:
            union(int(i), int(j))

    groups: dict[int, list[int]] = {}
    for idx in range(n):
        groups.setdefault(find(idx), []).append(idx)

    out: list[DuplicateGroup] = []
    for members in groups.values():
        if len(members) < 2:
            continue
        sub = sims[np.ix_(members, members)]
        mean_sim = float((sub.sum() + len(members)) / (len(members) ** 2))  # +1 to undo diag
        rep = tokens[members[0]]
        out.append(DuplicateGroup(
            representative=rep,
            members=[tokens[m] for m in members],
            mean_similarity=mean_sim,
        ))
    out.sort(key=lambda g: -len(g.members))
    return out


# ---------------------------------------------------------------------------
# Outlier surfacing (k-NN distance)
# ---------------------------------------------------------------------------
@dataclass
class Outlier:
    sample_token: str
    knn_distance: float
    rank: int


def find_outliers(
    embeddings: np.ndarray,
    tokens: Sequence[str],
    k: int = 10,
    top_pct: float = 0.05,
) -> list[Outlier]:
    if len(embeddings) <= k:
        return []
    norms = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    sims = norms @ norms.T
    np.fill_diagonal(sims, 1.0)
    # distance = 1 - sim; k-th nearest distance
    dists = 1.0 - sims
    np.fill_diagonal(dists, 0.0)
    kth = np.partition(dists, k, axis=1)[:, k]
    n_top = max(1, int(len(tokens) * top_pct))
    order = np.argsort(-kth)[:n_top]
    return [Outlier(tokens[i], float(kth[i]), rank) for rank, i in enumerate(order)]


# ---------------------------------------------------------------------------
# Failure-mode clustering
# ---------------------------------------------------------------------------
@dataclass
class Cluster:
    cluster_id: int
    label: str
    size: int
    members: list[str]
    centroid_2d: tuple[float, float]
    sample_2d: list[tuple[str, float, float]] = field(default_factory=list)


_FAILURE_PROMPTS = [
    "a clear daytime urban street",
    "a rainy night intersection with reflections",
    "a foggy highway with low visibility",
    "a construction zone with cones and barriers",
    "a tunnel with artificial lighting",
    "a dense traffic jam in a city",
    "a quiet residential road",
    "a parking lot with many vehicles",
    "a highway merge with fast traffic",
    "an empty road at dawn or dusk",
]


def cluster_failure_modes(
    clip_embeddings: np.ndarray,
    tokens: Sequence[str],
    encoder=None,
    min_cluster_size: int = 5,
) -> tuple[list[Cluster], np.ndarray]:
    """UMAP → HDBSCAN; label each cluster by nearest CLIP text prompt.

    Returns (clusters, umap_xy).
    """
    from sklearn.cluster import KMeans

    # 2D projection for the UI
    try:
        import umap  # type: ignore

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42, n_components=2)
        xy = reducer.fit_transform(clip_embeddings)
    except Exception as exc:  # noqa: BLE001
        log.warning("UMAP unavailable (%s); falling back to PCA", exc)
        from sklearn.decomposition import PCA

        xy = PCA(n_components=2, random_state=42).fit_transform(clip_embeddings)

    # Clustering — done in UMAP space because synthetic / domain-similar imagery
    # produces a dense CLIP cloud where HDBSCAN collapses everything into noise
    # or one giant cluster. UMAP gives separable structure.
    labels: np.ndarray
    try:
        import hdbscan  # type: ignore

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=1,
            cluster_selection_epsilon=0.3,
            prediction_data=False,
        )
        labels = clusterer.fit_predict(xy.astype(np.float64))
        if (labels >= 0).sum() < min_cluster_size or len(set(labels[labels >= 0])) < 2:
            raise RuntimeError("HDBSCAN found no clusters; falling back")
    except Exception as exc:  # noqa: BLE001
        log.warning("HDBSCAN unavailable / weak (%s); using KMeans(k=8)", exc)
        k = min(8, max(2, len(tokens) // max(min_cluster_size, 1)))
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(clip_embeddings)

    # Label clusters via nearest text prompt
    if encoder is not None:
        try:
            text_emb = encoder.encode_text(_FAILURE_PROMPTS)
        except Exception:  # noqa: BLE001
            text_emb = None
    else:
        text_emb = None

    clusters: list[Cluster] = []
    for cid in sorted(set(int(l) for l in labels)):
        if cid < 0:
            continue
        mask = labels == cid
        members = [tokens[i] for i in np.where(mask)[0]]
        centroid = clip_embeddings[mask].mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-8
        if text_emb is not None:
            scores = text_emb @ centroid
            label = _FAILURE_PROMPTS[int(np.argmax(scores))]
        else:
            label = f"cluster {cid}"
        cx, cy = xy[mask].mean(axis=0)
        sample_idx = np.where(mask)[0][:25]
        sample_2d = [(tokens[i], float(xy[i, 0]), float(xy[i, 1])) for i in sample_idx]
        clusters.append(
            Cluster(
                cluster_id=int(cid),
                label=label,
                size=int(mask.sum()),
                members=members,
                centroid_2d=(float(cx), float(cy)),
                sample_2d=sample_2d,
            )
        )
    clusters.sort(key=lambda c: -c.size)
    return clusters, xy


# ---------------------------------------------------------------------------
# Curation set assembly
# ---------------------------------------------------------------------------
@dataclass
class CurationSet:
    kept: list[str]
    dropped_duplicates: list[str]
    boosted_outliers: list[str]
    per_cluster_kept: dict[int, int]


def assemble_curation_set(
    tokens: Sequence[str],
    duplicates: list[DuplicateGroup],
    outliers: list[Outlier],
    clusters: list[Cluster],
    target_size: int,
) -> CurationSet:
    """Stratified across clusters; keeps cluster representatives, drops
    duplicate followers, boosts outliers."""
    drop = set()
    for g in duplicates:
        for m in g.members:
            if m != g.representative:
                drop.add(m)

    surviving = [t for t in tokens if t not in drop]
    if not clusters:
        kept = surviving[:target_size]
        return CurationSet(kept=kept, dropped_duplicates=sorted(drop),
                           boosted_outliers=[], per_cluster_kept={})

    # Quota per cluster proportional to size
    total = sum(c.size for c in clusters)
    kept: list[str] = []
    per_cluster: dict[int, int] = {}
    for c in clusters:
        quota = max(1, round(target_size * c.size / total))
        members_alive = [m for m in c.members if m not in drop]
        take = members_alive[:quota]
        kept.extend(take)
        per_cluster[c.cluster_id] = len(take)

    boosted = [o.sample_token for o in outliers if o.sample_token not in drop and o.sample_token not in kept]
    kept.extend(boosted[: max(1, target_size // 10)])

    # Truncate / dedupe
    seen = set()
    kept_unique = []
    for t in kept:
        if t not in seen:
            seen.add(t)
            kept_unique.append(t)
    kept_unique = kept_unique[:target_size]

    return CurationSet(
        kept=kept_unique,
        dropped_duplicates=sorted(drop),
        boosted_outliers=boosted,
        per_cluster_kept=per_cluster,
    )
