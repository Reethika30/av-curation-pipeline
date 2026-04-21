"""Precompute pipeline → bakes JSON + thumbnails into web/public/data/.

Usage::

    python -m precompute.run --source synthetic --n 400
    python -m precompute.run --source nuscenes --dataroot ./data/nuscenes

After this, the Next.js app reads everything statically — Vercel needs no
backend.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np

from pipeline import curation, embeddings, lineage, loaders, vector_store

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DATA = REPO_ROOT / "web" / "public" / "data"
THUMBS_DIR = REPO_ROOT / "web" / "public" / "thumbs"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
log = logging.getLogger("precompute")


def _thumb_b64(img, size: int = 96) -> str:
    im = img.copy()
    im.thumbnail((size, size))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=72, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _save_thumb(img, token: str, size: int = 192) -> str:
    THUMBS_DIR.mkdir(parents=True, exist_ok=True)
    out = THUMBS_DIR / f"{token}.jpg"
    im = img.copy()
    im.thumbnail((size, size))
    im.save(out, format="JPEG", quality=78, optimize=True)
    return f"/thumbs/{out.name}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["synthetic", "nuscenes"], default="synthetic")
    p.add_argument("--n", type=int, default=400, help="synthetic samples to generate")
    p.add_argument("--dataroot", type=str, default="./data/nuscenes")
    p.add_argument("--encoder", choices=["auto", "torch", "fallback"], default="auto")
    p.add_argument("--dup-threshold", type=float, default=0.97)
    p.add_argument("--target-size", type=int, default=120)
    p.add_argument("--max-samples", type=int, default=None,
                   help="cap on nuScenes samples loaded")
    args = p.parse_args()

    WEB_DATA.mkdir(parents=True, exist_ok=True)
    THUMBS_DIR.mkdir(parents=True, exist_ok=True)

    # ----- 1. Load -----
    log.info("Loading samples (source=%s)", args.source)
    if args.source == "synthetic":
        samples = loaders.synthetic_samples(n=args.n)
    else:
        samples = list(loaders.nuscenes_samples(args.dataroot, max_samples=args.max_samples))
    if not samples:
        raise SystemExit("No samples loaded — check --dataroot or use --source synthetic.")
    log.info("Loaded %d samples across %d scenes",
             len(samples), len({s.scene_name for s in samples}))

    # Persist the metadata table as parquet (a peek at the data-eng layer)
    table = loaders.samples_to_arrow(samples)
    import pyarrow.parquet as pq

    parquet_dir = REPO_ROOT / "data" / "tables"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, parquet_dir / "samples.parquet")
    log.info("Wrote %s", parquet_dir / "samples.parquet")

    # ----- 2. Embed -----
    encoder = embeddings.get_encoder(prefer=args.encoder)
    log.info("Encoder backend: %s", encoder.__class__.__name__)
    encoded = encoder.encode_images([s.image for s in samples])

    np.savez_compressed(WEB_DATA / "clip_embeddings.npz",
                        clip=encoded.clip_image,
                        dino=encoded.dinov2)

    # ----- 3. Vector index -----
    store = vector_store.VectorStore(
        dim=encoded.clip_image.shape[1],
        name="clip_image",
    )
    payloads = [
        {"sample_token": s.sample_token, "scene_name": s.scene_name,
         "weather": s.metadata["weather"], "time_of_day": s.metadata["time_of_day"]}
        for s in samples
    ]
    store.add(encoded.clip_image, payloads)
    log.info("Vector store backend: %s", store.backend)

    # ----- 4. Curation -----
    tokens = [s.sample_token for s in samples]
    log.info("Finding near-duplicates (cos >= %.2f) ...", args.dup_threshold)
    dup_groups = curation.find_near_duplicates(
        encoded.dinov2, tokens, threshold=args.dup_threshold
    )
    log.info("Surfacing outliers ...")
    outliers = curation.find_outliers(encoded.clip_image, tokens)
    log.info("Clustering failure modes ...")
    clusters, xy = curation.cluster_failure_modes(
        encoded.clip_image, tokens, encoder=encoder
    )
    log.info("Assembling curation set (target=%d) ...", args.target_size)
    curated = curation.assemble_curation_set(
        tokens, dup_groups, outliers, clusters, target_size=args.target_size
    )

    # ----- 5. Bake artifacts for the frontend -----
    log.info("Writing thumbnails for %d samples ...", len(samples))
    sample_records = []
    for s, (x, y) in zip(samples, xy):
        thumb = _save_thumb(s.image, s.sample_token)
        sample_records.append({
            "sample_token": s.sample_token,
            "scene_name": s.scene_name,
            "weather": s.metadata["weather"],
            "time_of_day": s.metadata["time_of_day"],
            "location": s.metadata["location"],
            "ego_speed_mps": round(s.metadata["ego_speed_mps"], 2),
            "thumb": thumb,
            "x": float(x),
            "y": float(y),
        })

    (WEB_DATA / "samples.json").write_text(json.dumps(sample_records), encoding="utf-8")
    (WEB_DATA / "duplicates.json").write_text(
        json.dumps([asdict(g) for g in dup_groups]), encoding="utf-8"
    )
    (WEB_DATA / "outliers.json").write_text(
        json.dumps([asdict(o) for o in outliers]), encoding="utf-8"
    )
    (WEB_DATA / "clusters.json").write_text(
        json.dumps([asdict(c) for c in clusters]), encoding="utf-8"
    )

    n_in_dups = sum(len(g.members) for g in dup_groups)
    summary = {
        "source": args.source,
        "encoder_backend": encoded.backend,
        "vector_backend": store.backend,
        "n_input_samples": len(samples),
        "n_scenes": len({s.scene_name for s in samples}),
        "n_duplicate_groups": len(dup_groups),
        "n_samples_in_duplicate_groups": n_in_dups,
        "duplicate_rate_pct": round(100.0 * n_in_dups / max(1, len(samples)), 2),
        "n_outliers": len(outliers),
        "n_clusters": len(clusters),
        "n_curated": len(curated.kept),
        "duplicate_threshold": args.dup_threshold,
        "target_size": args.target_size,
        "headline_metric": (
            f"Curated {len(samples)} raw frames → {len(curated.kept)} "
            f"high-value training samples; flagged {round(100 * n_in_dups / max(1, len(samples)), 1)}% "
            f"near-duplicates; surfaced {len(clusters)} failure-mode clusters."
        ),
    }
    (WEB_DATA / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (WEB_DATA / "curation_set.json").write_text(
        json.dumps({
            "kept": curated.kept,
            "dropped_duplicates": curated.dropped_duplicates,
            "boosted_outliers": curated.boosted_outliers,
            "per_cluster_kept": curated.per_cluster_kept,
        }), encoding="utf-8",
    )

    # ----- 6. Lineage -----
    record = lineage.make_record(
        source=args.source,
        encoder_backend=encoded.backend,
        n_input=len(samples),
        curated_tokens=curated.kept,
        n_dup_groups=len(dup_groups),
        n_outliers=len(outliers),
        n_clusters=len(clusters),
        duplicate_threshold=args.dup_threshold,
        target_size=args.target_size,
        repo_root=REPO_ROOT,
        extras={"vector_backend": store.backend},
    )
    lineage_path = WEB_DATA / "lineage.json"
    lineage.append_lineage(record, lineage_path)

    log.info("\n%s", json.dumps(summary, indent=2))
    log.info("Artifacts written to %s", WEB_DATA)


if __name__ == "__main__":
    main()
