"""Lineage tracking. Records each pipeline run's manifest hash, git SHA,
input source, encoder backend, parameters, and output stats so any
downstream model can be traced back to the exact data slice it saw."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class LineageRecord:
    run_id: str
    timestamp_utc: str
    git_sha: str | None
    python_version: str
    encoder_backend: str
    source: str
    n_input_samples: int
    n_curated_samples: int
    n_duplicate_groups: int
    n_outliers: int
    n_clusters: int
    duplicate_threshold: float
    target_size: int
    manifest_sha256: str
    extras: dict = field(default_factory=dict)


def _git_sha(cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=5
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return None


def _manifest_sha(tokens: list[str]) -> str:
    h = hashlib.sha256()
    for t in sorted(tokens):
        h.update(t.encode())
        h.update(b"\n")
    return h.hexdigest()


def make_record(
    *, source: str, encoder_backend: str, n_input: int, curated_tokens: list[str],
    n_dup_groups: int, n_outliers: int, n_clusters: int, duplicate_threshold: float,
    target_size: int, repo_root: Path, extras: dict | None = None,
) -> LineageRecord:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = _manifest_sha(curated_tokens)
    run_id = f"run-{ts}-{manifest[:8]}"
    return LineageRecord(
        run_id=run_id,
        timestamp_utc=ts,
        git_sha=_git_sha(repo_root),
        python_version=platform.python_version(),
        encoder_backend=encoder_backend,
        source=source,
        n_input_samples=n_input,
        n_curated_samples=len(curated_tokens),
        n_duplicate_groups=n_dup_groups,
        n_outliers=n_outliers,
        n_clusters=n_clusters,
        duplicate_threshold=duplicate_threshold,
        target_size=target_size,
        manifest_sha256=manifest,
        extras=extras or {},
    )


def append_lineage(record: LineageRecord, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    if path.exists():
        try:
            history = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            history = []
    history.append(asdict(record))
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")
