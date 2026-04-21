"""Dataset loaders.

Two sources are supported:

* ``synthetic`` — generates a deterministic, in-memory mini AV dataset that
  mimics the *shape* of nuScenes (synced multi-camera + LiDAR metadata, scene
  attributes, weather/time-of-day labels, plausible duplicate clusters).
  Useful for CI, Vercel demos, and machines without ~4 GB of free disk.

* ``nuscenes`` — wraps :mod:`nuscenes-devkit` against the *mini* split.

Both produce the same :class:`pyarrow.Table` schema so downstream code is
source-agnostic.
"""
from __future__ import annotations

import hashlib
import io
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow as pa
from PIL import Image, ImageDraw, ImageFilter
# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
SAMPLE_SCHEMA = pa.schema(
    [
        ("sample_token", pa.string()),
        ("scene_token", pa.string()),
        ("scene_name", pa.string()),
        ("timestamp", pa.int64()),
        ("camera", pa.string()),
        ("cam_path", pa.string()),
        ("lidar_path", pa.string()),
        ("location", pa.string()),
        ("weather", pa.string()),
        ("time_of_day", pa.string()),
        ("ego_speed_mps", pa.float32()),
        ("num_lidar_pts", pa.int32()),
    ]
)


@dataclass
class LoadedSample:
    """One synced multi-sensor sample, with the front-camera image in memory."""

    sample_token: str
    scene_name: str
    image: Image.Image
    metadata: dict


# ---------------------------------------------------------------------------
# Synthetic loader
# ---------------------------------------------------------------------------
_LOCATIONS = ["boston-seaport", "singapore-onenorth", "singapore-queenstown", "singapore-hollandvillage"]
_WEATHER = ["clear", "clear", "clear", "rain", "rain", "fog"]
_TOD = ["day", "day", "day", "night", "night", "dusk"]
_SCENE_TEMPLATES = [
    ("urban_intersection", (60, 90, 130)),
    ("highway_merge", (40, 60, 80)),
    ("tunnel", (15, 15, 25)),
    ("construction_zone", (180, 140, 60)),
    ("residential", (90, 130, 90)),
    ("parking_lot", (120, 120, 120)),
]


def _seeded_rng(token: str) -> random.Random:
    h = int(hashlib.md5(token.encode()).hexdigest()[:8], 16)
    return random.Random(h)


def _synthesize_image(token: str, scene: str, base_color: tuple[int, int, int],
                       weather: str, tod: str, size: int = 224) -> Image.Image:
    """Render a deterministic, visually distinctive 'driving scene' image.

    The point isn't realism — it's that DINOv2/CLIP get a stable, varied signal
    so clustering, duplicate detection and outlier surfacing produce meaningful
    output without needing real data.
    """
    rng = _seeded_rng(token)
    # Per-sample base-color jitter so non-duplicates spread in embedding space
    jitter = lambda c: max(0, min(255, c + rng.randint(-25, 25)))
    base = tuple(jitter(c) for c in base_color)
    img = Image.new("RGB", (size, size), base)
    draw = ImageDraw.Draw(img)

    # Sky gradient band
    for i in range(size // 2):
        shade = int(base[0] * (0.6 + 0.4 * i / (size / 2)))
        draw.line([(0, i), (size, i)], fill=(shade, shade + 5, shade + 15))

    # Time-of-day tinting
    if tod == "night":
        img = Image.eval(img, lambda v: int(v * 0.35))
        draw = ImageDraw.Draw(img)
    elif tod == "dusk":
        img = Image.eval(img, lambda v: int(v * 0.7))
        draw = ImageDraw.Draw(img)

    # Road
    horizon = size // 2 + rng.randint(-10, 10)
    draw.polygon(
        [(size * 0.3, horizon), (size * 0.7, horizon), (size, size), (0, size)],
        fill=(50, 50, 55),
    )
    # Lane markings
    for i in range(3):
        y = horizon + (size - horizon) * (0.2 + 0.3 * i)
        x_off = (size - horizon) * 0.05 * (i + 1)
        draw.line(
            [(size / 2 - x_off, y), (size / 2 + x_off, y + 6)],
            fill=(230, 220, 90), width=2,
        )

    # Vehicles (boxes) — count varies by scene
    n_vehicles = {"urban_intersection": 6, "highway_merge": 4, "tunnel": 2,
                  "construction_zone": 3, "residential": 1, "parking_lot": 8}.get(scene, 3)
    n_vehicles += rng.randint(-1, 2)
    n_vehicles = max(0, n_vehicles)
    for _ in range(n_vehicles):
        vx = rng.randint(int(size * 0.15), int(size * 0.85))
        vy = rng.randint(horizon + 4, size - 20)
        vw = rng.randint(14, 40)
        vh = rng.randint(10, 26)
        color = (rng.randint(40, 230), rng.randint(40, 230), rng.randint(40, 230))
        draw.rectangle([vx, vy, vx + vw, vy + vh], fill=color, outline=(20, 20, 20))
        # window
        draw.rectangle([vx + 2, vy + 2, vx + vw - 2, vy + vh // 2],
                       fill=(min(255, color[0] + 30),
                             min(255, color[1] + 30),
                             min(255, color[2] + 30)))

    # Construction cones
    if scene == "construction_zone":
        for _ in range(5):
            cx = rng.randint(int(size * 0.3), int(size * 0.7))
            cy = rng.randint(horizon + 10, size - 10)
            draw.polygon(
                [(cx, cy - 8), (cx - 4, cy + 4), (cx + 4, cy + 4)],
                fill=(255, 120, 0),
            )

    # Weather effects
    if weather == "rain":
        for _ in range(120):
            x = rng.randint(0, size)
            y = rng.randint(0, size)
            draw.line([(x, y), (x - 2, y + 6)], fill=(180, 200, 230), width=1)
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    elif weather == "fog":
        fog = Image.new("RGB", (size, size), (200, 200, 205))
        img = Image.blend(img, fog, 0.45)

    # Headlights at night
    if tod == "night":
        for _ in range(3):
            hx = rng.randint(int(size * 0.2), int(size * 0.8))
            hy = rng.randint(horizon + 8, size - 30)
            draw = ImageDraw.Draw(img)
            draw.ellipse([hx - 6, hy - 6, hx + 6, hy + 6], fill=(255, 240, 180))

    # Buildings on horizon (variable skyline)
    n_bld = rng.randint(2, 7)
    for _ in range(n_bld):
        bx = rng.randint(0, size)
        bw = rng.randint(20, 60)
        bh = rng.randint(20, max(21, horizon - 5))
        bcolor = (rng.randint(60, 140), rng.randint(60, 140), rng.randint(60, 150))
        draw.rectangle([bx, horizon - bh, bx + bw, horizon], fill=bcolor)

    # Per-sample pixel noise — pushes non-duplicates apart in DINOv2 space
    arr = np.asarray(img, dtype=np.int16)
    noise = np.random.default_rng(int(hashlib.md5(token.encode()).hexdigest()[:8], 16)) \
        .integers(-15, 16, arr.shape, dtype=np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    return img


def synthetic_samples(n: int = 400, n_scenes: int = 24,
                       duplicate_rate: float = 0.12, seed: int = 7) -> list[LoadedSample]:
    """Generate ``n`` samples spread across ``n_scenes``, with planted
    near-duplicates so curation has something to find."""
    rng = random.Random(seed)
    scenes: list[tuple[str, str, tuple[int, int, int], str, str, str]] = []
    for i in range(n_scenes):
        template, color = _SCENE_TEMPLATES[i % len(_SCENE_TEMPLATES)]
        scene_name = f"scene-{i:04d}-{template}"
        location = _LOCATIONS[i % len(_LOCATIONS)]
        weather = _WEATHER[i % len(_WEATHER)]
        tod = _TOD[i % len(_TOD)]
        scene_token = hashlib.md5(scene_name.encode()).hexdigest()[:16]
        scenes.append((scene_token, scene_name, color, location, weather, tod))

    samples: list[LoadedSample] = []
    base_ts = 1_700_000_000_000_000  # microseconds
    for idx in range(n):
        scene_token, scene_name, color, location, weather, tod = scenes[idx % n_scenes]
        is_dup = rng.random() < duplicate_rate and idx > 5
        if is_dup:
            # near-duplicate of an earlier frame in same scene
            siblings = [s for s in samples if s.metadata["scene_name"] == scene_name]
            if siblings:
                sib = rng.choice(siblings)
                token = f"sample_{idx:05d}_dup_{sib.sample_token[-6:]}"
                # tiny perturbation -> still very high cosine sim
                image = sib.image.copy().filter(ImageFilter.GaussianBlur(radius=0.3))
            else:
                token = f"sample_{idx:05d}"
                template = scene_name.split("-")[-1]
                image = _synthesize_image(token, template, color, weather, tod)
        else:
            token = f"sample_{idx:05d}"
            template = scene_name.split("-")[-1]
            image = _synthesize_image(token, template, color, weather, tod)

        meta = {
            "sample_token": token,
            "scene_token": scene_token,
            "scene_name": scene_name,
            "timestamp": base_ts + idx * 500_000,
            "camera": "CAM_FRONT",
            "cam_path": f"synthetic://{token}/CAM_FRONT.jpg",
            "lidar_path": f"synthetic://{token}/LIDAR_TOP.bin",
            "location": location,
            "weather": weather,
            "time_of_day": tod,
            "ego_speed_mps": float(rng.uniform(0.0, 22.0)),
            "num_lidar_pts": int(rng.randint(20_000, 120_000)),
        }
        samples.append(LoadedSample(sample_token=token, scene_name=scene_name,
                                     image=image, metadata=meta))
    return samples


# ---------------------------------------------------------------------------
# nuScenes loader (real data)
# ---------------------------------------------------------------------------
def nuscenes_samples(dataroot: str | Path, version: str = "v1.0-mini",
                      max_samples: int | None = None) -> Iterator[LoadedSample]:
    """Yield :class:`LoadedSample` from a local nuScenes-mini install.

    Imports happen lazily so machines without ``nuscenes-devkit`` can still
    use the synthetic path.
    """
    from nuscenes.nuscenes import NuScenes  # type: ignore

    nusc = NuScenes(version=version, dataroot=str(dataroot), verbose=False)
    n_yielded = 0
    for sample in nusc.sample:
        if max_samples is not None and n_yielded >= max_samples:
            break
        cam_token = sample["data"]["CAM_FRONT"]
        cam = nusc.get("sample_data", cam_token)
        scene = nusc.get("scene", sample["scene_token"])
        log = nusc.get("log", scene["log_token"])
        cam_path = Path(dataroot) / cam["filename"]
        try:
            img = Image.open(cam_path).convert("RGB").resize((224, 224))
        except FileNotFoundError:
            continue
        meta = {
            "sample_token": sample["token"],
            "scene_token": scene["token"],
            "scene_name": scene["name"],
            "timestamp": int(sample["timestamp"]),
            "camera": "CAM_FRONT",
            "cam_path": str(cam_path),
            "lidar_path": str(Path(dataroot) / nusc.get("sample_data",
                                                          sample["data"]["LIDAR_TOP"])["filename"]),
            "location": log["location"],
            "weather": _infer_weather(scene["description"]),
            "time_of_day": _infer_tod(scene["description"]),
            "ego_speed_mps": 0.0,  # would join with ego_pose for real value
            "num_lidar_pts": 0,
        }
        yield LoadedSample(sample_token=sample["token"], scene_name=scene["name"],
                            image=img, metadata=meta)
        n_yielded += 1


def _infer_weather(desc: str) -> str:
    d = desc.lower()
    if "rain" in d:
        return "rain"
    if "fog" in d:
        return "fog"
    return "clear"


def _infer_tod(desc: str) -> str:
    d = desc.lower()
    if "night" in d:
        return "night"
    if "dusk" in d or "dawn" in d:
        return "dusk"
    return "day"


# ---------------------------------------------------------------------------
# Arrow conversion
# ---------------------------------------------------------------------------
def samples_to_arrow(samples: list[LoadedSample]) -> pa.Table:
    cols: dict[str, list] = {f.name: [] for f in SAMPLE_SCHEMA}
    for s in samples:
        for k in cols:
            cols[k].append(s.metadata[k])
    arrays = [pa.array(cols[f.name], type=f.type) for f in SAMPLE_SCHEMA]
    return pa.Table.from_arrays(arrays, schema=SAMPLE_SCHEMA)


def image_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
