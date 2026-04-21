"""Embedding models.

Wraps DINOv2 and CLIP behind a uniform interface. If the heavy ML libs are
unavailable (Python 3.14 wheel gaps, offline environment, etc.) we fall back
to a deterministic hash-based feature extractor so the rest of the pipeline
still produces meaningful, reproducible structure for the demo.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


@dataclass
class EncodedBatch:
    dinov2: np.ndarray  # (N, D_dino)
    clip_image: np.ndarray  # (N, D_clip)
    backend: str  # "torch" or "fallback"


# ---------------------------------------------------------------------------
# Real backend (DINOv2 + CLIP via transformers / open_clip)
# ---------------------------------------------------------------------------
class TorchEncoder:
    """Lazy-loads DINOv2 (ViT-S/14) and CLIP (ViT-B/32). CPU-friendly."""

    def __init__(self, device: str | None = None):
        import torch  # noqa: F401  -- imported here so import errors surface immediately
        from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

        self.torch = __import__("torch")
        self.device = device or ("cuda" if self.torch.cuda.is_available() else "cpu")

        log.info("Loading DINOv2 (facebook/dinov2-small) on %s", self.device)
        self.dino_proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        self.dino = AutoModel.from_pretrained("facebook/dinov2-small").to(self.device).eval()

        log.info("Loading CLIP (openai/clip-vit-base-patch32) on %s", self.device)
        self.clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device).eval()

    def encode_images(self, images: Sequence[Image.Image], batch_size: int = 16) -> EncodedBatch:
        torch = self.torch
        all_dino, all_clip = [], []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                chunk = list(images[i : i + batch_size])

                d_in = self.dino_proc(images=chunk, return_tensors="pt").to(self.device)
                d_out = self.dino(**d_in).last_hidden_state[:, 0]  # CLS token
                d_out = torch.nn.functional.normalize(d_out, dim=-1)
                all_dino.append(d_out.cpu().numpy().astype(np.float32))

                c_in = self.clip_proc(images=chunk, return_tensors="pt").to(self.device)
                c_out = self.clip.get_image_features(**c_in)
                # transformers v5 returns BaseModelOutputWithPooling; v4 returns a Tensor
                if hasattr(c_out, "image_embeds"):
                    c_out = c_out.image_embeds  # v4 projected
                elif hasattr(c_out, "pooler_output") and c_out.pooler_output is not None:
                    c_out = c_out.pooler_output  # v5 projected (512-d for ViT-B/32)
                elif hasattr(c_out, "last_hidden_state"):
                    c_out = c_out.last_hidden_state[:, 0]
                c_out = torch.nn.functional.normalize(c_out, dim=-1)
                all_clip.append(c_out.cpu().numpy().astype(np.float32))

        return EncodedBatch(
            dinov2=np.concatenate(all_dino, axis=0),
            clip_image=np.concatenate(all_clip, axis=0),
            backend="torch",
        )

    def encode_text(self, prompts: Sequence[str]) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            t_in = self.clip_proc(text=list(prompts), return_tensors="pt", padding=True).to(self.device)
            t_out = self.clip.get_text_features(**t_in)
            if hasattr(t_out, "text_embeds"):
                t_out = t_out.text_embeds
            elif hasattr(t_out, "pooler_output") and t_out.pooler_output is not None:
                t_out = t_out.pooler_output
            elif hasattr(t_out, "last_hidden_state"):
                t_out = t_out.last_hidden_state[:, 0]
            t_out = torch.nn.functional.normalize(t_out, dim=-1)
        return t_out.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Fallback backend (deterministic hash + color histogram)
# ---------------------------------------------------------------------------
class FallbackEncoder:
    """Pure-NumPy substitute. Encodes structural color statistics + a
    seeded random projection of the image bytes — enough to get visually
    similar frames close together and weather/time-of-day separating cleanly.
    """

    DINO_DIM = 384
    CLIP_DIM = 512

    def __init__(self):
        self._proj_dino = np.random.default_rng(11).standard_normal(
            (self.DINO_DIM, 64 * 3)
        ).astype(np.float32)
        self._proj_clip = np.random.default_rng(22).standard_normal(
            (self.CLIP_DIM, 64 * 3)
        ).astype(np.float32)

    @staticmethod
    def _features(img: Image.Image) -> np.ndarray:
        small = img.convert("RGB").resize((8, 8))
        arr = np.asarray(small, dtype=np.float32).reshape(-1) / 255.0
        return arr

    def encode_images(self, images: Sequence[Image.Image], batch_size: int = 16) -> EncodedBatch:
        feats = np.stack([self._features(im) for im in images], axis=0)  # (N, 192)
        d = feats @ self._proj_dino.T
        c = feats @ self._proj_clip.T
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-8
        c /= np.linalg.norm(c, axis=1, keepdims=True) + 1e-8
        return EncodedBatch(dinov2=d.astype(np.float32),
                            clip_image=c.astype(np.float32),
                            backend="fallback")

    def encode_text(self, prompts: Sequence[str]) -> np.ndarray:
        rng = np.random.default_rng(33)
        out = np.zeros((len(prompts), self.CLIP_DIM), dtype=np.float32)
        for i, p in enumerate(prompts):
            seed = int(hashlib.md5(p.lower().encode()).hexdigest()[:8], 16)
            local = np.random.default_rng(seed).standard_normal(self.CLIP_DIM)
            out[i] = local + 0.1 * rng.standard_normal(self.CLIP_DIM)
        out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-8
        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def get_encoder(prefer: str = "auto"):
    """Return the best available encoder.

    ``prefer`` may be ``"auto"``, ``"torch"`` or ``"fallback"``.
    """
    if prefer == "fallback":
        return FallbackEncoder()
    try:
        return TorchEncoder()
    except Exception as exc:  # noqa: BLE001
        if prefer == "torch":
            raise
        log.warning("Falling back to deterministic encoder (%s)", exc)
        return FallbackEncoder()
