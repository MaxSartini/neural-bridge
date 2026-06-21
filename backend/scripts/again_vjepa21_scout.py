"""V-JEPA 2.1 MLX scout loading and feature helpers for AGAIN."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from backend.scripts.again_scout_sparse_pipeline import ScoutModelSpec, model_registry, safe_float


@dataclass(frozen=True)
class ScoutWindowFingerprint:
    dataset_name: str
    video_id: str
    clip_start_seconds: float
    clip_end_seconds: float
    scout_model_name: str
    checkpoint_sha256: str
    frame_count: int
    resolution: int
    stride_seconds: float
    preprocessing_version: str = "vjepa21_scout_384_center_crop_v1"

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.__dict__, sort_keys=True).encode("utf-8")).hexdigest()


def require_vjepa21_scout(spec: ScoutModelSpec) -> None:
    if spec.vjepa_version != "2.1":
        raise ValueError(f"Only V-JEPA 2.1 scout specs are allowed, got {spec.vjepa_version}")
    if spec.status != "ready":
        raise FileNotFoundError(f"Scout model is not ready: {spec.name} status={spec.status}")
    if "vjepa2-vitl-fpc64-256" in json.dumps(spec.__dict__).lower():
        raise ValueError("V-JEPA 2.x 256 community weights are not allowed for the V-JEPA 2.1 scout path")


def scout_spec_by_name(name: str) -> ScoutModelSpec:
    for spec in model_registry():
        if spec.name == name:
            require_vjepa21_scout(spec)
            return spec
    raise KeyError(f"Unknown scout model: {name}")


def load_lukasugar_vitb_model(spec: ScoutModelSpec, *, repo_root: Path) -> Any:
    """Load the converted ViT-B scout through lukasugar/vjepa2.1-mlx."""
    require_vjepa21_scout(spec)
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from vjepa2_1_mlx.models.vision_transformer import build_vjepa2_1_model  # noqa: WPS433
    import mlx.core as mx  # noqa: WPS433

    weights_path = Path(spec.mlx_path)
    if not weights_path.exists():
        raise FileNotFoundError(weights_path)
    model = build_vjepa2_1_model(spec.model_name)
    model.load_weights(str(weights_path), strict=True)
    mx.eval(model.parameters())
    return model


def load_dgrauet_vitl_model(spec: ScoutModelSpec, *, repo_root: Path) -> Any:
    """Load the preconverted ViT-L fallback scout through dgrauet/vjepa2-mlx."""
    require_vjepa21_scout(spec)
    src = repo_root / "packages" / "vjepa2-core-mlx" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from vjepa2_core_mlx.utils.weights import from_pretrained  # noqa: WPS433

    return from_pretrained(spec.mlx_path)


def pool_scout_tokens(output: np.ndarray) -> np.ndarray:
    """Return one pooled embedding per input window."""
    arr = np.asarray(output)
    if arr.ndim == 3:
        return np.mean(arr, axis=1)
    if arr.ndim == 2:
        return arr
    if arr.ndim > 3:
        return arr.reshape(arr.shape[0], -1, arr.shape[-1]).mean(axis=1)
    raise ValueError(f"Unsupported scout output shape: {arr.shape}")


def embedding_delta(previous: np.ndarray | None, current: np.ndarray) -> float:
    if previous is None:
        return 0.0
    prev = np.asarray(previous, dtype=np.float64).reshape(-1)
    curr = np.asarray(current, dtype=np.float64).reshape(-1)
    if prev.shape != curr.shape:
        raise ValueError(f"Embedding shape mismatch: {prev.shape} vs {curr.shape}")
    return float(np.linalg.norm(curr - prev) / max(1, curr.size))


def scout_window_fingerprint(
    *,
    spec: ScoutModelSpec,
    video_id: str,
    clip_start_seconds: float,
    clip_end_seconds: float,
    frame_count: int,
    resolution: int,
    stride_seconds: float,
) -> ScoutWindowFingerprint:
    require_vjepa21_scout(spec)
    return ScoutWindowFingerprint(
        dataset_name="AGAIN_cleaned",
        video_id=video_id,
        clip_start_seconds=round(safe_float(clip_start_seconds), 4),
        clip_end_seconds=round(safe_float(clip_end_seconds), 4),
        scout_model_name=spec.name,
        checkpoint_sha256=spec.checkpoint_sha256,
        frame_count=int(frame_count),
        resolution=int(resolution),
        stride_seconds=float(stride_seconds),
    )
