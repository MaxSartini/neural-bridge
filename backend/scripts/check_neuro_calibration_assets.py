"""Check local assets used by the neuro-behaviour calibration harness."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.neuro_roi_calibrator import TRIBE_CORTICAL_VERTICES  # noqa: E402
from app.services.subcortical_roi_adapter import (  # noqa: E402
    SubcorticalRoiAdapter,
    TRIBE_SUBCORTICAL_VOXELS,
)


ROOT = Path(__file__).resolve().parents[2]


def file_info(path: Path, min_size_bytes: int = 1) -> dict:
    size = path.stat().st_size if path.exists() and path.is_file() else 0
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": size,
        "min_size_bytes": min_size_bytes,
        "complete": path.exists() and size >= min_size_bytes,
    }


def main() -> None:
    atlas_dir = ROOT / "models" / "neuro_atlases"
    subcortical_dir = ROOT / "models" / "tribe" / "loganf26-tribev2-subcortical"
    mlx_dir = ROOT / "models" / "tribe-mlx" / "zimengxiong-tribev2-mlx"
    external_root = Path("/Volumes/onn. Drive/Neural Bridge/models/subcortical-upstream")
    subcortical_projection = SubcorticalRoiAdapter().project(
        [[0.0] * TRIBE_SUBCORTICAL_VOXELS]
    )

    checks = {
        "destrieux_surface_atlas": {
            "path": str(atlas_dir),
            "exists": atlas_dir.exists(),
            "expected_cortical_vertices": TRIBE_CORTICAL_VERTICES,
        },
        "subcortical_head": {
            "dir": str(subcortical_dir),
            "files": {
                "best.safetensors": file_info(subcortical_dir / "best.safetensors", 1_322_000_000),
                "best.ckpt": file_info(subcortical_dir / "best.ckpt", 600_000_000),
                "config.yaml": file_info(subcortical_dir / "config.yaml"),
                "build_args.json": file_info(subcortical_dir / "build_args.json"),
                "eval.json": file_info(subcortical_dir / "eval.json"),
            },
            "atlas_contract": {
                "voxel_count": subcortical_projection["voxel_count"],
                "region_count": len(subcortical_projection["region_labels"]),
                "voxel_order": subcortical_projection["voxel_order"],
            },
        },
        "subcortical_upstream_encoders": {
            "text_qwen3_0_6b": {
                "path": str(external_root / "Qwen-Qwen3-0.6B"),
                "complete": (external_root / "Qwen-Qwen3-0.6B" / "model.safetensors").exists(),
            },
            "audio_w2v_bert_2": {
                "path": str(external_root / "facebook-w2v-bert-2.0"),
                "complete": (external_root / "facebook-w2v-bert-2.0" / "model.safetensors").exists(),
            },
            "video_vjepa2_vitl": {
                "path": str(external_root / "facebook-vjepa2-vitl-fpc64-256"),
                "complete": (external_root / "facebook-vjepa2-vitl-fpc64-256" / "model.safetensors").exists(),
            },
        },
        "tribev2_mlx": {
            "dir": str(mlx_dir),
            "files": {
                "tribev2_mlx_float32.npz": file_info(mlx_dir / "tribev2_mlx_float32.npz", 708_000_000),
                "config.json": file_info(mlx_dir / "config.json"),
                "README.md": file_info(mlx_dir / "README.md"),
            },
        },
    }

    print(json.dumps(checks, indent=2))

    required = [
        checks["destrieux_surface_atlas"]["exists"],
        checks["subcortical_head"]["files"]["config.yaml"]["complete"],
        checks["subcortical_head"]["files"]["best.ckpt"]["complete"],
        checks["subcortical_head"]["files"]["eval.json"]["complete"],
        checks["tribev2_mlx"]["files"]["config.json"]["complete"],
    ]
    if not all(required):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
