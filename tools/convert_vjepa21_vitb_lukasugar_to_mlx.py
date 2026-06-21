#!/usr/bin/env python3
"""Convert official V-JEPA 2.1 ViT-B checkpoint into MLX safetensors."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.scripts.again_scout_sparse_pipeline import external_root, sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    root = external_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=root / "models" / "vjepa21-pytorch" / "scout" / "vitb" / "vjepa2_1_vitb_dist_vitG_384.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "models" / "vjepa21_mlx" / "scout" / "vitb",
    )
    parser.add_argument(
        "--lukasugar-repo",
        type=Path,
        default=PROJECT_ROOT / ".cache" / "vjepa21-mlx-repos" / "vjepa2.1-mlx",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.expanduser()
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "vjepa2_1_vitb_dist_vitG_384.safetensors"
    if destination.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite existing MLX weights: {destination}")
    if not source.exists():
        raise SystemExit(f"Source checkpoint missing: {source}")

    expected_size = 1664223428
    source_size = source.stat().st_size
    if source_size != expected_size:
        raise SystemExit(
            f"Source checkpoint size mismatch for V-JEPA 2.1 ViT-B: "
            f"expected {expected_size}, got {source_size}"
        )

    lukasugar_src = args.lukasugar_repo / "src"
    if str(lukasugar_src) not in sys.path:
        sys.path.insert(0, str(lukasugar_src))
    from vjepa2_1_mlx.utils.checkpoints import convert_checkpoint_to_mlx  # noqa: WPS433

    source_sha256 = sha256_file(source)
    convert_checkpoint_to_mlx(source, destination)
    converted_sha256 = sha256_file(destination)
    config = {
        "model_family": "V-JEPA",
        "vjepa_version": "2.1",
        "model_name": "vjepa2_1_vit_base_384",
        "checkpoint_name": "vjepa2_1_vitb_dist_vitG_384.pt",
        "tensor_layout": "vjepa2_1_lukasugar_mlx_port",
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "mlp_ratio": 4.0,
        "patch_size": 16,
        "tubelet_size": 2,
        "image_size": 384,
        "frames_per_clip": 64,
        "source_checkpoint_path": str(source),
        "source_checkpoint_sha256": source_sha256,
        "mlx_safetensors_path": str(destination),
        "mlx_safetensors_sha256": converted_sha256,
        "backend_repo": "lukasugar/vjepa2.1-mlx",
        "cuda_dependencies_installed": False,
        "mlx_optimized_output": True,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    manifest = {
        **config,
        "converted_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_size_bytes": source_size,
        "mlx_size_bytes": destination.stat().st_size,
        "intended_use": "AGAIN V-JEPA 2.1 ViT-B scout candidate-window selector",
    }
    (output_dir / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"converted={destination}")
    print(f"source_sha256={source_sha256}")
    print(f"mlx_sha256={converted_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
