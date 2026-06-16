"""Convert the exact cortical HF V-JEPA2 ViT-G checkpoint to MLX safetensors.

Default mode is dry-run. Use --apply to write the converted checkpoint.

This targets the current cortical TRIBE video encoder:
facebook/vjepa2-vitg-fpc64-256, 64 frames, 256px, hidden 1408, 40 layers.

The xocialize/vjepa2-mlx HF-style model keeps split query/key/value keys. The
only tensor-layout conversion needed for the local HF safetensors is the Conv3D
patch embedding weight:

  PyTorch/HF: (out, in, kt, kh, kw)
  MLX:        (out, kt, kh, kw, in)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


DEFAULT_MODEL_DIR = (
    "/Volumes/onn. Drive/Neural Bridge/models/cortical-upstream/"
    "facebook-vjepa2-vitg-fpc64-256"
)
DEFAULT_OUT_DIR = "models/upstream-encoders-mlx/facebook-vjepa2-vitg-fpc64-256"

EXPECTED = {
    "hidden_size": 1408,
    "num_hidden_layers": 40,
    "num_attention_heads": 22,
    "frames_per_clip": 64,
    "image_size": 256,
    "tubelet_size": 2,
}


def load_config(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def validate_config(config: dict[str, Any]) -> list[str]:
    errors = []
    for key, expected in EXPECTED.items():
        observed = config.get(key)
        if observed != expected:
            errors.append(f"{key}: expected {expected!r}, observed {observed!r}")
    return errors


def convert_tensor(key: str, value: np.ndarray, dtype: str) -> np.ndarray:
    if key == "encoder.embeddings.patch_embeddings.proj.weight":
        value = np.transpose(value, (0, 2, 3, 4, 1))
    if dtype == "float16":
        return value.astype(np.float16)
    if dtype == "float32":
        return value.astype(np.float32)
    raise ValueError(f"Unsupported dtype: {dtype}")


def inspect_weights(weights_path: Path) -> dict[str, Any]:
    required = {
        "encoder.embeddings.patch_embeddings.proj.weight": (1408, 3, 2, 16, 16),
        "encoder.layer.0.attention.query.weight": (1408, 1408),
        "encoder.layer.0.attention.key.weight": (1408, 1408),
        "encoder.layer.0.attention.value.weight": (1408, 1408),
        "encoder.layer.39.mlp.fc2.weight": (1408, 6144),
        "encoder.layernorm.weight": (1408,),
    }
    with safe_open(str(weights_path), framework="np") as bundle:
        keys = list(bundle.keys())
        missing = []
        mismatched = []
        observed = {}
        for key, shape in required.items():
            if key not in keys:
                missing.append(key)
                continue
            actual = tuple(bundle.get_tensor(key).shape)
            observed[key] = actual
            if actual != shape:
                mismatched.append({"key": key, "expected": shape, "observed": actual})
        return {
            "num_keys": len(keys),
            "required_shapes": observed,
            "missing_required_keys": missing,
            "mismatched_required_shapes": mismatched,
        }


def convert(model_dir: Path, out_dir: Path, dtype: str) -> dict[str, Any]:
    weights_path = model_dir / "model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing model.safetensors: {weights_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    output_weights = out_dir / "model.safetensors"
    output_config = out_dir / "config.json"

    converted: dict[str, np.ndarray] = {}
    with safe_open(str(weights_path), framework="np") as bundle:
        for key in bundle.keys():
            converted[key] = convert_tensor(key, bundle.get_tensor(key), dtype)

    save_file(converted, str(output_weights), metadata={"format": "mlx", "source": str(model_dir)})
    config = load_config(model_dir)
    config["mlx_converted_from"] = str(model_dir)
    config["mlx_weight_dtype"] = dtype
    output_config.write_text(json.dumps(config, indent=2), encoding="utf-8")

    return {
        "output_dir": str(out_dir),
        "output_weights": str(output_weights),
        "output_config": str(output_config),
        "num_keys": len(converted),
        "dtype": dtype,
        "output_size_mb": round(output_weights.stat().st_size / 1_000_000, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--apply", action="store_true", help="Write converted safetensors.")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    config = load_config(model_dir)
    config_errors = validate_config(config)
    weight_report = inspect_weights(model_dir / "model.safetensors")
    if config_errors or weight_report["missing_required_keys"] or weight_report["mismatched_required_shapes"]:
        print(json.dumps({
            "ok": False,
            "config_errors": config_errors,
            "weight_report": weight_report,
        }, indent=2))
        return 1

    report = {
        "ok": True,
        "apply": args.apply,
        "source_model_dir": str(model_dir),
        "out_dir": str(out_dir),
        "dtype": args.dtype,
        "config_subset": {key: config.get(key) for key in EXPECTED},
        "weight_report": weight_report,
        "conversion": None,
    }
    if args.apply:
        report["conversion"] = convert(model_dir, out_dir, args.dtype)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
