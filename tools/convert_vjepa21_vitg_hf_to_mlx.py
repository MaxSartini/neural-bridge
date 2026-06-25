#!/usr/bin/env python3
"""Dry-run-first V-JEPA 2.1 ViT-g/ViT-G checkpoint conversion scaffold.

This script is intentionally conservative. It does not download checkpoints,
does not import CUDA-specific packages, and does not write converted weights
unless --apply is passed. The intended output root is external storage:

  $NEURAL_BRIDGE_EXTERNAL_ROOT/models/vjepa21_mlx/vitg/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import mlx.core as mx
from safetensors.numpy import save_file


EXPECTED_VJEPA21_VITG = {
    "checkpoint_name": "vjepa2_1_vitg_384.pt",
    "model_name": "vjepa2_1_vit_giant_384",
    "arch_name": "vit_giant_xformers",
    "hidden_size": 1408,
    "num_hidden_layers": 40,
    "num_attention_heads": 22,
    "mlp_ratio": 48 / 11,
    "patch_size": 16,
    "tubelet_size": 2,
    "crop_size": 384,
    "frames_per_clip": 64,
    "checkpoint_key": "target_encoder",
    "uses_rope": True,
    "modality_embedding": True,
}


EXPECTED_VJEPA21_VITGIGANTIC = {
    "checkpoint_name": "vjepa2_1_vitG_384.pt",
    "model_name": "vjepa2_1_vit_gigantic_384",
    "arch_name": "vit_gigantic_xformers",
    "hidden_size": 1664,
    "num_hidden_layers": 48,
    "num_attention_heads": 26,
    "mlp_ratio": 64 / 13,
    "patch_size": 16,
    "tubelet_size": 2,
    "crop_size": 384,
    "frames_per_clip": 64,
    "checkpoint_key": "target_encoder",
    "uses_rope": True,
    "modality_embedding": True,
}


@dataclass(frozen=True)
class KeyMapping:
    source_pattern: str
    target_pattern: str
    conversion: str
    notes: str


KEY_MAPPING_PLAN = [
    KeyMapping(
        "module.* / backbone.* prefixes",
        "*",
        "strip_prefix",
        "Mirror upstream _clean_backbone_key behavior.",
    ),
    KeyMapping(
        "patch_embed.proj.weight",
        "patch_embed.proj.weight",
        "conv3d_o_i_t_h_w_to_o_t_h_w_i",
        "MLX Conv3d expects channel-last kernel layout.",
    ),
    KeyMapping(
        "patch_embed.proj.bias",
        "patch_embed.proj.bias",
        "copy",
        "Bias is one-dimensional.",
    ),
    KeyMapping(
        "patch_embed_img.proj.weight",
        "patch_embed_img.proj.weight",
        "conv3d_o_i_t_h_w_to_o_t_h_w_i",
        "V-JEPA 2.1 has a separate image temporal path.",
    ),
    KeyMapping(
        "blocks.{i}.norm*.{weight,bias}",
        "blocks.{i}.norm*.{weight,bias}",
        "copy",
        "LayerNorm parameters are one-dimensional.",
    ),
    KeyMapping(
        "blocks.{i}.attn.qkv.{weight,bias}",
        "blocks.{i}.attn.qkv.{weight,bias}",
        "copy_or_split_after_model_audit",
        "Luka MLX port uses fused qkv; Neuro Bridge V-JEPA2 code uses split q/k/v.",
    ),
    KeyMapping(
        "blocks.{i}.attn.{q,k,v}.{weight,bias}",
        "blocks.{i}.attn.{q,k,v}.{weight,bias}",
        "copy_or_fuse_after_model_audit",
        "Support both possible checkpoint layouts; fail on unexpected mix.",
    ),
    KeyMapping(
        "blocks.{i}.attn.proj.{weight,bias}",
        "blocks.{i}.attn.proj.{weight,bias}",
        "copy",
        "Linear weights are preserved unless target MLX module expects transpose.",
    ),
    KeyMapping(
        "blocks.{i}.mlp.fc*.{weight,bias}",
        "blocks.{i}.mlp.fc*.{weight,bias}",
        "copy",
        "MLP intermediate width must match hidden_size * mlp_ratio.",
    ),
    KeyMapping(
        "norms_block.*.{weight,bias}",
        "norms_block.*.{weight,bias}",
        "copy",
        "V-JEPA 2.1 hierarchical output norms must be preserved.",
    ),
    KeyMapping(
        "{img,video}_mod_embed",
        "{img,video}_mod_embed",
        "copy",
        "Modality embeddings are V-JEPA 2.1-specific distribution-sensitive parameters.",
    ),
]


def external_default_out_dir() -> Path:
    external_root = os.environ.get("NEURAL_BRIDGE_EXTERNAL_ROOT")
    if not external_root:
        raise SystemExit("NEURAL_BRIDGE_EXTERNAL_ROOT is required for the default output dir")
    return Path(external_root) / "models" / "vjepa21_mlx" / "vitg"


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_config(config: dict[str, Any], variant: str) -> dict[str, Any]:
    expected = EXPECTED_VJEPA21_VITGIGANTIC if variant == "vitG_gigantic" else EXPECTED_VJEPA21_VITG
    normalized = dict(expected)
    aliases = {
        "embed_dim": "hidden_size",
        "hidden_size": "hidden_size",
        "depth": "num_hidden_layers",
        "num_hidden_layers": "num_hidden_layers",
        "num_heads": "num_attention_heads",
        "num_attention_heads": "num_attention_heads",
        "image_size": "crop_size",
        "crop_size": "crop_size",
        "num_frames": "frames_per_clip",
        "frames_per_clip": "frames_per_clip",
    }
    for source_key, target_key in aliases.items():
        if source_key in config:
            normalized[target_key] = config[source_key]
    return normalized


def audit_config(config: dict[str, Any], variant: str) -> dict[str, Any]:
    expected = EXPECTED_VJEPA21_VITGIGANTIC if variant == "vitG_gigantic" else EXPECTED_VJEPA21_VITG
    normalized = normalize_config(config, variant)
    checks = {}
    for key, expected_value in expected.items():
        checks[key] = {
            "expected": expected_value,
            "observed": normalized.get(key),
            "matches": normalized.get(key) == expected_value,
        }
    return {"variant": variant, "normalized_config": normalized, "checks": checks}


def write_runtime_config(path: Path, normalized_config: dict[str, Any], run_manifest: dict[str, Any]) -> None:
    config = {
        "model_name": normalized_config["model_name"],
        "checkpoint_name": normalized_config["checkpoint_name"],
        "arch_name": normalized_config["arch_name"],
        "embed_dim": normalized_config["hidden_size"],
        "hidden_size": normalized_config["hidden_size"],
        "depth": normalized_config["num_hidden_layers"],
        "num_hidden_layers": normalized_config["num_hidden_layers"],
        "num_heads": normalized_config["num_attention_heads"],
        "num_attention_heads": normalized_config["num_attention_heads"],
        "mlp_ratio": normalized_config["mlp_ratio"],
        "img_size": normalized_config["crop_size"],
        "image_size": normalized_config["crop_size"],
        "crop_size": normalized_config["crop_size"],
        "num_frames": normalized_config["frames_per_clip"],
        "frames_per_clip": normalized_config["frames_per_clip"],
        "patch_size": normalized_config["patch_size"],
        "tubelet_size": normalized_config["tubelet_size"],
        "qkv_bias": True,
        "use_rope": normalized_config["uses_rope"],
        "interpolate_rope": True,
        "img_temporal_dim_size": 1,
        "modality_embedding": normalized_config["modality_embedding"],
        "n_output_distillation": 1,
        "checkpoint_key": run_manifest["checkpoint_key"],
        "mlx_weight_dtype": run_manifest["dtype"],
        "mlx_converted_from": run_manifest["checkpoint_path"],
        "tensor_layout": "vjepa2_1_mlx_port",
        "loader_note": (
            "Weights use lukasugar/vjepa2.1-mlx-style keys after stripping "
            "module./backbone. prefixes. Neuro Bridge's older mlx_vjepa2_cortical "
            "loader is V-JEPA2-specific and is not a direct loader for this artifact."
        ),
    }
    write_json(path, config)


def torch_tensor_to_numpy(key: str, tensor: Any, dtype: str) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    if array.ndim == 5 and key.endswith("weight"):
        array = array.transpose(0, 2, 3, 4, 1)
    elif array.ndim == 4 and key.endswith("weight"):
        array = array.transpose(0, 2, 3, 1)
    if dtype == "float16":
        array = array.astype(np.float16, copy=False)
    elif dtype == "float32" or dtype == "bfloat16":
        array = array.astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unsupported dtype for numpy safetensors conversion: {dtype}")
    return np.ascontiguousarray(array)


def load_torch_checkpoint_encoder(path: Path, checkpoint_key: str) -> dict[str, Any]:
    import torch  # Imported lazily to avoid making torch a config-audit dependency.

    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if checkpoint_key in checkpoint:
        state_dict = checkpoint[checkpoint_key]
    elif "target_encoder" in checkpoint:
        state_dict = checkpoint["target_encoder"]
    elif "ema_encoder" in checkpoint:
        state_dict = checkpoint["ema_encoder"]
    elif "encoder" in checkpoint:
        state_dict = checkpoint["encoder"]
    else:
        raise KeyError(f"unsupported checkpoint structure: {list(checkpoint.keys())}")
    cleaned = {}
    for key, value in state_dict.items():
        cleaned[key.replace("module.", "").replace("backbone.", "")] = value
    return cleaned


def write_mapping_plan(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_pattern", "target_pattern", "conversion", "notes"])
        writer.writeheader()
        for row in KEY_MAPPING_PLAN:
            writer.writerow(asdict(row))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", type=Path, default=None)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--variant", choices=["vitg_giant", "vitG_gigantic"], default="vitg_giant")
    parser.add_argument("--checkpoint-key", default=None)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--allow-unexpected", action="store_true")
    parser.add_argument("--apply", action="store_true", help="Actually write converted safetensors.")
    args = parser.parse_args()

    out_dir = args.out_dir or external_default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config_path)
    config_audit = audit_config(config, args.variant)
    checkpoint_key = args.checkpoint_key or config_audit["normalized_config"]["checkpoint_key"]

    run_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tool": "convert_vjepa21_vitg_hf_to_mlx.py",
        "variant": args.variant,
        "apply": args.apply,
        "weights_downloaded": False,
        "cuda_dependencies_installed": False,
        "cuda_assets_downloaded": False,
        "output_dir": str(out_dir),
        "config_path": str(args.config_path) if args.config_path else None,
        "checkpoint_path": str(args.checkpoint_path) if args.checkpoint_path else None,
        "checkpoint_key": checkpoint_key,
        "dtype": args.dtype,
        "allow_missing": args.allow_missing,
        "allow_unexpected": args.allow_unexpected,
    }

    write_mapping_plan(out_dir / "vjepa21_vitg_key_mapping_plan.csv")
    write_json(out_dir / "config_shape_audit.json", config_audit)
    write_runtime_config(out_dir / "config.json", config_audit["normalized_config"], run_manifest)

    if args.checkpoint_path is None:
        run_manifest["status"] = "dry_run_config_only"
        run_manifest["converted_weights_written"] = False
        write_json(out_dir / "conversion_manifest.json", run_manifest)
        print(json.dumps(run_manifest, indent=2, sort_keys=True))
        return 0

    state = load_torch_checkpoint_encoder(args.checkpoint_path, checkpoint_key)
    shape_rows = []
    converted = {}
    source_bytes = 0
    target_bytes = 0
    for key, tensor in sorted(state.items()):
        converted_array = torch_tensor_to_numpy(key, tensor, args.dtype)
        source_nbytes = int(tensor.numel() * tensor.element_size())
        target_nbytes = int(converted_array.nbytes)
        source_bytes += source_nbytes
        target_bytes += target_nbytes
        shape_rows.append(
            {
                "key": key,
                "source_shape": list(tensor.shape),
                "target_shape": list(converted_array.shape),
                "source_dtype": str(tensor.dtype),
                "target_dtype": str(converted_array.dtype),
                "source_nbytes": source_nbytes,
                "target_nbytes": target_nbytes,
                "conversion": "conv_transpose" if tensor.ndim in (4, 5) and key.endswith("weight") else "copy",
            }
        )
        if args.apply:
            if args.dtype == "bfloat16":
                converted[key] = mx.array(converted_array, dtype=mx.bfloat16)
            else:
                converted[key] = converted_array

    with (out_dir / "shape_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "key",
                "source_shape",
                "target_shape",
                "source_dtype",
                "target_dtype",
                "source_nbytes",
                "target_nbytes",
                "conversion",
            ],
        )
        writer.writeheader()
        writer.writerows(shape_rows)

    if args.apply:
        target = out_dir / "model.safetensors"
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing converted weights: {target}")
        if args.dtype == "bfloat16":
            mx.save_safetensors(str(target), converted)
        else:
            save_file(
                converted,
                str(target),
                metadata={
                    "format": "mlx",
                    "source": str(args.checkpoint_path),
                    "checkpoint_key": checkpoint_key,
                    "dtype": args.dtype,
                    "tensor_layout": "vjepa2_1_mlx_port",
                },
            )
        run_manifest["converted_weights_written"] = True
        run_manifest["converted_weights_path"] = str(target)
        run_manifest["converted_weights_size_bytes"] = target.stat().st_size
    else:
        run_manifest["converted_weights_written"] = False

    run_manifest["status"] = "completed"
    run_manifest["num_encoder_keys"] = len(state)
    run_manifest["source_encoder_tensor_bytes"] = source_bytes
    run_manifest["target_encoder_tensor_bytes"] = target_bytes
    write_json(out_dir / "conversion_manifest.json", run_manifest)
    print(json.dumps(run_manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
