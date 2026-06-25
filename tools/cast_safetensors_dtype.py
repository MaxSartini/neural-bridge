#!/usr/bin/env python3
"""Cast already-converted safetensors weights to a target storage dtype."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import mlx.core as mx
from safetensors import safe_open
from safetensors.numpy import save_file


def dtype_for_name(name: str) -> np.dtype:
    if name == "float16":
        return np.dtype(np.float16)
    if name == "float32":
        return np.dtype(np.float32)
    if name == "bfloat16":
        return np.dtype(np.float32)
    raise ValueError(f"Unsupported dtype: {name}")


def cast_file(source: Path, destination: Path, dtype_name: str) -> dict[str, object]:
    if dtype_name == "bfloat16":
        source_tensors = mx.load(str(source))
        input_counts: dict[str, int] = {}
        tensors = {}
        for key, tensor in source_tensors.items():
            input_counts[str(tensor.dtype)] = input_counts.get(str(tensor.dtype), 0) + 1
            tensors[key] = tensor.astype(mx.bfloat16)
        destination.parent.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(destination), tensors)
        return {
            "source": str(source),
            "destination": str(destination),
            "input_dtype_counts": input_counts,
            "output_dtype": dtype_name,
            "tensor_count": len(tensors),
        }

    dtype = dtype_for_name(dtype_name)
    tensors: dict[str, np.ndarray] = {}
    input_counts: dict[str, int] = {}
    with safe_open(str(source), framework="np") as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            input_counts[str(tensor.dtype)] = input_counts.get(str(tensor.dtype), 0) + 1
            tensors[key] = tensor.astype(dtype, copy=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(destination),
        metadata={"source": str(source), "cast_dtype": dtype_name, "format": "mlx"},
    )
    return {
        "source": str(source),
        "destination": str(destination),
        "input_dtype_counts": input_counts,
        "output_dtype": dtype_name,
        "tensor_count": len(tensors),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--copy-metadata", action="store_true")
    args = parser.parse_args()

    source_dir = args.source_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)

    reports = []
    for source in sorted(source_dir.glob("*.safetensors")):
        reports.append(cast_file(source, output_dir / source.name, args.dtype))

    if args.copy_metadata:
        output_dir.mkdir(parents=True, exist_ok=True)
        for source in sorted(source_dir.iterdir()):
            if source.is_file() and source.suffix != ".safetensors":
                shutil.copy2(source, output_dir / source.name)

    manifest = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "dtype": args.dtype,
        "files": reports,
    }
    (output_dir / "dtype_cast_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
