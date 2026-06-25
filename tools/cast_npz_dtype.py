#!/usr/bin/env python3
"""Cast numeric arrays in an NPZ bundle to a target dtype."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import mlx.core as mx


def dtype_for_name(name: str) -> np.dtype:
    if name == "float16":
        return np.dtype(np.float16)
    if name == "float32":
        return np.dtype(np.float32)
    if name == "bfloat16":
        return np.dtype(np.float32)
    raise ValueError(f"Unsupported dtype: {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)

    input_counts: dict[str, int] = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.dtype == "bfloat16":
        source_arrays = mx.load(str(source))
        arrays = {}
        for key, value in source_arrays.items():
            input_counts[str(value.dtype)] = input_counts.get(str(value.dtype), 0) + 1
            arrays[key] = value.astype(mx.bfloat16)
        mx.savez(str(output), **arrays)
        array_count = len(arrays)
    else:
        dtype = dtype_for_name(args.dtype)
        arrays_np: dict[str, np.ndarray] = {}
        with np.load(source) as bundle:
            for key in bundle.files:
                value = bundle[key]
                input_counts[str(value.dtype)] = input_counts.get(str(value.dtype), 0) + 1
                arrays_np[key] = value.astype(dtype, copy=False) if np.issubdtype(value.dtype, np.floating) else value
        np.savez(output, **arrays_np)
        array_count = len(arrays_np)
    manifest = {
        "source": str(source),
        "output": str(output),
        "dtype": args.dtype,
        "input_dtype_counts": input_counts,
        "array_count": array_count,
    }
    manifest_path = output.with_suffix(output.suffix + ".dtype_cast_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
