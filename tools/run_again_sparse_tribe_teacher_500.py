#!/usr/bin/env python3
"""Run an AGAIN sparse ViT-G/TRIBE teacher pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.scripts.again_sparse_tribe_teacher_500 import (  # noqa: E402
    SparseTeacherConfig,
    causal_relative_seconds_for_actual_window_hz,
    external_root,
    now_stamp,
    run_sparse_teacher_500,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestamp", default=now_stamp())
    parser.add_argument("--max-actual-windows", type=int, default=500)
    parser.add_argument(
        "--actual-window-hz",
        type=float,
        default=1.0,
        help="Sparse ViT-G/TRIBE actual-window cadence inside selected causal regions. Use 2.0 for 0.5s steps.",
    )
    parser.add_argument("--selector-validation-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--external-cache-root", type=Path, default=None)
    parser.add_argument("--run-label", default=None)
    parser.add_argument(
        "--teacher-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
        help="Dtype for V-JEPA 2.1 ViT-G inputs/weights and TRIBE MLX weights. bfloat16 is the safe default; cache fingerprints include this value.",
    )
    parser.add_argument(
        "--allow-cache-backfill",
        action="store_true",
        help="Allow loose legacy/physical-window cache backfill. Keep off for dtype speed tests.",
    )
    parser.add_argument(
        "--gpu-workers",
        type=int,
        default=1,
        help="Must stay 1 for MLX/V-JEPA on Apple Silicon. Multiple GPU workers contend on the same GPU.",
    )
    parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=0,
        help="CPU-side preprocessing worker count. The current sparse teacher keeps GPU forwards in the main owner only.",
    )
    parser.add_argument(
        "--ready-queue-max-size",
        type=int,
        default=4,
        help="Maximum prepared windows allowed before a GPU forward. Also caps the effective microbatch size.",
    )
    parser.add_argument(
        "--writer-queue-max-size",
        type=int,
        default=4,
        help="Reserved output queue bound recorded in manifests; writes remain synchronous in the safe sparse path.",
    )
    parser.add_argument(
        "--microbatch-size",
        type=int,
        default=1,
        help="Number of uncached windows stacked into one single-owner V-JEPA forward. Test 1, 2, then maybe 3.",
    )
    parser.add_argument(
        "--no-compile-encoder",
        action="store_true",
        help="Disable mx.compile wrapping for the fixed-shape V-JEPA selected-state forward.",
    )
    parser.add_argument(
        "--arm-window-budgets-json",
        default=None,
        help="Optional JSON object overriding selector arm window budgets, e.g. '{\"hybrid_top5_selected\":900}'.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    default_run_label = f"again_sparse_tribe_teacher_{args.max_actual_windows}"
    run_label = args.run_label or default_run_label
    run_title = f"AGAIN Sparse TRIBE Teacher {args.max_actual_windows}"
    output_root = args.output_root or Path("outputs") / f"{run_label}_{args.timestamp}"
    cache_root = args.external_cache_root or external_root() / "benchmarks" / "again" / f"{run_label}_{args.timestamp}"
    config_kwargs = {
        "max_actual_windows": args.max_actual_windows,
        "report_date": args.timestamp,
        "run_label": run_label,
        "run_title": run_title,
        "causal_relative_seconds": causal_relative_seconds_for_actual_window_hz(args.actual_window_hz),
        "teacher_dtype": args.teacher_dtype,
        "allow_cache_backfill": args.allow_cache_backfill,
        "gpu_workers": args.gpu_workers,
        "preprocess_workers": args.preprocess_workers,
        "ready_queue_max_size": args.ready_queue_max_size,
        "writer_queue_max_size": args.writer_queue_max_size,
        "microbatch_size": args.microbatch_size,
        "compile_encoder": not args.no_compile_encoder,
    }
    if args.selector_validation_root is not None:
        config_kwargs["selector_validation_root"] = args.selector_validation_root
    if args.arm_window_budgets_json is not None:
        budgets = json.loads(args.arm_window_budgets_json)
        if not isinstance(budgets, dict) or not all(isinstance(key, str) and isinstance(value, int) for key, value in budgets.items()):
            raise ValueError("--arm-window-budgets-json must be a JSON object of string arm names to integer budgets")
        if sum(budgets.values()) > args.max_actual_windows:
            raise ValueError("custom arm budgets must sum to <= --max-actual-windows")
        config_kwargs["arm_window_budgets"] = budgets
    result = run_sparse_teacher_500(
        output_root=output_root,
        external_cache_root=cache_root,
        config=SparseTeacherConfig(**config_kwargs),
    )
    manifest = result["manifest"]
    runtime = result["runtime_summary"]
    print(f"benchmark_mode={manifest['benchmark_mode']}")
    print(f"output_root={manifest['output_root']}")
    print(f"external_cache_root={manifest['external_cache_root']}")
    print(f"actual_unique_windows_queued={manifest['actual_unique_windows_queued']}")
    print(f"actual_successful_windows={manifest['actual_successful_windows']}")
    print(f"cache_hits={runtime['cache_hits']}")
    print(f"teacher_dtype={runtime.get('teacher_dtype')}")
    print(f"allow_cache_backfill={str(runtime.get('allow_cache_backfill')).lower()}")
    print(f"gpu_workers={runtime.get('gpu_workers')}")
    print(f"microbatch_size={runtime.get('microbatch_size')}")
    print(f"compile_encoder={str(runtime.get('compile_encoder')).lower()}")
    print(f"failed_windows={runtime['failed_windows']}")
    print(f"dense_again_vitg_encoding_run={str(manifest['dense_again_vitg_encoding_run']).lower()}")
    print(f"models_trained={str(manifest['models_trained']).lower()}")
    print(f"results_report={manifest['results_report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
