"""Compare old and MLX VEATIC TRIBE cache outputs for parity gating."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def numeric_compare(old: np.ndarray, new: np.ndarray) -> dict[str, Any]:
    item: dict[str, Any] = {
        "old_shape": list(old.shape),
        "new_shape": list(new.shape),
        "old_finite": bool(np.isfinite(old).all()),
        "new_finite": bool(np.isfinite(new).all()),
    }
    if old.shape != new.shape:
        return item
    old_flat = old.reshape(-1).astype(np.float64)
    new_flat = new.reshape(-1).astype(np.float64)
    item.update(
        {
            "mean_abs_diff": float(np.mean(np.abs(old_flat - new_flat))),
            "max_abs_diff": float(np.max(np.abs(old_flat - new_flat))),
            "old_mean": float(np.mean(old_flat)),
            "new_mean": float(np.mean(new_flat)),
            "old_std": float(np.std(old_flat)),
            "new_std": float(np.std(new_flat)),
            "corr": (
                float(np.corrcoef(old_flat, new_flat)[0, 1])
                if old_flat.size > 1 and np.std(old_flat) > 0 and np.std(new_flat) > 0
                else None
            ),
        }
    )
    return item


def compare_video(old_root: Path, new_root: Path, video_id: str) -> dict[str, Any]:
    old_raw = old_root / video_id / "tribe_raw_output.npz"
    new_raw = new_root / video_id / "tribe_raw_output.npz"
    report: dict[str, Any] = {
        "video_id": video_id,
        "old_raw_exists": old_raw.exists(),
        "new_raw_exists": new_raw.exists(),
        "arrays": {},
    }
    if not old_raw.exists() or not new_raw.exists():
        return report
    with np.load(old_raw) as old_bundle, np.load(new_raw) as new_bundle:
        for key in sorted(set(old_bundle.files) | set(new_bundle.files)):
            if key not in old_bundle.files or key not in new_bundle.files:
                report["arrays"][key] = {
                    "old_present": key in old_bundle.files,
                    "new_present": key in new_bundle.files,
                }
                continue
            old = np.asarray(old_bundle[key])
            new = np.asarray(new_bundle[key])
            if np.issubdtype(old.dtype, np.number) and np.issubdtype(new.dtype, np.number):
                report["arrays"][key] = numeric_compare(old, new)
            else:
                report["arrays"][key] = {
                    "old_shape": list(old.shape),
                    "new_shape": list(new.shape),
                    "same_values": bool(np.array_equal(old, new)) if old.shape == new.shape else False,
                }
    return report


def gate_status(report: dict[str, Any], min_prediction_corr: float, max_safeguard_diff: float) -> str:
    if not report["old_raw_exists"] or not report["new_raw_exists"]:
        return "missing"
    predictions = report["arrays"].get("predictions", {})
    if predictions.get("old_shape") != predictions.get("new_shape"):
        return "fail_shape"
    if not predictions.get("old_finite") or not predictions.get("new_finite"):
        return "fail_nonfinite"
    corr = predictions.get("corr")
    if corr is None or corr < min_prediction_corr:
        return "fail_prediction_corr"
    for key in ("modality_missing_flags", "segment_retention_features"):
        item = report["arrays"].get(key, {})
        if item.get("old_shape") != item.get("new_shape"):
            return f"fail_{key}_shape"
        if float(item.get("max_abs_diff", 0.0)) > max_safeguard_diff:
            return f"fail_{key}_diff"
    return "pass"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-cache", default="/Volumes/onn. Drive/Neural Bridge/benchmarks/veatic/tribe_cache")
    parser.add_argument("--new-cache", default="/Volumes/onn. Drive/Neural Bridge/benchmarks/veatic/tribe_cache_mlx")
    parser.add_argument("--video-ids", default="")
    parser.add_argument("--min-prediction-corr", type=float, default=0.999)
    parser.add_argument("--max-safeguard-diff", type=float, default=0.0)
    args = parser.parse_args()

    old_root = Path(args.old_cache).expanduser().resolve()
    new_root = Path(args.new_cache).expanduser().resolve()
    if args.video_ids:
        video_ids = [item.strip() for item in args.video_ids.split(",") if item.strip()]
    else:
        video_ids = sorted(
            {path.parent.name for path in old_root.glob("*/tribe_raw_output.npz")}
            & {path.parent.name for path in new_root.glob("*/tribe_raw_output.npz")},
            key=lambda value: int(value) if value.isdigit() else value,
        )
    reports = []
    for video_id in video_ids:
        report = compare_video(old_root, new_root, video_id)
        report["gate"] = gate_status(report, args.min_prediction_corr, args.max_safeguard_diff)
        reports.append(report)
    summary = {
        "old_cache": str(old_root),
        "new_cache": str(new_root),
        "video_count": len(reports),
        "passed": sum(1 for report in reports if report["gate"] == "pass"),
        "failed": [report for report in reports if report["gate"] != "pass"],
        "reports": reports,
    }
    print(json.dumps(summary, indent=2))
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
