#!/usr/bin/env python3
"""Profile V-JEPA 2.1 MLX TRIBE speed knobs on one real VEATIC video."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from probe_veatic_vjepa21_real_video_runtime import (  # noqa: E402
    ffprobe,
    load_manifest_rows,
    npz_prediction_stats,
    resolve_external_path,
    select_short_video,
    write_csv,
    write_json,
)


DEFAULT_VARIANTS = [
    {"variant": "native384_batch1", "image_size": 384, "window_batch_size": 1},
    {"variant": "override256_batch1", "image_size": 256, "window_batch_size": 1},
    {"variant": "override256_batch2", "image_size": 256, "window_batch_size": 2},
]


def row_from_variant(
    *,
    variant: dict[str, Any],
    video_id: str,
    video_path: Path,
    manifest_rows: list[dict[str, Any]],
    selected_video: dict[str, Any],
    metadata: dict[str, Any],
    external_output_root: Path,
    old_npz: Path,
) -> dict[str, Any]:
    from backend.app.config import Config
    from backend.app.services.tribe_adapter import TribeAdapter

    variant_name = str(variant["variant"])
    variant_root = external_output_root / variant_name
    output_dir = variant_root / video_id
    video_window_root = variant_root / "video_windows"
    video_window_root.mkdir(parents=True, exist_ok=True)
    os.environ["TRIBE_VIDEO_WINDOW_CACHE_DIR"] = str(video_window_root)
    os.environ["TRIBE_VIDEO_WINDOW_BATCH_SIZE"] = str(int(variant["window_batch_size"]))

    Config.TRIBE_MLX_ENABLED = True
    Config.NEURO_PRIOR_SAVE_RAW_OUTPUT = True
    Config.TRIBE_VIDEO_ENCODER_BACKEND = "mlx"
    Config.TRIBE_VIDEO_ENCODER_MLX_DIR = str(
        Path(Config.NEURAL_BRIDGE_EXTERNAL_ROOT).expanduser()
        / "models"
        / "vjepa21_mlx"
        / "vitg"
    )
    Config.TRIBE_VJEPA21_IMAGE_SIZE = int(variant["image_size"])
    Config.TRIBE_VIDEO_NUM_FRAMES = 64

    cache_suffix = f"speed-ablation-{external_output_root.name}-{variant_name}"

    class AblationTribeAdapter(TribeAdapter):
        def _config_update(self) -> dict[str, Any]:
            update = super()._config_update()
            if update.get("data.video_feature.name") == "MlxVjepa21Video":
                update["data.video_feature.cache_model_name"] = (
                    f"{update['data.video_feature.cache_model_name']}-{cache_suffix}"
                )
            return update

    adapter = AblationTribeAdapter()
    selected_config = adapter._config_update()
    started = time.time()
    error = ""
    result: dict[str, Any] = {}
    try:
        result = adapter.predict(
            stimulus_text="",
            stimulus_type="video",
            media_path=str(video_path),
            output_dir=str(output_dir),
            backend="tribe_mlx",
        )
    except Exception as exc:  # noqa: BLE001 - diagnostic probe should record failures
        error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        result = {"success": False, "error": error}
    elapsed = time.time() - started

    new_npz = output_dir / "tribe_raw_output.npz"
    new_stats = npz_prediction_stats(new_npz, prefix="new_")
    old_stats = npz_prediction_stats(old_npz, prefix="old_")
    window_rows = 0
    for cache_path in video_window_root.glob("*.npy"):
        try:
            window_rows += int(np.load(cache_path, mmap_mode="r").shape[0])
        except Exception:
            pass
    storage_bytes = sum(path.stat().st_size for path in variant_root.rglob("*") if path.is_file())
    success = bool(result.get("success")) and bool(new_stats.get("new_exists")) and int(new_stats.get("new_nonfinite_count", 1)) == 0
    feature_width_match = new_stats.get("new_feature_width") == old_stats.get("old_feature_width")
    temporal_rows_delta = (
        int(new_stats.get("new_temporal_rows", 0)) - int(old_stats.get("old_temporal_rows", 0))
        if old_stats.get("old_exists")
        else None
    )
    shape_compatible = bool(
        success
        and feature_width_match
        and old_stats.get("old_exists")
        and abs(int(temporal_rows_delta or 0)) <= 2
    )
    stats_sane = bool(success and float(new_stats.get("new_std", 0.0) or 0.0) > 0.0)

    return {
        "variant": variant_name,
        "video_id": video_id,
        "video_path": str(video_path),
        "manifest_rows": len(manifest_rows),
        "manifest_duration_seconds": selected_video.get("duration_seconds"),
        "ffprobe_duration_seconds": metadata.get("duration_seconds"),
        "fps": metadata.get("fps") or selected_video.get("fps"),
        "width": metadata.get("width"),
        "height": metadata.get("height"),
        "codec_name": metadata.get("codec_name"),
        "success": success,
        "backend_used": "tribe_mlx",
        "video_feature_name": selected_config.get("data.video_feature.name"),
        "image_size": Config.TRIBE_VJEPA21_IMAGE_SIZE,
        "frames_per_clip": Config.TRIBE_VIDEO_NUM_FRAMES,
        "window_batch_size": int(variant["window_batch_size"]),
        "frame_sampler": selected_config.get("data.video_feature.frame_sampler"),
        "internal_encoder_windows": window_rows,
        "encoding_time_seconds": round(elapsed, 3),
        "seconds_per_internal_window": round(elapsed / window_rows, 3) if window_rows else None,
        "storage_size_bytes": storage_bytes,
        "selected_model_config_json": json.dumps(
            {
                "video_encoder_mlx_dir": Config.TRIBE_VIDEO_ENCODER_MLX_DIR,
                "image_size": Config.TRIBE_VJEPA21_IMAGE_SIZE,
                "frames_per_clip": Config.TRIBE_VIDEO_NUM_FRAMES,
                "feature_frequency_hz": Config.TRIBE_FEATURE_FREQUENCY_HZ,
                "frame_sampler": selected_config.get("data.video_feature.frame_sampler"),
                "window_batch_size": int(variant["window_batch_size"]),
                "cache_suffix": cache_suffix,
                "cache_model_name": selected_config.get("data.video_feature.cache_model_name"),
            },
            sort_keys=True,
        ),
        "error": error or result.get("error", ""),
        "feature_width_matches_old": feature_width_match,
        "temporal_rows_delta_vs_old": temporal_rows_delta,
        "shape_compatible_with_old": shape_compatible,
        "stats_sane": stats_sane,
        "real_veatic_videos_encoded": 1,
        "full_veatic_encoding_run": False,
        "again_encoding_run": False,
        "benchmark_run": False,
        "models_trained": False,
        "old_veatic_cache_modified": False,
        "vjepa21_claim_made": False,
        **new_stats,
        **old_stats,
    }


def write_report(path: Path, rows: list[dict[str, Any]], output_root: Path, external_output_root: Path) -> None:
    successful = [row for row in rows if row.get("success")]
    native = next((row for row in rows if row.get("variant") == "native384_batch1" and row.get("success")), None)

    def speedup(row: dict[str, Any]) -> str:
        if not native:
            return "n/a"
        denom = float(row.get("encoding_time_seconds") or 0)
        if denom <= 0:
            return "n/a"
        return f"{float(native['encoding_time_seconds']) / denom:.2f}x"

    lines = [
        "# VEATIC V-JEPA 2.1 MLX Speed Ablation",
        "",
        "## Verdict",
        f"- Variants run: `{len(rows)}`",
        f"- Successful variants: `{len(successful)}`",
        "- Benchmark run: `false`",
        "- Models trained: `false`",
        "- Accuracy claim made: `false`",
        "",
        "## Results",
        "| variant | image | batch | success | shape | time_s | sec/window | speedup_vs_384_b1 | new mean/std |",
        "|---|---:|---:|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {image_size} | {window_batch_size} | {success} | {shape} | {time} | {spw} | {speedup} | {mean}/{std} |".format(
                variant=row.get("variant"),
                image_size=row.get("image_size"),
                window_batch_size=row.get("window_batch_size"),
                success=str(row.get("success")).lower(),
                shape=row.get("new_shape", ""),
                time=row.get("encoding_time_seconds"),
                spw=row.get("seconds_per_internal_window"),
                speedup=speedup(row),
                mean=round(float(row.get("new_mean", 0.0) or 0.0), 4) if row.get("new_mean") is not None else "n/a",
                std=round(float(row.get("new_std", 0.0) or 0.0), 4) if row.get("new_std") is not None else "n/a",
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "- This is a speed and output-health probe only.",
            "- It does not measure arousal prediction accuracy.",
            "- `256` is an inference override for the `vjepa2_1_vitg_384.pt` checkpoint, so it needs downstream validation before bulk use.",
            "- Batch-size gains are limited by Apple Silicon unified memory and may not scale monotonically.",
            "",
            "## Output Roots",
            f"- Tracked output root: `{output_root}`",
            f"- External output root: `{external_output_root}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("benchmarks/veatic/veatic_manifest_124_complete_20260616.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("benchmarks/veatic/veatic_manifest_124_complete_20260616.report.json"))
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument(
        "--include-384-batch2",
        action="store_true",
        help="Also try native 384px with batch size 2. This can exceed 32 GB memory.",
    )
    parser.add_argument(
        "--skip-native-384",
        action="store_true",
        help="Run only 256px override variants.",
    )
    parser.add_argument(
        "--only-batch1",
        action="store_true",
        help="Run only batch-size-1 variants.",
    )
    args = parser.parse_args()

    from backend.app.config import Config

    external_root = Path(Config.NEURAL_BRIDGE_EXTERNAL_ROOT).expanduser()
    output_root = ROOT / "outputs" / f"veatic_vjepa21_speed_ablation_{args.timestamp}"
    external_output_root = external_root / "benchmarks" / "veatic" / f"vjepa21_mlx_speed_ablation_{args.timestamp}"
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty output root: {output_root}")
    if external_output_root.exists() and any(external_output_root.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty external root: {external_output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    external_output_root.mkdir(parents=True, exist_ok=False)

    selected_video = select_short_video(args.report, args.video_id)
    video_id = str(selected_video["video_id"])
    manifest_rows = [row for row in load_manifest_rows(args.manifest) if str(row.get("video_id")) == video_id]
    if not manifest_rows:
        raise ValueError(f"No manifest rows found for video_id={video_id}")
    video_path = resolve_external_path(manifest_rows[0]["media_path"], external_root)
    if not video_path.exists():
        raise FileNotFoundError(f"Selected video missing: {video_path}")
    metadata = ffprobe(video_path)
    old_npz = external_root / "benchmarks" / "veatic" / "tribe_cache" / video_id / "tribe_raw_output.npz"

    variants = list(DEFAULT_VARIANTS)
    if args.skip_native_384:
        variants = [variant for variant in variants if int(variant["image_size"]) != 384]
    if args.only_batch1:
        variants = [variant for variant in variants if int(variant["window_batch_size"]) == 1]
    if args.include_384_batch2:
        variants.insert(1, {"variant": "native384_batch2", "image_size": 384, "window_batch_size": 2})

    rows: list[dict[str, Any]] = []
    for variant in variants:
        print(f"[speed-ablation] running {variant['variant']} image={variant['image_size']} batch={variant['window_batch_size']}", flush=True)
        rows.append(
            row_from_variant(
                variant=variant,
                video_id=video_id,
                video_path=video_path,
                manifest_rows=manifest_rows,
                selected_video=selected_video,
                metadata=metadata,
                external_output_root=external_output_root,
                old_npz=old_npz,
            )
        )
        write_csv(output_root / "veatic_vjepa21_speed_ablation_results.csv", rows)

    successful = [row for row in rows if row.get("success")]
    native = next((row for row in rows if row.get("variant") == "native384_batch1" and row.get("success")), None)
    for row in rows:
        if native and row.get("success") and float(row.get("encoding_time_seconds") or 0) > 0:
            row["speedup_vs_native384_batch1"] = round(
                float(native["encoding_time_seconds"]) / float(row["encoding_time_seconds"]),
                4,
            )
        else:
            row["speedup_vs_native384_batch1"] = None
    write_csv(output_root / "veatic_vjepa21_speed_ablation_results.csv", rows)

    recommendation = "needs_review"
    if successful:
        compatible = [row for row in successful if row.get("shape_compatible_with_old") and row.get("stats_sane")]
        if compatible:
            recommendation = min(compatible, key=lambda item: float(item.get("encoding_time_seconds") or 1e12)).get("variant", "needs_review")

    run_manifest = {
        "created_at": datetime.now().isoformat(),
        "output_root": str(output_root),
        "external_output_root": str(external_output_root),
        "selected_video": {
            "video_id": video_id,
            "video_path": str(video_path),
            "duration_seconds": selected_video.get("duration_seconds"),
            "fps": selected_video.get("fps"),
            "manifest_rows": len(manifest_rows),
            "metadata": metadata,
        },
        "variants": variants,
        "recommended_speed_variant_for_next_pilot": recommendation,
        "guardrails": {
            "real_veatic_videos_encoded_per_variant": 1,
            "full_veatic_encoding_run": False,
            "again_encoding_run": False,
            "benchmark_run": False,
            "models_trained": False,
            "old_veatic_cache_modified": False,
            "vjepa21_claim_made": False,
        },
    }
    write_json(output_root / "run_manifest.json", run_manifest)
    write_report(output_root / "veatic_vjepa21_speed_ablation_report.md", rows, output_root, external_output_root)
    print(json.dumps(run_manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
