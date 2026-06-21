#!/usr/bin/env python3
"""Small AGAIN TRIBE native-grid probe.

Runs a bounded TRIBE-MLX probe on selected AGAIN videos, records the native
prediction grid, and creates a pilot manifest aligned to continuous arousal.
It does not run full AGAIN encoding, train models, create a final manifest, or
modify VEATIC outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_ROOT))
PROJECT_ROOT = BACKEND_ROOT.parent

from app.config import Config  # noqa: E402
from app.services.tribe_adapter import TribeAdapter  # noqa: E402
from scripts.again_native_temporal_alignment import build_native_grid_manifest  # noqa: E402


ALIGNMENT_DIAGNOSIS = Path("outputs/again_alignment_offset_diagnosis_20260621_131041")
INVENTORY_AUDIT = Path("outputs/again_cleaned_inventory_audit_20260621_123531")
BOUNDARY_AUDIT = Path("outputs/again_video_boundary_audit_20260621_204520")
ALIGNMENT_POLICY = "use_annotation_covered_video_time_only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe AGAIN TRIBE native temporal grid.")
    parser.add_argument("--again-root", type=Path, default=default_again_root())
    parser.add_argument("--alignment-diagnosis-root", type=Path, default=ALIGNMENT_DIAGNOSIS)
    parser.add_argument("--inventory-audit-root", type=Path, default=INVENTORY_AUDIT)
    parser.add_argument("--boundary-audit-root", type=Path, default=BOUNDARY_AUDIT)
    parser.add_argument("--max-videos", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true", help="Do not invoke TRIBE; write planned probe rows only.")
    parser.add_argument(
        "--no-prepare-aligned-probe-clips",
        action="store_true",
        help="Disable temporary aligned probe clip creation for selected videos.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--video-num-frames", type=int, default=64)
    parser.add_argument("--video-dtype", default="float16")
    parser.add_argument("--mps-memory-fraction", type=float, default=0.35)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--external-cache-root", type=Path, default=None)
    return parser.parse_args()


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def external_root() -> Path:
    configured = os.environ.get("NEURAL_BRIDGE_EXTERNAL_ROOT", "").strip()
    if not configured:
        raise EnvironmentError("Set NEURAL_BRIDGE_EXTERNAL_ROOT to the Neural Bridge external assets root.")
    return Path(configured).expanduser()


def default_again_root() -> Path:
    return external_root() / "data" / "external" / "AGAIN" / "cleaned"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: clean_value(row.get(key, "")) for key in fieldnames})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_value(payload), indent=2, sort_keys=True), encoding="utf-8")


def clean_value(value: Any) -> Any:
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {key: clean_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_value(item) for item in value]
    return value


def safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        number = float(value)
        return number if np.isfinite(number) else None
    except Exception:
        return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_runtime(args: argparse.Namespace, cache_root: Path) -> dict[str, Any]:
    root = external_root()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HOME"] = str(root / "cache" / "huggingface")
    runtime_root = cache_root / "_runtime_cache"
    os.environ["TMPDIR"] = str(runtime_root / "tmp")
    feature_cache_dir = runtime_root / "tribev2"
    os.environ["TRIBE_CACHE_DIR"] = str(feature_cache_dir)
    os.environ["TRIBE_VIDEO_WINDOW_CACHE_DIR"] = str(feature_cache_dir / "video_windows")
    os.environ["TRIBE_MPS_CHUNKED_ATTENTION"] = "true"
    os.environ["TRIBE_MPS_ATTENTION_QUERY_CHUNK_SIZE"] = "128"
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.45"
    os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.25"

    Config.NEURO_PRIOR_MODE = "tribe_mlx"
    Config.TRIBE_MLX_ENABLED = True
    Config.NEURO_PRIOR_STRICT = True
    Config.TRIBE_DEVICE = "mps"
    Config.TRIBE_VIDEO_DEVICE = "mps"
    Config.TRIBE_VIDEO_DTYPE = args.video_dtype
    Config.TRIBE_VIDEO_NUM_FRAMES = args.video_num_frames
    Config.TRIBE_VIDEO_ENCODER_BACKEND = "mlx"
    Config.TRIBE_VIDEO_ENCODER_MLX_DIR = str(root / "models" / "vjepa21_mlx" / "vitg")
    Config.TRIBE_VIDEO_ENCODER_LOCAL_DIR = str(root / "models" / "vjepa21_mlx" / "vitg")
    Config.TRIBE_VJEPA21_IMAGE_SIZE = int(getattr(Config, "TRIBE_VJEPA21_IMAGE_SIZE", 256))
    Config.TRIBE_VJEPA21_COMPILE_ENCODER = bool(getattr(Config, "TRIBE_VJEPA21_COMPILE_ENCODER", True))
    Config.TRIBE_MPS_MEMORY_FRACTION = args.mps_memory_fraction
    Config.TRIBE_CACHE_DIR = str(feature_cache_dir)
    Config.TRIBE_APPLE_SILICON_SOURCE_DIR = str(PROJECT_ROOT / "external_models" / "tribev2-apple-silicon")
    Config.TRIBE_MLX_DIR = str(root / "models" / "tribe-mlx" / "zimengxiong-tribev2-mlx")
    Config.NEURO_PRIOR_SAVE_RAW_OUTPUT = True

    return {
        "backend": "tribe_mlx",
        "video_encoder_backend": "mlx",
        "device": "mps",
        "video_dtype": Config.TRIBE_VIDEO_DTYPE,
        "video_num_frames": Config.TRIBE_VIDEO_NUM_FRAMES,
        "video_encoder_mlx_dir": Config.TRIBE_VIDEO_ENCODER_MLX_DIR,
        "vjepa21_image_size": Config.TRIBE_VJEPA21_IMAGE_SIZE,
        "vjepa21_compile_encoder": Config.TRIBE_VJEPA21_COMPILE_ENCODER,
        "probe_cache_root": str(cache_root),
        "feature_cache_dir": str(feature_cache_dir),
        "tribe_video_window_cache_dir": os.environ["TRIBE_VIDEO_WINDOW_CACHE_DIR"],
    }


def assert_again_only_paths(*, output_root: Path, cache_root: Path) -> None:
    checked = {
        "output_root": str(output_root),
        "external_cache_root": str(cache_root),
    }
    for label, value in checked.items():
        lowered = value.lower()
        if "benchmarks/veatic" in lowered or "veatic/tribe_cache" in lowered:
            raise SystemExit(f"Refusing AGAIN probe with VEATIC path in {label}: {value}")
    if "again" not in {part.lower() for part in cache_root.parts}:
        raise SystemExit(f"AGAIN probe external cache root must include an AGAIN path component: {cache_root}")


def selected_probe_videos(args: argparse.Namespace) -> list[dict[str, Any]]:
    selected_path = args.alignment_diagnosis_root / "again_selected_video_inspection.csv"
    selected = read_csv_rows(selected_path)
    priority = ["closest_to_median", "smallest_mismatch", "largest_positive_mismatch", "random_seed_20260621"]
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for reason in priority:
        for row in selected:
            if row["video_id"] in seen:
                continue
            if reason in row.get("selection_reasons", ""):
                ordered.append(row)
                seen.add(row["video_id"])
            if len(ordered) >= args.max_videos:
                return ordered
    return ordered[: args.max_videos]


def boundary_recommendations(boundary_audit_root: Path) -> dict[str, dict[str, str]]:
    path = boundary_audit_root / "again_video_boundary_recommendations.csv"
    rows = read_csv_rows(path)
    return {row["video_id"]: row for row in rows if row.get("video_id")}


def apply_boundary_recommendations(selected: list[dict[str, Any]], boundaries: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for row in selected:
        video_id = row["video_id"]
        boundary = boundaries.get(video_id)
        if boundary is None:
            raise SystemExit(f"Missing boundary recommendation for selected video: {video_id}")
        if boundary.get("recommended_policy") != ALIGNMENT_POLICY:
            raise SystemExit(
                f"Unsupported boundary policy for {video_id}: {boundary.get('recommended_policy')}; "
                f"expected {ALIGNMENT_POLICY}"
            )
        merged.append({**row, **{f"boundary_{key}": value for key, value in boundary.items()}})
    return merged


def annotation_series(manifest_path: Path, video_ids: set[str]) -> dict[str, tuple[list[float], list[float]]]:
    times: dict[str, list[float]] = defaultdict(list)
    values: dict[str, list[float]] = defaultdict(list)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            video_id = row.get("video_id", "")
            if video_id not in video_ids:
                continue
            t = safe_float(row.get("time_start_seconds"))
            y = safe_float(row.get("arousal"))
            if t is None or y is None:
                continue
            times[video_id].append(t)
            values[video_id].append(y)
    return {video_id: (times[video_id], values[video_id]) for video_id in video_ids}


def prepare_aligned_probe_clip(source: Path, destination: Path, duration: float, force: bool) -> dict[str, Any]:
    if destination.exists() and not force:
        return {"success": True, "path": str(destination), "status": "cached", "duration_seconds": duration}
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-t",
        f"{duration:.3f}",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    started = time.perf_counter()
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=180)
    except Exception as exc:
        return {
            "success": False,
            "path": str(destination),
            "duration_seconds": duration,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "success": True,
        "path": str(destination),
        "duration_seconds": duration,
        "creation_time_seconds": round(time.perf_counter() - started, 3),
        "size_bytes": destination.stat().st_size if destination.exists() else 0,
    }


def probe_video(
    adapter: TribeAdapter,
    video: dict[str, Any],
    output_dir: Path,
    force: bool,
    dry_run: bool,
    prepare_probe_clip: bool,
) -> dict[str, Any]:
    raw_path = output_dir / "tribe_raw_output.npz"
    status_path = output_dir / "cache_status.json"
    source_media_path = Path(video["video_path"])
    media_path = source_media_path
    clip_status: dict[str, Any] = {"success": False, "status": "not_requested"}
    if prepare_probe_clip and not dry_run:
        clip_status = prepare_aligned_probe_clip(
            source_media_path,
            output_dir / "aligned_probe_clip.mp4",
            aligned_duration(video),
            force=force,
        )
        if clip_status.get("success"):
            media_path = Path(str(clip_status["path"]))
    status: dict[str, Any] = {
        "video_id": video["video_id"],
        "video_path": str(source_media_path),
        "probe_media_path": str(media_path),
        "probe_media_preprocessed": str(media_path) != str(source_media_path),
        "probe_media_preprocess_policy": (
            "trim_to_aligned_duration_and_write_duration_metadata"
            if str(media_path) != str(source_media_path)
            else "none"
        ),
        "probe_clip_creation": clip_status,
        "aligned_video_duration": aligned_duration(video),
        "aligned_context_end_seconds": aligned_context_end(video),
        "stimulus_sha256": sha256(source_media_path) if source_media_path.exists() else "",
        "raw_tribe_output_path": str(raw_path),
        "dry_run": dry_run,
    }
    if dry_run:
        status.update({"success": False, "error": "dry_run_no_tribe_invocation"})
        return status
    if raw_path.exists() and status_path.exists() and not force:
        try:
            previous = json.loads(status_path.read_text(encoding="utf-8"))
            if previous.get("stimulus_sha256") == status["stimulus_sha256"]:
                return {**previous, "status": "cached"}
        except Exception:
            pass
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result = adapter.predict(stimulus_type="video", media_path=str(media_path), output_dir=str(output_dir), backend="tribe_mlx")
    status["encoding_time_seconds"] = round(time.perf_counter() - started, 3)
    status["result_success"] = bool(result.get("success"))
    if not result.get("success"):
        status.update({"success": False, "error": result.get("error", "TRIBE probe failed")})
    elif not raw_path.exists():
        status.update({"success": False, "error": "raw TRIBE output missing"})
    else:
        status.update(inspect_raw_output(raw_path, status["aligned_video_duration"]))
        status["success"] = True
        status["output_size_bytes"] = raw_path.stat().st_size
    status_path.write_text(json.dumps(clean_value(status), indent=2), encoding="utf-8")
    return status


def inspect_raw_output(raw_path: Path, aligned_duration: float) -> dict[str, Any]:
    with np.load(raw_path) as bundle:
        keys = list(bundle.files)
        predictions_exists = "predictions" in bundle.files
        shape = list(bundle["predictions"].shape) if predictions_exists else []
    rows = int(shape[0]) if shape else 0
    width = int(shape[1]) if len(shape) >= 2 else None
    seconds_per_prediction = (aligned_duration / rows) if rows else None
    rate = (rows / aligned_duration) if aligned_duration > 0 and rows else None
    return {
        "predictions_key_exists": predictions_exists,
        "prediction_array_shape": shape,
        "number_of_prediction_rows": rows,
        "feature_width": width,
        "raw_keys": keys,
        "model_data_tr": None,
        "timestamps_available": False,
        "segment_start_end_times_available": False,
        "inferred_seconds_per_prediction": seconds_per_prediction,
        "inferred_prediction_rate_hz": rate,
        "timing_source": "inferred_duration_even_spacing",
        "timing_confidence": "medium" if rows else "unknown",
        "close_to_1hz": bool(rate is not None and abs(rate - 1.0) <= 0.15),
        "higher_than_1hz": bool(rate is not None and rate > 1.15),
        "segment_level": True,
    }


def aligned_duration(video: dict[str, Any]) -> float:
    start = safe_float(video.get("boundary_recommended_encode_start_seconds"))
    end = safe_float(video.get("boundary_recommended_encode_end_seconds"))
    if start is not None and end is not None:
        return max(0.0, end - start)
    video_duration = safe_float(video.get("video_duration_seconds")) or 0.0
    annotation_duration = safe_float(video.get("annotation_duration_seconds")) or 0.0
    return max(0.0, min(video_duration, annotation_duration))


def aligned_context_end(video: dict[str, Any]) -> float:
    end = safe_float(video.get("boundary_recommended_benchmark_end_seconds"))
    if end is not None:
        return end
    return aligned_duration(video)


def build_manifest_rows(
    probe_rows: list[dict[str, Any]],
    annotations: dict[str, tuple[list[float], list[float]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for probe in probe_rows:
        if not probe.get("success"):
            continue
        shape = probe.get("prediction_array_shape") or []
        video_id = probe["video_id"]
        times, values = annotations.get(video_id, ([], []))
        rows, grid = build_native_grid_manifest(
            dataset_name="AGAIN_cleaned",
            video_id=video_id,
            video_path=probe["video_path"],
            prediction_shape=shape,
            aligned_video_duration=float(probe["aligned_video_duration"]),
            annotation_times=times,
            arousal_values=values,
            alignment_policy=ALIGNMENT_POLICY,
            aligned_context_end_seconds=float(probe.get("aligned_context_end_seconds") or probe["aligned_video_duration"]),
            interpolation_method="linear",
        )
        manifest_rows.extend(rows)
        summary_rows.append(
            {
                "video_id": video_id,
                "prediction_rows": len(rows),
                "feature_width": probe.get("feature_width"),
                "seconds_per_prediction": grid.seconds_per_prediction,
                "prediction_rate_hz": grid.prediction_rate_hz,
                "timing_source": grid.timing_source,
                "timing_confidence": grid.timing_confidence,
                "aligned_video_duration": probe["aligned_video_duration"],
                "future_spike_feasible_rows": sum(1 for row in rows if row["future_spike_1_3s_feasible"]),
                "future_change_feasible_rows": sum(1 for row in rows if row["future_change_p3s_feasible"]),
            }
        )
    return manifest_rows, summary_rows


def report_text(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# AGAIN TRIBE Native Grid Probe",
            "",
            "## Executive Summary",
            "",
            f"- videos probed: {summary['videos_probed']}",
            f"- successful TRIBE probes: {summary['successful_probes']}",
            f"- recommended benchmark grid: {summary['decision']['recommended_benchmark_grid']}",
            f"- 1Hz still recommended: {summary['decision']['one_hz_still_recommended']}",
            f"- 2Hz/5Hz/30fps supported: {summary['decision']['higher_rates_supported']}",
            f"- temporary aligned probe clips created: {summary['probe_media']['temporary_aligned_probe_clips_created']}",
            "",
            "## Timing",
            "",
            f"- timing source: {summary['aggregate_grid']['dominant_timing_source']}",
            f"- timing confidence: {summary['aggregate_grid']['dominant_timing_confidence']}",
            f"- median inferred rate Hz: {summary['aggregate_grid']['median_prediction_rate_hz']}",
            "",
            "## Guardrails",
            "",
            "full_again_encoding_run=false",
            "benchmark_run=false",
            "models_trained=false",
            "final_manifest_created=false",
            "veatic_outputs_modified=false",
            "fake_high_rate_cortical_rows_created=false",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    stamp = now_stamp()
    output_root = args.output_root or Path("outputs") / f"again_tribe_native_grid_probe_{stamp}"
    cache_root = args.external_cache_root or external_root() / "benchmarks" / "again" / f"native_grid_probe_{stamp}"
    assert_again_only_paths(output_root=output_root, cache_root=cache_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"Output root exists and is non-empty: {output_root}")
    if cache_root.exists() and any(cache_root.iterdir()) and not args.force:
        raise SystemExit(f"External cache root exists and is non-empty: {cache_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    selected = apply_boundary_recommendations(selected_probe_videos(args), boundary_recommendations(args.boundary_audit_root))
    video_ids = {row["video_id"] for row in selected}
    annotations = annotation_series(args.inventory_audit_root / "again_manifest_proposal.csv", video_ids)
    runtime_contract = configure_runtime(args, cache_root)
    adapter = TribeAdapter()
    probe_rows = []
    for video in selected:
        probe_rows.append(
            probe_video(
                adapter,
                video,
                cache_root / video["video_id"],
                force=args.force,
                dry_run=args.dry_run,
                prepare_probe_clip=not args.no_prepare_aligned_probe_clips,
            )
        )
    manifest_rows, manifest_summary_rows = build_manifest_rows(probe_rows, annotations)
    successful = [row for row in probe_rows if row.get("success")]
    rates = [safe_float(row.get("inferred_prediction_rate_hz")) for row in successful]
    rates = [rate for rate in rates if rate is not None]
    close_to_1hz = bool(rates and all(abs(rate - 1.0) <= 0.15 for rate in rates))
    higher_than_1hz = bool(rates and any(rate > 1.15 for rate in rates))
    if not successful:
        recommended_grid = "blocked_no_successful_tribe_probe"
    elif close_to_1hz:
        recommended_grid = "tribe_native_approximately_1hz"
    elif higher_than_1hz:
        recommended_grid = "tribe_native_grid"
    else:
        recommended_grid = "tribe_native_segment_grid"
    summary = {
        "schema_version": "again_tribe_native_grid_probe_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "again_root": str(args.again_root),
        "output_root": str(output_root),
        "external_cache_root": str(cache_root),
        "runtime_contract": runtime_contract,
        "probe_media": {
            "temporary_aligned_probe_clips_created": not args.no_prepare_aligned_probe_clips and not args.dry_run,
            "reason": "AGAIN WebM files often report Duration:N/A; TRIBE/MoviePy requires duration metadata. Probe clips use audited annotation-covered boundaries.",
            "scope": "selected probe videos only",
            "final_again_media_created": False,
            "alignment_policy": ALIGNMENT_POLICY,
        },
        "videos_probed": len(selected),
        "successful_probes": len(successful),
        "probe_rows": probe_rows,
        "aggregate_grid": {
            "median_prediction_rate_hz": float(np.median(rates)) if rates else None,
            "min_prediction_rate_hz": float(np.min(rates)) if rates else None,
            "max_prediction_rate_hz": float(np.max(rates)) if rates else None,
            "dominant_timing_source": "inferred_duration_even_spacing" if successful else "unknown",
            "dominant_timing_confidence": "medium" if successful else "unknown",
        },
        "decision": {
            "recommended_benchmark_grid": recommended_grid,
            "one_hz_still_recommended": close_to_1hz,
            "higher_rates_supported": bool(higher_than_1hz),
            "two_hz_five_hz_thirtyfps_supported": False,
            "notes": "Do not fabricate high-rate cortical rows; use native TRIBE rows unless a real high-rate encoder path is added.",
        },
        "guardrails": {
            "full_again_encoding_run": False,
            "benchmark_run": False,
            "models_trained": False,
            "final_manifest_created": False,
            "veatic_outputs_modified": False,
            "fake_high_rate_cortical_rows_created": False,
        },
    }

    write_csv(output_root / "again_tribe_native_grid_probe.csv", probe_rows)
    write_csv(output_root / "again_native_grid_manifest_pilot.csv", manifest_rows)
    write_json(output_root / "again_native_grid_manifest_summary.json", {**summary, "manifest_summary_rows": manifest_summary_rows})
    (output_root / "again_native_grid_alignment_report.md").write_text(report_text(summary), encoding="utf-8")
    run_manifest = {
        "schema_version": "again_tribe_native_grid_probe_run_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "external_cache_root": str(cache_root),
        "dry_run": args.dry_run,
        "files_written": [
            str(output_root / "again_tribe_native_grid_probe.csv"),
            str(output_root / "again_native_grid_manifest_pilot.csv"),
            str(output_root / "again_native_grid_manifest_summary.json"),
            str(output_root / "again_native_grid_alignment_report.md"),
            str(output_root / "run_manifest.json"),
        ],
        **summary["guardrails"],
    }
    write_json(output_root / "run_manifest.json", run_manifest)
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "external_cache_root": str(cache_root),
                "videos_probed": len(selected),
                "successful_tribe_probes": len(successful),
                "recommended_benchmark_grid": recommended_grid,
                "timing_source": summary["aggregate_grid"]["dominant_timing_source"],
                "timing_confidence": summary["aggregate_grid"]["dominant_timing_confidence"],
                "one_hz_still_recommended": close_to_1hz,
                "two_hz_five_hz_thirtyfps_supported": False,
                "temporary_aligned_probe_clips_created": summary["probe_media"]["temporary_aligned_probe_clips_created"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
