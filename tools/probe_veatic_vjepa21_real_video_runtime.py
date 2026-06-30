#!/usr/bin/env python3
"""Run one real VEATIC video through the V-JEPA 2.1 MLX -> TRIBE path."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_external_path(raw: str, external_root: Path) -> Path:
    return Path(raw.replace("<external-assets-root>", str(external_root)))


def load_manifest_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_short_video(report_path: Path, video_id: str | None = None) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    videos = report["videos"]
    if video_id is not None:
        for video in videos:
            if str(video["video_id"]) == str(video_id):
                return video
        raise ValueError(f"video_id {video_id!r} not found in report {report_path}")
    return sorted(videos, key=lambda item: (float(item["duration_seconds"]), int(item["video_id"])))[0]


def ffprobe(path: Path) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate,avg_frame_rate,codec_name,duration,nb_frames",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        data = json.loads(proc.stdout or "{}")
        streams = data.get("streams") or []
        stream = streams[0] if streams else {}
    except Exception as exc:  # noqa: BLE001 - diagnostic path
        return {"ffprobe_error": str(exc)}

    def rate(value: str | None) -> float | None:
        if not value or value == "0/0":
            return None
        if "/" in value:
            num, den = value.split("/", 1)
            den_f = float(den)
            return float(num) / den_f if den_f else None
        return float(value)

    return {
        "codec_name": stream.get("codec_name", ""),
        "width": stream.get("width"),
        "height": stream.get("height"),
        "fps": rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate")),
        "duration_seconds": float(stream["duration"]) if stream.get("duration") else None,
        "nb_frames": int(stream["nb_frames"]) if str(stream.get("nb_frames", "")).isdigit() else None,
    }


def numeric_stats(array: np.ndarray, prefix: str = "") -> dict[str, Any]:
    arr = np.asarray(array)
    is_numeric = np.issubdtype(arr.dtype, np.number)
    row: dict[str, Any] = {
        f"{prefix}shape": json.dumps(list(arr.shape)),
        f"{prefix}ndim": int(arr.ndim),
        f"{prefix}dtype": str(arr.dtype),
        f"{prefix}storage_bytes": int(arr.nbytes),
    }
    if not is_numeric:
        return row
    finite = np.isfinite(arr)
    nonfinite_count = int((~finite).sum())
    row[f"{prefix}nonfinite_count"] = nonfinite_count
    if arr.size:
        clean = arr[finite] if finite.any() else np.asarray([], dtype=np.float64)
        row[f"{prefix}min"] = float(np.min(clean)) if clean.size else None
        row[f"{prefix}mean"] = float(np.mean(clean)) if clean.size else None
        row[f"{prefix}std"] = float(np.std(clean)) if clean.size else None
        row[f"{prefix}max"] = float(np.max(clean)) if clean.size else None
    return row


def npz_prediction_stats(path: Path, prefix: str = "") -> dict[str, Any]:
    if not path.exists():
        return {f"{prefix}exists": False}
    with np.load(path) as bundle:
        keys = list(bundle.files)
        row: dict[str, Any] = {
            f"{prefix}exists": True,
            f"{prefix}path": str(path),
            f"{prefix}array_keys": json.dumps(keys),
            f"{prefix}file_size_bytes": path.stat().st_size,
        }
        if "predictions" in bundle.files:
            predictions = np.asarray(bundle["predictions"])
            row.update(numeric_stats(predictions, prefix=prefix))
            row[f"{prefix}temporal_rows"] = int(predictions.shape[0]) if predictions.ndim >= 1 else 0
            row[f"{prefix}feature_width"] = int(predictions.shape[1]) if predictions.ndim >= 2 else None
        return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("benchmarks/veatic/veatic_manifest_124_complete_20260616.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("benchmarks/veatic/veatic_manifest_124_complete_20260616.report.json"))
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    from backend.app.config import Config
    from backend.app.services.tribe_adapter import TribeAdapter

    external_root = Path(Config.NEURAL_BRIDGE_EXTERNAL_ROOT).expanduser()
    output_root = ROOT / "outputs" / f"veatic_vjepa21_real_video_runtime_{args.timestamp}"
    external_output_root = external_root / "benchmarks" / "veatic" / f"vjepa21_mlx_real_video_runtime_{args.timestamp}"
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

    os.environ["TRIBE_VIDEO_WINDOW_CACHE_DIR"] = str(external_output_root / "video_windows")
    Config.TRIBE_MLX_ENABLED = True
    Config.NEURO_PRIOR_SAVE_RAW_OUTPUT = True
    Config.TRIBE_VIDEO_ENCODER_BACKEND = "mlx"
    Config.TRIBE_VIDEO_ENCODER_MLX_DIR = str(external_root / "models" / "vjepa21_mlx" / "vitg")
    Config.TRIBE_VJEPA21_IMAGE_SIZE = int(getattr(Config, "TRIBE_VJEPA21_IMAGE_SIZE", 384))
    Config.TRIBE_VIDEO_NUM_FRAMES = 64

    adapter = TribeAdapter()
    selected_config = adapter._config_update()
    started = time.time()
    result = adapter.predict(
        stimulus_text="",
        stimulus_type="video",
        media_path=str(video_path),
        output_dir=str(external_output_root / video_id),
        backend="tribe_mlx",
    )
    encoding_seconds = time.time() - started

    new_npz = external_output_root / video_id / "tribe_raw_output.npz"
    old_npz = external_root / "benchmarks" / "veatic" / "tribe_cache" / video_id / "tribe_raw_output.npz"
    new_stats = npz_prediction_stats(new_npz, prefix="new_")
    old_stats = npz_prediction_stats(old_npz, prefix="old_")
    metadata = ffprobe(video_path)
    storage_bytes = sum(path.stat().st_size for path in external_output_root.rglob("*") if path.is_file())

    success = bool(result.get("success")) and bool(new_stats.get("new_exists")) and int(new_stats.get("new_nonfinite_count", 1)) == 0
    shape_compatible = new_stats.get("new_feature_width") == 20484 and int(new_stats.get("new_temporal_rows", 0)) > 0
    old_shape_compatible = (
        bool(old_stats.get("old_exists"))
        and new_stats.get("new_feature_width") == old_stats.get("old_feature_width")
        and abs(int(new_stats.get("new_temporal_rows", 0)) - int(old_stats.get("old_temporal_rows", 0))) <= 2
    )
    stats_sane = bool(success and np.isfinite(float(new_stats.get("new_std", 0.0))) and float(new_stats.get("new_std", 0.0)) > 0.0)
    three_video_pilot_approved = bool(success and shape_compatible and stats_sane)

    runtime_probe_row = {
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
        "selected_model_config_json": json.dumps({
            "video_encoder_mlx_dir": Config.TRIBE_VIDEO_ENCODER_MLX_DIR,
            "image_size": Config.TRIBE_VJEPA21_IMAGE_SIZE,
            "frames_per_clip": Config.TRIBE_VIDEO_NUM_FRAMES,
            "cache_model_name": selected_config.get("data.video_feature.cache_model_name"),
        }, sort_keys=True),
        "image_size": Config.TRIBE_VJEPA21_IMAGE_SIZE,
        "frames_per_clip": Config.TRIBE_VIDEO_NUM_FRAMES,
        "encoding_time_seconds": round(encoding_seconds, 3),
        "storage_size_bytes": storage_bytes,
        "error": result.get("error", ""),
        "real_veatic_videos_encoded": 1,
        "full_veatic_encoding_run": False,
        "again_encoding_run": False,
        "benchmark_run": False,
        "models_trained": False,
        "old_veatic_cache_modified": False,
        "vjepa21_claim_made": False,
        **new_stats,
    }
    comparison_row = {
        "video_id": video_id,
        "new_vs_old_feature_width_match": new_stats.get("new_feature_width") == old_stats.get("old_feature_width"),
        "new_vs_old_temporal_rows_delta": (
            int(new_stats.get("new_temporal_rows", 0)) - int(old_stats.get("old_temporal_rows", 0))
            if old_stats.get("old_exists") else None
        ),
        "old_vs_new_shape_compatible": old_shape_compatible,
        "stats_sane_compared_with_old": stats_sane and bool(old_stats.get("old_exists")),
        **new_stats,
        **old_stats,
    }
    write_csv(output_root / "veatic_vjepa21_real_video_runtime.csv", [runtime_probe_row])
    write_csv(output_root / "veatic_vjepa21_old_vs_new_shape_stats.csv", [comparison_row])

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
        "guardrails": {
            "real_veatic_videos_encoded": 1,
            "full_veatic_encoding_run": False,
            "again_encoding_run": False,
            "benchmark_run": False,
            "models_trained": False,
            "old_veatic_cache_modified": False,
            "vjepa21_claim_made": False,
        },
        "success": success,
        "shape_compatible_with_downstream_tensor_export": shape_compatible,
        "old_vs_new_shape_compatible": old_shape_compatible,
        "three_video_pilot_approved": three_video_pilot_approved,
        "new_output_npz": str(new_npz),
        "old_output_npz_read_only": str(old_npz),
    }
    write_json(output_root / "run_manifest.json", run_manifest)

    report = "\n".join([
        "# VEATIC V-JEPA 2.1 MLX Real-Video Runtime Probe",
        "",
        "## Verdict",
        f"- V-JEPA 2.1 MLX TRIBE run on real VEATIC video: `{str(success).lower()}`",
        f"- Selected video: `{video_id}`",
        f"- Finite cortical predictions: `{str(int(new_stats.get('new_nonfinite_count', 1)) == 0).lower()}`",
        f"- Downstream shape compatible: `{str(shape_compatible).lower()}`",
        f"- Old-vs-new temporal shape roughly compatible: `{str(old_shape_compatible).lower()}`",
        f"- 3-video pilot approved: `{str(three_video_pilot_approved).lower()}`",
        "",
        "## Selected Video",
        f"- Path: `{video_path}`",
        f"- Duration: `{selected_video.get('duration_seconds')}` seconds",
        f"- FPS: `{metadata.get('fps') or selected_video.get('fps')}`",
        f"- Resolution: `{metadata.get('width')}x{metadata.get('height')}`",
        "",
        "## New V-JEPA 2.1 Output",
        f"- NPZ: `{new_npz}`",
        f"- Keys: `{new_stats.get('new_array_keys')}`",
        f"- Predictions shape: `{new_stats.get('new_shape')}`",
        f"- Temporal rows: `{new_stats.get('new_temporal_rows')}`",
        f"- Feature width: `{new_stats.get('new_feature_width')}`",
        f"- Nonfinite count: `{new_stats.get('new_nonfinite_count')}`",
        f"- Min/mean/std/max: `{new_stats.get('new_min')}` / `{new_stats.get('new_mean')}` / `{new_stats.get('new_std')}` / `{new_stats.get('new_max')}`",
        f"- Encoding time: `{round(encoding_seconds, 3)}` seconds",
        f"- Storage size: `{storage_bytes}` bytes",
        "",
        "## Old Cache Read-Only Comparison",
        f"- Old NPZ exists: `{str(bool(old_stats.get('old_exists'))).lower()}`",
        f"- Old shape: `{old_stats.get('old_shape')}`",
        f"- Old temporal rows: `{old_stats.get('old_temporal_rows')}`",
        f"- Old nonfinite count: `{old_stats.get('old_nonfinite_count')}`",
        f"- Old min/mean/std/max: `{old_stats.get('old_min')}` / `{old_stats.get('old_mean')}` / `{old_stats.get('old_std')}` / `{old_stats.get('old_max')}`",
        "",
        "## Guardrails",
        "- real_veatic_videos_encoded: `1`",
        "- full_veatic_encoding_run: `false`",
        "- again_encoding_run: `false`",
        "- benchmark_run: `false`",
        "- models_trained: `false`",
        "- old_veatic_cache_modified: `false`",
        "- vjepa21_claim_made: `false`",
        "",
        "No performance claims are made from this one-video runtime probe.",
        "",
    ])
    (output_root / "veatic_vjepa21_real_video_runtime_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(run_manifest, indent=2, sort_keys=True))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
