"""Local TRIBE activation viewer API.

This is a visualization surface only. It exposes downsampled cortical parcel
trajectories from cached TRIBE outputs without loading the models or running
inference.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from flask import jsonify, request, send_file

from . import neuro_viewer_bp
from ..services.cortical_roi_mapper import CorticalRoiMapper, TRIBE_CORTICAL_VERTICES
from ..utils.logger import get_logger

logger = get_logger("neural_bridge.api.neuro_viewer")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXTERNAL_ROOT = Path(
    os.environ.get("NEURAL_BRIDGE_EXTERNAL_ROOT", str(PROJECT_ROOT / "external_assets"))
).expanduser()
DEFAULT_CACHE_ROOT = Path(
    os.environ.get(
        "TRIBE_VIEWER_CACHE_ROOT",
        str(EXTERNAL_ROOT / "benchmarks" / "veatic" / "tribe_cache"),
    )
)
DEFAULT_MANIFEST_REPORT = Path(
    os.environ.get(
        "TRIBE_VIEWER_MANIFEST_REPORT",
        str(Path.cwd() / "benchmarks/veatic/veatic_manifest_1hz.report.json"),
    )
)


def _json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return {}


def _cache_root() -> Path:
    return Path(request.args.get("cache_root") or DEFAULT_CACHE_ROOT).expanduser()


def _video_dirs(cache_root: Path) -> list[Path]:
    if not cache_root.exists():
        return []
    return sorted(
        [item for item in cache_root.iterdir() if item.is_dir()],
        key=lambda item: int(item.name) if item.name.isdigit() else item.name,
    )


def _manifest_videos() -> dict[str, dict[str, Any]]:
    report = _json_file(DEFAULT_MANIFEST_REPORT)
    return {str(item.get("video_id")): item for item in report.get("videos", [])}


def _safe_video_dir(cache_root: Path, video_id: str) -> Path:
    if not video_id.replace("_", "").isalnum():
        raise ValueError("Invalid video_id")
    video_dir = (cache_root / video_id).resolve()
    if cache_root.resolve() not in video_dir.parents:
        raise ValueError("Invalid video path")
    return video_dir


def _normalize(values: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(values, dtype=np.float32))
    if arr.size == 0:
        return arr
    lo = float(np.percentile(arr, 2))
    hi = float(np.percentile(arr, 98))
    if math.isclose(lo, hi):
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _signed_intensity(values: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(values, dtype=np.float32))
    scale = float(np.percentile(np.abs(arr), 98))
    if scale <= 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip(arr / scale, -1.0, 1.0).astype(np.float32)


def _region_xy(label: str, index: int, count: int) -> dict[str, float]:
    hemi = "L" if label.startswith("L:") else "R" if label.startswith("R:") else ""
    name = label.split(":", 1)[-1].lower()
    base_x = 31 if hemi == "L" else 69 if hemi == "R" else 50
    if "occip" in name or "calcar" in name:
        y = 72
    elif "front" in name or "orbital" in name or "rectus" in name:
        y = 28
    elif "temporal" in name or "insul" in name:
        y = 58
    elif "cingul" in name or "precuneus" in name or "paracentral" in name:
        y = 42
    elif "pariet" in name or "intrapariet" in name or "postcentral" in name:
        y = 48
    else:
        y = 50
    offset = ((index % 5) - 2) * 4
    row_offset = ((index // 5) % 3 - 1) * 5
    x = base_x + (-offset if hemi == "L" else offset)
    return {"x": float(max(10, min(90, x))), "y": float(max(15, min(85, y + row_offset)))}


def _surface_mesh() -> dict[str, Any]:
    """Load the same fsaverage5 surface family used by TRIBE plotting."""
    import nibabel as nib
    from nilearn import datasets

    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    coords_all = []
    faces_all = []
    bg_all = []
    offset = 0
    for hemi in ("left", "right"):
        infl_xyz, _ = nib.load(getattr(fsaverage, f"infl_{hemi}")).darrays
        pial_xyz, faces = nib.load(getattr(fsaverage, f"pial_{hemi}")).darrays
        sulc = nib.load(getattr(fsaverage, f"sulc_{hemi}")).darrays[0].data.astype(np.float32)
        coords = infl_xyz.data.astype(np.float32) * 0.5 + pial_xyz.data.astype(np.float32) * 0.5
        if hemi == "left":
            coords[:, 0] = coords[:, 0] - coords[:, 0].max() - 4.0
        else:
            coords[:, 0] = coords[:, 0] - coords[:, 0].min() + 4.0
        coords_all.append(coords)
        faces_all.append(faces.data.astype(np.int32) + offset)
        bg_all.append(sulc)
        offset += coords.shape[0]

    coords = np.concatenate(coords_all, axis=0)
    faces = np.concatenate(faces_all, axis=0)
    bg = np.concatenate(bg_all, axis=0)
    bg = _normalize(bg)
    max_abs = float(np.max(np.abs(coords))) or 1.0
    coords = coords / max_abs
    return {
        "mesh": "fsaverage5",
        "coords": coords.round(5).tolist(),
        "faces": faces.tolist(),
        "background": bg.round(4).tolist(),
    }


def _vertex_activity(cortical: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(cortical, dtype=np.float32))
    scale = float(np.percentile(np.abs(arr), 99))
    if scale <= 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip(arr / scale, -1.0, 1.0).astype(np.float32)


def _cortical_regions(cortical: np.ndarray, max_regions: int) -> list[dict[str, Any]]:
    if cortical.ndim != 2 or cortical.shape[1] != TRIBE_CORTICAL_VERTICES:
        return []
    atlas = CorticalRoiMapper().load_destrieux_atlas()
    if atlas is None:
        return []
    rows: list[dict[str, Any]] = []
    offset = 0
    for hemisphere, label_map in (("L", atlas["left"]), ("R", atlas["right"])):
        for label_id, label in enumerate(atlas["labels"]):
            if label_id == 0:
                continue
            local_idx = np.where(label_map == label_id)[0]
            if local_idx.size == 0:
                continue
            idx = local_idx + offset
            trace = cortical[:, idx].mean(axis=1).astype(np.float32)
            rows.append(
                {
                    "label": f"{hemisphere}:{label}",
                    "vertex_count": int(idx.size),
                    "mean_abs": float(np.mean(np.abs(trace))),
                    "peak_abs": float(np.max(np.abs(trace))),
                    "trace": trace,
                }
            )
        offset += len(label_map)
    rows.sort(key=lambda row: row["mean_abs"], reverse=True)
    selected = rows[:max_regions]
    return [
        {
            "id": f"cortical-{index}",
            "label": row["label"],
            "kind": "cortical",
            "vertex_count": row["vertex_count"],
            "mean_abs": row["mean_abs"],
            "peak_abs": row["peak_abs"],
            "position": _region_xy(row["label"], index, len(selected)),
            "trace": _signed_intensity(row["trace"]).round(4).tolist(),
            "magnitude": _normalize(np.abs(row["trace"])).round(4).tolist(),
        }
        for index, row in enumerate(selected)
    ]


@neuro_viewer_bp.route("/progress", methods=["GET"])
def progress():
    cache_root = _cache_root()
    videos = []
    complete = 0
    for video_dir in _video_dirs(cache_root):
        status = _json_file(video_dir / "cache_status.json")
        raw_exists = (video_dir / "tribe_raw_output.npz").exists()
        is_complete = bool(status.get("complete")) and raw_exists
        complete += int(is_complete)
        videos.append(
            {
                "video_id": video_dir.name,
                "complete": is_complete,
                "raw_exists": raw_exists,
                "status": status.get("status", "unknown"),
                "duration_seconds": status.get("duration_seconds"),
                "manifest_rows": status.get("manifest_rows"),
            }
        )
    return jsonify(
        {
            "success": True,
            "data": {
                "cache_root": str(cache_root),
                "complete": complete,
                "total_seen": len(videos),
                "videos": videos,
            },
        }
    )


@neuro_viewer_bp.route("/videos", methods=["GET"])
def list_videos():
    cache_root = _cache_root()
    manifest = _manifest_videos()
    videos = []
    for video_dir in _video_dirs(cache_root):
        status = _json_file(video_dir / "cache_status.json")
        summary = _json_file(video_dir / "tribe_summary.json")
        raw_exists = (video_dir / "tribe_raw_output.npz").exists()
        video_id = video_dir.name
        videos.append(
            {
                "video_id": video_id,
                "complete": bool(status.get("complete")) and raw_exists,
                "raw_exists": raw_exists,
                "duration_seconds": status.get("duration_seconds") or manifest.get(video_id, {}).get("duration_seconds"),
                "manifest_rows": status.get("manifest_rows") or manifest.get(video_id, {}).get("manifest_rows"),
                "media_path": status.get("media_path") or manifest.get(video_id, {}).get("media_path"),
                "mean_activation_proxy": summary.get("mean_activation_proxy"),
                "temporal_variance_proxy": summary.get("temporal_variance_proxy"),
                "peak_response_proxy": summary.get("peak_response_proxy"),
                "backend": summary.get("backend") or status.get("contract", {}).get("backend"),
            }
        )
    videos.sort(key=lambda item: (not item["complete"], float(item["duration_seconds"] or 1e9), item["video_id"]))
    return jsonify({"success": True, "data": videos})


@neuro_viewer_bp.route("/videos/<video_id>/media", methods=["GET"])
def video_media(video_id: str):
    video_dir = _safe_video_dir(_cache_root(), video_id)
    status = _json_file(video_dir / "cache_status.json")
    media_path = Path(status.get("media_path") or "")
    if not media_path.exists():
        manifest = _manifest_videos()
        media_path = Path(manifest.get(video_id, {}).get("media_path") or "")
    if not media_path.exists():
        return jsonify({"success": False, "error": "Media file not found"}), 404
    return send_file(media_path, mimetype="video/mp4", conditional=True)


@neuro_viewer_bp.route("/videos/<video_id>/timeline", methods=["GET"])
def video_timeline(video_id: str):
    max_regions = max(6, min(80, request.args.get("max_regions", 28, type=int)))
    video_dir = _safe_video_dir(_cache_root(), video_id)
    raw_path = video_dir / "tribe_raw_output.npz"
    if not raw_path.exists():
        return jsonify({"success": False, "error": "TRIBE raw output not found"}), 404
    status = _json_file(video_dir / "cache_status.json")
    summary = _json_file(video_dir / "tribe_summary.json")
    with np.load(raw_path) as bundle:
        cortical = np.asarray(bundle["predictions"], dtype=np.float32)
    abs_cortical = np.abs(cortical)
    global_traces = {
        "mean": _signed_intensity(cortical.mean(axis=1)).round(4).tolist(),
        "mean_abs": _normalize(abs_cortical.mean(axis=1)).round(4).tolist(),
        "std": _normalize(cortical.std(axis=1)).round(4).tolist(),
        "peak_abs": _normalize(abs_cortical.max(axis=1)).round(4).tolist(),
        "p95_abs": _normalize(np.percentile(abs_cortical, 95, axis=1)).round(4).tolist(),
    }
    timepoints = int(cortical.shape[0])
    return jsonify(
        {
            "success": True,
            "data": {
                "video_id": video_id,
                "timepoints": timepoints,
                "timestamps_seconds": [float(index) for index in range(timepoints)],
                "contract": status.get("contract", {}),
                "summary": {
                    "mean_activation_proxy": summary.get("mean_activation_proxy"),
                    "temporal_variance_proxy": summary.get("temporal_variance_proxy"),
                    "peak_response_proxy": summary.get("peak_response_proxy"),
                    "event_quality": summary.get("event_quality"),
                    "segment_quality": summary.get("segment_quality"),
                },
                "global_traces": global_traces,
                "regions": {
                    "cortical": _cortical_regions(cortical, max_regions),
                },
            },
        }
    )


@neuro_viewer_bp.route("/videos/<video_id>/surface", methods=["GET"])
def video_surface(video_id: str):
    """Return real fsaverage5 mesh geometry plus TRIBE vertex activity traces."""
    video_dir = _safe_video_dir(_cache_root(), video_id)
    raw_path = video_dir / "tribe_raw_output.npz"
    if not raw_path.exists():
        return jsonify({"success": False, "error": "TRIBE raw output not found"}), 404
    with np.load(raw_path) as bundle:
        cortical = np.asarray(bundle["predictions"], dtype=np.float32)
    if cortical.ndim != 2 or cortical.shape[1] != TRIBE_CORTICAL_VERTICES:
        return jsonify(
            {
                "success": False,
                "error": f"Expected cortical predictions with {TRIBE_CORTICAL_VERTICES} vertices",
            }
        ), 422

    stride = max(1, min(4, request.args.get("time_stride", 1, type=int)))
    cortical = cortical[::stride]
    activity = _vertex_activity(cortical)
    mesh = _surface_mesh()
    return jsonify(
        {
            "success": True,
            "data": {
                "video_id": video_id,
                "contract": {
                    "source": "TRIBE v2 cached predictions",
                    "space": "fsaverage5",
                    "plotting_reference": "tribev2.plotting.PlotBrain(mesh='fsaverage5')",
                    "normalization": "signed vertex activity divided by per-video p99 absolute response",
                    "time_stride": stride,
                },
                "timepoints": int(activity.shape[0]),
                "timestamps_seconds": [float(index * stride) for index in range(activity.shape[0])],
                "surface": mesh,
                "activity": activity.round(4).tolist(),
            },
        }
    )
