"""Fail-fast gate before expensive OpenLAV TRIBE extraction.

This script intentionally avoids model inference. It checks that the benchmark
contract, dataset, cache paths, MPS exactness patch, and finalizer semantics are
ready before a multi-hour extraction run starts.
"""

from __future__ import annotations

import csv
import importlib.util
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from app.config import Config  # noqa: E402
from app.services.tribe_adapter import TribeAdapter  # noqa: E402


EXPECTED_VIDEO_FRAMES = 64
EXPECTED_EXTRACTION_CONTRACT = "official_64_frame_exact_chunked_attention"
EXPECTED_FEATURE_SCHEMA = "neuro_calibration_features_v2"
DTYPE_PARITY_GLOB = "vjepa_dtype_parity_64_mps*.json"
MIN_DTYPE_PARITY_REPORTS = 3
MAX_RELATIVE_RMSE_FOR_DTYPE_PARITY = 0.001
MIN_PEARSON_FOR_DTYPE_PARITY = 0.999999


def check(condition: bool, label: str, details: Any = None) -> dict[str, Any]:
    return {"ok": bool(condition), "label": label, "details": details}


def load_finalizer() -> Any:
    script = BACKEND / "scripts" / "watch_and_finalize_openlav.py"
    spec = importlib.util.spec_from_file_location("watch_and_finalize_openlav", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ffprobe_ok(video: Path) -> bool:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(video)],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def dtype_parity_reports() -> tuple[list[dict[str, Any]], list[str]]:
    report_dir = ROOT / "benchmarks" / "openlav"
    reports: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in sorted(report_dir.glob(DTYPE_PARITY_GLOB)):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data["_path"] = str(path)
            reports.append(data)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: {exc}")
    return reports, errors


def dtype_parity_summary(reports: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [report.get("metrics", {}) for report in reports if report.get("passes_contract")]
    rel_rmse = [float(metric.get("relative_rmse_to_reference_std", float("inf"))) for metric in metrics]
    pearson = [float(metric.get("pearson", 0.0)) for metric in metrics]
    return {
        "report_count": len(reports),
        "passing_report_count": sum(1 for report in reports if report.get("passes_contract")),
        "max_relative_rmse_to_reference_std": max(rel_rmse) if rel_rmse else None,
        "min_pearson": min(pearson) if pearson else None,
        "reports": [report.get("_path") for report in reports],
    }


def main() -> None:
    labels_path = Path(
        os.environ.get(
            "OPENLAV_LABELS",
            "/Volumes/onn. Drive/Neural Bridge/datasets/openlav_tools/export/video_data.csv",
        )
    )
    videos_dir = Path(
        os.environ.get(
            "OPENLAV_VIDEOS_DIR",
            "/Volumes/onn. Drive/Neural Bridge/datasets/openlav_videos",
        )
    )
    cache_dir = Path(
        os.environ.get(
            "OPENLAV_TRIBE_CACHE_DIR",
            "/Volumes/onn. Drive/Neural Bridge/benchmarks/openlav/tribe_cache",
        )
    )
    adapter = TribeAdapter()
    checks: list[dict[str, Any]] = []

    labels: list[dict[str, str]] = []
    if labels_path.exists():
        with labels_path.open(newline="", encoding="utf-8-sig") as handle:
            labels = list(csv.DictReader(handle))
    codes = [row.get("video_code", "") for row in labels]
    checks.append(check(labels_path.exists(), "OpenLAV labels file exists", str(labels_path)))
    checks.append(check(len(labels) == 188, "OpenLAV official label row count is 188", len(labels)))
    checks.append(check(len(codes) == len(set(codes)), "OpenLAV video codes are unique"))
    checks.append(check(all(row.get("valence") and row.get("arousal") for row in labels), "Targets include valence and arousal"))

    videos = sorted(videos_dir.glob("*.webm"))
    missing_videos = sorted(set(codes) - {path.stem for path in videos})
    zero_byte = [path.name for path in videos if path.stat().st_size == 0]
    checks.append(check(videos_dir.exists(), "OpenLAV videos directory exists", str(videos_dir)))
    checks.append(check(len(videos) == len(labels) == 188, "Video count matches labels", {"videos": len(videos), "labels": len(labels)}))
    checks.append(check(not missing_videos, "No labelled videos are missing", missing_videos[:10]))
    checks.append(check(not zero_byte, "No zero-byte videos", zero_byte[:10]))
    if videos:
        sampled = [videos[0], videos[len(videos) // 2], videos[-1]]
        checks.append(check(all(ffprobe_ok(path) for path in sampled), "ffprobe works on sampled videos", [path.name for path in sampled]))

    checks.append(check(Config.TRIBE_VIDEO_NUM_FRAMES == EXPECTED_VIDEO_FRAMES, "TRIBE uses official 64 video frames", Config.TRIBE_VIDEO_NUM_FRAMES))
    checks.append(check(Config.TRIBE_VIDEO_DTYPE in {"float32", "float16", "bfloat16"}, "TRIBE video dtype is explicit", Config.TRIBE_VIDEO_DTYPE))
    checks.append(check(os.environ.get("TRIBE_MPS_CHUNKED_ATTENTION", "false").lower() == "true", "Exact chunked MPS attention enabled"))
    checks.append(check(os.environ.get("TRIBE_VJEPA_SELECTIVE_HIDDEN_STATES", "true").lower() == "true", "Selective hidden-state capture enabled"))
    checks.append(check(Config.TRIBE_VIDEO_DEVICE.lower() == "mps", "Video extractor requested on MPS", Config.TRIBE_VIDEO_DEVICE))
    checks.append(check(Config.TRIBE_ALLOW_UNSAFE_VITG_MPS, "MPS ViT-G gate allows patched exact path", Config.TRIBE_ALLOW_UNSAFE_VITG_MPS))
    checks.append(check(str(Config.TRIBE_CACHE_DIR).startswith("/Volumes/"), "TRIBE cache is on external volume", Config.TRIBE_CACHE_DIR))
    checks.append(check(str(os.environ.get("TMPDIR", "")).startswith("/Volumes/"), "TMPDIR is on external volume", os.environ.get("TMPDIR")))
    checks.append(check(cache_dir.as_posix().startswith("/Volumes/"), "OpenLAV cache is on external volume", str(cache_dir)))

    try:
        import torch
        from neuralset.extractors.video import _HFVideoModel, _mps_chunked_sdpa_attention_forward

        checks.append(check(torch.backends.mps.is_available(), "Torch MPS is available"))
        checks.append(check(callable(_mps_chunked_sdpa_attention_forward), "Exact chunked SDPA patch imports"))
        checks.append(check("cache_n_layers" in inspect.signature(_HFVideoModel.__init__).parameters, "Selective layer patch is installed"))
    except Exception as exc:
        checks.append(check(False, "MPS/NeuralSet exactness patch check", f"{type(exc).__name__}: {exc}"))

    video_encoder = adapter._resolve_path(Config.TRIBE_VIDEO_ENCODER_LOCAL_DIR)
    checks.append(check(adapter._looks_like_encoder_model_dir(video_encoder), "Cortical V-JEPA2 encoder assets are complete", video_encoder))
    mlx_dir = Path(adapter._resolve_path(Config.TRIBE_MLX_DIR))
    checks.append(check((mlx_dir / "tribev2_mlx_float32.npz").exists(), "TRIBE MLX cortical head weights exist", str(mlx_dir)))

    if Config.TRIBE_VIDEO_DTYPE != "float32":
        reports, errors = dtype_parity_reports()
        summary = dtype_parity_summary(reports)
        checks.append(check(not errors, "V-JEPA dtype parity reports are readable", errors))
        checks.append(check(
            summary["passing_report_count"] >= MIN_DTYPE_PARITY_REPORTS,
            "Non-fp32 V-JEPA dtype has enough passing parity reports",
            summary,
        ))
        checks.append(check(
            summary["max_relative_rmse_to_reference_std"] is not None
            and summary["max_relative_rmse_to_reference_std"] <= MAX_RELATIVE_RMSE_FOR_DTYPE_PARITY,
            "Non-fp32 V-JEPA relative RMSE is below strict tolerance",
            summary,
        ))
        checks.append(check(
            summary["min_pearson"] is not None
            and summary["min_pearson"] >= MIN_PEARSON_FOR_DTYPE_PARITY,
            "Non-fp32 V-JEPA Pearson parity is above strict tolerance",
            summary,
        ))

    finalizer = load_finalizer()
    checks.append(check(finalizer.EXPECTED_VIDEO_FRAMES == EXPECTED_VIDEO_FRAMES, "Finalizer frame contract matches gate"))
    checks.append(check(finalizer.EXPECTED_EXTRACTION_CONTRACT == EXPECTED_EXTRACTION_CONTRACT, "Finalizer extraction contract matches gate"))
    checks.append(check(finalizer.EXPECTED_FEATURE_SCHEMA == EXPECTED_FEATURE_SCHEMA, "Finalizer feature schema matches gate"))

    complete_current = finalizer.completed_count()
    report = {
        "schema_version": "openlav_scientific_readiness_gate_v1",
        "ready": all(entry["ok"] for entry in checks),
        "current_complete_count": complete_current,
        "expected_contract": {
            "video_num_frames": EXPECTED_VIDEO_FRAMES,
            "video_dtype": Config.TRIBE_VIDEO_DTYPE,
            "dtype_parity_required": Config.TRIBE_VIDEO_DTYPE != "float32",
            "video_extraction_contract": EXPECTED_EXTRACTION_CONTRACT,
            "feature_schema": EXPECTED_FEATURE_SCHEMA,
            "benchmark_controls": [
                "grouped holdout",
                "mean baseline",
                "neutral neuro",
                "shuffled cortical",
                "shuffled subcortical",
                "shuffled labels",
                "regularized linear guardrail",
                "CatBoost primary model",
            ],
        },
        "checks": checks,
    }
    print(json.dumps(report, indent=2))
    if not report["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
