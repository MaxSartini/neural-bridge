"""Cache VEATIC TRIBE outputs for a small, resumable gated batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from app.config import Config  # noqa: E402
from app.services.tribe_adapter import TribeAdapter  # noqa: E402

RUN_MODES = ("cortical_fast_default",)


def external_root() -> Path:
    fallback = BACKEND_ROOT.parent / "external_assets"
    return Path(os.environ.get("NEURAL_BRIDGE_EXTERNAL_ROOT", str(fallback))).expanduser()


def external_path(*parts: str) -> str:
    return str(external_root().joinpath(*parts))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_videos(report_path: Path, limit: int, video_ids: set[str] | None) -> list[dict[str, Any]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    videos = report["videos"]
    if video_ids:
        videos = [video for video in videos if str(video["video_id"]) in video_ids]
    videos = sorted(videos, key=lambda item: (float(item["duration_seconds"]), int(item["video_id"])))
    return videos[:limit] if limit else videos


def has_required_raw(raw_path: Path) -> bool:
    if not raw_path.exists():
        return False
    try:
        with np.load(raw_path) as bundle:
            if "predictions" not in bundle.files:
                return False
        return True
    except (OSError, ValueError):
        return False


def configure_runtime(args: argparse.Namespace) -> dict[str, Any]:
    root = external_root()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HOME"] = str(root / "cache" / "huggingface")
    os.environ["TMPDIR"] = str(root / "tmp")
    os.environ["TRIBE_CACHE_DIR"] = str(root / "cache" / "tribev2")
    os.environ["TRIBE_VIDEO_WINDOW_CACHE_DIR"] = str(root / "cache" / "tribev2" / "video_windows")
    os.environ["TRIBE_MPS_CHUNKED_ATTENTION"] = "true"
    os.environ["TRIBE_MPS_ATTENTION_QUERY_CHUNK_SIZE"] = str(args.attention_query_chunk_size)
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = str(args.mps_high_watermark)
    os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = str(args.mps_low_watermark)

    Config.NEURO_PRIOR_MODE = "tribe_mlx"
    Config.TRIBE_MLX_ENABLED = True
    Config.NEURO_PRIOR_STRICT = True
    Config.TRIBE_DEVICE = "mps"
    Config.TRIBE_VIDEO_DEVICE = "mps"
    Config.TRIBE_VIDEO_DTYPE = args.video_dtype
    Config.TRIBE_VIDEO_NUM_FRAMES = args.video_num_frames
    Config.TRIBE_VIDEO_ENCODER_BACKEND = args.video_encoder_backend
    Config.TRIBE_VIDEO_ENCODER_MLX_DIR = args.cortical_video_encoder_mlx_dir
    Config.TRIBE_MPS_MEMORY_FRACTION = args.mps_memory_fraction
    Config.TRIBE_CACHE_DIR = os.environ["TRIBE_CACHE_DIR"]
    Config.TRIBE_APPLE_SILICON_SOURCE_DIR = args.apple_silicon_source_dir
    Config.TRIBE_MLX_DIR = args.tribe_mlx_dir
    Config.TRIBE_VIDEO_ENCODER_LOCAL_DIR = args.cortical_video_encoder_dir

    return {
        "run_mode": args.run_mode,
        "backend": "tribe_mlx",
        "device": "mps",
        "video_dtype": Config.TRIBE_VIDEO_DTYPE,
        "video_num_frames": Config.TRIBE_VIDEO_NUM_FRAMES,
        "video_encoder_backend": Config.TRIBE_VIDEO_ENCODER_BACKEND,
        "video_encoder_mlx_dir": Config.TRIBE_VIDEO_ENCODER_MLX_DIR,
        "mps_memory_fraction": Config.TRIBE_MPS_MEMORY_FRACTION,
        "mps_high_watermark": os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"],
        "mps_low_watermark": os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"],
        "attention": (
            "mlx_vjepa2_sdpa" if Config.TRIBE_VIDEO_ENCODER_BACKEND == "mlx"
            else "exact_mps_query_chunked_sdpa"
        ),
        "attention_query_chunk_size": os.environ["TRIBE_MPS_ATTENTION_QUERY_CHUNK_SIZE"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="benchmarks/veatic/veatic_manifest_1hz.report.json")
    parser.add_argument(
        "--cache-dir",
        default="",
        help=(
            "Defaults to the protected Torch/MPS cache for --video-encoder-backend torch "
            "and to a separate tribe_cache_mlx directory for MLX."
        ),
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--video-ids", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--run-mode",
        choices=RUN_MODES,
        default="cortical_fast_default",
        help="Current cortical-only VEATIC/TRIBE cache contract.",
    )
    parser.add_argument(
        "--cortical-only",
        action="store_true",
        help="Deprecated alias for --run-mode cortical_fast_default.",
    )
    parser.add_argument("--video-dtype", default="float16")
    parser.add_argument("--video-num-frames", type=int, default=64)
    parser.add_argument("--video-encoder-backend", choices=("auto", "mlx", "torch"), default="mlx")
    parser.add_argument("--mps-memory-fraction", type=float, default=0.35)
    parser.add_argument("--mps-high-watermark", type=float, default=0.45)
    parser.add_argument("--mps-low-watermark", type=float, default=0.25)
    parser.add_argument("--attention-query-chunk-size", type=int, default=128)
    parser.add_argument("--apple-silicon-source-dir", default=str(BACKEND_ROOT.parent / "external_models" / "tribev2-apple-silicon"))
    parser.add_argument("--tribe-mlx-dir", default=external_path("models", "tribe-mlx", "zimengxiong-tribev2-mlx"))
    parser.add_argument("--cortical-video-encoder-dir", default=external_path("models", "cortical-upstream", "facebook-vjepa2-vitg-fpc64-256"))
    parser.add_argument("--cortical-video-encoder-mlx-dir", default=external_path("models", "upstream-encoders-mlx", "facebook-vjepa2-vitg-fpc64-256"))
    args = parser.parse_args()
    if args.cortical_only:
        args.run_mode = "cortical_fast_default"
    if not args.cache_dir:
        args.cache_dir = (
            external_path("benchmarks", "veatic", "tribe_cache_mlx")
            if args.video_encoder_backend == "mlx"
            else external_path("benchmarks", "veatic", "tribe_cache")
        )

    report_path = Path(args.report).expanduser().resolve()
    cache_root = Path(args.cache_dir).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    video_ids = {item.strip() for item in args.video_ids.split(",") if item.strip()} or None
    videos = load_videos(report_path, args.limit, video_ids)
    contract = configure_runtime(args)
    adapter = TribeAdapter()
    statuses = []
    for index, video in enumerate(videos, start=1):
        video_id = str(video["video_id"])
        media_path = Path(video["media_path"]).expanduser().resolve()
        output = cache_root / video_id
        status_path = output / "cache_status.json"
        raw_path = output / "tribe_raw_output.npz"
        stimulus_hash = sha256(media_path)
        status = {
            "video_id": video_id,
            "media_path": str(media_path),
            "manifest_rows": video["manifest_rows"],
            "duration_seconds": video["duration_seconds"],
            "stimulus_sha256": stimulus_hash,
            "contract": contract,
            "complete": False,
        }
        if (
            not args.force
            and status_path.exists()
            and has_required_raw(raw_path)
        ):
            previous = json.loads(status_path.read_text(encoding="utf-8"))
            if previous.get("stimulus_sha256") == stimulus_hash and previous.get("contract") == contract:
                cached = {**status, "complete": True, "status": "cached"}
                statuses.append(cached)
                print(json.dumps({"progress": f"{index}/{len(videos)}", **cached}), flush=True)
                continue
        print(json.dumps({"progress": f"{index}/{len(videos)}", **status}), flush=True)
        statuses.append(status)
        if args.dry_run:
            continue
        output.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        started = time.perf_counter()
        result = adapter.predict(
            stimulus_type="video",
            media_path=str(media_path),
            output_dir=str(output),
            backend="tribe_mlx",
        )
        status["timings_seconds"] = {"tribe_inference": round(time.perf_counter() - started, 3)}
        status["result_success"] = bool(result.get("success"))
        status["raw_output_path"] = result.get("raw_output_path", "")
        if not result.get("success"):
            status["error"] = result.get("error", "TRIBE inference failed")
        elif not has_required_raw(raw_path):
            status["error"] = "Raw output missing required cortical predictions."
        else:
            with np.load(raw_path) as bundle:
                status["raw_shapes"] = {key: list(bundle[key].shape) for key in bundle.files}
            status["complete"] = True
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    summary = {
        "processed": len(statuses),
        "complete": sum(1 for item in statuses if item.get("complete")),
        "failed": [item for item in statuses if not item.get("complete")],
        "statuses": statuses,
    }
    batch_status_name = "batch_status.dry_run.json" if args.dry_run else "batch_status.json"
    (cache_root / batch_status_name).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if summary["failed"] and not args.dry_run:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
