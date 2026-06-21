"""Cache VEATIC TRIBE outputs for a small, resumable gated batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
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
MODALITY_ORDER = ("text", "audio", "video")
OPERATIONAL_CONTRACT_KEYS = {
    "attention_query_chunk_size",
    "clear_mlx_cache_each_video",
    "clear_mlx_cache_each_window",
    "feature_cache_dir",
    "mps_high_watermark",
    "mps_low_watermark",
    "mps_memory_fraction",
    "restart_every_n_videos",
    "resume_policy",
    "vjepa21_compile_encoder",
}


def external_root() -> Path:
    fallback = BACKEND_ROOT.parent / "external_assets"
    return Path(os.environ.get("NEURAL_BRIDGE_EXTERNAL_ROOT", str(fallback))).expanduser()


def external_path(*parts: str) -> str:
    return str(external_root().joinpath(*parts))


def protected_veatic_cache_dir() -> Path:
    return external_root().joinpath("benchmarks", "veatic", "tribe_cache").resolve()


def is_protected_veatic_cache_write(cache_root: Path, *, video_encoder_backend: str) -> bool:
    if video_encoder_backend != "mlx":
        return False
    try:
        return cache_root.resolve() == protected_veatic_cache_dir()
    except OSError:
        return False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_media_path(raw_path: str) -> Path:
    path_text = raw_path.replace("<external-assets-root>", str(external_root()))
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = (BACKEND_ROOT.parent / path).resolve()
    return path.resolve()


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def cache_identity_contract(contract: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(contract, dict):
        return {}
    return {key: value for key, value in contract.items() if key not in OPERATIONAL_CONTRACT_KEYS}


def contracts_match(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    return cache_identity_contract(left) == cache_identity_contract(right)


def matching_complete_cache(
    *,
    status_path: Path,
    raw_path: Path,
    stimulus_hash: str,
    contract: dict[str, Any],
) -> dict[str, Any] | None:
    if not status_path.exists() or not has_required_raw(raw_path):
        return None
    previous = read_json(status_path)
    if previous.get("stimulus_sha256") == stimulus_hash and contracts_match(previous.get("contract"), contract):
        if previous.get("complete"):
            return previous
    return None


def matching_failed_cache(
    *,
    status_path: Path,
    stimulus_hash: str,
    contract: dict[str, Any],
) -> dict[str, Any] | None:
    if not status_path.exists():
        return None
    previous = read_json(status_path)
    if previous.get("stimulus_sha256") != stimulus_hash or not contracts_match(previous.get("contract"), contract):
        return None
    if previous.get("complete"):
        return None
    if previous.get("error") or previous.get("result_success") is False:
        return previous
    return None


def configure_runtime(args: argparse.Namespace) -> dict[str, Any]:
    root = external_root()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HOME"] = str(root / "cache" / "huggingface")
    os.environ["TMPDIR"] = str(root / "tmp")
    feature_cache_dir = Path(args.feature_cache_dir).expanduser().resolve() if args.feature_cache_dir else root / "cache" / "tribev2"
    os.environ["TRIBE_CACHE_DIR"] = str(feature_cache_dir)
    os.environ["TRIBE_VIDEO_WINDOW_CACHE_DIR"] = str(feature_cache_dir / "video_windows")
    os.environ["TRIBE_MPS_CHUNKED_ATTENTION"] = "true"
    os.environ["TRIBE_MPS_ATTENTION_QUERY_CHUNK_SIZE"] = str(args.attention_query_chunk_size)
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = str(args.mps_high_watermark)
    os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = str(args.mps_low_watermark)
    os.environ["TRIBE_VIDEO_WINDOW_BATCH_SIZE"] = str(args.video_window_batch_size)
    os.environ["TRIBE_MLX_CLEAR_CACHE_EACH_WINDOW"] = str(args.clear_mlx_cache_each_window).lower()
    os.environ["TRIBE_MLX_CLEAR_CACHE_EACH_VIDEO"] = str(args.clear_mlx_cache_each_video).lower()
    os.environ["TRIBE_ENCODER_RESTART_EVERY_N_VIDEOS"] = str(args.restart_every_n_videos)

    Config.NEURO_PRIOR_MODE = "tribe_mlx"
    Config.TRIBE_MLX_ENABLED = True
    Config.NEURO_PRIOR_STRICT = True
    Config.TRIBE_DEVICE = "mps"
    Config.TRIBE_VIDEO_DEVICE = "mps"
    Config.TRIBE_VIDEO_DTYPE = args.video_dtype
    Config.TRIBE_VIDEO_NUM_FRAMES = args.video_num_frames
    Config.TRIBE_VIDEO_WINDOW_BATCH_SIZE = args.video_window_batch_size
    Config.TRIBE_VIDEO_FRAME_SAMPLER = args.video_frame_sampler
    Config.TRIBE_VIDEO_ENCODER_BACKEND = args.video_encoder_backend
    Config.TRIBE_VIDEO_ENCODER_MLX_DIR = args.cortical_video_encoder_mlx_dir
    Config.TRIBE_VJEPA21_IMAGE_SIZE = args.vjepa21_image_size
    Config.TRIBE_VJEPA21_COMPILE_ENCODER = args.vjepa21_compile_encoder
    Config.TRIBE_MLX_CLEAR_CACHE_EACH_WINDOW = args.clear_mlx_cache_each_window
    Config.TRIBE_MLX_CLEAR_CACHE_EACH_VIDEO = args.clear_mlx_cache_each_video
    Config.TRIBE_ENCODER_RESTART_EVERY_N_VIDEOS = max(0, int(args.restart_every_n_videos))
    Config.TRIBE_FEATURE_FREQUENCY_HZ = args.feature_frequency_hz
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
        "video_window_batch_size": Config.TRIBE_VIDEO_WINDOW_BATCH_SIZE,
        "video_frame_sampler": Config.TRIBE_VIDEO_FRAME_SAMPLER,
        "video_encoder_backend": Config.TRIBE_VIDEO_ENCODER_BACKEND,
        "video_encoder_mlx_dir": Config.TRIBE_VIDEO_ENCODER_MLX_DIR,
        "vjepa21_image_size": Config.TRIBE_VJEPA21_IMAGE_SIZE,
        "vjepa21_compile_encoder": Config.TRIBE_VJEPA21_COMPILE_ENCODER,
        "feature_frequency_hz": Config.TRIBE_FEATURE_FREQUENCY_HZ,
        "clear_mlx_cache_each_window": Config.TRIBE_MLX_CLEAR_CACHE_EACH_WINDOW,
        "clear_mlx_cache_each_video": Config.TRIBE_MLX_CLEAR_CACHE_EACH_VIDEO,
        "restart_every_n_videos": Config.TRIBE_ENCODER_RESTART_EVERY_N_VIDEOS,
        "resume_policy": {
            "skip_completed_matching_contract": True,
            "per_video_status_path": "cache_status.json",
            "per_window_checkpoints": True,
            "video_window_checkpoint_dir": os.environ["TRIBE_VIDEO_WINDOW_CACHE_DIR"],
        },
        "feature_cache_dir": Config.TRIBE_CACHE_DIR,
        "required_modalities": list(args.required_modalities),
        "mps_memory_fraction": Config.TRIBE_MPS_MEMORY_FRACTION,
        "mps_high_watermark": os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"],
        "mps_low_watermark": os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"],
        "attention": (
            "mlx_vjepa2_sdpa" if Config.TRIBE_VIDEO_ENCODER_BACKEND == "mlx"
            else "exact_mps_query_chunked_sdpa"
        ),
        "attention_query_chunk_size": os.environ["TRIBE_MPS_ATTENTION_QUERY_CHUNK_SIZE"],
    }


def modality_status(raw_path: Path) -> dict[str, Any]:
    if not raw_path.exists():
        return {
            "present_modalities": [],
            "missing_modalities": list(MODALITY_ORDER),
            "modality_missing_flags": None,
        }
    with np.load(raw_path) as bundle:
        if "feature_modality_present_flags" in bundle.files:
            present_flags = [bool(item) for item in bundle["feature_modality_present_flags"].tolist()]
            present = [name for name, is_present in zip(MODALITY_ORDER, present_flags) if is_present]
            missing = [name for name, is_present in zip(MODALITY_ORDER, present_flags) if not is_present]
            return {
                "present_modalities": present,
                "missing_modalities": missing,
                "modality_missing_flags": [int(not item) for item in present_flags],
                "feature_modality_present_flags": [int(item) for item in present_flags],
            }
        if "modality_missing_flags" not in bundle.files:
            return {
                "present_modalities": ["unknown"],
                "missing_modalities": ["unknown"],
                "modality_missing_flags": None,
            }
        flags = [bool(item) for item in bundle["modality_missing_flags"].tolist()]
    present = [name for name, missing in zip(MODALITY_ORDER, flags) if not missing]
    missing = [name for name, missing in zip(MODALITY_ORDER, flags) if missing]
    return {
        "present_modalities": present,
        "missing_modalities": missing,
        "modality_missing_flags": [int(item) for item in flags],
    }


def missing_required_modalities(raw_path: Path, required: tuple[str, ...]) -> list[str]:
    status = modality_status(raw_path)
    missing = set(status["missing_modalities"])
    return [item for item in required if item in missing]


def should_restart_process(
    *,
    encoded_since_restart: int,
    restart_every_n_videos: int,
    remaining_videos: int,
    dry_run: bool,
) -> bool:
    return (
        not dry_run
        and int(restart_every_n_videos) > 0
        and int(remaining_videos) > 0
        and int(encoded_since_restart) >= int(restart_every_n_videos)
    )


def batch_status_filename(*, dry_run: bool, partial: bool, worker_id: str = "") -> str:
    if worker_id:
        suffix = "partial" if partial else "dry_run" if dry_run else "complete"
        return f"batch_status.worker_{worker_id}.{suffix}.json"
    if partial:
        return "batch_status.partial.json"
    return "batch_status.dry_run.json" if dry_run else "batch_status.json"


def write_batch_status(
    cache_root: Path,
    statuses: list[dict[str, Any]],
    *,
    dry_run: bool,
    partial: bool = False,
    worker_id: str = "",
) -> dict[str, Any]:
    summary = {
        "worker_id": worker_id,
        "processed": len(statuses),
        "complete": sum(1 for item in statuses if item.get("complete")),
        "failed": [item for item in statuses if not item.get("complete")],
        "statuses": statuses,
    }
    (cache_root / batch_status_filename(dry_run=dry_run, partial=partial, worker_id=worker_id)).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def claim_path_for(output: Path) -> Path:
    return output / "encode_claim.json"


def read_claim(claim_path: Path) -> dict[str, Any] | None:
    try:
        return read_json(claim_path)
    except (OSError, json.JSONDecodeError):
        return None


def claim_is_stale(claim_path: Path, timeout_seconds: float, *, now: float | None = None) -> bool:
    claim = read_claim(claim_path)
    if not claim:
        return True
    claimed_at = float(claim.get("claimed_at_unix", 0.0) or 0.0)
    return (now or time.time()) - claimed_at > float(timeout_seconds)


def try_claim_video(
    *,
    output: Path,
    video_id: str,
    worker_id: str,
    contract: dict[str, Any],
    claim_timeout_seconds: float,
) -> dict[str, Any] | None:
    output.mkdir(parents=True, exist_ok=True)
    claim_path = claim_path_for(output)
    claim = {
        "claim_id": f"{worker_id or 'single'}-{os.getpid()}-{time.time_ns()}",
        "worker_id": worker_id or "single",
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "video_id": video_id,
        "claimed_at_unix": time.time(),
        "claim_timeout_seconds": float(claim_timeout_seconds),
        "contract_digest": hashlib.sha256(
            json.dumps(cache_identity_contract(contract), sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    while True:
        try:
            fd = os.open(str(claim_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError:
            if not claim_is_stale(claim_path, claim_timeout_seconds):
                return None
            try:
                claim_path.unlink()
            except FileNotFoundError:
                continue
            except OSError:
                return None
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(claim, handle, indent=2)
        return claim


def release_claim(output: Path, claim: dict[str, Any] | None) -> None:
    if not claim:
        return
    claim_path = claim_path_for(output)
    current = read_claim(claim_path)
    if current and current.get("claim_id") != claim.get("claim_id"):
        return
    try:
        claim_path.unlink()
    except FileNotFoundError:
        pass


def process_video(
    *,
    video: dict[str, Any],
    index: int,
    total: int,
    args: argparse.Namespace,
    adapter: TribeAdapter,
    cache_root: Path,
    contract: dict[str, Any],
    worker_id: str = "",
) -> tuple[dict[str, Any], bool]:
    video_id = str(video["video_id"])
    media_path = resolve_media_path(str(video["media_path"]))
    output = cache_root / video_id
    status_path = output / "cache_status.json"
    raw_path = output / "tribe_raw_output.npz"
    stimulus_hash = sha256(media_path)
    status = {
        "worker_id": worker_id,
        "video_id": video_id,
        "media_path": str(media_path),
        "manifest_rows": video["manifest_rows"],
        "duration_seconds": video["duration_seconds"],
        "stimulus_sha256": stimulus_hash,
        "contract": contract,
        "complete": False,
    }
    if not args.force:
        previous = matching_complete_cache(
            status_path=status_path,
            raw_path=raw_path,
            stimulus_hash=stimulus_hash,
            contract=contract,
        )
        if previous:
            cached = {**status, "complete": True, "status": "cached"}
            print(json.dumps({"progress": f"{index}/{total}", **cached}), flush=True)
            return cached, False
        if getattr(args, "queue_skip_failed", False):
            failed = matching_failed_cache(
                status_path=status_path,
                stimulus_hash=stimulus_hash,
                contract=contract,
            )
            if failed:
                skipped = {**status, "status": "previous_failed", "error": failed.get("error", "")}
                print(json.dumps({"progress": f"{index}/{total}", **skipped}), flush=True)
                return skipped, False

    claim = None
    if worker_id and not args.dry_run:
        claim = try_claim_video(
            output=output,
            video_id=video_id,
            worker_id=worker_id,
            contract=contract,
            claim_timeout_seconds=args.claim_timeout_seconds,
        )
        if claim is None:
            return {**status, "status": "claimed_by_other_worker"}, False
        status["claim"] = claim

    print(json.dumps({"progress": f"{index}/{total}", **status}), flush=True)
    if args.dry_run:
        return status, False
    try:
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
        elif missing := missing_required_modalities(raw_path, args.required_modalities):
            status["modality_status"] = modality_status(raw_path)
            status["error"] = f"Missing required modalities: {', '.join(missing)}"
        else:
            with np.load(raw_path) as bundle:
                status["raw_shapes"] = {key: list(bundle[key].shape) for key in bundle.files}
            status["modality_status"] = modality_status(raw_path)
            status["complete"] = True
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        return status, bool(status.get("complete"))
    finally:
        release_claim(output, claim)


def worker_argv(raw_argv: list[str], worker_id: int) -> list[str]:
    stripped: list[str] = []
    skip_next = False
    options_with_values = {"--workers", "--worker-id"}
    for arg in raw_argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg in options_with_values:
            skip_next = True
            continue
        if arg.startswith("--workers=") or arg.startswith("--worker-id="):
            continue
        stripped.append(arg)
    return [
        sys.executable,
        raw_argv[0],
        *stripped,
        "--workers",
        "1",
        "--worker-id",
        str(worker_id),
    ]


def launch_worker_processes(worker_count: int) -> None:
    processes: list[subprocess.Popen[Any]] = []
    try:
        for worker_index in range(worker_count):
            cmd = worker_argv(sys.argv, worker_index)
            print(json.dumps({
                "status": "starting_tribe_cache_worker",
                "worker_id": str(worker_index),
                "cmd": cmd,
            }), flush=True)
            processes.append(subprocess.Popen(cmd))
        failed = []
        for worker_index, process in enumerate(processes):
            return_code = process.wait()
            if return_code:
                failed.append({"worker_id": str(worker_index), "return_code": return_code})
        if failed:
            print(json.dumps({"status": "worker_failure", "failed_workers": failed}), flush=True)
            raise SystemExit(1)
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
        raise


def run_video_loop(
    *,
    videos: list[dict[str, Any]],
    args: argparse.Namespace,
    cache_root: Path,
    contract: dict[str, Any],
    worker_id: str = "",
) -> dict[str, Any]:
    adapter = TribeAdapter()
    statuses = []
    encoded_since_restart = 0
    for index, video in enumerate(videos, start=1):
        status, encoded = process_video(
            video=video,
            index=index,
            total=len(videos),
            args=args,
            adapter=adapter,
            cache_root=cache_root,
            contract=contract,
            worker_id=worker_id,
        )
        if status.get("status") != "claimed_by_other_worker":
            statuses.append(status)
        if encoded:
            encoded_since_restart += 1
        if should_restart_process(
            encoded_since_restart=encoded_since_restart,
            restart_every_n_videos=Config.TRIBE_ENCODER_RESTART_EVERY_N_VIDEOS,
            remaining_videos=len(videos) - index,
            dry_run=args.dry_run,
        ):
            summary = write_batch_status(
                cache_root,
                statuses,
                dry_run=args.dry_run,
                partial=True,
                worker_id=worker_id,
            )
            print(json.dumps({
                "status": "restarting_process_for_memory_hygiene",
                "worker_id": worker_id,
                "encoded_since_restart": encoded_since_restart,
                "restart_every_n_videos": Config.TRIBE_ENCODER_RESTART_EVERY_N_VIDEOS,
                "resume_policy": contract["resume_policy"],
            }), flush=True)
            print(json.dumps(summary, indent=2), flush=True)
            os.execv(sys.executable, [sys.executable, *sys.argv])
    summary = write_batch_status(cache_root, statuses, dry_run=args.dry_run, worker_id=worker_id)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


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
        "--workers",
        type=int,
        default=1,
        help="Launch this many independent video-queue workers. Each worker claims whole videos atomically.",
    )
    parser.add_argument(
        "--worker-id",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--claim-timeout-seconds",
        type=float,
        default=6 * 60 * 60,
        help="Reclaim abandoned per-video encode claims older than this many seconds.",
    )
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
    parser.add_argument("--video-window-batch-size", type=int, default=Config.TRIBE_VIDEO_WINDOW_BATCH_SIZE)
    parser.add_argument("--video-frame-sampler", choices=("moviepy", "ffmpeg"), default=Config.TRIBE_VIDEO_FRAME_SAMPLER)
    parser.add_argument("--video-encoder-backend", choices=("auto", "mlx", "torch"), default="mlx")
    parser.add_argument("--vjepa21-image-size", type=int, default=Config.TRIBE_VJEPA21_IMAGE_SIZE)
    parser.add_argument(
        "--vjepa21-compile-encoder",
        action=argparse.BooleanOptionalAction,
        default=Config.TRIBE_VJEPA21_COMPILE_ENCODER,
    )
    parser.add_argument(
        "--clear-mlx-cache-each-window",
        action=argparse.BooleanOptionalAction,
        default=Config.TRIBE_MLX_CLEAR_CACHE_EACH_WINDOW,
        help="Clear MLX cache after every video window. Slower; use only under memory pressure.",
    )
    parser.add_argument(
        "--clear-mlx-cache-each-video",
        action=argparse.BooleanOptionalAction,
        default=Config.TRIBE_MLX_CLEAR_CACHE_EACH_VIDEO,
        help="Clear MLX cache after each video while keeping the window loop hot.",
    )
    parser.add_argument(
        "--restart-every-n-videos",
        type=int,
        default=Config.TRIBE_ENCODER_RESTART_EVERY_N_VIDEOS,
        help="Automatically re-exec after this many newly encoded videos. 0 disables process recycling.",
    )
    parser.add_argument(
        "--feature-frequency-hz",
        type=float,
        default=Config.TRIBE_FEATURE_FREQUENCY_HZ,
        help="Optional TRIBE feature-grid override. Leave unset for checkpoint default.",
    )
    parser.add_argument("--mps-memory-fraction", type=float, default=0.35)
    parser.add_argument("--mps-high-watermark", type=float, default=0.45)
    parser.add_argument("--mps-low-watermark", type=float, default=0.25)
    parser.add_argument("--attention-query-chunk-size", type=int, default=128)
    parser.add_argument("--apple-silicon-source-dir", default=str(BACKEND_ROOT.parent / "external_models" / "tribev2-apple-silicon"))
    parser.add_argument("--tribe-mlx-dir", default=external_path("models", "tribe-mlx", "zimengxiong-tribev2-mlx"))
    parser.add_argument("--cortical-video-encoder-dir", default=external_path("models", "cortical-upstream", "facebook-vjepa2-vitg-fpc64-256"))
    parser.add_argument("--cortical-video-encoder-mlx-dir", default=external_path("models", "upstream-encoders-mlx", "facebook-vjepa2-vitg-fpc64-256"))
    parser.add_argument(
        "--feature-cache-dir",
        default="",
        help="Feature-extractor cache root. Use a fresh directory for uncached pilot runs.",
    )
    parser.add_argument(
        "--required-modalities",
        nargs="+",
        choices=MODALITY_ORDER,
        default=["video"],
        help="Fail a completed cache item if these modalities are absent from modality_missing_flags.",
    )
    parser.add_argument(
        "--require-multimodal",
        action="store_true",
        help="Shortcut for --required-modalities text audio video.",
    )
    args = parser.parse_args()
    if args.workers > 1 and not args.worker_id and not args.dry_run:
        launch_worker_processes(args.workers)
        return
    args.queue_skip_failed = bool(args.worker_id)
    if args.cortical_only:
        args.run_mode = "cortical_fast_default"
    if args.require_multimodal:
        args.required_modalities = list(MODALITY_ORDER)
    args.required_modalities = tuple(args.required_modalities)
    if not args.cache_dir:
        args.cache_dir = (
            external_path("benchmarks", "veatic", "tribe_cache_mlx")
            if args.video_encoder_backend == "mlx"
            else external_path("benchmarks", "veatic", "tribe_cache")
        )

    report_path = Path(args.report).expanduser().resolve()
    cache_root = Path(args.cache_dir).expanduser().resolve()
    if is_protected_veatic_cache_write(cache_root, video_encoder_backend=args.video_encoder_backend):
        raise SystemExit(
            "Refusing to write MLX/V-JEPA outputs into the protected VEATIC-124 cache: "
            f"{cache_root}. Use a separate cache root such as "
            "<external-root>/benchmarks/veatic/tribe_cache_mlx or a timestamped pilot directory."
        )
    cache_root.mkdir(parents=True, exist_ok=True)
    video_ids = {item.strip() for item in args.video_ids.split(",") if item.strip()} or None
    videos = load_videos(report_path, args.limit, video_ids)
    contract = configure_runtime(args)
    summary = run_video_loop(
        videos=videos,
        args=args,
        cache_root=cache_root,
        contract=contract,
        worker_id=args.worker_id,
    )
    if summary["failed"] and not args.dry_run:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
