"""Resumably cache full TRIBE and neuro-response IR output for OpenLAV."""

import argparse
import hashlib
import inspect
import json
import os
import sys
import time
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from app.config import Config  # noqa: E402
from app.services.neuro_response_ir import NeuroResponseIRBuilder  # noqa: E402
from app.services.tribe_adapter import TribeAdapter  # noqa: E402

RUN_MODES = ("cortical_fast_default", "full_research", "subcortical_ablation")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_video_encoder(adapter: TribeAdapter, configured_path: str) -> str:
    local = Path(adapter._resolve_path(configured_path))
    if not adapter._looks_like_encoder_model_dir(str(local)):
        raise RuntimeError(
            "Exact cortical video encoder is unavailable at "
            f"{local}. OpenLAV cortical inference requires "
            f"{Config.TRIBE_VIDEO_ENCODER_ID}; do not substitute the subcortical ViT-L encoder."
        )
    return str(local)


def validate_mps_runtime_contract() -> None:
    if Config.TRIBE_VIDEO_DEVICE != "mps" or Config.TRIBE_VIDEO_NUM_FRAMES != 64:
        return
    from neuralset.extractors.video import _HFVideoModel, _mps_chunked_sdpa_attention_forward

    del _mps_chunked_sdpa_attention_forward
    parameters = inspect.signature(_HFVideoModel.__init__).parameters
    if "cache_n_layers" not in parameters:
        raise RuntimeError(
            "The bounded exact 64-frame Neuralset MPS patch is missing. "
            "Refusing to run an unsafe unbounded V-JEPA2 path."
        )


def scientific_contract(contract: dict) -> dict:
    """Exclude exact runtime controls that cannot change model outputs."""
    return {
        key: value
        for key, value in contract.items()
        if key not in {"mps_memory_fraction", "video_attention_query_chunk_size"}
    }


def feature_schema(output_dir: Path) -> str | None:
    metadata_path = output_dir / "neuro_response_ir.json"
    if not metadata_path.exists():
        return None
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8")).get(
            "feature_contract", {}
        ).get("schema_version")
    except (OSError, json.JSONDecodeError):
        return None


def ir_feature_count(output_dir: Path) -> int | None:
    metadata_path = output_dir / "neuro_response_ir.json"
    if not metadata_path.exists():
        return None
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8")).get(
            "feature_contract", {}
        ).get("feature_count")
    except (OSError, json.JSONDecodeError):
        return None


def has_required_subcortical_output(output_dir: Path) -> bool:
    if not Config.TRIBE_ENABLE_SUBCORTICAL:
        return True
    raw_path = output_dir / "tribe_raw_output.npz"
    metadata_path = output_dir / "neuro_response_ir.json"
    arrays_path = output_dir / "neuro_response_ir.npz"
    if not raw_path.exists() or not metadata_path.exists() or not arrays_path.exists():
        return False
    try:
        import numpy as np

        with np.load(raw_path) as raw:
            if "subcortical_predictions" not in raw.files:
                return False
        with np.load(arrays_path) as arrays:
            if "subcortical_summary_features" not in arrays.files:
                return False
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        contract = metadata.get("feature_contract", {})
        return bool(contract.get("includes_subcortical")) and contract.get("feature_count") == 854
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def has_required_ir_output(output_dir: Path, *, require_subcortical: bool) -> bool:
    raw_path = output_dir / "tribe_raw_output.npz"
    metadata_path = output_dir / "neuro_response_ir.json"
    arrays_path = output_dir / "neuro_response_ir.npz"
    if not raw_path.exists() or not metadata_path.exists() or not arrays_path.exists():
        return False
    try:
        import numpy as np

        with np.load(raw_path) as raw:
            if "predictions" not in raw.files:
                return False
            if require_subcortical and "subcortical_predictions" not in raw.files:
                return False
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        contract = metadata.get("feature_contract", {})
        if contract.get("schema_version") != "neuro_calibration_features_v2":
            return False
        if bool(contract.get("includes_subcortical")) != require_subcortical:
            return False
        if require_subcortical and contract.get("feature_count") != 854:
            return False
        with np.load(arrays_path) as arrays:
            if "calibration_feature_vector" not in arrays.files:
                return False
            if require_subcortical and "subcortical_summary_features" not in arrays.files:
                return False
        return True
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--videos-dir",
        default="/Volumes/onn. Drive/Neural Bridge/datasets/openlav_videos",
    )
    parser.add_argument(
        "--cache-dir",
        default="/Volumes/onn. Drive/Neural Bridge/benchmarks/openlav/tribe_cache",
    )
    parser.add_argument("--backend", default=Config.NEURO_PRIOR_MODE)
    parser.add_argument(
        "--video-encoder-dir",
        default=os.environ.get(
            "TRIBE_VIDEO_ENCODER_LOCAL_DIR",
            "/Volumes/onn. Drive/Neural Bridge/models/cortical-upstream/facebook-vjepa2-vitg-fpc64-256",
        ),
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--run-mode",
        choices=RUN_MODES,
        default="cortical_fast_default",
        help=(
            "cortical_fast_default disables subcortical inference. "
            "full_research and subcortical_ablation explicitly enable it."
        ),
    )
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    Config.TRIBE_ENABLE_SUBCORTICAL = args.run_mode in {"full_research", "subcortical_ablation"}
    validate_mps_runtime_contract()
    videos = sorted(Path(args.videos_dir).expanduser().resolve().glob("*.webm"))
    videos = videos[args.offset:args.offset + args.limit if args.limit else None]
    cache_root = Path(args.cache_dir).expanduser().resolve()
    runtime_contract = {
        "mps_memory_fraction": Config.TRIBE_MPS_MEMORY_FRACTION,
        "video_attention_query_chunk_size": int(
            os.environ.get("TRIBE_MPS_ATTENTION_QUERY_CHUNK_SIZE", "0")
        ),
    }
    contract = {
        "run_mode": args.run_mode,
        "backend": args.backend,
        "tribe_model_id": Config.TRIBE_MODEL_ID,
        "subcortical_model_id": Config.TRIBE_SUBCORTICAL_MODEL_ID,
        "subcortical_enabled": Config.TRIBE_ENABLE_SUBCORTICAL,
        "tribe_device": Config.TRIBE_DEVICE,
        "video_device_requested": Config.TRIBE_VIDEO_DEVICE,
        "video_dtype": Config.TRIBE_VIDEO_DTYPE,
        "video_num_frames": Config.TRIBE_VIDEO_NUM_FRAMES,
        "video_attention_strategy": (
            "exact_mps_query_chunked_sdpa"
            if os.environ.get("TRIBE_MPS_CHUNKED_ATTENTION", "false").lower() == "true"
            else "model_default"
        ),
        "video_extraction_contract": (
            "official_64_frame_exact_chunked_attention"
            if Config.TRIBE_VIDEO_NUM_FRAMES == 64
            and os.environ.get("TRIBE_MPS_CHUNKED_ATTENTION", "false").lower() == "true"
            else "memory_bounded_non_official_frame_adaptation"
            if Config.TRIBE_VIDEO_NUM_FRAMES != 64
            else "official_64_frame"
        ),
        "ir_schema_version": NeuroResponseIRBuilder.SCHEMA_VERSION,
        "subcortical_policy": (
            "disabled_default_cortical_fast_path"
            if not Config.TRIBE_ENABLE_SUBCORTICAL
            else "explicit_experimental_research_mode"
        ),
    }
    adapter = TribeAdapter()
    exact_video_encoder = validate_video_encoder(adapter, args.video_encoder_dir) if not args.dry_run else str(
        Path(adapter._resolve_path(args.video_encoder_dir))
    )
    Config.TRIBE_VIDEO_ENCODER_LOCAL_DIR = exact_video_encoder
    contract["video_device_effective"] = adapter._config_update()[
        "data.video_feature.image.device"
    ]
    contract["cortical_video_encoder"] = exact_video_encoder
    statuses = []
    for index, video in enumerate(videos, start=1):
        output = cache_root / video.stem
        stimulus_hash = sha256(video)
        cache_key = hashlib.sha256(
            json.dumps({"stimulus_hash": stimulus_hash, **contract}, sort_keys=True).encode()
        ).hexdigest()
        status_path = output / "cache_status.json"
        if status_path.exists() and not args.force:
            previous = json.loads(status_path.read_text(encoding="utf-8"))
            previous_contract = scientific_contract(previous.get("model_contract", {}))
            if (
                previous.get("complete")
                and previous.get("stimulus_sha256") == stimulus_hash
                and previous_contract == contract
                and feature_schema(output) == "neuro_calibration_features_v2"
                and has_required_ir_output(output, require_subcortical=Config.TRIBE_ENABLE_SUBCORTICAL)
            ):
                statuses.append({"stimulus_id": video.stem, "status": "cached"})
                continue
        status = {
            "stimulus_id": video.stem,
            "video_path": str(video),
            "stimulus_sha256": stimulus_hash,
            "cache_key": cache_key,
            "model_contract": contract,
            "runtime_contract": runtime_contract,
            "complete": False,
        }
        statuses.append(status)
        print(json.dumps({"progress": f"{index}/{len(videos)}", **status}), flush=True)
        if args.dry_run:
            continue
        output.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        inference_started = time.perf_counter()
        result = adapter.predict(
            stimulus_type="video",
            media_path=str(video),
            output_dir=str(output),
            backend=args.backend,
        )
        status["timings_seconds"] = {
            "tribe_inference": round(time.perf_counter() - inference_started, 3)
        }
        if not result.get("success"):
            status["error"] = result.get("error", "TRIBE inference failed")
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
            continue
        raw_path = Path(result["raw_output_path"])
        if Config.TRIBE_ENABLE_SUBCORTICAL:
            try:
                import numpy as np

                with np.load(raw_path) as raw_bundle:
                    has_subcortical = "subcortical_predictions" in raw_bundle.files
            except (OSError, ValueError):
                has_subcortical = False
            if not has_subcortical:
                status["error"] = (
                    "Subcortical output missing from raw TRIBE archive while "
                    "TRIBE_ENABLE_SUBCORTICAL=true"
                )
                status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
                continue
        ir_started = time.perf_counter()
        NeuroResponseIRBuilder().build_from_npz(str(raw_path), str(output))
        if not has_required_ir_output(output, require_subcortical=Config.TRIBE_ENABLE_SUBCORTICAL):
            status["error"] = (
                "Neuro IR contract violation for selected run mode: expected cortical "
                "features only in cortical_fast_default, or 854-feature cortical+"
                "subcortical vector in explicit research/ablation mode."
            )
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
            continue
        status["timings_seconds"]["ir_build"] = round(
            time.perf_counter() - ir_started, 3
        )
        status["timings_seconds"]["total"] = round(
            status["timings_seconds"]["tribe_inference"]
            + status["timings_seconds"]["ir_build"],
            3,
        )
        status["complete"] = True
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    print(json.dumps({"processed": len(statuses), "statuses": statuses}, indent=2))
    failures = [
        status
        for status in statuses
        if status.get("status") != "cached" and not status.get("complete")
    ]
    if failures and not args.dry_run:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
