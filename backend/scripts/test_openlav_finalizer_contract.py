"""Smoke-test that OpenLAV finalization counts only current v2 cache rows."""

import json
import tempfile
from pathlib import Path

import watch_and_finalize_openlav as finalizer


def write_status(root: Path, name: str, complete: bool, frames: int, contract: str, schema: str | None) -> None:
    target = root / name
    target.mkdir(parents=True)
    (target / "cache_status.json").write_text(
        json.dumps({
            "complete": complete,
            "model_contract": {
                "video_num_frames": frames,
                "video_extraction_contract": contract,
            },
        }),
        encoding="utf-8",
    )
    if schema is not None:
        (target / "neuro_response_ir.json").write_text(
            json.dumps({"feature_contract": {"schema_version": schema}}),
            encoding="utf-8",
        )


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        finalizer.CACHE = root
        write_status(
            root,
            "current_complete",
            True,
            64,
            "official_64_frame_exact_chunked_attention",
            "neuro_calibration_features_v2",
        )
        write_status(
            root,
            "legacy_complete",
            True,
            32,
            "memory_bounded_32_frame_apple_silicon_adaptation",
            "neuro_calibration_features_v1",
        )
        write_status(
            root,
            "current_missing_ir",
            True,
            64,
            "official_64_frame_exact_chunked_attention",
            None,
        )
        assert finalizer.completed_count() == 1
    print(json.dumps({"openlav_finalizer_contract_ok": True}))


if __name__ == "__main__":
    main()
