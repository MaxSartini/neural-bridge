"""Smoke-test routing and plan generation without loading an encoder."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.neuro_bridge_pipeline import NeuroBridgePipeline  # noqa: E402


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        raw = root / "raw.npz"
        np.savez(raw, predictions=np.zeros((9, 20484), dtype=np.float32))
        result = NeuroBridgePipeline(str(BACKEND_ROOT), "/unused").run(
            str(raw),
            str(root / "run"),
            dry_run=True,
        )
        assert result["routing"]["selected_encoder"] == "moment_1_small"
        assert result["encoder_stage"]["name"] == "moment_1_small"
        assert (root / "run" / "neuro_bridge_plan.json").exists()
        print(json.dumps({"neuro_bridge_pipeline_ok": True}))


if __name__ == "__main__":
    main()
