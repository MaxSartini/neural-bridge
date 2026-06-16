"""Smoke-test fidelity-aware encoder routing."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.neuro_encoder_router import NeuroEncoderRouter  # noqa: E402


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        short = root / "short.npz"
        complete = root / "complete.npz"
        np.savez(short, schaefer400_trajectories=np.ones((9, 400), dtype=np.float32))
        np.savez(
            complete,
            schaefer400_trajectories=np.ones((160, 400), dtype=np.float32),
            tian50_trajectories=np.ones((160, 50), dtype=np.float32),
        )
        router = NeuroEncoderRouter()
        short_result = router.inspect(str(short))
        complete_result = router.inspect(str(complete))
        assert short_result["selected_encoder"] == "moment_1_small"
        assert complete_result["selected_encoder"] == "brain_jepa"
        assert short_result["brain_dit_compatible"] is False
        assert short_result["brainlm_compatible"] is False
        print(json.dumps({"neuro_encoder_router_ok": True}))


if __name__ == "__main__":
    main()
