"""Smoke-test faithful masked Brain-JEPA translation."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.brain_jepa_adapter import BrainJepaAdapter  # noqa: E402


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        source = Path(temporary) / "ir.npz"
        output = Path(temporary) / "brain_jepa.npz"
        safe_output = Path(temporary) / "brain_jepa.safetensors"
        trajectories = np.arange(9 * 400, dtype=np.float32).reshape(9, 400)
        np.savez(source, schaefer400_trajectories=trajectories)
        metadata = BrainJepaAdapter().build(str(source), str(output), str(safe_output))
        result = np.load(output)
        assert result["values"].shape == (1, 450, 160)
        assert result["roi_available_mask"].sum() == 400
        assert result["temporal_available_mask"].sum() == 9
        assert np.array_equal(result["values"][0, :400, :9], trajectories.T)
        assert np.count_nonzero(result["values"][0, 400:, :]) == 0
        assert safe_output.exists()
        assert metadata["production_eligible"] is False
        assert len(metadata["ineligibility_reasons"]) == 2

        complete_source = Path(temporary) / "complete_ir.npz"
        np.savez(
            complete_source,
            schaefer400_trajectories=np.ones((160, 400), dtype=np.float32),
            tian50_trajectories=np.ones((160, 50), dtype=np.float32),
        )
        complete_metadata = BrainJepaAdapter().build(
            str(complete_source),
            str(Path(temporary) / "complete_brain_jepa.npz"),
        )
        assert complete_metadata["production_eligible"] is True
        assert complete_metadata["ineligibility_reasons"] == []
        print(json.dumps({"brain_jepa_adapter_ok": True, **metadata}))


if __name__ == "__main__":
    main()
