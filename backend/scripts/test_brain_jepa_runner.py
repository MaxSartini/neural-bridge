"""Verify Brain-JEPA refuses incomplete translations by default."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.brain_jepa_runner import BrainJepaRunner  # noqa: E402


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "ir.npz"
        np.savez(source, schaefer400_trajectories=np.ones((9, 400), dtype=np.float32))
        runner = BrainJepaRunner("/not/run", "/not/run", "/not/run")
        try:
            runner.run(str(source), str(root / "out.safetensors"), str(root / "work"))
        except ValueError as error:
            assert "Refusing incomplete Brain-JEPA translation" in str(error)
            print(json.dumps({"brain_jepa_runner_gate_ok": True, "error": str(error)}))
            return
        raise AssertionError("Brain-JEPA runner accepted an incomplete translation")


if __name__ == "__main__":
    main()
