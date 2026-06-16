"""Smoke-test the translation-quality report."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.neuro_translation_report import NeuroTranslationReport  # noqa: E402


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        arrays = root / "ir.npz"
        metadata = root / "ir.json"
        np.savez(arrays, schaefer400_trajectories=np.ones((8, 400), dtype=np.float32))
        metadata.write_text(
            json.dumps(
                {
                    "source": {
                        "model": "TRIBE v2",
                        "representation": "test",
                        "space": "fsaverage5",
                        "shape": [8, 20484],
                        "sampling_hz": 2.0,
                        "source_sha256": "test",
                    },
                    "translations": {
                        "schaefer400_cortical": {
                            "shape": [8, 400],
                            "explained_variance": 0.75,
                            "vertex_coverage": 0.91,
                        }
                    },
                    "semantics": {"preserved": [], "not_inferred": []},
                    "arrays_path": str(arrays),
                }
            ),
            encoding="utf-8",
        )
        report = NeuroTranslationReport().build(str(metadata))
        assert report["source"]["duration_seconds"] == 4.0
        assert report["encoder_routing"]["selected_encoder"] == "moment_1_small"
        print(json.dumps({"neuro_translation_report_ok": True}))


if __name__ == "__main__":
    main()
