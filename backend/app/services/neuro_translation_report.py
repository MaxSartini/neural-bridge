"""Build an auditable report for TRIBE representation translations."""

import json
from pathlib import Path
from typing import Any, Dict

from app.services.neuro_encoder_router import NeuroEncoderRouter


class NeuroTranslationReport:
    """Summarize preserved meaning, measured loss, and encoder eligibility."""

    def build(self, metadata_path: str) -> Dict[str, Any]:
        source_path = Path(metadata_path).expanduser().resolve()
        metadata = json.loads(source_path.read_text(encoding="utf-8"))
        source = metadata["source"]
        translations = metadata["translations"]
        arrays_path = metadata["arrays_path"]
        router = NeuroEncoderRouter().inspect(arrays_path)

        time_steps = int(source["shape"][0])
        sampling_hz = float(source["sampling_hz"])
        report = {
            "schema_version": "neuro_translation_report_v1",
            "source_metadata_path": str(source_path),
            "source": {
                "model": source["model"],
                "representation": source["representation"],
                "space": source["space"],
                "shape": source["shape"],
                "sampling_hz": sampling_hz,
                "duration_seconds": time_steps / sampling_hz,
                "sha256": source["source_sha256"],
            },
            "translation_quality": {
                name: {
                    key: value
                    for key, value in details.items()
                    if key
                    in {
                        "atlas",
                        "shape",
                        "method",
                        "reconstruction_mae",
                        "reconstruction_rmse",
                        "explained_variance",
                        "vertex_coverage",
                    }
                }
                for name, details in translations.items()
            },
            "semantics": metadata["semantics"],
            "encoder_routing": router,
            "interpretation": (
                "This report measures representation preservation and compatibility. "
                "It does not establish emotion, behavior, or outcome validity."
            ),
        }
        return report

    def write(self, metadata_path: str, output_path: str) -> Dict[str, Any]:
        report = self.build(metadata_path)
        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
