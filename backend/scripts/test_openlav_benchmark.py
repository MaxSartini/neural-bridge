"""Synthetic smoke test for OpenLAV grouped ablation benchmark output."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "backend" / ".venv" / "bin" / "python"
RUNNER = ROOT / "backend" / "scripts" / "run_openlav_benchmark.py"


def main() -> None:
    rng = np.random.default_rng(7)
    feature_names = [
        "cortical::L:G_and_S_cingul-Ant::mean",
        "cortical::R:G_insular_short::std",
        "subcortical::Left Amygdala::mean",
        "subcortical::Right Accumbens::std",
        "missingness::text::is_missing",
        "global::mean",
        "quality::retention_ratio",
    ]
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        manifest = root / "manifest.jsonl"
        rows = []
        for index in range(16):
            vector = rng.normal(size=len(feature_names)).astype(np.float32)
            vector[4] = 1.0
            feature_path = root / f"stim_{index:02d}.npz"
            np.savez_compressed(feature_path, calibration_feature_vector=vector)
            feature_path.with_suffix(".json").write_text(
                json.dumps({"feature_contract": {"feature_names": feature_names}}),
                encoding="utf-8",
            )
            target = float(np.clip(0.5 + 0.15 * vector[0] - 0.1 * vector[2], 0.0, 1.0))
            rows.append({
                "stimulus_id": f"stim_{index:02d}",
                "feature_path": str(feature_path),
                "group": f"group_{index:02d}",
                "targets": {"valence": target, "arousal": 1.0 - target},
                "cache_contract": {
                    "model_contract_hash": "contract",
                    "feature_names_hash": "features",
                    "cache_key": f"cache_{index}",
                    "stimulus_sha256": f"sha_{index}",
                },
            })
        manifest.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        output = root / "benchmark.json"
        subprocess.run(
            [
                str(PYTHON),
                str(RUNNER),
                str(manifest),
                "--output",
                str(output),
                "--minimum-rows",
                "12",
                "--n-splits",
                "2",
            ],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["schema_version"] == "openlav_grouped_ablation_v3"
        assert report["feature_contract"]["feature_to_row_ratio"] == len(feature_names) / len(rows)
        assert report["feature_sets"]["ultra_compact_neuro"]["count"] == len(feature_names)
        assert report["feature_sets"]["compact_neuro_affect"]["count"] == len(feature_names)
        assert report["model_families"]["guardrail"] == "variance_threshold_standardized_ridge_cv"
        for axis in ("valence", "arousal"):
            assert "regularized_linear_controls" in report["targets"][axis]
            assert "cortical_plus_subcortical_calibrated" in report["targets"][axis]["regularized_linear_controls"]
            assert "ultra_compact_neuro" in report["targets"][axis]["regularized_linear_controls"]
            assert "compact_neuro_affect" in report["targets"][axis]["regularized_linear_controls"]
            assert "real_cortical_real_subcortical" in report["component_effects"][axis]
            assert "real_cortical_real_subcortical" not in report["sanity_checks"][axis]
            assert "random_gaussian_cortical_subcortical" in report["sanity_checks"][axis]
        assert output.with_suffix(".leakage.json").exists()
    print(json.dumps({"openlav_benchmark_ok": True}))


if __name__ == "__main__":
    main()
