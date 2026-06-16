"""Smoke-test ROI calibration on TRIBE-shaped synthetic cortical predictions."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.neuro_roi_calibrator import NeuroRoiCalibrator, TRIBE_CORTICAL_VERTICES  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(7)
    preds = rng.normal(0, 0.02, size=(4, TRIBE_CORTICAL_VERTICES)).astype("float32")
    profile = NeuroRoiCalibrator().calibrate_predictions(preds)

    assert "roi_summary" in profile
    assert "behavioural_axes" in profile
    assert profile["calibration_trace"]["method"] == "destrieux_surface_percentile_adapter_v1"
    for key in [
        "salience_score",
        "threat_score",
        "reward_score",
        "arousal_score",
        "uncertainty_score",
        "memory_relevance_score",
        "approach_bias",
        "avoidance_bias",
        "polarisation_risk",
        "virality_pressure",
    ]:
        assert 0.0 <= float(profile[key]) <= 1.0, (key, profile[key])

    print({
        "roi_calibrator_ok": True,
        "method": profile["calibration_trace"]["method"],
        "top_parcels": len(profile["roi_summary"]["top_parcels"]),
        "axes": list(profile["behavioural_axes"].keys()),
    })


if __name__ == "__main__":
    main()
