"""Smoke-test grouped held-out CatBoost neuro calibration."""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.neuro_calibration_model import NeuroCalibrationModel  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(42)
    features = rng.normal(size=(80, 12)).astype(np.float32)
    target = np.clip(0.5 + features[:, 0] * 0.15 - features[:, 1] * 0.08, 0.0, 1.0)
    groups = [f"participant-{index % 8}" for index in range(80)]
    feature_names = [f"f{index}" for index in range(features.shape[1])]
    with tempfile.TemporaryDirectory() as temporary:
        report = NeuroCalibrationModel().train(
            features,
            {"response_probability": target},
            groups,
            temporary,
            feature_names=feature_names,
            model_version="smoke_v1",
        )
        metrics = report["axes"]["response_probability"]
        assert metrics["n_train"] > metrics["n_test"] > 0
        assert metrics["mae"] < 0.2
        assert metrics["mean_baseline_mae"] >= 0
        assert report["feature_to_row_ratio"] > 0
        assert report["model_version"] == "smoke_v1"
        assert report["runtime_versions"]["catboost"] != "unavailable"
        assert Path(metrics["model_path"]).exists()
        predicted = NeuroCalibrationModel().predict(features[0], temporary, feature_names=feature_names)
        assert 0.0 <= predicted["response_probability"] <= 1.0
        try:
            NeuroCalibrationModel().predict(features[0], temporary, feature_names=list(reversed(feature_names)))
            raise AssertionError("Feature-order mismatch was not rejected")
        except ValueError:
            pass
        try:
            NeuroCalibrationModel().predict(features[0], temporary)
            raise AssertionError("Missing feature contract was not rejected")
        except ValueError:
            pass
        try:
            NeuroCalibrationModel().train(
                features,
                {"response_probability": np.full(features.shape[0], np.nan)},
                groups,
                temporary,
                feature_names=feature_names,
            )
            raise AssertionError("Non-finite target values were not rejected")
        except ValueError:
            pass
    print({"neuro_calibration_model_ok": True, "mae": metrics["mae"]})


if __name__ == "__main__":
    main()
