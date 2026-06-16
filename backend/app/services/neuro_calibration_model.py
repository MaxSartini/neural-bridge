"""Grouped held-out supervised calibration for frozen neuro-response features."""

from __future__ import annotations

import json
import hashlib
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np


class NeuroCalibrationModel:
    """Train one conservative CatBoost regressor per observed response axis."""

    SCHEMA_VERSION = "neuro_catboost_calibrator_v1"

    def predict(
        self,
        features: np.ndarray,
        model_dir: str,
        feature_names: Iterable[str] | None = None,
    ) -> Dict[str, float]:
        """Run only explicitly trained local calibration axes."""
        from catboost import CatBoostRegressor

        vector = np.nan_to_num(np.asarray(features, dtype=np.float32)).reshape(1, -1)
        root = Path(model_dir).expanduser().resolve()
        report = json.loads((root / "calibration_report.json").read_text(encoding="utf-8"))
        expected = int(report["feature_shape"][1])
        if vector.shape[1] != expected:
            raise ValueError(f"Expected {expected} calibration features, got {vector.shape[1]}")
        if feature_names is None:
            raise ValueError("Calibration prediction requires feature names to verify column ordering")
        supplied_hash = self._feature_name_hash(feature_names)
        if supplied_hash != report.get("feature_name_sha256"):
            raise ValueError("Calibration feature names or ordering do not match the trained model")
        output = {}
        for axis, details in report["axes"].items():
            model = CatBoostRegressor()
            model.load_model(details["model_path"])
            output[axis] = float(np.clip(model.predict(vector)[0], 0.0, 1.0))
        return output

    def train(
        self,
        features: np.ndarray,
        targets: Dict[str, np.ndarray],
        groups: Iterable[str],
        output_dir: str,
        feature_names: Iterable[str] | None = None,
        test_size: float = 0.25,
        seed: int = 33,
        model_version: str = "unversioned",
    ) -> Dict[str, Any]:
        from catboost import CatBoostRegressor
        from sklearn.model_selection import GroupShuffleSplit

        x = np.nan_to_num(np.asarray(features, dtype=np.float32))
        group_array = np.asarray(list(groups))
        if x.ndim != 2 or x.shape[0] < 20:
            raise ValueError("Calibration requires at least 20 rows of 2D frozen features")
        if group_array.shape[0] != x.shape[0] or np.unique(group_array).size < 2:
            raise ValueError("Calibration requires aligned rows from at least two holdout groups")
        names = list(feature_names or [f"feature_{index}" for index in range(x.shape[1])])
        if len(names) != x.shape[1] or len(set(names)) != len(names):
            raise ValueError("Calibration feature names must be unique and aligned with feature columns")

        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(x, groups=group_array))
        target_dir = Path(output_dir).expanduser().resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        metrics: Dict[str, Any] = {}
        if not targets:
            raise ValueError("Calibration requires at least one target axis")

        for axis, target_values in sorted(targets.items()):
            y = np.asarray(target_values, dtype=np.float32)
            if y.shape != (x.shape[0],):
                raise ValueError(f"Target {axis} must have shape {(x.shape[0],)}, got {y.shape}")
            if not np.all(np.isfinite(y)):
                raise ValueError(f"Target {axis} contains non-finite values")
            if np.any((y < 0.0) | (y > 1.0)):
                raise ValueError(f"Target {axis} must be normalized to [0, 1]")
            model_args = dict(
                iterations=300,
                depth=6,
                learning_rate=0.04,
                loss_function="RMSE",
                random_seed=seed,
                verbose=False,
                allow_writing_files=False,
                thread_count=4,
            )
            validation_model = CatBoostRegressor(**model_args)
            validation_model.fit(x[train_idx], y[train_idx])
            predicted = np.clip(validation_model.predict(x[test_idx]), 0.0, 1.0)
            observed = y[test_idx]
            error = predicted - observed
            model_path = target_dir / f"{axis}.cbm"
            deployment_model = CatBoostRegressor(**model_args)
            deployment_model.fit(x, y)
            deployment_model.save_model(str(model_path))
            metrics[axis] = {
                "n_train": int(train_idx.size),
                "n_test": int(test_idx.size),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "brier": float(np.mean(np.square(error))),
                "mean_baseline_mae": float(
                    np.mean(np.abs(np.mean(y[train_idx]) - observed))
                ),
                "mean_baseline_rmse": float(
                    np.sqrt(np.mean(np.square(np.mean(y[train_idx]) - observed)))
                ),
                "prediction_mean": float(np.mean(predicted)),
                "observed_mean": float(np.mean(observed)),
                "model_path": str(model_path),
            }

        report = {
            "schema_version": self.SCHEMA_VERSION,
            "model_version": model_version,
            "feature_shape": list(x.shape),
            "feature_names": names,
            "feature_name_sha256": self._feature_name_hash(names),
            "group_count": int(np.unique(group_array).size),
            "train_group_sha256": self._feature_name_hash(
                sorted(str(value) for value in set(group_array[train_idx]))
            ),
            "test_group_sha256": self._feature_name_hash(
                sorted(str(value) for value in set(group_array[test_idx]))
            ),
            "feature_to_row_ratio": float(x.shape[1] / x.shape[0]),
            "split": "group_shuffle_holdout",
            "deployment_fit": "all_rows_after_holdout_evaluation",
            "seed": seed,
            "runtime_versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "catboost": self._package_version("catboost"),
                "scikit_learn": self._package_version("scikit-learn"),
            },
            "axes": metrics,
            "limitations": [
                "Promotion requires repeated participant, experiment, time, topic, and modality holdouts.",
                "This model estimates association, not causal behavioral effects.",
                "High feature-to-row ratios require simple-model baselines, dimensionality reduction, and repeated grouped validation.",
            ],
        }
        (target_dir / "calibration_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        return report

    @staticmethod
    def _feature_name_hash(feature_names: Iterable[str]) -> str:
        encoded = json.dumps(list(feature_names), ensure_ascii=False, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _package_version(package: str) -> str:
        try:
            return version(package)
        except PackageNotFoundError:
            return "unavailable"
