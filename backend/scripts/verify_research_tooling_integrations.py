#!/usr/bin/env python3
"""Run a small real-package smoke test for the research-tooling stack."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from integrations import (  # noqa: E402
    AcceleratedObjectiveResult,
    MLflowRun,
    RunProvenance,
    TrainOnlyStudySpec,
    collect_mlx,
    explain_model_behavior,
    require_accelerator,
    run_train_only_study,
    write_table,
)


def main() -> int:
    import mlflow
    import mlx.core as mx
    import optuna
    import polars as pl
    import shap

    mlx_status = require_accelerator("mlx")
    mps_status = require_accelerator("mps")
    with tempfile.TemporaryDirectory(prefix="neural-bridge-research-tooling-") as raw_tmp:
        tmp = Path(raw_tmp)

        table_path = write_table(
            pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}),
            tmp / "features.parquet",
        )
        mlx_table = collect_mlx(table_path, columns=["a", "b"])

        def objective(trial, train_indices, validation_indices):
            scale = trial.suggest_float("scale", 0.5, 1.0)
            value = mx.mean(mx.asarray([*train_indices, *validation_indices]) * scale)
            mx.eval(value)
            return AcceleratedObjectiveResult(float(np.asarray(value)), "mlx")

        study = run_train_only_study(
            TrainOnlyStudySpec("research-tooling-smoke", 2, 23, "mlx", load_if_exists=False),
            objective,
            inner_train_indices=[0, 1, 2],
            inner_validation_indices=[3, 4],
        )

        provenance = RunProvenance(
            git_commit="smoke-only-not-a-canonical-run",
            dataset_manifest_sha256="smoke-dataset",
            split_manifest_sha256="smoke-split",
            feature_manifest_sha256="smoke-features",
            target="smoke",
            architecture="smoke",
            validation_protocol="inner_train_validation_only",
            seed=23,
            accelerator_backend="mlx",
        )
        with MLflowRun(
            tracking_uri=f"sqlite:///{tmp / 'mlflow.db'}",
            experiment_name="neural-bridge-smoke",
            run_name="real-upstream-smoke",
            provenance=provenance,
            artifact_location=(tmp / "artifacts").as_uri(),
        ) as tracked:
            tracked.log_metrics({"best_value": study.best_value})
            mlflow_run_id = tracked.run_id

        weights = mx.asarray([0.5, -0.25], dtype=mx.float32)
        explanation = explain_model_behavior(
            lambda values: values @ weights,
            background=np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
            samples=np.asarray([[2.0, 1.0]], dtype=np.float32),
            feature_names=["a", "b"],
            background_is_inner_train_only=True,
        )

        result = {
            "accelerators": {"mlx": mlx_status.detail, "mps": mps_status.detail},
            "versions": {
                "mlflow": mlflow.__version__,
                "optuna": optuna.__version__,
                "polars": pl.__version__,
                "shap": shap.__version__,
            },
            "checks": {
                "polars_to_mlx_shape": list(mlx_table.shape),
                "optuna_trials": len(study.trials),
                "mlflow_run_created": bool(mlflow_run_id),
                "shap_values_shape": list(explanation.explanation.values.shape),
                "all_passed": True,
            },
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
