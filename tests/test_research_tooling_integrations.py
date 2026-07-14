"""Real-package contracts for the optional research-tooling integrations."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

pytest.importorskip("mlx.core")
mlflow = pytest.importorskip("mlflow")
pytest.importorskip("optuna")
pl = pytest.importorskip("polars")
pytest.importorskip("shap")

from integrations import (  # noqa: E402
    AcceleratedObjectiveResult,
    MLflowRun,
    RunProvenance,
    TrainOnlyStudySpec,
    collect_mlx,
    explain_model_behavior,
    require_accelerator,
    run_train_only_study,
    scan_table,
    write_table,
)


def test_apple_accelerators_execute_real_probes() -> None:
    assert require_accelerator("mlx").available
    assert "gpu" in require_accelerator("mlx").detail.lower()
    assert require_accelerator("mps").available


def test_polars_round_trip_and_direct_mlx_handoff(tmp_path: Path) -> None:
    source = tmp_path / "features.parquet"
    frame = pl.DataFrame({"feature_a": [1.0, 2.0], "feature_b": [3.0, 4.0]})
    write_table(frame, source)

    assert scan_table(source, columns=["feature_b"]).collect().columns == ["feature_b"]
    values = collect_mlx(source, columns=["feature_a", "feature_b"])

    np.testing.assert_allclose(np.asarray(values), [[1.0, 3.0], [2.0, 4.0]])


def test_optuna_objective_runs_on_mlx_and_records_split_contract() -> None:
    import mlx.core as mx

    spec = TrainOnlyStudySpec(
        study_name="neural-bridge-real-mlx-test",
        n_trials=3,
        sampler_seed=17,
        accelerator_backend="mlx",
        load_if_exists=False,
    )

    def objective(trial, train_indices, validation_indices):
        scale = trial.suggest_float("scale", 0.25, 1.0)
        values = mx.asarray([*train_indices, *validation_indices], dtype=mx.float32)
        score = mx.mean(values * scale)
        mx.eval(score)
        return AcceleratedObjectiveResult(float(np.asarray(score)), "mlx")

    study = run_train_only_study(
        spec,
        objective,
        inner_train_indices=[0, 1, 2],
        inner_validation_indices=[3, 4],
    )

    assert len(study.trials) == 3
    assert study.user_attrs["neural_bridge.scope"] == "inner_train_validation_only"
    assert study.user_attrs["neural_bridge.accelerator_backend"] == "mlx"
    assert all(
        trial.user_attrs["neural_bridge.accelerator_backend"] == "mlx"
        for trial in study.trials
    )


def test_optuna_rejects_split_overlap() -> None:
    spec = TrainOnlyStudySpec("overlap", 1, 1, "mlx", load_if_exists=False)
    with pytest.raises(ValueError, match="overlap"):
        run_train_only_study(
            spec,
            lambda *_: AcceleratedObjectiveResult(0.0, "mlx"),
            inner_train_indices=[0, 1],
            inner_validation_indices=[1, 2],
        )


def test_mlflow_records_accelerator_and_noncanonical_provenance(tmp_path: Path) -> None:
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifact = tmp_path / "result.txt"
    artifact.write_text("real integration artifact\n", encoding="utf-8")
    provenance = RunProvenance(
        git_commit="a" * 40,
        dataset_manifest_sha256="b" * 64,
        split_manifest_sha256="c" * 64,
        feature_manifest_sha256="d" * 64,
        target="integration_test_target",
        architecture="integration_test_head",
        validation_protocol="inner_train_validation_only",
        seed=7,
        accelerator_backend="mlx",
    )

    with MLflowRun(
        tracking_uri=tracking_uri,
        experiment_name="neural-bridge-integration-tests",
        run_name="real-mlx-run",
        provenance=provenance,
        artifact_location=(tmp_path / "artifacts").as_uri(),
    ) as run:
        run.log_metrics({"score": 0.75})
        run.log_artifact(artifact, artifact_path="smoke")
        run_id = run.run_id

    assert run_id is not None
    stored = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri).get_run(run_id)
    assert stored.data.tags["neural_bridge.accelerator_verified"] == "true"
    assert stored.data.tags["neural_bridge.canonical_evidence"] == "false"
    assert stored.data.params["accelerator_backend"] == "mlx"
    assert stored.data.metrics["score"] == pytest.approx(0.75)


def test_official_shap_uses_mlx_prediction_batches() -> None:
    import mlx.core as mx

    weights = mx.asarray([0.5, -0.25, 1.0], dtype=mx.float32)

    def predict_mlx(values):
        return values @ weights

    bundle = explain_model_behavior(
        predict_mlx,
        background=np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
        samples=np.asarray([[2.0, 1.0, 0.5]], dtype=np.float32),
        feature_names=["a", "b", "c"],
        background_is_inner_train_only=True,
    )

    assert bundle.predictor_backend == "mlx_gpu"
    assert bundle.explanation.values.shape == (1, 3)
    assert "not causal" in bundle.claim_boundary


def test_shap_rejects_nontraining_background() -> None:
    with pytest.raises(ValueError, match="inner-training"):
        explain_model_behavior(
            lambda values: values,
            background=np.ones((1, 2)),
            samples=np.ones((1, 2)),
            feature_names=["a", "b"],
            background_is_inner_train_only=False,
        )
