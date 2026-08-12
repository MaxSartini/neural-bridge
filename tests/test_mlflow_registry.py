from __future__ import annotations

from pathlib import Path

import pytest

from neural_bridge.mlflow_registry import (
    log_completed_output,
    phase_roots,
    scientific_metrics_from_object,
    scientific_run_data_from_file,
    start_run,
    status,
    sync_existing,
    tracking_uri,
)


def test_phase_roots_excludes_mlflow_storage(tmp_path: Path) -> None:
    (tmp_path / "runs" / "again" / "phase-01").mkdir(parents=True)
    (tmp_path / "runs" / "mlflow" / "again").mkdir(parents=True)

    assert phase_roots(tmp_path) == [("again", tmp_path / "runs" / "again" / "phase-01")]


def test_tracking_uri_uses_absolute_sqlite_path(tmp_path: Path) -> None:
    # Build the expectation the way the implementation does rather than gluing a
    # literal "/" onto a Path. The old form asserted a POSIX separator, so it
    # failed on Windows against a correct absolute path — a red suite for anyone
    # developing there, while CI stayed green on ubuntu.
    database = tmp_path / "mlflow.db"
    assert tracking_uri(database) == f"sqlite:///{database.absolute()}"
    assert Path(tracking_uri(database).removeprefix("sqlite:///")).is_absolute()


def test_scientific_metrics_exclude_inventory_and_summarise_results(tmp_path: Path) -> None:
    phase = tmp_path / "phase"
    phase.mkdir()
    (phase / "result.json").write_text(
        '{"rows": 99, "real_spearman": 0.4, "checks": {"audit_pass": true}, '
        '"records": [{"pr_auc": 0.2}, {"pr_auc": 0.4}]}'
    )
    (phase / "inventory_summary.json").write_text('{"bytes": 999}')
    (phase / "fold_metrics.csv").write_text("fold,pr_auc,seed\n0,0.3,1\n1,0.5,2\n")

    json_metrics, _ = scientific_run_data_from_file(phase / "result.json")
    csv_metrics, _ = scientific_run_data_from_file(phase / "fold_metrics.csv")

    assert json_metrics["model.continuous.spearman"] == 0.4
    assert json_metrics["quality.audit_pass"] == 1.0
    assert json_metrics["model.event.pr_auc"] == pytest.approx(0.3)
    assert csv_metrics["model.event.pr_auc"] == pytest.approx(0.4)
    assert "rows" not in json_metrics
    assert "bytes" not in json_metrics


def test_scientific_metrics_from_object_keeps_live_success_measures() -> None:
    metrics = scientific_metrics_from_object(
        {
            "records": [
                {"pooled_pr_auc": 0.2, "skill_delta_vs_ar": -0.1, "fold": 0},
                {"pooled_pr_auc": 0.4, "skill_delta_vs_ar": 0.1, "fold": 1},
            ]
        }
    )

    assert metrics["model.event.pr_auc"] == pytest.approx(0.3)
    assert metrics["delta_vs_ar.event.average_precision_skill"] == 0.0
    assert "fold" not in metrics


def test_result_parameters_keep_only_invariant_configuration(tmp_path: Path) -> None:
    result = tmp_path / "summary.csv"
    result.write_text(
        "target,architecture,seed,pr_auc,pca_width\n"
        "spike,temporal,1,0.3,128\n"
        "spike,temporal,2,0.4,128\n"
    )

    metrics, parameters = scientific_run_data_from_file(result)

    assert metrics["model.event.pr_auc"] == pytest.approx(0.35)
    assert parameters == {"architecture": "temporal", "pca_width": "128", "target": "spike"}


def test_native_run_references_external_output_without_copying(tmp_path: Path) -> None:
    database = tmp_path / "mlflow.db"
    artifact_root = tmp_path / "mlflow-artifacts"
    output = tmp_path / "result.json"
    output.write_text("{}")

    with start_run(
        "veatic-2.1",
        "stage1-ar-benchmark",
        database=database,
        mlflow_artifact_root=artifact_root,
    ):
        log_completed_output(
            output,
            parameters={"sources": "tribe_cortical"},
            metrics={"records": 1},
        )

    import mlflow

    run = mlflow.MlflowClient().search_runs(["1"])[0]
    assert run.data.tags["neural_bridge.output_path"] == str(output)
    assert run.data.params["sources"] == "tribe_cortical"
    assert run.data.metrics["records"] == 1
    assert not list(artifact_root.rglob("result.json"))
    assert status(database)["runs"] == 1


def test_sync_existing_creates_result_runs_without_inventory_metrics(tmp_path: Path) -> None:
    source = tmp_path / "artifacts" / "runs" / "again" / "phase-07" / "metrics"
    source.mkdir(parents=True)
    result = source / "result.json"
    result.write_text('{"real_spearman": 0.3, "ar_spearman": 0.2, "rows": 99}')
    (source / "fold_metrics.csv").write_text("fold,pr_auc\n0,0.4\n1,0.6\n")
    database = tmp_path / "mlflow.db"

    summary = sync_existing(
        artifact_root=tmp_path / "artifacts",
        database=database,
        mlflow_artifact_root=tmp_path / "mlflow-artifacts",
    )

    import mlflow

    runs = mlflow.MlflowClient().search_runs(["1"])
    assert summary["results"] == 1
    assert len(runs) == 1
    assert runs[0].data.tags["neural_bridge.source_path"] == str(source.parent)
    assert runs[0].data.tags["neural_bridge.source_file_count"] == "2"
    assert runs[0].data.metrics == {
        "baseline_ar.continuous.spearman": 0.2,
        "comparison.ar": 0.2,
        "comparison.delta_vs_ar": pytest.approx(0.1),
        "comparison.model": 0.3,
        "model.event.pr_auc": 0.5,
        "model.continuous.spearman": 0.3,
    }
    assert runs[0].data.tags["neural_bridge.primary_measure"] == "continuous.spearman"

    second = sync_existing(
        artifact_root=tmp_path / "artifacts",
        database=database,
        mlflow_artifact_root=tmp_path / "mlflow-artifacts",
    )
    assert second["created"] == 0
    assert second["updated"] == 1
    assert second["pruned"] == 0
