from __future__ import annotations

from pathlib import Path

from neural_bridge.mlflow_registry import (
    log_completed_output,
    phase_inventory,
    phase_roots,
    start_run,
    tracking_uri,
)


def test_phase_inventory_ignores_aliases_and_is_stable(tmp_path: Path) -> None:
    phase = tmp_path / "runs" / "again" / "phase-01"
    phase.mkdir(parents=True)
    (phase / "result.json").write_text("{}")
    (phase / "model.ckpt").write_bytes(b"model")
    (phase / "alias").symlink_to(phase, target_is_directory=True)

    first = phase_inventory(phase)
    second = phase_inventory(phase)

    assert first == second
    assert first.files == 2
    assert first.json_files == 1
    assert first.checkpoints == 1


def test_phase_roots_excludes_mlflow_storage(tmp_path: Path) -> None:
    (tmp_path / "runs" / "again" / "phase-01").mkdir(parents=True)
    (tmp_path / "runs" / "mlflow" / "again").mkdir(parents=True)

    assert phase_roots(tmp_path) == [("again", tmp_path / "runs" / "again" / "phase-01")]


def test_tracking_uri_uses_absolute_sqlite_path(tmp_path: Path) -> None:
    assert tracking_uri(tmp_path / "mlflow.db") == f"sqlite:///{tmp_path}/mlflow.db"


def test_native_run_references_external_output_without_copying(tmp_path: Path) -> None:
    database = tmp_path / "mlflow.db"
    artifact_root = tmp_path / "mlflow-artifacts"
    output = tmp_path / "result.json"
    output.write_text("{}")

    with start_run(
        "veatic-2.1",
        "event-target-screen",
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
