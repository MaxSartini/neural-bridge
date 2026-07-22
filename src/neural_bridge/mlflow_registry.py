"""Track current and historical Neural Bridge experiments with MLflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ARTIFACT_ROOT = Path("/Volumes/onn. Drive/Neural Bridge Artifacts")
TRACKING_DB = ARTIFACT_ROOT / "indexes" / "mlflow" / "mlflow.db"
MLFLOW_ARTIFACT_ROOT = ARTIFACT_ROOT / "runs" / "mlflow"


@dataclass(frozen=True)
class PhaseInventory:
    files: int
    bytes: int
    json_files: int
    manifests: int
    checkpoints: int
    digest: str


def tracking_uri(database: Path = TRACKING_DB) -> str:
    database = database.expanduser().absolute()
    return f"sqlite:///{database}"


def phase_inventory(root: Path) -> PhaseInventory:
    digest = hashlib.sha256()
    counts = {"files": 0, "bytes": 0, "json_files": 0, "manifests": 0, "checkpoints": 0}
    for directory, names, files in os.walk(root, followlinks=False):
        current = Path(directory)
        names[:] = sorted(name for name in names if not (current / name).is_symlink())
        for name in sorted(files):
            path = current / name
            if path.is_symlink() or name == ".DS_Store":
                continue
            stat = path.stat()
            relative = path.relative_to(root).as_posix()
            counts["files"] += 1
            counts["bytes"] += stat.st_size
            counts["json_files"] += path.suffix.lower() == ".json"
            counts["manifests"] += "manifest" in name.lower()
            counts["checkpoints"] += any(
                token in name.lower() for token in ("checkpoint", "ckpt", "best_model")
            )
            digest.update(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return PhaseInventory(**counts, digest=digest.hexdigest())


def phase_roots(artifact_root: Path = ARTIFACT_ROOT) -> list[tuple[str, Path]]:
    runs = artifact_root / "runs"
    phases: list[tuple[str, Path]] = []
    for programme in sorted(path for path in runs.iterdir() if path.is_dir()):
        if programme.name == "mlflow":
            continue
        phases.extend(
            (programme.name, phase)
            for phase in sorted(path for path in programme.iterdir() if path.is_dir())
        )
    return phases


def configure_tracking(database: Path = TRACKING_DB) -> Any:
    import mlflow

    database.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri(database))
    return mlflow


def _experiment(client: Any, programme: str, artifact_root: Path) -> str:
    name = f"Neural Bridge / {programme}"
    existing = client.get_experiment_by_name(name)
    if existing is not None:
        return existing.experiment_id
    location = (artifact_root / programme).absolute()
    location.mkdir(parents=True, exist_ok=True)
    return client.create_experiment(name, artifact_location=location.as_uri())


def sync_existing(
    artifact_root: Path = ARTIFACT_ROOT,
    database: Path = TRACKING_DB,
    mlflow_artifact_root: Path = MLFLOW_ARTIFACT_ROOT,
) -> dict[str, object]:
    mlflow = configure_tracking(database)
    client = mlflow.MlflowClient()
    created = 0
    updated = 0
    programmes: set[str] = set()

    for programme, phase in phase_roots(artifact_root):
        programmes.add(programme)
        experiment_id = _experiment(client, programme, mlflow_artifact_root)
        source = str(phase.absolute())
        existing = {
            run.data.tags.get("neural_bridge.source_path"): run
            for run in client.search_runs(
                [experiment_id],
                filter_string="tags.`neural_bridge.import_kind` = 'existing_phase'",
                max_results=1000,
            )
        }.get(source)
        inventory = phase_inventory(phase)
        tags = {
            "mlflow.runName": phase.name,
            "neural_bridge.programme": programme,
            "neural_bridge.phase": phase.name,
            "neural_bridge.source_path": source,
            "neural_bridge.import_kind": "existing_phase",
            "neural_bridge.payload_policy": "reference_in_place",
            "neural_bridge.inventory_sha256": inventory.digest,
        }
        if existing is None:
            run = client.create_run(experiment_id, tags=tags)
            created += 1
        else:
            run = existing
            for key, value in tags.items():
                client.set_tag(run.info.run_id, key, value)
            updated += 1
        timestamp = int(time.time() * 1000)
        for key, value in {
            "inventory.files": inventory.files,
            "inventory.bytes": inventory.bytes,
            "inventory.json_files": inventory.json_files,
            "inventory.manifests": inventory.manifests,
            "inventory.checkpoints": inventory.checkpoints,
        }.items():
            client.log_metric(run.info.run_id, key, value, timestamp=timestamp)
        client.set_terminated(run.info.run_id)

    return {
        "created": created,
        "database": str(database),
        "phases": created + updated,
        "programmes": len(programmes),
        "tracking_uri": tracking_uri(database),
        "updated": updated,
    }


def start_run(
    programme: str,
    phase: str,
    *,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
    database: Path = TRACKING_DB,
    mlflow_artifact_root: Path = MLFLOW_ARTIFACT_ROOT,
) -> AbstractContextManager[Any]:
    """Start a native run for new work; callers log only genuine parameters and metrics."""
    mlflow = configure_tracking(database)
    experiment_id = _experiment(mlflow.MlflowClient(), programme, mlflow_artifact_root)
    run_tags = {"neural_bridge.programme": programme, "neural_bridge.phase": phase}
    run_tags.update(tags or {})
    return mlflow.start_run(
        experiment_id=experiment_id,
        run_name=run_name or phase,
        tags=run_tags,
    )


def log_completed_output(
    output: Path,
    *,
    parameters: dict[str, object],
    metrics: dict[str, float | int],
    tags: dict[str, str] | None = None,
) -> None:
    """Attach a completed external result to the active run without copying its payload."""
    import mlflow

    output = output.expanduser().resolve(strict=True)
    mlflow.log_params(parameters)
    mlflow.log_metrics(metrics)
    mlflow.set_tags({"neural_bridge.output_path": str(output), **(tags or {})})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("sync-existing",))
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--database", type=Path, default=TRACKING_DB)
    parser.add_argument("--mlflow-artifact-root", type=Path, default=MLFLOW_ARTIFACT_ROOT)
    return parser


def main() -> int:
    args = _parser().parse_args()
    summary = sync_existing(args.artifact_root, args.database, args.mlflow_artifact_root)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
