"""Track current and historical Neural Bridge experiments with MLflow."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ARTIFACT_ROOT = Path("/Volumes/onn. Drive/Neural Bridge Artifacts")
TRACKING_DB = ARTIFACT_ROOT / "indexes" / "mlflow" / "mlflow.db"
MLFLOW_ARTIFACT_ROOT = ARTIFACT_ROOT / "runs" / "mlflow"
IMPORT_SCHEMA = "neural_bridge_scientific_results_v1"


@dataclass
class MetricStats:
    count: int = 0
    total: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)


@dataclass
class ParameterValue:
    value: str | None = None
    varies: bool = False

    def add(self, value: object) -> None:
        text = str(value)
        if self.value is None and not self.varies:
            self.value = text
        elif text != self.value:
            self.varies = True
            self.value = None


_EVIDENCE_NAME_TOKENS = (
    "ablation",
    "audit",
    "bootstrap",
    "comparison",
    "control",
    "evaluation",
    "gate",
    "leaderboard",
    "metric",
    "promotion",
    "result",
    "score",
    "screen",
    "selection",
    "stability",
    "summary",
)
_IGNORED_EVIDENCE_PARTS = {
    "_checkpoint",
    "mlflow-artifacts",
    "models",
    "predictions",
    "raw_cache",
    "video_windows",
}
_METRIC_TOKENS = (
    "accuracy",
    "average_precision",
    "auprc",
    "auroc",
    "brier",
    "correlation",
    "delta",
    "duration",
    "elapsed",
    "error",
    "f1",
    "gain",
    "lift",
    "loss",
    "mae",
    "mse",
    "pearson",
    "pr_auc",
    "precision",
    "prevalence",
    "r2",
    "recall",
    "rmse",
    "roc_auc",
    "skill",
    "spearman",
    "throughput",
    "uplift",
)
_DECISION_TOKENS = (
    "audit_pass",
    "beats_",
    "credible",
    "gate",
    "no_harm",
    "passed",
    "positive_fold",
    "promotable",
    "proven",
    "stable",
    "verification_pass",
    "wins_",
)
_GENERIC_STAT_KEYS = {"count", "max", "mean", "median", "min", "n", "std", "wins"}
_PARAMETER_TOKENS = (
    "alpha",
    "architecture",
    "candidate",
    "checkpoint",
    "control",
    "epoch",
    "fold",
    "form",
    "head",
    "lane",
    "model",
    "pca",
    "representation",
    "seed",
    "source",
    "target",
    "threshold",
)


def tracking_uri(database: Path = TRACKING_DB) -> str:
    database = database.expanduser().absolute()
    return f"sqlite:///{database}"


def _normalise_key(value: str) -> str:
    key = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_.-").lower()
    return re.sub(r"_+", "_", key)[:240]


def _is_metric_key(key: str, parents: tuple[str, ...]) -> bool:
    context = ".".join((*parents[-2:], key)).lower()
    return any(token in context for token in (*_METRIC_TOKENS, *_DECISION_TOKENS)) or (
        key.lower() in _GENERIC_STAT_KEYS
        and any(token in context for token in ("vs_", "control", "real", "ar", "primary"))
    )


def _metric_name(key: str, parents: tuple[str, ...]) -> str:
    normalised = _normalise_key(key)
    if key.lower() not in _GENERIC_STAT_KEYS:
        return normalised
    parent = next((part for part in reversed(parents) if part not in {"metrics", "summary"}), "")
    return _normalise_key(f"{parent}.{key}")


def _is_parameter_key(key: str) -> bool:
    lowered = key.lower()
    return any(token in lowered for token in _PARAMETER_TOKENS)


def _collect_parameter(
    key: str,
    value: object,
    parameters: dict[str, ParameterValue],
) -> None:
    if (
        _is_parameter_key(key)
        and isinstance(value, (str, int, float))
        and not isinstance(value, bool)
    ):
        parameters[_normalise_key(key)].add(value)


def _collect_json_values(
    value: object,
    values: dict[str, MetricStats],
    *,
    parameters: dict[str, ParameterValue] | None = None,
    parents: tuple[str, ...] = (),
) -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            if isinstance(item, bool) and _is_metric_key(key, parents):
                values[_metric_name(key, parents)].add(float(item))
            elif isinstance(item, (int, float)) and not isinstance(item, bool):
                number = float(item)
                if math.isfinite(number) and _is_metric_key(key, parents):
                    values[_metric_name(key, parents)].add(number)
                elif parameters is not None:
                    _collect_parameter(key, item, parameters)
            elif isinstance(item, str) and parameters is not None:
                _collect_parameter(key, item, parameters)
            else:
                _collect_json_values(
                    item,
                    values,
                    parameters=parameters,
                    parents=(*parents, key),
                )
    elif isinstance(value, list):
        for item in value:
            _collect_json_values(item, values, parameters=parameters, parents=parents)


def _evidence_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink() or path.suffix.lower() not in {".csv", ".json"}:
            continue
        relative_parts = set(path.relative_to(root).parts[:-1])
        if relative_parts & _IGNORED_EVIDENCE_PARTS:
            continue
        evidence_context = f"{root.name}/{path.relative_to(root).as_posix()}".lower()
        if "inventory" in path.stem.lower() or not any(
            token in evidence_context for token in _EVIDENCE_NAME_TOKENS
        ):
            continue
        yield path


def _collect_csv_values(
    path: Path,
    values: dict[str, MetricStats],
    parameters: dict[str, ParameterValue] | None = None,
) -> None:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            for raw_key, raw_value in row.items():
                key = raw_key or ""
                if not raw_value:
                    continue
                if not _is_metric_key(key, (path.stem,)):
                    if parameters is not None:
                        _collect_parameter(key, raw_value, parameters)
                    continue
                try:
                    number = float(raw_value)
                except ValueError:
                    continue
                if math.isfinite(number):
                    values[_metric_name(key, (path.stem,))].add(number)


def scientific_metrics_from_object(value: object) -> dict[str, float]:
    """Return transparent min/mean/max summaries of genuine scientific measures."""
    values: dict[str, MetricStats] = defaultdict(MetricStats)
    _collect_json_values(value, values)
    return _summarise_metric_values(values)


def _summarise_metric_values(values: Mapping[str, MetricStats]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, observations in sorted(values.items()):
        if observations.count == 0:
            continue
        if observations.count == 1:
            metrics[key] = observations.total
            continue
        metrics[f"{key}.min"] = observations.minimum
        metrics[f"{key}.mean"] = observations.total / observations.count
        metrics[f"{key}.max"] = observations.maximum
        metrics[f"{key}.n"] = float(observations.count)
    return metrics


def scientific_run_data_from_file(path: Path) -> tuple[dict[str, float], dict[str, str]]:
    """Return scientific metrics and only parameters invariant across one result source."""
    values: dict[str, MetricStats] = defaultdict(MetricStats)
    parameters: dict[str, ParameterValue] = defaultdict(ParameterValue)
    if path.suffix.lower() == ".json":
        _collect_json_values(
            json.loads(path.read_text(encoding="utf-8")),
            values,
            parameters=parameters,
        )
    elif path.suffix.lower() == ".csv":
        _collect_csv_values(path, values, parameters)
    invariant_parameters = {
        key: value.value
        for key, value in sorted(parameters.items())
        if not value.varies and value.value is not None
    }
    return _summarise_metric_values(values), invariant_parameters


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _log_run_data(
    client: Any,
    run_id: str,
    metrics: Mapping[str, float],
    parameters: Mapping[str, str],
) -> None:
    from mlflow.entities import Metric, Param

    timestamp = int(time.time() * 1000)
    entities = [
        Metric(key=key, value=value, timestamp=timestamp, step=0)
        for key, value in sorted(metrics.items())
    ]
    for start in range(0, len(entities), 1000):
        client.log_batch(
            run_id,
            metrics=entities[start : start + 1000],
            params=[Param(key=key, value=value) for key, value in sorted(parameters.items())]
            if start == 0
            else [],
        )


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
    active_sources: dict[str, set[str]] = defaultdict(set)

    for programme, phase in phase_roots(artifact_root):
        programmes.add(programme)
        experiment_id = _experiment(client, programme, mlflow_artifact_root)
        existing_by_source = {
            run.data.tags.get("neural_bridge.source_path"): run
            for run in client.search_runs(
                [experiment_id],
                filter_string="tags.`neural_bridge.import_kind` = 'existing_result'",
                max_results=10_000,
            )
        }
        for source_path in _evidence_files(phase):
            source = str(source_path.absolute())
            existing = existing_by_source.get(source)
            try:
                digest = _file_sha256(source_path)
            except OSError:
                continue
            if (
                existing is not None
                and existing.data.tags.get("neural_bridge.source_sha256") == digest
                and existing.data.tags.get("neural_bridge.import_schema") == IMPORT_SCHEMA
            ):
                active_sources[experiment_id].add(source)
                updated += 1
                continue
            try:
                scientific_metrics, parameters = scientific_run_data_from_file(source_path)
            except (OSError, UnicodeError, csv.Error, json.JSONDecodeError):
                continue
            if not scientific_metrics:
                continue
            active_sources[experiment_id].add(source)
            if existing is not None:
                client.delete_run(existing.info.run_id)
                existing = None
            relative = source_path.relative_to(phase).as_posix()
            tags = {
                "mlflow.runName": f"{phase.name} / {relative}"[-240:],
                "neural_bridge.programme": programme,
                "neural_bridge.phase": phase.name,
                "neural_bridge.phase_root": str(phase.absolute()),
                "neural_bridge.source_path": source,
                "neural_bridge.source_sha256": digest,
                "neural_bridge.import_kind": "existing_result",
                "neural_bridge.import_schema": IMPORT_SCHEMA,
                "neural_bridge.payload_policy": "reference_in_place",
                "neural_bridge.science_metric_count": str(len(scientific_metrics)),
            }
            if existing is None:
                run = client.create_run(experiment_id, tags=tags)
                _log_run_data(client, run.info.run_id, scientific_metrics, parameters)
                created += 1
            else:
                run = existing
                for key, value in tags.items():
                    client.set_tag(run.info.run_id, key, value)
                updated += 1
            client.set_terminated(run.info.run_id)

    pruned = 0
    for experiment in client.search_experiments():
        if experiment.name == "Default":
            continue
        for run in client.search_runs([experiment.experiment_id], max_results=10_000):
            import_kind = run.data.tags.get("neural_bridge.import_kind")
            source = run.data.tags.get("neural_bridge.source_path")
            if import_kind == "existing_phase" or (
                import_kind == "existing_result"
                and source not in active_sources[experiment.experiment_id]
            ):
                client.delete_run(run.info.run_id)
                pruned += 1

    return {
        "created": created,
        "database": str(database),
        "pruned": pruned,
        "programmes": len(programmes),
        "results": created + updated,
        "tracking_uri": tracking_uri(database),
        "updated": updated,
    }


def status(database: Path = TRACKING_DB) -> dict[str, object]:
    mlflow = configure_tracking(database)
    client = mlflow.MlflowClient()
    experiments = [
        experiment
        for experiment in client.search_experiments()
        if experiment.name != "Default"
    ]
    runs = sum(
        len(client.search_runs([experiment.experiment_id], max_results=10_000))
        for experiment in experiments
    )
    return {
        "database": str(database),
        "experiments": len(experiments),
        "runs": runs,
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
    parser.add_argument("command", choices=("status", "sync-existing"))
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--database", type=Path, default=TRACKING_DB)
    parser.add_argument("--mlflow-artifact-root", type=Path, default=MLFLOW_ARTIFACT_ROOT)
    return parser


def main() -> int:
    args = _parser().parse_args()
    summary = (
        status(args.database)
        if args.command == "status"
        else sync_existing(args.artifact_root, args.database, args.mlflow_artifact_root)
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
