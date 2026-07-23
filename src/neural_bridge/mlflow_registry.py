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
from functools import lru_cache
from pathlib import Path
from typing import Any

ARTIFACT_ROOT = Path("/Volumes/onn. Drive/Neural Bridge Artifacts")
TRACKING_DB = ARTIFACT_ROOT / "indexes" / "mlflow" / "mlflow.db"
MLFLOW_ARTIFACT_ROOT = ARTIFACT_ROOT / "runs" / "mlflow"
IMPORT_SCHEMA = "neural_bridge_scientific_results_v4"


@dataclass
class MetricStats:
    count: int = 0
    total: float = 0.0

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value


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
_METRIC_BASES = (
    ("ranking.top_5pct_lift", ("top_5pct_lift", "top5_lift")),
    ("event.average_precision_skill", ("average_precision_skill", "skill_delta")),
    ("event.pr_auc", ("pr_auc", "auprc", "average_precision")),
    ("event.roc_auc", ("roc_auc", "auroc")),
    ("event.prevalence", ("prevalence",)),
    ("continuous.spearman", ("spearman",)),
    ("continuous.pearson", ("pearson", "correlation")),
    ("continuous.mae", ("mae",)),
    ("continuous.rmse", ("rmse",)),
)


def tracking_uri(database: Path = TRACKING_DB) -> str:
    database = database.expanduser().absolute()
    return f"sqlite:///{database}"


@lru_cache(maxsize=4096)
def _normalise_key(value: str) -> str:
    key = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_.-").lower()
    return re.sub(r"_+", "_", key)[:240]


@lru_cache(maxsize=8192)
def _canonical_decision_metric(key: str, parents: tuple[str, ...]) -> str | None:
    context = _normalise_key(".".join((*parents, key)))
    if "audit_pass" in context or "verification_pass" in context:
        return "quality.audit_pass"
    if "no_harm" in context:
        return "quality.no_harm_pass"
    if "leakage" in context and ("pass" in context or "check" in context):
        return "quality.leakage_pass"
    if "promotable" in context:
        return "selection.promotable"
    if "credible" in context:
        return "selection.credible"
    if key == "passed" and not parents:
        return "selection.passed"
    if "checks" in parents and isinstance(key, str):
        return "quality.check_pass_rate"
    return None


@lru_cache(maxsize=8192)
def _canonical_metric_name(key: str, parents: tuple[str, ...]) -> str | None:
    decision = _canonical_decision_metric(key, parents)
    if decision is not None:
        return decision
    raw = _normalise_key(".".join((*parents[-2:], key)))
    if "train_loss" in raw:
        return "training.train_loss"
    if "validation_loss" in raw or "val_loss" in raw:
        return "training.validation_loss"
    if "duration_seconds" in raw or "elapsed_seconds" in raw:
        return "runtime.duration_seconds"
    base = next(
        (name for name, aliases in _METRIC_BASES if any(alias in raw for alias in aliases)),
        None,
    )
    if base is None:
        return None
    stat = next(
        (name for name in ("std", "min", "max") if re.search(rf"(^|[._]){name}[._]", raw)),
        None,
    )
    if stat is not None:
        return f"stability.{base}.{stat}"
    if any(token in raw for token in ("minus_raw", "vs_raw", "delta_vs_raw")):
        lane = "delta_vs_raw"
    elif any(token in raw for token in ("minus_best_control", "vs_best_control", "vs_control")):
        lane = "delta_vs_control"
    elif any(token in raw for token in ("minus_ar", "vs_ar", "delta_vs_frozen_ar")):
        lane = "delta_vs_ar"
    elif raw.startswith("ar_") or ".ar_" in raw or "frozen_ar" in raw:
        lane = "baseline_ar"
    elif raw.startswith("raw_") or ".raw_" in raw:
        lane = "baseline_raw"
    elif "control" in raw:
        lane = "baseline_control"
    elif "validation" in raw:
        lane = "validation"
    else:
        lane = "model"
    return f"{lane}.{base}"


@lru_cache(maxsize=1024)
def _canonical_parameter_key(key: str) -> str | None:
    raw = _normalise_key(key)
    exact = {
        "architecture": "architecture",
        "best_architecture": "architecture",
        "best_epoch": "best_epoch",
        "candidate": "candidate",
        "checkpoint_min_epoch": "checkpoint_min_epoch",
        "control_type": "control",
        "epochs_run": "epochs_run",
        "fold": "fold",
        "form": "form",
        "head": "head",
        "inner_fold": "inner_fold",
        "max_epochs": "max_epochs",
        "min_epochs": "min_epochs",
        "model": "model",
        "model_head": "head",
        "outer_fold": "outer_fold",
        "pca_variance_target": "pca_variance_target",
        "pca_width": "pca_width",
        "representation": "representation",
        "ridge_alpha": "regularization_alpha",
        "seed": "seed",
        "source": "representation",
        "target": "target",
        "target_name": "target",
        "threshold": "threshold",
    }
    return exact.get(raw)


def _collect_parameter(
    key: str,
    value: object,
    parameters: dict[str, ParameterValue],
) -> None:
    canonical = _canonical_parameter_key(key)
    if (
        canonical is not None
        and isinstance(value, (str, int, float))
        and not isinstance(value, bool)
    ):
        parameters[canonical].add(value)


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
            metric_name = _canonical_metric_name(key, parents)
            if isinstance(item, bool) and metric_name is not None:
                values[metric_name].add(float(item))
            elif isinstance(item, (int, float)) and not isinstance(item, bool):
                number = float(item)
                if math.isfinite(number) and metric_name is not None:
                    values[metric_name].add(number)
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


def _bundle_root(phase: Path, source: Path) -> Path:
    relative = source.relative_to(phase)
    directories = relative.parts[:-1]
    markers = {"diagnostics", "metrics", "promotion", "score_parts", "training_curves"}
    for index, part in enumerate(directories):
        if part in markers:
            return phase.joinpath(*directories[:index])
    return source if source.parent == phase else source.parent


def _evidence_bundles(phase: Path) -> dict[Path, tuple[Path, ...]]:
    grouped: dict[Path, list[Path]] = defaultdict(list)
    for source in _evidence_files(phase):
        grouped[_bundle_root(phase, source)].append(source)
    return {root: tuple(paths) for root, paths in sorted(grouped.items())}


def _collect_csv_values(
    path: Path,
    values: dict[str, MetricStats],
    parameters: dict[str, ParameterValue] | None = None,
) -> None:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        metric_fields = {
            field: metric
            for field in reader.fieldnames or ()
            if (metric := _canonical_metric_name(field, (path.stem,))) is not None
        }
        parameter_fields = {
            field: parameter
            for field in reader.fieldnames or ()
            if (parameter := _canonical_parameter_key(field)) is not None
        }
        for row in reader:
            for key, metric_name in metric_fields.items():
                raw_value = row.get(key)
                if not raw_value:
                    continue
                try:
                    number = float(raw_value)
                except ValueError:
                    continue
                if math.isfinite(number):
                    values[metric_name].add(number)
            if parameters is not None:
                for key in parameter_fields:
                    raw_value = row.get(key)
                    if raw_value:
                        _collect_parameter(key, raw_value, parameters)


def scientific_metrics_from_object(value: object) -> dict[str, float]:
    """Return standardized summaries of genuine scientific measures."""
    values: dict[str, MetricStats] = defaultdict(MetricStats)
    _collect_json_values(value, values)
    return _summarise_metric_values(values)


def _summarise_metric_values(values: Mapping[str, MetricStats]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, observations in sorted(values.items()):
        if observations.count == 0:
            continue
        metrics[key] = observations.total / observations.count
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


def scientific_run_data_from_files(
    paths: Iterable[Path],
) -> tuple[dict[str, float], dict[str, str]]:
    """Combine the result tables belonging to one executed experiment."""
    values: dict[str, MetricStats] = defaultdict(MetricStats)
    parameters: dict[str, ParameterValue] = defaultdict(ParameterValue)
    for path in paths:
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


def _bundle_sha256(root: Path, sources: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for source in sources:
        digest.update(str(source.relative_to(root) if root.is_dir() else source.name).encode())
        digest.update(_file_sha256(source).encode())
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


def _evidence_level(path: Path) -> str:
    context = path.as_posix().lower()
    if any(token in context for token in ("confirmation", "final-benchmark", "locked")):
        return "confirmation"
    if any(token in context for token in ("screen", "discovery", "ablation")):
        return "discovery"
    return "development"


def _task(metrics: Mapping[str, float], parameters: Mapping[str, str]) -> str:
    target = parameters.get("target", "").lower()
    if "valence" in target:
        return "valence"
    if any(".event." in key for key in metrics):
        return "spike"
    if any(".continuous." in key for key in metrics):
        return "continuous"
    return "other"


def _comparison_metrics(metrics: Mapping[str, float]) -> tuple[str | None, dict[str, float]]:
    candidates = []
    for priority, measure in enumerate(("continuous.spearman", "event.pr_auc")):
        score = metrics.get(f"model.{measure}", metrics.get(f"validation.{measure}"))
        if score is None:
            continue
        matched_baselines = sum(
            f"baseline_{lane}.{measure}" in metrics for lane in ("raw", "ar", "control")
        )
        candidates.append((matched_baselines, priority, measure))
    primary = max(candidates)[2] if candidates else None
    if primary is None:
        return None, {}
    score = metrics.get(f"model.{primary}", metrics.get(f"validation.{primary}"))
    raw = metrics.get(f"baseline_raw.{primary}")
    ar = metrics.get(f"baseline_ar.{primary}")
    control = metrics.get(f"baseline_control.{primary}")
    comparison = {
        key: value
        for key, value in {
            "comparison.model": score,
            "comparison.raw": raw,
            "comparison.ar": ar,
            "comparison.best_control": control,
            "comparison.delta_vs_raw": None if score is None or raw is None else score - raw,
            "comparison.delta_vs_ar": None if score is None or ar is None else score - ar,
            "comparison.delta_vs_best_control": (
                None if score is None or control is None else score - control
            ),
        }.items()
        if value is not None
    }
    return primary, comparison


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
        for bundle_root, source_paths in _evidence_bundles(phase).items():
            source = str(bundle_root.absolute())
            existing = existing_by_source.get(source)
            try:
                digest = _bundle_sha256(bundle_root, source_paths)
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
                scientific_metrics, parameters = scientific_run_data_from_files(source_paths)
            except (OSError, UnicodeError, csv.Error, json.JSONDecodeError):
                continue
            if not scientific_metrics:
                continue
            primary_measure, comparison = _comparison_metrics(scientific_metrics)
            scientific_metrics.update(comparison)
            active_sources[experiment_id].add(source)
            if existing is not None:
                client.delete_run(existing.info.run_id)
                existing = None
            relative = bundle_root.relative_to(phase).as_posix()
            tags = {
                "mlflow.runName": f"{phase.name} / {relative}"[-240:],
                "neural_bridge.programme": programme,
                "neural_bridge.phase": phase.name,
                "neural_bridge.phase_root": str(phase.absolute()),
                "neural_bridge.source_path": source,
                "neural_bridge.source_file_count": str(len(source_paths)),
                "neural_bridge.source_sha256": digest,
                "neural_bridge.import_kind": "existing_result",
                "neural_bridge.import_schema": IMPORT_SCHEMA,
                "neural_bridge.evidence_level": _evidence_level(bundle_root),
                "neural_bridge.task": _task(scientific_metrics, parameters),
                "neural_bridge.primary_measure": primary_measure or "none",
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
        experiment for experiment in client.search_experiments() if experiment.name != "Default"
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
