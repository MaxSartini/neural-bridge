"""Explicit MLflow tracking with Neural Bridge provenance and device contracts."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ._optional import require_upstream
from .acceleration import require_accelerator


@dataclass(frozen=True)
class RunProvenance:
    git_commit: str
    dataset_manifest_sha256: str
    split_manifest_sha256: str
    feature_manifest_sha256: str
    target: str
    architecture: str
    validation_protocol: str
    seed: int
    accelerator_backend: str
    frozen_ar_sha256: str | None = None
    claim_status: str = "exploratory"
    extra: Mapping[str, str | int | float | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        required = {
            "git_commit": self.git_commit,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "split_manifest_sha256": self.split_manifest_sha256,
            "feature_manifest_sha256": self.feature_manifest_sha256,
            "target": self.target,
            "architecture": self.architecture,
            "validation_protocol": self.validation_protocol,
            "accelerator_backend": self.accelerator_backend,
        }
        missing = sorted(key for key, value in required.items() if not str(value).strip())
        if missing:
            raise ValueError(f"Missing required run provenance: {', '.join(missing)}")
        if self.accelerator_backend not in {"mlx", "mps"}:
            raise ValueError("accelerator_backend must be 'mlx' or 'mps'")
        if self.claim_status != "exploratory":
            raise ValueError(
                "Research-tooling runs must start as exploratory; canonical promotion "
                "uses the separate evidence workflow."
            )


class MLflowRun:
    """Context manager over the official MLflow fluent tracking API."""

    def __init__(
        self,
        *,
        tracking_uri: str,
        experiment_name: str,
        run_name: str,
        provenance: RunProvenance,
        tags: Mapping[str, str] | None = None,
        artifact_location: str | None = None,
    ) -> None:
        if not tracking_uri.strip() or not experiment_name.strip() or not run_name.strip():
            raise ValueError("tracking_uri, experiment_name, and run_name are required")
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.provenance = provenance
        self.tags = dict(tags or {})
        self.artifact_location = artifact_location
        self.run_id: str | None = None
        self._mlflow: Any = None
        self._active_run: Any = None
        self._previous_tracking_uri: str | None = None

    def __enter__(self) -> "MLflowRun":
        accelerator = require_accelerator(self.provenance.accelerator_backend)
        mlflow = require_upstream("mlflow")
        self._mlflow = mlflow
        self._previous_tracking_uri = mlflow.get_tracking_uri()
        mlflow.set_tracking_uri(self.tracking_uri)
        if self.artifact_location:
            client = mlflow.tracking.MlflowClient(tracking_uri=self.tracking_uri)
            if client.get_experiment_by_name(self.experiment_name) is None:
                client.create_experiment(
                    self.experiment_name,
                    artifact_location=self.artifact_location,
                )
        mlflow.set_experiment(self.experiment_name)
        effective_tags = {
            "neural_bridge.claim_status": "exploratory",
            "neural_bridge.canonical_evidence": "false",
            "neural_bridge.accelerator_verified": "true",
            "neural_bridge.accelerator_detail": accelerator.detail,
            **self.tags,
        }
        self._active_run = mlflow.start_run(run_name=self.run_name, tags=effective_tags)
        self.run_id = self._active_run.info.run_id
        params = asdict(self.provenance)
        extra = params.pop("extra")
        params.update({f"extra.{key}": value for key, value in extra.items()})
        mlflow.log_params({key: value for key, value in params.items() if value is not None})
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._mlflow is None:
            return
        self._mlflow.end_run(status="FAILED" if exc_type is not None else "FINISHED")
        if self._previous_tracking_uri is not None:
            self._mlflow.set_tracking_uri(self._previous_tracking_uri)
        self._active_run = None

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        self._require_active()
        clean: dict[str, float] = {}
        for name, value in metrics.items():
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"Metric {name!r} is not finite")
            clean[str(name)] = numeric
        self._mlflow.log_metrics(clean, step=step)

    def log_artifact(self, path: str | Path, *, artifact_path: str | None = None) -> None:
        self._require_active()
        artifact = Path(path)
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        self._mlflow.log_artifact(str(artifact), artifact_path=artifact_path)

    def _require_active(self) -> None:
        if self._active_run is None:
            raise RuntimeError("MLflowRun must be entered before logging")
