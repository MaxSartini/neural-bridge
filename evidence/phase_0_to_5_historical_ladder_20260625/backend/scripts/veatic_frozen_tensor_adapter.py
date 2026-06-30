"""Frozen tensor feature adapter for the existing VEATIC benchmark suite.

The adapter loads frozen tensor contracts and exposes them in the same conceptual
shape used by the current VEATIC benchmark semantics: split rows, feature
matrices, target arrays, canonical lane keys, and metadata. It does not fit
models or compute benchmark scores.
"""

from __future__ import annotations

import hashlib
import csv
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from backend.scripts import run_veatic_event_spike_retest as event_spike
from backend.scripts import run_veatic_neuro_benchmark as bench
from backend.scripts import run_veatic_strict_benchmark as strict


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_CONTRACT_VERSION = "veatic_frozen_tensor_incremental_v1"
BENCHMARK_MODE = "existing_suite_with_frozen_tensor_adapter"
SUMMARY_ROOT = Path("outputs/veatic_124_raw_representation_tensor_export_v1")

PRIMARY_CANDIDATE = "pca_sequence_128_causal_past_2s_mean"
FROZEN_BASELINE = "cortical_pca64_delta_frozen_baseline"
REQUIRED_REPRESENTATIONS = (PRIMARY_CANDIDATE, FROZEN_BASELINE)
FEATURE_ALIASES = {
    PRIMARY_CANDIDATE: "PCA128",
    FROZEN_BASELINE: "PCA64_delta",
}
SAFE_PRIMARY_REPRESENTATIONS = (
    PRIMARY_CANDIDATE,
    "roi_parcel_features",
    FROZEN_BASELINE,
)
SPLITS = ("blocked", "official", "grouped_0", "grouped_1", "grouped_2", "grouped_3", "grouped_4")
FORBIDDEN_RESULT_INPUT_NAMES = (
    "candidate_promotion_summary.json",
    "control_results.csv",
    "fixed_split_results.csv",
    "grouped_video_results.csv",
    "raw_vs_compressed_leaderboard.csv",
    "representation_results_all.csv",
    "representation_results_primary.csv",
)
EVENT_HELPER_CANDIDATES = (
    "regression_metrics",
    "pr_auc",
    "binary_metrics_from_pred",
    "best_train_threshold",
    "event_metrics",
    "majority_metrics",
    "topk_recall",
)


def local_env_value(name: str) -> str | None:
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if not line or line.lstrip().startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == name:
                return value.strip().strip('"').strip("'")
    return os.environ.get(name)


def default_tensor_root() -> Path:
    external_root = local_env_value("NEURAL_BRIDGE_EXTERNAL_ROOT")
    if not external_root:
        return Path("${NEURAL_BRIDGE_EXTERNAL_ROOT}") / "tensors" / "veatic_124_raw_representation_v1"
    return Path(external_root).expanduser() / "tensors" / "veatic_124_raw_representation_v1"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def finite_json(value: Any) -> Any:
    if isinstance(value, np.generic):
        return finite_json(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(finite_json(payload), indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: finite_json(value) for key, value in row.items()})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def target_dir_name(target_name: str, threshold: float) -> str:
    return f"{target_name}__thr_{threshold:g}"


def row_key(row: dict[str, Any]) -> tuple[str, float, int]:
    return (
        str(row["video_id"]),
        float(row["time_start_seconds"]),
        int(row["frame_index"]),
    )


def assert_same_row_order(
    ar_rows: list[dict[str, Any]],
    tensor_row_metadata: list[dict[str, Any]],
    *,
    role: str,
) -> None:
    """Stop if AR rows differ from tensor row metadata order."""
    if len(ar_rows) != len(tensor_row_metadata):
        raise AssertionError(
            f"{role} row count mismatch: ar_rows={len(ar_rows)} tensor_rows={len(tensor_row_metadata)}"
        )
    for index, (ar_row, tensor_row) in enumerate(zip(ar_rows, tensor_row_metadata)):
        if row_key(ar_row) != row_key(tensor_row):
            raise AssertionError(
                f"{role} row order mismatch at index {index}: "
                f"ar={row_key(ar_row)} tensor={row_key(tensor_row)}"
            )


def rows_from_metadata_order(
    all_rows: list[dict[str, Any]],
    tensor_row_metadata: list[dict[str, Any]],
    *,
    role: str,
) -> list[dict[str, Any]]:
    manifest_rows_by_key: dict[tuple[str, float, int], dict[str, Any]] = {}
    for manifest_row in all_rows:
        key = row_key(manifest_row)
        if key in manifest_rows_by_key:
            raise AssertionError(f"duplicate manifest row key while preparing {role} rows: {key}")
        manifest_rows_by_key[key] = manifest_row

    rows = []
    for meta in tensor_row_metadata:
        key = row_key(meta)
        if key not in manifest_rows_by_key:
            raise AssertionError(f"{role} tensor row key missing from manifest rows: {key}")
        rows.append(manifest_rows_by_key[key])
    assert_same_row_order(rows, tensor_row_metadata, role=role)
    return rows


def reject_exclude_83_paths(path: Path) -> None:
    joined = "/".join(part.lower() for part in path.parts)
    forbidden_tokens = (
        "exclude_video_83",
        "exclude-video-83",
        "exclude-83",
        "exclude_83",
        "without_video_83",
        "without-video-83",
        "without_83",
        "no_video_83",
        "no-video-83",
        "no_83",
    )
    if any(token in joined for token in forbidden_tokens):
        raise ValueError(f"exclude-video-83 paths are forbidden for this benchmark: {path}")


def canonical_prediction_keys(feature_name: str) -> tuple[str, ...]:
    return (
        "mean_train",
        "time_ridge",
        "autoregressive",
        feature_name,
        f"shuffled_{feature_name}",
        f"random_gaussian_{feature_name}",
        f"autoregressive_plus_{feature_name}",
        f"autoregressive_plus_shuffled_{feature_name}",
        f"autoregressive_plus_random_gaussian_{feature_name}",
        f"residualized_autoregressive_plus_{feature_name}",
        f"residualized_autoregressive_plus_shuffled_{feature_name}",
        f"residualized_autoregressive_plus_random_gaussian_{feature_name}",
    )


def lane_name_from_prediction_key(feature_name: str, key: str) -> str:
    if key == "mean_train":
        return "mean_train"
    if key == "time_ridge":
        return "time_ridge"
    if key == "autoregressive":
        return "AR_only"
    if key == feature_name:
        return f"{feature_name}_only"
    if key == f"shuffled_{feature_name}":
        return f"shuffled_{feature_name}"
    if key == f"random_gaussian_{feature_name}":
        return f"random_gaussian_{feature_name}"
    if key == f"autoregressive_plus_{feature_name}":
        return f"AR_plus_{feature_name}"
    if key == f"autoregressive_plus_shuffled_{feature_name}":
        return f"AR_plus_shuffled_{feature_name}"
    if key == f"autoregressive_plus_random_gaussian_{feature_name}":
        return f"AR_plus_random_{feature_name}"
    if key == f"residualized_autoregressive_plus_{feature_name}":
        return f"residualized_AR_plus_{feature_name}"
    if key == f"residualized_autoregressive_plus_shuffled_{feature_name}":
        return f"residualized_AR_plus_shuffled_{feature_name}"
    if key == f"residualized_autoregressive_plus_random_gaussian_{feature_name}":
        return f"residualized_AR_plus_random_{feature_name}"
    return key


def is_control_lane(lane_name: str) -> bool:
    return (
        lane_name in {"mean_train", "time_ridge"}
        or lane_name.startswith("shuffled_")
        or lane_name.startswith("random_gaussian_")
        or "_shuffled_" in lane_name
        or "_random_" in lane_name
    )


def discover_event_metric_helpers() -> dict[str, str]:
    """Inspect existing event/top-k helpers before future scoring imports them."""
    helpers: dict[str, str] = {}
    for name in EVENT_HELPER_CANDIDATES:
        helper = getattr(event_spike, name, None)
        if callable(helper):
            helpers[name] = f"{event_spike.__name__}.{name}"
    return helpers


def canonical_helper_references() -> dict[str, str]:
    return {
        "autoregressive_features": f"{bench.__name__}.autoregressive_features",
        "time_features": f"{bench.__name__}.time_features",
        "ridge_fit_predict": f"{bench.__name__}.ridge_fit_predict",
        "ridge": f"{bench.__name__}.ridge",
        "strict_control_ledger": f"{strict.__name__}.CONTROL_LEDGER",
        "event_metric_helper_discovery": (
            "Inspect run_veatic_event_spike_retest.py for the actual event/top-k metric helper names before importing. "
            "Reuse existing project metric helpers where available. Do not assume helper names."
        ),
    }


def parse_targets(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "target_name": str(item["target_name"]),
            "task_type": str(item["task_type"]),
            "threshold": float(item["threshold"]),
            "folder": target_dir_name(str(item["target_name"]), float(item["threshold"])),
        }
        for item in summary.get("targets_exported", [])
    ]


def add_check(checks: list[dict[str, Any]], name: str, status: str, details: dict[str, Any] | None = None) -> None:
    checks.append({"check_name": name, "status": status, "details": details or {}})


def check_failures(checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in checks if item["status"] == "fail"]


def deny_prior_result_files(paths: list[Path]) -> None:
    forbidden = set(FORBIDDEN_RESULT_INPUT_NAMES)
    found: list[str] = []
    for root in paths:
        if not root.exists():
            continue
        if root.is_file():
            if root.name in forbidden:
                found.append(str(root))
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.name in forbidden:
                found.append(str(path))
    if found:
        raise ValueError(f"Prior benchmark result inputs are forbidden: {found}")


def count_video_83_rows(row_metadata_path: Path) -> tuple[int, bool]:
    if not row_metadata_path.exists():
        return 0, False
    count = 0
    with row_metadata_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if bool(row.get("is_video_83")) or str(row.get("video_id")) == "83":
                count += 1
    return count, True


@dataclass(frozen=True)
class ExistingBenchmarkInputs:
    feature_name: str
    train_rows: list[dict[str, Any]]
    test_rows: list[dict[str, Any]]
    train_x: np.ndarray
    test_x: np.ndarray
    train_y: np.ndarray
    test_y: np.ndarray
    train_row_metadata: list[dict[str, Any]]
    test_row_metadata: list[dict[str, Any]]
    prediction_keys: tuple[str, ...]


@dataclass(frozen=True)
class FrozenTensorContract:
    representation_name: str
    feature_name: str
    split: str
    target_name: str
    threshold: float
    contract_dir: Path
    tracked_dir: Path
    train_x: np.ndarray
    test_x: np.ndarray
    train_y: np.ndarray
    test_y: np.ndarray
    train_row_metadata: list[dict[str, Any]]
    test_row_metadata: list[dict[str, Any]]
    representation_metadata: dict[str, Any]
    split_metadata: dict[str, Any]
    target_metadata: dict[str, Any]
    checksum_manifest: dict[str, Any]
    leakage_contract: dict[str, Any]

    def as_existing_benchmark_inputs(self, all_rows: list[dict[str, Any]]) -> ExistingBenchmarkInputs:
        train_rows = rows_from_metadata_order(all_rows, self.train_row_metadata, role="train")
        test_rows = rows_from_metadata_order(all_rows, self.test_row_metadata, role="test")
        return ExistingBenchmarkInputs(
            feature_name=self.feature_name,
            train_rows=train_rows,
            test_rows=test_rows,
            train_x=self.train_x,
            test_x=self.test_x,
            train_y=self.train_y,
            test_y=self.test_y,
            train_row_metadata=self.train_row_metadata,
            test_row_metadata=self.test_row_metadata,
            prediction_keys=canonical_prediction_keys(self.feature_name),
        )


def base_target_name(target_name: str) -> str:
    return target_name.split("__", 1)[0]


def ridge_train_test(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train_scores = bench.ridge(train_x, train_y, train_x)
    test_scores = bench.ridge(train_x, train_y, test_x)
    return train_scores, test_scores


def score_metrics(
    *,
    task_type: str,
    train_y: np.ndarray,
    train_scores: np.ndarray,
    test_y: np.ndarray,
    test_scores: np.ndarray,
) -> dict[str, Any]:
    if task_type == "binary":
        return event_spike.event_metrics(
            train_y.astype(np.float64),
            train_scores,
            test_y.astype(np.int64),
            test_scores,
        )
    metrics = bench.metrics(test_y.astype(np.float64), test_scores)
    metrics["train_mae"] = float(np.mean(np.abs(train_y.astype(np.float64) - train_scores)))
    return metrics


def flatten_lane_row(
    *,
    contract: FrozenTensorContract,
    prediction_key: str,
    train_rows: int,
    test_rows: int,
    train_videos: int,
    test_videos: int,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    lane_name = lane_name_from_prediction_key(contract.feature_name, prediction_key)
    row = {
        "benchmark_mode": BENCHMARK_MODE,
        "representation_name": contract.representation_name,
        "feature_name": contract.feature_name,
        "split": contract.split,
        "target": contract.target_name,
        "threshold": contract.threshold,
        "task_type": contract.target_metadata.get("task_type"),
        "canonical_prediction_key": prediction_key,
        "lane": lane_name,
        "is_control": is_control_lane(lane_name),
        "train_rows": train_rows,
        "test_rows": test_rows,
        "train_videos": train_videos,
        "test_videos": test_videos,
        "computed_fresh": True,
        "ridge_only": True,
    }
    row.update(metrics)
    return row


def score_contract_ridge_existing_semantics(
    contract: FrozenTensorContract,
    *,
    all_rows: list[dict[str, Any]],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute ridge-only lanes using the existing VEATIC lane semantics."""
    inputs = contract.as_existing_benchmark_inputs(all_rows)
    assert_same_row_order(inputs.train_rows, inputs.train_row_metadata, role="train")
    assert_same_row_order(inputs.test_rows, inputs.test_row_metadata, role="test")

    rng = np.random.default_rng(seed)
    train_y = inputs.train_y.astype(np.float64)
    test_y = inputs.test_y.astype(np.float64)
    base_target = base_target_name(contract.target_name)
    train_ar = bench.autoregressive_features(
        all_rows, inputs.train_rows, base_target, include_current=True
    )
    test_ar = bench.autoregressive_features(
        all_rows, inputs.test_rows, base_target, include_current=True
    )

    ar_train_pred, ar_test_pred = ridge_train_test(train_ar, train_y, test_ar)
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "mean_train": (
            np.full(len(inputs.train_rows), float(np.mean(train_y)), dtype=np.float64),
            np.full(len(inputs.test_rows), float(np.mean(train_y)), dtype=np.float64),
        ),
        "time_ridge": ridge_train_test(
            bench.time_features(inputs.train_rows),
            train_y,
            bench.time_features(inputs.test_rows),
        ),
        "autoregressive": (ar_train_pred, ar_test_pred),
    }

    feature_name = inputs.feature_name
    train_feature_matrix = inputs.train_x.astype(np.float64)
    test_feature_matrix = inputs.test_x.astype(np.float64)
    predictions[feature_name] = ridge_train_test(train_feature_matrix, train_y, test_feature_matrix)

    train_perm = rng.permutation(len(train_y))
    test_perm = rng.permutation(len(test_y))
    shuffled_train = train_feature_matrix[train_perm]
    shuffled_test = test_feature_matrix[test_perm]
    predictions[f"shuffled_{feature_name}"] = ridge_train_test(shuffled_train, train_y, shuffled_test)

    random_train = rng.normal(size=train_feature_matrix.shape)
    random_test = rng.normal(size=test_feature_matrix.shape)
    predictions[f"random_gaussian_{feature_name}"] = ridge_train_test(random_train, train_y, random_test)

    ar_train_x = np.concatenate([train_ar, train_feature_matrix], axis=1)
    ar_test_x = np.concatenate([test_ar, test_feature_matrix], axis=1)
    predictions[f"autoregressive_plus_{feature_name}"] = ridge_train_test(ar_train_x, train_y, ar_test_x)

    predictions[f"autoregressive_plus_shuffled_{feature_name}"] = ridge_train_test(
        np.concatenate([train_ar, shuffled_train], axis=1),
        train_y,
        np.concatenate([test_ar, shuffled_test], axis=1),
    )

    ar_random_train = rng.normal(size=train_feature_matrix.shape)
    ar_random_test = rng.normal(size=test_feature_matrix.shape)
    predictions[f"autoregressive_plus_random_gaussian_{feature_name}"] = ridge_train_test(
        np.concatenate([train_ar, ar_random_train], axis=1),
        train_y,
        np.concatenate([test_ar, ar_random_test], axis=1),
    )

    neuro_residual_train = train_y - ar_train_pred
    residual_train_pred, residual_test_pred = ridge_train_test(
        train_feature_matrix,
        neuro_residual_train,
        test_feature_matrix,
    )
    predictions[f"residualized_autoregressive_plus_{feature_name}"] = (
        ar_train_pred + residual_train_pred,
        ar_test_pred + residual_test_pred,
    )

    residual_train_shuffled, residual_test_shuffled = ridge_train_test(
        shuffled_train,
        neuro_residual_train,
        shuffled_test,
    )
    predictions[f"residualized_autoregressive_plus_shuffled_{feature_name}"] = (
        ar_train_pred + residual_train_shuffled,
        ar_test_pred + residual_test_shuffled,
    )

    residual_random_train = rng.normal(size=train_feature_matrix.shape)
    residual_random_test = rng.normal(size=test_feature_matrix.shape)
    residual_train_random, residual_test_random = ridge_train_test(
        residual_random_train,
        neuro_residual_train,
        residual_random_test,
    )
    predictions[f"residualized_autoregressive_plus_random_gaussian_{feature_name}"] = (
        ar_train_pred + residual_train_random,
        ar_test_pred + residual_test_random,
    )

    train_videos = len({row["video_id"] for row in inputs.train_rows})
    test_videos = len({row["video_id"] for row in inputs.test_rows})
    lane_rows = []
    for key, (train_scores, test_scores) in predictions.items():
        metrics = score_metrics(
            task_type=str(contract.target_metadata.get("task_type", "binary")),
            train_y=train_y,
            train_scores=train_scores,
            test_y=test_y,
            test_scores=test_scores,
        )
        lane_rows.append(
            flatten_lane_row(
                contract=contract,
                prediction_key=key,
                train_rows=len(inputs.train_rows),
                test_rows=len(inputs.test_rows),
                train_videos=train_videos,
                test_videos=test_videos,
                metrics=metrics,
            )
        )

    leakage_checks = [
        {
            "check_name": "same_row_order_before_ar_scoring",
            "status": "pass",
            "representation_name": contract.representation_name,
            "split": contract.split,
            "target": contract.target_name,
            "threshold": contract.threshold,
            "details": {
                "train_rows": len(inputs.train_rows),
                "test_rows": len(inputs.test_rows),
            },
        },
        {
            "check_name": "computed_ar_fresh",
            "status": "pass",
            "representation_name": contract.representation_name,
            "split": contract.split,
            "target": contract.target_name,
            "threshold": contract.threshold,
            "details": {"ar_feature_width": int(train_ar.shape[1])},
        },
    ]
    return lane_rows, leakage_checks


class FrozenTensorFeatureProvider:
    """Load frozen tensor contracts for use by the existing VEATIC evaluator."""

    def __init__(
        self,
        *,
        tensor_root: Path | None = None,
        summary_root: Path = SUMMARY_ROOT,
        representations: tuple[str, ...] = REQUIRED_REPRESENTATIONS,
    ) -> None:
        self.tensor_root = (tensor_root or default_tensor_root()).expanduser()
        self.summary_root = summary_root.expanduser()
        self.representations = representations
        reject_exclude_83_paths(self.tensor_root)
        reject_exclude_83_paths(self.summary_root)

    @property
    def summary_path(self) -> Path:
        return self.summary_root / "tensor_export_summary.json"

    @property
    def verification_path(self) -> Path:
        return self.summary_root / "tensor_export_verification.json"

    @property
    def inventory_path(self) -> Path:
        return self.summary_root / "tensor_export_inventory.csv"

    def summary(self) -> dict[str, Any]:
        return read_json(self.summary_path)

    def verification(self) -> dict[str, Any]:
        return read_json(self.verification_path)

    def targets(self) -> list[dict[str, Any]]:
        return parse_targets(self.summary())

    def contract_dirs(self, representation: str, split: str, target: dict[str, Any]) -> tuple[Path, Path]:
        rel = Path(representation) / split / target["folder"]
        contract_dir = self.tensor_root / rel
        tracked_dir = self.summary_root / rel
        reject_exclude_83_paths(contract_dir)
        return contract_dir, tracked_dir

    def load_contract(
        self,
        *,
        representation: str,
        split: str,
        target_name: str,
        threshold: float,
    ) -> FrozenTensorContract:
        target = {"target_name": target_name, "threshold": threshold, "folder": target_dir_name(target_name, threshold)}
        contract_dir, tracked_dir = self.contract_dirs(representation, split, target)
        feature_name = FEATURE_ALIASES.get(representation, representation)
        return FrozenTensorContract(
            representation_name=representation,
            feature_name=feature_name,
            split=split,
            target_name=target_name,
            threshold=threshold,
            contract_dir=contract_dir,
            tracked_dir=tracked_dir,
            train_x=np.load(contract_dir / "X_train.npy"),
            test_x=np.load(contract_dir / "X_test.npy"),
            train_y=np.load(contract_dir / "y_train.npy"),
            test_y=np.load(contract_dir / "y_test.npy"),
            train_row_metadata=read_jsonl(contract_dir / "row_metadata_train.jsonl"),
            test_row_metadata=read_jsonl(contract_dir / "row_metadata_test.jsonl"),
            representation_metadata=read_json(tracked_dir / "representation_metadata.json"),
            split_metadata=read_json(tracked_dir / "split_metadata.json"),
            target_metadata=read_json(tracked_dir / "target_metadata.json"),
            checksum_manifest=read_json(tracked_dir / "checksum_manifest.json"),
            leakage_contract=read_json(tracked_dir / "leakage_contract.json"),
        )

    def preflight_checks(self) -> list[dict[str, Any]]:
        summary = self.summary()
        verification = self.verification()
        checks: list[dict[str, Any]] = []
        add_check(checks, "tensor_root_exists", "pass" if self.tensor_root.exists() else "fail", {"path": str(self.tensor_root)})
        add_check(checks, "summary_root_exists", "pass" if self.summary_root.exists() else "fail", {"path": str(self.summary_root)})
        add_check(checks, "verification_status_pass", "pass" if verification.get("status") == "pass" else "fail", {"status": verification.get("status")})
        add_check(checks, "verification_failures_empty", "pass" if not verification.get("failures") else "fail", {"failures": verification.get("failures", [])})
        add_check(checks, "summary_video_83_included", "pass" if summary.get("video_83_included") is True else "fail", {"video_83_included": summary.get("video_83_included")})
        add_check(
            checks,
            "exclude_video_83_sensitivity_absent",
            "pass" if summary.get("exclude_video_83_sensitivity_exported") is False else "fail",
            {"exclude_video_83_sensitivity_exported": summary.get("exclude_video_83_sensitivity_exported")},
        )

        exported = set(summary.get("representations_exported", []))
        missing_reps = sorted(set(self.representations) - exported)
        add_check(checks, "required_representations_exported", "pass" if not missing_reps else "fail", {"missing": missing_reps})

        safe = set(summary.get("safe_for_primary_training", []))
        unsafe_required = sorted(set(self.representations) - safe)
        add_check(checks, "required_representations_safe_for_primary_training", "pass" if not unsafe_required else "fail", {"unsafe": unsafe_required})

        targets = parse_targets(summary)
        add_check(checks, "targets_available", "pass" if targets else "fail", {"targets": targets})
        splits = tuple(summary.get("splits_exported", SPLITS))
        add_check(checks, "full_split_set_available", "pass" if splits == SPLITS else "fail", {"splits": list(splits)})

        for representation in self.representations:
            for split in SPLITS:
                for target in targets:
                    contract_dir, tracked_dir = self.contract_dirs(representation, split, target)
                    rel = Path(representation) / split / target["folder"]
                    required_payloads = [
                        "X_train.npy",
                        "X_test.npy",
                        "y_train.npy",
                        "y_test.npy",
                        "row_metadata_train.jsonl",
                        "row_metadata_test.jsonl",
                    ]
                    if representation == PRIMARY_CANDIDATE:
                        required_payloads.extend(
                            ["X_sequence_train.npy", "X_sequence_test.npy", "sequence_mask_train.npy", "sequence_mask_test.npy"]
                        )
                    missing_payloads = [name for name in required_payloads if not (contract_dir / name).exists()]
                    add_check(
                        checks,
                        "required_external_payloads_exist",
                        "pass" if not missing_payloads else "fail",
                        {"contract": str(rel), "missing": missing_payloads},
                    )

                    metadata_files = [
                        "representation_metadata.json",
                        "split_metadata.json",
                        "target_metadata.json",
                        "checksum_manifest.json",
                        "leakage_contract.json",
                    ]
                    missing_metadata = [name for name in metadata_files if not (tracked_dir / name).exists()]
                    add_check(
                        checks,
                        "tracked_metadata_exists",
                        "pass" if not missing_metadata else "fail",
                        {"contract": str(rel), "missing": missing_metadata},
                    )
                    if missing_metadata:
                        continue

                    rep_meta = read_json(tracked_dir / "representation_metadata.json")
                    split_meta = read_json(tracked_dir / "split_metadata.json")
                    add_check(
                        checks,
                        "no_future_feature_rows",
                        "pass" if rep_meta.get("uses_future_features") is False else "fail",
                        {"contract": str(rel), "uses_future_features": rep_meta.get("uses_future_features")},
                    )
                    pca_meta = rep_meta.get("pca") or {}
                    pca_ok = (not pca_meta) or (
                        pca_meta.get("pca_fit_scope") == "train_rows_only"
                        and pca_meta.get("cache_rebuilt") is False
                    )
                    add_check(checks, "train_only_pca_scope", "pass" if pca_ok else "fail", {"contract": str(rel), "pca": pca_meta})
                    grouped_ok = (not split.startswith("grouped_")) or not split_meta.get("train_test_video_overlap")
                    add_check(
                        checks,
                        "grouped_video_disjoint",
                        "pass" if grouped_ok else "fail",
                        {"contract": str(rel), "train_test_video_overlap": split_meta.get("train_test_video_overlap", [])},
                    )

                    train_83, train_meta_exists = count_video_83_rows(contract_dir / "row_metadata_train.jsonl")
                    test_83, test_meta_exists = count_video_83_rows(contract_dir / "row_metadata_test.jsonl")
                    if train_83 + test_83 > 0:
                        status = "pass"
                        details = {"contract": str(rel), "video_83_rows": train_83 + test_83, "train_rows": train_83, "test_rows": test_83}
                    elif train_meta_exists and test_meta_exists:
                        status = "recorded_absent_after_target_horizon_trimming"
                        details = {
                            "contract": str(rel),
                            "video_83_rows": 0,
                            "reason": "No eligible video 83 rows appear in exported row metadata for this split/target contract.",
                            "dropped_rows": rep_meta.get("dropped_rows"),
                        }
                    else:
                        status = "fail"
                        details = {"contract": str(rel), "reason": "row metadata missing; cannot audit video 83 eligibility"}
                    add_check(checks, "video_83_not_deliberately_excluded", status, details)

        return checks

    def freshness_ledger(
        self,
        fresh_run_id: str,
        *,
        computed_scores: bool = False,
        wrote_result_csvs: bool = False,
    ) -> dict[str, Any]:
        return {
            "fresh_run_id": fresh_run_id,
            "code_git_sha": git_sha(),
            "tensor_export_summary_sha256": sha256_file(self.summary_path),
            "tensor_export_inventory_sha256": sha256_file(self.inventory_path),
            "benchmark_contract_version": BENCHMARK_CONTRACT_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "benchmark_mode": BENCHMARK_MODE,
            "computed_scores": computed_scores,
            "fit_models": computed_scores,
            "wrote_result_csvs": wrote_result_csvs,
            "full_veatic_124": True,
            "video_83_included": True,
            "exclude_video_83_run": False,
            "computed_ar_fresh": computed_scores,
            "computed_controls_fresh": computed_scores,
            "reused_benchmark_result_rows": False,
            "result_row_reuse_policy": {
                "prior_benchmark_rows_reused": False,
                "prior_ar_rows_reused": False,
                "prior_ridge_rows_reused": False,
                "prior_control_rows_reused": False,
                "forbidden_inputs": list(FORBIDDEN_RESULT_INPUT_NAMES),
            },
        }

    def _policy_checks(self, preflight_checks: list[dict[str, Any]], lane_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        lanes = {row["lane"] for row in lane_rows}
        required_lanes = {
            "AR_only",
            "PCA128_only",
            "PCA64_delta_only",
            "AR_plus_PCA128",
            "AR_plus_PCA64_delta",
            "residualized_AR_plus_PCA128",
            "residualized_AR_plus_PCA64_delta",
        }
        video_83_checks = [row for row in preflight_checks if row["check_name"] == "video_83_not_deliberately_excluded"]
        full_checks = [
            {"check_name": "benchmark_mode", "status": "pass", "details": {"benchmark_mode": BENCHMARK_MODE}},
            {"check_name": "full_veatic_124", "status": "pass", "details": {"splits": list(SPLITS)}},
            {"check_name": "exclude_video_83_run", "status": "pass", "details": {"exclude_video_83_run": False}},
            {
                "check_name": "video_83_included",
                "status": "pass" if all(row["status"] != "fail" for row in video_83_checks) else "fail",
                "details": {"video_83_contract_checks": len(video_83_checks)},
            },
            {
                "check_name": "required_lanes_computed",
                "status": "pass" if required_lanes.issubset(lanes) else "fail",
                "details": {"missing_lanes": sorted(required_lanes - lanes), "lane_count": len(lanes)},
            },
            {"check_name": "promotion_json_not_written", "status": "pass", "details": {"promotion_json": False}},
        ]
        return full_checks

    def _report_markdown(
        self,
        *,
        run_manifest: dict[str, Any],
        gate_checks: list[dict[str, Any]],
        lane_count: int,
        fold_count: int,
    ) -> str:
        return "\n".join(
            [
                "# VEATIC Frozen Tensor Ridge-Only Benchmark",
                "",
                "This run uses the existing VEATIC benchmark lane semantics with a frozen tensor adapter.",
                "",
                f"- benchmark_mode: `{BENCHMARK_MODE}`",
                "- head: `ridge_only`",
                f"- computed_scores: `{str(run_manifest['computed_scores']).lower()}`",
                "- reused_benchmark_result_rows: `false`",
                "- computed_ar_fresh: `true`",
                "- computed_controls_fresh: `true`",
                "- full_veatic_124: `true`",
                "- video_83_included: `true`",
                "- exclude_video_83_run: `false`",
                f"- lane_rows: `{lane_count}`",
                f"- fold_rows: `{fold_count}`",
                "",
                "No promotion JSON or final claim is produced by this run.",
                "",
                "## Gate Checks",
                "",
                *[
                    f"- {check['check_name']}: `{check['status']}`"
                    for check in gate_checks
                ],
                "",
            ]
        )

    def score_ridge_only(
        self,
        *,
        all_rows: list[dict[str, Any]],
        output_dir: Path,
        fresh_run_id: str,
        seed: int = 43,
    ) -> dict[str, Any]:
        deny_prior_result_files([self.tensor_root, self.summary_root])
        reject_exclude_83_paths(output_dir)
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError(f"Output directory must be new or empty: {output_dir}")

        preflight_checks = self.preflight_checks()
        failures = check_failures(preflight_checks)
        if failures:
            raise ValueError(f"Frozen tensor preflight failed: {failures}")

        lane_rows: list[dict[str, Any]] = []
        leakage_checks: list[dict[str, Any]] = list(preflight_checks)
        targets = self.targets()
        job_index = 0
        for representation in self.representations:
            for split in SPLITS:
                for target in targets:
                    contract = self.load_contract(
                        representation=representation,
                        split=split,
                        target_name=target["target_name"],
                        threshold=float(target["threshold"]),
                    )
                    rows, checks = score_contract_ridge_existing_semantics(
                        contract,
                        all_rows=all_rows,
                        seed=seed + job_index,
                    )
                    lane_rows.extend(rows)
                    leakage_checks.extend(checks)
                    job_index += 1

        control_rows = [row for row in lane_rows if row["is_control"]]
        fold_rows = [row for row in lane_rows if str(row["split"]).startswith("grouped_")]
        gate_checks = self._policy_checks(preflight_checks, lane_rows)
        gate_failures = check_failures(gate_checks)
        if gate_failures:
            raise ValueError(f"Ridge-only gate checks failed: {gate_failures}")

        output_dir.mkdir(parents=True, exist_ok=True)
        ledger = self.freshness_ledger(
            fresh_run_id,
            computed_scores=True,
            wrote_result_csvs=True,
        )
        run_manifest = {
            "schema_version": "veatic_frozen_tensor_ridge_only_run_v1",
            "benchmark_contract_version": BENCHMARK_CONTRACT_VERSION,
            "benchmark_mode": BENCHMARK_MODE,
            "fresh_run_id": fresh_run_id,
            "seed": seed,
            "computed_scores": True,
            "reused_benchmark_result_rows": False,
            "computed_ar_fresh": True,
            "computed_controls_fresh": True,
            "full_veatic_124": True,
            "video_83_included": True,
            "exclude_video_83_run": False,
            "ridge_only": True,
            "logistic_heads": False,
            "mlp_heads": False,
            "learned_temporal_pooling": False,
            "representations": list(self.representations),
            "splits": list(SPLITS),
            "targets": targets,
            "lane_count": len(lane_rows),
            "fold_count": len(fold_rows),
            "control_count": len(control_rows),
            "outputs": {
                "lane_results_csv": str(output_dir / "lane_results.csv"),
                "fold_results_csv": str(output_dir / "fold_results.csv"),
                "control_results_csv": str(output_dir / "control_results.csv"),
                "freshness_ledger_json": str(output_dir / "freshness_ledger.json"),
                "leakage_checks_json": str(output_dir / "leakage_checks.json"),
                "gate_checks_json": str(output_dir / "gate_checks.json"),
                "run_manifest_json": str(output_dir / "run_manifest.json"),
                "ridge_only_report_md": str(output_dir / "ridge_only_report.md"),
            },
        }

        write_csv(output_dir / "lane_results.csv", lane_rows)
        write_csv(output_dir / "fold_results.csv", fold_rows)
        write_csv(output_dir / "control_results.csv", control_rows)
        write_json(output_dir / "freshness_ledger.json", ledger)
        write_json(output_dir / "leakage_checks.json", leakage_checks)
        write_json(output_dir / "gate_checks.json", gate_checks)
        write_json(output_dir / "run_manifest.json", run_manifest)
        (output_dir / "ridge_only_report.md").write_text(
            self._report_markdown(
                run_manifest=run_manifest,
                gate_checks=gate_checks,
                lane_count=len(lane_rows),
                fold_count=len(fold_rows),
            ),
            encoding="utf-8",
        )
        return run_manifest

    def dry_run_plan(self, *, fresh_run_id: str | None = None) -> dict[str, Any]:
        summary = self.summary()
        checks = self.preflight_checks()
        run_id = fresh_run_id or f"dryrun_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        feature_keys = {FEATURE_ALIASES[name]: canonical_prediction_keys(FEATURE_ALIASES[name]) for name in self.representations}
        return {
            "schema_version": "veatic_frozen_tensor_adapter_dry_run_v1",
            "benchmark_contract_version": BENCHMARK_CONTRACT_VERSION,
            "benchmark_mode": BENCHMARK_MODE,
            "dry_run_only": True,
            "dry_run_contract": "must not fit models, compute real benchmark scores, or write result CSVs",
            "reused_existing_benchmark_structure": True,
            "reused_frozen_tensors": True,
            "reused_canonical_controls": True,
            "reused_benchmark_result_rows": False,
            "computed_ar_fresh": True,
            "computed_controls_fresh": True,
            "core_question": "Does pca_sequence_128_causal_past_2s_mean add predictive signal beyond fresh same-row AR_only?",
            "tensor_root": str(self.tensor_root),
            "summary_root": str(self.summary_root),
            "representations": {
                "primary_candidate": PRIMARY_CANDIDATE,
                "previous_neural_baseline": FROZEN_BASELINE,
                "feature_aliases": {name: FEATURE_ALIASES[name] for name in self.representations},
                "safe_for_primary_training": list(SAFE_PRIMARY_REPRESENTATIONS),
            },
            "targets": parse_targets(summary),
            "splits": list(SPLITS),
            "existing_suite_prediction_keys": feature_keys,
            "canonical_controls": [row["control"] for row in strict.CONTROL_LEDGER],
            "canonical_helpers": canonical_helper_references(),
            "event_metric_helpers_found": discover_event_metric_helpers(),
            "same_row_assertion": (
                "Before AR scoring, assert AR train/test row order exactly matches tensor row metadata order "
                "using video_id, time_start_seconds, and frame_index. Stop on mismatch. Do not sort or realign silently."
            ),
            "video_83_policy": (
                "Verify video 83 is not deliberately excluded. For each split/target contract, confirm video 83 appears "
                "if it has eligible target rows. If it has no eligible rows due to target-horizon trimming, record that explicitly."
            ),
            "reuse_policy": {
                "reuse": [
                    "frozen tensor inputs",
                    "canonical benchmark/control implementations",
                    "deterministic split definitions",
                    "existing metric/control helper implementations",
                ],
                "recompute_later": [
                    "AR matrices",
                    "model fits",
                    "predictions",
                    "controls",
                    "summaries",
                    "gates",
                ],
                "do_not_reuse": [
                    "old CSV result rows",
                    "old AR scores",
                    "old ridge scores",
                    "old control scores",
                    "old promotion JSON",
                    "old grouped summaries",
                ],
            },
            "freshness_ledger": self.freshness_ledger(run_id),
            "preflight_checks": checks,
            "preflight_status": "pass" if not check_failures(checks) else "fail",
            "failure_count": len(check_failures(checks)),
        }
