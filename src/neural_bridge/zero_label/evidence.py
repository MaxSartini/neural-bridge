"""Recompute and verify the sealed AGAIN zero-label confirmation closure.

This module verifies concluded evidence. It does not reopen the locked pool,
refit the historical model, or promote a new claim.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PANELS = (1, 2, 3, 4, 5)
SEEDS = (20260721, 20260722, 20260723)
PRIMARY = "video_supervised_temporal"
CURRENT_ROW = "video_supervised_current_row"
DIAGNOSTICS_ONLY = "diagnostics_only_supervised_temporal"
NO_VIDEO = "no_video_supervised_temporal"
SHUFFLED = "sequence_shuffled_supervised_temporal"
PERMUTED = "label_permutation_supervised_temporal"
TEACHER = "phase7_ar_assisted_teacher_ceiling"
LANES = (PRIMARY, CURRENT_ROW, DIAGNOSTICS_ONLY, NO_VIDEO, SHUFFLED, PERMUTED, TEACHER)
ZERO_LABEL_LANES = LANES[:-1]
FALSE_SIGNAL_CONTROLS = (DIAGNOSTICS_ONLY, NO_VIDEO, SHUFFLED, PERMUTED)
REQUIRED_METRICS = (
    "pooled_continuous_spearman",
    "top_5pct_true_future_movement_lift",
    "training_q90_future_event_pr_auc",
)
FIRST30_METRICS = tuple(f"first30_{metric}" for metric in REQUIRED_METRICS)
RUNTIME_ARTIFACTS = {
    "locked_confirmation_result.json": "metrics/locked_confirmation_result.json",
    "locked_confirmation_rows.csv": "metrics/locked_confirmation_rows.csv",
    "aggregate_metrics.json": "metrics/aggregate_metrics.json",
    "paired_video_bootstrap.json": "metrics/paired_video_bootstrap.json",
    "implementation_freeze.json": "manifests/implementation_freeze.json",
    "zero_label_prediction_seal.json": "manifests/zero_label_prediction_seal.json",
}
IGNORED_NAMES = {".DS_Store", ".pytest_cache", "__pycache__"}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_identity(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    files = 0
    size_bytes = 0
    paths = [root] if root.is_file() or root.is_symlink() else sorted(root.rglob("*"))
    for path in paths:
        if any(part in IGNORED_NAMES for part in path.relative_to(root).parts):
            continue
        if path.is_dir() and not path.is_symlink():
            continue
        relative = "." if path == root else path.relative_to(root).as_posix()
        if path.is_symlink():
            digest.update(f"L\0{relative}\0{os.readlink(path)}\n".encode())
            continue
        size = path.stat().st_size
        digest.update(f"F\0{relative}\0{size}\0{_file_sha256(path)}\n".encode())
        files += 1
        size_bytes += size
    return {"sha256_tree": digest.hexdigest(), "files": files, "size_bytes": size_bytes}


def _scored_row_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row["stage"],
        row["split_digest"],
        int(row["panel"]),
        row["lane"],
        row["row_type"],
        str(row["seed_or_group"]),
        row["cold_start_policy"],
    )


def _strongest_control(
    aggregate: Mapping[str, Mapping[str, float]], metric: str
) -> tuple[str, float]:
    lane = max(FALSE_SIGNAL_CONTROLS, key=lambda item: float(aggregate[item][metric]))
    return lane, float(aggregate[lane][metric])


def _panel_deltas(frame: pd.DataFrame, *, control_lane: str, metric: str) -> list[float]:
    ensembles = frame[frame["row_type"] == "ensemble"]
    deltas = []
    for panel in PANELS:
        panel_rows = ensembles[ensembles["panel"] == panel].set_index("lane")
        deltas.append(float(panel_rows.loc[PRIMARY, metric] - panel_rows.loc[control_lane, metric]))
    return deltas


def _compute_verdict(
    frame: pd.DataFrame,
    *,
    aggregate: Mapping[str, Mapping[str, float]],
    bootstrap: Mapping[str, Mapping[str, float]],
    audit_pass: bool,
) -> dict[str, Any]:
    expected = len(PANELS) * len(LANES) * (len(SEEDS) + 1)
    keys = [_scored_row_key(row) for row in frame.to_dict(orient="records")]
    scope_pass = (
        len(frame) == expected
        and len(set(keys)) == expected
        and set(frame["panel"].astype(int)) == set(PANELS)
        and set(frame["lane"].astype(str)) == set(LANES)
        and set(frame["row_type"].astype(str)) == {"member", "ensemble"}
    )
    finite_pass = all(
        np.isfinite(frame[metric].to_numpy(dtype=float)).all() for metric in REQUIRED_METRICS
    )
    metric_results: dict[str, dict[str, Any]] = {}
    tier1_passes = []
    tier2_passes = []
    tier3_passes = []
    for metric, first30_metric in zip(REQUIRED_METRICS, FIRST30_METRICS, strict=True):
        control_lane, control_value = _strongest_control(aggregate, metric)
        aggregate_delta = float(aggregate[PRIMARY][metric]) - control_value
        panel_deltas = _panel_deltas(frame, control_lane=control_lane, metric=metric)
        panel_wins = sum(delta > 0 for delta in panel_deltas)
        shuffled_delta = float(aggregate[PRIMARY][metric]) - float(aggregate[SHUFFLED][metric])
        permuted_delta = float(aggregate[PRIMARY][metric]) - float(aggregate[PERMUTED][metric])
        tier1_metric = (
            aggregate_delta > 0
            and float(np.median(panel_deltas)) > 0
            and shuffled_delta > 0
            and permuted_delta > 0
        )
        lower = float(bootstrap[metric]["lower_95_one_sided"])
        tier2_metric = tier1_metric and panel_wins >= 4 and lower > 0
        first30_lane, first30_control = _strongest_control(aggregate, first30_metric)
        first30_delta = float(aggregate[PRIMARY][first30_metric]) - first30_control
        first30_deltas = _panel_deltas(
            frame, control_lane=first30_lane, metric=first30_metric
        )
        first30_wins = sum(delta > 0 for delta in first30_deltas)
        tier3_metric = first30_delta > 0 and first30_wins >= 4
        metric_results[metric] = {
            "strongest_control": control_lane,
            "primary": float(aggregate[PRIMARY][metric]),
            "control": control_value,
            "aggregate_delta": aggregate_delta,
            "panel_deltas": panel_deltas,
            "panel_wins": f"{panel_wins}/5",
            "panel_median_delta": float(np.median(panel_deltas)),
            "shuffled_delta": shuffled_delta,
            "label_permutation_delta": permuted_delta,
            "bootstrap_lower_95_one_sided": lower,
            "tier1_metric_pass": tier1_metric,
            "tier2_metric_pass": tier2_metric,
            "first30_strongest_control": first30_lane,
            "first30_aggregate_delta": first30_delta,
            "first30_panel_deltas": first30_deltas,
            "first30_panel_wins": f"{first30_wins}/5",
            "tier3_metric_pass": tier3_metric,
        }
        tier1_passes.append(tier1_metric)
        tier2_passes.append(tier2_metric)
        tier3_passes.append(tier3_metric)
    tier1 = scope_pass and finite_pass and audit_pass and all(tier1_passes)
    return {
        "rows_expected": expected,
        "rows_actual": len(frame),
        "scope_pass": scope_pass,
        "finite_pass": finite_pass,
        "audit_pass": audit_pass,
        "tier1_zero_label_deployment_signal_confirmed": tier1,
        "tier2_high_consistency_confirmation": tier1 and all(tier2_passes),
        "tier3_first30_cold_start_confirmation": tier1 and all(tier3_passes),
        "metric_results": metric_results,
        "teacher_retention_is_gate": False,
        "phase7_teacher_is_ceiling_not_threshold": True,
        "failed_tier1_gates": [] if tier1 else ["recomputed_tier1_failure"],
    }


def _equivalent(left: Any, right: Any, tolerance: float) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _equivalent(left[key], right[key], tolerance) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _equivalent(a, b, tolerance) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return bool(np.isclose(left, right, rtol=0.0, atol=tolerance))
    return left == right


def _audit_policy_pass(audit: Mapping[str, Any]) -> bool:
    records = audit.get("audits", [])
    zero_label = [record for record in records if record.get("lane") in ZERO_LABEL_LANES]
    teacher = [record for record in records if record.get("lane") == TEACHER]
    return bool(
        audit.get("audit_pass")
        and len(records) == 21
        and len(zero_label) == 18
        and len(teacher) == 3
        and all(record.get("all_finite") for record in records)
        and all(record.get("labels_loaded_by_predictor") is False for record in zero_label)
        and all(
            record.get("observed_response_inputs_at_inference") is False
            for record in zero_label
        )
        and all(record.get("opened_after_zero_label_prediction_seal") is True for record in teacher)
    )


def verify_locked_confirmation(
    root: Path,
    *,
    external_root: Path | None = None,
    registry_path: Path | None = None,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Verify compact evidence and optionally the complete external artifact tree."""
    summary = _load_json(root / "locked_confirmation_summary.json")
    result = _load_json(root / "metrics/locked_confirmation_result.json")
    aggregate = _load_json(root / "metrics/aggregate_metrics.json")
    bootstrap = _load_json(root / "metrics/paired_video_bootstrap.json")
    audit = _load_json(root / "diagnostics/locked_confirmation_audit.json")
    freeze = _load_json(root / "manifests/implementation_freeze.json")
    seal = _load_json(root / "manifests/zero_label_prediction_seal.json")
    frame = pd.read_csv(root / "metrics/locked_confirmation_rows.csv")

    expected_hashes = summary["runtime_artifact_sha256"]
    hash_pass = all(
        expected_hashes[name] == _file_sha256(root / relative)
        for name, relative in RUNTIME_ARTIFACTS.items()
    )
    recomputed = _compute_verdict(
        frame,
        aggregate=aggregate,
        bootstrap=bootstrap,
        audit_pass=bool(audit.get("audit_pass")),
    )
    comparable_result = {key: result[key] for key in recomputed}
    freeze_pass = bool(
        freeze.get("candidate") == PRIMARY
        and freeze.get("panels") == list(PANELS)
        and freeze.get("seeds") == list(SEEDS)
        and freeze.get("lanes") == list(LANES)
        and freeze.get("false_signal_controls") == list(FALSE_SIGNAL_CONTROLS)
        and freeze.get("teacher_retention_is_gate") is False
        and freeze.get("heldout_hyperparameter_search") is False
        and freeze.get("member_selection") is False
        and freeze.get("weight_search") is False
        and freeze.get("rows") == {"member": 105, "ensemble": 35, "total": 140}
    )
    split = summary["split"]
    checks = {
        "runtime_artifact_hashes": hash_pass,
        "verdict_recomputed": _equivalent(recomputed, comparable_result, tolerance),
        "inference_audit": _audit_policy_pass(audit),
        "prediction_sealed_before_labels": bool(
            seal.get("sealed_before_teacher_ceiling_and_locked_label_join")
            and len(seal.get("prediction_files", {})) == 18
        ),
        "implementation_frozen": freeze_pass,
        "locked_split_untouched": bool(
            split.get("development_videos") == 696
            and split.get("locked_videos") == 299
            and result.get("locked_video_count") == 299
            and split.get("development_digest") == result.get("development_split_digest")
            and split.get("locked_digest") == result.get("locked_split_digest")
            and split.get("target_identity_digest") == result.get("target_identity_digest")
        ),
    }
    external_check: dict[str, Any] | None = None
    if external_root is not None:
        if registry_path is None:
            raise ValueError("registry_path is required with external_root")
        registered = _load_json(registry_path)
        actual = _tree_identity(external_root)
        external_check = {
            **actual,
            "verification_pass": all(
                actual[key] == registered[key] for key in ("sha256_tree", "files", "size_bytes")
            ),
        }
        checks["external_artifact_tree"] = bool(external_check["verification_pass"])

    scientific_pass = bool(
        recomputed["tier1_zero_label_deployment_signal_confirmed"]
        and recomputed["tier2_high_consistency_confirmation"]
        and recomputed["tier3_first30_cold_start_confirmation"]
    )
    return {
        "schema_version": "neural_bridge_zero_label_evidence_verification_v1",
        "checks": checks,
        "verification_pass": all(checks.values()),
        "scientific_pass": scientific_pass,
        "rows": len(frame),
        "locked_videos": result["locked_video_count"],
        "external_artifact": external_check,
    }
