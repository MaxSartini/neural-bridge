"""Contracts and verdict for the locked zero-label direct-supervised confirmation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from backend.scripts import again_zero_label_deployment_stage_a as stage_a


SCHEMA_VERSION = "again_zero_label_direct_supervised_locked_confirmation_v1"
AUTHORIZATION = "explicit_user_authorization_20260715_continue_with_all"
PREREGISTRATION = Path(
    "docs/zero_label_video_only_direct_supervised_locked_confirmation_preregistration.md"
)
PANELS = (1, 2, 3, 4, 5)
SEEDS = (20260721, 20260722, 20260723)
GROUP_NAME = "20260721_20260722_20260723"
PRIMARY = "video_supervised_temporal"
CURRENT_ROW = "video_supervised_current_row"
DIAGNOSTICS_ONLY = "diagnostics_only_supervised_temporal"
NO_VIDEO = "no_video_supervised_temporal"
SHUFFLED = "sequence_shuffled_supervised_temporal"
PERMUTED = "label_permutation_supervised_temporal"
TEACHER = "phase7_ar_assisted_teacher_ceiling"
LANES = (
    PRIMARY,
    CURRENT_ROW,
    DIAGNOSTICS_ONLY,
    NO_VIDEO,
    SHUFFLED,
    PERMUTED,
    TEACHER,
)
ZERO_LABEL_LANES = tuple(lane for lane in LANES if lane != TEACHER)
FALSE_SIGNAL_CONTROLS = (DIAGNOSTICS_ONLY, NO_VIDEO, SHUFFLED, PERMUTED)
REQUIRED_METRICS = stage_a.REQUIRED_METRICS
FIRST30_METRICS = tuple(f"first30_{metric}" for metric in REQUIRED_METRICS)
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260724


def implementation_freeze_manifest(*, preregistration_sha256: str) -> Mapping[str, Any]:
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "authorization": AUTHORIZATION,
        "preregistration": str(PREREGISTRATION),
        "preregistration_sha256": preregistration_sha256,
        "stage": "locked_confirmation",
        "candidate": PRIMARY,
        "panels": list(PANELS),
        "seeds": list(SEEDS),
        "ensemble": {"members": list(SEEDS), "weights": [1 / 3, 1 / 3, 1 / 3]},
        "lanes": list(LANES),
        "false_signal_controls": list(FALSE_SIGNAL_CONTROLS),
        "required_metrics": list(REQUIRED_METRICS),
        "teacher_retention_is_gate": False,
        "absolute_teacher_parity_required": False,
        "model": {
            "hidden": stage_a.HIDDEN,
            "optimizer": "mlx.optimizers.AdamW",
            "learning_rate": stage_a.LEARNING_RATE,
            "weight_decay": stage_a.WEIGHT_DECAY,
            "max_epochs": stage_a.MAX_EPOCHS,
            "patience": stage_a.PATIENCE,
            "batch_size": stage_a.BATCH_SIZE,
            "gradient_clip": stage_a.GRAD_CLIP,
            "loss": "weighted_huber_to_hard_future_movement",
            "checkpoint_selection": "inner_grouped_video_hard_target_huber_only",
        },
        "features": {
            "source_family": stage_a.PCA_FAMILY,
            "pca_width": stage_a.PCA_WIDTH,
            "sequence_window_rows": stage_a.WINDOW_ROWS,
            "diagnostics_width": stage_a.DIAGNOSTIC_WIDTH,
            "outer_train_only": True,
        },
        "controls": {
            "diagnostics_only": "zero_all_predicted_cortical_pca_channels",
            "no_video": "zero_pca_and_diagnostics_keep_masks_and_time_only",
            "sequence_shuffle": "deterministic_whole_video_input_donor_resampling",
            "label_permutation": "deterministic_whole_video_hard_target_donor_resampling",
        },
        "cold_start": {
            "prediction_starts_at_row0": True,
            "missing_history": "zeros_plus_explicit_mask",
            "observed_response_inputs": False,
            "teacher_forcing_ratio": 0.0,
        },
        "event_threshold": {"quantile": stage_a.EVENT_QUANTILE, "fit_scope": "development_only"},
        "bootstrap": {
            "unit": "video",
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "one_sided_lower_quantile": 0.05,
        },
        "hardware": "mlx_gpu_mps",
        "cpu_fallback": False,
        "rows": {"member": 105, "ensemble": 35, "total": 140},
        "heldout_hyperparameter_search": False,
        "member_selection": False,
        "weight_search": False,
    }
    return {**manifest, "implementation_freeze_digest": stage_a.canonical_digest(manifest)}


def scored_row_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row["stage"],
        row["split_digest"],
        int(row["panel"]),
        row["lane"],
        row["row_type"],
        str(row["seed_or_group"]),
        row["cold_start_policy"],
    )


def strongest_control(
    aggregate: Mapping[str, Mapping[str, float]], metric: str
) -> tuple[str, float]:
    lane = max(FALSE_SIGNAL_CONTROLS, key=lambda item: float(aggregate[item][metric]))
    return lane, float(aggregate[lane][metric])


def _panel_deltas(
    frame: pd.DataFrame, *, control_lane: str, metric: str
) -> list[float]:
    ensembles = frame[frame["row_type"] == "ensemble"]
    deltas: list[float] = []
    for panel in PANELS:
        panel_rows = ensembles[ensembles["panel"] == panel].set_index("lane")
        deltas.append(float(panel_rows.loc[PRIMARY, metric] - panel_rows.loc[control_lane, metric]))
    return deltas


def compute_verdict(
    frame: pd.DataFrame,
    *,
    aggregate: Mapping[str, Mapping[str, float]],
    bootstrap: Mapping[str, Mapping[str, float]],
    audit_pass: bool,
) -> Mapping[str, Any]:
    expected = len(PANELS) * len(LANES) * (len(SEEDS) + 1)
    keys = [scored_row_key(row) for row in frame.to_dict(orient="records")]
    scope_pass = (
        len(frame) == expected
        and len(set(keys)) == expected
        and set(frame["panel"].astype(int)) == set(PANELS)
        and set(frame["lane"].astype(str)) == set(LANES)
        and set(frame["row_type"].astype(str)) == {"member", "ensemble"}
    )
    finite_pass = bool(
        all(np.isfinite(frame[metric].to_numpy(dtype=float)).all() for metric in REQUIRED_METRICS)
    )

    metric_results: dict[str, Mapping[str, Any]] = {}
    tier1_metric_passes: list[bool] = []
    tier2_metric_passes: list[bool] = []
    tier3_metric_passes: list[bool] = []
    for metric, first30_metric in zip(REQUIRED_METRICS, FIRST30_METRICS, strict=True):
        control_lane, control_value = strongest_control(aggregate, metric)
        aggregate_delta = float(aggregate[PRIMARY][metric]) - control_value
        panel_deltas = _panel_deltas(frame, control_lane=control_lane, metric=metric)
        panel_wins = sum(delta > 0 for delta in panel_deltas)
        panel_median = float(np.median(panel_deltas))
        shuffled_delta = float(aggregate[PRIMARY][metric]) - float(aggregate[SHUFFLED][metric])
        permuted_delta = float(aggregate[PRIMARY][metric]) - float(aggregate[PERMUTED][metric])
        tier1_metric = bool(
            aggregate_delta > 0
            and panel_median > 0
            and shuffled_delta > 0
            and permuted_delta > 0
        )
        lower = float(bootstrap[metric]["lower_95_one_sided"])
        tier2_metric = bool(tier1_metric and panel_wins >= 4 and lower > 0)

        first30_lane, first30_control_value = strongest_control(aggregate, first30_metric)
        first30_delta = float(aggregate[PRIMARY][first30_metric]) - first30_control_value
        first30_panel_deltas = _panel_deltas(
            frame, control_lane=first30_lane, metric=first30_metric
        )
        first30_wins = sum(delta > 0 for delta in first30_panel_deltas)
        tier3_metric = bool(first30_delta > 0 and first30_wins >= 4)

        metric_results[metric] = {
            "strongest_control": control_lane,
            "primary": float(aggregate[PRIMARY][metric]),
            "control": control_value,
            "aggregate_delta": aggregate_delta,
            "panel_deltas": panel_deltas,
            "panel_wins": f"{panel_wins}/5",
            "panel_median_delta": panel_median,
            "shuffled_delta": shuffled_delta,
            "label_permutation_delta": permuted_delta,
            "bootstrap_lower_95_one_sided": lower,
            "tier1_metric_pass": tier1_metric,
            "tier2_metric_pass": tier2_metric,
            "first30_strongest_control": first30_lane,
            "first30_aggregate_delta": first30_delta,
            "first30_panel_deltas": first30_panel_deltas,
            "first30_panel_wins": f"{first30_wins}/5",
            "tier3_metric_pass": tier3_metric,
        }
        tier1_metric_passes.append(tier1_metric)
        tier2_metric_passes.append(tier2_metric)
        tier3_metric_passes.append(tier3_metric)

    tier1 = bool(scope_pass and finite_pass and audit_pass and all(tier1_metric_passes))
    tier2 = bool(tier1 and all(tier2_metric_passes))
    tier3 = bool(tier1 and all(tier3_metric_passes))
    mechanism = {
        metric: {
            "primary_minus_current_row": float(aggregate[PRIMARY][metric])
            - float(aggregate[CURRENT_ROW][metric]),
            "primary_minus_diagnostics_only": float(aggregate[PRIMARY][metric])
            - float(aggregate[DIAGNOSTICS_ONLY][metric]),
        }
        for metric in REQUIRED_METRICS
    }
    failed: list[str] = []
    if not scope_pass:
        failed.append("exact_140_row_scope")
    if not finite_pass:
        failed.append("finite_required_metrics")
    if not audit_pass:
        failed.append("all_provenance_inference_and_hardware_audits")
    if not all(tier1_metric_passes):
        failed.append("tier1_all_endpoints_beat_false_signal_and_no_video_controls")
    return {
        "schema_version": SCHEMA_VERSION,
        "rows_expected": expected,
        "rows_actual": int(len(frame)),
        "scope_pass": scope_pass,
        "finite_pass": finite_pass,
        "audit_pass": bool(audit_pass),
        "tier1_zero_label_deployment_signal_confirmed": tier1,
        "tier2_high_consistency_confirmation": tier2,
        "tier3_first30_cold_start_confirmation": tier3,
        "metric_results": metric_results,
        "mechanism_results": mechanism,
        "teacher_retention_is_gate": False,
        "phase7_teacher_is_ceiling_not_threshold": True,
        "failed_tier1_gates": failed,
    }


def relative_gain(value: float, baseline: float) -> float:
    if not math.isfinite(value) or not math.isfinite(baseline) or baseline == 0:
        return math.nan
    return (value - baseline) / abs(baseline)
