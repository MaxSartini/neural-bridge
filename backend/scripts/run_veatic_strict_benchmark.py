"""Unified VEATIC-124 strict benchmark suite.

This is the consolidated entrypoint for the current Neural Bridge VEATIC
benchmark contract. It does not shell out to the older retest scripts. Instead,
it reuses their validated helpers in one coordinated pass so fixed splits,
grouped-video folds, controls, event masks, balanced sampling, offset
diagnostics, and causal context windows are produced from the same loaded
context and split matrices.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.scripts import run_veatic_event_conditioned_retest as conditioned  # noqa: E402
from backend.scripts import run_veatic_event_spike_retest as retest  # noqa: E402
from backend.scripts import run_veatic_temporal_fairness_benchmark as fairness  # noqa: E402


bench = retest.bench

FEATURE_MODES: tuple[tuple[str, str, str], ...] = (
    ("cortical_pca64_delta", "cortical_pca64_delta", "primary"),
    ("cortical_pca_64", "cortical_pca_64", "primary"),
    ("cortical_global_delta", "cortical_global_delta", "secondary"),
    ("cortical_fast_default", "cortical_global", "secondary"),
)
TARGETS: tuple[tuple[str, int | None, float, str], ...] = (
    ("arousal__future_spike_1_3s", None, 0.05, "primary"),
    ("arousal__future_spike_1_3s", None, 0.075, "primary"),
    ("arousal__future_change_p3s_movement", 3, 0.05, "primary"),
    ("arousal__future_change_p3s_movement", 3, 0.075, "primary"),
    ("arousal__future_change_p2s_movement", 2, 0.05, "secondary"),
)
CAUSAL_WINDOWS: tuple[tuple[str, float, float, str], ...] = (
    ("current_only_0s", 0.0, 0.0, "last"),
    ("causal_past_1s", -1.0, 0.0, "mean_std_last_slope"),
    ("causal_past_2s", -2.0, 0.0, "mean_std_last_slope"),
    ("causal_past_3s", -3.0, 0.0, "mean_std_last_slope"),
)
DIAGNOSTIC_WINDOWS: tuple[tuple[str, float, float, str], ...] = (
    ("future_3s_diagnostic", 0.0, 3.0, "mean_std_last_slope"),
    ("symmetric_2s_diagnostic", -2.0, 2.0, "mean_std_last_slope"),
)
CORE_OFFSETS: tuple[float, ...] = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
FULL_OFFSETS: tuple[float, ...] = (
    -8.0,
    -6.0,
    -4.0,
    -3.0,
    -2.5,
    -2.0,
    -1.5,
    -1.0,
    -0.5,
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    4.0,
    6.0,
    8.0,
)
CONTROL_LEDGER: tuple[dict[str, str], ...] = (
    {
        "control": "autoregressive_affect_history",
        "model_key": "ar",
        "layer": "event_masks,timing_grid",
        "purpose": "Tests whether cortical/TRIBE adds value beyond current and recent arousal history plus time features.",
    },
    {
        "control": "shuffled_cortical_rows",
        "model_key": "shuffled",
        "layer": "event_masks,timing_grid",
        "purpose": "Destroys row-level cortical/stimulus identity while preserving feature scale and width.",
    },
    {
        "control": "split_local_shuffled_cortical_rows",
        "model_key": "shuffled",
        "layer": "event_masks,timing_grid",
        "purpose": "Shuffles train and test feature rows inside the already-created split instead of before splitting.",
    },
    {
        "control": "random_gaussian_features",
        "model_key": "random",
        "layer": "event_masks,timing_grid",
        "purpose": "Checks whether lift comes from adding high-dimensional nuisance features.",
    },
    {
        "control": "split_local_random_gaussian_features",
        "model_key": "random",
        "layer": "event_masks,timing_grid",
        "purpose": "Generates Gaussian controls at the same split-local feature shape as real cortical inputs.",
    },
    {
        "control": "majority_classifier",
        "model_key": "majority",
        "layer": "event_masks,balanced_event_vs_stable",
        "purpose": "Checks event metrics against train-prevalence-only predictions.",
    },
    {
        "control": "timestamp_only",
        "model_key": "timestamp,timestamp_only",
        "layer": "blocked_primary_event_controls,timing_grid",
        "purpose": "Checks whether absolute/normalized time explains the result.",
    },
    {
        "control": "video_id_time_only",
        "model_key": "video_time,video_id_time_only",
        "layer": "blocked_primary_event_controls,timing_grid",
        "purpose": "Checks whether video identity plus within-video time explains the result.",
    },
    {
        "control": "label_shuffle_across_videos",
        "model_key": "label_shuffle_across_videos",
        "layer": "blocked_primary_event_controls",
        "purpose": "Breaks label-feature relation globally while preserving the feature matrix.",
    },
    {
        "control": "label_shuffle_within_video",
        "model_key": "label_shuffle_within_video",
        "layer": "blocked_primary_event_controls",
        "purpose": "Breaks within-video label timing while preserving per-video label distribution.",
    },
    {
        "control": "feature_shuffle_across_videos",
        "model_key": "feature_shuffle_across_videos",
        "layer": "blocked_primary_event_controls",
        "purpose": "Breaks cortical row identity globally while preserving labels and AR features.",
    },
    {
        "control": "feature_shuffle_within_video",
        "model_key": "feature_shuffle_within_video",
        "layer": "blocked_primary_event_controls",
        "purpose": "Breaks cortical timing inside each video while preserving per-video feature distribution.",
    },
    {
        "control": "blocked_temporal_gap_holdout",
        "model_key": "blocked",
        "layer": "split_contract",
        "purpose": "Leaves out the blocked temporal-gap test rows defined by the VEATIC manifest.",
    },
    {
        "control": "official_70_30_holdout",
        "model_key": "official",
        "layer": "split_contract",
        "purpose": "Keeps the manifest official 70/30 split as a second fixed-split check.",
    },
    {
        "control": "grouped_video_k_fold_holdout",
        "model_key": "grouped_",
        "layer": "split_contract",
        "purpose": "Leaves whole videos out of training so train/test rows never share a video.",
    },
    {
        "control": "zero_change_baseline",
        "model_key": "zero",
        "layer": "continuous_diagnostics",
        "purpose": "Keeps continuous future-change metrics diagnostic when stable frames dominate.",
    },
    {
        "control": "single_backend_policy",
        "model_key": "pca_backend,ridge_backend",
        "layer": "run_contract",
        "purpose": "Avoids mixing CPU/MPS thresholded results without a separate device consistency audit.",
    },
)


def default_cache_dir() -> Path:
    root = os.environ.get("NEURAL_BRIDGE_EXTERNAL_ROOT")
    if root:
        return Path(root).expanduser() / "benchmarks" / "veatic" / "tribe_cache"
    return ROOT / "external_assets" / "benchmarks" / "veatic" / "tribe_cache"


def contract_manifest(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": "veatic_strict_contract_v1",
        "claim_scope": "VEATIC-124 cortical/TRIBE arousal event and spike ranking",
        "manifest": str(Path(args.manifest).expanduser()),
        "report": str(Path(args.report).expanduser()),
        "cache_dir": str(Path(args.cache_dir).expanduser()),
        "feature_modes": [item[0] for item in selected_features(args)],
        "targets": [
            {"target": target, "horizon_seconds": horizon, "threshold": threshold, "tier": tier}
            for target, horizon, threshold, tier in selected_targets(args)
        ],
        "splits": ["blocked_temporal_gap", "official_70_30", f"grouped_video_{args.grouped_folds}_fold"],
        "controls": [row["control"] for row in CONTROL_LEDGER],
        "control_ledger": list(CONTROL_LEDGER),
        "event_masks": [
            "all_frames",
            "stable_negative_only",
            "event_only",
            "pre_event_1s",
            "pre_event_2s",
            "pre_event_3s",
            "pre_event_5s",
            "event_plus_pre_3s",
            "balanced_event_vs_stable",
        ],
        "timing_policy": {
            "primary_alignment": "current_0s",
            "offset_grid": list(FULL_OFFSETS if args.full_offsets else CORE_OFFSETS),
            "offset_grid_usage": "diagnostic_only",
            "causal_windows": [item[0] for item in CAUSAL_WINDOWS],
            "future_or_symmetric_windows": (
                [item[0] for item in DIAGNOSTIC_WINDOWS] if args.include_diagnostic_windows else []
            ),
            "future_or_symmetric_usage": "diagnostic_only",
        },
        "leakage_rules": [
            "PCA and transforms fit on train rows only for scored splits/folds.",
            "Decision thresholds selected on train predictions only.",
            "Grouped-video folds keep train and test videos disjoint.",
            "Positive-only event/pre-event masks are recall/top-k diagnostics, not PR-AUC claims.",
            "Balanced event-vs-stable rows carry event-conditioned PR-AUC claims.",
            "Offset and future-inclusive context grids are diagnostics unless a future train-only policy survives controls.",
        ],
        "backend_policy": {
            "pca_backend": args.pca_backend,
            "ridge_backend": args.ridge_backend,
            "seed": args.seed,
            "device_consistency_rule": (
                "Final reports should use one backend policy. CPU/MPS mixing requires a separate "
                "score, threshold, and threshold-flip audit before promotion."
            ),
        },
    }


def selected_features(args: argparse.Namespace) -> tuple[tuple[str, str, str], ...]:
    if args.primary_only:
        return tuple(item for item in FEATURE_MODES if item[2] == "primary")
    return FEATURE_MODES


def selected_targets(args: argparse.Namespace) -> tuple[tuple[str, int | None, float, str], ...]:
    if args.primary_only:
        return tuple(item for item in TARGETS if item[3] == "primary")
    return TARGETS


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fieldnames:
            handle.write("\n")
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finite(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def diff(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            return "NA"
        return f"{float(value):.4f}"
    return str(value)


def target_rows(
    ctx: retest.RetestContext,
    rows: list[dict[str, Any]],
    target: str,
    horizon: int | None,
    threshold: float,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray | None]:
    if target == "arousal__future_spike_1_3s":
        selected, y = retest.future_spike_rows(ctx, rows, threshold)
        return selected, y.astype(np.float64), None
    if target.startswith("arousal__future_change_p") and target.endswith("_movement"):
        if horizon is None:
            raise ValueError(f"horizon required for {target}")
        selected, continuous = retest.future_change_rows(ctx, rows, horizon)
        y = (np.abs(continuous) >= threshold).astype(np.float64)
        return selected, y, continuous.astype(np.float64)
    raise ValueError(f"Unsupported strict-suite target: {target}")


def split_specs(
    ctx: retest.RetestContext,
    grouped_folds: int,
) -> list[tuple[str, list[str], list[dict[str, Any]], list[dict[str, Any]]]]:
    specs: list[tuple[str, list[str], list[dict[str, Any]], list[dict[str, Any]]]] = []
    for label, split_name in (("blocked", "blocked_temporal_gap"), ("official", "official_70_30")):
        train_rows, test_rows, _gap = retest.fixed_rows(ctx.accepted_rows, split_name)
        specs.append((label, [], train_rows, test_rows))
    for fold_label, held, train_rows, test_rows in fairness.grouped_video_folds(ctx.accepted_rows, grouped_folds):
        specs.append((fold_label, held, train_rows, test_rows))
    return specs


def add_event_layer(
    *,
    ctx: retest.RetestContext,
    split_label: str,
    feature_label: str,
    target: str,
    horizon: int | None,
    threshold: float,
    target_tier: str,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    matrices: dict[str, Any],
    rng: np.random.Generator,
    event_rows: list[dict[str, Any]],
    balanced_rows: list[dict[str, Any]],
    per_video_rows: list[dict[str, Any]],
    continuous_rows: list[dict[str, Any]],
) -> None:
    train_selected, train_y, _train_continuous = target_rows(ctx, train_rows, target, horizon, threshold)
    test_selected, test_y, test_continuous = target_rows(ctx, test_rows, target, horizon, threshold)
    include_anti_leakage = split_label == "blocked" and target_tier == "primary"
    scores = conditioned.model_score_sets(
        ctx,
        train_selected,
        train_y,
        test_selected,
        test_y,
        matrices,
        rng,
        include_anti_leakage=include_anti_leakage,
    )
    if scores is None:
        return
    stable_magnitude = conditioned.event_magnitude_for_rows(
        ctx,
        scores["test_rows"],
        target,
        horizon,
    )
    masks = conditioned.build_region_masks(
        scores["test_rows"],
        scores["true_test_y"].astype(np.int64),
        stable_magnitude,
        threshold,
    )
    base = {
        "split": split_label,
        "feature_mode": feature_label,
        "target": target,
        "threshold": threshold,
        "target_tier": target_tier,
        "layer": "event_masks",
    }
    conditioned.evaluate_event_masks(event_rows, base, scores, masks)
    conditioned.evaluate_balanced_sampling(balanced_rows, base, scores, masks)
    if test_continuous is not None:
        continuous_base = {**base, "layer": "continuous_diagnostics"}
        conditioned.evaluate_continuous_masks(continuous_rows, continuous_base, scores, test_continuous, masks)
    if split_label == "blocked" and target_tier == "primary":
        conditioned.per_video_rows(per_video_rows, base, scores, masks)


def add_timing_layer(
    *,
    ctx: retest.RetestContext,
    split_label: str,
    held: list[str],
    feature_label: str,
    target: str,
    horizon: int | None,
    threshold: float,
    target_tier: str,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    matrices: dict[str, Any],
    offsets: tuple[float, ...],
    include_diagnostic_windows: bool,
    timing_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
) -> None:
    train_selected, train_y, _ = target_rows(ctx, train_rows, target, horizon, threshold)
    test_selected, test_y, _ = target_rows(ctx, test_rows, target, horizon, threshold)
    train_table = fairness.subset_table(train_rows, matrices["train_matrix"])
    test_table = fairness.subset_table(test_rows, matrices["test_matrix"])
    for window_name, start, end, aggregation in CAUSAL_WINDOWS:
        row, scores = fairness.evaluate_context(
            ctx,
            split_label,
            feature_label,
            target,
            horizon,
            threshold,
            train_selected,
            train_y,
            test_selected,
            test_y,
            train_table,
            test_table,
            window_name,
            start,
            end,
            "causal",
            aggregation,
        )
        if row is not None:
            row["target_tier"] = target_tier
            row["held_out_video_ids"] = ",".join(held)
            timing_rows.append(row)
        if scores is not None and window_name == "current_only_0s" and target_tier == "primary":
            score_rows.extend(fairness.score_records(row or {}, scores))
    if include_diagnostic_windows:
        for window_name, start, end, aggregation in DIAGNOSTIC_WINDOWS:
            row, _scores = fairness.evaluate_context(
                ctx,
                split_label,
                feature_label,
                target,
                horizon,
                threshold,
                train_selected,
                train_y,
                test_selected,
                test_y,
                train_table,
                test_table,
                window_name,
                start,
                end,
                "non_causal_diagnostic",
                aggregation,
            )
            if row is not None:
                row["target_tier"] = target_tier
                row["held_out_video_ids"] = ",".join(held)
                timing_rows.append(row)
    for offset in offsets:
        row, _scores = fairness.evaluate_offset(
            ctx,
            split_label,
            feature_label,
            target,
            horizon,
            threshold,
            train_selected,
            train_y,
            test_selected,
            test_y,
            train_table,
            test_table,
            offset,
            "offset_grid_diagnostic" if offset else "current_0s_offset_baseline",
        )
        if row is not None:
            row["target_tier"] = target_tier
            row["held_out_video_ids"] = ",".join(held)
            timing_rows.append(row)


def summarize_model_deltas(event_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    keys = ("split", "feature_mode", "target", "threshold", "mask", "task_type")
    for row in event_rows:
        grouped[tuple(row.get(key) for key in keys)][str(row["model"])] = row
    out = []
    for key, models in grouped.items():
        real = models.get("real")
        if not real:
            continue
        record = {name: key[index] for index, name in enumerate(keys)}
        for metric in ("pr_auc", "f1", "recall", "top_10pct_recall"):
            record[f"real_{metric}"] = real.get(metric)
            for control in ("ar", "shuffled", "random"):
                control_value = models.get(control, {}).get(metric)
                record[f"{control}_{metric}"] = control_value
                record[f"real_vs_{control}_{metric}"] = diff(real.get(metric), control_value)
        out.append(record)
    return out


def summarize_control_coverage(
    event_rows: list[dict[str, Any]],
    balanced_rows: list[dict[str, Any]],
    timing_rows: list[dict[str, Any]],
    continuous_rows: list[dict[str, Any]],
    splits: list[str],
) -> list[dict[str, Any]]:
    event_models = {str(row.get("model")) for row in event_rows}
    balanced_models = {str(row.get("model")) for row in balanced_rows}
    continuous_models = {str(row.get("model")) for row in continuous_rows}
    timing_metrics = set()
    for row in timing_rows:
        for key in row:
            if key.endswith("_pr_auc"):
                timing_metrics.add(key.removesuffix("_pr_auc"))
    output = []
    for item in CONTROL_LEDGER:
        keys = [key.strip() for key in item["model_key"].split(",")]
        present_event = any(key in event_models for key in keys)
        present_balanced = any(key in balanced_models for key in keys)
        present_continuous = any(key in continuous_models for key in keys)
        present_timing = any(key in timing_metrics for key in keys)
        present_split = item["layer"] == "split_contract" and any(
            split == item["model_key"] or split.startswith(item["model_key"]) for split in splits
        )
        present_contract = item["layer"] == "run_contract"
        output.append(
            {
                **item,
                "present_event_masks": present_event,
                "present_balanced_sampling": present_balanced,
                "present_continuous_diagnostics": present_continuous,
                "present_timing_grid": present_timing,
                "present_split_contract": present_split,
                "present_contract_only": present_contract,
                "covered": present_event or present_balanced or present_continuous or present_timing or present_split or present_contract,
            }
        )
    return output


def summarize_timing(timing_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in timing_rows:
        if row.get("split") == "blocked" and row.get("target_tier") == "primary":
            grouped[(row["feature_mode"], row["target"], float(row["threshold"]))].append(row)
    out = []
    for (feature, target, threshold), rows in sorted(grouped.items()):
        current = [
            row
            for row in rows
            if row.get("arm") in {"causal_context_window", "current_0s_offset_baseline"}
            and float(row.get("offset_seconds", 999.0)) == 0.0
            and row.get("window_name") in {None, "current_only_0s"}
        ]
        current_best = max(current, key=lambda row: row.get("real_pr_auc") or -1.0) if current else None
        causal = [
            row
            for row in rows
            if row.get("arm") == "causal_context_window" and row.get("window_name") != "current_only_0s"
        ]
        offsets = [row for row in rows if row.get("arm") == "offset_grid_diagnostic"]
        best_causal = max(causal, key=lambda row: row.get("real_pr_auc") or -1.0) if causal else None
        best_offset = max(offsets, key=lambda row: row.get("real_pr_auc") or -1.0) if offsets else None
        out.append(
            {
                "feature_mode": feature,
                "target": target,
                "threshold": threshold,
                "current_0s_real_pr_auc": current_best.get("real_pr_auc") if current_best else None,
                "best_causal_window": best_causal.get("window_name") if best_causal else None,
                "best_causal_real_pr_auc": best_causal.get("real_pr_auc") if best_causal else None,
                "best_causal_gain_vs_0s": diff(
                    best_causal.get("real_pr_auc") if best_causal else None,
                    current_best.get("real_pr_auc") if current_best else None,
                ),
                "best_offset_seconds": best_offset.get("offset_seconds") if best_offset else None,
                "best_offset_real_pr_auc": best_offset.get("real_pr_auc") if best_offset else None,
                "best_offset_gain_vs_0s": diff(
                    best_offset.get("real_pr_auc") if best_offset else None,
                    current_best.get("real_pr_auc") if current_best else None,
                ),
                "offset_usage": "diagnostic_only",
            }
        )
    return out


def write_report(
    path: Path,
    summary: dict[str, Any],
    event_delta_rows: list[dict[str, Any]],
    balanced_rows: list[dict[str, Any]],
    timing_summary: list[dict[str, Any]],
    control_coverage: list[dict[str, Any]],
    outputs: dict[str, Path],
) -> None:
    blocked_focus = [
        row
        for row in event_delta_rows
        if row.get("split") == "blocked"
        and row.get("mask") == "all_frames"
        and row.get("target") == "arousal__future_spike_1_3s"
    ]
    blocked_focus.sort(key=lambda row: row.get("real_vs_ar_pr_auc") or -1.0, reverse=True)
    balanced_focus = [
        row
        for row in balanced_rows
        if row.get("split") == "blocked"
        and row.get("mask") == "balanced_event_vs_stable"
        and row.get("model") == "real"
    ]
    balanced_focus.sort(key=lambda row: row.get("pr_auc_mean") or -1.0, reverse=True)
    lines = [
        "# VEATIC-124 Strict Benchmark Suite",
        "",
        "## Verdict",
        "",
        summary["executive_summary"],
        "",
        "## Contract",
        "",
        "- Primary alignment: current 0s.",
        "- Offset grids and future/symmetric windows are diagnostics only.",
        "- Thresholds, PCA, and transforms are train-only.",
        "- Grouped-video folds keep videos disjoint.",
        "- Balanced event-vs-stable rows carry event-conditioned PR-AUC claims.",
        "",
        "## Controls",
        "",
        "| Control | Event masks | Balanced | Continuous | Timing | Split | Purpose |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in control_coverage:
        lines.append(
            f"| `{row['control']}` | {row['present_event_masks']} | "
            f"{row['present_balanced_sampling']} | {row['present_continuous_diagnostics']} | "
            f"{row['present_timing_grid']} | "
            f"{row['present_split_contract']} | {row['purpose']} |"
        )
    lines.extend(
        [
            "",
            "## Strongest Blocked Spike Rows",
            "",
            "| Feature | Thr | Real PR-AUC | vs AR | vs shuffled | vs random |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in blocked_focus[:12]:
        lines.append(
            f"| {row['feature_mode']} | {fmt(row['threshold'])} | {fmt(row.get('real_pr_auc'))} | "
            f"{fmt(row.get('real_vs_ar_pr_auc'))} | {fmt(row.get('real_vs_shuffled_pr_auc'))} | "
            f"{fmt(row.get('real_vs_random_pr_auc'))} |"
        )
    lines.extend(
        [
            "",
            "## Balanced Event-vs-Stable",
            "",
            "| Feature | Target | Thr | Ratio | PR-AUC mean | F1 mean | Recall mean |",
            "|---|---|---:|---|---:|---:|---:|",
        ]
    )
    for row in balanced_focus[:16]:
        lines.append(
            f"| {row['feature_mode']} | `{row['target']}` | {fmt(row['threshold'])} | {row['negative_ratio']} | "
            f"{fmt(row.get('pr_auc_mean'))} | {fmt(row.get('f1_mean'))} | {fmt(row.get('recall_mean'))} |"
        )
    lines.extend(
        [
            "",
            "## Timing Diagnostics",
            "",
            "| Feature | Target | Thr | 0s PR-AUC | Best causal | Causal gain | Best offset | Offset gain |",
            "|---|---|---:|---:|---|---:|---:|---:|",
        ]
    )
    for row in timing_summary:
        lines.append(
            f"| {row['feature_mode']} | `{row['target']}` | {fmt(row['threshold'])} | "
            f"{fmt(row.get('current_0s_real_pr_auc'))} | {row.get('best_causal_window') or 'NA'} | "
            f"{fmt(row.get('best_causal_gain_vs_0s'))} | {fmt(row.get('best_offset_seconds'))} | "
            f"{fmt(row.get('best_offset_gain_vs_0s'))} |"
        )
    lines.extend(["", "## Output Files", ""])
    for label, output in outputs.items():
        lines.append(f"- {label}: `{output}`")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    start = time.monotonic()
    bench.PCA_BACKEND = args.pca_backend
    bench.RIDGE_BACKEND = args.ridge_backend
    ctx = retest.RetestContext(
        Path(args.manifest).expanduser().resolve(),
        Path(args.report).expanduser().resolve(),
        Path(args.cache_dir).expanduser().resolve(),
    )
    splits = split_specs(ctx, args.grouped_folds)
    offsets = FULL_OFFSETS if args.full_offsets else CORE_OFFSETS
    rng = np.random.default_rng(args.seed)
    event_rows: list[dict[str, Any]] = []
    balanced_rows: list[dict[str, Any]] = []
    per_video_rows: list[dict[str, Any]] = []
    continuous_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    split_metadata: list[dict[str, Any]] = []

    for feature_label, feature_mode, _feature_tier in selected_features(args):
        base_feature_sets = ctx.base_feature_sets(feature_mode)
        for split_label, held, train_rows, test_rows in splits:
            matrices = retest.split_matrices(ctx, base_feature_sets, train_rows, test_rows, feature_mode)
            split_metadata.append(
                {
                    "feature_mode": feature_label,
                    "split": split_label,
                    "held_out_video_ids": ",".join(held),
                    "train_rows": len(train_rows),
                    "test_rows": len(test_rows),
                    "matrix_feature_key": matrices["feature_key"],
                    "matrix_width": int(matrices["train_matrix"].shape[1]),
                    "transform_metadata": matrices["metadata"],
                }
            )
            for target, horizon, threshold, target_tier in selected_targets(args):
                seed = rng.integers(0, 2**32 - 1)
                local_rng = np.random.default_rng(int(seed))
                add_event_layer(
                    ctx=ctx,
                    split_label=split_label,
                    feature_label=feature_label,
                    target=target,
                    horizon=horizon,
                    threshold=threshold,
                    target_tier=target_tier,
                    train_rows=train_rows,
                    test_rows=test_rows,
                    matrices=matrices,
                    rng=local_rng,
                    event_rows=event_rows,
                    balanced_rows=balanced_rows,
                    per_video_rows=per_video_rows,
                    continuous_rows=continuous_rows,
                )
                add_timing_layer(
                    ctx=ctx,
                    split_label=split_label,
                    held=held,
                    feature_label=feature_label,
                    target=target,
                    horizon=horizon,
                    threshold=threshold,
                    target_tier=target_tier,
                    train_rows=train_rows,
                    test_rows=test_rows,
                    matrices=matrices,
                    offsets=offsets,
                    include_diagnostic_windows=args.include_diagnostic_windows,
                    timing_rows=timing_rows,
                    score_rows=score_rows,
                )
            print(f"[INFO] finished feature={feature_label} split={split_label}", flush=True)

    event_delta_rows = summarize_model_deltas(event_rows)
    timing_summary = summarize_timing(timing_rows)
    split_labels = [item[0] for item in splits]
    control_coverage = summarize_control_coverage(event_rows, balanced_rows, timing_rows, continuous_rows, split_labels)
    summary = {
        "schema_version": "veatic_strict_benchmark_v1",
        "elapsed_seconds": time.monotonic() - start,
        "accepted_videos": len({str(row["video_id"]) for row in ctx.accepted_rows}),
        "accepted_rows": len(ctx.accepted_rows),
        "feature_modes": [item[0] for item in selected_features(args)],
        "splits": split_labels,
        "event_rows": len(event_rows),
        "balanced_rows": len(balanced_rows),
        "per_video_rows": len(per_video_rows),
        "continuous_rows": len(continuous_rows),
        "timing_rows": len(timing_rows),
        "score_rows": len(score_rows),
        "control_coverage": control_coverage,
        "executive_summary": (
            "Unified strict run completed with fixed splits, grouped-video folds, train-only "
            "thresholding/PCA, explicit real-vs-control comparisons, anti-leakage shuffles, "
            "balanced event-vs-stable rows, causal context checks, and diagnostic-only offset grids."
        ),
        "contract": contract_manifest(args),
    }
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "summary_json": out_dir / "veatic_strict_summary.json",
        "contract_json": out_dir / "veatic_strict_contract.json",
        "event_masks_csv": out_dir / "event_masks.csv",
        "event_deltas_csv": out_dir / "event_model_deltas.csv",
        "balanced_sampling_csv": out_dir / "balanced_event_vs_stable.csv",
        "per_video_csv": out_dir / "per_video_event_rows.csv",
        "continuous_diagnostics_csv": out_dir / "continuous_diagnostics.csv",
        "timing_grid_csv": out_dir / "timing_context_offset_grid.csv",
        "timing_summary_csv": out_dir / "timing_summary.csv",
        "control_coverage_csv": out_dir / "control_coverage.csv",
        "split_metadata_json": out_dir / "split_transform_metadata.json",
        "score_records_csv": out_dir / "score_records_primary_current_0s.csv",
        "report_md": out_dir / "veatic_strict_benchmark_report.md",
    }
    outputs["summary_json"].write_text(json.dumps(bench.json_safe(summary), indent=2), encoding="utf-8")
    outputs["contract_json"].write_text(json.dumps(bench.json_safe(summary["contract"]), indent=2), encoding="utf-8")
    outputs["split_metadata_json"].write_text(json.dumps(bench.json_safe(split_metadata), indent=2), encoding="utf-8")
    write_csv(outputs["event_masks_csv"], event_rows)
    write_csv(outputs["event_deltas_csv"], event_delta_rows)
    write_csv(outputs["balanced_sampling_csv"], balanced_rows)
    write_csv(outputs["per_video_csv"], per_video_rows)
    write_csv(outputs["continuous_diagnostics_csv"], continuous_rows)
    write_csv(outputs["timing_grid_csv"], timing_rows)
    write_csv(outputs["timing_summary_csv"], timing_summary)
    write_csv(outputs["control_coverage_csv"], control_coverage)
    write_csv(outputs["score_records_csv"], score_rows)
    write_report(outputs["report_md"], summary, event_delta_rows, balanced_rows, timing_summary, control_coverage, outputs)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the unified VEATIC-124 strict benchmark suite.")
    parser.add_argument("--manifest", default="benchmarks/veatic/veatic_manifest_124_complete_20260616.jsonl")
    parser.add_argument("--report", default="benchmarks/veatic/veatic_manifest_124_complete_20260616.report.json")
    parser.add_argument("--cache-dir", default=str(default_cache_dir()))
    parser.add_argument(
        "--output-dir",
        default=f"outputs/veatic_124_strict_benchmark_{datetime.now().strftime('%Y%m%d_%H%M')}",
    )
    parser.add_argument("--grouped-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--pca-backend", default="mps_gram", choices=("auto", "mps_power", "mps_gram", "cpu_svd"))
    parser.add_argument("--ridge-backend", default="cpu_pinv", choices=("auto", "mps_solve", "cpu_pinv"))
    parser.add_argument("--primary-only", action="store_true", help="Run only primary feature/target rows.")
    parser.add_argument("--full-offsets", action="store_true", help="Use the wider offset grid.")
    parser.add_argument(
        "--include-diagnostic-windows",
        action="store_true",
        help="Include future/symmetric diagnostic windows. These are never final-score rows.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the strict contract without loading cache files.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps(contract_manifest(args), indent=2))
        return
    run_suite(args)


if __name__ == "__main__":
    main()
