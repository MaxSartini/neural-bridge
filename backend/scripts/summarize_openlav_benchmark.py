"""Create a compact Markdown diagnostic report from an OpenLAV benchmark JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRIC_KEYS = ("mae", "rmse", "pearson", "spearman")


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def delta(condition: dict[str, Any]) -> Any:
    return condition.get("paired_mae_delta_vs_mean_baseline", {}).get("mean")


def metric(condition: dict[str, Any], key: str) -> Any:
    if not condition:
        return None
    value = condition.get(key)
    if isinstance(value, dict):
        return value.get("mean")
    return value


def ci(condition: dict[str, Any], key: str) -> str:
    interval = condition.get(f"{key}_bootstrap_95_ci")
    if not interval:
        return "n/a"
    return f"[{fmt(interval[0])}, {fmt(interval[1])}]"


def metric_row(name: str, condition: dict[str, Any], mean_baseline: dict[str, Any], semantic_baseline: dict[str, Any] | None) -> str:
    if "skipped" in condition:
        return f"| {name} | skipped | {condition['skipped']} | | | | | | | | |\n"
    semantic_delta = "n/a"
    condition_mae = metric(condition, "mae")
    semantic_mae = metric(semantic_baseline, "mae") if semantic_baseline else None
    if semantic_mae is not None and condition_mae is not None:
        semantic_delta = fmt(condition_mae - semantic_mae)
    return (
        f"| {name} | {fmt(condition_mae)} | {fmt(metric(condition, 'rmse'))} | "
        f"{fmt(metric(condition, 'pearson'))} | {fmt(metric(condition, 'spearman'))} | "
        f"{fmt(delta(condition))} | {semantic_delta} | {ci(condition, 'mae')} | "
        f"{ci(condition, 'pearson')} | {ci(condition, 'spearman')} |\n"
    )


def top_importance(section: dict[str, Any], condition: str, kind: str) -> str:
    report = section.get(condition)
    if not report or "skipped" in report:
        return f"- `{condition}` {kind}: skipped\n"
    dominance = report.get("missingness_or_duration_dominates")
    lines = [
        f"- `{condition}` {kind}: missingness/duration dominates = {fmt(dominance)}; "
        f"neuro={fmt(report.get('neuro_positive_importance', report.get('neuro_importance')))}, "
        f"nuisance={fmt(report.get('nuisance_positive_importance', report.get('nuisance_importance')))}"
    ]
    for item in report.get("top_features", [])[:8]:
        value = item.get("mean_mae_increase_when_permuted", item.get("mean_catboost_importance"))
        lines.append(f"  - {item.get('feature')} ({item.get('family')}): {fmt(value)}")
    return "\n".join(lines) + "\n"


def append_feature_counts(lines: list[str], report: dict[str, Any]) -> None:
    lines.append("## Feature Sets\n\n")
    lines.append("| Condition | Count | Feature/row ratio | Rationale |\n")
    lines.append("|---|---:|---:|---|\n")
    for name, info in report.get("feature_sets", {}).items():
        lines.append(
            f"| {name} | {fmt(info.get('count'))} | {fmt(info.get('feature_to_row_ratio'))} | "
            f"{info.get('rationale', '')} |\n"
        )
    lines.append("\n")


def append_reading_guide(lines: list[str], report: dict[str, Any]) -> None:
    targets = ", ".join(sorted(report.get("targets", {}).keys())) or "arousal and valence"
    lines.append("## How To Read This Benchmark\n\n")
    lines.append(
        "- Ground truth: OpenLAV provides observed human affect ratings for each video. "
        "The benchmark is not trying to beat OpenLAV; it is trying to predict those human labels from stimulus-derived features.\n"
    )
    lines.append(
        f"- Targets predicted: `{targets}`. In this current video-level benchmark, targets are full-video aggregate human ratings, "
        "not segment-level labels and not per-participant rows.\n"
    )
    lines.append(
        "- Label scale: OpenLAV factor scores are normalized by this pipeline with the recorded dataset min/max bounds, "
        "so model predictions and labels are compared on the same reproducible numeric scale.\n"
    )
    lines.append(
        "- MAE: mean absolute error. Lower is better. It is the average distance between the model's predicted rating "
        "and the OpenLAV human aggregate rating.\n"
    )
    lines.append(
        "- Pearson/Spearman: correlation metrics. Higher is better. Pearson measures linear tracking of the human labels; "
        "Spearman measures whether the model ranks videos similarly to humans even if the scale is imperfect.\n"
    )
    lines.append(
        "- Baselines: baselines are alternative prediction methods, not the dataset. The mean baseline predicts the training-fold average. "
        "Video/container and handcrafted non-neuro baselines use simpler non-neural stimulus metadata. Neuro conditions use TRIBE-derived cortical/subcortical features.\n"
    )
    lines.append(
        "- Success ladder: weak success means beating the mean baseline; meaningful success means beating strong non-neuro baselines; "
        "strong success means neuro improves additive or residualized tests beyond non-neuro-only predictions; invalid neuro lift means shuffled neuro performs similarly to real neuro.\n"
    )
    lines.append(
        "- Residualized tests matter because they ask whether neuro features explain what is left after simpler media/source confounds "
        "such as duration, bitrate, frame count, resolution, compression, and extraction quality have already been used.\n"
    )
    lines.append("\n")


def append_target(lines: list[str], axis: str, target: dict[str, Any], sanity: dict[str, Any], effects: dict[str, Any]) -> None:
    lines.append(f"## Target: {axis}\n\n")
    semantic_baseline = target.get("semantic_baseline")
    mean_baseline = target["mean_baseline"]
    headline = [
        "mean_baseline",
        "compact_global_quality",
        "cortical_only",
        "subcortical_only",
        "cortical_plus_subcortical_calibrated",
        "compact_cortical_salience",
        "compact_subcortical_affective",
        "compact_neuro_affect",
        "ultra_compact_neuro",
        "neutral_neuro",
        "shuffled_cortical",
        "shuffled_subcortical",
        "shuffled_labels",
    ]
    lines.append("| Condition | MAE | RMSE | Pearson | Spearman | Delta vs mean MAE | Delta vs semantic MAE | MAE 95% CI | Pearson 95% CI | Spearman 95% CI |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|---|---|\n")
    for name in headline:
        if name in target:
            lines.append(metric_row(name, target[name], mean_baseline, semantic_baseline))
    for group_name in ("strict_split_local_controls", "cortical_subcortical_component_controls"):
        for name, condition in target.get(group_name, {}).items():
            lines.append(metric_row(name, condition, mean_baseline, semantic_baseline))
    lines.append("\n")

    if sanity:
        lines.append("### Control Sanity\n\n")
        lines.append("| Control | Status | Delta vs mean MAE | Expected |\n")
        lines.append("|---|---|---:|---|\n")
        for name, item in sanity.items():
            lines.append(
                f"| {name} | {item.get('status', 'unknown')} | "
                f"{fmt(item.get('paired_mae_delta_vs_mean_baseline'))} | {item.get('expected', item.get('reason', ''))} |\n"
            )
        lines.append("\n")

    if effects:
        lines.append("### Real Feature Effects\n\n")
        lines.append("| Condition | Status | Delta vs mean MAE |\n")
        lines.append("|---|---|---:|\n")
        for name, item in effects.items():
            lines.append(
                f"| {name} | {item.get('status', 'unknown')} | "
                f"{fmt(item.get('paired_mae_delta_vs_mean_baseline'))} |\n"
            )
        lines.append("\n")

    if target.get("regularized_linear_controls"):
        lines.append("### Regularized Linear Guardrail\n\n")
        lines.append("| Condition | MAE | Pearson | Delta vs mean MAE |\n")
        lines.append("|---|---:|---:|---:|\n")
        for name, condition in target["regularized_linear_controls"].items():
            if "skipped" in condition:
                lines.append(f"| {name} | skipped | | |\n")
            else:
                lines.append(
                    f"| {name} | {fmt(condition.get('mae'))} | "
                    f"{fmt(condition.get('pearson'))} | {fmt(delta(condition))} |\n"
                )
        lines.append("\n")

    if target.get("permutation_importance") or target.get("feature_importance"):
        lines.append("### Importance Diagnostics\n\n")
        for condition in (
            "ultra_compact_neuro",
            "compact_neuro_affect",
            "compact_subcortical_affective",
            "compact_global_quality",
            "cortical_plus_subcortical_calibrated",
        ):
            lines.append(top_importance(target.get("permutation_importance", {}), condition, "permutation"))
            lines.append(top_importance(target.get("feature_importance", {}), condition, "catboost"))
        lines.append("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark")
    parser.add_argument("--manifest-report")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark).expanduser().resolve()
    report = json.loads(benchmark_path.read_text(encoding="utf-8"))
    manifest_report = None
    if args.manifest_report:
        manifest_report = json.loads(Path(args.manifest_report).expanduser().read_text(encoding="utf-8"))

    lines: list[str] = []
    lines.append(f"# OpenLAV Benchmark Diagnostic Report\n\n")
    lines.append(f"- Benchmark: `{benchmark_path}`\n")
    lines.append(f"- Schema: `{report.get('schema_version')}`\n")
    lines.append(f"- Rows: {fmt(report.get('rows'))}\n")
    lines.append(f"- Groups: {fmt(report.get('groups'))}\n")
    if manifest_report:
        lines.append(f"- Accepted rows: {fmt(manifest_report.get('rows'))}\n")
        lines.append(f"- Missing/rejected first-N rows: {fmt(len(manifest_report.get('missing', [])))}\n")
    lines.append(f"- Split: {report.get('split', {})}\n")
    lines.append("\n")

    lines.append("## Contract And Leakage\n\n")
    lines.append(f"- Contract audit: `{json.dumps(report.get('contract_audit', {}), sort_keys=True)}`\n")
    leakage = report.get("leakage_audit", {})
    overlaps = leakage.get("group_overlap_by_split", [])
    has_group_overlap = any(len(x) > 0 for x in overlaps)
    lines.append(f"- Train/test group overlap detected: {fmt(has_group_overlap)}\n")
    lines.append(f"- Duplicate feature vectors: {fmt(leakage.get('duplicate_feature_vectors'))}\n")
    lines.append(f"- Unavailable conditions: `{json.dumps(report.get('unavailable_conditions', {}), sort_keys=True)}`\n")
    lines.append("\n")

    append_reading_guide(lines, report)
    append_feature_counts(lines, report)
    for axis, target in report.get("targets", {}).items():
        append_target(
            lines,
            axis,
            target,
            report.get("sanity_checks", {}).get(axis, {}),
            report.get("component_effects", {}).get(axis, {}),
        )

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"summary": str(output), "schema": report.get("schema_version")}))


if __name__ == "__main__":
    main()
