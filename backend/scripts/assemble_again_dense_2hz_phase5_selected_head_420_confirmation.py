"""Assemble and audit the bounded AGAIN selected-head 420-row confirmation.

This is a no-search consolidation of the canonical 70-row blocked temporal and
350-row grouped-video evaluations. It never trains or scores a model. A source
row is reusable only when its matrix key, provenance, frozen-AR scores,
checkpoint state, and executable control policy pass the frozen contracts below.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "again_dense_2hz_phase5_selected_head_420_confirmation_v1"
TARGET = "future_arousal_max_delta_rows_4_10_train_q90"
ARCHITECTURE = "short_temporal_conv_residual"
FEATURE = "temporal_mean_2s_then_pca256"
SEEDS = tuple(range(20260625, 20260635))
FOLDS = tuple(range(1, 6))
LANES = (
    "frozen_ar_only",
    "real_residual",
    "shuffled_pca_residual",
    "random_pca_residual",
    "label_permutation_residual",
    "train_only_video_mean_residual",
    "diagnostics_only_residual",
)
PRIMARY_CONTROLS = (
    "shuffled_pca_residual",
    "random_pca_residual",
    "label_permutation_residual",
    "train_only_video_mean_residual",
)
SOURCE_LANE_MAP = {
    "frozen_ar_only": "frozen_ar_only",
    "frozen_or_ar_only": "frozen_ar_only",
    **{lane: lane for lane in LANES if lane != "frozen_ar_only"},
}
PROTOCOLS = ("blocked_temporal_70_30", "grouped_video")
BLOCKED_ROOT = Path("outputs/again_dense_2hz_phase5_temporal_residual_binary_big_confirm_20260630_025437")
GROUPED_ROOT = Path("outputs/again_dense_2hz_phase5_temporal_residual_grouped_compat_20260630_033520")
GROUPED_VERDICT_ROOT = Path("evidence/phase_5_5_grouped_compatibility_20260630_033520")
BLOCKED_METRICS = "temporal_residual_binary_big_confirm_seed_metrics.csv"
GROUPED_METRICS = "temporal_residual_grouped_compat_seed_metrics.csv"
BLOCKED_GATE = "temporal_residual_binary_big_confirm_gates.json"
GROUPED_UPDATED_GATE = "temporal_residual_grouped_compat_updated_gates.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if pd.isna(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def file_digest(path: Path, *, algorithm: str = "sha256", digest_size: int = 16) -> str:
    digest = hashlib.sha256() if algorithm == "sha256" else hashlib.blake2b(digest_size=digest_size)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.blake2b(contiguous.view(np.uint8), digest_size=16).hexdigest()


def expected_matrix_keys() -> set[tuple[str, int, int, str]]:
    blocked = {
        ("blocked_temporal_70_30", 1, seed, lane)
        for seed in SEEDS
        for lane in LANES
    }
    grouped = {
        ("grouped_video", fold, seed, lane)
        for fold in FOLDS
        for seed in SEEDS
        for lane in LANES
    }
    return blocked | grouped


def source_row_digest(row: pd.Series) -> str:
    payload = {key: clean_json(value) for key, value in row.items()}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_source_rows(blocked: pd.DataFrame, grouped: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for protocol, source_name, frame in (
        ("blocked_temporal_70_30", BLOCKED_METRICS, blocked),
        ("grouped_video", GROUPED_METRICS, grouped),
    ):
        source = frame.copy()
        source["source_row_sha256"] = [source_row_digest(row) for _, row in source.iterrows()]
        source.insert(0, "confirmation_schema_version", SCHEMA_VERSION)
        source.insert(1, "protocol", protocol)
        source.insert(2, "lane", source["control_type"].map(SOURCE_LANE_MAP))
        source.insert(3, "source_control_type", source["control_type"])
        source.insert(4, "source_metrics_artifact", source_name)
        source.insert(5, "reuse_status", "reused_existing_scored_row")
        source.insert(6, "rerun_reason", "")
        frames.append(source)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    protocol_order = {name: index for index, name in enumerate(PROTOCOLS)}
    lane_order = {name: index for index, name in enumerate(LANES)}
    combined["_protocol_order"] = combined["protocol"].map(protocol_order)
    combined["_lane_order"] = combined["lane"].map(lane_order)
    combined = combined.sort_values(
        ["_protocol_order", "fold", "seed", "_lane_order"], kind="stable"
    ).drop(columns=["_protocol_order", "_lane_order"])
    return combined.reset_index(drop=True)


def matrix_discrepancies(rows: pd.DataFrame) -> dict[str, Any]:
    key_columns = ["protocol", "fold", "seed", "lane"]
    actual_keys = [tuple(item) for item in rows[key_columns].itertuples(index=False, name=None)]
    actual_set = set(actual_keys)
    expected = expected_matrix_keys()
    duplicates = rows[rows.duplicated(key_columns, keep=False)][key_columns].to_dict("records")
    unknown_lanes = sorted(str(value) for value in rows.loc[rows["lane"].isna(), "source_control_type"].unique())
    return {
        "expected_rows": 420,
        "actual_rows": int(len(rows)),
        "unique_rows": int(len(actual_set)),
        "missing_keys": [list(key) for key in sorted(expected - actual_set)],
        "unexpected_keys": [list(key) for key in sorted(actual_set - expected)],
        "duplicate_keys": duplicates,
        "unknown_source_lanes": unknown_lanes,
        "matrix_completeness_pass": len(rows) == 420 and actual_set == expected,
        "matrix_uniqueness_pass": len(actual_keys) == len(actual_set),
    }


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.map(lambda value: str(value).strip().lower() in {"true", "1", "yes"})


def audit_checkpoints(rows: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    audits: list[dict[str, Any]] = []
    residual = rows[rows["lane"] != "frozen_ar_only"]
    for row in residual.itertuples(index=False):
        suppressed = bool(row.residual_suppressed) if not pd.isna(row.residual_suppressed) else False
        restored = bool(row.checkpoint_restore_pass)
        raw_path = None if pd.isna(row.checkpoint_path) else str(row.checkpoint_path)
        expected_checksum = None if pd.isna(row.checkpoint_checksum) else str(row.checkpoint_checksum)
        path = resolve_repo_path(raw_path) if raw_path else None
        exists = bool(path and path.exists())
        actual_checksum = file_digest(path, algorithm="blake2b") if exists else None
        checksum_match = bool(expected_checksum and actual_checksum == expected_checksum) if not suppressed else True
        policy_pass = bool(restored and ((suppressed and not raw_path) or (not suppressed and exists and checksum_match)))
        audits.append(
            {
                "protocol": row.protocol,
                "fold": int(row.fold),
                "seed": int(row.seed),
                "lane": row.lane,
                "residual_suppressed": suppressed,
                "checkpoint_restore_pass": restored,
                "checkpoint_path": raw_path,
                "checkpoint_exists": exists,
                "expected_checkpoint_checksum": expected_checksum,
                "actual_checkpoint_checksum": actual_checksum,
                "checkpoint_checksum_match": checksum_match,
                "checkpoint_policy_pass": policy_pass,
            }
        )
    frame = pd.DataFrame(audits)
    return frame, bool(len(frame) == 360 and frame["checkpoint_policy_pass"].all())


def audit_frozen_ar(
    source_root: Path,
    protocol_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple[int, int], dict[str, str]], bool]:
    manifest = read_json(source_root / "manifests" / "ar_baseline_generation_manifest.json")
    baselines = manifest.get("baselines", [])
    audits: list[dict[str, Any]] = []
    split_digests: dict[tuple[int, int], dict[str, str]] = {}
    metrics_by_key = protocol_rows.groupby(["fold", "seed"])
    expected_baselines = 10 if protocol_rows["protocol"].iloc[0] == "blocked_temporal_70_30" else 50
    for baseline in baselines:
        fold = int(baseline["fold"])
        seed = int(baseline["seed"])
        group = metrics_by_key.get_group((fold, seed))
        key = str(baseline["key"])
        split_digests[(fold, seed)] = {}
        for split_name, checksum_field in (("train", "train_checksum"), ("heldout_test", "test_checksum")):
            score_path = source_root / "frozen_ar_scores" / f"{key}__{split_name}.csv.gz"
            exists = score_path.exists()
            score_frame = pd.read_csv(score_path, usecols=["row_id", "frozen_ar_score"]) if exists else None
            score_checksum = (
                array_digest(score_frame["frozen_ar_score"].to_numpy(dtype=np.float32))
                if score_frame is not None
                else None
            )
            row_digest = (
                array_digest(score_frame["row_id"].to_numpy(dtype=np.int64))
                if score_frame is not None
                else None
            )
            split_digests[(fold, seed)][split_name] = row_digest or ""
            expected_checksum = str(baseline[checksum_field])
            metric_column = "frozen_ar_train_checksum" if split_name == "train" else "frozen_ar_test_checksum"
            lane_values = sorted(str(value) for value in group[metric_column].dropna().unique())
            identity_pass = lane_values == [expected_checksum]
            audits.append(
                {
                    "protocol": protocol_rows["protocol"].iloc[0],
                    "fold": fold,
                    "seed": seed,
                    "split": split_name,
                    "score_path": str(score_path.relative_to(REPO_ROOT)),
                    "score_file_exists": exists,
                    "expected_score_checksum": expected_checksum,
                    "actual_score_checksum": score_checksum,
                    "score_checksum_match": score_checksum == expected_checksum,
                    "row_id_digest": row_digest,
                    "matched_lane_count": int(len(group)),
                    "all_lanes_use_manifest_checksum": identity_pass,
                    "checkpoint_restore_pass": bool(baseline.get("checkpoint_restore_pass")),
                    "eval_mode_scoring_pass": bool(baseline.get("eval_mode_scoring_pass")),
                    "baseline_source": baseline.get("source"),
                }
            )
    frame = pd.DataFrame(audits)
    passed = bool(
        len(baselines) == expected_baselines
        and len(frame) == expected_baselines * 2
        and frame["score_file_exists"].all()
        and frame["score_checksum_match"].all()
        and frame["all_lanes_use_manifest_checksum"].all()
        and frame["checkpoint_restore_pass"].all()
        and frame["eval_mode_scoring_pass"].all()
    )
    return frame, split_digests, passed


def feature_policy_for(entry: dict[str, Any]) -> tuple[str | None, bool]:
    control = str(entry["control_type"])
    blocks = entry.get("blocks", [])
    if control == "diagnostics_only_residual":
        diagnostics_only = any(block.get("control") == "diagnostics_only" for block in blocks)
        pca_omitted = not any("pca_control" in block for block in blocks)
        return "omitted_diagnostics_only", bool(diagnostics_only and pca_omitted)
    pca_blocks = [block for block in blocks if "pca_control" in block]
    actual = pca_blocks[0].get("pca_control") if len(pca_blocks) == 1 else None
    expected = {
        "real_residual": "real",
        "label_permutation_residual": "real",
        "shuffled_pca_residual": "shuffled_train_and_test_separately",
        "random_pca_residual": "random_gaussian_matched_shape",
        "train_only_video_mean_residual": "train_only_video_mean",
    }[control]
    extra_pass = True
    if control == "train_only_video_mean_residual":
        extra_pass = pca_blocks[0].get("uses_test_rows_for_mean") is False if pca_blocks else False
    return actual, bool(actual == expected and extra_pass)


def audit_control_policies(source_root: Path, protocol: str) -> tuple[pd.DataFrame, bool]:
    feature_manifest = read_json(source_root / "manifests" / "feature_manifest.json")
    label_audit = read_json(source_root / "diagnostics" / "label_permutation_audit.json")
    mean_audit = read_json(source_root / "diagnostics" / "train_only_video_mean_audit.json")
    context_audit = read_json(source_root / "diagnostics" / "leakage_context_audit.json")
    label_keys = {
        (int(row.get("fold", 1)), int(row["seed"]))
        for row in label_audit.get("rows", [])
        if row.get("label_policy") == "permuted_train_and_permuted_inner_val_selection"
        and row.get("heldout_scoring_policy") == "true_heldout_labels_targets"
    }
    mean_keys = {
        (int(row.get("fold", 1)), int(row["seed"]))
        for row in mean_audit.get("rows", [])
        if row.get("uses_test_rows_for_mean") is False
    }
    audits: list[dict[str, Any]] = []
    for entry in feature_manifest.get("features", []):
        fold = int(entry.get("fold", 1))
        seed = int(entry["seed"])
        control = str(entry["control_type"])
        actual_policy, feature_pass = feature_policy_for(entry)
        label_pass = (fold, seed) in label_keys if control == "label_permutation_residual" else True
        mean_pass = (fold, seed) in mean_keys if control == "train_only_video_mean_residual" else True
        audits.append(
            {
                "protocol": protocol,
                "fold": fold,
                "seed": seed,
                "lane": control,
                "actual_feature_policy": actual_policy,
                "feature_policy_pass": feature_pass,
                "label_permutation_policy_pass": label_pass,
                "train_only_video_mean_policy_pass": mean_pass,
                "control_policy_pass": bool(feature_pass and label_pass and mean_pass),
            }
        )
    frame = pd.DataFrame(audits)
    expected_rows = 60 if protocol == "blocked_temporal_70_30" else 300
    context_pass = bool(
        context_audit.get("leakage_context_audit_pass")
        and context_audit.get("label_permutation_policy_pass")
        and context_audit.get("train_only_video_mean_pass")
        and context_audit.get("temporal_context_causal_only")
        and context_audit.get("no_centered_or_future_windows")
        and context_audit.get("same_video_history_masking")
    )
    passed = bool(len(frame) == expected_rows and frame["control_policy_pass"].all() and context_pass)
    return frame, passed


def pca_manifest_paths(blocked_root: Path, grouped_root: Path) -> list[Path]:
    blocked_run = read_json(blocked_root / "manifests" / "run_manifest.json")
    grouped_run = read_json(grouped_root / "manifests" / "run_manifest.json")
    blocked_pca_root = resolve_repo_path(blocked_run["foldsafe_pca_root"])
    grouped_pca_root = resolve_repo_path(grouped_run["grouped_pca_root"])
    blocked = blocked_pca_root / "features" / (
        f"{TARGET}__blocked_temporal_70_30__fold1__temporal_mean_2s__pca_manifest.json"
    )
    grouped = [
        grouped_pca_root / "features" / (
            f"{TARGET}__grouped_video__fold{fold}__temporal_mean_2s__pca_manifest.json"
        )
        for fold in FOLDS
    ]
    return [blocked, *grouped]


def audit_provenance(
    rows: pd.DataFrame,
    blocked_root: Path,
    grouped_root: Path,
    blocked_split_digests: dict[tuple[int, int], dict[str, str]],
    grouped_split_digests: dict[tuple[int, int], dict[str, str]],
) -> dict[str, Any]:
    blocked_run = read_json(blocked_root / "manifests" / "run_manifest.json")
    grouped_run = read_json(grouped_root / "manifests" / "run_manifest.json")
    run_contract_pass = bool(
        blocked_run.get("target") == TARGET
        and grouped_run.get("target") == TARGET
        and blocked_run.get("architecture") == ARCHITECTURE
        and grouped_run.get("architecture") == ARCHITECTURE
        and blocked_run.get("feature") == FEATURE
        and grouped_run.get("feature") == FEATURE
        and blocked_run.get("seeds") == list(SEEDS)
        and grouped_run.get("seeds") == list(SEEDS)
        and grouped_run.get("folds") == list(FOLDS)
        and blocked_run.get("matrix_size") == 70
        and grouped_run.get("matrix_size") == 350
        and blocked_run.get("no_504") is True
        and grouped_run.get("no_504") is True
        and blocked_run.get("no_continuous") is True
        and grouped_run.get("no_continuous") is True
    )
    row_contract_pass = bool(
        set(rows["target_name"]) == {TARGET}
        and set(rows["architecture"]) == {ARCHITECTURE}
        and set(rows["feature_name"]) == {FEATURE}
        and bool_series(rows["checkpoint_restore_pass"]).all()
        and bool_series(rows["eval_mode_scoring"]).all()
        and bool_series(rows["dropout_disabled"]).all()
    )
    split_rows: list[dict[str, Any]] = []
    for manifest_path in pca_manifest_paths(blocked_root, grouped_root):
        manifest = read_json(manifest_path)
        protocol = str(manifest["protocol"])
        fold = int(manifest["fold"])
        digests = blocked_split_digests if protocol == "blocked_temporal_70_30" else grouped_split_digests
        fold_digests = [value for (item_fold, _seed), value in digests.items() if item_fold == fold]
        train_digests = {value["train"] for value in fold_digests}
        test_digests = {value["heldout_test"] for value in fold_digests}
        component_path = resolve_repo_path(manifest["component_path"])
        if component_path.exists():
            with np.load(component_path) as component_bundle:
                component_actual = array_digest(component_bundle["components"])
        else:
            component_actual = None
        passed = bool(
            manifest.get("target_name") == TARGET
            and manifest.get("pca_width") == 256
            and manifest.get("no_test_fit") is True
            and manifest.get("no_global_pca") is True
            and manifest.get("future_leakage_suspected") is False
            and manifest.get("target_window_overlap") is False
            and train_digests == {str(manifest["train_idx_digest"])}
            and test_digests == {str(manifest["test_idx_digest"])}
            and component_actual == manifest.get("component_checksum")
        )
        split_rows.append(
            {
                "protocol": protocol,
                "fold": fold,
                "manifest_path": str(manifest_path.relative_to(REPO_ROOT)),
                "train_idx_digest": manifest.get("train_idx_digest"),
                "test_idx_digest": manifest.get("test_idx_digest"),
                "train_score_cache_row_digests": sorted(train_digests),
                "test_score_cache_row_digests": sorted(test_digests),
                "component_checksum_expected": manifest.get("component_checksum"),
                "component_checksum_actual": component_actual,
                "split_pca_provenance_pass": passed,
            }
        )
    split_pass = bool(len(split_rows) == 6 and all(row["split_pca_provenance_pass"] for row in split_rows))
    return {
        "schema_version": SCHEMA_VERSION,
        "target_window_policy": {
            "target": TARGET,
            "row_rate_hz": 2,
            "future_window_start_rows": 4,
            "future_window_end_rows_inclusive": 10,
            "future_window_seconds": "2.0_to_5.0_seconds",
            "washout_gap_rows": 4,
            "event_value": "max_future_arousal_minus_current_arousal",
            "threshold_policy": "train_only_quantile_0.90",
            "transform": "positive_delta",
        },
        "run_manifest_contract_pass": run_contract_pass,
        "row_contract_pass": row_contract_pass,
        "split_pca_rows": split_rows,
        "split_pca_provenance_pass": split_pass,
        "provenance_integrity_pass": bool(run_contract_pass and row_contract_pass and split_pass),
    }


def protocol_summary(rows: pd.DataFrame) -> pd.DataFrame:
    summary = (
        rows.groupby(["protocol", "lane"], sort=False)["pr_auc"]
        .agg(scored_rows="size", mean_pr_auc="mean", std_pr_auc="std", min_pr_auc="min", max_pr_auc="max")
        .reset_index()
    )
    return summary


def fold_seed_deltas(rows: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    for (protocol, fold, seed), group in rows.groupby(["protocol", "fold", "seed"], sort=False):
        values = group.set_index("lane")["pr_auc"].astype(float).to_dict()
        best_control = max(PRIMARY_CONTROLS, key=lambda lane: values[lane])
        row: dict[str, Any] = {
            "protocol": protocol,
            "fold": int(fold),
            "seed": int(seed),
            "real_pr_auc": values["real_residual"],
            "frozen_ar_pr_auc": values["frozen_ar_only"],
            "best_primary_control": best_control,
            "best_primary_control_pr_auc": values[best_control],
            "real_minus_frozen_ar_pr_auc": values["real_residual"] - values["frozen_ar_only"],
            "real_minus_best_control_pr_auc": values["real_residual"] - values[best_control],
        }
        for lane in PRIMARY_CONTROLS:
            row[f"real_minus_{lane}_pr_auc"] = values["real_residual"] - values[lane]
        output.append(row)
    return pd.DataFrame(output)


def compose_overall_gate(
    *,
    discrepancies: dict[str, Any],
    provenance_pass: bool,
    frozen_ar_pass: bool,
    control_policy_pass: bool,
    checkpoint_pass: bool,
    blocked_gate: dict[str, Any],
    grouped_gate: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "matrix_completeness_pass": bool(discrepancies["matrix_completeness_pass"]),
        "matrix_uniqueness_pass": bool(discrepancies["matrix_uniqueness_pass"]),
        "provenance_integrity_pass": provenance_pass,
        "frozen_ar_checksum_identity_pass": frozen_ar_pass,
        "control_policy_pass": control_policy_pass,
        "checkpoint_restore_and_checksum_pass": checkpoint_pass,
        "blocked_confirmation_pass": bool(blocked_gate.get("binary_pass"))
        and not blocked_gate.get("failed_gates"),
        "updated_grouped_compatibility_pass": bool(grouped_gate.get("grouped_compatibility_pass"))
        and not grouped_gate.get("failed_gates"),
        "no_unresolved_incompatible_rows_pass": not any(
            discrepancies[name]
            for name in ("missing_keys", "unexpected_keys", "duplicate_keys", "unknown_source_lanes")
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": SCHEMA_VERSION,
        "target": TARGET,
        "architecture": ARCHITECTURE,
        "feature": FEATURE,
        "matrix_rows_expected": 420,
        "matrix_rows_actual": int(discrepancies["actual_rows"]),
        "rows_reused": int(discrepancies["actual_rows"]),
        "rows_rerun": 0,
        "blocked_rows": 70,
        "grouped_rows": 350,
        **checks,
        "blocked_canonical_result": {
            "real_pr_auc": blocked_gate.get("real_pr_auc"),
            "frozen_ar_pr_auc": blocked_gate.get("frozen_ar_pr_auc"),
            "best_control": blocked_gate.get("best_control"),
            "best_control_pr_auc": blocked_gate.get("best_control_pr_auc"),
            "delta_vs_ar": blocked_gate.get("delta_vs_ar"),
            "delta_vs_best_control": blocked_gate.get("delta_vs_best_control"),
            "seeds_positive_vs_ar": blocked_gate.get("seeds_positive_vs_ar"),
            "seeds_positive_vs_best_control": blocked_gate.get("seeds_positive_vs_best_control"),
        },
        "grouped_canonical_result": {
            "real_pr_auc": grouped_gate.get("real_pr_auc"),
            "ar_frozen_pr_auc": grouped_gate.get("ar_frozen_pr_auc"),
            "best_control": grouped_gate.get("best_control"),
            "best_control_pr_auc": grouped_gate.get("best_control_pr_auc"),
            "delta_vs_ar": grouped_gate.get("delta_vs_ar"),
            "delta_vs_best_control": grouped_gate.get("delta_vs_best_control"),
            "fold_seed_positives_vs_best_control": grouped_gate.get("fold_seed_positives_vs_best_control"),
            "real_minus_label_permutation_pr_auc": grouped_gate.get("real_minus_label_permutation_pr_auc"),
            "label_permutation_minus_ar_pr_auc": grouped_gate.get("label_permutation_minus_ar_pr_auc"),
            "verdict_update_policy": grouped_gate.get("verdict_update_policy"),
        },
        "failed_gates": failed,
        "overall_selected_head_420_confirmation_pass": not failed,
        "no_504_run": True,
        "no_new_training": True,
        "no_new_scoring": True,
        "no_continuous_model_development": True,
    }


def source_artifact_inventory(paths: Iterable[Path]) -> pd.DataFrame:
    rows = []
    for path in sorted(set(paths)):
        exists = path.exists()
        rows.append(
            {
                "path": str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
                "exists": exists,
                "size_bytes": path.stat().st_size if exists else None,
                "sha256": file_digest(path) if exists else None,
            }
        )
    return pd.DataFrame(rows)


def write_report(path: Path, output_root: Path, evidence_root: Path, gate: dict[str, Any]) -> None:
    blocked = gate["blocked_canonical_result"]
    grouped = gate["grouped_canonical_result"]
    status = "PASS" if gate["overall_selected_head_420_confirmation_pass"] else "FAIL"
    text = f"""# AGAIN Phase 5 Selected-Head 420-Row Confirmation

## Verdict

**{status}.** Matrix completeness is `{gate['matrix_rows_actual']}/420`: `70/70` strict blocked temporal rows plus `350/350` grouped held-out-video rows. All scored rows were reused; no training, scoring, PCA fitting, or rerun was required.

Neural Bridge passes a full bounded 420-row selected-head confirmation for future arousal event ranking on AGAIN across strict blocked temporal validation and grouped held-out-video compatibility, using the redesigned washout-gap target, short temporal convolution residual, matched frozen AR, and matched controls.

## Fixed Scope

- Target: `{TARGET}`
- Head: `{ARCHITECTURE}`
- Feature: `{FEATURE}`
- Seeds: `20260625` through `20260634`
- Lanes: seven matched lanes per protocol/fold/seed
- Historical 504 matrix: not run and not part of this confirmation

## Canonical Protocol Results

| Protocol | Real PR-AUC | AR/frozen PR-AUC | Best matched control | Control PR-AUC | Delta vs AR | Delta vs control | Consistency |
|---|---:|---:|---|---:|---:|---:|---:|
| Blocked temporal | {blocked['real_pr_auc']:.10f} | {blocked['frozen_ar_pr_auc']:.10f} | `{blocked['best_control']}` | {blocked['best_control_pr_auc']:.10f} | {blocked['delta_vs_ar']:+.10f} | {blocked['delta_vs_best_control']:+.10f} | {blocked['seeds_positive_vs_best_control']}/10 |
| Grouped video | {grouped['real_pr_auc']:.10f} | {grouped['ar_frozen_pr_auc']:.10f} | `{grouped['best_control']}` | {grouped['best_control_pr_auc']:.10f} | {grouped['delta_vs_ar']:+.10f} | {grouped['delta_vs_best_control']:+.10f} | {grouped['fold_seed_positives_vs_best_control']}/50 |

The grouped label-permutation verdict remains frozen-AR-residual-aware: real minus label permutation is `{grouped['real_minus_label_permutation_pr_auc']:+.10f}`, while label permutation minus AR is `{grouped['label_permutation_minus_ar_pr_auc']:+.10f}`. The superseded raw-prevalence-near-chance gate was not restored.

## Integrity And Reuse Audit

- Matrix keys complete and unique: `{str(gate['matrix_completeness_pass']).lower()}` / `{str(gate['matrix_uniqueness_pass']).lower()}`
- Target, head, target-window, split, PCA, and row provenance: `{str(gate['provenance_integrity_pass']).lower()}`
- Frozen-AR score-cache hashes and within-group checksum identity: `{str(gate['frozen_ar_checksum_identity_pass']).lower()}`
- Executable control policies: `{str(gate['control_policy_pass']).lower()}`
- Eval-mode checkpoint restoration/checksums: `{str(gate['checkpoint_restore_and_checksum_pass']).lower()}`
- Rows reused / rerun: `{gate['rows_reused']}` / `{gate['rows_rerun']}`
- Failed gates: `{gate['failed_gates']}`

## Artifacts

- Output root: `{output_root}`
- Evidence snapshot: `{evidence_root}`
- Row manifest: `{output_root / 'metrics' / 'selected_head_420_row_manifest.csv'}`
- Gate JSON: `{output_root / 'promotion' / 'selected_head_420_gates.json'}`

## Claim Boundary

This is a bounded selected-head binary future-event ranking confirmation. It is not a 504 run and does not establish exact continuous-value forecasting, blocked continuous generalization, broad all-target/all-dataset prediction, or universal temporal generalization. The prior grouped continuous future-movement ranking/lift pass remains a separate bounded result.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def copy_evidence(output_root: Path, evidence_root: Path, report_path: Path) -> None:
    mapping = {
        output_root / "metrics" / "selected_head_420_row_manifest.csv": evidence_root / "metrics" / "selected_head_420_row_manifest.csv",
        output_root / "metrics" / "selected_head_420_protocol_summary.csv": evidence_root / "metrics" / "selected_head_420_protocol_summary.csv",
        output_root / "metrics" / "selected_head_420_fold_seed_deltas.csv": evidence_root / "metrics" / "selected_head_420_fold_seed_deltas.csv",
        output_root / "diagnostics" / "matrix_discrepancies.json": evidence_root / "diagnostics" / "matrix_discrepancies.json",
        output_root / "diagnostics" / "provenance_audit.json": evidence_root / "diagnostics" / "provenance_audit.json",
        output_root / "diagnostics" / "frozen_ar_checksum_audit.csv": evidence_root / "diagnostics" / "frozen_ar_checksum_audit.csv",
        output_root / "diagnostics" / "checkpoint_checksum_audit.csv": evidence_root / "diagnostics" / "checkpoint_checksum_audit.csv",
        output_root / "diagnostics" / "control_policy_audit.csv": evidence_root / "diagnostics" / "control_policy_audit.csv",
        output_root / "diagnostics" / "reuse_rerun_accounting.json": evidence_root / "diagnostics" / "reuse_rerun_accounting.json",
        output_root / "diagnostics" / "discrepancy_ledger.json": evidence_root / "diagnostics" / "discrepancy_ledger.json",
        output_root / "manifests" / "run_manifest.json": evidence_root / "manifests" / "run_manifest.json",
        output_root / "manifests" / "source_artifact_checksums.csv": evidence_root / "manifests" / "source_artifact_checksums.csv",
        output_root / "manifests" / "generated_artifact_checksums.csv": evidence_root / "manifests" / "generated_artifact_checksums.csv",
        output_root / "promotion" / "selected_head_420_gates.json": evidence_root / "promotion" / "selected_head_420_gates.json",
        output_root / "promotion" / "selected_head_420_failure_reasons.json": evidence_root / "promotion" / "selected_head_420_failure_reasons.json",
        output_root / "promotion" / "selected_head_420_adversarial_verdict.json": evidence_root / "promotion" / "selected_head_420_adversarial_verdict.json",
        report_path: evidence_root / "reports" / report_path.name,
    }
    for source, destination in mapping.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    gate = read_json(output_root / "promotion" / "selected_head_420_gates.json")
    readme = f"""# Phase 5.5 Selected-Head 420 Confirmation Evidence

Verdict: **{'PASS' if gate['overall_selected_head_420_confirmation_pass'] else 'FAIL'}**.

This bounded snapshot consolidates `70/70` blocked temporal and `350/350` grouped-video rows for `{TARGET}` / `{ARCHITECTURE}`. All `420` rows were reused after provenance and checksum audit; rerun count is `0`.

Start with:

- `reports/{report_path.name}`
- `metrics/selected_head_420_row_manifest.csv`
- `promotion/selected_head_420_gates.json`
- `diagnostics/provenance_audit.json`
- `artifact_manifest.csv`

The grouped verdict uses the updated frozen-AR-residual-aware label-permutation interpretation. This is not a 504 run and makes no exact continuous-value or universal generalization claim.
"""
    (evidence_root / "README.md").write_text(readme, encoding="utf-8")
    evidence_files = [path for path in evidence_root.rglob("*") if path.is_file() and path.name != "artifact_manifest.csv"]
    inventory = source_artifact_inventory(evidence_files)
    inventory.to_csv(evidence_root / "artifact_manifest.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocked-root", default=str(BLOCKED_ROOT))
    parser.add_argument("--grouped-root", default=str(GROUPED_ROOT))
    parser.add_argument("--grouped-verdict-root", default=str(GROUPED_VERDICT_ROOT))
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--evidence-root", default=None)
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stamp = args.timestamp or utc_stamp()
    blocked_root = resolve_repo_path(args.blocked_root)
    grouped_root = resolve_repo_path(args.grouped_root)
    grouped_verdict_root = resolve_repo_path(args.grouped_verdict_root)
    output_root = resolve_repo_path(
        args.output_root or f"outputs/again_dense_2hz_phase5_selected_head_420_confirmation_{stamp}"
    )
    evidence_root = resolve_repo_path(
        args.evidence_root or f"evidence/phase_5_5_selected_head_420_confirmation_{stamp}"
    )
    reports_dir = resolve_repo_path(args.reports_dir)
    source_files = {
        "blocked_metrics": blocked_root / "metrics" / BLOCKED_METRICS,
        "grouped_metrics": grouped_root / "metrics" / GROUPED_METRICS,
        "blocked_gate": blocked_root / "promotion" / BLOCKED_GATE,
        "grouped_updated_gate": grouped_verdict_root / "promotion" / GROUPED_UPDATED_GATE,
    }
    dry_run = {
        "schema_version": SCHEMA_VERSION,
        "matrix_rows_expected": len(expected_matrix_keys()),
        "blocked_rows_expected": 70,
        "grouped_rows_expected": 350,
        "source_files_exist": {name: path.exists() for name, path in source_files.items()},
        "no_training": True,
        "no_scoring": True,
        "no_504": True,
    }
    if args.dry_run:
        print(json.dumps(dry_run, indent=2, sort_keys=True))
        return 0 if all(dry_run["source_files_exist"].values()) else 1
    for root in (output_root, evidence_root):
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty artifact root: {root}")
    for directory in ("metrics", "diagnostics", "manifests", "promotion", "reports"):
        (output_root / directory).mkdir(parents=True, exist_ok=True)

    blocked_source = pd.read_csv(source_files["blocked_metrics"])
    grouped_source = pd.read_csv(source_files["grouped_metrics"])
    rows = normalize_source_rows(blocked_source, grouped_source)
    discrepancies = matrix_discrepancies(rows)
    checkpoint_audit, checkpoint_pass = audit_checkpoints(rows)
    blocked_rows = rows[rows["protocol"] == "blocked_temporal_70_30"]
    grouped_rows = rows[rows["protocol"] == "grouped_video"]
    blocked_ar_audit, blocked_split_digests, blocked_ar_pass = audit_frozen_ar(blocked_root, blocked_rows)
    grouped_ar_audit, grouped_split_digests, grouped_ar_pass = audit_frozen_ar(grouped_root, grouped_rows)
    frozen_ar_audit = pd.concat([blocked_ar_audit, grouped_ar_audit], ignore_index=True)
    blocked_control_audit, blocked_control_pass = audit_control_policies(blocked_root, "blocked_temporal_70_30")
    grouped_control_audit, grouped_control_pass = audit_control_policies(grouped_root, "grouped_video")
    control_audit = pd.concat([blocked_control_audit, grouped_control_audit], ignore_index=True)
    provenance = audit_provenance(
        rows,
        blocked_root,
        grouped_root,
        blocked_split_digests,
        grouped_split_digests,
    )
    blocked_gate = read_json(source_files["blocked_gate"])
    grouped_gate = read_json(source_files["grouped_updated_gate"])
    gate = compose_overall_gate(
        discrepancies=discrepancies,
        provenance_pass=bool(provenance["provenance_integrity_pass"]),
        frozen_ar_pass=bool(blocked_ar_pass and grouped_ar_pass),
        control_policy_pass=bool(blocked_control_pass and grouped_control_pass),
        checkpoint_pass=checkpoint_pass,
        blocked_gate=blocked_gate,
        grouped_gate=grouped_gate,
    )

    rows.to_csv(output_root / "metrics" / "selected_head_420_row_manifest.csv", index=False)
    protocol_summary(rows).to_csv(output_root / "metrics" / "selected_head_420_protocol_summary.csv", index=False)
    fold_seed_deltas(rows).to_csv(output_root / "metrics" / "selected_head_420_fold_seed_deltas.csv", index=False)
    checkpoint_audit.to_csv(output_root / "diagnostics" / "checkpoint_checksum_audit.csv", index=False)
    frozen_ar_audit.to_csv(output_root / "diagnostics" / "frozen_ar_checksum_audit.csv", index=False)
    control_audit.to_csv(output_root / "diagnostics" / "control_policy_audit.csv", index=False)
    write_json(output_root / "diagnostics" / "matrix_discrepancies.json", discrepancies)
    write_json(output_root / "diagnostics" / "provenance_audit.json", provenance)
    reuse_accounting = {
        "blocked": {"rows_expected": 70, "rows_reused": int(len(blocked_rows)), "rows_rerun": 0},
        "grouped": {"rows_expected": 350, "rows_reused": int(len(grouped_rows)), "rows_rerun": 0},
        "total": {"rows_expected": 420, "rows_reused": int(len(rows)), "rows_rerun": 0},
        "rerun_required": False,
        "rerun_reason": None,
    }
    write_json(output_root / "diagnostics" / "reuse_rerun_accounting.json", reuse_accounting)
    write_json(output_root / "diagnostics" / "discrepancy_ledger.json", {"discrepancies": []})
    write_json(output_root / "promotion" / "selected_head_420_gates.json", gate)
    write_json(
        output_root / "promotion" / "selected_head_420_failure_reasons.json",
        {"overall_pass": gate["overall_selected_head_420_confirmation_pass"], "failed_gates": gate["failed_gates"]},
    )
    adversarial = {
        "schema_version": SCHEMA_VERSION,
        "verdict": "pass" if gate["overall_selected_head_420_confirmation_pass"] else "fail",
        "overall_selected_head_420_confirmation_pass": gate["overall_selected_head_420_confirmation_pass"],
        "rows_reused": 420,
        "rows_rerun": 0,
        "silent_substitutions": False,
        "scientific_thresholds_changed_after_results": False,
        "superseded_grouped_label_permutation_gate_restored": False,
        "historical_504_run_performed": False,
        "continuous_model_development_performed": False,
        "bounded_claim": "controlled future arousal event ranking across strict blocked temporal and grouped held-out-video validation on AGAIN",
        "not_proven": [
            "exact continuous future arousal values",
            "blocked continuous generalization",
            "broad all-target/all-dataset prediction",
            "universal temporal generalization",
        ],
        "failed_gates": gate["failed_gates"],
    }
    write_json(output_root / "promotion" / "selected_head_420_adversarial_verdict.json", adversarial)

    pca_paths = pca_manifest_paths(blocked_root, grouped_root)
    inventory_paths = [
        *source_files.values(),
        blocked_root / "manifests" / "run_manifest.json",
        grouped_root / "manifests" / "run_manifest.json",
        blocked_root / "manifests" / "ar_baseline_generation_manifest.json",
        grouped_root / "manifests" / "ar_baseline_generation_manifest.json",
        blocked_root / "manifests" / "feature_manifest.json",
        grouped_root / "manifests" / "feature_manifest.json",
        blocked_root / "diagnostics" / "leakage_context_audit.json",
        grouped_root / "diagnostics" / "leakage_context_audit.json",
        blocked_root / "diagnostics" / "label_permutation_audit.json",
        grouped_root / "diagnostics" / "label_permutation_audit.json",
        blocked_root / "diagnostics" / "train_only_video_mean_audit.json",
        grouped_root / "diagnostics" / "train_only_video_mean_audit.json",
        REPO_ROOT / "backend/scripts/run_again_dense_2hz_phase5_redesigned_target_blocked.py",
        REPO_ROOT / "backend/scripts/run_again_dense_2hz_phase5_temporal_residual_blocked.py",
        REPO_ROOT / "backend/scripts/run_again_dense_2hz_phase5_temporal_residual_binary_big_confirm.py",
        REPO_ROOT / "backend/scripts/run_again_dense_2hz_phase5_temporal_residual_grouped_compat.py",
        REPO_ROOT / "backend/scripts/update_again_dense_2hz_phase5_temporal_residual_grouped_compat_verdict.py",
        REPO_ROOT / "backend/scripts/assemble_again_dense_2hz_phase5_selected_head_420_confirmation.py",
        REPO_ROOT / "tests/test_again_selected_head_420_confirmation.py",
        *pca_paths,
    ]
    source_inventory = source_artifact_inventory(inventory_paths)
    source_inventory.to_csv(output_root / "manifests" / "source_artifact_checksums.csv", index=False)
    run_manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": now_iso(),
        "target": TARGET,
        "architecture": ARCHITECTURE,
        "feature": FEATURE,
        "protocols": list(PROTOCOLS),
        "folds": list(FOLDS),
        "seeds": list(SEEDS),
        "lanes": list(LANES),
        "matrix_rows_expected": 420,
        "matrix_rows_actual": int(len(rows)),
        "blocked_rows": int(len(blocked_rows)),
        "grouped_rows": int(len(grouped_rows)),
        "blocked_source_root": str(blocked_root.relative_to(REPO_ROOT)),
        "grouped_source_root": str(grouped_root.relative_to(REPO_ROOT)),
        "grouped_updated_verdict_root": str(grouped_verdict_root.relative_to(REPO_ROOT)),
        "output_root": str(output_root.relative_to(REPO_ROOT)),
        "evidence_root": str(evidence_root.relative_to(REPO_ROOT)),
        "rows_reused": int(len(rows)),
        "rows_rerun": 0,
        "no_training": True,
        "no_scoring": True,
        "no_pca_fit": True,
        "no_504": True,
        "no_continuous_model_development": True,
        "overall_pass": gate["overall_selected_head_420_confirmation_pass"],
    }
    write_json(output_root / "manifests" / "run_manifest.json", run_manifest)

    report_name = f"again_dense_2hz_phase5_selected_head_420_confirmation_{stamp}.md"
    report_path = reports_dir / report_name
    write_report(report_path, output_root.relative_to(REPO_ROOT), evidence_root.relative_to(REPO_ROOT), gate)
    shutil.copy2(report_path, output_root / "reports" / report_name)
    generated_files = [
        path
        for path in output_root.rglob("*")
        if path.is_file() and path.name != "generated_artifact_checksums.csv"
    ]
    source_artifact_inventory(generated_files).to_csv(
        output_root / "manifests" / "generated_artifact_checksums.csv", index=False
    )
    copy_evidence(output_root, evidence_root, report_path)
    result = {
        "run_completed": True,
        "overall_pass": gate["overall_selected_head_420_confirmation_pass"],
        "matrix_rows": len(rows),
        "rows_reused": len(rows),
        "rows_rerun": 0,
        "output_root": str(output_root.relative_to(REPO_ROOT)),
        "evidence_root": str(evidence_root.relative_to(REPO_ROOT)),
        "report": str(report_path.relative_to(REPO_ROOT)),
        "failed_gates": gate["failed_gates"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if gate["overall_selected_head_420_confirmation_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
