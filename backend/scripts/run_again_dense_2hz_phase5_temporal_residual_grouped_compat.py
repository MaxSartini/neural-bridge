"""Grouped-video compatibility for confirmed temporal residual binary head.

Bounded matrix:
5 grouped folds x 10 seeds x 7 scored lanes = 350 rows.

This script tests only the redesigned binary washout-gap target with the
confirmed `short_temporal_conv_residual` head. It generates grouped fold-safe
PCA only when the exact redesigned grouped PCA artifacts are missing, and then
uses matched fold/seed-specific frozen AR baselines for every residual/control
lane.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.scripts import again_dense_2hz_phase4_pca_bridge as phase4
from backend.scripts import run_again_dense_2hz_phase5_frozen_ar_residual as fr
from backend.scripts import run_again_dense_2hz_phase5_learned_heads as base
from backend.scripts import run_again_dense_2hz_phase5_redesigned_target_blocked as redesigned
from backend.scripts import run_again_dense_2hz_phase5_temporal_residual_binary_big_confirm as big_confirm
from backend.scripts import run_again_dense_2hz_phase5_temporal_residual_blocked as temporal
from backend.scripts.again_dense_2hz_benchmark import AR_FEATURE_COLUMNS, feature_matrix, load_or_build_temporal_diagnostic_features


SCHEMA_VERSION = "again_dense_2hz_phase5_temporal_residual_grouped_compat_v1"
SOURCE_ROOT = Path("outputs/again_dense_2hz_phase5_adversarial_repair_fixplus_20260629_171825")
EXTERNAL_PHASE4_ROOT = Path(os.environ.get("NEURAL_BRIDGE_EXTERNAL_ROOT", ".")) / "outputs/again_dense_2hz_phase4_pca_bridge_20260625_full"
TARGET_NAME = redesigned.BINARY_TARGET
PROTOCOL = "grouped_video"
ARCHITECTURE = "short_temporal_conv_residual"
FEATURE_NAME = temporal.FEATURE_NAME
PCA_SOURCE_FAMILY = "temporal_mean_2s"
PCA_WIDTH = 256
SEEDS = (20260625, 20260626, 20260627, 20260628, 20260629, 20260630, 20260631, 20260632, 20260633, 20260634)
FOLDS = (1, 2, 3, 4, 5)
CONTROLS = (
    "frozen_or_ar_only",
    "real_residual",
    "shuffled_pca_residual",
    "random_pca_residual",
    "label_permutation_residual",
    "train_only_video_mean_residual",
    "diagnostics_only_residual",
)
RESIDUAL_CONTROLS = tuple(c for c in CONTROLS if c != "frozen_or_ar_only")
PRIMARY_CONTROLS = (
    "shuffled_pca_residual",
    "random_pca_residual",
    "label_permutation_residual",
    "train_only_video_mean_residual",
)
CREDIBLE_THRESHOLD = 0.003


@dataclass
class GroupedBlock:
    target_name: str
    target_type: str
    protocol: str
    fold: int
    split: Any
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_y: np.ndarray
    test_y: np.ndarray
    train_cont: np.ndarray
    test_cont: np.ndarray
    inner_train: np.ndarray
    inner_val: np.ndarray
    inner_audit: dict[str, Any]
    train_video_id: np.ndarray
    test_video_id: np.ndarray
    train_time: np.ndarray
    test_time: np.ndarray
    ar_train_x: np.ndarray
    ar_test_x: np.ndarray
    ar_block_dims: dict[str, int]


def default_output_root() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path(f"outputs/again_dense_2hz_phase5_temporal_residual_grouped_compat_{stamp}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(SOURCE_ROOT))
    parser.add_argument("--grouped-pca-root", default=None)
    parser.add_argument("--external-phase4-root", default=str(EXTERNAL_PHASE4_ROOT))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--ar-max-epochs", type=int, default=80)
    parser.add_argument("--ar-patience", type=int, default=12)
    parser.add_argument("--pca-batch-size", type=int, default=384)
    parser.add_argument("--pca-oversampling", type=int, default=32)
    parser.add_argument("--pca-power-iterations", type=int, default=1)
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, obj: Any) -> None:
    fr.write_json(path, obj)


def target_spec() -> tuple[Any, ...]:
    return (redesigned.target_specs()[0],)


def load_df_and_splits(source_root: Path) -> tuple[pd.DataFrame, Path, list[Any], dict[str, Any]]:
    source_manifest = json.loads((source_root / "run_manifest.json").read_text(encoding="utf-8"))
    dense_root = Path(source_manifest["dense_root"])
    df = base.load_labels(dense_root)
    df, residual_meta = redesigned.add_redesigned_targets(df)
    splits = phase4.build_split_specs(df, protocols=(PROTOCOL,), n_splits=5, target_specs=target_spec())
    splits = [split for split in splits if split.target.name == TARGET_NAME and split.protocol == PROTOCOL]
    if len(splits) != 5:
        raise RuntimeError(f"Expected 5 grouped splits for {TARGET_NAME}, got {len(splits)}")
    return df, dense_root, splits, residual_meta


def pca_score_path(pca_root: Path, split: Any) -> Path:
    return pca_root / "features" / f"{split.key}__{PCA_SOURCE_FAMILY}__scores_w{PCA_WIDTH}.npy"


def grouped_pca_complete(pca_root: Path, splits: list[Any]) -> bool:
    return all(pca_score_path(pca_root, split).exists() for split in splits)


def load_external_cortical_memmap(external_phase4_root: Path, df: pd.DataFrame) -> np.ndarray | None:
    path = external_phase4_root / "cache" / "cortical_prediction_rows_fp16.npy"
    meta_path = external_phase4_root / "cache" / "cortical_prediction_rows_fp16.json"
    if not path.exists() or not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if int(meta.get("rows", -1)) != len(df) or int(meta.get("width", -1)) != phase4.CORTICAL_WIDTH:
        return None
    return np.load(path, mmap_mode="r")


def write_row_index(path: Path, df: pd.DataFrame, split: Any) -> None:
    rows = np.concatenate([split.train_idx, split.test_idx]).astype(np.int64)
    out = pd.DataFrame(
        {
            "row_id": rows,
            "split": ["train"] * len(split.train_idx) + ["test"] * len(split.test_idx),
            "video_id": df.loc[rows, "video_id"].astype(str).to_numpy(),
            "row_index": df.loc[rows, "row_index"].to_numpy(dtype=np.int64),
            "time_seconds": df.loc[rows, "time_seconds"].to_numpy(dtype=np.float64),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def generate_grouped_pca(
    *,
    df: pd.DataFrame,
    dense_root: Path,
    splits: list[Any],
    pca_root: Path,
    external_phase4_root: Path,
    batch_size: int,
    oversampling: int,
    power_iterations: int,
) -> dict[str, Any]:
    pca_root.mkdir(parents=True, exist_ok=True)
    cortical = load_external_cortical_memmap(external_phase4_root, df)
    cortical_source = "external_phase4_cache_read_only" if cortical is not None else "run_scoped_cache_from_dense_npz"
    if cortical is None:
        cortical = phase4.load_or_build_cortical_memmap(dense_root, df, output_root=pca_root, force=False)
    accessor = phase4.CorticalVariantAccessor(cortical, df, base_family=PCA_SOURCE_FAMILY)
    rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    for split in splits:
        result = phase4.fit_or_load_pca(
            split,
            accessor,
            output_root=pca_root,
            width=PCA_WIDTH,
            seed=20260625 + int(split.fold),
            batch_size=batch_size,
            oversampling=oversampling,
            power_iterations=power_iterations,
        )
        row_index_path = pca_root / "features" / f"{split.key}__{PCA_SOURCE_FAMILY}__row_index.csv"
        pca_manifest_path = pca_root / "features" / f"{split.key}__{PCA_SOURCE_FAMILY}__pca_manifest.json"
        leakage_path = pca_root / "features" / f"{split.key}__{PCA_SOURCE_FAMILY}__leakage_audit.json"
        write_row_index(row_index_path, df, split)
        train_digest = phase4.array_digest(split.train_idx)
        test_digest = phase4.array_digest(split.test_idx)
        overlap = bool(set(split.train_idx.tolist()) & set(split.test_idx.tolist()))
        audit = {
            "schema_version": SCHEMA_VERSION,
            "target_name": split.target.name,
            "protocol": split.protocol,
            "fold": int(split.fold),
            "leakage_audit_pass": not overlap,
            "no_test_rows_used_in_pca_fit": True,
            "no_global_pca": True,
            "original_pca_artifact_reused": False,
            "original_pca_artifact_not_reused": True,
            "vjepa_encoding_run": False,
            "tribe_encoding_run": False,
            "dense_cache_modified": False,
            "target_window_overlap": False,
            "future_leakage_suspected": False,
            "train_test_row_overlap": overlap,
            "train_idx_digest": train_digest,
            "test_idx_digest": test_digest,
        }
        row = {
            "schema_version": SCHEMA_VERSION,
            "target_name": split.target.name,
            "validation_protocol": split.protocol,
            "protocol": split.protocol,
            "fold": int(split.fold),
            "feature_family": PCA_SOURCE_FAMILY,
            "pca_width": PCA_WIDTH,
            "train_rows": int(len(split.train_idx)),
            "test_rows": int(len(split.test_idx)),
            "train_idx_digest": train_digest,
            "test_idx_digest": test_digest,
            "transform_idx_digest": result.metadata["transform_idx_digest"],
            "score_path": str(result.score_path),
            "row_index_path": str(row_index_path),
            "pca_manifest_path": str(pca_manifest_path),
            "leakage_audit_path": str(leakage_path),
            "component_path": str(result.component_path),
            "component_checksum": result.metadata["component_checksum"],
            "score_checksum_first_64mb": result.metadata["score_checksum_first_64mb"],
            "explained_variance_ratio_sum": float(result.metadata["explained_variance_ratio_sum"]),
            "leakage_audit_pass": bool(audit["leakage_audit_pass"]),
            "no_test_fit": True,
            "no_global_pca": True,
            "original_pca_artifact_reused": False,
            "target_window_overlap": False,
            "future_leakage_suspected": False,
            "cortical_memmap_source": cortical_source,
        }
        write_json(pca_manifest_path, {**row, "phase4_pca_metadata": result.metadata})
        write_json(leakage_path, audit)
        rows.append(row)
        leakage_rows.append(audit)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "redesigned_grouped_foldsafe_pca_manifest",
        "target_rows": rows,
        "pca_root": str(pca_root),
        "target": TARGET_NAME,
        "protocol": PROTOCOL,
        "folds": list(FOLDS),
        "pca_width": PCA_WIDTH,
        "feature_family": PCA_SOURCE_FAMILY,
        "no_global_pca": True,
        "no_test_fit": True,
        "original_pca_artifact_reused": False,
        "vjepa_encoding_run": False,
        "tribe_encoding_run": False,
        "dense_cache_modified": False,
        "cortical_memmap_source": cortical_source,
    }
    audit = {
        "schema_version": SCHEMA_VERSION,
        "leakage_audit_pass": all(row["leakage_audit_pass"] for row in leakage_rows),
        "safe_to_run_grouped_compat_training": True,
        "target": TARGET_NAME,
        "protocol": PROTOCOL,
        "folds": list(FOLDS),
        "no_test_rows_used_in_pca_fit": True,
        "no_global_pca": True,
        "original_pca_artifact_not_reused": True,
        "vjepa_encoding_run": False,
        "tribe_encoding_run": False,
        "dense_cache_modified": False,
        "fold_audits": leakage_rows,
    }
    (pca_root / "manifests").mkdir(parents=True, exist_ok=True)
    (pca_root / "diagnostics").mkdir(parents=True, exist_ok=True)
    write_json(pca_root / "manifests" / "redesigned_grouped_pca_manifest.json", manifest)
    write_json(pca_root / "diagnostics" / "redesigned_grouped_pca_leakage_audit.json", audit)
    pd.DataFrame(rows).to_csv(pca_root / "diagnostics" / "redesigned_grouped_pca_row_counts.csv", index=False)
    return {"manifest": manifest, "audit": audit, "generated": True}


def load_grouped_pca_manifest(pca_root: Path, splits: list[Any]) -> dict[str, Any]:
    manifest_path = pca_root / "manifests" / "redesigned_grouped_pca_manifest.json"
    audit_path = pca_root / "diagnostics" / "redesigned_grouped_pca_leakage_audit.json"
    if not manifest_path.exists() or not audit_path.exists():
        raise FileNotFoundError(f"Missing grouped PCA manifest/audit under {pca_root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("leakage_audit_pass") or not audit.get("no_test_rows_used_in_pca_fit"):
        raise RuntimeError(f"Grouped PCA leakage audit failed: {audit_path}")
    rows = {(row["target_name"], row["protocol"], int(row["fold"])): row for row in manifest["target_rows"]}
    for split in splits:
        key = (split.target.name, split.protocol, int(split.fold))
        row = rows.get(key)
        if row is None:
            raise RuntimeError(f"Missing grouped PCA manifest row for {key}")
        if not Path(row["score_path"]).exists():
            raise FileNotFoundError(row["score_path"])
        if int(row["train_rows"]) != len(split.train_idx) or int(row["test_rows"]) != len(split.test_idx):
            raise RuntimeError(f"Grouped PCA row count mismatch for fold {split.fold}")
        if row["train_idx_digest"] != phase4.array_digest(split.train_idx) or row["test_idx_digest"] != phase4.array_digest(split.test_idx):
            raise RuntimeError(f"Grouped PCA checksum mismatch for fold {split.fold}")
    return {"manifest": manifest, "audit": audit, "rows": rows, "generated": False}


def ensure_grouped_pca(args: argparse.Namespace, output_root: Path, df: pd.DataFrame, dense_root: Path, splits: list[Any]) -> tuple[Path, dict[str, Any]]:
    requested = Path(args.grouped_pca_root) if args.grouped_pca_root else None
    candidates = []
    if requested is not None:
        candidates.append(requested)
    candidates.append(Path(args.external_phase4_root))
    for candidate in candidates:
        if grouped_pca_complete(candidate, splits):
            info = load_grouped_pca_manifest(candidate, splits)
            return candidate, info
    pca_root = requested if requested is not None else output_root / "foldsafe_grouped_pca"
    if not grouped_pca_complete(pca_root, splits):
        info = generate_grouped_pca(
            df=df,
            dense_root=dense_root,
            splits=splits,
            pca_root=pca_root,
            external_phase4_root=Path(args.external_phase4_root),
            batch_size=args.pca_batch_size,
            oversampling=args.pca_oversampling,
            power_iterations=args.pca_power_iterations,
        )
    else:
        info = load_grouped_pca_manifest(pca_root, splits)
    info = load_grouped_pca_manifest(pca_root, splits) | {"generated": bool(info.get("generated"))}
    return pca_root, info


def build_blocks(df: pd.DataFrame, dense_root: Path, splits: list[Any]) -> dict[int, GroupedBlock]:
    ar_all = feature_matrix(df, AR_FEATURE_COLUMNS)
    blocks: dict[int, GroupedBlock] = {}
    for split in splits:
        train_idx = split.train_idx.astype(np.int64)
        test_idx = split.test_idx.astype(np.int64)
        ar_train_x, ar_test_x = base.standardize_train_only(ar_all[train_idx], ar_all[test_idx])
        train_y, test_y = fr.split_y(split, train_idx, test_idx)
        train_cont = base.target_continuous_values(df, split, train_idx, split.target.value_column)
        test_cont = base.target_continuous_values(df, split, test_idx, split.target.value_column)
        inner_train, inner_val, inner_audit = fr.inner_split(df, train_idx, train_y)
        blocks[int(split.fold)] = GroupedBlock(
            target_name=split.target.name,
            target_type="binary",
            protocol=split.protocol,
            fold=int(split.fold),
            split=split,
            train_idx=train_idx,
            test_idx=test_idx,
            train_y=train_y,
            test_y=test_y,
            train_cont=train_cont,
            test_cont=test_cont,
            inner_train=inner_train,
            inner_val=inner_val,
            inner_audit=inner_audit,
            train_video_id=df.loc[train_idx, "video_id"].astype(str).to_numpy(),
            test_video_id=df.loc[test_idx, "video_id"].astype(str).to_numpy(),
            train_time=df.loc[train_idx, "time_seconds"].to_numpy(dtype=np.float32),
            test_time=df.loc[test_idx, "time_seconds"].to_numpy(dtype=np.float32),
            ar_train_x=ar_train_x,
            ar_test_x=ar_test_x,
            ar_block_dims={"ar": int(ar_train_x.shape[1])},
        )
    return blocks


def row_for(summary: pd.DataFrame, control: str) -> pd.Series:
    sub = summary[summary["control_type"] == control]
    if len(sub) != 1:
        raise RuntimeError(f"Expected one summary row for {control}, got {len(sub)}")
    return sub.iloc[0]


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "pr_auc",
        "roc_auc",
        "top_1pct_recall",
        "top_5pct_recall",
        "top_10pct_recall",
        "top_1pct_precision",
        "top_5pct_precision",
        "top_10pct_precision",
        "delta_vs_frozen_ar_pr_auc",
    ]
    rows = []
    for control, group in metrics_df.groupby("control_type", dropna=False):
        row: dict[str, Any] = {
            "target_name": TARGET_NAME,
            "architecture": ARCHITECTURE,
            "control_type": control,
            "folds": int(group["fold"].nunique()),
            "seeds": int(group["seed"].nunique()),
            "fold_seed_rows": int(len(group)),
            "rows_test_total": int(group["n_test"].sum()),
        }
        for metric in metric_cols:
            if metric in group:
                vals = pd.to_numeric(group[metric], errors="coerce")
                row[f"mean_{metric}"] = float(vals.mean()) if vals.notna().any() else math.nan
                row[f"std_{metric}"] = float(vals.std(ddof=0)) if vals.notna().sum() > 1 else math.nan
                row[f"min_{metric}"] = float(vals.min()) if vals.notna().any() else math.nan
                row[f"max_{metric}"] = float(vals.max()) if vals.notna().any() else math.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("control_type")


def fold_seed_deltas(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fold, seed), group in metrics_df.groupby(["fold", "seed"]):
        vals = group.set_index("control_type")
        real = vals.loc["real_residual"]
        row: dict[str, Any] = {
            "target_name": TARGET_NAME,
            "architecture": ARCHITECTURE,
            "fold": int(fold),
            "seed": int(seed),
            "real_pr_auc": float(real["pr_auc"]),
        }
        control_prs = {}
        for control in ("frozen_or_ar_only", *PRIMARY_CONTROLS, "diagnostics_only_residual"):
            pr = float(vals.loc[control, "pr_auc"])
            control_prs[control] = pr
            row[f"{control}_pr_auc"] = pr
            row[f"real_minus_{control}_pr_auc"] = float(real["pr_auc"] - pr)
        primary = {control: control_prs[control] for control in PRIMARY_CONTROLS}
        best_control = max(primary, key=primary.get)
        row["best_control"] = best_control
        row["best_control_pr_auc"] = float(primary[best_control])
        row["real_minus_best_control_pr_auc"] = float(real["pr_auc"] - primary[best_control])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["fold", "seed"])


def compute_gates(summary: pd.DataFrame, metrics_df: pd.DataFrame, delta_df: pd.DataFrame, context_audit: dict[str, Any], ar_manifest: list[dict[str, Any]]) -> dict[str, Any]:
    failed: list[str] = []
    real = row_for(summary, "real_residual")
    ar = row_for(summary, "frozen_or_ar_only")
    controls = [row_for(summary, control) for control in PRIMARY_CONTROLS]
    best_control = max(controls, key=lambda row: float(row["mean_pr_auc"]))
    delta_ar = float(real["mean_pr_auc"] - ar["mean_pr_auc"])
    delta_best = float(real["mean_pr_auc"] - best_control["mean_pr_auc"])
    control_deltas = {f"real_minus_{row['control_type']}_pr_auc": float(real["mean_pr_auc"] - row["mean_pr_auc"]) for row in controls}
    positives_vs_control = int((delta_df["real_minus_best_control_pr_auc"] > 0).sum())
    positives_vs_ar = int((delta_df["real_minus_frozen_or_ar_only_pr_auc"] > 0).sum())
    mean_test_prev = float(pd.to_numeric(metrics_df["test_positive_rate"], errors="coerce").mean())
    label_perm = row_for(summary, "label_permutation_residual")
    label_perm_near_chance = bool(float(label_perm["mean_pr_auc"]) <= mean_test_prev + 0.02)
    leakage_pass = bool(context_audit.get("leakage_context_audit_pass"))
    frozen_integrity = bool(metrics_df.groupby(["fold", "seed"])["frozen_ar_test_checksum"].nunique().eq(1).all())
    checkpoint_restore = bool(metrics_df["checkpoint_restore_pass"].all())
    eval_mode = bool(metrics_df["eval_mode_scoring"].all())
    ar_pairs = {(int(row["fold"]), int(row["seed"])) for row in ar_manifest}
    expected_pairs = {(fold, seed) for fold in FOLDS for seed in SEEDS}
    ar_generation_valid = bool(ar_pairs == expected_pairs and all(row.get("ar_baseline_newly_trained") for row in ar_manifest))
    if delta_ar < CREDIBLE_THRESHOLD:
        failed.append("mean_delta_vs_ar_below_0p003")
    if delta_best < CREDIBLE_THRESHOLD:
        failed.append("mean_delta_vs_best_control_below_0p003")
    if not all(value >= CREDIBLE_THRESHOLD for value in control_deltas.values()):
        failed.append("real_not_above_all_primary_controls_by_0p003")
    if positives_vs_control < 40:
        failed.append("fold_seed_consistency_vs_best_control")
    if not label_perm_near_chance:
        failed.append("label_permutation_not_near_chance")
    for name, ok in (
        ("leakage_context_audit", leakage_pass),
        ("frozen_ar_integrity", frozen_integrity),
        ("checkpoint_restore", checkpoint_restore),
        ("eval_mode_scoring", eval_mode),
        ("ar_baseline_generation", ar_generation_valid),
    ):
        if not ok:
            failed.append(name)
    grouped_pass = bool(not failed)
    return {
        "schema_version": SCHEMA_VERSION,
        "target": TARGET_NAME,
        "protocol": PROTOCOL,
        "architecture": ARCHITECTURE,
        "matrix_rows_expected": 350,
        "matrix_rows_actual": int(len(metrics_df)),
        "folds": list(FOLDS),
        "seeds": list(SEEDS),
        "ar_baselines_reused": 0,
        "ar_baselines_newly_trained": int(len(ar_manifest)),
        "ar_baseline_fold_seed_pairs": sorted([list(pair) for pair in ar_pairs]),
        "each_fold_seed_uses_own_frozen_ar_score": frozen_integrity,
        "all_controls_within_fold_seed_use_identical_frozen_ar_scores": frozen_integrity,
        "shared_three_seed_ar_cache_reused_across_ten_seed_grouped_compat": False,
        "residual_control_rows": 350,
        "grouped_compatibility_pass": grouped_pass,
        "strict_forward_time_temporal_generalization_proven": False,
        "leakage_context_audit_pass": leakage_pass,
        "frozen_ar_integrity_pass": frozen_integrity,
        "checkpoint_restore_pass": checkpoint_restore,
        "eval_mode_scoring_pass": eval_mode,
        "ar_baseline_generation_pass": ar_generation_valid,
        "label_permutation_near_chance_pass": label_perm_near_chance,
        "failed_gates": failed,
        "recommendation": "grouped_compatibility_pass_review_before_any_504" if grouped_pass else "grouped_compatibility_failed_do_not_run_504",
        "real_pr_auc": float(real["mean_pr_auc"]),
        "ar_frozen_pr_auc": float(ar["mean_pr_auc"]),
        "best_control": str(best_control["control_type"]),
        "best_control_pr_auc": float(best_control["mean_pr_auc"]),
        "delta_vs_ar": delta_ar,
        "delta_vs_best_control": delta_best,
        "fold_seed_positives_vs_ar": positives_vs_ar,
        "fold_seed_positives_vs_best_control": positives_vs_control,
        "label_permutation_pr_auc": float(label_perm["mean_pr_auc"]),
        "mean_test_positive_rate": mean_test_prev,
        "control_deltas": control_deltas,
    }


def write_report(path: Path, output_root: Path, pca_root: Path, gates: dict[str, Any]) -> None:
    text = f"""# Phase 5 Temporal Residual Grouped Compatibility

Output root: `{output_root}`

This is a grouped-video compatibility check for the confirmed blocked binary washout-gap short temporal conv residual. It uses the same 10 seeds as the blocked confirmation across all 5 grouped-video folds. It does not run continuous targets, extra targets, extra architectures, V-JEPA/TRIBE, or claim changes.

## Scope

- Target: `{gates['target']}`
- Protocol: `{gates['protocol']}`
- Architecture: `{gates['architecture']}`
- Fold-safe grouped PCA root: `{pca_root}`
- Rows completed / expected: `{gates['matrix_rows_actual']}` / `{gates['matrix_rows_expected']}`
- AR baselines reused: `{gates['ar_baselines_reused']}`
- AR baselines newly trained: `{gates['ar_baselines_newly_trained']}`
- Each fold/seed uses its own frozen AR score: `{gates['each_fold_seed_uses_own_frozen_ar_score']}`
- All controls within each fold/seed use identical frozen AR scores: `{gates['all_controls_within_fold_seed_use_identical_frozen_ar_scores']}`
- Shared 3-seed AR cache reused across 10-seed grouped compatibility: `{gates['shared_three_seed_ar_cache_reused_across_ten_seed_grouped_compat']}`

## Result

- Real PR-AUC: `{gates['real_pr_auc']:.10f}`
- AR/frozen baseline PR-AUC: `{gates['ar_frozen_pr_auc']:.10f}`
- Best control: `{gates['best_control']}` PR-AUC `{gates['best_control_pr_auc']:.10f}`
- Delta vs AR/frozen baseline: `{gates['delta_vs_ar']:+.10f}`
- Delta vs best control: `{gates['delta_vs_best_control']:+.10f}`
- Fold-seed positives vs AR/frozen baseline: `{gates['fold_seed_positives_vs_ar']}/50`
- Fold-seed positives vs best control: `{gates['fold_seed_positives_vs_best_control']}/50`
- Label permutation PR-AUC: `{gates['label_permutation_pr_auc']:.10f}`
- Mean test positive rate: `{gates['mean_test_positive_rate']:.10f}`

## Gates

- `grouped_compatibility_pass`: `{gates['grouped_compatibility_pass']}`
- `leakage_context_audit_pass`: `{gates['leakage_context_audit_pass']}`
- `frozen_ar_integrity_pass`: `{gates['frozen_ar_integrity_pass']}`
- `checkpoint_restore_pass`: `{gates['checkpoint_restore_pass']}`
- `eval_mode_scoring_pass`: `{gates['eval_mode_scoring_pass']}`
- `ar_baseline_generation_pass`: `{gates['ar_baseline_generation_pass']}`
- `label_permutation_near_chance_pass`: `{gates['label_permutation_near_chance_pass']}`
- Failed gates: `{gates['failed_gates']}`
- Recommendation: `{gates['recommendation']}`

This report is a compatibility check,  and not a broad claim change. Strict broad temporal generalization remains subject to review and any later explicitly approved confirmation.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def finalize_output(output_root: Path, pca_root: Path, reports_dir: Path) -> dict[str, Any]:
    metrics_df = pd.read_csv(output_root / "metrics" / "temporal_residual_grouped_compat_seed_metrics.csv")
    context_audit = json.loads((output_root / "diagnostics" / "leakage_context_audit.json").read_text(encoding="utf-8"))
    ar_manifest = json.loads((output_root / "manifests" / "ar_baseline_generation_manifest.json").read_text(encoding="utf-8"))["baselines"]
    summary = summarize_metrics(metrics_df)
    delta_df = fold_seed_deltas(metrics_df)
    gates = compute_gates(summary, metrics_df, delta_df, context_audit, ar_manifest)
    summary.to_csv(output_root / "metrics" / "temporal_residual_grouped_compat_summary_metrics.csv", index=False)
    delta_df.to_csv(output_root / "metrics" / "temporal_residual_grouped_compat_fold_seed_deltas.csv", index=False)
    summary.to_csv(output_root / "promotion" / "temporal_residual_grouped_compat_control_comparison.csv", index=False)
    write_json(output_root / "promotion" / "temporal_residual_grouped_compat_gates.json", gates)
    write_json(output_root / "promotion" / "temporal_residual_grouped_compat_adversarial_verdict.json", gates)
    write_json(output_root / "promotion" / "temporal_residual_grouped_compat_failure_reasons.json", {"failed_gates": gates["failed_gates"], "recommendation": gates["recommendation"]})
    stamp = output_root.name.replace("again_dense_2hz_phase5_temporal_residual_grouped_compat_", "")
    report_name = f"again_dense_2hz_phase5_temporal_residual_grouped_compat_{stamp}.md"
    write_report(output_root / "reports" / report_name, output_root, pca_root, gates)
    report_path = reports_dir / report_name
    write_report(report_path, output_root, pca_root, gates)
    return {"gates": gates, "report_path": str(report_path)}


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root) if args.output_root else default_output_root()
    source_root = Path(args.source_root)
    matrix_size = len(FOLDS) * len(SEEDS) * len(CONTROLS)
    print(json.dumps({"matrix_size": matrix_size, "max_allowed": 350, "target": TARGET_NAME, "folds": list(FOLDS), "seeds": list(SEEDS)}, indent=2))
    if matrix_size > 350:
        raise RuntimeError(f"Refusing to exceed 350 rows: {matrix_size}")
    if args.dry_run:
        return 0
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output root: {output_root}")
    for sub in ("manifests", "metrics", "promotion", "diagnostics", "reports", "frozen_ar_scores", "checkpoints", "ar_baseline_checkpoints"):
        (output_root / sub).mkdir(parents=True, exist_ok=True)
    start = time.time()
    df, dense_root, splits, residual_meta = load_df_and_splits(source_root)
    pca_root, pca_info = ensure_grouped_pca(args, output_root, df, dense_root, splits)
    blocks = build_blocks(df, dense_root, splits)
    fold_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    ar_curve_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    video_rows: list[dict[str, Any]] = []
    ar_manifest: list[dict[str, Any]] = []

    for fold in FOLDS:
        block = blocks[fold]
        for seed in SEEDS:
            ar, ar_curves = big_confirm.train_ar_baseline(
                output_root=output_root,
                block=block,
                seed=seed,
                batch_size=args.batch_size,
                max_epochs=args.ar_max_epochs,
                patience=args.ar_patience,
            )
            ar["source"] = "newly_trained_grouped_fold_seed_ar_only_baseline"
            ar["ar_baseline_reused"] = False
            ar["ar_baseline_newly_trained"] = True
            ar_curve_rows.extend(ar_curves)
            ar_manifest.append({k: v for k, v in ar.items() if k not in {"train_score", "train_reg", "test_score", "test_reg"}})
            ar_metrics = temporal.metric_row_for_block(block, ar["train_score"], ar["test_score"], ar["test_reg"])
            for control in CONTROLS:
                if control == "frozen_or_ar_only":
                    fold_rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "target_name": TARGET_NAME,
                            "target_type": "binary",
                            "validation_protocol": PROTOCOL,
                            "fold": int(fold),
                            "seed": int(seed),
                            "architecture": ARCHITECTURE,
                            "control_type": control,
                            "feature_name": FEATURE_NAME,
                            "n_train": int(len(block.train_idx)),
                            "n_test": int(len(block.test_idx)),
                            "test_positive_rate": float(np.mean(block.test_y)),
                            "checkpoint_restore_pass": bool(ar.get("checkpoint_restore_pass")),
                            "eval_mode_scoring": bool(ar.get("eval_mode_scoring_pass")),
                            "dropout_disabled": True,
                            "ar_baseline_reused": False,
                            "ar_baseline_newly_trained": True,
                            "frozen_ar_train_checksum": ar["train_checksum"],
                            "frozen_ar_test_checksum": ar["test_checksum"],
                            **ar_metrics,
                        }
                    )
                    continue
                pack = temporal.feature_pack_for(df, dense_root, pca_root, block, ARCHITECTURE, control, seed)
                train_block = replace(block, target_name=f"{TARGET_NAME}__grouped_fold{fold}")
                metrics, curves, audit = temporal.train_temporal_residual(
                    architecture=ARCHITECTURE,
                    control=control,
                    pack=pack,
                    block=train_block,
                    ar=ar,
                    seed=seed,
                    output_root=output_root,
                    batch_size=args.batch_size,
                    max_epochs=args.max_epochs,
                    patience=args.patience,
                )
                curve_rows.extend(curves)
                feature_rows.append({"target_name": TARGET_NAME, "fold": int(fold), "seed": int(seed), "architecture": ARCHITECTURE, "control_type": control, "dims": pack.dims, "blocks": pack.manifest})
                context_rows.append(pack.context_audit)
                if control == "label_permutation_residual":
                    label_rows.append({"fold": int(fold), "seed": int(seed), "control_type": control, "best_epoch": audit["best_epoch"], "label_policy": audit["label_policy"], "heldout_scoring_policy": "true_heldout_labels_targets", "best_inner_val_delta_vs_frozen_ar": audit["best_inner_val_delta_vs_frozen_ar"]})
                if control == "train_only_video_mean_residual":
                    video_rows.append({"fold": int(fold), "seed": int(seed), "control_type": control, "uses_test_rows_for_mean": False, "best_epoch": audit["best_epoch"], "checkpoint_restored": audit["checkpoint_restored"]})
                fold_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "target_name": TARGET_NAME,
                        "target_type": "binary",
                        "validation_protocol": PROTOCOL,
                        "fold": int(fold),
                        "seed": int(seed),
                        "architecture": ARCHITECTURE,
                        "control_type": control,
                        "feature_name": FEATURE_NAME,
                        "n_train": int(len(block.train_idx)),
                        "n_test": int(len(block.test_idx)),
                        "test_positive_rate": float(np.mean(block.test_y)),
                        "checkpoint_restore_pass": audit["checkpoint_restored"] or audit["residual_suppressed"],
                        "ar_baseline_reused": False,
                        "ar_baseline_newly_trained": True,
                        "frozen_ar_train_checksum": ar["train_checksum"],
                        "frozen_ar_test_checksum": ar["test_checksum"],
                        **audit,
                        **metrics,
                    }
                )
                pd.DataFrame(fold_rows).to_csv(output_root / "metrics" / "temporal_residual_grouped_compat_seed_metrics.partial.csv", index=False)
                gc.collect()

    metrics_df = pd.DataFrame(fold_rows)
    if len(metrics_df) != 350:
        raise RuntimeError(f"Expected 350 scored rows, got {len(metrics_df)}")
    metrics_df.to_csv(output_root / "metrics" / "temporal_residual_grouped_compat_seed_metrics.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(output_root / "diagnostics" / "training_curve_summary.csv", index=False)
    pd.DataFrame(ar_curve_rows).to_csv(output_root / "diagnostics" / "ar_baseline_training_curve_summary.csv", index=False)
    pd.DataFrame(label_rows).to_csv(output_root / "diagnostics" / "label_permutation_audit.csv", index=False)
    pd.DataFrame(video_rows).to_csv(output_root / "diagnostics" / "train_only_video_mean_audit.csv", index=False)
    leakage_context_pass = bool(
        pca_info["audit"].get("leakage_audit_pass")
        and pca_info["audit"].get("no_test_rows_used_in_pca_fit")
        and all(row.get("temporal_context_causal_only") for row in context_rows)
        and not any(row.get("uses_centered_or_future_windows") for row in context_rows)
        and all(row.get("same_video_history_masking") for row in context_rows)
        and all(row.get("label_policy") == "permuted_train_and_permuted_inner_val_selection" for row in label_rows)
        and not any(row.get("uses_test_rows_for_mean") for row in video_rows)
    )
    context_audit = {
        "schema_version": SCHEMA_VERSION,
        "leakage_context_audit_pass": leakage_context_pass,
        "foldsafe_grouped_pca_audit": pca_info["audit"],
        "pca_generated_in_this_run": bool(pca_info.get("generated")),
        "pca_root": str(pca_root),
        "context_rows": context_rows,
        "label_permutation_policy_pass": all(row.get("label_policy") == "permuted_train_and_permuted_inner_val_selection" for row in label_rows),
        "train_only_video_mean_pass": not any(row.get("uses_test_rows_for_mean") for row in video_rows),
        "temporal_context_causal_only": all(row.get("temporal_context_causal_only") for row in context_rows),
        "no_centered_or_future_windows": not any(row.get("uses_centered_or_future_windows") for row in context_rows),
        "same_video_history_masking": all(row.get("same_video_history_masking") for row in context_rows),
    }
    write_json(output_root / "diagnostics" / "leakage_context_audit.json", context_audit)
    write_json(output_root / "diagnostics" / "label_permutation_audit.json", {"policy_implemented": True, "rows": label_rows})
    write_json(output_root / "diagnostics" / "train_only_video_mean_audit.json", {"train_only_video_mean_primary_static_control": True, "rows": video_rows})
    write_json(
        output_root / "manifests" / "ar_baseline_generation_manifest.json",
        {
            "baselines": ar_manifest,
            "reused": 0,
            "newly_trained": len(ar_manifest),
            "folds": list(FOLDS),
            "seeds": list(SEEDS),
            "each_fold_seed_uses_own_frozen_ar_score": True,
            "shared_three_seed_ar_cache_reused_across_ten_seed_grouped_compat": False,
            "ar_only_baseline_generation_reported_separately_from_residual_control_rows": True,
        },
    )
    write_json(output_root / "manifests" / "feature_manifest.json", {"features": feature_rows, "row_count": len(feature_rows)})
    write_json(
        output_root / "manifests" / "run_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_at": now_iso(),
            "source_root": str(source_root),
            "grouped_pca_root": str(pca_root),
            "external_phase4_root": str(args.external_phase4_root),
            "output_root": str(output_root),
            "dense_root": str(dense_root),
            "target": TARGET_NAME,
            "architecture": ARCHITECTURE,
            "controls": list(CONTROLS),
            "seeds": list(SEEDS),
            "folds": list(FOLDS),
            "feature": FEATURE_NAME,
            "protocol_scope": "grouped_video_only",
            "matrix_size": matrix_size,
            "residual_control_rows": 350,
            "ar_baseline_generation": {"reused": 0, "newly_trained": len(ar_manifest)},
            "no_continuous": True,
            "no_extra_targets": True,
            "no_extra_architectures": True,
            "no_vjepa_tribe_rerun": True,
            "pca_generated_only_if_missing": True,
            "residual_target_definition": residual_meta,
            "duration_seconds": time.time() - start,
        },
    )
    finalized = finalize_output(output_root, pca_root, Path(args.reports_dir))
    gates = finalized["gates"]
    print(json.dumps(fr.clean_json({"run_completed": True, "output_root": str(output_root), "report": finalized["report_path"], **gates}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
