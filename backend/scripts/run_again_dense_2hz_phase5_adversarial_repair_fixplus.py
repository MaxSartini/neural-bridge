"""Canonical Phase 5 adversarial repair + fix-plus retrain.

This runner consumes the existing dense AGAIN 2Hz cache and Phase 4 fold-safe
PCA artifacts only. It does not run V-JEPA, TRIBE, global PCA, or modify the
canonical Phase 4/Phase 5 outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.scripts import run_again_dense_2hz_phase5_learned_heads as base
from backend.scripts.again_dense_2hz_benchmark import (
    AR_FEATURE_COLUMNS,
    DEFAULT_DENSE_ROOT,
    QUALITY_FEATURE_COLUMNS,
    TIME_FEATURE_COLUMNS,
    clean_json,
    feature_matrix,
    load_or_build_temporal_diagnostic_features,
    write_json,
)
from backend.scripts.again_dense_2hz_phase4_pca_bridge import array_digest, split_fingerprint


REPAIR_SCHEMA_VERSION = "again_dense_2hz_phase5_adversarial_repair_fixplus_v1"
DEFAULT_REPAIR_CONTROLS = (
    "ar_only_head",
    "pca_only_real",
    "pca_only_shuffled",
    "pca_only_random",
    "ar_plus_shuffled_pca",
    "ar_plus_random_pca",
    "diag_only",
    "shuffled_diag_only",
    "time_only",
    "quality_only",
    "video_mean_pca_oracle_diagnostic",
    "label_permutation",
)
REPAIR_MODELS = ("gated_ar_pca_mlp",)
REPAIR_LOSSES = ("regression", "binary", "regression_plus_binary")
REPAIR_SEEDS = (20260625, 20260626, 20260627)
REPAIR_GRADIENT_CLIP = 1.0
INNER_AUDIT_ROWS: list[dict[str, Any]] = []
WITHIN_VIDEO_ROWS: list[dict[str, Any]] = []
CURRENT_LABELS_DF: pd.DataFrame | None = None


_orig_load_labels = base.load_labels


def load_labels_capture(dense_root: Path) -> pd.DataFrame:
    global CURRENT_LABELS_DF
    df = _orig_load_labels(dense_root)
    CURRENT_LABELS_DF = df
    return df


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _spearman(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(scores) == 0:
        return math.nan
    return float(np.corrcoef(_rank(y_true), _rank(scores))[0, 1])


def _ndcg(relevance: np.ndarray, scores: np.ndarray) -> float:
    if len(relevance) == 0:
        return math.nan
    rel = np.asarray(relevance, dtype=np.float64)
    rel = rel - min(0.0, float(np.nanmin(rel)))
    if float(np.nansum(rel)) <= 0:
        return math.nan
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="mergesort")
    ideal = np.argsort(-rel, kind="mergesort")
    discounts = 1.0 / np.log2(np.arange(2, len(rel) + 2, dtype=np.float64))
    dcg = float(np.sum(rel[order] * discounts))
    idcg = float(np.sum(rel[ideal] * discounts))
    return dcg / idcg if idcg > 0 else math.nan


def _top_fraction_metrics(y_binary: np.ndarray, y_cont: np.ndarray, scores: np.ndarray, frac: float) -> dict[str, Any]:
    if len(scores) == 0:
        return {}
    k = max(1, int(math.ceil(len(scores) * frac)))
    top_idx = np.argsort(-scores, kind="mergesort")[:k]
    base_rate = float(np.mean(y_binary)) if len(y_binary) else math.nan
    precision = float(np.mean(y_binary[top_idx])) if k else math.nan
    recall = float(np.sum(y_binary[top_idx]) / np.sum(y_binary)) if np.sum(y_binary) > 0 else math.nan
    movement = float(np.mean(y_cont[top_idx])) if k else math.nan
    baseline_movement = float(np.mean(y_cont)) if len(y_cont) else math.nan
    return {
        f"top_{int(frac * 100)}pct_precision": precision,
        f"top_{int(frac * 100)}pct_recall": recall,
        f"top_{int(frac * 100)}pct_lift": precision / base_rate if base_rate and math.isfinite(base_rate) else math.nan,
        f"top_{int(frac * 100)}pct_avg_true_movement": movement,
        f"top_{int(frac * 100)}pct_avg_true_movement_lift": movement - baseline_movement
        if math.isfinite(movement) and math.isfinite(baseline_movement)
        else math.nan,
        f"top_{int(frac * 100)}pct_n": int(k),
    }


def ranking_metrics(y_binary: np.ndarray, y_cont: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {
        "spearman_future_movement": _spearman(y_cont, scores),
        "ndcg_binary": _ndcg(y_binary.astype(float), scores),
        "ndcg_future_movement": _ndcg(y_cont, scores),
    }
    for frac in (0.01, 0.05, 0.10):
        out.update(_top_fraction_metrics(y_binary, y_cont, scores, frac))
    return out


def temporal_inner_validation_relative_split(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    y_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    groups = df.loc[train_idx, "video_id"].astype(str).to_numpy()
    times = df.loc[train_idx, "time_seconds"].to_numpy(dtype=np.float64)
    inner_train_parts: list[np.ndarray] = []
    inner_val_parts: list[np.ndarray] = []
    for group in np.unique(groups):
        rel = np.flatnonzero(groups == group)
        rel = rel[np.argsort(times[rel], kind="mergesort")]
        if len(rel) < 6:
            continue
        cutoff = max(1, min(len(rel) - 1, int(math.floor(len(rel) * 0.80))))
        inner_train_parts.append(rel[:cutoff])
        inner_val_parts.append(rel[cutoff:])
    if inner_train_parts and inner_val_parts:
        inner_train = np.concatenate(inner_train_parts).astype(np.int64)
        inner_val = np.concatenate(inner_val_parts).astype(np.int64)
        if len(np.unique(y_train[inner_train])) >= 2 and len(np.unique(y_train[inner_val])) >= 2:
            strict = True
            for group in np.unique(groups):
                tr = inner_train[groups[inner_train] == group]
                va = inner_val[groups[inner_val] == group]
                if len(tr) and len(va) and not (np.max(times[tr]) < np.min(times[va])):
                    strict = False
                    break
            return inner_train, inner_val, {
                "inner_validation_strategy": "inner_temporal_within_outer_train_by_video_80_20",
                "inner_overlap_rows": int(np.intersect1d(inner_train, inner_val).size),
                "inner_strict_forward_time": bool(strict),
                "inner_train_rows": int(len(inner_train)),
                "inner_val_rows": int(len(inner_val)),
            }
    order = np.argsort(times, kind="mergesort")
    cutoff = max(1, min(len(order) - 1, int(math.floor(len(order) * 0.80))))
    inner_train = order[:cutoff].astype(np.int64)
    inner_val = order[cutoff:].astype(np.int64)
    if len(np.unique(y_train[inner_train])) >= 2 and len(np.unique(y_train[inner_val])) >= 2:
        return inner_train, inner_val, {
            "inner_validation_strategy": "inner_temporal_global_outer_train_80_20",
            "inner_overlap_rows": int(np.intersect1d(inner_train, inner_val).size),
            "inner_strict_forward_time": bool(np.max(times[inner_train]) < np.min(times[inner_val])),
            "inner_train_rows": int(len(inner_train)),
            "inner_val_rows": int(len(inner_val)),
        }
    rel = np.arange(len(train_idx), dtype=np.int64)
    return rel, rel, {
        "inner_validation_strategy": "inner_fallback_train_resubstitution",
        "inner_overlap_rows": int(len(rel)),
        "inner_strict_forward_time": False,
        "inner_train_rows": int(len(rel)),
        "inner_val_rows": int(len(rel)),
    }


def control_metadata(control: str | None) -> dict[str, Any]:
    control_name = control or "real_ar_pca_diag"
    pca_control_type = "none"
    diag_control_type = "none"
    uses_ar = control_name in {
        "real_ar_pca_diag",
        "ar_only_head",
        "ar_plus_shuffled_pca",
        "ar_plus_random_pca",
        "label_permutation",
    }
    uses_pca = control_name in {
        "real_ar_pca_diag",
        "pca_only_real",
        "pca_only_shuffled",
        "pca_only_random",
        "ar_plus_shuffled_pca",
        "ar_plus_random_pca",
        "video_mean_pca_oracle_diagnostic",
        "label_permutation",
    }
    uses_diag = control_name in {"real_ar_pca_diag", "ar_plus_shuffled_pca", "ar_plus_random_pca", "diag_only", "shuffled_diag_only", "label_permutation"}
    if control_name in {"real_ar_pca_diag", "pca_only_real", "label_permutation"}:
        pca_control_type = "real"
    elif control_name in {"pca_only_shuffled", "ar_plus_shuffled_pca"}:
        pca_control_type = "shuffled"
    elif control_name in {"pca_only_random", "ar_plus_random_pca"}:
        pca_control_type = "random"
    elif control_name == "video_mean_pca_oracle_diagnostic":
        pca_control_type = "video_mean_oracle"
    if control_name in {"real_ar_pca_diag", "ar_plus_shuffled_pca", "ar_plus_random_pca", "diag_only", "label_permutation"}:
        diag_control_type = "real"
    elif control_name == "shuffled_diag_only":
        diag_control_type = "shuffled"
    matched = control_name in {"ar_plus_shuffled_pca", "ar_plus_random_pca"}
    return {
        "control_type": control_name,
        "uses_ar_feature_block": bool(uses_ar),
        "uses_pca_feature_block": bool(uses_pca),
        "uses_temporal_diagnostics_block": bool(uses_diag),
        "pca_control_type": pca_control_type,
        "diag_control_type": diag_control_type,
        "label_permutation": bool(control_name == "label_permutation"),
        "oracle_diagnostic": bool(control_name == "video_mean_pca_oracle_diagnostic"),
        "matched_control": bool(matched),
    }


def row_dict_base_repair(
    split: Any,
    feature: str,
    model: str,
    loss: str,
    seed: int | None,
    control: str | None,
    n_train: int,
    n_test: int,
    feature_width: int,
) -> dict[str, Any]:
    row = base._ORIGINAL_ROW_DICT_BASE(split, feature, model, loss, seed, control, n_train, n_test, feature_width)
    row["schema_version"] = REPAIR_SCHEMA_VERSION
    row.update(control_metadata(control))
    return row


def assemble_feature_blocks_repair(
    df: pd.DataFrame,
    dense_root: Path,
    phase4_root: Path,
    split: Any,
    spec: Any,
    *,
    include_ar: bool,
    include_temporal_diagnostics: bool,
    control: str | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int], list[dict[str, Any]]]:
    train_idx, test_idx, pca_train, pca_test, source_path = base.load_phase4_scores(df, phase4_root, split, spec)
    control_name = control or "real_ar_pca_diag"
    blocks_train: list[np.ndarray] = []
    blocks_test: list[np.ndarray] = []
    block_dims: dict[str, int] = {}
    feature_manifest: list[dict[str, Any]] = []
    meta = control_metadata(control)

    def add_ar() -> None:
        ar = feature_matrix(df, AR_FEATURE_COLUMNS)
        blocks_train.append(ar[train_idx])
        blocks_test.append(ar[test_idx])
        block_dims["ar"] = ar.shape[1]
        feature_manifest.append({"block": "ar", "columns": list(AR_FEATURE_COLUMNS), "width": ar.shape[1], **meta})

    def add_diag(shuffle: bool = False) -> None:
        diag = load_or_build_temporal_diagnostic_features(dense_root, df)
        d_train = diag[train_idx].copy()
        d_test = diag[test_idx].copy()
        if shuffle:
            d_train = d_train[rng.permutation(len(d_train))]
            d_test = d_test[rng.permutation(len(d_test))]
        blocks_train.append(d_train)
        blocks_test.append(d_test)
        block_dims["diagnostics"] = d_train.shape[1]
        feature_manifest.append(
            {
                "block": "temporal_diagnostics",
                "source": str(dense_root / "_derived" / "temporal_diagnostics_summary_features.npy"),
                "width": d_train.shape[1],
                **meta,
            }
        )

    def add_pca(kind: str) -> None:
        p_train = pca_train.copy()
        p_test = pca_test.copy()
        if kind == "shuffled":
            p_train = p_train[rng.permutation(len(p_train))]
            p_test = p_test[rng.permutation(len(p_test))]
        elif kind == "random":
            p_train = rng.normal(0, 1, size=p_train.shape).astype(np.float32)
            p_test = rng.normal(0, 1, size=p_test.shape).astype(np.float32)
        elif kind == "video_mean_oracle":
            all_idx = np.concatenate([train_idx, test_idx])
            all_pca = np.concatenate([pca_train, pca_test], axis=0)
            videos = df.loc[all_idx, "video_id"].astype(str).to_numpy()
            means = {video: all_pca[videos == video].mean(axis=0) for video in np.unique(videos)}
            p_train = np.vstack([means[v] for v in df.loc[train_idx, "video_id"].astype(str).to_numpy()]).astype(np.float32)
            p_test = np.vstack([means[v] for v in df.loc[test_idx, "video_id"].astype(str).to_numpy()]).astype(np.float32)
        blocks_train.append(p_train)
        blocks_test.append(p_test)
        block_dims["pca"] = p_train.shape[1]
        feature_manifest.append(
            {
                "block": "phase4_fold_safe_pca",
                "feature_name": spec.name,
                "source_family": spec.source_family,
                "source_path": str(source_path),
                "source_checksum": base.file_digest(source_path),
                "width": p_train.shape[1],
                **meta,
            }
        )

    if control_name == "ar_only_head":
        add_ar()
    elif control_name == "pca_only_real":
        add_pca("real")
    elif control_name == "pca_only_shuffled":
        add_pca("shuffled")
    elif control_name == "pca_only_random":
        add_pca("random")
    elif control_name == "ar_plus_shuffled_pca":
        add_ar()
        add_pca("shuffled")
        add_diag()
    elif control_name == "ar_plus_random_pca":
        add_ar()
        add_pca("random")
        add_diag()
    elif control_name == "diag_only":
        add_diag()
    elif control_name == "shuffled_diag_only":
        add_diag(shuffle=True)
    elif control_name == "time_only":
        time_x = feature_matrix(df, TIME_FEATURE_COLUMNS)
        blocks_train.append(time_x[train_idx])
        blocks_test.append(time_x[test_idx])
        block_dims["diagnostics"] = time_x.shape[1]
        feature_manifest.append({"block": "time", "columns": list(TIME_FEATURE_COLUMNS), "width": time_x.shape[1], **meta})
    elif control_name == "quality_only":
        q = feature_matrix(df, QUALITY_FEATURE_COLUMNS)
        blocks_train.append(q[train_idx])
        blocks_test.append(q[test_idx])
        block_dims["diagnostics"] = q.shape[1]
        feature_manifest.append({"block": "quality_motion_luma", "columns": list(QUALITY_FEATURE_COLUMNS), "width": q.shape[1], **meta})
    elif control_name == "video_mean_pca_oracle_diagnostic":
        add_pca("video_mean_oracle")
    elif control_name in {"real_ar_pca_diag", "label_permutation"}:
        add_ar()
        add_pca("real")
        add_diag()
    else:
        raise ValueError(f"Unsupported repair control: {control_name}")

    if not blocks_train:
        raise ValueError(f"No feature blocks assembled for control={control}")
    return (
        train_idx,
        test_idx,
        np.concatenate(blocks_train, axis=1).astype(np.float32, copy=False),
        np.concatenate(blocks_test, axis=1).astype(np.float32, copy=False),
        block_dims,
        feature_manifest,
    )


def train_mlx_head_repair(
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_y_binary: np.ndarray,
    test_y_binary: np.ndarray,
    train_y_continuous: np.ndarray,
    test_y_continuous: np.ndarray,
    df: pd.DataFrame,
    train_idx: np.ndarray,
    config: Any,
    block_dims: dict[str, int],
    checkpoint_dir: Path,
    run_key: str,
) -> Any:
    base.require_mlx()
    base.mx.random.seed(int(config.seed))
    train_x, test_x = base.standardize_train_only(train_x, test_x)
    inner_train, inner_val, audit = temporal_inner_validation_relative_split(df, train_idx, train_y_binary)
    model = base.make_model(config, train_x.shape[1], block_dims)
    optimizer = (
        base.optim.AdamW(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
        if config.weight_decay > 0
        else base.optim.Adam(learning_rate=config.learning_rate)
    )
    optimizer_name = "mlx.optimizers.AdamW" if config.weight_decay > 0 else "mlx.optimizers.Adam"
    batch_size = max(128, int(config.batch_size))
    rng = np.random.default_rng(config.seed)
    curves: list[dict[str, Any]] = []
    best_val = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / f"{run_key}__best.npz"

    def loss_fn(model_obj: Any, xb: Any, yb: Any, yr: Any) -> Any:
        out = model_obj(xb)
        if out.ndim == 1:
            out = out[:, None]
        if config.loss_name == "binary":
            return base.mx.mean(base.nn.losses.binary_cross_entropy(out[:, :1], yb, with_logits=True))
        if config.loss_name == "regression":
            return base.mx.mean(base.nn.losses.huber_loss(out[:, :1], yr, delta=1.0))
        reg = base.mx.mean(base.nn.losses.huber_loss(out[:, :1], yr, delta=1.0))
        bce = base.mx.mean(base.nn.losses.binary_cross_entropy(out[:, 1:2], yb, with_logits=True))
        return reg + float(config.lambda_binary) * bce

    loss_and_grad = base.nn.value_and_grad(model, loss_fn)
    final_val_pr = math.nan
    final_train_loss = math.nan
    early_stopping_reason = "max_epochs_reached"
    for epoch in range(1, int(config.max_epochs) + 1):
        order = rng.permutation(inner_train)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, len(order), batch_size):
            rel = order[start : start + batch_size]
            xb = base.mx.array(train_x[rel], dtype=base.mx.float32)
            yb = base.mx.array(train_y_binary[rel].astype(np.float32)[:, None], dtype=base.mx.float32)
            yr = base.mx.array(train_y_continuous[rel].astype(np.float32)[:, None], dtype=base.mx.float32)
            loss, grads = loss_and_grad(model, xb, yb, yr)
            if REPAIR_GRADIENT_CLIP > 0:
                grads, _grad_norm = base.optim.clip_grad_norm(grads, REPAIR_GRADIENT_CLIP)
            optimizer.update(model, grads)
            base.mx.eval(model.parameters(), optimizer.state)
            epoch_loss += float(np.asarray(loss))
            batches += 1
        val_out = model(base.mx.array(train_x[inner_val], dtype=base.mx.float32))
        base.mx.eval(val_out)
        val_score, _ = base.select_score_columns(np.asarray(val_out, dtype=np.float32), config.loss_name)
        final_val_pr = average_precision_score(train_y_binary[inner_val], val_score) if len(np.unique(train_y_binary[inner_val])) > 1 else math.nan
        final_train_loss = epoch_loss / max(1, batches)
        row = {
            "epoch": epoch,
            "train_loss": final_train_loss,
            "inner_validation_pr_auc": final_val_pr,
            **audit,
        }
        curves.append(row)
        if math.isfinite(final_val_pr) and final_val_pr > best_val:
            best_val = float(final_val_pr)
            best_epoch = epoch
            model.save_weights(str(best_checkpoint_path))
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= int(config.patience):
            early_stopping_reason = "patience_exhausted"
            break

    if not best_checkpoint_path.exists():
        raise RuntimeError("Best checkpoint was not saved; refusing to score unrepaired Phase 5 model.")
    model.load_weights(str(best_checkpoint_path))
    train_out = model(base.mx.array(train_x, dtype=base.mx.float32))
    test_out = model(base.mx.array(test_x, dtype=base.mx.float32))
    val_out = model(base.mx.array(train_x[inner_val], dtype=base.mx.float32))
    base.mx.eval(train_out, test_out, val_out)
    train_scores, train_reg = base.select_score_columns(np.asarray(train_out, dtype=np.float32), config.loss_name)
    test_scores, test_reg = base.select_score_columns(np.asarray(test_out, dtype=np.float32), config.loss_name)
    restored_val_score, _ = base.select_score_columns(np.asarray(val_out, dtype=np.float32), config.loss_name)
    restored_val_pr = (
        average_precision_score(train_y_binary[inner_val], restored_val_score)
        if len(np.unique(train_y_binary[inner_val])) > 1
        else math.nan
    )
    checkpoint_checksum = base.file_digest(best_checkpoint_path)
    overfit_flag = bool(math.isfinite(best_val) and math.isfinite(final_val_pr) and final_val_pr < best_val - 0.0025)
    audit_row = {
        "run_key": run_key,
        "seed": int(config.seed),
        "model": config.model_name,
        "loss": config.loss_name,
        "checkpoint_restored_for_test_scoring": True,
        "checkpoint_path": str(best_checkpoint_path),
        "checkpoint_checksum": checkpoint_checksum,
        "best_epoch": int(best_epoch),
        "epochs_run": int(len(curves)),
        "best_inner_validation_pr_auc": best_val,
        "final_inner_validation_pr_auc": final_val_pr,
        "restored_inner_validation_pr_auc": restored_val_pr,
        "early_stopping_reason": early_stopping_reason,
        "overfit_flag": overfit_flag,
        **audit,
    }
    INNER_AUDIT_ROWS.append(audit_row)
    return base.TrainResult(
        train_scores=train_scores,
        test_scores=test_scores,
        train_regression_scores=train_reg,
        test_regression_scores=test_reg,
        config={
            "backend": "mlx",
            "optimizer": optimizer_name,
            "gradient_clip": REPAIR_GRADIENT_CLIP,
            **audit_row,
        },
        curves=curves,
        checkpoint_path=str(best_checkpoint_path),
        checkpoint_checksum=checkpoint_checksum,
    )


def train_head_repair(backend: str, *args: Any, **kwargs: Any) -> Any:
    if backend != "mlx":
        raise ValueError("Repair runner requires MLX backend; CPU/MPS fallback is refused.")
    return train_mlx_head_repair(*args, **kwargs)


def metric_records_from_scores_repair(
    split: Any,
    feature: str,
    model: str,
    loss: str,
    seed: int | None,
    control: str | None,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    train_y: np.ndarray,
    test_y: np.ndarray,
    train_cont: np.ndarray,
    test_cont: np.ndarray,
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    train_reg_scores: np.ndarray,
    test_reg_scores: np.ndarray,
    feature_width: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = base._ORIGINAL_METRIC_RECORDS_FROM_SCORES(
        split,
        feature,
        model,
        loss,
        seed,
        control,
        train_idx,
        test_idx,
        train_y,
        test_y,
        train_cont,
        test_cont,
        train_scores,
        test_scores,
        train_reg_scores,
        test_reg_scores,
        feature_width,
        extra,
    )
    row["schema_version"] = REPAIR_SCHEMA_VERSION
    row.update(control_metadata(control))
    row.update(ranking_metrics(test_y.astype(int), test_cont.astype(float), test_scores.astype(float)))
    if len(test_cont) and len(test_reg_scores):
        row["continuous_rmse"] = float(math.sqrt(mean_squared_error(test_cont, test_reg_scores)))
        row["continuous_spearman"] = _spearman(test_cont, test_reg_scores)
        row["continuous_regression_ndcg"] = _ndcg(test_cont, test_reg_scores)
    if CURRENT_LABELS_DF is not None and len(test_idx):
        video_ids = CURRENT_LABELS_DF.loc[test_idx, "video_id"].astype(str).to_numpy()
        per_video_rows: list[dict[str, Any]] = []
        for video_id in np.unique(video_ids):
            mask = video_ids == video_id
            if int(mask.sum()) < 3:
                continue
            yv = test_y[mask].astype(int)
            cv = test_cont[mask].astype(float)
            sv = test_scores[mask].astype(float)
            rv = test_reg_scores[mask].astype(float)
            video_row = {
                "target_name": split.target.name,
                "validation_protocol": split.protocol,
                "fold": split.fold,
                "feature_name": feature,
                "model_head": model,
                "loss_name": loss,
                "seed": seed,
                "video_id": video_id,
                "n_rows": int(mask.sum()),
                "event_count": int(np.sum(yv)),
                **control_metadata(control),
                **ranking_metrics(yv, cv, sv),
                "continuous_mae": float(mean_absolute_error(cv, rv)) if len(cv) else math.nan,
                "continuous_rmse": float(math.sqrt(mean_squared_error(cv, rv))) if len(cv) else math.nan,
                "continuous_spearman": _spearman(cv, rv),
            }
            if len(np.unique(yv)) > 1:
                video_row["pr_auc"] = float(average_precision_score(yv, sv))
                video_row["roc_auc"] = float(roc_auc_score(yv, sv))
            else:
                video_row["pr_auc"] = math.nan
                video_row["roc_auc"] = math.nan
            per_video_rows.append(video_row)
            WITHIN_VIDEO_ROWS.append(video_row)
        if per_video_rows:
            for metric in ("pr_auc", "top_1pct_recall", "top_5pct_recall", "top_10pct_recall", "spearman_future_movement"):
                values = [float(v.get(metric, math.nan)) for v in per_video_rows if math.isfinite(float(v.get(metric, math.nan)))]
                row[f"within_video_macro_{metric}"] = float(np.mean(values)) if values else math.nan
            row["within_video_video_count"] = int(len(per_video_rows))
    return row


def promotion_gates_repair(summary: pd.DataFrame, phase4_root: Path, target_name: str) -> dict[str, Any]:
    gates: dict[str, Any] = {
        "schema_version": REPAIR_SCHEMA_VERSION,
        "gate_name": "corrected_matched_control_gate",
        "target_name": target_name,
        "real_control_type": "real_ar_pca_diag",
        "matched_control_types": ["ar_plus_shuffled_pca", "ar_plus_random_pca"],
        "holy_shit_pass_retired": True,
        "recommendation": "insufficient_evidence",
    }
    if summary.empty:
        gates["failure"] = "no_summary_rows"
        return gates
    real = summary[(summary["target_name"] == target_name) & (summary["control_type"] == "real_ar_pca_diag")]
    if real.empty:
        gates["failure"] = "missing_real_ar_pca_diag"
        return gates
    real = real.sort_values("mean_pr_auc", ascending=False)
    best = real.iloc[0].to_dict()
    gates["best_real"] = clean_json(best)
    matched = summary[
        (summary["target_name"] == target_name)
        & (summary["feature_name"] == best["feature_name"])
        & (summary["model_head"] == best["model_head"])
        & (summary["loss_name"] == best["loss_name"])
        & (summary["control_type"].isin(gates["matched_control_types"]))
    ].copy()
    for protocol in ("grouped_video", "blocked_temporal_70_30"):
        real_p = real[
            (real["validation_protocol"] == protocol)
            & (real["feature_name"] == best["feature_name"])
            & (real["model_head"] == best["model_head"])
            & (real["loss_name"] == best["loss_name"])
        ]
        ctrl_p = matched[matched["validation_protocol"] == protocol]
        if real_p.empty or ctrl_p.empty:
            gates[f"{protocol}_has_matched_control"] = False
            continue
        best_real = real_p.sort_values("mean_pr_auc", ascending=False).iloc[0]
        best_ctrl = ctrl_p.sort_values("mean_pr_auc", ascending=False).iloc[0]
        delta = float(best_real["mean_pr_auc"]) - float(best_ctrl["mean_pr_auc"])
        gates[f"{protocol}_real_pr_auc"] = float(best_real["mean_pr_auc"])
        gates[f"{protocol}_best_matched_control_pr_auc"] = float(best_ctrl["mean_pr_auc"])
        gates[f"{protocol}_best_matched_control_type"] = str(best_ctrl["control_type"])
        gates[f"{protocol}_real_minus_matched_control_pr_auc"] = delta
        gates[f"{protocol}_beats_best_matched_control"] = bool(delta > 0)
    grouped_ok = bool(gates.get("grouped_video_beats_best_matched_control"))
    blocked_ok = bool(gates.get("blocked_temporal_70_30_beats_best_matched_control"))
    gates["corrected_grouped_support"] = grouped_ok
    gates["corrected_blocked_temporal_support"] = blocked_ok
    gates["strict_forward_time_temporal_generalization_proven"] = bool(grouped_ok and blocked_ok)
    gates["recommendation"] = "repair_pass" if grouped_ok and blocked_ok else "repair_required"
    return clean_json(gates)


def patch_base_module() -> None:
    if not hasattr(base, "_ORIGINAL_ROW_DICT_BASE"):
        base._ORIGINAL_ROW_DICT_BASE = base.row_dict_base
    if not hasattr(base, "_ORIGINAL_METRIC_RECORDS_FROM_SCORES"):
        base._ORIGINAL_METRIC_RECORDS_FROM_SCORES = base.metric_records_from_scores
    base.PHASE5_SCHEMA_VERSION = REPAIR_SCHEMA_VERSION
    base.DEFAULT_MODELS = REPAIR_MODELS
    base.DEFAULT_LOSSES = REPAIR_LOSSES
    base.PHASE5_DEFAULT_SEEDS = REPAIR_SEEDS
    base.load_labels = load_labels_capture
    base.assemble_feature_blocks = assemble_feature_blocks_repair
    base.row_dict_base = row_dict_base_repair
    base.train_mlx_head = train_mlx_head_repair
    base.train_head = train_head_repair
    base.metric_records_from_scores = metric_records_from_scores_repair
    base.promotion_gates = promotion_gates_repair


def default_phase4_root() -> str:
    external = os.environ.get("NEURAL_BRIDGE_EXTERNAL_ROOT")
    if external:
        return str(Path(external) / "outputs" / "again_dense_2hz_phase4_pca_bridge_20260625_full")
    return "outputs/again_dense_2hz_phase4_pca_bridge_20260625_full"


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser()
    parser.description = __doc__
    for action in parser._actions:
        if action.dest == "phase4_root":
            action.required = False
    parser.set_defaults(
        phase4_root=default_phase4_root(),
        output_root="",
        models=",".join(REPAIR_MODELS),
        losses=",".join(REPAIR_LOSSES),
        controls=",".join(DEFAULT_REPAIR_CONTROLS),
        seeds=",".join(str(seed) for seed in REPAIR_SEEDS),
        include_ar=True,
        include_temporal_diagnostics=True,
        fast_controls=False,
        max_epochs=180,
        patience=24,
        learning_rate=3e-4,
        weight_decay=1e-4,
        dropout=0.1,
        batch_size=8192,
    )
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    return parser


def validate_phase4_feature_payload(args: argparse.Namespace) -> None:
    phase4_root = Path(args.phase4_root)
    features_dir = phase4_root / "features"
    if not features_dir.is_dir():
        raise FileNotFoundError(
            "Missing Phase 4 fold-safe PCA feature payload. "
            f"Expected directory: {features_dir}. "
            "Mount the external root or pass --phase4-root to the artifact root that contains features/*.npy."
        )

    targets = base.parse_csv(args.targets)
    protocols = base.parse_csv(args.validation_protocols)
    features = base.parse_csv(args.features)
    missing: list[str] = []
    for target in targets:
        for protocol in protocols:
            for feature_name in features:
                source_family = base.feature_spec(feature_name).source_family
                pattern = f"{target}__{protocol}__fold*__{source_family}__scores_w256.npy"
                if not any(features_dir.glob(pattern)):
                    missing.append(pattern)
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "" if len(missing) <= 8 else f", ... ({len(missing)} missing patterns total)"
        raise FileNotFoundError(
            "Phase 4 feature payload is incomplete for this repair run. "
            f"Missing patterns under {features_dir}: {preview}{suffix}"
        )


def write_repair_artifacts(output_root: Path) -> None:
    metrics = output_root / "metrics"
    diagnostics = output_root / "diagnostics"
    pd.DataFrame(INNER_AUDIT_ROWS).to_csv(diagnostics / "blocked_inner_validation_audit.csv", index=False)
    write_json(diagnostics / "blocked_inner_validation_audit.json", INNER_AUDIT_ROWS)
    pd.DataFrame(WITHIN_VIDEO_ROWS).to_csv(metrics / "phase5_within_video_metrics.csv", index=False)
    write_json(
        diagnostics / "checkpoint_restore_audit.json",
        {
            "status": "pass" if INNER_AUDIT_ROWS and all(row.get("checkpoint_restored_for_test_scoring") for row in INNER_AUDIT_ROWS) else "fail",
            "runs": INNER_AUDIT_ROWS,
        },
    )


def run_repair(args: argparse.Namespace) -> dict[str, Any]:
    global REPAIR_GRADIENT_CLIP
    REPAIR_GRADIENT_CLIP = float(args.gradient_clip)
    patch_base_module()
    if not args.output_root:
        args.output_root = str(base.default_output_root("again_dense_2hz_phase5_adversarial_repair_fixplus"))
    output_root = Path(args.output_root)
    if not args.dry_run:
        validate_phase4_feature_payload(args)
        output_root.mkdir(parents=True, exist_ok=False)
    start = time.time()
    result = base.run_phase5(args, dry_run=args.dry_run)
    if not args.dry_run:
        write_repair_artifacts(output_root)
        summary = {
            "schema_version": REPAIR_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "output_root": str(output_root),
            "runtime_seconds_repair_wrapper": time.time() - start,
            "checkpoint_restored_for_test_scoring": bool(
                INNER_AUDIT_ROWS and all(row.get("checkpoint_restored_for_test_scoring") for row in INNER_AUDIT_ROWS)
            ),
            "inner_validation_audit": str(output_root / "diagnostics" / "blocked_inner_validation_audit.json"),
            "within_video_metrics": str(output_root / "metrics" / "phase5_within_video_metrics.csv"),
        }
        write_json(output_root / "repair_fixplus_summary.json", summary)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_repair(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
