"""Blocked redesigned-target temporal/event-context residual diagnostic.

Bounded matrix:
2 targets x 3 seeds x 4 architectures x 7 controls = 168 rows.

This runner uses only the fold-safe redesigned-target PCA256 artifacts. It does
not run grouped validation, V-JEPA/TRIBE, PCA fitting, AR training, or
extra targets. Frozen AR remains the baseline floor.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.scripts import run_again_dense_2hz_phase5_continuous_residual_blocked as continuous_run
from backend.scripts import run_again_dense_2hz_phase5_frozen_ar_residual as fr
from backend.scripts import run_again_dense_2hz_phase5_learned_heads as base
from backend.scripts import run_again_dense_2hz_phase5_redesigned_target_blocked as redesigned
from backend.scripts.again_dense_2hz_benchmark import load_or_build_temporal_diagnostic_features


SCHEMA_VERSION = "again_dense_2hz_phase5_temporal_residual_blocked_v1"
SOURCE_ROOT = Path("outputs/again_dense_2hz_phase5_adversarial_repair_fixplus_20260629_171825")
FOLDSAFE_PCA_ROOT = Path("outputs/again_dense_2hz_phase5_redesigned_target_foldsafe_pca_20260630_005312")
PREVIOUS_REDESIGNED_ROOT = Path("outputs/again_dense_2hz_phase5_redesigned_target_blocked_20260630_010721")
PROTOCOL = "blocked_temporal_70_30"
FOLD = 1
FEATURE_NAME = "temporal_mean_2s_then_pca256"
PCA_SOURCE_FAMILY = "temporal_mean_2s"
PCA_WIDTH = 256
SEEDS = (20260625, 20260626, 20260627)
ARCHITECTURES = (
    "current_row_mlp_residual",
    "delta_feature_mlp_residual",
    "short_temporal_conv_residual",
    "low_ar_confidence_temporal_residual",
)
CONTROLS = (
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
BINARY_TARGET = redesigned.BINARY_TARGET
CONTINUOUS_TARGET = redesigned.CONTINUOUS_TARGET
TARGET_TYPES = {BINARY_TARGET: "binary", CONTINUOUS_TARGET: "continuous"}
WEAK_THRESHOLD = 0.001
DO_NO_HARM_FLOOR = -0.0005
CONV_WINDOW_ROWS = 5
DELTA_HISTORY_ROWS = 8


@dataclass
class FeaturePack:
    train_x: np.ndarray
    test_x: np.ndarray
    dims: dict[str, int]
    manifest: list[dict[str, Any]]
    context_audit: dict[str, Any]


def default_output_root() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path(f"outputs/again_dense_2hz_phase5_temporal_residual_blocked_{stamp}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(SOURCE_ROOT))
    parser.add_argument("--foldsafe-pca-root", default=str(FOLDSAFE_PCA_ROOT))
    parser.add_argument("--previous-redesigned-root", default=str(PREVIOUS_REDESIGNED_ROOT))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, obj: Any) -> None:
    fr.write_json(path, obj)


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return math.nan
    return out if math.isfinite(out) else math.nan


def load_pca_manifest(pca_root: Path) -> dict[str, Any]:
    audit = redesigned.validate_pca_root(pca_root)
    manifest_path = pca_root / "manifests" / "redesigned_pca_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = {row["target_name"]: row for row in manifest["target_rows"]}
    for target in (BINARY_TARGET, CONTINUOUS_TARGET):
        row = rows[target]
        if not row.get("leakage_audit_pass"):
            raise RuntimeError(f"PCA leakage audit failed for {target}")
        if not row.get("no_test_fit") or row.get("original_pca_artifact_reused"):
            raise RuntimeError(f"PCA fit policy failed for {target}")
        if row.get("target_window_overlap") or row.get("future_leakage_suspected"):
            raise RuntimeError(f"Target overlap/leakage audit failed for {target}")
    return {"audit": audit, "manifest": manifest, "rows": rows}


def block_for_target(blocks: dict[str, redesigned.RunBlock], target_name: str) -> redesigned.RunBlock:
    block = blocks[target_name]
    if block.protocol != PROTOCOL or int(block.fold) != FOLD:
        raise RuntimeError(f"Unexpected block protocol/fold for {target_name}: {block.protocol}/fold{block.fold}")
    return block


def verify_pca_rows(pca_root: Path, block: redesigned.RunBlock, pca_rows: dict[str, Any]) -> dict[str, Any]:
    row = pca_rows[block.target_name]
    train_digest = redesigned.phase4.array_digest(block.train_idx)
    test_digest = redesigned.phase4.array_digest(block.test_idx)
    if int(row["train_rows"]) != len(block.train_idx) or int(row["test_rows"]) != len(block.test_idx):
        raise RuntimeError(f"Row count mismatch for {block.target_name}")
    if row["train_idx_digest"] != train_digest or row["test_idx_digest"] != test_digest:
        raise RuntimeError(f"Row index checksum mismatch for {block.target_name}")
    row_index_path = Path(row["row_index_path"])
    if not row_index_path.exists():
        raise FileNotFoundError(row_index_path)
    row_index = pd.read_csv(row_index_path)
    expected = np.concatenate([block.train_idx, block.test_idx]).astype(np.int64)
    if "row_id" in row_index:
        observed = row_index["row_id"].to_numpy(dtype=np.int64)
    else:
        observed = row_index.iloc[:, 0].to_numpy(dtype=np.int64)
    if not np.array_equal(observed, expected):
        raise RuntimeError(f"Fold-safe PCA row-index file does not match split rows for {block.target_name}")
    return {
        "target_name": block.target_name,
        "train_rows": int(len(block.train_idx)),
        "test_rows": int(len(block.test_idx)),
        "train_idx_digest": train_digest,
        "test_idx_digest": test_digest,
        "row_index_verified": True,
        "score_path": row["score_path"],
        "explained_variance_ratio_sum": safe_float(row.get("explained_variance_ratio_sum")),
    }


def build_blocks(source_root: Path, pca_root: Path) -> tuple[dict[str, redesigned.RunBlock], pd.DataFrame, Path, dict[str, Any]]:
    blocks, df, dense_root, residual_meta = redesigned.build_blocks(source_root, pca_root)
    expected = {BINARY_TARGET, CONTINUOUS_TARGET}
    if set(blocks) != expected:
        raise RuntimeError(f"Unexpected blocks: {sorted(blocks)}")
    return blocks, df, dense_root, residual_meta


def load_pca_for_block(
    df: pd.DataFrame,
    pca_root: Path,
    block: redesigned.RunBlock,
) -> tuple[np.ndarray, np.ndarray, Path]:
    spec = base.feature_spec(FEATURE_NAME)
    train_idx, test_idx, pca_train, pca_test, source_path = base.load_phase4_scores(df, pca_root, block.split, spec)
    if not np.array_equal(train_idx, block.train_idx) or not np.array_equal(test_idx, block.test_idx):
        raise RuntimeError(f"PCA row mismatch for {block.target_name}")
    return pca_train.astype(np.float32, copy=False), pca_test.astype(np.float32, copy=False), source_path


def load_or_cache_frozen_ar(
    previous_root: Path,
    source_root: Path,
    output_root: Path,
    block: redesigned.RunBlock,
    seed: int,
    batch_size: int,
) -> dict[str, Any]:
    key = f"{block.target_name}__{block.protocol}__fold{block.fold}__seed{seed}__{fr.LOSS_NAME}"
    prev_dir = previous_root / "frozen_ar_scores"
    paths = {
        "train": prev_dir / f"{key}__train.csv.gz",
        "heldout_test": prev_dir / f"{key}__heldout_test.csv.gz",
        "inner_val": prev_dir / f"{key}__inner_val.csv.gz",
    }
    if all(path.exists() for path in paths.values()):
        train = pd.read_csv(paths["train"])
        test = pd.read_csv(paths["heldout_test"])
        if np.array_equal(train["row_id"].to_numpy(dtype=np.int64), block.train_idx.astype(np.int64)) and np.array_equal(
            test["row_id"].to_numpy(dtype=np.int64), block.test_idx.astype(np.int64)
        ):
            out_dir = output_root / "frozen_ar_scores"
            out_dir.mkdir(parents=True, exist_ok=True)
            for split_name, src in paths.items():
                shutil.copy2(src, out_dir / f"{key}__{split_name}.csv.gz")
            train_score = train["frozen_ar_score"].to_numpy(dtype=np.float32)
            train_reg = train["frozen_ar_continuous_prediction"].to_numpy(dtype=np.float32)
            test_score = test["frozen_ar_score"].to_numpy(dtype=np.float32)
            test_reg = test["frozen_ar_continuous_prediction"].to_numpy(dtype=np.float32)
            return {
                "key": key,
                "target_name": block.target_name,
                "protocol": block.protocol,
                "fold": int(block.fold),
                "seed": int(seed),
                "loss": fr.LOSS_NAME,
                "source": "reused_previous_redesigned_frozen_ar_score_cache",
                "ar_retrained": False,
                "train_score": train_score,
                "train_reg": train_reg,
                "test_score": test_score,
                "test_reg": test_reg,
                "train_checksum": fr.hash_array(train_score),
                "test_checksum": fr.hash_array(test_score),
            }
    ar = redesigned.cache_frozen_ar(source_root, output_root, block, seed, batch_size)
    ar["source"] = "re_forwarded_saved_ar_only_best_checkpoint_cache_missing"
    return ar


def binary_metrics(block: redesigned.RunBlock, train_score: np.ndarray, test_score: np.ndarray, test_reg: np.ndarray) -> dict[str, Any]:
    return redesigned.binary_metric_row(block, train_score, test_score, test_reg)


def continuous_metrics(block: redesigned.RunBlock, train_score: np.ndarray, test_score: np.ndarray, test_reg: np.ndarray) -> dict[str, Any]:
    return redesigned.continuous_metric_row(block, train_score, test_score, test_reg)


def metric_row_for_block(
    block: redesigned.RunBlock,
    train_score: np.ndarray,
    test_score: np.ndarray,
    test_reg: np.ndarray,
) -> dict[str, Any]:
    if block.target_type == "binary":
        return binary_metrics(block, train_score, test_score, test_reg)
    return continuous_metrics(block, train_score, test_score, test_reg)


def add_deltas(metrics: dict[str, Any], ar_metrics: dict[str, Any], target_type: str) -> None:
    if target_type == "binary":
        redesigned.add_binary_deltas(metrics, ar_metrics)
    else:
        redesigned.add_continuous_deltas(metrics, ar_metrics)


def pca_control_arrays(
    pca_train: np.ndarray,
    pca_test: np.ndarray,
    block: redesigned.RunBlock,
    control: str,
    seed: int,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    if control == "diagnostics_only_residual":
        return None, None, {"pca_control": "omitted_diagnostics_only"}
    rng = np.random.default_rng(int(seed) + 1009 * (CONTROLS.index(control) + 1) + 17 * block.fold)
    if control == "shuffled_pca_residual":
        return (
            pca_train[rng.permutation(len(pca_train))].astype(np.float32, copy=False),
            pca_test[rng.permutation(len(pca_test))].astype(np.float32, copy=False),
            {"pca_control": "shuffled_train_and_test_separately"},
        )
    if control == "random_pca_residual":
        return (
            rng.normal(0.0, 1.0, size=pca_train.shape).astype(np.float32),
            rng.normal(0.0, 1.0, size=pca_test.shape).astype(np.float32),
            {"pca_control": "random_gaussian_matched_shape"},
        )
    if control == "train_only_video_mean_residual":
        train_video = block.train_video_id.astype(str)
        test_video = block.test_video_id.astype(str)
        means = {video: pca_train[train_video == video].mean(axis=0) for video in np.unique(train_video)}
        global_mean = pca_train.mean(axis=0)
        return (
            np.vstack([means.get(v, global_mean) for v in train_video]).astype(np.float32),
            np.vstack([means.get(v, global_mean) for v in test_video]).astype(np.float32),
            {"pca_control": "train_only_video_mean", "uses_test_rows_for_mean": False},
        )
    return pca_train, pca_test, {"pca_control": "real"}


def same_video_prev_lookup(
    row_idx: np.ndarray,
    video_id: np.ndarray,
    values: np.ndarray,
    *,
    max_lag: int,
) -> tuple[dict[int, np.ndarray], dict[int, bool]]:
    row_to_pos = {int(row): pos for pos, row in enumerate(row_idx)}
    out: dict[int, np.ndarray] = {}
    available: dict[int, bool] = {}
    width = int(values.shape[1])
    for pos, row in enumerate(row_idx):
        histories = []
        ok = True
        for lag in range(1, max_lag + 1):
            prev_row = int(row) - lag
            prev_pos = row_to_pos.get(prev_row)
            if prev_pos is None or str(video_id[prev_pos]) != str(video_id[pos]):
                histories.append(np.zeros(width, dtype=np.float32))
                ok = False
            else:
                histories.append(values[prev_pos])
        out[int(row)] = np.vstack(histories).astype(np.float32)
        available[int(row)] = ok
    return out, available


def causal_delta_features(
    current: np.ndarray,
    row_idx: np.ndarray,
    video_id: np.ndarray,
    *,
    max_lag: int = DELTA_HISTORY_ROWS,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    histories, full_available = same_video_prev_lookup(row_idx, video_id, current, max_lag=max_lag)
    delta1 = np.zeros_like(current)
    delta4 = np.zeros_like(current)
    delta_early_late = np.zeros_like(current)
    flags = np.zeros((len(row_idx), 3), dtype=np.float32)
    for i, row in enumerate(row_idx):
        hist = histories[int(row)]
        has1 = full_available[int(row)] or bool(np.any(hist[0]))
        has4 = bool(np.any(hist[:4])) and all(str(video_id[i]) == str(video_id[max(0, i - j)]) for j in range(1, min(i + 1, 5)))
        has8 = full_available[int(row)]
        if has1:
            delta1[i] = current[i] - hist[0]
            flags[i, 0] = 1.0
        if has4:
            delta4[i] = current[i] - hist[:4].mean(axis=0)
            flags[i, 1] = 1.0
        if has8:
            delta_early_late[i] = hist[:4].mean(axis=0) - hist[4:8].mean(axis=0)
            flags[i, 2] = 1.0
    audit = {
        "causal_only": True,
        "max_past_lag_rows": int(max_lag),
        "uses_future_rows": False,
        "same_video_history_masking": True,
        "full_history_available_rate": float(np.mean(flags[:, 2])) if len(flags) else math.nan,
    }
    return np.concatenate([current, delta1, delta4, delta_early_late, flags], axis=1).astype(np.float32), flags, audit


def causal_sequence_features(
    current: np.ndarray,
    row_idx: np.ndarray,
    video_id: np.ndarray,
    *,
    window_rows: int = CONV_WINDOW_ROWS,
) -> tuple[np.ndarray, dict[str, Any]]:
    histories, _full_available = same_video_prev_lookup(row_idx, video_id, current, max_lag=window_rows - 1)
    seq = np.zeros((len(row_idx), window_rows, current.shape[1]), dtype=np.float32)
    flags = np.zeros((len(row_idx), window_rows), dtype=np.float32)
    for i, row in enumerate(row_idx):
        hist = histories[int(row)]
        seq[i, -1] = current[i]
        flags[i, -1] = 1.0
        for lag in range(1, window_rows):
            prev = hist[lag - 1]
            seq[i, window_rows - 1 - lag] = prev
            if np.any(prev):
                flags[i, window_rows - 1 - lag] = 1.0
    flat = seq.reshape(len(row_idx), window_rows * current.shape[1])
    audit = {
        "causal_only": True,
        "window_rows": int(window_rows),
        "uses_future_rows": False,
        "same_video_history_masking": True,
        "mean_context_row_availability": float(np.mean(flags)) if len(flags) else math.nan,
    }
    return flat.astype(np.float32), audit


def feature_pack_for(
    df: pd.DataFrame,
    dense_root: Path,
    pca_root: Path,
    block: redesigned.RunBlock,
    architecture: str,
    control: str,
    seed: int,
) -> FeaturePack:
    pca_train, pca_test, pca_path = load_pca_for_block(df, pca_root, block)
    diag_all = load_or_build_temporal_diagnostic_features(dense_root, df)
    diag_train = diag_all[block.train_idx].astype(np.float32, copy=True)
    diag_test = diag_all[block.test_idx].astype(np.float32, copy=True)
    pca_train_ctrl, pca_test_ctrl, pca_meta = pca_control_arrays(pca_train, pca_test, block, control, seed)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    dims: dict[str, int] = {}
    manifest: list[dict[str, Any]] = []
    context_audit: dict[str, Any] = {
        "architecture": architecture,
        "control": control,
        "target_name": block.target_name,
        "temporal_context_causal_only": True,
        "uses_centered_or_future_windows": False,
        "same_video_history_masking": True,
        "label_permutation_policy_required": control == "label_permutation_residual",
        "train_only_video_mean_policy_required": control == "train_only_video_mean_residual",
    }

    if architecture == "short_temporal_conv_residual":
        if pca_train_ctrl is not None and pca_test_ctrl is not None:
            train_seq, train_audit = causal_sequence_features(pca_train_ctrl, block.train_idx, block.train_video_id)
            test_seq, test_audit = causal_sequence_features(pca_test_ctrl, block.test_idx, block.test_video_id)
            train_parts.append(train_seq)
            test_parts.append(test_seq)
            dims["sequence"] = int(train_seq.shape[1])
            dims["sequence_window"] = CONV_WINDOW_ROWS
            dims["sequence_channels"] = int(pca_train_ctrl.shape[1])
            context_audit["pca_sequence_train"] = train_audit
            context_audit["pca_sequence_test"] = test_audit
            manifest.append(
                {
                    "block": "causal_pca_sequence_flat",
                    "source_path": str(pca_path),
                    "source_checksum": base.file_digest(pca_path, digest_size=16),
                    "window_rows": CONV_WINDOW_ROWS,
                    "channels": int(pca_train_ctrl.shape[1]),
                    **pca_meta,
                }
            )
            train_parts.append(diag_train)
            test_parts.append(diag_test)
            dims["diagnostics"] = int(diag_train.shape[1])
            manifest.append({"block": "current_temporal_diagnostics", "width": int(diag_train.shape[1])})
        else:
            train_seq, train_audit = causal_sequence_features(diag_train, block.train_idx, block.train_video_id)
            test_seq, test_audit = causal_sequence_features(diag_test, block.test_idx, block.test_video_id)
            train_parts.append(train_seq)
            test_parts.append(test_seq)
            dims["sequence"] = int(train_seq.shape[1])
            dims["sequence_window"] = CONV_WINDOW_ROWS
            dims["sequence_channels"] = int(diag_train.shape[1])
            context_audit["diagnostics_sequence_train"] = train_audit
            context_audit["diagnostics_sequence_test"] = test_audit
            manifest.append(
                {
                    "block": "causal_diagnostics_sequence_flat",
                    "window_rows": CONV_WINDOW_ROWS,
                    "channels": int(diag_train.shape[1]),
                    "control": "diagnostics_only",
                }
            )
    elif architecture in {"delta_feature_mlp_residual", "low_ar_confidence_temporal_residual"}:
        if pca_train_ctrl is not None and pca_test_ctrl is not None:
            pca_train_delta, _flags_train, train_audit = causal_delta_features(pca_train_ctrl, block.train_idx, block.train_video_id)
            pca_test_delta, _flags_test, test_audit = causal_delta_features(pca_test_ctrl, block.test_idx, block.test_video_id)
            train_parts.append(pca_train_delta)
            test_parts.append(pca_test_delta)
            dims["pca"] = int(pca_train_delta.shape[1])
            context_audit["pca_delta_train"] = train_audit
            context_audit["pca_delta_test"] = test_audit
            manifest.append(
                {
                    "block": "current_pca_plus_causal_pca_deltas",
                    "source_path": str(pca_path),
                    "source_checksum": base.file_digest(pca_path, digest_size=16),
                    "max_lag_rows": DELTA_HISTORY_ROWS,
                    "width": int(pca_train_delta.shape[1]),
                    **pca_meta,
                }
            )
        diag_train_delta, _df_train, diag_train_audit = causal_delta_features(diag_train, block.train_idx, block.train_video_id)
        diag_test_delta, _df_test, diag_test_audit = causal_delta_features(diag_test, block.test_idx, block.test_video_id)
        train_parts.append(diag_train_delta)
        test_parts.append(diag_test_delta)
        dims["diagnostics"] = int(diag_train_delta.shape[1])
        context_audit["diagnostics_delta_train"] = diag_train_audit
        context_audit["diagnostics_delta_test"] = diag_test_audit
        manifest.append({"block": "current_diagnostics_plus_causal_diagnostics_deltas", "width": int(diag_train_delta.shape[1])})
    elif architecture == "current_row_mlp_residual":
        if pca_train_ctrl is not None and pca_test_ctrl is not None:
            train_parts.append(pca_train_ctrl)
            test_parts.append(pca_test_ctrl)
            dims["pca"] = int(pca_train_ctrl.shape[1])
            manifest.append(
                {
                    "block": "current_fold_safe_pca",
                    "source_path": str(pca_path),
                    "source_checksum": base.file_digest(pca_path, digest_size=16),
                    "width": int(pca_train_ctrl.shape[1]),
                    **pca_meta,
                }
            )
        train_parts.append(diag_train)
        test_parts.append(diag_test)
        dims["diagnostics"] = int(diag_train.shape[1])
        manifest.append({"block": "current_temporal_diagnostics", "width": int(diag_train.shape[1])})
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    if not train_parts:
        raise RuntimeError(f"No features assembled for {architecture}/{control}")
    train_x = np.concatenate(train_parts, axis=1).astype(np.float32, copy=False)
    test_x = np.concatenate(test_parts, axis=1).astype(np.float32, copy=False)
    train_x, test_x = base.standardize_train_only(train_x, test_x)
    if control == "train_only_video_mean_residual" and pca_meta.get("uses_test_rows_for_mean") is not False:
        raise RuntimeError("train_only_video_mean_residual attempted to use test rows")
    context_audit["input_width"] = int(train_x.shape[1])
    context_audit["train_rows"] = int(len(train_x))
    context_audit["test_rows"] = int(len(test_x))
    return FeaturePack(train_x, test_x, dims, manifest, context_audit)


class TemporalResidualHead(base.nn.Module):
    def __init__(
        self,
        input_dim: int,
        architecture: str,
        *,
        hidden: int = 64,
        sequence_window: int = 0,
        sequence_channels: int = 0,
        alpha_initial_logit: float = -4.0,
        alpha_cap: float = 0.12,
        gate_bias: float = 4.0,
    ):
        super().__init__()
        self.architecture = architecture
        self.sequence_window = int(sequence_window)
        self.sequence_channels = int(sequence_channels)
        self.alpha = base.mx.array([float(alpha_initial_logit)], dtype=base.mx.float32)
        self.alpha_cap = float(alpha_cap)
        self.gate_bias = float(gate_bias)
        self.gate = base.nn.Linear(input_dim, 1)
        self.conf_gate = base.nn.Linear(3, 1)
        if architecture == "short_temporal_conv_residual":
            if self.sequence_window <= 0 or self.sequence_channels <= 0:
                raise ValueError("short_temporal_conv_residual requires sequence dims")
            self.conv = base.nn.Linear(self.sequence_channels * 3, hidden)
            remaining = input_dim - self.sequence_window * self.sequence_channels
            self.post = base.nn.Linear(hidden + max(0, remaining), hidden)
        else:
            self.layers = [base.nn.Linear(input_dim, hidden), base.nn.Linear(hidden, hidden)]
        self.out = base.nn.Linear(hidden, 2)

    def alpha_value(self) -> Any:
        return base.mx.sigmoid(self.alpha) * self.alpha_cap

    def confidence_features(self, ar_score: Any) -> Any:
        return base.mx.concatenate(
            [
                ar_score[:, None],
                base.mx.abs(ar_score[:, None]),
                base.mx.sigmoid(ar_score[:, None]) * (1.0 - base.mx.sigmoid(ar_score[:, None])),
            ],
            axis=1,
        )

    def gate_value(self, x: Any, ar_score: Any) -> Any:
        gate = base.mx.sigmoid(self.gate(x) - self.gate_bias)
        if self.architecture == "low_ar_confidence_temporal_residual":
            confidence_gate = base.mx.sigmoid(self.conf_gate(self.confidence_features(ar_score)))
            gate = gate * confidence_gate
        return gate

    def hidden(self, x: Any) -> Any:
        if self.architecture == "short_temporal_conv_residual":
            seq_width = self.sequence_window * self.sequence_channels
            seq = x[:, :seq_width].reshape((x.shape[0], self.sequence_window, self.sequence_channels))
            extra = x[:, seq_width:]
            padded = base.mx.concatenate(
                [base.mx.zeros((x.shape[0], 2, self.sequence_channels), dtype=x.dtype), seq],
                axis=1,
            )
            conv_rows = []
            for pos in range(self.sequence_window):
                window = padded[:, pos : pos + 3, :].reshape((x.shape[0], self.sequence_channels * 3))
                conv_rows.append(base.nn.gelu(self.conv(window)))
            h = conv_rows[-1]
            if extra.shape[1] > 0:
                h = base.mx.concatenate([h, extra], axis=1)
            return base.nn.gelu(self.post(h))
        h = x
        for layer in self.layers:
            h = base.nn.gelu(layer(h))
        return h

    def __call__(self, x: Any, ar_score: Any, ar_reg: Any) -> Any:
        residual = self.out(self.hidden(x))
        scale = self.alpha_value() * self.gate_value(x, ar_score)
        reg = ar_reg[:, None] + scale * residual[:, 0:1]
        binary = ar_score[:, None] + scale * residual[:, 1:2]
        return base.mx.concatenate([reg, binary], axis=1)


def forward_residual(
    model: TemporalResidualHead,
    x: np.ndarray,
    ar_score: np.ndarray,
    ar_reg: np.ndarray,
    *,
    batch_size: int,
    target_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores: list[np.ndarray] = []
    regs: list[np.ndarray] = []
    gates: list[np.ndarray] = []
    if hasattr(model, "eval"):
        model.eval()
    for start in range(0, len(x), batch_size):
        xb = base.mx.array(x[start : start + batch_size], dtype=base.mx.float32)
        ab = base.mx.array(ar_score[start : start + batch_size], dtype=base.mx.float32)
        rb = base.mx.array(ar_reg[start : start + batch_size], dtype=base.mx.float32)
        out = model(xb, ab, rb)
        gate = model.gate_value(xb, ab)
        base.mx.eval(out, gate)
        out_np = np.asarray(out, dtype=np.float32)
        score = out_np[:, 1] if target_type == "binary" else out_np[:, 0]
        reg = out_np[:, 0]
        scores.append(score.astype(np.float32, copy=False))
        regs.append(reg.astype(np.float32, copy=False))
        gates.append(np.asarray(gate, dtype=np.float32).reshape(-1))
    return np.concatenate(scores), np.concatenate(regs), np.concatenate(gates)


def top5_lift_delta(
    block: redesigned.RunBlock,
    ar_train_score: np.ndarray,
    ar_test_score: np.ndarray,
    ar_test_reg: np.ndarray,
    train_score: np.ndarray,
    test_score: np.ndarray,
    test_reg: np.ndarray,
) -> tuple[float, float, float]:
    ar_m = continuous_metrics(block, ar_train_score, ar_test_score, ar_test_reg)
    m = continuous_metrics(block, train_score, test_score, test_reg)
    return (
        safe_float(m["top_5pct_continuous_lift"] - ar_m["top_5pct_continuous_lift"]),
        safe_float(m["continuous_spearman"] - ar_m["continuous_spearman"]),
        safe_float(m["top_10pct_continuous_lift"] - ar_m["top_10pct_continuous_lift"]),
    )


def train_temporal_residual(
    *,
    architecture: str,
    control: str,
    pack: FeaturePack,
    block: redesigned.RunBlock,
    ar: dict[str, Any],
    seed: int,
    output_root: Path,
    batch_size: int,
    max_epochs: int,
    patience: int,
    hyperparameters: dict[str, float | int] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    base.require_mlx()
    base.mx.random.seed(int(seed))
    hp = dict(hyperparameters or {})
    model = TemporalResidualHead(
        pack.train_x.shape[1],
        architecture,
        hidden=int(hp.get("hidden", 64)),
        sequence_window=int(pack.dims.get("sequence_window", 0)),
        sequence_channels=int(pack.dims.get("sequence_channels", 0)),
        alpha_initial_logit=float(hp.get("alpha_initial_logit", -4.0)),
        alpha_cap=float(hp.get("alpha_cap", 0.12)),
        gate_bias=float(hp.get("gate_bias", 4.0)),
    )
    optimizer = base.optim.AdamW(
        learning_rate=float(hp.get("learning_rate", 2e-4)),
        weight_decay=float(hp.get("weight_decay", 1e-4)),
    )
    inner_train = block.inner_train
    inner_val = block.inner_val
    rng = np.random.default_rng(int(seed) + 10007 * (ARCHITECTURES.index(architecture) + 1) + 503 * (CONTROLS.index(control) + 1))
    train_y_metric = block.train_y.copy()
    train_cont_metric = block.train_cont.copy()
    selection_y = block.train_y.copy()
    selection_cont = block.train_cont.copy()
    if control == "label_permutation_residual":
        perm = rng.permutation(len(train_y_metric))
        train_y_metric = train_y_metric[perm]
        train_cont_metric = train_cont_metric[perm]
        selection_y = train_y_metric
        selection_cont = train_cont_metric
        label_policy = "permuted_train_and_permuted_inner_val_selection"
    else:
        label_policy = "true_train_and_true_inner_val_selection"
    ar_train_score = ar["train_score"].astype(np.float32)
    ar_train_reg = ar["train_reg"].astype(np.float32)
    ar_test_score = ar["test_score"].astype(np.float32)
    ar_test_reg = ar["test_reg"].astype(np.float32)
    if block.target_type == "binary":
        ar_inner_metric = average_precision_score(selection_y[inner_val], ar_train_score[inner_val])
        best_key: tuple[float, float, float] = (0.0, 0.0, 0.0)
        selection_name = "inner_val_pr_auc_delta_vs_frozen_ar"
    else:
        ar_inner_metrics = continuous_run.continuous_metric_row(
            selection_y[inner_train],
            ar_train_score[inner_train],
            selection_y[inner_val],
            ar_train_score[inner_val],
            selection_cont[inner_val],
            ar_train_reg[inner_val],
        )
        best_key = (0.0, 0.0, 0.0)
        selection_name = "inner_val_top_5pct_lift_delta_vs_frozen_ar"
    q80 = float(np.quantile(train_cont_metric[inner_train], 0.80))
    q90 = float(np.quantile(train_cont_metric[inner_train], 0.90))
    best_epoch = 0
    best_path = output_root / "checkpoints" / (
        f"{block.target_name}__{architecture}__{control}__seed{seed}__best.npz"
    )
    best_path.parent.mkdir(parents=True, exist_ok=True)
    curves: list[dict[str, Any]] = []
    stale = 0
    early_stop = "max_epochs_reached"
    suppressed = True

    def loss_fn(model_obj: TemporalResidualHead, xb: Any, ar_b: Any, ar_r: Any, yb: Any, yr: Any, wb: Any) -> Any:
        out = model_obj(xb, ar_b, ar_r)
        reg_loss = base.mx.mean(base.nn.losses.huber_loss(out[:, 0:1], yr, delta=1.0) * wb)
        if block.target_type == "continuous":
            anchor = 0.03 * base.mx.mean((out[:, 0:1] - ar_r[:, None]) * (out[:, 0:1] - ar_r[:, None]))
            return reg_loss + anchor + 0.01 * base.mx.mean(model_obj.alpha_value() * model_obj.alpha_value())
        bce = base.nn.losses.binary_cross_entropy(out[:, 1:2], yb, with_logits=True)
        return (
            reg_loss
            + float(hp.get("lambda_binary", 0.5)) * base.mx.mean(bce * wb)
            + 0.01 * base.mx.mean(model_obj.alpha_value() * model_obj.alpha_value())
        )

    loss_and_grad = base.nn.value_and_grad(model, loss_fn)
    for epoch in range(1, int(max_epochs) + 1):
        if hasattr(model, "train"):
            model.train()
        order = rng.permutation(inner_train)
        total = 0.0
        batches = 0
        for start in range(0, len(order), batch_size):
            rel = order[start : start + batch_size]
            xb = base.mx.array(pack.train_x[rel], dtype=base.mx.float32)
            ab = base.mx.array(ar_train_score[rel], dtype=base.mx.float32)
            rb = base.mx.array(ar_train_reg[rel], dtype=base.mx.float32)
            yb_np = train_y_metric[rel].astype(np.float32)[:, None]
            yr_np = train_cont_metric[rel].astype(np.float32)[:, None]
            if block.target_type == "continuous":
                weights = 1.0 + 1.0 * (yr_np >= q80).astype(np.float32) + 2.0 * (yr_np >= q90).astype(np.float32)
            else:
                weights = np.ones_like(yb_np, dtype=np.float32)
            loss, grads = loss_and_grad(
                model,
                xb,
                ab,
                rb,
                base.mx.array(yb_np, dtype=base.mx.float32),
                base.mx.array(yr_np, dtype=base.mx.float32),
                base.mx.array(weights, dtype=base.mx.float32),
            )
            grads, _ = base.optim.clip_grad_norm(grads, 1.0)
            optimizer.update(model, grads)
            base.mx.eval(model.parameters(), optimizer.state)
            total += float(np.asarray(loss))
            batches += 1
        if hasattr(model, "eval"):
            model.eval()
        val_score, val_reg, val_gate = forward_residual(
            model,
            pack.train_x[inner_val],
            ar_train_score[inner_val],
            ar_train_reg[inner_val],
            batch_size=batch_size,
            target_type=block.target_type,
        )
        if block.target_type == "binary":
            val_metric = (
                average_precision_score(selection_y[inner_val], val_score)
                if len(np.unique(selection_y[inner_val])) > 1
                else math.nan
            )
            delta = safe_float(val_metric - ar_inner_metric)
            current_key = (delta if math.isfinite(delta) else -math.inf, 0.0, 0.0)
        else:
            val_metrics = continuous_run.continuous_metric_row(
                selection_y[inner_train],
                ar_train_score[inner_train],
                selection_y[inner_val],
                val_score,
                selection_cont[inner_val],
                val_reg,
            )
            continuous_run.add_delta_metrics(val_metrics, ar_inner_metrics)
            current_key = (
                safe_float(val_metrics["delta_vs_frozen_ar_top_5pct_continuous_lift"]),
                safe_float(val_metrics["delta_vs_frozen_ar_continuous_spearman"]),
                safe_float(val_metrics["delta_vs_frozen_ar_top_10pct_continuous_lift"]),
            )
            delta = current_key[0]
            val_metric = current_key[0]
        curves.append(
            {
                "target_name": block.target_name,
                "target_type": block.target_type,
                "architecture": architecture,
                "control_type": control,
                "seed": int(seed),
                "epoch": int(epoch),
                "train_loss": total / max(1, batches),
                "inner_val_selection_metric": val_metric,
                "inner_val_delta_vs_frozen_ar": delta,
                "selection_metric_name": selection_name,
                "alpha": float(np.asarray(model.alpha_value())[0]),
                "gate_mean": float(np.mean(val_gate)),
                "gate_p95": float(np.quantile(val_gate, 0.95)),
                "label_policy": label_policy,
            }
        )
        if current_key > best_key:
            model.save_weights(str(best_path))
            best_key = current_key
            best_epoch = epoch
            stale = 0
            suppressed = False
        else:
            stale += 1
        if stale >= int(patience):
            early_stop = "patience_exhausted"
            break

    if suppressed:
        train_score = ar_train_score
        train_reg = ar_train_reg
        test_score = ar_test_score
        test_reg = ar_test_reg
        gate = np.zeros_like(test_score)
        checkpoint_restored = False
        checkpoint_checksum = None
        alpha_final = 0.0
    else:
        model.load_weights(str(best_path))
        if hasattr(model, "eval"):
            model.eval()
        train_score, train_reg, _ = forward_residual(
            model, pack.train_x, ar_train_score, ar_train_reg, batch_size=batch_size, target_type=block.target_type
        )
        test_score, test_reg, gate = forward_residual(
            model, pack.test_x, ar_test_score, ar_test_reg, batch_size=batch_size, target_type=block.target_type
        )
        checkpoint_restored = True
        checkpoint_checksum = base.file_digest(best_path)
        alpha_final = float(np.asarray(model.alpha_value())[0])
    metrics = metric_row_for_block(block, train_score, test_score, test_reg)
    ar_metrics = metric_row_for_block(block, ar_train_score, ar_test_score, ar_test_reg)
    add_deltas(metrics, ar_metrics, block.target_type)
    audit = {
        "best_epoch": int(best_epoch),
        "epochs_run": int(len(curves)),
        "best_inner_val_delta_vs_frozen_ar": float(best_key[0]),
        "best_inner_val_secondary_delta": float(best_key[1]),
        "early_stopping_reason": early_stop,
        "residual_suppressed": bool(suppressed),
        "checkpoint_restored": bool(checkpoint_restored),
        "checkpoint_path": str(best_path) if checkpoint_restored else None,
        "checkpoint_checksum": checkpoint_checksum,
        "eval_mode_scoring": True,
        "dropout_disabled": True,
        "label_policy": label_policy,
        "alpha_final": float(alpha_final),
        "hyperparameters": {
            "hidden": int(hp.get("hidden", 64)),
            "learning_rate": float(hp.get("learning_rate", 2e-4)),
            "weight_decay": float(hp.get("weight_decay", 1e-4)),
            "alpha_initial_logit": float(hp.get("alpha_initial_logit", -4.0)),
            "alpha_cap": float(hp.get("alpha_cap", 0.12)),
            "gate_bias": float(hp.get("gate_bias", 4.0)),
            "lambda_binary": float(hp.get("lambda_binary", 0.5)),
        },
        "gate_mean": float(np.mean(gate)) if len(gate) else math.nan,
        "gate_p05": float(np.quantile(gate, 0.05)) if len(gate) else math.nan,
        "gate_p50": float(np.quantile(gate, 0.50)) if len(gate) else math.nan,
        "gate_p95": float(np.quantile(gate, 0.95)) if len(gate) else math.nan,
    }
    return metrics, curves, audit


def summarize_metrics(fold_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["target_name", "target_type", "architecture", "control_type"]
    metric_cols = [
        "pr_auc",
        "roc_auc",
        "top_1pct_recall",
        "top_5pct_recall",
        "top_10pct_recall",
        "top_1pct_precision",
        "top_5pct_precision",
        "top_10pct_precision",
        "continuous_spearman",
        "continuous_pearson",
        "top_1pct_continuous_lift",
        "top_5pct_continuous_lift",
        "top_10pct_continuous_lift",
        "top_1pct_avg_true_movement",
        "top_5pct_avg_true_movement",
        "top_10pct_avg_true_movement",
        "ndcg_at_1pct",
        "ndcg_at_5pct",
        "ndcg_at_10pct",
        "delta_vs_frozen_ar_pr_auc",
        "delta_vs_frozen_ar_top_5pct_continuous_lift",
        "delta_vs_frozen_ar_continuous_spearman",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in fold_df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["seeds"] = int(group["seed"].nunique())
        row["rows_test_total"] = int(group["n_test"].sum())
        for metric in metric_cols:
            if metric in group:
                vals = pd.to_numeric(group[metric], errors="coerce")
                row[f"mean_{metric}"] = float(vals.mean()) if vals.notna().any() else math.nan
                row[f"min_{metric}"] = float(vals.min()) if vals.notna().any() else math.nan
                row[f"max_{metric}"] = float(vals.max()) if vals.notna().any() else math.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["target_name", "architecture", "control_type"])


def seed_deltas(fold_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (target, arch, seed), group in fold_df.groupby(["target_name", "architecture", "seed"]):
        vals = group.set_index("control_type")
        if "real_residual" not in vals.index:
            continue
        target_type = str(group["target_type"].iloc[0])
        real = vals.loc["real_residual"]
        row: dict[str, Any] = {"target_name": target, "target_type": target_type, "architecture": arch, "seed": int(seed)}
        if target_type == "binary":
            row["real_pr_auc"] = float(real["pr_auc"])
            for control in ("frozen_ar_only", *PRIMARY_CONTROLS):
                row[f"{control}_pr_auc"] = float(vals.loc[control, "pr_auc"])
                row[f"real_minus_{control}_pr_auc"] = float(real["pr_auc"] - vals.loc[control, "pr_auc"])
        else:
            row["real_top_5pct_lift"] = float(real["top_5pct_continuous_lift"])
            row["real_spearman"] = float(real["continuous_spearman"])
            for control in ("frozen_ar_only", *PRIMARY_CONTROLS):
                row[f"{control}_top_5pct_lift"] = float(vals.loc[control, "top_5pct_continuous_lift"])
                row[f"real_minus_{control}_top_5pct_lift"] = float(real["top_5pct_continuous_lift"] - vals.loc[control, "top_5pct_continuous_lift"])
                row[f"real_minus_{control}_spearman"] = float(real["continuous_spearman"] - vals.loc[control, "continuous_spearman"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["target_name", "architecture", "seed"])


def row_for(summary: pd.DataFrame, target: str, architecture: str, control: str) -> pd.Series:
    sub = summary[(summary["target_name"] == target) & (summary["architecture"] == architecture) & (summary["control_type"] == control)]
    if len(sub) != 1:
        raise RuntimeError(f"Expected one row for {target}/{architecture}/{control}, got {len(sub)}")
    return sub.iloc[0]


def compute_gates(summary: pd.DataFrame, fold_df: pd.DataFrame, seed_df: pd.DataFrame, leakage_pass: bool) -> dict[str, Any]:
    failed: list[str] = []

    binary_real = summary[(summary["target_name"] == BINARY_TARGET) & (summary["control_type"] == "real_residual")]
    b_real = binary_real.sort_values("mean_pr_auc", ascending=False).iloc[0]
    b_arch = str(b_real["architecture"])
    b_ar = row_for(summary, BINARY_TARGET, b_arch, "frozen_ar_only")
    b_controls = [row_for(summary, BINARY_TARGET, b_arch, control) for control in PRIMARY_CONTROLS]
    b_best_ctrl = max(b_controls, key=lambda row: float(row["mean_pr_auc"]))
    b_deltas = {
        "real_minus_frozen_ar_pr_auc": float(b_real["mean_pr_auc"] - b_ar["mean_pr_auc"]),
        "real_minus_best_control_pr_auc": float(b_real["mean_pr_auc"] - b_best_ctrl["mean_pr_auc"]),
    }
    for row in b_controls:
        b_deltas[f"real_minus_{row['control_type']}_pr_auc"] = float(b_real["mean_pr_auc"] - row["mean_pr_auc"])
    b_seed = seed_df[(seed_df["target_name"] == BINARY_TARGET) & (seed_df["architecture"] == b_arch)]
    b_seed_cols = [f"real_minus_{control}_pr_auc" for control in ("frozen_ar_only", *PRIMARY_CONTROLS)]
    b_seed_positive = int((b_seed[b_seed_cols].min(axis=1) > 0).sum())
    b_threshold = bool(b_deltas["real_minus_frozen_ar_pr_auc"] >= WEAK_THRESHOLD and all(b_deltas[f"real_minus_{c}_pr_auc"] >= WEAK_THRESHOLD for c in PRIMARY_CONTROLS))
    binary_pass = bool(leakage_pass and b_threshold and b_seed_positive >= 2)
    if not b_threshold:
        failed.append("binary_min_delta_threshold")
    if b_seed_positive < 2:
        failed.append("binary_seed_consistency")

    continuous_real = summary[(summary["target_name"] == CONTINUOUS_TARGET) & (summary["control_type"] == "real_residual")]
    c_real = continuous_real.sort_values("mean_top_5pct_continuous_lift", ascending=False).iloc[0]
    c_arch = str(c_real["architecture"])
    c_ar = row_for(summary, CONTINUOUS_TARGET, c_arch, "frozen_ar_only")
    c_controls = [row_for(summary, CONTINUOUS_TARGET, c_arch, control) for control in PRIMARY_CONTROLS]
    c_best_ctrl = max(c_controls, key=lambda row: float(row["mean_top_5pct_continuous_lift"]))
    c_deltas = {
        "real_minus_frozen_ar_top_5pct_lift": float(c_real["mean_top_5pct_continuous_lift"] - c_ar["mean_top_5pct_continuous_lift"]),
        "real_minus_best_control_top_5pct_lift": float(c_real["mean_top_5pct_continuous_lift"] - c_best_ctrl["mean_top_5pct_continuous_lift"]),
        "real_minus_frozen_ar_spearman": float(c_real["mean_continuous_spearman"] - c_ar["mean_continuous_spearman"]),
    }
    for row in c_controls:
        c_deltas[f"real_minus_{row['control_type']}_top_5pct_lift"] = float(c_real["mean_top_5pct_continuous_lift"] - row["mean_top_5pct_continuous_lift"])
    c_seed = seed_df[(seed_df["target_name"] == CONTINUOUS_TARGET) & (seed_df["architecture"] == c_arch)]
    c_seed_cols = [f"real_minus_{control}_top_5pct_lift" for control in ("frozen_ar_only", *PRIMARY_CONTROLS)]
    c_seed_positive = int((c_seed[c_seed_cols].min(axis=1) > 0).sum())
    c_threshold = bool(
        c_deltas["real_minus_frozen_ar_top_5pct_lift"] >= WEAK_THRESHOLD
        and all(c_deltas[f"real_minus_{control}_top_5pct_lift"] >= WEAK_THRESHOLD for control in PRIMARY_CONTROLS)
    )
    continuous_pass = bool(leakage_pass and c_threshold and c_deltas["real_minus_frozen_ar_spearman"] > 0 and c_seed_positive >= 2)
    if not c_threshold:
        failed.append("continuous_min_delta_threshold")
    if not c_deltas["real_minus_frozen_ar_spearman"] > 0:
        failed.append("continuous_spearman_delta")
    if c_seed_positive < 2:
        failed.append("continuous_seed_consistency")
    if not leakage_pass:
        failed.append("leakage_or_context_audit")

    frozen_integrity = bool(fold_df.groupby(["target_name", "architecture", "seed"])["frozen_ar_test_checksum"].nunique().max() == 1)
    checkpoint_restore = bool(fold_df["checkpoint_restore_pass"].all())
    eval_mode = bool(fold_df["eval_mode_scoring"].all())
    if not frozen_integrity:
        failed.append("frozen_ar_integrity")
    if not checkpoint_restore:
        failed.append("checkpoint_restore")
    if not eval_mode:
        failed.append("eval_mode_scoring")
    recommendation = (
        "temporal_residual_pass_review_before_grouped_or_504"
        if binary_pass and continuous_pass
        else "temporal_residual_blocked_failed_do_not_run_grouped_or_504"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "matrix_rows_expected": 168,
        "matrix_rows_actual": int(len(fold_df)),
        "protocol": PROTOCOL,
        "targets": [BINARY_TARGET, CONTINUOUS_TARGET],
        "architectures": list(ARCHITECTURES),
        "eval_mode_scoring_pass": eval_mode,
        "checkpoint_restore_pass": checkpoint_restore,
        "frozen_ar_integrity_pass": frozen_integrity,
        "leakage_context_audit_pass": bool(leakage_pass),
        "binary_pass": binary_pass,
        "continuous_pass": continuous_pass,
        "strict_forward_time_temporal_generalization_proven": False,
        "grouped_started": False,
        "recommendation": recommendation,
        "failed_gates": failed,
        "binary": {
            "target": BINARY_TARGET,
            "best_architecture": b_arch,
            "real_pr_auc": float(b_real["mean_pr_auc"]),
            "frozen_ar_pr_auc": float(b_ar["mean_pr_auc"]),
            "best_control": str(b_best_ctrl["control_type"]),
            "best_control_pr_auc": float(b_best_ctrl["mean_pr_auc"]),
            "seed_positive_count": b_seed_positive,
            "deltas": b_deltas,
        },
        "continuous": {
            "target": CONTINUOUS_TARGET,
            "best_architecture": c_arch,
            "real_spearman": float(c_real["mean_continuous_spearman"]),
            "frozen_ar_spearman": float(c_ar["mean_continuous_spearman"]),
            "real_top_5pct_lift": float(c_real["mean_top_5pct_continuous_lift"]),
            "frozen_ar_top_5pct_lift": float(c_ar["mean_top_5pct_continuous_lift"]),
            "best_control": str(c_best_ctrl["control_type"]),
            "best_control_top_5pct_lift": float(c_best_ctrl["mean_top_5pct_continuous_lift"]),
            "seed_positive_count": c_seed_positive,
            "deltas": c_deltas,
        },
    }


def write_report(path: Path, output_root: Path, gates: dict[str, Any]) -> None:
    b = gates["binary"]
    c = gates["continuous"]
    text = f"""# Phase 5 Temporal Residual Blocked Summary

Output root: `{output_root}`

This is a bounded blocked-only temporal/event-context residual diagnostic over the redesigned targets. It uses the fold-safe redesigned PCA256 artifacts and keeps frozen AR as the baseline floor. It does not run grouped, extra targets, V-JEPA/TRIBE/PCA, AR retraining, or claim changes.

## Binary Washout-Gap Target

- Target: `{b['target']}`
- Best architecture: `{b['best_architecture']}`
- Real PR-AUC: `{b['real_pr_auc']:.10f}`
- Frozen AR PR-AUC: `{b['frozen_ar_pr_auc']:.10f}`
- Best control: `{b['best_control']}` PR-AUC `{b['best_control_pr_auc']:.10f}`
- Delta vs frozen AR: `{b['deltas']['real_minus_frozen_ar_pr_auc']:+.10f}`
- Delta vs best control: `{b['deltas']['real_minus_best_control_pr_auc']:+.10f}`
- Seed positive count: `{b['seed_positive_count']}/3`
- Binary pass: `{gates['binary_pass']}`

## Continuous AR-Residualized Target

- Target: `{c['target']}`
- Best architecture: `{c['best_architecture']}`
- Real Spearman: `{c['real_spearman']:.10f}`
- Frozen AR Spearman: `{c['frozen_ar_spearman']:.10f}`
- Spearman delta vs frozen AR: `{c['deltas']['real_minus_frozen_ar_spearman']:+.10f}`
- Real top 5pct lift: `{c['real_top_5pct_lift']:.10f}`
- Frozen AR top 5pct lift: `{c['frozen_ar_top_5pct_lift']:.10f}`
- Best control: `{c['best_control']}` top 5pct lift `{c['best_control_top_5pct_lift']:.10f}`
- Delta vs frozen AR: `{c['deltas']['real_minus_frozen_ar_top_5pct_lift']:+.10f}`
- Delta vs best control: `{c['deltas']['real_minus_best_control_top_5pct_lift']:+.10f}`
- Seed positive count: `{c['seed_positive_count']}/3`
- Continuous pass: `{gates['continuous_pass']}`

## Gates

- `leakage_context_audit_pass`: `{gates['leakage_context_audit_pass']}`
- `frozen_ar_integrity_pass`: `{gates['frozen_ar_integrity_pass']}`
- `checkpoint_restore_pass`: `{gates['checkpoint_restore_pass']}`
- `eval_mode_scoring_pass`: `{gates['eval_mode_scoring_pass']}`
- Failed gates: `{gates['failed_gates']}`
- Recommendation: `{gates['recommendation']}`

Strict forward-time temporal generalization remains unproven. This diagnostic should not trigger grouped unless the blocked gates pass cleanly and the result is reviewed.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def finalize_output(output_root: Path, reports_dir: Path) -> dict[str, Any]:
    fold_df = pd.read_csv(output_root / "metrics" / "temporal_residual_blocked_seed_metrics.csv")
    context_audit = json.loads((output_root / "diagnostics" / "leakage_context_audit.json").read_text(encoding="utf-8"))
    summary = summarize_metrics(fold_df)
    seed_df = seed_deltas(fold_df)
    gates = compute_gates(summary, fold_df, seed_df, bool(context_audit.get("leakage_context_audit_pass")))
    summary.to_csv(output_root / "metrics" / "temporal_residual_blocked_summary_metrics.csv", index=False)
    summary.to_csv(output_root / "metrics" / "temporal_residual_blocked_architecture_comparison.csv", index=False)
    seed_df.to_csv(output_root / "metrics" / "temporal_residual_blocked_seed_deltas.csv", index=False)
    summary.to_csv(output_root / "promotion" / "temporal_residual_blocked_control_comparison.csv", index=False)
    write_json(output_root / "promotion" / "temporal_residual_blocked_gates.json", gates)
    write_json(output_root / "promotion" / "temporal_residual_blocked_adversarial_verdict.json", gates)
    write_json(
        output_root / "promotion" / "temporal_residual_blocked_failure_reasons.json",
        {"failed_gates": gates["failed_gates"], "recommendation": gates["recommendation"]},
    )
    best_heads = {"binary": gates["binary"], "continuous": gates["continuous"]}
    write_json(output_root / "promotion" / "temporal_residual_blocked_best_architectures.json", best_heads)
    stamp = output_root.name.replace("again_dense_2hz_phase5_temporal_residual_blocked_", "")
    report_name = f"again_dense_2hz_phase5_temporal_residual_blocked_summary_{stamp}.md"
    write_report(output_root / "reports" / report_name, output_root, gates)
    report_path = reports_dir / report_name
    write_report(report_path, output_root, gates)
    return {"gates": gates, "report_path": str(report_path)}


def matrix_rows() -> list[tuple[str, int, str, str]]:
    return [(target, seed, arch, control) for target in TARGET_TYPES for seed in SEEDS for arch in ARCHITECTURES for control in CONTROLS]


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root) if args.output_root else default_output_root()
    source_root = Path(args.source_root)
    pca_root = Path(args.foldsafe_pca_root)
    previous_root = Path(args.previous_redesigned_root)
    matrix = matrix_rows()
    print(json.dumps({"matrix_size": len(matrix), "max_allowed": 168, "targets": list(TARGET_TYPES), "architectures": list(ARCHITECTURES)}, indent=2))
    if len(matrix) > 168:
        raise RuntimeError(f"Refusing to exceed 168 rows: {len(matrix)}")
    pca_info = load_pca_manifest(pca_root)
    if args.dry_run:
        return 0
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output root: {output_root}")
    start = time.time()
    for sub in ("manifests", "metrics", "promotion", "diagnostics", "reports", "frozen_ar_scores", "checkpoints"):
        (output_root / sub).mkdir(parents=True, exist_ok=True)

    blocks, df, dense_root, residual_meta = build_blocks(source_root, pca_root)
    split_audits = [verify_pca_rows(pca_root, block_for_target(blocks, target), pca_info["rows"]) for target in TARGET_TYPES]
    fold_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    video_mean_rows: list[dict[str, Any]] = []
    ar_manifest: list[dict[str, Any]] = []

    for target_name in (BINARY_TARGET, CONTINUOUS_TARGET):
        block = block_for_target(blocks, target_name)
        ar_by_seed: dict[int, dict[str, Any]] = {}
        ar_metrics_by_seed: dict[int, dict[str, Any]] = {}
        for seed in SEEDS:
            ar = load_or_cache_frozen_ar(previous_root, source_root, output_root, block, seed, args.batch_size)
            ar_by_seed[seed] = ar
            ar_metrics = metric_row_for_block(block, ar["train_score"], ar["test_score"], ar["test_reg"])
            ar_metrics_by_seed[seed] = ar_metrics
            ar_manifest.append({k: v for k, v in ar.items() if k not in {"train_score", "train_reg", "test_score", "test_reg"}})
        for architecture in ARCHITECTURES:
            for seed in SEEDS:
                ar = ar_by_seed[seed]
                ar_metrics = ar_metrics_by_seed[seed]
                for control in CONTROLS:
                    if control == "frozen_ar_only":
                        row = {
                            "schema_version": SCHEMA_VERSION,
                            "target_name": target_name,
                            "target_type": block.target_type,
                            "validation_protocol": PROTOCOL,
                            "fold": FOLD,
                            "seed": int(seed),
                            "architecture": architecture,
                            "control_type": control,
                            "feature_name": FEATURE_NAME,
                            "loss_objective": "frozen_ar_only_eval_mode",
                            "n_train": int(len(block.train_idx)),
                            "n_test": int(len(block.test_idx)),
                            "checkpoint_restore_pass": True,
                            "eval_mode_scoring": True,
                            "dropout_disabled": True,
                            "ar_retrained": False,
                            "frozen_ar_train_checksum": ar["train_checksum"],
                            "frozen_ar_test_checksum": ar["test_checksum"],
                            **ar_metrics,
                        }
                        fold_rows.append(row)
                        continue
                    pack = feature_pack_for(df, dense_root, pca_root, block, architecture, control, seed)
                    metrics, curves, audit = train_temporal_residual(
                        architecture=architecture,
                        control=control,
                        pack=pack,
                        block=block,
                        ar=ar,
                        seed=seed,
                        output_root=output_root,
                        batch_size=args.batch_size,
                        max_epochs=args.max_epochs,
                        patience=args.patience,
                    )
                    for curve in curves:
                        curve_rows.append(curve)
                    feature_rows.append(
                        {
                            "target_name": target_name,
                            "target_type": block.target_type,
                            "architecture": architecture,
                            "control_type": control,
                            "seed": int(seed),
                            "dims": pack.dims,
                            "blocks": pack.manifest,
                        }
                    )
                    context_rows.append(pack.context_audit)
                    if control == "label_permutation_residual":
                        label_rows.append(
                            {
                                "target_name": target_name,
                                "target_type": block.target_type,
                                "architecture": architecture,
                                "seed": int(seed),
                                "best_epoch": audit["best_epoch"],
                                "label_policy": audit["label_policy"],
                                "heldout_scoring_policy": "true_heldout_labels_targets",
                                "best_inner_val_delta_vs_frozen_ar": audit["best_inner_val_delta_vs_frozen_ar"],
                            }
                        )
                    if control == "train_only_video_mean_residual":
                        video_mean_rows.append(
                            {
                                "target_name": target_name,
                                "target_type": block.target_type,
                                "architecture": architecture,
                                "seed": int(seed),
                                "uses_test_rows_for_mean": False,
                                "best_epoch": audit["best_epoch"],
                                "checkpoint_restored": audit["checkpoint_restored"],
                            }
                        )
                    fold_rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "target_name": target_name,
                            "target_type": block.target_type,
                            "validation_protocol": PROTOCOL,
                            "fold": FOLD,
                            "seed": int(seed),
                            "architecture": architecture,
                            "control_type": control,
                            "feature_name": FEATURE_NAME,
                            "loss_objective": "binary_regression_plus_binary" if block.target_type == "binary" else "continuous_huber_top_percent_weighted",
                            "n_train": int(len(block.train_idx)),
                            "n_test": int(len(block.test_idx)),
                            "checkpoint_restore_pass": audit["checkpoint_restored"] or audit["residual_suppressed"],
                            "ar_retrained": False,
                            "frozen_ar_train_checksum": ar["train_checksum"],
                            "frozen_ar_test_checksum": ar["test_checksum"],
                            **audit,
                            **metrics,
                        }
                    )
                    pd.DataFrame(fold_rows).to_csv(output_root / "metrics" / "temporal_residual_blocked_seed_metrics.partial.csv", index=False)
                    gc.collect()

    fold_df = pd.DataFrame(fold_rows)
    if len(fold_df) != 168:
        raise RuntimeError(f"Expected 168 rows, got {len(fold_df)}")
    fold_df.to_csv(output_root / "metrics" / "temporal_residual_blocked_seed_metrics.csv", index=False)
    fold_df.to_csv(output_root / "metrics" / "temporal_residual_blocked_fold_metrics.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(output_root / "diagnostics" / "training_curve_summary.csv", index=False)
    pd.DataFrame(label_rows).to_csv(output_root / "diagnostics" / "label_permutation_audit.csv", index=False)
    pd.DataFrame(video_mean_rows).to_csv(output_root / "diagnostics" / "train_only_video_mean_audit.csv", index=False)
    leakage_context_pass = bool(
        pca_info["audit"].get("leakage_audit_pass")
        and pca_info["audit"].get("no_test_rows_used_in_pca_fit")
        and pca_info["audit"].get("row_counts_match_redesigned_split")
        and all(row.get("temporal_context_causal_only") for row in context_rows)
        and not any(row.get("uses_centered_or_future_windows") for row in context_rows)
        and all(row.get("same_video_history_masking") for row in context_rows)
        and all(row.get("label_policy") == "permuted_train_and_permuted_inner_val_selection" for row in label_rows)
        and not any(row.get("uses_test_rows_for_mean") for row in video_mean_rows)
    )
    context_audit = {
        "schema_version": SCHEMA_VERSION,
        "leakage_context_audit_pass": leakage_context_pass,
        "foldsafe_pca_audit": pca_info["audit"],
        "split_row_audits": split_audits,
        "context_rows": context_rows,
        "label_permutation_policy_pass": all(row.get("label_policy") == "permuted_train_and_permuted_inner_val_selection" for row in label_rows),
        "train_only_video_mean_pass": not any(row.get("uses_test_rows_for_mean") for row in video_mean_rows),
        "temporal_context_causal_only": all(row.get("temporal_context_causal_only") for row in context_rows),
        "no_centered_or_future_windows": not any(row.get("uses_centered_or_future_windows") for row in context_rows),
        "same_video_history_masking": all(row.get("same_video_history_masking") for row in context_rows),
    }
    write_json(output_root / "diagnostics" / "leakage_context_audit.json", context_audit)
    write_json(output_root / "diagnostics" / "label_permutation_audit.json", {"policy_implemented": True, "rows": label_rows})
    write_json(output_root / "diagnostics" / "train_only_video_mean_audit.json", {"train_only_video_mean_primary_static_control": True, "rows": video_mean_rows})
    write_json(output_root / "manifests" / "frozen_ar_manifest.json", {"ar_retrained": False, "scores": ar_manifest})
    write_json(output_root / "manifests" / "feature_manifest.json", {"features": feature_rows, "row_count": len(feature_rows)})
    write_json(
        output_root / "manifests" / "run_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_at": now_iso(),
            "source_root": str(source_root),
            "foldsafe_pca_root": str(pca_root),
            "previous_redesigned_root": str(previous_root),
            "output_root": str(output_root),
            "dense_root": str(dense_root),
            "targets": list(TARGET_TYPES),
            "architectures": list(ARCHITECTURES),
            "controls": list(CONTROLS),
            "feature": FEATURE_NAME,
            "protocol_scope": "blocked_temporal_70_30_only",
            "matrix_size": len(matrix),
            "no_grouped": True,
            "no_extra_targets": True,
            "no_vjepa_tribe_pca_rerun": True,
            "no_pca_refit": True,
            "no_ar_retraining": True,
            "residual_target_definition": residual_meta,
            "duration_seconds": time.time() - start,
        },
    )
    finalized = finalize_output(output_root, Path(args.reports_dir))
    gates = finalized["gates"]
    print(json.dumps(fr.clean_json({"run_completed": True, "output_root": str(output_root), "report": finalized["report_path"], **gates}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
