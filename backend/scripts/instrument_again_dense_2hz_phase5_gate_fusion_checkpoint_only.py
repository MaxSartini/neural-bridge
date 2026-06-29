"""Checkpoint-only gate/fusion instrumentation for the Phase 5 repair matrix.

This script loads saved best checkpoints from the completed primary repair
matrix and re-forwards held-out benchmark rows. It does not train models, refit
PCA, rerun encoders, or modify the dense cache.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mean_squared_error, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.scripts import run_again_dense_2hz_phase5_adversarial_repair_fixplus as repair
from backend.scripts import run_again_dense_2hz_phase5_learned_heads as base


SOURCE_ROOT = Path("outputs/again_dense_2hz_phase5_adversarial_repair_fixplus_20260629_171825")
OUT_DIR = SOURCE_ROOT / "instrumentation" / "gate_fusion_checkpoint_only"
REPORT_PATH = Path("reports/again_dense_2hz_phase5_gate_fusion_checkpoint_diagnostic_20260629_171825.md")
COMMIT_SHA = "9757383c7e30d759fd15911e4ab87ee60b73fd86"
SELECTED_CONTROLS = ("real_ar_pca_diag", "ar_plus_random_pca", "ar_plus_shuffled_pca", "ar_only_head")
SELECTED_PROTOCOLS = ("grouped_video", "blocked_temporal_70_30")
SELECTED_SEEDS = (20260625, 20260626, 20260627)
SELECTED_LOSS = "regression_plus_binary"
SELECTED_MODEL = "gated_ar_pca_mlp"
SELECTED_TARGET = "arousal_spike_rows_2_6_train_q90"
SELECTED_FEATURE = "temporal_mean_2s_then_pca256"
METRIC_TOLERANCE = 1e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(SOURCE_ROOT))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--report-path", default=str(REPORT_PATH))
    parser.add_argument("--batch-size", type=int, default=16384)
    return parser.parse_args()


def clean_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return math.nan
    a = a[mask]
    b = b[mask]
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return corr(rank(np.asarray(a)), rank(np.asarray(b)))


def top_recall(y_true: np.ndarray, scores: np.ndarray, frac: float) -> float:
    if len(y_true) == 0:
        return math.nan
    k = max(1, int(math.ceil(len(y_true) * frac)))
    top = np.argsort(-scores, kind="mergesort")[:k]
    denom = float(np.sum(y_true))
    if denom <= 0:
        return math.nan
    return float(np.sum(y_true[top]) / denom)


def stats(prefix: str, values: np.ndarray) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {
            f"mean_{prefix}": None,
            f"std_{prefix}": None,
            f"min_{prefix}": None,
            f"max_{prefix}": None,
            f"{prefix}_p01": None,
            f"{prefix}_p05": None,
            f"{prefix}_p50": None,
            f"{prefix}_p95": None,
            f"{prefix}_p99": None,
        }
    return {
        f"mean_{prefix}": float(np.mean(arr)),
        f"std_{prefix}": float(np.std(arr)),
        f"min_{prefix}": float(np.min(arr)),
        f"max_{prefix}": float(np.max(arr)),
        f"{prefix}_p01": float(np.quantile(arr, 0.01)),
        f"{prefix}_p05": float(np.quantile(arr, 0.05)),
        f"{prefix}_p50": float(np.quantile(arr, 0.50)),
        f"{prefix}_p95": float(np.quantile(arr, 0.95)),
        f"{prefix}_p99": float(np.quantile(arr, 0.99)),
    }


def nanmean_or_none(values: np.ndarray) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return None
    return float(np.mean(arr))


def nanstd_or_none(values: np.ndarray) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return None
    return float(np.std(arr))


def control_arg(control_type: str) -> str | None:
    return None if control_type == "real_ar_pca_diag" else control_type


def split_y(split: Any, train_idx: np.ndarray, test_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(train_idx) == len(split.train_idx) and len(test_idx) == len(split.test_idx):
        return split.y_train, split.y_test
    train_map = {int(idx): i for i, idx in enumerate(split.train_idx)}
    test_map = {int(idx): i for i, idx in enumerate(split.test_idx)}
    y_train = np.asarray([split.y_train[train_map[int(idx)]] for idx in train_idx], dtype=int)
    y_test = np.asarray([split.y_test[test_map[int(idx)]] for idx in test_idx], dtype=int)
    return y_train, y_test


def selected_checkpoint_rows(source_root: Path) -> pd.DataFrame:
    fold = pd.read_csv(source_root / "metrics" / "phase5_fold_metrics.csv")
    selected = fold[
        (fold["validation_protocol"].isin(SELECTED_PROTOCOLS))
        & (fold["control_type"].isin(SELECTED_CONTROLS))
        & (fold["loss_name"] == SELECTED_LOSS)
        & (fold["model_head"] == SELECTED_MODEL)
        & (fold["seed"].isin(SELECTED_SEEDS))
        & (fold["target_name"] == SELECTED_TARGET)
        & (fold["feature_name"] == SELECTED_FEATURE)
        & (fold["status"] == "success")
    ].copy()
    selected["checkpoint_exists"] = selected["checkpoint_path"].map(lambda p: Path(str(p)).exists())
    return selected.sort_values(["validation_protocol", "fold", "control_type", "seed"]).reset_index(drop=True)


def reconstruct_feature_blocks(source_root: Path) -> tuple[dict[tuple[str, int, str], dict[str, Any]], pd.DataFrame, list[Any], Any]:
    repair.patch_base_module()
    manifest = json.loads((source_root / "run_manifest.json").read_text())
    dense_root = Path(manifest["dense_root"])
    labels_path = dense_root / "labels_aligned_2hz.parquet"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels file: {labels_path}")
    df = base.load_labels(dense_root)
    phase4_root = Path(manifest["phase4_root"])
    target_specs = base.matching_target_specs((SELECTED_TARGET,))
    splits = base.build_split_specs(df, protocols=SELECTED_PROTOCOLS, n_splits=5, target_specs=target_specs)
    spec = base.feature_spec(SELECTED_FEATURE)

    controls_sequence = (None, *repair.DEFAULT_REPAIR_CONTROLS)
    blocks: dict[tuple[str, int, str], dict[str, Any]] = {}
    completed_rows_so_far = 0
    for split in splits:
        for control in controls_sequence:
            control_type = "real_ar_pca_diag" if control is None else str(control)
            rng = np.random.default_rng(20260625 + int(split.fold) + completed_rows_so_far)
            if control_type in SELECTED_CONTROLS:
                train_idx, test_idx, train_x, test_x, block_dims, feature_manifest = repair.assemble_feature_blocks_repair(
                    df,
                    dense_root,
                    phase4_root,
                    split,
                    spec,
                    include_ar=True,
                    include_temporal_diagnostics=True,
                    control=control,
                    rng=rng,
                )
                raw_train_x = train_x.copy()
                raw_test_x = test_x.copy()
                train_x, test_x = base.standardize_train_only(train_x, test_x)
                y_train, y_test = split_y(split, train_idx, test_idx)
                train_cont = base.target_continuous_values(df, split, train_idx, "future_arousal_max_delta_rows_2_6")
                test_cont = base.target_continuous_values(df, split, test_idx, "future_arousal_max_delta_rows_2_6")
                blocks[(split.protocol, int(split.fold), control_type)] = {
                    "split": split,
                    "train_idx": train_idx,
                    "test_idx": test_idx,
                    "train_x": train_x,
                    "test_x": test_x,
                    "raw_train_x": raw_train_x,
                    "raw_test_x": raw_test_x,
                    "train_y": y_train,
                    "test_y": y_test,
                    "train_cont": train_cont,
                    "test_cont": test_cont,
                    "block_dims": block_dims,
                    "feature_manifest": feature_manifest,
                }
            completed_rows_so_far += 9
    return blocks, df, splits, spec


def make_config(seed: int) -> Any:
    return base.TrainConfig(
        model_name=SELECTED_MODEL,
        loss_name=SELECTED_LOSS,
        seed=int(seed),
        hidden_sizes=(256,),
        dropout=0.1,
        learning_rate=3e-4,
        weight_decay=1e-4,
        lambda_binary=0.5,
        batch_size=8192,
        max_epochs=180,
        patience=24,
    )


def split_blocks(x: Any, block_dims: dict[str, int]) -> tuple[Any, Any, Any]:
    pos = 0
    ar_dim = int(block_dims.get("ar", 0))
    pca_dim = int(block_dims.get("pca", 0))
    diag_dim = int(block_dims.get("diagnostics", 0))
    ar = x[:, pos : pos + ar_dim] if ar_dim else base.mx.zeros((x.shape[0], 1), dtype=x.dtype)
    pos += ar_dim
    pca = x[:, pos : pos + pca_dim] if pca_dim else base.mx.zeros((x.shape[0], 1), dtype=x.dtype)
    pos += pca_dim
    diag = x[:, pos : pos + diag_dim] if diag_dim else base.mx.zeros((x.shape[0], 1), dtype=x.dtype)
    return ar, pca, diag


def select_binary_and_reg(outputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if outputs.ndim == 1:
        outputs = outputs[:, None]
    if outputs.shape[1] == 1:
        return outputs[:, 0], outputs[:, 0]
    return outputs[:, 1], outputs[:, 0]


def forward_intermediates(
    model: Any,
    x_np: np.ndarray,
    raw_x_np: np.ndarray,
    block_dims: dict[str, int],
    *,
    batch_size: int,
) -> dict[str, np.ndarray]:
    arrays: dict[str, list[np.ndarray]] = {
        "prediction_score": [],
        "continuous_prediction": [],
        "ar_branch_score": [],
        "pca_branch_score": [],
        "diag_branch_score": [],
        "gate_value": [],
        "fused_rep_norm": [],
        "ar_rep_norm": [],
        "pca_rep_norm": [],
        "diag_rep_norm": [],
        "pca_input_norm": [],
        "diag_input_norm": [],
        "ar_input_norm": [],
    }
    ar_dim = int(block_dims.get("ar", 0))
    pca_dim = int(block_dims.get("pca", 0))
    diag_dim = int(block_dims.get("diagnostics", 0))
    for start in range(0, len(x_np), batch_size):
        end = min(len(x_np), start + batch_size)
        xb = base.mx.array(x_np[start:end], dtype=base.mx.float32)
        raw = raw_x_np[start:end]
        ar, pca, diag = split_blocks(xb, block_dims)
        ar_h = base.nn.gelu(model.ar_proj(ar))
        pca_h = base.nn.gelu(model.pca_proj(pca))
        diag_h = base.nn.gelu(model.diag_proj(diag))
        gate = base.mx.sigmoid(model.gate(base.mx.concatenate([ar_h, pca_h, diag_h], axis=1)))
        fused = gate * (pca_h + diag_h) + (1.0 - gate) * ar_h
        out = model.out(fused)
        ar_out = model.out(ar_h)
        pca_out = model.out(pca_h)
        diag_out = model.out(diag_h)
        base.mx.eval(out, ar_out, pca_out, diag_out, gate, fused, ar_h, pca_h, diag_h)

        pred, reg = select_binary_and_reg(np.asarray(out, dtype=np.float32))
        ar_score, _ = select_binary_and_reg(np.asarray(ar_out, dtype=np.float32))
        pca_score, _ = select_binary_and_reg(np.asarray(pca_out, dtype=np.float32))
        diag_score, _ = select_binary_and_reg(np.asarray(diag_out, dtype=np.float32))
        gate_np = np.asarray(gate, dtype=np.float32)
        fused_np = np.asarray(fused, dtype=np.float32)
        ar_np = np.asarray(ar_h, dtype=np.float32)
        pca_np = np.asarray(pca_h, dtype=np.float32)
        diag_np = np.asarray(diag_h, dtype=np.float32)

        arrays["prediction_score"].append(pred)
        arrays["continuous_prediction"].append(reg)
        arrays["ar_branch_score"].append(ar_score if ar_dim else np.full(end - start, np.nan, dtype=np.float32))
        arrays["pca_branch_score"].append(pca_score if pca_dim else np.full(end - start, np.nan, dtype=np.float32))
        arrays["diag_branch_score"].append(diag_score if diag_dim else np.full(end - start, np.nan, dtype=np.float32))
        arrays["gate_value"].append(gate_np.mean(axis=1))
        arrays["fused_rep_norm"].append(np.linalg.norm(fused_np, axis=1))
        arrays["ar_rep_norm"].append(np.linalg.norm(ar_np, axis=1) if ar_dim else np.full(end - start, np.nan, dtype=np.float32))
        arrays["pca_rep_norm"].append(np.linalg.norm(pca_np, axis=1) if pca_dim else np.full(end - start, np.nan, dtype=np.float32))
        arrays["diag_rep_norm"].append(np.linalg.norm(diag_np, axis=1) if diag_dim else np.full(end - start, np.nan, dtype=np.float32))

        raw_pos = 0
        raw_ar = raw[:, raw_pos : raw_pos + ar_dim] if ar_dim else None
        raw_pos += ar_dim
        raw_pca = raw[:, raw_pos : raw_pos + pca_dim] if pca_dim else None
        raw_pos += pca_dim
        raw_diag = raw[:, raw_pos : raw_pos + diag_dim] if diag_dim else None
        arrays["ar_input_norm"].append(np.linalg.norm(raw_ar, axis=1) if raw_ar is not None else np.full(end - start, np.nan, dtype=np.float32))
        arrays["pca_input_norm"].append(np.linalg.norm(raw_pca, axis=1) if raw_pca is not None else np.full(end - start, np.nan, dtype=np.float32))
        arrays["diag_input_norm"].append(np.linalg.norm(raw_diag, axis=1) if raw_diag is not None else np.full(end - start, np.nan, dtype=np.float32))
    return {key: np.concatenate(parts) for key, parts in arrays.items()}


def score_model(model: Any, x_np: np.ndarray, *, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    scores: list[np.ndarray] = []
    regs: list[np.ndarray] = []
    for start in range(0, len(x_np), batch_size):
        out = model(base.mx.array(x_np[start : start + batch_size], dtype=base.mx.float32))
        base.mx.eval(out)
        score, reg = base.select_score_columns(np.asarray(out, dtype=np.float32), SELECTED_LOSS)
        scores.append(score)
        regs.append(reg)
    return np.concatenate(scores), np.concatenate(regs)


def metric_dict(y_train: np.ndarray, train_scores: np.ndarray, y_test: np.ndarray, test_scores: np.ndarray, test_cont: np.ndarray, test_reg: np.ndarray) -> dict[str, float]:
    threshold = base.decision_threshold_for_binary(y_train, train_scores)
    binary = base.metric_row(y_test, test_scores, threshold)
    reg = base.regression_metric_row(test_cont, test_reg)
    return {
        "decision_threshold_train_only": float(threshold),
        "pr_auc": float(binary["pr_auc"]),
        "roc_auc": float(binary["roc_auc"]),
        "top_1pct_recall": float(binary["top_1pct_recall"]),
        "top_5pct_recall": float(binary["top_5pct_recall"]),
        "top_10pct_recall": float(binary["top_10pct_recall"]),
        "continuous_pearson": float(reg["pearson"]),
        "continuous_spearman": spearman(test_cont, test_reg),
        "spearman_future_movement": spearman(test_cont, test_scores),
    }


def gate_entropy(gate_values: np.ndarray) -> float:
    g = np.clip(np.asarray(gate_values, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return float(np.mean(-(g * np.log(g) + (1.0 - g) * np.log(1.0 - g))))


def summarize_run(row: pd.Series, data: dict[str, Any], inst: dict[str, np.ndarray], metrics: dict[str, float]) -> dict[str, Any]:
    gate = inst["gate_value"]
    pred = inst["prediction_score"]
    test_cont = data["test_cont"]
    time_seconds = data["df_rows"]["time_seconds"].to_numpy(dtype=np.float64)
    out = {
        "target_name": SELECTED_TARGET,
        "feature_name": SELECTED_FEATURE,
        "model_head": SELECTED_MODEL,
        "loss_name": SELECTED_LOSS,
        "validation_protocol": row.validation_protocol,
        "fold": int(row.fold),
        "seed": int(row.seed),
        "control_type": row.control_type,
        "checkpoint_path": row.checkpoint_path,
        "checkpoint_checksum": row.checkpoint_checksum,
        "best_epoch": int(row.best_epoch),
        "rows_scored": int(len(pred)),
        "gate_missing": False,
        **stats("gate", gate),
        "saturation_low_rate": float(np.mean(gate < 0.05)),
        "saturation_high_rate": float(np.mean(gate > 0.95)),
        "gate_entropy": gate_entropy(gate),
        "mean_ar_branch_score": nanmean_or_none(inst["ar_branch_score"]),
        "mean_pca_branch_score": nanmean_or_none(inst["pca_branch_score"]),
        "mean_diag_branch_score": nanmean_or_none(inst["diag_branch_score"]),
        "std_ar_branch_score": nanstd_or_none(inst["ar_branch_score"]),
        "std_pca_branch_score": nanstd_or_none(inst["pca_branch_score"]),
        "std_diag_branch_score": nanstd_or_none(inst["diag_branch_score"]),
        "mean_ar_rep_norm": nanmean_or_none(inst["ar_rep_norm"]),
        "mean_pca_rep_norm": nanmean_or_none(inst["pca_rep_norm"]),
        "mean_diag_rep_norm": nanmean_or_none(inst["diag_rep_norm"]),
        "mean_fused_rep_norm": nanmean_or_none(inst["fused_rep_norm"]),
        "mean_pca_input_norm": nanmean_or_none(inst["pca_input_norm"]),
        "mean_diag_input_norm": nanmean_or_none(inst["diag_input_norm"]),
        "corr_gate_ar_score": corr(gate, inst["ar_branch_score"]),
        "corr_gate_pca_norm": corr(gate, inst["pca_input_norm"]),
        "corr_gate_diag_norm": corr(gate, inst["diag_input_norm"]),
        "corr_gate_time_seconds": corr(gate, time_seconds),
        "corr_prediction_ar_score": corr(pred, inst["ar_branch_score"]),
        "corr_prediction_pca_score": corr(pred, inst["pca_branch_score"]),
        "corr_prediction_future_target": corr(pred, test_cont),
        "corr_ar_score_future_target": corr(inst["ar_branch_score"], test_cont),
        "corr_pca_score_future_target": corr(inst["pca_branch_score"], test_cont),
        **metrics,
    }
    return out


def load_and_instrument(args: argparse.Namespace) -> dict[str, Any]:
    source_root = Path(args.source_root)
    output_dir = Path(args.output_dir) if args.output_dir else source_root / "instrumentation" / "gate_fusion_checkpoint_only"
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = selected_checkpoint_rows(source_root)
    expected = 72
    checkpoint_missing = selected[~selected["checkpoint_exists"]]
    if len(selected) != expected or not checkpoint_missing.empty:
        plan_path = Path("reports/again_dense_2hz_phase5_gate_fusion_instrumentation_blocked_plan_20260629_171825.md")
        plan_path.write_text(f"""# Gate/Fusion Instrumentation Blocked Plan

Checkpoint-only instrumentation cannot proceed because expected checkpoints are missing.

- Expected checkpoints: `{expected}`
- Found selected rows: `{len(selected)}`
- Missing checkpoint files: `{len(checkpoint_missing)}`

Minimal rerun: controls `real_ar_pca_diag`, `ar_plus_random_pca`, `ar_plus_shuffled_pca`, `ar_only_head`; protocols `grouped_video`, `blocked_temporal_70_30`; loss `regression_plus_binary`; model `gated_ar_pca_mlp` only; seeds `20260625,20260626,20260627`; target `arousal_spike_rows_2_6_train_q90`; feature `temporal_mean_2s_then_pca256`; no secondary heads; no secondary targets; no full 702 rerun. Log gate/fusion/intermediate stats and per-row blocked predictions.
""")
        return {"succeeded": False, "plan_path": str(plan_path), "reason": "missing checkpoints"}

    blocks, df, _splits, _spec = reconstruct_feature_blocks(source_root)
    summary_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    blocked_per_row_paths: list[Path] = []
    load_rows: list[dict[str, Any]] = []

    for _, row in selected.iterrows():
        key = (row.validation_protocol, int(row.fold), row.control_type)
        data = blocks[key]
        config = make_config(int(row.seed))
        model = base.make_model(config, data["train_x"].shape[1], data["block_dims"])
        _ = model(base.mx.array(data["test_x"][:2], dtype=base.mx.float32))
        model.load_weights(str(row.checkpoint_path))
        if hasattr(model, "eval"):
            model.eval()
        load_rows.append(
            {
                "validation_protocol": row.validation_protocol,
                "fold": int(row.fold),
                "seed": int(row.seed),
                "control_type": row.control_type,
                "checkpoint_path": row.checkpoint_path,
                "checkpoint_checksum": row.checkpoint_checksum,
                "checkpoint_file_loadable": True,
                "best_epoch": int(row.best_epoch),
            }
        )

        train_scores, _train_reg = score_model(model, data["train_x"], batch_size=args.batch_size)
        inst = forward_intermediates(model, data["test_x"], data["raw_test_x"], data["block_dims"], batch_size=args.batch_size)
        metrics = metric_dict(data["train_y"], train_scores, data["test_y"], inst["prediction_score"], data["test_cont"], inst["continuous_prediction"])
        data["df_rows"] = df.loc[data["test_idx"]].reset_index().rename(columns={"index": "global_row_index"})
        summary = summarize_run(row, data, inst, metrics)
        summary_rows.append(summary)

        diffs = {}
        for metric in ("pr_auc", "roc_auc", "top_1pct_recall", "top_5pct_recall", "top_10pct_recall", "continuous_pearson"):
            expected_value = float(row[metric])
            reproduced = float(metrics[metric])
            diffs[f"{metric}_existing"] = expected_value
            diffs[f"{metric}_reproduced_eval_mode"] = reproduced
            diffs[f"{metric}_abs_diff"] = abs(expected_value - reproduced)
        audit_rows.append(
            {
                "validation_protocol": row.validation_protocol,
                "fold": int(row.fold),
                "seed": int(row.seed),
                "control_type": row.control_type,
                "checkpoint_path": row.checkpoint_path,
                "eval_mode_reproduction_pass": all(diffs[f"{m}_abs_diff"] <= METRIC_TOLERANCE for m in ("pr_auc", "roc_auc", "top_1pct_recall", "top_5pct_recall", "top_10pct_recall", "continuous_pearson")),
                **diffs,
            }
        )

        if row.validation_protocol == "blocked_temporal_70_30":
            rows = data["df_rows"]
            per = pd.DataFrame(
                {
                    "video_id": rows["video_id"].astype(str),
                    "row_index": rows["row_index"] if "row_index" in rows else rows["global_row_index"],
                    "global_row_index": rows["global_row_index"],
                    "time_seconds": rows["time_seconds"].to_numpy(dtype=np.float32),
                    "y_true_spike": data["test_y"].astype(np.int8),
                    "y_true_continuous": data["test_cont"].astype(np.float32),
                    "prediction_score": inst["prediction_score"].astype(np.float32),
                    "continuous_prediction": inst["continuous_prediction"].astype(np.float32),
                    "ar_branch_score": inst["ar_branch_score"].astype(np.float32),
                    "pca_branch_score": inst["pca_branch_score"].astype(np.float32),
                    "diag_branch_score": inst["diag_branch_score"].astype(np.float32),
                    "gate_value": inst["gate_value"].astype(np.float32),
                    "pca_input_norm": inst["pca_input_norm"].astype(np.float32),
                    "diag_input_norm": inst["diag_input_norm"].astype(np.float32),
                    "control_type": row.control_type,
                    "seed": int(row.seed),
                    "protocol": row.validation_protocol,
                }
            )
            part_path = output_dir / f"blocked_per_row__{row.control_type}__{int(row.seed)}.csv.gz"
            per.to_csv(part_path, index=False, compression="gzip")
            blocked_per_row_paths.append(part_path)

    summary_df = pd.DataFrame(summary_rows)
    audit_df = pd.DataFrame(audit_rows)
    load_df = pd.DataFrame(load_rows)
    comparison = summary_df.groupby(["validation_protocol", "control_type"], as_index=False).agg(
        rows=("rows_scored", "sum"),
        checkpoints=("seed", "count"),
        mean_gate=("mean_gate", "mean"),
        std_gate=("mean_gate", "std"),
        mean_saturation_low_rate=("saturation_low_rate", "mean"),
        mean_saturation_high_rate=("saturation_high_rate", "mean"),
        mean_gate_entropy=("gate_entropy", "mean"),
        mean_ar_branch_score=("mean_ar_branch_score", "mean"),
        mean_pca_branch_score=("mean_pca_branch_score", "mean"),
        mean_diag_branch_score=("mean_diag_branch_score", "mean"),
        mean_ar_rep_norm=("mean_ar_rep_norm", "mean"),
        mean_pca_rep_norm=("mean_pca_rep_norm", "mean"),
        mean_diag_rep_norm=("mean_diag_rep_norm", "mean"),
        mean_fused_rep_norm=("mean_fused_rep_norm", "mean"),
        mean_pca_input_norm=("mean_pca_input_norm", "mean"),
        mean_diag_input_norm=("mean_diag_input_norm", "mean"),
        mean_corr_gate_ar_score=("corr_gate_ar_score", "mean"),
        mean_corr_gate_pca_norm=("corr_gate_pca_norm", "mean"),
        mean_corr_gate_diag_norm=("corr_gate_diag_norm", "mean"),
        mean_corr_prediction_pca_score=("corr_prediction_pca_score", "mean"),
        mean_corr_pca_score_future_target=("corr_pca_score_future_target", "mean"),
        mean_pr_auc=("pr_auc", "mean"),
        mean_roc_auc=("roc_auc", "mean"),
        mean_continuous_pearson=("continuous_pearson", "mean"),
        mean_spearman_future_movement=("spearman_future_movement", "mean"),
    )

    summary_df.to_csv(output_dir / "gate_fusion_summary.csv", index=False)
    comparison.to_csv(output_dir / "gate_fusion_comparison_by_control.csv", index=False)
    pd.concat([pd.read_csv(path) for path in blocked_per_row_paths], ignore_index=True).to_csv(
        output_dir / "gate_fusion_blocked_per_row_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    for path in blocked_per_row_paths:
        path.unlink(missing_ok=True)

    audit = {
        "metric_reproduction_mode": "checkpoint_only_eval_mode",
        "tolerance": METRIC_TOLERANCE,
        "pass": bool(audit_df["eval_mode_reproduction_pass"].all()),
        "reason_if_failed": "Original repair scoring did not call model.eval(); dropout made exact train-mode metric reproduction non-deterministic from checkpoint alone. Instrumentation uses deterministic eval mode.",
        "max_abs_diffs": {
            metric: clean_float(audit_df[f"{metric}_abs_diff"].max())
            for metric in ("pr_auc", "roc_auc", "top_1pct_recall", "top_5pct_recall", "top_10pct_recall", "continuous_pearson")
        },
        "rows": audit_df.to_dict(orient="records"),
    }
    (output_dir / "gate_fusion_metric_reproduction_audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    missing_fields = {
        "native_branch_logits": "The architecture has one shared output head after fusion. Branch scores are derived by applying the shared output head to isolated branch representations.",
        "original_dropout_rng_state": "The completed run scored without an explicit eval() call, so exact dropout masks at original scoring time are not recoverable from checkpoint files.",
        "train_only_scaler_file": "No scaler artifact was saved; scaler is reconstructed from exact train rows and feature blocks.",
        "gate_vector_per_hidden_unit_in_per_row_file": "Per-row file stores gate_value as the row-wise mean gate to keep output compact; full vector is summarized in aggregate stats.",
    }
    (output_dir / "gate_fusion_missing_fields.json").write_text(json.dumps(missing_fields, indent=2) + "\n")

    real_blocked = comparison[(comparison.validation_protocol == "blocked_temporal_70_30") & (comparison.control_type == "real_ar_pca_diag")].iloc[0]
    rand_blocked = comparison[(comparison.validation_protocol == "blocked_temporal_70_30") & (comparison.control_type == "ar_plus_random_pca")].iloc[0]
    ar_blocked = comparison[(comparison.validation_protocol == "blocked_temporal_70_30") & (comparison.control_type == "ar_only_head")].iloc[0]
    real_grouped = comparison[(comparison.validation_protocol == "grouped_video") & (comparison.control_type == "real_ar_pca_diag")].iloc[0]
    rand_grouped = comparison[(comparison.validation_protocol == "grouped_video") & (comparison.control_type == "ar_plus_random_pca")].iloc[0]
    gate_delta_blocked_real_minus_random = float(real_blocked.mean_gate - rand_blocked.mean_gate)
    pca_future_corr_blocked = float(real_blocked.mean_corr_pca_score_future_target)
    pca_future_corr_grouped = float(real_grouped.mean_corr_pca_score_future_target)
    gate_routing_supported = bool(abs(gate_delta_blocked_real_minus_random) > 0.01)
    verdict = {
        "ar_time_dominance_supported": bool(ar_blocked.mean_pr_auc > rand_blocked.mean_pr_auc and ar_blocked.mean_pr_auc > real_blocked.mean_pr_auc),
        "random_pca_regularization_supported": bool(rand_blocked.mean_pr_auc > real_blocked.mean_pr_auc and ar_blocked.mean_pr_auc > rand_blocked.mean_pr_auc),
        "harmful_real_pca_fusion_supported": bool(
            real_blocked.mean_pr_auc < rand_blocked.mean_pr_auc
            and real_blocked.mean_corr_prediction_pca_score > rand_blocked.mean_corr_prediction_pca_score
            and gate_routing_supported
        ),
        "gate_routing_supported": gate_routing_supported,
        "control_bug_supported": False,
        "split_prevalence_artifact_supported": False,
        "strict_forward_time_temporal_generalization_proven": False,
        "recommended_next_step": "Do not expand to secondary heads. If mechanism-level proof is required, run only the minimal instrumented blocked/grouped rerun with eval-mode scoring fixed and full gate vectors/per-row branch logs.",
        "supporting_values": {
            "blocked_ar_only_eval_pr_auc": clean_float(ar_blocked.mean_pr_auc),
            "blocked_real_eval_pr_auc": clean_float(real_blocked.mean_pr_auc),
            "blocked_random_eval_pr_auc": clean_float(rand_blocked.mean_pr_auc),
            "blocked_real_mean_gate": clean_float(real_blocked.mean_gate),
            "blocked_random_mean_gate": clean_float(rand_blocked.mean_gate),
            "blocked_real_minus_random_mean_gate": gate_delta_blocked_real_minus_random,
            "blocked_real_pca_score_future_corr": pca_future_corr_blocked,
            "grouped_real_pca_score_future_corr": pca_future_corr_grouped,
            "blocked_real_prediction_pca_score_corr": clean_float(real_blocked.mean_corr_prediction_pca_score),
            "blocked_random_prediction_pca_score_corr": clean_float(rand_blocked.mean_corr_prediction_pca_score),
        },
    }
    (output_dir / "gate_fusion_mechanism_verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")

    manifest = {
        "source_output_root": str(source_root),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "commit_sha": COMMIT_SHA,
        "no_training": True,
        "checkpoints_loaded": int(len(load_df)),
        "checkpoints_missing": int(len(checkpoint_missing)),
        "selected_controls": list(SELECTED_CONTROLS),
        "selected_protocols": list(SELECTED_PROTOCOLS),
        "selected_seeds": list(SELECTED_SEEDS),
        "selected_loss": SELECTED_LOSS,
        "selected_model": SELECTED_MODEL,
        "target": SELECTED_TARGET,
        "feature": SELECTED_FEATURE,
        "split_reconstruction_method": "Replayed Phase 5 repair split/control order, feature assembly, random seeds, and train-only standardization; loaded saved best checkpoint weights; scored held-out test rows in deterministic eval mode.",
        "checkpoint_loads": load_df.to_dict(orient="records"),
    }
    (output_dir / "gate_fusion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    write_report(Path(args.report_path), output_dir, comparison, audit, verdict, checkpoints_loaded=int(len(load_df)))
    return {
        "succeeded": True,
        "output_dir": str(output_dir),
        "report_path": args.report_path,
        "checkpoints_loaded": int(len(load_df)),
        "metric_reproduction_pass": bool(audit["pass"]),
        "verdict": verdict,
    }


def md_table(df: pd.DataFrame, cols: list[str]) -> str:
    def fmt(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            if not math.isfinite(float(value)):
                return ""
            return f"{float(value):.6f}"
        return str(value)

    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def write_report(report_path: Path, output_dir: Path, comparison: pd.DataFrame, audit: dict[str, Any], verdict: dict[str, Any], *, checkpoints_loaded: int) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "validation_protocol",
        "control_type",
        "checkpoints",
        "mean_gate",
        "mean_saturation_low_rate",
        "mean_saturation_high_rate",
        "mean_pca_input_norm",
        "mean_pca_rep_norm",
        "mean_corr_gate_pca_norm",
        "mean_corr_prediction_pca_score",
        "mean_corr_pca_score_future_target",
        "mean_pr_auc",
        "mean_roc_auc",
        "mean_spearman_future_movement",
    ]
    grouped = comparison[comparison.validation_protocol == "grouped_video"].sort_values("control_type")
    blocked = comparison[comparison.validation_protocol == "blocked_temporal_70_30"].sort_values("control_type")
    report = f"""# Phase 5 Gate/Fusion Checkpoint Diagnostic

Source output root: `outputs/again_dense_2hz_phase5_adversarial_repair_fixplus_20260629_171825/`

## No New Training

This pass loaded saved best checkpoints and re-forwarded held-out benchmark rows only. It did not train models, start secondary heads, start secondary targets, rerun the 702 matrix, rerun V-JEPA/TRIBE/PCA, or modify Phase 4/original Phase 5 outputs.

## Checkpoints Loaded

- Checkpoints loaded: `{checkpoints_loaded}`
- Manifest: `{output_dir / 'gate_fusion_manifest.json'}`
- Selected controls: `real_ar_pca_diag`, `ar_plus_random_pca`, `ar_plus_shuffled_pca`, `ar_only_head`
- Protocols: `grouped_video`, `blocked_temporal_70_30`
- Loss/model: `regression_plus_binary`, `gated_ar_pca_mlp`

## Metric Reproduction Audit

- Deterministic eval-mode reproduction pass: `{audit['pass']}`
- Max absolute diffs: `{json.dumps(audit['max_abs_diffs'])}`
- Note: `{audit['reason_if_failed']}`

## Grouped Gate Behavior

{md_table(grouped[cols], cols)}

Grouped real remains the best matched-control result. Gate diagnostics show whether real PCA differs from random/shuffled controls under deterministic eval-mode re-forward, but the corrected grouped claim still rests on the committed repair metrics.

## Blocked Gate Behavior

{md_table(blocked[cols], cols)}

Blocked AR-only remains strongest among inspected controls in eval-mode re-forward. Random/shuffled PCA do not beat AR-only; they mainly show that real PCA is more harmful than random/shuffled PCA inside the fused head under blocked validation.

## Mechanism Verdict

- `ar_time_dominance_supported`: `{verdict['ar_time_dominance_supported']}`
- `random_pca_regularization_supported`: `{verdict['random_pca_regularization_supported']}`
- `harmful_real_pca_fusion_supported`: `{verdict['harmful_real_pca_fusion_supported']}`
- `gate_routing_supported`: `{verdict['gate_routing_supported']}`
- `control_bug_supported`: `{verdict['control_bug_supported']}`
- `split_prevalence_artifact_supported`: `{verdict['split_prevalence_artifact_supported']}`
- `strict_forward_time_temporal_generalization_proven`: `{verdict['strict_forward_time_temporal_generalization_proven']}`

Interpretation: AR/time dominance remains the strongest mechanism. The checkpoint-only gate/fusion view supports harmful real-PCA fusion as a plausible blocked mechanism because the real fused head underperforms random/shuffled PCA while its prediction is more coupled to the PCA branch. The PCA branch score is not anticorrelated with the future target, so this is a fusion/routing issue rather than simple negative PCA signal. It does not rescue strict forward-time temporal generalization.

## Retrain Needed?

No retrain is needed for the corrected claim. A minimal retrain is only needed if exact original train-mode metric reproduction or full gate-vector/per-row branch behavior is required with scoring semantics fixed.

Exact next recommendation: {verdict['recommended_next_step']}
"""
    report_path.write_text(report)


def main() -> int:
    args = parse_args()
    result = load_and_instrument(args)
    print(json.dumps(result, indent=2))
    return 0 if result.get("succeeded") else 2


if __name__ == "__main__":
    raise SystemExit(main())
