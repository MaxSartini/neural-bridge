#!/usr/bin/env python3
"""Execute the explicitly authorized 96-row zero-label deployment Stage A screen."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.scripts import again_dense_2hz_phase4_pca_bridge as phase4  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase5_frozen_ar_residual as frozen  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase5_redesigned_target_blocked as redesigned  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase5_temporal_residual_blocked as temporal  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase5_temporal_residual_grouped_compat as grouped  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase6_fixed_blend_fresh5 as fixed  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase7_continuous_checkpoint_ensemble_diagnostic as phase7  # noqa: E402
from backend.scripts import run_again_dense_2hz_zero_label_deployment_stage0 as stage0  # noqa: E402
from backend.scripts import again_zero_label_deployment_stage_a as stage_a  # noqa: E402


DEFAULT_STAGE0_ROOT = REPO_ROOT / "evidence/zero_label_video_only_deployment_stage0_20260714"
DEFAULT_EXTERNAL_PHASE4_ROOT = Path(
    "/Volumes/onn. Drive/Neural Bridge/outputs/again_dense_2hz_phase4_pca_bridge_20260625_full"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/Volumes/onn. Drive/Neural Bridge/outputs/again_dense_2hz_zero_label_deployment_stage_a_20260714"
)
AR_COLUMNS = tuple(grouped.AR_FEATURE_COLUMNS)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage0-root", default=str(DEFAULT_STAGE0_ROOT))
    parser.add_argument("--dense-root", default=str(stage0.DEFAULT_DENSE_ROOT))
    parser.add_argument("--external-phase4-root", default=str(DEFAULT_EXTERNAL_PHASE4_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--batch-size", type=int, default=stage_a.BATCH_SIZE)
    parser.add_argument("--pca-batch-size", type=int, default=384)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fold", type=int, choices=stage_a.OUTER_FOLDS, default=None)
    return parser.parse_args()


def load_stage0(stage0_root: Path) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    result = json.loads((stage0_root / "stage0_result.json").read_text(encoding="utf-8"))
    splits = json.loads((stage0_root / "split_manifest.json").read_text(encoding="utf-8"))
    target = json.loads((stage0_root / "target_identity_manifest.json").read_text(encoding="utf-8"))
    if not result.get("stage0_pass") or result.get("failed_contracts"):
        raise RuntimeError("Canonical Stage 0 evidence does not pass")
    if target.get("target_identity_digest") != "446906dff30be33f204de0f973207975":
        raise RuntimeError("Canonical Stage 0 target identity changed")
    if splits.get("development_digest") != "cf65a766cd827e6201544dd753049cb4":
        raise RuntimeError("Canonical Stage 0 development split changed")
    if splits.get("locked_digest") != "ded8bc2bf079fef91ae5c253b9a9ac2e":
        raise RuntimeError("Canonical Stage 0 locked split changed")
    return result, splits, target


def implementation_preflight(
    *,
    stage0_root: Path,
    dense_root: Path,
    external_phase4_root: Path,
    output_root: Path,
) -> Mapping[str, Any]:
    stage0_result, splits, target = load_stage0(stage0_root)
    freeze = stage_a.implementation_freeze_manifest()
    manifest = {
        **freeze,
        "stage0_result_sha256": stage_a.file_sha256(stage0_root / "stage0_result.json"),
        "split_manifest_sha256": stage_a.file_sha256(stage0_root / "split_manifest.json"),
        "target_manifest_sha256": stage_a.file_sha256(stage0_root / "target_identity_manifest.json"),
        "target_identity_digest": target["target_identity_digest"],
        "development_split_digest": splits["development_digest"],
        "locked_split_digest": splits["locked_digest"],
        "dense_root": str(dense_root),
        "external_phase4_root": str(external_phase4_root),
        "output_root": str(output_root),
        "stage0_pass": stage0_result["stage0_pass"],
        "authorized": True,
        "stage_b_authorized": False,
    }
    code_paths = {
        "utility": REPO_ROOT / "backend/scripts/again_zero_label_deployment_stage_a.py",
        "runner": REPO_ROOT
        / "backend/scripts/run_again_dense_2hz_zero_label_deployment_stage_a.py",
        "contracts": REPO_ROOT / "tests/test_again_zero_label_deployment_stage_a.py",
    }
    manifest["implementation_code_sha256"] = {
        name: stage_a.file_sha256(path) for name, path in code_paths.items()
    }
    manifest["signed_implementation_digest"] = stage_a.canonical_digest(manifest)
    return manifest


def _teacher_feature_raw(
    *,
    df: pd.DataFrame,
    indices: np.ndarray,
    pca_scores: np.ndarray,
    diagnostics_all: np.ndarray,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    videos = df.loc[indices, "video_id"].astype(str).to_numpy()
    sequence, audit = temporal.causal_sequence_features(pca_scores, indices, videos)
    raw = np.concatenate(
        [sequence, diagnostics_all[indices].astype(np.float32)], axis=1
    ).astype(np.float32)
    return raw, audit


def _make_block(
    *,
    df: pd.DataFrame,
    values: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fold: int,
    target_name: str,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ar_all = grouped.feature_matrix(df, AR_COLUMNS)
    ar_train_raw = ar_all[train_idx].astype(np.float32)
    ar_test_raw = ar_all[test_idx].astype(np.float32)
    ar_mean = np.nanmean(ar_train_raw, axis=0).astype(np.float32)
    ar_std = np.nanstd(ar_train_raw, axis=0).astype(np.float32)
    ar_std[~np.isfinite(ar_std) | (ar_std < 1e-6)] = 1.0
    ar_train = ((np.nan_to_num(ar_train_raw) - ar_mean) / ar_std).astype(np.float32)
    ar_test = ((np.nan_to_num(ar_test_raw) - ar_mean) / ar_std).astype(np.float32)
    train_cont = values[train_idx].astype(np.float32)
    test_cont = values[test_idx].astype(np.float32)
    threshold = float(np.quantile(train_cont[np.isfinite(train_cont)], stage_a.EVENT_QUANTILE))
    train_y = (train_cont >= threshold).astype(np.int64)
    test_y = (test_cont >= threshold).astype(np.int64)
    inner_train, inner_val, inner_audit = frozen.inner_split(df, train_idx, train_y)
    block = redesigned.RunBlock(
        target_name=target_name,
        target_type="continuous",
        protocol="zero_label_grouped_video",
        fold=int(fold),
        split=None,
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
        ar_train_x=ar_train,
        ar_test_x=ar_test,
        ar_block_dims={"ar": int(ar_train.shape[1])},
    )
    return block, ar_mean, ar_std, ar_train_raw, ar_test_raw


def _teacher_pack(
    *,
    df: pd.DataFrame,
    block: Any,
    pca_view: stage_a.PcaView,
    diagnostics_all: np.ndarray,
) -> tuple[Any, np.ndarray, np.ndarray]:
    train_raw, train_audit = _teacher_feature_raw(
        df=df,
        indices=block.train_idx,
        pca_scores=pca_view.scores_for(block.train_idx),
        diagnostics_all=diagnostics_all,
    )
    test_raw, test_audit = _teacher_feature_raw(
        df=df,
        indices=block.test_idx,
        pca_scores=pca_view.scores_for(block.test_idx),
        diagnostics_all=diagnostics_all,
    )
    train_x, test_x, mean, std = stage_a.standardize_train_only(train_raw, test_raw)
    pack = temporal.FeaturePack(
        train_x=train_x,
        test_x=test_x,
        dims={
            "sequence": stage_a.WINDOW_ROWS * stage_a.PCA_WIDTH,
            "sequence_window": stage_a.WINDOW_ROWS,
            "sequence_channels": stage_a.PCA_WIDTH,
            "diagnostics": stage_a.DIAGNOSTIC_WIDTH,
        },
        manifest=[
            {
                "block": "canonical_phase7_teacher_pca_sequence_plus_diagnostics",
                "pca_component_sha256": stage_a.file_sha256(pca_view.component_path),
            }
        ],
        context_audit={
            "temporal_context_causal_only": True,
            "same_video_history_masking": True,
            "uses_centered_or_future_windows": False,
            "train": train_audit,
            "test": test_audit,
        },
    )
    return pack, mean, std


@dataclass
class TeacherMember:
    seed: int
    root: Path
    block: Any
    pack: Any
    ar_mean: np.ndarray
    ar_std: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    residual_audit: Mapping[str, Any]
    ar: Mapping[str, Any]


def _train_teacher_member(
    *,
    df: pd.DataFrame,
    values: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    pca_view: stage_a.PcaView,
    diagnostics_all: np.ndarray,
    seed: int,
    root: Path,
    fold: int,
    batch_size: int,
) -> tuple[TeacherMember, np.ndarray]:
    block, ar_mean, ar_std, _, _ = _make_block(
        df=df,
        values=values,
        train_idx=train_idx,
        test_idx=test_idx,
        fold=fold,
        target_name=f"{stage0.TARGET_NAME}__teacher_{root.name}",
    )
    pack, feature_mean, feature_std = _teacher_pack(
        df=df, block=block, pca_view=pca_view, diagnostics_all=diagnostics_all
    )
    phase7.train_continuous_ar_inner_only(
        block=block,
        seed=seed,
        output_root=root,
        batch_size=batch_size,
        max_epochs=80,
        patience=12,
    )
    ar = phase7.load_continuous_ar(root, block, seed, batch_size)
    _, curves, audit = temporal.train_temporal_residual(
        architecture=phase7.ARCHITECTURE,
        control="real_residual",
        pack=pack,
        block=block,
        ar=ar,
        seed=seed,
        output_root=root / "residual",
        batch_size=batch_size,
        max_epochs=int(phase7.PARAMS["max_epochs"]),
        patience=int(phase7.PARAMS["patience"]),
        hyperparameters=phase7.PARAMS,
    )
    restored = fixed.restored_scores(
        audit=dict(audit),
        params=phase7.PARAMS,
        pack=pack,
        block=block,
        ar=dict(ar),
        batch_size=batch_size,
    )
    write_json(
        root / "teacher_member_manifest.json",
        {
            "seed": seed,
            "train_idx_digest": stage_a.array_digest(train_idx),
            "test_idx_digest": stage_a.array_digest(test_idx),
            "pca_component_sha256": stage_a.file_sha256(pca_view.component_path),
            "ar_checkpoint": str(root / "ar_baseline_checkpoints" / f"seed{seed}__best.npz"),
            "ar_checkpoint_sha256": stage_a.file_sha256(
                root / "ar_baseline_checkpoints" / f"seed{seed}__best.npz"
            ),
            "residual_audit": audit,
            "teacher_score_digest": stage_a.array_digest(restored["test_reg"].astype(np.float32)),
            "curves": curves,
            "outer_test_video_entered_fit": False,
        },
    )
    member = TeacherMember(
        seed=seed,
        root=root,
        block=block,
        pack=pack,
        ar_mean=ar_mean,
        ar_std=ar_std,
        feature_mean=feature_mean,
        feature_std=feature_std,
        residual_audit=dict(audit),
        ar=dict(ar),
    )
    return member, restored["test_reg"].astype(np.float32)


def _load_ar_model(member: TeacherMember) -> Any:
    checkpoint = member.root / "ar_baseline_checkpoints" / f"seed{member.seed}__best.npz"
    config = phase7.confirm.ar_config(
        member.seed, max_epochs=80, patience=12, batch_size=stage_a.BATCH_SIZE
    )
    model = phase7.base.make_model(
        config, member.block.ar_train_x.shape[1], member.block.ar_block_dims
    )
    _ = model(phase7.base.mx.array(member.block.ar_train_x[:2], dtype=phase7.base.mx.float32))
    model.load_weights(str(checkpoint))
    if hasattr(model, "eval"):
        model.eval()
    return model


def _score_teacher_member(
    *,
    member: TeacherMember,
    test_feature_raw: np.ndarray,
    test_ar_raw: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    feature_x = ((np.nan_to_num(test_feature_raw) - member.feature_mean) / member.feature_std).astype(
        np.float32
    )
    ar_x = ((np.nan_to_num(test_ar_raw) - member.ar_mean) / member.ar_std).astype(np.float32)
    ar_model = _load_ar_model(member)
    ar_score, ar_reg = frozen.score_existing_model(ar_model, ar_x, batch_size)
    audit = dict(member.residual_audit)
    if audit.get("residual_suppressed"):
        return ar_reg.astype(np.float32)
    residual_model = temporal.TemporalResidualHead(
        feature_x.shape[1],
        phase7.ARCHITECTURE,
        hidden=int(phase7.PARAMS["hidden"]),
        sequence_window=stage_a.WINDOW_ROWS,
        sequence_channels=stage_a.PCA_WIDTH,
        alpha_initial_logit=float(phase7.PARAMS["alpha_initial_logit"]),
        alpha_cap=float(phase7.PARAMS["alpha_cap"]),
        gate_bias=float(phase7.PARAMS["gate_bias"]),
    )
    _ = residual_model(
        phase7.base.mx.array(feature_x[:2], dtype=phase7.base.mx.float32),
        phase7.base.mx.array(ar_score[:2], dtype=phase7.base.mx.float32),
        phase7.base.mx.array(ar_reg[:2], dtype=phase7.base.mx.float32),
    )
    residual_model.load_weights(str(audit["checkpoint_path"]))
    if hasattr(residual_model, "eval"):
        residual_model.eval()
    _, prediction, _ = temporal.forward_residual(
        residual_model,
        feature_x,
        ar_score,
        ar_reg,
        batch_size=batch_size,
        target_type="continuous",
    )
    return prediction.astype(np.float32)


def _save_prediction(
    *,
    path: Path,
    df: pd.DataFrame,
    indices: np.ndarray,
    prediction: np.ndarray,
    fold: int,
    lane: str,
    seed: int,
    target_identity_digest: str,
    metadata: Mapping[str, Any],
) -> str:
    if len(indices) != len(prediction) or not np.isfinite(prediction).all():
        raise RuntimeError(f"Prediction coverage failed for fold{fold}/{lane}/seed{seed}")
    table = pd.DataFrame(
        {
            "split_id": f"stage_a_fold{fold}",
            "video_id": df.loc[indices, "video_id"].astype(str).to_numpy(),
            "row_index": indices.astype(np.int64),
            "row_id": stage_a.row_ids(df)[indices],
            "target_identity_digest": target_identity_digest,
            "prediction": prediction.astype(np.float32),
            "fold": int(fold),
            "lane": lane,
            "seed": int(seed),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    checksum = stage_a.file_sha256(path)
    write_json(
        path.with_suffix(".manifest.json"),
        {
            "prediction_path": str(path),
            "prediction_sha256": checksum,
            "prediction_rows": len(table),
            "prediction_digest": stage_a.array_digest(prediction.astype(np.float32)),
            "labels_loaded_by_predictor": False,
            "prediction_sealed_before_label_join": True,
            **dict(metadata),
        },
    )
    return checksum


def _load_prediction(path: Path) -> np.ndarray:
    table = pd.read_parquet(path)
    return table["prediction"].to_numpy(dtype=np.float32)


def _score_lane(
    *,
    df: pd.DataFrame,
    values: np.ndarray,
    hard_mask: np.ndarray,
    compatibility_mask: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    prediction: np.ndarray,
) -> Mapping[str, Any]:
    train_hard = hard_mask[train_idx]
    test_hard = hard_mask[test_idx]
    full = stage_a.score_prediction(
        train_values=values[train_idx][train_hard],
        test_values=values[test_idx][test_hard],
        prediction=prediction[test_hard],
        time_seconds=df.loc[test_idx, "time_seconds"].to_numpy(dtype=np.float32)[test_hard],
    )
    train_compat = compatibility_mask[train_idx]
    test_compat = compatibility_mask[test_idx]
    compat = stage_a.score_prediction(
        train_values=values[train_idx][train_hard],
        test_values=values[test_idx][test_compat],
        prediction=prediction[test_compat],
        time_seconds=df.loc[test_idx, "time_seconds"].to_numpy(dtype=np.float32)[test_compat],
    )
    return {**full, **{f"compat_{key}": value for key, value in compat.items()}}


def run_fold(
    *,
    fold: int,
    df: pd.DataFrame,
    values: np.ndarray,
    valid: np.ndarray,
    split_manifest: Mapping[str, Any],
    target_identity_digest: str,
    accessor: phase4.CorticalVariantAccessor,
    diagnostics_all: np.ndarray,
    root: Path,
    batch_size: int,
    pca_batch_size: int,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    train_all, test_all, outer_record = stage_a.split_rows(df, split_manifest, fold)
    inference_df = df[["video_id", "time_seconds"]].copy()
    if set(inference_df.columns) != {"video_id", "time_seconds"}:
        raise RuntimeError("Held-out inference metadata firewall contains unexpected columns")
    hard_mask = stage_a.valid_target_mask(df, values, valid)
    compat_mask = stage_a.teacher_compatibility_mask(df, hard_mask)
    fold_root = root / f"fold{fold}"
    pca_root = fold_root / "pca"
    outer_pca = stage_a.fit_or_load_pca_view(
        accessor=accessor,
        train_idx=train_all,
        test_idx=test_all,
        output_root=pca_root,
        name=f"stage_a_fold{fold}__outer__{stage_a.PCA_FAMILY}",
        seed=20260714 + fold,
        batch_size=pca_batch_size,
    )
    train_features = stage_a.build_video_features(
        inference_df,
        train_all,
        outer_pca.train_scores,
        diagnostics_all[train_all].astype(np.float32),
    )
    test_features = stage_a.build_video_features(
        inference_df,
        test_all,
        outer_pca.test_scores,
        diagnostics_all[test_all].astype(np.float32),
    )

    # Cross-fitted teacher soft targets are generated only inside the outer training pool.
    soft_target = np.full(len(train_all), np.nan, dtype=np.float32)
    train_position = np.full(len(df), -1, dtype=np.int64)
    train_position[train_all] = np.arange(len(train_all), dtype=np.int64)
    teacher_ownership: list[Mapping[str, Any]] = []
    for nested_index, (nested_train_all, nested_test_all, nested_record) in enumerate(
        stage_a.three_way_teacher_rows(df, outer_record), 1
    ):
        nested_pca = stage_a.fit_or_load_pca_view(
            accessor=accessor,
            train_idx=nested_train_all,
            test_idx=nested_test_all,
            output_root=pca_root,
            name=f"stage_a_fold{fold}__teacher{nested_index}__{stage_a.PCA_FAMILY}",
            seed=20260730 + fold * 10 + nested_index,
            batch_size=pca_batch_size,
        )
        nested_train = nested_train_all[compat_mask[nested_train_all]]
        nested_test = nested_test_all[compat_mask[nested_test_all]]
        member_scores: list[np.ndarray] = []
        for seed in stage_a.SEEDS:
            _, score = _train_teacher_member(
                df=df,
                values=values,
                train_idx=nested_train,
                test_idx=nested_test,
                pca_view=nested_pca,
                diagnostics_all=diagnostics_all,
                seed=seed,
                root=fold_root / "teachers" / f"nested{nested_index}" / f"seed{seed}",
                fold=fold,
                batch_size=batch_size,
            )
            member_scores.append(score)
        ensemble = stage_a.ensemble_predictions(member_scores)
        positions = train_position[nested_test]
        if np.any(positions < 0) or np.isfinite(soft_target[positions]).any():
            raise RuntimeError("Cross-fitted teacher ownership overlap or escape")
        soft_target[positions] = ensemble
        teacher_ownership.append(
            {
                "nested_fold": nested_index,
                "train_video_digest": nested_record["train_digest"],
                "test_video_digest": nested_record["test_digest"],
                "test_row_digest": stage_a.array_digest(nested_test),
                "teacher_score_digest": stage_a.array_digest(ensemble),
                "outer_test_video_entered_fit": False,
            }
        )
    soft_mask = np.isfinite(soft_target) & compat_mask[train_all]
    if not np.array_equal(np.flatnonzero(soft_mask), np.flatnonzero(compat_mask[train_all])):
        raise RuntimeError("Cross-fitted teacher did not cover every compatible outer-training row")
    soft_mean = float(np.mean(soft_target[soft_mask]))
    soft_std = float(np.std(soft_target[soft_mask]))
    if not math.isfinite(soft_std) or soft_std < 1e-6:
        raise RuntimeError("Cross-fitted teacher soft targets are degenerate")
    soft_standardized = (soft_target - soft_mean) / soft_std
    np.savez_compressed(
        fold_root / "teacher_soft_targets.npz",
        train_idx=train_all,
        soft_target=soft_target,
        soft_standardized=soft_standardized,
        soft_mask=soft_mask,
    )

    # Train the outer teacher on training rows only. A training-only dummy test
    # avoids opening held-out AR inputs/labels before all candidate predictions seal.
    outer_train_compat = train_all[compat_mask[train_all]]
    dummy_test = outer_train_compat[: max(128, min(2048, len(outer_train_compat)))]
    outer_members: dict[int, TeacherMember] = {}
    for seed in stage_a.SEEDS:
        member, _ = _train_teacher_member(
            df=df,
            values=values,
            train_idx=outer_train_compat,
            test_idx=dummy_test,
            pca_view=outer_pca,
            diagnostics_all=diagnostics_all,
            seed=seed,
            root=fold_root / "teachers" / "outer" / f"seed{seed}",
            fold=fold,
            batch_size=batch_size,
        )
        outer_members[seed] = member

    prediction_root = fold_root / "predictions"
    predictions: dict[str, dict[int, np.ndarray]] = {lane: {} for lane in stage_a.LANES}
    audits: list[Mapping[str, Any]] = []
    hard_train_mask = hard_mask[train_all]
    current_arousal = df["arousal"].to_numpy(dtype=np.float32)
    current_loss_mask = df.loc[train_all, "label_available"].to_numpy(dtype=bool) & np.isfinite(
        current_arousal[train_all]
    )
    shuffled_train_x, shuffled_train_map = stage_a.reassign_video_sequences(
        train_features.x_temporal,
        train_features.video_id,
        f"stage_a_fold{fold}|sequence_shuffle|train",
    )
    shuffled_test_x, shuffled_test_map = stage_a.reassign_video_sequences(
        test_features.x_temporal,
        test_features.video_id,
        f"stage_a_fold{fold}|sequence_shuffle|test",
    )
    permuted_soft, label_map = stage_a.permute_video_targets(
        soft_standardized,
        train_features.video_id,
        f"stage_a_fold{fold}|label_permutation",
    )
    no_video_train = train_features.x_temporal.copy()
    no_video_test = test_features.x_temporal.copy()
    no_video_train[:, : stage_a.WINDOW_ROWS * stage_a.PCA_WIDTH + stage_a.DIAGNOSTIC_WIDTH] = 0.0
    no_video_test[:, : stage_a.WINDOW_ROWS * stage_a.PCA_WIDTH + stage_a.DIAGNOSTIC_WIDTH] = 0.0

    for seed in stage_a.SEEDS:
        scalar_specs = {
            "video_distilled_temporal": (
                train_features.x_temporal,
                test_features.x_temporal,
                soft_standardized,
                soft_mask,
                True,
                True,
            ),
            "video_supervised_temporal": (
                train_features.x_temporal,
                test_features.x_temporal,
                values[train_all],
                hard_train_mask,
                True,
                True,
            ),
            "video_supervised_current_row": (
                train_features.x_current,
                test_features.x_current,
                values[train_all],
                hard_train_mask,
                False,
                True,
            ),
            "sequence_shuffled_video": (
                shuffled_train_x,
                shuffled_test_x,
                soft_standardized,
                soft_mask,
                True,
                True,
            ),
            "video_label_permutation": (
                train_features.x_temporal,
                test_features.x_temporal,
                permuted_soft,
                np.isfinite(permuted_soft),
                True,
                True,
            ),
        }
        for lane, (train_x, test_x, target, loss_mask, temporal_context, weighted) in scalar_specs.items():
            result = stage_a.train_scalar_model(
                train_x=train_x,
                test_x=test_x,
                train_target=target,
                train_loss_mask=loss_mask,
                train_video_id=train_features.video_id,
                temporal_context=temporal_context,
                seed=seed,
                checkpoint_path=fold_root / "models" / lane / f"seed{seed}.npz",
                namespace=f"stage_a_fold{fold}|{lane}|seed{seed}",
                weighted_huber=weighted,
                batch_size=batch_size,
            )
            predictions[lane][seed] = result.test_prediction
            _save_prediction(
                path=prediction_root / lane / f"seed{seed}.parquet",
                df=inference_df,
                indices=test_all,
                prediction=result.test_prediction,
                fold=fold,
                lane=lane,
                seed=seed,
                target_identity_digest=target_identity_digest,
                metadata={
                    "checkpoint_sha256": result.checkpoint_sha256,
                    "best_epoch": result.best_epoch,
                    "best_validation_loss": result.best_validation_loss,
                    "hard_outcome_used_in_loss": lane.startswith("video_supervised"),
                    "teacher_soft_target_used_in_loss": lane in {
                        "video_distilled_temporal",
                        "sequence_shuffled_video",
                        "video_label_permutation",
                    },
                },
            )
            audits.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "lane": lane,
                    "checkpoint_sha256": result.checkpoint_sha256,
                    "all_finite": bool(np.isfinite(result.test_prediction).all()),
                    "labels_loaded_by_predictor": False,
                }
            )

        # H2 and its no-video anchor predict current response from video only,
        # then feed only self-predicted response state through the frozen teacher.
        h2_predictions: dict[str, tuple[np.ndarray, Mapping[str, Any]]] = {}
        for lane, state_train_x, state_test_x in (
            ("video_closed_loop_rollout", train_features.x_temporal, test_features.x_temporal),
            ("no_video_closed_loop_persistence", no_video_train, no_video_test),
        ):
            state = stage_a.train_scalar_model(
                train_x=state_train_x,
                test_x=state_test_x,
                train_target=current_arousal[train_all],
                train_loss_mask=current_loss_mask,
                train_video_id=train_features.video_id,
                temporal_context=True,
                seed=seed,
                checkpoint_path=fold_root / "models" / lane / f"state_seed{seed}.npz",
                namespace=f"stage_a_fold{fold}|{lane}|state|seed{seed}",
                weighted_huber=False,
                batch_size=batch_size,
            )
            train_median = float(np.median(current_arousal[train_all][current_loss_mask]))
            test_ar_raw, rollout_audit = stage_a.own_prediction_ar_features(
                state.test_prediction, test_features.video_id, train_median
            )
            teacher_feature_test_raw, _ = _teacher_feature_raw(
                df=inference_df,
                indices=test_all,
                pca_scores=outer_pca.test_scores,
                diagnostics_all=diagnostics_all,
            )
            prediction = _score_teacher_member(
                member=outer_members[seed],
                test_feature_raw=teacher_feature_test_raw,
                test_ar_raw=test_ar_raw,
                batch_size=batch_size,
            )
            predictions[lane][seed] = prediction
            h2_predictions[lane] = (prediction, rollout_audit)
            _save_prediction(
                path=prediction_root / lane / f"seed{seed}.parquet",
                df=inference_df,
                indices=test_all,
                prediction=prediction,
                fold=fold,
                lane=lane,
                seed=seed,
                target_identity_digest=target_identity_digest,
                metadata={
                    "state_checkpoint_sha256": state.checkpoint_sha256,
                    "hard_outcome_used_in_state_loss": False,
                    "current_arousal_used_in_training_loss_only": True,
                    **dict(rollout_audit),
                },
            )
            audits.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "lane": lane,
                    **dict(rollout_audit),
                    "labels_loaded_by_predictor": False,
                }
            )

    # All zero-label candidate/control predictions are sealed before the
    # privileged teacher ceiling opens observed held-out AR context.
    zero_label_prediction_paths = sorted(
        path
        for lane in stage_a.ZERO_LABEL_LANES
        for path in (prediction_root / lane).glob("seed*.parquet")
    )
    if len(zero_label_prediction_paths) != len(stage_a.ZERO_LABEL_LANES) * len(stage_a.SEEDS):
        raise RuntimeError("Candidate/control prediction seal is incomplete")
    seal = {
        str(path): stage_a.file_sha256(path) for path in zero_label_prediction_paths
    }
    write_json(
        fold_root / "candidate_prediction_seal.json",
        {
            "fold": fold,
            "prediction_files": seal,
            "sealed_before_teacher_ceiling_and_label_join": True,
        },
    )

    # Privileged teacher ceiling: observed outer-test AR is opened only now.
    observed_ar_all = grouped.feature_matrix(df, AR_COLUMNS)
    teacher_feature_test_raw, _ = _teacher_feature_raw(
        df=inference_df,
        indices=test_all,
        pca_scores=outer_pca.test_scores,
        diagnostics_all=diagnostics_all,
    )
    for seed in stage_a.SEEDS:
        prediction = _score_teacher_member(
            member=outer_members[seed],
            test_feature_raw=teacher_feature_test_raw,
            test_ar_raw=observed_ar_all[test_all].astype(np.float32),
            batch_size=batch_size,
        )
        predictions["phase7_ar_assisted_teacher_ceiling"][seed] = prediction
        _save_prediction(
            path=prediction_root / "phase7_ar_assisted_teacher_ceiling" / f"seed{seed}.parquet",
            df=inference_df,
            indices=test_all,
            prediction=prediction,
            fold=fold,
            lane="phase7_ar_assisted_teacher_ceiling",
            seed=seed,
            target_identity_digest=target_identity_digest,
            metadata={
                "non_deployable_teacher_ceiling": True,
                "observed_arousal_context_used": True,
                "opened_after_zero_label_prediction_seal": True,
            },
        )

    rows: list[Mapping[str, Any]] = []
    for lane in stage_a.LANES:
        for seed in stage_a.SEEDS:
            prediction_path = prediction_root / lane / f"seed{seed}.parquet"
            prediction = _load_prediction(prediction_path)
            metrics = _score_lane(
                df=df,
                values=values,
                hard_mask=hard_mask,
                compatibility_mask=compat_mask,
                train_idx=train_all,
                test_idx=test_all,
                prediction=prediction,
            )
            rollout = next(
                (
                    audit
                    for audit in audits
                    if audit.get("lane") == lane and int(audit.get("seed", -1)) == seed
                ),
                {},
            )
            rows.append(
                {
                    "stage": "stage_a",
                    "split_digest": outer_record["test_digest"],
                    "fold": fold,
                    "lane": lane,
                    "row_type": "member",
                    "seed_or_group": str(seed),
                    "cold_start_policy": "row0_zero_history_no_label_burnin",
                    "prediction_sha256": stage_a.file_sha256(prediction_path),
                    "score_mask_digest": stage_a.array_digest(hard_mask[test_all].astype(np.uint8)),
                    "compatibility_mask_digest": stage_a.array_digest(
                        compat_mask[test_all].astype(np.uint8)
                    ),
                    "teacher_forcing_ratio": rollout.get("teacher_forcing_ratio", math.nan),
                    "cross_video_state_carry": rollout.get("cross_video_state_carry", math.nan),
                    "rollout_all_finite": rollout.get("all_finite", True),
                    **metrics,
                }
            )
        ensemble = stage_a.ensemble_predictions(
            [predictions[lane][seed] for seed in stage_a.SEEDS]
        )
        ensemble_path = prediction_root / lane / f"ensemble_{stage_a.GROUP_NAME}.parquet"
        _save_prediction(
            path=ensemble_path,
            df=inference_df,
            indices=test_all,
            prediction=ensemble,
            fold=fold,
            lane=lane,
            seed=0,
            target_identity_digest=target_identity_digest,
            metadata={"member_seeds": list(stage_a.SEEDS), "weights": [1 / 3, 1 / 3, 1 / 3]},
        )
        metrics = _score_lane(
            df=df,
            values=values,
            hard_mask=hard_mask,
            compatibility_mask=compat_mask,
            train_idx=train_all,
            test_idx=test_all,
            prediction=ensemble,
        )
        lane_audits = [audit for audit in audits if audit.get("lane") == lane]
        rows.append(
            {
                "stage": "stage_a",
                "split_digest": outer_record["test_digest"],
                "fold": fold,
                "lane": lane,
                "row_type": "ensemble",
                "seed_or_group": stage_a.GROUP_NAME,
                "cold_start_policy": "row0_zero_history_no_label_burnin",
                "prediction_sha256": stage_a.file_sha256(ensemble_path),
                "score_mask_digest": stage_a.array_digest(hard_mask[test_all].astype(np.uint8)),
                "compatibility_mask_digest": stage_a.array_digest(
                    compat_mask[test_all].astype(np.uint8)
                ),
                "teacher_forcing_ratio": (
                    0.0 if lane in {"video_closed_loop_rollout", "no_video_closed_loop_persistence"} else math.nan
                ),
                "cross_video_state_carry": (
                    0 if lane in {"video_closed_loop_rollout", "no_video_closed_loop_persistence"} else math.nan
                ),
                "rollout_all_finite": all(audit.get("all_finite", True) for audit in lane_audits),
                **metrics,
            }
        )
    write_json(
        fold_root / "teacher_crossfit_ownership.json",
        {
            "fold": fold,
            "records": teacher_ownership,
            "soft_target_coverage": int(soft_mask.sum()),
            "soft_target_digest": stage_a.array_digest(soft_target[soft_mask]),
            "outer_test_video_entered_teacher_fit": False,
        },
    )
    pd.DataFrame(rows).to_csv(fold_root / "metrics_rows.csv", index=False)
    write_json(fold_root / "audits.json", {"fold": fold, "audits": audits})
    return rows, audits


def report_text(result: Mapping[str, Any], root: Path) -> str:
    winner = result.get("locked_winner") or "none"
    return f"""# Zero-label video-only deployment bridge — Stage A

- Output: `{root}`
- Matrix: `{result['rows_actual']}/{result['rows_expected']}`
- Stage A continuation pass: `{result['stage_a_pass']}`
- Locked Stage A winner: `{winner}`
- Stage B authorized: `False`
- Failed gates: `{result['failed_gates']}`

This is a cold-start, zero-label inference development screen. It does not
promote a deployment claim and does not change the canonical AR-assisted Phase 7
result. Stage B remains locked pending a separate explicit authorization.
"""


def main() -> int:
    args = parse_args()
    stage0_root = Path(args.stage0_root)
    dense_root = Path(args.dense_root)
    external_phase4_root = Path(args.external_phase4_root)
    root = Path(args.output_root)
    preflight = implementation_preflight(
        stage0_root=stage0_root,
        dense_root=dense_root,
        external_phase4_root=external_phase4_root,
        output_root=root,
    )
    print(json.dumps(preflight, indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    device = stage_a.require_mlx_gpu()
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise FileExistsError(f"{root} is non-empty; pass --resume to continue")
    for subdir in ("manifests", "metrics", "diagnostics", "reports"):
        (root / subdir).mkdir(parents=True, exist_ok=True)
    write_json(root / "manifests" / "implementation_freeze.json", preflight)
    started = time.time()
    _, split_manifest, target_manifest = load_stage0(stage0_root)
    df = stage0.base.load_labels(dense_root)
    values, valid = stage0.redesigned.future_max_delta(df, 4, 10)
    if stage_a.array_digest(values.astype(np.float64)) != target_manifest["value_array_digest"]:
        raise RuntimeError("Stage A target values differ from the Stage 0 target lock")
    cortical = grouped.load_external_cortical_memmap(external_phase4_root, df)
    if cortical is None:
        raise FileNotFoundError("The external Phase 4 cortical memmap is required")
    accessor = phase4.CorticalVariantAccessor(cortical, df, base_family=stage_a.PCA_FAMILY)
    diagnostics_all = np.load(
        dense_root / "_derived/temporal_diagnostics_summary_features.npy", mmap_mode="r"
    )
    folds = (args.fold,) if args.fold is not None else stage_a.OUTER_FOLDS
    all_rows: list[Mapping[str, Any]] = []
    all_audits: list[Mapping[str, Any]] = []
    for fold in folds:
        fold_rows, fold_audits = run_fold(
            fold=int(fold),
            df=df,
            values=values,
            valid=valid,
            split_manifest=split_manifest,
            target_identity_digest=target_manifest["target_identity_digest"],
            accessor=accessor,
            diagnostics_all=diagnostics_all,
            root=root,
            batch_size=args.batch_size,
            pca_batch_size=args.pca_batch_size,
        )
        all_rows.extend(fold_rows)
        all_audits.extend(fold_audits)
        pd.DataFrame(all_rows).to_csv(root / "metrics" / "stage_a_rows.partial.csv", index=False)
        gc.collect()
    if args.fold is not None:
        print(json.dumps({"fold_completed": args.fold, "rows": len(all_rows), "root": str(root)}, indent=2))
        return 0
    frame = pd.DataFrame(all_rows)
    audit_pass = bool(
        len(all_audits) >= len(stage_a.OUTER_FOLDS) * len(stage_a.ZERO_LABEL_LANES) * len(stage_a.SEEDS)
        and all(audit.get("labels_loaded_by_predictor") is False for audit in all_audits)
        and all(audit.get("all_finite", True) for audit in all_audits)
    )
    result = dict(stage_a.compute_stage_a_verdict(frame, audit_pass))
    result.update(
        {
            "duration_seconds": time.time() - started,
            "accelerator_detail": device,
            "target_identity_digest": target_manifest["target_identity_digest"],
            "development_split_digest": split_manifest["development_digest"],
            "implementation_freeze_digest": preflight["implementation_freeze_digest"],
        }
    )
    frame.to_csv(root / "metrics" / "stage_a_rows.csv", index=False)
    write_json(root / "metrics" / "stage_a_result.json", result)
    write_json(root / "diagnostics" / "stage_a_audit.json", {"audit_pass": audit_pass, "audits": all_audits})
    report = report_text(result, root)
    report_path = root / "reports" / "again_dense_2hz_zero_label_deployment_stage_a_20260714.md"
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps({"run_completed": True, "output_root": str(root), **result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
