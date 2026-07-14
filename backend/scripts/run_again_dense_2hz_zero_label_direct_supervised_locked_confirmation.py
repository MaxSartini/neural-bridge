#!/usr/bin/env python3
"""Run the locked 299-video zero-label direct-supervised confirmation."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.scripts import again_dense_2hz_phase4_pca_bridge as phase4  # noqa: E402
from backend.scripts import again_zero_label_deployment_stage_a as stage_a  # noqa: E402
from backend.scripts import (  # noqa: E402
    again_zero_label_direct_supervised_locked_confirmation as locked,
)
from backend.scripts import run_again_dense_2hz_zero_label_deployment_stage0 as stage0  # noqa: E402
from backend.scripts import run_again_dense_2hz_zero_label_deployment_stage_a as stage_a_runner  # noqa: E402


DEFAULT_STAGE0_ROOT = REPO_ROOT / "evidence/zero_label_video_only_deployment_stage0_20260714"
EXTERNAL_ROOT = Path(os.environ.get("NEURAL_BRIDGE_EXTERNAL_ROOT", REPO_ROOT / "external_assets"))
DEFAULT_EXTERNAL_PHASE4_ROOT = EXTERNAL_ROOT / "outputs/again_dense_2hz_phase4_pca_bridge_20260625_full"
DEFAULT_OUTPUT_ROOT = EXTERNAL_ROOT / "outputs/again_dense_2hz_zero_label_direct_supervised_locked_confirm_20260715"
PREREGISTRATION_PATH = REPO_ROOT / locked.PREREGISTRATION


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    stage_a_runner.write_json(path, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage0-root", default=str(DEFAULT_STAGE0_ROOT))
    parser.add_argument("--dense-root", default=str(stage0.DEFAULT_DENSE_ROOT))
    parser.add_argument("--external-phase4-root", default=str(DEFAULT_EXTERNAL_PHASE4_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--batch-size", type=int, default=stage_a.BATCH_SIZE)
    parser.add_argument("--pca-batch-size", type=int, default=384)
    parser.add_argument("--bootstrap-resamples", type=int, default=locked.BOOTSTRAP_RESAMPLES)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def implementation_preflight(
    *,
    stage0_root: Path,
    dense_root: Path,
    external_phase4_root: Path,
    output_root: Path,
) -> Mapping[str, Any]:
    stage0_result, splits, target = stage_a_runner.load_stage0(stage0_root)
    prereg_sha = stage_a.file_sha256(PREREGISTRATION_PATH)
    freeze = dict(locked.implementation_freeze_manifest(preregistration_sha256=prereg_sha))
    manifest = {
        **freeze,
        "stage0_result_sha256": stage_a.file_sha256(stage0_root / "stage0_result.json"),
        "split_manifest_sha256": stage_a.file_sha256(stage0_root / "split_manifest.json"),
        "target_manifest_sha256": stage_a.file_sha256(stage0_root / "target_identity_manifest.json"),
        "target_identity_digest": target["target_identity_digest"],
        "development_split_digest": splits["development_digest"],
        "locked_split_digest": splits["locked_digest"],
        "locked_video_count": int(splits["locked_count"]),
        "dense_root": str(dense_root),
        "external_phase4_root": str(external_phase4_root),
        "output_root": str(output_root),
        "stage0_pass": bool(stage0_result["stage0_pass"]),
        "authorized": True,
    }
    code_paths = {
        "stage_a_utility": REPO_ROOT / "backend/scripts/again_zero_label_deployment_stage_a.py",
        "stage_a_runner": REPO_ROOT / "backend/scripts/run_again_dense_2hz_zero_label_deployment_stage_a.py",
        "locked_utility": REPO_ROOT
        / "backend/scripts/again_zero_label_direct_supervised_locked_confirmation.py",
        "locked_runner": Path(__file__).resolve(),
        "contracts": REPO_ROOT
        / "tests/test_again_zero_label_direct_supervised_locked_confirmation.py",
    }
    manifest["implementation_code_sha256"] = {
        name: stage_a.file_sha256(path) for name, path in code_paths.items()
    }
    manifest["signed_implementation_digest"] = stage_a.canonical_digest(manifest)
    return manifest


def split_indices(
    df: pd.DataFrame, split_manifest: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, Mapping[int, np.ndarray]]:
    train_idx = stage_a.indices_for_videos(df, split_manifest["stage_b"]["train_videos"])
    locked_idx = stage_a.indices_for_videos(df, split_manifest["locked_videos"])
    if len(set(df.loc[train_idx, "video_id"]) & set(df.loc[locked_idx, "video_id"])):
        raise RuntimeError("Development and locked video ownership overlap")
    panels: dict[int, np.ndarray] = {}
    for record in split_manifest["stage_b"]["panels"]:
        panel = int(record["panel"])
        panels[panel] = stage_a.indices_for_videos(df, record["test_videos"])
    panel_rows = np.concatenate([panels[panel] for panel in locked.PANELS])
    if set(panel_rows.tolist()) != set(locked_idx.tolist()) or len(panel_rows) != len(locked_idx):
        raise RuntimeError("Frozen Stage B panels do not partition the locked rows")
    return train_idx, locked_idx, panels


def _feature_controls(
    train_features: stage_a.VideoFeatures,
    test_features: stage_a.VideoFeatures,
) -> Mapping[str, tuple[np.ndarray, np.ndarray]]:
    pca_stop = stage_a.WINDOW_ROWS * stage_a.PCA_WIDTH
    content_stop = pca_stop + stage_a.DIAGNOSTIC_WIDTH
    diagnostics_train = train_features.x_temporal.copy()
    diagnostics_test = test_features.x_temporal.copy()
    diagnostics_train[:, :pca_stop] = 0.0
    diagnostics_test[:, :pca_stop] = 0.0
    no_video_train = train_features.x_temporal.copy()
    no_video_test = test_features.x_temporal.copy()
    no_video_train[:, :content_stop] = 0.0
    no_video_test[:, :content_stop] = 0.0
    return {
        locked.DIAGNOSTICS_ONLY: (diagnostics_train, diagnostics_test),
        locked.NO_VIDEO: (no_video_train, no_video_test),
    }


def _save_zero_label_prediction(
    *,
    root: Path,
    inference_df: pd.DataFrame,
    locked_idx: np.ndarray,
    lane: str,
    seed: int,
    prediction: np.ndarray,
    target_identity_digest: str,
    metadata: Mapping[str, Any],
) -> Path:
    path = root / "predictions" / lane / f"seed{seed}.parquet"
    stage_a_runner._save_prediction(
        path=path,
        df=inference_df,
        indices=locked_idx,
        prediction=prediction,
        fold=0,
        lane=lane,
        seed=seed,
        target_identity_digest=target_identity_digest,
        metadata=metadata,
    )
    return path


def train_zero_label_lanes(
    *,
    df: pd.DataFrame,
    values: np.ndarray,
    hard_mask: np.ndarray,
    train_idx: np.ndarray,
    locked_idx: np.ndarray,
    train_features: stage_a.VideoFeatures,
    locked_features: stage_a.VideoFeatures,
    root: Path,
    target_identity_digest: str,
    batch_size: int,
) -> tuple[dict[str, dict[int, np.ndarray]], list[Mapping[str, Any]]]:
    inference_df = df[["video_id", "time_seconds"]].copy()
    hard_train_mask = hard_mask[train_idx]
    controls = _feature_controls(train_features, locked_features)
    shuffled_train, shuffled_train_map = stage_a.reassign_video_sequences(
        train_features.x_temporal,
        train_features.video_id,
        "locked_confirm|sequence_shuffle|train",
    )
    shuffled_test, shuffled_test_map = stage_a.reassign_video_sequences(
        locked_features.x_temporal,
        locked_features.video_id,
        "locked_confirm|sequence_shuffle|test",
    )
    permuted_target, label_map = stage_a.permute_video_targets(
        values[train_idx],
        train_features.video_id,
        "locked_confirm|hard_label_permutation",
    )
    predictions: dict[str, dict[int, np.ndarray]] = {
        lane: {} for lane in locked.ZERO_LABEL_LANES
    }
    audits: list[Mapping[str, Any]] = []
    for seed in locked.SEEDS:
        specs = {
            locked.PRIMARY: (
                train_features.x_temporal,
                locked_features.x_temporal,
                values[train_idx],
                hard_train_mask,
                True,
            ),
            locked.CURRENT_ROW: (
                train_features.x_current,
                locked_features.x_current,
                values[train_idx],
                hard_train_mask,
                False,
            ),
            locked.DIAGNOSTICS_ONLY: (
                controls[locked.DIAGNOSTICS_ONLY][0],
                controls[locked.DIAGNOSTICS_ONLY][1],
                values[train_idx],
                hard_train_mask,
                True,
            ),
            locked.NO_VIDEO: (
                controls[locked.NO_VIDEO][0],
                controls[locked.NO_VIDEO][1],
                values[train_idx],
                hard_train_mask,
                True,
            ),
            locked.SHUFFLED: (
                shuffled_train,
                shuffled_test,
                values[train_idx],
                hard_train_mask,
                True,
            ),
            locked.PERMUTED: (
                train_features.x_temporal,
                locked_features.x_temporal,
                permuted_target,
                np.isfinite(permuted_target),
                True,
            ),
        }
        for lane, (train_x, test_x, target, loss_mask, temporal_context) in specs.items():
            model = stage_a.train_scalar_model(
                train_x=train_x,
                test_x=test_x,
                train_target=target,
                train_loss_mask=loss_mask,
                train_video_id=train_features.video_id,
                temporal_context=temporal_context,
                seed=seed,
                checkpoint_path=root / "models" / lane / f"seed{seed}.npz",
                namespace=f"locked_confirm|{lane}|seed{seed}",
                weighted_huber=True,
                batch_size=batch_size,
            )
            prediction = model.test_prediction.astype(np.float32)
            predictions[lane][seed] = prediction
            path = _save_zero_label_prediction(
                root=root,
                inference_df=inference_df,
                locked_idx=locked_idx,
                lane=lane,
                seed=seed,
                prediction=prediction,
                target_identity_digest=target_identity_digest,
                metadata={
                    "checkpoint_sha256": model.checkpoint_sha256,
                    "best_epoch": model.best_epoch,
                    "best_validation_loss": model.best_validation_loss,
                    "observed_response_inputs_at_inference": False,
                    "hard_outcome_used_in_training_loss_only": lane != locked.PERMUTED,
                    "whole_video_hard_target_permutation": lane == locked.PERMUTED,
                    "whole_video_sequence_shuffle": lane == locked.SHUFFLED,
                },
            )
            audits.append(
                {
                    "seed": seed,
                    "lane": lane,
                    "prediction_sha256": stage_a.file_sha256(path),
                    "checkpoint_sha256": model.checkpoint_sha256,
                    "all_finite": bool(np.isfinite(prediction).all()),
                    "labels_loaded_by_predictor": False,
                    "observed_response_inputs_at_inference": False,
                }
            )
    write_json(
        root / "manifests" / "matched_control_mappings.json",
        {
            "sequence_shuffle_train": shuffled_train_map,
            "sequence_shuffle_locked": shuffled_test_map,
            "hard_target_permutation": label_map,
        },
    )
    return predictions, audits


def seal_zero_label_predictions(
    *, root: Path, predictions: Mapping[str, Mapping[int, np.ndarray]]
) -> Mapping[str, str]:
    expected = len(locked.ZERO_LABEL_LANES) * len(locked.SEEDS)
    paths = sorted(
        root / "predictions" / lane / f"seed{seed}.parquet"
        for lane in locked.ZERO_LABEL_LANES
        for seed in locked.SEEDS
    )
    if len(paths) != expected or not all(path.exists() for path in paths):
        raise RuntimeError("Zero-label prediction seal is incomplete")
    if any(len(predictions[lane]) != len(locked.SEEDS) for lane in locked.ZERO_LABEL_LANES):
        raise RuntimeError("In-memory zero-label prediction matrix is incomplete")
    seal = {str(path): stage_a.file_sha256(path) for path in paths}
    write_json(
        root / "manifests" / "zero_label_prediction_seal.json",
        {
            "prediction_files": seal,
            "sealed_before_teacher_ceiling_and_locked_label_join": True,
        },
    )
    return seal


def train_teacher_ceiling(
    *,
    df: pd.DataFrame,
    values: np.ndarray,
    compatibility_mask: np.ndarray,
    train_idx: np.ndarray,
    locked_idx: np.ndarray,
    pca_view: stage_a.PcaView,
    diagnostics_all: np.ndarray,
    root: Path,
    target_identity_digest: str,
    batch_size: int,
) -> tuple[dict[int, np.ndarray], list[Mapping[str, Any]]]:
    inference_df = df[["video_id", "time_seconds"]].copy()
    teacher: dict[int, np.ndarray] = {}
    audits: list[Mapping[str, Any]] = []
    train_compat = train_idx[compatibility_mask[train_idx]]
    test_compat = locked_idx[compatibility_mask[locked_idx]]
    for seed in locked.SEEDS:
        _, prediction_compat = stage_a_runner._train_teacher_member(
            df=df,
            values=values,
            train_idx=train_compat,
            test_idx=test_compat,
            pca_view=pca_view,
            diagnostics_all=diagnostics_all,
            seed=seed,
            root=root / "teachers" / f"seed{seed}",
            fold=0,
            batch_size=batch_size,
        )
        prediction = np.full(len(locked_idx), np.nan, dtype=np.float32)
        locked_position = {int(row): pos for pos, row in enumerate(locked_idx)}
        compat_positions = np.asarray([locked_position[int(row)] for row in test_compat], dtype=np.int64)
        prediction[compat_positions] = prediction_compat
        # The full hard-target scorer requires finite coverage. Rows without AR
        # compatibility are ceiling-ineligible and receive the development mean;
        # compatibility metrics remain the ceiling authority.
        fill = float(np.mean(prediction_compat))
        prediction[~np.isfinite(prediction)] = fill
        teacher[seed] = prediction
        path = root / "predictions" / locked.TEACHER / f"seed{seed}.parquet"
        stage_a_runner._save_prediction(
            path=path,
            df=inference_df,
            indices=locked_idx,
            prediction=prediction,
            fold=0,
            lane=locked.TEACHER,
            seed=seed,
            target_identity_digest=target_identity_digest,
            metadata={
                "non_deployable_teacher_ceiling": True,
                "observed_arousal_context_used": True,
                "opened_after_zero_label_prediction_seal": True,
                "compatibility_rows": int(len(test_compat)),
            },
        )
        audits.append(
            {
                "seed": seed,
                "lane": locked.TEACHER,
                "prediction_sha256": stage_a.file_sha256(path),
                "all_finite": bool(np.isfinite(prediction).all()),
                "opened_after_zero_label_prediction_seal": True,
            }
        )
    return teacher, audits


def _panel_position(
    locked_idx: np.ndarray, panels: Mapping[int, np.ndarray]
) -> Mapping[int, np.ndarray]:
    position = {int(row): pos for pos, row in enumerate(locked_idx)}
    return {
        panel: np.asarray([position[int(row)] for row in panels[panel]], dtype=np.int64)
        for panel in locked.PANELS
    }


def score_matrix(
    *,
    df: pd.DataFrame,
    values: np.ndarray,
    hard_mask: np.ndarray,
    compatibility_mask: np.ndarray,
    train_idx: np.ndarray,
    locked_idx: np.ndarray,
    panels: Mapping[int, np.ndarray],
    predictions: Mapping[str, Mapping[int, np.ndarray]],
    root: Path,
    target_identity_digest: str,
    split_manifest: Mapping[str, Any],
) -> tuple[pd.DataFrame, Mapping[str, Mapping[str, float]], Mapping[str, np.ndarray]]:
    inference_df = df[["video_id", "time_seconds"]].copy()
    all_predictions: dict[str, dict[int, np.ndarray]] = {
        lane: dict(seed_map) for lane, seed_map in predictions.items()
    }
    ensemble_predictions: dict[str, np.ndarray] = {}
    for lane in locked.LANES:
        ensemble = stage_a.ensemble_predictions(
            [all_predictions[lane][seed] for seed in locked.SEEDS]
        )
        ensemble_predictions[lane] = ensemble
        path = root / "predictions" / lane / f"ensemble_{locked.GROUP_NAME}.parquet"
        stage_a_runner._save_prediction(
            path=path,
            df=inference_df,
            indices=locked_idx,
            prediction=ensemble,
            fold=0,
            lane=lane,
            seed=0,
            target_identity_digest=target_identity_digest,
            metadata={"member_seeds": list(locked.SEEDS), "weights": [1 / 3, 1 / 3, 1 / 3]},
        )

    aggregate: dict[str, Mapping[str, float]] = {}
    for lane in locked.LANES:
        aggregate[lane] = dict(
            stage_a_runner._score_lane(
                df=df,
                values=values,
                hard_mask=hard_mask,
                compatibility_mask=compatibility_mask,
                train_idx=train_idx,
                test_idx=locked_idx,
                prediction=ensemble_predictions[lane],
            )
        )

    panel_positions = _panel_position(locked_idx, panels)
    panel_records = {int(record["panel"]): record for record in split_manifest["stage_b"]["panels"]}
    rows: list[Mapping[str, Any]] = []
    for panel in locked.PANELS:
        test_idx = panels[panel]
        positions = panel_positions[panel]
        for lane in locked.LANES:
            for seed in locked.SEEDS:
                path = root / "predictions" / lane / f"seed{seed}.parquet"
                metrics = stage_a_runner._score_lane(
                    df=df,
                    values=values,
                    hard_mask=hard_mask,
                    compatibility_mask=compatibility_mask,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    prediction=all_predictions[lane][seed][positions],
                )
                rows.append(
                    {
                        "stage": "locked_confirmation",
                        "split_digest": panel_records[panel]["split_digest"],
                        "panel": panel,
                        "lane": lane,
                        "row_type": "member",
                        "seed_or_group": str(seed),
                        "cold_start_policy": "row0_zero_history_no_label_burnin",
                        "prediction_sha256": stage_a.file_sha256(path),
                        "score_mask_digest": stage_a.array_digest(hard_mask[test_idx].astype(np.uint8)),
                        "compatibility_mask_digest": stage_a.array_digest(
                            compatibility_mask[test_idx].astype(np.uint8)
                        ),
                        **metrics,
                    }
                )
            ensemble_path = root / "predictions" / lane / f"ensemble_{locked.GROUP_NAME}.parquet"
            metrics = stage_a_runner._score_lane(
                df=df,
                values=values,
                hard_mask=hard_mask,
                compatibility_mask=compatibility_mask,
                train_idx=train_idx,
                test_idx=test_idx,
                prediction=ensemble_predictions[lane][positions],
            )
            rows.append(
                {
                    "stage": "locked_confirmation",
                    "split_digest": panel_records[panel]["split_digest"],
                    "panel": panel,
                    "lane": lane,
                    "row_type": "ensemble",
                    "seed_or_group": locked.GROUP_NAME,
                    "cold_start_policy": "row0_zero_history_no_label_burnin",
                    "prediction_sha256": stage_a.file_sha256(ensemble_path),
                    "score_mask_digest": stage_a.array_digest(hard_mask[test_idx].astype(np.uint8)),
                    "compatibility_mask_digest": stage_a.array_digest(
                        compatibility_mask[test_idx].astype(np.uint8)
                    ),
                    **metrics,
                }
            )
    return pd.DataFrame(rows), aggregate, ensemble_predictions


def paired_video_bootstrap(
    *,
    df: pd.DataFrame,
    values: np.ndarray,
    hard_mask: np.ndarray,
    train_idx: np.ndarray,
    locked_idx: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    aggregate: Mapping[str, Mapping[str, float]],
    resamples: int,
) -> Mapping[str, Mapping[str, float]]:
    if resamples != locked.BOOTSTRAP_RESAMPLES:
        raise ValueError(f"Locked bootstrap requires {locked.BOOTSTRAP_RESAMPLES} resamples")
    train_values = values[train_idx][hard_mask[train_idx]]
    valid_positions = np.flatnonzero(hard_mask[locked_idx])
    valid_videos = df.loc[locked_idx, "video_id"].astype(str).to_numpy()[valid_positions]
    test_values = values[locked_idx][valid_positions]
    times = df.loc[locked_idx, "time_seconds"].to_numpy(dtype=np.float32)[valid_positions]
    unique_videos = np.asarray(sorted(set(valid_videos)), dtype=object)
    blocks = {
        video: np.flatnonzero(valid_videos == video).astype(np.int64) for video in unique_videos
    }
    rng = np.random.default_rng(locked.BOOTSTRAP_SEED)
    results: dict[str, Mapping[str, float]] = {}
    for metric in locked.REQUIRED_METRICS:
        control_lane, _ = locked.strongest_control(aggregate, metric)
        primary = predictions[locked.PRIMARY][valid_positions]
        control = predictions[control_lane][valid_positions]
        deltas = np.empty(resamples, dtype=np.float64)
        for iteration in range(resamples):
            sampled = rng.choice(unique_videos, size=len(unique_videos), replace=True)
            take = np.concatenate([blocks[str(video)] for video in sampled])
            primary_metric = stage_a.score_prediction(
                train_values=train_values,
                test_values=test_values[take],
                prediction=primary[take],
                time_seconds=times[take],
            )[metric]
            control_metric = stage_a.score_prediction(
                train_values=train_values,
                test_values=test_values[take],
                prediction=control[take],
                time_seconds=times[take],
            )[metric]
            deltas[iteration] = float(primary_metric) - float(control_metric)
        results[metric] = {
            "control_lane": control_lane,
            "resamples": int(resamples),
            "lower_95_one_sided": float(np.quantile(deltas, 0.05)),
            "median_delta": float(np.median(deltas)),
            "mean_delta": float(np.mean(deltas)),
            "positive_fraction": float(np.mean(deltas > 0)),
        }
    return results


def report_text(result: Mapping[str, Any], root: Path) -> str:
    tier1 = result["tier1_zero_label_deployment_signal_confirmed"]
    tier2 = result["tier2_high_consistency_confirmation"]
    tier3 = result["tier3_first30_cold_start_confirmation"]
    lines = [
        "# AGAIN zero-label direct-supervised locked confirmation",
        "",
        f"Output root: `{root}`",
        "",
        f"- exact matrix: `{result['rows_actual']}/{result['rows_expected']}`",
        f"- Tier 1 baseline-beating deployment signal: **{str(tier1).lower()}**",
        f"- Tier 2 high-consistency confirmation: **{str(tier2).lower()}**",
        f"- Tier 3 first-30-second confirmation: **{str(tier3).lower()}**",
        f"- failed Tier 1 gates: `{json.dumps(result['failed_tier1_gates'])}`",
        "- teacher retention was report-only and Phase 7 was a ceiling, not a pass threshold.",
        "",
        "## Required endpoints",
        "",
    ]
    for metric, record in result["metric_results"].items():
        lines.extend(
            [
                f"### `{metric}`",
                "",
                f"- primary / strongest control: `{record['primary']:.10f}` / `{record['control']:.10f}`",
                f"- strongest control: `{record['strongest_control']}`",
                f"- aggregate delta: `{record['aggregate_delta']:+.10f}`",
                f"- panel wins: `{record['panel_wins']}`",
                f"- one-sided bootstrap lower 95%: `{record['bootstrap_lower_95_one_sided']:+.10f}`",
                f"- first-30 panel wins: `{record['first30_panel_wins']}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


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
    _, split_manifest, target_manifest = stage_a_runner.load_stage0(stage0_root)
    df = stage0.base.load_labels(dense_root)
    values, valid = stage0.redesigned.future_max_delta(df, 4, 10)
    if stage_a.array_digest(values.astype(np.float64)) != target_manifest["value_array_digest"]:
        raise RuntimeError("Locked target values differ from the Stage 0 target lock")
    train_idx, locked_idx, panels = split_indices(df, split_manifest)
    cortical = stage_a_runner.grouped.load_external_cortical_memmap(external_phase4_root, df)
    if cortical is None:
        raise FileNotFoundError("The external Phase 4 cortical memmap is required")
    accessor = phase4.CorticalVariantAccessor(cortical, df, base_family=stage_a.PCA_FAMILY)
    diagnostics_all = np.load(
        dense_root / "_derived/temporal_diagnostics_summary_features.npy", mmap_mode="r"
    )
    pca_view = stage_a.fit_or_load_pca_view(
        accessor=accessor,
        train_idx=train_idx,
        test_idx=locked_idx,
        output_root=root / "pca",
        name=f"locked_confirmation__{stage_a.PCA_FAMILY}",
        seed=20260715,
        batch_size=args.pca_batch_size,
    )
    inference_df = df[["video_id", "time_seconds"]].copy()
    train_features = stage_a.build_video_features(
        inference_df,
        train_idx,
        pca_view.train_scores,
        diagnostics_all[train_idx].astype(np.float32),
    )
    locked_features = stage_a.build_video_features(
        inference_df,
        locked_idx,
        pca_view.test_scores,
        diagnostics_all[locked_idx].astype(np.float32),
    )
    hard_mask = stage_a.valid_target_mask(df, values, valid)
    compatibility_mask = stage_a.teacher_compatibility_mask(df, hard_mask)
    predictions, audits = train_zero_label_lanes(
        df=df,
        values=values,
        hard_mask=hard_mask,
        train_idx=train_idx,
        locked_idx=locked_idx,
        train_features=train_features,
        locked_features=locked_features,
        root=root,
        target_identity_digest=target_manifest["target_identity_digest"],
        batch_size=args.batch_size,
    )
    seal_zero_label_predictions(root=root, predictions=predictions)
    teacher, teacher_audits = train_teacher_ceiling(
        df=df,
        values=values,
        compatibility_mask=compatibility_mask,
        train_idx=train_idx,
        locked_idx=locked_idx,
        pca_view=pca_view,
        diagnostics_all=diagnostics_all,
        root=root,
        target_identity_digest=target_manifest["target_identity_digest"],
        batch_size=args.batch_size,
    )
    predictions[locked.TEACHER] = teacher
    audits.extend(teacher_audits)
    frame, aggregate, ensembles = score_matrix(
        df=df,
        values=values,
        hard_mask=hard_mask,
        compatibility_mask=compatibility_mask,
        train_idx=train_idx,
        locked_idx=locked_idx,
        panels=panels,
        predictions=predictions,
        root=root,
        target_identity_digest=target_manifest["target_identity_digest"],
        split_manifest=split_manifest,
    )
    frame.to_csv(root / "metrics" / "locked_confirmation_rows.csv", index=False)
    write_json(root / "metrics" / "aggregate_metrics.json", aggregate)
    bootstrap = paired_video_bootstrap(
        df=df,
        values=values,
        hard_mask=hard_mask,
        train_idx=train_idx,
        locked_idx=locked_idx,
        predictions=ensembles,
        aggregate=aggregate,
        resamples=args.bootstrap_resamples,
    )
    write_json(root / "metrics" / "paired_video_bootstrap.json", bootstrap)
    audit_pass = bool(
        len(audits) == len(locked.LANES) * len(locked.SEEDS)
        and all(audit.get("all_finite", False) for audit in audits)
        and all(
            audit.get("labels_loaded_by_predictor") is False
            for audit in audits
            if audit["lane"] in locked.ZERO_LABEL_LANES
        )
        and len(locked_idx) > 0
        and len(set(df.loc[locked_idx, "video_id"])) == 299
    )
    result = dict(
        locked.compute_verdict(
            frame,
            aggregate=aggregate,
            bootstrap=bootstrap,
            audit_pass=audit_pass,
        )
    )
    result.update(
        {
            "duration_seconds": time.time() - started,
            "accelerator_detail": device,
            "target_identity_digest": target_manifest["target_identity_digest"],
            "development_split_digest": split_manifest["development_digest"],
            "locked_split_digest": split_manifest["locked_digest"],
            "locked_video_count": 299,
            "implementation_freeze_digest": preflight["implementation_freeze_digest"],
            "signed_implementation_digest": preflight["signed_implementation_digest"],
        }
    )
    write_json(root / "metrics" / "locked_confirmation_result.json", result)
    write_json(root / "diagnostics" / "locked_confirmation_audit.json", {"audit_pass": audit_pass, "audits": audits})
    report = report_text(result, root)
    report_path = root / "reports" / "again_dense_2hz_zero_label_direct_supervised_locked_confirmation_20260715.md"
    report_path.write_text(report, encoding="utf-8")
    gc.collect()
    print(json.dumps({"run_completed": True, "output_root": str(root), **result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
