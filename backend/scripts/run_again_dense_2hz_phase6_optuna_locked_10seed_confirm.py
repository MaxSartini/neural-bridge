#!/usr/bin/env python3
"""Confirm one locked Optuna winner across the canonical 10 blocked seeds.

This runner performs no optimization. It reuses the canonical seed-specific
frozen-AR caches and original rows, then trains the single pilot-locked
configuration for real and matched residual-control lanes on MLX.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND = REPO_ROOT / "backend"
for candidate in (str(REPO_ROOT), str(BACKEND)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from integrations import MLflowRun, RunProvenance  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase5_frozen_ar_residual as fr  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase5_temporal_residual_binary_big_confirm as confirm  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase5_temporal_residual_blocked as temporal  # noqa: E402


SCHEMA_VERSION = "again_dense_2hz_phase6_optuna_locked_10seed_confirm_v1"
SEEDS = confirm.SEEDS
FOLLOWUP_SEEDS = SEEDS[1:]
CONTROLS = (
    "real_residual",
    "shuffled_pca_residual",
    "random_pca_residual",
    "label_permutation_residual",
    "train_only_video_mean_residual",
    "diagnostics_only_residual",
)
PRIMARY_CONTROLS = confirm.PRIMARY_CONTROLS
LOCKED_WINNER = REPO_ROOT / "outputs/again_dense_2hz_phase6_optuna_selected_head_pilot_20260714_135902/manifests/locked_optuna_winner.json"
LOCKED_WINNER_SHA256 = "cf1f783105f5fc80df0639cbfd66d2487e2ba11f679e325cf6f9684099f98d0a"
CANONICAL_ROOT = REPO_ROOT / "outputs/again_dense_2hz_phase5_temporal_residual_binary_big_confirm_20260630_025437"
CANONICAL_METRICS = REPO_ROOT / "evidence/phase_5_5_binary_blocked_confirmation_20260630_025437/metrics/temporal_residual_binary_big_confirm_seed_metrics.csv"
CANONICAL_METRICS_SHA256 = "3b7c3aaf773a762bdd11c04342b41c9e7d44c7b7f868908d3abb95e93bf45a41"
EXPECTED_PARAMS: dict[str, float | int] = {
    "alpha_cap": 0.16,
    "alpha_initial_logit": -3.0,
    "gate_bias": 5.0,
    "hidden": 64,
    "lambda_binary": 0.8,
    "learning_rate": 0.00010528366155183298,
    "weight_decay": 0.00020452569809101856,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(confirm.SOURCE_ROOT))
    parser.add_argument("--foldsafe-pca-root", default=str(confirm.FOLDSAFE_PCA_ROOT))
    parser.add_argument("--canonical-root", default=str(CANONICAL_ROOT))
    parser.add_argument("--canonical-metrics", default=str(CANONICAL_METRICS))
    parser.add_argument("--locked-winner", default=str(LOCKED_WINNER))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def default_output_root() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path(f"outputs/again_dense_2hz_phase6_optuna_locked_10seed_confirm_{stamp}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def load_locked_winner(path: Path) -> tuple[dict[str, float | int], dict[str, Any]]:
    digest = sha256_file(path)
    if path.resolve() == LOCKED_WINNER.resolve() and digest != LOCKED_WINNER_SHA256:
        raise RuntimeError(f"Locked winner checksum mismatch: {digest}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = payload.get("params")
    if params != EXPECTED_PARAMS:
        raise RuntimeError(f"Locked parameters changed: {params!r}")
    if payload.get("trial_number") != 15:
        raise RuntimeError("Expected locked pilot trial 15")
    if payload.get("heldout_scored_during_study") is not False:
        raise RuntimeError("Pilot manifest does not attest held-out exclusion")
    return params, {"path": str(path), "sha256": digest, **payload}


def load_canonical_metrics(path: Path) -> pd.DataFrame:
    digest = sha256_file(path)
    if path.resolve() == CANONICAL_METRICS.resolve() and digest != CANONICAL_METRICS_SHA256:
        raise RuntimeError(f"Canonical metrics checksum mismatch: {digest}")
    frame = pd.read_csv(path)
    expected = {(seed, lane) for seed in SEEDS for lane in confirm.CONTROLS}
    actual = set(zip(frame["seed"].astype(int), frame["control_type"].astype(str)))
    if len(frame) != 70 or actual != expected:
        raise RuntimeError("Canonical metrics do not contain the exact 10 x 7 matrix")
    frame.attrs["sha256"] = digest
    return frame


def max_positive_contribution(values: pd.Series) -> float:
    positive = pd.to_numeric(values, errors="coerce")
    positive = positive[positive > 0]
    total = float(positive.sum())
    if total <= 0 or positive.empty:
        return math.inf
    return float(positive.max() / total)


def compute_result(
    tuned: pd.DataFrame,
    canonical: pd.DataFrame,
    *,
    provenance_pass: bool,
    leakage_pass: bool,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    summary = confirm.summarize_metrics(tuned)
    seed_df = confirm.seed_deltas(tuned)
    original = canonical[canonical["control_type"] == "real_residual"][
        ["seed", "pr_auc"]
    ].rename(columns={"pr_auc": "original_pr_auc"})
    seed_df = seed_df.merge(original, on="seed", validate="one_to_one")
    seed_df["tuned_minus_original_pr_auc"] = (
        seed_df["real_pr_auc"] - seed_df["original_pr_auc"]
    )
    followup = seed_df[seed_df["seed"].isin(FOLLOWUP_SEEDS)]
    full_delta = float(seed_df["tuned_minus_original_pr_auc"].mean())
    followup_delta = float(followup["tuned_minus_original_pr_auc"].mean())
    full_positive = int((seed_df["tuned_minus_original_pr_auc"] > 0).sum())
    followup_positive = int((followup["tuned_minus_original_pr_auc"] > 0).sum())
    tuned_row = confirm.row_for(summary, "real_residual")
    ar_row = confirm.row_for(summary, "frozen_ar_only")
    control_rows = [confirm.row_for(summary, lane) for lane in PRIMARY_CONTROLS]
    best_control = max(control_rows, key=lambda row: float(row["mean_pr_auc"]))
    primary_control_pass = all(
        float(tuned_row["mean_pr_auc"]) > float(row["mean_pr_auc"])
        for row in control_rows
    )
    positive_ar = int((seed_df["real_minus_frozen_ar_only_pr_auc"] > 0).sum())
    positive_best = int((seed_df["real_minus_best_control_pr_auc"] > 0).sum())
    contribution = max_positive_contribution(seed_df["tuned_minus_original_pr_auc"])
    checks = {
        "followup_mean_delta_at_least_0_001": followup_delta >= 0.001,
        "followup_positive_at_least_6_of_9": followup_positive >= 6,
        "full_mean_delta_positive": full_delta > 0,
        "full_positive_at_least_7_of_10": full_positive >= 7,
        "tuned_mean_exceeds_frozen_ar": float(tuned_row["mean_pr_auc"]) > float(ar_row["mean_pr_auc"]),
        "tuned_mean_exceeds_all_primary_controls": primary_control_pass,
        "positive_vs_ar_at_least_8_of_10": positive_ar >= 8,
        "positive_vs_best_control_at_least_8_of_10": positive_best >= 8,
        "single_seed_contribution_at_most_0_40": contribution <= 0.40,
        "locked_provenance_pass": provenance_pass,
        "leakage_and_runtime_audits_pass": leakage_pass,
    }
    failed = [name for name, passed in checks.items() if not passed]
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": confirm.TARGET_NAME,
        "architecture": confirm.ARCHITECTURE,
        "protocol": confirm.PROTOCOL,
        "seeds": list(SEEDS),
        "followup_seeds": list(FOLLOWUP_SEEDS),
        "rows_expected": 70,
        "rows_actual": int(len(tuned)),
        "tuned_pr_auc": float(tuned_row["mean_pr_auc"]),
        "original_pr_auc": float(original["original_pr_auc"].mean()),
        "frozen_ar_pr_auc": float(ar_row["mean_pr_auc"]),
        "best_control": str(best_control["control_type"]),
        "best_control_pr_auc": float(best_control["mean_pr_auc"]),
        "tuned_minus_original_pr_auc": full_delta,
        "followup_tuned_minus_original_pr_auc": followup_delta,
        "tuned_minus_frozen_ar_pr_auc": float(tuned_row["mean_pr_auc"] - ar_row["mean_pr_auc"]),
        "tuned_minus_best_control_pr_auc": float(tuned_row["mean_pr_auc"] - best_control["mean_pr_auc"]),
        "seeds_positive_vs_original": full_positive,
        "followup_seeds_positive_vs_original": followup_positive,
        "seeds_positive_vs_frozen_ar": positive_ar,
        "seeds_positive_vs_best_control": positive_best,
        "max_seed_contribution_vs_original": contribution,
        "checks": checks,
        "failed_gates": failed,
        "locked_improvement_pass": not failed,
        "grouped_started": False,
        "claim_promoted": False,
    }
    return result, summary, seed_df


def report_text(result: dict[str, Any], output_root: Path) -> str:
    return f"""# Phase 6 Optuna Locked-Winner 10-Seed Confirmation

Output root: `{output_root}`

This confirmation applies one pilot-locked configuration unchanged across the
canonical blocked seeds. It performs no optimization, encoder/PCA work,
grouped evaluation, 420 rerun, or continuous modeling.

## Result

- tuned PR-AUC: `{result['tuned_pr_auc']:.10f}`
- canonical original PR-AUC: `{result['original_pr_auc']:.10f}`
- frozen AR PR-AUC: `{result['frozen_ar_pr_auc']:.10f}`
- best tuned matched control: `{result['best_control']}` / `{result['best_control_pr_auc']:.10f}`
- tuned minus original, all 10 seeds: `{result['tuned_minus_original_pr_auc']:+.10f}`
- tuned minus original, nine follow-up seeds: `{result['followup_tuned_minus_original_pr_auc']:+.10f}`
- tuned minus frozen AR: `{result['tuned_minus_frozen_ar_pr_auc']:+.10f}`
- tuned minus best control: `{result['tuned_minus_best_control_pr_auc']:+.10f}`
- positive vs original: `{result['seeds_positive_vs_original']}/10`
- positive vs original on follow-up seeds: `{result['followup_seeds_positive_vs_original']}/9`
- positive vs frozen AR / best control: `{result['seeds_positive_vs_frozen_ar']}/10` / `{result['seeds_positive_vs_best_control']}/10`
- maximum positive-seed contribution vs original: `{result['max_seed_contribution_vs_original']:.4f}`

## Prespecified Verdict

- locked improvement pass: `{result['locked_improvement_pass']}`
- failed gates: `{result['failed_gates']}`

This result is exploratory until reviewed and deliberately promoted. A pass may
justify a later grouped locked-winner confirmation; it does not itself change
the canonical 420-row claim.
"""


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root) if args.output_root else default_output_root()
    source_root = Path(args.source_root)
    pca_root = Path(args.foldsafe_pca_root)
    canonical_root = Path(args.canonical_root)
    canonical_metrics_path = Path(args.canonical_metrics)
    locked_path = Path(args.locked_winner)
    params, locked_manifest = load_locked_winner(locked_path)
    canonical = load_canonical_metrics(canonical_metrics_path)
    dry = {
        "schema_version": SCHEMA_VERSION,
        "target": confirm.TARGET_NAME,
        "architecture": confirm.ARCHITECTURE,
        "seeds": list(SEEDS),
        "followup_seeds": list(FOLLOWUP_SEEDS),
        "rows": 70,
        "locked_trial": locked_manifest["trial_number"],
        "locked_params": params,
        "per_seed_optimization": False,
        "reuse_canonical_original_rows": True,
        "reuse_all_canonical_frozen_ar_scores": True,
        "accelerator": "mlx",
    }
    print(json.dumps(dry, indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(output_root)
    for subdir in (
        "manifests",
        "metrics",
        "diagnostics",
        "reports",
        "promotion",
        "checkpoints",
        "frozen_ar_scores",
        "mlflow_artifacts",
    ):
        (output_root / subdir).mkdir(parents=True, exist_ok=True)

    started = time.time()
    pca_info = temporal.load_pca_manifest(pca_root)
    blocks, df, dense_root, residual_meta = temporal.build_blocks(source_root, pca_root)
    block = temporal.block_for_target(blocks, confirm.TARGET_NAME)
    split_audit = temporal.verify_pca_rows(pca_root, block, pca_info["rows"])
    canonical_ar_checksums = (
        canonical[canonical["control_type"] == "frozen_ar_only"]
        .set_index("seed")["frozen_ar_test_checksum"]
        .to_dict()
    )
    fold_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    video_rows: list[dict[str, Any]] = []
    ar_rows: list[dict[str, Any]] = []

    for seed in SEEDS:
        ar = confirm.load_reused_ar_scores(
            canonical_root, output_root, block, seed
        )
        if ar is None:
            raise RuntimeError(f"Missing provenance-compatible canonical AR cache for seed {seed}")
        checksum_match = ar["test_checksum"] == canonical_ar_checksums[int(seed)]
        if not checksum_match:
            raise RuntimeError(f"Canonical frozen-AR checksum mismatch for seed {seed}")
        ar_rows.append(
            {
                **{k: v for k, v in ar.items() if k not in {"train_score", "train_reg", "test_score", "test_reg"}},
                "canonical_checksum_match": checksum_match,
            }
        )
        ar_metrics = temporal.metric_row_for_block(
            block, ar["train_score"], ar["test_score"], ar["test_reg"]
        )
        fold_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "target_name": confirm.TARGET_NAME,
                "target_type": "binary",
                "validation_protocol": confirm.PROTOCOL,
                "fold": confirm.FOLD,
                "seed": int(seed),
                "architecture": confirm.ARCHITECTURE,
                "control_type": "frozen_ar_only",
                "feature_name": confirm.FEATURE_NAME,
                "n_train": int(len(block.train_idx)),
                "n_test": int(len(block.test_idx)),
                "checkpoint_restore_pass": True,
                "eval_mode_scoring": True,
                "frozen_ar_train_checksum": ar["train_checksum"],
                "frozen_ar_test_checksum": ar["test_checksum"],
                **ar_metrics,
            }
        )
        for control in CONTROLS:
            pack = temporal.feature_pack_for(
                df, dense_root, pca_root, block, confirm.ARCHITECTURE, control, seed
            )
            metrics, curves, audit = temporal.train_temporal_residual(
                architecture=confirm.ARCHITECTURE,
                control=control,
                pack=pack,
                block=block,
                ar=ar,
                seed=seed,
                output_root=output_root,
                batch_size=args.batch_size,
                max_epochs=args.max_epochs,
                patience=args.patience,
                hyperparameters=params,
            )
            curve_rows.extend(curves)
            feature_rows.append(
                {"seed": int(seed), "control_type": control, "dims": pack.dims, "blocks": pack.manifest}
            )
            context_rows.append(pack.context_audit)
            if control == "label_permutation_residual":
                label_rows.append({"seed": int(seed), "label_policy": audit["label_policy"]})
            if control == "train_only_video_mean_residual":
                video_rows.append({"seed": int(seed), "uses_test_rows_for_mean": False})
            fold_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "target_name": confirm.TARGET_NAME,
                    "target_type": "binary",
                    "validation_protocol": confirm.PROTOCOL,
                    "fold": confirm.FOLD,
                    "seed": int(seed),
                    "architecture": confirm.ARCHITECTURE,
                    "control_type": control,
                    "feature_name": confirm.FEATURE_NAME,
                    "n_train": int(len(block.train_idx)),
                    "n_test": int(len(block.test_idx)),
                    "checkpoint_restore_pass": audit["checkpoint_restored"] or audit["residual_suppressed"],
                    "eval_mode_scoring": True,
                    "frozen_ar_train_checksum": ar["train_checksum"],
                    "frozen_ar_test_checksum": ar["test_checksum"],
                    **audit,
                    **metrics,
                }
            )
            pd.DataFrame(fold_rows).to_csv(
                output_root / "metrics/locked_10seed_metrics.partial.csv", index=False
            )
            gc.collect()

    fold_df = pd.DataFrame(fold_rows)
    if len(fold_df) != 70:
        raise RuntimeError(f"Expected 70 tuned rows, got {len(fold_df)}")
    label_pass = all(
        row["label_policy"] == "permuted_train_and_permuted_inner_val_selection"
        for row in label_rows
    )
    leakage_pass = bool(
        pca_info["audit"].get("leakage_audit_pass")
        and pca_info["audit"].get("no_test_rows_used_in_pca_fit")
        and split_audit.get("row_index_verified")
        and all(row.get("temporal_context_causal_only") for row in context_rows)
        and not any(row.get("uses_centered_or_future_windows") for row in context_rows)
        and all(row.get("same_video_history_masking") for row in context_rows)
        and label_pass
        and not any(row["uses_test_rows_for_mean"] for row in video_rows)
        and fold_df["checkpoint_restore_pass"].all()
        and fold_df["eval_mode_scoring"].all()
        and all(row["canonical_checksum_match"] for row in ar_rows)
    )
    provenance_pass = bool(
        locked_manifest["sha256"] == LOCKED_WINNER_SHA256
        and locked_manifest["params"] == EXPECTED_PARAMS
        and canonical.attrs["sha256"] == CANONICAL_METRICS_SHA256
    )
    result, summary, seed_df = compute_result(
        fold_df, canonical, provenance_pass=provenance_pass, leakage_pass=leakage_pass
    )
    result["duration_seconds"] = time.time() - started
    result["accelerator_detail"] = "Device(gpu, 0)"
    result["locked_winner"] = locked_manifest
    result["canonical_metrics_sha256"] = canonical.attrs["sha256"]

    fold_df.to_csv(output_root / "metrics/locked_10seed_metrics.csv", index=False)
    summary.to_csv(output_root / "metrics/locked_10seed_summary.csv", index=False)
    seed_df.to_csv(output_root / "metrics/locked_10seed_seed_deltas.csv", index=False)
    canonical.to_csv(output_root / "metrics/canonical_original_70row_reference.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(output_root / "diagnostics/training_curves.csv", index=False)
    fr.write_json(output_root / "diagnostics/leakage_and_runtime_audit.json", {
        "pass": leakage_pass,
        "pca_audit": pca_info["audit"],
        "split_audit": split_audit,
        "label_permutation_policy_pass": label_pass,
        "canonical_ar_rows": ar_rows,
    })
    fr.write_json(output_root / "manifests/locked_winner_provenance.json", locked_manifest)
    fr.write_json(output_root / "manifests/feature_manifest.json", {"features": feature_rows})
    fr.write_json(output_root / "promotion/locked_10seed_verdict.json", result)
    fr.write_json(output_root / "metrics/locked_10seed_result.json", result)

    report = report_text(result, output_root)
    report_name = f"again_dense_2hz_phase6_optuna_locked_10seed_confirm_{output_root.name.rsplit('_', 2)[-2]}_{output_root.name.rsplit('_', 1)[-1]}.md"
    (output_root / "reports" / report_name).write_text(report, encoding="utf-8")
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / report_name
    report_path.write_text(report, encoding="utf-8")
    run_manifest = {
        "schema_version": SCHEMA_VERSION,
        "git_commit": git_commit(),
        "output_root": str(output_root),
        "target": confirm.TARGET_NAME,
        "architecture": confirm.ARCHITECTURE,
        "seeds": list(SEEDS),
        "followup_seeds": list(FOLLOWUP_SEEDS),
        "controls": list(CONTROLS),
        "locked_params": params,
        "per_seed_optimization": False,
        "canonical_original_retrained": False,
        "canonical_frozen_ar_retrained": False,
        "no_vjepa_tribe_pca_rerun": True,
        "no_grouped": True,
        "no_420_rerun": True,
        "no_continuous": True,
        "residual_target_definition": residual_meta,
        "duration_seconds": result["duration_seconds"],
    }
    fr.write_json(output_root / "manifests/run_manifest.json", run_manifest)

    tracking_uri = f"sqlite:///{(output_root / 'mlflow.db').resolve()}"
    provenance = RunProvenance(
        git_commit=run_manifest["git_commit"],
        dataset_manifest_sha256=sha256_file(dense_root / "_run/global_run_metadata.json"),
        split_manifest_sha256=hashlib.sha256(
            np.concatenate([block.train_idx, block.test_idx]).astype(np.int64).tobytes()
        ).hexdigest(),
        feature_manifest_sha256=sha256_file(pca_root / "manifests/redesigned_pca_manifest.json"),
        target=confirm.TARGET_NAME,
        architecture=confirm.ARCHITECTURE,
        validation_protocol="blocked_temporal_70_30_locked_10seed_confirmation",
        seed=0,
        accelerator_backend="mlx",
        frozen_ar_sha256=hashlib.sha256(
            "".join(row["test_checksum"] for row in ar_rows).encode("utf-8")
        ).hexdigest(),
        extra={"locked_trial": 15, "seed_count": 10, "followup_seed_count": 9},
    )
    with MLflowRun(
        tracking_uri=tracking_uri,
        experiment_name="neural-bridge-phase6-optuna-locked-10seed-confirm",
        run_name="locked-winner-confirmation",
        provenance=provenance,
        artifact_location=(output_root / "mlflow_artifacts").resolve().as_uri(),
    ) as run:
        run.log_metrics({
            "tuned_pr_auc": result["tuned_pr_auc"],
            "tuned_minus_original_pr_auc": result["tuned_minus_original_pr_auc"],
            "followup_tuned_minus_original_pr_auc": result["followup_tuned_minus_original_pr_auc"],
            "tuned_minus_frozen_ar_pr_auc": result["tuned_minus_frozen_ar_pr_auc"],
            "tuned_minus_best_control_pr_auc": result["tuned_minus_best_control_pr_auc"],
        })
        run.log_artifact(output_root / "metrics/locked_10seed_result.json", artifact_path="metrics")
        run.log_artifact(report_path, artifact_path="reports")

    print(json.dumps({"run_completed": True, "output_root": str(output_root), "report": str(report_path), **result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
