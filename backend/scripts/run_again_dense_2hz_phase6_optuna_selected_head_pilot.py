#!/usr/bin/env python3
"""Bounded Optuna pilot around the proven AGAIN selected head.

The study tunes only on the existing outer-train inner split. The held-out
blocked test is scored once, after the winning parameters are locked. This is
an exploratory calibration experiment and never mutates Phase 5/5.5 evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND = REPO_ROOT / "backend"
for candidate in (str(REPO_ROOT), str(BACKEND)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from integrations import (  # noqa: E402
    AcceleratedObjectiveResult,
    MLflowRun,
    RunProvenance,
    TrainOnlyStudySpec,
    run_train_only_study,
)
from backend.scripts import run_again_dense_2hz_phase5_learned_heads as base  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase5_temporal_residual_binary_big_confirm as confirm  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase5_temporal_residual_blocked as temporal  # noqa: E402


SCHEMA_VERSION = "again_dense_2hz_phase6_optuna_selected_head_pilot_v1"
SEED = 20260625
N_TRIALS = 16
ARCHITECTURE = "short_temporal_conv_residual"
CONTROLS = (
    "real_residual",
    "shuffled_pca_residual",
    "random_pca_residual",
    "label_permutation_residual",
    "train_only_video_mean_residual",
    "diagnostics_only_residual",
)
ORIGINAL_PARAMS: dict[str, float | int] = {
    "hidden": 64,
    "learning_rate": 2e-4,
    "weight_decay": 1e-4,
    "alpha_initial_logit": -4.0,
    "alpha_cap": 0.12,
    "gate_bias": 4.0,
    "lambda_binary": 0.5,
}
CANONICAL_METRICS = REPO_ROOT / "evidence/phase_5_5_binary_blocked_confirmation_20260630_025437/metrics/temporal_residual_binary_big_confirm_seed_metrics.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(confirm.SOURCE_ROOT))
    parser.add_argument("--foldsafe-pca-root", default=str(confirm.FOLDSAFE_PCA_ROOT))
    parser.add_argument("--previous-temporal-root", default=str(confirm.PREVIOUS_TEMPORAL_ROOT))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def default_output_root() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path(f"outputs/again_dense_2hz_phase6_optuna_selected_head_pilot_{stamp}")


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values).tobytes()).hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def sampled_parameters(trial: Any) -> dict[str, float | int]:
    return {
        "hidden": trial.suggest_categorical("hidden", [48, 64, 96, 128]),
        "learning_rate": trial.suggest_float("learning_rate", 8e-5, 5e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 2e-5, 5e-4, log=True),
        "alpha_initial_logit": trial.suggest_categorical("alpha_initial_logit", [-5.0, -4.0, -3.0]),
        "alpha_cap": trial.suggest_categorical("alpha_cap", [0.08, 0.12, 0.16, 0.20]),
        "gate_bias": trial.suggest_categorical("gate_bias", [3.0, 4.0, 5.0]),
        "lambda_binary": trial.suggest_categorical("lambda_binary", [0.35, 0.5, 0.65, 0.8]),
    }


def promising_followup(
    *, tuned_pr: float, original_pr: float, ar_pr: float, best_control_pr: float
) -> bool:
    return bool(
        tuned_pr - original_pr >= 0.001
        and tuned_pr - ar_pr >= 0.001
        and tuned_pr - best_control_pr >= 0.001
    )


@dataclass(frozen=True)
class InnerPack:
    train_x: np.ndarray
    dims: dict[str, int]


def train_inner_only(
    *,
    trial: Any,
    params: dict[str, float | int],
    pack: InnerPack,
    block: Any,
    ar: dict[str, Any],
    inner_train: tuple[int, ...],
    inner_validation: tuple[int, ...],
    batch_size: int,
    max_epochs: int,
    patience: int,
) -> float:
    """Return best inner-validation delta without touching held-out arrays."""

    base.require_mlx()
    base.mx.random.seed(SEED)
    model = temporal.TemporalResidualHead(
        pack.train_x.shape[1],
        ARCHITECTURE,
        hidden=int(params["hidden"]),
        sequence_window=int(pack.dims.get("sequence_window", 0)),
        sequence_channels=int(pack.dims.get("sequence_channels", 0)),
        alpha_initial_logit=float(params["alpha_initial_logit"]),
        alpha_cap=float(params["alpha_cap"]),
        gate_bias=float(params["gate_bias"]),
    )
    optimizer = base.optim.AdamW(
        learning_rate=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )
    train_idx = np.asarray(inner_train, dtype=np.int64)
    val_idx = np.asarray(inner_validation, dtype=np.int64)
    ar_train_score = ar["train_score"].astype(np.float32)
    ar_train_reg = ar["train_reg"].astype(np.float32)
    ar_val_pr = average_precision_score(block.train_y[val_idx], ar_train_score[val_idx])
    rng = np.random.default_rng(SEED + 10007 * (temporal.ARCHITECTURES.index(ARCHITECTURE) + 1))

    def loss_fn(model_obj: Any, xb: Any, ar_b: Any, ar_r: Any, yb: Any, yr: Any) -> Any:
        out = model_obj(xb, ar_b, ar_r)
        reg = base.mx.mean(base.nn.losses.huber_loss(out[:, 0:1], yr, delta=1.0))
        bce = base.mx.mean(
            base.nn.losses.binary_cross_entropy(out[:, 1:2], yb, with_logits=True)
        )
        alpha_penalty = 0.01 * base.mx.mean(model_obj.alpha_value() ** 2)
        return reg + float(params["lambda_binary"]) * bce + alpha_penalty

    loss_and_grad = base.nn.value_and_grad(model, loss_fn)
    best_delta = 0.0
    stale = 0
    for epoch in range(1, max_epochs + 1):
        if hasattr(model, "train"):
            model.train()
        order = rng.permutation(train_idx)
        for start in range(0, len(train_idx), batch_size):
            rel = order[start : start + batch_size]
            loss, grads = loss_and_grad(
                model,
                base.mx.array(pack.train_x[rel], dtype=base.mx.float32),
                base.mx.array(ar_train_score[rel], dtype=base.mx.float32),
                base.mx.array(ar_train_reg[rel], dtype=base.mx.float32),
                base.mx.array(block.train_y[rel].astype(np.float32)[:, None], dtype=base.mx.float32),
                base.mx.array(block.train_cont[rel].astype(np.float32)[:, None], dtype=base.mx.float32),
            )
            grads, _ = base.optim.clip_grad_norm(grads, 1.0)
            optimizer.update(model, grads)
            base.mx.eval(loss, model.parameters(), optimizer.state)
        val_score, _val_reg, _gate = temporal.forward_residual(
            model,
            pack.train_x[val_idx],
            ar_train_score[val_idx],
            ar_train_reg[val_idx],
            batch_size=batch_size,
            target_type="binary",
        )
        val_pr = average_precision_score(block.train_y[val_idx], val_score)
        delta = float(val_pr - ar_val_pr)
        trial.report(delta, epoch)
        if delta > best_delta:
            best_delta = delta
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    return best_delta


def canonical_seed_metrics() -> dict[str, float]:
    rows = pd.read_csv(CANONICAL_METRICS)
    selected = rows[rows["seed"].eq(SEED)].set_index("control_type")
    return {
        "canonical_original_pr_auc": float(selected.loc["real_residual", "pr_auc"]),
        "canonical_frozen_ar_pr_auc": float(selected.loc["frozen_ar_only", "pr_auc"]),
    }


def report_text(result: dict[str, Any]) -> str:
    return f"""# Phase 6 Optuna Selected-Head Pilot

This is an exploratory one-seed calibration pilot around the already-proven
AGAIN selected head. Optuna saw only the inner train/validation partition; the
blocked held-out test was scored once after the winner was locked. The
canonical 420-row result is unchanged.

## Scope

- seed: `{SEED}`
- trials: `{result['trial_count']}`
- target: `{confirm.TARGET_NAME}`
- head: `{ARCHITECTURE}`
- feature: `{confirm.FEATURE_NAME}`
- MLX device: `{result['accelerator_detail']}`
- V-JEPA/TRIBE/PCA rerun: `false`

## Result

- canonical stored original PR-AUC: `{result['canonical_original_pr_auc']:.10f}`
- fresh original reproduction PR-AUC: `{result['original_pr_auc']:.10f}`
- Optuna-tuned PR-AUC: `{result['tuned_pr_auc']:.10f}`
- frozen AR PR-AUC: `{result['frozen_ar_pr_auc']:.10f}`
- best tuned matched control: `{result['best_control']}` / `{result['best_control_pr_auc']:.10f}`
- tuned minus fresh original: `{result['tuned_minus_original']:+.10f}`
- tuned minus frozen AR: `{result['tuned_minus_ar']:+.10f}`
- tuned minus best control: `{result['tuned_minus_best_control']:+.10f}`
- reproduction absolute difference: `{result['reproduction_abs_difference']:.10f}`
- promising enough to justify a multi-seed Optuna follow-up: `{result['promising_followup']}`

## Locked Winner

```json
{json.dumps(result['best_params'], indent=2, sort_keys=True)}
```

This single seed cannot promote a new claim. It only measures whether Optuna
adds enough value to justify a bounded multi-seed follow-up.
"""


def main() -> int:
    args = parse_args()
    if args.trials < 2 or args.trials > 32:
        raise ValueError("Pilot trials must be between 2 and 32")
    output_root = Path(args.output_root) if args.output_root else default_output_root()
    pca_root = Path(args.foldsafe_pca_root)
    previous_root = Path(args.previous_temporal_root)
    source_root = Path(args.source_root)
    pca_info = temporal.load_pca_manifest(pca_root)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "seed": SEED,
                    "trials": args.trials,
                    "target": confirm.TARGET_NAME,
                    "architecture": ARCHITECTURE,
                    "heldout_in_objective": False,
                },
                indent=2,
            )
        )
        return 0
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(output_root)
    for subdir in ("manifests", "metrics", "diagnostics", "reports", "checkpoints", "frozen_ar_scores", "mlflow_artifacts"):
        (output_root / subdir).mkdir(parents=True, exist_ok=True)

    started = time.time()
    blocks, df, dense_root, residual_meta = temporal.build_blocks(source_root, pca_root)
    block = temporal.block_for_target(blocks, confirm.TARGET_NAME)
    split_audit = temporal.verify_pca_rows(pca_root, block, pca_info["rows"])
    ar, _curves = confirm.obtain_ar_baseline(
        previous_root=previous_root,
        output_root=output_root,
        block=block,
        seed=SEED,
        batch_size=args.batch_size,
        ar_max_epochs=80,
        ar_patience=12,
    )
    real_pack = temporal.feature_pack_for(
        df, dense_root, pca_root, block, ARCHITECTURE, "real_residual", SEED
    )
    inner_pack = InnerPack(real_pack.train_x, real_pack.dims)
    tracking_uri = f"sqlite:///{(output_root / 'mlflow.db').resolve()}"
    provenance_base = {
        "git_commit": git_commit(),
        "dataset_manifest_sha256": file_digest(dense_root / "_run/global_run_metadata.json"),
        "split_manifest_sha256": array_digest(
            np.concatenate([block.train_idx, block.test_idx]).astype(np.int64)
        ),
        "feature_manifest_sha256": file_digest(
            pca_root / "manifests/redesigned_pca_manifest.json"
        ),
        "target": confirm.TARGET_NAME,
        "architecture": ARCHITECTURE,
        "validation_protocol": "blocked_temporal_70_30_inner_train_validation_only",
        "seed": SEED,
        "accelerator_backend": "mlx",
        "frozen_ar_sha256": ar["train_checksum"],
    }

    def objective(trial: Any, inner_train: tuple[int, ...], inner_val: tuple[int, ...]) -> AcceleratedObjectiveResult:
        params = sampled_parameters(trial)
        value = train_inner_only(
            trial=trial,
            params=params,
            pack=inner_pack,
            block=block,
            ar=ar,
            inner_train=inner_train,
            inner_validation=inner_val,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
        )
        provenance = RunProvenance(
            **provenance_base,
            extra={"trial_number": int(trial.number), **params},
        )
        with MLflowRun(
            tracking_uri=tracking_uri,
            experiment_name="neural-bridge-phase6-optuna-selected-head-pilot",
            run_name=f"trial-{trial.number:03d}",
            provenance=provenance,
            artifact_location=(output_root / "mlflow_artifacts").resolve().as_uri(),
        ) as run:
            run.log_metrics({"inner_val_delta_vs_frozen_ar_pr_auc": value})
        return AcceleratedObjectiveResult(value, "mlx")

    study = run_train_only_study(
        TrainOnlyStudySpec(
            study_name="again-phase6-selected-head-pilot",
            n_trials=args.trials,
            sampler_seed=SEED,
            accelerator_backend="mlx",
            storage=f"sqlite:///{(output_root / 'optuna.db').resolve()}",
            load_if_exists=True,
            initial_trials=(ORIGINAL_PARAMS,),
        ),
        objective,
        inner_train_indices=block.inner_train,
        inner_validation_indices=block.inner_val,
    )
    best_params = {key: value for key, value in study.best_params.items()}
    (output_root / "manifests/locked_optuna_winner.json").write_text(
        json.dumps(
            {
                "study_name": study.study_name,
                "trial_number": int(study.best_trial.number),
                "inner_val_delta_vs_frozen_ar_pr_auc": float(study.best_value),
                "params": best_params,
                "heldout_scored_during_study": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    ar_metrics = temporal.metric_row_for_block(
        block, ar["train_score"], ar["test_score"], ar["test_reg"]
    )
    original_metrics, original_curves, original_audit = temporal.train_temporal_residual(
        architecture=ARCHITECTURE,
        control="real_residual",
        pack=real_pack,
        block=block,
        ar=ar,
        seed=SEED,
        output_root=output_root / "original",
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        hyperparameters=ORIGINAL_PARAMS,
    )
    rows: list[dict[str, Any]] = [
        {"lane": "frozen_ar_only", **ar_metrics},
        {"lane": "original_real_residual", **original_audit, **original_metrics},
    ]
    tuned_curves: list[dict[str, Any]] = []
    for control in CONTROLS:
        pack = (
            real_pack
            if control == "real_residual"
            else temporal.feature_pack_for(
                df, dense_root, pca_root, block, ARCHITECTURE, control, SEED
            )
        )
        metrics, curves, audit = temporal.train_temporal_residual(
            architecture=ARCHITECTURE,
            control=control,
            pack=pack,
            block=block,
            ar=ar,
            seed=SEED,
            output_root=output_root / "optuna",
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            hyperparameters=best_params,
        )
        tuned_curves.extend(curves)
        rows.append({"lane": f"optuna_{control}", **audit, **metrics})

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_root / "metrics/pilot_heldout_metrics.csv", index=False)
    pd.DataFrame(original_curves).to_csv(
        output_root / "diagnostics/original_training_curve.csv", index=False
    )
    pd.DataFrame(tuned_curves).to_csv(
        output_root / "diagnostics/optuna_training_curves.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "trial": int(trial.number),
                "state": str(trial.state.name),
                "value": trial.value,
                **trial.params,
            }
            for trial in study.trials
        ]
    ).to_csv(output_root / "metrics/optuna_trials.csv", index=False)

    by_lane = metrics_df.set_index("lane")
    tuned_pr = float(by_lane.loc["optuna_real_residual", "pr_auc"])
    original_pr = float(by_lane.loc["original_real_residual", "pr_auc"])
    ar_pr = float(by_lane.loc["frozen_ar_only", "pr_auc"])
    control_pr = {
        control: float(by_lane.loc[f"optuna_{control}", "pr_auc"])
        for control in CONTROLS
        if control != "real_residual"
    }
    best_control = max(control_pr, key=control_pr.get)
    canonical = canonical_seed_metrics()
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "trial_count": len(study.trials),
        "seed": SEED,
        "target": confirm.TARGET_NAME,
        "architecture": ARCHITECTURE,
        "best_trial": int(study.best_trial.number),
        "best_inner_val_delta_vs_frozen_ar_pr_auc": float(study.best_value),
        "best_params": best_params,
        "original_params": ORIGINAL_PARAMS,
        "original_pr_auc": original_pr,
        "tuned_pr_auc": tuned_pr,
        "frozen_ar_pr_auc": ar_pr,
        "best_control": best_control,
        "best_control_pr_auc": control_pr[best_control],
        "tuned_minus_original": tuned_pr - original_pr,
        "tuned_minus_ar": tuned_pr - ar_pr,
        "tuned_minus_best_control": tuned_pr - control_pr[best_control],
        **canonical,
        "reproduction_abs_difference": abs(
            original_pr - canonical["canonical_original_pr_auc"]
        ),
        "promising_followup": promising_followup(
            tuned_pr=tuned_pr,
            original_pr=original_pr,
            ar_pr=ar_pr,
            best_control_pr=control_pr[best_control],
        ),
        "accelerator_detail": "Device(gpu, 0)",
        "heldout_scored_during_study": False,
        "split_audit": split_audit,
        "pca_audit": pca_info["audit"],
        "residual_target_definition": residual_meta,
        "duration_seconds": time.time() - started,
    }
    (output_root / "metrics/pilot_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = report_text(result)
    stamp = output_root.name.rsplit("_", 2)[-2] + "_" + output_root.name.rsplit("_", 1)[-1]
    report_name = f"again_dense_2hz_phase6_optuna_selected_head_pilot_{stamp}.md"
    (output_root / "reports" / report_name).write_text(report, encoding="utf-8")
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / report_name).write_text(report, encoding="utf-8")
    (output_root / "manifests/run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "git_commit": git_commit(),
                "output_root": str(output_root),
                "target": confirm.TARGET_NAME,
                "architecture": ARCHITECTURE,
                "seed": SEED,
                "trials": args.trials,
                "controls": list(CONTROLS),
                "no_vjepa_tribe_pca_rerun": True,
                "no_grouped": True,
                "no_420_rerun": True,
                "no_claim_change": True,
                "heldout_scored_during_study": False,
                "duration_seconds": result["duration_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
