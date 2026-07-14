#!/usr/bin/env python3
"""Stage A of the robust Optuna 720-row plan.

Searches only inner train/validation evidence across five development seeds,
locks one configuration, then evaluates it on five reserved inner-validation
seeds. No blocked held-out or grouped test scores are read or produced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

from integrations import AcceleratedObjectiveResult, MLflowRun, RunProvenance, TrainOnlyStudySpec, run_train_only_study  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase5_temporal_residual_binary_big_confirm as confirm  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase5_temporal_residual_blocked as temporal  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase6_optuna_selected_head_pilot as pilot  # noqa: E402


SCHEMA_VERSION = "again_dense_2hz_phase6_robust_multiseed_optuna_v1"
N_TRIALS = 24
DEVELOPMENT_SEEDS = confirm.SEEDS[:5]
VALIDATION_SEEDS = confirm.SEEDS[5:]
ORIGINAL_PARAMS: dict[str, float | int] = {
    **pilot.ORIGINAL_PARAMS,
    "max_epochs": 40,
    "patience": 8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(confirm.SOURCE_ROOT))
    parser.add_argument("--foldsafe-pca-root", default=str(confirm.FOLDSAFE_PCA_ROOT))
    parser.add_argument("--canonical-root", default=str(REPO_ROOT / "outputs/again_dense_2hz_phase5_temporal_residual_binary_big_confirm_20260630_025437"))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def default_output_root() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path(f"outputs/again_dense_2hz_phase6_robust_multiseed_optuna_{stamp}")


def sampled_parameters(trial: Any) -> dict[str, float | int]:
    return {
        **pilot.sampled_parameters(trial),
        "max_epochs": trial.suggest_categorical("max_epochs", [40, 60]),
        "patience": trial.suggest_categorical("patience", [8, 12]),
    }


def robust_objective(deltas: list[float]) -> float:
    values = np.asarray(deltas, dtype=np.float64)
    if values.shape != (5,) or not np.isfinite(values).all():
        raise ValueError("Robust objective requires five finite seed deltas")
    return float(
        0.50 * np.min(values)
        + 0.50 * np.quantile(values, 0.25)
        + 0.10 * np.mean(values)
    )


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def array_sha(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values).tobytes()).hexdigest()


def evaluate_params(
    params: dict[str, float | int],
    *,
    seeds: tuple[int, ...],
    pack: pilot.InnerPack,
    block: Any,
    ar_by_seed: dict[int, dict[str, Any]],
    batch_size: int,
) -> dict[int, float]:
    out: dict[int, float] = {}
    for seed in seeds:
        out[int(seed)] = pilot.train_inner_only(
            trial=None,
            params=params,
            pack=pack,
            block=block,
            ar=ar_by_seed[int(seed)],
            inner_train=tuple(int(v) for v in block.inner_train),
            inner_validation=tuple(int(v) for v in block.inner_val),
            batch_size=batch_size,
            max_epochs=int(params["max_epochs"]),
            patience=int(params["patience"]),
            training_seed=int(seed),
            report_to_trial=False,
        )
    return out


def validation_gate(candidate: dict[int, float], original: dict[int, float]) -> dict[str, Any]:
    paired = np.asarray([candidate[s] - original[s] for s in VALIDATION_SEEDS], dtype=np.float64)
    candidate_values = [candidate[s] for s in VALIDATION_SEEDS]
    original_values = [original[s] for s in VALIDATION_SEEDS]
    checks = {
        "robust_objective_gain_at_least_0_001": robust_objective(candidate_values) - robust_objective(original_values) >= 0.001,
        "mean_delta_improves": float(np.mean(candidate_values)) > float(np.mean(original_values)),
        "paired_wins_at_least_4_of_5": int((paired > 0).sum()) >= 4,
    }
    return {
        "candidate_robust_objective": robust_objective(candidate_values),
        "original_robust_objective": robust_objective(original_values),
        "robust_objective_gain": robust_objective(candidate_values) - robust_objective(original_values),
        "candidate_mean_delta_vs_ar": float(np.mean(candidate_values)),
        "original_mean_delta_vs_ar": float(np.mean(original_values)),
        "paired_candidate_minus_original": {str(seed): float(candidate[seed] - original[seed]) for seed in VALIDATION_SEEDS},
        "paired_wins": int((paired > 0).sum()),
        "checks": checks,
        "failed_gates": [name for name, passed in checks.items() if not passed],
        "stage_a_pass": all(checks.values()),
    }


def report_text(result: dict[str, Any], output_root: Path) -> str:
    gate = result["validation_gate"]
    return f"""# Phase 6 Robust Multi-Seed Optuna — Stage A

Output root: `{output_root}`

This is inner-train/validation-only development. No blocked held-out or grouped
test score was read or produced.

## Study

- trials: `{result['trial_count']}`
- development seeds: `{result['development_seeds']}`
- reserved validation seeds: `{result['validation_seeds']}`
- best trial: `{result['best_trial']}`
- best development robust objective: `{result['best_development_objective']:.10f}`

## Reserved Inner-Validation Gate

- candidate robust objective: `{gate['candidate_robust_objective']:.10f}`
- original robust objective: `{gate['original_robust_objective']:.10f}`
- gain: `{gate['robust_objective_gain']:+.10f}`
- paired wins: `{gate['paired_wins']}/5`
- Stage A pass: `{gate['stage_a_pass']}`
- failed gates: `{gate['failed_gates']}`

A pass authorizes only the preregistered 15-seed blocked Stage B. A failure
stops the 720-row campaign before held-out scoring.
"""


def main() -> int:
    args = parse_args()
    if args.trials < 2 or args.trials > 40:
        raise ValueError("Trials must be between 2 and 40")
    dry = {
        "schema_version": SCHEMA_VERSION,
        "trials": args.trials,
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "validation_seeds": list(VALIDATION_SEEDS),
        "heldout_scores_read": False,
        "grouped_scores_read": False,
        "original_enqueued_trial_zero": True,
        "accelerator": "mlx",
    }
    print(json.dumps(dry, indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    output_root = Path(args.output_root) if args.output_root else default_output_root()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(output_root)
    for subdir in ("manifests", "metrics", "reports", "frozen_ar_scores", "mlflow_artifacts"):
        (output_root / subdir).mkdir(parents=True, exist_ok=True)

    started = time.time()
    source_root = Path(args.source_root)
    pca_root = Path(args.foldsafe_pca_root)
    canonical_root = Path(args.canonical_root)
    blocks, df, dense_root, _meta = temporal.build_blocks(source_root, pca_root)
    block = temporal.block_for_target(blocks, confirm.TARGET_NAME)
    real = temporal.feature_pack_for(df, dense_root, pca_root, block, confirm.ARCHITECTURE, "real_residual", DEVELOPMENT_SEEDS[0])
    pack = pilot.InnerPack(real.train_x, real.dims)
    ar_by_seed: dict[int, dict[str, Any]] = {}
    for seed in (*DEVELOPMENT_SEEDS, *VALIDATION_SEEDS):
        ar = confirm.load_reused_ar_scores(canonical_root, output_root, block, int(seed))
        if ar is None:
            raise RuntimeError(f"Missing canonical frozen AR cache for seed {seed}")
        ar_by_seed[int(seed)] = ar

    trial_seed_rows: list[dict[str, Any]] = []
    tracking_uri = f"sqlite:///{(output_root / 'mlflow.db').resolve()}"

    def objective(trial: Any, inner_train: tuple[int, ...], inner_val: tuple[int, ...]) -> AcceleratedObjectiveResult:
        params = sampled_parameters(trial)
        deltas: list[float] = []
        for seed in DEVELOPMENT_SEEDS:
            delta = pilot.train_inner_only(
                trial=trial,
                params=params,
                pack=pack,
                block=block,
                ar=ar_by_seed[int(seed)],
                inner_train=inner_train,
                inner_validation=inner_val,
                batch_size=args.batch_size,
                max_epochs=int(params["max_epochs"]),
                patience=int(params["patience"]),
                training_seed=int(seed),
                report_to_trial=False,
            )
            deltas.append(delta)
            trial_seed_rows.append({"trial": int(trial.number), "seed": int(seed), "inner_val_delta_vs_frozen_ar": delta, **params})
        value = robust_objective(deltas)
        provenance = RunProvenance(
            git_commit=git_commit(),
            dataset_manifest_sha256=hashlib.sha256(str(dense_root).encode()).hexdigest(),
            split_manifest_sha256=array_sha(np.concatenate([block.inner_train, block.inner_val])),
            feature_manifest_sha256=array_sha(real.train_x[: min(1024, len(real.train_x))]),
            target=confirm.TARGET_NAME,
            architecture=confirm.ARCHITECTURE,
            validation_protocol="inner_only_robust_multiseed_development",
            seed=0,
            accelerator_backend="mlx",
            extra={"trial": int(trial.number), "seed_count": 5, **params},
        )
        with MLflowRun(
            tracking_uri=tracking_uri,
            experiment_name="neural-bridge-phase6-robust-multiseed-optuna",
            run_name=f"trial-{trial.number:03d}",
            provenance=provenance,
            artifact_location=(output_root / "mlflow_artifacts").resolve().as_uri(),
        ) as run:
            run.log_metrics({"robust_inner_val_objective": value, "mean_inner_val_delta_vs_ar": float(np.mean(deltas)), "min_inner_val_delta_vs_ar": float(np.min(deltas))})
        return AcceleratedObjectiveResult(value, "mlx")

    study = run_train_only_study(
        TrainOnlyStudySpec(
            study_name="again-phase6-robust-multiseed-optuna",
            n_trials=args.trials,
            sampler_seed=20260714,
            accelerator_backend="mlx",
            storage=f"sqlite:///{(output_root / 'optuna.db').resolve()}",
            load_if_exists=True,
            initial_trials=(ORIGINAL_PARAMS,),
        ),
        objective,
        inner_train_indices=block.inner_train,
        inner_validation_indices=block.inner_val,
    )
    best_params = dict(study.best_params)
    candidate_validation = evaluate_params(best_params, seeds=VALIDATION_SEEDS, pack=pack, block=block, ar_by_seed=ar_by_seed, batch_size=args.batch_size)
    original_validation = evaluate_params(ORIGINAL_PARAMS, seeds=VALIDATION_SEEDS, pack=pack, block=block, ar_by_seed=ar_by_seed, batch_size=args.batch_size)
    gate = validation_gate(candidate_validation, original_validation)
    result = {
        **dry,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "trial_count": len(study.trials),
        "best_trial": int(study.best_trial.number),
        "best_development_objective": float(study.best_value),
        "best_params": best_params,
        "original_params": ORIGINAL_PARAMS,
        "candidate_validation_deltas": {str(k): v for k, v in candidate_validation.items()},
        "original_validation_deltas": {str(k): v for k, v in original_validation.items()},
        "validation_gate": gate,
        "duration_seconds": time.time() - started,
        "accelerator_detail": "Device(gpu, 0)",
        "next_stage_authorized": bool(gate["stage_a_pass"]),
    }
    pd.DataFrame(trial_seed_rows).to_csv(output_root / "metrics/trial_seed_inner_validation.csv", index=False)
    pd.DataFrame([{"trial": int(t.number), "state": t.state.name, "value": t.value, **t.params} for t in study.trials]).to_csv(output_root / "metrics/optuna_trials.csv", index=False)
    (output_root / "metrics/result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "manifests/locked_winner.json").write_text(json.dumps({"trial": int(study.best_trial.number), "params": best_params, "development_objective": float(study.best_value), "validation_gate": gate, "heldout_scores_read": False}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = report_text(result, output_root)
    name = f"again_dense_2hz_phase6_robust_multiseed_optuna_{output_root.name.rsplit('_', 2)[-2]}_{output_root.name.rsplit('_', 1)[-1]}.md"
    (output_root / "reports" / name).write_text(report, encoding="utf-8")
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / name
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps({"run_completed": True, "output_root": str(output_root), "report": str(report_path), **result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
