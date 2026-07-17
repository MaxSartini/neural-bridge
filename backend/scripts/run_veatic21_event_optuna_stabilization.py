#!/usr/bin/env python3
"""Run preregistered VEATIC event Optuna stabilization and fresh-seed showdown."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np

from backend.integrations import MLflowRun, RunProvenance, require_accelerator
from backend.integrations._optional import require_upstream
from backend.scripts import run_veatic21_again_parity_discovery as parity
from backend.scripts import run_veatic21_endstate as runner
from backend.scripts import veatic21_distilled_program as program
from backend.scripts import veatic21_execution as execution
from backend.scripts import veatic21_modeling as modeling


SCHEMA_VERSION = "veatic21_event_optuna_stabilization_v1"
PREREGISTRATION = Path(
    "docs/veatic21_event_optuna_stabilization_preregistration_20260717.md"
)
TRIAL_COUNT = 50
SAMPLER_SEED = 20260717
SEARCH_PANELS = ((1, 1), (2, 2), (3, 3), (4, 1), (5, 2))
ALL_PANELS = tuple((outer, inner) for outer in parity.OUTER_FOLDS for inner in parity.INNER_FOLDS)
HELDOUT_PANELS = tuple(panel for panel in ALL_PANELS if panel not in SEARCH_PANELS)
SEARCH_SEEDS = parity.SEEDS
SHOWDOWN_SEEDS = (20260719, 20260720, 20260721, 20260722, 20260723)
LANES = ("tuned", "original")
RECIPE = next(
    recipe
    for recipe in parity.RECIPES
    if recipe.name == "temporal_mean_2s_pca256_again_clean_joint"
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
ORIGINAL_TRAINING = parity.TrainingSettings()
SEARCH_SPACE = {
    "hidden": [48, 64, 96, 128],
    "learning_rate": [5e-5, 5e-4],
    "weight_decay": [1e-5, 1e-3],
    "alpha_initial_logit": [-6.0, -5.0, -4.0, -3.0],
    "alpha_cap": [0.04, 0.06, 0.08, 0.12, 0.16],
    "gate_bias": [3.0, 4.0, 5.0, 6.0],
    "lambda_binary": [0.35, 0.5, 0.65, 0.8, 1.0],
}


class StabilizationError(RuntimeError):
    pass


@dataclass(frozen=True)
class StudyContext:
    args: argparse.Namespace
    seal: Any
    plan: Any
    dataset: Any
    preflight: Mapping[str, Any]
    target: np.ndarray
    target_valid: np.ndarray
    ar_x: np.ndarray
    ar_context: np.ndarray


@dataclass(frozen=True)
class PanelData:
    outer: int
    inner: int
    train_rows: np.ndarray
    test_rows: np.ndarray
    train_x: np.ndarray
    test_x: np.ndarray
    train_ar_x: np.ndarray
    test_ar_x: np.ndarray
    train_event: np.ndarray
    test_event: np.ndarray
    train_continuous: np.ndarray
    train_valid: np.ndarray
    test_valid: np.ndarray
    train_videos: np.ndarray
    threshold: float
    panel_stats: Mapping[str, Any]
    pca_audit: Mapping[str, Any]
    pca_provenance: Mapping[str, Any]

    @property
    def key(self) -> tuple[int, int]:
        return self.outer, self.inner


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).resolve().parents[2]
    ).strip()


def external_root() -> Path:
    value = os.environ.get("NEURAL_BRIDGE_EXTERNAL_ROOT")
    return Path(value) if value else Path("/Volumes/onn. Drive/Neural Bridge")


def build_parser() -> argparse.ArgumentParser:
    root = external_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=root / "cache/veatic_h100_tribe_v2_mlx_compact_20260716",
    )
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=root / "cache/veatic_h100_vjepa21_compact_20260716",
    )
    parser.add_argument(
        "--shared-derived-root",
        type=Path,
        default=root / "outputs/veatic21_endstate_shared_derived_20260717",
    )
    parser.add_argument(
        "--source-parity-root",
        type=Path,
        default=root / "outputs/veatic21_again_parity_inner_discovery_20260717",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "outputs/veatic21_event_optuna_stabilization_20260717",
    )
    parser.add_argument(
        "--identity-manifest",
        type=Path,
        default=runner.compact.DEFAULT_IDENTITY_MANIFEST,
    )
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--stage", choices=("search", "showdown", "all"), default="all")
    parser.add_argument("--skip-checksums", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    return parser


def dry_schedule() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "trial_count": TRIAL_COUNT,
        "search_panels": [list(panel) for panel in SEARCH_PANELS],
        "heldout_panels": [list(panel) for panel in HELDOUT_PANELS],
        "search_seeds": list(SEARCH_SEEDS),
        "showdown_seeds": list(SHOWDOWN_SEEDS),
        "showdown_lanes": list(LANES),
        "search_member_fits": TRIAL_COUNT * len(SEARCH_PANELS) * len(SEARCH_SEEDS),
        "showdown_residual_member_fits": len(ALL_PANELS) * len(SHOWDOWN_SEEDS) * len(LANES),
        "showdown_ar_fits": len(ALL_PANELS) * len(SHOWDOWN_SEEDS),
        "checkpoint_selection_from_epoch": parity.CHECKPOINT_ELIGIBLE_EPOCH,
        "early_stop_not_before_epoch": ORIGINAL_TRAINING.min_epochs,
        "patience": ORIGINAL_TRAINING.patience,
        "max_epochs": ORIGINAL_TRAINING.max_epochs,
        "outer_test_scores_used": False,
        "explicitly_nonpromotable": True,
    }


def build_context(args: argparse.Namespace) -> StudyContext:
    cache = runner.compact.Veatic21CompactCache(
        args.cache_root,
        upstream_root=args.upstream_root,
        identity_manifest_path=args.identity_manifest,
        verify_checksums=not bool(args.skip_checksums),
    )
    report = cache.validate()
    seal = runner.dataset_seal_from_report(report)
    plan = runner.build_plan(seal)
    dataset = runner.materialize_dense_dataset(
        cache=cache, seal=seal, shared_root=args.shared_derived_root
    )
    preflight = parity.preflight_quality_alignment_audit(cache, dataset, plan)
    if not preflight["passed"]:
        raise StabilizationError(f"preflight failed: {preflight['checks']}")
    ar_x, ar_context, _ = program.canonical_ar_history_features(
        dataset.arousal, dataset.video_id
    )
    return StudyContext(
        args=args,
        seal=seal,
        plan=plan,
        dataset=dataset,
        preflight=preflight,
        target=np.asarray(dataset.target_values[parity.TARGET_NAME], dtype=np.float32),
        target_valid=np.asarray(dataset.target_valid[parity.TARGET_NAME], dtype=bool),
        ar_x=ar_x,
        ar_context=ar_context,
    )


def prepare_panel(context: StudyContext, outer: int, inner: int) -> PanelData:
    prepared = execution._prepare_features(
        dataset=context.dataset,
        plan=context.plan,
        recipe=parity._feature_recipe(RECIPE),
        outer_fold=outer,
        inner_fold=inner,
        args=context.args,
    )
    pca_audit = parity.validate_veatic_pca_provenance(
        prepared=prepared,
        dataset=context.dataset,
        plan=context.plan,
        shared_derived_root=context.args.shared_derived_root,
    )
    train_rows = prepared.train_rows
    test_rows = prepared.test_rows
    train_cont = np.maximum(context.target[train_rows], 0.0).astype(np.float32)
    test_cont = np.maximum(context.target[test_rows], 0.0).astype(np.float32)
    train_valid = (
        context.target_valid[train_rows]
        & context.dataset.quality_valid[train_rows]
        & context.ar_context[train_rows]
    )
    test_valid = (
        context.target_valid[test_rows]
        & context.dataset.quality_valid[test_rows]
        & context.ar_context[test_rows]
    )
    threshold = float(np.quantile(train_cont[train_valid], 0.90))
    train_event = (train_cont >= threshold).astype(np.float32)
    test_event = (test_cont >= threshold).astype(np.float32)
    if threshold <= 0 or np.unique(test_event[test_valid]).size != 2:
        raise StabilizationError(f"invalid event panel {(outer, inner)}")
    panel_stats = parity.event_panel_stats(
        labels=test_event,
        valid=test_valid,
        videos=prepared.test.video_id,
        global_rows=test_rows,
    )
    expected = next(
        row
        for row in context.preflight["event_panels"]
        if row["outer_fold"] == outer and row["inner_fold"] == inner
    )
    if (
        not math.isclose(threshold, float(expected["event_threshold"]), abs_tol=1e-12)
        or panel_stats["heldout_label_digest"] != expected["heldout_label_digest"]
        or panel_stats["heldout_valid_row_digest"] != expected["heldout_valid_row_digest"]
    ):
        raise StabilizationError(f"panel {(outer, inner)} drifted from preflight")
    return PanelData(
        outer=outer,
        inner=inner,
        train_rows=train_rows,
        test_rows=test_rows,
        train_x=parity.clean_or_enriched_view(
            prepared.train.x_temporal,
            pca_width=RECIPE.pca_width,
            input_variant=RECIPE.input_variant,
        ),
        test_x=parity.clean_or_enriched_view(
            prepared.test.x_temporal,
            pca_width=RECIPE.pca_width,
            input_variant=RECIPE.input_variant,
        ),
        train_ar_x=context.ar_x[train_rows],
        test_ar_x=context.ar_x[test_rows],
        train_event=train_event,
        test_event=test_event,
        train_continuous=train_cont,
        train_valid=train_valid,
        test_valid=test_valid,
        train_videos=prepared.train.video_id,
        threshold=threshold,
        panel_stats=panel_stats,
        pca_audit=pca_audit,
        pca_provenance=dict(prepared.provenance),
    )


def ownership(panel: PanelData, seed: int, *, search: bool) -> Any:
    programme = "again-method-parity" if search else "event-optuna-stabilization"
    namespace = (
        f"veatic21|{programme}|outer{panel.outer}|inner{panel.inner}|seed{seed}"
    )
    return program.build_inner_video_ownership(panel.train_videos, namespace=namespace)


def params_to_training(params: Mapping[str, Any]) -> parity.TrainingSettings:
    return parity.TrainingSettings(
        learning_rate=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )


def params_to_hyperparameters(
    params: Mapping[str, Any],
) -> parity.ResidualHyperparameters:
    return parity.ResidualHyperparameters(
        hidden=int(params["hidden"]),
        alpha_initial_logit=float(params["alpha_initial_logit"]),
        alpha_cap=float(params["alpha_cap"]),
        gate_bias=float(params["gate_bias"]),
        lambda_binary=float(params["lambda_binary"]),
    )


def sample_parameters(trial: Any) -> dict[str, float | int]:
    return {
        "hidden": trial.suggest_categorical("hidden", SEARCH_SPACE["hidden"]),
        "learning_rate": trial.suggest_float(
            "learning_rate", *SEARCH_SPACE["learning_rate"], log=True
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay", *SEARCH_SPACE["weight_decay"], log=True
        ),
        "alpha_initial_logit": trial.suggest_categorical(
            "alpha_initial_logit", SEARCH_SPACE["alpha_initial_logit"]
        ),
        "alpha_cap": trial.suggest_categorical("alpha_cap", SEARCH_SPACE["alpha_cap"]),
        "gate_bias": trial.suggest_categorical("gate_bias", SEARCH_SPACE["gate_bias"]),
        "lambda_binary": trial.suggest_categorical(
            "lambda_binary", SEARCH_SPACE["lambda_binary"]
        ),
    }


def robust_objective(panel_rows: Sequence[Mapping[str, float]]) -> tuple[float, dict[str, float]]:
    deltas = np.asarray([row["ensemble_delta_vs_ar"] for row in panel_rows], dtype=float)
    member_delta = float(np.mean([row["member_mean_delta_vs_ar"] for row in panel_rows]))
    uplift = float(np.mean([row["ensemble_uplift"] for row in panel_rows]))
    metrics = {
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "q25_delta": float(np.quantile(deltas, 0.25)),
        "worst_delta": float(np.min(deltas)),
        "std_delta": float(np.std(deltas, ddof=0)),
        "win_rate": float(np.mean(deltas > 0)),
        "member_mean_delta": member_delta,
        "ensemble_uplift": uplift,
    }
    value = (
        0.30 * metrics["mean_delta"]
        + 0.20 * metrics["median_delta"]
        + 0.15 * metrics["q25_delta"]
        + 0.15 * metrics["worst_delta"]
        - 0.10 * metrics["std_delta"]
        + 0.05 * member_delta
        + 0.05 * uplift
        + 0.002 * metrics["win_rate"]
    )
    return float(value), metrics


def run_identity(context: StudyContext) -> dict[str, Any]:
    prereg = PREREGISTRATION.resolve()
    source_identity = json.loads(
        (context.args.source_parity_root / "run_identity.json").read_text(encoding="utf-8")
    )
    payload = {
        **dry_schedule(),
        "dataset_seal_digest": runner.canonical_digest(context.seal.manifest()),
        "plan_digest": context.plan.digest,
        "quality_alignment_zero_event_preflight_digest": context.preflight["digest"],
        "recipe": asdict(RECIPE) | {"digest": RECIPE.digest},
        "original_params": ORIGINAL_PARAMS,
        "search_space": SEARCH_SPACE,
        "preregistration": str(prereg),
        "preregistration_sha256": runner.file_sha256(prereg),
        "executor_sha256": runner.file_sha256(Path(__file__)),
        "parity_executor_sha256": runner.file_sha256(Path(parity.__file__)),
        "source_parity_run_identity_digest": source_identity["digest"],
        "cache_root": str(context.args.cache_root.resolve()),
        "upstream_root": str(context.args.upstream_root.resolve()),
        "shared_derived_root": str(context.args.shared_derived_root.resolve()),
        "git_commit": git_commit(),
    }
    return {**payload, "digest": runner.canonical_digest(payload)}


def initialize_output(context: StudyContext) -> dict[str, Any]:
    identity = run_identity(context)
    root = context.args.output_root
    root.mkdir(parents=True, exist_ok=True)
    identity_path = root / "run_identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise StabilizationError("run identity mismatch")
    else:
        runner.atomic_json(identity_path, identity)
    runner.atomic_json(root / "schedule.json", dry_schedule())
    runner.atomic_json(
        root / "quality_alignment_zero_event_preflight.json", dict(context.preflight)
    )
    return identity


def load_search_ar(
    context: StudyContext, panel: PanelData, seed: int, panel_ownership: Any
) -> parity.DualScores:
    root = (
        context.args.source_parity_root
        / "ar"
        / f"outer_{panel.outer}"
        / f"inner_{panel.inner}"
        / f"seed_{seed}"
    )
    manifest_path = root / "scores.json"
    score_path = root / "scores.npz"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = manifest["identity"]
    expected = {
        "kind": "veatic_only_dual_task_frozen_ar7",
        "seed": seed,
        "outer_fold": panel.outer,
        "inner_fold": panel.inner,
        "target": parity.TARGET_NAME,
        "ownership_digest": panel_ownership.digest,
        "train_x_digest": runner.array_digest(panel.train_ar_x),
        "test_x_digest": runner.array_digest(panel.test_ar_x),
        "event_digest": runner.array_digest(panel.train_event),
        "continuous_digest": runner.array_digest(panel.train_continuous),
        "valid_digest": runner.array_digest(panel.train_valid.astype(np.uint8)),
        "settings": asdict(ORIGINAL_TRAINING),
    }
    if any(identity.get(key) != value for key, value in expected.items()):
        raise StabilizationError(f"source frozen AR identity drift for {panel.key}, seed {seed}")
    scores = parity._load_score_bundle(score_path, manifest_path, identity)
    if scores is None:
        raise StabilizationError("source frozen AR bundle missing")
    return scores


def evaluate_residual(
    *,
    context: StudyContext,
    identity: Mapping[str, Any],
    panel: PanelData,
    seed: int,
    frozen_ar: parity.DualScores,
    panel_ownership: Any,
    params: Mapping[str, Any],
    output_root: Path,
    stage: str,
    lane: str,
) -> parity.DualScores:
    return parity.fit_dual_residual(
        x_train=panel.train_x,
        x_test=panel.test_x,
        event=panel.train_event,
        continuous=panel.train_continuous,
        valid=panel.train_valid,
        ownership=panel_ownership,
        frozen_ar=frozen_ar,
        pca_width=RECIPE.pca_width,
        seed=seed,
        output_root=output_root,
        identity_base={
            "run_identity_digest": identity["digest"],
            "stage": stage,
            "lane": lane,
            "outer_fold": panel.outer,
            "inner_fold": panel.inner,
            "target": parity.TARGET_NAME,
            "recipe": asdict(RECIPE),
            "recipe_digest": RECIPE.digest,
            "train_video_ids": panel.train_videos.tolist(),
            "pca_provenance": dict(panel.pca_provenance),
        },
        settings=params_to_training(params),
        hyperparameters=params_to_hyperparameters(params),
    )


def evaluate_search_trial(
    *,
    context: StudyContext,
    identity: Mapping[str, Any],
    panels: Sequence[PanelData],
    trial: Any,
    params: Mapping[str, Any],
) -> tuple[float, dict[str, float]]:
    panel_rows: list[dict[str, float]] = []
    frozen_digests: list[str] = []
    for panel in panels:
        member_logits: list[np.ndarray] = []
        ar_logits: list[np.ndarray] = []
        member_scores: list[float] = []
        ar_scores: list[float] = []
        labels = panel.test_event[panel.test_valid]
        for seed in SEARCH_SEEDS:
            panel_ownership = ownership(panel, seed, search=True)
            frozen_ar = load_search_ar(context, panel, seed, panel_ownership)
            frozen_digests.append(frozen_ar.identity_digest)
            residual = evaluate_residual(
                context=context,
                identity=identity,
                panel=panel,
                seed=seed,
                frozen_ar=frozen_ar,
                panel_ownership=panel_ownership,
                params=params,
                output_root=(
                    context.args.output_root
                    / "search"
                    / "trials"
                    / f"trial_{trial.number:04d}"
                    / f"outer_{panel.outer}"
                    / f"inner_{panel.inner}"
                    / f"seed_{seed}"
                ),
                stage="search",
                lane=f"trial_{trial.number}",
            )
            member_logit = residual.test_event_logit[panel.test_valid]
            ar_logit = frozen_ar.test_event_logit[panel.test_valid]
            member_logits.append(member_logit)
            ar_logits.append(ar_logit)
            member_scores.append(parity.pr_auc(labels, member_logit))
            ar_scores.append(parity.pr_auc(labels, ar_logit))
        ensemble_score = parity.pr_auc(labels, np.mean(np.stack(member_logits), axis=0))
        ar_ensemble_score = parity.pr_auc(labels, np.mean(np.stack(ar_logits), axis=0))
        panel_rows.append(
            {
                "outer_fold": float(panel.outer),
                "inner_fold": float(panel.inner),
                "ensemble_pr_auc": ensemble_score,
                "ar_ensemble_pr_auc": ar_ensemble_score,
                "ensemble_delta_vs_ar": ensemble_score - ar_ensemble_score,
                "member_mean_delta_vs_ar": float(np.mean(member_scores) - np.mean(ar_scores)),
                "ensemble_uplift": ensemble_score - float(np.mean(member_scores)),
            }
        )
    value, metrics = robust_objective(panel_rows)
    summary = {
        "trial": trial.number,
        "params": dict(params),
        "objective": value,
        "metrics": metrics,
        "panel_rows": panel_rows,
        "outer_test_scores_used": False,
        "completed_at": utc_now(),
    }
    summary_path = (
        context.args.output_root / "search" / "trials" / f"trial_{trial.number:04d}" / "summary.json"
    )
    runner.atomic_json(summary_path, summary)
    provenance = RunProvenance(
        git_commit=identity["git_commit"],
        dataset_manifest_sha256=identity["dataset_seal_digest"],
        split_manifest_sha256=identity["plan_digest"],
        feature_manifest_sha256=runner.canonical_digest(
            [panel.pca_provenance for panel in panels]
        ),
        target=parity.TARGET_NAME,
        architecture=RECIPE.name,
        validation_protocol="five_prespecified_inner_panels_three_seed_equal_ensemble",
        seed=SAMPLER_SEED,
        accelerator_backend="mlx",
        frozen_ar_sha256=runner.canonical_digest(frozen_digests),
        extra={f"param.{key}": value for key, value in params.items()},
    )
    with MLflowRun(
        tracking_uri=f"sqlite:///{(context.args.output_root / 'mlflow.db').resolve()}",
        experiment_name="veatic21-event-optuna-stabilization",
        run_name=f"trial-{trial.number:04d}",
        provenance=provenance,
        tags={"neural_bridge.stage": "search"},
        artifact_location=(context.args.output_root / "mlflow-artifacts").resolve().as_uri(),
    ) as run:
        run.log_metrics({"objective": value, **metrics})
        run.log_artifact(summary_path, artifact_path="search")
    trial.set_user_attr("neural_bridge.metrics", metrics)
    trial.set_user_attr("neural_bridge.panel_rows", panel_rows)
    return value, metrics


def completed_trials(study: Any, optuna: Any) -> list[Any]:
    return [
        trial
        for trial in study.get_trials(deepcopy=False)
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]


def run_search(
    context: StudyContext, identity: Mapping[str, Any], panels: Sequence[PanelData]
) -> dict[str, Any]:
    require_accelerator("mlx")
    optuna = require_upstream("optuna")
    sampler = optuna.samplers.TPESampler(
        seed=SAMPLER_SEED, n_startup_trials=10, multivariate=True
    )
    study = optuna.create_study(
        study_name="veatic21-event-optuna-stabilization",
        direction="maximize",
        sampler=sampler,
        storage=f"sqlite:///{(context.args.output_root / 'optuna.db').resolve()}",
        load_if_exists=True,
    )
    contract = {
        "run_identity_digest": identity["digest"],
        "scope": "inner_only_five_panel_three_seed_ensemble",
        "trial_count": TRIAL_COUNT,
        "outer_test_scores_used": False,
    }
    for key, value in contract.items():
        existing = study.user_attrs.get(key)
        if existing is not None and existing != value:
            raise StabilizationError(f"Optuna study contract drift: {key}")
        study.set_user_attr(key, value)
    if not study.trials:
        study.enqueue_trial(ORIGINAL_PARAMS)

    def objective(trial: Any) -> float:
        params = sample_parameters(trial)
        value, _ = evaluate_search_trial(
            context=context,
            identity=identity,
            panels=panels,
            trial=trial,
            params=params,
        )
        return value

    complete = completed_trials(study, optuna)
    if len(complete) > TRIAL_COUNT:
        raise StabilizationError("Optuna study exceeded preregistered trial count")
    remaining = TRIAL_COUNT - len(complete)
    if remaining:
        study.optimize(objective, n_trials=remaining, gc_after_trial=True)
    complete = completed_trials(study, optuna)
    if len(complete) != TRIAL_COUNT:
        raise StabilizationError(f"expected {TRIAL_COUNT} complete trials, got {len(complete)}")
    trial_rows = [
        {
            "number": trial.number,
            "value": float(trial.value),
            "params": dict(trial.params),
            "user_attrs": dict(trial.user_attrs),
        }
        for trial in complete
    ]
    runner.atomic_json(context.args.output_root / "search" / "completed_trials.json", trial_rows)
    frozen = {
        "schema_version": SCHEMA_VERSION,
        "trial_number": int(study.best_trial.number),
        "objective": float(study.best_value),
        "params": dict(study.best_params),
        "completed_trial_count": len(complete),
        "completed_study_digest": runner.canonical_digest(trial_rows),
        "run_identity_digest": identity["digest"],
        "preregistration_sha256": identity["preregistration_sha256"],
        "frozen_before_showdown": True,
        "frozen_at": utc_now(),
    }
    frozen_path = context.args.output_root / "frozen_candidate.json"
    if frozen_path.exists():
        previous = json.loads(frozen_path.read_text(encoding="utf-8"))
        comparable = {key: value for key, value in frozen.items() if key != "frozen_at"}
        old_comparable = {key: value for key, value in previous.items() if key != "frozen_at"}
        if comparable != old_comparable:
            raise StabilizationError("frozen candidate drift")
        frozen = previous
    else:
        runner.atomic_json(frozen_path, frozen)
    return frozen


def train_showdown_ar(
    context: StudyContext,
    identity: Mapping[str, Any],
    panel: PanelData,
    seed: int,
    panel_ownership: Any,
) -> parity.DualScores:
    return parity.fit_frozen_dual_ar(
        x_train=panel.train_ar_x,
        x_test=panel.test_ar_x,
        event=panel.train_event,
        continuous=panel.train_continuous,
        valid=panel.train_valid,
        videos=panel.train_videos,
        ownership=panel_ownership,
        seed=seed,
        output_root=(
            context.args.output_root
            / "showdown"
            / "ar"
            / f"outer_{panel.outer}"
            / f"inner_{panel.inner}"
            / f"seed_{seed}"
        ),
        identity_base={
            "run_identity_digest": identity["digest"],
            "stage": "showdown",
            "outer_fold": panel.outer,
            "inner_fold": panel.inner,
            "target": parity.TARGET_NAME,
        },
        settings=ORIGINAL_TRAINING,
    )


def comparison(
    ensemble_rows: Sequence[Mapping[str, Any]],
    member_rows: Sequence[Mapping[str, Any]],
    panels: Sequence[tuple[int, int]],
    left: str,
    right: str,
) -> dict[str, Any]:
    panel_set = set(panels)
    selected = [row for row in ensemble_rows if (row["outer_fold"], row["inner_fold"]) in panel_set]
    left_rows = {(row["outer_fold"], row["inner_fold"]): row for row in selected if row["lane"] == left}
    if right == "ar":
        differences = {
            key: float(row["pr_auc"] - row["ar_pr_auc"]) for key, row in left_rows.items()
        }
    else:
        right_rows = {
            (row["outer_fold"], row["inner_fold"]): row
            for row in selected
            if row["lane"] == right
        }
        differences = {
            key: float(left_rows[key]["pr_auc"] - right_rows[key]["pr_auc"])
            for key in left_rows
        }
    values = np.asarray(list(differences.values()), dtype=float)
    outer_means = [
        float(np.mean([value for (outer_fold, _), value in differences.items() if outer_fold == outer]))
        for outer in parity.OUTER_FOLDS
    ]
    left_members = {
        (row["outer_fold"], row["inner_fold"], row["seed"]): row
        for row in member_rows
        if row["lane"] == left and (row["outer_fold"], row["inner_fold"]) in panel_set
    }
    if right == "ar":
        member_differences = [row["pr_auc"] - row["ar_pr_auc"] for row in left_members.values()]
    else:
        right_members = {
            (row["outer_fold"], row["inner_fold"], row["seed"]): row
            for row in member_rows
            if row["lane"] == right and (row["outer_fold"], row["inner_fold"]) in panel_set
        }
        member_differences = [
            left_members[key]["pr_auc"] - right_members[key]["pr_auc"]
            for key in left_members
        ]
    member_values = np.asarray(member_differences, dtype=float)
    return {
        "left": left,
        "right": right,
        "panel_count": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "wins": int(np.sum(values > 0)),
        "std": float(np.std(values, ddof=0)),
        "outer_positive_means": int(np.sum(np.asarray(outer_means) > 0)),
        "outer_means": outer_means,
        "member_count": len(member_values),
        "member_mean": float(np.mean(member_values)),
        "member_wins": int(np.sum(member_values > 0)),
    }


def summarize_showdown(
    member_rows: Sequence[Mapping[str, Any]], ensemble_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    primary_tuned_original = comparison(
        ensemble_rows, member_rows, HELDOUT_PANELS, "tuned", "original"
    )
    primary_tuned_ar = comparison(
        ensemble_rows, member_rows, HELDOUT_PANELS, "tuned", "ar"
    )
    original_ar = comparison(
        ensemble_rows, member_rows, HELDOUT_PANELS, "original", "ar"
    )
    tuned_heldout = [
        row
        for row in ensemble_rows
        if row["lane"] == "tuned"
        and (row["outer_fold"], row["inner_fold"]) in set(HELDOUT_PANELS)
    ]
    checks = {
        "tuned_mean_exceeds_original": primary_tuned_original["mean"] > 0,
        "tuned_median_exceeds_original": primary_tuned_original["median"] > 0,
        "tuned_wins_at_least_7_of_10_vs_original": primary_tuned_original["wins"] >= 7,
        "tuned_outer_means_at_least_4_of_5_vs_original": primary_tuned_original["outer_positive_means"] >= 4,
        "tuned_mean_exceeds_ar": primary_tuned_ar["mean"] > 0,
        "tuned_median_exceeds_ar": primary_tuned_ar["median"] > 0,
        "tuned_wins_at_least_7_of_10_vs_ar": primary_tuned_ar["wins"] >= 7,
        "tuned_outer_means_at_least_4_of_5_vs_ar": primary_tuned_ar["outer_positive_means"] >= 4,
        "tuned_delta_variability_no_worse_than_original": primary_tuned_ar["std"] <= original_ar["std"],
        "tuned_mean_ensemble_uplift_positive": float(
            np.mean([row["ensemble_uplift_over_member_mean"] for row in tuned_heldout])
        ) > 0,
        "tuned_member_mean_exceeds_original": primary_tuned_original["member_mean"] > 0,
        "tuned_member_wins_at_least_30_of_50_vs_original": primary_tuned_original["member_wins"] >= 30,
        "tuned_member_mean_exceeds_ar": primary_tuned_ar["member_mean"] > 0,
        "tuned_member_wins_at_least_30_of_50_vs_ar": primary_tuned_ar["member_wins"] >= 30,
    }
    return {
        "primary_heldout_10": {
            "tuned_vs_original": primary_tuned_original,
            "tuned_vs_ar": primary_tuned_ar,
            "original_vs_ar": original_ar,
        },
        "secondary_all_15": {
            "tuned_vs_original": comparison(
                ensemble_rows, member_rows, ALL_PANELS, "tuned", "original"
            ),
            "tuned_vs_ar": comparison(
                ensemble_rows, member_rows, ALL_PANELS, "tuned", "ar"
            ),
            "original_vs_ar": comparison(
                ensemble_rows, member_rows, ALL_PANELS, "original", "ar"
            ),
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def audit_showdown(
    member_rows: Sequence[Mapping[str, Any]], ensemble_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected_members = {
        (outer, inner, lane, seed)
        for outer, inner in ALL_PANELS
        for lane in LANES
        for seed in SHOWDOWN_SEEDS
    }
    actual_members = {
        (row["outer_fold"], row["inner_fold"], row["lane"], row["seed"])
        for row in member_rows
    }
    expected_ensembles = {
        (outer, inner, lane) for outer, inner in ALL_PANELS for lane in LANES
    }
    actual_ensembles = {
        (row["outer_fold"], row["inner_fold"], row["lane"]) for row in ensemble_rows
    }
    checks = {
        "member_matrix_complete": actual_members == expected_members,
        "ensemble_matrix_complete": actual_ensembles == expected_ensembles,
        "outer_test_closed": all(row["outer_test_scores_used"] is False for row in [*member_rows, *ensemble_rows]),
        "finite_metrics": all(
            math.isfinite(float(row[key]))
            for row in [*member_rows, *ensemble_rows]
            for key in ("pr_auc", "ar_pr_auc", "delta_vs_ar")
        ),
        "frozen_ar_shared_across_lanes": all(
            len(
                {
                    row["ar_prediction_digest"]
                    for row in member_rows
                    if row["outer_fold"] == outer
                    and row["inner_fold"] == inner
                    and row["seed"] == seed
                }
            )
            == 1
            for outer, inner in ALL_PANELS
            for seed in SHOWDOWN_SEEDS
        ),
        "label_digest_aligned_across_lanes": all(
            len(
                {
                    row["heldout_label_digest"]
                    for row in ensemble_rows
                    if row["outer_fold"] == outer and row["inner_fold"] == inner
                }
            )
            == 1
            for outer, inner in ALL_PANELS
        ),
        "zero_event_policy_preserved": all(
            row["undefined_per_video_pr_auc_score_filled"] is False
            and row["zero_event_videos_excluded_from_pooled_negatives"] is False
            for row in [*member_rows, *ensemble_rows]
        ),
    }
    return {"checks": checks, "passed": all(checks.values())}


def run_showdown(
    context: StudyContext,
    identity: Mapping[str, Any],
    panels: Sequence[PanelData],
    frozen: Mapping[str, Any],
) -> dict[str, Any]:
    if not frozen.get("frozen_before_showdown") or frozen["run_identity_digest"] != identity["digest"]:
        raise StabilizationError("candidate was not frozen under current run identity")
    require_accelerator("mlx")
    member_rows: list[dict[str, Any]] = []
    predictions: dict[tuple[int, int, str, int], tuple[np.ndarray, np.ndarray]] = {}
    params_by_lane = {"tuned": frozen["params"], "original": ORIGINAL_PARAMS}
    for panel in panels:
        labels = panel.test_event[panel.test_valid]
        for seed in SHOWDOWN_SEEDS:
            panel_ownership = ownership(panel, seed, search=False)
            frozen_ar = train_showdown_ar(context, identity, panel, seed, panel_ownership)
            ar_logit = frozen_ar.test_event_logit[panel.test_valid]
            ar_score = parity.pr_auc(labels, ar_logit)
            for lane in LANES:
                residual = evaluate_residual(
                    context=context,
                    identity=identity,
                    panel=panel,
                    seed=seed,
                    frozen_ar=frozen_ar,
                    panel_ownership=panel_ownership,
                    params=params_by_lane[lane],
                    output_root=(
                        context.args.output_root
                        / "showdown"
                        / "models"
                        / f"outer_{panel.outer}"
                        / f"inner_{panel.inner}"
                        / lane
                        / f"seed_{seed}"
                    ),
                    stage="showdown",
                    lane=lane,
                )
                logit = residual.test_event_logit[panel.test_valid]
                score = parity.pr_auc(labels, logit)
                row = {
                    "outer_fold": panel.outer,
                    "inner_fold": panel.inner,
                    "lane": lane,
                    "seed": seed,
                    "pr_auc": score,
                    "ar_pr_auc": ar_score,
                    "delta_vs_ar": score - ar_score,
                    "residual_best_epoch": residual.best_epoch,
                    "residual_suppressed": residual.suppressed,
                    "frozen_ar_identity_digest": frozen_ar.identity_digest,
                    "residual_identity_digest": residual.identity_digest,
                    "prediction_digest": runner.array_digest(logit),
                    "ar_prediction_digest": runner.array_digest(ar_logit),
                    "event_threshold": panel.threshold,
                    **dict(panel.panel_stats),
                    **dict(panel.pca_audit),
                    "outer_test_scores_used": False,
                    "explicitly_nonpromotable": True,
                }
                member_rows.append(row)
                predictions[(panel.outer, panel.inner, lane, seed)] = (logit, ar_logit)
                runner.atomic_json(
                    context.args.output_root / "showdown" / "member_rows.partial.json",
                    member_rows,
                )
    runner.atomic_json(context.args.output_root / "showdown" / "member_rows.json", member_rows)
    ensemble_rows: list[dict[str, Any]] = []
    member_lookup = {
        (row["outer_fold"], row["inner_fold"], row["lane"], row["seed"]): row
        for row in member_rows
    }
    for panel in panels:
        labels = panel.test_event[panel.test_valid]
        for lane in LANES:
            bundles = [predictions[(panel.outer, panel.inner, lane, seed)] for seed in SHOWDOWN_SEEDS]
            logit = np.mean(np.stack([bundle[0] for bundle in bundles]), axis=0)
            ar_logit = np.mean(np.stack([bundle[1] for bundle in bundles]), axis=0)
            score = parity.pr_auc(labels, logit)
            ar_score = parity.pr_auc(labels, ar_logit)
            members = [member_lookup[(panel.outer, panel.inner, lane, seed)] for seed in SHOWDOWN_SEEDS]
            ensemble_rows.append(
                {
                    "outer_fold": panel.outer,
                    "inner_fold": panel.inner,
                    "lane": lane,
                    "member_seeds": list(SHOWDOWN_SEEDS),
                    "pr_auc": score,
                    "ar_pr_auc": ar_score,
                    "delta_vs_ar": score - ar_score,
                    "member_mean_pr_auc": float(np.mean([row["pr_auc"] for row in members])),
                    "ensemble_uplift_over_member_mean": score
                    - float(np.mean([row["pr_auc"] for row in members])),
                    "prediction_digest": runner.array_digest(logit),
                    "ar_prediction_digest": runner.array_digest(ar_logit),
                    **dict(panel.panel_stats),
                    **dict(panel.pca_audit),
                    "outer_test_scores_used": False,
                    "explicitly_nonpromotable": True,
                }
            )
    runner.atomic_json(context.args.output_root / "showdown" / "ensemble_rows.json", ensemble_rows)
    audit = audit_showdown(member_rows, ensemble_rows)
    summary = summarize_showdown(member_rows, ensemble_rows)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "run_identity_digest": identity["digest"],
        "frozen_candidate": dict(frozen),
        "member_rows": len(member_rows),
        "ensemble_rows": len(ensemble_rows),
        "audit": audit,
        "summary": summary,
        "passed": audit["passed"] and summary["passed"],
        "outer_test_scores_used": False,
        "explicitly_nonpromotable": True,
        "completed_at": utc_now(),
    }
    runner.atomic_json(context.args.output_root / "result.json", result)
    return result


def report_text(result: Mapping[str, Any], output_root: Path) -> str:
    primary = result["summary"]["primary_heldout_10"]
    secondary = result["summary"]["secondary_all_15"]
    return f"""# VEATIC 2.1 Event Optuna Stabilization

Status: `{'PASS' if result['passed'] else 'FAIL'}`  
Run identity: `{result['run_identity_digest']}`  
Output root: `{output_root.resolve()}`  
Outer-test scores used: `false`

## Frozen Optuna candidate

- trials: `{result['frozen_candidate']['completed_trial_count']}`
- best trial: `{result['frozen_candidate']['trial_number']}`
- development objective: `{result['frozen_candidate']['objective']:.10f}`
- parameters: `{json.dumps(result['frozen_candidate']['params'], sort_keys=True)}`

## Primary held-back 10 panels

- tuned minus original: mean `{primary['tuned_vs_original']['mean']:+.10f}`, median `{primary['tuned_vs_original']['median']:+.10f}`, wins `{primary['tuned_vs_original']['wins']}/10`, positive outer means `{primary['tuned_vs_original']['outer_positive_means']}/5`;
- tuned minus AR: mean `{primary['tuned_vs_ar']['mean']:+.10f}`, median `{primary['tuned_vs_ar']['median']:+.10f}`, wins `{primary['tuned_vs_ar']['wins']}/10`, positive outer means `{primary['tuned_vs_ar']['outer_positive_means']}/5`;
- original minus AR: mean `{primary['original_vs_ar']['mean']:+.10f}`, median `{primary['original_vs_ar']['median']:+.10f}`.

## Secondary all 15 panels

- tuned minus original: mean `{secondary['tuned_vs_original']['mean']:+.10f}`, median `{secondary['tuned_vs_original']['median']:+.10f}`, wins `{secondary['tuned_vs_original']['wins']}/15`;
- tuned minus AR: mean `{secondary['tuned_vs_ar']['mean']:+.10f}`, median `{secondary['tuned_vs_ar']['median']:+.10f}`, wins `{secondary['tuned_vs_ar']['wins']}/15`.

## Gate and audit

- gate checks: `{json.dumps(result['summary']['checks'], sort_keys=True)}`
- audit: `{json.dumps(result['audit']['checks'], sort_keys=True)}`

This is inner-only exploratory evidence. It does not authorize outer confirmation.
"""


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.dry_run and args.audit_only:
        raise StabilizationError("--dry-run and --audit-only are mutually exclusive")
    if args.dry_run:
        return {"status": "dry_run", **dry_schedule()}
    context = build_context(args)
    identity = initialize_output(context)
    if args.audit_only:
        member_rows = json.loads(
            (args.output_root / "showdown/member_rows.json").read_text(encoding="utf-8")
        )
        ensemble_rows = json.loads(
            (args.output_root / "showdown/ensemble_rows.json").read_text(encoding="utf-8")
        )
        return {
            "status": "audited",
            "audit": audit_showdown(member_rows, ensemble_rows),
            "summary": summarize_showdown(member_rows, ensemble_rows),
        }
    modeling.require_mlx_gpu()
    search_panels = [prepare_panel(context, *panel) for panel in SEARCH_PANELS]
    frozen_path = args.output_root / "frozen_candidate.json"
    if args.stage in {"search", "all"}:
        frozen = run_search(context, identity, search_panels)
    else:
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if args.stage == "search":
        return {"status": "search_complete", "frozen_candidate": frozen}
    panels = [prepare_panel(context, *panel) for panel in ALL_PANELS]
    result = run_showdown(context, identity, panels, frozen)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    report = args.reports_dir / "veatic21_event_optuna_stabilization_20260717.md"
    report.write_text(report_text(result, args.output_root), encoding="utf-8")
    return {**result, "report": str(report)}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except Exception as exc:
        print(f"VEATIC event Optuna stabilization failed: {exc}", file=os.sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
