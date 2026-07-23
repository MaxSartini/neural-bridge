"""VEATIC 2.1 foundation commands."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from .contracts import AROUSAL_SPIKE_1_3S, CandidateSpec, CellSpec
from .controls import build_control_plan, run_control_program, write_control_plan
from .data import CanonicalSubstrate
from .evidence import atomic_write_json, digest_json, load_json, sha256_file
from .head_training import (
    build_head_family_screen,
    run_head_family_screen,
    select_head_family,
    write_head_family_screen,
    write_head_family_selection,
)
from .pca_cache import fit_event_pca_cache, load_event_pca_scaler
from .preregistration import (
    benchmark_partition_mask,
    build_event_preregistration,
    calibrate_event_preregistration,
)
from .recipe_resolution import (
    resolve_stopped_training_recipe,
    write_training_recipe_resolution,
)
from .runner import run_confirmation_cell, verify_confirmation_cell
from .stability import build_stability_plan, run_stability_program, write_stability_plan
from .stage1 import (
    Stage1CellConfig,
    build_stage1_plan,
    probe_stage1_capacity,
    run_stage1_ar_benchmark,
    run_stage1_discovery_cell,
    write_stage1_plan,
)
from .stage2 import (
    build_stage2_pca_screen,
    run_stage2_pca_screen,
    select_stage2_pca_width,
    write_stage2_pca_screen,
    write_stage2_pca_selection,
)
from .supervised_projection import (
    build_supervised_projection_screen,
    probe_supervised_projection_capacity,
    run_supervised_projection_screen,
    select_supervised_projection,
    write_supervised_projection_screen,
    write_supervised_projection_selection,
)
from .training_recipe import (
    build_training_recipe_plan,
    run_training_recipe_program,
    write_training_recipe_plan,
)

_ARTIFACT_ROOT = Path("/Volumes/onn. Drive/Neural Bridge Artifacts")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m neural_bridge.veatic21")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke", help="run one non-promotable sealed foundation cell")
    smoke.add_argument("--output", type=Path)
    smoke.add_argument("--fold", type=int, default=0)
    smoke.add_argument("--seed", type=int, default=20_260_721)
    preregister = subparsers.add_parser(
        "preregister-event",
        help="freeze the VEATIC-only event discovery matrix without opening labels",
    )
    preregister.add_argument("--output", type=Path)
    calibrate = subparsers.add_parser(
        "calibrate-event",
        help="derive event targets from benchmark-train labels while benchmark test remains sealed",
    )
    calibrate.add_argument("--preregistration", type=Path, required=True)
    calibrate.add_argument("--output", type=Path)
    pca = subparsers.add_parser(
        "fit-event-pca",
        help="fit reusable label-blind cortical PCA bases inside benchmark-train folds",
    )
    pca.add_argument("--preregistration", type=Path, required=True)
    pca.add_argument("--output", type=Path)
    pca.add_argument("--fold", action="append", type=int, choices=range(5))
    stage1 = subparsers.add_parser(
        "prepare-stage1",
        help="probe local capacity and seal the executable Stage-1 child plan",
    )
    stage1.add_argument("--preregistration", type=Path)
    stage1.add_argument("--calibration", type=Path)
    stage1.add_argument("--pca-manifest", type=Path)
    stage1.add_argument("--ar-benchmark", type=Path)
    stage1.add_argument("--output", type=Path)
    ar_benchmark = subparsers.add_parser(
        "benchmark-stage1-ar",
        help="benchmark fresh AR across every registered target, fold, and comparison seed",
    )
    ar_benchmark.add_argument("--preregistration", type=Path)
    ar_benchmark.add_argument("--calibration", type=Path)
    ar_benchmark.add_argument("--output", type=Path)
    prepare_stage2 = subparsers.add_parser(
        "prepare-stage2-pca",
        help="seal the train-only target shortlist and fixed-PCA screening matrix",
    )
    prepare_stage2.add_argument("--ar-benchmark", type=Path)
    prepare_stage2.add_argument("--plan", type=Path)
    prepare_stage2.add_argument("--executor-request", type=Path)
    prepare_stage2.add_argument("--output", type=Path)
    select_stage2 = subparsers.add_parser(
        "select-stage2-pca",
        help="select and seal the fixed-PCA width after the complete registered screen",
    )
    select_stage2.add_argument("--summary", type=Path)
    select_stage2.add_argument("--screen", type=Path)
    select_stage2.add_argument("--output", type=Path)
    stage2 = subparsers.add_parser(
        "benchmark-stage2-pca",
        help="run the complete resumable fixed-PCA screen without opening the sealed tail",
    )
    stage2.add_argument("--preregistration", type=Path)
    stage2.add_argument("--calibration", type=Path)
    stage2.add_argument("--pca-manifest", type=Path)
    stage2.add_argument("--plan", type=Path)
    stage2.add_argument("--screen", type=Path)
    stage2.add_argument("--output", type=Path)
    prepare_supervised = subparsers.add_parser(
        "prepare-supervised-projection",
        help="seal the matched PCA-512 versus supervised-bottleneck matrix",
    )
    prepare_supervised.add_argument("--pca-selection", type=Path)
    prepare_supervised.add_argument("--pca-summary", type=Path)
    prepare_supervised.add_argument("--pca-screen", type=Path)
    prepare_supervised.add_argument("--plan", type=Path)
    prepare_supervised.add_argument("--pca-manifest", type=Path)
    prepare_supervised.add_argument("--output", type=Path)
    supervised = subparsers.add_parser(
        "benchmark-supervised-projection",
        help="run the sequential matched representation screen without opening the tail",
    )
    supervised.add_argument("--preregistration", type=Path)
    supervised.add_argument("--calibration", type=Path)
    supervised.add_argument("--pca-manifest", type=Path)
    supervised.add_argument("--plan", type=Path)
    supervised.add_argument("--screen", type=Path)
    supervised.add_argument("--output", type=Path)
    select_supervised = subparsers.add_parser(
        "select-supervised-projection",
        help="seal the representation decision after the complete matched screen",
    )
    select_supervised.add_argument("--summary", type=Path)
    select_supervised.add_argument("--screen", type=Path)
    select_supervised.add_argument("--output", type=Path)
    prepare_heads = subparsers.add_parser(
        "prepare-head-family",
        help="seal the matched causal versus gated PCA-512 head-family matrix",
    )
    prepare_heads.add_argument("--representation-selection", type=Path)
    prepare_heads.add_argument("--representation-summary", type=Path)
    prepare_heads.add_argument("--representation-screen", type=Path)
    prepare_heads.add_argument("--plan", type=Path)
    prepare_heads.add_argument("--pca-manifest", type=Path)
    prepare_heads.add_argument("--output", type=Path)
    heads = subparsers.add_parser(
        "benchmark-head-family",
        help="run the sequential gated-head screen against verified causal evidence",
    )
    heads.add_argument("--preregistration", type=Path)
    heads.add_argument("--calibration", type=Path)
    heads.add_argument("--pca-manifest", type=Path)
    heads.add_argument("--plan", type=Path)
    heads.add_argument("--screen", type=Path)
    heads.add_argument("--output", type=Path)
    select_heads = subparsers.add_parser(
        "select-head-family",
        help="seal the paired causal-versus-gated head-family decision",
    )
    select_heads.add_argument("--summary", type=Path)
    select_heads.add_argument("--screen", type=Path)
    select_heads.add_argument("--baseline-summary", type=Path)
    select_heads.add_argument("--output", type=Path)
    prepare_recipe = subparsers.add_parser(
        "prepare-training-recipe",
        help="seal the staged VEATIC-only numeric training-recipe gates",
    )
    prepare_recipe.add_argument("--plan", type=Path)
    prepare_recipe.add_argument("--representation-summary", type=Path)
    prepare_recipe.add_argument("--representation-selection", type=Path)
    prepare_recipe.add_argument("--head-screen", type=Path)
    prepare_recipe.add_argument("--head-selection", type=Path)
    prepare_recipe.add_argument("--pca-manifest", type=Path)
    prepare_recipe.add_argument("--output", type=Path)
    recipe = subparsers.add_parser(
        "benchmark-training-recipe",
        help="run all staged recipe gates with one sequential MLX worker",
    )
    recipe.add_argument("--preregistration", type=Path)
    recipe.add_argument("--calibration", type=Path)
    recipe.add_argument("--pca-manifest", type=Path)
    recipe.add_argument("--plan", type=Path)
    recipe.add_argument("--output", type=Path)
    resolve_recipe = subparsers.add_parser(
        "resolve-training-recipe",
        help="seal a conservative resolution of the stopped numeric sweep",
    )
    resolve_recipe.add_argument("--plan", type=Path)
    resolve_recipe.add_argument("--baseline-summary", type=Path)
    resolve_recipe.add_argument("--run-root", type=Path)
    resolve_recipe.add_argument("--output", type=Path)
    prepare_stability = subparsers.add_parser(
        "prepare-stability",
        help="seal the fixed nine-seed stability panel for the retained recipe",
    )
    prepare_stability.add_argument("--preregistration", type=Path)
    prepare_stability.add_argument("--recipe-plan", type=Path)
    prepare_stability.add_argument("--recipe-selection", type=Path)
    prepare_stability.add_argument("--pca-manifest", type=Path)
    prepare_stability.add_argument("--output", type=Path)
    stability = subparsers.add_parser(
        "benchmark-stability",
        help="run the retained recipe over the fixed stability seeds with one MLX worker",
    )
    stability.add_argument("--preregistration", type=Path)
    stability.add_argument("--calibration", type=Path)
    stability.add_argument("--pca-manifest", type=Path)
    stability.add_argument("--plan", type=Path)
    stability.add_argument("--output", type=Path)
    prepare_controls = subparsers.add_parser(
        "prepare-controls",
        help="seal the lifecycle-complete matched control panel before stability resumes",
    )
    prepare_controls.add_argument("--preregistration", type=Path)
    prepare_controls.add_argument("--recipe-plan", type=Path)
    prepare_controls.add_argument("--recipe-selection", type=Path)
    prepare_controls.add_argument("--baseline-summary", type=Path)
    prepare_controls.add_argument("--pca-manifest", type=Path)
    prepare_controls.add_argument("--crosswalk", type=Path)
    prepare_controls.add_argument("--output", type=Path)
    controls = subparsers.add_parser(
        "benchmark-controls",
        help="run all matched controls with one MLX worker before further stability",
    )
    controls.add_argument("--preregistration", type=Path)
    controls.add_argument("--calibration", type=Path)
    controls.add_argument("--pca-manifest", type=Path)
    controls.add_argument("--plan", type=Path)
    controls.add_argument("--baseline-summary", type=Path)
    controls.add_argument("--baseline-root", type=Path)
    controls.add_argument("--output", type=Path)
    cell = subparsers.add_parser(
        "run-stage1-cell",
        help="train one VEATIC-only learned spike discovery cell without opening the sealed tail",
    )
    cell.add_argument("--target", required=True)
    cell.add_argument("--fold", type=int, required=True, choices=range(5))
    cell.add_argument("--seed", type=int, required=True)
    cell.add_argument("--pca-width", type=int, required=True, choices=(64, 128, 256, 512))
    cell.add_argument(
        "--head-family",
        required=True,
        choices=(
            "frozen_ar_plus_causal_temporal_residual",
            "frozen_ar_plus_gated_multiscale_temporal_residual",
        ),
    )
    cell.add_argument("--hidden-width", type=int, required=True)
    cell.add_argument("--learning-rate", type=float, required=True)
    cell.add_argument("--weight-decay", type=float, required=True)
    cell.add_argument("--residual-logit-cap", type=float, required=True)
    cell.add_argument("--batch-rows", type=int, required=True)
    cell.add_argument("--preregistration", type=Path)
    cell.add_argument("--calibration", type=Path)
    cell.add_argument("--pca-manifest", type=Path)
    cell.add_argument("--plan", type=Path)
    cell.add_argument("--output", type=Path)
    return parser


def _default_output(repo_root: Path) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return _ARTIFACT_ROOT / "runs/veatic-2.1/foundation-smoke" / stamp


def _default_preregistration_output(repo_root: Path) -> Path:
    return _ARTIFACT_ROOT / "preregistrations/veatic-2.1/event-spike-v1.json"


def _default_calibration_output(repo_root: Path) -> Path:
    return _ARTIFACT_ROOT / "preregistrations/veatic-2.1/event-spike-v1-calibration.json"


def _default_pca_output(repo_root: Path) -> Path:
    return _ARTIFACT_ROOT / "features/veatic-2.1/neural-bridge/cortical-pca-v1"


def _default_stage1_output(repo_root: Path) -> Path:
    return _ARTIFACT_ROOT / "preregistrations/veatic-2.1/stage1-child-plan.json"


def _default_ar_benchmark_output() -> Path:
    return _ARTIFACT_ROOT / "runs/veatic-2.1/stage1-ar-benchmark"


def _default_stage2_screen_output() -> Path:
    return _ARTIFACT_ROOT / "preregistrations/veatic-2.1/stage2-pca-screen.json"


def _default_stage2_run_output() -> Path:
    return _ARTIFACT_ROOT / "runs/veatic-2.1/stage2-pca-screen"


def _default_stage2_selection_output() -> Path:
    return _ARTIFACT_ROOT / "preregistrations/veatic-2.1/stage2-pca-selection.json"


def _default_supervised_screen_output() -> Path:
    return _ARTIFACT_ROOT / "preregistrations/veatic-2.1/supervised-projection-screen.json"


def _default_supervised_run_output() -> Path:
    return _ARTIFACT_ROOT / "runs/veatic-2.1/supervised-projection-screen"


def _default_supervised_selection_output() -> Path:
    return _ARTIFACT_ROOT / "preregistrations/veatic-2.1/supervised-projection-selection.json"


def _default_head_screen_output() -> Path:
    return _ARTIFACT_ROOT / "preregistrations/veatic-2.1/head-family-screen.json"


def _default_head_run_output() -> Path:
    return _ARTIFACT_ROOT / "runs/veatic-2.1/head-family-screen"


def _default_head_selection_output() -> Path:
    return _ARTIFACT_ROOT / "preregistrations/veatic-2.1/head-family-selection.json"


def _default_training_recipe_plan_output() -> Path:
    return _ARTIFACT_ROOT / "preregistrations/veatic-2.1/training-recipe-plan.json"


def _default_training_recipe_run_output() -> Path:
    return _ARTIFACT_ROOT / "runs/veatic-2.1/training-recipe"


def _default_training_recipe_selection_output() -> Path:
    return _ARTIFACT_ROOT / "preregistrations/veatic-2.1/training-recipe-selection.json"


def _default_stability_plan_output() -> Path:
    return _ARTIFACT_ROOT / "preregistrations/veatic-2.1/stability-plan.json"


def _default_stability_run_output() -> Path:
    return _ARTIFACT_ROOT / "runs/veatic-2.1/stability"


def _default_control_plan_output() -> Path:
    return _ARTIFACT_ROOT / "preregistrations/veatic-2.1/control-plan.json"


def _default_control_run_output() -> Path:
    return _ARTIFACT_ROOT / "runs/veatic-2.1/matched-controls"


def _default_lifecycle_control_crosswalk() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "internal/active/veatic21-lifecycle-control-crosswalk.md"
    )


def _default_executor_validation_request() -> Path:
    return _ARTIFACT_ROOT / "runs/veatic-2.1/stage1-executor-validation/cell-001/request.json"


def _default_stage1_cell_output(config: Stage1CellConfig) -> Path:
    config_sha256 = digest_json(
        {
            name: list(value) if isinstance(value, tuple) else value
            for name, value in config.__dict__.items()
        }
    )
    return (
        _ARTIFACT_ROOT
        / "runs/veatic-2.1/stage1-discovery"
        / config.target_name
        / f"fold-{config.fold}"
        / f"seed-{config.seed}"
        / f"{config.head_family}__pca-{config.pca_width}__{config_sha256[:16]}"
    )


def _owned_rows(features, mask) -> dict[str, list[int]]:
    rows = {video: [] for video in sorted(set(features.video_id.astype(str)), key=int)}
    for video, row, owned in zip(features.video_id, features.row_index, mask, strict=True):
        if owned:
            rows[str(video)].append(int(row))
    return rows


def _prepare_stage1(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[3]
    preregistration_path = args.preregistration or _default_preregistration_output(repo_root)
    calibration_path = args.calibration or _default_calibration_output(repo_root)
    pca_manifest_path = args.pca_manifest or (_default_pca_output(repo_root) / "manifest.json")
    output = (args.output or _default_stage1_output(repo_root)).expanduser().resolve()
    preregistration = load_json(preregistration_path.expanduser().resolve())
    calibration = load_json(calibration_path.expanduser().resolve())
    pca_manifest = load_json(pca_manifest_path.expanduser().resolve())
    ar_benchmark_path = args.ar_benchmark or (_default_ar_benchmark_output() / "summary.json")
    ar_benchmark = (
        load_json(ar_benchmark_path.expanduser().resolve())
        if ar_benchmark_path.expanduser().resolve().is_file()
        else None
    )
    capacity = probe_stage1_capacity(pca_manifest)
    plan = build_stage1_plan(
        preregistration,
        calibration,
        pca_manifest,
        capacity,
        ar_benchmark,
    )
    write_stage1_plan(output, plan)
    print(
        json.dumps(
            {
                "output": str(output),
                "plan_sha256": plan["plan_sha256"],
                "purpose": plan["purpose"],
                "schema": plan["schema"],
            },
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    args = _parser().parse_args()
    if args.command == "prepare-stage1":
        return _prepare_stage1(args)
    if args.command == "prepare-stage2-pca":
        ar_path = (
            (args.ar_benchmark or (_default_ar_benchmark_output() / "summary.json"))
            .expanduser()
            .resolve()
        )
        plan_path = (args.plan or _default_stage1_output(Path.cwd())).expanduser().resolve()
        executor_path = (
            (args.executor_request or _default_executor_validation_request()).expanduser().resolve()
        )
        output = (args.output or _default_stage2_screen_output()).expanduser().resolve()
        screen = build_stage2_pca_screen(
            load_json(ar_path),
            load_json(plan_path),
            load_json(executor_path),
            executor_request_sha256=sha256_file(executor_path),
        )
        write_stage2_pca_screen(output, screen)
        print(
            json.dumps(
                {
                    "benchmark_test_labels_accessed": False,
                    "expected_cells": screen["matrix"]["expected_cells"],
                    "output": str(output),
                    "screen_sha256": screen["screen_sha256"],
                    "selected_targets": screen["target_shortlist"]["selected_targets"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "select-stage2-pca":
        summary_path = (
            (args.summary or (_default_stage2_run_output() / "summary.json")).expanduser().resolve()
        )
        screen_path = (args.screen or _default_stage2_screen_output()).expanduser().resolve()
        output = (args.output or _default_stage2_selection_output()).expanduser().resolve()
        selection = select_stage2_pca_width(
            load_json(summary_path),
            load_json(screen_path),
        )
        write_stage2_pca_selection(output, selection)
        print(
            json.dumps(
                {
                    "benchmark_test_labels_accessed": False,
                    "output": str(output),
                    "selected_pca_width": selection["selected_pca_width"],
                    "selection_sha256": selection["selection_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "prepare-supervised-projection":
        pca_selection_path = (
            (args.pca_selection or _default_stage2_selection_output()).expanduser().resolve()
        )
        pca_summary_path = (
            (args.pca_summary or (_default_stage2_run_output() / "summary.json"))
            .expanduser()
            .resolve()
        )
        pca_screen_path = (
            (args.pca_screen or _default_stage2_screen_output()).expanduser().resolve()
        )
        plan_path = (args.plan or _default_stage1_output(Path.cwd())).expanduser().resolve()
        pca_root = _default_pca_output(Path.cwd())
        pca_manifest_path = (
            (args.pca_manifest or (pca_root / "manifest.json")).expanduser().resolve()
        )
        preregistration = load_json(_default_preregistration_output(Path.cwd()))
        pca_manifest = load_json(pca_manifest_path)
        source_mean, _, _ = load_event_pca_scaler(preregistration, pca_manifest, pca_root, fold=0)
        capacity = probe_supervised_projection_capacity(source_width=len(source_mean))
        screen = build_supervised_projection_screen(
            load_json(pca_selection_path),
            load_json(pca_summary_path),
            load_json(pca_screen_path),
            load_json(plan_path),
            pca_manifest,
            capacity,
        )
        output = (args.output or _default_supervised_screen_output()).expanduser().resolve()
        write_supervised_projection_screen(output, screen)
        print(
            json.dumps(
                {
                    "benchmark_test_labels_accessed": False,
                    "batch_rows": screen["matched_recipe"]["batch_rows"],
                    "expected_cells": screen["matrix"]["expected_cells"],
                    "output": str(output),
                    "screen_sha256": screen["screen_sha256"],
                    "worker_count": 1,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "select-supervised-projection":
        summary_path = (
            (args.summary or (_default_supervised_run_output() / "summary.json"))
            .expanduser()
            .resolve()
        )
        screen_path = (args.screen or _default_supervised_screen_output()).expanduser().resolve()
        output = (args.output or _default_supervised_selection_output()).expanduser().resolve()
        selection = select_supervised_projection(load_json(summary_path), load_json(screen_path))
        write_supervised_projection_selection(output, selection)
        print(
            json.dumps(
                {
                    "benchmark_test_labels_accessed": False,
                    "output": str(output),
                    "selected_representation": selection["selected_representation"],
                    "selection_sha256": selection["selection_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "prepare-head-family":
        representation_selection_path = (
            (args.representation_selection or _default_supervised_selection_output())
            .expanduser()
            .resolve()
        )
        representation_summary_path = (
            (args.representation_summary or (_default_supervised_run_output() / "summary.json"))
            .expanduser()
            .resolve()
        )
        representation_screen_path = (
            (args.representation_screen or _default_supervised_screen_output())
            .expanduser()
            .resolve()
        )
        plan_path = (args.plan or _default_stage1_output(Path.cwd())).expanduser().resolve()
        pca_root = _default_pca_output(Path.cwd())
        pca_manifest_path = (
            (args.pca_manifest or (pca_root / "manifest.json")).expanduser().resolve()
        )
        screen = build_head_family_screen(
            load_json(representation_selection_path),
            load_json(representation_summary_path),
            load_json(representation_screen_path),
            load_json(plan_path),
            load_json(pca_manifest_path),
        )
        output = (args.output or _default_head_screen_output()).expanduser().resolve()
        write_head_family_screen(output, screen)
        print(
            json.dumps(
                {
                    "benchmark_test_labels_accessed": False,
                    "expected_candidate_cells": screen["matrix"]["expected_candidate_cells"],
                    "output": str(output),
                    "screen_sha256": screen["screen_sha256"],
                    "worker_count": 1,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "select-head-family":
        summary_path = (
            (args.summary or (_default_head_run_output() / "summary.json")).expanduser().resolve()
        )
        screen_path = (args.screen or _default_head_screen_output()).expanduser().resolve()
        baseline_path = (
            (args.baseline_summary or (_default_supervised_run_output() / "summary.json"))
            .expanduser()
            .resolve()
        )
        selection = select_head_family(
            load_json(summary_path), load_json(screen_path), load_json(baseline_path)
        )
        output = (args.output or _default_head_selection_output()).expanduser().resolve()
        write_head_family_selection(output, selection)
        print(
            json.dumps(
                {
                    "benchmark_test_labels_accessed": False,
                    "output": str(output),
                    "selected_head_family": selection["selected_head_family"],
                    "selection_sha256": selection["selection_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "prepare-training-recipe":
        pca_root = _default_pca_output(Path.cwd())
        plan = build_training_recipe_plan(
            load_json((args.plan or _default_stage1_output(Path.cwd())).expanduser().resolve()),
            load_json(
                (
                    args.representation_summary
                    or (_default_supervised_run_output() / "summary.json")
                )
                .expanduser()
                .resolve()
            ),
            load_json(
                (args.representation_selection or _default_supervised_selection_output())
                .expanduser()
                .resolve()
            ),
            load_json((args.head_screen or _default_head_screen_output()).expanduser().resolve()),
            load_json(
                (args.head_selection or _default_head_selection_output()).expanduser().resolve()
            ),
            load_json(
                (args.pca_manifest or (pca_root / "manifest.json")).expanduser().resolve()
            ),
        )
        output = (args.output or _default_training_recipe_plan_output()).expanduser().resolve()
        write_training_recipe_plan(output, plan)
        print(
            json.dumps(
                {
                    "backend": "mlx",
                    "benchmark_test_labels_accessed": False,
                    "expected_new_cells": plan["matrix"]["expected_new_cells"],
                    "output": str(output),
                    "plan_sha256": plan["plan_sha256"],
                    "worker_count": 1,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "resolve-training-recipe":
        resolution = resolve_stopped_training_recipe(
            load_json(
                (args.plan or _default_training_recipe_plan_output()).expanduser().resolve()
            ),
            load_json(
                (args.baseline_summary or (_default_supervised_run_output() / "summary.json"))
                .expanduser()
                .resolve()
            ),
            (args.run_root or _default_training_recipe_run_output()).expanduser().resolve(),
        )
        output = (args.output or _default_training_recipe_selection_output()).expanduser().resolve()
        write_training_recipe_resolution(output, resolution)
        print(
            json.dumps(
                {
                    "benchmark_test_labels_accessed": False,
                    "completed_new_cells": resolution["completed_new_cells"],
                    "output": str(output),
                    "resolution_sha256": resolution["resolution_sha256"],
                    "selected_hidden_width": resolution["selected_recipe"]["hidden_width"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "prepare-stability":
        pca_root = _default_pca_output(Path.cwd())
        plan = build_stability_plan(
            load_json(
                (args.preregistration or _default_preregistration_output(Path.cwd()))
                .expanduser()
                .resolve()
            ),
            load_json(
                (args.recipe_plan or _default_training_recipe_plan_output())
                .expanduser()
                .resolve()
            ),
            load_json(
                (args.recipe_selection or _default_training_recipe_selection_output())
                .expanduser()
                .resolve()
            ),
            load_json(
                (args.pca_manifest or (pca_root / "manifest.json")).expanduser().resolve()
            ),
        )
        output = (args.output or _default_stability_plan_output()).expanduser().resolve()
        write_stability_plan(output, plan)
        print(
            json.dumps(
                {
                    "backend": "mlx",
                    "benchmark_test_labels_accessed": False,
                    "expected_cells": plan["matrix"]["expected_cells"],
                    "output": str(output),
                    "plan_sha256": plan["plan_sha256"],
                    "worker_count": 1,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "prepare-controls":
        pca_root = _default_pca_output(Path.cwd())
        plan = build_control_plan(
            load_json(
                (args.preregistration or _default_preregistration_output(Path.cwd()))
                .expanduser()
                .resolve()
            ),
            load_json(
                (args.recipe_plan or _default_training_recipe_plan_output())
                .expanduser()
                .resolve()
            ),
            load_json(
                (args.recipe_selection or _default_training_recipe_selection_output())
                .expanduser()
                .resolve()
            ),
            load_json(
                (args.baseline_summary or (_default_supervised_run_output() / "summary.json"))
                .expanduser()
                .resolve()
            ),
            load_json(
                (args.pca_manifest or (pca_root / "manifest.json")).expanduser().resolve()
            ),
            (args.crosswalk or _default_lifecycle_control_crosswalk()).expanduser().resolve(),
        )
        output = (args.output or _default_control_plan_output()).expanduser().resolve()
        write_control_plan(output, plan)
        print(
            json.dumps(
                {
                    "backend": "mlx",
                    "benchmark_test_labels_accessed": False,
                    "expected_new_cells": plan["matrix"]["expected_new_cells"],
                    "output": str(output),
                    "plan_sha256": plan["plan_sha256"],
                    "worker_count": 1,
                },
                sort_keys=True,
            )
        )
        return 0
    substrate = CanonicalSubstrate.from_repo()
    if args.command == "benchmark-stage1-ar":
        preregistration = load_json(
            (args.preregistration or _default_preregistration_output(substrate.repo_root))
            .expanduser()
            .resolve()
        )
        calibration = load_json(
            (args.calibration or _default_calibration_output(substrate.repo_root))
            .expanduser()
            .resolve()
        )
        output = (args.output or _default_ar_benchmark_output()).expanduser().resolve()
        summary = run_stage1_ar_benchmark(
            substrate,
            preregistration,
            calibration,
            output,
        )
        print(
            json.dumps(
                {
                    "benchmark_test_labels_accessed": False,
                    "completed_cells": summary["completed_cells"],
                    "invalid_cells": summary["invalid_cells"],
                    "output": str(output),
                    "target_count": summary["target_count"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "benchmark-stage2-pca":
        preregistration = load_json(
            (args.preregistration or _default_preregistration_output(substrate.repo_root))
            .expanduser()
            .resolve()
        )
        calibration = load_json(
            (args.calibration or _default_calibration_output(substrate.repo_root))
            .expanduser()
            .resolve()
        )
        pca_root = _default_pca_output(substrate.repo_root)
        pca_manifest = load_json(
            (args.pca_manifest or (pca_root / "manifest.json")).expanduser().resolve()
        )
        plan = load_json(
            (args.plan or _default_stage1_output(substrate.repo_root)).expanduser().resolve()
        )
        screen = load_json((args.screen or _default_stage2_screen_output()).expanduser().resolve())
        output = (args.output or _default_stage2_run_output()).expanduser().resolve()
        summary = run_stage2_pca_screen(
            substrate,
            preregistration,
            calibration,
            pca_manifest,
            plan,
            screen,
            pca_root,
            output,
            progress=lambda record: print(json.dumps(record, sort_keys=True), flush=True),
        )
        print(
            json.dumps(
                {
                    "benchmark_test_labels_accessed": False,
                    "completed_cells": summary["completed_cells"],
                    "expected_cells": summary["expected_cells"],
                    "output": str(output),
                    "summary_sha256": summary["summary_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "benchmark-supervised-projection":
        preregistration = load_json(
            (args.preregistration or _default_preregistration_output(substrate.repo_root))
            .expanduser()
            .resolve()
        )
        calibration = load_json(
            (args.calibration or _default_calibration_output(substrate.repo_root))
            .expanduser()
            .resolve()
        )
        pca_root = _default_pca_output(substrate.repo_root)
        pca_manifest = load_json(
            (args.pca_manifest or (pca_root / "manifest.json")).expanduser().resolve()
        )
        plan = load_json(
            (args.plan or _default_stage1_output(substrate.repo_root)).expanduser().resolve()
        )
        screen = load_json(
            (args.screen or _default_supervised_screen_output()).expanduser().resolve()
        )
        output = (args.output or _default_supervised_run_output()).expanduser().resolve()
        summary = run_supervised_projection_screen(
            substrate,
            preregistration,
            calibration,
            pca_manifest,
            plan,
            screen,
            pca_root,
            output,
            progress=lambda record: print(json.dumps(record, sort_keys=True), flush=True),
        )
        print(
            json.dumps(
                {
                    "benchmark_test_labels_accessed": False,
                    "completed_cells": summary["completed_cells"],
                    "expected_cells": summary["expected_cells"],
                    "output": str(output),
                    "summary_sha256": summary["summary_sha256"],
                    "worker_count": 1,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "benchmark-head-family":
        preregistration = load_json(
            (args.preregistration or _default_preregistration_output(substrate.repo_root))
            .expanduser()
            .resolve()
        )
        calibration = load_json(
            (args.calibration or _default_calibration_output(substrate.repo_root))
            .expanduser()
            .resolve()
        )
        pca_root = _default_pca_output(substrate.repo_root)
        pca_manifest = load_json(
            (args.pca_manifest or (pca_root / "manifest.json")).expanduser().resolve()
        )
        plan = load_json(
            (args.plan or _default_stage1_output(substrate.repo_root)).expanduser().resolve()
        )
        screen = load_json((args.screen or _default_head_screen_output()).expanduser().resolve())
        output = (args.output or _default_head_run_output()).expanduser().resolve()
        summary = run_head_family_screen(
            substrate,
            preregistration,
            calibration,
            pca_manifest,
            plan,
            screen,
            pca_root,
            output,
            progress=lambda record: print(json.dumps(record, sort_keys=True), flush=True),
        )
        print(
            json.dumps(
                {
                    "benchmark_test_labels_accessed": False,
                    "completed_cells": summary["completed_cells"],
                    "expected_cells": summary["expected_cells"],
                    "output": str(output),
                    "summary_sha256": summary["summary_sha256"],
                    "worker_count": 1,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "benchmark-training-recipe":
        preregistration = load_json(
            (args.preregistration or _default_preregistration_output(substrate.repo_root))
            .expanduser()
            .resolve()
        )
        calibration = load_json(
            (args.calibration or _default_calibration_output(substrate.repo_root))
            .expanduser()
            .resolve()
        )
        pca_root = _default_pca_output(substrate.repo_root)
        pca_manifest = load_json(
            (args.pca_manifest or (pca_root / "manifest.json")).expanduser().resolve()
        )
        plan = load_json(
            (args.plan or _default_training_recipe_plan_output()).expanduser().resolve()
        )
        output = (args.output or _default_training_recipe_run_output()).expanduser().resolve()
        summary = run_training_recipe_program(
            substrate,
            preregistration,
            calibration,
            pca_manifest,
            plan,
            pca_root,
            output,
            progress=lambda record: print(json.dumps(record, sort_keys=True), flush=True),
        )
        print(
            json.dumps(
                {
                    "backend": "mlx",
                    "benchmark_test_labels_accessed": False,
                    "completed_new_cells": summary["completed_new_cells"],
                    "output": str(output),
                    "summary_sha256": summary["summary_sha256"],
                    "worker_count": 1,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "benchmark-stability":
        preregistration = load_json(
            (args.preregistration or _default_preregistration_output(substrate.repo_root))
            .expanduser()
            .resolve()
        )
        calibration = load_json(
            (args.calibration or _default_calibration_output(substrate.repo_root))
            .expanduser()
            .resolve()
        )
        pca_root = _default_pca_output(substrate.repo_root)
        pca_manifest = load_json(
            (args.pca_manifest or (pca_root / "manifest.json")).expanduser().resolve()
        )
        plan = load_json((args.plan or _default_stability_plan_output()).expanduser().resolve())
        output = (args.output or _default_stability_run_output()).expanduser().resolve()
        summary = run_stability_program(
            substrate,
            preregistration,
            calibration,
            pca_manifest,
            plan,
            pca_root,
            output,
            progress=lambda record: print(json.dumps(record, sort_keys=True), flush=True),
        )
        print(
            json.dumps(
                {
                    "backend": "mlx",
                    "benchmark_test_labels_accessed": False,
                    "completed_cells": summary["completed_cells"],
                    "output": str(output),
                    "summary_sha256": summary["summary_sha256"],
                    "worker_count": 1,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "benchmark-controls":
        preregistration = load_json(
            (args.preregistration or _default_preregistration_output(substrate.repo_root))
            .expanduser()
            .resolve()
        )
        calibration = load_json(
            (args.calibration or _default_calibration_output(substrate.repo_root))
            .expanduser()
            .resolve()
        )
        pca_root = _default_pca_output(substrate.repo_root)
        pca_manifest = load_json(
            (args.pca_manifest or (pca_root / "manifest.json")).expanduser().resolve()
        )
        plan = load_json((args.plan or _default_control_plan_output()).expanduser().resolve())
        baseline_root = (
            args.baseline_root or _default_supervised_run_output()
        ).expanduser().resolve()
        baseline_summary = load_json(
            (args.baseline_summary or (baseline_root / "summary.json")).expanduser().resolve()
        )
        output = (args.output or _default_control_run_output()).expanduser().resolve()
        summary = run_control_program(
            substrate,
            preregistration,
            calibration,
            pca_manifest,
            plan,
            baseline_summary,
            pca_root,
            baseline_root,
            output,
            progress=lambda record: print(json.dumps(record, sort_keys=True), flush=True),
        )
        print(
            json.dumps(
                {
                    "all_gates_pass": summary["all_gates_pass"],
                    "backend": "mlx",
                    "benchmark_test_labels_accessed": False,
                    "completed_new_cells": summary["completed_new_cells"],
                    "output": str(output),
                    "summary_sha256": summary["summary_sha256"],
                    "worker_count": 1,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "run-stage1-cell":
        config = Stage1CellConfig(
            target_name=args.target,
            fold=args.fold,
            seed=args.seed,
            pca_width=args.pca_width,
            head_family=args.head_family,
            hidden_width=args.hidden_width,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            residual_logit_cap=args.residual_logit_cap,
            batch_rows=args.batch_rows,
        )
        preregistration = load_json(
            (args.preregistration or _default_preregistration_output(substrate.repo_root))
            .expanduser()
            .resolve()
        )
        calibration = load_json(
            (args.calibration or _default_calibration_output(substrate.repo_root))
            .expanduser()
            .resolve()
        )
        pca_root = _default_pca_output(substrate.repo_root)
        pca_manifest = load_json(
            (args.pca_manifest or (pca_root / "manifest.json")).expanduser().resolve()
        )
        plan = load_json(
            (args.plan or _default_stage1_output(substrate.repo_root)).expanduser().resolve()
        )
        output = (args.output or _default_stage1_cell_output(config)).expanduser().resolve()
        metrics = run_stage1_discovery_cell(
            substrate,
            preregistration,
            calibration,
            pca_manifest,
            plan,
            pca_root,
            output,
            config,
        )
        print(json.dumps({"output": str(output), **metrics}, sort_keys=True))
        return 0
    if args.command == "preregister-event":
        output = (
            (args.output or _default_preregistration_output(substrate.repo_root))
            .expanduser()
            .resolve()
        )
        features = substrate.load_features(substrate.video_ids, ("diagnostics_only",))
        manifest = build_event_preregistration(substrate.identity, features)
        atomic_write_json(output, manifest)
        print(
            json.dumps(
                {
                    "label_values_accessed": manifest["label_values_accessed"],
                    "output": str(output),
                    "preregistration_sha256": manifest["preregistration_sha256"],
                    "schema": manifest["schema"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "calibrate-event":
        preregistration = load_json(args.preregistration.expanduser().resolve())
        output = (
            (args.output or _default_calibration_output(substrate.repo_root)).expanduser().resolve()
        )
        all_features = substrate.load_features(substrate.video_ids, ("diagnostics_only",))
        train_mask = benchmark_partition_mask(all_features, preregistration["split"], "train")
        features = all_features.subset(train_mask)
        labels = substrate.load_labels(
            substrate.video_ids,
            row_indices=_owned_rows(all_features, train_mask),
            stage="event_target_calibration_benchmark_train_only",
        )
        calibration = calibrate_event_preregistration(preregistration, features, labels)
        atomic_write_json(output, calibration)
        print(
            json.dumps(
                {
                    "calibration_sha256": calibration["calibration_sha256"],
                    "benchmark_test_labels_accessed": calibration["benchmark_test_labels_accessed"],
                    "output": str(output),
                    "schema": calibration["schema"],
                    "targets": len(calibration["target_hypotheses"]),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "fit-event-pca":
        preregistration = load_json(args.preregistration.expanduser().resolve())
        output = (args.output or _default_pca_output(substrate.repo_root)).expanduser().resolve()
        features = substrate.load_features(substrate.video_ids, ("tribe_cortical",))
        result = fit_event_pca_cache(
            features,
            preregistration,
            output,
            folds=args.fold,
            progress=lambda record: print(json.dumps(record, sort_keys=True), flush=True),
        )
        print(
            json.dumps(
                {
                    "benchmark_test_labels_accessed": False,
                    "folds": len(result["folds"]),
                    "manifest_sha256": result["manifest_sha256"],
                    "output": str(output),
                    "schema": result["schema"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command != "smoke":
        raise AssertionError("unreachable command")
    output = (args.output or _default_output(substrate.repo_root)).expanduser().resolve()
    cell = CellSpec(
        target=AROUSAL_SPIKE_1_3S,
        outer_fold=args.fold,
        seed=args.seed,
        promotable=False,
    )
    candidates = (
        CandidateSpec(
            name="tribe-cortical-pca64-c1",
            representation="tribe_cortical",
            pca_width=64,
            regularization_c=1.0,
            pca_solver="incremental",
            pca_batch_rows=128,
        ),
    )
    first = run_confirmation_cell(
        substrate,
        output,
        cell=cell,
        candidates=candidates,
        pause_after_seal=True,
    )
    result = (
        run_confirmation_cell(substrate, output, cell=cell, candidates=candidates)
        if first["status"] == "predictions_sealed"
        else first
    )
    verification = verify_confirmation_cell(
        substrate,
        output,
        cell=cell,
        candidates=candidates,
    )
    summary = {
        "status": result["status"],
        "audit_pass": result["audit"]["audit_pass"],
        "verification_pass": verification["verification_pass"],
        "promotable": result["metrics"]["promotable"],
        "rows": result["metrics"]["row_count"],
        "output": str(output),
    }
    print(json.dumps(summary, sort_keys=True))
    return (
        0
        if summary["audit_pass"] and summary["verification_pass"] and not summary["promotable"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
