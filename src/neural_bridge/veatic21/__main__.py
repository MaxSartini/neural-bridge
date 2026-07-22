"""VEATIC 2.1 foundation commands."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from neural_bridge.mlflow_registry import log_completed_output, start_run

from .contracts import AROUSAL_SPIKE_1_3S, CandidateSpec, CellSpec
from .data import CanonicalSubstrate
from .event_screen import run_event_target_screen, write_event_target_screen
from .evidence import atomic_write_json, load_json
from .pca_cache import fit_event_pca_cache
from .preregistration import (
    benchmark_partition_mask,
    build_event_preregistration,
    calibrate_event_preregistration,
)
from .runner import run_confirmation_cell, verify_confirmation_cell
from .stage1 import build_stage1_plan, probe_stage1_capacity, write_stage1_plan


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
    screen = subparsers.add_parser(
        "screen-event",
        help="run supervised target/representation screening on benchmark-train videos only",
    )
    screen.add_argument("--preregistration", type=Path, required=True)
    screen.add_argument("--calibration", type=Path, required=True)
    screen.add_argument(
        "--source",
        action="append",
        choices=("vjepa_temporal_mean", "tribe_grouped_mean", "tribe_cortical"),
        required=True,
    )
    screen.add_argument("--output", type=Path)
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
    stage1.add_argument("--output", type=Path)
    return parser


def _default_output(repo_root: Path) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return repo_root / "artifacts/runs/veatic-2.1/foundation-smoke" / stamp


def _default_preregistration_output(repo_root: Path) -> Path:
    return repo_root / "artifacts/preregistrations/veatic-2.1/event-spike-v1.json"


def _default_calibration_output(repo_root: Path) -> Path:
    return repo_root / "artifacts/preregistrations/veatic-2.1/event-spike-v1-calibration.json"


def _default_screen_output(repo_root: Path, sources: list[str]) -> Path:
    source_key = "-".join(sorted(sources))
    return repo_root / f"artifacts/runs/veatic-2.1/event-target-screen/{source_key}.json"


def _default_pca_output(repo_root: Path) -> Path:
    return repo_root / "artifacts/features/veatic-2.1/neural-bridge/cortical-pca-v1"


def _default_stage1_output(repo_root: Path) -> Path:
    return repo_root / "artifacts/preregistrations/veatic-2.1/stage1-child-plan.json"


def _owned_rows(features, mask) -> dict[str, list[int]]:
    rows = {video: [] for video in sorted(set(features.video_id.astype(str)), key=int)}
    for video, row, owned in zip(features.video_id, features.row_index, mask, strict=True):
        if owned:
            rows[str(video)].append(int(row))
    return rows


def main() -> int:
    args = _parser().parse_args()
    substrate = CanonicalSubstrate.from_repo()
    if args.command == "preregister-event":
        output = (
            args.output or _default_preregistration_output(substrate.repo_root)
        ).expanduser().resolve()
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
            args.output or _default_calibration_output(substrate.repo_root)
        ).expanduser().resolve()
        all_features = substrate.load_features(substrate.video_ids, ("diagnostics_only",))
        train_mask = benchmark_partition_mask(
            all_features, preregistration["split"], "train"
        )
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
                    "benchmark_test_labels_accessed": calibration[
                        "benchmark_test_labels_accessed"
                    ],
                    "output": str(output),
                    "schema": calibration["schema"],
                    "targets": len(calibration["target_hypotheses"]),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "screen-event":
        preregistration = load_json(args.preregistration.expanduser().resolve())
        calibration = load_json(args.calibration.expanduser().resolve())
        sources = list(dict.fromkeys(args.source))
        output = (
            args.output or _default_screen_output(substrate.repo_root, sources)
        ).expanduser().resolve()
        with start_run(
            "veatic-2.1",
            "event-target-screen",
            run_name=output.stem,
            tags={"neural_bridge.execution_kind": "native"},
        ):
            all_features = substrate.load_features(substrate.video_ids, sources)
            train_mask = benchmark_partition_mask(
                all_features, preregistration["split"], "train"
            )
            features = all_features.subset(train_mask)
            labels = substrate.load_labels(
                substrate.video_ids,
                row_indices=_owned_rows(all_features, train_mask),
                stage="event_screen_benchmark_train_only",
            )
            result = run_event_target_screen(
                features,
                labels,
                preregistration,
                calibration,
                sources=sources,
            )
            write_event_target_screen(output, result)
            log_completed_output(
                output,
                parameters={
                    "folds": len(preregistration["split"]["inner_grouped_video_folds"]),
                    "sources": ",".join(sources),
                    "targets": len(calibration["target_hypotheses"]),
                },
                metrics={"records": len(result["records"])},
                tags={
                    "neural_bridge.result_sha256": result["screen_sha256"],
                    "neural_bridge.schema": result["schema"],
                },
            )
        print(
            json.dumps(
                {
                    "benchmark_test_labels_accessed": result[
                        "benchmark_test_labels_accessed"
                    ],
                    "output": str(output),
                    "records": len(result["records"]),
                    "schema": result["schema"],
                    "screen_sha256": result["screen_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "fit-event-pca":
        preregistration = load_json(args.preregistration.expanduser().resolve())
        output = (
            args.output or _default_pca_output(substrate.repo_root)
        ).expanduser().resolve()
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
    if args.command == "prepare-stage1":
        preregistration_path = args.preregistration or _default_preregistration_output(
            substrate.repo_root
        )
        calibration_path = args.calibration or _default_calibration_output(
            substrate.repo_root
        )
        pca_manifest_path = args.pca_manifest or (
            _default_pca_output(substrate.repo_root) / "manifest.json"
        )
        output = (args.output or _default_stage1_output(substrate.repo_root)).expanduser().resolve()
        preregistration = load_json(preregistration_path.expanduser().resolve())
        calibration = load_json(calibration_path.expanduser().resolve())
        pca_manifest = load_json(pca_manifest_path.expanduser().resolve())
        capacity = probe_stage1_capacity(pca_manifest)
        plan = build_stage1_plan(preregistration, calibration, pca_manifest, capacity)
        write_stage1_plan(output, plan)
        print(
            json.dumps(
                {
                    "output": str(output),
                    "plan_sha256": plan["plan_sha256"],
                    "schema": plan["schema"],
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
            name="tribe-grouped-mean-pca8-c1",
            representation="tribe_grouped_mean",
            pca_width=8,
            regularization_c=1.0,
        ),
        CandidateSpec(
            name="vjepa-temporal-mean-pca8-c1",
            representation="vjepa_temporal_mean",
            pca_width=8,
            regularization_c=1.0,
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
