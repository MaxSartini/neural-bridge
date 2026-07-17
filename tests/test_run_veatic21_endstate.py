from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import numpy as np
import pytest

from backend.scripts import run_veatic21_endstate as runner
from backend.scripts import veatic21_compact_cache as compact
from backend.scripts import veatic21_discovery as discovery
from backend.scripts import veatic21_endstate_contract as endstate


def _row_counts() -> dict[str, int]:
    # Exact canonical total, while still varying grouped-video weights.
    return {str(video): 167 if video < 73 else 166 for video in range(124)}


@pytest.fixture(scope="module")
def seal() -> runner.DatasetSeal:
    value = runner.synthetic_dataset_seal(_row_counts())
    assert value.total_rows == compact.EXPECTED_TOTAL_ROWS
    return value


@pytest.fixture(scope="module")
def plan(seal: runner.DatasetSeal) -> discovery.NestedDiscoveryPlan:
    return runner.build_plan(seal)


@pytest.fixture(scope="module")
def selections(
    plan: discovery.NestedDiscoveryPlan,
) -> discovery.DiscoverySelectionArtifact:
    return runner.dry_run_selection(plan)


def _args(tmp_path: Path, *, stage: str = "all", dry_run: bool = True) -> Namespace:
    return runner.build_parser().parse_args(
        [
            "--cache-root",
            str(tmp_path / "cache"),
            "--upstream-root",
            str(tmp_path / "upstream"),
            "--output-root",
            str(tmp_path / "output"),
            "--shared-derived-root",
            str(tmp_path / "derived"),
            "--stage",
            stage,
            *(["--dry-run"] if dry_run else []),
        ]
    )


def test_cli_contract_and_full_dry_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    runner.validate_cli_args(args)
    assert args.stage == "all"
    assert args.dry_run is True
    assert args.smoke is False

    report = compact.Veatic21ValidationReport(
        status="pass",
        video_ids=tuple(_row_counts()),
        video_count=124,
        total_rows=compact.EXPECTED_TOTAL_ROWS,
        row_hz=2.0,
        prediction_width=compact.PREDICTION_WIDTH,
        row_plan_sha256=compact.ROW_PLAN_SHA256,
        model_sha256=compact.MODEL_SHA256,
        dataset_fingerprint_sha256="f" * 64,
        row_counts=_row_counts(),
    )

    class FakeCache:
        def validate(self) -> compact.Veatic21ValidationReport:
            return report

    monkeypatch.setattr(runner, "_canonical_cache", lambda _args: FakeCache())
    summary = runner.run(args)
    assert summary["results"]["discovery"]["status"] == "planned"
    assert summary["results"]["confirmation"]["expected_rows"] == 3920
    assert summary["results"]["final"]["global_selection_count"] == 12
    assert summary["promotable"] is False
    assert summary["canonical_gates_passed"] is False
    manifest = json.loads((args.output_root / "run_manifest.json").read_text())
    assert manifest["confirmation_expected_rows"] == 3920
    assert manifest["discovery_expected_rows"] == 3240


def test_all_124_have_five_outer_and_three_inner_folds_with_no_reserve(
    plan: discovery.NestedDiscoveryPlan,
) -> None:
    assert plan.video_count == 124
    assert endstate.RESERVED_VIDEO_COUNT == 0
    assert len(plan.outer_folds) == 5
    seen: list[str] = []
    for outer in plan.outer_folds:
        assert len(outer.inner_folds) == 3
        assert not set(outer.train_videos) & set(outer.test_videos)
        assert len(outer.train_videos) + len(outer.test_videos) == 124
        seen.extend(outer.test_videos)
        for inner in outer.inner_folds:
            runner.assert_no_outer_leakage(
                plan=plan,
                outer_fold=outer.outer_fold,
                fit_videos=inner.train_videos,
                validation_videos=inner.validation_videos,
            )
    assert len(seen) == len(set(seen)) == 124


def test_outer_test_leakage_is_rejected(plan: discovery.NestedDiscoveryPlan) -> None:
    outer = plan.outer_folds[0]
    with pytest.raises(runner.EndStateRunError, match="outer-test"):
        runner.assert_no_outer_leakage(
            plan=plan,
            outer_fold=outer.outer_fold,
            fit_videos=outer.train_videos + (outer.test_videos[0],),
        )


def test_pca_plan_fits_max_width_once_and_seals_leading_slices(
    plan: discovery.NestedDiscoveryPlan,
) -> None:
    requests = runner.pca_requests_for_plan(plan)
    # Three families x (5 outer confirmation + 5*3 inner discovery).
    assert len(requests) == 60
    assert all(request.parent_width == 256 for request in requests)
    temporal = [
        request
        for request in requests
        if request.scope == "inner_discovery"
        and request.outer_fold == 1
        and request.inner_fold == 1
        and request.feature_family == "temporal_mean_2s"
    ]
    assert len(temporal) == 1
    assert temporal[0].requested_widths == (64, 256)
    assert not set(temporal[0].fit_videos) & set(temporal[0].held_out_videos)

    parent = {
        "identity_sha256": "parent-seal",
        "identity": {"pca_width": 256},
    }
    sliced = runner.pca_slice_manifest(parent_metadata=parent, requested_width=64)
    assert sliced["parent_pca_identity_sha256"] == "parent-seal"
    assert sliced["slice_width"] == 64
    assert sliced["component_range"] == [0, 64]
    assert sliced["policy"] == "leading_components_only_no_refit"


def test_exact_confirmation_row_counts_and_unique_accounting(
    plan: discovery.NestedDiscoveryPlan,
    selections: discovery.DiscoverySelectionArtifact,
) -> None:
    cells = runner.build_confirmation_matrix(selections, plan)
    audit = runner.audit_confirmation_matrix(cells, plan=plan)
    assert audit.passed
    assert audit.observed_rows == audit.expected_rows == 3920
    counts = dict(audit.endpoint_counts)
    assert counts == {
        "privileged_binary": 1680,
        "privileged_continuous": 1680,
        "zero_label_continuous": 560,
    }
    assert len({cell.key for cell in cells}) == 3920


def test_binary_is_true_bce_and_zero_event_reuses_continuous_predictions(
    plan: discovery.NestedDiscoveryPlan,
    selections: discovery.DiscoverySelectionArtifact,
) -> None:
    cells = runner.build_confirmation_matrix(selections, plan)
    binary = [cell for cell in cells if cell.endpoint == "privileged_binary"]
    zero = [cell for cell in cells if cell.endpoint == "zero_label_continuous"]
    assert binary
    assert all(cell.objective == "bce_with_frozen_ar_logit_offset" for cell in binary)
    assert all(cell.endpoint != "zero_label_binary" for cell in cells)
    assert all(cell.event_from_continuous_prediction for cell in zero)
    assert all(cell.objective == "direct_huber_continuous" for cell in zero)


def test_zero_label_schema_and_teacher_sealing_order(
    plan: discovery.NestedDiscoveryPlan,
    selections: discovery.DiscoverySelectionArtifact,
) -> None:
    runner.assert_zero_label_schema(
        (
            "pca256_lag0_0",
            "temporal_diagnostic_0",
            "history_available_lag0",
            "log1p_time_seconds",
        )
    )
    with pytest.raises(runner.EndStateRunError, match="zero-label feature audit"):
        runner.assert_zero_label_schema(("observed_arousal_lag1",))

    cells = runner.build_confirmation_matrix(selections, plan)
    for target in plan.targets:
        for outer in plan.outer_folds:
            zero = [
                cell
                for cell in cells
                if cell.target == target
                and cell.outer_fold == outer.outer_fold
                and cell.endpoint == runner.ZERO_ENDPOINT
            ]
            first_teacher = next(
                index
                for index, cell in enumerate(zero)
                if cell.lane == "privileged_teacher_ceiling"
            )
            assert all(cell.response_free for cell in zero[:first_teacher])
            assert all(cell.descriptive_only for cell in zero[first_teacher:])


def test_fixed_member_and_ensemble_accounting(
    plan: discovery.NestedDiscoveryPlan,
    selections: discovery.DiscoverySelectionArtifact,
) -> None:
    cells = runner.build_confirmation_matrix(selections, plan)
    members = [cell for cell in cells if cell.row_kind == runner.MEMBER_KIND]
    ensembles = [cell for cell in cells if cell.row_kind == runner.ENSEMBLE_KIND]
    assert len(members) == 2 * 1260 + 420
    assert len(ensembles) == 2 * 420 + 140
    assert all(cell.seed is not None and len(cell.member_seeds) == 1 for cell in members)
    assert all(cell.seed is None and len(cell.member_seeds) == 3 for cell in ensembles)
    privileged = [cell for cell in ensembles if cell.endpoint != runner.ZERO_ENDPOINT]
    zero = [cell for cell in ensembles if cell.endpoint == runner.ZERO_ENDPOINT]
    assert {cell.ensemble_group for cell in privileged} == {1, 2, 3}
    assert {cell.ensemble_group for cell in zero} == {1}


def test_stage_resume_identity_is_restart_safe_and_rejects_drift(
    tmp_path: Path,
    seal: runner.DatasetSeal,
    plan: discovery.NestedDiscoveryPlan,
) -> None:
    args = _args(tmp_path, stage="discovery")
    identity = runner.build_run_identity(args=args, seal=seal, plan=plan)
    runner.write_or_verify_run_identity(args.output_root, identity)
    runner.write_or_verify_run_identity(args.output_root, identity)

    drifted_args = _args(tmp_path, stage="discovery")
    drifted_args.max_epochs += 1
    drifted = runner.build_run_identity(args=drifted_args, seal=seal, plan=plan)
    assert drifted.digest != identity.digest
    with pytest.raises(runner.EndStateRunError, match="resume identity mismatch"):
        runner.write_or_verify_run_identity(args.output_root, drifted)


def test_prediction_shard_is_atomic_and_identity_bound(
    tmp_path: Path,
    plan: discovery.NestedDiscoveryPlan,
    selections: discovery.DiscoverySelectionArtifact,
) -> None:
    cell = runner.build_confirmation_matrix(selections, plan, smoke=True)[0]
    path = runner.prediction_path(tmp_path, cell)
    manifest = runner.seal_prediction_shard(
        path=path,
        cell=cell,
        row_indices=np.arange(8),
        video_ids=np.asarray(["1"] * 4 + ["2"] * 4),
        y_true=np.linspace(0.0, 1.0, 8),
        prediction=np.linspace(0.1, 0.9, 8),
        event_threshold=0.75,
        checkpoint=None,
        run_identity_digest="run-seal",
        extra_provenance={"fit_scope": "outer_train_only"},
    )
    assert manifest["row_count"] == 8
    runner.verify_prediction_shard(
        path, cell=cell, run_identity_digest="run-seal"
    )
    with pytest.raises(runner.EndStateRunError, match="different run identity"):
        runner.verify_prediction_shard(
            path, cell=cell, run_identity_digest="another-run"
        )


def test_final_export_contract_is_all124_fresh_and_not_evidence(
    tmp_path: Path,
    seal: runner.DatasetSeal,
    plan: discovery.NestedDiscoveryPlan,
    selections: discovery.DiscoverySelectionArtifact,
) -> None:
    args = _args(tmp_path, stage="final")
    identity = runner.build_run_identity(args=args, seal=seal, plan=plan)
    global_selections = runner.select_global_recipes(selections, plan)
    epochs = {
        (item.target, item.protocol, item.recipe_order): (7, 9, 11, 10, 8)
        for item in global_selections
    }
    payload = runner.build_final_export_contract(
        identity=identity,
        plan=plan,
        selections=selections,
        best_epochs=epochs,
        max_epochs=args.max_epochs,
    )
    runner.audit_final_export_contract(payload)
    assert payload["video_count"] == 124
    assert payload["reserve_count"] == 0
    assert payload["all_video_refit"] is True
    assert payload["fresh_pca"] is True
    assert payload["fresh_normalizers"] is True
    assert payload["fresh_target_specific_ar"] is True
    assert payload["fresh_all_video_q90_thresholds"] is True
    assert payload["no_in_sample_metric_claim"] is True
    assert payload["zero_label_response_inputs"] is False
    assert len(payload["global_selections"]) == 12
    assert len(payload["fixed_epochs"]) == 12
    assert {row["fixed_epoch"] for row in payload["fixed_epochs"]} == {9}


def test_smoke_is_noncanonical_but_keeps_all_matched_lanes(
    plan: discovery.NestedDiscoveryPlan,
    selections: discovery.DiscoverySelectionArtifact,
) -> None:
    cells = runner.build_confirmation_matrix(selections, plan, smoke=True)
    audit = runner.audit_confirmation_matrix(cells, plan=plan, smoke=True)
    assert audit.expected_rows == 42
    assert len({cell.lane for cell in cells if cell.endpoint == "privileged_continuous"}) == 7
    assert len({cell.lane for cell in cells if cell.endpoint == "privileged_binary"}) == 7
    assert len({cell.lane for cell in cells if cell.endpoint == runner.ZERO_ENDPOINT}) == 7
