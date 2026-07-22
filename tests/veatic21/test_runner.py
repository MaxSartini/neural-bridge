from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import cast

import numpy as np
import pytest

import neural_bridge.veatic21.contracts as contracts
import neural_bridge.veatic21.runner as runner
from neural_bridge.veatic21.contracts import (
    CANONICAL_DATASET,
    CONTROL_LANES,
    CandidateSpec,
    CellSpec,
    FeatureRows,
    LabelRows,
    SubstrateIdentity,
    TargetSpec,
)
from neural_bridge.veatic21.data import CanonicalSubstrate
from neural_bridge.veatic21.evidence import (
    atomic_save_npz,
    atomic_write_json,
    load_json,
    sha256_file,
)
from neural_bridge.veatic21.protocol import freeze_final_recipe
from neural_bridge.veatic21.runner import (
    refit_all_124,
    run_confirmation_cell,
    verify_confirmation_cell,
)
from neural_bridge.veatic21.stage1 import CheckpointSelector, build_stage1_plan


def test_stage1_checkpoint_policy_keeps_epoch_one_and_has_no_epoch_ceiling() -> None:
    selector = CheckpointSelector()
    assert selector.observe(1, 0.2)
    for epoch in range(2, 401):
        selector.observe(epoch, 0.1)
    assert selector.best_epoch == 1
    assert selector.should_stop(400, optimizer_converged=True)

    selector.observe(401, 0.3)
    assert selector.best_epoch == 401
    assert not selector.should_stop(10_000, optimizer_converged=False)


def test_stage1_plan_encodes_registered_checkpoint_and_fold_matrix() -> None:
    preregistration = {
        "preregistration_sha256": "pre",
        "heads": {
            "label_assisted_discovery": [
                "frozen_ar_plus_causal_temporal_residual",
                "frozen_ar_plus_gated_multiscale_temporal_residual",
            ]
        },
        "training": {
            "checkpoint_eligibility": "every_completed_validation_from_epoch_1",
            "checkpoint_metric": "inner_average_precision_skill_delta_vs_frozen_ar",
            "comparison_seed_panel": [1, 2, 3],
            "last_checkpoint_preference": False,
            "minimum_epochs_before_termination": 50,
        },
    }
    calibration = {
        "benchmark_test_labels_accessed": False,
        "calibration_sha256": "cal",
        "preregistration_sha256": "pre",
        "schema": "veatic21_event_calibration_v12",
        "target_hypotheses": [
            {
                "horizon_rows": [2, 4],
                "label": "arousal",
                "name": "target",
                "train_quantile": 0.9,
                "transform": "absolute",
            }
        ],
    }
    pca_manifest = {
        "folds": [{"candidate_widths": [64, 128], "directory": "fold-0", "fold": 0}],
        "manifest_sha256": "pca",
        "preregistration_sha256": "pre",
    }
    plan = build_stage1_plan(
        preregistration,
        calibration,
        pca_manifest,
        {"backend": "mlx", "safe_batch_rows_by_hidden_width": {"64": 128}},
    )
    assert plan["checkpoint_policy"] == {
        "eligible_from_epoch": 1,
        "maximum_epochs": None,
        "minimum_epochs_before_termination": 50,
        "optimizer_convergence_required": True,
        "plateau_patience_epochs": 50,
        "selection_metric": "inner_average_precision_skill_delta_vs_frozen_ar",
        "tie_break": "earliest_checkpoint",
    }
    assert plan["matrix"]["folds"][0]["candidate_pca_widths"] == [64, 128]


class _MemorySubstrate:
    def __init__(
        self,
        features: FeatureRows,
        labels: LabelRows,
        identity: SubstrateIdentity,
    ) -> None:
        self._features = features
        self._labels = labels
        self.identity = identity
        self.feature_loads = 0
        self.label_stages: list[str] = []

    @property
    def video_ids(self) -> tuple[str, ...]:
        return self.identity.video_ids

    @staticmethod
    def _requested(video_ids: Iterable[str | int] | str | int) -> set[str]:
        if isinstance(video_ids, (str, int)):
            return {str(video_ids)}
        return {str(video_id) for video_id in video_ids}

    def load_features(
        self,
        video_ids: Iterable[str | int] | str | int,
        representations: Iterable[str],
    ) -> FeatureRows:
        self.feature_loads += 1
        mask = np.isin(self._features.video_id.astype(str), list(self._requested(video_ids)))
        requested = tuple(representations)
        return FeatureRows(
            video_id=self._features.video_id[mask],
            row_index=self._features.row_index[mask],
            time_seconds=self._features.time_seconds[mask],
            quality_eligible=self._features.quality_eligible[mask],
            representations={
                name: self._features.representations[name][mask] for name in requested
            },
        )

    def load_labels(
        self,
        video_ids: Iterable[str | int] | str | int,
        *,
        access_callback: Callable[[str], None] | None = None,
        stage: str = "load_labels",
    ) -> LabelRows:
        self.label_stages.append(stage)
        if access_callback is not None:
            access_callback(stage)
        mask = np.isin(self._labels.video_id.astype(str), list(self._requested(video_ids)))
        return LabelRows(
            video_id=self._labels.video_id[mask],
            row_index=self._labels.row_index[mask],
            time_seconds=self._labels.time_seconds[mask],
            arousal=self._labels.arousal[mask],
            valence=self._labels.valence[mask],
        )


def _identity() -> SubstrateIdentity:
    return SubstrateIdentity(
        video_ids=tuple(str(video_id) for video_id in range(CANONICAL_DATASET.video_count)),
        row_count=CANONICAL_DATASET.row_count,
        exclusion_count=CANONICAL_DATASET.exclusion_count,
        row_hz=CANONICAL_DATASET.row_hz,
        vjepa_artifact_id="veatic-2.1-memory-vjepa",
        vjepa_sha256_tree="a" * 64,
        vjepa_file_count=1,
        vjepa_size_bytes=1,
        tribe_artifact_id="veatic-2.1-memory-tribe",
        tribe_sha256_tree="b" * 64,
        tribe_file_count=1,
        tribe_size_bytes=1,
        row_plan_sha256="c" * 64,
        source_tree_sha256="d" * 64,
        encoder_model_sha256="e" * 64,
    )


def _small_substrate() -> _MemorySubstrate:
    per_video = 12
    video_number = np.repeat(np.arange(CANONICAL_DATASET.video_count), per_video)
    video_id = video_number.astype(str)
    row_index = np.tile(np.arange(per_video), CANONICAL_DATASET.video_count)
    time_seconds = row_index.astype(np.float64) / CANONICAL_DATASET.row_hz
    pattern = np.array([0.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 4.0, 0.0, 2.0, 0.0])
    scale = 1.0 + (video_number % 7) / 5.0
    arousal = pattern[row_index] * scale + (video_number % 3) / 100.0
    representation = np.column_stack(
        (
            arousal,
            np.sin((row_index + 1) * (video_number + 1) * 0.07),
            row_index / (per_video - 1),
        )
    )
    diagnostics = np.column_stack(
        (np.cos((row_index + 1) * 0.3), (video_number % 5) / 5.0)
    )
    features = FeatureRows(
        video_id=video_id,
        row_index=row_index,
        time_seconds=time_seconds,
        quality_eligible=np.ones(len(video_id), dtype=bool),
        representations={
            "vjepa_temporal_mean": representation,
            "diagnostics_only": diagnostics,
        },
    )
    labels = LabelRows(
        video_id=video_id,
        row_index=row_index,
        time_seconds=time_seconds,
        arousal=arousal,
        valence=np.cos(row_index * 0.2),
    )
    return _MemorySubstrate(features, labels, _identity())


def _candidate() -> CandidateSpec:
    return CandidateSpec(
        name="synthetic-vjepa",
        representation="vjepa_temporal_mean",
        pca_width=2,
        regularization_c=1.0,
        max_iter=1_000,
        tolerance=1e-4,
    )


def test_video_mean_uses_only_each_rows_causal_prefix() -> None:
    values = np.asarray(((10.0,), (1.0,), (30.0,), (3.0,), (100.0,)))
    video_id = np.asarray(("1", "0", "1", "0", "1"))
    row_index = np.asarray((0, 0, 1, 1, 2))

    result = runner._causal_video_means(values, video_id, row_index)

    np.testing.assert_allclose(result[:, 0], (10.0, 1.0, 20.0, 2.0, 140.0 / 3.0))


def test_cortical_candidates_require_declared_bounded_incremental_pca() -> None:
    with pytest.raises(ValueError, match="incremental PCA"):
        CandidateSpec(
            name="unsafe-cortical",
            representation="tribe_cortical",
            pca_width=4,
            regularization_c=1.0,
        ).validate()

    candidate = CandidateSpec(
        name="bounded-cortical",
        representation="tribe_cortical",
        pca_width=4,
        regularization_c=1.0,
        pca_solver="incremental",
        pca_batch_rows=8,
    )
    candidate.validate()


def test_incremental_transform_is_bounded_and_matches_dense_application(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = np.random.default_rng(7).normal(size=(24, 11))
    selected = np.ones(len(values), dtype=bool)
    selected[[2, 9, 17, 23]] = False
    batch_sizes: list[int] = []
    original_partial_fit = runner.IncrementalPCA.partial_fit

    def record_partial_fit(
        self: runner.IncrementalPCA, batch: np.ndarray, *args: object, **kwargs: object
    ) -> runner.IncrementalPCA:
        batch_sizes.append(len(batch))
        return original_partial_fit(self, batch, *args, **kwargs)

    monkeypatch.setattr(runner.IncrementalPCA, "partial_fit", record_partial_fit)
    transform = runner._fit_transform(
        values,
        3,
        seed=11,
        rows=selected,
        solver="incremental",
        batch_rows=6,
    )

    assert batch_sizes and max(batch_sizes) <= 6 and min(batch_sizes) >= 3
    dense = (
        (values - transform.scaler_mean) / transform.scaler_scale - transform.pca_mean
    ) @ transform.pca_components.T
    np.testing.assert_allclose(transform.apply(values), dense, rtol=1e-12, atol=1e-12)


def test_incremental_fit_ignores_rows_outside_its_training_mask() -> None:
    values = np.random.default_rng(13).normal(size=(30, 12))
    train_mask = np.arange(len(values)) < 20
    changed = values.copy()
    changed[~train_mask] = 1_000_000.0

    first = runner._fit_transform(
        values,
        3,
        seed=5,
        rows=train_mask,
        solver="incremental",
        batch_rows=10,
    )
    second = runner._fit_transform(
        changed,
        3,
        seed=5,
        rows=train_mask,
        solver="incremental",
        batch_rows=10,
    )

    for left, right in zip(
        (first.scaler_mean, first.scaler_scale, first.pca_mean, first.pca_components),
        (second.scaler_mean, second.scaler_scale, second.pca_mean, second.pca_components),
        strict=True,
    ):
        np.testing.assert_array_equal(left, right)


def test_label_null_is_deterministic_nonzero_rotation_within_each_video() -> None:
    labels = np.asarray((0, 1, 1, 0, 1, 0, 0, 1), dtype=np.int8)
    videos = np.asarray(("0", "0", "0", "0", "1", "1", "1", "1"))
    rows = np.asarray((0, 1, 2, 3, 0, 1, 2, 3))
    first = runner._circular_permute_labels(
        labels, videos, rows, runner._lane_rng(29, "label_permutation")
    )
    second = runner._circular_permute_labels(
        labels, videos, rows, runner._lane_rng(29, "label_permutation")
    )

    np.testing.assert_array_equal(first, second)
    for video in ("0", "1"):
        mask = videos == video
        source = labels[mask]
        shifted = first[mask]
        assert int(source.sum()) == int(shifted.sum())
        assert any(
            np.array_equal(shifted, np.roll(source, shift))
            for shift in range(1, len(source))
        )


def test_null_rng_streams_are_independent_and_random_is_not_a_matched_control() -> None:
    expected = runner._lane_rng(31, "random").random(8)
    sequence_rng = runner._lane_rng(31, "sequence_shuffled")
    sequence_rng.random(100)
    actual = runner._lane_rng(31, "random").random(8)

    np.testing.assert_array_equal(actual, expected)
    assert "random" not in runner.MATCHED_CONTROL_LANES


def test_feature_rows_validation_checks_representations_in_bounded_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row_count = 600
    rows = FeatureRows(
        video_id=np.full(row_count, "0"),
        row_index=np.arange(row_count),
        time_seconds=np.arange(row_count, dtype=np.float64) / 2.0,
        quality_eligible=np.ones(row_count, dtype=np.bool_),
        representations={"tribe_cortical": np.zeros((row_count, 4), dtype=np.float16)},
    )
    checked_rows: list[int] = []
    original_isfinite = np.isfinite

    def record_isfinite(values: np.ndarray) -> np.ndarray:
        if values.ndim == 2:
            checked_rows.append(len(values))
        return original_isfinite(values)

    monkeypatch.setattr(contracts.np, "isfinite", record_isfinite)
    rows.validate()
    assert checked_rows == [256, 256, 88]


@pytest.mark.parametrize(
    "candidates",
    [
        (),
        (
            CandidateSpec(
                name="invalid",
                representation="vjepa_temporal_mean",
                pca_width=0,
                regularization_c=1.0,
            ),
        ),
    ],
)
def test_confirmation_validates_candidates_before_writing(
    tmp_path: Path,
    candidates: tuple[CandidateSpec, ...],
) -> None:
    substrate = cast(CanonicalSubstrate, _small_substrate())
    target = TargetSpec("synthetic-spike", "arousal", (1, 2), 0.5)
    cell = CellSpec(target=target, outer_fold=0, seed=20_260_721)
    output_dir = tmp_path / "cell"

    with pytest.raises(ValueError):
        run_confirmation_cell(substrate, output_dir, cell=cell, candidates=candidates)
    assert not output_dir.exists()


def test_confirmation_pause_resume_audit_and_tamper_refusal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = _small_substrate()
    substrate = cast(CanonicalSubstrate, memory)
    target = TargetSpec("synthetic-spike", "arousal", (1, 2), 0.5)
    cell = CellSpec(target=target, outer_fold=0, seed=20_260_721)
    output_dir = tmp_path / "cell"

    paused = run_confirmation_cell(
        substrate,
        output_dir,
        cell=cell,
        candidates=(_candidate(),),
        pause_after_seal=True,
    )
    assert paused["status"] == "predictions_sealed"
    assert paused["resumed"] is False
    prediction_path = output_dir / "predictions.npz"
    prediction_sha256 = sha256_file(prediction_path)
    prediction_mtime = prediction_path.stat().st_mtime_ns
    feature_loads = memory.feature_loads
    assert memory.label_stages == ["outer_train_labels_opened_for_nested_discovery"]
    model_files = load_json(output_dir / "fit.json")["model_files"]
    assert set(model_files) == set(runner.FITTED_PRESEAL_LANES)
    assert all((output_dir / filename).is_file() for filename in model_files.values())

    def refuse_refit(*args: object, **kwargs: object) -> None:
        raise AssertionError("resume retrained a sealed cell")

    monkeypatch.setattr(runner, "_fit_linear", refuse_refit)
    resumed = run_confirmation_cell(
        substrate,
        output_dir,
        cell=cell,
        candidates=(_candidate(),),
    )

    assert resumed["status"] == "audited"
    assert resumed["resumed"] is True
    assert memory.feature_loads == feature_loads
    assert sha256_file(prediction_path) == prediction_sha256
    assert prediction_path.stat().st_mtime_ns == prediction_mtime
    assert memory.label_stages[-1] == "outer_test_labels_opened_after_prediction_seal"
    metrics = resumed["metrics"]
    audit = resumed["audit"]
    assert set(CONTROL_LANES).issubset(metrics["pooled_pr_auc"])
    assert metrics["promotable"] is False
    assert metrics["scientific_claim"] is None
    assert audit["audit_pass"] is True
    assert audit["non_promotable_smoke"] is True
    assert audit["result_scope"] == "plumbing_smoke"
    assert audit["label_access_events"].index("prediction_seal_written") < audit[
        "label_access_events"
    ].index("outer_test_labels_opened_after_prediction_seal")
    verification = verify_confirmation_cell(
        substrate,
        output_dir,
        cell=cell,
        candidates=(_candidate(),),
    )
    assert verification["verification_pass"] is True
    assert verification["failures"] == []

    with np.load(prediction_path, allow_pickle=False) as stored:
        tampered = {name: np.asarray(stored[name]) for name in stored.files}
    tampered[runner.PRIMARY] = tampered[runner.PRIMARY].copy()
    tampered[runner.PRIMARY][0] += 0.01
    atomic_save_npz(prediction_path, tampered)
    seal_path = output_dir / "prediction_seal.json"
    tampered_seal = load_json(seal_path)
    tampered_seal["prediction_sha256"] = sha256_file(prediction_path)
    atomic_write_json(seal_path, tampered_seal)
    label_stages = list(memory.label_stages)
    with pytest.raises(RuntimeError, match="prediction seal changed"):
        run_confirmation_cell(
            substrate,
            output_dir,
            cell=cell,
            candidates=(_candidate(),),
        )
    assert memory.label_stages == label_stages


def _run_audited_cell(
    output_dir: Path,
) -> tuple[_MemorySubstrate, CanonicalSubstrate, CellSpec, tuple[CandidateSpec, ...]]:
    memory = _small_substrate()
    substrate = cast(CanonicalSubstrate, memory)
    target = TargetSpec("synthetic-spike", "arousal", (1, 2), 0.5)
    cell = CellSpec(target=target, outer_fold=0, seed=20_260_721)
    candidates = (_candidate(),)
    result = run_confirmation_cell(
        substrate,
        output_dir,
        cell=cell,
        candidates=candidates,
    )
    assert result["audit"]["audit_pass"] is True
    return memory, substrate, cell, candidates


def test_verifier_replays_preseal_predictions_instead_of_trusting_resealed_scores(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "cell"
    _, substrate, cell, candidates = _run_audited_cell(output_dir)
    prediction_path = output_dir / "predictions.npz"
    with np.load(prediction_path, allow_pickle=False) as stored:
        predictions = {name: np.asarray(stored[name]) for name in stored.files}
    predictions[runner.PRIMARY] = predictions[runner.PRIMARY].copy()
    predictions[runner.PRIMARY][0] += 0.01
    atomic_save_npz(prediction_path, predictions)

    seal_path = output_dir / "prediction_seal.json"
    seal = load_json(seal_path)
    seal["prediction_sha256"] = sha256_file(prediction_path)
    atomic_write_json(seal_path, seal)
    state_path = output_dir / "state.json"
    state = load_json(state_path)
    state["prediction_seal_sha256"] = sha256_file(seal_path)
    atomic_write_json(state_path, state)

    verification = verify_confirmation_cell(
        substrate,
        output_dir,
        cell=cell,
        candidates=candidates,
    )

    assert verification["verification_pass"] is False
    assert "prediction_replay_primary" in verification["failures"]


def test_verifier_recomputes_frozen_ar_instead_of_trusting_evaluator_scores(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "cell"
    _, substrate, cell, candidates = _run_audited_cell(output_dir)
    evaluator_path = output_dir / "evaluator_controls.npz"
    with np.load(evaluator_path, allow_pickle=False) as stored:
        evaluator = {name: np.asarray(stored[name]) for name in stored.files}
    evaluator["target_specific_frozen_ar"] = evaluator[
        "target_specific_frozen_ar"
    ].copy()
    evaluator["target_specific_frozen_ar"][0] += 0.01
    atomic_save_npz(evaluator_path, evaluator)

    evaluator_seal_path = output_dir / "evaluator_control_seal.json"
    evaluator_seal = load_json(evaluator_seal_path)
    evaluator_seal["sha256"] = sha256_file(evaluator_path)
    atomic_write_json(evaluator_seal_path, evaluator_seal)
    state_path = output_dir / "state.json"
    state = load_json(state_path)
    state["evaluator_control_seal_sha256"] = sha256_file(evaluator_seal_path)
    atomic_write_json(state_path, state)

    verification = verify_confirmation_cell(
        substrate,
        output_dir,
        cell=cell,
        candidates=candidates,
    )

    assert verification["verification_pass"] is False
    assert "frozen_ar_replay" in verification["failures"]


def test_refit_all_124_is_blocked_until_a_verified_promotion_gate(tmp_path: Path) -> None:
    memory = _small_substrate()
    candidate = _candidate()
    discovery = [
        {
            "candidate": candidate.name,
            "outer_fold": fold,
            "pooled_pr_auc": 0.6 + fold / 100.0,
            "discovery_digest": hashlib.sha256(f"fold-{fold}".encode()).hexdigest(),
        }
        for fold in range(5)
    ]
    recipe = freeze_final_recipe(
        (candidate,),
        discovery,
        refit_seed=123,
    )
    target = TargetSpec("synthetic-spike", "arousal", (1, 2), 0.75)
    output_dir = tmp_path / "final"

    with pytest.raises(RuntimeError, match="verifiable promotion gate"):
        refit_all_124(
            cast(CanonicalSubstrate, memory),
            output_dir,
            recipe=recipe,
            target=target,
        )

    assert not output_dir.exists()
    assert memory.feature_loads == 0
    assert memory.label_stages == []
