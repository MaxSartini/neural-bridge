"""Numerical executors for the locked VEATIC 2.1 end-state runner.

This module is intentionally downstream of the immutable planning contracts.
It consumes an already validated :class:`DenseDataset`, fits every numerical
artifact from the rows owned by the supplied split, and returns compact values
for the runner to seal.  Full feature/control matrices are never persisted.
"""

from __future__ import annotations

import json
import math
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from backend.scripts import veatic21_controls as controls
from backend.scripts import veatic21_discovery as discovery
from backend.scripts import veatic21_distilled_program as program
from backend.scripts import veatic21_endstate_contract as endstate
from backend.scripts import veatic21_evaluation as evaluation
from backend.scripts import veatic21_features as features
from backend.scripts import veatic21_pca as pca


EXECUTION_SCHEMA_VERSION = "veatic21_numerical_execution_v1"
_FEATURE_CACHE_LIMIT = 3
_FEATURE_CACHE: "OrderedDict[str, PreparedFeatures]" = OrderedDict()


class Veatic21ExecutionError(RuntimeError):
    """Raised when an executor cannot preserve the end-state contract."""


@dataclass(frozen=True)
class PreparedFeatures:
    train_rows: np.ndarray
    test_rows: np.ndarray
    train: features.Veatic21Features
    test: features.Veatic21Features
    recipe: discovery.RecipeSpec
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class FrozenArBundle:
    train_prediction: np.ndarray
    test_prediction: np.ndarray
    identity: Mapping[str, Any]
    checkpoint_path: Path
    best_epoch: int
    cache_hit: bool
    provenance: Mapping[str, Any]


def _atomic_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _modeling() -> Any:
    # Imported lazily because veatic21_modeling exposes the public wrappers
    # which call back into this module.
    from backend.scripts import veatic21_modeling as modeling

    return modeling


def _runner() -> Any:
    # The runner imports veatic21_modeling at module load time.  A lazy import
    # here avoids a module-initialization cycle while retaining its dataclasses
    # and canonical path helpers as the sole orchestration authority.
    from backend.scripts import run_veatic21_endstate as runner

    return runner


def _require_executor_contract(
    *, serial: bool, pca_parent_width: int, pca_slice_policy: str
) -> None:
    if not serial:
        raise Veatic21ExecutionError("VEATIC 2.1 numerical execution must be serial")
    if int(pca_parent_width) != 256:
        raise Veatic21ExecutionError("VEATIC 2.1 requires one PCA-256 parent fit")
    if pca_slice_policy != "leading_components_only":
        raise Veatic21ExecutionError("VEATIC 2.1 requires leading-component PCA slices")


def _shared_root(args: Any) -> Path:
    root = getattr(args, "shared_derived_root", None)
    return Path(root if root is not None else Path(args.output_root) / "derived").resolve()


def _indices_for_videos(video_ids: np.ndarray, selected: Sequence[str]) -> np.ndarray:
    mask = np.isin(np.asarray(video_ids, dtype=str), list(map(str, selected)))
    rows = np.flatnonzero(mask).astype(np.int64)
    if not len(rows):
        raise Veatic21ExecutionError("grouped ownership produced an empty row split")
    return rows


def _recipe(plan: discovery.NestedDiscoveryPlan, name: str) -> discovery.RecipeSpec:
    for item in plan.recipes:
        if item.name == str(name):
            return item
    raise Veatic21ExecutionError(f"unknown selected recipe {name!r}")


def _target_axis(target_name: str) -> str:
    return "arousal" if str(target_name).startswith("future_arousal") else "valence"


def _target_arrays(dataset: Any, target_name: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        values = np.asarray(dataset.target_values[target_name], dtype=np.float32)
        valid = np.asarray(dataset.target_valid[target_name], dtype=bool)
    except KeyError as exc:
        raise Veatic21ExecutionError(f"dense dataset lacks target {target_name}") from exc
    if values.shape != (dataset.rows,) or valid.shape != (dataset.rows,):
        raise Veatic21ExecutionError("dense target arrays are not row aligned")
    if not np.isfinite(values[valid]).all():
        raise Veatic21ExecutionError("dense target contains non-finite eligible rows")
    return values, valid


def _pca_seed(identity: str) -> int:
    return 1 + int(str(identity)[:8], 16) % (2**31 - 2)


def _request_for_scope(
    plan: discovery.NestedDiscoveryPlan,
    *,
    outer_fold: int,
    inner_fold: int | None,
    feature_family: str,
) -> Any:
    scope = "inner_discovery" if inner_fold is not None else "outer_confirmation"
    matches = [
        item
        for item in _runner().pca_requests_for_plan(plan)
        if item.scope == scope
        and item.outer_fold == int(outer_fold)
        and item.inner_fold == inner_fold
        and item.feature_family == feature_family
    ]
    if len(matches) != 1:
        raise Veatic21ExecutionError("PCA request plan did not resolve uniquely")
    return matches[0]


def _finite_pca_scores(transform: pca.PcaTransform, width: int) -> np.ndarray:
    values = np.asarray(transform.values[:, : int(width)], dtype=np.float32).copy()
    family_valid = np.asarray(transform.family_valid_mask, dtype=bool)
    if values.shape != (len(transform.row_indices), int(width)):
        raise Veatic21ExecutionError("PCA leading slice shape drift")
    invalid = ~family_valid
    if np.any(np.isfinite(values[invalid])):
        raise Veatic21ExecutionError("family-invalid PCA rows were not explicitly masked")
    values[invalid] = 0.0
    if not np.isfinite(values).all():
        raise Veatic21ExecutionError("PCA transform contains unexplained non-finite values")
    return values


def _sigmoid(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    clipped = np.clip(array, -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)


def _cache_prepared(key: str, value: PreparedFeatures) -> PreparedFeatures:
    _FEATURE_CACHE[key] = value
    _FEATURE_CACHE.move_to_end(key)
    while len(_FEATURE_CACHE) > _FEATURE_CACHE_LIMIT:
        _FEATURE_CACHE.popitem(last=False)
    return value


def _prepare_features(
    *,
    dataset: Any,
    plan: discovery.NestedDiscoveryPlan,
    recipe: discovery.RecipeSpec,
    outer_fold: int,
    inner_fold: int | None,
    args: Any,
) -> PreparedFeatures:
    request = _request_for_scope(
        plan,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        feature_family=recipe.feature_family,
    )
    cache_key = program.canonical_digest(
        {
            "dataset": dataset.artifact_digest,
            "request": request.parent_identity,
            "recipe": recipe.digest,
            "shared_derived_root": str(_shared_root(args)),
        }
    )
    cached = _FEATURE_CACHE.get(cache_key)
    if cached is not None:
        _FEATURE_CACHE.move_to_end(cache_key)
        return cached

    train_rows = _indices_for_videos(dataset.video_id, request.fit_videos)
    if inner_fold is None:
        test_videos = plan.outer(outer_fold).test_videos
    else:
        test_videos = plan.inner(outer_fold, inner_fold).validation_videos
    test_rows = _indices_for_videos(dataset.video_id, test_videos)
    held_out_rows = _indices_for_videos(dataset.video_id, request.held_out_videos)
    accessor = pca.Veatic21PcaAccessor(
        dataset.cortical,
        dataset.video_id,
        base_family=recipe.feature_family,
        cache_digest=dataset.dataset_seal_digest,
    )
    fitted = pca.fit_or_load_pca(
        accessor,
        train_row_indices=train_rows,
        held_out_row_indices=held_out_rows,
        quality_mask=np.asarray(dataset.quality_valid, dtype=bool),
        output_root=_shared_root(args) / "pca",
        artifact_key=request.parent_identity,
        width=int(request.parent_width),
        seed=_pca_seed(request.parent_identity),
        contract_digest=plan.digest,
    )
    train_transform = fitted.transform(accessor, train_rows)
    test_transform = fitted.transform(accessor, test_rows)
    train_scores = _finite_pca_scores(train_transform, recipe.pca_width)
    test_scores = _finite_pca_scores(test_transform, recipe.pca_width)
    train_block = features.build_veatic21_features(
        row_idx=dataset.row_idx[train_rows],
        video_id=dataset.video_id[train_rows],
        time_seconds=dataset.time_seconds[train_rows],
        pca_scores=train_scores,
        diagnostics=dataset.diagnostics[train_rows],
        pca_width=recipe.pca_width,
        pca_row_idx=train_transform.row_indices,
        diagnostic_row_idx=dataset.row_idx[train_rows],
    )
    test_block = features.build_veatic21_features(
        row_idx=dataset.row_idx[test_rows],
        video_id=dataset.video_id[test_rows],
        time_seconds=dataset.time_seconds[test_rows],
        pca_scores=test_scores,
        diagnostics=dataset.diagnostics[test_rows],
        pca_width=recipe.pca_width,
        pca_row_idx=test_transform.row_indices,
        diagnostic_row_idx=dataset.row_idx[test_rows],
    )
    slice_manifest = _runner().pca_slice_manifest(
        parent_metadata=fitted.metadata,
        requested_width=recipe.pca_width,
    )
    provenance = {
        "pca_parent_identity": str(fitted.metadata["identity_sha256"]),
        "pca_component_path": str(fitted.component_path.resolve()),
        "pca_component_sha256": _modeling().file_sha256(fitted.component_path),
        "pca_metadata_path": str(fitted.metadata_path.resolve()),
        "pca_cache_hit": bool(fitted.cache_hit),
        "pca_slice": slice_manifest,
        "feature_schema_digest": train_block.schema_digest,
        "train_rows_digest": program.array_digest(train_rows),
        "test_rows_digest": program.array_digest(test_rows),
        "delta_invalid_rows_zero_filled_after_sealed_transform": True,
    }
    return _cache_prepared(
        cache_key,
        PreparedFeatures(
            train_rows=train_rows,
            test_rows=test_rows,
            train=train_block,
            test=test_block,
            recipe=recipe,
            provenance=provenance,
        ),
    )


def _view(block: features.Veatic21Features, head: str) -> np.ndarray:
    return block.x_current if head == "current_row_mlp" else block.x_temporal


def _spec(
    *, recipe: discovery.RecipeSpec, objective: str, input_dim: int, residual: bool
) -> Any:
    modeling = _modeling()
    return modeling.ModelSpec(
        head=recipe.head,
        objective=objective,
        input_dim=int(input_dim),
        pca_width=int(recipe.pca_width),
        condition_on_frozen_offset=bool(residual),
    )


def _inner_ownership(
    video_ids: np.ndarray, *, namespace: str
) -> program.InnerVideoOwnership:
    return program.build_inner_video_ownership(
        np.asarray(video_ids, dtype=str), namespace=namespace
    )


def _training_kwargs(args: Any) -> dict[str, Any]:
    return {
        "batch_size": int(args.batch_size),
        "max_epochs": int(args.max_epochs),
        "patience": int(args.patience),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
    }


def _fixed_training_kwargs(args: Any) -> dict[str, Any]:
    return {
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
    }


def _frozen_ar_paths(root: Path) -> tuple[Path, Path]:
    return Path(root) / "scores.npz", Path(root) / "scores.json"


def _frozen_ar(
    *,
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_target: np.ndarray,
    train_valid: np.ndarray,
    train_video_ids: np.ndarray,
    ownership: program.InnerVideoOwnership,
    objective: str,
    seed: int,
    outer_fold: int,
    target_name: str,
    run_identity_digest: str,
    output_root: Path,
    args: Any,
) -> FrozenArBundle:
    modeling = _modeling()
    spec = modeling.ModelSpec(
        head=modeling.AR_HEAD,
        objective=objective,
        input_dim=7,
        pca_width=None,
        condition_on_frozen_offset=False,
    )
    x_train = np.asarray(train_x, dtype=np.float32)
    x_test = np.asarray(test_x, dtype=np.float32)
    target = np.asarray(train_target, dtype=np.float32)
    valid = np.asarray(train_valid, dtype=bool)
    videos = np.asarray(train_video_ids, dtype=str)
    inner_train, inner_val = ownership.eligible_indices(videos, valid)
    identity = {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "kind": "honest_neural_ar7",
        "run_identity_digest": run_identity_digest,
        "outer_fold": int(outer_fold),
        "target": target_name,
        "objective": objective,
        "seed": int(seed),
        "ownership_digest": ownership.digest,
        "train_x_digest": program.array_digest(x_train),
        "test_x_digest": program.array_digest(x_test),
        "target_digest": program.array_digest(target),
        "valid_digest": program.array_digest(valid.astype(np.uint8)),
    }
    scores_path, manifest_path = _frozen_ar_paths(output_root)
    identity_digest = program.canonical_digest(identity)
    if scores_path.exists() or manifest_path.exists():
        if not scores_path.is_file() or not manifest_path.is_file():
            raise Veatic21ExecutionError("incomplete frozen-AR score cache")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("identity") != identity or manifest.get("identity_digest") != identity_digest:
            raise Veatic21ExecutionError("frozen-AR resume identity mismatch")
        if manifest.get("score_file_sha256") != modeling.file_sha256(scores_path):
            raise Veatic21ExecutionError("frozen-AR score checksum drift")
        with np.load(scores_path, allow_pickle=False) as bundle:
            train_prediction = np.asarray(bundle["train_prediction"], dtype=np.float32)
            test_prediction = np.asarray(bundle["test_prediction"], dtype=np.float32)
        if (
            train_prediction.shape != (len(x_train),)
            or test_prediction.shape != (len(x_test),)
            or manifest.get("train_prediction_digest")
            != program.array_digest(train_prediction)
            or manifest.get("test_prediction_digest")
            != program.array_digest(test_prediction)
        ):
            raise Veatic21ExecutionError("frozen-AR cached score shape/digest drift")
        checkpoint = Path(manifest["final_checkpoint"])
        if not checkpoint.is_file() or manifest.get("final_checkpoint_sha256") != modeling.file_sha256(checkpoint):
            raise Veatic21ExecutionError("frozen-AR final checkpoint drift")
        return FrozenArBundle(
            train_prediction=train_prediction,
            test_prediction=test_prediction,
            identity=dict(manifest["frozen_ar_identity"]),
            checkpoint_path=checkpoint,
            best_epoch=int(manifest["best_epoch"]),
            cache_hit=True,
            provenance=manifest,
        )

    output_root = Path(output_root)
    primary = modeling.train_scalar_model(
        train_x=x_train,
        test_x=x_test,
        train_target=target,
        train_loss_mask=valid,
        inner_train_idx=inner_train,
        inner_val_idx=inner_val,
        spec=spec,
        seed=int(seed),
        checkpoint_path=output_root / "selection" / "ar.npz",
        artifact_identity={**identity, "phase": "inner_epoch_selection"},
        refit_after_selection=False,
        **_training_kwargs(args),
    )
    best_epoch = int(primary.best_epoch)
    honest = np.full(len(x_train), np.nan, dtype=np.float32)
    ownership_train, ownership_val = ownership.row_masks(videos)
    honest[ownership_val] = primary.train_prediction[ownership_val]
    crossfit_provenance: list[dict[str, Any]] = []
    for scope in program.build_ar_crossfit_video_folds(ownership, fold_count=5):
        fit_mask = valid & np.isin(videos, list(scope.fit_videos))
        prediction_mask = ownership_train & np.isin(videos, list(scope.prediction_videos))
        crossfit = modeling.refit_scalar_model_fixed_epochs(
            train_x=x_train,
            train_target=target,
            train_loss_mask=fit_mask,
            spec=spec,
            seed=int(seed) + scope.fold * 100_003,
            epochs=best_epoch,
            checkpoint_path=output_root / "crossfit" / f"fold_{scope.fold}.npz",
            artifact_identity={
                **identity,
                "phase": "whole_video_crossfit",
                "scope_digest": scope.digest,
            },
            **_fixed_training_kwargs(args),
        )
        honest[prediction_mask] = crossfit.train_prediction[prediction_mask]
        crossfit_provenance.append(
            {
                "scope_digest": scope.digest,
                "fit_videos": list(scope.fit_videos),
                "prediction_videos": list(scope.prediction_videos),
                "checkpoint_sha256": crossfit.checkpoint_sha256,
                "prediction_rows": int(np.count_nonzero(prediction_mask)),
            }
        )
    if not np.isfinite(honest).all():
        raise Veatic21ExecutionError("neural AR cross-fitting left non-finite train scores")

    combined_x = np.concatenate([x_train, x_test], axis=0)
    combined_target = np.concatenate(
        [target, np.zeros(len(x_test), dtype=np.float32)], axis=0
    )
    combined_mask = np.concatenate(
        [valid, np.zeros(len(x_test), dtype=bool)], axis=0
    )
    final = modeling.refit_scalar_model_fixed_epochs(
        train_x=combined_x,
        train_target=combined_target,
        train_loss_mask=combined_mask,
        spec=spec,
        seed=int(seed),
        epochs=best_epoch,
        checkpoint_path=output_root / "final" / "ar.npz",
        artifact_identity={**identity, "phase": "all_owned_train_final_inference_ar"},
        **_fixed_training_kwargs(args),
    )
    final_test = final.train_prediction[len(x_train) :].astype(np.float32)
    frozen_identity_payload = {
        "outer_fold": int(outer_fold),
        "target_name": target_name,
        "seed": int(seed),
        "objective": objective,
        "model_family": "mlx_target_specific_neural_ar7",
        "checkpoint_digest": final.checkpoint_sha256,
        "ownership_digest": ownership.digest,
        "train_prediction_digest": program.array_digest(honest),
        "test_prediction_digest": program.array_digest(final_test),
        "all_outer_train_predictions_out_of_video_fit": True,
    }
    frozen_identity = {
        **frozen_identity_payload,
        "identity_digest": program.canonical_digest(frozen_identity_payload),
    }
    _atomic_npz(
        scores_path,
        train_prediction=honest.astype(np.float32),
        test_prediction=final_test,
    )
    manifest = {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "identity": identity,
        "identity_digest": identity_digest,
        "best_epoch": best_epoch,
        "selection_checkpoint": str(primary.checkpoint_path.resolve()),
        "selection_checkpoint_sha256": primary.checkpoint_sha256,
        "final_checkpoint": str(final.checkpoint_path.resolve()),
        "final_checkpoint_sha256": final.checkpoint_sha256,
        "train_prediction_digest": program.array_digest(honest),
        "test_prediction_digest": program.array_digest(final_test),
        "crossfit": crossfit_provenance,
        "frozen_ar_identity": frozen_identity,
        "score_file": str(scores_path.resolve()),
        "score_file_sha256": modeling.file_sha256(scores_path),
        "all_outer_train_predictions_out_of_video_fit": True,
        "final_test_fit_uses_outer_train_only": True,
    }
    _atomic_json(manifest_path, manifest)
    return FrozenArBundle(
        train_prediction=honest.astype(np.float32),
        test_prediction=final_test,
        identity=frozen_identity,
        checkpoint_path=final.checkpoint_path,
        best_epoch=best_epoch,
        cache_hit=False,
        provenance=manifest,
    )


def _train_head(
    *,
    prepared: PreparedFeatures,
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_target: np.ndarray,
    train_valid: np.ndarray,
    ownership: program.InnerVideoOwnership,
    objective: str,
    seed: int,
    checkpoint_path: Path,
    artifact_identity: Mapping[str, Any],
    args: Any,
    frozen_train_offset: np.ndarray | None = None,
    frozen_test_offset: np.ndarray | None = None,
    head_override: str | None = None,
) -> Any:
    modeling = _modeling()
    head = str(head_override or prepared.recipe.head)
    recipe = (
        prepared.recipe
        if head == prepared.recipe.head
        else discovery.RecipeSpec(
            order=prepared.recipe.order,
            name=prepared.recipe.name + "__" + head,
            feature_family=prepared.recipe.feature_family,
            pca_width=prepared.recipe.pca_width,
            head=head,
            causal_rows=1 if head == "current_row_mlp" else 5,
            complexity_score=prepared.recipe.complexity_score,
            payload_json=prepared.recipe.payload_json,
            digest=program.canonical_digest(
                {"recipe": prepared.recipe.digest, "head_override": head}
            ),
        )
    )
    spec = _spec(
        recipe=recipe,
        objective=objective,
        input_dim=int(train_x.shape[1]),
        residual=frozen_train_offset is not None,
    )
    inner_train, inner_val = ownership.eligible_indices(
        prepared.train.video_id, np.asarray(train_valid, dtype=bool)
    )
    return modeling.train_scalar_model(
        train_x=np.asarray(train_x, dtype=np.float32),
        test_x=np.asarray(test_x, dtype=np.float32),
        train_target=np.asarray(train_target, dtype=np.float32),
        train_loss_mask=np.asarray(train_valid, dtype=bool),
        inner_train_idx=inner_train,
        inner_val_idx=inner_val,
        spec=spec,
        seed=int(seed),
        checkpoint_path=Path(checkpoint_path),
        artifact_identity=dict(artifact_identity),
        frozen_train_offset=frozen_train_offset,
        frozen_test_offset=frozen_test_offset,
        refit_after_selection=True,
        **_training_kwargs(args),
    )


def _event_threshold(values: np.ndarray, valid: np.ndarray) -> float:
    eligible = np.asarray(values, dtype=np.float32)[np.asarray(valid, dtype=bool)]
    if len(eligible) < 2 or not np.isfinite(eligible).all():
        raise Veatic21ExecutionError("event threshold has insufficient train-only values")
    return evaluation.train_q90_threshold(eligible)


def _score_metrics(
    *, y_true: np.ndarray, prediction: np.ndarray, event_threshold: float
) -> evaluation.MetricScores:
    return evaluation.score_end_state_metrics(
        y_true=np.asarray(y_true, dtype=np.float32),
        prediction=np.asarray(prediction, dtype=np.float32),
        event_threshold=float(event_threshold),
    )


def _label_permutation(
    *,
    prepared: PreparedFeatures,
    target: np.ndarray,
    valid: np.ndarray,
    namespace: controls.ControlNamespace,
    privileged: bool,
    frozen_ar_identity: Mapping[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any]]:
    mask = np.asarray(valid, dtype=bool)
    eligible = np.flatnonzero(mask).astype(np.int64)
    control = controls.build_whole_video_label_permutation(
        row_idx=prepared.train.row_idx[eligible],
        video_id=prepared.train.video_id[eligible],
        time_seconds=prepared.train.time_seconds[eligible],
        target=np.asarray(target, dtype=np.float32)[eligible],
        namespace=namespace,
        privileged=privileged,
        frozen_ar_identity=frozen_ar_identity,
    )
    output = np.zeros(len(target), dtype=np.float32)
    output[eligible] = control.target
    return output, mask, control.record


def _control_record(control: controls.SealedFeatureControl) -> Mapping[str, Any]:
    return {
        "record": dict(control.record),
        "parameter_digests": {
            name: controls.array_digest(value)
            for name, value in sorted(control.parameter_arrays.items())
        },
        "full_control_matrices_persisted": False,
    }


def _probability_if_binary(result: Any, objective: str) -> np.ndarray:
    if objective == _modeling().BINARY:
        if result.test_probability is None:
            raise Veatic21ExecutionError("binary model did not return probabilities")
        return np.asarray(result.test_probability, dtype=np.float32)
    return np.asarray(result.test_prediction, dtype=np.float32)


def execute_nested_discovery(
    *,
    args: Any,
    plan: discovery.NestedDiscoveryPlan,
    dataset: Any,
    run_identity_digest: str,
    output_root: Path,
    pca_parent_width: int,
    pca_slice_policy: str,
    serial: bool,
) -> Sequence[discovery.DiscoveryScoreRow]:
    """Execute the exact nested matrix, or its explicitly bounded smoke slice."""

    _require_executor_contract(
        serial=serial,
        pca_parent_width=pca_parent_width,
        pca_slice_policy=pca_slice_policy,
    )
    if dataset is None:
        raise Veatic21ExecutionError("nested discovery requires the validated dense dataset")
    modeling = _modeling()
    modeling.require_mlx_gpu()
    target_names = plan.targets[:1] if bool(args.smoke) else plan.targets
    outer_folds = plan.outer_folds[:1] if bool(args.smoke) else plan.outer_folds
    seeds = plan.discovery_seeds[:1] if bool(args.smoke) else plan.discovery_seeds
    score_rows: list[discovery.DiscoveryScoreRow] = []

    for target_name in target_names:
        all_target, all_target_valid = _target_arrays(dataset, target_name)
        axis_signal = np.asarray(
            dataset.arousal if _target_axis(target_name) == "arousal" else dataset.valence,
            dtype=np.float32,
        )
        ar_x_all, ar_context_all, _ = program.canonical_ar_history_features(
            axis_signal, dataset.video_id
        )
        for outer in outer_folds:
            for inner in outer.inner_folds:
                for recipe in plan.recipes:
                    prepared = _prepare_features(
                        dataset=dataset,
                        plan=plan,
                        recipe=recipe,
                        outer_fold=outer.outer_fold,
                        inner_fold=inner.fold,
                        args=args,
                    )
                    train_target = all_target[prepared.train_rows]
                    test_target = all_target[prepared.test_rows]
                    base_train_valid = (
                        all_target_valid[prepared.train_rows]
                        & dataset.quality_valid[prepared.train_rows]
                    )
                    base_test_valid = (
                        all_target_valid[prepared.test_rows]
                        & dataset.quality_valid[prepared.test_rows]
                    )
                    for protocol in plan.protocols:
                        objective = (
                            modeling.BINARY
                            if protocol == discovery.PRIVILEGED_BINARY
                            else modeling.CONTINUOUS
                        )
                        privileged = protocol != discovery.ZERO_LABEL_CONTINUOUS
                        train_valid = base_train_valid.copy()
                        test_valid = base_test_valid.copy()
                        train_offset: np.ndarray | None = None
                        test_offset: np.ndarray | None = None
                        threshold = _event_threshold(train_target, train_valid)
                        model_target = train_target.copy()
                        if objective == modeling.BINARY:
                            model_target = (train_target >= threshold).astype(np.float32)
                        seed_protocol = list(plan.protocols).index(protocol) * 10_000_019
                        for seed in seeds:
                            namespace = (
                                f"veatic21|discovery|outer{outer.outer_fold}|inner{inner.fold}|"
                                f"{target_name}|{protocol}|seed{seed}"
                            )
                            ownership = _inner_ownership(
                                prepared.train.video_id, namespace=namespace
                            )
                            if privileged:
                                train_valid = base_train_valid & ar_context_all[prepared.train_rows]
                                test_valid = base_test_valid & ar_context_all[prepared.test_rows]
                                threshold = _event_threshold(train_target, train_valid)
                                model_target = (
                                    (train_target >= threshold).astype(np.float32)
                                    if objective == modeling.BINARY
                                    else train_target.copy()
                                )
                                ar = _frozen_ar(
                                    train_x=ar_x_all[prepared.train_rows],
                                    test_x=ar_x_all[prepared.test_rows],
                                    train_target=model_target,
                                    train_valid=train_valid,
                                    train_video_ids=prepared.train.video_id,
                                    ownership=ownership,
                                    objective=objective,
                                    seed=int(seed) + seed_protocol,
                                    outer_fold=outer.outer_fold,
                                    target_name=target_name,
                                    run_identity_digest=run_identity_digest,
                                    output_root=(
                                        Path(output_root)
                                        / "ar"
                                        / target_name
                                        / protocol
                                        / f"outer_{outer.outer_fold}"
                                        / f"inner_{inner.fold}"
                                        / f"seed_{seed}"
                                    ),
                                    args=args,
                                )
                                train_offset = ar.train_prediction
                                test_offset = ar.test_prediction
                            train_x = _view(prepared.train, recipe.head)
                            test_x = _view(prepared.test, recipe.head)
                            result = _train_head(
                                prepared=prepared,
                                train_x=train_x,
                                test_x=test_x,
                                train_target=model_target,
                                train_valid=train_valid,
                                ownership=ownership,
                                objective=objective,
                                seed=int(seed) + seed_protocol,
                                checkpoint_path=(
                                    Path(output_root)
                                    / "models"
                                    / target_name
                                    / protocol
                                    / f"outer_{outer.outer_fold}"
                                    / f"inner_{inner.fold}"
                                    / recipe.name
                                    / f"seed_{seed}.npz"
                                ),
                                artifact_identity={
                                    "schema_version": EXECUTION_SCHEMA_VERSION,
                                    "stage": "nested_discovery",
                                    "run_identity_digest": run_identity_digest,
                                    "plan_digest": plan.digest,
                                    "target": target_name,
                                    "protocol": protocol,
                                    "outer_fold": outer.outer_fold,
                                    "inner_fold": inner.fold,
                                    "recipe_digest": recipe.digest,
                                    "seed": int(seed),
                                    "ownership_digest": ownership.digest,
                                },
                                args=args,
                                frozen_train_offset=train_offset,
                                frozen_test_offset=test_offset,
                            )
                            prediction = _probability_if_binary(result, objective)[test_valid]
                            scored = _score_metrics(
                                y_true=test_target[test_valid],
                                prediction=prediction,
                                event_threshold=threshold,
                            )
                            metrics = (
                                {discovery.TRAIN_Q90_PR_AUC: scored.train_q90_pr_auc}
                                if protocol == discovery.PRIVILEGED_BINARY
                                else {
                                    discovery.SPEARMAN: scored.spearman,
                                    discovery.TOP5_LIFT: scored.top_5pct_lift,
                                }
                            )
                            score_rows.append(
                                discovery.make_discovery_score_row(
                                    plan,
                                    target=target_name,
                                    protocol=protocol,
                                    outer_fold=outer.outer_fold,
                                    recipe=recipe.name,
                                    inner_fold=inner.fold,
                                    seed=int(seed),
                                    metrics=metrics,
                                )
                            )
    return tuple(score_rows)


def _confirmation_ensemble(
    *, cell: Any, output_root: Path, run_identity_digest: str
) -> Mapping[str, Any]:
    runner = _runner()
    arrays: list[np.ndarray] = []
    reference: dict[str, np.ndarray] | None = None
    manifests: list[Mapping[str, Any]] = []
    for seed in cell.member_seeds:
        member_path = (
            Path(output_root)
            / "predictions"
            / cell.endpoint
            / cell.target
            / f"fold_{cell.outer_fold}"
            / cell.lane
            / f"seed_{seed}.npz"
        )
        if not member_path.is_file() or not member_path.with_suffix(".json").is_file():
            raise Veatic21ExecutionError("ensemble member prediction is not sealed")
        manifest = json.loads(member_path.with_suffix(".json").read_text(encoding="utf-8"))
        if manifest.get("run_identity_digest") != run_identity_digest:
            raise Veatic21ExecutionError("ensemble member belongs to another run identity")
        with np.load(member_path, allow_pickle=False) as bundle:
            current = {name: np.asarray(bundle[name]) for name in bundle.files}
        if reference is None:
            reference = current
        elif any(
            not np.array_equal(current[name], reference[name])
            for name in ("row_index", "video_id", "y_true")
        ):
            raise Veatic21ExecutionError("ensemble member row/truth alignment drift")
        arrays.append(np.asarray(current["prediction"], dtype=np.float32))
        manifests.append(manifest)
    if reference is None:
        raise Veatic21ExecutionError("ensemble has no members")
    prediction = np.mean(np.stack(arrays, axis=0), axis=0, dtype=np.float64).astype(np.float32)
    return {
        "row_indices": np.asarray(reference["row_index"], dtype=np.int64),
        "video_ids": np.asarray(reference["video_id"], dtype=str),
        "y_true": np.asarray(reference["y_true"], dtype=np.float32),
        "prediction": prediction,
        "event_threshold": float(manifests[0]["event_threshold"]),
        "checkpoint": None,
        "provenance": {
            "schema_version": EXECUTION_SCHEMA_VERSION,
            "kind": "fixed_arithmetic_mean_ensemble",
            "member_seeds": list(cell.member_seeds),
            "member_manifest_digests": [item["manifest_digest"] for item in manifests],
            "post_confirmation_weight_search": False,
        },
    }


def execute_confirmation_cell(
    *,
    args: Any,
    plan: discovery.NestedDiscoveryPlan,
    selection: Any,
    cell: Any,
    dataset: Any,
    run_identity_digest: str,
    output_root: Path,
    pca_parent_width: int,
    pca_slice_policy: str,
    serial: bool,
) -> Mapping[str, Any]:
    """Train or assemble one exact confirmation matrix cell."""

    _require_executor_contract(
        serial=serial,
        pca_parent_width=pca_parent_width,
        pca_slice_policy=pca_slice_policy,
    )
    if dataset is None:
        raise Veatic21ExecutionError("confirmation requires the validated dense dataset")
    if cell.row_kind == _runner().ENSEMBLE_KIND:
        return _confirmation_ensemble(
            cell=cell,
            output_root=output_root,
            run_identity_digest=run_identity_digest,
        )

    modeling = _modeling()
    modeling.require_mlx_gpu()
    recipe = _recipe(plan, cell.recipe)
    prepared = _prepare_features(
        dataset=dataset,
        plan=plan,
        recipe=recipe,
        outer_fold=cell.outer_fold,
        inner_fold=None,
        args=args,
    )
    target_all, target_valid_all = _target_arrays(dataset, cell.target)
    train_target = target_all[prepared.train_rows]
    test_target = target_all[prepared.test_rows]
    base_train_valid = (
        target_valid_all[prepared.train_rows]
        & dataset.quality_valid[prepared.train_rows]
    )
    base_test_valid = (
        target_valid_all[prepared.test_rows]
        & dataset.quality_valid[prepared.test_rows]
    )
    seed = int(cell.seed)
    namespace = controls.ControlNamespace(
        target=cell.target,
        fold=cell.outer_fold,
        seed=seed,
        endpoint=cell.endpoint,
        lane=cell.lane,
    )
    ownership = program.build_member_inner_video_ownership(
        prepared.train.video_id,
        outer_fold=cell.outer_fold,
        target_name=cell.target,
        seed=seed,
    )
    objective = modeling.BINARY if cell.endpoint == discovery.PRIVILEGED_BINARY else modeling.CONTINUOUS
    privileged = cell.endpoint != discovery.ZERO_LABEL_CONTINUOUS or cell.descriptive_only
    train_valid = base_train_valid.copy()
    test_valid = base_test_valid.copy()
    threshold = _event_threshold(train_target, train_valid)
    model_target = (
        (train_target >= threshold).astype(np.float32)
        if objective == modeling.BINARY
        else train_target.copy()
    )
    ar: FrozenArBundle | None = None
    train_offset: np.ndarray | None = None
    test_offset: np.ndarray | None = None
    if privileged:
        axis = np.asarray(
            dataset.arousal if _target_axis(cell.target) == "arousal" else dataset.valence,
            dtype=np.float32,
        )
        ar_x_all, ar_context_all, ar_audit = program.canonical_ar_history_features(
            axis, dataset.video_id
        )
        train_valid &= ar_context_all[prepared.train_rows]
        test_valid &= ar_context_all[prepared.test_rows]
        threshold = _event_threshold(train_target, train_valid)
        model_target = (
            (train_target >= threshold).astype(np.float32)
            if objective == modeling.BINARY
            else train_target.copy()
        )
        ar = _frozen_ar(
            train_x=ar_x_all[prepared.train_rows],
            test_x=ar_x_all[prepared.test_rows],
            train_target=model_target,
            train_valid=train_valid,
            train_video_ids=prepared.train.video_id,
            ownership=ownership,
            objective=objective,
            seed=seed,
            outer_fold=cell.outer_fold,
            target_name=cell.target,
            run_identity_digest=run_identity_digest,
            output_root=(
                Path(output_root)
                / "frozen_ar"
                / cell.endpoint
                / cell.target
                / f"fold_{cell.outer_fold}"
                / f"seed_{seed}"
            ),
            args=args,
        )
        train_offset = ar.train_prediction
        test_offset = ar.test_prediction
    else:
        ar_audit = None

    checkpoint: Path | None = None
    best_epoch: int | None = None
    control_provenance: Mapping[str, Any] | None = None
    if cell.lane == "frozen_ar_only":
        if ar is None:
            raise Veatic21ExecutionError("frozen_ar_only is not a privileged lane")
        raw_prediction = ar.test_prediction
        prediction = (
            _sigmoid(raw_prediction)
            if objective == modeling.BINARY
            else raw_prediction
        )
        checkpoint = ar.checkpoint_path
        best_epoch = ar.best_epoch
    else:
        train_block = prepared.train
        test_block = prepared.test
        lane_target = model_target
        lane_valid = train_valid
        head = recipe.head
        if cell.lane in ("shuffled_pca_residual", "sequence_shuffled_supervised_temporal"):
            control = controls.build_sequence_shuffled_pca_control(
                prepared.train,
                prepared.test,
                namespace=namespace,
                privileged=privileged,
                frozen_ar_identity=(ar.identity if ar is not None else None),
            )
            train_block, test_block = control.train, control.test
            control_provenance = _control_record(control)
        elif cell.lane == "random_pca_residual":
            if ar is None:
                raise Veatic21ExecutionError("random PCA residual requires frozen AR")
            control = controls.build_matched_random_pca_control(
                prepared.train,
                prepared.test,
                namespace=namespace,
                frozen_ar_identity=ar.identity,
                train_fit_mask=train_valid,
            )
            train_block, test_block = control.train, control.test
            control_provenance = _control_record(control)
        elif cell.lane == "train_only_video_mean_residual":
            if ar is None:
                raise Veatic21ExecutionError("video-mean residual requires frozen AR")
            control = controls.build_train_only_video_mean_control(
                prepared.train,
                prepared.test,
                namespace=namespace,
                frozen_ar_identity=ar.identity,
                train_fit_mask=train_valid,
            )
            train_block, test_block = control.train, control.test
            control_provenance = _control_record(control)
        elif cell.lane in (
            "diagnostics_only_residual",
            "diagnostics_only_supervised_temporal",
        ):
            control = controls.build_diagnostics_only_control(
                prepared.train,
                prepared.test,
                namespace=namespace,
                privileged=privileged,
                frozen_ar_identity=(ar.identity if ar is not None else None),
            )
            train_block, test_block = control.train, control.test
            control_provenance = _control_record(control)
        elif cell.lane == "no_video_supervised_temporal":
            control = controls.build_no_video_control(
                prepared.train, prepared.test, namespace=namespace
            )
            train_block, test_block = control.train, control.test
            control_provenance = _control_record(control)
        elif cell.lane in (
            "label_permutation_residual",
            "label_permutation_supervised_temporal",
        ):
            lane_target, lane_valid, record = _label_permutation(
                prepared=prepared,
                target=model_target,
                valid=train_valid,
                namespace=namespace,
                privileged=privileged,
                frozen_ar_identity=(ar.identity if ar is not None else None),
            )
            control_provenance = {
                "record": dict(record),
                "full_control_matrices_persisted": False,
            }
        elif cell.lane == "video_supervised_current_row":
            head = "current_row_mlp"
        elif cell.lane not in (
            "real_residual",
            "video_supervised_temporal",
            "privileged_teacher_ceiling",
        ):
            raise Veatic21ExecutionError(f"unsupported confirmation lane {cell.lane}")

        train_x = _view(train_block, head)
        test_x = _view(test_block, head)
        result = _train_head(
            prepared=prepared,
            train_x=train_x,
            test_x=test_x,
            train_target=lane_target,
            train_valid=lane_valid,
            ownership=ownership,
            objective=objective,
            seed=seed,
            checkpoint_path=_runner().checkpoint_path(output_root, cell),
            artifact_identity={
                "schema_version": EXECUTION_SCHEMA_VERSION,
                "stage": "confirmation",
                "run_identity_digest": run_identity_digest,
                "selection_digest": selection.artifact_digest,
                "cell_key": cell.key,
                "recipe_digest": recipe.digest,
                "ownership_digest": ownership.digest,
                "frozen_ar_identity": ar.identity if ar is not None else None,
                "control_record_digest": (
                    program.canonical_digest(control_provenance)
                    if control_provenance is not None
                    else None
                ),
            },
            args=args,
            frozen_train_offset=train_offset,
            frozen_test_offset=test_offset,
            head_override=head,
        )
        prediction = _probability_if_binary(result, objective)
        checkpoint = result.checkpoint_path
        best_epoch = int(result.best_epoch)

    if not np.any(test_valid):
        raise Veatic21ExecutionError("confirmation cell has no eligible held-out rows")
    protocol = {
        discovery.PRIVILEGED_CONTINUOUS: discovery.PRIVILEGED_CONTINUOUS,
        discovery.PRIVILEGED_BINARY: discovery.PRIVILEGED_BINARY,
        discovery.ZERO_LABEL_CONTINUOUS: discovery.ZERO_LABEL_CONTINUOUS,
    }[cell.endpoint]
    return {
        "row_indices": prepared.test_rows[test_valid],
        "video_ids": prepared.test.video_id[test_valid],
        "y_true": test_target[test_valid],
        "prediction": np.asarray(prediction, dtype=np.float32)[test_valid],
        "event_threshold": threshold,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "provenance": {
            "schema_version": EXECUTION_SCHEMA_VERSION,
            "protocol": protocol,
            "recipe": recipe.name,
            "recipe_order": recipe.order,
            "best_epoch": best_epoch,
            "outer_train_only_event_threshold": True,
            "ownership_digest": ownership.digest,
            "pca": dict(prepared.provenance),
            "frozen_ar_identity": ar.identity if ar is not None else None,
            "frozen_ar_cache_hit": ar.cache_hit if ar is not None else None,
            "ar_feature_audit": ar_audit,
            "control": control_provenance,
            "response_free_inference": bool(cell.response_free),
            "descriptive_only": bool(cell.descriptive_only),
            "full_control_matrices_persisted": False,
        },
    }


def _all_video_features(
    *, dataset: Any, plan: discovery.NestedDiscoveryPlan, recipe: discovery.RecipeSpec, args: Any
) -> tuple[features.Veatic21Features, Mapping[str, Any]]:
    identity_payload = {
        "scope": "all_124_refit",
        "plan_digest": plan.digest,
        "feature_family": recipe.feature_family,
        "parent_width": 256,
        "fit_videos": sorted(np.unique(dataset.video_id).astype(str).tolist()),
    }
    identity = program.canonical_digest(identity_payload)
    accessor = pca.Veatic21PcaAccessor(
        dataset.cortical,
        dataset.video_id,
        base_family=recipe.feature_family,
        cache_digest=dataset.dataset_seal_digest,
    )
    fitted = pca.fit_or_load_pca(
        accessor,
        train_row_indices=dataset.row_idx,
        held_out_row_indices=np.zeros(0, dtype=np.int64),
        quality_mask=np.asarray(dataset.quality_valid, dtype=bool),
        output_root=Path(args.output_root) / "final" / "pca",
        artifact_key=identity,
        width=256,
        seed=_pca_seed(identity),
        contract_digest=plan.digest,
    )
    transformed = fitted.transform(accessor, dataset.row_idx)
    scores = _finite_pca_scores(transformed, recipe.pca_width)
    block = features.build_veatic21_features(
        row_idx=dataset.row_idx,
        video_id=dataset.video_id,
        time_seconds=dataset.time_seconds,
        pca_scores=scores,
        diagnostics=dataset.diagnostics,
        pca_width=recipe.pca_width,
        pca_row_idx=transformed.row_indices,
        diagnostic_row_idx=dataset.row_idx,
    )
    return block, {
        "parent_identity": str(fitted.metadata["identity_sha256"]),
        "component": str(fitted.component_path.resolve()),
        "metadata": str(fitted.metadata_path.resolve()),
        "slice_width": recipe.pca_width,
        "fit_all_124": True,
    }


def execute_all124_refit(
    *,
    args: Any,
    plan: discovery.NestedDiscoveryPlan,
    export_contract: Mapping[str, Any],
    dataset: Any,
    output_root: Path,
    pca_parent_width: int,
    pca_slice_policy: str,
    serial: bool,
    score_training_rows: bool,
) -> Mapping[str, Any]:
    """Refit frozen selections from scratch on all 124 videos without scoring."""

    _require_executor_contract(
        serial=serial,
        pca_parent_width=pca_parent_width,
        pca_slice_policy=pca_slice_policy,
    )
    if score_training_rows:
        raise Veatic21ExecutionError("all-124 refit cannot report in-sample metrics")
    if dataset is None or dataset.rows <= 0:
        raise Veatic21ExecutionError("all-124 refit requires the validated dense dataset")
    if int(export_contract.get("video_count", 0)) != 124 or not export_contract.get("all_video_refit"):
        raise Veatic21ExecutionError("all-124 export contract is incomplete")
    modeling = _modeling()
    modeling.require_mlx_gpu()
    fixed_epochs = {
        (str(row["target"]), str(row["protocol"])): int(row["fixed_epoch"])
        for row in export_contract["fixed_epochs"]
    }
    selected = {
        (str(row["target"]), str(row["protocol"])): str(row["selected_recipe"])
        for row in export_contract["global_selections"]
    }
    artifacts: set[Path] = set()
    manifests: list[dict[str, Any]] = []
    feature_cache: dict[str, tuple[features.Veatic21Features, Mapping[str, Any]]] = {}

    for key, recipe_name in sorted(selected.items()):
        target_name, protocol = key
        recipe = _recipe(plan, recipe_name)
        if recipe_name not in feature_cache:
            feature_cache[recipe_name] = _all_video_features(
                dataset=dataset, plan=plan, recipe=recipe, args=args
            )
        block, pca_provenance = feature_cache[recipe_name]
        target, valid = _target_arrays(dataset, target_name)
        valid = valid & np.asarray(dataset.quality_valid, dtype=bool)
        threshold = _event_threshold(target, valid)
        objective = (
            modeling.BINARY if protocol == discovery.PRIVILEGED_BINARY else modeling.CONTINUOUS
        )
        model_target = (
            (target >= threshold).astype(np.float32)
            if objective == modeling.BINARY
            else target.copy()
        )
        privileged = protocol != discovery.ZERO_LABEL_CONTINUOUS
        if privileged:
            axis = dataset.arousal if _target_axis(target_name) == "arousal" else dataset.valence
            ar_x, ar_context, _ = program.canonical_ar_history_features(axis, dataset.video_id)
            valid &= ar_context
        else:
            ar_x = None
        seeds = (
            endstate.ZERO_LABEL_CONFIRMATION_SEEDS
            if protocol == discovery.ZERO_LABEL_CONTINUOUS
            else endstate.PRIVILEGED_CONFIRMATION_SEEDS
        )
        epochs = fixed_epochs[key]
        for seed in seeds:
            train_offset: np.ndarray | None = None
            ar_checkpoint: Path | None = None
            if privileged:
                ownership = program.build_inner_video_ownership(
                    dataset.video_id,
                    namespace=f"veatic21|all124|{target_name}|{protocol}|seed{seed}",
                )
                ar = _frozen_ar(
                    train_x=ar_x,
                    test_x=np.empty((0, 7), dtype=np.float32),
                    train_target=model_target,
                    train_valid=valid,
                    train_video_ids=dataset.video_id,
                    ownership=ownership,
                    objective=objective,
                    seed=int(seed),
                    outer_fold=0,
                    target_name=target_name,
                    run_identity_digest=str(export_contract["run_identity_digest"]),
                    output_root=Path(output_root) / "ar" / target_name / protocol / f"seed_{seed}",
                    args=args,
                )
                train_offset = ar.train_prediction
                ar_checkpoint = ar.checkpoint_path
                artifacts.update(
                    {ar.checkpoint_path, ar.checkpoint_path.with_suffix(".json"), ar.checkpoint_path.with_name(ar.checkpoint_path.stem + "__normalization.npz")}
                )
            x = _view(block, recipe.head)
            spec = _spec(
                recipe=recipe,
                objective=objective,
                input_dim=x.shape[1],
                residual=privileged,
            )
            checkpoint = Path(output_root) / "models" / target_name / protocol / f"seed_{seed}.npz"
            result = modeling.refit_scalar_model_fixed_epochs(
                train_x=x,
                train_target=model_target,
                train_loss_mask=valid,
                spec=spec,
                seed=int(seed),
                epochs=epochs,
                checkpoint_path=checkpoint,
                artifact_identity={
                    "schema_version": EXECUTION_SCHEMA_VERSION,
                    "stage": "all_124_refit",
                    "export_contract_digest": export_contract["export_contract_digest"],
                    "target": target_name,
                    "protocol": protocol,
                    "recipe_digest": recipe.digest,
                    "seed": int(seed),
                    "event_threshold": threshold,
                    "pca": pca_provenance,
                    "frozen_ar_checkpoint": str(ar_checkpoint) if ar_checkpoint else None,
                    "in_sample_metrics_reported": False,
                },
                frozen_train_offset=train_offset,
                **_fixed_training_kwargs(args),
            )
            artifacts.update(
                {result.checkpoint_path, result.manifest_path, result.normalization_path}
            )
            model_manifest = {
                "schema_version": EXECUTION_SCHEMA_VERSION,
                "target": target_name,
                "protocol": protocol,
                "recipe": recipe.name,
                "seed": int(seed),
                "fixed_epoch": epochs,
                "checkpoint": str(result.checkpoint_path.resolve()),
                "checkpoint_sha256": result.checkpoint_sha256,
                "event_threshold": threshold,
                "pca": pca_provenance,
                "frozen_ar_checkpoint": str(ar_checkpoint.resolve()) if ar_checkpoint else None,
                "all_124_refit": True,
                "in_sample_metrics_reported": False,
            }
            model_manifest_path = checkpoint.with_name(checkpoint.stem + "__export.json")
            _atomic_json(model_manifest_path, model_manifest)
            artifacts.add(model_manifest_path)
            manifests.append(model_manifest)
        component = Path(pca_provenance["component"])
        metadata = Path(pca_provenance["metadata"])
        artifacts.update({component, metadata})

    index_path = Path(output_root) / "model_index.json"
    _atomic_json(
        index_path,
        {
            "schema_version": EXECUTION_SCHEMA_VERSION,
            "export_contract_digest": export_contract["export_contract_digest"],
            "models": manifests,
            "model_count": len(manifests),
            "all_124_refit": True,
            "in_sample_metrics_reported": False,
        },
    )
    artifacts.add(index_path)
    return {"artifacts": [str(path.resolve()) for path in sorted(artifacts)]}


__all__ = [
    "Veatic21ExecutionError",
    "execute_all124_refit",
    "execute_confirmation_cell",
    "execute_nested_discovery",
]
