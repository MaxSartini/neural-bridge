"""Nested discovery, sealed confirmation, resume, audit, and final refit."""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expit
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .contracts import (
    CANONICAL_DATASET,
    CONTROL_LANES,
    CandidateSpec,
    CellSpec,
    FeatureRows,
    FrozenRecipe,
    FrozenWinner,
    LabelRows,
    SubstrateIdentity,
    TargetSpec,
    VideoSplit,
)
from .data import CanonicalSubstrate
from .evidence import (
    atomic_save_npz,
    atomic_write_json,
    create_prediction_seal,
    digest_json,
    load_json,
    per_video_pr_auc,
    pooled_pr_auc,
    row_identity_digest,
    sha256_file,
    source_tree_digest,
    verify_prediction_seal,
)
from .protocol import (
    assert_row_alignment,
    build_video_splits,
    causal_ar_features,
    event_labels,
    fit_event_threshold,
    freeze_winner,
    future_target_values,
    target_support_mask,
)

PRIMARY = "primary"
PRESEAL_LANES = (
    PRIMARY,
    "sequence_shuffled",
    "random",
    "video_mean",
    "diagnostics_only",
    "label_permutation",
)
FITTED_PRESEAL_LANES = tuple(lane for lane in PRESEAL_LANES if lane != "random")
MATCHED_CONTROL_LANES = tuple(lane for lane in CONTROL_LANES if lane != "random")
DEFAULT_TRANSFORM_BATCH_ROWS = 1_024


def _selected_indices(size: int, rows: np.ndarray | None) -> np.ndarray:
    if rows is None:
        return np.arange(size, dtype=np.int64)
    selection = np.asarray(rows)
    if np.issubdtype(selection.dtype, np.bool_):
        if selection.shape != (size,):
            raise ValueError("row mask has the wrong shape")
        return np.flatnonzero(selection)
    if selection.ndim != 1 or not np.issubdtype(selection.dtype, np.integer):
        raise ValueError("row selection must be a boolean mask or integer vector")
    indices = selection.astype(np.int64, copy=False)
    if np.any(indices < 0) or np.any(indices >= size):
        raise ValueError("row selection is outside the matrix")
    return indices


def _batches(
    indices: np.ndarray,
    batch_rows: int,
    *,
    minimum_rows: int = 1,
) -> Sequence[np.ndarray]:
    if batch_rows < minimum_rows:
        raise ValueError("batch_rows must cover the minimum batch size")
    if not len(indices):
        return ()
    batch_count = (len(indices) + batch_rows - 1) // batch_rows
    if len(indices) // batch_count < minimum_rows:
        raise ValueError("selected rows cannot form bounded batches that each cover the PCA width")
    base_size, larger_batches = divmod(len(indices), batch_count)
    batches: list[np.ndarray] = []
    start = 0
    for batch_index in range(batch_count):
        stop = start + base_size + (batch_index < larger_batches)
        batches.append(indices[start:stop])
        start = stop
    return batches


@dataclass(frozen=True)
class LinearTransform:
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    pca_mean: np.ndarray
    pca_components: np.ndarray
    batch_rows: int = DEFAULT_TRANSFORM_BATCH_ROWS

    def apply(self, values: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
        matrix = np.asarray(values)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.scaler_mean):
            raise ValueError("transform input has the wrong feature width")
        indices = _selected_indices(len(matrix), rows)
        output = np.empty((len(indices), len(self.pca_components)), dtype=np.float64)
        offset = 0
        for batch_indices in _batches(indices, self.batch_rows):
            batch = np.asarray(matrix[batch_indices], dtype=np.float64)
            scaled = (batch - self.scaler_mean) / self.scaler_scale
            count = len(batch_indices)
            output[offset : offset + count] = (scaled - self.pca_mean) @ self.pca_components.T
            offset += count
        return output


@dataclass(frozen=True)
class LinearHead:
    coefficient: np.ndarray
    intercept: float

    def predict(self, values: np.ndarray) -> np.ndarray:
        return expit(values @ self.coefficient + self.intercept).astype(np.float64)


@dataclass(frozen=True)
class FittedLinear:
    transform: LinearTransform
    head: LinearHead

    def predict(self, values: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
        return self.head.predict(self.transform.apply(values, rows))

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "scaler_mean": self.transform.scaler_mean,
            "scaler_scale": self.transform.scaler_scale,
            "pca_mean": self.transform.pca_mean,
            "pca_components": self.transform.pca_components,
            "batch_rows": np.asarray([self.transform.batch_rows], dtype=np.int64),
            "coefficient": self.head.coefficient,
            "intercept": np.asarray([self.head.intercept], dtype=np.float64),
        }

    @classmethod
    def from_arrays(cls, arrays: Mapping[str, np.ndarray]) -> FittedLinear:
        return cls(
            transform=LinearTransform(
                scaler_mean=np.asarray(arrays["scaler_mean"], dtype=np.float64),
                scaler_scale=np.asarray(arrays["scaler_scale"], dtype=np.float64),
                pca_mean=np.asarray(arrays["pca_mean"], dtype=np.float64),
                pca_components=np.asarray(arrays["pca_components"], dtype=np.float64),
                batch_rows=int(
                    np.asarray(
                        arrays.get(
                            "batch_rows",
                            np.asarray([DEFAULT_TRANSFORM_BATCH_ROWS], dtype=np.int64),
                        )
                    ).item()
                ),
            ),
            head=LinearHead(
                coefficient=np.asarray(arrays["coefficient"], dtype=np.float64),
                intercept=float(np.asarray(arrays["intercept"]).item()),
            ),
        )


def _fit_transform(
    values: np.ndarray,
    width: int,
    *,
    seed: int,
    rows: np.ndarray | None = None,
    solver: str = "randomized",
    batch_rows: int | None = None,
) -> LinearTransform:
    matrix = np.asarray(values)
    if matrix.ndim != 2:
        raise ValueError("a fitted representation requires a matrix")
    indices = _selected_indices(len(matrix), rows)
    if len(indices) < 3:
        raise ValueError("a fitted representation requires a matrix with at least three rows")
    maximum = min(len(indices) - 1, matrix.shape[1])
    if not 0 < width <= maximum:
        raise ValueError(f"PCA width {width} exceeds fitted rank ceiling {maximum}")
    if solver == "incremental":
        if batch_rows is None or batch_rows < width:
            raise ValueError("incremental PCA requires batch_rows >= width")
        incremental_batch_rows = batch_rows
        batches = _batches(indices, incremental_batch_rows, minimum_rows=width)
        scaler = StandardScaler()
        for batch_indices in batches:
            scaler.partial_fit(np.asarray(matrix[batch_indices], dtype=np.float64))
    elif solver == "randomized":
        selected = matrix if rows is None else matrix[indices]
        scaler = StandardScaler().fit(np.asarray(selected, dtype=np.float64))
    else:
        raise ValueError(f"unsupported PCA solver: {solver}")
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    scale[scale == 0.0] = 1.0
    if solver == "incremental":
        pca = IncrementalPCA(n_components=width, batch_size=incremental_batch_rows)
        for batch_indices in batches:
            batch = np.asarray(matrix[batch_indices], dtype=np.float64)
            pca.partial_fit((batch - scaler.mean_) / scale)
        transform_batch_rows = incremental_batch_rows
    else:
        selected = matrix if rows is None else matrix[indices]
        dense = np.asarray(selected, dtype=np.float64)
        scaled = (dense - scaler.mean_) / scale
        dense_solver = "full" if width == maximum else "randomized"
        pca = PCA(n_components=width, svd_solver=dense_solver, random_state=seed).fit(scaled)
        transform_batch_rows = DEFAULT_TRANSFORM_BATCH_ROWS
    return LinearTransform(
        scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
        scaler_scale=scale,
        pca_mean=np.asarray(pca.mean_, dtype=np.float64),
        pca_components=np.asarray(pca.components_, dtype=np.float64),
        batch_rows=transform_batch_rows,
    )


def _fit_head(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    regularization_c: float,
    max_iter: int,
    tolerance: float,
    seed: int,
) -> LinearHead:
    labels = np.asarray(labels, dtype=np.int8)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("binary head fitting requires both target classes")
    model = LogisticRegression(
        C=regularization_c,
        max_iter=max_iter,
        random_state=seed,
        solver="lbfgs",
        tol=tolerance,
    ).fit(values, labels)
    if int(model.n_iter_[0]) >= max_iter:
        raise RuntimeError("linear head did not satisfy the declared convergence limit")
    return LinearHead(
        coefficient=np.asarray(model.coef_[0], dtype=np.float64),
        intercept=float(model.intercept_[0]),
    )


def _fit_linear(
    train_x: np.ndarray,
    train_y: np.ndarray,
    candidate: CandidateSpec,
    *,
    seed: int,
    width: int | None = None,
    rows: np.ndarray | None = None,
    solver: str | None = None,
    batch_rows: int | None = None,
) -> FittedLinear:
    candidate.validate()
    selected_solver = solver or candidate.pca_solver
    selected_batch_rows = candidate.pca_batch_rows if solver is None else batch_rows
    transform = _fit_transform(
        train_x,
        width or candidate.pca_width,
        seed=seed,
        rows=rows,
        solver=selected_solver,
        batch_rows=selected_batch_rows,
    )
    head = _fit_head(
        transform.apply(train_x, rows),
        train_y,
        regularization_c=candidate.regularization_c,
        max_iter=candidate.max_iter,
        tolerance=candidate.tolerance,
        seed=seed,
    )
    return FittedLinear(transform=transform, head=head)


def _fit_standardized_linear(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    regularization_c: float,
    max_iter: int,
    tolerance: float,
    seed: int,
) -> FittedLinear:
    """Fit a small direct baseline without routing it through cortical PCA."""

    matrix = np.asarray(train_x, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) < 3:
        raise ValueError("a direct linear baseline requires at least three matrix rows")
    scaler = StandardScaler().fit(matrix)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    scale[scale == 0.0] = 1.0
    width = matrix.shape[1]
    transform = LinearTransform(
        scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
        scaler_scale=scale,
        pca_mean=np.zeros(width, dtype=np.float64),
        pca_components=np.eye(width, dtype=np.float64),
    )
    head = _fit_head(
        transform.apply(matrix),
        train_y,
        regularization_c=regularization_c,
        max_iter=max_iter,
        tolerance=tolerance,
        seed=seed,
    )
    return FittedLinear(transform=transform, head=head)


def _array_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, values in sorted(arrays.items()):
        array = np.ascontiguousarray(values)
        digest.update(f"{name}\0{array.dtype.str}\0{array.shape}\0".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _video_mask(video_id: np.ndarray, videos: Sequence[str]) -> np.ndarray:
    return np.isin(video_id.astype(str), np.asarray(tuple(videos), dtype=str))


def _permute_within_video(
    values: np.ndarray, video_id: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    output = np.empty_like(values)
    for video in sorted(set(video_id.astype(str)), key=lambda item: int(item)):
        indices = np.flatnonzero(video_id.astype(str) == video)
        output[indices] = values[rng.permutation(indices)]
    return output


def _lane_rng(seed: int, lane: str) -> np.random.Generator:
    payload = hashlib.sha256(f"veatic21:{seed}:{lane}".encode()).digest()
    return np.random.default_rng(int.from_bytes(payload[:8], "big"))


def _circular_permute_labels(
    labels: np.ndarray,
    video_id: np.ndarray,
    row_index: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Rotate labels within each video, preserving prevalence and serial structure."""

    labels = np.asarray(labels)
    videos = video_id.astype(str)
    rows = np.asarray(row_index)
    if labels.ndim != 1 or videos.shape != labels.shape or rows.shape != labels.shape:
        raise ValueError("label-permutation inputs must be aligned vectors")
    output = np.empty_like(labels)
    for video in sorted(set(videos), key=int):
        positions = np.flatnonzero(videos == video)
        ordered = positions[np.argsort(rows[positions], kind="stable")]
        if len(ordered) < 2:
            output[ordered] = labels[ordered]
            continue
        shift = int(rng.integers(1, len(ordered)))
        output[ordered] = labels[np.roll(ordered, shift)]
    return output


def _causal_video_means(
    values: np.ndarray,
    video_id: np.ndarray,
    row_index: np.ndarray,
) -> np.ndarray:
    """Return each video's expanding mean through the current row only."""

    values = np.asarray(values, dtype=np.float64)
    video_id = video_id.astype(str)
    row_index = np.asarray(row_index)
    if values.ndim != 2 or video_id.shape != (len(values),) or row_index.shape != (len(values),):
        raise ValueError("causal video-mean inputs must be aligned row vectors and a matrix")
    output = np.empty_like(values)
    for video in set(video_id.astype(str)):
        positions = np.flatnonzero(video_id == video)
        ordered = positions[np.argsort(row_index[positions], kind="stable")]
        cumulative = np.cumsum(values[ordered], axis=0, dtype=np.float64)
        output[ordered] = cumulative / np.arange(1, len(ordered) + 1)[:, None]
    return output


def _align_prediction_rows(
    labels: LabelRows, video_id: np.ndarray, row_index: np.ndarray, values: np.ndarray
) -> np.ndarray:
    lookup = {
        (str(video), int(row)): index
        for index, (video, row) in enumerate(zip(labels.video_id, labels.row_index, strict=True))
    }
    try:
        indices = np.asarray(
            [
                lookup[(str(video), int(row))]
                for video, row in zip(video_id, row_index, strict=True)
            ],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError(f"sealed prediction row is absent from labels: {error.args[0]}") from error
    return values[indices]


def _veatic_import_boundary_pass(source_root: Path) -> bool:
    for path in source_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(
                alias.name == "neural_bridge.again" or alias.name.startswith("neural_bridge.again.")
                for alias in node.names
            ):
                return False
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "neural_bridge.again" or module.startswith("neural_bridge.again."):
                    return False
                if node.level and (module == "again" or module.startswith("again.")):
                    return False
    return True


def run_nested_discovery(
    features: FeatureRows,
    labels: LabelRows,
    split: VideoSplit,
    cell: CellSpec,
    candidates: Sequence[CandidateSpec],
) -> FrozenWinner:
    """Select one fold-local recipe without opening the outer-test labels."""

    if not candidates:
        raise ValueError("nested discovery requires at least one candidate")
    if len({candidate.name for candidate in candidates}) != len(candidates):
        raise ValueError("candidate names must be unique")
    features.validate()
    labels.validate()
    assert_row_alignment(features, labels)
    target_values = future_target_values(labels, cell.target)
    support = target_support_mask(features, cell.target)
    ar_values, ar_available = causal_ar_features(labels, cell.target)
    ar_values = np.concatenate([ar_values, ar_available.astype(np.float64)], axis=1)
    records: list[dict[str, object]] = []
    for inner_fold, (inner_train_videos, inner_validation_videos) in enumerate(split.inner_splits):
        train_mask = (
            _video_mask(features.video_id, inner_train_videos) & features.quality_eligible & support
        )
        validation_mask = (
            _video_mask(features.video_id, inner_validation_videos)
            & features.quality_eligible
            & support
        )
        if np.any(train_mask & validation_mask):
            raise ValueError("inner training and validation rows overlap")
        threshold = fit_event_threshold(target_values, train_mask, cell.target)
        targets = event_labels(target_values, threshold)
        fitted_ar = _fit_standardized_linear(
            ar_values[train_mask],
            targets[train_mask],
            regularization_c=1.0,
            max_iter=5_000,
            tolerance=1e-6,
            seed=cell.seed + inner_fold,
        )
        ar_score = pooled_pr_auc(
            targets[validation_mask], fitted_ar.predict(ar_values[validation_mask])
        )
        for candidate in candidates:
            candidate.validate()
            if candidate.representation not in features.representations:
                raise ValueError(f"representation was not loaded: {candidate.representation}")
            matrix = features.representations[candidate.representation]
            fitted = _fit_linear(
                matrix,
                targets[train_mask],
                candidate,
                seed=cell.seed + inner_fold,
                rows=train_mask,
            )
            score = pooled_pr_auc(targets[validation_mask], fitted.predict(matrix, validation_mask))
            records.append(
                {
                    "candidate": candidate.name,
                    "inner_fold": inner_fold,
                    "pooled_pr_auc": score,
                    "frozen_ar_pr_auc": ar_score,
                    "delta_vs_frozen_ar": score - ar_score,
                    "threshold": threshold,
                    "train_rows": int(train_mask.sum()),
                    "validation_rows": int(validation_mask.sum()),
                }
            )
    return freeze_winner(
        candidates,
        records,
        cell=cell,
        split_digest=split.digest,
        selection_metric="pooled_pr_auc",
        tie_break="mean_desc_candidate_name_asc",
    )


def _fit_preseal_lanes(
    train_features: FeatureRows,
    test_features: FeatureRows,
    train_labels: LabelRows,
    winner: FrozenWinner,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, FittedLinear],
    FittedLinear,
    dict[str, Any],
]:
    assert_row_alignment(train_features, train_labels)
    candidate = winner.candidate
    target = winner.cell.target
    train_support = target_support_mask(train_features, target)
    test_support = target_support_mask(test_features, target)
    train_mask = train_features.quality_eligible & train_support
    test_mask = test_features.quality_eligible & test_support
    target_values = future_target_values(train_labels, target)
    threshold = fit_event_threshold(target_values, train_mask, target)
    targets = event_labels(target_values, threshold)
    train_video = train_features.video_id[train_mask].astype(str)
    test_video = test_features.video_id[test_mask].astype(str)
    train_x = train_features.representations[candidate.representation]
    test_x = test_features.representations[candidate.representation]
    train_y = targets[train_mask]
    train_row = train_features.row_index[train_mask]
    test_row = test_features.row_index[test_mask]

    primary = _fit_linear(
        train_x,
        train_y,
        candidate,
        seed=winner.cell.seed,
        rows=train_mask,
    )
    train_z = primary.transform.apply(train_x, train_mask)
    test_z = primary.transform.apply(test_x, test_mask)

    sequence_rng = _lane_rng(winner.cell.seed, "sequence_shuffled")
    shuffled_train = _permute_within_video(train_z, train_video, sequence_rng)
    shuffled_test = _permute_within_video(test_z, test_video, sequence_rng)
    shuffled_head = _fit_head(
        shuffled_train,
        train_y,
        regularization_c=candidate.regularization_c,
        max_iter=candidate.max_iter,
        tolerance=candidate.tolerance,
        seed=winner.cell.seed,
    )
    label_permutation_head = _fit_head(
        train_z,
        _circular_permute_labels(
            train_y,
            train_video,
            train_row,
            _lane_rng(winner.cell.seed, "label_permutation"),
        ),
        regularization_c=candidate.regularization_c,
        max_iter=candidate.max_iter,
        tolerance=candidate.tolerance,
        seed=winner.cell.seed,
    )
    video_mean_train = _causal_video_means(train_z, train_video, train_row)
    video_mean_test = _causal_video_means(test_z, test_video, test_row)
    video_mean_head = _fit_head(
        video_mean_train,
        train_y,
        regularization_c=candidate.regularization_c,
        max_iter=candidate.max_iter,
        tolerance=candidate.tolerance,
        seed=winner.cell.seed,
    )
    diagnostics_train = train_features.representations["diagnostics_only"]
    diagnostics_test = test_features.representations["diagnostics_only"]
    diagnostics_recipe = replace(
        candidate,
        name=f"{candidate.name}__diagnostics-only",
        pca_width=min(candidate.pca_width, diagnostics_train.shape[1]),
    )
    diagnostics = _fit_linear(
        diagnostics_train,
        train_y,
        diagnostics_recipe,
        seed=winner.cell.seed,
        rows=train_mask,
        solver="randomized",
    )

    ar_values, ar_available = causal_ar_features(train_labels, target)
    ar_values = np.concatenate([ar_values, ar_available.astype(np.float64)], axis=1)
    frozen_ar = _fit_standardized_linear(
        ar_values[train_mask],
        train_y,
        regularization_c=1.0,
        max_iter=candidate.max_iter,
        tolerance=candidate.tolerance,
        seed=winner.cell.seed,
    )

    predictions = {
        "video_id": test_features.video_id[test_mask].astype("U16"),
        "row_index": test_features.row_index[test_mask].astype(np.int32),
        "time_seconds": test_features.time_seconds[test_mask].astype(np.float32),
        PRIMARY: primary.head.predict(test_z),
        "sequence_shuffled": shuffled_head.predict(shuffled_test),
        "random": _lane_rng(winner.cell.seed, "random").random(
            int(test_mask.sum()), dtype=np.float64
        ),
        "video_mean": video_mean_head.predict(video_mean_test),
        "diagnostics_only": diagnostics.predict(diagnostics_test, test_mask),
        "label_permutation": label_permutation_head.predict(test_z),
    }
    fitted_lanes = {
        PRIMARY: primary,
        "sequence_shuffled": FittedLinear(primary.transform, shuffled_head),
        "video_mean": FittedLinear(primary.transform, video_mean_head),
        "diagnostics_only": diagnostics,
        "label_permutation": FittedLinear(primary.transform, label_permutation_head),
    }
    model_digests = {name: _array_digest(model.arrays()) for name, model in fitted_lanes.items()}
    model_digests["random"] = digest_json(
        {
            "generator": "numpy.default_rng_namespaced_sha256",
            "seed": winner.cell.seed,
            "lane": "random",
        }
    )
    model_digests["target_specific_frozen_ar"] = _array_digest(frozen_ar.arrays())
    fit_manifest = {
        "schema": "veatic21_confirmation_fit_v1",
        "candidate": asdict(candidate),
        "target_threshold": threshold,
        "target_threshold_scope": "outer_train_only",
        "representation_fit_scope": "outer_train_only",
        "head_fit_scope": "outer_train_only",
        "control_semantics": {
            "sequence_shuffled": "within_video_feature_permutation_outer_train_fitted",
            "random": "seeded_uniform_chance_diagnostic_no_fit",
            "video_mean": "causal_prefix_mean_outer_train_fitted",
            "diagnostics_only": "outer_train_fitted",
            "label_permutation": "nonzero_within_video_circular_shift_outer_train_fitted",
            "target_specific_frozen_ar": "outer_train_fitted_postseal_label_side_evaluation",
        },
        "train_rows": int(train_mask.sum()),
        "prediction_rows": int(test_mask.sum()),
        "train_quality_exclusions": int((~train_features.quality_eligible).sum()),
        "test_quality_exclusions": int((~test_features.quality_eligible).sum()),
        "model_files": {lane: f"model_{lane}.npz" for lane in FITTED_PRESEAL_LANES},
        "model_sha256": model_digests,
        "winner_sha256": winner.digest,
    }
    return predictions, fitted_lanes, frozen_ar, fit_manifest


def _request_payload(
    identity: SubstrateIdentity,
    split: VideoSplit,
    cell: CellSpec,
    candidates: Sequence[CandidateSpec],
) -> dict[str, Any]:
    return {
        "schema": "veatic21_confirmation_request_v1",
        "programme": "veatic-2.1",
        "cell": asdict(cell),
        "candidates": [
            asdict(candidate) for candidate in sorted(candidates, key=lambda item: item.name)
        ],
        "split": asdict(split),
        "substrate": asdict(identity),
        "protocol": {
            "group_key": "video_id",
            "outer_split": "sorted_video_ids_seeded_permutation_array_split",
            "inner_split": "outer_train_only_sorted_video_ids_seeded_permutation_array_split",
            "selection_metric": "pooled_pr_auc",
            "controls": list(CONTROL_LANES),
            "outer_labels_closed_during_selection": True,
        },
    }


def _load_fitted_linear(path: Path) -> FittedLinear:
    with np.load(path, allow_pickle=False) as arrays:
        return FittedLinear.from_arrays({name: arrays[name] for name in arrays.files})


def _strict_resume_check(
    cell_root: Path,
    *,
    request_digest: str,
    split_digest: str,
    substrate_digest: str,
    code_digest: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    request = load_json(cell_root / "request.json")
    if digest_json(request) != request_digest:
        raise RuntimeError("refusing resume: request contract changed")
    state = load_json(cell_root / "state.json")
    if state.get("request_sha256") != request_digest:
        raise RuntimeError("refusing resume: state request digest differs")
    seal_path = cell_root / "prediction_seal.json"
    if sha256_file(seal_path) != state.get("prediction_seal_sha256"):
        raise RuntimeError("refusing resume: prediction seal changed")
    seal = load_json(seal_path)
    if seal.get("cell_sha256") != state.get("cell_sha256"):
        raise RuntimeError("refusing resume: prediction seal cell digest differs")
    checks = verify_prediction_seal(cell_root, seal)
    if not checks["pass"]:
        raise RuntimeError(f"refusing resume: invalid prediction seal: {checks['failures']}")
    expected = {
        "split_sha256": split_digest,
        "substrate_sha256": substrate_digest,
        "code_sha256": code_digest,
    }
    mismatches = [name for name, value in expected.items() if seal.get(name) != value]
    if mismatches:
        raise RuntimeError(f"refusing resume: changed dependencies: {mismatches}")
    fit_path = cell_root / "fit.json"
    ar_path = cell_root / "target_specific_frozen_ar.npz"
    model_hashes = seal.get("model_sha256", {})
    if not isinstance(model_hashes, Mapping):
        raise RuntimeError("refusing resume: model digests are not a mapping")
    if sha256_file(fit_path) != model_hashes.get("fit_manifest"):
        raise RuntimeError("refusing resume: fit manifest changed")
    fit = load_json(fit_path)
    model_files = fit.get("model_files")
    if not isinstance(model_files, Mapping) or set(model_files) != set(FITTED_PRESEAL_LANES):
        raise RuntimeError("refusing resume: fitted pre-seal model inventory changed")
    for lane in FITTED_PRESEAL_LANES:
        filename = str(model_files[lane])
        if Path(filename).name != filename:
            raise RuntimeError("refusing resume: invalid fitted model path")
        model = _load_fitted_linear(cell_root / filename)
        if _array_digest(model.arrays()) != model_hashes.get(lane):
            raise RuntimeError(f"refusing resume: fitted model changed: {lane}")
    ar_digest = _array_digest(_load_fitted_linear(ar_path).arrays())
    if ar_digest != model_hashes.get("target_specific_frozen_ar"):
        raise RuntimeError("refusing resume: frozen AR model changed")
    return state, seal


def _score_sealed_cell(
    substrate: CanonicalSubstrate,
    cell_root: Path,
    split: VideoSplit,
    cell: CellSpec,
    state: dict[str, Any],
    seal: dict[str, Any],
) -> dict[str, Any]:
    events = list(state.get("label_access_events", []))

    def record(stage: str) -> None:
        events.append(stage)

    if "prediction_seal_written" not in events:
        raise RuntimeError("held-out scoring requires an already-written prediction seal")
    heldout_labels = substrate.load_labels(
        split.test_video_ids,
        access_callback=record,
        stage="outer_test_labels_opened_after_prediction_seal",
    )
    with np.load(cell_root / str(seal["prediction_file"]), allow_pickle=False) as arrays:
        predictions = {name: np.asarray(arrays[name]) for name in arrays.files}
    fit = load_json(cell_root / "fit.json")
    target_values = future_target_values(heldout_labels, cell.target)
    all_targets = event_labels(target_values, float(fit["target_threshold"]))
    y_true = _align_prediction_rows(
        heldout_labels,
        predictions["video_id"],
        predictions["row_index"],
        all_targets,
    ).astype(np.int8)
    ar_values, ar_available = causal_ar_features(heldout_labels, cell.target)
    ar_values = np.concatenate([ar_values, ar_available.astype(np.float64)], axis=1)
    ar_x = _align_prediction_rows(
        heldout_labels,
        predictions["video_id"],
        predictions["row_index"],
        ar_values,
    )
    frozen_ar = _load_fitted_linear(cell_root / "target_specific_frozen_ar.npz")
    ar_scores = frozen_ar.predict(ar_x)
    evaluator_path = cell_root / "evaluator_controls.npz"
    atomic_save_npz(
        evaluator_path,
        {
            "video_id": predictions["video_id"],
            "row_index": predictions["row_index"],
            "target_specific_frozen_ar": ar_scores,
        },
    )
    evaluator_seal = {
        "schema": "veatic21_evaluator_control_seal_v1",
        "file": evaluator_path.name,
        "sha256": sha256_file(evaluator_path),
        "row_identity_sha256": seal["row_identity_sha256"],
        "outer_labels_opened_after_prediction_seal": True,
        "frozen_ar_fit_scope": "outer_train_only",
    }
    atomic_write_json(cell_root / "evaluator_control_seal.json", evaluator_seal)

    lane_scores = {lane: np.asarray(predictions[lane], dtype=np.float64) for lane in PRESEAL_LANES}
    lane_scores["target_specific_frozen_ar"] = ar_scores
    pooled = {lane: pooled_pr_auc(y_true, scores) for lane, scores in lane_scores.items()}
    control_best = max(MATCHED_CONTROL_LANES, key=lambda lane: pooled[lane])
    per_video = per_video_pr_auc(predictions["video_id"].astype(str), y_true, lane_scores[PRIMARY])
    zero_event_videos = []
    zero_event_negative_rows = 0
    for video in sorted(set(predictions["video_id"].astype(str)), key=int):
        mask = predictions["video_id"].astype(str) == video
        if not y_true[mask].any():
            zero_event_videos.append(video)
            zero_event_negative_rows += int(mask.sum())
    metrics = {
        "schema": "veatic21_confirmation_metrics_v1",
        "row_count": len(y_true),
        "positive_rows": int(y_true.sum()),
        "negative_rows": int((y_true == 0).sum()),
        "pooled_pr_auc": pooled,
        "analytic_chance_pr_auc": float(np.mean(y_true)),
        "primary_minus_strongest_control": pooled[PRIMARY] - pooled[control_best],
        "strongest_control": control_best,
        "strongest_control_scope": "matched_controls_only_random_is_chance_diagnostic",
        "primary_per_video_pr_auc": per_video,
        "undefined_per_video_count": sum(value is None for value in per_video.values()),
        "zero_event_videos": zero_event_videos,
        "zero_event_negative_rows_retained_in_pooled_metric": zero_event_negative_rows,
        "scientific_claim": None,
        "promotable": cell.promotable,
    }
    atomic_write_json(cell_root / "metrics.json", metrics)

    source_root = Path(__file__).parent
    gates = {
        "canonical_video_count": len(substrate.identity.video_ids) == CANONICAL_DATASET.video_count,
        "canonical_row_count": substrate.identity.row_count == CANONICAL_DATASET.row_count,
        "locked_exclusion_count": (
            substrate.identity.exclusion_count == CANONICAL_DATASET.exclusion_count
        ),
        "group_disjoint_outer_split": set(split.train_video_ids).isdisjoint(split.test_video_ids),
        "prediction_seal_valid": bool(verify_prediction_seal(cell_root, seal)["pass"]),
        "prediction_sealed_before_outer_label_access": events.index("prediction_seal_written")
        < events.index("outer_test_labels_opened_after_prediction_seal"),
        "one_to_one_prediction_coverage": len(y_true)
        == len(
            set(
                zip(
                    predictions["video_id"].tolist(),
                    predictions["row_index"].tolist(),
                    strict=True,
                )
            )
        ),
        "all_controls_present": set(CONTROL_LANES).issubset(lane_scores),
        "eligible_zero_event_negatives_retained": zero_event_negative_rows
        == sum(
            int((predictions["video_id"].astype(str) == video).sum()) for video in zero_event_videos
        ),
        "undefined_per_video_scores_not_zero_filled": all(
            per_video[video] is None for video in zero_event_videos
        ),
        "veatic_only_artifacts": substrate.identity.vjepa_artifact_id.startswith("veatic-2.1-")
        and substrate.identity.tribe_artifact_id.startswith("veatic-2.1-"),
        "no_again_import_boundary": _veatic_import_boundary_pass(source_root),
        "scope_declaration_consistent": metrics["promotable"] is cell.promotable,
    }
    audit = {
        "schema": "veatic21_confirmation_audit_v1",
        "audit_pass": all(gates.values()),
        "gates": gates,
        "label_access_events": events,
        "fit_ownership": {
            "target_threshold": "outer_train_only",
            "normalizer": "outer_train_only",
            "pca": "outer_train_only",
            "heads": "outer_train_only",
            "target_specific_frozen_ar": "outer_train_only",
            "outer_labels": "opened_only_after_prediction_seal",
        },
        "inherited_fitted_artifacts": [],
        "result_scope": "plumbing_smoke" if not cell.promotable else "declared_confirmation_cell",
        "non_promotable_smoke": not cell.promotable,
    }
    if not audit["audit_pass"]:
        raise RuntimeError(
            f"confirmation audit failed: {[name for name, passed in gates.items() if not passed]}"
        )
    atomic_write_json(cell_root / "audit.json", audit)
    completed_state = {
        "schema": "veatic21_cell_state_v1",
        "status": "audited",
        "request_sha256": state["request_sha256"],
        "cell_sha256": seal["cell_sha256"],
        "prediction_seal_sha256": sha256_file(cell_root / "prediction_seal.json"),
        "evaluator_control_seal_sha256": sha256_file(cell_root / "evaluator_control_seal.json"),
        "metrics_sha256": sha256_file(cell_root / "metrics.json"),
        "audit_sha256": sha256_file(cell_root / "audit.json"),
        "label_access_events": events,
    }
    atomic_write_json(cell_root / "state.json", completed_state)
    return {"status": "audited", "resumed": True, "metrics": metrics, "audit": audit}


def run_confirmation_cell(
    substrate: CanonicalSubstrate,
    output_dir: Path,
    *,
    cell: CellSpec,
    candidates: Sequence[CandidateSpec],
    pause_after_seal: bool = False,
) -> dict[str, Any]:
    """Run or strictly resume one nested held-out-video confirmation cell."""

    cell.validate()
    candidates = tuple(candidates)
    if not candidates:
        raise ValueError("confirmation requires at least one candidate")
    if len({candidate.name for candidate in candidates}) != len(candidates):
        raise ValueError("candidate names must be unique")
    for candidate in candidates:
        candidate.validate()
    substrate.identity.validate()
    splits = build_video_splits(
        substrate.video_ids,
        outer_folds=cell.outer_folds,
        inner_folds=cell.inner_folds,
        split_seed=cell.split_seed,
    )
    split = next((item for item in splits if item.outer_fold == cell.outer_fold), None)
    if split is None:
        raise ValueError(f"outer fold {cell.outer_fold} was not planned")
    request = _request_payload(substrate.identity, split, cell, candidates)
    request_digest = digest_json(request)
    substrate_digest = digest_json(asdict(substrate.identity))
    code_digest = source_tree_digest(Path(__file__).parent)
    state_path = output_dir / "state.json"

    if state_path.exists():
        state, seal = _strict_resume_check(
            output_dir,
            request_digest=request_digest,
            split_digest=split.digest,
            substrate_digest=substrate_digest,
            code_digest=code_digest,
        )
        if state.get("status") == "predictions_sealed":
            return _score_sealed_cell(substrate, output_dir, split, cell, state, seal)
        if state.get("status") != "audited":
            raise RuntimeError(f"unsupported resume state: {state.get('status')}")
        for filename, field in (
            ("prediction_seal.json", "prediction_seal_sha256"),
            ("evaluator_control_seal.json", "evaluator_control_seal_sha256"),
            ("metrics.json", "metrics_sha256"),
            ("audit.json", "audit_sha256"),
        ):
            if sha256_file(output_dir / filename) != state.get(field):
                raise RuntimeError(f"refusing resume: audited artifact changed: {filename}")
        audit = load_json(output_dir / "audit.json")
        if audit.get("audit_pass") is not True:
            raise RuntimeError("refusing resume: saved audit does not pass")
        return {
            "status": "audited",
            "resumed": True,
            "metrics": load_json(output_dir / "metrics.json"),
            "audit": audit,
        }

    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("refusing to overwrite an unsealed confirmation directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "request.json", request)
    events: list[str] = []

    def record(stage: str) -> None:
        events.append(stage)

    atomic_write_json(
        state_path,
        {
            "schema": "veatic21_cell_state_v1",
            "status": "planned",
            "request_sha256": request_digest,
            "label_access_events": events,
        },
    )
    representations = tuple(
        sorted({candidate.representation for candidate in candidates} | {"diagnostics_only"})
    )
    train_features = substrate.load_features(split.train_video_ids, representations)
    test_features = substrate.load_features(split.test_video_ids, representations)
    train_labels = substrate.load_labels(
        split.train_video_ids,
        access_callback=record,
        stage="outer_train_labels_opened_for_nested_discovery",
    )
    winner = run_nested_discovery(train_features, train_labels, split, cell, candidates)
    atomic_write_json(
        output_dir / "discovery.json",
        {
            "schema": "veatic21_nested_discovery_v1",
            "winner": asdict(winner),
            "outer_labels_opened": False,
            "promotable": False,
        },
    )
    predictions, fitted_lanes, frozen_ar, fit_manifest = _fit_preseal_lanes(
        train_features,
        test_features,
        train_labels,
        winner,
    )
    model_files = fit_manifest["model_files"]
    if not isinstance(model_files, Mapping):
        raise TypeError("fit manifest model_files must be a mapping")
    for lane, model in fitted_lanes.items():
        atomic_save_npz(output_dir / str(model_files[lane]), model.arrays())
    ar_path = output_dir / "target_specific_frozen_ar.npz"
    atomic_save_npz(ar_path, frozen_ar.arrays())
    fit_path = output_dir / "fit.json"
    atomic_write_json(fit_path, fit_manifest)
    prediction_path = output_dir / "predictions.npz"
    atomic_save_npz(prediction_path, predictions)
    model_digests = dict(fit_manifest["model_sha256"])
    model_digests["fit_manifest"] = sha256_file(fit_path)
    cell_digest = digest_json(
        {
            "request_sha256": request_digest,
            "winner_sha256": winner.digest,
            "fit_sha256": model_digests["fit_manifest"],
        }
    )
    seal = create_prediction_seal(
        prediction_path,
        row_digest=row_identity_digest(predictions["video_id"], predictions["row_index"]),
        row_count=len(predictions["video_id"]),
        cell_digest=cell_digest,
        split_digest=split.digest,
        winner_digest=winner.digest,
        substrate_digest=substrate_digest,
        code_digest=code_digest,
        model_digests=model_digests,
        lanes=PRESEAL_LANES,
        promotable=cell.promotable,
    )
    atomic_write_json(output_dir / "prediction_seal.json", seal)
    events.append("prediction_seal_written")
    sealed_state = {
        "schema": "veatic21_cell_state_v1",
        "status": "predictions_sealed",
        "request_sha256": request_digest,
        "cell_sha256": cell_digest,
        "prediction_seal_sha256": sha256_file(output_dir / "prediction_seal.json"),
        "label_access_events": events,
    }
    atomic_write_json(state_path, sealed_state)
    if pause_after_seal:
        return {"status": "predictions_sealed", "resumed": False, "seal": seal}
    return _score_sealed_cell(substrate, output_dir, split, cell, sealed_state, seal)


def _equivalent(left: object, right: object, *, tolerance: float) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        left_items = sorted(left.items(), key=lambda item: str(item[0]))
        right_items = sorted(right.items(), key=lambda item: str(item[0]))
        return len(left_items) == len(right_items) and all(
            left_key == right_key and _equivalent(left_value, right_value, tolerance=tolerance)
            for (left_key, left_value), (right_key, right_value) in zip(
                left_items, right_items, strict=True
            )
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _equivalent(a, b, tolerance=tolerance) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return bool(np.isclose(left, right, rtol=0.0, atol=tolerance))
    return left == right


def _replay_preseal_predictions(
    cell_root: Path,
    *,
    fit: Mapping[str, Any],
    cell: CellSpec,
    candidate: CandidateSpec,
    train_features: FeatureRows,
    test_features: FeatureRows,
) -> dict[str, np.ndarray]:
    """Replay every pre-seal lane from persisted models and canonical inputs."""

    train_support = target_support_mask(train_features, cell.target)
    test_support = target_support_mask(test_features, cell.target)
    train_mask = train_features.quality_eligible & train_support
    test_mask = test_features.quality_eligible & test_support
    train_video = train_features.video_id[train_mask].astype(str)
    test_video = test_features.video_id[test_mask].astype(str)
    test_row = test_features.row_index[test_mask]
    train_x = train_features.representations[candidate.representation]
    test_x = test_features.representations[candidate.representation]

    model_files = fit.get("model_files")
    if not isinstance(model_files, Mapping):
        raise ValueError("fit manifest has no fitted model inventory")
    models = {
        lane: _load_fitted_linear(cell_root / str(model_files[lane]))
        for lane in FITTED_PRESEAL_LANES
    }

    sequence_model = models["sequence_shuffled"]
    sequence_train = sequence_model.transform.apply(train_x, train_mask)
    sequence_test = sequence_model.transform.apply(test_x, test_mask)
    sequence_rng = _lane_rng(cell.seed, "sequence_shuffled")
    _permute_within_video(sequence_train, train_video, sequence_rng)
    shuffled_test = _permute_within_video(sequence_test, test_video, sequence_rng)

    video_mean_model = models["video_mean"]
    video_mean_test = _causal_video_means(
        video_mean_model.transform.apply(test_x, test_mask),
        test_video,
        test_row,
    )
    diagnostics_test = test_features.representations["diagnostics_only"]
    return {
        PRIMARY: models[PRIMARY].predict(test_x, test_mask),
        "sequence_shuffled": sequence_model.head.predict(shuffled_test),
        "random": _lane_rng(cell.seed, "random").random(int(test_mask.sum()), dtype=np.float64),
        "video_mean": video_mean_model.head.predict(video_mean_test),
        "diagnostics_only": models["diagnostics_only"].predict(diagnostics_test, test_mask),
        "label_permutation": models["label_permutation"].predict(test_x, test_mask),
    }


def verify_confirmation_cell(
    substrate: CanonicalSubstrate,
    output_dir: Path,
    *,
    cell: CellSpec,
    candidates: Sequence[CandidateSpec],
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Recompute a completed cell from sealed predictions and canonical rows."""

    cell.validate()
    substrate.identity.validate()
    splits = build_video_splits(
        substrate.video_ids,
        outer_folds=cell.outer_folds,
        inner_folds=cell.inner_folds,
        split_seed=cell.split_seed,
    )
    split = splits[cell.outer_fold]
    request = _request_payload(substrate.identity, split, cell, candidates)
    state, seal = _strict_resume_check(
        output_dir,
        request_digest=digest_json(request),
        split_digest=split.digest,
        substrate_digest=digest_json(asdict(substrate.identity)),
        code_digest=source_tree_digest(Path(__file__).parent),
    )
    failures: list[str] = []
    if state.get("status") != "audited":
        failures.append("state_not_audited")
    for filename, field in (
        ("prediction_seal.json", "prediction_seal_sha256"),
        ("evaluator_control_seal.json", "evaluator_control_seal_sha256"),
        ("metrics.json", "metrics_sha256"),
        ("audit.json", "audit_sha256"),
    ):
        if sha256_file(output_dir / filename) != state.get(field):
            failures.append(f"state_hash_{filename}")

    fit = load_json(output_dir / "fit.json")
    fit_candidate = fit.get("candidate")
    candidate = next(
        (item for item in candidates if asdict(item) == fit_candidate),
        None,
    )
    if candidate is None:
        failures.append("fit_candidate")
    representations = tuple(
        sorted({candidate.representation for candidate in candidates} | {"diagnostics_only"})
    )
    train_features = substrate.load_features(split.train_video_ids, representations)
    test_features = substrate.load_features(split.test_video_ids, representations)
    train_labels = substrate.load_labels(split.train_video_ids, stage="verification_outer_train")
    assert_row_alignment(train_features, train_labels)
    train_support = target_support_mask(train_features, cell.target)
    train_values = future_target_values(train_labels, cell.target)
    recomputed_threshold = fit_event_threshold(
        train_values,
        train_features.quality_eligible & train_support,
        cell.target,
    )
    if not np.isclose(
        recomputed_threshold,
        float(fit["target_threshold"]),
        rtol=0.0,
        atol=tolerance,
    ):
        failures.append("target_threshold")

    with np.load(output_dir / str(seal["prediction_file"]), allow_pickle=False) as arrays:
        predictions = {name: np.asarray(arrays[name]) for name in arrays.files}
    expected_test_mask = test_features.quality_eligible & target_support_mask(
        test_features, cell.target
    )
    expected_video = test_features.video_id[expected_test_mask].astype(str)
    expected_row = test_features.row_index[expected_test_mask]
    prediction_rows_match = np.array_equal(
        predictions["video_id"].astype(str), expected_video
    ) and np.array_equal(predictions["row_index"], expected_row)
    if not prediction_rows_match:
        failures.append("sealed_row_coverage")
    if candidate is not None:
        replayed = _replay_preseal_predictions(
            output_dir,
            fit=fit,
            cell=cell,
            candidate=candidate,
            train_features=train_features,
            test_features=test_features,
        )
        for lane in PRESEAL_LANES:
            saved = np.asarray(predictions[lane], dtype=np.float64)
            if saved.shape != replayed[lane].shape or not np.allclose(
                saved,
                replayed[lane],
                rtol=0.0,
                atol=tolerance,
            ):
                failures.append(f"prediction_replay_{lane}")

    evaluator_seal = load_json(output_dir / "evaluator_control_seal.json")
    evaluator_path = output_dir / str(evaluator_seal.get("file"))
    if sha256_file(evaluator_path) != evaluator_seal.get("sha256"):
        failures.append("evaluator_control_sha256")
    if evaluator_seal.get("row_identity_sha256") != seal.get("row_identity_sha256"):
        failures.append("evaluator_row_identity")
    with np.load(evaluator_path, allow_pickle=False) as arrays:
        evaluator = {name: np.asarray(arrays[name]) for name in arrays.files}
    evaluator_rows_match = np.array_equal(
        evaluator["video_id"].astype(str), predictions["video_id"].astype(str)
    ) and np.array_equal(evaluator["row_index"], predictions["row_index"])
    if not evaluator_rows_match:
        failures.append("evaluator_row_coverage")

    heldout_labels = substrate.load_labels(split.test_video_ids, stage="verification_outer_test")
    target_values = future_target_values(heldout_labels, cell.target)
    y_true = _align_prediction_rows(
        heldout_labels,
        predictions["video_id"],
        predictions["row_index"],
        event_labels(target_values, recomputed_threshold),
    ).astype(np.int8)
    ar_values, ar_available = causal_ar_features(heldout_labels, cell.target)
    ar_values = np.concatenate([ar_values, ar_available.astype(np.float64)], axis=1)
    ar_x = _align_prediction_rows(
        heldout_labels,
        predictions["video_id"],
        predictions["row_index"],
        ar_values,
    )
    replayed_ar = _load_fitted_linear(output_dir / "target_specific_frozen_ar.npz").predict(ar_x)
    saved_ar = np.asarray(evaluator["target_specific_frozen_ar"], dtype=np.float64)
    if saved_ar.shape != replayed_ar.shape or not np.allclose(
        saved_ar,
        replayed_ar,
        rtol=0.0,
        atol=tolerance,
    ):
        failures.append("frozen_ar_replay")
    lane_scores = {lane: np.asarray(predictions[lane], dtype=np.float64) for lane in PRESEAL_LANES}
    lane_scores["target_specific_frozen_ar"] = np.asarray(
        evaluator["target_specific_frozen_ar"], dtype=np.float64
    )
    pooled = {lane: pooled_pr_auc(y_true, scores) for lane, scores in lane_scores.items()}
    strongest = max(MATCHED_CONTROL_LANES, key=lambda lane: pooled[lane])
    per_video = per_video_pr_auc(predictions["video_id"].astype(str), y_true, lane_scores[PRIMARY])
    zero_event_videos = []
    zero_event_rows = 0
    for video in sorted(set(predictions["video_id"].astype(str)), key=int):
        mask = predictions["video_id"].astype(str) == video
        if not y_true[mask].any():
            zero_event_videos.append(video)
            zero_event_rows += int(mask.sum())
    recomputed_metrics = {
        "schema": "veatic21_confirmation_metrics_v1",
        "row_count": len(y_true),
        "positive_rows": int(y_true.sum()),
        "negative_rows": int((y_true == 0).sum()),
        "pooled_pr_auc": pooled,
        "analytic_chance_pr_auc": float(np.mean(y_true)),
        "primary_minus_strongest_control": pooled[PRIMARY] - pooled[strongest],
        "strongest_control": strongest,
        "strongest_control_scope": "matched_controls_only_random_is_chance_diagnostic",
        "primary_per_video_pr_auc": per_video,
        "undefined_per_video_count": sum(value is None for value in per_video.values()),
        "zero_event_videos": zero_event_videos,
        "zero_event_negative_rows_retained_in_pooled_metric": zero_event_rows,
        "scientific_claim": None,
        "promotable": cell.promotable,
    }
    saved_metrics = load_json(output_dir / "metrics.json")
    if not _equivalent(saved_metrics, recomputed_metrics, tolerance=tolerance):
        failures.append("metrics_recomputation")
    audit = load_json(output_dir / "audit.json")
    events = audit.get("label_access_events", [])
    if (
        not isinstance(events, list)
        or "prediction_seal_written" not in events
        or "outer_test_labels_opened_after_prediction_seal" not in events
        or events.index("prediction_seal_written")
        >= events.index("outer_test_labels_opened_after_prediction_seal")
    ):
        failures.append("label_access_order")
    if audit.get("audit_pass") is not True or not all(audit.get("gates", {}).values()):
        failures.append("audit_policy")
    return {
        "verification_pass": not failures,
        "failures": failures,
        "recomputed_metrics": recomputed_metrics,
    }


def predict_exported_model(model_path: Path, values: np.ndarray) -> np.ndarray:
    """Apply a portable final VEATIC linear export."""

    return _load_fitted_linear(model_path).predict(values)


def refit_all_124(
    substrate: CanonicalSubstrate,
    output_dir: Path,
    *,
    recipe: FrozenRecipe,
    target: TargetSpec,
) -> dict[str, Any]:
    """Refuse export until a preregistered, sealed promotion gate exists."""

    raise RuntimeError(
        "VEATIC 2.1 foundation export is disabled until a preregistered, "
        "verifiable promotion gate is implemented"
    )
