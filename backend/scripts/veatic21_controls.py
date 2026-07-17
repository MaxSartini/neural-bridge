"""Deterministic, fold-safe matched controls for the VEATIC 2.1 program.

The builders in this module operate on the sealed video-only feature contract
from :mod:`backend.scripts.veatic21_features`.  They never accept response
features and they keep every output row in the recipient video's original
position.  Controls that need a donor sequence reassign whole videos and
resample the donor trajectory onto recipient-relative time; they never shuffle
individual rows.

Every artifact carries a target/fold/seed/endpoint/lane namespace, construction
parameters, array digests, and (for privileged residual lanes) the exact frozen
AR identity.  Optional persistence is fail-closed: an existing artifact is
reused only when its complete sealed record and frozen-AR identity match.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from backend.scripts import veatic21_features as feature_contract


CONTROL_SCHEMA_NAME = "veatic21_matched_controls_v1"
CONTROL_SCHEMA_VERSION = 1
LINEAR_RESAMPLE_POLICY = "whole_video_normalized_time_linear"
NEAREST_RESAMPLE_POLICY = "whole_video_normalized_time_nearest"


class Veatic21ControlError(ValueError):
    """Raised when a control violates split, schema, or reuse identity."""


@dataclass(frozen=True)
class ControlNamespace:
    """Complete identity of one target/fold/seed/endpoint/lane control."""

    target: str
    fold: int
    seed: int
    endpoint: str
    lane: str

    def __post_init__(self) -> None:
        for field_name in ("target", "endpoint", "lane"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise Veatic21ControlError(
                    f"Control namespace {field_name} must be a non-empty trimmed string"
                )
        for field_name in ("fold", "seed"):
            value = getattr(self, field_name)
            if isinstance(value, (bool, np.bool_)):
                raise Veatic21ControlError(
                    f"Control namespace {field_name} must be a non-negative integer"
                )
            try:
                integer = int(value)
            except (TypeError, ValueError) as exc:
                raise Veatic21ControlError(
                    f"Control namespace {field_name} must be a non-negative integer"
                ) from exc
            if integer != value or integer < 0:
                raise Veatic21ControlError(
                    f"Control namespace {field_name} must be a non-negative integer"
                )
            object.__setattr__(self, field_name, integer)

    def as_dict(self) -> dict[str, object]:
        return {
            "target": self.target,
            "fold": self.fold,
            "seed": self.seed,
            "endpoint": self.endpoint,
            "lane": self.lane,
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.as_dict())


@dataclass(frozen=True)
class SealedFeatureControl:
    """One aligned train/test feature control and its reproducibility record."""

    namespace: ControlNamespace
    train: feature_contract.Veatic21Features
    test: feature_contract.Veatic21Features
    record: Mapping[str, Any]
    parameter_arrays: Mapping[str, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class SealedTargetControl:
    """One whole-video label-permutation target in original recipient row order."""

    namespace: ControlNamespace
    row_idx: np.ndarray
    video_id: np.ndarray
    time_seconds: np.ndarray
    target: np.ndarray
    record: Mapping[str, Any]


@dataclass(frozen=True)
class ControlAudit:
    passed: bool
    checks: tuple[str, ...]
    record_digest: str


@dataclass(frozen=True)
class ControlPaths:
    manifest: Path
    arrays: Path


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def array_digest(values: np.ndarray | Sequence[object]) -> str:
    """Hash shape, dtype, and values without object-array pointer instability."""

    array = np.asarray(values)
    header = _canonical_json({"shape": list(array.shape), "dtype": array.dtype.str})
    digest = hashlib.sha256(header.encode("utf-8"))
    if array.dtype.kind in {"O", "U", "S"}:
        digest.update(
            _canonical_json([str(value) for value in array.reshape(-1).tolist()]).encode(
                "utf-8"
            )
        )
    else:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def feature_block_digest(block: feature_contract.Veatic21Features) -> str:
    payload = {
        "row_idx": array_digest(block.row_idx),
        "video_id": array_digest(block.video_id),
        "time_seconds": array_digest(block.time_seconds),
        "x_temporal": array_digest(block.x_temporal),
        "x_current": array_digest(block.x_current),
        "history_mask": array_digest(block.history_mask),
        "temporal_feature_names": list(block.temporal_feature_names),
        "current_feature_names": list(block.current_feature_names),
        "schema_digest": block.schema_digest,
        "pca_width": int(block.pca_width),
    }
    return canonical_digest(payload)


def _identity_payload(
    identity: str | Mapping[str, Any] | None,
    *,
    required: bool,
) -> dict[str, Any] | None:
    if identity is None:
        if required:
            raise Veatic21ControlError(
                "Privileged control requires the exact frozen AR identity"
            )
        return None
    if not required:
        raise Veatic21ControlError(
            "A zero-label control must not carry a frozen AR identity"
        )
    if isinstance(identity, str):
        if not identity or identity != identity.strip():
            raise Veatic21ControlError("Frozen AR identity digest must be non-empty")
        return {"identity_digest": identity}
    payload = dict(identity)
    digest = payload.get("identity_digest")
    if not isinstance(digest, str) or not digest or digest != digest.strip():
        raise Veatic21ControlError("Frozen AR identity mapping lacks identity_digest")
    # Full identities emitted by veatic21_distilled_program use this exact seal.
    if len(payload) > 1:
        unsigned = {key: value for key, value in payload.items() if key != "identity_digest"}
        expected = hashlib.blake2b(
            _canonical_json(unsigned).encode("utf-8"), digest_size=16
        ).hexdigest()
        if digest != expected:
            raise Veatic21ControlError("Frozen AR identity payload digest mismatch")
    # Canonical JSON conversion also proves the payload is persistable.
    _canonical_json(payload)
    return payload


def _expected_identity_matches(
    recorded: Mapping[str, Any] | None,
    expected: str | Mapping[str, Any] | None,
) -> None:
    if expected is None:
        return
    expected_payload = _identity_payload(expected, required=True)
    if recorded is None:
        raise Veatic21ControlError("Persisted control is missing frozen AR identity")
    if isinstance(expected, Mapping):
        if dict(recorded) != expected_payload:
            raise Veatic21ControlError("Frozen AR reuse identity mismatch")
    elif str(recorded.get("identity_digest", "")) != str(
        expected_payload["identity_digest"]
    ):
        raise Veatic21ControlError("Frozen AR reuse identity mismatch")


def _audit_source_pair(
    train: feature_contract.Veatic21Features,
    test: feature_contract.Veatic21Features,
) -> int:
    feature_contract.audit_veatic21_features(train)
    feature_contract.audit_veatic21_features(test)
    if train.pca_width != test.pca_width or train.schema_digest != test.schema_digest:
        raise Veatic21ControlError("Train/test feature schema mismatch")
    train_videos = set(np.asarray(train.video_id, dtype=str).tolist())
    test_videos = set(np.asarray(test.video_id, dtype=str).tolist())
    overlap = sorted(train_videos & test_videos)
    if overlap:
        raise Veatic21ControlError(
            f"Outer train/test video overlap in matched control: {overlap}"
        )
    return int(train.pca_width)


def _current_pca(block: feature_contract.Veatic21Features) -> np.ndarray:
    return np.asarray(block.x_current[:, : block.pca_width], dtype=np.float32)


def _diagnostics(block: feature_contract.Veatic21Features) -> np.ndarray:
    start = int(block.pca_width)
    stop = start + feature_contract.DIAGNOSTIC_WIDTH
    return np.asarray(block.x_current[:, start:stop], dtype=np.float32)


def _build_like(
    source: feature_contract.Veatic21Features,
    *,
    current_pca: np.ndarray,
    diagnostics: np.ndarray,
    preserve_diagnostics: bool,
) -> feature_contract.Veatic21Features:
    current = np.asarray(current_pca, dtype=np.float32)
    diagnostic_values = np.asarray(diagnostics, dtype=np.float32)
    expected_pca_shape = (len(source.row_idx), int(source.pca_width))
    expected_diagnostic_shape = (
        len(source.row_idx),
        feature_contract.DIAGNOSTIC_WIDTH,
    )
    if current.shape != expected_pca_shape:
        raise Veatic21ControlError(
            f"Control PCA shape drift: expected {expected_pca_shape}, got {current.shape}"
        )
    if diagnostic_values.shape != expected_diagnostic_shape:
        raise Veatic21ControlError(
            "Control diagnostic shape drift: "
            f"expected {expected_diagnostic_shape}, got {diagnostic_values.shape}"
        )
    if not np.isfinite(current).all() or not np.isfinite(diagnostic_values).all():
        raise Veatic21ControlError("Control construction produced non-finite features")
    output = feature_contract.build_veatic21_features(
        row_idx=source.row_idx,
        video_id=source.video_id,
        time_seconds=source.time_seconds,
        pca_scores=current,
        diagnostics=diagnostic_values,
        pca_width=source.pca_width,
        pca_row_idx=source.row_idx,
        diagnostic_row_idx=source.row_idx,
    )
    if not np.array_equal(output.row_idx, source.row_idx):
        raise Veatic21ControlError("Control changed recipient row order")
    if not np.array_equal(output.video_id, source.video_id):
        raise Veatic21ControlError("Control changed recipient video ownership")
    if not np.array_equal(output.time_seconds, source.time_seconds):
        raise Veatic21ControlError("Control changed recipient video time")
    if not np.array_equal(output.history_mask, source.history_mask):
        raise Veatic21ControlError("Control changed causal history availability")
    if preserve_diagnostics and not np.array_equal(
        _diagnostics(output), _diagnostics(source)
    ):
        raise Veatic21ControlError("Control failed to preserve non-PCA diagnostics")
    if not np.array_equal(output.x_temporal[:, -2:], source.x_temporal[:, -2:]):
        raise Veatic21ControlError("Control changed recipient time fields")
    return output


def _namespace_seed(namespace: ControlNamespace, *parts: str) -> int:
    payload = "|".join([namespace.digest, *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def _donor_mapping(
    video_id: np.ndarray,
    *,
    namespace: ControlNamespace,
    split: str,
    purpose: str,
) -> dict[str, str]:
    unique = sorted(set(np.asarray(video_id, dtype=str).tolist()))
    if not unique:
        raise Veatic21ControlError("Whole-video control has no videos")
    if len(unique) == 1:
        return {unique[0]: unique[0]}
    ordered = sorted(
        unique,
        key=lambda video: (
            _namespace_seed(namespace, split, purpose, video),
            video,
        ),
    )
    mapping = {
        recipient: ordered[(position + 1) % len(ordered)]
        for position, recipient in enumerate(ordered)
    }
    _audit_mapping(mapping, unique)
    return mapping


def _audit_mapping(mapping: Mapping[str, str], videos: Sequence[str]) -> None:
    unique = set(str(video) for video in videos)
    normalized = {str(key): str(value) for key, value in mapping.items()}
    if set(normalized) != unique or set(normalized.values()) != unique:
        raise Veatic21ControlError("Whole-video donor mapping is not a permutation")
    if len(unique) > 1 and any(
        recipient == donor for recipient, donor in normalized.items()
    ):
        raise Veatic21ControlError(
            "Whole-video donor mapping contains a self-map with multiple videos"
        )


def _relative_time(times: np.ndarray) -> np.ndarray:
    values = np.asarray(times, dtype=np.float64)
    if len(values) <= 1:
        return np.zeros(len(values), dtype=np.float64)
    duration = float(values[-1] - values[0])
    if not duration > 0.0:
        raise Veatic21ControlError("Video time does not increase for resampling")
    return np.clip((values - values[0]) / duration, 0.0, 1.0)


def _linear_resample(
    values: np.ndarray,
    video_id: np.ndarray,
    time_seconds: np.ndarray,
    mapping: Mapping[str, str],
) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    videos = np.asarray(video_id, dtype=str)
    times = np.asarray(time_seconds, dtype=np.float32)
    if source.ndim != 2 or source.shape[0] != len(videos):
        raise Veatic21ControlError("Whole-video PCA source shape mismatch")
    if not np.isfinite(source).all():
        raise Veatic21ControlError("Whole-video PCA source contains non-finite values")
    _audit_mapping(mapping, sorted(set(videos.tolist())))
    output = np.empty_like(source)
    for recipient, donor in mapping.items():
        recipient_rows = np.flatnonzero(videos == recipient)
        donor_rows = np.flatnonzero(videos == donor)
        recipient_progress = _relative_time(times[recipient_rows])
        donor_progress = _relative_time(times[donor_rows])
        donor_values = source[donor_rows]
        if len(donor_rows) == 1:
            output[recipient_rows] = donor_values[0]
            continue
        right = np.searchsorted(donor_progress, recipient_progress, side="left")
        right = np.clip(right, 1, len(donor_rows) - 1)
        left = right - 1
        span = donor_progress[right] - donor_progress[left]
        if np.any(span <= 0.0):
            raise Veatic21ControlError("Donor video time is not strictly increasing")
        weight = ((recipient_progress - donor_progress[left]) / span).reshape(-1, 1)
        interpolated = donor_values[left] * (1.0 - weight) + donor_values[right] * weight
        output[recipient_rows] = interpolated.astype(np.float32)
    if not np.isfinite(output).all():
        raise Veatic21ControlError("Whole-video PCA resampling produced non-finite values")
    return output


def _nearest_resample(
    values: np.ndarray,
    video_id: np.ndarray,
    time_seconds: np.ndarray,
    mapping: Mapping[str, str],
) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    videos = np.asarray(video_id, dtype=str)
    times = np.asarray(time_seconds, dtype=np.float32)
    _audit_mapping(mapping, sorted(set(videos.tolist())))
    output = np.empty_like(source)
    for recipient, donor in mapping.items():
        recipient_rows = np.flatnonzero(videos == recipient)
        donor_rows = np.flatnonzero(videos == donor)
        recipient_progress = _relative_time(times[recipient_rows])
        donor_progress = _relative_time(times[donor_rows])
        distances = np.abs(
            recipient_progress.reshape(-1, 1) - donor_progress.reshape(1, -1)
        )
        nearest = np.argmin(distances, axis=1)
        output[recipient_rows] = source[donor_rows[nearest]]
    return output


def _fit_mask(mask: np.ndarray | Sequence[bool] | None, rows: int) -> np.ndarray:
    if mask is None:
        output = np.ones(rows, dtype=bool)
    else:
        raw = np.asarray(mask)
        if raw.shape != (rows,) or raw.dtype.kind != "b":
            raise Veatic21ControlError("Outer-training fit mask must be a 1D bool array")
        output = raw.astype(bool, copy=True)
    if int(np.sum(output)) < 2:
        raise Veatic21ControlError(
            "Matched control requires at least two outer-training fit rows"
        )
    return output


def _parameter_metadata(parameters: Mapping[str, np.ndarray]) -> dict[str, object]:
    metadata: dict[str, object] = {}
    for raw_name, raw_values in sorted(parameters.items()):
        name = str(raw_name)
        if not name or not name.replace("_", "").isalnum():
            raise Veatic21ControlError(f"Invalid parameter-array name {name!r}")
        values = np.asarray(raw_values)
        if values.dtype.kind in {"f", "c"} and not np.isfinite(values).all():
            raise Veatic21ControlError(f"Parameter array {name!r} is non-finite")
        metadata[name] = {
            "shape": list(values.shape),
            "dtype": values.dtype.str,
            "digest": array_digest(values),
        }
    return metadata


def _seal_record(payload: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(payload)
    record["record_digest"] = canonical_digest(record)
    return record


def _finalize_feature_control(
    *,
    namespace: ControlNamespace,
    kind: str,
    train_source: feature_contract.Veatic21Features,
    test_source: feature_contract.Veatic21Features,
    train_output: feature_contract.Veatic21Features,
    test_output: feature_contract.Veatic21Features,
    construction: Mapping[str, Any],
    parameter_arrays: Mapping[str, np.ndarray],
    privileged: bool,
    frozen_ar_identity: str | Mapping[str, Any] | None,
) -> SealedFeatureControl:
    width = _audit_source_pair(train_source, test_source)
    _audit_source_pair(train_output, test_output)
    identity = _identity_payload(frozen_ar_identity, required=privileged)
    parameters = {
        str(name): np.asarray(values).copy()
        for name, values in parameter_arrays.items()
    }
    record = _seal_record(
        {
            "schema_name": CONTROL_SCHEMA_NAME,
            "schema_version": CONTROL_SCHEMA_VERSION,
            "artifact_type": "feature_control",
            "control_kind": kind,
            "namespace": namespace.as_dict(),
            "namespace_digest": namespace.digest,
            "privileged": bool(privileged),
            "frozen_ar_identity": identity,
            "pca_width": width,
            "feature_schema_digest": train_output.schema_digest,
            "source_train_digest": feature_block_digest(train_source),
            "source_test_digest": feature_block_digest(test_source),
            "output_train_digest": feature_block_digest(train_output),
            "output_test_digest": feature_block_digest(test_output),
            "train_row_digest": array_digest(train_output.row_idx),
            "test_row_digest": array_digest(test_output.row_idx),
            "train_video_digest": array_digest(train_output.video_id),
            "test_video_digest": array_digest(test_output.video_id),
            "construction": dict(construction),
            "parameter_arrays": _parameter_metadata(parameters),
            "row_order_preserved": True,
            "response_free_feature_schema": True,
        }
    )
    control = SealedFeatureControl(
        namespace=namespace,
        train=train_output,
        test=test_output,
        record=record,
        parameter_arrays=parameters,
    )
    audit_feature_control(control, expected_frozen_ar_identity=frozen_ar_identity)
    return control


def build_sequence_shuffled_pca_control(
    train: feature_contract.Veatic21Features,
    test: feature_contract.Veatic21Features,
    *,
    namespace: ControlNamespace,
    privileged: bool = False,
    frozen_ar_identity: str | Mapping[str, Any] | None = None,
) -> SealedFeatureControl:
    """Reassign intact PCA video trajectories and preserve recipient metadata."""

    _audit_source_pair(train, test)
    train_mapping = _donor_mapping(
        train.video_id,
        namespace=namespace,
        split="outer_train",
        purpose="sequence_shuffled_pca",
    )
    test_mapping = _donor_mapping(
        test.video_id,
        namespace=namespace,
        split="outer_test",
        purpose="sequence_shuffled_pca",
    )
    shuffled_train = _linear_resample(
        _current_pca(train), train.video_id, train.time_seconds, train_mapping
    )
    shuffled_test = _linear_resample(
        _current_pca(test), test.video_id, test.time_seconds, test_mapping
    )
    train_output = _build_like(
        train,
        current_pca=shuffled_train,
        diagnostics=_diagnostics(train),
        preserve_diagnostics=True,
    )
    test_output = _build_like(
        test,
        current_pca=shuffled_test,
        diagnostics=_diagnostics(test),
        preserve_diagnostics=True,
    )
    construction = {
        "resample_policy": LINEAR_RESAMPLE_POLICY,
        "row_shuffle": False,
        "replace_blocks": ["pca"],
        "preserve_blocks": ["diagnostics", "history_mask", "time"],
        "train_donor_mapping": train_mapping,
        "test_donor_mapping": test_mapping,
        "train_mapping_digest": canonical_digest(train_mapping),
        "test_mapping_digest": canonical_digest(test_mapping),
        "single_video_self_map_only": True,
    }
    return _finalize_feature_control(
        namespace=namespace,
        kind="sequence_shuffled_pca",
        train_source=train,
        test_source=test,
        train_output=train_output,
        test_output=test_output,
        construction=construction,
        parameter_arrays={},
        privileged=privileged,
        frozen_ar_identity=frozen_ar_identity,
    )


def build_matched_random_pca_control(
    train: feature_contract.Veatic21Features,
    test: feature_contract.Veatic21Features,
    *,
    namespace: ControlNamespace,
    frozen_ar_identity: str | Mapping[str, Any],
    train_fit_mask: np.ndarray | Sequence[bool] | None = None,
) -> SealedFeatureControl:
    """Generate random PCA rows matched only to outer-training mean and scale."""

    _audit_source_pair(train, test)
    train_pca = _current_pca(train)
    fit = _fit_mask(train_fit_mask, len(train_pca))
    fit_values = train_pca[fit].astype(np.float64)
    fit_mean = np.mean(fit_values, axis=0).astype(np.float32)
    fit_std = np.std(fit_values, axis=0).astype(np.float32)
    if not np.isfinite(fit_mean).all() or not np.isfinite(fit_std).all():
        raise Veatic21ControlError("Outer-training random-control fit is non-finite")

    train_seed = _namespace_seed(namespace, "matched_random_pca", "outer_train")
    test_seed = _namespace_seed(namespace, "matched_random_pca", "outer_test")
    train_noise = np.random.default_rng(train_seed).standard_normal(train_pca.shape)
    test_noise = np.random.default_rng(test_seed).standard_normal(
        (len(test.row_idx), train.pca_width)
    )
    noise_mean = np.mean(train_noise[fit], axis=0)
    noise_std = np.std(train_noise[fit], axis=0)
    if np.any(noise_std < 1e-12) or not np.isfinite(noise_std).all():
        raise Veatic21ControlError("Random-control normalization is degenerate")
    train_noise = (train_noise - noise_mean) / noise_std
    random_train = (train_noise * fit_std + fit_mean).astype(np.float32)
    random_test = (test_noise * fit_std + fit_mean).astype(np.float32)

    train_output = _build_like(
        train,
        current_pca=random_train,
        diagnostics=_diagnostics(train),
        preserve_diagnostics=True,
    )
    test_output = _build_like(
        test,
        current_pca=random_test,
        diagnostics=_diagnostics(test),
        preserve_diagnostics=True,
    )
    parameters = {
        "fit_mean": fit_mean,
        "fit_std": fit_std,
        "fit_mask": fit.astype(np.uint8),
        "train_noise_mean": noise_mean.astype(np.float64),
        "train_noise_std": noise_std.astype(np.float64),
    }
    construction = {
        "fit_scope": "outer_train_only",
        "test_values_used_for_fit": False,
        "distribution_match": "per_pca_component_mean_and_population_std",
        "train_fit_rows": int(np.sum(fit)),
        "train_fit_row_digest": array_digest(np.asarray(train.row_idx)[fit]),
        "train_rng_seed": train_seed,
        "test_rng_seed": test_seed,
        "replace_blocks": ["pca"],
        "preserve_blocks": ["diagnostics", "history_mask", "time"],
    }
    return _finalize_feature_control(
        namespace=namespace,
        kind="matched_random_pca",
        train_source=train,
        test_source=test,
        train_output=train_output,
        test_output=test_output,
        construction=construction,
        parameter_arrays=parameters,
        privileged=True,
        frozen_ar_identity=frozen_ar_identity,
    )


def build_train_only_video_mean_control(
    train: feature_contract.Veatic21Features,
    test: feature_contract.Veatic21Features,
    *,
    namespace: ControlNamespace,
    frozen_ar_identity: str | Mapping[str, Any],
    train_fit_mask: np.ndarray | Sequence[bool] | None = None,
) -> SealedFeatureControl:
    """Replace PCA with train-video means; unseen held-out videos use train global."""

    _audit_source_pair(train, test)
    train_pca = _current_pca(train)
    fit = _fit_mask(train_fit_mask, len(train_pca))
    global_mean = np.mean(train_pca[fit], axis=0, dtype=np.float64).astype(np.float32)
    if not np.isfinite(global_mean).all():
        raise Veatic21ControlError("Outer-training global PCA mean is non-finite")

    training_videos = sorted(set(np.asarray(train.video_id, dtype=str).tolist()))
    means: list[np.ndarray] = []
    fallback_videos: list[str] = []
    for video in training_videos:
        rows = (np.asarray(train.video_id, dtype=str) == video) & fit
        if np.any(rows):
            value = np.mean(train_pca[rows], axis=0, dtype=np.float64).astype(np.float32)
        else:
            value = global_mean.copy()
            fallback_videos.append(video)
        means.append(value)
    mean_matrix = np.stack(means, axis=0).astype(np.float32)
    lookup = {video: mean_matrix[index] for index, video in enumerate(training_videos)}
    mean_train = np.stack(
        [lookup[str(video)] for video in np.asarray(train.video_id, dtype=str)], axis=0
    ).astype(np.float32)
    # Split audit guarantees every held-out video is unseen by outer training.
    mean_test = np.repeat(global_mean.reshape(1, -1), len(test.row_idx), axis=0)

    train_output = _build_like(
        train,
        current_pca=mean_train,
        diagnostics=_diagnostics(train),
        preserve_diagnostics=True,
    )
    test_output = _build_like(
        test,
        current_pca=mean_test,
        diagnostics=_diagnostics(test),
        preserve_diagnostics=True,
    )
    construction = {
        "fit_scope": "outer_train_only",
        "test_values_used_for_fit": False,
        "held_out_policy": "global_outer_train_mean",
        "training_video_order": training_videos,
        "training_videos_without_fit_rows": fallback_videos,
        "held_out_global_fallback_rows": int(len(test.row_idx)),
        "train_fit_rows": int(np.sum(fit)),
        "train_fit_row_digest": array_digest(np.asarray(train.row_idx)[fit]),
        "replace_blocks": ["pca"],
        "preserve_blocks": ["diagnostics", "history_mask", "time"],
    }
    parameters = {
        "global_mean": global_mean,
        "training_video_means": mean_matrix,
        "fit_mask": fit.astype(np.uint8),
    }
    return _finalize_feature_control(
        namespace=namespace,
        kind="train_only_video_mean_pca",
        train_source=train,
        test_source=test,
        train_output=train_output,
        test_output=test_output,
        construction=construction,
        parameter_arrays=parameters,
        privileged=True,
        frozen_ar_identity=frozen_ar_identity,
    )


def build_diagnostics_only_control(
    train: feature_contract.Veatic21Features,
    test: feature_contract.Veatic21Features,
    *,
    namespace: ControlNamespace,
    privileged: bool = False,
    frozen_ar_identity: str | Mapping[str, Any] | None = None,
) -> SealedFeatureControl:
    """Zero PCA while retaining diagnostics, causal mask, and video time."""

    _audit_source_pair(train, test)
    train_output = _build_like(
        train,
        current_pca=np.zeros_like(_current_pca(train)),
        diagnostics=_diagnostics(train),
        preserve_diagnostics=True,
    )
    test_output = _build_like(
        test,
        current_pca=np.zeros_like(_current_pca(test)),
        diagnostics=_diagnostics(test),
        preserve_diagnostics=True,
    )
    return _finalize_feature_control(
        namespace=namespace,
        kind="diagnostics_only",
        train_source=train,
        test_source=test,
        train_output=train_output,
        test_output=test_output,
        construction={
            "zero_blocks": ["pca"],
            "preserve_blocks": ["diagnostics", "history_mask", "time"],
        },
        parameter_arrays={},
        privileged=privileged,
        frozen_ar_identity=frozen_ar_identity,
    )


def build_no_video_control(
    train: feature_contract.Veatic21Features,
    test: feature_contract.Veatic21Features,
    *,
    namespace: ControlNamespace,
) -> SealedFeatureControl:
    """Zero PCA and diagnostics while retaining shape, mask, and time fields."""

    _audit_source_pair(train, test)
    train_output = _build_like(
        train,
        current_pca=np.zeros_like(_current_pca(train)),
        diagnostics=np.zeros_like(_diagnostics(train)),
        preserve_diagnostics=False,
    )
    test_output = _build_like(
        test,
        current_pca=np.zeros_like(_current_pca(test)),
        diagnostics=np.zeros_like(_diagnostics(test)),
        preserve_diagnostics=False,
    )
    return _finalize_feature_control(
        namespace=namespace,
        kind="no_video",
        train_source=train,
        test_source=test,
        train_output=train_output,
        test_output=test_output,
        construction={
            "zero_blocks": ["pca", "diagnostics"],
            "preserve_blocks": ["history_mask", "time"],
        },
        parameter_arrays={},
        privileged=False,
        frozen_ar_identity=None,
    )


def _target_metadata(
    row_idx: np.ndarray | Sequence[int],
    video_id: np.ndarray | Sequence[str],
    time_seconds: np.ndarray | Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Reuse the exact ordering/time contract used by the feature block builder.
    try:
        return feature_contract._aligned_metadata(row_idx, video_id, time_seconds)
    except feature_contract.Veatic21FeatureAuditError as exc:
        raise Veatic21ControlError(str(exc)) from exc


def build_whole_video_label_permutation(
    *,
    row_idx: np.ndarray | Sequence[int],
    video_id: np.ndarray | Sequence[str],
    time_seconds: np.ndarray | Sequence[float],
    target: np.ndarray | Sequence[float],
    namespace: ControlNamespace,
    privileged: bool = False,
    frozen_ar_identity: str | Mapping[str, Any] | None = None,
) -> SealedTargetControl:
    """Permute complete label trajectories without ever shuffling target rows."""

    idx, videos, times = _target_metadata(row_idx, video_id, time_seconds)
    raw_target = np.asarray(target)
    if raw_target.ndim != 1 or len(raw_target) != len(idx):
        raise Veatic21ControlError("Label-permutation target shape mismatch")
    if not np.issubdtype(raw_target.dtype, np.number):
        raise Veatic21ControlError("Label-permutation target must be numeric")
    values = raw_target.astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise Veatic21ControlError(
            "Label-permutation target must be finite; pass only target-valid rows"
        )
    identity = _identity_payload(frozen_ar_identity, required=privileged)
    mapping = _donor_mapping(
        videos,
        namespace=namespace,
        split="target_rows",
        purpose="whole_video_label_permutation",
    )
    permuted = _nearest_resample(values, videos, times, mapping).astype(np.float32)
    if permuted.shape != values.shape or not np.isfinite(permuted).all():
        raise Veatic21ControlError("Label permutation produced shape/non-finite drift")
    record = _seal_record(
        {
            "schema_name": CONTROL_SCHEMA_NAME,
            "schema_version": CONTROL_SCHEMA_VERSION,
            "artifact_type": "target_control",
            "control_kind": "whole_video_label_permutation",
            "namespace": namespace.as_dict(),
            "namespace_digest": namespace.digest,
            "privileged": bool(privileged),
            "frozen_ar_identity": identity,
            "row_digest": array_digest(idx),
            "video_digest": array_digest(videos),
            "time_digest": array_digest(times),
            "source_target_digest": array_digest(values),
            "output_target_digest": array_digest(permuted),
            "construction": {
                "resample_policy": NEAREST_RESAMPLE_POLICY,
                "row_shuffle": False,
                "donor_mapping": mapping,
                "mapping_digest": canonical_digest(mapping),
                "single_video_self_map_only": True,
            },
            "row_order_preserved": True,
        }
    )
    control = SealedTargetControl(
        namespace=namespace,
        row_idx=idx.copy(),
        video_id=videos.copy(),
        time_seconds=times.copy(),
        target=permuted,
        record=record,
    )
    audit_target_control(control, expected_frozen_ar_identity=frozen_ar_identity)
    return control


def _audit_record(record: Mapping[str, Any], namespace: ControlNamespace) -> None:
    if record.get("schema_name") != CONTROL_SCHEMA_NAME:
        raise Veatic21ControlError("Control schema name mismatch")
    if int(record.get("schema_version", -1)) != CONTROL_SCHEMA_VERSION:
        raise Veatic21ControlError("Control schema version mismatch")
    if record.get("namespace") != namespace.as_dict():
        raise Veatic21ControlError("Control namespace mismatch")
    if record.get("namespace_digest") != namespace.digest:
        raise Veatic21ControlError("Control namespace digest mismatch")
    supplied = str(record.get("record_digest", ""))
    unsigned = {key: value for key, value in record.items() if key != "record_digest"}
    if not supplied or supplied != canonical_digest(unsigned):
        raise Veatic21ControlError("Control record digest mismatch")


def audit_feature_control(
    control: SealedFeatureControl,
    *,
    expected_namespace: ControlNamespace | None = None,
    expected_frozen_ar_identity: str | Mapping[str, Any] | None = None,
) -> ControlAudit:
    """Fail closed on schema, output, mapping, parameter, or identity drift."""

    if expected_namespace is not None and control.namespace != expected_namespace:
        raise Veatic21ControlError("Control reuse namespace mismatch")
    _audit_record(control.record, control.namespace)
    _audit_source_pair(control.train, control.test)
    if control.record.get("artifact_type") != "feature_control":
        raise Veatic21ControlError("Expected a feature-control artifact")
    if int(control.record.get("pca_width", -1)) != control.train.pca_width:
        raise Veatic21ControlError("Persisted PCA width mismatch")
    if control.record.get("feature_schema_digest") != control.train.schema_digest:
        raise Veatic21ControlError("Persisted feature schema digest mismatch")
    expected_outputs = {
        "output_train_digest": feature_block_digest(control.train),
        "output_test_digest": feature_block_digest(control.test),
        "train_row_digest": array_digest(control.train.row_idx),
        "test_row_digest": array_digest(control.test.row_idx),
        "train_video_digest": array_digest(control.train.video_id),
        "test_video_digest": array_digest(control.test.video_id),
    }
    for field_name, expected in expected_outputs.items():
        if control.record.get(field_name) != expected:
            raise Veatic21ControlError(f"Control output identity mismatch: {field_name}")

    privileged = bool(control.record.get("privileged", False))
    recorded_identity = control.record.get("frozen_ar_identity")
    if privileged:
        _identity_payload(recorded_identity, required=True)
    elif recorded_identity is not None:
        raise Veatic21ControlError("Zero-label record unexpectedly contains frozen AR")
    _expected_identity_matches(recorded_identity, expected_frozen_ar_identity)

    expected_parameters = _parameter_metadata(control.parameter_arrays)
    if control.record.get("parameter_arrays") != expected_parameters:
        raise Veatic21ControlError("Control parameter-array digest mismatch")

    kind = str(control.record.get("control_kind", ""))
    construction = control.record.get("construction")
    if not isinstance(construction, Mapping):
        raise Veatic21ControlError("Control construction record is missing")
    if kind == "sequence_shuffled_pca":
        for split, block in (("train", control.train), ("test", control.test)):
            raw_mapping = construction.get(f"{split}_donor_mapping")
            if not isinstance(raw_mapping, Mapping):
                raise Veatic21ControlError("Sequence control donor mapping is missing")
            _audit_mapping(raw_mapping, sorted(set(block.video_id.astype(str).tolist())))
            if construction.get(f"{split}_mapping_digest") != canonical_digest(
                dict(raw_mapping)
            ):
                raise Veatic21ControlError("Sequence control mapping digest mismatch")

    return ControlAudit(
        passed=True,
        checks=(
            "namespace",
            "split_disjoint",
            "response_free_schema",
            "shape_and_finite",
            "output_digests",
            "parameter_digests",
            "frozen_ar_identity",
            "mapping_policy",
        ),
        record_digest=str(control.record["record_digest"]),
    )


def audit_target_control(
    control: SealedTargetControl,
    *,
    expected_namespace: ControlNamespace | None = None,
    expected_frozen_ar_identity: str | Mapping[str, Any] | None = None,
) -> ControlAudit:
    if expected_namespace is not None and control.namespace != expected_namespace:
        raise Veatic21ControlError("Target-control reuse namespace mismatch")
    _audit_record(control.record, control.namespace)
    idx, videos, times = _target_metadata(
        control.row_idx, control.video_id, control.time_seconds
    )
    values = np.asarray(control.target)
    if values.shape != (len(idx),) or values.dtype != np.float32:
        raise Veatic21ControlError("Target-control shape/dtype drift")
    if not np.isfinite(values).all():
        raise Veatic21ControlError("Target-control contains non-finite values")
    expected = {
        "row_digest": array_digest(idx),
        "video_digest": array_digest(videos),
        "time_digest": array_digest(times),
        "output_target_digest": array_digest(values),
    }
    for field_name, digest in expected.items():
        if control.record.get(field_name) != digest:
            raise Veatic21ControlError(f"Target-control identity mismatch: {field_name}")
    construction = control.record.get("construction")
    if not isinstance(construction, Mapping):
        raise Veatic21ControlError("Target-control construction record is missing")
    mapping = construction.get("donor_mapping")
    if not isinstance(mapping, Mapping):
        raise Veatic21ControlError("Target-control donor mapping is missing")
    _audit_mapping(mapping, sorted(set(videos.tolist())))
    if construction.get("mapping_digest") != canonical_digest(dict(mapping)):
        raise Veatic21ControlError("Target-control mapping digest mismatch")
    privileged = bool(control.record.get("privileged", False))
    recorded_identity = control.record.get("frozen_ar_identity")
    if privileged:
        _identity_payload(recorded_identity, required=True)
    elif recorded_identity is not None:
        raise Veatic21ControlError("Zero-label target record contains frozen AR")
    _expected_identity_matches(recorded_identity, expected_frozen_ar_identity)
    return ControlAudit(
        passed=True,
        checks=(
            "namespace",
            "row_identity",
            "finite_target",
            "whole_video_mapping",
            "output_digest",
            "frozen_ar_identity",
        ),
        record_digest=str(control.record["record_digest"]),
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def feature_control_paths(root: Path | str, control: SealedFeatureControl) -> ControlPaths:
    base = Path(root)
    stem = f"{control.record['control_kind']}__{control.namespace.digest}"
    return ControlPaths(base / f"{stem}.json", base / f"{stem}.npz")


def target_control_paths(root: Path | str, control: SealedTargetControl) -> ControlPaths:
    base = Path(root)
    stem = f"{control.record['control_kind']}__{control.namespace.digest}"
    return ControlPaths(base / f"{stem}.json", base / f"{stem}.npz")


def _feature_arrays(control: SealedFeatureControl) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for split, block in (("train", control.train), ("test", control.test)):
        arrays[f"{split}_row_idx"] = np.asarray(block.row_idx)
        arrays[f"{split}_video_id"] = np.asarray(block.video_id, dtype=str)
        arrays[f"{split}_time_seconds"] = np.asarray(block.time_seconds)
        arrays[f"{split}_x_temporal"] = np.asarray(block.x_temporal)
        arrays[f"{split}_x_current"] = np.asarray(block.x_current)
        arrays[f"{split}_history_mask"] = np.asarray(block.history_mask)
    for name, values in control.parameter_arrays.items():
        arrays[f"parameter__{name}"] = np.asarray(values)
    return arrays


def persist_feature_control(
    control: SealedFeatureControl,
    root: Path | str,
) -> ControlPaths:
    """Persist or exactly reuse one feature control; never overwrite mismatch."""

    audit_feature_control(control)
    paths = feature_control_paths(root, control)
    if paths.manifest.exists() or paths.arrays.exists():
        if not paths.manifest.exists() or not paths.arrays.exists():
            raise Veatic21ControlError("Incomplete persisted feature-control artifact")
        loaded = load_feature_control(
            paths.manifest,
            expected_namespace=control.namespace,
            expected_frozen_ar_identity=control.record.get("frozen_ar_identity"),
        )
        if loaded.record != control.record:
            raise Veatic21ControlError("Feature-control reuse record mismatch")
        return paths
    _atomic_npz(paths.arrays, _feature_arrays(control))
    manifest = {
        "manifest_schema": "veatic21_control_manifest_v1",
        "artifact_type": "feature_control",
        "arrays_file": paths.arrays.name,
        "arrays_sha256": _file_sha256(paths.arrays),
        "record": dict(control.record),
    }
    manifest["manifest_digest"] = canonical_digest(manifest)
    _atomic_json(paths.manifest, manifest)
    return paths


def _loaded_block(
    arrays: Mapping[str, np.ndarray],
    *,
    split: str,
    width: int,
    schema_digest: str,
) -> feature_contract.Veatic21Features:
    block = feature_contract.Veatic21Features(
        row_idx=np.asarray(arrays[f"{split}_row_idx"]),
        video_id=np.asarray(arrays[f"{split}_video_id"], dtype=str),
        time_seconds=np.asarray(arrays[f"{split}_time_seconds"]),
        x_temporal=np.asarray(arrays[f"{split}_x_temporal"]),
        x_current=np.asarray(arrays[f"{split}_x_current"]),
        history_mask=np.asarray(arrays[f"{split}_history_mask"]),
        temporal_feature_names=feature_contract.feature_names(width, view="temporal"),
        current_feature_names=feature_contract.feature_names(width, view="current"),
        schema_digest=schema_digest,
        pca_width=width,
    )
    feature_contract.audit_veatic21_features(block)
    return block


def _load_manifest(path: Path, *, artifact_type: str) -> tuple[dict[str, Any], Path]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Veatic21ControlError(f"Cannot read control manifest {path}") from exc
    supplied = str(manifest.get("manifest_digest", ""))
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    if not supplied or supplied != canonical_digest(unsigned):
        raise Veatic21ControlError("Control manifest digest mismatch")
    if manifest.get("artifact_type") != artifact_type:
        raise Veatic21ControlError("Control manifest artifact type mismatch")
    arrays_name = manifest.get("arrays_file")
    if not isinstance(arrays_name, str) or Path(arrays_name).name != arrays_name:
        raise Veatic21ControlError("Control manifest arrays path is unsafe")
    arrays_path = path.parent / arrays_name
    if not arrays_path.is_file() or _file_sha256(arrays_path) != manifest.get(
        "arrays_sha256"
    ):
        raise Veatic21ControlError("Control arrays file checksum mismatch")
    return manifest, arrays_path


def load_feature_control(
    manifest_path: Path | str,
    *,
    expected_namespace: ControlNamespace | None = None,
    expected_frozen_ar_identity: str | Mapping[str, Any] | None = None,
) -> SealedFeatureControl:
    path = Path(manifest_path)
    manifest, arrays_path = _load_manifest(path, artifact_type="feature_control")
    record = manifest.get("record")
    if not isinstance(record, Mapping):
        raise Veatic21ControlError("Feature-control manifest record is missing")
    namespace = ControlNamespace(**dict(record["namespace"]))
    if expected_namespace is not None and namespace != expected_namespace:
        raise Veatic21ControlError("Feature-control reuse namespace mismatch")
    width = int(record["pca_width"])
    schema_digest = str(record["feature_schema_digest"])
    try:
        with np.load(arrays_path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        raise Veatic21ControlError("Cannot load feature-control arrays") from exc
    train = _loaded_block(
        arrays, split="train", width=width, schema_digest=schema_digest
    )
    test = _loaded_block(arrays, split="test", width=width, schema_digest=schema_digest)
    parameter_metadata = record.get("parameter_arrays")
    if not isinstance(parameter_metadata, Mapping):
        raise Veatic21ControlError("Feature-control parameter metadata is missing")
    parameters: dict[str, np.ndarray] = {}
    for name in parameter_metadata:
        key = f"parameter__{name}"
        if key not in arrays:
            raise Veatic21ControlError(f"Feature-control parameter {name!r} is missing")
        parameters[str(name)] = arrays[key]
    expected_keys = {
        *(f"{split}_{name}" for split in ("train", "test") for name in (
            "row_idx",
            "video_id",
            "time_seconds",
            "x_temporal",
            "x_current",
            "history_mask",
        )),
        *(f"parameter__{name}" for name in parameter_metadata),
    }
    if set(arrays) != expected_keys:
        raise Veatic21ControlError("Feature-control arrays contain schema drift")
    control = SealedFeatureControl(namespace, train, test, dict(record), parameters)
    audit_feature_control(
        control,
        expected_namespace=expected_namespace,
        expected_frozen_ar_identity=expected_frozen_ar_identity,
    )
    return control


def persist_target_control(
    control: SealedTargetControl,
    root: Path | str,
) -> ControlPaths:
    audit_target_control(control)
    paths = target_control_paths(root, control)
    if paths.manifest.exists() or paths.arrays.exists():
        if not paths.manifest.exists() or not paths.arrays.exists():
            raise Veatic21ControlError("Incomplete persisted target-control artifact")
        loaded = load_target_control(
            paths.manifest,
            expected_namespace=control.namespace,
            expected_frozen_ar_identity=control.record.get("frozen_ar_identity"),
        )
        if loaded.record != control.record:
            raise Veatic21ControlError("Target-control reuse record mismatch")
        return paths
    _atomic_npz(
        paths.arrays,
        {
            "row_idx": control.row_idx,
            "video_id": np.asarray(control.video_id, dtype=str),
            "time_seconds": control.time_seconds,
            "target": control.target,
        },
    )
    manifest = {
        "manifest_schema": "veatic21_control_manifest_v1",
        "artifact_type": "target_control",
        "arrays_file": paths.arrays.name,
        "arrays_sha256": _file_sha256(paths.arrays),
        "record": dict(control.record),
    }
    manifest["manifest_digest"] = canonical_digest(manifest)
    _atomic_json(paths.manifest, manifest)
    return paths


def load_target_control(
    manifest_path: Path | str,
    *,
    expected_namespace: ControlNamespace | None = None,
    expected_frozen_ar_identity: str | Mapping[str, Any] | None = None,
) -> SealedTargetControl:
    path = Path(manifest_path)
    manifest, arrays_path = _load_manifest(path, artifact_type="target_control")
    record = manifest.get("record")
    if not isinstance(record, Mapping):
        raise Veatic21ControlError("Target-control manifest record is missing")
    namespace = ControlNamespace(**dict(record["namespace"]))
    if expected_namespace is not None and namespace != expected_namespace:
        raise Veatic21ControlError("Target-control reuse namespace mismatch")
    try:
        with np.load(arrays_path, allow_pickle=False) as archive:
            if set(archive.files) != {"row_idx", "video_id", "time_seconds", "target"}:
                raise Veatic21ControlError("Target-control arrays contain schema drift")
            control = SealedTargetControl(
                namespace=namespace,
                row_idx=np.asarray(archive["row_idx"]),
                video_id=np.asarray(archive["video_id"], dtype=str),
                time_seconds=np.asarray(archive["time_seconds"]),
                target=np.asarray(archive["target"]),
                record=dict(record),
            )
    except (OSError, ValueError, KeyError) as exc:
        raise Veatic21ControlError("Cannot load target-control arrays") from exc
    audit_target_control(
        control,
        expected_namespace=expected_namespace,
        expected_frozen_ar_identity=expected_frozen_ar_identity,
    )
    return control

