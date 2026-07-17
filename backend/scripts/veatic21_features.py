"""Label-free VEATIC 2.1 video feature construction and audit contracts.

This module deliberately knows nothing about VEATIC response annotations.  It
only combines fold-safe PCA scores derived from video, TRIBE temporal
diagnostics, and causal video-time metadata.  The same builder can therefore
be used by arousal, valence, and future combined-dataset experiments without
silently introducing a response-side input.

The temporal view is ordered from lag 4 through the current row (lag 0).
Unavailable history is zero-filled and accompanied by one availability bit per
lag.  Every video starts with the cold-start mask ``[0, 0, 0, 0, 1]``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Literal, Sequence

import numpy as np


SCHEMA_NAME = "veatic21_video_only_features_v1"
SCHEMA_VERSION = 1
ALLOWED_PCA_WIDTHS = (64, 128, 256)
WINDOW_ROWS = 5
DIAGNOSTIC_WIDTH = 53
TIME_FEATURE_NAMES = ("log1p_time_seconds", "within_video_time_fraction")
COLD_START_POLICY = "same_video_zero_fill_current_row_available_no_label_burnin"

# Substring matching is intentional: a compound field such as
# ``normalized_future_arousal`` must fail just as decisively as ``arousal``.
FORBIDDEN_RESPONSE_TOKENS = (
    "arousal",
    "valence",
    "label",
    "target",
    "future_",
    "ground_truth",
    "teacher",
    "response",
    "ar_score",
    "ar_reg",
)


class Veatic21FeatureAuditError(ValueError):
    """Raised when a video-only feature block violates its sealed contract."""


@dataclass(frozen=True)
class Veatic21Features:
    """Aligned label-free current-row and five-row temporal feature views."""

    row_idx: np.ndarray
    video_id: np.ndarray
    time_seconds: np.ndarray
    x_temporal: np.ndarray
    x_current: np.ndarray
    history_mask: np.ndarray
    temporal_feature_names: tuple[str, ...]
    current_feature_names: tuple[str, ...]
    schema_digest: str
    pca_width: int

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Compatibility alias for consumers whose primary view is temporal."""

        return self.temporal_feature_names


@dataclass(frozen=True)
class Veatic21FeatureAudit:
    """Successful fail-closed audit result.

    Invalid inputs never produce a report: :func:`audit_veatic21_features`
    raises :class:`Veatic21FeatureAuditError` at the first failed invariant.
    """

    passed: bool
    checks: tuple[str, ...]
    schema_digest: str
    pca_width: int
    rows: int
    videos: int


def _checked_pca_width(pca_width: int) -> int:
    if isinstance(pca_width, (bool, np.bool_)):
        raise ValueError("PCA width must be one of 64, 128, or 256")
    try:
        width = int(pca_width)
    except (TypeError, ValueError) as exc:
        raise ValueError("PCA width must be one of 64, 128, or 256") from exc
    if width != pca_width or width not in ALLOWED_PCA_WIDTHS:
        raise ValueError(
            f"Unsupported PCA width {pca_width!r}; expected one of {ALLOWED_PCA_WIDTHS}"
        )
    return width


def feature_names(
    pca_width: int,
    *,
    view: Literal["temporal", "current"] = "temporal",
) -> tuple[str, ...]:
    """Return the ordered, response-free feature names for one model view."""

    width = _checked_pca_width(pca_width)
    diagnostic_names = tuple(
        f"temporal_diagnostic_{column}" for column in range(DIAGNOSTIC_WIDTH)
    )
    if view == "temporal":
        pca_names = tuple(
            f"pca{width}_lag{lag}_{column}"
            for lag in range(WINDOW_ROWS - 1, -1, -1)
            for column in range(width)
        )
        mask_names = tuple(
            f"history_available_lag{lag}" for lag in range(WINDOW_ROWS - 1, -1, -1)
        )
    elif view == "current":
        pca_names = tuple(f"pca{width}_lag0_{column}" for column in range(width))
        mask_names = ("history_available_lag0",)
    else:
        raise ValueError(f"Unknown feature view {view!r}")
    return pca_names + diagnostic_names + mask_names + TIME_FEATURE_NAMES


def feature_schema(pca_width: int) -> dict[str, object]:
    """Return the canonical schema payload used to seal feature compatibility."""

    width = _checked_pca_width(pca_width)
    temporal_names = feature_names(width, view="temporal")
    current_names = feature_names(width, view="current")
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "pca_width": width,
        "allowed_pca_widths": list(ALLOWED_PCA_WIDTHS),
        "window_rows": WINDOW_ROWS,
        "causal_order": [4, 3, 2, 1, 0],
        "diagnostic_width": DIAGNOSTIC_WIDTH,
        "time_feature_names": list(TIME_FEATURE_NAMES),
        "cold_start_policy": COLD_START_POLICY,
        "temporal_width": len(temporal_names),
        "current_width": len(current_names),
        "temporal_feature_names": list(temporal_names),
        "current_feature_names": list(current_names),
    }


def feature_schema_digest(pca_width: int) -> str:
    """Return a deterministic SHA-256 digest for a PCA-width-specific schema."""

    payload = json.dumps(
        feature_schema(pca_width),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_feature_names(names: Sequence[str]) -> None:
    """Reject response, target, label, teacher, and autoregressive fields."""

    for raw_name in names:
        name = str(raw_name).strip().lower()
        if not name:
            raise Veatic21FeatureAuditError("Feature names must be non-empty")
        if any(token in name for token in FORBIDDEN_RESPONSE_TOKENS):
            raise Veatic21FeatureAuditError(
                f"Forbidden response-side feature token in {raw_name!r}"
            )


def _row_indices(values: Sequence[int] | np.ndarray, *, field: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
        raise Veatic21FeatureAuditError(f"{field} must be a one-dimensional integer array")
    idx = raw.astype(np.int64, copy=False)
    if len(idx) == 0:
        raise Veatic21FeatureAuditError("Feature blocks must contain at least one row")
    if np.any(np.diff(idx) <= 0):
        raise Veatic21FeatureAuditError(
            f"{field} must be unique and strictly increasing for row alignment"
        )
    return idx


def _aligned_metadata(
    row_idx: Sequence[int] | np.ndarray,
    video_id: Sequence[str] | np.ndarray,
    time_seconds: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = _row_indices(row_idx, field="row_idx")
    raw_videos = np.asarray(video_id, dtype=object)
    if raw_videos.ndim != 1 or len(raw_videos) != len(idx):
        raise Veatic21FeatureAuditError("video_id is misaligned with row_idx")
    if any(value is None or not str(value).strip() for value in raw_videos):
        raise Veatic21FeatureAuditError("video_id contains a missing or empty value")
    videos = np.asarray([str(value) for value in raw_videos], dtype=str)

    raw_times = np.asarray(time_seconds)
    if raw_times.ndim != 1 or len(raw_times) != len(idx):
        raise Veatic21FeatureAuditError("time_seconds is misaligned with row_idx")
    try:
        times = raw_times.astype(np.float32, copy=False)
    except (TypeError, ValueError) as exc:
        raise Veatic21FeatureAuditError("time_seconds must be numeric") from exc
    if not np.isfinite(times).all():
        raise Veatic21FeatureAuditError("time_seconds contains non-finite data")
    if np.any(times < 0.0):
        raise Veatic21FeatureAuditError("time_seconds must be non-negative")

    closed_videos: set[str] = set()
    for pos in range(1, len(idx)):
        previous = str(videos[pos - 1])
        current = str(videos[pos])
        if current == previous:
            if not times[pos] > times[pos - 1]:
                raise Veatic21FeatureAuditError(
                    f"time_seconds is not strictly increasing within video {current!r}"
                )
            continue
        closed_videos.add(previous)
        if current in closed_videos:
            raise Veatic21FeatureAuditError(
                f"video {current!r} appears in multiple blocks; causal alignment is ambiguous"
            )
    return idx, videos, times


def _assert_source_alignment(
    source_row_idx: Sequence[int] | np.ndarray | None,
    expected: np.ndarray,
    *,
    field: str,
) -> None:
    if source_row_idx is None:
        return
    actual = _row_indices(source_row_idx, field=field)
    if not np.array_equal(actual, expected):
        raise Veatic21FeatureAuditError(f"{field} is misaligned with row_idx")


def _time_features(video_id: np.ndarray, time_seconds: np.ndarray) -> np.ndarray:
    durations: dict[str, float] = {}
    for video, time_value in zip(video_id, time_seconds, strict=True):
        key = str(video)
        durations[key] = max(durations.get(key, 0.0), float(time_value))
    denom = np.asarray(
        [max(durations[str(video)], 0.5) for video in video_id], dtype=np.float32
    )
    fraction = np.clip(time_seconds / denom, 0.0, 1.0)
    return np.column_stack(
        [np.log1p(np.maximum(time_seconds, 0.0)), fraction]
    ).astype(np.float32)


def _expected_history_mask(video_id: np.ndarray) -> np.ndarray:
    rows = len(video_id)
    mask = np.zeros((rows, WINDOW_ROWS), dtype=np.float32)
    for pos in range(rows):
        for offset, lag in enumerate(range(WINDOW_ROWS - 1, -1, -1)):
            previous = pos - lag
            if previous >= 0 and str(video_id[previous]) == str(video_id[pos]):
                mask[pos, offset] = 1.0
    return mask


def _expected_sequence(current_pca: np.ndarray, video_id: np.ndarray) -> np.ndarray:
    rows, width = current_pca.shape
    sequence = np.zeros((rows, WINDOW_ROWS, width), dtype=np.float32)
    for pos in range(rows):
        for offset, lag in enumerate(range(WINDOW_ROWS - 1, -1, -1)):
            previous = pos - lag
            if previous >= 0 and str(video_id[previous]) == str(video_id[pos]):
                sequence[pos, offset] = current_pca[previous]
    return sequence


def audit_veatic21_features(features: Veatic21Features) -> Veatic21FeatureAudit:
    """Fail closed unless a feature block exactly matches the video-only schema."""

    width = _checked_pca_width(features.pca_width)
    idx, videos, times = _aligned_metadata(
        features.row_idx, features.video_id, features.time_seconds
    )
    rows = len(idx)
    temporal_names = feature_names(width, view="temporal")
    current_names = feature_names(width, view="current")
    supplied_temporal_names = tuple(str(name) for name in features.temporal_feature_names)
    supplied_current_names = tuple(str(name) for name in features.current_feature_names)
    validate_feature_names(supplied_temporal_names + supplied_current_names)
    if len(set(supplied_temporal_names)) != len(supplied_temporal_names) or len(
        set(supplied_current_names)
    ) != len(supplied_current_names):
        raise Veatic21FeatureAuditError("Feature names are not unique")
    if tuple(features.temporal_feature_names) != temporal_names:
        raise Veatic21FeatureAuditError("Temporal feature names do not match the sealed schema")
    if tuple(features.current_feature_names) != current_names:
        raise Veatic21FeatureAuditError("Current-row feature names do not match the sealed schema")
    expected_digest = feature_schema_digest(width)
    if features.schema_digest != expected_digest:
        raise Veatic21FeatureAuditError("Feature schema digest mismatch")

    expected_temporal_shape = (rows, WINDOW_ROWS * width + DIAGNOSTIC_WIDTH + WINDOW_ROWS + 2)
    expected_current_shape = (rows, width + DIAGNOSTIC_WIDTH + 1 + 2)
    if tuple(features.x_temporal.shape) != expected_temporal_shape:
        raise Veatic21FeatureAuditError(
            f"Temporal feature matrix is misaligned: expected {expected_temporal_shape}, "
            f"got {features.x_temporal.shape}"
        )
    if tuple(features.x_current.shape) != expected_current_shape:
        raise Veatic21FeatureAuditError(
            f"Current-row feature matrix is misaligned: expected {expected_current_shape}, "
            f"got {features.x_current.shape}"
        )
    if tuple(features.history_mask.shape) != (rows, WINDOW_ROWS):
        raise Veatic21FeatureAuditError("History mask is misaligned with feature rows")
    if features.x_temporal.dtype != np.float32 or features.x_current.dtype != np.float32:
        raise Veatic21FeatureAuditError("Feature matrices must use float32")
    if features.history_mask.dtype != np.float32:
        raise Veatic21FeatureAuditError("History mask must use float32")
    if not np.isfinite(features.x_temporal).all():
        raise Veatic21FeatureAuditError("Temporal feature matrix contains non-finite data")
    if not np.isfinite(features.x_current).all():
        raise Veatic21FeatureAuditError("Current-row feature matrix contains non-finite data")
    if not np.isfinite(features.history_mask).all():
        raise Veatic21FeatureAuditError("History mask contains non-finite data")

    expected_mask = _expected_history_mask(videos)
    if not np.array_equal(features.history_mask, expected_mask):
        raise Veatic21FeatureAuditError(
            "History mask violates same-video causal alignment or carries cross-video history"
        )
    sequence = features.x_temporal[:, : WINDOW_ROWS * width].reshape(
        rows, WINDOW_ROWS, width
    )
    current_pca = features.x_current[:, :width]
    expected_sequence = _expected_sequence(current_pca, videos)
    if not np.array_equal(sequence, expected_sequence):
        raise Veatic21FeatureAuditError(
            "Temporal PCA sequence violates same-video causal alignment or carries cross-video history"
        )

    temporal_diagnostics = features.x_temporal[
        :, WINDOW_ROWS * width : WINDOW_ROWS * width + DIAGNOSTIC_WIDTH
    ]
    current_diagnostics = features.x_current[:, width : width + DIAGNOSTIC_WIDTH]
    if not np.array_equal(temporal_diagnostics, current_diagnostics):
        raise Veatic21FeatureAuditError("Diagnostic rows are misaligned between feature views")
    temporal_mask = features.x_temporal[
        :,
        WINDOW_ROWS * width
        + DIAGNOSTIC_WIDTH : WINDOW_ROWS * width
        + DIAGNOSTIC_WIDTH
        + WINDOW_ROWS,
    ]
    current_mask = features.x_current[:, width + DIAGNOSTIC_WIDTH : width + DIAGNOSTIC_WIDTH + 1]
    if not np.array_equal(temporal_mask, expected_mask):
        raise Veatic21FeatureAuditError("Temporal availability columns are misaligned")
    if not np.array_equal(current_mask[:, 0], expected_mask[:, -1]):
        raise Veatic21FeatureAuditError("Current-row availability column is misaligned")

    expected_time = _time_features(videos, times)
    if not np.array_equal(features.x_temporal[:, -2:], expected_time):
        raise Veatic21FeatureAuditError("Temporal time features are misaligned")
    if not np.array_equal(features.x_current[:, -2:], expected_time):
        raise Veatic21FeatureAuditError("Current-row time features are misaligned")

    starts = np.concatenate(([True], videos[1:] != videos[:-1]))
    cold_start_mask = np.asarray([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    if not np.all(features.history_mask[starts] == cold_start_mask):
        raise Veatic21FeatureAuditError("Video row 0 cold-start mask was not preserved")
    if not np.all(sequence[starts, :-1] == 0.0):
        raise Veatic21FeatureAuditError("Video row 0 contains non-zero pre-video history")

    return Veatic21FeatureAudit(
        passed=True,
        checks=(
            "response_free_schema",
            "schema_digest",
            "row_alignment",
            "finite_values",
            "same_video_causal_history",
            "diagnostic_alignment",
            "time_alignment",
            "row0_cold_start",
        ),
        schema_digest=expected_digest,
        pca_width=width,
        rows=rows,
        videos=len(set(videos.tolist())),
    )


def build_veatic21_features(
    *,
    row_idx: Sequence[int] | np.ndarray,
    video_id: Sequence[str] | np.ndarray,
    time_seconds: Sequence[float] | np.ndarray,
    pca_scores: np.ndarray,
    diagnostics: np.ndarray,
    pca_width: int | None = None,
    pca_row_idx: Sequence[int] | np.ndarray | None = None,
    diagnostic_row_idx: Sequence[int] | np.ndarray | None = None,
) -> Veatic21Features:
    """Build and audit aligned temporal/current video-only feature views.

    ``pca_row_idx`` and ``diagnostic_row_idx`` should be supplied when their
    source artifacts expose row identifiers.  Any disagreement with
    ``row_idx`` fails closed instead of trusting positional coincidence.
    """

    idx, videos, times = _aligned_metadata(row_idx, video_id, time_seconds)
    _assert_source_alignment(pca_row_idx, idx, field="pca_row_idx")
    _assert_source_alignment(diagnostic_row_idx, idx, field="diagnostic_row_idx")

    raw_pca = np.asarray(pca_scores)
    if raw_pca.ndim != 2 or raw_pca.shape[0] != len(idx):
        raise Veatic21FeatureAuditError("pca_scores is misaligned with row_idx")
    inferred_width = _checked_pca_width(raw_pca.shape[1])
    if pca_width is not None and _checked_pca_width(pca_width) != inferred_width:
        raise Veatic21FeatureAuditError(
            f"Declared PCA width {pca_width} does not match score width {inferred_width}"
        )
    if not np.isfinite(raw_pca).all():
        raise Veatic21FeatureAuditError("pca_scores contains non-finite data")
    pca = raw_pca.astype(np.float32, copy=False)
    if not np.isfinite(pca).all():
        raise Veatic21FeatureAuditError("pca_scores overflowed float32")

    raw_diagnostics = np.asarray(diagnostics)
    if tuple(raw_diagnostics.shape) != (len(idx), DIAGNOSTIC_WIDTH):
        raise Veatic21FeatureAuditError(
            f"diagnostics is misaligned: expected {(len(idx), DIAGNOSTIC_WIDTH)}, "
            f"got {raw_diagnostics.shape}"
        )
    if not np.isfinite(raw_diagnostics).all():
        raise Veatic21FeatureAuditError("diagnostics contains non-finite data")
    diagnostic_values = raw_diagnostics.astype(np.float32, copy=False)
    if not np.isfinite(diagnostic_values).all():
        raise Veatic21FeatureAuditError("diagnostics overflowed float32")

    sequence = _expected_sequence(pca, videos)
    history_mask = _expected_history_mask(videos)
    time_features = _time_features(videos, times)
    x_temporal = np.concatenate(
        [
            sequence.reshape(len(idx), WINDOW_ROWS * inferred_width),
            diagnostic_values,
            history_mask,
            time_features,
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    x_current = np.concatenate(
        [pca, diagnostic_values, history_mask[:, -1:], time_features], axis=1
    ).astype(np.float32, copy=False)

    block = Veatic21Features(
        row_idx=idx.copy(),
        video_id=videos.copy(),
        time_seconds=times.copy(),
        x_temporal=x_temporal,
        x_current=x_current,
        history_mask=history_mask,
        temporal_feature_names=feature_names(inferred_width, view="temporal"),
        current_feature_names=feature_names(inferred_width, view="current"),
        schema_digest=feature_schema_digest(inferred_width),
        pca_width=inferred_width,
    )
    audit_veatic21_features(block)
    return block


__all__ = [
    "ALLOWED_PCA_WIDTHS",
    "COLD_START_POLICY",
    "DIAGNOSTIC_WIDTH",
    "FORBIDDEN_RESPONSE_TOKENS",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "TIME_FEATURE_NAMES",
    "WINDOW_ROWS",
    "Veatic21FeatureAudit",
    "Veatic21FeatureAuditError",
    "Veatic21Features",
    "audit_veatic21_features",
    "build_veatic21_features",
    "feature_names",
    "feature_schema",
    "feature_schema_digest",
    "validate_feature_names",
]
