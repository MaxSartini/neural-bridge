"""Leakage-safe dense 2 Hz future targets for the VEATIC 2.1 substrate.

The new compact VEATIC 2.1 cache is the authority for its aligned native
``arousal`` and ``valence`` annotations.  The historical 1 Hz manifest is only
an identity/source cross-check.  This module adds a new, AGAIN-compatible
temporal contract: a row is scored from the complete future window at offsets
4 through 10 (2.0 through 5.0 seconds at 2 Hz).  The omitted offsets 1 through
3 are the washout gap.

Valence is intentionally not treated as arousal.  It exposes signed rise,
signed drop, and absolute-movement targets, with separately transformed event
scores.  Event thresholds are fit on training rows only and then locked for
every other partition.
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


CONTRACT_VERSION = "veatic_dense_2hz_future_targets_v1"
ROW_FREQUENCY_HZ = 2.0
ROW_PERIOD_SECONDS = 0.5
FUTURE_START_ROW = 4
FUTURE_END_ROW = 10
FUTURE_ROW_OFFSETS = tuple(range(FUTURE_START_ROW, FUTURE_END_ROW + 1))
FUTURE_SECOND_OFFSETS = tuple(offset / ROW_FREQUENCY_HZ for offset in FUTURE_ROW_OFFSETS)

AROUSAL_FUTURE_MAX_DELTA = "future_arousal_max_delta_rows_4_10"
VALENCE_FUTURE_SIGNED_RISE = "future_valence_signed_rise_rows_4_10"
VALENCE_FUTURE_SIGNED_DROP = "future_valence_signed_drop_rows_4_10"
VALENCE_FUTURE_RISE_MAGNITUDE = "future_valence_rise_magnitude_rows_4_10"
VALENCE_FUTURE_DROP_MAGNITUDE = "future_valence_drop_magnitude_rows_4_10"
VALENCE_FUTURE_MAX_ABS_MOVEMENT = "future_valence_max_abs_movement_rows_4_10"

AROUSAL_FUTURE_MAX_DELTA_EVENT = f"{AROUSAL_FUTURE_MAX_DELTA}_train_q90"
VALENCE_FUTURE_RISE_EVENT = "future_valence_rise_magnitude_rows_4_10_train_q90"
VALENCE_FUTURE_DROP_EVENT = "future_valence_drop_magnitude_rows_4_10_train_q90"
VALENCE_FUTURE_MAX_ABS_MOVEMENT_EVENT = f"{VALENCE_FUTURE_MAX_ABS_MOVEMENT}_train_q90"

CONTINUOUS_TARGET_NAMES = (
    AROUSAL_FUTURE_MAX_DELTA,
    VALENCE_FUTURE_SIGNED_RISE,
    VALENCE_FUTURE_SIGNED_DROP,
    VALENCE_FUTURE_RISE_MAGNITUDE,
    VALENCE_FUTURE_DROP_MAGNITUDE,
    VALENCE_FUTURE_MAX_ABS_MOVEMENT,
)
EVENT_TARGET_NAMES = (
    AROUSAL_FUTURE_MAX_DELTA_EVENT,
    VALENCE_FUTURE_RISE_EVENT,
    VALENCE_FUTURE_DROP_EVENT,
    VALENCE_FUTURE_MAX_ABS_MOVEMENT_EVENT,
)
DERIVED_TARGET_NAMES = CONTINUOUS_TARGET_NAMES + EVENT_TARGET_NAMES

TARGET_CONTRACTS: dict[str, dict[str, Any]] = {
    AROUSAL_FUTURE_MAX_DELTA: {
        "type": "continuous",
        "native_source": "arousal",
        "semantic": "max(future_arousal - current_arousal)",
        "signed": True,
    },
    VALENCE_FUTURE_SIGNED_RISE: {
        "type": "continuous",
        "native_source": "valence",
        "semantic": "max(future_valence - current_valence)",
        "signed": True,
        "direction": "rise",
    },
    VALENCE_FUTURE_SIGNED_DROP: {
        "type": "continuous",
        "native_source": "valence",
        "semantic": "min(future_valence - current_valence)",
        "signed": True,
        "direction": "drop",
    },
    VALENCE_FUTURE_RISE_MAGNITUDE: {
        "type": "continuous",
        "native_source": "valence",
        "semantic": "max(max(future_valence - current_valence), 0)",
        "signed": False,
        "direction": "rise_magnitude",
    },
    VALENCE_FUTURE_DROP_MAGNITUDE: {
        "type": "continuous",
        "native_source": "valence",
        "semantic": "max(-min(future_valence - current_valence), 0)",
        "signed": False,
        "direction": "drop_magnitude",
    },
    VALENCE_FUTURE_MAX_ABS_MOVEMENT: {
        "type": "continuous",
        "native_source": "valence",
        "semantic": "max(abs(future_valence - current_valence))",
        "signed": False,
        "direction": "bidirectional_movement",
    },
    AROUSAL_FUTURE_MAX_DELTA_EVENT: {
        "type": "binary_event",
        "source_target": AROUSAL_FUTURE_MAX_DELTA,
        "score_transform": "positive_delta",
        "threshold_policy": "train_quantile",
    },
    VALENCE_FUTURE_RISE_EVENT: {
        "type": "binary_event",
        "source_target": VALENCE_FUTURE_SIGNED_RISE,
        "score_transform": "positive_delta",
        "threshold_policy": "train_quantile",
    },
    VALENCE_FUTURE_DROP_EVENT: {
        "type": "binary_event",
        "source_target": VALENCE_FUTURE_SIGNED_DROP,
        "score_transform": "negative_delta_magnitude",
        "threshold_policy": "train_quantile",
    },
    VALENCE_FUTURE_MAX_ABS_MOVEMENT_EVENT: {
        "type": "binary_event",
        "source_target": VALENCE_FUTURE_MAX_ABS_MOVEMENT,
        "score_transform": "absolute_movement",
        "threshold_policy": "train_quantile",
    },
}


@dataclass(frozen=True)
class Veatic21TargetResult:
    """Augmented manifest rows plus the executable target contract."""

    rows: list[dict[str, Any]]
    contract: dict[str, Any]


@dataclass(frozen=True)
class _EventSpec:
    name: str
    source_target: str
    transform: str


_EVENT_SPECS = (
    _EventSpec(AROUSAL_FUTURE_MAX_DELTA_EVENT, AROUSAL_FUTURE_MAX_DELTA, "positive_delta"),
    _EventSpec(VALENCE_FUTURE_RISE_EVENT, VALENCE_FUTURE_SIGNED_RISE, "positive_delta"),
    _EventSpec(VALENCE_FUTURE_DROP_EVENT, VALENCE_FUTURE_SIGNED_DROP, "negative_delta_magnitude"),
    _EventSpec(
        VALENCE_FUTURE_MAX_ABS_MOVEMENT_EVENT,
        VALENCE_FUTURE_MAX_ABS_MOVEMENT,
        "absolute_movement",
    ),
)


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _event_score(value: float, transform: str) -> float:
    if transform == "positive_delta":
        return max(value, 0.0)
    if transform == "negative_delta_magnitude":
        return max(-value, 0.0)
    if transform == "absolute_movement":
        return abs(value)
    raise ValueError(f"Unknown event score transform: {transform}")


def _resolve_train_mask(
    rows: Sequence[Mapping[str, Any]],
    *,
    train_mask: Sequence[bool] | None,
    split_name: str,
    train_value: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if train_mask is not None:
        if len(train_mask) != len(rows):
            raise ValueError(f"train_mask has {len(train_mask)} rows; expected {len(rows)}")
        resolved = np.asarray(train_mask, dtype=bool)
        return resolved, {"source": "explicit_train_mask"}

    resolved_values: list[bool] = []
    for index, row in enumerate(rows):
        splits = row.get("splits")
        if not isinstance(splits, Mapping) or split_name not in splits:
            raise ValueError(f"row {index} is missing splits.{split_name}")
        resolved_values.append(str(splits[split_name]) == train_value)
    return np.asarray(resolved_values, dtype=bool), {
        "source": f"splits.{split_name}",
        "train_value": train_value,
    }


def build_veatic21_targets(
    rows: Sequence[Mapping[str, Any]],
    *,
    train_mask: Sequence[bool] | None = None,
    split_name: str = "blocked_temporal_gap",
    train_value: str = "train",
    event_quantile: float = 0.90,
    build_events: bool = True,
) -> Veatic21TargetResult:
    """Add dense 2 Hz washout targets without mutating source manifest rows.

    A valid target requires the current annotation and every exact future point
    at row offsets 4, 5, ..., 10 within the same video.  Missing timestamps,
    unavailable annotations, and video ends are masked rather than imputed.

    ``train_mask`` is the preferred interface for fold-specific construction.
    When omitted, the canonical manifest split at
    ``splits.blocked_temporal_gap`` is used.  Event thresholds never inspect
    values outside that training partition.
    """

    if not rows:
        raise ValueError("VEATIC 2.1 target construction requires at least one row")
    if not 0.0 <= event_quantile <= 1.0:
        raise ValueError("event_quantile must be in [0, 1]")

    output_rows: list[dict[str, Any]] = [copy.deepcopy(dict(row)) for row in rows]
    derived_names = DERIVED_TARGET_NAMES if build_events else CONTINUOUS_TARGET_NAMES
    groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
    native_values: dict[str, list[float | None]] = {
        "arousal": [None] * len(output_rows),
        "valence": [None] * len(output_rows),
    }

    for index, row in enumerate(output_rows):
        video_id = row.get("video_id")
        if video_id is None or str(video_id) == "":
            raise ValueError(f"row {index} has no video_id")

        frequency = _finite_float(row.get("sampling_frequency_hz"))
        if frequency is None or not math.isclose(frequency, ROW_FREQUENCY_HZ, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"row {index} sampling_frequency_hz={row.get('sampling_frequency_hz')!r}; "
                f"the VEATIC 2.1 target contract requires exactly {ROW_FREQUENCY_HZ} Hz"
            )

        time_seconds = _finite_float(row.get("time_start_seconds"))
        if time_seconds is None:
            raise ValueError(f"row {index} has no finite time_start_seconds")
        scaled_time = time_seconds * ROW_FREQUENCY_HZ
        tick = int(round(scaled_time))
        if not math.isclose(scaled_time, tick, rel_tol=0.0, abs_tol=1e-7):
            raise ValueError(
                f"row {index} time_start_seconds={time_seconds} is not on the exact 2 Hz grid"
            )
        groups[str(video_id)].append((tick, index))

        targets = row.get("targets")
        if not isinstance(targets, Mapping):
            raise ValueError(f"row {index} has no targets mapping")
        mutable_targets = dict(targets)
        masks = row.get("target_masks", {})
        if not isinstance(masks, Mapping):
            raise ValueError(f"row {index} target_masks must be a mapping when present")
        mutable_masks = dict(masks)
        collisions = set(derived_names).intersection(mutable_targets)
        mask_collisions = {f"target_mask_{name}" for name in derived_names}.intersection(mutable_masks)
        if collisions or mask_collisions:
            names = sorted(collisions | mask_collisions)
            raise ValueError(f"row {index} already contains VEATIC 2.1 derived targets: {names}")

        for native_name in native_values:
            if native_name not in mutable_targets:
                raise ValueError(f"row {index} is missing native targets.{native_name}")
            native_values[native_name][index] = _finite_float(mutable_targets[native_name])
        for target_name in derived_names:
            mutable_targets[target_name] = None
            mutable_masks[f"target_mask_{target_name}"] = False
        row["targets"] = mutable_targets
        row["target_masks"] = mutable_masks

    for video_id, tick_rows in groups.items():
        ordered = sorted(tick_rows)
        ticks = [tick for tick, _index in ordered]
        if len(set(ticks)) != len(ticks):
            raise ValueError(f"video {video_id!r} contains duplicate 2 Hz timestamps")
        index_at_tick = dict(ordered)

        for tick, row_index in ordered:
            future_indices = [index_at_tick.get(tick + offset) for offset in FUTURE_ROW_OFFSETS]
            if any(index is None for index in future_indices):
                continue
            exact_future_indices = [int(index) for index in future_indices if index is not None]
            row_targets = output_rows[row_index]["targets"]
            row_masks = output_rows[row_index]["target_masks"]

            current_arousal = native_values["arousal"][row_index]
            future_arousal = [native_values["arousal"][index] for index in exact_future_indices]
            if current_arousal is not None and all(value is not None for value in future_arousal):
                arousal_deltas = [float(value) - current_arousal for value in future_arousal if value is not None]
                row_targets[AROUSAL_FUTURE_MAX_DELTA] = float(max(arousal_deltas))
                row_masks[f"target_mask_{AROUSAL_FUTURE_MAX_DELTA}"] = True

            current_valence = native_values["valence"][row_index]
            future_valence = [native_values["valence"][index] for index in exact_future_indices]
            if current_valence is not None and all(value is not None for value in future_valence):
                valence_deltas = [float(value) - current_valence for value in future_valence if value is not None]
                row_targets[VALENCE_FUTURE_SIGNED_RISE] = float(max(valence_deltas))
                row_targets[VALENCE_FUTURE_SIGNED_DROP] = float(min(valence_deltas))
                row_targets[VALENCE_FUTURE_RISE_MAGNITUDE] = float(
                    max(max(valence_deltas), 0.0)
                )
                row_targets[VALENCE_FUTURE_DROP_MAGNITUDE] = float(
                    max(-min(valence_deltas), 0.0)
                )
                row_targets[VALENCE_FUTURE_MAX_ABS_MOVEMENT] = float(max(abs(value) for value in valence_deltas))
                row_masks[f"target_mask_{VALENCE_FUTURE_SIGNED_RISE}"] = True
                row_masks[f"target_mask_{VALENCE_FUTURE_SIGNED_DROP}"] = True
                row_masks[f"target_mask_{VALENCE_FUTURE_RISE_MAGNITUDE}"] = True
                row_masks[f"target_mask_{VALENCE_FUTURE_DROP_MAGNITUDE}"] = True
                row_masks[f"target_mask_{VALENCE_FUTURE_MAX_ABS_MOVEMENT}"] = True

    threshold_contract: dict[str, dict[str, Any]] = {}
    if build_events:
        resolved_train_mask, train_partition = _resolve_train_mask(
            output_rows,
            train_mask=train_mask,
            split_name=split_name,
            train_value=train_value,
        )
        for event_spec in _EVENT_SPECS:
            source_mask_name = f"target_mask_{event_spec.source_target}"
            fit_scores: list[float] = []
            for index, row in enumerate(output_rows):
                if not resolved_train_mask[index] or not row["target_masks"][source_mask_name]:
                    continue
                source_value = _finite_float(row["targets"][event_spec.source_target])
                if source_value is not None:
                    fit_scores.append(_event_score(source_value, event_spec.transform))
            if not fit_scores:
                raise ValueError(
                    f"no valid training rows are available to fit event threshold {event_spec.name!r}"
                )
            threshold = float(np.quantile(np.asarray(fit_scores, dtype=np.float64), event_quantile))

            event_mask_name = f"target_mask_{event_spec.name}"
            for row in output_rows:
                if not row["target_masks"][source_mask_name]:
                    continue
                source_value = _finite_float(row["targets"][event_spec.source_target])
                if source_value is None:
                    continue
                score = _event_score(source_value, event_spec.transform)
                row["targets"][event_spec.name] = int(score >= threshold)
                row["target_masks"][event_mask_name] = True

            threshold_contract[event_spec.name] = {
                "source_target": event_spec.source_target,
                "score_transform": event_spec.transform,
                "quantile": float(event_quantile),
                "threshold": threshold,
                "fit_row_count": len(fit_scores),
                "fit_partition": copy.deepcopy(train_partition),
            }

    contract = {
        "schema_version": CONTRACT_VERSION,
        "row_frequency_hz": ROW_FREQUENCY_HZ,
        "row_period_seconds": ROW_PERIOD_SECONDS,
        "future_row_offsets": list(FUTURE_ROW_OFFSETS),
        "future_second_offsets": list(FUTURE_SECOND_OFFSETS),
        "washout_row_offsets": [1, 2, 3],
        "washout_seconds": [0.5, 1.0, 1.5],
        "future_window_seconds_inclusive": [2.0, 5.0],
        "again_timing_equivalence": "rows_4_10_at_2hz",
        "within_video_only": True,
        "full_future_window_required": True,
        "missing_or_end_value": None,
        "mask_container": "target_masks",
        "mask_name_policy": "target_mask_<target_name>",
        "native_targets_preserved": ["arousal", "valence"],
        "target_contracts": {
            name: copy.deepcopy(TARGET_CONTRACTS[name]) for name in derived_names
        },
        "event_targets_built": bool(build_events),
        "event_threshold_fit_scope": (
            "requested_training_partition" if build_events else "deferred_to_outer_training_fold"
        ),
        "event_thresholds": threshold_contract,
    }
    return Veatic21TargetResult(rows=output_rows, contract=contract)


__all__ = [
    "AROUSAL_FUTURE_MAX_DELTA",
    "AROUSAL_FUTURE_MAX_DELTA_EVENT",
    "CONTINUOUS_TARGET_NAMES",
    "CONTRACT_VERSION",
    "DERIVED_TARGET_NAMES",
    "EVENT_TARGET_NAMES",
    "FUTURE_ROW_OFFSETS",
    "FUTURE_SECOND_OFFSETS",
    "TARGET_CONTRACTS",
    "VALENCE_FUTURE_DROP_EVENT",
    "VALENCE_FUTURE_DROP_MAGNITUDE",
    "VALENCE_FUTURE_MAX_ABS_MOVEMENT",
    "VALENCE_FUTURE_MAX_ABS_MOVEMENT_EVENT",
    "VALENCE_FUTURE_RISE_EVENT",
    "VALENCE_FUTURE_RISE_MAGNITUDE",
    "VALENCE_FUTURE_SIGNED_DROP",
    "VALENCE_FUTURE_SIGNED_RISE",
    "Veatic21TargetResult",
    "build_veatic21_targets",
]
