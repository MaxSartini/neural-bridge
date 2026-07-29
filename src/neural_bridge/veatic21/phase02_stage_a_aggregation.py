"""Inner-only Stage A aggregation and prospective Stage B registry construction."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, cast

import numpy as np

from neural_bridge.veatic21.contracts import (
    PHASE01_ROOT,
    PHASE02_BENCHMARK_ROOT,
    PHASE02_REGISTRATION_ROOT,
    PHASE02_REGISTRATION_SHA256,
    PHASE02_STAGE_A_SATURATED_ROOT,
    REPOSITORY_ROOT,
)
from neural_bridge.veatic21.data import load_json, reject_forbidden_runtime_path, sha256_file
from neural_bridge.veatic21.phase00 import _write_json, canonical_json_bytes
from neural_bridge.veatic21.phase02_features import build_causal_history, feature_names
from neural_bridge.veatic21.phase02_metrics import (
    binary_ranking_and_probability_metrics_fast,
)
from neural_bridge.veatic21.phase02_registration import _blocked_row_masks

AGGREGATION_REGISTRATION = REPOSITORY_ROOT / (
    "internal/active/veatic21-phase02-registration/stage-a-aggregation-registration.json"
)
AGGREGATION_ROOT = PHASE02_BENCHMARK_ROOT / ("stage-a-aggregation-stage-b-registration")
AGGREGATION_EXECUTOR_BACKTEST_ROOT = PHASE02_BENCHMARK_ROOT / (
    "stage-a-aggregation-executor-backtest-v2-end-to-end"
)
SELECTED_AGGREGATION_EXECUTOR = REPOSITORY_ROOT / (
    "internal/active/veatic21-phase02-registration/selected-aggregation-executor.json"
)
RESCUE_MAIN_ROOT = PHASE02_STAGE_A_SATURATED_ROOT.parent / (
    "stage-a-convergence-rescue/main-hardware-saturated"
)

STAGE_A_VERIFICATION_SHA256 = "32467b1cbe223a7297cb90b4546e71ac56478c834a720ec5af90775cfc01afb4"
STAGE_A_LEDGER_SHA256 = "95bf4d4c18b38372ca81af0ee8210a9b18da942db12f6162c8c678a0a1b9d342"
RESCUE_VERIFICATION_SHA256 = "5a86e7e9ed2dd8f2be7a0d754482ba79fc74c695cd9d0c461440978d98fcec9b"
RESCUE_LEDGER_SHA256 = "c4eb95b038a0db6d17abf8dc0cf36152592b69fd3104030cbe65855ed3beda47"
EXPECTED_STAGE_A_UNITS = 40_824
EXPECTED_STAGE_A_CELLS = 8_573_040
EXPECTED_RESCUE_CELLS = 113_392
EXPECTED_RESCUE_ELIGIBLE = 82_566
EXPECTED_RESCUE_INVALID = 30_826
EXPECTED_FEATURE_SETS = 126
FINALISTS_PER_TARGET_SCOPE = 12
FEATURE_FORMS = (
    "current_only",
    "raw_levels_with_availability_mask",
    "level_and_first_difference",
    "causal_rolling_summary",
    "combined_levels_differences_summaries",
    "raw_sequence_with_availability_mask",
)
HISTORY_DEPTHS = tuple(range(1, 22))
MODEL_FAMILIES = ("continuous_ridge", "event_logistic_l2")
REGULARIZATION_MULTIPLIERS = (
    1e-6,
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1.0,
    10.0,
    100.0,
    1000.0,
)
TARGETS = tuple(f"s01_e{index:02d}" for index in range(1, 22))
UNIT_PATTERN = re.compile(
    r"^(?P<sequence>\d{5})_(?P<protocol>blocked|grouped)_r"
    r"(?P<repeat>na|\d{2})_o(?P<outer>\d{2})_i(?P<inner>\d{2})_"
    r"(?P<form>.+)_d(?P<depth>\d{2})_"
    r"(?P<family>continuous_ridge|event_logistic_l2)$"
)


@dataclass(frozen=True)
class UnitSource:
    unit_id: str
    path: str
    sha256: str
    rescue_path: str | None
    rescue_sha256: str | None


@dataclass(frozen=True)
class ScopeTask:
    scope_index: int
    protocol: str
    repeat: int | None
    outer_fold: int
    expected_inner_folds: int
    units: tuple[UnitSource, ...]
    shard_root: str


@dataclass(frozen=True)
class BenchmarkChunk:
    index: int
    sources: tuple[UnitSource, ...]
    output_path: str


_BASELINE_WORKER_ARRAYS: dict[str, Any] | None = None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _strict_json_bytes(payload: bytes, path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value} in {path}")

    value = json.loads(payload, parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, Any], value)


def _scope_id(protocol: str, repeat: int | None, outer_fold: int) -> str:
    repeat_code = "na" if repeat is None else f"{repeat:02d}"
    return f"{protocol}_r{repeat_code}_o{outer_fold:02d}"


def _parse_unit_id(unit_id: str) -> dict[str, Any]:
    matched = UNIT_PATTERN.fullmatch(unit_id)
    if matched is None:
        raise ValueError(f"invalid Stage A unit identity: {unit_id}")
    values = matched.groupdict()
    return {
        "sequence": int(values["sequence"]),
        "protocol": values["protocol"],
        "repeat": None if values["repeat"] == "na" else int(values["repeat"]),
        "outer_fold": int(values["outer"]),
        "inner_fold": int(values["inner"]),
        "feature_form": values["form"],
        "history_depth": int(values["depth"]),
        "model_family": values["family"],
    }


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL line {line_number}: {path}")
            yield _strict_json_bytes(line.encode(), path)


def _read_gzip_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank gzip JSONL line {line_number}: {path}")
            yield _strict_json_bytes(line.encode(), path)


def _atomic_gzip_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as raw:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0
            ) as compressed:
                for row in rows:
                    compressed.write(canonical_json_bytes(row) + b"\n")
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_concat(path: Path, sources: Iterable[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            for source in sources:
                with source.open("rb") as handle:
                    shutil.copyfileobj(handle, output, length=4 * 1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _sha256_lines(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _mean_and_se(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean_value = float(np.mean(array))
    standard_error = (
        0.0 if len(array) == 1 else float(np.std(array, ddof=1) / math.sqrt(len(array)))
    )
    return mean_value, standard_error


def _history_region(depth: int) -> str:
    if 1 <= depth <= 7:
        return "low"
    if 8 <= depth <= 14:
        return "mid"
    if 15 <= depth <= 21:
        return "high"
    raise ValueError(f"unregistered history depth: {depth}")


def _capacity_key(row: dict[str, Any]) -> tuple[Any, ...]:
    brier = row.get("mean_brier")
    family_rank = MODEL_FAMILIES.index(cast(str, row["model_family"]))
    return (
        brier is None,
        math.inf if brier is None else float(brier),
        int(row["history_depth_rows"]),
        int(row["feature_count"]),
        family_rank,
        -float(row["regularization_multiplier"]),
        str(row["aggregate_configuration_id"]),
    )


def select_one_standard_error(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Select one complete configuration by the frozen inner-only rule."""

    eligible = [row for row in rows if row["disposition"] == "eligible_for_selection"]
    if not eligible:
        raise ValueError("one-standard-error selection has no complete candidate")
    best = min(
        eligible,
        key=lambda row: (
            -float(row["mean_raw_pr_auc"]),
            float(row["standard_error_raw_pr_auc"]),
            str(row["aggregate_configuration_id"]),
        ),
    )
    threshold = float(best["mean_raw_pr_auc"]) - float(best["standard_error_raw_pr_auc"])
    one_se = [row for row in eligible if float(row["mean_raw_pr_auc"]) >= threshold]
    selected = min(one_se, key=_capacity_key)
    return {
        **selected,
        "one_standard_error_best_configuration_id": best["aggregate_configuration_id"],
        "one_standard_error_best_mean_raw_pr_auc": best["mean_raw_pr_auc"],
        "one_standard_error_threshold": threshold,
        "one_standard_error_set_size": len(one_se),
    }


def _global_feature_rank(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) != EXPECTED_FEATURE_SETS:
        raise ValueError(f"expected {EXPECTED_FEATURE_SETS} feature sets, found {len(rows)}")
    eligible = [row for row in rows if row["disposition"] == "eligible_feature_set"]
    if len(eligible) < FINALISTS_PER_TARGET_SCOPE:
        raise ValueError("fewer than 12 complete feature sets remain")
    best = min(
        eligible,
        key=lambda row: (
            -float(cast(dict[str, Any], row["representative"])["mean_raw_pr_auc"]),
            float(cast(dict[str, Any], row["representative"])["standard_error_raw_pr_auc"]),
            str(row["feature_set_id"]),
        ),
    )
    best_rep = cast(dict[str, Any], best["representative"])
    threshold = float(best_rep["mean_raw_pr_auc"]) - float(best_rep["standard_error_raw_pr_auc"])

    def rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
        representative = cast(dict[str, Any], row["representative"])
        within = float(representative["mean_raw_pr_auc"]) >= threshold
        if within:
            return (0, *_capacity_key(representative), str(row["feature_set_id"]))
        return (
            1,
            -float(representative["mean_raw_pr_auc"]),
            float(representative["standard_error_raw_pr_auc"]),
            *_capacity_key(representative),
            str(row["feature_set_id"]),
        )

    ranked = sorted(eligible, key=rank_key)
    for rank, row in enumerate(ranked, 1):
        row["global_rank"] = rank
        row["within_global_one_standard_error"] = (
            float(cast(dict[str, Any], row["representative"])["mean_raw_pr_auc"]) >= threshold
        )
        row["global_one_standard_error_threshold"] = threshold
    return ranked


def select_stratified_finalists(
    feature_sets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Retain exactly 12 feature sets while satisfying all marginal strata."""

    ranked = _global_feature_rank(feature_sets)
    selected: dict[str, dict[str, Any]] = {}
    reasons: dict[str, list[str]] = defaultdict(list)

    for form in FEATURE_FORMS:
        matches = [row for row in ranked if row["feature_form"] == form]
        if not matches:
            raise ValueError(f"no eligible feature set for required form {form}")
        row = matches[0]
        selected[cast(str, row["feature_set_id"])] = row
        reasons[cast(str, row["feature_set_id"])].append(f"feature_form:{form}")

    for region in ("low", "mid", "high"):
        matches = [row for row in ranked if row["history_region"] == region]
        if not matches:
            raise ValueError(f"no eligible feature set for required region {region}")
        row = matches[0]
        selected[cast(str, row["feature_set_id"])] = row
        reasons[cast(str, row["feature_set_id"])].append(f"history_region:{region}")

    for row in ranked:
        if len(selected) == FINALISTS_PER_TARGET_SCOPE:
            break
        identifier = cast(str, row["feature_set_id"])
        if identifier not in selected:
            selected[identifier] = row
            reasons[identifier].append("global_rank_fill")

    if len(selected) != FINALISTS_PER_TARGET_SCOPE:
        raise ValueError("stratified selection did not yield exactly 12 finalists")
    finalists = sorted(selected.values(), key=lambda row: int(row["global_rank"]))
    for finalist_rank, row in enumerate(finalists, 1):
        row["finalist_rank"] = finalist_rank
        row["retention_reasons"] = reasons[cast(str, row["feature_set_id"])]
    if {row["feature_form"] for row in finalists} != set(FEATURE_FORMS):
        raise ValueError("feature-form coverage failed")
    if {row["history_region"] for row in finalists} != {"low", "mid", "high"}:
        raise ValueError("history-region coverage failed")
    return finalists


def _aggregate_id(
    scope_id: str,
    target: str,
    form: str,
    depth: int,
    family: str,
    regularization_index: int,
) -> str:
    return f"{scope_id}__{target}__{form}__d{depth:02d}__{family}__reg{regularization_index:02d}"


def _feature_set_id(scope_id: str, target: str, form: str, depth: int) -> str:
    return f"{scope_id}__{target}__{form}__d{depth:02d}"


def _validated_cell(
    stage_a_record: dict[str, Any],
    rescue_by_id: dict[str, dict[str, Any]],
    *,
    original_unit_id: str,
    original_unit_sha256: str,
) -> tuple[dict[str, Any], str]:
    configuration_id = cast(str, stage_a_record.get("configuration_id"))
    disposition = stage_a_record.get("disposition")
    if disposition == "eligible_for_inner_aggregation":
        _require(stage_a_record.get("converged") is True, "eligible Stage A cell not converged")
        _require(configuration_id not in rescue_by_id, "converged Stage A cell was rescued")
        return stage_a_record, "immutable_stage_a"
    _require(
        disposition == "protected_from_pruning_requires_16x_budget",
        f"unexpected Stage A disposition: {disposition}",
    )
    rescue = rescue_by_id.get(configuration_id)
    _require(rescue is not None, f"missing linked rescue cell: {configuration_id}")
    _require(rescue.get("original_unit_id") == original_unit_id, "rescue unit link changed")
    _require(
        rescue.get("original_unit_result_sha256") == original_unit_sha256,
        "rescue original unit hash changed",
    )
    _require(
        rescue.get("original_configuration_id") == configuration_id,
        "rescue configuration link changed",
    )
    _require(
        rescue.get("disposition")
        in {
            "eligible_for_inner_aggregation",
            "invalid_nonconverged_after_registered_maximum_budget",
        },
        "unexpected rescue disposition",
    )
    return rescue, "linked_rescue"


def _boundary_disposition(family_rows: list[dict[str, Any]], family: str) -> dict[str, Any]:
    eligible = [row for row in family_rows if row["disposition"] == "eligible_for_selection"]
    if not eligible:
        return {
            "model_family": family,
            "disposition": "no_complete_family_configuration",
            "expansion_multiplier": None,
            "family_winner": None,
        }
    winner = select_one_standard_error(eligible)
    index = int(winner["regularization_index"])
    expansion = 1e-7 if index == 0 else 1e4 if index == 9 else None
    return {
        "model_family": family,
        "disposition": (
            "registered_low_edge_expansion"
            if index == 0
            else "registered_high_edge_expansion"
            if index == 9
            else "interior_winner_no_expansion"
        ),
        "expansion_multiplier": expansion,
        "family_winner": {
            key: winner[key]
            for key in (
                "aggregate_configuration_id",
                "regularization_index",
                "regularization_multiplier",
                "mean_raw_pr_auc",
                "standard_error_raw_pr_auc",
                "mean_brier",
                "one_standard_error_threshold",
                "one_standard_error_set_size",
            )
        },
    }


def _aggregate_scope(task: ScopeTask) -> dict[str, Any]:
    scope_id = _scope_id(task.protocol, task.repeat, task.outer_fold)
    expected_unit_count = (
        len(FEATURE_FORMS) * len(HISTORY_DEPTHS) * len(MODEL_FAMILIES) * task.expected_inner_folds
    )
    _require(len(task.units) == expected_unit_count, f"unit count changed for {scope_id}")
    cells: dict[tuple[str, str, int, str, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    exclusion_rows: list[dict[str, Any]] = []
    admitted_ids: list[str] = []
    all_original_ids: set[str] = set()
    rescued_ids: set[str] = set()
    stage_a_bytes = 0
    rescue_bytes = 0

    for source in sorted(task.units, key=lambda item: _parse_unit_id(item.unit_id)["sequence"]):
        unit_meta = _parse_unit_id(source.unit_id)
        payload = Path(source.path).read_bytes()
        stage_a_bytes += len(payload)
        _require(hashlib.sha256(payload).hexdigest() == source.sha256, "Stage A unit hash changed")
        result = _strict_json_bytes(payload, Path(source.path))
        _require(result.get("outer_test_scores_opened") is False, "outer score access detected")
        _require(result.get("cortical_values_opened") is False, "cortical access detected")
        _require(
            result.get("unit")
            == {
                "unit_id": source.unit_id,
                "sequence": unit_meta["sequence"],
                "protocol": unit_meta["protocol"],
                "split_index": cast(dict[str, Any], result["unit"])["split_index"],
                "repeat": unit_meta["repeat"],
                "outer_fold": unit_meta["outer_fold"],
                "inner_fold": unit_meta["inner_fold"],
                "feature_form": unit_meta["feature_form"],
                "history_depth": unit_meta["history_depth"],
                "model_family": unit_meta["model_family"],
            },
            "Stage A unit metadata mismatch",
        )
        rescue_by_id: dict[str, dict[str, Any]] = {}
        if source.rescue_path is not None:
            _require(source.rescue_sha256 is not None, "rescue hash missing")
            rescue_payload = Path(source.rescue_path).read_bytes()
            rescue_bytes += len(rescue_payload)
            _require(
                hashlib.sha256(rescue_payload).hexdigest() == source.rescue_sha256,
                "rescue unit hash changed",
            )
            rescue_result = _strict_json_bytes(rescue_payload, Path(source.rescue_path))
            _require(
                rescue_result.get("outer_test_scores_opened") is False,
                "rescue outer score access detected",
            )
            _require(
                rescue_result.get("cortical_values_opened") is False,
                "rescue cortical access detected",
            )
            _require(
                rescue_result.get("aggregation_or_pruning_performed") is False,
                "rescue artifact was aggregated before authorization",
            )
            for record in cast(list[dict[str, Any]], rescue_result["records"]):
                identifier = cast(str, record["original_configuration_id"])
                _require(identifier not in rescue_by_id, "duplicate rescue configuration")
                rescue_by_id[identifier] = record
                rescued_ids.add(identifier)

        records = cast(list[dict[str, Any]], result.get("records"))
        _require(len(records) == 210, "Stage A unit configuration count changed")
        used_rescue_ids: set[str] = set()
        for original in records:
            original_id = cast(str, original["configuration_id"])
            _require(original_id not in all_original_ids, "duplicate Stage A configuration")
            all_original_ids.add(original_id)
            selected, source_kind = _validated_cell(
                original,
                rescue_by_id,
                original_unit_id=source.unit_id,
                original_unit_sha256=source.sha256,
            )
            if source_kind == "linked_rescue":
                used_rescue_ids.add(original_id)
            suffix = original_id.rsplit("__reg", 1)[-1]
            _require(len(suffix) == 2 and suffix.isdigit(), "invalid regularization identity")
            regularization_index = int(suffix)
            target = cast(str, original["candidate_id"])
            key = (
                target,
                cast(str, original["feature_form"]),
                int(original["history_depth_rows"]),
                cast(str, original["model_family"]),
                regularization_index,
            )
            inner_fold = int(unit_meta["inner_fold"])
            _require(inner_fold not in cells[key], "duplicate inner cell")
            valid = selected.get("disposition") == "eligible_for_inner_aggregation"
            if valid:
                _require(selected.get("converged") is True, "admitted source did not converge")
                admitted_ids.append(original_id)
            else:
                _require(
                    selected.get("disposition")
                    == "invalid_nonconverged_after_registered_maximum_budget",
                    "nonconverged cell was not given the frozen invalid disposition",
                )
                exclusion_rows.append(
                    {
                        "scope_id": scope_id,
                        "protocol": task.protocol,
                        "repeat": task.repeat,
                        "outer_fold": task.outer_fold,
                        "inner_fold": inner_fold,
                        "original_configuration_id": original_id,
                        "original_unit_id": source.unit_id,
                        "original_unit_sha256": source.sha256,
                        "rescue_cell_identity_sha256": selected["rescue_cell_identity_sha256"],
                        "rescue_disposition": selected["disposition"],
                        "scientific_interpretation": "incomplete_invalid_not_negative",
                    }
                )
            cells[key][inner_fold] = {
                "inner_fold": inner_fold,
                "original_configuration_id": original_id,
                "source_kind": source_kind,
                "source_disposition": selected["disposition"],
                "raw_pr_auc": selected["raw_pr_auc"] if valid else None,
                "brier": selected["brier"] if valid else None,
                "roc_auc": selected["roc_auc"] if valid else None,
                "prevalence": selected["prevalence"],
                "train_rows": selected["train_rows"],
                "validation_rows": selected["validation_rows"],
                "regularization_scale": selected["regularization_scale"],
                "regularization_value": selected["regularization_value"],
                "split_sha256": result["split_sha256"],
                "stage_a_unit_sha256": source.sha256,
                "rescue_cell_identity_sha256": selected.get("rescue_cell_identity_sha256"),
            }
        _require(used_rescue_ids == set(rescue_by_id), "rescue unit coverage mismatch")

    expected_keys = len(TARGETS) * EXPECTED_FEATURE_SETS * len(MODEL_FAMILIES) * 10
    _require(len(cells) == expected_keys, f"aggregate key coverage changed for {scope_id}")
    aggregate_rows: list[dict[str, Any]] = []
    feature_groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    ordered_admitted: list[str] = []
    ordered_excluded: list[str] = []
    for target in TARGETS:
        for form in FEATURE_FORMS:
            for depth in HISTORY_DEPTHS:
                feature_count = len(feature_names(form, depth))
                for family in MODEL_FAMILIES:
                    for regularization_index, multiplier in enumerate(REGULARIZATION_MULTIPLIERS):
                        key = (target, form, depth, family, regularization_index)
                        inner = cells[key]
                        _require(
                            set(inner) == set(range(task.expected_inner_folds)),
                            "inner-fold coverage changed",
                        )
                        values = [inner[index] for index in range(task.expected_inner_folds)]
                        invalid_count = sum(
                            value["source_disposition"]
                            == "invalid_nonconverged_after_registered_maximum_budget"
                            for value in values
                        )
                        for value in values:
                            if value["source_disposition"] == "eligible_for_inner_aggregation":
                                ordered_admitted.append(value["original_configuration_id"])
                            else:
                                ordered_excluded.append(value["original_configuration_id"])
                        raw_values = [
                            float(value["raw_pr_auc"])
                            for value in values
                            if value["raw_pr_auc"] is not None
                        ]
                        brier_values = [
                            float(value["brier"]) for value in values if value["brier"] is not None
                        ]
                        complete = invalid_count == 0
                        mean_raw, standard_error = (
                            _mean_and_se(raw_values) if complete else (None, None)
                        )
                        mean_roc = (
                            float(np.mean([float(value["roc_auc"]) for value in values]))
                            if complete
                            else None
                        )
                        mean_prevalence = float(
                            np.mean([float(value["prevalence"]) for value in values])
                        )
                        row: dict[str, Any] = {
                            "schema_version": "veatic21_phase02_stage_a_aggregate_v1",
                            "aggregate_configuration_id": _aggregate_id(
                                scope_id,
                                target,
                                form,
                                depth,
                                family,
                                regularization_index,
                            ),
                            "scope_id": scope_id,
                            "protocol": task.protocol,
                            "repeat": task.repeat,
                            "outer_fold": task.outer_fold,
                            "candidate_id": target,
                            "feature_form": form,
                            "history_depth_rows": depth,
                            "history_region": _history_region(depth),
                            "feature_count": feature_count,
                            "model_family": family,
                            "regularization_index": regularization_index,
                            "regularization_multiplier": multiplier,
                            "expected_inner_folds": task.expected_inner_folds,
                            "eligible_inner_folds": task.expected_inner_folds - invalid_count,
                            "invalid_inner_folds": invalid_count,
                            "disposition": (
                                "eligible_for_selection"
                                if complete
                                else "excluded_incomplete_invalid_not_negative"
                            ),
                            "mean_raw_pr_auc": mean_raw,
                            "standard_error_raw_pr_auc": standard_error,
                            "mean_brier": (
                                float(np.mean(brier_values))
                                if complete and len(brier_values) == len(values)
                                else None
                            ),
                            "mean_roc_auc": mean_roc,
                            "mean_prevalence": mean_prevalence,
                            "mean_ar_uplift_over_chance": (
                                None if mean_raw is None else mean_raw - mean_prevalence
                            ),
                            "positive_uplift_inner_folds": (
                                None
                                if not complete
                                else sum(
                                    float(value["raw_pr_auc"]) > float(value["prevalence"])
                                    for value in values
                                )
                            ),
                            "inner": values,
                            "outer_test_scores_opened": False,
                            "cortical_values_opened": False,
                        }
                        aggregate_rows.append(row)
                        feature_groups[(target, form, depth)].append(row)

    _require(set(admitted_ids) == set(ordered_admitted), "admission accounting mismatch")
    _require(
        {row["original_configuration_id"] for row in exclusion_rows} == set(ordered_excluded),
        "exclusion accounting mismatch",
    )
    feature_rows: list[dict[str, Any]] = []
    finalist_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    for target in TARGETS:
        target_features: list[dict[str, Any]] = []
        for form in FEATURE_FORMS:
            for depth in HISTORY_DEPTHS:
                configurations = feature_groups[(target, form, depth)]
                complete = [
                    row for row in configurations if row["disposition"] == "eligible_for_selection"
                ]
                feature_row: dict[str, Any] = {
                    "schema_version": "veatic21_phase02_stage_a_feature_disposition_v1",
                    "feature_set_id": _feature_set_id(scope_id, target, form, depth),
                    "scope_id": scope_id,
                    "protocol": task.protocol,
                    "repeat": task.repeat,
                    "outer_fold": task.outer_fold,
                    "candidate_id": target,
                    "feature_form": form,
                    "history_depth_rows": depth,
                    "history_region": _history_region(depth),
                    "feature_count": len(feature_names(form, depth)),
                    "registered_linear_configurations": 20,
                    "complete_linear_configurations": len(complete),
                    "excluded_linear_configurations": 20 - len(complete),
                    "disposition": (
                        "eligible_feature_set"
                        if complete
                        else "excluded_no_complete_linear_configuration"
                    ),
                    "representative": select_one_standard_error(complete) if complete else None,
                    "family_boundary_dispositions": [
                        _boundary_disposition(
                            [row for row in configurations if row["model_family"] == family],
                            family,
                        )
                        for family in MODEL_FAMILIES
                    ],
                    "outer_test_scores_opened": False,
                    "cortical_values_opened": False,
                }
                feature_rows.append(feature_row)
                target_features.append(feature_row)
        finalists = select_stratified_finalists(target_features)
        for finalist in finalists:
            finalist_row = {
                **finalist,
                "schema_version": "veatic21_phase02_stage_b_finalist_v1",
                "stage_b_disposition": "prospectively_registered_not_executed",
            }
            finalist_rows.append(finalist_row)
            for boundary in cast(list[dict[str, Any]], finalist["family_boundary_dispositions"]):
                boundary_rows.append(
                    {
                        "schema_version": "veatic21_phase02_stage_b_boundary_disposition_v1",
                        "feature_set_id": finalist["feature_set_id"],
                        "scope_id": scope_id,
                        "candidate_id": target,
                        "feature_form": finalist["feature_form"],
                        "history_depth_rows": finalist["history_depth_rows"],
                        **boundary,
                        "executed": False,
                        "outer_test_scores_opened": False,
                        "cortical_values_opened": False,
                    }
                )

    shard_root = Path(task.shard_root)
    prefix = f"{task.scope_index:02d}_{scope_id}"
    aggregate_path = shard_root / "aggregates" / f"{prefix}.jsonl.gz"
    exclusion_path = shard_root / "exclusions" / f"{prefix}.jsonl.gz"
    feature_path = shard_root / "feature-dispositions" / f"{prefix}.jsonl.gz"
    finalist_path = shard_root / "finalists" / f"{prefix}.jsonl.gz"
    boundary_path = shard_root / "boundaries" / f"{prefix}.jsonl.gz"
    _atomic_gzip_jsonl(aggregate_path, aggregate_rows)
    _atomic_gzip_jsonl(
        exclusion_path,
        sorted(exclusion_rows, key=lambda row: str(row["original_configuration_id"])),
    )
    _atomic_gzip_jsonl(feature_path, feature_rows)
    _atomic_gzip_jsonl(finalist_path, finalist_rows)
    _atomic_gzip_jsonl(boundary_path, boundary_rows)
    return {
        "scope_index": task.scope_index,
        "scope_id": scope_id,
        "aggregate_path": str(aggregate_path),
        "aggregate_sha256": sha256_file(aggregate_path),
        "aggregate_rows": len(aggregate_rows),
        "exclusion_path": str(exclusion_path),
        "exclusion_sha256": sha256_file(exclusion_path),
        "excluded_cells": len(exclusion_rows),
        "feature_path": str(feature_path),
        "feature_sha256": sha256_file(feature_path),
        "feature_rows": len(feature_rows),
        "finalist_path": str(finalist_path),
        "finalist_sha256": sha256_file(finalist_path),
        "finalist_rows": len(finalist_rows),
        "boundary_path": str(boundary_path),
        "boundary_sha256": sha256_file(boundary_path),
        "boundary_rows": len(boundary_rows),
        "admitted_cells": len(ordered_admitted),
        "admitted_configuration_ids_sha256": _sha256_lines(ordered_admitted),
        "excluded_configuration_ids_sha256": _sha256_lines(ordered_excluded),
        "sorted_admitted_configuration_ids_sha256": _sha256_lines(sorted(ordered_admitted)),
        "sorted_excluded_configuration_ids_sha256": _sha256_lines(sorted(ordered_excluded)),
        "rescued_cells": len(rescued_ids),
        "stage_a_bytes_hashed": stage_a_bytes,
        "rescue_bytes_hashed": rescue_bytes,
    }


def _next_power_of_two(value: float) -> int:
    if value <= 1:
        return 1
    return 1 << math.ceil(math.log2(value))


def _largest_power_of_two_not_exceeding(value: float) -> int:
    if value < 1:
        return 1
    return 1 << math.floor(math.log2(value))


def _nonlinear_candidates(family: str, input_dim: int, train_rows: int) -> list[dict[str, Any]]:
    base_width = _next_power_of_two(math.sqrt(input_dim))
    width_cap = _next_power_of_two(math.sqrt(train_rows))
    widths = sorted(
        {
            max(1, min(width_cap, base_width // 2)),
            min(width_cap, base_width),
            min(width_cap, base_width * 2),
            min(width_cap, base_width * 4),
        }
    )
    base_batch = _largest_power_of_two_not_exceeding(math.sqrt(train_rows))
    batches = sorted({max(1, base_batch // 2), base_batch, base_batch * 2})
    dropouts = sorted(
        {0.0, min(0.5, 1.0 / math.sqrt(input_dim)), min(0.5, 2.0 / math.sqrt(input_dim))}
    )
    learning_rates = [factor / math.sqrt(train_rows) for factor in (0.25, 1.0, 4.0, 16.0)]
    base: dict[str, Any] = {
        "family": family,
        "width": base_width,
        "layers": 1,
        "activation": "relu" if family == "event_mlp" else "gru_native_tanh_sigmoid",
        "dropout": 0.0,
        "optimizer": "adamw",
        "learning_rate": 1.0 / math.sqrt(train_rows),
        "batch_size": base_batch,
    }
    axes: list[tuple[str, list[Any]]] = [
        ("width", widths),
        ("layers", [1, 2, 3] if family == "event_mlp" else [1, 2]),
        ("dropout", dropouts),
        ("optimizer", ["adamw", "sgd_nesterov"]),
        ("learning_rate", learning_rates),
        ("batch_size", batches),
    ]
    if family == "event_mlp":
        axes.insert(2, ("activation", ["relu", "gelu", "tanh"]))
    unique: dict[bytes, dict[str, Any]] = {}
    unique[canonical_json_bytes(base)] = base
    for name, values in axes:
        for value in values:
            candidate = {**base, name: value}
            unique[canonical_json_bytes(candidate)] = candidate
    return [unique[key] for key in sorted(unique)]


def _stage_b_candidates(finalist: dict[str, Any], inner: dict[str, Any]) -> list[dict[str, Any]]:
    input_dim = int(finalist["feature_count"])
    train_rows = int(inner["train_rows"])
    base_budget = _next_power_of_two(math.sqrt(train_rows))
    candidates: list[dict[str, Any]] = []
    for boundary in cast(list[dict[str, Any]], finalist["family_boundary_dispositions"]):
        multiplier = boundary["expansion_multiplier"]
        if multiplier is not None:
            candidates.append(
                {
                    "family": boundary["model_family"],
                    "search_role": "registered_linear_boundary_expansion",
                    "regularization_multiplier": multiplier,
                    "regularization_value": float(multiplier)
                    * float(inner["regularization_scale"]),
                    "initial_update_budget": (
                        base_budget
                        if boundary["model_family"] == "continuous_ridge"
                        else 4 * base_budget
                    ),
                    "maximum_convergence_budget": 16 * base_budget,
                }
            )
    for multiplier in REGULARIZATION_MULTIPLIERS:
        for l1_ratio in (0.25, 0.5, 0.75, 1.0):
            candidates.append(
                {
                    "family": "event_elastic_net",
                    "search_role": "complete_elastic_net_grid",
                    "regularization_multiplier": multiplier,
                    "regularization_value": multiplier * float(inner["regularization_scale"]),
                    "l1_ratio": l1_ratio,
                    "initial_update_budget": base_budget,
                    "undertraining_recovery_maximum_budget": 2 * base_budget,
                }
            )
    for candidate in _nonlinear_candidates("event_mlp", input_dim, train_rows):
        candidates.append(
            {
                **candidate,
                "search_role": "one_factor_at_a_time",
                "initial_update_budget": base_budget,
                "undertraining_recovery_maximum_budget": 2 * base_budget,
            }
        )
    if finalist["feature_form"] == "raw_sequence_with_availability_mask":
        for candidate in _nonlinear_candidates("event_gru", input_dim, train_rows):
            candidates.append(
                {
                    **candidate,
                    "search_role": "one_factor_at_a_time",
                    "initial_update_budget": base_budget,
                    "undertraining_recovery_maximum_budget": 2 * base_budget,
                }
            )
    unique: dict[bytes, dict[str, Any]] = {}
    for candidate in candidates:
        unique[canonical_json_bytes(candidate)] = candidate
    return [unique[key] for key in sorted(unique)]


def _work_registry_rows(finalist_path: Path) -> Iterator[dict[str, Any]]:
    sequence = 0
    for finalist in _read_gzip_jsonl(finalist_path):
        representative = cast(dict[str, Any], finalist["representative"])
        for inner in cast(list[dict[str, Any]], representative["inner"]):
            candidates = _stage_b_candidates(finalist, inner)
            candidate_ids = [
                hashlib.sha256(
                    canonical_json_bytes(
                        {
                            "feature_set_id": finalist["feature_set_id"],
                            "inner_fold": inner["inner_fold"],
                            "candidate": candidate,
                        }
                    )
                ).hexdigest()
                for candidate in candidates
            ]
            yield {
                "schema_version": "veatic21_phase02_stage_b_work_unit_v1",
                "sequence": sequence,
                "work_unit_id": (
                    f"{sequence:05d}__{finalist['feature_set_id']}__i{int(inner['inner_fold']):02d}"
                ),
                "scope_id": finalist["scope_id"],
                "protocol": finalist["protocol"],
                "repeat": finalist["repeat"],
                "outer_fold": finalist["outer_fold"],
                "inner_fold": inner["inner_fold"],
                "candidate_id": finalist["candidate_id"],
                "feature_set_id": finalist["feature_set_id"],
                "feature_form": finalist["feature_form"],
                "history_depth_rows": finalist["history_depth_rows"],
                "feature_count": finalist["feature_count"],
                "train_rows": inner["train_rows"],
                "validation_rows": inner["validation_rows"],
                "split_sha256": inner["split_sha256"],
                "regularization_scale": inner["regularization_scale"],
                "candidate_count": len(candidates),
                "candidate_ids_sha256": _sha256_lines(candidate_ids),
                "candidates": candidates,
                "gru_disposition": (
                    "registered"
                    if finalist["feature_form"] == "raw_sequence_with_availability_mask"
                    else "not_applicable_nonsequence_feature_form"
                ),
                "execution_status": "not_executed",
                "outer_test_scores_opened": False,
                "cortical_values_opened": False,
            }
            sequence += 1


def _load_development_arrays() -> dict[str, Any]:
    with np.load(PHASE01_ROOT / "aligned-labels.npz", allow_pickle=False) as payload:
        arousal = payload["arousal"].astype(np.float32)
        video_id = payload["video_id"].astype(np.int16)
        row_index = payload["row_index"].astype(np.int32)
    with np.load(PHASE01_ROOT / "target-substrate.npz", allow_pickle=False) as payload:
        starts = payload["candidate_start_rows"].astype(int)
        active = np.flatnonzero(starts == 1)
        values = payload["continuous_future_maximum_increase"][active].astype(np.float32)
        masks = payload["valid_mask"][active].astype(bool)
        ends = payload["candidate_end_rows"][active].astype(int)
    _require(tuple(ends.tolist()) == HISTORY_DEPTHS, "active target ends changed")
    return {
        "arousal": arousal,
        "video_id": video_id,
        "row_index": row_index,
        "values": values,
        "masks": masks,
        "ends": ends,
        "history": build_causal_history(arousal, video_id, row_index, max_depth=21),
        "splits": load_json(PHASE02_REGISTRATION_ROOT / "split-registry.json"),
    }


def _development_masks(
    arrays: dict[str, Any],
    protocol: str,
    repeat: int | None,
    outer_fold: int,
    inner_fold: int,
) -> tuple[np.ndarray, np.ndarray]:
    video_id = cast(np.ndarray, arrays["video_id"])
    active_masks = cast(np.ndarray, arrays["masks"])
    if protocol == "grouped":
        splits = cast(list[dict[str, Any]], cast(dict[str, Any], arrays["splits"])["grouped"])
        matches = [
            split
            for split in splits
            if split["repeat"] == repeat and split["outer_fold"] == outer_fold
        ]
        _require(len(matches) == 1, "grouped development split lookup failed")
        split = matches[0]
        validation_videos = set(split["inner_validation_video_folds"][inner_fold])
        training_videos = set(split["train_videos"]) - validation_videos
        train_owner = np.isin(video_id, list(training_videos))
        validation_owner = np.isin(video_id, list(validation_videos))
        return train_owner[:, None] & active_masks.T, validation_owner[:, None] & active_masks.T
    splits = cast(list[dict[str, Any]], cast(dict[str, Any], arrays["splits"])["blocked"])
    matches = [split for split in splits if split["outer_fold"] == outer_fold]
    _require(len(matches) == 1 and inner_fold == 0, "blocked development split lookup failed")
    split = matches[0]
    row_counts = {int(video): int(np.sum(video_id == video)) for video in np.unique(video_id)}
    train = np.zeros_like(active_masks.T)
    validation = np.zeros_like(active_masks.T)
    for target, target_end in enumerate(cast(np.ndarray, arrays["ends"])):
        masks = _blocked_row_masks(
            video_id,
            cast(np.ndarray, arrays["row_index"]),
            row_counts,
            int(target_end),
            int(split["test_block_index"]),
            int(split["block_count"]),
        )
        train[:, target] = masks["inner_train"] & active_masks[target]
        validation[:, target] = masks["inner_validation"] & active_masks[target]
    return train, validation


def _per_video_summary(
    labels: np.ndarray,
    scores: np.ndarray,
    video_ids: np.ndarray,
) -> dict[str, Any]:
    values: list[float] = []
    undefined = 0
    positive_uplift = 0
    for video in sorted(np.unique(video_ids).tolist()):
        owned = video_ids == video
        metric = binary_ranking_and_probability_metrics_fast(
            labels[owned], scores[owned], probability=False
        )
        pr_auc = metric["raw_pr_auc"]
        prevalence = metric["prevalence"]
        if pr_auc is None:
            undefined += 1
        else:
            values.append(float(pr_auc))
            positive_uplift += float(pr_auc) > float(cast(float, prevalence))
    return {
        "defined_videos": len(values),
        "undefined_single_class_videos": undefined,
        "median_raw_pr_auc_defined": median(values) if values else None,
        "positive_uplift_defined_videos": positive_uplift,
    }


def _baseline_record(
    descriptor: dict[str, Any],
    arrays: dict[str, Any],
    train_masks: np.ndarray,
    validation_masks: np.ndarray,
) -> dict[str, Any]:
    target_index = int(descriptor["target_index"])
    depth = int(descriptor["history_depth_rows"])
    train = train_masks[:, target_index]
    validation = validation_masks[:, target_index]
    values = cast(np.ndarray, arrays["values"])[target_index]
    threshold = float(np.quantile(values[train], 0.90))
    labels = values >= threshold
    owned_labels = labels[validation]
    owned_videos = cast(np.ndarray, arrays["video_id"])[validation]
    history = arrays["history"]
    scores = {
        "training_prevalence_constant": np.full(
            int(validation.sum()), float(np.mean(labels[train])), dtype=np.float64
        ),
        "current_arousal_rank": history.levels[validation, 0],
        "previous_delta_rank": history.deltas[validation, 0],
        "causal_trailing_mean_rank": history.rolling_mean[validation, depth - 1],
        "causal_slope_rank": history.rolling_slope[validation, depth - 1],
    }
    metrics: dict[str, Any] = {}
    for name, score in scores.items():
        aggregate = binary_ranking_and_probability_metrics_fast(
            owned_labels, np.asarray(score), probability=False
        )
        metrics[name] = {
            **aggregate,
            "per_video": _per_video_summary(owned_labels, np.asarray(score), owned_videos),
        }
    available = history.available[validation, 1 : depth + 1]
    return {
        "schema_version": "veatic21_phase02_development_baseline_v1",
        **descriptor,
        "train_threshold_q90": threshold,
        "train_rows": int(train.sum()),
        "validation_rows": int(validation.sum()),
        "full_history_lag_availability_fraction": float(np.mean(available)),
        "baselines": metrics,
        "outer_test_scores_opened": False,
        "cortical_values_opened": False,
    }


def _initialize_baseline_worker() -> None:
    global _BASELINE_WORKER_ARRAYS
    _BASELINE_WORKER_ARRAYS = _load_development_arrays()


def _baseline_scope_task(descriptors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    arrays = _BASELINE_WORKER_ARRAYS
    _require(arrays is not None, "baseline process was not initialized")
    _require(bool(descriptors), "baseline process received an empty scope")
    first = descriptors[0]
    ownership = (
        cast(str, first["protocol"]),
        cast(int | None, first["repeat"]),
        int(first["outer_fold"]),
        int(first["inner_fold"]),
    )
    _require(
        all(
            (
                descriptor["protocol"],
                descriptor["repeat"],
                descriptor["outer_fold"],
                descriptor["inner_fold"],
            )
            == ownership
            for descriptor in descriptors
        ),
        "baseline process scope mixed split ownership",
    )
    train_masks, validation_masks = _development_masks(arrays, *ownership)
    return [
        _baseline_record(descriptor, arrays, train_masks, validation_masks)
        for descriptor in descriptors
    ]


def _baseline_rows(finalist_path: Path, workers: int) -> list[dict[str, Any]]:
    descriptors: dict[tuple[Any, ...], dict[str, Any]] = {}
    for finalist in _read_gzip_jsonl(finalist_path):
        representative = cast(dict[str, Any], finalist["representative"])
        for inner in cast(list[dict[str, Any]], representative["inner"]):
            key = (
                finalist["scope_id"],
                inner["inner_fold"],
                finalist["candidate_id"],
                finalist["history_depth_rows"],
            )
            descriptors[key] = {
                "scope_id": finalist["scope_id"],
                "protocol": finalist["protocol"],
                "repeat": finalist["repeat"],
                "outer_fold": finalist["outer_fold"],
                "inner_fold": inner["inner_fold"],
                "candidate_id": finalist["candidate_id"],
                "target_index": TARGETS.index(cast(str, finalist["candidate_id"])),
                "history_depth_rows": finalist["history_depth_rows"],
            }
    ordered = [descriptors[key] for key in sorted(descriptors)]
    grouped: dict[tuple[str, int | None, int, int], list[dict[str, Any]]] = defaultdict(list)
    for descriptor in ordered:
        ownership = (
            cast(str, descriptor["protocol"]),
            cast(int | None, descriptor["repeat"]),
            int(descriptor["outer_fold"]),
            int(descriptor["inner_fold"]),
        )
        grouped[ownership].append(descriptor)
    scope_descriptors = [grouped[key] for key in sorted(grouped)]
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_initialize_baseline_worker
    ) as executor:
        chunks = list(executor.map(_baseline_scope_task, scope_descriptors))
    return [row for chunk in chunks for row in chunk]


def _history_offsets(form: str, depth: int) -> list[int]:
    if form == "current_only":
        return [0]
    if form == "causal_rolling_summary":
        oldest = max(depth - 1, 1)
        return list(range(-oldest, 1))
    return list(range(-depth, 1))


def _dominance_rows(
    finalist_path: Path,
    baseline_rows: list[dict[str, Any]],
    target_registry: dict[str, dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    baseline_map = {
        (
            row["scope_id"],
            row["inner_fold"],
            row["candidate_id"],
            row["history_depth_rows"],
        ): row
        for row in baseline_rows
    }
    for finalist in _read_gzip_jsonl(finalist_path):
        representative = cast(dict[str, Any], finalist["representative"])
        target = target_registry[cast(str, finalist["candidate_id"])]
        baselines = [
            baseline_map[
                (
                    finalist["scope_id"],
                    inner["inner_fold"],
                    finalist["candidate_id"],
                    finalist["history_depth_rows"],
                )
            ]
            for inner in cast(list[dict[str, Any]], representative["inner"])
        ]
        target_offsets = list(
            range(int(target["future_start_rows"]), int(target["future_end_rows"]) + 1)
        )
        gap_offsets = list(range(1, int(target["future_start_rows"])))
        yield {
            "schema_version": "veatic21_phase02_ar_dominance_overlap_v1",
            "feature_set_id": finalist["feature_set_id"],
            "scope_id": finalist["scope_id"],
            "protocol": finalist["protocol"],
            "repeat": finalist["repeat"],
            "outer_fold": finalist["outer_fold"],
            "candidate_id": finalist["candidate_id"],
            "feature_form": finalist["feature_form"],
            "history_depth_rows": finalist["history_depth_rows"],
            "history_offsets_rows": _history_offsets(
                cast(str, finalist["feature_form"]),
                int(finalist["history_depth_rows"]),
            ),
            "target_offsets_rows": target_offsets,
            "intervening_gap_offsets_rows": gap_offsets,
            "history_target_row_overlap_count": 0,
            "nearest_history_to_target_separation_rows": 1,
            "nearest_history_to_target_separation_seconds": 0.5,
            "ar_mean_inner_raw_pr_auc": representative["mean_raw_pr_auc"],
            "analytic_chance_mean_inner": representative["mean_prevalence"],
            "ar_uplift_over_chance_mean_inner": representative["mean_ar_uplift_over_chance"],
            "ar_positive_uplift_inner_folds": representative["positive_uplift_inner_folds"],
            "ar_registered_inner_folds": representative["expected_inner_folds"],
            "simple_causal_history_baselines": baselines,
            "ar_per_video_consistency": {
                "status": "pending_immutable_finalist_predictions",
                "reason": (
                    "Stage A stored metrics but not predictions; refitting a converged Stage A "
                    "cell is forbidden"
                ),
            },
            "prospective_washout_candidate_scored": False,
            "outer_test_scores_opened": False,
            "cortical_values_opened": False,
        }


def _source_tasks(output_root: Path) -> tuple[ScopeTask, ...]:
    stage_a_ledger_path = PHASE02_STAGE_A_SATURATED_ROOT / ("append-only-experiment-ledger.jsonl")
    rescue_ledger_path = RESCUE_MAIN_ROOT / "append-only-experiment-ledger.jsonl"
    _require(sha256_file(stage_a_ledger_path) == STAGE_A_LEDGER_SHA256, "Stage A ledger changed")
    _require(sha256_file(rescue_ledger_path) == RESCUE_LEDGER_SHA256, "rescue ledger changed")
    rescue_entries: dict[str, tuple[str, str]] = {}
    for entry in _read_jsonl(rescue_ledger_path):
        unit_id = cast(str, entry["original_unit_id"])
        _require(unit_id not in rescue_entries, "duplicate rescue ledger original unit")
        _require(entry.get("outer_test_scores_opened") is False, "outer access in rescue ledger")
        _require(entry.get("cortical_values_opened") is False, "cortical access in rescue ledger")
        rescue_entries[unit_id] = (
            cast(str, entry["unit_result_path"]),
            cast(str, entry["unit_result_sha256"]),
        )

    grouped: dict[tuple[str, int | None, int], list[UnitSource]] = defaultdict(list)
    stage_a_ids: set[str] = set()
    for entry in _read_jsonl(stage_a_ledger_path):
        unit_id = cast(str, entry["unit_id"])
        _require(unit_id not in stage_a_ids, "duplicate Stage A ledger unit")
        stage_a_ids.add(unit_id)
        _require(entry.get("outer_test_scores_opened") is False, "outer access in Stage A ledger")
        meta = _parse_unit_id(unit_id)
        key = (meta["protocol"], meta["repeat"], meta["outer_fold"])
        rescue = rescue_entries.get(unit_id)
        grouped[key].append(
            UnitSource(
                unit_id=unit_id,
                path=cast(str, entry["unit_result_path"]),
                sha256=cast(str, entry["unit_result_sha256"]),
                rescue_path=None if rescue is None else rescue[0],
                rescue_sha256=None if rescue is None else rescue[1],
            )
        )
    _require(len(stage_a_ids) == EXPECTED_STAGE_A_UNITS, "Stage A ledger count changed")
    _require(set(rescue_entries) <= stage_a_ids, "rescue ledger has unknown Stage A unit")
    expected_scopes = 42
    _require(len(grouped) == expected_scopes, "outer-scope count changed")
    ordered_keys = sorted(
        grouped,
        key=lambda key: (0 if key[0] == "blocked" else 1, -1 if key[1] is None else key[1], key[2]),
    )
    return tuple(
        ScopeTask(
            scope_index=index,
            protocol=key[0],
            repeat=key[1],
            outer_fold=key[2],
            expected_inner_folds=1 if key[0] == "blocked" else 4,
            units=tuple(grouped[key]),
            shard_root=str(output_root / "shards"),
        )
        for index, key in enumerate(ordered_keys)
    )


def _aggregation_code_sha256() -> str:
    path = Path(__file__)
    return sha256_file(path)


def _pressure_snapshot() -> dict[str, Any]:
    commands = {
        "thermal": ["/usr/bin/pmset", "-g", "therm"],
        "power": ["/usr/bin/pmset", "-g", "custom"],
        "swap": ["/usr/sbin/sysctl", "vm.swapusage"],
        "memory": ["/usr/bin/memory_pressure", "-Q"],
    }
    result: dict[str, Any] = {}
    for name, command in commands.items():
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        result[name] = completed.stdout.strip()
        result[f"{name}_returncode"] = completed.returncode
    return result


def _benchmark_chunk(chunk: BenchmarkChunk) -> dict[str, Any]:
    unit_digests: list[tuple[str, str]] = []
    materialized_rows: list[dict[str, Any]] = []
    source_bytes = 0
    cells = 0
    for source in chunk.sources:
        path = Path(source.path)
        payload = path.read_bytes()
        source_bytes += len(payload)
        _require(hashlib.sha256(payload).hexdigest() == source.sha256, "benchmark source changed")
        result = _strict_json_bytes(payload, path)
        records = cast(list[dict[str, Any]], result["records"])
        _require(len(records) == 210, "benchmark Stage A cell count changed")
        rescues: dict[str, dict[str, Any]] = {}
        if source.rescue_path is not None:
            rescue_path = Path(source.rescue_path)
            rescue_payload = rescue_path.read_bytes()
            source_bytes += len(rescue_payload)
            _require(source.rescue_sha256 is not None, "benchmark rescue hash absent")
            _require(
                hashlib.sha256(rescue_payload).hexdigest() == source.rescue_sha256,
                "benchmark rescue source changed",
            )
            rescue = _strict_json_bytes(rescue_payload, rescue_path)
            for rescue_record in cast(list[dict[str, Any]], rescue["records"]):
                identifier = cast(str, rescue_record["original_configuration_id"])
                _require(identifier not in rescues, "duplicate benchmark rescue cell")
                rescues[identifier] = rescue_record
        cells += len(records)
        normalized: list[dict[str, Any]] = []
        used: set[str] = set()
        for record in records:
            selected, source_kind = _validated_cell(
                record,
                rescues,
                original_unit_id=source.unit_id,
                original_unit_sha256=source.sha256,
            )
            if source_kind == "linked_rescue":
                used.add(cast(str, record["configuration_id"]))
            normalized.append(
                {
                    "configuration_id": record["configuration_id"],
                    "source_kind": source_kind,
                    "disposition": selected["disposition"],
                    "raw_pr_auc": selected["raw_pr_auc"],
                    "brier": selected["brier"],
                    "prevalence": selected["prevalence"],
                }
            )
        _require(used == set(rescues), "benchmark rescue coverage changed")
        materialized_rows.extend(normalized)
        unit_digests.append(
            (
                source.unit_id,
                hashlib.sha256(
                    canonical_json_bytes(
                        {
                            "unit_id": source.unit_id,
                            "stage_a_sha256": source.sha256,
                            "records": normalized,
                        }
                    )
                ).hexdigest(),
            )
        )
    output_path = Path(chunk.output_path)
    _atomic_gzip_jsonl(output_path, materialized_rows)
    return {
        "chunk_index": chunk.index,
        "source_bytes": source_bytes,
        "cells": cells,
        "unit_digests": unit_digests,
        "compressed_output_bytes": output_path.stat().st_size,
        "compressed_output_sha256": sha256_file(output_path),
    }


def _benchmark_once(
    sources: tuple[UnitSource, ...], workers: int, output_root: Path
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=False)
    chunks = tuple(
        BenchmarkChunk(
            index,
            tuple(sources[index::workers]),
            str(output_root / f"shard-{index:02d}.jsonl.gz"),
        )
        for index in range(workers)
    )
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_benchmark_chunk, chunks))
    elapsed = time.monotonic() - started
    unit_digests = sorted(
        (item for result in results for item in result["unit_digests"]),
        key=lambda item: item[0],
    )
    return {
        "workers": workers,
        "elapsed_seconds": elapsed,
        "units_per_second": len(sources) / elapsed,
        "source_mib_per_second": (
            sum(int(result["source_bytes"]) for result in results) / (1024 * 1024) / elapsed
        ),
        "compressed_output_mib_per_second": (
            sum(int(result["compressed_output_bytes"]) for result in results)
            / (1024 * 1024)
            / elapsed
        ),
        "compressed_output_bytes": sum(
            int(result["compressed_output_bytes"]) for result in results
        ),
        "units": len(unit_digests),
        "cells": sum(int(result["cells"]) for result in results),
        "normalized_digest": _sha256_lines(
            f"{unit_id}:{digest}" for unit_id, digest in unit_digests
        ),
    }


def _registered_baseline_benchmark_groups(
    tasks: tuple[ScopeTask, ...],
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for task in tasks:
        descriptors: list[dict[str, Any]] = []
        for target_index, target in enumerate(TARGETS):
            for depth in HISTORY_DEPTHS:
                descriptors.append(
                    {
                        "scope_id": _scope_id(task.protocol, task.repeat, task.outer_fold),
                        "protocol": task.protocol,
                        "repeat": task.repeat,
                        "outer_fold": task.outer_fold,
                        "inner_fold": 0,
                        "candidate_id": target,
                        "target_index": target_index,
                        "history_depth_rows": depth,
                    }
                )
        groups.append(descriptors)
    _require(len(groups) == 42, "baseline benchmark scope count changed")
    _require(
        sum(len(group) for group in groups) == 18_522,
        "baseline benchmark row count changed",
    )
    return groups


def _benchmark_baseline_scope(task: tuple[int, list[dict[str, Any]], str]) -> dict[str, Any]:
    index, descriptors, output_path_value = task
    rows = _baseline_scope_task(descriptors)
    output_path = Path(output_path_value)
    _atomic_gzip_jsonl(output_path, rows)
    return {
        "scope_index": index,
        "rows": len(rows),
        "output_bytes": output_path.stat().st_size,
        "output_sha256": sha256_file(output_path),
    }


def _benchmark_baselines_once(
    groups: list[list[dict[str, Any]]], workers: int, output_root: Path
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=False)
    tasks = [
        (index, descriptors, str(output_root / f"scope-{index:02d}.jsonl.gz"))
        for index, descriptors in enumerate(groups)
    ]
    started = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_initialize_baseline_worker
    ) as executor:
        results = list(executor.map(_benchmark_baseline_scope, tasks))
    elapsed = time.monotonic() - started
    results.sort(key=lambda row: int(row["scope_index"]))
    rows = sum(int(row["rows"]) for row in results)
    output_bytes = sum(int(row["output_bytes"]) for row in results)
    return {
        "workers": workers,
        "elapsed_seconds": elapsed,
        "baseline_rows": rows,
        "baseline_rows_per_second": rows / elapsed,
        "compressed_output_bytes": output_bytes,
        "compressed_output_mib_per_second": output_bytes / (1024 * 1024) / elapsed,
        "normalized_digest": _sha256_lines(
            f"{int(row['scope_index']):02d}:{row['output_sha256']}" for row in results
        ),
    }


def _select_worker_summary(
    summaries: list[dict[str, Any]], throughput_key: str
) -> tuple[dict[str, Any], float]:
    fastest = max(float(row[throughput_key]) for row in summaries)
    plateau = [row for row in summaries if float(row[throughput_key]) >= 0.97 * fastest]
    return min(plateau, key=lambda row: int(row["workers"])), fastest


def run_phase02_stage_a_aggregation_executor_backtest(
    *, output_root: Path = AGGREGATION_EXECUTOR_BACKTEST_ROOT
) -> dict[str, Any]:
    """Measure both CPU/I-O workload shapes on immutable real VEATIC data."""

    output_root = reject_forbidden_runtime_path(output_root)
    _require(
        output_root == AGGREGATION_EXECUTOR_BACKTEST_ROOT,
        "aggregation executor backtest must use its canonical root",
    )
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to reuse aggregation backtest root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    tasks = _source_tasks(output_root)
    sources = tuple(source for task in tasks[:3] for source in task.units)
    _require(len(sources) == 1_512, "aggregation benchmark coverage changed")
    baseline_groups = _registered_baseline_benchmark_groups(tasks)
    candidates = [1, 2, 4, 8, 12]
    request = {
        "schema_version": "veatic21_phase02_stage_a_aggregation_backtest_request_v2",
        "aggregation_code_sha256": _aggregation_code_sha256(),
        "aggregation_registration_sha256": sha256_file(AGGREGATION_REGISTRATION),
        "stage_a_ledger_sha256": STAGE_A_LEDGER_SHA256,
        "rescue_ledger_sha256": RESCUE_LEDGER_SHA256,
        "candidate_workers": candidates,
        "timed_repetitions": 3,
        "source_pipeline": {
            "representative_units": len(sources),
            "operations": (
                "immutable hashing, strict JSON parsing, exact rescue-link resolution, "
                "metric/disposition row materialization, and deterministic gzip compression"
            ),
            "coverage": (
                "both blocked scopes plus one complete four-inner-fold grouped scope; all "
                "forms, histories, linear families, targets, and linked rescues"
            ),
        },
        "analytic_pipeline": {
            "representative_rows": sum(len(group) for group in baseline_groups),
            "operations": (
                "fresh per-process VEATIC history loading, split masks, q90 labels, five "
                "aggregate baselines, defined-only per-video metrics, and gzip compression"
            ),
            "coverage": (
                "all 42 outer scopes at inner fold zero, all targets, all histories, and both "
                "protocols"
            ),
        },
        "selection": (
            "select source-pipeline and analytic-pipeline workers independently by fastest "
            "median throughput; within three percent choose fewer processes"
        ),
        "outer_test_scores_opened": False,
        "cortical_values_opened": False,
    }
    _write_json(output_root / "request.json", request)
    orders = (
        (12, 8, 4, 2, 1),
        (1, 2, 4, 8, 12),
        (4, 12, 2, 8, 1),
    )
    source_repetitions: list[dict[str, Any]] = []
    analytic_repetitions: list[dict[str, Any]] = []
    pressure_before = _pressure_snapshot()
    for repetition, order in enumerate(orders):
        for workers in order:
            source_repetitions.append(
                {
                    "repetition": repetition,
                    **_benchmark_once(
                        sources,
                        workers,
                        output_root
                        / "source-pipeline"
                        / f"repetition-{repetition}"
                        / f"workers-{workers}",
                    ),
                }
            )
        for workers in order:
            analytic_repetitions.append(
                {
                    "repetition": repetition,
                    **_benchmark_baselines_once(
                        baseline_groups,
                        workers,
                        output_root
                        / "analytic-pipeline"
                        / f"repetition-{repetition}"
                        / f"workers-{workers}",
                    ),
                }
            )
    pressure_after = _pressure_snapshot()
    _require(
        len({row["normalized_digest"] for row in source_repetitions}) == 1,
        "worker topology changed normalized source evidence",
    )
    _require(
        len({row["normalized_digest"] for row in analytic_repetitions}) == 1,
        "worker topology changed normalized analytic evidence",
    )
    source_summaries: list[dict[str, Any]] = []
    analytic_summaries: list[dict[str, Any]] = []
    for workers in candidates:
        source_rows = [row for row in source_repetitions if row["workers"] == workers]
        analytic_rows = [row for row in analytic_repetitions if row["workers"] == workers]
        source_summaries.append(
            {
                "workers": workers,
                "median_units_per_second": median(
                    [float(row["units_per_second"]) for row in source_rows]
                ),
                "median_source_mib_per_second": median(
                    [float(row["source_mib_per_second"]) for row in source_rows]
                ),
                "median_compressed_output_mib_per_second": median(
                    [float(row["compressed_output_mib_per_second"]) for row in source_rows]
                ),
                "elapsed_seconds": [row["elapsed_seconds"] for row in source_rows],
                "normalized_digest": source_rows[0]["normalized_digest"],
            }
        )
        analytic_summaries.append(
            {
                "workers": workers,
                "median_baseline_rows_per_second": median(
                    [float(row["baseline_rows_per_second"]) for row in analytic_rows]
                ),
                "median_compressed_output_mib_per_second": median(
                    [float(row["compressed_output_mib_per_second"]) for row in analytic_rows]
                ),
                "elapsed_seconds": [row["elapsed_seconds"] for row in analytic_rows],
                "normalized_digest": analytic_rows[0]["normalized_digest"],
            }
        )
    selected_source, fastest_source = _select_worker_summary(
        source_summaries, "median_units_per_second"
    )
    selected_analytic, fastest_analytic = _select_worker_summary(
        analytic_summaries, "median_baseline_rows_per_second"
    )
    result = {
        "schema_version": "veatic21_phase02_stage_a_aggregation_backtest_result_v2",
        "status": "PASS",
        "request_sha256": sha256_file(output_root / "request.json"),
        "source_repetitions": source_repetitions,
        "analytic_repetitions": analytic_repetitions,
        "source_candidate_summaries": source_summaries,
        "analytic_candidate_summaries": analytic_summaries,
        "fastest_source_median_units_per_second": fastest_source,
        "fastest_analytic_median_rows_per_second": fastest_analytic,
        "selected_aggregation_workers": selected_source["workers"],
        "selected_baseline_workers": selected_analytic["workers"],
        "selected_source_median_units_per_second": selected_source["median_units_per_second"],
        "selected_analytic_median_rows_per_second": selected_analytic[
            "median_baseline_rows_per_second"
        ],
        "selected_source_within_fastest_fraction": (
            float(selected_source["median_units_per_second"]) / fastest_source
        ),
        "selected_analytic_within_fastest_fraction": (
            float(selected_analytic["median_baseline_rows_per_second"]) / fastest_analytic
        ),
        "source_identity_gate": "PASS",
        "analytic_identity_gate": "PASS",
        "pressure_before": pressure_before,
        "pressure_after": pressure_after,
        "outer_test_scores_opened": False,
        "cortical_values_opened": False,
    }
    _write_json(output_root / "result.json", result)
    return result


def run_phase02_stage_a_aggregation(
    *,
    output_root: Path = AGGREGATION_ROOT,
    aggregation_workers: int | None = None,
    baseline_workers: int | None = None,
) -> dict[str, Any]:
    """Aggregate immutable Stage A evidence and freeze Stage B without executing it."""

    output_root = reject_forbidden_runtime_path(output_root)
    _require(output_root == AGGREGATION_ROOT, "aggregation must use its canonical root")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to reuse aggregation root: {output_root}")
    started = time.monotonic()
    frozen_executor = load_json(SELECTED_AGGREGATION_EXECUTOR)
    backtest_request_path = AGGREGATION_EXECUTOR_BACKTEST_ROOT / "request.json"
    backtest_result_path = AGGREGATION_EXECUTOR_BACKTEST_ROOT / "result.json"
    _require(
        frozen_executor.get("aggregation_code_sha256") == _aggregation_code_sha256(),
        "selected aggregation executor code identity changed",
    )
    _require(
        frozen_executor.get("backtest_request_sha256") == sha256_file(backtest_request_path),
        "aggregation executor backtest request changed",
    )
    _require(
        frozen_executor.get("backtest_result_sha256") == sha256_file(backtest_result_path),
        "aggregation executor backtest result changed",
    )
    frozen_aggregation_workers = int(frozen_executor["selected_aggregation_workers"])
    frozen_baseline_workers = int(frozen_executor["selected_baseline_workers"])
    _require(
        aggregation_workers is None or aggregation_workers == frozen_aggregation_workers,
        "main source-pipeline worker override differs from frozen executor",
    )
    _require(
        baseline_workers is None or baseline_workers == frozen_baseline_workers,
        "main analytic-pipeline worker override differs from frozen executor",
    )
    selected_aggregation_workers = frozen_aggregation_workers
    selected_baseline_workers = frozen_baseline_workers
    _require(
        1 <= selected_aggregation_workers <= (os.cpu_count() or 1),
        "invalid aggregation worker count",
    )
    _require(
        1 <= selected_baseline_workers <= (os.cpu_count() or 1),
        "invalid baseline worker count",
    )
    _require(
        sha256_file(PHASE02_REGISTRATION_ROOT / "experiment-registration.json")
        == PHASE02_REGISTRATION_SHA256,
        "Phase 02 registration changed",
    )
    _require(
        sha256_file(PHASE02_STAGE_A_SATURATED_ROOT / "verification.json")
        == STAGE_A_VERIFICATION_SHA256,
        "Stage A verification changed",
    )
    _require(
        sha256_file(RESCUE_MAIN_ROOT / "verification.json") == RESCUE_VERIFICATION_SHA256,
        "rescue verification changed",
    )
    policy = load_json(AGGREGATION_REGISTRATION)
    _require(
        policy.get("registration_status")
        == "prospective_before_stage_a_aggregation_and_stage_b_execution",
        "aggregation policy is not prospective",
    )
    output_root.mkdir(parents=True, exist_ok=False)
    request = {
        "schema_version": "veatic21_phase02_stage_a_aggregation_request_v1",
        "output_root": str(output_root),
        "aggregation_registration_path": str(AGGREGATION_REGISTRATION),
        "aggregation_registration_sha256": sha256_file(AGGREGATION_REGISTRATION),
        "aggregation_code_sha256": _aggregation_code_sha256(),
        "selected_aggregation_executor_path": str(SELECTED_AGGREGATION_EXECUTOR),
        "selected_aggregation_executor_sha256": sha256_file(SELECTED_AGGREGATION_EXECUTOR),
        "aggregation_executor_backtest_request_sha256": sha256_file(backtest_request_path),
        "aggregation_executor_backtest_result_sha256": sha256_file(backtest_result_path),
        "phase02_registration_sha256": PHASE02_REGISTRATION_SHA256,
        "stage_a_verification_sha256": STAGE_A_VERIFICATION_SHA256,
        "stage_a_ledger_sha256": STAGE_A_LEDGER_SHA256,
        "rescue_verification_sha256": RESCUE_VERIFICATION_SHA256,
        "rescue_ledger_sha256": RESCUE_LEDGER_SHA256,
        "aggregation_workers": selected_aggregation_workers,
        "baseline_workers": selected_baseline_workers,
        "worker_selection": "independently measured real-data source and analytic pipelines",
        "gpu_used": False,
        "gpu_disposition": (
            "JSON hashing, aggregation, sorting, and analytic metrics are CPU/I-O workloads "
            "without a numerically valid MLX acceleration path"
        ),
        "expected_stage_a_cells": EXPECTED_STAGE_A_CELLS,
        "expected_invalid_cells": EXPECTED_RESCUE_INVALID,
        "stage_b_execution_authorized": False,
        "outer_test_scores_opened": False,
        "cortical_values_opened": False,
        "prospective_washout_candidates_opened": False,
    }
    _write_json(output_root / "request.json", request)
    tasks = _source_tasks(output_root)
    with ProcessPoolExecutor(max_workers=selected_aggregation_workers) as executor:
        scope_summaries = list(executor.map(_aggregate_scope, tasks))
    scope_summaries.sort(key=lambda row: int(row["scope_index"]))
    _write_json(output_root / "scope-summaries.json", scope_summaries)

    def combine(key: str, destination: str) -> Path:
        path = output_root / destination
        _atomic_concat(path, (Path(cast(str, row[key])) for row in scope_summaries))
        return path

    aggregate_path = combine("aggregate_path", "stage-a-configuration-aggregates.jsonl.gz")
    exclusion_path = combine("exclusion_path", "stage-a-invalid-exclusions.jsonl.gz")
    feature_path = combine("feature_path", "stage-a-feature-dispositions.jsonl.gz")
    finalist_path = combine("finalist_path", "stage-b-finalists.jsonl.gz")
    boundary_path = combine("boundary_path", "stage-b-boundary-dispositions.jsonl.gz")
    work_path = output_root / "stage-b-work-registry.jsonl.gz"
    _atomic_gzip_jsonl(work_path, _work_registry_rows(finalist_path))
    work_row_count = 0
    stage_b_cells = 0
    for work_row in _read_gzip_jsonl(work_path):
        work_row_count += 1
        stage_b_cells += int(work_row["candidate_count"])
    _require(work_row_count == 40_824, "Stage B work-unit count changed")

    baseline_rows = _baseline_rows(finalist_path, selected_baseline_workers)
    baseline_path = output_root / "development-simple-baselines.jsonl.gz"
    _atomic_gzip_jsonl(baseline_path, baseline_rows)
    candidate_registry = load_json(PHASE01_ROOT / "candidate-registry.json")
    target_registry = {
        cast(str, row["candidate_id"]): row
        for row in cast(list[dict[str, Any]], candidate_registry["candidates"])
        if row.get("phase02_active") is True
    }
    _require(set(target_registry) == set(TARGETS), "active target registry changed")
    dominance_path = output_root / "development-ar-dominance-overlap.jsonl.gz"
    _atomic_gzip_jsonl(
        dominance_path,
        _dominance_rows(finalist_path, baseline_rows, target_registry),
    )
    dominance_count = sum(1 for _ in _read_gzip_jsonl(dominance_path))

    admitted_cells = sum(int(row["admitted_cells"]) for row in scope_summaries)
    invalid_cells = sum(int(row["excluded_cells"]) for row in scope_summaries)
    rescued_cells = sum(int(row["rescued_cells"]) for row in scope_summaries)
    _require(
        admitted_cells + invalid_cells == EXPECTED_STAGE_A_CELLS,
        "cell disposition total changed",
    )
    _require(invalid_cells == EXPECTED_RESCUE_INVALID, "invalid rescue count changed")
    _require(rescued_cells == EXPECTED_RESCUE_CELLS, "rescue admission source count changed")
    files = {
        "request.json": sha256_file(output_root / "request.json"),
        "scope-summaries.json": sha256_file(output_root / "scope-summaries.json"),
        aggregate_path.name: sha256_file(aggregate_path),
        exclusion_path.name: sha256_file(exclusion_path),
        feature_path.name: sha256_file(feature_path),
        finalist_path.name: sha256_file(finalist_path),
        boundary_path.name: sha256_file(boundary_path),
        work_path.name: sha256_file(work_path),
        baseline_path.name: sha256_file(baseline_path),
        dominance_path.name: sha256_file(dominance_path),
    }
    summary = {
        "schema_version": "veatic21_phase02_stage_a_aggregation_summary_v1",
        "status": "COMPLETE_PENDING_INDEPENDENT_VERIFICATION",
        "stage_a_cells": EXPECTED_STAGE_A_CELLS,
        "admitted_cells": admitted_cells,
        "original_stage_a_admissions": admitted_cells - EXPECTED_RESCUE_ELIGIBLE,
        "linked_converged_rescue_admissions": EXPECTED_RESCUE_ELIGIBLE,
        "invalid_nonconverged_exclusions": invalid_cells,
        "aggregate_configurations": sum(int(row["aggregate_rows"]) for row in scope_summaries),
        "feature_set_dispositions": sum(int(row["feature_rows"]) for row in scope_summaries),
        "stage_b_finalists": sum(int(row["finalist_rows"]) for row in scope_summaries),
        "stage_b_work_units": work_row_count,
        "stage_b_registered_cells": stage_b_cells,
        "boundary_dispositions": sum(int(row["boundary_rows"]) for row in scope_summaries),
        "development_baseline_rows": len(baseline_rows),
        "dominance_overlap_rows": dominance_count,
        "aggregation_workers": selected_aggregation_workers,
        "baseline_workers": selected_baseline_workers,
        "elapsed_seconds": time.monotonic() - started,
        "stage_b_executed": False,
        "outer_test_scores_opened": False,
        "cortical_values_opened": False,
        "prospective_washout_candidates_scored": False,
        "files": files,
    }
    _write_json(output_root / "summary.json", summary)
    manifest = {
        "schema_version": "veatic21_phase02_stage_a_aggregation_manifest_v1",
        "files": {
            **files,
            "summary.json": sha256_file(output_root / "summary.json"),
        },
        "outer_test_scores_opened": False,
        "cortical_values_opened": False,
    }
    _write_json(output_root / "artifact-manifest.json", manifest)
    return summary
