"""Independent verifier for Stage A aggregation and the prospective Stage B registry."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, cast

import numpy as np

from neural_bridge.veatic21.contracts import (
    PHASE02_STAGE_A_SATURATED_ROOT,
)
from neural_bridge.veatic21.data import load_json, sha256_file
from neural_bridge.veatic21.phase00 import _write_json, canonical_json_bytes

OUTPUT_ROOT = PHASE02_STAGE_A_SATURATED_ROOT.parent / ("stage-a-aggregation-stage-b-registration")
BACKTEST_ROOT = PHASE02_STAGE_A_SATURATED_ROOT.parent / (
    "stage-a-aggregation-executor-backtest-v2-end-to-end"
)
RESCUE_ROOT = PHASE02_STAGE_A_SATURATED_ROOT.parent / (
    "stage-a-convergence-rescue/main-hardware-saturated"
)
STAGE_A_LEDGER_SHA256 = "95bf4d4c18b38372ca81af0ee8210a9b18da942db12f6162c8c678a0a1b9d342"
RESCUE_LEDGER_SHA256 = "c4eb95b038a0db6d17abf8dc0cf36152592b69fd3104030cbe65855ed3beda47"
FORMS = (
    "current_only",
    "raw_levels_with_availability_mask",
    "level_and_first_difference",
    "causal_rolling_summary",
    "combined_levels_differences_summaries",
    "raw_sequence_with_availability_mask",
)
DEPTHS = tuple(range(1, 22))
FAMILIES = ("continuous_ridge", "event_logistic_l2")
MULTIPLIERS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
TARGETS = tuple(f"s01_e{index:02d}" for index in range(1, 22))
UNIT_PATTERN = re.compile(
    r"^(?P<sequence>\d{5})_(?P<protocol>blocked|grouped)_r"
    r"(?P<repeat>na|\d{2})_o(?P<outer>\d{2})_i(?P<inner>\d{2})_"
    r"(?P<form>.+)_d(?P<depth>\d{2})_"
    r"(?P<family>continuous_ridge|event_logistic_l2)$"
)


@dataclass(frozen=True)
class VerifyUnit:
    unit_id: str
    path: str
    sha256: str
    rescue_path: str | None
    rescue_sha256: str | None


@dataclass(frozen=True)
class VerifyScope:
    index: int
    protocol: str
    repeat: int | None
    outer_fold: int
    inner_folds: int
    units: tuple[VerifyUnit, ...]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _strict(payload: bytes, path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON value {value}: {path}")

    value = json.loads(payload, parse_constant=reject)
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return cast(dict[str, Any], value)


def _jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            _require(bool(line.strip()), f"blank JSONL line: {path}")
            yield _strict(line.encode(), path)


def _gzip_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            _require(bool(line.strip()), f"blank gzip JSONL line: {path}")
            yield _strict(line.encode(), path)


def _next(iterator: Iterator[dict[str, Any]], name: str) -> dict[str, Any]:
    try:
        return next(iterator)
    except StopIteration as error:
        raise ValueError(f"premature end of {name}") from error


def _exhausted(iterator: Iterator[dict[str, Any]], name: str) -> None:
    try:
        next(iterator)
    except StopIteration:
        return
    raise ValueError(f"unexpected trailing row in {name}")


def _line_digest(values: list[str] | Iterator[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _select_backtest_workers(
    summaries: list[dict[str, Any]], throughput_key: str
) -> tuple[int, float]:
    fastest = max(float(row[throughput_key]) for row in summaries)
    plateau = [row for row in summaries if float(row[throughput_key]) >= 0.97 * fastest]
    return min(int(row["workers"]) for row in plateau), fastest


def verify_phase02_stage_a_aggregation_executor_backtest(
    *, output_root: Path = BACKTEST_ROOT
) -> dict[str, Any]:
    """Independently audit both real-data worker matrices and their selection."""

    _require(output_root == BACKTEST_ROOT, "aggregation backtest verifier root changed")
    request_path = output_root / "request.json"
    result_path = output_root / "result.json"
    request = load_json(request_path)
    result = load_json(result_path)
    candidates = cast(list[int], request["candidate_workers"])
    _require(candidates == [1, 2, 4, 8, 12], "backtest worker candidates changed")
    _require(request["timed_repetitions"] == 3, "backtest repetition count changed")
    _require(result["request_sha256"] == sha256_file(request_path), "request hash changed")
    source_repetitions = cast(list[dict[str, Any]], result["source_repetitions"])
    analytic_repetitions = cast(list[dict[str, Any]], result["analytic_repetitions"])
    expected_pairs = {(repetition, workers) for repetition in range(3) for workers in candidates}
    _require(
        {(int(row["repetition"]), int(row["workers"])) for row in source_repetitions}
        == expected_pairs,
        "source backtest matrix coverage changed",
    )
    _require(
        {(int(row["repetition"]), int(row["workers"])) for row in analytic_repetitions}
        == expected_pairs,
        "analytic backtest matrix coverage changed",
    )
    _require(
        len({row["normalized_digest"] for row in source_repetitions}) == 1,
        "source normalized identity changed across workers",
    )
    _require(
        len({row["normalized_digest"] for row in analytic_repetitions}) == 1,
        "analytic normalized identity changed across workers",
    )

    canonical_source_digest: str | None = None
    canonical_analytic_digest: str | None = None
    source_rows_audited = 0
    analytic_rows_audited = 0
    for row in source_repetitions:
        repetition = int(row["repetition"])
        workers = int(row["workers"])
        root = output_root / "source-pipeline" / f"repetition-{repetition}" / f"workers-{workers}"
        paths = sorted(root.glob("shard-*.jsonl.gz"))
        _require(len(paths) == workers, "source shard count changed")
        normalized: list[tuple[str, bytes]] = []
        for path in paths:
            for value in _gzip_jsonl(path):
                identifier = cast(str, value["configuration_id"])
                normalized.append((identifier, canonical_json_bytes(value)))
        _require(len(normalized) == 317_520, "source materialized row count changed")
        _require(
            len({identifier for identifier, _ in normalized}) == 317_520,
            "source materialized identity duplicated",
        )
        digest = hashlib.sha256()
        for identifier, payload in sorted(normalized, key=lambda item: item[0]):
            digest.update(identifier.encode())
            digest.update(b"\0")
            digest.update(payload)
            digest.update(b"\n")
        candidate_digest = digest.hexdigest()
        if canonical_source_digest is None:
            canonical_source_digest = candidate_digest
        _require(candidate_digest == canonical_source_digest, "source output changed by workers")
        source_rows_audited += len(normalized)

    for row in analytic_repetitions:
        repetition = int(row["repetition"])
        workers = int(row["workers"])
        root = output_root / "analytic-pipeline" / f"repetition-{repetition}" / f"workers-{workers}"
        paths = sorted(root.glob("scope-*.jsonl.gz"))
        _require(len(paths) == 42, "analytic scope shard count changed")
        scope_hashes: list[str] = []
        row_count = 0
        for scope_index, path in enumerate(paths):
            scope_hashes.append(f"{scope_index:02d}:{sha256_file(path)}")
            for value in _gzip_jsonl(path):
                _require(value["outer_test_scores_opened"] is False, "outer analytic access")
                _require(value["cortical_values_opened"] is False, "cortical analytic access")
                row_count += 1
        _require(row_count == 18_522, "analytic materialized row count changed")
        candidate_digest = _line_digest(scope_hashes)
        _require(
            candidate_digest == row["normalized_digest"],
            "analytic normalized digest changed",
        )
        if canonical_analytic_digest is None:
            canonical_analytic_digest = candidate_digest
        _require(
            candidate_digest == canonical_analytic_digest,
            "analytic output changed by workers",
        )
        analytic_rows_audited += row_count

    source_summaries = cast(list[dict[str, Any]], result["source_candidate_summaries"])
    analytic_summaries = cast(list[dict[str, Any]], result["analytic_candidate_summaries"])
    for workers in candidates:
        source_rows = [row for row in source_repetitions if row["workers"] == workers]
        analytic_rows = [row for row in analytic_repetitions if row["workers"] == workers]
        source_summary = next(row for row in source_summaries if row["workers"] == workers)
        analytic_summary = next(row for row in analytic_summaries if row["workers"] == workers)
        _require(
            source_summary["median_units_per_second"]
            == median(float(row["units_per_second"]) for row in source_rows),
            "source median changed",
        )
        _require(
            analytic_summary["median_baseline_rows_per_second"]
            == median(float(row["baseline_rows_per_second"]) for row in analytic_rows),
            "analytic median changed",
        )
    selected_source, fastest_source = _select_backtest_workers(
        source_summaries, "median_units_per_second"
    )
    selected_analytic, fastest_analytic = _select_backtest_workers(
        analytic_summaries, "median_baseline_rows_per_second"
    )
    _require(
        result["selected_aggregation_workers"] == selected_source,
        "source worker selection changed",
    )
    _require(
        result["selected_baseline_workers"] == selected_analytic,
        "analytic worker selection changed",
    )
    for boundary in ("pressure_before", "pressure_after"):
        pressure = cast(dict[str, Any], result[boundary])
        _require("used = 0.00M" in pressure["swap"], "backtest used swap")
        _require("lowpowermode         0" in pressure["power"], "low-power mode enabled")
        _require("No thermal warning" in pressure["thermal"], "thermal warning detected")
        _require("No performance warning" in pressure["thermal"], "performance warning")
    verification = {
        "schema_version": "veatic21_phase02_stage_a_aggregation_backtest_verification_v1",
        "status": "PASS",
        "request_sha256": sha256_file(request_path),
        "result_sha256": sha256_file(result_path),
        "verifier_code_sha256": sha256_file(Path(__file__)),
        "source_rows_audited": source_rows_audited,
        "analytic_rows_audited": analytic_rows_audited,
        "source_output_sha256": canonical_source_digest,
        "analytic_output_sha256": canonical_analytic_digest,
        "selected_aggregation_workers": selected_source,
        "selected_baseline_workers": selected_analytic,
        "fastest_source_median_units_per_second": fastest_source,
        "fastest_analytic_median_rows_per_second": fastest_analytic,
        "outer_test_scores_opened": False,
        "cortical_values_opened": False,
    }
    _write_json(output_root / "verification.json", verification)
    return verification


def _unit_meta(unit_id: str) -> dict[str, Any]:
    match = UNIT_PATTERN.fullmatch(unit_id)
    _require(match is not None, f"invalid unit identity: {unit_id}")
    values = cast(re.Match[str], match).groupdict()
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


def _scopes() -> tuple[VerifyScope, ...]:
    stage_ledger = PHASE02_STAGE_A_SATURATED_ROOT / "append-only-experiment-ledger.jsonl"
    rescue_ledger = RESCUE_ROOT / "append-only-experiment-ledger.jsonl"
    _require(sha256_file(stage_ledger) == STAGE_A_LEDGER_SHA256, "Stage A ledger changed")
    _require(sha256_file(rescue_ledger) == RESCUE_LEDGER_SHA256, "rescue ledger changed")
    rescues: dict[str, tuple[str, str]] = {}
    for row in _jsonl(rescue_ledger):
        identifier = cast(str, row["original_unit_id"])
        _require(identifier not in rescues, "duplicate rescue unit")
        rescues[identifier] = (
            cast(str, row["unit_result_path"]),
            cast(str, row["unit_result_sha256"]),
        )
    grouped: dict[tuple[str, int | None, int], list[VerifyUnit]] = defaultdict(list)
    ids: set[str] = set()
    for row in _jsonl(stage_ledger):
        identifier = cast(str, row["unit_id"])
        _require(identifier not in ids, "duplicate Stage A unit")
        ids.add(identifier)
        meta = _unit_meta(identifier)
        rescue = rescues.get(identifier)
        grouped[(meta["protocol"], meta["repeat"], meta["outer_fold"])].append(
            VerifyUnit(
                identifier,
                cast(str, row["unit_result_path"]),
                cast(str, row["unit_result_sha256"]),
                None if rescue is None else rescue[0],
                None if rescue is None else rescue[1],
            )
        )
    _require(len(ids) == 40_824, "Stage A unit count changed")
    _require(set(rescues) <= ids, "unknown original unit in rescue")
    _require(len(grouped) == 42, "scope count changed")
    keys = sorted(
        grouped,
        key=lambda key: (
            0 if key[0] == "blocked" else 1,
            -1 if key[1] is None else key[1],
            key[2],
        ),
    )
    return tuple(
        VerifyScope(
            index,
            key[0],
            key[1],
            key[2],
            1 if key[0] == "blocked" else 4,
            tuple(grouped[key]),
        )
        for index, key in enumerate(keys)
    )


def _verify_source_scope(scope: VerifyScope) -> dict[str, Any]:
    cells: dict[tuple[str, str, int, str, int], dict[int, tuple[str, bool]]] = defaultdict(dict)
    rescue_cells = 0
    for source in scope.units:
        meta = _unit_meta(source.unit_id)
        payload = Path(source.path).read_bytes()
        _require(hashlib.sha256(payload).hexdigest() == source.sha256, "source hash changed")
        unit = _strict(payload, Path(source.path))
        _require(unit.get("outer_test_scores_opened") is False, "outer source access")
        _require(unit.get("cortical_values_opened") is False, "cortical source access")
        rescues: dict[str, dict[str, Any]] = {}
        if source.rescue_path is not None:
            _require(source.rescue_sha256 is not None, "missing rescue hash")
            rescue_payload = Path(source.rescue_path).read_bytes()
            _require(
                hashlib.sha256(rescue_payload).hexdigest() == source.rescue_sha256,
                "rescue source hash changed",
            )
            rescue_unit = _strict(rescue_payload, Path(source.rescue_path))
            for rescue in cast(list[dict[str, Any]], rescue_unit["records"]):
                identifier = cast(str, rescue["original_configuration_id"])
                _require(identifier not in rescues, "duplicate rescue cell")
                rescues[identifier] = rescue
                rescue_cells += 1
        used: set[str] = set()
        records = cast(list[dict[str, Any]], unit["records"])
        _require(len(records) == 210, "source record count changed")
        for record in records:
            identifier = cast(str, record["configuration_id"])
            regularization_index = int(identifier.rsplit("__reg", 1)[-1])
            disposition = record["disposition"]
            eligible = disposition == "eligible_for_inner_aggregation"
            if not eligible:
                _require(
                    disposition == "protected_from_pruning_requires_16x_budget",
                    "unexpected Stage A source disposition",
                )
                _require(identifier in rescues, "missing rescue record")
                rescue = rescues[identifier]
                used.add(identifier)
                eligible = rescue["disposition"] == "eligible_for_inner_aggregation"
                _require(
                    eligible
                    or rescue["disposition"]
                    == "invalid_nonconverged_after_registered_maximum_budget",
                    "unexpected rescue disposition",
                )
            else:
                _require(identifier not in rescues, "converged Stage A cell was rescued")
            key = (
                cast(str, record["candidate_id"]),
                cast(str, record["feature_form"]),
                int(record["history_depth_rows"]),
                cast(str, record["model_family"]),
                regularization_index,
            )
            inner = int(meta["inner_fold"])
            _require(inner not in cells[key], "duplicate source inner cell")
            cells[key][inner] = (identifier, eligible)
        _require(used == set(rescues), "unconsumed rescue source")
    admitted: list[str] = []
    excluded: list[str] = []
    for target in TARGETS:
        for form in FORMS:
            for depth in DEPTHS:
                for family in FAMILIES:
                    for regularization_index in range(10):
                        values = cells[(target, form, depth, family, regularization_index)]
                        _require(
                            set(values) == set(range(scope.inner_folds)),
                            "source inner coverage changed",
                        )
                        for inner in range(scope.inner_folds):
                            identifier, eligible = values[inner]
                            (admitted if eligible else excluded).append(identifier)
    return {
        "scope_index": scope.index,
        "admitted": len(admitted),
        "excluded": len(excluded),
        "rescued": rescue_cells,
        "admitted_digest": _line_digest(admitted),
        "excluded_digest": _line_digest(excluded),
        "sorted_admitted_digest": _line_digest(sorted(admitted)),
        "sorted_excluded_digest": _line_digest(sorted(excluded)),
    }


def _mean_se(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return (
        float(np.mean(array)),
        0.0 if len(array) == 1 else float(np.std(array, ddof=1) / math.sqrt(len(array))),
    )


def _capacity(row: dict[str, Any]) -> tuple[Any, ...]:
    brier = row["mean_brier"]
    return (
        brier is None,
        math.inf if brier is None else float(brier),
        int(row["history_depth_rows"]),
        int(row["feature_count"]),
        FAMILIES.index(cast(str, row["model_family"])),
        -float(row["regularization_multiplier"]),
        str(row["aggregate_configuration_id"]),
    )


def _one_se(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], float, int, str]:
    complete = [row for row in rows if row["disposition"] == "eligible_for_selection"]
    _require(bool(complete), "feature set lost every complete configuration")
    best = min(
        complete,
        key=lambda row: (
            -float(row["mean_raw_pr_auc"]),
            float(row["standard_error_raw_pr_auc"]),
            str(row["aggregate_configuration_id"]),
        ),
    )
    threshold = float(best["mean_raw_pr_auc"]) - float(best["standard_error_raw_pr_auc"])
    within = [row for row in complete if float(row["mean_raw_pr_auc"]) >= threshold]
    return (
        min(within, key=_capacity),
        threshold,
        len(within),
        cast(str, best["aggregate_configuration_id"]),
    )


def _independent_finalists(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [row for row in features if row["disposition"] == "eligible_feature_set"]
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

    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        representative = cast(dict[str, Any], row["representative"])
        if float(representative["mean_raw_pr_auc"]) >= threshold:
            return (0, *_capacity(representative), str(row["feature_set_id"]))
        return (
            1,
            -float(representative["mean_raw_pr_auc"]),
            float(representative["standard_error_raw_pr_auc"]),
            *_capacity(representative),
            str(row["feature_set_id"]),
        )

    ranked = sorted(eligible, key=key)
    chosen: dict[str, dict[str, Any]] = {}
    for form in FORMS:
        row = next(row for row in ranked if row["feature_form"] == form)
        chosen[cast(str, row["feature_set_id"])] = row
    for region in ("low", "mid", "high"):
        row = next(row for row in ranked if row["history_region"] == region)
        chosen[cast(str, row["feature_set_id"])] = row
    for row in ranked:
        if len(chosen) == 12:
            break
        chosen.setdefault(cast(str, row["feature_set_id"]), row)
    _require(len(chosen) == 12, "independent finalist count changed")
    return [row for row in ranked if row["feature_set_id"] in chosen]


def _verify_aggregation_and_selection(
    scope_summaries: list[dict[str, Any]], source: list[dict[str, Any]]
) -> dict[str, int]:
    aggregates = _gzip_jsonl(OUTPUT_ROOT / "stage-a-configuration-aggregates.jsonl.gz")
    features = _gzip_jsonl(OUTPUT_ROOT / "stage-a-feature-dispositions.jsonl.gz")
    finalists = _gzip_jsonl(OUTPUT_ROOT / "stage-b-finalists.jsonl.gz")
    aggregate_count = 0
    feature_count = 0
    finalist_count = 0
    for scope_index in range(42):
        admitted_ids: list[str] = []
        excluded_ids: list[str] = []
        for target in TARGETS:
            calculated_features: list[dict[str, Any]] = []
            for form in FORMS:
                for depth in DEPTHS:
                    configurations: list[dict[str, Any]] = []
                    for family in FAMILIES:
                        for regularization_index in range(10):
                            row = _next(aggregates, "configuration aggregates")
                            aggregate_count += 1
                            _require(
                                row["candidate_id"] == target, "aggregate target order changed"
                            )
                            _require(row["feature_form"] == form, "aggregate form order changed")
                            _require(
                                row["history_depth_rows"] == depth, "aggregate depth order changed"
                            )
                            _require(
                                row["model_family"] == family, "aggregate family order changed"
                            )
                            _require(
                                row["regularization_index"] == regularization_index,
                                "aggregate regularization order changed",
                            )
                            inner = cast(list[dict[str, Any]], row["inner"])
                            invalid = [
                                value
                                for value in inner
                                if value["source_disposition"]
                                == "invalid_nonconverged_after_registered_maximum_budget"
                            ]
                            for value in inner:
                                identifier = cast(str, value["original_configuration_id"])
                                (excluded_ids if value in invalid else admitted_ids).append(
                                    identifier
                                )
                            if invalid:
                                _require(
                                    row["disposition"]
                                    == "excluded_incomplete_invalid_not_negative",
                                    "invalid aggregate was admitted",
                                )
                                _require(row["mean_raw_pr_auc"] is None, "invalid aggregate scored")
                            else:
                                values = [float(value["raw_pr_auc"]) for value in inner]
                                mean, standard_error = _mean_se(values)
                                _require(
                                    math.isclose(
                                        float(row["mean_raw_pr_auc"]), mean, abs_tol=1e-15
                                    ),
                                    "aggregate mean changed",
                                )
                                _require(
                                    math.isclose(
                                        float(row["standard_error_raw_pr_auc"]),
                                        standard_error,
                                        abs_tol=1e-15,
                                    ),
                                    "aggregate standard error changed",
                                )
                            _require(row["outer_test_scores_opened"] is False, "outer access flag")
                            _require(row["cortical_values_opened"] is False, "cortical access flag")
                            configurations.append(row)
                    stored_feature = _next(features, "feature dispositions")
                    feature_count += 1
                    complete = [
                        row
                        for row in configurations
                        if row["disposition"] == "eligible_for_selection"
                    ]
                    _require(
                        stored_feature["complete_linear_configurations"] == len(complete),
                        "complete configuration count changed",
                    )
                    if complete:
                        winner, threshold, set_size, best_id = _one_se(configurations)
                        stored = cast(dict[str, Any], stored_feature["representative"])
                        _require(
                            stored["aggregate_configuration_id"]
                            == winner["aggregate_configuration_id"],
                            "feature representative changed",
                        )
                        _require(
                            math.isclose(
                                float(stored["one_standard_error_threshold"]),
                                threshold,
                                abs_tol=1e-15,
                            ),
                            "one-standard-error threshold changed",
                        )
                        _require(
                            stored["one_standard_error_set_size"] == set_size,
                            "one-standard-error set size changed",
                        )
                        _require(
                            stored["one_standard_error_best_configuration_id"] == best_id,
                            "one-standard-error best identity changed",
                        )
                        for field in (
                            "mean_raw_pr_auc",
                            "standard_error_raw_pr_auc",
                            "mean_brier",
                            "history_depth_rows",
                            "feature_count",
                            "model_family",
                            "regularization_index",
                            "regularization_multiplier",
                        ):
                            _require(
                                stored[field] == winner[field],
                                f"stored feature representative field changed: {field}",
                            )
                    boundaries = cast(
                        list[dict[str, Any]],
                        stored_feature["family_boundary_dispositions"],
                    )
                    _require(len(boundaries) == 2, "boundary family coverage changed")
                    for family, boundary in zip(FAMILIES, boundaries, strict=True):
                        family_complete = [
                            row
                            for row in configurations
                            if row["model_family"] == family
                            and row["disposition"] == "eligible_for_selection"
                        ]
                        _require(boundary["model_family"] == family, "boundary family changed")
                        if not family_complete:
                            _require(
                                boundary["disposition"] == "no_complete_family_configuration",
                                "missing family boundary disposition changed",
                            )
                            continue
                        family_winner, _, _, _ = _one_se(family_complete)
                        index = int(family_winner["regularization_index"])
                        expected_multiplier = 1e-7 if index == 0 else 1e4 if index == 9 else None
                        _require(
                            boundary["expansion_multiplier"] == expected_multiplier,
                            "boundary expansion multiplier changed",
                        )
                        _require(
                            cast(dict[str, Any], boundary["family_winner"])[
                                "aggregate_configuration_id"
                            ]
                            == family_winner["aggregate_configuration_id"],
                            "boundary family winner changed",
                        )
                    calculated_features.append(stored_feature)
            expected_finalists = _independent_finalists(calculated_features)
            stored_finalists = [_next(finalists, "Stage B finalists") for _ in range(12)]
            finalist_count += 12
            _require(
                [row["feature_set_id"] for row in stored_finalists]
                == [row["feature_set_id"] for row in expected_finalists],
                "stratified Stage B finalists changed",
            )
            _require(
                {row["feature_form"] for row in stored_finalists} == set(FORMS),
                "finalist form coverage failed",
            )
            _require(
                {row["history_region"] for row in stored_finalists} == {"low", "mid", "high"},
                "finalist region coverage failed",
            )
        expected_source = source[scope_index]
        _require(
            _line_digest(admitted_ids) == expected_source["admitted_digest"],
            "aggregate admission identity changed",
        )
        _require(
            _line_digest(excluded_ids) == expected_source["excluded_digest"],
            "aggregate exclusion identity changed",
        )
        stored_scope = scope_summaries[scope_index]
        _require(
            stored_scope["admitted_configuration_ids_sha256"] == expected_source["admitted_digest"],
            "stored scope admission digest changed",
        )
        _require(
            stored_scope["excluded_configuration_ids_sha256"] == expected_source["excluded_digest"],
            "stored scope exclusion digest changed",
        )
    _exhausted(aggregates, "configuration aggregates")
    _exhausted(features, "feature dispositions")
    _exhausted(finalists, "Stage B finalists")
    return {
        "aggregate_configurations": aggregate_count,
        "feature_dispositions": feature_count,
        "finalists": finalist_count,
    }


def _next_power(value: float) -> int:
    return 1 if value <= 1 else 1 << math.ceil(math.log2(value))


def _floor_power(value: float) -> int:
    return 1 if value < 1 else 1 << math.floor(math.log2(value))


def _nonlinear(family: str, input_dim: int, train_rows: int) -> list[dict[str, Any]]:
    width = _next_power(math.sqrt(input_dim))
    cap = _next_power(math.sqrt(train_rows))
    widths = sorted(
        {max(1, min(cap, width // 2)), min(cap, width), min(cap, 2 * width), min(cap, 4 * width)}
    )
    batch = _floor_power(math.sqrt(train_rows))
    batches = sorted({max(1, batch // 2), batch, 2 * batch})
    dropouts = sorted({0.0, min(0.5, 1 / math.sqrt(input_dim)), min(0.5, 2 / math.sqrt(input_dim))})
    rates = [factor / math.sqrt(train_rows) for factor in (0.25, 1.0, 4.0, 16.0)]
    base: dict[str, Any] = {
        "family": family,
        "width": width,
        "layers": 1,
        "activation": "relu" if family == "event_mlp" else "gru_native_tanh_sigmoid",
        "dropout": 0.0,
        "optimizer": "adamw",
        "learning_rate": 1 / math.sqrt(train_rows),
        "batch_size": batch,
    }
    axes: list[tuple[str, list[Any]]] = [
        ("width", widths),
        ("layers", [1, 2, 3] if family == "event_mlp" else [1, 2]),
        ("dropout", dropouts),
        ("optimizer", ["adamw", "sgd_nesterov"]),
        ("learning_rate", rates),
        ("batch_size", batches),
    ]
    if family == "event_mlp":
        axes.insert(2, ("activation", ["relu", "gelu", "tanh"]))
    values = {canonical_json_bytes(base): base}
    for name, candidates in axes:
        for value in candidates:
            candidate = {**base, name: value}
            values[canonical_json_bytes(candidate)] = candidate
    return [values[key] for key in sorted(values)]


def _work_candidates(finalist: dict[str, Any], inner: dict[str, Any]) -> list[dict[str, Any]]:
    input_dim = int(finalist["feature_count"])
    train_rows = int(inner["train_rows"])
    budget = _next_power(math.sqrt(train_rows))
    values: list[dict[str, Any]] = []
    for boundary in cast(list[dict[str, Any]], finalist["family_boundary_dispositions"]):
        multiplier = boundary["expansion_multiplier"]
        if multiplier is not None:
            values.append(
                {
                    "family": boundary["model_family"],
                    "search_role": "registered_linear_boundary_expansion",
                    "regularization_multiplier": multiplier,
                    "regularization_value": float(multiplier)
                    * float(inner["regularization_scale"]),
                    "initial_update_budget": budget
                    if boundary["model_family"] == "continuous_ridge"
                    else 4 * budget,
                    "maximum_convergence_budget": 16 * budget,
                }
            )
    for multiplier in MULTIPLIERS:
        for ratio in (0.25, 0.5, 0.75, 1.0):
            values.append(
                {
                    "family": "event_elastic_net",
                    "search_role": "complete_elastic_net_grid",
                    "regularization_multiplier": multiplier,
                    "regularization_value": multiplier * float(inner["regularization_scale"]),
                    "l1_ratio": ratio,
                    "initial_update_budget": budget,
                    "undertraining_recovery_maximum_budget": 2 * budget,
                }
            )
    for family in ("event_mlp", "event_gru"):
        if (
            family == "event_gru"
            and finalist["feature_form"] != "raw_sequence_with_availability_mask"
        ):
            continue
        for candidate in _nonlinear(family, input_dim, train_rows):
            values.append(
                {
                    **candidate,
                    "search_role": "one_factor_at_a_time",
                    "initial_update_budget": budget,
                    "undertraining_recovery_maximum_budget": 2 * budget,
                }
            )
    unique = {canonical_json_bytes(value): value for value in values}
    return [unique[key] for key in sorted(unique)]


def _verify_stage_b_work() -> dict[str, int]:
    finalist_rows = list(_gzip_jsonl(OUTPUT_ROOT / "stage-b-finalists.jsonl.gz"))
    finalists = {cast(str, row["feature_set_id"]): row for row in finalist_rows}
    _require(len(finalists) == 10_584, "finalist identity count changed")
    boundaries = _gzip_jsonl(OUTPUT_ROOT / "stage-b-boundary-dispositions.jsonl.gz")
    boundary_count = 0
    for finalist in finalist_rows:
        for expected in cast(list[dict[str, Any]], finalist["family_boundary_dispositions"]):
            stored = _next(boundaries, "boundary dispositions")
            _require(
                stored["feature_set_id"] == finalist["feature_set_id"],
                "boundary feature identity changed",
            )
            for field in (
                "model_family",
                "disposition",
                "expansion_multiplier",
                "family_winner",
            ):
                _require(stored[field] == expected[field], f"boundary field changed: {field}")
            _require(stored["executed"] is False, "boundary expansion was executed")
            boundary_count += 1
    _exhausted(boundaries, "boundary dispositions")
    _require(boundary_count == 21_168, "boundary disposition count changed")
    work_count = 0
    cell_count = 0
    for work in _gzip_jsonl(OUTPUT_ROOT / "stage-b-work-registry.jsonl.gz"):
        _require(work["sequence"] == work_count, "Stage B work sequence changed")
        finalist = finalists[cast(str, work["feature_set_id"])]
        representative = cast(dict[str, Any], finalist["representative"])
        inner_matches = [
            row
            for row in cast(list[dict[str, Any]], representative["inner"])
            if row["inner_fold"] == work["inner_fold"]
        ]
        _require(len(inner_matches) == 1, "Stage B work inner ownership changed")
        expected = _work_candidates(finalist, inner_matches[0])
        _require(work["candidates"] == expected, "Stage B candidate registry changed")
        identifiers = [
            hashlib.sha256(
                canonical_json_bytes(
                    {
                        "feature_set_id": finalist["feature_set_id"],
                        "inner_fold": work["inner_fold"],
                        "candidate": candidate,
                    }
                )
            ).hexdigest()
            for candidate in expected
        ]
        _require(
            work["candidate_ids_sha256"] == _line_digest(identifiers),
            "Stage B candidate identity digest changed",
        )
        _require(work["execution_status"] == "not_executed", "Stage B was executed")
        _require(work["outer_test_scores_opened"] is False, "outer Stage B access")
        _require(work["cortical_values_opened"] is False, "cortical Stage B access")
        work_count += 1
        cell_count += len(expected)
    _require(work_count == 40_824, "Stage B work count changed")
    return {
        "work_units": work_count,
        "registered_cells": cell_count,
        "boundary_dispositions": boundary_count,
    }


def _verify_exclusion_registry(source: list[dict[str, Any]]) -> int:
    rows = _gzip_jsonl(OUTPUT_ROOT / "stage-a-invalid-exclusions.jsonl.gz")
    total = 0
    for scope_index in range(42):
        expected_count = int(source[scope_index]["excluded"])
        identifiers: list[str] = []
        for _ in range(expected_count):
            row = _next(rows, "invalid exclusion registry")
            _require(
                row["scientific_interpretation"] == "incomplete_invalid_not_negative",
                "invalid exclusion was interpreted as negative evidence",
            )
            _require(
                row["rescue_disposition"] == "invalid_nonconverged_after_registered_maximum_budget",
                "invalid exclusion disposition changed",
            )
            identifiers.append(cast(str, row["original_configuration_id"]))
        _require(identifiers == sorted(identifiers), "scope exclusions are not sorted")
        _require(
            _line_digest(identifiers) == source[scope_index]["sorted_excluded_digest"],
            "exclusion registry identity changed",
        )
        total += len(identifiers)
    _exhausted(rows, "invalid exclusion registry")
    return total


def _verify_decomposition() -> dict[str, int]:
    baselines: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in _gzip_jsonl(OUTPUT_ROOT / "development-simple-baselines.jsonl.gz"):
        key = (
            row["scope_id"],
            row["inner_fold"],
            row["candidate_id"],
            row["history_depth_rows"],
        )
        _require(key not in baselines, "duplicate development baseline")
        for name, metric in cast(dict[str, dict[str, Any]], row["baselines"]).items():
            raw_pr_auc = metric["raw_pr_auc"]
            _require(raw_pr_auc is None or 0 <= float(raw_pr_auc) <= 1, "invalid baseline PR-AUC")
            per_video = cast(dict[str, Any], metric["per_video"])
            _require(
                int(per_video["defined_videos"]) + int(per_video["undefined_single_class_videos"])
                > 0,
                "baseline video accounting empty",
            )
            if name == "training_prevalence_constant":
                _require(
                    math.isclose(float(raw_pr_auc), float(metric["prevalence"]), abs_tol=1e-15),
                    "analytic chance does not equal event prevalence",
                )
        _require(row["outer_test_scores_opened"] is False, "outer baseline access")
        _require(row["cortical_values_opened"] is False, "cortical baseline access")
        baselines[key] = row
    dominance_count = 0
    for row in _gzip_jsonl(OUTPUT_ROOT / "development-ar-dominance-overlap.jsonl.gz"):
        history = set(cast(list[int], row["history_offsets_rows"]))
        target = set(cast(list[int], row["target_offsets_rows"]))
        gap = cast(list[int], row["intervening_gap_offsets_rows"])
        _require(not history & target, "history/target row overlap detected")
        _require(not gap, "no-washout decomposition contains a gap")
        _require(row["history_target_row_overlap_count"] == 0, "overlap count changed")
        _require(
            math.isclose(
                float(row["ar_uplift_over_chance_mean_inner"]),
                float(row["ar_mean_inner_raw_pr_auc"]) - float(row["analytic_chance_mean_inner"]),
                abs_tol=1e-15,
            ),
            "AR uplift arithmetic changed",
        )
        embedded = cast(list[dict[str, Any]], row["simple_causal_history_baselines"])
        for baseline in embedded:
            key = (
                baseline["scope_id"],
                baseline["inner_fold"],
                baseline["candidate_id"],
                baseline["history_depth_rows"],
            )
            _require(baselines[key] == baseline, "dominance baseline link changed")
        _require(row["outer_test_scores_opened"] is False, "outer dominance access")
        _require(row["cortical_values_opened"] is False, "cortical dominance access")
        dominance_count += 1
    _require(dominance_count == 10_584, "dominance row count changed")
    return {"baseline_rows": len(baselines), "dominance_rows": dominance_count}


def verify_phase02_stage_a_aggregation(*, output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    _require(output_root == OUTPUT_ROOT, "aggregation verifier root changed")
    request = load_json(output_root / "request.json")
    summary = load_json(output_root / "summary.json")
    manifest = load_json(output_root / "artifact-manifest.json")
    _require(request["outer_test_scores_opened"] is False, "outer request access")
    _require(request["cortical_values_opened"] is False, "cortical request access")
    _require(request["stage_b_execution_authorized"] is False, "Stage B authorization changed")
    for name, digest in cast(dict[str, str], manifest["files"]).items():
        _require(sha256_file(output_root / name) == digest, f"manifest hash changed: {name}")
    scope_summaries = cast(list[dict[str, Any]], load_json(output_root / "scope-summaries.json"))
    _require(len(scope_summaries) == 42, "scope summary count changed")
    for row in scope_summaries:
        for path_key, hash_key in (
            ("aggregate_path", "aggregate_sha256"),
            ("exclusion_path", "exclusion_sha256"),
            ("feature_path", "feature_sha256"),
            ("finalist_path", "finalist_sha256"),
            ("boundary_path", "boundary_sha256"),
        ):
            _require(
                sha256_file(Path(cast(str, row[path_key]))) == row[hash_key],
                f"scope shard hash changed: {path_key}",
            )
    scopes = _scopes()
    workers = int(request["aggregation_workers"])
    with ProcessPoolExecutor(max_workers=workers) as executor:
        source = list(executor.map(_verify_source_scope, scopes))
    source.sort(key=lambda row: int(row["scope_index"]))
    _require(sum(int(row["admitted"]) for row in source) == 8_542_214, "admission count changed")
    _require(sum(int(row["excluded"]) for row in source) == 30_826, "exclusion count changed")
    _require(sum(int(row["rescued"]) for row in source) == 113_392, "rescue count changed")
    for index, row in enumerate(source):
        stored = scope_summaries[index]
        _require(
            stored["sorted_admitted_configuration_ids_sha256"] == row["sorted_admitted_digest"],
            "sorted admission digest changed",
        )
        _require(
            stored["sorted_excluded_configuration_ids_sha256"] == row["sorted_excluded_digest"],
            "sorted exclusion digest changed",
        )
    exclusion_registry_count = _verify_exclusion_registry(source)
    aggregation = _verify_aggregation_and_selection(scope_summaries, source)
    work = _verify_stage_b_work()
    decomposition = _verify_decomposition()
    _require(summary["aggregate_configurations"] == 2_222_640, "aggregate total changed")
    _require(summary["stage_b_finalists"] == 10_584, "finalist total changed")
    _require(summary["stage_b_work_units"] == work["work_units"], "work total changed")
    _require(
        summary["stage_b_registered_cells"] == work["registered_cells"],
        "Stage B cell total changed",
    )
    verification = {
        "schema_version": "veatic21_phase02_stage_a_aggregation_verification_v1",
        "status": "PASS",
        "request_sha256": sha256_file(output_root / "request.json"),
        "summary_sha256": sha256_file(output_root / "summary.json"),
        "manifest_sha256": sha256_file(output_root / "artifact-manifest.json"),
        "verifier_code_sha256": sha256_file(Path(__file__)),
        "source_admissions": 8_542_214,
        "source_invalid_exclusions": 30_826,
        "exclusion_registry_rows": exclusion_registry_count,
        "source_rescue_cells": 113_392,
        **aggregation,
        **work,
        **decomposition,
        "all_admissions_exclusions_independently_rederived": True,
        "all_selections_independently_rederived": True,
        "stage_b_executed": False,
        "outer_test_scores_opened": False,
        "cortical_values_opened": False,
        "prospective_washout_candidates_scored": False,
    }
    _write_json(output_root / "verification.json", verification)
    return verification
