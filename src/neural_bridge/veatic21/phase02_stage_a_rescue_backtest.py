"""Registered systems backtest for the sparse VEATIC Stage A rescue executor."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import traceback
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean, median
from typing import cast

from neural_bridge.veatic21.contracts import REPOSITORY_ROOT
from neural_bridge.veatic21.data import load_json, reject_forbidden_runtime_path, sha256_file
from neural_bridge.veatic21.phase00 import _write_json
from neural_bridge.veatic21.phase02_stage_a_rescue import (
    RescueUnit,
    load_rescue_registry,
    rescue_solver_code_identity,
)
from neural_bridge.veatic21.phase02_stage_a_rescue_executor import (
    RescueExecutorConfiguration,
    RescueRunSelection,
    rescue_configuration_from_dict,
    rescue_executor_code_identity,
    run_rescue_executor,
)
from neural_bridge.veatic21.phase02_stage_a_rescue_registration import (
    RESCUE_CELL_REGISTRY,
    RESCUE_REGISTRATION_ROOT,
    RESCUE_REGISTRATION_VERIFICATION,
    RESCUE_UNIT_REGISTRY,
)

RESCUE_EXECUTOR_BACKTEST_ROOT = RESCUE_REGISTRATION_ROOT.parent / "executor-backtest"
RESCUE_EXECUTOR_BACKTEST_REGISTRATION = REPOSITORY_ROOT / (
    "internal/active/veatic21-phase02-registration/"
    "rescue-executor-backtest-registration.json"
)
SIX_GIBIBYTES = 6 * 1024**3


def _as_int(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"expected int, received {type(value).__name__}")
    return value


def _as_float(value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"expected number, received {type(value).__name__}")
    return float(value)


def _hashed_order(unit: RescueUnit) -> tuple[str, int]:
    return hashlib.sha256(unit.original_unit_id.encode()).hexdigest(), unit.rescue_unit_sequence


def _cell_count_band(count: int) -> str:
    if count == 1:
        return "1"
    if count <= 4:
        return "2-4"
    if count <= 8:
        return "5-8"
    if count <= 16:
        return "9-16"
    return "17-41"


def _history_band(depth: int) -> str:
    if depth <= 2:
        return "1-2"
    if depth <= 5:
        return "3-5"
    if depth <= 10:
        return "6-10"
    if depth <= 15:
        return "11-15"
    return "16-21"


def _representative_tokens(unit: RescueUnit) -> set[str]:
    cell = unit.cells[0]
    tokens = {
        f"family:{unit.model_family}",
        f"protocol:{cell.protocol}",
        f"feature:{cell.feature_form}",
        f"history:{_history_band(cell.history_depth)}",
        f"cells:{_cell_count_band(len(unit.cells))}",
    }
    tokens.update(f"target:{item.candidate_id}" for item in unit.cells)
    tokens.update(f"reg:{item.regularization_index}" for item in unit.cells)
    return tokens


def _representative_selection(
    units: tuple[RescueUnit, ...], size: int, *, minimum_logistic: int
) -> tuple[int, ...]:
    """Cover workload strata first, then fill by an immutable hash—not scientific scores."""

    if size > len(units):
        raise ValueError("representative request exceeds the registry")
    chosen: list[RescueUnit] = []
    chosen_ids: set[int] = set()
    logistic = sorted(
        (unit for unit in units if unit.model_family == "event_logistic_l2"),
        key=_hashed_order,
    )
    for unit in logistic[:minimum_logistic]:
        chosen.append(unit)
        chosen_ids.add(unit.rescue_unit_sequence)
    uncovered = set().union(*(_representative_tokens(unit) for unit in units))
    for unit in chosen:
        uncovered.difference_update(_representative_tokens(unit))
    remaining = [unit for unit in units if unit.rescue_unit_sequence not in chosen_ids]
    while uncovered and len(chosen) < size:
        unit = min(
            remaining,
            key=lambda item: (
                -len(_representative_tokens(item).intersection(uncovered)),
                _hashed_order(item),
            ),
        )
        chosen.append(unit)
        chosen_ids.add(unit.rescue_unit_sequence)
        uncovered.difference_update(_representative_tokens(unit))
        remaining.remove(unit)
    for unit in sorted(remaining, key=_hashed_order):
        if len(chosen) == size:
            break
        chosen.append(unit)
    return tuple(sorted(unit.rescue_unit_sequence for unit in chosen))


def _selection_coverage(
    units: tuple[RescueUnit, ...], sequences: tuple[int, ...]
) -> dict[str, object]:
    selected = [units[index] for index in sequences]

    def values(name: str) -> list[object]:
        if name == "model_family":
            return [unit.model_family for unit in selected]
        return [getattr(unit.cells[0], name) for unit in selected]

    return {
        "rescue_units": len(selected),
        "rescue_cells": sum(len(unit.cells) for unit in selected),
        "model_families": sorted(set(cast(list[str], values("model_family")))),
        "protocols": sorted(set(cast(list[str], values("protocol")))),
        "feature_forms": sorted(set(cast(list[str], values("feature_form")))),
        "history_depth_rows": sorted(set(cast(list[int], values("history_depth")))),
        "cell_count_bands": sorted({_cell_count_band(len(unit.cells)) for unit in selected}),
        "candidate_ids": sorted({cell.candidate_id for unit in selected for cell in unit.cells}),
        "regularization_indices": sorted(
            {cell.regularization_index for unit in selected for cell in unit.cells}
        ),
        "logistic_units": sum(unit.model_family == "event_logistic_l2" for unit in selected),
        "ridge_units": sum(unit.model_family == "continuous_ridge" for unit in selected),
    }


def register_rescue_executor_backtest() -> dict[str, object]:
    """Freeze representative cells and a staged full-host systems search before timing."""

    units = load_rescue_registry()
    equivalence = _representative_selection(units, 24, minimum_logistic=8)
    timed = _representative_selection(units, 192, minimum_logistic=32)
    warmup = _representative_selection(units, 12, minimum_logistic=4)
    if RESCUE_EXECUTOR_BACKTEST_REGISTRATION.exists():
        existing = load_json(RESCUE_EXECUTOR_BACKTEST_REGISTRATION)
        frozen = cast(dict[str, object], existing["representative_units"])
        if (
            frozen["equivalence_rescue_unit_sequences"] != list(equivalence)
            or frozen["timed_rescue_unit_sequences"] != list(timed)
            or frozen["warmup_rescue_unit_sequences"] != list(warmup)
        ):
            raise ValueError("frozen rescue backtest representative selection changed")
        return existing

    sysctl = subprocess.run(
        ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string", "hw.ncpu", "hw.memsize"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    registration: dict[str, object] = {
        "schema_version": "veatic21_phase02_stage_a_rescue_executor_backtest_registration_v1",
        "registered_at_utc": datetime.now(UTC).isoformat(),
        "purpose": "select_fastest_safe_numerically_equivalent_sparse_rescue_executor",
        "host": {
            "cpu_brand": sysctl[0],
            "logical_cpu_count": int(sysctl[1]),
            "memory_bytes": int(sysctl[2]),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "power_source_required": "AC Power",
            "low_power_mode_required": 0,
            "background_taskpolicy_allowed": False,
        },
        "input_identity": {
            "rescue_cell_registry_sha256": sha256_file(RESCUE_CELL_REGISTRY),
            "rescue_unit_registry_sha256": sha256_file(RESCUE_UNIT_REGISTRY),
            "rescue_registration_verification_sha256": sha256_file(
                RESCUE_REGISTRATION_VERIFICATION
            ),
            "rescue_solver_code_sha256": rescue_solver_code_identity(),
            "rescue_executor_code_sha256": rescue_executor_code_identity(),
        },
        "representative_units": {
            "method": (
                "coverage_greedy_over_family_protocol_feature_history_cell_count_target_and_"
                "regularization_then_sha256_fill_without_scientific_scores"
            ),
            "equivalence_rescue_unit_sequences": list(equivalence),
            "timed_rescue_unit_sequences": list(timed),
            "warmup_rescue_unit_sequences": list(warmup),
            "equivalence_coverage": _selection_coverage(units, equivalence),
            "timed_coverage": _selection_coverage(units, timed),
            "warmup_coverage": _selection_coverage(units, warmup),
        },
        "reference_configuration": asdict(
            RescueExecutorConfiguration(
                id="reference_1p1s_1m_b1_uncompiled",
                mlx_lanes=1,
                gpu_streams_per_lane=1,
                metric_workers_per_lane=1,
                cell_batch_size=1,
                compiled_update_blocks=False,
                fast_metrics=True,
                pipeline_depth=2,
            )
        ),
        "search": {
            "stage_1_cell_batch_sizes": [1, 4, 8, 16, 32, 64],
            "stage_1_fixed_topology": {"mlx_lanes": 3, "gpu_streams_per_lane": 1},
            "stage_1_metric_workers_per_lane": 2,
            "stage_2_safe_topologies": [
                [1, 1],
                [1, 2],
                [1, 4],
                [1, 8],
                [1, 12],
                [2, 1],
                [2, 2],
                [2, 4],
                [2, 6],
                [3, 1],
                [3, 2],
                [3, 4],
                [4, 1],
                [4, 2],
                [4, 3],
                [6, 1],
                [6, 2],
                [8, 1],
                [12, 1],
            ],
            "stage_2_metric_workers_per_lane": 2,
            "stage_3_metric_workers_per_lane": [1, 2, 4, 8],
            "stage_3_topology_advancement": (
                "top_four_plus_all_within_five_percent_of_stage_2_fastest"
            ),
            "stage_4_compilation_values": [False, True],
            "stage_4_configuration_advancement": (
                "top_four_plus_all_within_five_percent_of_stage_3_fastest"
            ),
            "finalist_advancement": "top_eight_plus_all_within_three_percent_of_fastest",
            "timed_repetitions_for_finalists": 3,
            "pipeline_depth": 4,
            "fast_metrics": True,
        },
        "equivalence_contract": {
            "float_absolute_tolerance": 1e-5,
            "integer_boolean_string_null_and_structure": "exact",
            "convergence_disposition_and_iteration_count": "exact",
            "same_configuration_repeat_normalized_artifact": "bitwise_exact",
            "runtime_and_execution_provenance_excluded": True,
        },
        "gates": {
            "minimum_memory_headroom_bytes": SIX_GIBIBYTES,
            "thermal_warning_allowed": False,
            "performance_warning_allowed": False,
            "determinism_required": True,
            "resume_required": True,
            "canonical_ledger_exact_coverage_required": True,
            "outer_test_scores_allowed": False,
            "cortical_values_allowed": False,
            "aggregation_or_pruning_allowed": False,
        },
        "selection_rule": {
            "primary": "median_rescue_cells_per_second_across_three_final_repetitions",
            "near_tie_fraction": 0.03,
            "near_tie_preference": [
                "fewer_total_metal_streams",
                "fewer_mlx_processes",
                "lower_peak_mlx_memory",
                "larger_cell_batch",
            ],
            "gpu_saturation": (
                "mean_gpu_at_least_90_percent_or_next_safe_concurrency_gain_below_5_percent_"
                "or_registered_12_stream_ceiling"
            ),
            "scientific_scores_used": False,
        },
        "outer_test_scores_opened": False,
        "cortical_values_opened": False,
        "aggregation_or_pruning_performed": False,
    }
    RESCUE_EXECUTOR_BACKTEST_REGISTRATION.parent.mkdir(parents=True, exist_ok=True)
    _write_json(RESCUE_EXECUTOR_BACKTEST_REGISTRATION, registration)
    return registration


def _selection(value: object) -> RescueRunSelection:
    if not isinstance(value, list) or not value:
        raise ValueError("rescue selection must be a non-empty list")
    sequences = tuple(_as_int(item) for item in value)
    return RescueRunSelection(sequences)


def _unit_paths(root: Path) -> list[Path]:
    registry = load_json(root / "work-unit-registry.json")
    units = cast(list[dict[str, object]], registry["units"])
    units.sort(key=lambda item: _as_int(item["rescue_unit_sequence"]))
    return [
        root
        / "units"
        / f"{_as_int(unit['rescue_unit_sequence']):05d}_{unit['original_unit_id']}.json"
        for unit in units
    ]


def _record_mismatch(
    stats: dict[str, object], path: str, reference: object, candidate: object
) -> None:
    stats["mismatch_count"] = _as_int(stats["mismatch_count"]) + 1
    examples = cast(list[dict[str, object]], stats["mismatch_examples"])
    if len(examples) < 100:
        examples.append({"path": path, "reference": reference, "candidate": candidate})


def _compare_value(
    reference: object,
    candidate: object,
    *,
    path: str,
    float_tolerance: float,
    stats: dict[str, object],
) -> None:
    if isinstance(reference, dict) and isinstance(candidate, dict):
        if set(reference) != set(candidate):
            _record_mismatch(stats, f"{path}.keys", sorted(reference), sorted(candidate))
            return
        for key in sorted(reference):
            _compare_value(
                reference[key],
                candidate[key],
                path=f"{path}.{key}",
                float_tolerance=float_tolerance,
                stats=stats,
            )
        return
    if isinstance(reference, list) and isinstance(candidate, list):
        if len(reference) != len(candidate):
            _record_mismatch(stats, f"{path}.length", len(reference), len(candidate))
            return
        for index, (left, right) in enumerate(zip(reference, candidate, strict=True)):
            _compare_value(
                left,
                right,
                path=f"{path}[{index}]",
                float_tolerance=float_tolerance,
                stats=stats,
            )
        return
    if type(reference) is int and type(candidate) is int:
        if reference != candidate:
            _record_mismatch(stats, path, reference, candidate)
        return
    if (
        isinstance(reference, (int, float))
        and not isinstance(reference, bool)
        and isinstance(candidate, (int, float))
        and not isinstance(candidate, bool)
    ):
        difference = abs(float(reference) - float(candidate))
        stats["max_float_absolute_difference"] = max(
            _as_float(stats["max_float_absolute_difference"]), difference
        )
        if not math.isfinite(difference) or difference > float_tolerance:
            _record_mismatch(stats, path, reference, candidate)
        return
    if reference != candidate:
        _record_mismatch(stats, path, reference, candidate)


def _normalized_unit(value: dict[str, object]) -> dict[str, object]:
    normalized = dict(value)
    normalized.pop("runtime_seconds", None)
    normalized.pop("execution_provenance", None)
    return normalized


def compare_rescue_output_to_reference(
    *, reference_root: Path, candidate_root: Path, float_tolerance: float
) -> dict[str, object]:
    stats: dict[str, object] = {
        "status": "RUNNING",
        "units_compared": 0,
        "mismatch_count": 0,
        "max_float_absolute_difference": 0.0,
        "mismatch_examples": [],
    }
    reference: dict[int, Path] = {}
    for path in _unit_paths(reference_root):
        rescue_unit = cast(dict[str, object], load_json(path)["rescue_unit"])
        reference[_as_int(rescue_unit["rescue_unit_sequence"])] = path
    candidates = _unit_paths(candidate_root)
    for path in candidates:
        value = load_json(path)
        sequence = _as_int(cast(dict[str, object], value["rescue_unit"])["rescue_unit_sequence"])
        reference_path = reference.get(sequence)
        if reference_path is None:
            _record_mismatch(stats, f"unit[{sequence}]", "present", "missing")
            continue
        _compare_value(
            _normalized_unit(load_json(reference_path)),
            _normalized_unit(value),
            path=f"unit[{sequence}]",
            float_tolerance=float_tolerance,
            stats=stats,
        )
        stats["units_compared"] = _as_int(stats["units_compared"]) + 1
    if len(candidates) != len(reference):
        _record_mismatch(stats, "unit_count", len(reference), len(candidates))
    stats["status"] = "PASS" if stats["mismatch_count"] == 0 else "FAIL"
    stats["outer_test_scores_opened"] = False
    stats["cortical_values_opened"] = False
    stats["aggregation_or_pruning_performed"] = False
    return stats


def _normalized_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in _unit_paths(root):
        value = _normalized_unit(load_json(path))
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        )
        digest.update(b"\0")
    return digest.hexdigest()


def _resume_snapshot(root: Path) -> dict[str, object]:
    paths = _unit_paths(root)
    ledgers = sorted((root / "shards").glob("shard-*/append-only-ledger.jsonl"))
    return {
        "unit_sha256": {path.name: sha256_file(path) for path in paths},
        "shard_ledger_sha256": {str(path): sha256_file(path) for path in ledgers},
        "canonical_ledger_sha256": sha256_file(root / "append-only-experiment-ledger.jsonl"),
    }


def _access_gate(root: Path) -> dict[str, object]:
    failures: list[str] = []
    for path in _unit_paths(root):
        value = load_json(path)
        provenance = cast(dict[str, object], value.get("execution_provenance", {}))
        for name in (
            "outer_test_scores_opened",
            "cortical_values_opened",
            "aggregation_or_pruning_performed",
        ):
            if value.get(name) is not False or provenance.get(name) is not False:
                failures.append(f"{path.name}:{name}")
    return {"status": "PASS" if not failures else "FAIL", "failures": failures[:100]}


def _run_candidate_once(
    *,
    output_root: Path,
    configuration: RescueExecutorConfiguration,
    selection: RescueRunSelection,
    warmup: RescueRunSelection | None,
) -> dict[str, object]:
    return run_rescue_executor(
        output_root=output_root,
        configuration=configuration,
        selection=selection,
        warmup_selection=warmup,
        resource_interval_seconds=0.1,
    )


def _equivalence_and_resume(
    *,
    root: Path,
    configuration: RescueExecutorConfiguration,
    reference_root: Path,
    selection: RescueRunSelection,
    tolerance: float,
) -> tuple[dict[str, object], bool]:
    _run_candidate_once(
        output_root=root,
        configuration=configuration,
        selection=selection,
        warmup=None,
    )
    equivalence = compare_rescue_output_to_reference(
        reference_root=reference_root,
        candidate_root=root,
        float_tolerance=tolerance,
    )
    before = _resume_snapshot(root)
    _run_candidate_once(
        output_root=root,
        configuration=configuration,
        selection=selection,
        warmup=None,
    )
    return equivalence, before == _resume_snapshot(root)


def _thermal_pass(state: dict[str, object]) -> bool:
    resources = cast(dict[str, object], state["resource_summary"])
    after = cast(dict[str, object], resources.get("pressure_after", {}))
    thermal = str(after.get("thermal", ""))
    return (
        "No thermal warning level has been recorded" in thermal
        and "No performance warning level has been recorded" in thermal
    )


def _timed_summary(
    *,
    configuration: RescueExecutorConfiguration,
    roots: list[Path],
    equivalence: dict[str, object],
    resume_pass: bool,
) -> dict[str, object]:
    states = [load_json(root / "run-state.json") for root in roots]
    throughputs = [_as_float(state["rescue_cells_per_second"]) for state in states]
    digests = [_normalized_digest(root) for root in roots]
    resources = [cast(dict[str, object], state["resource_summary"]) for state in states]
    headroom = [item.get("estimated_minimum_memory_headroom_bytes") for item in resources]
    sample_counts = [_as_int(item["sample_count"]) for item in resources]
    gpu = [item.get("gpu_device_utilization_mean_percent") for item in resources]
    memory_pass = all(
        isinstance(value, int) and not isinstance(value, bool) and value >= SIX_GIBIBYTES
        for value in headroom
    ) and all(count > 0 for count in sample_counts)
    ledger_pass = all(
        _as_int(state["ledger_lines"]) == _as_int(state["rescue_units_total"])
        and _as_int(state["unique_rescue_units"]) == _as_int(state["rescue_units_total"])
        and _as_int(state["unique_rescue_cells"]) == _as_int(state["rescue_cells"])
        for state in states
    )
    access_pass = all(_access_gate(root)["status"] == "PASS" for root in roots)
    deterministic = len(set(digests)) == 1
    thermal = all(_thermal_pass(state) for state in states)
    eligible = all(
        (
            equivalence["status"] == "PASS",
            resume_pass,
            memory_pass,
            ledger_pass,
            access_pass,
            deterministic,
            thermal,
        )
    )
    return {
        "configuration": asdict(configuration),
        "status": "PASS" if eligible else "FAIL",
        "eligible_without_saturation_gate": eligible,
        "equivalence": equivalence,
        "resume_gate": "PASS" if resume_pass else "FAIL",
        "determinism_gate": "PASS" if deterministic else "FAIL",
        "ledger_gate": "PASS" if ledger_pass else "FAIL",
        "access_gate": "PASS" if access_pass else "FAIL",
        "memory_gate": "PASS" if memory_pass else "FAIL",
        "thermal_gate": "PASS" if thermal else "FAIL",
        "normalized_unit_digests": digests,
        "rescue_cells_per_second": throughputs,
        "median_rescue_cells_per_second": median(throughputs),
        "elapsed_seconds": [_as_float(state["elapsed_seconds"]) for state in states],
        "gpu_utilization_mean_percent_by_repetition": gpu,
        "gpu_utilization_mean_percent": (
            fmean(_as_float(value) for value in gpu if value is not None)
            if any(value is not None for value in gpu)
            else None
        ),
        "estimated_minimum_memory_headroom_bytes_by_repetition": headroom,
        "peak_summed_mlx_active_memory_bytes_by_repetition": [
            sum(
                _as_int(shard["peak_mlx_active_memory_bytes"])
                for shard in cast(list[dict[str, object]], state["shards"])
            )
            for state in states
        ],
        "sample_counts": sample_counts,
        "outer_test_scores_opened": False,
        "cortical_values_opened": False,
        "aggregation_or_pruning_performed": False,
    }


def _candidate_id(configuration: RescueExecutorConfiguration, stage: str) -> str:
    compiled = "compiled" if configuration.compiled_update_blocks else "uncompiled"
    return (
        f"{stage}_{configuration.mlx_lanes}p{configuration.gpu_streams_per_lane}s_"
        f"{configuration.metric_workers_per_lane}m_b{configuration.cell_batch_size}_{compiled}"
    )


def _config(
    *, lanes: int, streams: int, metrics: int, batch: int, compiled: bool, stage: str
) -> RescueExecutorConfiguration:
    base = RescueExecutorConfiguration(
        id="pending",
        mlx_lanes=lanes,
        gpu_streams_per_lane=streams,
        metric_workers_per_lane=metrics,
        cell_batch_size=batch,
        compiled_update_blocks=compiled,
        fast_metrics=True,
        pipeline_depth=max(4, streams),
    )
    return replace(base, id=_candidate_id(base, stage))


def _eligible_sorted(summaries: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        (item for item in summaries if item.get("eligible_without_saturation_gate") is True),
        key=lambda item: -_as_float(item["median_rescue_cells_per_second"]),
    )


def _advance(
    summaries: list[dict[str, object]], *, minimum: int, fraction: float
) -> list[dict[str, object]]:
    eligible = _eligible_sorted(summaries)
    if not eligible:
        raise ValueError("no eligible rescue executor candidate to advance")
    fastest = _as_float(eligible[0]["median_rescue_cells_per_second"])
    ids = {
        cast(dict[str, object], item["configuration"])["id"]
        for item in eligible
        if _as_float(item["median_rescue_cells_per_second"]) >= fastest * (1 - fraction)
    }
    ids.update(cast(dict[str, object], item["configuration"])["id"] for item in eligible[:minimum])
    return [
        item
        for item in eligible
        if cast(dict[str, object], item["configuration"])["id"] in ids
    ]


def _select_final(summaries: list[dict[str, object]]) -> dict[str, object]:
    eligible = _eligible_sorted(summaries)
    if not eligible:
        return {"status": "FAIL", "reason": "no_finalist_passed_all_gates"}
    fastest = _as_float(eligible[0]["median_rescue_cells_per_second"])
    tied = [
        item
        for item in eligible
        if _as_float(item["median_rescue_cells_per_second"]) >= fastest * 0.97
    ]

    def key(item: dict[str, object]) -> tuple[int, int, int, int]:
        config = cast(dict[str, object], item["configuration"])
        return (
            _as_int(config["mlx_lanes"]) * _as_int(config["gpu_streams_per_lane"]),
            _as_int(config["mlx_lanes"]),
            max(cast(list[int], item["peak_summed_mlx_active_memory_bytes_by_repetition"])),
            -_as_int(config["cell_batch_size"]),
        )

    selected = min(tied, key=key)
    config = cast(dict[str, object], selected["configuration"])
    concurrency = _as_int(config["mlx_lanes"]) * _as_int(config["gpu_streams_per_lane"])
    gpu = selected.get("gpu_utilization_mean_percent")
    higher = [
        item
        for item in eligible
        if _as_int(cast(dict[str, object], item["configuration"])["mlx_lanes"])
        * _as_int(cast(dict[str, object], item["configuration"])["gpu_streams_per_lane"])
        > concurrency
    ]
    if gpu is not None and _as_float(gpu) >= 90:
        saturation = {"status": "PASS", "reason": "mean_gpu_at_least_ninety_percent"}
    elif higher:
        next_concurrency = min(
            _as_int(cast(dict[str, object], item["configuration"])["mlx_lanes"])
            * _as_int(cast(dict[str, object], item["configuration"])["gpu_streams_per_lane"])
            for item in higher
        )
        next_speed = max(
            _as_float(item["median_rescue_cells_per_second"])
            for item in higher
            if _as_int(cast(dict[str, object], item["configuration"])["mlx_lanes"])
            * _as_int(
                cast(dict[str, object], item["configuration"])["gpu_streams_per_lane"]
            )
            == next_concurrency
        )
        gain = (next_speed - _as_float(selected["median_rescue_cells_per_second"])) / _as_float(
            selected["median_rescue_cells_per_second"]
        )
        saturation = {
            "status": "PASS" if gain < 0.05 else "FAIL",
            "reason": "next_safe_concurrency_gain_below_five_percent",
            "next_throughput_gain_fraction": gain,
        }
    elif concurrency == 12:
        saturation = {"status": "PASS", "reason": "registered_twelve_stream_ceiling"}
    else:
        saturation = {"status": "FAIL", "reason": "saturation_not_demonstrated"}
    return {
        "status": saturation["status"],
        "selected_configuration": config,
        "selected_configuration_id": config["id"],
        "selected_executor_sha256": rescue_executor_code_identity(),
        "selected_solver_sha256": rescue_solver_code_identity(),
        "selected_median_rescue_cells_per_second": selected[
            "median_rescue_cells_per_second"
        ],
        "fastest_observed_rescue_cells_per_second": fastest,
        "tie_within_three_percent_applied": selected is not eligible[0],
        "saturation_gate": saturation,
    }


def run_rescue_executor_backtest(
    *, output_root: Path = RESCUE_EXECUTOR_BACKTEST_ROOT
) -> dict[str, object]:
    """Run the registered staged systems search without inspecting scientific outcomes."""

    output_root = reject_forbidden_runtime_path(output_root)
    if output_root != RESCUE_EXECUTOR_BACKTEST_ROOT:
        raise ValueError(f"rescue executor backtest must use {RESCUE_EXECUTOR_BACKTEST_ROOT}")
    registration = load_json(RESCUE_EXECUTOR_BACKTEST_REGISTRATION)
    inputs = cast(dict[str, object], registration["input_identity"])
    if inputs != {
        "rescue_cell_registry_sha256": sha256_file(RESCUE_CELL_REGISTRY),
        "rescue_unit_registry_sha256": sha256_file(RESCUE_UNIT_REGISTRY),
        "rescue_registration_verification_sha256": sha256_file(
            RESCUE_REGISTRATION_VERIFICATION
        ),
        "rescue_solver_code_sha256": rescue_solver_code_identity(),
        "rescue_executor_code_sha256": rescue_executor_code_identity(),
    }:
        raise ValueError("registered rescue executor input identity changed")
    representatives = cast(dict[str, object], registration["representative_units"])
    equivalence_selection = _selection(representatives["equivalence_rescue_unit_sequences"])
    timed_selection = _selection(representatives["timed_rescue_unit_sequences"])
    warmup_selection = _selection(representatives["warmup_rescue_unit_sequences"])
    reference_configuration = rescue_configuration_from_dict(
        cast(dict[str, object], registration["reference_configuration"])
    )
    search = cast(dict[str, object], registration["search"])
    equivalence_contract = cast(dict[str, object], registration["equivalence_contract"])
    tolerance = _as_float(equivalence_contract["float_absolute_tolerance"])
    output_root.mkdir(parents=True, exist_ok=True)
    request = {
        "schema_version": "veatic21_phase02_stage_a_rescue_executor_backtest_request_v1",
        "registration_sha256": sha256_file(RESCUE_EXECUTOR_BACKTEST_REGISTRATION),
        "solver_sha256": rescue_solver_code_identity(),
        "executor_sha256": rescue_executor_code_identity(),
        "reference_configuration": asdict(reference_configuration),
        "equivalence_selection": equivalence_selection.json_value(),
        "timed_selection": timed_selection.json_value(),
        "warmup_selection": warmup_selection.json_value(),
        "outer_test_scores_opened": False,
        "cortical_values_opened": False,
        "aggregation_or_pruning_performed": False,
    }
    request_path = output_root / "request.json"
    if request_path.exists() and load_json(request_path) != request:
        raise ValueError("rescue backtest request identity changed")
    _write_json(request_path, request)
    reference_root = output_root / "reference"
    _run_candidate_once(
        output_root=reference_root,
        configuration=reference_configuration,
        selection=equivalence_selection,
        warmup=None,
    )
    stages: dict[str, list[dict[str, object]]] = {}

    def evaluate(
        stage: str,
        configurations: list[RescueExecutorConfiguration],
        repetitions: int,
    ) -> list[dict[str, object]]:
        summaries: list[dict[str, object]] = []
        for configuration in configurations:
            root = output_root / "stages" / stage / configuration.id
            try:
                equivalence, resume_pass = _equivalence_and_resume(
                    root=root / "equivalence",
                    configuration=configuration,
                    reference_root=reference_root,
                    selection=equivalence_selection,
                    tolerance=tolerance,
                )
                timed_roots: list[Path] = []
                for repetition in range(repetitions):
                    timed_root = root / "timed" / f"repetition-{repetition:02d}"
                    _run_candidate_once(
                        output_root=timed_root,
                        configuration=configuration,
                        selection=timed_selection,
                        warmup=warmup_selection,
                    )
                    timed_roots.append(timed_root)
                summary = _timed_summary(
                    configuration=configuration,
                    roots=timed_roots,
                    equivalence=equivalence,
                    resume_pass=resume_pass,
                )
            except Exception as error:
                summary = cast(
                    dict[str, object],
                    {
                        "configuration": asdict(configuration),
                        "status": "ERROR",
                        "eligible_without_saturation_gate": False,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                        "outer_test_scores_opened": False,
                        "cortical_values_opened": False,
                        "aggregation_or_pruning_performed": False,
                    },
                )
            summaries.append(summary)
            _write_json(root / "candidate-state.json", summary)
            _write_json(output_root / "stage-summaries.json", {**stages, stage: summaries})
        stages[stage] = summaries
        _write_json(output_root / "stage-summaries.json", stages)
        return summaries

    batch_configs = [
        _config(lanes=3, streams=1, metrics=2, batch=batch, compiled=False, stage="s1")
        for batch in cast(list[int], search["stage_1_cell_batch_sizes"])
    ]
    stage1 = evaluate("stage-1-cell-batch", batch_configs, 1)
    best_batch = _as_int(
        cast(dict[str, object], _eligible_sorted(stage1)[0]["configuration"])["cell_batch_size"]
    )
    topology_configs = [
        _config(
            lanes=_as_int(pair[0]),
            streams=_as_int(pair[1]),
            metrics=2,
            batch=best_batch,
            compiled=False,
            stage="s2",
        )
        for pair in cast(list[list[int]], search["stage_2_safe_topologies"])
    ]
    stage2 = evaluate("stage-2-topology", topology_configs, 1)
    advanced_topologies = _advance(stage2, minimum=4, fraction=0.05)
    metric_configs: list[RescueExecutorConfiguration] = []
    for item in advanced_topologies:
        value = cast(dict[str, object], item["configuration"])
        for workers in cast(list[int], search["stage_3_metric_workers_per_lane"]):
            metric_configs.append(
                _config(
                    lanes=_as_int(value["mlx_lanes"]),
                    streams=_as_int(value["gpu_streams_per_lane"]),
                    metrics=workers,
                    batch=best_batch,
                    compiled=False,
                    stage="s3",
                )
            )
    stage3 = evaluate("stage-3-metric-workers", metric_configs, 1)
    advanced_metrics = _advance(stage3, minimum=4, fraction=0.05)
    compilation_configs: list[RescueExecutorConfiguration] = []
    for item in advanced_metrics:
        value = cast(dict[str, object], item["configuration"])
        for compiled in cast(list[bool], search["stage_4_compilation_values"]):
            compilation_configs.append(
                _config(
                    lanes=_as_int(value["mlx_lanes"]),
                    streams=_as_int(value["gpu_streams_per_lane"]),
                    metrics=_as_int(value["metric_workers_per_lane"]),
                    batch=best_batch,
                    compiled=compiled,
                    stage="s4",
                )
            )
    stage4 = evaluate("stage-4-compilation", compilation_configs, 1)
    finalists = _advance(stage4, minimum=8, fraction=0.03)
    finalist_configs = [
        replace(
            rescue_configuration_from_dict(cast(dict[str, object], item["configuration"])),
            id=_candidate_id(
                rescue_configuration_from_dict(cast(dict[str, object], item["configuration"])),
                "final",
            ),
        )
        for item in finalists
    ]
    final_summaries = evaluate(
        "stage-5-final-repetitions",
        finalist_configs,
        _as_int(search["timed_repetitions_for_finalists"]),
    )
    selection = _select_final(final_summaries)
    result = {
        "schema_version": "veatic21_phase02_stage_a_rescue_executor_backtest_result_v1",
        "status": "PASS" if selection["status"] == "PASS" else "FAIL",
        "registration_sha256": sha256_file(RESCUE_EXECUTOR_BACKTEST_REGISTRATION),
        "request_sha256": sha256_file(request_path),
        "solver_sha256": rescue_solver_code_identity(),
        "executor_sha256": rescue_executor_code_identity(),
        "stage_candidate_counts": {name: len(items) for name, items in stages.items()},
        "stages": stages,
        "selection": selection,
        "scientific_scores_used_for_executor_selection": False,
        "outer_test_scores_opened": False,
        "cortical_values_opened": False,
        "aggregation_or_pruning_performed": False,
    }
    _write_json(output_root / "result.json", result)
    return result
