"""Hardware-saturating executor for the sparse VEATIC Stage A rescue registry."""

from __future__ import annotations

import hashlib
import importlib
import json
import multiprocessing
import os
import threading
import time
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, cast

from neural_bridge.veatic21.contracts import REPOSITORY_ROOT
from neural_bridge.veatic21.data import load_json, reject_forbidden_runtime_path, sha256_file
from neural_bridge.veatic21.phase00 import _write_json, _write_text
from neural_bridge.veatic21.phase02_features import feature_names
from neural_bridge.veatic21.phase02_stage_a_executor import (
    NUMERICAL_THREAD_ENVIRONMENT,
    _host_pressure_snapshot,
    _resource_sampler,
    _resource_summary,
)
from neural_bridge.veatic21.phase02_stage_a_rescue import (
    RescueUnit,
    finalize_rescue_unit,
    load_rescue_registry,
    load_rescue_runtime_inputs,
    prepare_registered_rescue_unit,
    rescue_solver_code_identity,
    solve_rescue_unit,
)
from neural_bridge.veatic21.phase02_stage_a_rescue_registration import (
    RESCUE_CELL_REGISTRY,
    RESCUE_REGISTRATION_VERIFICATION,
    RESCUE_UNIT_REGISTRY,
)

RESCUE_EXECUTOR_SOURCE_FILES = (
    "contracts.py",
    "data.py",
    "phase00.py",
    "phase01.py",
    "phase02_features.py",
    "phase02_metrics.py",
    "phase02_registration.py",
    "phase02_stage_a.py",
    "phase02_stage_a_rescue_registration.py",
    "phase02_stage_a_rescue.py",
    "phase02_stage_a_rescue_executor.py",
)


@dataclass(frozen=True)
class RescueExecutorConfiguration:
    id: str
    mlx_lanes: int
    gpu_streams_per_lane: int
    metric_workers_per_lane: int
    cell_batch_size: int
    compiled_update_blocks: bool
    fast_metrics: bool
    pipeline_depth: int = 4

    def validate(self) -> None:
        if not 1 <= self.mlx_lanes <= 12:
            raise ValueError("mlx_lanes must be in 1..12")
        if not 1 <= self.gpu_streams_per_lane <= 12:
            raise ValueError("gpu_streams_per_lane must be in 1..12")
        if self.mlx_lanes * self.gpu_streams_per_lane > 12:
            raise ValueError("total concurrent Metal streams must not exceed host CPU cores")
        if not 1 <= self.metric_workers_per_lane <= 12:
            raise ValueError("metric_workers_per_lane must be in 1..12")
        if self.cell_batch_size not in {1, 4, 8, 16, 32, 64}:
            raise ValueError("cell_batch_size is outside the registered safe set")
        if self.pipeline_depth < self.gpu_streams_per_lane:
            raise ValueError("pipeline depth must cover all Metal streams")


@dataclass(frozen=True)
class RescueRunSelection:
    rescue_unit_sequences: tuple[int, ...]

    def validate(self, unit_count: int) -> None:
        if not self.rescue_unit_sequences:
            raise ValueError("rescue selection is empty")
        if tuple(sorted(set(self.rescue_unit_sequences))) != self.rescue_unit_sequences:
            raise ValueError("rescue selection must be sorted and unique")
        if self.rescue_unit_sequences[0] < 0 or self.rescue_unit_sequences[-1] >= unit_count:
            raise ValueError("rescue selection is outside the frozen registry")

    def json_value(self) -> dict[str, object]:
        return {"rescue_unit_sequences": list(self.rescue_unit_sequences)}


class _Barrier(Protocol):
    def wait(self) -> object: ...


def rescue_executor_code_identity() -> str:
    digest = hashlib.sha256()
    package = REPOSITORY_ROOT / "src/neural_bridge/veatic21"
    for filename in RESCUE_EXECUTOR_SOURCE_FILES:
        path = package / filename
        digest.update(filename.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def rescue_configuration_from_dict(value: dict[str, object]) -> RescueExecutorConfiguration:
    expected = {
        "id",
        "mlx_lanes",
        "gpu_streams_per_lane",
        "metric_workers_per_lane",
        "cell_batch_size",
        "compiled_update_blocks",
        "fast_metrics",
        "pipeline_depth",
    }
    if set(value) != expected:
        raise ValueError("rescue executor configuration schema changed")
    for name in ("compiled_update_blocks", "fast_metrics"):
        if not isinstance(value[name], bool):
            raise TypeError(f"{name} must be bool")
    for name in (
        "mlx_lanes",
        "gpu_streams_per_lane",
        "metric_workers_per_lane",
        "cell_batch_size",
        "pipeline_depth",
    ):
        if not isinstance(value[name], int) or isinstance(value[name], bool):
            raise TypeError(f"{name} must be int")
    if not isinstance(value["id"], str):
        raise TypeError("configuration id must be str")
    result = RescueExecutorConfiguration(
        id=value["id"],
        mlx_lanes=cast(int, value["mlx_lanes"]),
        gpu_streams_per_lane=cast(int, value["gpu_streams_per_lane"]),
        metric_workers_per_lane=cast(int, value["metric_workers_per_lane"]),
        cell_batch_size=cast(int, value["cell_batch_size"]),
        compiled_update_blocks=cast(bool, value["compiled_update_blocks"]),
        fast_metrics=cast(bool, value["fast_metrics"]),
        pipeline_depth=cast(int, value["pipeline_depth"]),
    )
    result.validate()
    return result


def deterministic_weighted_shards(
    units: tuple[RescueUnit, ...], lanes: int
) -> tuple[tuple[RescueUnit, ...], ...]:
    """Balance immutable rescue units by registered cell-budget work, deterministically."""

    if lanes < 1 or len(units) < lanes:
        raise ValueError("each MLX lane needs at least one rescue unit")
    shards: list[list[RescueUnit]] = [[] for _ in range(lanes)]
    loads = [0] * lanes

    for unit in sorted(
        units, key=lambda item: (-rescue_unit_work_weight(item), item.rescue_unit_sequence)
    ):
        lane = min(range(lanes), key=lambda index: (loads[index], index))
        shards[lane].append(unit)
        loads[lane] += rescue_unit_work_weight(unit)
    for shard in shards:
        shard.sort(key=lambda item: item.rescue_unit_sequence)
    actual = [unit.rescue_unit_sequence for shard in shards for unit in shard]
    expected = [unit.rescue_unit_sequence for unit in units]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError("weighted rescue sharding lost or duplicated a unit")
    return tuple(tuple(shard) for shard in shards)


def rescue_unit_work_weight(unit: RescueUnit) -> int:
    """Proxy MLX work as rows × feature width × registered maximum cell updates."""

    cell = unit.cells[0]
    feature_count_with_intercept = len(feature_names(cell.feature_form, cell.history_depth)) + 1
    solve_work = sum(
        item.train_rows * item.rescue_maximum_budget * feature_count_with_intercept
        for item in unit.cells
    )
    preparation_work = 20_657 * feature_count_with_intercept
    return solve_work + preparation_work


def _unit_path(root: Path, unit: RescueUnit) -> Path:
    return root / "units" / f"{unit.rescue_unit_sequence:05d}_{unit.original_unit_id}.json"


def _append_rescue_ledger(path: Path, unit_path: Path, result: dict[str, object]) -> None:
    rescue_unit = cast(dict[str, object], result["rescue_unit"])
    entry = {
        "schema_version": "veatic21_phase02_stage_a_rescue_ledger_v1",
        "rescue_unit_sequence": rescue_unit["rescue_unit_sequence"],
        "original_unit_id": rescue_unit["original_unit_id"],
        "original_unit_result_sha256": rescue_unit["original_unit_result_sha256"],
        "cell_count": result["cell_count"],
        "unit_result_path": str(unit_path),
        "unit_result_sha256": sha256_file(unit_path),
        "outer_test_scores_opened": False,
        "cortical_values_opened": False,
        "aggregation_or_pruning_performed": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _ledger_sequences(path: Path) -> set[int]:
    if not path.exists():
        return set()
    result: set[int] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            sequence = value.get("rescue_unit_sequence")
            if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence in result:
                raise ValueError(f"invalid or duplicate rescue ledger row: {path}")
            result.add(sequence)
    return result


def _validate_stored_unit(
    path: Path,
    unit: RescueUnit,
    *,
    executor_sha256: str,
    configuration: RescueExecutorConfiguration,
    shard_index: int,
) -> dict[str, object]:
    stored = load_json(path)
    rescue_value = cast(dict[str, object], stored.get("rescue_unit", {}))
    provenance = cast(dict[str, object], stored.get("execution_provenance", {}))
    if (
        rescue_value.get("rescue_unit_sequence") != unit.rescue_unit_sequence
        or rescue_value.get("original_unit_id") != unit.original_unit_id
        or rescue_value.get("original_unit_result_sha256") != unit.original_unit_result_sha256
        or stored.get("rescue_solver_code_sha256") != rescue_solver_code_identity()
        or provenance.get("executor_sha256") != executor_sha256
        or provenance.get("configuration_id") != configuration.id
        or provenance.get("shard_index") != shard_index
    ):
        raise ValueError(f"stored rescue unit identity mismatch: {path}")
    return stored


def _run_rescue_shard(
    *,
    output_root_text: str,
    shard_index: int,
    rescue_unit_sequences: list[int],
    warmup_sequences: list[int],
    configuration_value: dict[str, object],
    executor_sha256: str,
    start_barrier: _Barrier | None,
) -> dict[str, object]:
    mx = importlib.import_module("mlx.core")
    configuration = rescue_configuration_from_dict(configuration_value)
    output_root = reject_forbidden_runtime_path(Path(output_root_text))
    unit_root = output_root / "units"
    unit_root.mkdir(parents=True, exist_ok=True)
    shard_root = output_root / "shards" / f"shard-{shard_index:02d}"
    shard_root.mkdir(parents=True, exist_ok=True)
    ledger_path = shard_root / "append-only-ledger.jsonl"
    ledger_sequences = _ledger_sequences(ledger_path)

    registry = load_rescue_registry()
    by_sequence = {unit.rescue_unit_sequence: unit for unit in registry}
    units = tuple(by_sequence[sequence] for sequence in rescue_unit_sequences)
    warmup_units = tuple(by_sequence[sequence] for sequence in warmup_sequences)
    missing: list[RescueUnit] = []
    completed = 0
    for unit in units:
        path = _unit_path(output_root, unit)
        if path.exists():
            stored = _validate_stored_unit(
                path,
                unit,
                executor_sha256=executor_sha256,
                configuration=configuration,
                shard_index=shard_index,
            )
            if unit.rescue_unit_sequence not in ledger_sequences:
                _append_rescue_ledger(ledger_path, path, stored)
                ledger_sequences.add(unit.rescue_unit_sequence)
            completed += 1
        else:
            missing.append(unit)

    inputs, stage_a_by_sequence = load_rescue_runtime_inputs()
    thread_stream = mx.new_thread_local_stream(mx.gpu)

    def compute(
        unit: RescueUnit,
        *,
        warmup: bool,
        metric_pool: ThreadPoolExecutor,
    ) -> tuple[RescueUnit, dict[str, object]]:
        stage_a_unit, prepared = prepare_registered_rescue_unit(
            inputs, stage_a_by_sequence, unit
        )
        with mx.stream(thread_stream):
            predictions, diagnostics, solve_seconds = solve_rescue_unit(
                inputs,
                stage_a_unit,
                unit,
                prepared,
                cell_batch_size=configuration.cell_batch_size,
                compiled_update_blocks=configuration.compiled_update_blocks,
            )
        provenance: dict[str, object] = {
            "schema_version": "veatic21_phase02_stage_a_rescue_execution_v1",
            "executor_sha256": executor_sha256,
            "configuration_id": configuration.id,
            "mlx_lanes": configuration.mlx_lanes,
            "gpu_streams_per_lane": configuration.gpu_streams_per_lane,
            "metric_workers_per_lane": configuration.metric_workers_per_lane,
            "cell_batch_size": configuration.cell_batch_size,
            "compiled_update_blocks": configuration.compiled_update_blocks,
            "fast_metrics": configuration.fast_metrics,
            "pipeline_depth": configuration.pipeline_depth,
            "shard_index": shard_index,
            "warmup": warmup,
            "outer_test_scores_opened": False,
            "cortical_values_opened": False,
            "aggregation_or_pruning_performed": False,
        }
        result = finalize_rescue_unit(
            inputs,
            stage_a_unit,
            unit,
            prepared,
            predictions,
            diagnostics,
            solve_seconds=solve_seconds,
            metric_executor=metric_pool,
            fast_metrics=configuration.fast_metrics,
            execution_provenance=provenance,
        )
        return unit, result

    warmup_started = time.monotonic()
    pending: deque[Future[tuple[RescueUnit, dict[str, object]]]] = deque()
    timed_started_ns = 0
    timed_finished_ns = 0
    last_sequence: int | None = None
    with (
        ThreadPoolExecutor(
            max_workers=configuration.metric_workers_per_lane,
            thread_name_prefix=f"rescue-metrics-{shard_index}",
        ) as metric_pool,
        ThreadPoolExecutor(
            max_workers=configuration.gpu_streams_per_lane,
            thread_name_prefix=f"rescue-mlx-{shard_index}",
        ) as gpu_pool,
    ):
        for unit in warmup_units:
            compute(unit, warmup=True, metric_pool=metric_pool)
        warmup_seconds = time.monotonic() - warmup_started
        if start_barrier is not None:
            start_barrier.wait()
        mx.reset_peak_memory()
        timed_started_ns = time.monotonic_ns()
        for unit in missing:
            pending.append(gpu_pool.submit(compute, unit, warmup=False, metric_pool=metric_pool))
            if len(pending) >= configuration.pipeline_depth:
                finished_unit, result = pending.popleft().result()
                path = _unit_path(output_root, finished_unit)
                _write_json(path, result)
                _append_rescue_ledger(ledger_path, path, result)
                ledger_sequences.add(finished_unit.rescue_unit_sequence)
                completed += 1
                last_sequence = finished_unit.rescue_unit_sequence
                _write_shard_progress(
                    shard_root,
                    shard_index=shard_index,
                    executor_sha256=executor_sha256,
                    configuration_id=configuration.id,
                    total=len(units),
                    completed=completed,
                    last_sequence=last_sequence,
                )
        while pending:
            finished_unit, result = pending.popleft().result()
            path = _unit_path(output_root, finished_unit)
            _write_json(path, result)
            _append_rescue_ledger(ledger_path, path, result)
            ledger_sequences.add(finished_unit.rescue_unit_sequence)
            completed += 1
            last_sequence = finished_unit.rescue_unit_sequence
            _write_shard_progress(
                shard_root,
                shard_index=shard_index,
                executor_sha256=executor_sha256,
                configuration_id=configuration.id,
                total=len(units),
                completed=completed,
                last_sequence=last_sequence,
            )
        timed_finished_ns = time.monotonic_ns()

    elapsed = (timed_finished_ns - timed_started_ns) / 1_000_000_000
    state: dict[str, object] = {
        "schema_version": "veatic21_phase02_stage_a_rescue_shard_state_v1",
        "status": "COMPLETE",
        "shard_index": shard_index,
        "executor_sha256": executor_sha256,
        "configuration_id": configuration.id,
        "rescue_units_total": len(units),
        "rescue_units_completed": completed,
        "rescue_units_executed_this_call": len(missing),
        "rescue_cells": sum(len(unit.cells) for unit in units),
        "warmup_units": len(warmup_units),
        "warmup_seconds": warmup_seconds,
        "timed_started_monotonic_ns": timed_started_ns,
        "timed_finished_monotonic_ns": timed_finished_ns,
        "elapsed_seconds": elapsed,
        "rescue_units_per_second": len(missing) / elapsed if elapsed else None,
        "rescue_cells_per_second": (
            sum(len(unit.cells) for unit in missing) / elapsed if elapsed else None
        ),
        "peak_mlx_active_memory_bytes": int(mx.get_peak_memory()),
        "final_mlx_active_memory_bytes": int(mx.get_active_memory()),
        "final_mlx_cache_memory_bytes": int(mx.get_cache_memory()),
        "ledger_sha256": sha256_file(ledger_path),
        "outer_test_scores_opened": False,
        "cortical_values_opened": False,
        "aggregation_or_pruning_performed": False,
    }
    _write_json(shard_root / "shard-state.json", state)
    return state


def _write_shard_progress(
    shard_root: Path,
    *,
    shard_index: int,
    executor_sha256: str,
    configuration_id: str,
    total: int,
    completed: int,
    last_sequence: int | None,
) -> None:
    _write_json(
        shard_root / "shard-state.json",
        {
            "schema_version": "veatic21_phase02_stage_a_rescue_shard_state_v1",
            "status": "RUNNING",
            "shard_index": shard_index,
            "executor_sha256": executor_sha256,
            "configuration_id": configuration_id,
            "rescue_units_total": total,
            "rescue_units_completed": completed,
            "last_rescue_unit_sequence": last_sequence,
            "outer_test_scores_opened": False,
            "cortical_values_opened": False,
            "aggregation_or_pruning_performed": False,
        },
    )


def _merge_rescue_ledgers(
    output_root: Path,
    *,
    lane_count: int,
    expected: tuple[RescueUnit, ...],
) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    seen_units: set[int] = set()
    seen_cells: set[str] = set()
    for shard_index in range(lane_count):
        path = output_root / "shards" / f"shard-{shard_index:02d}" / "append-only-ledger.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                entry = cast(dict[str, object], json.loads(line))
                sequence = entry.get("rescue_unit_sequence")
                if not isinstance(sequence, int) or sequence in seen_units:
                    raise ValueError(f"invalid/duplicate rescue ledger row {path}:{line_number}")
                unit_path = Path(str(entry["unit_result_path"]))
                if sha256_file(unit_path) != entry["unit_result_sha256"]:
                    raise ValueError(f"rescue unit artifact hash mismatch: {sequence}")
                result = load_json(unit_path)
                records = cast(list[dict[str, object]], result["records"])
                identities = [str(record["rescue_cell_identity_sha256"]) for record in records]
                if len(identities) != len(set(identities)) or seen_cells.intersection(identities):
                    raise ValueError("duplicate rescue cell in canonical merge")
                seen_cells.update(identities)
                seen_units.add(sequence)
                entries.append(entry)
    expected_units = {unit.rescue_unit_sequence for unit in expected}
    expected_cells = {
        cell.rescue_cell_identity_sha256 for unit in expected for cell in unit.cells
    }
    if seen_units != expected_units or seen_cells != expected_cells:
        raise ValueError("canonical rescue merge does not exactly cover the selection")
    entries.sort(key=lambda entry: int(cast(int, entry["rescue_unit_sequence"])))
    path = output_root / "append-only-experiment-ledger.jsonl"
    _write_text(
        path,
        "".join(
            json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n"
            for entry in entries
        ),
    )
    return {
        "ledger_lines": len(entries),
        "unique_rescue_units": len(seen_units),
        "unique_rescue_cells": len(seen_cells),
        "canonical_ledger_sha256": sha256_file(path),
    }


def run_rescue_executor(
    *,
    output_root: Path,
    configuration: RescueExecutorConfiguration,
    selection: RescueRunSelection,
    warmup_selection: RescueRunSelection | None = None,
    resource_interval_seconds: float = 0.5,
) -> dict[str, object]:
    """Execute a deterministic selection with measured CPU, GPU, memory, and resume evidence."""

    configuration.validate()
    output_root = reject_forbidden_runtime_path(output_root)
    registry = load_rescue_registry()
    selection.validate(len(registry))
    selected = tuple(registry[index] for index in selection.rescue_unit_sequences)
    warmup_units: tuple[RescueUnit, ...] = ()
    if warmup_selection is not None:
        warmup_selection.validate(len(registry))
        warmup_units = tuple(registry[index] for index in warmup_selection.rescue_unit_sequences)
    shards = deterministic_weighted_shards(selected, configuration.mlx_lanes)
    warmup_shards = (
        deterministic_weighted_shards(warmup_units, configuration.mlx_lanes)
        if warmup_units
        else tuple(() for _ in range(configuration.mlx_lanes))
    )
    executor_sha256 = rescue_executor_code_identity()
    request = {
        "schema_version": "veatic21_phase02_stage_a_rescue_executor_request_v1",
        "rescue_solver_code_sha256": rescue_solver_code_identity(),
        "executor_sha256": executor_sha256,
        "rescue_cell_registry_sha256": sha256_file(RESCUE_CELL_REGISTRY),
        "rescue_unit_registry_sha256": sha256_file(RESCUE_UNIT_REGISTRY),
        "rescue_registration_verification_sha256": sha256_file(
            RESCUE_REGISTRATION_VERIFICATION
        ),
        "configuration": asdict(configuration),
        "selection": selection.json_value(),
        "warmup_selection": (
            warmup_selection.json_value() if warmup_selection is not None else None
        ),
        "rescue_units": len(selected),
        "rescue_cells": sum(len(unit.cells) for unit in selected),
        "outer_test_scores_opened": False,
        "cortical_values_opened": False,
        "aggregation_or_pruning_performed": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    request_path = output_root / "request.json"
    if request_path.exists() and load_json(request_path) != request:
        raise ValueError("rescue executor request identity changed")
    _write_json(request_path, request)
    work_registry = {
        "schema_version": "veatic21_phase02_stage_a_rescue_work_registry_v1",
        "executor_sha256": executor_sha256,
        "units": [
            {
                "rescue_unit_sequence": unit.rescue_unit_sequence,
                "original_unit_id": unit.original_unit_id,
                "original_unit_result_sha256": unit.original_unit_result_sha256,
                "cell_count": len(unit.cells),
                "shard_index": shard_index,
            }
            for shard_index, shard in enumerate(shards)
            for unit in shard
        ],
    }
    work_registry_path = output_root / "work-unit-registry.json"
    if work_registry_path.exists() and load_json(work_registry_path) != work_registry:
        raise ValueError("rescue work registry identity changed")
    _write_json(work_registry_path, work_registry)
    _write_json(
        output_root / "run-state.json",
        {
            "schema_version": "veatic21_phase02_stage_a_rescue_executor_state_v1",
            "status": "RUNNING",
            "executor_sha256": executor_sha256,
            "configuration_id": configuration.id,
            "rescue_units_total": len(selected),
            "rescue_units_completed": 0,
            "outer_test_scores_opened": False,
            "cortical_values_opened": False,
            "aggregation_or_pruning_performed": False,
        },
    )

    previous_environment = {name: os.environ.get(name) for name in NUMERICAL_THREAD_ENVIRONMENT}
    os.environ.update(NUMERICAL_THREAD_ENVIRONMENT)
    samples: list[dict[str, float | int | None]] = []
    stop_sampler = threading.Event()
    sampler = threading.Thread(
        target=_resource_sampler,
        args=(stop_sampler, os.getpid(), samples, resource_interval_seconds),
        daemon=True,
        name="rescue-resource-sampler",
    )
    pressure_before = _host_pressure_snapshot()
    sampler.start()
    states: list[dict[str, object]] = []
    try:
        context = multiprocessing.get_context("spawn")
        with context.Manager() as manager:
            start_barrier = cast(_Barrier, manager.Barrier(configuration.mlx_lanes))
            with ProcessPoolExecutor(
                max_workers=configuration.mlx_lanes, mp_context=context
            ) as pool:
                futures = [
                    pool.submit(
                        _run_rescue_shard,
                        output_root_text=str(output_root),
                        shard_index=shard_index,
                        rescue_unit_sequences=[unit.rescue_unit_sequence for unit in shard],
                        warmup_sequences=[
                            unit.rescue_unit_sequence for unit in warmup_shards[shard_index]
                        ],
                        configuration_value=asdict(configuration),
                        executor_sha256=executor_sha256,
                        start_barrier=start_barrier,
                    )
                    for shard_index, shard in enumerate(shards)
                ]
                while not all(future.done() for future in futures):
                    time.sleep(0.5)
                states = [future.result() for future in futures]
    finally:
        stop_sampler.set()
        sampler.join(timeout=5)
        for name, value in previous_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    timed_started_ns = min(int(cast(int, state["timed_started_monotonic_ns"])) for state in states)
    timed_finished_ns = max(
        int(cast(int, state["timed_finished_monotonic_ns"])) for state in states
    )
    elapsed = (timed_finished_ns - timed_started_ns) / 1_000_000_000
    timed_samples = [
        sample
        for sample in samples
        if timed_started_ns / 1_000_000_000
        <= float(sample["monotonic_seconds"] or 0)
        <= timed_finished_ns / 1_000_000_000
    ]
    merge = _merge_rescue_ledgers(
        output_root,
        lane_count=configuration.mlx_lanes,
        expected=selected,
    )
    resources = _resource_summary(timed_samples)
    resources["pressure_before"] = pressure_before
    resources["pressure_after"] = _host_pressure_snapshot()
    _write_text(
        output_root / "resource-samples.jsonl",
        "".join(
            json.dumps(sample, sort_keys=True, separators=(",", ":")) + "\n"
            for sample in timed_samples
        ),
    )
    _write_json(output_root / "resource-summary.json", resources)
    final = {
        "schema_version": "veatic21_phase02_stage_a_rescue_executor_state_v1",
        "status": "COMPLETE",
        "executor_sha256": executor_sha256,
        "configuration_id": configuration.id,
        "rescue_units_total": len(selected),
        "rescue_units_completed": len(selected),
        "rescue_cells": sum(len(unit.cells) for unit in selected),
        "elapsed_seconds": elapsed,
        "rescue_units_per_second": len(selected) / elapsed,
        "rescue_cells_per_second": sum(len(unit.cells) for unit in selected) / elapsed,
        "timed_started_monotonic_ns": timed_started_ns,
        "timed_finished_monotonic_ns": timed_finished_ns,
        "shards": states,
        "resource_summary": resources,
        **merge,
        "request_sha256": sha256_file(request_path),
        "work_unit_registry_sha256": sha256_file(work_registry_path),
        "outer_test_scores_opened": False,
        "cortical_values_opened": False,
        "aggregation_or_pruning_performed": False,
    }
    _write_json(output_root / "run-state.json", final)
    return final
