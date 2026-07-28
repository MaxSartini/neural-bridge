"""Coordinated hardware-saturating executor for VEATIC Phase 02 Stage A."""

from __future__ import annotations

import hashlib
import importlib
import json
import multiprocessing
import os
import re
import subprocess
import threading
import time
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Protocol, cast

from neural_bridge.veatic21.contracts import (
    PHASE02_EXECUTOR_BACKTEST_REGISTRATION,
    PHASE02_REGISTRATION_ROOT,
    PHASE02_REGISTRATION_SHA256,
    REPOSITORY_ROOT,
)
from neural_bridge.veatic21.data import load_json, reject_forbidden_runtime_path, sha256_file
from neural_bridge.veatic21.phase00 import _write_json, _write_text
from neural_bridge.veatic21.phase02_stage_a import (
    REGULARIZATION_MULTIPLIERS,
    STAGE_A_FAMILIES,
    StageAPrepared,
    StageAWorkUnit,
    _append_ledger,
    _ledger_unit_ids,
    _load_inputs,
    _stage_a_code_identity,
    enumerate_stage_a_work_units,
    finalize_stage_a_unit,
    prepare_stage_a_unit,
    solve_stage_a_prepared,
)

EXECUTOR_SOURCE_FILES = (
    "contracts.py",
    "data.py",
    "phase00.py",
    "phase01.py",
    "phase02_features.py",
    "phase02_metrics.py",
    "phase02_registration.py",
    "phase02_stage_a.py",
    "phase02_stage_a_executor.py",
)
NUMERICAL_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


@dataclass(frozen=True)
class ExecutorConfiguration:
    id: str
    mlx_lanes: int
    gpu_streams_per_lane: int
    metric_workers_per_lane: int
    pair_cache: bool
    compiled_ridge_update_blocks: bool
    compiled_logistic_update_blocks: bool
    fast_metrics: bool
    pipeline_depth: int = 4

    def validate(self) -> None:
        if not (1 <= self.mlx_lanes <= 12):
            raise ValueError("mlx_lanes must be in 1..12")
        if not (1 <= self.gpu_streams_per_lane <= 12):
            raise ValueError("gpu_streams_per_lane must be in 1..12")
        if self.mlx_lanes * self.gpu_streams_per_lane > 12:
            raise ValueError("total concurrent GPU streams must not exceed host CPU cores")
        if not (1 <= self.metric_workers_per_lane <= 12):
            raise ValueError("metric_workers_per_lane must be in 1..12")
        if self.pipeline_depth < 1:
            raise ValueError("pipeline_depth must be positive")
        if self.pipeline_depth < self.gpu_streams_per_lane:
            raise ValueError("pipeline_depth must cover every GPU stream")


@dataclass(frozen=True)
class ExecutorRunSelection:
    sequence_ranges_inclusive: tuple[tuple[int, int], ...]

    def contains(self, sequence: int) -> bool:
        return any(start <= sequence <= end for start, end in self.sequence_ranges_inclusive)


class _Barrier(Protocol):
    def wait(self) -> object: ...


def executor_code_identity() -> str:
    digest = hashlib.sha256()
    package = REPOSITORY_ROOT / "src/neural_bridge/veatic21"
    for filename in EXECUTOR_SOURCE_FILES:
        path = package / filename
        digest.update(filename.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def pair_stage_a_units(
    units: tuple[StageAWorkUnit, ...] | list[StageAWorkUnit],
) -> tuple[tuple[StageAWorkUnit, StageAWorkUnit], ...]:
    if len(units) % 2:
        raise ValueError("Stage A unit selection must contain complete ridge/logistic pairs")
    pairs: list[tuple[StageAWorkUnit, StageAWorkUnit]] = []
    for index in range(0, len(units), 2):
        ridge, logistic = units[index : index + 2]
        left = asdict(ridge)
        right = asdict(logistic)
        left.pop("model_family")
        right.pop("model_family")
        left.pop("unit_id")
        right.pop("unit_id")
        left.pop("sequence")
        right.pop("sequence")
        if (
            left != right
            or ridge.model_family != "continuous_ridge"
            or logistic.model_family != "event_logistic_l2"
            or logistic.sequence != ridge.sequence + 1
        ):
            raise ValueError("Stage A work registry lost paired model-family ordering")
        pairs.append((ridge, logistic))
    return tuple(pairs)


def deterministic_pair_shards(
    pairs: tuple[tuple[StageAWorkUnit, StageAWorkUnit], ...], lanes: int
) -> tuple[tuple[tuple[StageAWorkUnit, StageAWorkUnit], ...], ...]:
    if lanes < 1:
        raise ValueError("lanes must be positive")
    shards: list[list[tuple[StageAWorkUnit, StageAWorkUnit]]] = [[] for _ in range(lanes)]
    for index, pair in enumerate(pairs):
        shards[index % lanes].append(pair)
    flattened = [unit.unit_id for shard in shards for pair in shard for unit in pair]
    expected = [unit.unit_id for pair in pairs for unit in pair]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(expected):
        raise ValueError("deterministic sharding lost or duplicated a Stage A unit")
    return tuple(tuple(shard) for shard in shards)


def _unit_from_dict(value: dict[str, object]) -> StageAWorkUnit:
    return StageAWorkUnit(
        unit_id=str(value["unit_id"]),
        sequence=int(cast(int, value["sequence"])),
        protocol=str(value["protocol"]),
        split_index=int(cast(int, value["split_index"])),
        repeat=(None if value["repeat"] is None else int(cast(int, value["repeat"]))),
        outer_fold=int(cast(int, value["outer_fold"])),
        inner_fold=int(cast(int, value["inner_fold"])),
        feature_form=str(value["feature_form"]),
        history_depth=int(cast(int, value["history_depth"])),
        model_family=str(value["model_family"]),
    )


def _configuration_from_dict(value: dict[str, object]) -> ExecutorConfiguration:
    expected = {
        "id",
        "mlx_lanes",
        "gpu_streams_per_lane",
        "metric_workers_per_lane",
        "pair_cache",
        "compiled_ridge_update_blocks",
        "compiled_logistic_update_blocks",
        "fast_metrics",
        "pipeline_depth",
    }
    if set(value) != expected:
        raise ValueError("executor configuration has an unexpected schema")
    boolean_fields = (
        "pair_cache",
        "compiled_ridge_update_blocks",
        "compiled_logistic_update_blocks",
        "fast_metrics",
    )
    if any(not isinstance(value[name], bool) for name in boolean_fields):
        raise TypeError("executor configuration boolean fields must be bool")
    integer_fields = (
        "mlx_lanes",
        "gpu_streams_per_lane",
        "metric_workers_per_lane",
        "pipeline_depth",
    )
    if any(
        not isinstance(value[name], int) or isinstance(value[name], bool) for name in integer_fields
    ):
        raise TypeError("executor configuration integer fields must be int")
    if not isinstance(value["id"], str):
        raise TypeError("executor configuration id must be str")
    return ExecutorConfiguration(
        id=value["id"],
        mlx_lanes=cast(int, value["mlx_lanes"]),
        gpu_streams_per_lane=cast(int, value["gpu_streams_per_lane"]),
        metric_workers_per_lane=cast(int, value["metric_workers_per_lane"]),
        pair_cache=cast(bool, value["pair_cache"]),
        compiled_ridge_update_blocks=cast(bool, value["compiled_ridge_update_blocks"]),
        compiled_logistic_update_blocks=cast(bool, value["compiled_logistic_update_blocks"]),
        fast_metrics=cast(bool, value["fast_metrics"]),
        pipeline_depth=cast(int, value["pipeline_depth"]),
    )


def _publish_completed_pair(
    pending: deque[Future[list[tuple[StageAWorkUnit, Path, Future[dict[str, object]]]]]],
    ledger_path: Path,
    ledger_ids: set[str],
) -> list[StageAWorkUnit]:
    completed: list[StageAWorkUnit] = []
    for unit, unit_path, finalizer in pending.popleft().result():
        result = finalizer.result()
        _write_json(unit_path, result)
        _append_ledger(ledger_path, unit_path, result)
        ledger_ids.add(unit.unit_id)
        completed.append(unit)
    return completed


def _run_executor_shard(
    *,
    output_root_text: str,
    shard_index: int,
    pair_values: list[list[dict[str, object]]],
    warmup_pair_values: list[list[dict[str, object]]],
    configuration_value: dict[str, object],
    executor_sha256: str,
    start_barrier: _Barrier | None,
) -> dict[str, object]:
    mx = importlib.import_module("mlx.core")

    configuration = _configuration_from_dict(configuration_value)
    configuration.validate()
    output_root = reject_forbidden_runtime_path(Path(output_root_text))
    unit_root = output_root / "units"
    unit_root.mkdir(parents=True, exist_ok=True)
    shard_root = output_root / "shards" / f"shard-{shard_index:02d}"
    shard_root.mkdir(parents=True, exist_ok=True)
    ledger_path = shard_root / "append-only-ledger.jsonl"
    ledger_ids = _ledger_unit_ids(ledger_path)
    pairs = [(_unit_from_dict(pair[0]), _unit_from_dict(pair[1])) for pair in pair_values]
    warmup_pairs = [
        (_unit_from_dict(pair[0]), _unit_from_dict(pair[1])) for pair in warmup_pair_values
    ]
    expected_units = [unit for pair in pairs for unit in pair]
    pending_pairs: list[tuple[StageAWorkUnit, tuple[StageAWorkUnit, ...]]] = []
    completed = 0
    for ridge, logistic in pairs:
        missing: list[StageAWorkUnit] = []
        for unit in (ridge, logistic):
            unit_path = unit_root / f"{unit.unit_id}.json"
            if unit_path.exists():
                stored = load_json(unit_path)
                provenance = cast(dict[str, object], stored.get("execution_provenance", {}))
                if (
                    cast(dict[str, object], stored.get("unit", {})).get("unit_id") != unit.unit_id
                    or stored.get("registration_sha256") != PHASE02_REGISTRATION_SHA256
                    or provenance.get("executor_sha256") != executor_sha256
                    or provenance.get("configuration_id") != configuration.id
                    or provenance.get("shard_index") != shard_index
                ):
                    raise ValueError(f"executor unit identity mismatch: {unit_path}")
                if unit.unit_id not in ledger_ids:
                    _append_ledger(ledger_path, unit_path, stored)
                    ledger_ids.add(unit.unit_id)
                completed += 1
            else:
                missing.append(unit)
        if missing:
            pending_pairs.append((ridge, tuple(missing)))

    inputs = _load_inputs() if pending_pairs or warmup_pairs else None
    warmup_started = time.monotonic()
    timed_started_ns = 0
    timed_finished_ns = 0
    last_unit: StageAWorkUnit | None = None
    pending: deque[Future[list[tuple[StageAWorkUnit, Path, Future[dict[str, object]]]]]] = deque()

    with (
        ThreadPoolExecutor(
            max_workers=configuration.metric_workers_per_lane,
            thread_name_prefix=f"metrics-{shard_index}",
        ) as metric_pool,
        ThreadPoolExecutor(
            max_workers=max(2, configuration.gpu_streams_per_lane * 2),
            thread_name_prefix=f"finalize-{shard_index}",
        ) as finalizer_pool,
        ThreadPoolExecutor(
            max_workers=configuration.gpu_streams_per_lane,
            thread_name_prefix=f"mlx-{shard_index}",
        ) as gpu_pool,
    ):
        thread_stream = mx.new_thread_local_stream(mx.gpu)

        def compute_pair(
            anchor: StageAWorkUnit,
            units: tuple[StageAWorkUnit, ...],
            *,
            warmup: bool,
        ) -> list[tuple[StageAWorkUnit, Path, Future[dict[str, object]]]]:
            if inputs is None:
                raise RuntimeError("executor inputs were not initialized")
            if configuration.pair_cache:
                shared_prepared: StageAPrepared | None = prepare_stage_a_unit(inputs, anchor)
            else:
                shared_prepared = None
            outputs: list[tuple[StageAWorkUnit, Path, Future[dict[str, object]]]] = []
            for unit in units:
                prepared = (
                    shared_prepared
                    if shared_prepared is not None
                    else prepare_stage_a_unit(inputs, unit)
                )
                with mx.stream(thread_stream):
                    compiled_update_blocks = (
                        configuration.compiled_ridge_update_blocks
                        if unit.model_family == "continuous_ridge"
                        else configuration.compiled_logistic_update_blocks
                    )
                    predictions, solver, solve_seconds = solve_stage_a_prepared(
                        inputs,
                        unit,
                        prepared,
                        compiled_update_blocks=compiled_update_blocks,
                    )
                provenance: dict[str, object] = {
                    "schema_version": "veatic21_phase02_stage_a_execution_v3",
                    "executor_sha256": executor_sha256,
                    "configuration_id": configuration.id,
                    "mlx_lanes": configuration.mlx_lanes,
                    "gpu_streams_per_lane": configuration.gpu_streams_per_lane,
                    "metric_workers_per_lane": configuration.metric_workers_per_lane,
                    "pair_cache": configuration.pair_cache,
                    "compiled_ridge_update_blocks": (configuration.compiled_ridge_update_blocks),
                    "compiled_logistic_update_blocks": (
                        configuration.compiled_logistic_update_blocks
                    ),
                    "fast_metrics": configuration.fast_metrics,
                    "pipeline_depth": configuration.pipeline_depth,
                    "shard_index": shard_index,
                    "warmup": warmup,
                    "outer_test_scores_opened": False,
                    "cortical_values_opened": False,
                }
                future = finalizer_pool.submit(
                    finalize_stage_a_unit,
                    inputs,
                    unit,
                    prepared,
                    predictions,
                    solver,
                    solve_seconds=solve_seconds,
                    metric_executor=metric_pool,
                    fast_metrics=configuration.fast_metrics,
                    execution_provenance=provenance,
                )
                outputs.append((unit, unit_root / f"{unit.unit_id}.json", future))
            return outputs

        warmup_futures = [
            gpu_pool.submit(compute_pair, ridge, (ridge, logistic), warmup=True)
            for ridge, logistic in warmup_pairs
        ]
        for future in warmup_futures:
            for _, _, finalizer in future.result():
                finalizer.result()
        warmup_seconds = time.monotonic() - warmup_started
        if start_barrier is not None:
            start_barrier.wait()
        mx.reset_peak_memory()
        timed_started_ns = time.monotonic_ns()

        for anchor, job_missing in pending_pairs:
            pending.append(gpu_pool.submit(compute_pair, anchor, job_missing, warmup=False))
            if len(pending) >= configuration.pipeline_depth:
                published = _publish_completed_pair(pending, ledger_path, ledger_ids)
                completed += len(published)
                last_unit = published[-1]
                _write_json(
                    shard_root / "shard-state.json",
                    {
                        "schema_version": "veatic21_phase02_stage_a_shard_state_v3",
                        "status": "RUNNING",
                        "shard_index": shard_index,
                        "executor_sha256": executor_sha256,
                        "configuration_id": configuration.id,
                        "work_units_total": len(expected_units),
                        "work_units_completed": completed,
                        "last_unit_id": last_unit.unit_id,
                        "outer_test_scores_opened": False,
                        "cortical_values_opened": False,
                    },
                )
        while pending:
            published = _publish_completed_pair(pending, ledger_path, ledger_ids)
            completed += len(published)
            last_unit = published[-1]
            _write_json(
                shard_root / "shard-state.json",
                {
                    "schema_version": "veatic21_phase02_stage_a_shard_state_v3",
                    "status": "RUNNING",
                    "shard_index": shard_index,
                    "executor_sha256": executor_sha256,
                    "configuration_id": configuration.id,
                    "work_units_total": len(expected_units),
                    "work_units_completed": completed,
                    "last_unit_id": last_unit.unit_id,
                    "outer_test_scores_opened": False,
                    "cortical_values_opened": False,
                },
            )
        timed_finished_ns = time.monotonic_ns()

    elapsed = (timed_finished_ns - timed_started_ns) / 1_000_000_000
    state = {
        "schema_version": "veatic21_phase02_stage_a_shard_state_v3",
        "status": "COMPLETE",
        "shard_index": shard_index,
        "executor_sha256": executor_sha256,
        "configuration_id": configuration.id,
        "work_units_total": len(expected_units),
        "work_units_completed": completed,
        "work_units_executed_this_call": sum(len(missing) for _, missing in pending_pairs),
        "warmup_pairs": len(warmup_pairs),
        "warmup_seconds": warmup_seconds,
        "timed_started_monotonic_ns": timed_started_ns,
        "timed_finished_monotonic_ns": timed_finished_ns,
        "elapsed_seconds": elapsed,
        "work_units_per_second": (
            sum(len(missing) for _, missing in pending_pairs) / elapsed if elapsed else None
        ),
        "peak_mlx_active_memory_bytes": int(mx.get_peak_memory()),
        "final_mlx_active_memory_bytes": int(mx.get_active_memory()),
        "final_mlx_cache_memory_bytes": int(mx.get_cache_memory()),
        "ledger_sha256": sha256_file(ledger_path),
        "outer_test_scores_opened": False,
        "cortical_values_opened": False,
    }
    _write_json(shard_root / "shard-state.json", state)
    return state


def _gpu_device_utilization() -> int | None:
    completed = subprocess.run(
        ["ioreg", "-r", "-d", "1", "-c", "AGXAccelerator"],
        check=False,
        capture_output=True,
        text=True,
    )
    match = re.search(r'"Device Utilization %"=(\d+)', completed.stdout)
    return int(match.group(1)) if match else None


def _system_memory_free_percent() -> int | None:
    completed = subprocess.run(
        ["/usr/bin/memory_pressure", "-Q"],
        check=False,
        capture_output=True,
        text=True,
    )
    match = re.search(r"System-wide memory free percentage:\s*(\d+)%", completed.stdout)
    return int(match.group(1)) if match else None


def _host_memory_bytes() -> int:
    completed = subprocess.run(
        ["/usr/sbin/sysctl", "-n", "hw.memsize"],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(completed.stdout.strip())


def _host_pressure_snapshot() -> dict[str, object]:
    commands = {
        "thermal": ["/usr/bin/pmset", "-g", "therm"],
        "power": ["/usr/bin/pmset", "-g", "custom"],
        "swap": ["/usr/sbin/sysctl", "vm.swapusage"],
    }
    snapshot: dict[str, object] = {"system_memory_free_percent": _system_memory_free_percent()}
    for name, command in commands.items():
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        snapshot[name] = completed.stdout.strip()
        snapshot[f"{name}_returncode"] = completed.returncode
    return snapshot


def _child_resource_usage(parent_pid: int) -> tuple[float, int]:
    completed = subprocess.run(
        ["ps", "-axo", "ppid=,%cpu=,rss="],
        check=True,
        capture_output=True,
        text=True,
    )
    cpu = 0.0
    rss_kib = 0
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) == 3 and int(fields[0]) == parent_pid:
            cpu += float(fields[1])
            rss_kib += int(fields[2])
    return cpu, rss_kib * 1024


def _resource_sampler(
    stop: threading.Event,
    parent_pid: int,
    samples: list[dict[str, float | int | None]],
    interval_seconds: float,
) -> None:
    while not stop.wait(interval_seconds):
        cpu, rss = _child_resource_usage(parent_pid)
        samples.append(
            {
                "monotonic_seconds": time.monotonic(),
                "gpu_device_utilization_percent": _gpu_device_utilization(),
                "worker_cpu_percent": cpu,
                "worker_rss_bytes": rss,
                "system_memory_free_percent": _system_memory_free_percent(),
            }
        )


def _resource_summary(
    samples: list[dict[str, float | int | None]],
) -> dict[str, object]:
    gpu = [
        float(value)
        for sample in samples
        if (value := sample["gpu_device_utilization_percent"]) is not None
    ]
    cpu = [float(sample["worker_cpu_percent"] or 0.0) for sample in samples]
    rss = [int(sample["worker_rss_bytes"] or 0) for sample in samples]
    memory_free = [
        float(value)
        for sample in samples
        if (value := sample["system_memory_free_percent"]) is not None
    ]
    host_memory = _host_memory_bytes()
    return {
        "sample_count": len(samples),
        "gpu_device_utilization_mean_percent": fmean(gpu) if gpu else None,
        "gpu_device_utilization_median_percent": median(gpu) if gpu else None,
        "gpu_device_utilization_max_percent": max(gpu) if gpu else None,
        "worker_cpu_mean_percent": fmean(cpu) if cpu else None,
        "worker_cpu_max_percent": max(cpu) if cpu else None,
        "worker_rss_peak_bytes": max(rss) if rss else None,
        "system_memory_free_min_percent": min(memory_free) if memory_free else None,
        "system_memory_free_median_percent": median(memory_free) if memory_free else None,
        "estimated_minimum_memory_headroom_bytes": (
            int(host_memory * min(memory_free) / 100) if memory_free else None
        ),
        "host_memory_bytes": host_memory,
    }


def _merge_shard_ledgers(
    output_root: Path,
    *,
    lane_count: int,
    expected_unit_ids: set[str],
) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    seen: set[str] = set()
    for shard_index in range(lane_count):
        path = output_root / "shards" / f"shard-{shard_index:02d}" / ("append-only-ledger.jsonl")
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                entry = json.loads(line)
                unit_id = entry.get("unit_id")
                if not isinstance(unit_id, str) or unit_id in seen:
                    raise ValueError(
                        f"duplicate/invalid executor ledger entry {path}:{line_number}"
                    )
                unit_path = Path(str(entry["unit_result_path"]))
                if sha256_file(unit_path) != entry["unit_result_sha256"]:
                    raise ValueError(f"executor ledger artifact hash mismatch: {unit_id}")
                seen.add(unit_id)
                entries.append(entry)
    if seen != expected_unit_ids:
        raise ValueError("canonical executor ledger does not exactly cover selected units")
    entries.sort(key=lambda entry: int(str(entry["unit_id"]).split("_", 1)[0]))
    canonical = output_root / "append-only-experiment-ledger.jsonl"
    _write_text(
        canonical,
        "".join(
            json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n" for entry in entries
        ),
    )
    return {
        "ledger_lines": len(entries),
        "unique_unit_ids": len(seen),
        "canonical_ledger_sha256": sha256_file(canonical),
    }


def run_hardware_saturated_executor(
    *,
    output_root: Path,
    configuration: ExecutorConfiguration,
    selection: ExecutorRunSelection,
    warmup_selection: ExecutorRunSelection | None = None,
    resource_interval_seconds: float = 1.0,
) -> dict[str, object]:
    """Run a deterministic sharded Stage A selection with measured host utilization."""

    configuration.validate()
    output_root = reject_forbidden_runtime_path(output_root)
    registration = load_json(PHASE02_REGISTRATION_ROOT / "experiment-registration.json")
    split_registry = load_json(PHASE02_REGISTRATION_ROOT / "split-registry.json")
    all_units = enumerate_stage_a_work_units(registration, split_registry)
    selected = tuple(unit for unit in all_units if selection.contains(unit.sequence))
    if not selected:
        raise ValueError("executor selection is empty")
    pairs = pair_stage_a_units(selected)
    if len(pairs) < configuration.mlx_lanes:
        raise ValueError("executor selection must provide at least one pair per MLX lane")
    shards = deterministic_pair_shards(pairs, configuration.mlx_lanes)
    warmup_selected = tuple(
        unit
        for unit in all_units
        if warmup_selection is not None and warmup_selection.contains(unit.sequence)
    )
    warmup_pairs = pair_stage_a_units(warmup_selected) if warmup_selected else ()
    warmup_shards = deterministic_pair_shards(warmup_pairs, configuration.mlx_lanes)
    executor_sha256 = executor_code_identity()
    backtest_registration_sha256 = sha256_file(PHASE02_EXECUTOR_BACKTEST_REGISTRATION)
    request = {
        "schema_version": "veatic21_phase02_stage_a_executor_request_v3",
        "scientific_registration_sha256": PHASE02_REGISTRATION_SHA256,
        "executor_backtest_registration_sha256": backtest_registration_sha256,
        "stage_a_solver_code_sha256": _stage_a_code_identity(),
        "executor_sha256": executor_sha256,
        "configuration": asdict(configuration),
        "selection": asdict(selection),
        "warmup_selection": (asdict(warmup_selection) if warmup_selection is not None else None),
        "warmup_work_units": len(warmup_selected),
        "work_units": len(selected),
        "configuration_evaluations": len(selected) * len(REGULARIZATION_MULTIPLIERS) * 21,
        "families": list(STAGE_A_FAMILIES),
        "outer_test_scores_opened": False,
        "cortical_values_opened": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    request_path = output_root / "request.json"
    if request_path.exists() and load_json(request_path) != request:
        raise ValueError("executor request identity changed")
    _write_json(request_path, request)
    registry = {
        "schema_version": "veatic21_phase02_stage_a_executor_registry_v3",
        "executor_sha256": executor_sha256,
        "units": [
            {**asdict(unit), "shard_index": pair_index % configuration.mlx_lanes}
            for pair_index, pair in enumerate(pairs)
            for unit in pair
        ],
    }
    registry_path = output_root / "work-unit-registry.json"
    if registry_path.exists() and load_json(registry_path) != registry:
        raise ValueError("executor work-unit registry identity changed")
    _write_json(registry_path, registry)
    _write_json(
        output_root / "run-state.json",
        {
            "schema_version": "veatic21_phase02_stage_a_executor_state_v3",
            "status": "RUNNING",
            "executor_sha256": executor_sha256,
            "configuration_id": configuration.id,
            "work_units_total": len(selected),
            "work_units_completed": 0,
            "outer_test_scores_opened": False,
            "cortical_values_opened": False,
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
        name="stage-a-resource-sampler",
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
                        _run_executor_shard,
                        output_root_text=str(output_root),
                        shard_index=shard_index,
                        pair_values=[[asdict(pair[0]), asdict(pair[1])] for pair in shard],
                        warmup_pair_values=[
                            [asdict(pair[0]), asdict(pair[1])]
                            for pair in warmup_shards[shard_index]
                        ],
                        configuration_value=asdict(configuration),
                        executor_sha256=executor_sha256,
                        start_barrier=start_barrier,
                    )
                    for shard_index, shard in enumerate(shards)
                ]
                while not all(future.done() for future in futures):
                    time.sleep(1.0)
                    completed = 0
                    for shard_index in range(configuration.mlx_lanes):
                        path = (
                            output_root
                            / "shards"
                            / f"shard-{shard_index:02d}"
                            / ("shard-state.json")
                        )
                        if path.exists():
                            completed += int(load_json(path).get("work_units_completed", 0))
                    _write_json(
                        output_root / "run-state.json",
                        {
                            "schema_version": "veatic21_phase02_stage_a_executor_state_v3",
                            "status": "RUNNING",
                            "executor_sha256": executor_sha256,
                            "configuration_id": configuration.id,
                            "work_units_total": len(selected),
                            "work_units_completed": completed,
                            "outer_test_scores_opened": False,
                            "cortical_values_opened": False,
                        },
                    )
                states = [future.result() for future in futures]
    finally:
        stop_sampler.set()
        sampler.join(timeout=5)
        for name, value in previous_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    timed_started_ns = min(cast(int, state["timed_started_monotonic_ns"]) for state in states)
    timed_finished_ns = max(cast(int, state["timed_finished_monotonic_ns"]) for state in states)
    elapsed = (timed_finished_ns - timed_started_ns) / 1_000_000_000
    timed_samples = [
        sample
        for sample in samples
        if timed_started_ns / 1_000_000_000
        <= float(sample["monotonic_seconds"] or 0.0)
        <= timed_finished_ns / 1_000_000_000
    ]
    merge = _merge_shard_ledgers(
        output_root,
        lane_count=configuration.mlx_lanes,
        expected_unit_ids={unit.unit_id for unit in selected},
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
        "schema_version": "veatic21_phase02_stage_a_executor_state_v3",
        "status": "COMPLETE",
        "executor_sha256": executor_sha256,
        "configuration_id": configuration.id,
        "work_units_total": len(selected),
        "work_units_completed": len(selected),
        "configuration_evaluations": len(selected) * len(REGULARIZATION_MULTIPLIERS) * 21,
        "elapsed_seconds": elapsed,
        "work_units_per_second": len(selected) / elapsed,
        "timed_started_monotonic_ns": timed_started_ns,
        "timed_finished_monotonic_ns": timed_finished_ns,
        "shards": states,
        "resource_summary": resources,
        **merge,
        "request_sha256": sha256_file(request_path),
        "work_unit_registry_sha256": sha256_file(registry_path),
        "outer_test_scores_opened": False,
        "cortical_values_opened": False,
    }
    _write_json(output_root / "run-state.json", final)
    return final
