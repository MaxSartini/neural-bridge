"""Hardware backtest and resumable executor for the frozen VEATIC Stage B registry."""

from __future__ import annotations

import gzip
import hashlib
import importlib
import json
import multiprocessing
import os
import statistics
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from neural_bridge.veatic21.contracts import REPOSITORY_ROOT
from neural_bridge.veatic21.data import load_json, sha256_file
from neural_bridge.veatic21.phase00 import _write_json, canonical_json_bytes
from neural_bridge.veatic21.phase02_stage_a import _load_inputs
from neural_bridge.veatic21.phase02_stage_a_executor import (
    _gpu_device_utilization,
    _host_pressure_snapshot,
    _resource_summary,
    _system_memory_free_percent,
)
from neural_bridge.veatic21.phase02_stage_b import (
    EXPECTED_AGGREGATION_VERIFICATION_SHA256,
    EXPECTED_WORK_REGISTRY_SHA256,
    STAGE_B_BACKTEST_ROOT,
    STAGE_B_EXECUTION_REGISTRATION,
    STAGE_B_MAIN_ROOT,
    PreparedStageBUnit,
    StageBWorkUnit,
    candidate_cell_id,
    execute_stage_b_cell,
    iter_work_units,
    prepare_stage_b_unit,
    stage_b_code_identity,
)

mx = importlib.import_module("mlx.core")
SELECTED_STAGE_B_EXECUTOR = REPOSITORY_ROOT / (
    "internal/active/veatic21-phase02-registration/selected-stage-b-executor.json"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class StageBTopology:
    cpu_preparation_workers: int
    mlx_stream_lanes: int

    @property
    def id(self) -> str:
        return f"cpu{self.cpu_preparation_workers:02d}_mlx{self.mlx_stream_lanes:02d}"

    def validate(self) -> None:
        _require(
            self.cpu_preparation_workers in {1, 2, 4, 8, 12},
            "unregistered Stage B CPU worker count",
        )
        _require(
            self.mlx_stream_lanes in {1, 2, 4, 8, 12},
            "unregistered Stage B MLX lane count",
        )


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode())
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)  # ty: ignore[invalid-argument-type]
        with Path(temporary).open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _strict_record(path: Path) -> dict[str, Any]:
    value = load_json(path)
    _require(value.get("schema_version") == "veatic21_phase02_stage_b_cell_v1", "bad cell")
    _require(value.get("outer_test_scores_opened") is False, "outer access changed")
    _require(value.get("cortical_values_opened") is False, "cortical access changed")
    return value


def _append_unique_ledger(path: Path, value: dict[str, Any], *, identity_key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = value[identity_key]
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                existing = json.loads(line)
                if existing.get(identity_key) == identity:
                    _require(existing == value, f"ledger identity collision: {identity}")
                    return
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _registration() -> dict[str, Any]:
    registration = load_json(STAGE_B_EXECUTION_REGISTRATION)
    _require(
        registration.get("registration_status") == "prospective_before_any_stage_b_fit",
        "Stage B execution registration is not prospective",
    )
    identity = cast(dict[str, Any], registration["input_identity"])
    _require(
        identity["stage_b_work_registry_sha256"] == EXPECTED_WORK_REGISTRY_SHA256,
        "Stage B registry pin changed",
    )
    _require(
        identity["stage_a_aggregation_verification_sha256"]
        == EXPECTED_AGGREGATION_VERIFICATION_SHA256,
        "Stage B aggregation verification pin changed",
    )
    return registration


def _registered_cell_pairs(
    registration: dict[str, Any], name: str = "representative_cells"
) -> tuple[tuple[int, str], ...]:
    backtest = cast(dict[str, Any], registration["systems_backtest"])
    values = cast(list[list[Any]], backtest[name])
    pairs = tuple((int(value[0]), str(value[1])) for value in values)
    _require(len(pairs) == len(set(pairs)) and bool(pairs), f"invalid {name}")
    return pairs


def _resolve_cells(
    pairs: tuple[tuple[int, str], ...],
) -> tuple[tuple[StageBWorkUnit, dict[str, Any]], ...]:
    wanted = dict(pairs)
    units = {unit.sequence: unit for unit in iter_work_units() if unit.sequence in wanted}
    _require(set(units) == set(wanted), "registered Stage B backtest work unit is missing")
    resolved: list[tuple[StageBWorkUnit, dict[str, Any]]] = []
    for sequence, cell_identity in pairs:
        unit = units[sequence]
        matches = [
            candidate
            for candidate in unit.candidates
            if candidate_cell_id(unit, candidate) == cell_identity
        ]
        _require(len(matches) == 1, "registered Stage B backtest cell is not unique")
        resolved.append((unit, matches[0]))
    return tuple(resolved)


def _prepare_cells(
    cells: tuple[tuple[StageBWorkUnit, dict[str, Any]], ...],
    *,
    inputs: Any,
    workers: int,
) -> dict[int, PreparedStageBUnit]:
    units = {unit.sequence: unit for unit, _ in cells}

    def prepare(unit: StageBWorkUnit) -> tuple[int, PreparedStageBUnit]:
        return unit.sequence, prepare_stage_b_unit(unit, inputs=inputs)

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="stage-b-prep") as pool:
        values = list(pool.map(prepare, units.values()))
    return dict(values)


_lane_state = threading.local()


def _execute_on_lane(
    value: tuple[PreparedStageBUnit, dict[str, Any]],
) -> tuple[dict[str, Any], np.ndarray, dict[str, np.ndarray]]:
    if not hasattr(_lane_state, "stream"):
        _lane_state.stream = mx.new_stream(mx.gpu)
    with mx.stream(_lane_state.stream):
        return execute_stage_b_cell(*value)


def _normalized_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "runtime_seconds"}


def _normalized_identity(records: list[dict[str, Any]]) -> str:
    normalized = sorted(
        (_normalized_record(record) for record in records),
        key=lambda row: str(row["candidate_cell_id"]),
    )
    return hashlib.sha256(canonical_json_bytes(normalized)).hexdigest()


def _publish_backtest_cell(
    root: Path,
    record: dict[str, Any],
    prediction: np.ndarray,
    checkpoint: dict[str, np.ndarray],
) -> dict[str, Any]:
    cell_id = str(record["candidate_cell_id"])
    json_path = root / "cells" / f"{cell_id}.json"
    npz_path = root / "cells" / f"{cell_id}.npz"
    arrays = {
        "validation_prediction": prediction.astype(np.float32),
        "validation_row_indices": checkpoint["validation_row_indices"],
        **{
            f"checkpoint__{name}": value
            for name, value in checkpoint.items()
            if name != "validation_row_indices"
        },
    }
    _atomic_npz(npz_path, arrays)
    _atomic_json(json_path, record)
    manifest = {
        "candidate_cell_id": cell_id,
        "record_sha256": sha256_file(json_path),
        "artifact_sha256": sha256_file(npz_path),
        "checkpoint_sha256": record["checkpoint_sha256"],
        "validation_prediction_sha256": record["validation_prediction_sha256"],
    }
    _append_unique_ledger(
        root / "append-only-ledger.jsonl",
        manifest,
        identity_key="candidate_cell_id",
    )
    return manifest


def _resume_audit(root: Path, manifests: list[dict[str, Any]]) -> dict[str, Any]:
    started = time.monotonic()
    for manifest in manifests:
        cell_id = str(manifest["candidate_cell_id"])
        json_path = root / "cells" / f"{cell_id}.json"
        npz_path = root / "cells" / f"{cell_id}.npz"
        _require(sha256_file(json_path) == manifest["record_sha256"], "resume JSON changed")
        _require(sha256_file(npz_path) == manifest["artifact_sha256"], "resume NPZ changed")
        record = _strict_record(json_path)
        _require(record["candidate_cell_id"] == cell_id, "resume cell identity changed")
        with np.load(npz_path, allow_pickle=False) as payload:
            _require("validation_prediction" in payload.files, "resume prediction is missing")
            _require(
                "validation_row_indices" in payload.files,
                "resume validation row identities are missing",
            )
    return {
        "status": "PASS",
        "cells_reused": len(manifests),
        "seconds": time.monotonic() - started,
    }


def _pressure_configuration() -> dict[str, int]:
    device = cast(dict[str, Any], mx.device_info())
    physical = int(device["memory_size"])
    memory_limit = int(device["max_recommended_working_set_size"])
    cache_limit = physical // 8
    mx.set_memory_limit(memory_limit)
    mx.set_cache_limit(cache_limit)
    return {
        "memory_limit_bytes": memory_limit,
        "cache_limit_bytes": cache_limit,
        "physical_memory_bytes": physical,
    }


def _stage_b_resource_sampler(
    stop: threading.Event,
    pid: int,
    samples: list[dict[str, float | int | None]],
    interval_seconds: float,
) -> None:
    while not stop.wait(interval_seconds):
        completed = subprocess.run(
            ["ps", "-p", str(pid), "-o", "%cpu=,rss="],
            check=True,
            capture_output=True,
            text=True,
        )
        fields = completed.stdout.split()
        cpu = float(fields[0]) if len(fields) == 2 else 0.0
        rss = int(fields[1]) * 1024 if len(fields) == 2 else 0
        samples.append(
            {
                "monotonic_seconds": time.monotonic(),
                "gpu_device_utilization_percent": _gpu_device_utilization(),
                "worker_cpu_percent": cpu,
                "worker_rss_bytes": rss,
                "system_memory_free_percent": _system_memory_free_percent(),
            }
        )


def _power_state() -> tuple[str, int | None]:
    battery = subprocess.run(
        ["/usr/bin/pmset", "-g", "batt"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout
    custom = subprocess.run(
        ["/usr/bin/pmset", "-g", "custom"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout
    source = "AC Power" if "AC Power" in battery else "Battery Power"
    low_power: int | None = None
    for line in custom.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0] == "lowpowermode":
            low_power = int(fields[1])
    return source, low_power


def _run_cells(
    cells: tuple[tuple[StageBWorkUnit, dict[str, Any]], ...],
    *,
    inputs: Any,
    topology: StageBTopology,
    publication_root: Path | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prepared = _prepare_cells(
        cells,
        inputs=inputs,
        workers=topology.cpu_preparation_workers,
    )
    work = [(prepared[unit.sequence], candidate) for unit, candidate in cells]
    with ThreadPoolExecutor(
        max_workers=topology.mlx_stream_lanes,
        thread_name_prefix="stage-b-mlx",
    ) as pool:
        outputs = list(pool.map(_execute_on_lane, work))
    records = [value[0] for value in outputs]
    manifests = []
    if publication_root is not None:
        for record, prediction, checkpoint in outputs:
            manifests.append(
                _publish_backtest_cell(publication_root, record, prediction, checkpoint)
            )
    return records, manifests


def run_stage_b_topology_repetition(
    *,
    topology_value: dict[str, int],
    repetition: int,
    output_root_text: str,
) -> dict[str, Any]:
    """Fresh-process worker for one registered topology/repetition."""

    topology = StageBTopology(**topology_value)
    topology.validate()
    registration = _registration()
    timed_cells = _resolve_cells(_registered_cell_pairs(registration))
    warmup_cells = _resolve_cells(_registered_cell_pairs(registration, "warmup_cells"))
    pressure_limits = _pressure_configuration()
    root = Path(output_root_text) / "topologies" / topology.id / f"repeat-{repetition:02d}"
    root.mkdir(parents=True, exist_ok=True)

    warmup_inputs = _load_inputs()
    _run_cells(
        warmup_cells,
        inputs=warmup_inputs,
        topology=topology,
        publication_root=None,
    )
    mx.synchronize()
    mx.clear_cache()

    samples: list[dict[str, float | int | None]] = []
    stop = threading.Event()
    sampler = threading.Thread(
        target=_stage_b_resource_sampler,
        args=(stop, os.getpid(), samples, 0.5),
        daemon=True,
        name="stage-b-resource-sampler",
    )
    pressure_before = _host_pressure_snapshot()
    power_source, low_power_mode = _power_state()
    sampler.start()
    started = time.monotonic()
    try:
        inputs = _load_inputs()
        records, manifests = _run_cells(
            timed_cells,
            inputs=inputs,
            topology=topology,
            publication_root=root,
        )
        mx.synchronize()
    finally:
        elapsed = time.monotonic() - started
        stop.set()
        sampler.join(timeout=5)
    resume = _resume_audit(root, manifests)
    ledger_path = root / "append-only-ledger.jsonl"
    resources = _resource_summary(samples)
    resources["pressure_before"] = pressure_before
    resources["pressure_after"] = _host_pressure_snapshot()
    resources["mlx_peak_memory_bytes"] = int(mx.get_peak_memory())
    resources["mlx_active_memory_bytes"] = int(mx.get_active_memory())
    resources["mlx_cache_memory_bytes"] = int(mx.get_cache_memory())
    result = {
        "schema_version": "veatic21_phase02_stage_b_topology_repetition_v1",
        "status": "COMPLETE",
        "topology": asdict(topology),
        "topology_id": topology.id,
        "repetition": repetition,
        "cells": len(records),
        "elapsed_seconds": elapsed,
        "cells_per_second": len(records) / elapsed,
        "normalized_evidence_sha256": _normalized_identity(records),
        "cell_manifests": manifests,
        "append_only_ledger_lines": len(manifests),
        "append_only_ledger_sha256": sha256_file(ledger_path),
        "resume": resume,
        "resources": resources,
        "mlx_limits": pressure_limits,
        "power_source": power_source,
        "low_power_mode": low_power_mode,
        "outer_test_scores_opened": False,
        "cortical_values_opened": False,
        "prospective_washout_candidates_opened": False,
    }
    _atomic_json(root / "result.json", result)
    return result


def _topologies(registration: dict[str, Any]) -> tuple[StageBTopology, ...]:
    backtest = cast(dict[str, Any], registration["systems_backtest"])
    cpu = cast(list[int], backtest["cpu_preparation_workers"])
    lanes = cast(list[int], backtest["mlx_stream_lanes"])
    values = tuple(StageBTopology(workers, lane) for workers in cpu for lane in lanes)
    for topology in values:
        topology.validate()
    return values


def _eligible_resources(result: dict[str, Any], registration: dict[str, Any]) -> bool:
    gates = cast(dict[str, Any], registration["resource_gates"])
    resources = cast(dict[str, Any], result["resources"])
    pressure_before = cast(dict[str, Any], resources["pressure_before"])
    pressure_after = cast(dict[str, Any], resources["pressure_after"])
    estimated_headroom = resources.get("estimated_minimum_memory_headroom_bytes")
    thermal = str(pressure_after.get("thermal", ""))
    swap_before = str(pressure_before.get("swap", ""))
    swap_after = str(pressure_after.get("swap", ""))
    gpu_max = resources.get("gpu_device_utilization_max_percent")
    return (
        result["resume"]["status"] == "PASS"
        and int(resources.get("sample_count", 0)) > 0
        and estimated_headroom is not None
        and int(estimated_headroom) >= int(gates["minimum_memory_headroom_bytes"])
        and "used = 0.00M" in swap_before
        and "used = 0.00M" in swap_after
        and "No thermal warning" in thermal
        and "No performance warning" in thermal
        and gpu_max is not None
        and float(gpu_max) > 0
        and result.get("power_source") == "AC Power"
        and result.get("low_power_mode") == 0
    )


def run_stage_b_executor_backtest() -> dict[str, Any]:
    """Run/resume the complete registered Stage B topology matrix."""

    registration = _registration()
    registration_sha = sha256_file(STAGE_B_EXECUTION_REGISTRATION)
    code_sha = stage_b_code_identity()
    backtest = cast(dict[str, Any], registration["systems_backtest"])
    repetitions = int(backtest["measured_repetitions"])
    request = {
        "schema_version": "veatic21_phase02_stage_b_backtest_request_v1",
        "stage_b_execution_registration_sha256": registration_sha,
        "stage_b_code_sha256": code_sha,
        "work_registry_sha256": EXPECTED_WORK_REGISTRY_SHA256,
        "aggregation_verification_sha256": EXPECTED_AGGREGATION_VERIFICATION_SHA256,
        "representative_cells": len(_registered_cell_pairs(registration)),
        "topologies": [asdict(value) for value in _topologies(registration)],
        "repetitions": repetitions,
        "fresh_process_per_topology_repetition": True,
        "outer_test_scores_opened": False,
        "cortical_values_opened": False,
        "prospective_washout_candidates_opened": False,
    }
    STAGE_B_BACKTEST_ROOT.mkdir(parents=True, exist_ok=True)
    request_path = STAGE_B_BACKTEST_ROOT / "request.json"
    if request_path.exists():
        _require(load_json(request_path) == request, "Stage B backtest request changed")
    else:
        _write_json(request_path, request)

    results: list[dict[str, Any]] = []
    context = multiprocessing.get_context("spawn")
    for topology in _topologies(registration):
        for repetition in range(repetitions):
            path = (
                STAGE_B_BACKTEST_ROOT
                / "topologies"
                / topology.id
                / f"repeat-{repetition:02d}"
                / "result.json"
            )
            if path.exists():
                result = load_json(path)
            else:
                with ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:
                    result = pool.submit(
                        run_stage_b_topology_repetition,
                        topology_value=asdict(topology),
                        repetition=repetition,
                        output_root_text=str(STAGE_B_BACKTEST_ROOT),
                    ).result()
            results.append(result)

    reference_identities = {str(value["normalized_evidence_sha256"]) for value in results}
    identity_pass = len(reference_identities) == 1
    summaries: list[dict[str, Any]] = []
    for topology in _topologies(registration):
        owned = [value for value in results if value["topology_id"] == topology.id]
        rates = [float(value["cells_per_second"]) for value in owned]
        eligible = identity_pass and all(
            _eligible_resources(value, registration) for value in owned
        )
        summaries.append(
            {
                "topology": asdict(topology),
                "topology_id": topology.id,
                "median_cells_per_second": statistics.median(rates),
                "minimum_cells_per_second": min(rates),
                "maximum_cells_per_second": max(rates),
                "eligible": eligible,
            }
        )
    eligible = [value for value in summaries if value["eligible"]]
    _require(bool(eligible), "no eligible Stage B topology")
    fastest = max(float(value["median_cells_per_second"]) for value in eligible)
    plateau = [
        value for value in eligible if float(value["median_cells_per_second"]) >= 0.97 * fastest
    ]
    selected = min(
        plateau,
        key=lambda value: (
            int(value["topology"]["mlx_stream_lanes"]),
            int(value["topology"]["cpu_preparation_workers"]),
            str(value["topology_id"]),
        ),
    )
    result = {
        "schema_version": "veatic21_phase02_stage_b_backtest_result_v1",
        "status": "PASS" if identity_pass else "FAIL",
        "request_sha256": sha256_file(request_path),
        "registration_sha256": registration_sha,
        "stage_b_code_sha256": code_sha,
        "normalized_evidence_sha256": next(iter(reference_identities)) if identity_pass else None,
        "topology_summaries": summaries,
        "absolute_fastest_median_cells_per_second": fastest,
        "selected_topology": selected,
        "repetitions": len(results),
        "outer_test_scores_opened": False,
        "cortical_values_opened": False,
        "prospective_washout_candidates_opened": False,
    }
    _atomic_json(STAGE_B_BACKTEST_ROOT / "result.json", result)
    return result


def _selected_executor() -> dict[str, Any]:
    _require(SELECTED_STAGE_B_EXECUTOR.is_file(), "selected Stage B executor is not frozen")
    selected = load_json(SELECTED_STAGE_B_EXECUTOR)
    _require(selected.get("eligible_for_main") is True, "Stage B main executor is ineligible")
    _require(selected["stage_b_code_sha256"] == stage_b_code_identity(), "Stage B code changed")
    _require(
        selected["execution_registration_sha256"] == sha256_file(STAGE_B_EXECUTION_REGISTRATION),
        "Stage B execution registration changed",
    )
    _require(
        selected["backtest_result_sha256"] == sha256_file(STAGE_B_BACKTEST_ROOT / "result.json"),
        "Stage B backtest result changed",
    )
    _require(
        selected["backtest_verification_sha256"]
        == sha256_file(STAGE_B_BACKTEST_ROOT / "verification.json"),
        "Stage B backtest verification changed",
    )
    return selected


def _unit_bundle_paths(root: Path, sequence: int) -> tuple[Path, Path, Path]:
    stem = f"{sequence:05d}"
    return (
        root / "units" / f"{stem}.json.gz",
        root / "units" / f"{stem}.npz",
        root / "units" / f"{stem}.manifest.json",
    )


def _publish_unit_bundle(
    root: Path,
    unit: StageBWorkUnit,
    outputs: list[tuple[dict[str, Any], np.ndarray, dict[str, np.ndarray]]],
) -> dict[str, Any]:
    json_path, npz_path, manifest_path = _unit_bundle_paths(root, unit.sequence)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    records = sorted((value[0] for value in outputs), key=lambda row: row["candidate_cell_id"])
    descriptor, temporary = tempfile.mkstemp(prefix=f".{json_path.name}.", dir=json_path.parent)
    try:
        with (
            os.fdopen(descriptor, "wb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle,
        ):
            for record in records:
                handle.write(canonical_json_bytes(record) + b"\n")
        with Path(temporary).open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, json_path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    arrays: dict[str, np.ndarray] = {}
    for record, prediction, checkpoint in outputs:
        cell = str(record["candidate_cell_id"])
        arrays[f"{cell}__prediction"] = prediction.astype(np.float32)
        arrays[f"{cell}__validation_row_indices"] = checkpoint["validation_row_indices"]
        for name, value in checkpoint.items():
            if name == "validation_row_indices":
                continue
            arrays[f"{cell}__checkpoint__{name}"] = value
    _atomic_npz(npz_path, arrays)
    manifest = {
        "schema_version": "veatic21_phase02_stage_b_unit_manifest_v1",
        "work_unit_id": unit.work_unit_id,
        "work_unit_sequence": unit.sequence,
        "candidate_cells": len(records),
        "candidate_cell_ids_sha256": hashlib.sha256(
            "".join(f"{row['candidate_cell_id']}\n" for row in records).encode()
        ).hexdigest(),
        "records_sha256": sha256_file(json_path),
        "artifacts_sha256": sha256_file(npz_path),
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def _valid_existing_bundle(root: Path, unit: StageBWorkUnit) -> dict[str, Any] | None:
    json_path, npz_path, manifest_path = _unit_bundle_paths(root, unit.sequence)
    if not (json_path.is_file() and npz_path.is_file() and manifest_path.is_file()):
        return None
    manifest = load_json(manifest_path)
    if (
        manifest.get("work_unit_id") != unit.work_unit_id
        or manifest.get("candidate_cells") != len(unit.candidates)
        or manifest.get("records_sha256") != sha256_file(json_path)
        or manifest.get("artifacts_sha256") != sha256_file(npz_path)
    ):
        return None
    return manifest


def _merge_main_ledgers(
    root: Path,
    *,
    lane_count: int,
    expected_sequences: set[int],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    seen: set[int] = set()
    for lane in range(lane_count):
        path = root / "ledgers" / f"shard-{lane:02d}.jsonl"
        _require(path.is_file(), f"missing Stage B shard ledger: {lane}")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                value = cast(dict[str, Any], json.loads(line))
                sequence = int(value["work_unit_sequence"])
                _require(sequence not in seen, f"duplicate Stage B work unit: {sequence}")
                _, _, manifest_path = _unit_bundle_paths(root, sequence)
                _require(
                    value == load_json(manifest_path),
                    f"Stage B ledger manifest changed: {sequence}",
                )
                seen.add(sequence)
                entries.append(value)
    _require(seen == expected_sequences, "Stage B shard ledgers do not exactly cover main")
    entries.sort(key=lambda value: int(value["work_unit_sequence"]))
    canonical = root / "append-only-experiment-ledger.jsonl"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{canonical.name}.", dir=root)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for value in entries:
                handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, canonical)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return {
        "ledger_lines": len(entries),
        "canonical_ledger_sha256": sha256_file(canonical),
    }


def run_stage_b_main() -> dict[str, Any]:
    """Run/resume exact Stage B units only after the selected executor is verified and pushed."""

    selected = _selected_executor()
    topology = StageBTopology(**cast(dict[str, int], selected["topology"]))
    topology.validate()
    units = list(iter_work_units())
    request = {
        "schema_version": "veatic21_phase02_stage_b_main_request_v1",
        "stage_b_code_sha256": stage_b_code_identity(),
        "execution_registration_sha256": sha256_file(STAGE_B_EXECUTION_REGISTRATION),
        "selected_executor_sha256": sha256_file(SELECTED_STAGE_B_EXECUTOR),
        "work_registry_sha256": EXPECTED_WORK_REGISTRY_SHA256,
        "topology": asdict(topology),
        "work_units": len(units),
        "candidate_cells": sum(len(unit.candidates) for unit in units),
        "outer_test_scores_opened": False,
        "cortical_values_opened": False,
        "prospective_washout_candidates_opened": False,
    }
    STAGE_B_MAIN_ROOT.mkdir(parents=True, exist_ok=True)
    request_path = STAGE_B_MAIN_ROOT / "request.json"
    if request_path.exists():
        _require(load_json(request_path) == request, "Stage B main request changed")
    else:
        _atomic_json(request_path, request)
    inputs = _load_inputs()
    manifests: list[dict[str, Any]] = []
    started = time.monotonic()
    with (
        ThreadPoolExecutor(
            max_workers=topology.cpu_preparation_workers,
            thread_name_prefix="stage-b-main-prep",
        ) as preparation_pool,
        ThreadPoolExecutor(
            max_workers=topology.mlx_stream_lanes,
            thread_name_prefix="stage-b-main-mlx",
        ) as mlx_pool,
    ):
        for start in range(0, len(units), topology.cpu_preparation_workers):
            batch = units[start : start + topology.cpu_preparation_workers]
            pending = [
                unit for unit in batch if _valid_existing_bundle(STAGE_B_MAIN_ROOT, unit) is None
            ]
            if pending:
                prepared_values = list(
                    preparation_pool.map(
                        lambda unit: prepare_stage_b_unit(unit, inputs=inputs),
                        pending,
                    )
                )
                prepared = {value.unit.sequence: value for value in prepared_values}
                cells = tuple(
                    (unit, candidate) for unit in pending for candidate in unit.candidates
                )
                work = [(prepared[unit.sequence], candidate) for unit, candidate in cells]
                outputs = list(mlx_pool.map(_execute_on_lane, work))
                offset = 0
                for unit in pending:
                    count = len(unit.candidates)
                    manifests.append(
                        _publish_unit_bundle(
                            STAGE_B_MAIN_ROOT,
                            unit,
                            outputs[offset : offset + count],
                        )
                    )
                    offset += count
            for unit in batch:
                existing = _valid_existing_bundle(STAGE_B_MAIN_ROOT, unit)
                _require(existing is not None, "Stage B unit publication failed")
                _append_unique_ledger(
                    STAGE_B_MAIN_ROOT
                    / "ledgers"
                    / f"shard-{unit.sequence % topology.mlx_stream_lanes:02d}.jsonl",
                    cast(dict[str, Any], existing),
                    identity_key="work_unit_sequence",
                )
                if all(value.get("work_unit_sequence") != unit.sequence for value in manifests):
                    manifests.append(cast(dict[str, Any], existing))
    ledger = _merge_main_ledgers(
        STAGE_B_MAIN_ROOT,
        lane_count=topology.mlx_stream_lanes,
        expected_sequences={unit.sequence for unit in units},
    )
    summary = {
        "schema_version": "veatic21_phase02_stage_b_main_summary_v1",
        "status": "COMPLETE_PENDING_INDEPENDENT_VERIFICATION",
        "request_sha256": sha256_file(request_path),
        "work_units": len(units),
        "candidate_cells": sum(int(value["candidate_cells"]) for value in manifests),
        "elapsed_seconds": time.monotonic() - started,
        "manifest_identity_sha256": hashlib.sha256(
            canonical_json_bytes(sorted(manifests, key=lambda value: value["work_unit_sequence"]))
        ).hexdigest(),
        **ledger,
        "outer_test_scores_opened": False,
        "cortical_values_opened": False,
        "prospective_washout_candidates_opened": False,
    }
    _atomic_json(STAGE_B_MAIN_ROOT / "summary.json", summary)
    return summary
