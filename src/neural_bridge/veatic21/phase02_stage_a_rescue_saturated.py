"""Frozen hardware-saturated executor for the complete VEATIC Stage A rescue."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

from neural_bridge.veatic21.contracts import REPOSITORY_ROOT
from neural_bridge.veatic21.data import load_json, reject_forbidden_runtime_path, sha256_file
from neural_bridge.veatic21.phase02_stage_a_rescue import (
    load_rescue_registry,
    rescue_solver_code_identity,
)
from neural_bridge.veatic21.phase02_stage_a_rescue_backtest import (
    RESCUE_EXECUTOR_BACKTEST_REGISTRATION,
    RESCUE_EXECUTOR_BACKTEST_ROOT,
)
from neural_bridge.veatic21.phase02_stage_a_rescue_backtest_verify import (
    verify_rescue_executor_backtest,
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
    RESCUE_REGISTRATION_VERIFICATION,
    RESCUE_ROOT,
    RESCUE_UNIT_REGISTRY,
)

SELECTED_RESCUE_EXECUTOR = REPOSITORY_ROOT / (
    "internal/active/veatic21-phase02-registration/selected-rescue-executor.json"
)
RESCUE_MAIN_ROOT = RESCUE_ROOT / "main-hardware-saturated"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sequence_digest(sequences: tuple[int, ...]) -> str:
    payload = json.dumps(list(sequences), separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _pretty_json_sha256(value: object) -> str:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    return hashlib.sha256(payload).hexdigest()


def _expected_main_request(
    configuration: RescueExecutorConfiguration,
    selection: RescueRunSelection,
    warmup: RescueRunSelection,
    *,
    rescue_cells: int,
) -> dict[str, object]:
    return {
        "schema_version": "veatic21_phase02_stage_a_rescue_executor_request_v1",
        "rescue_solver_code_sha256": rescue_solver_code_identity(),
        "executor_sha256": rescue_executor_code_identity(),
        "rescue_cell_registry_sha256": sha256_file(RESCUE_CELL_REGISTRY),
        "rescue_unit_registry_sha256": sha256_file(RESCUE_UNIT_REGISTRY),
        "rescue_registration_verification_sha256": sha256_file(
            RESCUE_REGISTRATION_VERIFICATION
        ),
        "configuration": {
            "id": configuration.id,
            "mlx_lanes": configuration.mlx_lanes,
            "gpu_streams_per_lane": configuration.gpu_streams_per_lane,
            "metric_workers_per_lane": configuration.metric_workers_per_lane,
            "cell_batch_size": configuration.cell_batch_size,
            "compiled_update_blocks": configuration.compiled_update_blocks,
            "fast_metrics": configuration.fast_metrics,
            "pipeline_depth": configuration.pipeline_depth,
        },
        "selection": selection.json_value(),
        "warmup_selection": warmup.json_value(),
        "rescue_units": len(selection.rescue_unit_sequences),
        "rescue_cells": rescue_cells,
        "outer_test_scores_opened": False,
        "cortical_values_opened": False,
        "aggregation_or_pruning_performed": False,
    }


def verify_selected_rescue_executor() -> dict[str, object]:
    """Verify immutable backtest evidence and the exact complete-rescue request."""

    frozen = load_json(SELECTED_RESCUE_EXECUTOR)
    _require(
        sha256_file(Path(__file__)) == frozen.get("launcher_source_sha256"),
        "current rescue launcher differs from the frozen launcher",
    )
    evidence_paths = {
        RESCUE_EXECUTOR_BACKTEST_REGISTRATION: frozen.get(
            "executor_backtest_registration_sha256"
        ),
        RESCUE_EXECUTOR_BACKTEST_ROOT / "request.json": frozen.get(
            "executor_backtest_request_sha256"
        ),
        RESCUE_EXECUTOR_BACKTEST_ROOT / "stage-summaries.json": frozen.get(
            "executor_backtest_stage_summaries_sha256"
        ),
        RESCUE_EXECUTOR_BACKTEST_ROOT / "result.json": frozen.get(
            "executor_backtest_result_sha256"
        ),
        RESCUE_EXECUTOR_BACKTEST_ROOT / "verification.json": frozen.get(
            "executor_backtest_verification_sha256"
        ),
    }
    for path, expected in evidence_paths.items():
        _require(sha256_file(path) == expected, f"frozen rescue evidence changed: {path}")

    verification = verify_rescue_executor_backtest(write_verification=False)
    _require(verification.get("status") == "PASS", "rescue backtest verification failed")
    _require(
        verification == load_json(RESCUE_EXECUTOR_BACKTEST_ROOT / "verification.json"),
        "stored rescue backtest verification changed",
    )
    selection = cast(dict[str, object], verification["selection"])
    configuration_value = cast(dict[str, object], frozen.get("configuration"))
    _require(
        selection.get("selected_configuration") == configuration_value,
        "frozen rescue configuration changed",
    )
    _require(
        rescue_solver_code_identity() == frozen.get("rescue_solver_sha256"),
        "selected rescue solver source changed",
    )
    _require(
        rescue_executor_code_identity() == frozen.get("rescue_executor_sha256"),
        "selected rescue executor source changed",
    )
    configuration = rescue_configuration_from_dict(configuration_value)

    registry = load_rescue_registry()
    sequences = tuple(unit.rescue_unit_sequence for unit in registry)
    expected_sequences = tuple(range(len(registry)))
    _require(sequences == expected_sequences, "rescue registry sequence coverage changed")
    main_identity = cast(dict[str, object], frozen.get("main_selection"))
    _require(main_identity.get("start_inclusive") == 0, "main rescue start changed")
    _require(
        main_identity.get("stop_exclusive") == len(registry), "main rescue stop changed"
    )
    _require(main_identity.get("count") == len(registry), "main rescue unit count changed")
    _require(
        main_identity.get("sequence_sha256") == _sequence_digest(sequences),
        "main rescue selection digest changed",
    )
    main_selection = RescueRunSelection(sequences)

    registration = load_json(RESCUE_EXECUTOR_BACKTEST_REGISTRATION)
    registered_warmup = tuple(
        cast(list[int], cast(dict[str, object], registration["representative_units"])[
            "warmup_rescue_unit_sequences"
        ])
    )
    frozen_warmup = cast(dict[str, object], frozen.get("warmup_selection"))
    _require(
        frozen_warmup.get("rescue_unit_sequences") == list(registered_warmup),
        "frozen warmup selection changed",
    )
    _require(
        frozen_warmup.get("sequence_sha256") == _sequence_digest(registered_warmup),
        "warmup selection digest changed",
    )
    warmup_selection = RescueRunSelection(registered_warmup)

    _require(str(RESCUE_MAIN_ROOT) == frozen.get("main_output_root"), "main root changed")
    rescue_cells = sum(len(unit.cells) for unit in registry)
    _require(rescue_cells == frozen.get("main_rescue_cells"), "main rescue cell count changed")
    expected_request = _expected_main_request(
        configuration,
        main_selection,
        warmup_selection,
        rescue_cells=rescue_cells,
    )
    _require(
        _pretty_json_sha256(expected_request) == frozen.get("main_request_sha256"),
        "exact main rescue request identity changed",
    )
    return {
        "schema_version": "veatic21_phase02_selected_rescue_executor_verification_v1",
        "status": "PASS",
        "configuration": configuration_value,
        "main_selection": main_identity,
        "warmup_selection": warmup_selection.json_value(),
        "main_rescue_units": len(registry),
        "main_rescue_cells": rescue_cells,
        "main_request_sha256": frozen["main_request_sha256"],
        "selected_rescue_executor_registration_sha256": sha256_file(
            SELECTED_RESCUE_EXECUTOR
        ),
        "outer_test_scores_opened": False,
        "cortical_values_opened": False,
        "aggregation_or_pruning_performed": False,
    }


def run_phase02_stage_a_rescue_saturated(
    *, output_root: Path = RESCUE_MAIN_ROOT
) -> dict[str, object]:
    """Run or safely resume every frozen rescue unit with the selected executor."""

    output_root = reject_forbidden_runtime_path(output_root)
    _require(output_root == RESCUE_MAIN_ROOT, "complete rescue must use its canonical root")
    verified = verify_selected_rescue_executor()
    configuration = rescue_configuration_from_dict(
        cast(dict[str, object], verified["configuration"])
    )
    main = RescueRunSelection(
        tuple(unit.rescue_unit_sequence for unit in load_rescue_registry())
    )
    warmup = RescueRunSelection(
        tuple(cast(list[int], cast(dict[str, object], verified["warmup_selection"])[
            "rescue_unit_sequences"
        ]))
    )
    result = run_rescue_executor(
        output_root=output_root,
        configuration=configuration,
        selection=main,
        warmup_selection=warmup,
    )
    _require(result.get("request_sha256") == verified["main_request_sha256"], "run request changed")
    return result
