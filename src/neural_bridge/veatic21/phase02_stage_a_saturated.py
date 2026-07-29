"""Frozen hardware-saturated main executor for VEATIC Phase 02 Stage A."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from neural_bridge.veatic21.contracts import (
    PHASE02_EXECUTOR_BACKTEST_REGISTRATION,
    PHASE02_EXECUTOR_BACKTEST_ROOT,
    PHASE02_REGISTRATION_SHA256,
    PHASE02_STAGE_A_SATURATED_ROOT,
    REPOSITORY_ROOT,
)
from neural_bridge.veatic21.data import load_json, reject_forbidden_runtime_path, sha256_file
from neural_bridge.veatic21.phase02_stage_a_executor import (
    ExecutorRunSelection,
    _configuration_from_dict,
    executor_code_identity,
    run_hardware_saturated_executor,
)

SELECTED_EXECUTOR = REPOSITORY_ROOT / (
    "internal/active/veatic21-phase02-registration/selected-executor.json"
)


def _selection(value: object) -> ExecutorRunSelection:
    ranges = cast(list[list[int]], value)
    if not ranges or any(len(pair) != 2 for pair in ranges):
        raise ValueError("selected executor sequence ranges are malformed")
    return ExecutorRunSelection(tuple((int(pair[0]), int(pair[1])) for pair in ranges))


def verify_phase02_selected_executor() -> dict[str, object]:
    """Verify the immutable backtest result and selected main executor identity."""

    frozen = load_json(SELECTED_EXECUTOR)
    if sha256_file(Path(__file__)) != frozen["launcher_source_sha256"]:
        raise ValueError("current launcher code differs from the frozen main launcher")
    backtest_result_path = PHASE02_EXECUTOR_BACKTEST_ROOT / "result.json"
    request_path = PHASE02_EXECUTOR_BACKTEST_ROOT / "request.json"
    summaries_path = PHASE02_EXECUTOR_BACKTEST_ROOT / "candidate-summaries.json"
    if str(PHASE02_EXECUTOR_BACKTEST_ROOT) != frozen["executor_backtest_root"]:
        raise ValueError("selected executor backtest root changed")
    expected_hashes = {
        request_path: frozen["executor_backtest_request_sha256"],
        summaries_path: frozen["executor_backtest_candidate_summaries_sha256"],
        backtest_result_path: frozen["executor_backtest_result_sha256"],
        PHASE02_EXECUTOR_BACKTEST_REGISTRATION: frozen["executor_backtest_registration_sha256"],
    }
    for path, expected in expected_hashes.items():
        if sha256_file(path) != expected:
            raise ValueError(f"selected executor evidence identity changed: {path}")

    result = load_json(backtest_result_path)
    selection = cast(dict[str, object], result["selection"])
    configuration_value = cast(dict[str, object], frozen["configuration"])
    configuration = _configuration_from_dict(configuration_value)
    configuration.validate()
    if result.get("status") != "PASS" or selection.get("status") != "PASS":
        raise ValueError("selected executor backtest did not pass")
    if result.get("candidate_count") != 10:
        raise ValueError("selected executor backtest candidate coverage changed")
    if result.get("scientific_registration_sha256") != PHASE02_REGISTRATION_SHA256:
        raise ValueError("selected executor scientific registration changed")
    if result.get("registration_sha256") != frozen["executor_backtest_registration_sha256"]:
        raise ValueError("selected executor registration identity changed")
    if selection.get("selected_configuration") != configuration_value:
        raise ValueError("selected executor configuration changed")
    if selection.get("selected_executor_sha256") != frozen["executor_sha256"]:
        raise ValueError("selected executor source identity changed in backtest evidence")
    if executor_code_identity() != frozen["executor_sha256"]:
        raise ValueError("current executor code differs from the frozen selected executor")
    candidates = cast(list[dict[str, object]], result["candidates"])
    if any(candidate.get("status") != "PASS" for candidate in candidates):
        raise ValueError("selected executor matrix contains a failed candidate")
    if any(
        cast(dict[str, object], candidate.get("equivalence", {})).get("status") != "PASS"
        for candidate in candidates
    ):
        raise ValueError("selected executor matrix contains an equivalence failure")
    required_candidate_gates = (
        "determinism_gate",
        "resume_gate",
        "ledger_gate",
        "access_gate",
        "memory_gate",
        "thermal_gate",
    )
    if any(
        candidate.get(gate) != "PASS"
        for candidate in candidates
        for gate in required_candidate_gates
    ):
        raise ValueError("selected executor matrix contains a failed safety gate")
    if any(
        cast(dict[str, object], candidate["configuration"])["compiled_ridge_update_blocks"]
        is not False
        or cast(dict[str, object], candidate["configuration"])["compiled_logistic_update_blocks"]
        is not False
        for candidate in candidates
    ):
        raise ValueError("selected executor supplement was not wholly uncompiled")
    if selection.get("saturation_gate") != {
        "reason": "mean_gpu_device_utilization_at_least_ninety_percent",
        "status": "PASS",
    }:
        raise ValueError("selected executor saturation gate changed")
    if selection.get("tie_within_three_percent_applied") is not True:
        raise ValueError("selected executor tie disposition changed")
    if result.get("outer_test_scores_opened") is not False:
        raise ValueError("selected executor evidence opened outer-test scores")
    if result.get("cortical_values_opened") is not False:
        raise ValueError("selected executor evidence opened cortical values")
    if str(PHASE02_STAGE_A_SATURATED_ROOT) != frozen["main_output_root"]:
        raise ValueError("selected executor main output root changed")
    main_selection = _selection(frozen["main_sequence_ranges_inclusive"])
    warmup_selection = _selection(frozen["warmup_sequence_ranges_inclusive"])
    if main_selection.sequence_ranges_inclusive != ((0, 40823),):
        raise ValueError("selected executor does not cover the complete Stage A registry")
    return {
        "schema_version": "veatic21_phase02_selected_executor_verification_v1",
        "status": "PASS",
        "selected_executor_sha256": frozen["executor_sha256"],
        "selected_executor_registration_sha256": sha256_file(SELECTED_EXECUTOR),
        "configuration": configuration_value,
        "main_selection": main_selection.json_value(),
        "warmup_selection": warmup_selection.json_value(),
        "main_work_units": frozen["main_work_units"],
        "main_configuration_evaluations": frozen["main_configuration_evaluations"],
        "outer_test_scores_opened": False,
        "cortical_values_opened": False,
    }


def run_phase02_stage_a_saturated(
    *, output_root: Path = PHASE02_STAGE_A_SATURATED_ROOT
) -> dict[str, object]:
    """Run or safely resume the complete Stage A matrix with the frozen executor."""

    output_root = reject_forbidden_runtime_path(output_root)
    if output_root != PHASE02_STAGE_A_SATURATED_ROOT:
        raise ValueError(f"saturated Stage A must use the canonical root: {output_root}")
    verification = verify_phase02_selected_executor()
    configuration = _configuration_from_dict(cast(dict[str, object], verification["configuration"]))
    main_selection = _selection(
        cast(dict[str, object], verification["main_selection"])["sequence_ranges_inclusive"]
    )
    warmup_selection = _selection(
        cast(dict[str, object], verification["warmup_selection"])["sequence_ranges_inclusive"]
    )
    return run_hardware_saturated_executor(
        output_root=output_root,
        configuration=configuration,
        selection=main_selection,
        warmup_selection=warmup_selection,
    )
