from __future__ import annotations

from neural_bridge.veatic21.contracts import CURRENT_STATE
from neural_bridge.veatic21.data import load_json, sha256_file
from neural_bridge.veatic21.phase02_stage_a_rescue_registration import (
    COMPACT_RESCUE_REGISTRATION,
    CONFIGURATION_PATTERN,
    EXPECTED_UNDERTRAINED_BY_FAMILY,
    EXPECTED_UNDERTRAINED_CELLS,
    RESCUE_CELL_REGISTRY,
    RESCUE_REGISTRATION_VERIFICATION,
    RESCUE_UNIT_REGISTRY,
    _canonical_sha256,
)


def test_rescue_registration_freezes_complete_verified_undertrained_count() -> None:
    assert EXPECTED_UNDERTRAINED_CELLS == 113_392
    assert sum(EXPECTED_UNDERTRAINED_BY_FAMILY.values()) == EXPECTED_UNDERTRAINED_CELLS
    assert EXPECTED_UNDERTRAINED_BY_FAMILY == {
        "continuous_ridge": 112_754,
        "event_logistic_l2": 638,
    }


def test_rescue_configuration_parser_is_exact() -> None:
    match = CONFIGURATION_PATTERN.search("00000_unit__s01_e21__reg09")
    assert match is not None
    assert match.groups() == ("s01_e21", "09")
    assert CONFIGURATION_PATTERN.search("00000_unit__s02_e21__reg09") is None


def test_rescue_cell_identity_is_canonical() -> None:
    assert _canonical_sha256({"b": 2, "a": 1}) == _canonical_sha256({"a": 1, "b": 2})


def test_compact_rescue_registration_freezes_verified_external_registries() -> None:
    registration = load_json(COMPACT_RESCUE_REGISTRATION)
    external = registration["external_registration"]
    verification = load_json(RESCUE_REGISTRATION_VERIFICATION)

    assert registration["master_specification_version"] == "2.2"
    assert registration["coverage"]["undertrained_cells"] == EXPECTED_UNDERTRAINED_CELLS
    assert external["undertrained_cell_registry_sha256"] == sha256_file(RESCUE_CELL_REGISTRY)
    assert external["affected_unit_registry_sha256"] == sha256_file(RESCUE_UNIT_REGISTRY)
    assert external["verification_sha256"] == sha256_file(RESCUE_REGISTRATION_VERIFICATION)
    assert verification["status"] == "PASS"
    assert registration["rescue_execution_authorized"] is False
    assert sha256_file(COMPACT_RESCUE_REGISTRATION) in CURRENT_STATE.read_text()
