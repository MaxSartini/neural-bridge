from __future__ import annotations

from neural_bridge.veatic21.phase02_stage_a_rescue_registration import (
    CONFIGURATION_PATTERN,
    EXPECTED_UNDERTRAINED_BY_FAMILY,
    EXPECTED_UNDERTRAINED_CELLS,
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
