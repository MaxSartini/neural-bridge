from __future__ import annotations

from neural_bridge.veatic21.phase05 import PHASE05_CHECKS, phase05_transition


def test_phase05_transition_authorizes_exactly_phase06_after_claim_pass() -> None:
    checks = dict.fromkeys(PHASE05_CHECKS, True)
    assert phase05_transition(checks, claim_pass=True, legal_persistence_dominates=False) == (
        True,
        False,
    )


def test_phase05_transition_authorizes_washout_design_after_persistence_dominance() -> None:
    checks = dict.fromkeys(PHASE05_CHECKS, True)
    assert phase05_transition(checks, claim_pass=False, legal_persistence_dominates=True) == (
        False,
        True,
    )


def test_phase05_transition_requires_complete_integrity_matrix() -> None:
    checks = dict.fromkeys(PHASE05_CHECKS, True)
    checks[PHASE05_CHECKS[0]] = False
    assert phase05_transition(checks, claim_pass=True, legal_persistence_dominates=False) == (
        False,
        False,
    )
