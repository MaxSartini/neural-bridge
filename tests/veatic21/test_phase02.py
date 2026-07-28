from __future__ import annotations

from neural_bridge.veatic21.phase02 import PHASE02_CHECKS, phase03_authorized


def test_phase03_authorization_requires_exact_complete_phase02_matrix() -> None:
    complete = dict.fromkeys(PHASE02_CHECKS, True)
    assert phase03_authorized(complete)
    complete[PHASE02_CHECKS[0]] = False
    assert not phase03_authorized(complete)
    assert not phase03_authorized(dict.fromkeys(PHASE02_CHECKS[:-1], True))
