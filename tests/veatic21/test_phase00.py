from __future__ import annotations

from neural_bridge.veatic21.contracts import MANDATORY_CHECKS
from neural_bridge.veatic21.phase00 import phase01_authorized


def test_phase01_authorization_requires_every_mandatory_check() -> None:
    complete = dict.fromkeys(MANDATORY_CHECKS, True)
    assert phase01_authorized(complete) is True

    failed = dict(complete)
    failed[MANDATORY_CHECKS[0]] = False
    assert phase01_authorized(failed) is False

    missing = dict(complete)
    missing.pop(MANDATORY_CHECKS[-1])
    assert phase01_authorized(missing) is False

    extra = dict(complete)
    extra["unregistered_check"] = True
    assert phase01_authorized(extra) is False
