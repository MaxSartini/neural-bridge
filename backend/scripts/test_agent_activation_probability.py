"""Smoke-test operational agent modifiers used by the OASIS runner."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_parallel_simulation import _agent_activation_probability  # noqa: E402


def main() -> None:
    base = {
        "activity_level": 0.7,
        "posts_per_hour": 0.2,
        "comments_per_hour": 0.2,
        "response_delay_min": 30,
        "response_delay_max": 90,
        "active_hours": [12],
    }
    baseline, active = _agent_activation_probability(base, 12, 60)
    faster = dict(base, posts_per_hour=1.0, comments_per_hour=2.0, response_delay_min=5, response_delay_max=15)
    conditioned, _ = _agent_activation_probability(faster, 12, 60)
    off_hour, active_off_hour = _agent_activation_probability(faster, 2, 60)
    assert active and not active_off_hour
    assert conditioned > baseline
    assert off_hour < conditioned
    print({"agent_activation_probability_ok": True})


if __name__ == "__main__":
    main()
