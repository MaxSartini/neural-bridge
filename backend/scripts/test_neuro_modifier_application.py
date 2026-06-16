"""Smoke-test conservative shared neuro-modifier application."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Config  # noqa: E402
from app.services.simulation_config_generator import (  # noqa: E402
    AgentActivityConfig,
    SimulationConfigGenerator,
)


def make_agent() -> AgentActivityConfig:
    return AgentActivityConfig(
        agent_id=1,
        entity_uuid="one",
        entity_name="Agent One",
        entity_type="Voter",
        activity_level=0.5,
        sentiment_bias=-0.4,
    )


def main() -> None:
    generator = SimulationConfigGenerator(api_key="test", base_url="http://localhost:1234/v1")
    modifiers = {"activity_multiplier": 1.4, "sentiment_shift": 0.5}
    original = Config.NEURO_PRIOR_SHARED_SENTIMENT_SHIFT
    try:
        Config.NEURO_PRIOR_SHARED_SENTIMENT_SHIFT = False
        conservative = generator._apply_neuro_modifiers_to_agent(make_agent(), modifiers, "Voter")
        assert abs(conservative.activity_level - 0.7) < 1e-9
        assert conservative.sentiment_bias == -0.4

        Config.NEURO_PRIOR_SHARED_SENTIMENT_SHIFT = True
        ablation = generator._apply_neuro_modifiers_to_agent(make_agent(), modifiers, "Voter")
        assert abs(ablation.sentiment_bias - 0.1) < 1e-9
    finally:
        Config.NEURO_PRIOR_SHARED_SENTIMENT_SHIFT = original
    print({"neuro_modifier_application_ok": True})


if __name__ == "__main__":
    main()
