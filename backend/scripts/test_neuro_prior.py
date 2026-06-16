"""Smoke-test neuro-prior model and mapper without requiring LM Studio/TRIBE."""

import tempfile
from pathlib import Path

from app.config import Config
from app.services.neuro_prior_mapper import NeuroPriorMapper
from app.services.neuro_prior_models import NeuroPriorProfile, neutral_profile
from app.services import neuro_prior_service


def main() -> None:
    profile = NeuroPriorProfile(
        mode="proxy",
        salience_score=0.8,
        threat_score=0.7,
        reward_score=0.2,
        arousal_score=0.75,
        uncertainty_score=0.65,
        polarisation_risk=0.8,
        virality_pressure=0.7,
        confidence=0.6,
    )
    modifiers = NeuroPriorMapper().map_to_modifiers(profile)
    assert 0.5 <= modifiers["activity_multiplier"] <= 2.0
    assert modifiers["viral_threshold_shift"] <= 0
    neutral = neutral_profile("disabled", "text", "test")
    assert neutral.to_dict()["mode"] == "disabled"
    neutral_modifiers = NeuroPriorMapper().map_to_modifiers(neutral)
    for key in (
        "activity_multiplier",
        "posting_multiplier",
        "commenting_multiplier",
        "sharing_multiplier",
        "response_speed_multiplier",
        "polarisation_multiplier",
        "memory_persistence_multiplier",
    ):
        assert neutral_modifiers[key] == 1.0, (key, neutral_modifiers[key])
    for key in ("sentiment_shift", "risk_aversion_shift", "trust_shift", "echo_chamber_shift"):
        assert neutral_modifiers[key] == 0.0, (key, neutral_modifiers[key])

    captured = {}

    class FakeAdapter:
        def predict(self, **kwargs):
            captured.update(kwargs)
            return {"success": True, "raw_backend": "fake"}

    original_adapter = neuro_prior_service.TribeAdapter
    original_mode = Config.NEURO_PRIOR_MODE
    try:
        neuro_prior_service.TribeAdapter = FakeAdapter
        Config.NEURO_PRIOR_MODE = "tribe_mlx"
        with tempfile.NamedTemporaryFile(suffix=".wav") as media:
            result = neuro_prior_service.NeuroPriorService().generate(
                stimulus_text="",
                stimulus_type="audio",
                media_path=media.name,
            )
            assert captured["media_path"] == media.name
            assert captured["stimulus_type"] == "audio"
            assert result.mode == "tribe_mlx"
    finally:
        neuro_prior_service.TribeAdapter = original_adapter
        Config.NEURO_PRIOR_MODE = original_mode
    print("neuro-prior smoke test passed")


if __name__ == "__main__":
    main()
