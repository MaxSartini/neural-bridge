"""Dataclass models for stimulus-level neuro-prior profiles."""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List


def clamp01(value: Any) -> float:
    """Clamp numeric-like values into the inclusive [0, 1] range."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.5
    return max(0.0, min(1.0, numeric))


@dataclass
class NeuroPriorProfile:
    mode: str = "proxy"
    stimulus_type: str = "text"
    salience_score: float = 0.5
    threat_score: float = 0.5
    reward_score: float = 0.5
    arousal_score: float = 0.5
    uncertainty_score: float = 0.5
    memory_relevance_score: float = 0.5
    approach_bias: float = 0.5
    avoidance_bias: float = 0.5
    polarisation_risk: float = 0.5
    virality_pressure: float = 0.5
    confidence: float = 0.0
    dominant_neural_interpretation: str = ""
    behavioural_prior_summary: str = ""
    roi_summary: Dict[str, Any] = field(default_factory=dict)
    behavioural_axes: Dict[str, Any] = field(default_factory=dict)
    calibration_trace: Dict[str, Any] = field(default_factory=dict)
    limitations: List[str] = field(default_factory=list)
    raw_backend: str = ""
    raw_output_path: str = ""
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self) -> None:
        for key in (
            "salience_score",
            "threat_score",
            "reward_score",
            "arousal_score",
            "uncertainty_score",
            "memory_relevance_score",
            "approach_bias",
            "avoidance_bias",
            "polarisation_risk",
            "virality_pressure",
            "confidence",
        ):
            setattr(self, key, clamp01(getattr(self, key)))

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        for key, value in list(data.items()):
            if key.endswith("_score") or key in {
                "approach_bias",
                "avoidance_bias",
                "polarisation_risk",
                "virality_pressure",
                "confidence",
            }:
                data[key] = clamp01(value)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NeuroPriorProfile":
        allowed = set(cls.__dataclass_fields__.keys())
        clean = {key: value for key, value in (data or {}).items() if key in allowed}
        return cls(**clean)


def neutral_profile(mode: str, stimulus_type: str, reason: str) -> NeuroPriorProfile:
    return NeuroPriorProfile(
        mode=mode,
        stimulus_type=stimulus_type or "text",
        confidence=0.0,
        behavioural_prior_summary="Neutral neuro-prior profile used.",
        limitations=[reason],
    )
