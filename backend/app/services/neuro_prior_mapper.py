"""Experimental fallback mapping from neuro-prior profiles to modifiers.

These hand-authored formulas are unvalidated and must not be treated as
scientific behavioral claims or active simulation conditioning by default.
"""

from typing import Any, Dict

from .neuro_prior_models import NeuroPriorProfile


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


class NeuroPriorMapper:
    def map_to_modifiers(self, profile: NeuroPriorProfile) -> Dict[str, Any]:
        p = profile if isinstance(profile, NeuroPriorProfile) else NeuroPriorProfile.from_dict(profile or {})

        salience = p.salience_score
        threat = p.threat_score
        reward = p.reward_score
        arousal = p.arousal_score
        uncertainty = p.uncertainty_score
        memory = p.memory_relevance_score
        approach = p.approach_bias
        avoidance = p.avoidance_bias
        polarisation = p.polarisation_risk
        virality = p.virality_pressure
        confidence = p.confidence

        sentiment_shift = (reward + approach - threat - avoidance) * 0.18 * confidence
        trust_shift = (reward + approach - threat - uncertainty) * 0.16 * confidence
        risk_shift = (threat + uncertainty + avoidance - reward) * 0.18 * confidence
        activation = (salience + arousal + virality) / 3.0

        persona_instruction = self._persona_instruction(p)

        return {
            "schema_version": "heuristic_neuro_modifiers_v1",
            "status": "experimental_unvalidated_fallback_only",
            "active_conditioning_eligible": False,
            "sentiment_shift": _clamp(sentiment_shift, -0.5, 0.5),
            "activity_multiplier": self._confidence_weighted_multiplier(0.75 + activation * 0.75, confidence),
            "posting_multiplier": self._confidence_weighted_multiplier(0.8 + (salience * 0.35) + (virality * 0.45), confidence),
            "commenting_multiplier": self._confidence_weighted_multiplier(0.8 + (arousal * 0.35) + (uncertainty * 0.3), confidence),
            "sharing_multiplier": self._confidence_weighted_multiplier(0.8 + (virality * 0.5) + (approach * 0.2) + (threat * 0.1), confidence),
            "response_speed_multiplier": self._confidence_weighted_multiplier(0.75 + (salience * 0.45) + (arousal * 0.25), confidence),
            "risk_aversion_shift": _clamp(risk_shift, -0.5, 0.5),
            "trust_shift": _clamp(trust_shift, -0.5, 0.5),
            "polarisation_multiplier": self._confidence_weighted_multiplier(0.8 + (polarisation * 0.6) + (threat * salience * 0.25), confidence),
            "echo_chamber_shift": _clamp(((polarisation - 0.5) * 0.45 + (threat - reward) * 0.1) * confidence, -0.3, 0.4),
            "viral_threshold_shift": int(round((-4 * virality - 2 * salience + 3) * confidence)),
            "memory_persistence_multiplier": self._confidence_weighted_multiplier(0.75 + memory * 0.8, confidence),
            "persona_instruction": persona_instruction,
            "neuro_influence_confidence": confidence,
        }

    def _persona_instruction(self, profile: NeuroPriorProfile) -> str:
        if profile.confidence < 0.2:
            return "Use a neutral behavioural prior; preserve persona-specific reactions."
        parts = []
        if profile.salience_score >= 0.65:
            parts.append("Treat this stimulus as high-salience; agents should notice and revisit it more often.")
        if profile.threat_score >= 0.65:
            parts.append("Elevate threat sensitivity: defensive, risk-averse, suspicious, or avoidant reactions become more likely.")
        if profile.reward_score >= 0.65:
            parts.append("Elevate reward framing: hopeful, supportive, opportunity-seeking, and sharing-oriented reactions become more likely.")
        if profile.uncertainty_score >= 0.65:
            parts.append("Represent ambiguity through questions, speculation, disagreement, and lower trust.")
        if profile.polarisation_risk >= 0.65:
            parts.append("Increase group-splitting and identity-reinforcing interpretations without making every agent extreme.")
        if profile.virality_pressure >= 0.65:
            parts.append("Increase posting and sharing pressure, especially for journalists, influencers, activists, and opposition actors.")
        if not parts:
            parts.append("Use a neutral behavioural prior; preserve persona-specific reactions.")
        return " ".join(parts)

    @staticmethod
    def _confidence_weighted_multiplier(raw_multiplier: float, confidence: float) -> float:
        return _clamp(1.0 + (raw_multiplier - 1.0) * confidence, 0.5, 2.0)
