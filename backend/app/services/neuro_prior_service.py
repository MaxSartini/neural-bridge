"""Coordinate TRIBE and proxy neuro-prior generation."""

from typing import Optional

from ..config import Config
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger
from .neuro_prior_models import NeuroPriorProfile, neutral_profile
from .tribe_adapter import TribeAdapter

logger = get_logger('neural_bridge.neuro_prior')


class NeuroPriorService:
    def generate(
        self,
        stimulus_text: str,
        stimulus_type: str = "text",
        simulation_requirement: str = "",
        document_text: str = "",
        simulation_id: str = "",
        output_dir: Optional[str] = None,
        media_path: Optional[str] = None,
    ) -> NeuroPriorProfile:
        stimulus_type = stimulus_type or "text"
        mode = getattr(Config, "NEURO_PRIOR_MODE", "proxy")
        if mode == "disabled":
            return neutral_profile("disabled", stimulus_type, "Neuro-prior mode disabled.")

        errors = []
        for backend in self._backend_order(mode):
            if backend == "disabled":
                return neutral_profile("disabled", stimulus_type, "Neuro-prior backend priority reached disabled.")
            if backend == "proxy":
                return self._generate_proxy(stimulus_text, stimulus_type, simulation_requirement, document_text)

            try:
                result = TribeAdapter().predict(
                    stimulus_text=stimulus_text,
                    stimulus_type=stimulus_type,
                    media_path=media_path,
                    output_dir=output_dir,
                    backend=backend,
                )
                if result.get("success"):
                    result["mode"] = backend
                    result["stimulus_type"] = stimulus_type
                    return NeuroPriorProfile.from_dict(result)
                errors.append(f"{backend}: {result.get('error', 'unknown failure')}")
            except Exception as exc:
                logger.warning(f"Neuro-prior backend failed ({backend}): {exc}")
                errors.append(f"{backend}: {exc}")
                if Config.NEURO_PRIOR_STRICT:
                    raise

            if Config.NEURO_PRIOR_STRICT:
                raise RuntimeError("; ".join(errors))

        if Config.NEURO_PRIOR_FALLBACK_TO_PROXY:
            profile = self._generate_proxy(stimulus_text, stimulus_type, simulation_requirement, document_text)
            profile.limitations.append("TRIBE unavailable; used Qwen/LM Studio proxy fallback. " + "; ".join(errors))
            return profile

        return neutral_profile("neutral", stimulus_type, "No neuro-prior backend succeeded. " + "; ".join(errors))

    def _backend_order(self, mode: str) -> list:
        if mode in {"apple_silicon_tribe", "official_tribe", "tribe_mlx", "proxy", "disabled"}:
            ordered = [mode]
            for item in Config.NEURO_PRIOR_BACKEND_PRIORITY.split(","):
                item = item.strip()
                if item and item not in ordered:
                    ordered.append(item)
            return ordered
        return [item.strip() for item in Config.NEURO_PRIOR_BACKEND_PRIORITY.split(",") if item.strip()]

    def _generate_proxy(
        self,
        stimulus_text: str,
        stimulus_type: str,
        simulation_requirement: str,
        document_text: str,
    ) -> NeuroPriorProfile:
        system_prompt = (
            "You are a neuroscience-informed behavioural simulation calibration model.\n"
            "You are not diagnosing humans.\n"
            "You are not reading minds.\n"
            "You are not measuring emotion directly.\n"
            "You are estimating population-level affective and salience priors from a stimulus for use in an agent-based social simulation.\n"
            "Return only valid JSON.\n"
            "No markdown.\n"
            "No prose outside JSON.\n"
            "All numeric scores must be between 0.0 and 1.0."
        )
        user_prompt = f"""Analyze the following stimulus for population-level behavioural priors.

The goal is to predict likely public reaction dynamics, such as public outcry, campaign effectiveness, backlash risk, trust, fear, reward, engagement, polarisation, and virality.

Evaluate the stimulus through these biologically inspired dimensions:
- salience: how strongly the content captures attention
- threat: perceived danger, loss, anger, moral violation, reputational harm, fear, or risk
- reward: perceived benefit, hope, status, gain, excitement, relief, or opportunity
- arousal: emotional activation intensity
- uncertainty: ambiguity, confusion, instability, missing information
- memory relevance: likelihood that the content sticks and recurs in later discussion
- approach bias: tendency toward acceptance, sharing, support, curiosity, optimism
- avoidance bias: tendency toward rejection, defence, anger, fear, suspicion, withdrawal
- polarisation risk: likelihood of splitting groups into opposed camps
- virality pressure: likelihood of spreading rapidly because of novelty, outrage, humour, fear, reward, or identity relevance

Return this exact JSON schema:
{{
  "salience_score": 0.0,
  "threat_score": 0.0,
  "reward_score": 0.0,
  "arousal_score": 0.0,
  "uncertainty_score": 0.0,
  "memory_relevance_score": 0.0,
  "approach_bias": 0.0,
  "avoidance_bias": 0.0,
  "polarisation_risk": 0.0,
  "virality_pressure": 0.0,
  "confidence": 0.0,
  "dominant_neural_interpretation": "",
  "behavioural_prior_summary": "",
  "limitations": []
}}

Stimulus:
{stimulus_text[:12000]}

Simulation context:
{simulation_requirement[:4000]}

Document context excerpt:
{document_text[:4000]}"""
        default = neutral_profile("proxy", stimulus_type, "Proxy JSON generation failed.").to_dict()
        try:
            data = LLMClient().chat_json_safe(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=1600,
                default=default,
            )
            data["mode"] = "proxy"
            data["stimulus_type"] = stimulus_type
            data["raw_backend"] = "qwen_lm_studio_proxy"
            return NeuroPriorProfile.from_dict(data)
        except Exception as exc:
            logger.warning(f"Proxy neuro-prior generation failed: {exc}")
            return neutral_profile("proxy", stimulus_type, f"Proxy generation failed: {exc}")
