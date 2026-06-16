"""Auditable paired LLM-only vs neuro-conditioned StockNet benchmark."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.llm_client import LLMClient
from .neuro_prior_service import NeuroPriorService


@dataclass
class StockNetCase:
    stimulus_id: str
    date: str
    stock_symbol: str
    company_name: str
    sector: str
    tweets: List[str]
    movement_percent: float
    observed_probability_up: float

    @property
    def stimulus(self) -> str:
        return "\n".join(f"- {tweet}" for tweet in self.tweets)


class StockNetPairedBenchmark:
    """Run identical prediction prompts with and without a TRIBE-derived prior."""

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        neuro_service: Optional[NeuroPriorService] = None,
        require_real_neuro: bool = True,
    ) -> None:
        self.llm = llm or LLMClient(timeout=600.0)
        self.neuro_service = neuro_service or NeuroPriorService()
        self.require_real_neuro = require_real_neuro

    def load_cases(
        self,
        path: str,
        count: int = 3,
        min_tweets: int = 2,
        holdout_fraction: float = 0.2,
        offset: int = 0,
        sampling: str = "spaced",
    ) -> List[StockNetCase]:
        frame = pd.read_parquet(path)
        eligible = frame[frame["num_tweets"] >= min_tweets].sort_values(["date", "stock_symbol"])
        split = int(len(eligible) * (1.0 - holdout_fraction))
        holdout = eligible.iloc[split:]
        # Deterministic chronological holdout. No realized movement is shown to the model.
        if count == 0:
            rows = holdout.iloc[offset:]
        elif sampling == "sequential":
            rows = holdout.iloc[offset:offset + count]
        elif sampling == "spaced":
            available = holdout.iloc[offset:]
            positions = np.linspace(0, len(available) - 1, num=min(count, len(available)), dtype=int)
            rows = available.iloc[positions]
        else:
            raise ValueError(f"Unknown sampling method: {sampling}")
        return [
            StockNetCase(
                stimulus_id=f"stocknet:{row.date.date()}:{row.stock_symbol}",
                date=str(row.date.date()),
                stock_symbol=str(row.stock_symbol),
                company_name=str(row.company_name),
                sector=str(row.sector),
                tweets=[str(item) for item in row.tweets],
                movement_percent=float(row.movement_percent),
                observed_probability_up=float(row.label),
            )
            for row in rows.itertuples(index=False)
        ]

    def run(self, cases: List[StockNetCase]) -> Dict[str, Any]:
        records: List[Dict[str, Any]] = []
        case_outputs: List[Dict[str, Any]] = []
        for case in cases:
            baseline = self._predict(case, neuro_prior=None)
            prior = self.neuro_service.generate(
                stimulus_text=case.stimulus,
                stimulus_type="text",
                simulation_requirement="Forecast population reaction and resulting stock movement direction.",
            ).to_dict()
            if self.require_real_neuro and prior.get("mode") not in {
                "apple_silicon_tribe",
                "official_tribe",
                "tribe_mlx",
            }:
                raise RuntimeError(
                    "Neuro benchmark requires a real TRIBE backend; "
                    f"received mode={prior.get('mode')!r}, raw_backend={prior.get('raw_backend')!r}"
                )
            conditioned = self._predict(case, neuro_prior=prior)
            observed = {"probability_up": case.observed_probability_up}
            metadata = {
                "date": case.date,
                "stock_symbol": case.stock_symbol,
                "movement_percent": case.movement_percent,
                "neuro_mode": prior.get("mode"),
                "neuro_raw_backend": prior.get("raw_backend"),
            }
            records.extend([
                {
                    "stimulus_id": case.stimulus_id,
                    "condition": "llm_only",
                    "predicted": {"probability_up": baseline["probability_up"]},
                    "observed": observed,
                    "metadata": metadata,
                },
                {
                    "stimulus_id": case.stimulus_id,
                    "condition": "neuro_conditioned",
                    "predicted": {"probability_up": conditioned["probability_up"]},
                    "observed": observed,
                    "metadata": metadata,
                },
            ])
            case_outputs.append({
                "case": asdict(case),
                "llm_only": baseline,
                "neuro_prior": prior,
                "neuro_conditioned": conditioned,
            })
        return {"records": records, "cases": case_outputs}

    def _predict(self, case: StockNetCase, neuro_prior: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        compact_prior = None
        if neuro_prior is not None:
            compact_prior = {
                key: neuro_prior.get(key)
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
                    "behavioural_prior_summary",
                    "limitations",
                )
            }
        prior_text = (
            "No neuro-behavioural prior is available. Base the forecast only on the stimulus."
            if neuro_prior is None
            else "Use this uncertain population-level neuro-behavioural prior as one weak input, not as fact:\n"
            + json.dumps(compact_prior, ensure_ascii=False)
        )
        parse_failure = {"_parse_failure": True}
        messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a calibrated forecasting component. Return only JSON. "
                        "Do not invent market data. A probability near 0.5 is appropriate when evidence is weak."
                    ),
                },
                {
                    "role": "user",
                    "content": f"""Forecast whether {case.company_name} ({case.stock_symbol}), sector {case.sector},
will have positive movement for the StockNet observation associated with these public posts.
You cannot see the realized movement.

{prior_text}

Posts:
{case.stimulus[:8000]}

Return exactly:
{{"probability_up": 0.0, "confidence": 0.0, "reasoning": "brief evidence-based explanation"}}
Both numeric values must be between 0 and 1.""",
                },
            ]
        result = parse_failure
        for max_tokens in (3000, 6000):
            result = self.llm.chat_json_safe(
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
                default=parse_failure,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "stocknet_forecast",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "probability_up": {"type": "number"},
                                "confidence": {"type": "number"},
                                "reasoning": {"type": "string"},
                            },
                            "required": ["probability_up", "confidence", "reasoning"],
                            "additionalProperties": False,
                        },
                    },
                },
            )
            if not result.get("_parse_failure"):
                break
        if result.get("_parse_failure"):
            raise RuntimeError(f"Model output was not valid JSON for {case.stimulus_id}")
        result["probability_up"] = max(0.0, min(1.0, float(result.get("probability_up", 0.5))))
        result["confidence"] = max(0.0, min(1.0, float(result.get("confidence", 0.0))))
        return result

    @staticmethod
    def write_jsonl(records: List[Dict[str, Any]], path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
            encoding="utf-8",
        )
