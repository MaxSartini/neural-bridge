"""Deterministic smoke test for the paired StockNet benchmark."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.neuro_prior_models import NeuroPriorProfile  # noqa: E402
from app.services.stocknet_paired_benchmark import StockNetCase, StockNetPairedBenchmark  # noqa: E402


class FakeLlm:
    def chat_json_safe(self, messages, **kwargs):
        conditioned = "neuro-behavioural prior as one weak input" in messages[1]["content"]
        return {"probability_up": 0.8 if conditioned else 0.6, "confidence": 0.5, "reasoning": "test"}


class FakeNeuro:
    def generate(self, **kwargs):
        return NeuroPriorProfile(threat_score=0.7, confidence=0.4)


def main() -> None:
    runner = StockNetPairedBenchmark(
        llm=FakeLlm(), neuro_service=FakeNeuro(), require_real_neuro=False
    )
    result = runner.run([
        StockNetCase("s1", "2020-01-01", "ABC", "ABC Inc", "Tech", ["good news"], 0.01, 1.0)
    ])
    assert len(result["records"]) == 2
    assert result["records"][0]["condition"] == "llm_only"
    assert result["records"][1]["predicted"]["probability_up"] == 0.8
    try:
        StockNetPairedBenchmark(llm=FakeLlm(), neuro_service=FakeNeuro()).run([
            StockNetCase("s2", "2020-01-02", "ABC", "ABC Inc", "Tech", ["good news"], 0.01, 1.0)
        ])
        raise AssertionError("Proxy prior was accepted as a real-neuro benchmark")
    except RuntimeError:
        pass
    print({"stocknet_paired_benchmark_ok": True})


if __name__ == "__main__":
    main()
