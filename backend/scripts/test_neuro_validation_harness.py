"""Smoke-test the local neuro validation harness."""

import json
import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.neuro_validation_harness import NeuroValidationHarness  # noqa: E402


def main() -> None:
    records = [
        {
            "stimulus_id": "s1",
            "condition": "baseline",
            "predicted": {"sentiment": 0.45, "virality": 0.40},
            "observed": {"sentiment": 0.70, "virality": 0.65},
        },
        {
            "stimulus_id": "s1",
            "condition": "neuro",
            "predicted": {"sentiment": 0.68, "virality": 0.62},
            "observed": {"sentiment": 0.70, "virality": 0.65},
        },
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "records.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
        loaded = NeuroValidationHarness().load_records(str(path))
        result = NeuroValidationHarness().score_records(loaded)

    assert result["record_count"] == 2
    assert result["comparison_to_baseline"]["neuro"]["sentiment"]["mae_improvement_pct"] > 0
    assert result["comparison_to_baseline"]["neuro"]["virality"]["mae_improvement_pct"] > 0
    assert result["conditions"]["neuro"]["axes"]["sentiment"]["brier"] >= 0
    assert result["conditions"]["neuro"]["axes"]["sentiment"]["log_loss"] >= 0
    assert "pearson" in result["conditions"]["neuro"]["axes"]["sentiment"]
    assert "spearman" in result["conditions"]["neuro"]["axes"]["sentiment"]
    paired = NeuroValidationHarness().paired_comparison(
        loaded,
        candidate_condition="neuro",
        axes=["sentiment", "virality"],
        bootstrap_samples=100,
    )
    assert paired["matched_stimuli"] == 1
    assert paired["axes"]["sentiment"]["candidate_win_rate"] == 1.0
    try:
        NeuroValidationHarness().score_records(loaded + [loaded[0]])
        raise AssertionError("Duplicate validation record was not rejected")
    except ValueError:
        pass
    print({"validation_harness_ok": True, "comparison": result["comparison_to_baseline"]})


if __name__ == "__main__":
    main()
