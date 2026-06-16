"""Local validation harness for neuro-conditioned simulation outputs.

The harness scores predicted population-level axes against observed outcomes.
It is deliberately small and file-based so consulting/demo projects can build a
private calibration dataset without cloud services.
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_AXES = (
    "sentiment",
    "virality",
    "polarisation",
    "trust",
    "risk_aversion",
    "attention",
)


@dataclass
class ValidationRecord:
    stimulus_id: str
    condition: str
    predicted: Dict[str, float]
    observed: Dict[str, float]
    metadata: Dict[str, Any]


class NeuroValidationHarness:
    """Score baseline vs neuro-conditioned predictions against outcomes."""

    def load_records(self, path: str) -> List[ValidationRecord]:
        p = Path(path)
        if p.suffix.lower() == ".jsonl":
            return self._load_jsonl(p)
        if p.suffix.lower() == ".json":
            return self._load_json(p)
        if p.suffix.lower() == ".csv":
            return self._load_csv(p)
        raise ValueError(f"Unsupported validation file type: {p.suffix}")

    def score_records(
        self,
        records: Iterable[ValidationRecord],
        axes: Optional[Iterable[str]] = None,
        baseline_condition: str = "baseline",
    ) -> Dict[str, Any]:
        axes = tuple(axes or DEFAULT_AXES)
        rows = list(records)
        self._validate_records(rows)
        by_condition: Dict[str, Dict[str, Any]] = {}

        for condition in sorted({row.condition for row in rows}):
            subset = [row for row in rows if row.condition == condition]
            by_condition[condition] = self._score_subset(subset, axes)

        comparison = {}
        if baseline_condition in by_condition:
            base = by_condition[baseline_condition]
            for condition, metrics in by_condition.items():
                if condition == baseline_condition:
                    continue
                comparison[condition] = self._compare_to_baseline(metrics, base, axes)

        return {
            "record_count": len(rows),
            "axes": list(axes),
            "baseline_condition": baseline_condition,
            "conditions": by_condition,
            "comparison_to_baseline": comparison,
        }

    def _score_subset(self, rows: List[ValidationRecord], axes: Iterable[str]) -> Dict[str, Any]:
        metrics = {"count": len(rows), "axes": {}}
        for axis in axes:
            pairs = [
                (row.predicted.get(axis), row.observed.get(axis))
                for row in rows
                if axis in row.predicted and axis in row.observed
            ]
            pairs = [(float(p), float(o)) for p, o in pairs if p is not None and o is not None]
            if not pairs:
                continue
            errors = [p - o for p, o in pairs]
            abs_errors = [abs(e) for e in errors]
            squared = [e * e for e in errors]
            directional = [
                self._direction(p) == self._direction(o)
                for p, o in pairs
                if self._direction(p) != 0 and self._direction(o) != 0
            ]
            metrics["axes"][axis] = {
                "n": len(pairs),
                "mae": sum(abs_errors) / len(abs_errors),
                "rmse": math.sqrt(sum(squared) / len(squared)),
                "pearson": self._pearson(pairs),
                "spearman": self._spearman(pairs),
                "brier": sum(squared) / len(squared),
                "log_loss": self._log_loss(pairs),
                "expected_calibration_error": self._ece(pairs),
                "bias": sum(errors) / len(errors),
                "directional_accuracy": (
                    sum(1 for item in directional if item) / len(directional)
                    if directional else None
                ),
            }
        return metrics

    def _compare_to_baseline(self, candidate: Dict[str, Any], baseline: Dict[str, Any], axes: Iterable[str]) -> Dict[str, Any]:
        out = {}
        for axis in axes:
            cand_axis = candidate.get("axes", {}).get(axis)
            base_axis = baseline.get("axes", {}).get(axis)
            if not cand_axis or not base_axis:
                continue
            base_mae = base_axis.get("mae")
            cand_mae = cand_axis.get("mae")
            if base_mae is None or cand_mae is None or base_mae == 0:
                continue
            out[axis] = {
                "mae_delta": cand_mae - base_mae,
                "mae_improvement_pct": (base_mae - cand_mae) / base_mae * 100.0,
                "candidate_mae": cand_mae,
                "baseline_mae": base_mae,
            }
        return out

    def paired_comparison(
        self,
        records: Iterable[ValidationRecord],
        candidate_condition: str,
        baseline_condition: str = "baseline",
        axes: Optional[Iterable[str]] = None,
        bootstrap_samples: int = 1000,
        seed: int = 33,
    ) -> Dict[str, Any]:
        """Compare conditions only on matched stimuli and report uncertainty."""
        axes = tuple(axes or DEFAULT_AXES)
        rows = list(records)
        self._validate_records(rows)
        keyed = {(row.stimulus_id, row.condition): row for row in rows}
        common = sorted({
            row.stimulus_id
            for row in rows
            if (row.stimulus_id, baseline_condition) in keyed
            and (row.stimulus_id, candidate_condition) in keyed
        })
        result: Dict[str, Any] = {"matched_stimuli": len(common), "axes": {}}
        rng = random.Random(seed)
        for axis in axes:
            improvements = []
            for stimulus_id in common:
                baseline = keyed[(stimulus_id, baseline_condition)]
                candidate = keyed[(stimulus_id, candidate_condition)]
                if axis not in baseline.predicted or axis not in candidate.predicted or axis not in candidate.observed:
                    continue
                if baseline.observed.get(axis) != candidate.observed.get(axis):
                    raise ValueError(
                        f"Observed {axis} differs across matched conditions for {stimulus_id}"
                    )
                observed = candidate.observed[axis]
                improvements.append(
                    abs(baseline.predicted[axis] - observed)
                    - abs(candidate.predicted[axis] - observed)
                )
            if not improvements:
                continue
            bootstrap = [
                sum(rng.choice(improvements) for _ in improvements) / len(improvements)
                for _ in range(max(1, bootstrap_samples))
            ]
            bootstrap.sort()
            result["axes"][axis] = {
                "n": len(improvements),
                "mean_absolute_error_improvement": sum(improvements) / len(improvements),
                "candidate_win_rate": sum(value > 0 for value in improvements) / len(improvements),
                "bootstrap_95_ci": [
                    bootstrap[int(0.025 * (len(bootstrap) - 1))],
                    bootstrap[int(0.975 * (len(bootstrap) - 1))],
                ],
            }
        return result

    @staticmethod
    def _validate_records(rows: List[ValidationRecord]) -> None:
        seen = set()
        for row in rows:
            key = (row.stimulus_id, row.condition)
            if not row.stimulus_id:
                raise ValueError("Validation records require a non-empty stimulus_id")
            if key in seen:
                raise ValueError(f"Duplicate validation record for stimulus/condition: {key}")
            seen.add(key)
            for family, values in (("predicted", row.predicted), ("observed", row.observed)):
                for axis, value in values.items():
                    if not math.isfinite(float(value)):
                        raise ValueError(f"{family} {axis} is not finite for {key}")
                    if not 0.0 <= float(value) <= 1.0:
                        raise ValueError(f"{family} {axis} must be normalized to [0, 1] for {key}")

    def _direction(self, value: float, neutral: float = 0.5, deadband: float = 0.05) -> int:
        if value > neutral + deadband:
            return 1
        if value < neutral - deadband:
            return -1
        return 0

    @staticmethod
    def _log_loss(pairs: List[tuple[float, float]]) -> float:
        epsilon = 1e-7
        losses = []
        for predicted, observed in pairs:
            probability = min(1.0 - epsilon, max(epsilon, predicted))
            target = min(1.0, max(0.0, observed))
            losses.append(-(target * math.log(probability) + (1.0 - target) * math.log(1.0 - probability)))
        return sum(losses) / len(losses)

    @staticmethod
    def _ece(pairs: List[tuple[float, float]], bins: int = 10) -> float:
        total = len(pairs)
        error = 0.0
        for index in range(bins):
            low = index / bins
            high = (index + 1) / bins
            selected = [
                pair for pair in pairs
                if low <= pair[0] < high or (index == bins - 1 and pair[0] == 1.0)
            ]
            if selected:
                predicted_mean = sum(pair[0] for pair in selected) / len(selected)
                observed_mean = sum(pair[1] for pair in selected) / len(selected)
                error += len(selected) / total * abs(predicted_mean - observed_mean)
        return error

    @staticmethod
    def _pearson(pairs: List[tuple[float, float]]) -> Optional[float]:
        if len(pairs) < 2:
            return None
        predicted = [pair[0] for pair in pairs]
        observed = [pair[1] for pair in pairs]
        p_mean = sum(predicted) / len(predicted)
        o_mean = sum(observed) / len(observed)
        numerator = sum((p - p_mean) * (o - o_mean) for p, o in pairs)
        denominator = math.sqrt(
            sum((p - p_mean) ** 2 for p in predicted)
            * sum((o - o_mean) ** 2 for o in observed)
        )
        return numerator / denominator if denominator else None

    @classmethod
    def _spearman(cls, pairs: List[tuple[float, float]]) -> Optional[float]:
        if len(pairs) < 2:
            return None
        predicted_ranks = cls._ranks([pair[0] for pair in pairs])
        observed_ranks = cls._ranks([pair[1] for pair in pairs])
        return cls._pearson(list(zip(predicted_ranks, observed_ranks)))

    @staticmethod
    def _ranks(values: List[float]) -> List[float]:
        order = sorted(range(len(values)), key=values.__getitem__)
        ranks = [0.0] * len(values)
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and values[order[end]] == values[order[start]]:
                end += 1
            rank = (start + end - 1) / 2.0 + 1.0
            for index in order[start:end]:
                ranks[index] = rank
            start = end
        return ranks

    def _load_jsonl(self, path: Path) -> List[ValidationRecord]:
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(self._record_from_dict(json.loads(line)))
        return rows

    def _load_json(self, path: Path) -> List[ValidationRecord]:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("records", [])
        return [self._record_from_dict(row) for row in data]

    def _load_csv(self, path: Path) -> List[ValidationRecord]:
        rows = []
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                predicted = {
                    key.removeprefix("predicted_"): float(value)
                    for key, value in row.items()
                    if key.startswith("predicted_") and value not in ("", None)
                }
                observed = {
                    key.removeprefix("observed_"): float(value)
                    for key, value in row.items()
                    if key.startswith("observed_") and value not in ("", None)
                }
                rows.append(ValidationRecord(
                    stimulus_id=row.get("stimulus_id", ""),
                    condition=row.get("condition", "unknown"),
                    predicted=predicted,
                    observed=observed,
                    metadata={k: v for k, v in row.items() if not k.startswith(("predicted_", "observed_"))},
                ))
        return rows

    def _record_from_dict(self, data: Dict[str, Any]) -> ValidationRecord:
        return ValidationRecord(
            stimulus_id=str(data.get("stimulus_id", "")),
            condition=str(data.get("condition", "unknown")),
            predicted={k: float(v) for k, v in (data.get("predicted") or {}).items()},
            observed={k: float(v) for k, v in (data.get("observed") or {}).items()},
            metadata=dict(data.get("metadata") or {}),
        )
