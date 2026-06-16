"""Prepare leakage-aware human-choice prediction cases from Psych-101."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator


CHOICE_PATTERN = re.compile(r"<<([^<>\n]+)>>")


class Psych101BenchmarkBuilder:
    """Stream participant transcripts into next-choice prediction cases."""

    def __init__(self, max_context_chars: int = 32000):
        self.max_context_chars = max(1000, int(max_context_chars))

    def build(
        self,
        source_path: str,
        output_path: str,
        max_cases: int = 100,
        split: str = "participant_holdout",
        max_cases_per_group: int = 5,
    ) -> Dict[str, Any]:
        if split not in {"participant_holdout", "experiment_holdout"}:
            raise ValueError(f"Unsupported split: {split}")
        source = Path(source_path).expanduser().resolve()
        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        experiments = set()
        participants = set()
        group_counts: Dict[str, int] = {}
        with source.open("r", encoding="utf-8") as rows, target.open("w", encoding="utf-8") as output:
            for line in rows:
                record = json.loads(line)
                partition = self._partition(record, split)
                if partition != "test":
                    continue
                group = str(record["participant"] if split == "participant_holdout" else record["experiment"])
                for case in self._cases(record, split):
                    if group_counts.get(group, 0) >= max_cases_per_group:
                        break
                    output.write(json.dumps(case) + "\n")
                    count += 1
                    group_counts[group] = group_counts.get(group, 0) + 1
                    experiments.add(str(record["experiment"]))
                    participants.add(str(record["participant"]))
                    if max_cases > 0 and count >= max_cases:
                        break
                if max_cases > 0 and count >= max_cases:
                    break

        metadata = {
            "schema_version": "psych101_choice_benchmark_v1",
            "source_path": str(source),
            "output_path": str(target),
            "split": split,
            "partition": "test",
            "cases": count,
            "experiments": len(experiments),
            "participants": len(participants),
            "max_context_chars": self.max_context_chars,
            "max_cases_per_group": max_cases_per_group,
            "metrics": ["choice_nll", "top1_accuracy", "brier_score", "calibration_error"],
        }
        target.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata

    def _cases(self, record: Dict[str, Any], split: str) -> Iterator[Dict[str, Any]]:
        text = str(record["text"])
        for trial_index, match in enumerate(CHOICE_PATTERN.finditer(text)):
            prefix = text[: match.start()]
            yield {
                "case_id": self._case_id(record, trial_index),
                "experiment": str(record["experiment"]),
                "participant": str(record["participant"]),
                "split": split,
                "trial_index": trial_index,
                "prompt": prefix[-self.max_context_chars :],
                "observed_choice": match.group(1).strip(),
            }

    @staticmethod
    def _partition(record: Dict[str, Any], split: str) -> str:
        key = record["participant"] if split == "participant_holdout" else record["experiment"]
        bucket = int(hashlib.sha256(str(key).encode("utf-8")).hexdigest()[:8], 16) % 10
        if bucket == 0:
            return "test"
        if bucket == 1:
            return "validation"
        return "train"

    @staticmethod
    def _case_id(record: Dict[str, Any], trial_index: int) -> str:
        raw = f"{record['experiment']}|{record['participant']}|{trial_index}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
