from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / "internal/active/veatic21-phase00-registration/experiment-registration.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_phase00_registration_is_prospective_complete_and_model_free() -> None:
    value = json.loads(REGISTRATION.read_text())
    assert value["status"] == "registered_not_executed"
    assert value["inputs"]["video_id_start_inclusive"] == 0
    assert value["inputs"]["video_id_end_inclusive"] == 123
    assert value["inputs"]["video_count"] == 124
    assert value["inputs"]["row_hz"] == 2.0
    assert value["inputs"]["forbidden_file"] == "vjepa21_hidden_states.npz"
    assert value["execution"]["later_phases_authorized"] is False


def test_phase00_registration_pins_authority_and_implementation_hashes() -> None:
    value = json.loads(REGISTRATION.read_text())
    authority = value["authority"]
    for key in ("master", "protocol", "current_state"):
        path = Path(authority[f"{key}_path"])
        assert _sha256(path) == authority[f"{key}_sha256"]
    implementation = value["implementation"]
    expected = {
        "benchmark_py_sha256": ROOT / "src/neural_bridge/veatic21/benchmark.py",
        "bundle_py_sha256": ROOT / "src/neural_bridge/veatic21/bundle.py",
        "phase00_py_sha256": ROOT / "src/neural_bridge/veatic21/phase00.py",
        "main_py_sha256": ROOT / "src/neural_bridge/veatic21/__main__.py",
    }
    for key, path in expected.items():
        assert _sha256(path) == implementation[key]


def test_selected_executor_is_the_fastest_recorded_repeated_median() -> None:
    value = json.loads(REGISTRATION.read_text())
    executor = value["executor_backtest"]
    medians: dict[int, float] = {}
    for workers, durations in executor["topology_repeat_end_to_end_seconds"].items():
        ordered = sorted(durations)
        midpoint = len(ordered) // 2
        if len(ordered) % 2:
            medians[int(workers)] = ordered[midpoint]
        else:
            medians[int(workers)] = (ordered[midpoint - 1] + ordered[midpoint]) / 2
    assert executor["selected_workers"] == min(medians, key=lambda workers: medians[workers])
    assert executor["selected_median_end_to_end_seconds"] == medians[12]
