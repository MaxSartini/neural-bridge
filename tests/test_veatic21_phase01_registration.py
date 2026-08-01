from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / "internal/active/veatic21-phase01-registration/experiment-registration.json"
DATA_CONTRACT = ROOT / "internal/active/veatic21-phase01-registration/phase01-data-contract.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_phase01_registration_is_prospective_and_cortical_blind() -> None:
    value = json.loads(REGISTRATION.read_text())
    assert value["status"] == "registered_not_executed"
    assert value["inputs"]["video_count"] == 124
    assert value["inputs"]["total_rows"] == 20_657
    assert value["descriptive_audit"]["no_cortical_outcome_read"] is True
    assert value["execution"]["phase02_or_cortical_scoring_authorized"] is False


def test_phase01_registration_pins_current_authority_and_code() -> None:
    value = json.loads(REGISTRATION.read_text())
    authority = value["authority"]
    implementation = value["implementation"]
    for key in ("agents", "master", "combination", "protocol", "current_state"):
        assert _sha256(Path(authority[f"{key}_path"])) == authority[f"{key}_sha256"]
    expected = {
        "phase01_py_sha256": ROOT / "src/neural_bridge/veatic21/phase01.py",
        "main_py_sha256": ROOT / "src/neural_bridge/veatic21/__main__.py",
        "bundle_py_sha256": ROOT / "src/neural_bridge/veatic21/bundle.py",
    }
    for key, path in expected.items():
        assert _sha256(path) == implementation[key]


def test_blocked_split_means_rows_within_all_supported_videos() -> None:
    value = json.loads(REGISTRATION.read_text())
    split = value["split_derivation"]
    assert "within each video" in split["blocked_semantics"]
    assert split["blocked_outer_train_fraction_audit"] != [0.7]
    assert "at least 30% outer test" in split["selection_rule"]
    assert "at least 20 whole test videos" in split["selection_rule"]


def test_phase01_binds_supported_event_targets_to_geometry() -> None:
    value = json.loads(REGISTRATION.read_text())
    target = value["target_derivation"]
    output = value["supervised_combination_output"]
    assert (
        "nonzero-washout maximum-positive-arousal geometry"
        in target["event_quantile_selection_rule"]
    )
    assert "at least 50% of videos" in target["event_quantile_selection_rule"]
    assert "nonzero washout" in output["event_target_binding"]
    assert "simple-history residualization" in output["continuous_target_binding"]


def test_data_contract_forbids_cortical_values() -> None:
    value = json.loads(DATA_CONTRACT.read_text())
    assert set(value["forbidden_value_reads"]) == {
        "cortical_prediction",
        "temporal_diagnostics53",
        "tribe_grouped_video_feature",
    }
    assert not set(value["forbidden_value_reads"]).intersection(value["npz_allowlist"])


def test_selected_executor_is_fastest_repeated_median() -> None:
    value = json.loads(REGISTRATION.read_text())["executor_backtest"]
    medians = {
        int(workers): sorted(durations)[len(durations) // 2]
        for workers, durations in value["topology_repeat_end_to_end_seconds"].items()
    }
    assert value["selected_workers"] == min(medians, key=medians.__getitem__)
    assert value["selected_median_end_to_end_seconds"] == medians[6]
