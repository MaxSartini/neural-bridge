import math
import random

import numpy as np

from backend.scripts.again_real_scout_selector_validation import (
    compute_label_vectors,
    evaluate_selector,
    score_rows,
    selected_intervals,
)


def test_pre_spike_labels_are_evaluation_only_shifted_from_spikes():
    rows = [
        {
            "time_start_seconds": str(i),
            "future_spike_1_3s_ge_0.05": "true" if i == 8 else "false",
        }
        for i in range(12)
    ]

    labels = compute_label_vectors(rows, threshold=0.05)

    assert labels["spike"][8]
    assert labels["pre_spike_2s"][6]
    assert labels["pre_spike_4s"][4]
    assert labels["pre_spike_6s"][2]
    assert labels["pre_spike_8s"][0]


def test_selected_intervals_expand_and_merge():
    intervals = selected_intervals([10, 12, 40], duration=50)

    assert intervals == [(2.0, 16.0), (32.0, 44.0)]


def test_hybrid_selector_evaluation_can_beat_random_when_signal_matches_labels():
    rows = []
    for i in range(30):
        positive = i in {10, 20}
        rows.append(
            {
                "video_id": "v1",
                "time_start_seconds": i,
                "future_spike_1_3s_ge_0.05": "true" if positive else "false",
                "telemetry_change_z": 5.0 if positive else 0.0,
                "cheap_video_audio_z": 5.0 if positive else 0.0,
                "scout_novelty_z": 5.0 if positive else 0.0,
                "vjepa_l_novelty_z": 5.0 if positive else "",
                "vjepa_b_novelty_z": "",
            }
        )

    hybrid = evaluate_selector(
        selector_name="hybrid_telemetry_video_audio_vjepa_b",
        budget_name="top5pct",
        rows_by_vid={"v1": rows},
        rng=random.Random(1),
    )

    assert hybrid["spike_recall"] == 1.0
    assert hybrid["deployable_selector"] is True
    assert not hybrid["selector_is_oracle"]


def test_random_coverage_matched_to_hybrid_matches_expanded_coverage():
    rows = []
    for i in range(120):
        rows.append(
            {
                "video_id": "v1",
                "time_start_seconds": i,
                "future_spike_1_3s_ge_0.05": "false",
                "telemetry_change_z": 10.0 if i in {50, 51, 52, 53, 54, 55} else 0.0,
                "cheap_video_audio_z": 10.0 if i in {50, 51, 52, 53, 54, 55} else 0.0,
                "scout_novelty_z": 10.0 if i in {50, 51, 52, 53, 54, 55} else 0.0,
                "vjepa_l_novelty_z": 10.0 if i in {50, 51, 52, 53, 54, 55} else "",
                "vjepa_b_novelty_z": "",
            }
        )

    hybrid = evaluate_selector(
        selector_name="hybrid_telemetry_video_audio_vjepa_b",
        budget_name="top5pct",
        rows_by_vid={"v1": rows},
        rng=random.Random(1),
    )
    matched = evaluate_selector(
        selector_name="random_coverage_matched_to_hybrid",
        budget_name="top5pct",
        rows_by_vid={"v1": rows},
        rng=random.Random(1),
    )

    assert matched["selector_is_control"] is True
    assert matched["deployable_selector"] is False
    assert matched["coverage_matched_to_selector"] == "hybrid_telemetry_video_audio_vjepa_b"
    assert math.isclose(
        matched["coverage_match_target_percent_of_video"],
        hybrid["selected_percent_of_video"],
        abs_tol=1e-9,
    )
    assert abs(matched["selected_percent_of_video"] - hybrid["selected_percent_of_video"]) <= 12 / 120


def test_scout_novelty_prefers_canonical_then_vitl_then_vitb_alias():
    scores = score_rows(
        [
            {"scout_novelty_z": 3.0, "vjepa_l_novelty_z": 9.0, "vjepa_b_novelty_z": 1.0},
            {"scout_novelty_z": "", "vjepa_l_novelty_z": 4.0, "vjepa_b_novelty_z": 1.0},
            {"scout_novelty_z": "", "vjepa_l_novelty_z": "", "vjepa_b_novelty_z": 2.0},
        ],
        {"scout_novelty_z": 1.0},
        rng=random.Random(1),
    )

    assert np.allclose(scores, [3.0, 4.0, 2.0])
