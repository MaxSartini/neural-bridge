from __future__ import annotations

import numpy as np

from neural_bridge.veatic21.contracts import LabelRows
from neural_bridge.veatic21.event_screen import (
    _ar_features,
    _causal_forms,
    summarize_event_target_screen,
    targets_from_calibration,
)


def test_causal_forms_never_cross_video_boundaries() -> None:
    projected = np.asarray(((1.0,), (2.0,), (10.0,), (20.0,)), dtype=np.float32)
    forms = _causal_forms(
        projected,
        np.asarray(("a", "a", "b", "b")),
        np.asarray((0, 1, 0, 1)),
        (1,),
    )

    delta, available = forms["first_difference_w1"]
    assert available.tolist() == [False, True, False, True]
    assert delta[:, 0].tolist() == [0.0, 1.0, 0.0, 10.0]


def test_causal_forms_never_bridge_excluded_rows() -> None:
    projected = np.asarray(((1.0,), (3.0,), (4.0,)), dtype=np.float32)
    forms = _causal_forms(
        projected,
        np.asarray(("a", "a", "a")),
        np.asarray((0, 2, 3)),
        (3,),
    )

    _, mean_available = forms["causal_mean_w3"]
    _, delta_available = forms["current_minus_past_mean_w3"]
    assert not mean_available.any()
    assert not delta_available.any()


def test_ar_features_include_current_and_only_same_video_history() -> None:
    labels = LabelRows(
        video_id=np.asarray(("a", "a", "b", "b")),
        row_index=np.asarray((0, 1, 0, 1)),
        time_seconds=np.asarray((0.0, 0.5, 0.0, 0.5)),
        arousal=np.asarray((1.0, 2.0, 10.0, 20.0)),
        valence=np.zeros(4),
    )

    matrix, complete_history = _ar_features(labels, 1)

    assert matrix[:, 0].tolist() == [1.0, 2.0, 10.0, 20.0]
    assert matrix[:, 1].tolist() == [0.0, 1.0, 0.0, 10.0]
    assert matrix[:, 2].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert complete_history.tolist() == [False, True, False, True]


def test_targets_are_materialized_from_veatic_calibration() -> None:
    calibration = {
        "schema": "veatic21_event_calibration_v12",
        "benchmark_test_labels_accessed": False,
        "target_hypotheses": [
            {
                "name": "derived",
                "label": "arousal",
                "horizon_rows": [1, 2, 3],
                "train_quantile": 0.875,
                "transform": "positive",
            }
        ],
    }

    targets = targets_from_calibration(calibration)

    assert len(targets) == 1
    assert targets[0].horizon_rows == (1, 2, 3)
    assert targets[0].quantile == 0.875


def test_event_screen_summary_selects_complete_family_configuration() -> None:
    records = []
    for fold in range(2):
        for form, delta in (("current", 0.1), ("causal", 0.2)):
            records.append(
                {
                    "event_prevalence": 0.1,
                    "fold": fold,
                    "form": form,
                    "pca_width": 64,
                    "pooled_pr_auc": 0.3 + delta,
                    "ridge_alpha": 1.0,
                    "skill_delta_vs_ar": delta + fold * 0.01,
                    "source": "tribe_cortical",
                    "target": "spike",
                    "validation_rows": 10,
                }
            )
    screen = {
        "benchmark_test_labels_accessed": False,
        "fold_count": 2,
        "records": records,
        "schema": "veatic21_event_target_screen_v12",
        "screen_sha256": "screen",
        "sources": ["tribe_cortical"],
        "target_count": 1,
    }

    summary = summarize_event_target_screen(screen)

    assert summary["family_count"] == 1
    assert summary["selection_status"] == "diagnostic_families_not_frozen"
    assert summary["families"][0]["form"] == "causal"
    assert summary["families"][0]["positive_delta_fold_count"] == 2
