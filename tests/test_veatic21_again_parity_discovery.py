from __future__ import annotations

import json

import numpy as np
import pytest

from backend.scripts import run_veatic21_again_parity_discovery as parity


def test_full_schedule_is_exact_270_members_and_90_ensembles() -> None:
    members = parity.member_keys()
    ensembles = parity.ensemble_keys()
    assert len(members) == parity.EXPECTED_MEMBER_ROWS == 270
    assert len(ensembles) == parity.EXPECTED_ENSEMBLE_ROWS == 90
    assert len(set(members)) == len(members)
    assert len(set(ensembles)) == len(ensembles)


def test_clean_and_enriched_views_preserve_fixed_feature_contract() -> None:
    width = 64
    enriched_width = 5 * width + 53 + 5 + 2
    values = np.arange(3 * enriched_width, dtype=np.float32).reshape(3, enriched_width)
    clean = parity.clean_or_enriched_view(
        values, pca_width=width, input_variant="again_clean"
    )
    enriched = parity.clean_or_enriched_view(
        values, pca_width=width, input_variant="veatic_enriched"
    )
    assert clean.shape == (3, 5 * width + 53)
    assert enriched.shape == values.shape
    assert np.array_equal(clean, values[:, : clean.shape[1]])
    assert np.array_equal(enriched, values)


def test_zero_event_video_is_not_assigned_a_fake_score() -> None:
    labels = np.asarray([0, 0, 0, 1, 0, 1], dtype=np.float32)
    valid = np.ones(6, dtype=bool)
    videos = np.asarray(["zero", "zero", "event", "event", "event", "event"])
    rows = np.arange(6, dtype=np.int64)
    stats = parity.event_panel_stats(
        labels=labels, valid=valid, videos=videos, global_rows=rows
    )
    assert stats["zero_event_video_count"] == 1
    assert stats["zero_event_video_ids"] == ["zero"]
    assert stats["pooled_valid_rows"] == 6
    assert stats["undefined_per_video_pr_auc_score_filled"] is False
    assert stats["zero_event_videos_excluded_from_pooled_negatives"] is False
    assert stats["event_metric_policy"] == parity.EVENT_METRIC_POLICY


def test_pr_auc_fails_closed_when_whole_panel_has_one_class() -> None:
    with pytest.raises(parity.ParityDiscoveryError, match="both classes"):
        parity.pr_auc(np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.float32))


def test_smoke_output_audit_requires_zero_event_and_veatic_pca_contract(tmp_path) -> None:
    outer, inner, recipe, seed = parity.member_keys(smoke=True)[0]
    row = {
        "outer_fold": outer,
        "inner_fold": inner,
        "recipe": recipe,
        "seed": seed,
        "real_pr_auc": 0.2,
        "ar_pr_auc": 0.19,
        "delta_vs_ar_pr_auc": 0.01,
        "outer_test_scores_used": False,
        "no_again_artifact_reuse": True,
        "event_metric_policy": parity.EVENT_METRIC_POLICY,
        "undefined_per_video_pr_auc_score_filled": False,
        "zero_event_videos_excluded_from_pooled_negatives": False,
        "heldout_label_digest": "labels",
        "heldout_valid_row_digest": "rows",
        "pooled_valid_rows": 100,
        "pooled_positive_rows": 10,
        "event_video_count": 3,
        "zero_event_video_count": 1,
        "pca_dataset_id": "veatic-124-v2.1",
        "pca_fresh_veatic_only": True,
        "pca_from_again": False,
        "pca_from_original_veatic": False,
    }
    (tmp_path / "member_rows.json").write_text(json.dumps([row]), encoding="utf-8")
    audit = parity.audit_outputs(tmp_path, smoke=True)
    assert audit["passed"] is True
