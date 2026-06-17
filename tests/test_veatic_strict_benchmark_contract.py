import json

import numpy as np

from backend.scripts import run_veatic_strict_benchmark as strict


def test_strict_contract_primary_plan_is_current_v2():
    parser = strict.build_parser()
    args = parser.parse_args(["--dry-run", "--primary-only"])
    contract = strict.contract_manifest(args)

    assert contract["schema_version"] == "veatic_strict_contract_v1"
    assert contract["manifest"].endswith("veatic_manifest_124_complete_20260616.jsonl")
    assert contract["modality_contract"]["current_v2_cache_scope"] == "video_dominant_visual_cortical_cache"
    assert contract["modality_contract"]["required_for_current_v2_claim"] == ["video"]
    assert contract["feature_modes"] == ["cortical_pca64_delta", "cortical_pca_64"]
    assert "grouped_video_5_fold" in contract["splits"]
    assert contract["timing_policy"]["primary_alignment"] == "current_0s"
    assert contract["timing_policy"]["offset_grid_usage"] == "diagnostic_only"
    assert "Balanced event-vs-stable rows carry event-conditioned PR-AUC claims." in contract["leakage_rules"]
    controls = set(contract["controls"])
    assert "split_local_shuffled_cortical_rows" in controls
    assert "split_local_random_gaussian_features" in controls
    assert "label_shuffle_within_video" in controls
    assert "label_shuffle_across_videos" in controls
    assert "feature_shuffle_within_video" in controls
    assert "feature_shuffle_across_videos" in controls
    assert "blocked_temporal_gap_holdout" in controls
    assert "official_70_30_holdout" in controls
    assert "grouped_video_k_fold_holdout" in controls
    assert "zero_change_baseline" in controls
    assert "single_backend_policy" in controls


def test_strict_contract_full_plan_keeps_secondary_rows_but_not_descoped_modalities():
    parser = strict.build_parser()
    args = parser.parse_args(["--dry-run"])
    contract = strict.contract_manifest(args)

    feature_modes = set(contract["feature_modes"])
    assert {"cortical_global_delta", "cortical_fast_default"}.issubset(feature_modes)
    assert feature_modes <= {
        "cortical_pca64_delta",
        "cortical_pca_64",
        "cortical_global_delta",
        "cortical_global",
        "cortical_fast_default",
    }
    target_names = {target["target"] for target in contract["targets"]}
    assert target_names <= {
        "arousal__future_spike_1_3s",
        "arousal__future_change_p2s_movement",
        "arousal__future_change_p3s_movement",
    }


def test_modality_coverage_reports_video_only_and_multimodal_cache_rows(tmp_path):
    video_only = tmp_path / "0"
    multimodal = tmp_path / "83"
    video_only.mkdir()
    multimodal.mkdir()

    np.savez(video_only / "tribe_raw_output.npz", predictions=np.zeros((1, 3)), modality_missing_flags=np.array([1, 1, 0]))
    np.savez(multimodal / "tribe_raw_output.npz", predictions=np.zeros((1, 3)), modality_missing_flags=np.array([0, 0, 0]))
    (video_only / "tribe_summary.json").write_text(
        json.dumps({"event_quality": {"type_counts": {"Video": 1}, "missing_text": True, "missing_audio": True, "missing_video": False}}),
        encoding="utf-8",
    )
    (multimodal / "tribe_summary.json").write_text(
        json.dumps({"event_quality": {"type_counts": {"Video": 1, "Audio": 1, "Word": 2}, "missing_text": False, "missing_audio": False, "missing_video": False}}),
        encoding="utf-8",
    )

    coverage = strict.summarize_modality_coverage(tmp_path)

    assert coverage["raw_output_count"] == 2
    assert coverage["video_only_count"] == 1
    assert coverage["multimodal_text_audio_video_count"] == 1
    assert coverage["coverage_rows"] == [
        {
            "present_modalities": "text+audio+video",
            "missing_modalities": "none",
            "count": 1,
            "example_video_id": "83",
        },
        {
            "present_modalities": "video",
            "missing_modalities": "text+audio",
            "count": 1,
            "example_video_id": "0",
        },
    ]
