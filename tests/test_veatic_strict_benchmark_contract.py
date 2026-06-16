from backend.scripts import run_veatic_strict_benchmark as strict


def test_strict_contract_primary_plan_is_current_v2():
    parser = strict.build_parser()
    args = parser.parse_args(["--dry-run", "--primary-only"])
    contract = strict.contract_manifest(args)

    assert contract["schema_version"] == "veatic_strict_contract_v1"
    assert contract["manifest"].endswith("veatic_manifest_124_complete_20260616.jsonl")
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

    assert "cortical_global_delta" in contract["feature_modes"]
    assert "cortical_fast_default" in contract["feature_modes"]
    assert all("subcortical" not in feature for feature in contract["feature_modes"])
    assert all("OpenLAV" not in str(target) for target in contract["targets"])
