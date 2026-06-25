# VEATIC-124 Raw Representation Audit

## Scope

- No video re-encoding was performed.
- All representations use cached `tribe_raw_output.npz` cortical predictions.
- Frozen `cortical_pca64_delta` is retained as the comparison baseline.
- Current 0s alignment remains primary; causal windows are row-matched when they drop history rows.

## Run Summary

- Mode: `primary-audit`
- Cache videos audited: 124
- Raw outputs present: 124
- Suspicious/resampled videos: `83`
- Candidate count: 14
- Leakage audit status: `pass`

## Frozen Reference Baseline

| Split | Target | Thr | Frozen PR-AUC | AR PR-AUC | N | Events |
|---|---|---:|---:|---:|---:|---:|
| `blocked` | `arousal__future_spike_1_3s` | 0.0500 | 0.2536 | 0.1969 | 2956 | 392.0000 |
| `blocked` | `arousal__future_spike_1_3s` | 0.0750 | 0.1614 | 0.1151 | 2956 | 228.0000 |
| `blocked` | `arousal__future_change_p3s_movement` | 0.0500 | 0.3762 | 0.3376 | 2708 | 668.0000 |
| `blocked` | `arousal__future_change_p3s_movement` | 0.0750 | 0.2503 | 0.2729 | 2708 | 363.0000 |

## Promotion Verdict

- `pca_sequence_128_causal_past_2s_mean` on `arousal__future_spike_1_3s` @ `0.075`: grouped PR-AUC gain `0.0408`.
- `roi_parcel_features` on `arousal__future_change_p3s_movement` @ `0.05`: grouped PR-AUC gain `0.0400`.
- `pca_sequence_128_causal_past_2s_mean` on `arousal__future_spike_1_3s` @ `0.05`: grouped PR-AUC gain `0.0393`.
- `topk_vertices_512` on `arousal__future_spike_1_3s` @ `0.075`: grouped PR-AUC gain `0.0225`.
- `cortical_pca_64` on `arousal__future_spike_1_3s` @ `0.075`: grouped PR-AUC gain `0.0198`.
- `roi_parcel_features` on `arousal__future_spike_1_3s` @ `0.075`: grouped PR-AUC gain `0.0196`.
- `roi_parcel_features` on `arousal__future_spike_1_3s` @ `0.05`: grouped PR-AUC gain `0.0180`.
- `topk_vertices_512` on `arousal__future_spike_1_3s` @ `0.05`: grouped PR-AUC gain `0.0169`.
- `cortical_pca_64` on `arousal__future_spike_1_3s` @ `0.05`: grouped PR-AUC gain `0.0141`.
- `pca_current_128` on `arousal__future_spike_1_3s` @ `0.075`: grouped PR-AUC gain `0.0122`.

## Leaderboard

| Candidate | Split | Target | Thr | Head | PR-AUC | F1 | vs AR | vs shuffled | vs random |
|---|---|---|---:|---|---:|---:|---:|---:|---:|
| `roi_parcel_features` | `grouped_2` | `arousal__future_change_p3s_movement` | 0.0500 | `ridge_score` | 0.4967 | 0.4953 | -0.0011 | 0.0057 | 0.0012 |
| `topk_vertices_512` | `grouped_2` | `arousal__future_change_p3s_movement` | 0.0500 | `ridge_score` | 0.4919 | 0.4959 | -0.0059 | -0.0025 | 0.0283 |
| `roi_parcel_features` | `grouped_0` | `arousal__future_change_p3s_movement` | 0.0500 | `ridge_score` | 0.4890 | 0.5083 | 0.0095 | 0.0215 | 0.0268 |
| `cortical_pca_64` | `grouped_2` | `arousal__future_change_p3s_movement` | 0.0500 | `ridge_score` | 0.4843 | 0.4791 | -0.0138 | -0.0106 | -0.0031 |
| `pca_current_128` | `grouped_2` | `arousal__future_change_p3s_movement` | 0.0500 | `ridge_score` | 0.4753 | 0.4764 | -0.0229 | -0.0018 | -0.0044 |
| `cortical_pca64_delta` | `grouped_2` | `arousal__future_change_p3s_movement` | 0.0500 | `ridge_score` | 0.4727 | 0.4577 | -0.0255 | 0.0051 | 0.0076 |
| `topk_vertices_512` | `grouped_0` | `arousal__future_change_p3s_movement` | 0.0500 | `ridge_score` | 0.4723 | 0.4972 | -0.0105 | -0.0064 | 0.0424 |
| `cortical_pca_64` | `grouped_0` | `arousal__future_change_p3s_movement` | 0.0500 | `ridge_score` | 0.4683 | 0.4892 | -0.0118 | -0.0088 | -0.0035 |
| `pca_sequence_128_causal_past_2s_mean` | `grouped_2` | `arousal__future_change_p3s_movement` | 0.0500 | `ridge_score` | 0.4633 | 0.4628 | -0.0324 | -0.0160 | -0.0215 |
| `roi_parcel_features` | `grouped_video_mean` | `arousal__future_change_p3s_movement` | 0.0500 | `ridge_score` | 0.4621 | 0.4798 | 0.0007 | 0.0090 | 0.0107 |
| `pca_current_256` | `grouped_2` | `arousal__future_change_p3s_movement` | 0.0500 | `ridge_score` | 0.4618 | 0.4538 | -0.0360 | -0.0192 | -0.0169 |
| `pca_sequence_128_causal_past_2s_mean_std_last_slope` | `grouped_2` | `arousal__future_change_p3s_movement` | 0.0500 | `ridge_score` | 0.4617 | 0.4485 | -0.0327 | -0.0068 | 0.0061 |

## Output Files

- inventory_csv: `$NEURAL_BRIDGE_EXTERNAL_ROOT/outputs/veatic_124_raw_representation_audit_primary_20260620_152411/raw_cache_inventory.csv`
- inventory_summary_json: `$NEURAL_BRIDGE_EXTERNAL_ROOT/outputs/veatic_124_raw_representation_audit_primary_20260620_152411/raw_cache_inventory_summary.json`
- inventory_report_md: `$NEURAL_BRIDGE_EXTERNAL_ROOT/outputs/veatic_124_raw_representation_audit_primary_20260620_152411/raw_cache_inventory_report.md`
- candidates_json: `$NEURAL_BRIDGE_EXTERNAL_ROOT/outputs/veatic_124_raw_representation_audit_primary_20260620_152411/representation_candidates.json`
- all_results_csv: `$NEURAL_BRIDGE_EXTERNAL_ROOT/outputs/veatic_124_raw_representation_audit_primary_20260620_152411/representation_results_all.csv`
- primary_results_csv: `$NEURAL_BRIDGE_EXTERNAL_ROOT/outputs/veatic_124_raw_representation_audit_primary_20260620_152411/representation_results_primary.csv`
- grouped_results_csv: `$NEURAL_BRIDGE_EXTERNAL_ROOT/outputs/veatic_124_raw_representation_audit_primary_20260620_152411/grouped_video_results.csv`
- fixed_results_csv: `$NEURAL_BRIDGE_EXTERNAL_ROOT/outputs/veatic_124_raw_representation_audit_primary_20260620_152411/fixed_split_results.csv`
- control_results_csv: `$NEURAL_BRIDGE_EXTERNAL_ROOT/outputs/veatic_124_raw_representation_audit_primary_20260620_152411/control_results.csv`
- matched_context_csv: `$NEURAL_BRIDGE_EXTERNAL_ROOT/outputs/veatic_124_raw_representation_audit_primary_20260620_152411/matched_row_context_results.csv`
- stability_csv: `$NEURAL_BRIDGE_EXTERNAL_ROOT/outputs/veatic_124_raw_representation_audit_primary_20260620_152411/supervised_feature_selection_stability.csv`
- leaderboard_csv: `$NEURAL_BRIDGE_EXTERNAL_ROOT/outputs/veatic_124_raw_representation_audit_primary_20260620_152411/raw_vs_compressed_leaderboard.csv`
- promotion_json: `$NEURAL_BRIDGE_EXTERNAL_ROOT/outputs/veatic_124_raw_representation_audit_primary_20260620_152411/candidate_promotion_summary.json`
- leakage_json: `$NEURAL_BRIDGE_EXTERNAL_ROOT/outputs/veatic_124_raw_representation_audit_primary_20260620_152411/leakage_audit.json`
- run_manifest_json: `$NEURAL_BRIDGE_EXTERNAL_ROOT/outputs/veatic_124_raw_representation_audit_primary_20260620_152411/run_manifest.json`
- report_md: `$NEURAL_BRIDGE_EXTERNAL_ROOT/outputs/veatic_124_raw_representation_audit_primary_20260620_152411/representation_audit_report.md`
