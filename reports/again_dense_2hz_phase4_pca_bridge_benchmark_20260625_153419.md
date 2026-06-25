# AGAIN Dense 2Hz Phase 4 PCA Bridge Benchmark

- Scope: dense true-2Hz AGAIN H100 TRIBE cortical cache.
- No raw video decode, V-JEPA run, or TRIBE run was performed.
- This is PCA bridge benchmarking, not learned-head training.
- targets: `['arousal_spike_rows_2_6_train_q90', 'arousal_delta_p2rows_train_q90', 'arousal_abs_delta_p4rows_train_q90']`
- validation protocols: `['grouped_video', 'blocked_temporal_70_30']`
- model lanes and controls: `['AR_only', 'PCA_only', 'AR_plus_PCA', 'residualized_AR_plus_PCA', 'PCA_plus_temporal_diagnostics', 'AR_plus_PCA_plus_temporal_diagnostics', 'residualized_AR_plus_PCA_plus_temporal_diagnostics', 'shuffled_PCA_control', 'shuffled_temporal_diagnostics_control', 'random_matched_PCA_control', 'timestamp_video_time_only_control', 'quality_motion_luma_only_control', 'AR_plus_shuffled_PCA_control', 'AR_plus_random_matched_PCA_control', 'AR_plus_timestamp_video_time_control', 'AR_plus_quality_motion_luma_control']`
- ridge alpha grid: `[0.1, 1.0, 10.0, 100.0]`

## Top Grouped-Video Rows

### arousal_abs_delta_p4rows_train_q90
- `residualized_AR_plus_PCA_plus_temporal_diagnostics` / `cortical_pca256_causal_past_0p5s_mean` / width `256`: PR-AUC `12.47%`, delta vs AR `0.65 pp`
- `residualized_AR_plus_PCA_plus_temporal_diagnostics` / `cortical_pca256_current` / width `256`: PR-AUC `12.47%`, delta vs AR `0.65 pp`
- `AR_plus_PCA_plus_temporal_diagnostics` / `cortical_pca256_causal_past_0p5s_mean` / width `256`: PR-AUC `12.45%`, delta vs AR `0.63 pp`
- `AR_plus_PCA_plus_temporal_diagnostics` / `cortical_pca256_current` / width `256`: PR-AUC `12.45%`, delta vs AR `0.63 pp`
- `AR_plus_PCA_plus_temporal_diagnostics` / `cortical_pca256_causal_past_1s_mean` / width `256`: PR-AUC `12.44%`, delta vs AR `0.62 pp`
- `residualized_AR_plus_PCA_plus_temporal_diagnostics` / `cortical_pca256_causal_past_1s_mean` / width `256`: PR-AUC `12.44%`, delta vs AR `0.62 pp`
- `residualized_AR_plus_PCA_plus_temporal_diagnostics` / `cortical_pca192_causal_past_0p5s_mean` / width `192`: PR-AUC `12.41%`, delta vs AR `0.60 pp`
- `residualized_AR_plus_PCA_plus_temporal_diagnostics` / `cortical_pca192_current` / width `192`: PR-AUC `12.41%`, delta vs AR `0.60 pp`
- `AR_plus_PCA_plus_temporal_diagnostics` / `cortical_pca192_current` / width `192`: PR-AUC `12.40%`, delta vs AR `0.58 pp`
- `AR_plus_PCA_plus_temporal_diagnostics` / `cortical_pca192_causal_past_0p5s_mean` / width `192`: PR-AUC `12.40%`, delta vs AR `0.58 pp`
- `AR_plus_PCA` / `cortical_pca256_causal_past_1s_mean` / width `256`: PR-AUC `12.39%`, delta vs AR `0.57 pp`
- `AR_plus_PCA_plus_temporal_diagnostics` / `cortical_pca192_causal_past_1s_mean` / width `192`: PR-AUC `12.38%`, delta vs AR `0.56 pp`
### arousal_delta_p2rows_train_q90
- `AR_plus_timestamp_video_time_control` / `cortical_pca192_causal_past_1s_std` / width `192`: PR-AUC `21.03%`, delta vs AR `0.19 pp`
- `AR_plus_timestamp_video_time_control` / `cortical_pca64_current` / width `64`: PR-AUC `21.03%`, delta vs AR `0.19 pp`
- `AR_plus_timestamp_video_time_control` / `cortical_pca192_causal_past_2s_std` / width `192`: PR-AUC `21.03%`, delta vs AR `0.19 pp`
- `AR_plus_timestamp_video_time_control` / `cortical_pca64_causal_past_0p5s_mean` / width `64`: PR-AUC `21.03%`, delta vs AR `0.19 pp`
- `AR_plus_timestamp_video_time_control` / `cortical_pca128_causal_past_2s_slope` / width `128`: PR-AUC `21.03%`, delta vs AR `0.19 pp`
- `AR_plus_timestamp_video_time_control` / `cortical_pca256_causal_past_3s_std` / width `256`: PR-AUC `21.03%`, delta vs AR `0.19 pp`
- `AR_plus_timestamp_video_time_control` / `cortical_pca192_causal_past_1s_slope` / width `192`: PR-AUC `21.03%`, delta vs AR `0.19 pp`
- `AR_plus_timestamp_video_time_control` / `cortical_pca256_causal_past_3s_slope` / width `256`: PR-AUC `21.03%`, delta vs AR `0.19 pp`
- `AR_plus_timestamp_video_time_control` / `cortical_pca128_causal_past_2s_std` / width `128`: PR-AUC `21.03%`, delta vs AR `0.19 pp`
- `AR_plus_timestamp_video_time_control` / `cortical_pca256_causal_past_3s_mean` / width `256`: PR-AUC `21.03%`, delta vs AR `0.19 pp`
- `AR_plus_timestamp_video_time_control` / `cortical_pca256_current` / width `256`: PR-AUC `21.03%`, delta vs AR `0.19 pp`
- `AR_plus_timestamp_video_time_control` / `cortical_pca256_causal_past_2s_std` / width `256`: PR-AUC `21.03%`, delta vs AR `0.19 pp`
### arousal_spike_rows_2_6_train_q90
- `AR_plus_PCA_plus_temporal_diagnostics` / `temporal_mean_2s_then_pca256` / width `256`: PR-AUC `17.16%`, delta vs AR `2.44 pp`
- `AR_plus_PCA_plus_temporal_diagnostics` / `cortical_pca256_causal_past_3s_mean` / width `256`: PR-AUC `17.07%`, delta vs AR `2.34 pp`
- `AR_plus_PCA` / `temporal_mean_2s_then_pca256` / width `256`: PR-AUC `17.00%`, delta vs AR `2.27 pp`
- `AR_plus_PCA` / `cortical_pca256_causal_past_3s_mean` / width `256`: PR-AUC `16.96%`, delta vs AR `2.23 pp`
- `AR_plus_PCA_plus_temporal_diagnostics` / `cortical_pca256_causal_past_2s_mean` / width `256`: PR-AUC `16.93%`, delta vs AR `2.20 pp`
- `AR_plus_PCA_plus_temporal_diagnostics` / `cortical_pca192_causal_past_3s_mean` / width `192`: PR-AUC `16.91%`, delta vs AR `2.19 pp`
- `AR_plus_PCA_plus_temporal_diagnostics` / `cortical_pca256_causal_past_1s_mean` / width `256`: PR-AUC `16.91%`, delta vs AR `2.19 pp`
- `AR_plus_PCA_plus_temporal_diagnostics` / `cortical_pca192_causal_past_2s_mean` / width `192`: PR-AUC `16.86%`, delta vs AR `2.13 pp`
- `AR_plus_PCA_plus_temporal_diagnostics` / `temporal_mean_2s_then_pca192` / width `192`: PR-AUC `16.85%`, delta vs AR `2.13 pp`
- `AR_plus_PCA_plus_temporal_diagnostics` / `cortical_pca256_causal_past_0p5s_mean` / width `256`: PR-AUC `16.83%`, delta vs AR `2.10 pp`
- `AR_plus_PCA_plus_temporal_diagnostics` / `cortical_pca256_current` / width `256`: PR-AUC `16.83%`, delta vs AR `2.10 pp`
- `AR_plus_PCA_plus_temporal_diagnostics` / `cortical_pca192_causal_past_1s_mean` / width `192`: PR-AUC `16.83%`, delta vs AR `2.10 pp`

## Limitations

- PCA uses deterministic randomized SVD with MLX-backed batch matmul where available.
- Blocked temporal support is diagnostic; grouped-video remains the primary gate.
- Phase 3 raw-cortical deltas are joined from the latest tracked report when available.
