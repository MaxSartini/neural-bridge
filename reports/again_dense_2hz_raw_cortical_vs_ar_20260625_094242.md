# AGAIN Dense 2Hz Raw Cortical vs AR

## Scope

- This benchmark uses the dense H100 AGAIN cache with true 0.5s row-level targets.
- Saved `time_seconds` values are used directly; no 1Hz fallback was used.
- No V-JEPA/TRIBE re-encoding was performed.
- Binary event thresholds are selected inside each train split from continuous future-label movement.
- This is not PCA bridge training.

## Coverage

- rows: `243575`
- labeled rows: `243441`
- targets: `['arousal_spike_rows_2_6_train_q90', 'arousal_delta_p2rows_train_q90', 'arousal_abs_delta_p4rows_train_q90']`
- validation protocols: `['blocked_temporal_70_30', 'grouped_video']`
- ridge alpha selection: `train_only_inner_validation`
- ridge alpha grid: `[0.1, 1.0, 10.0, 100.0]`
- ridge backend: per-fold `mlx_primal_conjugate_gradient`

## Lane Summary

- `arousal_abs_delta_p4rows_train_q90` / `blocked_temporal_70_30` / `AR_only`: folds `1`, PR-AUC `11.60%`, ROC-AUC `45.90%`, F1 `21.13%`
- `arousal_abs_delta_p4rows_train_q90` / `blocked_temporal_70_30` / `AR_plus_raw_cortical`: folds `1`, PR-AUC `13.33%`, ROC-AUC `52.51%`, F1 `20.60%`
- `arousal_abs_delta_p4rows_train_q90` / `blocked_temporal_70_30` / `AR_plus_raw_cortical_plus_temporal_diagnostics`: folds `1`, PR-AUC `13.42%`, ROC-AUC `52.78%`, F1 `20.27%`
- `arousal_abs_delta_p4rows_train_q90` / `blocked_temporal_70_30` / `AR_plus_temporal_diagnostics`: folds `1`, PR-AUC `11.95%`, ROC-AUC `48.01%`, F1 `20.51%`
- `arousal_abs_delta_p4rows_train_q90` / `blocked_temporal_70_30` / `quality_motion_luma_only_control`: folds `1`, PR-AUC `12.46%`, ROC-AUC `50.72%`, F1 `21.96%`
- `arousal_abs_delta_p4rows_train_q90` / `blocked_temporal_70_30` / `random_matched_feature_control`: folds `1`, PR-AUC `12.05%`, ROC-AUC `49.65%`, F1 `19.80%`
- `arousal_abs_delta_p4rows_train_q90` / `blocked_temporal_70_30` / `raw_cortical_only`: folds `1`, PR-AUC `14.56%`, ROC-AUC `54.94%`, F1 `21.67%`
- `arousal_abs_delta_p4rows_train_q90` / `blocked_temporal_70_30` / `shuffled_cortical_control`: folds `1`, PR-AUC `12.27%`, ROC-AUC `50.27%`, F1 `19.19%`
- `arousal_abs_delta_p4rows_train_q90` / `blocked_temporal_70_30` / `shuffled_temporal_diagnostics_control`: folds `1`, PR-AUC `12.16%`, ROC-AUC `50.02%`, F1 `21.64%`
- `arousal_abs_delta_p4rows_train_q90` / `blocked_temporal_70_30` / `temporal_diagnostics_only`: folds `1`, PR-AUC `13.42%`, ROC-AUC `52.66%`, F1 `21.09%`
- `arousal_abs_delta_p4rows_train_q90` / `blocked_temporal_70_30` / `timestamp_video_time_only_control`: folds `1`, PR-AUC `12.53%`, ROC-AUC `49.55%`, F1 `21.14%`
- `arousal_abs_delta_p4rows_train_q90` / `grouped_video` / `AR_only`: folds `5`, PR-AUC `11.82%`, ROC-AUC `53.98%`, F1 `19.55%`
- `arousal_abs_delta_p4rows_train_q90` / `grouped_video` / `AR_plus_raw_cortical`: folds `5`, PR-AUC `12.73%`, ROC-AUC `56.82%`, F1 `19.40%`
- `arousal_abs_delta_p4rows_train_q90` / `grouped_video` / `AR_plus_raw_cortical_plus_temporal_diagnostics`: folds `5`, PR-AUC `12.70%`, ROC-AUC `56.80%`, F1 `19.44%`
- `arousal_abs_delta_p4rows_train_q90` / `grouped_video` / `AR_plus_temporal_diagnostics`: folds `5`, PR-AUC `11.73%`, ROC-AUC `54.09%`, F1 `18.50%`
- `arousal_abs_delta_p4rows_train_q90` / `grouped_video` / `quality_motion_luma_only_control`: folds `5`, PR-AUC `10.67%`, ROC-AUC `51.18%`, F1 `18.22%`
- `arousal_abs_delta_p4rows_train_q90` / `grouped_video` / `random_matched_feature_control`: folds `5`, PR-AUC `10.12%`, ROC-AUC `50.10%`, F1 `17.59%`
- `arousal_abs_delta_p4rows_train_q90` / `grouped_video` / `raw_cortical_only`: folds `5`, PR-AUC `12.65%`, ROC-AUC `56.25%`, F1 `19.20%`
- `arousal_abs_delta_p4rows_train_q90` / `grouped_video` / `shuffled_cortical_control`: folds `5`, PR-AUC `10.07%`, ROC-AUC `49.86%`, F1 `17.42%`
- `arousal_abs_delta_p4rows_train_q90` / `grouped_video` / `shuffled_temporal_diagnostics_control`: folds `5`, PR-AUC `10.10%`, ROC-AUC `49.96%`, F1 `18.28%`
- `arousal_abs_delta_p4rows_train_q90` / `grouped_video` / `temporal_diagnostics_only`: folds `5`, PR-AUC `11.08%`, ROC-AUC `52.39%`, F1 `17.68%`
- `arousal_abs_delta_p4rows_train_q90` / `grouped_video` / `timestamp_video_time_only_control`: folds `5`, PR-AUC `11.30%`, ROC-AUC `52.63%`, F1 `18.28%`
- `arousal_delta_p2rows_train_q90` / `blocked_temporal_70_30` / `AR_only`: folds `1`, PR-AUC `26.19%`, ROC-AUC `70.70%`, F1 `34.64%`
- `arousal_delta_p2rows_train_q90` / `blocked_temporal_70_30` / `AR_plus_raw_cortical`: folds `1`, PR-AUC `20.25%`, ROC-AUC `65.15%`, F1 `26.39%`
- `arousal_delta_p2rows_train_q90` / `blocked_temporal_70_30` / `AR_plus_raw_cortical_plus_temporal_diagnostics`: folds `1`, PR-AUC `20.43%`, ROC-AUC `64.91%`, F1 `26.66%`
- `arousal_delta_p2rows_train_q90` / `blocked_temporal_70_30` / `AR_plus_temporal_diagnostics`: folds `1`, PR-AUC `23.93%`, ROC-AUC `67.71%`, F1 `29.65%`
- `arousal_delta_p2rows_train_q90` / `blocked_temporal_70_30` / `quality_motion_luma_only_control`: folds `1`, PR-AUC `10.38%`, ROC-AUC `49.36%`, F1 `19.41%`
- `arousal_delta_p2rows_train_q90` / `blocked_temporal_70_30` / `random_matched_feature_control`: folds `1`, PR-AUC `11.04%`, ROC-AUC `50.71%`, F1 `18.30%`
- `arousal_delta_p2rows_train_q90` / `blocked_temporal_70_30` / `raw_cortical_only`: folds `1`, PR-AUC `12.14%`, ROC-AUC `54.30%`, F1 `19.11%`
- `arousal_delta_p2rows_train_q90` / `blocked_temporal_70_30` / `shuffled_cortical_control`: folds `1`, PR-AUC `10.54%`, ROC-AUC `49.25%`, F1 `18.67%`
- `arousal_delta_p2rows_train_q90` / `blocked_temporal_70_30` / `shuffled_temporal_diagnostics_control`: folds `1`, PR-AUC `10.84%`, ROC-AUC `50.11%`, F1 `19.19%`
- `arousal_delta_p2rows_train_q90` / `blocked_temporal_70_30` / `temporal_diagnostics_only`: folds `1`, PR-AUC `10.73%`, ROC-AUC `49.27%`, F1 `16.70%`
- `arousal_delta_p2rows_train_q90` / `blocked_temporal_70_30` / `timestamp_video_time_only_control`: folds `1`, PR-AUC `10.57%`, ROC-AUC `47.76%`, F1 `15.57%`
- `arousal_delta_p2rows_train_q90` / `grouped_video` / `AR_only`: folds `5`, PR-AUC `20.84%`, ROC-AUC `65.66%`, F1 `30.10%`
- `arousal_delta_p2rows_train_q90` / `grouped_video` / `AR_plus_raw_cortical`: folds `5`, PR-AUC `20.19%`, ROC-AUC `66.09%`, F1 `27.46%`
- `arousal_delta_p2rows_train_q90` / `grouped_video` / `AR_plus_raw_cortical_plus_temporal_diagnostics`: folds `5`, PR-AUC `20.24%`, ROC-AUC `66.07%`, F1 `27.46%`
- `arousal_delta_p2rows_train_q90` / `grouped_video` / `AR_plus_temporal_diagnostics`: folds `5`, PR-AUC `20.57%`, ROC-AUC `65.41%`, F1 `28.34%`
- `arousal_delta_p2rows_train_q90` / `grouped_video` / `quality_motion_luma_only_control`: folds `5`, PR-AUC `10.38%`, ROC-AUC `50.63%`, F1 `18.23%`
- `arousal_delta_p2rows_train_q90` / `grouped_video` / `random_matched_feature_control`: folds `5`, PR-AUC `9.88%`, ROC-AUC `49.71%`, F1 `17.24%`
- `arousal_delta_p2rows_train_q90` / `grouped_video` / `raw_cortical_only`: folds `5`, PR-AUC `13.26%`, ROC-AUC `57.51%`, F1 `19.73%`
- `arousal_delta_p2rows_train_q90` / `grouped_video` / `shuffled_cortical_control`: folds `5`, PR-AUC `9.93%`, ROC-AUC `49.77%`, F1 `17.47%`
- `arousal_delta_p2rows_train_q90` / `grouped_video` / `shuffled_temporal_diagnostics_control`: folds `5`, PR-AUC `10.06%`, ROC-AUC `50.00%`, F1 `18.08%`
- `arousal_delta_p2rows_train_q90` / `grouped_video` / `temporal_diagnostics_only`: folds `5`, PR-AUC `11.05%`, ROC-AUC `52.51%`, F1 `17.93%`
- `arousal_delta_p2rows_train_q90` / `grouped_video` / `timestamp_video_time_only_control`: folds `5`, PR-AUC `10.39%`, ROC-AUC `50.13%`, F1 `18.02%`
- `arousal_spike_rows_2_6_train_q90` / `blocked_temporal_70_30` / `AR_only`: folds `1`, PR-AUC `20.36%`, ROC-AUC `67.04%`, F1 `19.60%`
- `arousal_spike_rows_2_6_train_q90` / `blocked_temporal_70_30` / `AR_plus_raw_cortical`: folds `1`, PR-AUC `16.77%`, ROC-AUC `62.08%`, F1 `23.64%`
- `arousal_spike_rows_2_6_train_q90` / `blocked_temporal_70_30` / `AR_plus_raw_cortical_plus_temporal_diagnostics`: folds `1`, PR-AUC `16.80%`, ROC-AUC `61.87%`, F1 `22.77%`
- `arousal_spike_rows_2_6_train_q90` / `blocked_temporal_70_30` / `AR_plus_temporal_diagnostics`: folds `1`, PR-AUC `18.63%`, ROC-AUC `64.28%`, F1 `24.19%`
- `arousal_spike_rows_2_6_train_q90` / `blocked_temporal_70_30` / `quality_motion_luma_only_control`: folds `1`, PR-AUC `10.67%`, ROC-AUC `48.90%`, F1 `18.50%`
- `arousal_spike_rows_2_6_train_q90` / `blocked_temporal_70_30` / `random_matched_feature_control`: folds `1`, PR-AUC `11.03%`, ROC-AUC `49.74%`, F1 `18.45%`
- `arousal_spike_rows_2_6_train_q90` / `blocked_temporal_70_30` / `raw_cortical_only`: folds `1`, PR-AUC `12.43%`, ROC-AUC `53.38%`, F1 `18.48%`
- `arousal_spike_rows_2_6_train_q90` / `blocked_temporal_70_30` / `shuffled_cortical_control`: folds `1`, PR-AUC `11.05%`, ROC-AUC `49.96%`, F1 `17.92%`
- `arousal_spike_rows_2_6_train_q90` / `blocked_temporal_70_30` / `shuffled_temporal_diagnostics_control`: folds `1`, PR-AUC `11.20%`, ROC-AUC `50.16%`, F1 `19.92%`
- `arousal_spike_rows_2_6_train_q90` / `blocked_temporal_70_30` / `temporal_diagnostics_only`: folds `1`, PR-AUC `11.33%`, ROC-AUC `50.75%`, F1 `15.35%`
- `arousal_spike_rows_2_6_train_q90` / `blocked_temporal_70_30` / `timestamp_video_time_only_control`: folds `1`, PR-AUC `11.14%`, ROC-AUC `47.21%`, F1 `17.00%`
- `arousal_spike_rows_2_6_train_q90` / `grouped_video` / `AR_only`: folds `5`, PR-AUC `14.73%`, ROC-AUC `58.44%`, F1 `20.56%`
- `arousal_spike_rows_2_6_train_q90` / `grouped_video` / `AR_plus_raw_cortical`: folds `5`, PR-AUC `17.03%`, ROC-AUC `63.82%`, F1 `23.48%`
- `arousal_spike_rows_2_6_train_q90` / `grouped_video` / `AR_plus_raw_cortical_plus_temporal_diagnostics`: folds `5`, PR-AUC `17.14%`, ROC-AUC `63.79%`, F1 `23.46%`
- `arousal_spike_rows_2_6_train_q90` / `grouped_video` / `AR_plus_temporal_diagnostics`: folds `5`, PR-AUC `15.04%`, ROC-AUC `60.91%`, F1 `20.79%`
- `arousal_spike_rows_2_6_train_q90` / `grouped_video` / `quality_motion_luma_only_control`: folds `5`, PR-AUC `10.29%`, ROC-AUC `49.62%`, F1 `17.84%`
- `arousal_spike_rows_2_6_train_q90` / `grouped_video` / `random_matched_feature_control`: folds `5`, PR-AUC `10.09%`, ROC-AUC `50.12%`, F1 `17.22%`
- `arousal_spike_rows_2_6_train_q90` / `grouped_video` / `raw_cortical_only`: folds `5`, PR-AUC `13.66%`, ROC-AUC `58.27%`, F1 `20.41%`
- `arousal_spike_rows_2_6_train_q90` / `grouped_video` / `shuffled_cortical_control`: folds `5`, PR-AUC `10.04%`, ROC-AUC `50.11%`, F1 `17.40%`
- `arousal_spike_rows_2_6_train_q90` / `grouped_video` / `shuffled_temporal_diagnostics_control`: folds `5`, PR-AUC `10.00%`, ROC-AUC `49.82%`, F1 `18.11%`
- `arousal_spike_rows_2_6_train_q90` / `grouped_video` / `temporal_diagnostics_only`: folds `5`, PR-AUC `10.97%`, ROC-AUC `52.61%`, F1 `17.75%`
- `arousal_spike_rows_2_6_train_q90` / `grouped_video` / `timestamp_video_time_only_control`: folds `5`, PR-AUC `10.30%`, ROC-AUC `50.20%`, F1 `18.11%`

## Promotion Gates

- `arousal_abs_delta_p4rows_train_q90`: raw beats AR `True`, AR+raw beats AR `True`, strict raw pass `True`, strict AR+raw pass `True`
- `arousal_delta_p2rows_train_q90`: raw beats AR `False`, AR+raw beats AR `False`, strict raw pass `False`, strict AR+raw pass `False`
- `arousal_spike_rows_2_6_train_q90`: raw beats AR `False`, AR+raw beats AR `True`, strict raw pass `False`, strict AR+raw pass `True`

## Limitations

- Raw cortical lanes use fixed label-free cortical block mean/std summaries for computational safety when modelling the 20,484 vertex output.
- Temporal diagnostics are non-PCA causal cache diagnostics, not learned bridge features.
- Promotion requires grouped-video wins over AR and nuisance controls; raw cortical losing is a valid outcome.

## Guardrails

- vjepa_encoding_run=`False`
- tribe_encoding_run=`False`
- pca_run=`False`
- models_trained=`True`
