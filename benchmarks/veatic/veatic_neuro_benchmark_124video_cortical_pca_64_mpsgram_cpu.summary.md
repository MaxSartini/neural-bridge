# VEATIC Temporal Dynamics Benchmark

## How To Read This
This benchmark predicts VEATIC human valence/arousal annotations from cached TRIBE features and controls.
Raw state prediction is expected to be dominated by temporal persistence. The stronger test is whether real neuro features add value beyond autoregressive affect-history baselines and beyond shuffled/random controls.

## Dataset / Cache
- Accepted videos: 124
- Accepted rows: 10357
- Rejected cache entries: 0
- Feature sets: {'cortical_pca_64': 64}
- Run mode: cortical_fast_default
- Feature mode: cortical_pca_64
- Subcortical enabled: False

## Scientific Contract
- TRIBE extraction contract unchanged.
- Default subcortical policy: Subcortical disabled for cortical_fast_default. It remains available as explicit full_research/subcortical_ablation, but current OpenLAV/VEATIC evidence does not justify it as default compute.
- Subcortical remains available for explicit `full_research` and `subcortical_ablation` runs.
- Subcortical is disabled in the default run because current OpenLAV/VEATIC evidence does not show stable additive lift over compact cortical features, while it adds inference time, memory pressure, crash risk, and benchmark complexity.
- Expected benefit: lower runtime and memory pressure by skipping the separate subcortical model branch and ROI projection; exact speedup depends on video length and cache state.
- Event threshold: 0.05
- Autoregressive features use only current/past labels relative to the prediction horizon.
- Residualization is fit inside each split/fold only.

## mode_a_official_veatic_70_30

- Gap rows: 0
### arousal__future_state_p1s

Top conditions by `mae`:
- autoregressive: 0.0160
- autoregressive_plus_random_gaussian_cortical_pca_64: 0.0161
- residualized_autoregressive_plus_random_gaussian_cortical_pca_64: 0.0162
- autoregressive_plus_shuffled_cortical_pca_64: 0.0163
- residualized_autoregressive_plus_shuffled_cortical_pca_64: 0.0164
- residualized_autoregressive_plus_cortical_pca_64: 0.0166
- residualized_autoregressive_plus_shuffled_combined: mae=0.0160, rmse=0.0261, pearson=0.9956, spearman=0.9957

### arousal__future_change_p1s

Top conditions by `mae`:
- autoregressive: 0.0161
- autoregressive_plus_random_gaussian_cortical_pca_64: 0.0162
- residualized_autoregressive_plus_shuffled_cortical_pca_64: 0.0163
- residualized_autoregressive_plus_random_gaussian_cortical_pca_64: 0.0163
- autoregressive_plus_shuffled_cortical_pca_64: 0.0164
- zero_change_or_residual: 0.0166
- residualized_autoregressive_plus_shuffled_combined: mae=0.0161, rmse=0.0262, pearson=0.3098, spearman=0.2958

### arousal__residual_future_p1s_persistence

Top conditions by `mae`:
- autoregressive: 0.0161
- autoregressive_plus_random_gaussian_cortical_pca_64: 0.0162
- autoregressive_plus_shuffled_cortical_pca_64: 0.0162
- residualized_autoregressive_plus_random_gaussian_cortical_pca_64: 0.0163
- residualized_autoregressive_plus_shuffled_cortical_pca_64: 0.0163
- zero_change_or_residual: 0.0166
- residualized_autoregressive_plus_shuffled_combined: mae=0.0161, rmse=0.0262, pearson=0.3098, spearman=0.2958

### arousal__event_future_spike_1_3s

Top conditions by `f1`:
- autoregressive_plus_cortical_pca_64: 0.3865
- residualized_autoregressive_plus_cortical_pca_64: 0.3603
- residualized_autoregressive_plus_random_gaussian_cortical_pca_64: 0.3013
- autoregressive: 0.2948
- autoregressive_plus_random_gaussian_cortical_pca_64: 0.2926
- cortical_pca_64: 0.2904
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.7815, f1=0.2948

### arousal__event_future_drop_1_3s

Top conditions by `f1`:
- autoregressive: 0.3553
- autoregressive_plus_shuffled_cortical_pca_64: 0.3492
- autoregressive_plus_random_gaussian_cortical_pca_64: 0.3461
- residualized_autoregressive_plus_shuffled_cortical_pca_64: 0.3461
- residualized_autoregressive_plus_random_gaussian_cortical_pca_64: 0.3461
- residualized_autoregressive_plus_cortical_pca_64: 0.3277
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.8576, f1=0.3553

### arousal__event_trend_reversal_1_3s

Top conditions by `f1`:
- autoregressive_plus_shuffled_cortical_pca_64: 0.1493
- residualized_autoregressive_plus_shuffled_cortical_pca_64: 0.1493
- residualized_autoregressive_plus_random_gaussian_cortical_pca_64: 0.1294
- autoregressive: 0.1194
- autoregressive_plus_cortical_pca_64: 0.1194
- autoregressive_plus_random_gaussian_cortical_pca_64: 0.1194
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.9346, f1=0.1194

### arousal__event_peak_onset_1_3s

Top conditions by `f1`:
- autoregressive_plus_shuffled_cortical_pca_64: 0.1325
- residualized_autoregressive_plus_shuffled_cortical_pca_64: 0.1325
- autoregressive: 0.1192
- autoregressive_plus_cortical_pca_64: 0.1192
- autoregressive_plus_random_gaussian_cortical_pca_64: 0.1192
- residualized_autoregressive_plus_cortical_pca_64: 0.1192
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.9509, f1=0.1192

### arousal__event_recovery_onset_1_3s

Top conditions by `f1`:
- autoregressive_plus_cortical_pca_64: 0.1287
- autoregressive: 0.1170
- residualized_autoregressive_plus_random_gaussian_cortical_pca_64: 0.1170
- autoregressive_plus_shuffled_cortical_pca_64: 0.0936
- autoregressive_plus_random_gaussian_cortical_pca_64: 0.0936
- residualized_autoregressive_plus_shuffled_cortical_pca_64: 0.0936
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.9489, f1=0.1170

### valence__future_state_p1s

Top conditions by `mae`:
- persistence_current_value: 0.0171
- autoregressive_plus_cortical_pca_64: 0.0182
- autoregressive_plus_shuffled_cortical_pca_64: 0.0363
- autoregressive: 0.0365
- residualized_autoregressive_plus_shuffled_cortical_pca_64: 0.0366
- residualized_autoregressive_plus_random_gaussian_cortical_pca_64: 0.0368
- residualized_autoregressive_plus_shuffled_combined: mae=0.0365, rmse=0.0461, pearson=0.9964, spearman=0.9954

### valence__future_change_p1s

Top conditions by `mae`:
- mean_train: 0.0171
- zero_change_or_residual: 0.0171
- time_ridge: 0.0173
- random_gaussian_cortical_pca_64: 0.0174
- shuffled_cortical_pca_64: 0.0176
- cortical_pca_64: 0.0181
- residualized_autoregressive_plus_shuffled_combined: mae=0.0366, rmse=0.0462, pearson=0.2259, spearman=0.1887

### valence__residual_future_p1s_persistence

Top conditions by `mae`:
- mean_train: 0.0171
- zero_change_or_residual: 0.0171
- time_ridge: 0.0173
- shuffled_cortical_pca_64: 0.0174
- random_gaussian_cortical_pca_64: 0.0175
- cortical_pca_64: 0.0181
- residualized_autoregressive_plus_shuffled_combined: mae=0.0366, rmse=0.0462, pearson=0.2259, spearman=0.1887

### valence__event_future_drop_1_3s

Top conditions by `f1`:
- autoregressive_plus_shuffled_cortical_pca_64: 0.2063
- residualized_autoregressive_plus_random_gaussian_cortical_pca_64: 0.2063
- mean_train: 0.2019
- residualized_autoregressive_plus_cortical_pca_64: 0.1993
- residualized_autoregressive_plus_shuffled_cortical_pca_64: 0.1923
- autoregressive: 0.1899
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.7662, f1=0.1899

## mode_b_blocked_temporal_gap

- Gap rows: 1040
### arousal__future_state_p1s

Top conditions by `mae`:
- persistence_current_value: 0.0166
- residualized_autoregressive_plus_cortical_pca_64: 0.0251
- autoregressive_plus_random_gaussian_cortical_pca_64: 0.0261
- autoregressive: 0.0267
- residualized_autoregressive_plus_random_gaussian_cortical_pca_64: 0.0267
- autoregressive_plus_shuffled_cortical_pca_64: 0.0272
- residualized_autoregressive_plus_shuffled_combined: mae=0.0267, rmse=0.0345, pearson=0.9951, spearman=0.9953

### arousal__future_change_p1s

Top conditions by `mae`:
- zero_change_or_residual: 0.0166
- time_ridge: 0.0170
- mean_train: 0.0170
- shuffled_cortical_pca_64: 0.0171
- random_gaussian_cortical_pca_64: 0.0174
- cortical_pca_64: 0.0174
- residualized_autoregressive_plus_shuffled_combined: mae=0.0261, rmse=0.0340, pearson=0.2525, spearman=0.2141

### arousal__residual_future_p1s_persistence

Top conditions by `mae`:
- zero_change_or_residual: 0.0166
- time_ridge: 0.0170
- mean_train: 0.0170
- shuffled_cortical_pca_64: 0.0173
- random_gaussian_cortical_pca_64: 0.0173
- cortical_pca_64: 0.0174
- residualized_autoregressive_plus_shuffled_combined: mae=0.0261, rmse=0.0340, pearson=0.2525, spearman=0.2141

### arousal__event_future_spike_1_3s

Top conditions by `f1`:
- cortical_pca_64: 0.2814
- autoregressive_plus_cortical_pca_64: 0.2597
- residualized_autoregressive_plus_cortical_pca_64: 0.2381
- mean_train: 0.2342
- residualized_autoregressive_plus_random_gaussian_cortical_pca_64: 0.2186
- autoregressive: 0.2121
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.7537, f1=0.2121

### arousal__event_future_drop_1_3s

Top conditions by `f1`:
- cortical_pca_64: 0.2216
- mean_train: 0.2079
- autoregressive_plus_cortical_pca_64: 0.2036
- residualized_autoregressive_plus_cortical_pca_64: 0.2006
- autoregressive_plus_random_gaussian_cortical_pca_64: 0.1916
- autoregressive: 0.1826
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.8153, f1=0.1826

### arousal__event_trend_reversal_1_3s

Top conditions by `f1`:
- autoregressive_plus_cortical_pca_64: 0.1294
- autoregressive: 0.1194
- autoregressive_plus_random_gaussian_cortical_pca_64: 0.1194
- residualized_autoregressive_plus_cortical_pca_64: 0.1194
- autoregressive_plus_shuffled_cortical_pca_64: 0.1095
- residualized_autoregressive_plus_shuffled_cortical_pca_64: 0.1095
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.9346, f1=0.1194

### arousal__event_peak_onset_1_3s

Top conditions by `f1`:
- autoregressive: 0.1325
- autoregressive_plus_shuffled_cortical_pca_64: 0.1325
- residualized_autoregressive_plus_shuffled_cortical_pca_64: 0.1325
- autoregressive_plus_cortical_pca_64: 0.1192
- autoregressive_plus_random_gaussian_cortical_pca_64: 0.1192
- residualized_autoregressive_plus_cortical_pca_64: 0.1192
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.9516, f1=0.1325

### arousal__event_recovery_onset_1_3s

Top conditions by `f1`:
- autoregressive_plus_cortical_pca_64: 0.0702
- autoregressive_plus_shuffled_cortical_pca_64: 0.0585
- residualized_autoregressive_plus_shuffled_cortical_pca_64: 0.0585
- cortical_pca_64: 0.0468
- random_gaussian_cortical_pca_64: 0.0468
- autoregressive: 0.0351
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.9442, f1=0.0351

### valence__future_state_p1s

Top conditions by `mae`:
- persistence_current_value: 0.0171
- autoregressive_plus_cortical_pca_64: 0.0702
- autoregressive_plus_shuffled_cortical_pca_64: 0.1282
- residualized_autoregressive_plus_random_gaussian_cortical_pca_64: 0.1318
- residualized_autoregressive_plus_shuffled_cortical_pca_64: 0.1319
- autoregressive: 0.1319
- residualized_autoregressive_plus_shuffled_combined: mae=0.1319, rmse=0.1514, pearson=0.9826, spearman=0.9741

### valence__future_change_p1s

Top conditions by `mae`:
- mean_train: 0.0171
- zero_change_or_residual: 0.0171
- time_ridge: 0.0172
- shuffled_cortical_pca_64: 0.0174
- random_gaussian_cortical_pca_64: 0.0176
- cortical_pca_64: 0.0182
- residualized_autoregressive_plus_shuffled_combined: mae=0.1322, rmse=0.1518, pearson=0.1353, spearman=0.1112

### valence__residual_future_p1s_persistence

Top conditions by `mae`:
- mean_train: 0.0171
- zero_change_or_residual: 0.0171
- time_ridge: 0.0172
- random_gaussian_cortical_pca_64: 0.0174
- shuffled_cortical_pca_64: 0.0175
- cortical_pca_64: 0.0182
- residualized_autoregressive_plus_shuffled_combined: mae=0.1322, rmse=0.1518, pearson=0.1353, spearman=0.1112

### valence__event_future_drop_1_3s

Top conditions by `f1`:
- autoregressive_plus_shuffled_cortical_pca_64: 0.3540
- residualized_autoregressive_plus_random_gaussian_cortical_pca_64: 0.3517
- autoregressive: 0.3472
- residualized_autoregressive_plus_shuffled_cortical_pca_64: 0.3472
- residualized_autoregressive_plus_cortical_pca_64: 0.3427
- autoregressive_plus_random_gaussian_cortical_pca_64: 0.3360
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.8041, f1=0.3472

## mode_c_leave_video_out

### arousal__future_state_p1s

Top conditions by `mae`:
- autoregressive: 0.0170
- residualized_autoregressive_plus_cortical_pca_64: 0.0170
- autoregressive_plus_shuffled_cortical_pca_64: 0.0172
- residualized_autoregressive_plus_shuffled_cortical_pca_64: 0.0172
- autoregressive_plus_cortical_pca_64: 0.0172
- residualized_autoregressive_plus_random_gaussian_cortical_pca_64: 0.0172
- residualized_autoregressive_plus_shuffled_combined: mae=0.0170, pearson=0.9937, rmse=0.0277, spearman=0.9917

### arousal__future_change_p1s

Top conditions by `mae`:
- autoregressive: 0.0170
- residualized_autoregressive_plus_cortical_pca_64: 0.0170
- autoregressive_plus_random_gaussian_cortical_pca_64: 0.0172
- autoregressive_plus_cortical_pca_64: 0.0172
- residualized_autoregressive_plus_random_gaussian_cortical_pca_64: 0.0172
- residualized_autoregressive_plus_shuffled_cortical_pca_64: 0.0172
- residualized_autoregressive_plus_shuffled_combined: mae=0.0170, pearson=0.3333, rmse=0.0277, spearman=0.3407

### arousal__residual_future_p1s_persistence

Top conditions by `mae`:
- autoregressive: 0.0170
- residualized_autoregressive_plus_cortical_pca_64: 0.0170
- autoregressive_plus_random_gaussian_cortical_pca_64: 0.0172
- residualized_autoregressive_plus_shuffled_cortical_pca_64: 0.0172
- autoregressive_plus_shuffled_cortical_pca_64: 0.0172
- autoregressive_plus_cortical_pca_64: 0.0172
- residualized_autoregressive_plus_shuffled_combined: mae=0.0170, pearson=0.3333, rmse=0.0277, spearman=0.3407

### arousal__event_future_spike_1_3s

Top conditions by `f1`:
- residualized_autoregressive_plus_cortical_pca_64: 0.4164
- autoregressive_plus_cortical_pca_64: 0.4155
- autoregressive: 0.3899
- autoregressive_plus_random_gaussian_cortical_pca_64: 0.3860
- autoregressive_plus_shuffled_cortical_pca_64: 0.3840
- residualized_autoregressive_plus_shuffled_cortical_pca_64: 0.3835
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.7994, f1=0.3899

### arousal__event_future_drop_1_3s

Top conditions by `f1`:
- residualized_autoregressive_plus_random_gaussian_cortical_pca_64: 0.3345
- autoregressive: 0.3320
- autoregressive_plus_random_gaussian_cortical_pca_64: 0.3307
- residualized_autoregressive_plus_shuffled_cortical_pca_64: 0.3242
- autoregressive_plus_shuffled_cortical_pca_64: 0.3215
- residualized_autoregressive_plus_cortical_pca_64: 0.3129
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.8552, f1=0.3320

### arousal__event_trend_reversal_1_3s

Top conditions by `f1`:
- residualized_autoregressive_plus_cortical_pca_64: 0.1838
- autoregressive_plus_cortical_pca_64: 0.1824
- autoregressive: 0.1818
- autoregressive_plus_random_gaussian_cortical_pca_64: 0.1693
- residualized_autoregressive_plus_random_gaussian_cortical_pca_64: 0.1685
- autoregressive_plus_shuffled_cortical_pca_64: 0.1648
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.9342, f1=0.1818

### arousal__event_peak_onset_1_3s

Top conditions by `f1`:
- autoregressive: 0.1701
- residualized_autoregressive_plus_cortical_pca_64: 0.1662
- autoregressive_plus_random_gaussian_cortical_pca_64: 0.1661
- residualized_autoregressive_plus_random_gaussian_cortical_pca_64: 0.1632
- autoregressive_plus_cortical_pca_64: 0.1627
- autoregressive_plus_shuffled_cortical_pca_64: 0.1571
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.9508, f1=0.1701

### arousal__event_recovery_onset_1_3s

Top conditions by `f1`:
- autoregressive_plus_cortical_pca_64: 0.2086
- residualized_autoregressive_plus_cortical_pca_64: 0.2056
- autoregressive: 0.1964
- autoregressive_plus_shuffled_cortical_pca_64: 0.1937
- residualized_autoregressive_plus_shuffled_cortical_pca_64: 0.1903
- residualized_autoregressive_plus_random_gaussian_cortical_pca_64: 0.1901
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.9510, f1=0.1964

### valence__future_state_p1s

Top conditions by `mae`:
- autoregressive: 0.0182
- autoregressive_plus_random_gaussian_cortical_pca_64: 0.0184
- residualized_autoregressive_plus_random_gaussian_cortical_pca_64: 0.0185
- autoregressive_plus_shuffled_cortical_pca_64: 0.0186
- residualized_autoregressive_plus_shuffled_cortical_pca_64: 0.0186
- residualized_autoregressive_plus_cortical_pca_64: 0.0187
- residualized_autoregressive_plus_shuffled_combined: mae=0.0182, pearson=0.9948, rmse=0.0335, spearman=0.9936

### valence__future_change_p1s

Top conditions by `mae`:
- autoregressive: 0.0182
- residualized_autoregressive_plus_shuffled_cortical_pca_64: 0.0184
- autoregressive_plus_shuffled_cortical_pca_64: 0.0184
- autoregressive_plus_random_gaussian_cortical_pca_64: 0.0185
- residualized_autoregressive_plus_random_gaussian_cortical_pca_64: 0.0185
- residualized_autoregressive_plus_cortical_pca_64: 0.0187
- residualized_autoregressive_plus_shuffled_combined: mae=0.0182, pearson=0.3068, rmse=0.0335, spearman=0.3656

### valence__residual_future_p1s_persistence

Top conditions by `mae`:
- autoregressive: 0.0182
- autoregressive_plus_random_gaussian_cortical_pca_64: 0.0184
- residualized_autoregressive_plus_random_gaussian_cortical_pca_64: 0.0185
- residualized_autoregressive_plus_shuffled_cortical_pca_64: 0.0185
- autoregressive_plus_shuffled_cortical_pca_64: 0.0185
- residualized_autoregressive_plus_cortical_pca_64: 0.0187
- residualized_autoregressive_plus_shuffled_combined: mae=0.0182, pearson=0.3068, rmse=0.0335, spearman=0.3656

### valence__event_future_drop_1_3s

Top conditions by `f1`:
- autoregressive: 0.4411
- autoregressive_plus_random_gaussian_cortical_pca_64: 0.4326
- residualized_autoregressive_plus_cortical_pca_64: 0.4326
- residualized_autoregressive_plus_random_gaussian_cortical_pca_64: 0.4295
- autoregressive_plus_shuffled_cortical_pca_64: 0.4283
- residualized_autoregressive_plus_shuffled_cortical_pca_64: 0.4271
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.8235, f1=0.4411

## Lead / Lag Diagnostics

### arousal__future_change_p1s
- cortical_pca_64: best_offset=-2s, max_abs_pearson=0.1388, best_feature=pca64_component_2, best_pearson=-0.1388

### arousal__future_change_p2s
- cortical_pca_64: best_offset=-1s, max_abs_pearson=0.1586, best_feature=pca64_component_2, best_pearson=-0.1586

### arousal__future_change_p3s
- cortical_pca_64: best_offset=0s, max_abs_pearson=0.1734, best_feature=pca64_component_2, best_pearson=-0.1734

### valence__future_change_p1s
- cortical_pca_64: best_offset=0s, max_abs_pearson=0.0756, best_feature=pca64_component_52, best_pearson=-0.0756

### valence__future_change_p2s
- cortical_pca_64: best_offset=1s, max_abs_pearson=0.0712, best_feature=pca64_component_16, best_pearson=-0.0712

### valence__future_change_p3s
- cortical_pca_64: best_offset=1s, max_abs_pearson=0.0799, best_feature=pca64_component_16, best_pearson=-0.0799

## Feature / Permutation Importance Diagnostics

### mode_a_official_veatic_70_30

### mode_b_blocked_temporal_gap

## Interpretation Guardrails
- Do not claim neuro-additive value unless real neuro beats autoregressive-only and shuffled/random controls.
- Do not make anatomical emotion claims from feature importance; use feature contribution language only.
- This 20-video run is a gated diagnostic, not investor-grade proof.
