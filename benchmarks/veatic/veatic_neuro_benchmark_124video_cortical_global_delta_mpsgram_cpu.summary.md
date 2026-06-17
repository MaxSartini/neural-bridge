# VEATIC Temporal Dynamics Benchmark

## How To Read This
This benchmark predicts VEATIC human valence/arousal annotations from cached TRIBE features and controls.
Raw state prediction is expected to be dominated by temporal persistence. The stronger test is whether real neuro features add value beyond autoregressive affect-history baselines and beyond shuffled/random controls.

## Dataset / Cache
- Accepted videos: 124
- Accepted rows: 10357
- Rejected cache entries: 0
- Feature sets: {'cortical_global_delta': 36}
- Run mode: cortical_fast_default
- Feature mode: cortical_global_delta
- Subcortical enabled: False

## Scientific Contract
- TRIBE extraction contract unchanged.
- Default subcortical policy: Subcortical disabled for cortical_fast_default. It remains available as explicit full_research/subcortical_ablation, but current VEATIC evidence does not justify it as default compute.
- Subcortical remains available for explicit `full_research` and `subcortical_ablation` runs.
- Subcortical is disabled in the default run because current VEATIC evidence does not show stable additive lift over compact cortical features, while it adds inference time, memory pressure, crash risk, and benchmark complexity.
- Expected benefit: lower runtime and memory pressure by skipping the separate subcortical model branch and ROI projection; exact speedup depends on video length and cache state.
- Event threshold: 0.05
- Autoregressive features use only current/past labels relative to the prediction horizon.
- Residualization is fit inside each split/fold only.

## mode_a_official_veatic_70_30

- Gap rows: 0
### arousal__future_state_p1s

Top conditions by `mae`:
- autoregressive: 0.0160
- autoregressive_plus_shuffled_cortical_global_delta: 0.0161
- residualized_autoregressive_plus_shuffled_cortical_global_delta: 0.0161
- autoregressive_plus_random_gaussian_cortical_global_delta: 0.0161
- residualized_autoregressive_plus_random_gaussian_cortical_global_delta: 0.0161
- residualized_autoregressive_plus_cortical_global_delta: 0.0163
- residualized_autoregressive_plus_shuffled_combined: mae=0.0160, rmse=0.0261, pearson=0.9956, spearman=0.9957

### arousal__future_change_p1s

Top conditions by `mae`:
- autoregressive_plus_shuffled_cortical_global_delta: 0.0161
- autoregressive: 0.0161
- residualized_autoregressive_plus_shuffled_cortical_global_delta: 0.0161
- autoregressive_plus_random_gaussian_cortical_global_delta: 0.0161
- residualized_autoregressive_plus_random_gaussian_cortical_global_delta: 0.0162
- residualized_autoregressive_plus_cortical_global_delta: 0.0164
- residualized_autoregressive_plus_shuffled_combined: mae=0.0161, rmse=0.0262, pearson=0.3098, spearman=0.2958

### arousal__residual_future_p1s_persistence

Top conditions by `mae`:
- autoregressive_plus_random_gaussian_cortical_global_delta: 0.0160
- autoregressive: 0.0161
- residualized_autoregressive_plus_random_gaussian_cortical_global_delta: 0.0161
- residualized_autoregressive_plus_shuffled_cortical_global_delta: 0.0162
- autoregressive_plus_shuffled_cortical_global_delta: 0.0163
- residualized_autoregressive_plus_cortical_global_delta: 0.0164
- residualized_autoregressive_plus_shuffled_combined: mae=0.0161, rmse=0.0262, pearson=0.3098, spearman=0.2958

### arousal__event_future_spike_1_3s

Top conditions by `f1`:
- autoregressive_plus_cortical_global_delta: 0.3231
- residualized_autoregressive_plus_cortical_global_delta: 0.3144
- autoregressive_plus_shuffled_cortical_global_delta: 0.3057
- autoregressive_plus_random_gaussian_cortical_global_delta: 0.3035
- residualized_autoregressive_plus_shuffled_cortical_global_delta: 0.2991
- autoregressive: 0.2948
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.7815, f1=0.2948

### arousal__event_future_drop_1_3s

Top conditions by `f1`:
- autoregressive_plus_random_gaussian_cortical_global_delta: 0.3583
- autoregressive: 0.3553
- autoregressive_plus_shuffled_cortical_global_delta: 0.3492
- residualized_autoregressive_plus_cortical_global_delta: 0.3492
- residualized_autoregressive_plus_shuffled_cortical_global_delta: 0.3492
- residualized_autoregressive_plus_random_gaussian_cortical_global_delta: 0.3461
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.8576, f1=0.3553

### arousal__event_trend_reversal_1_3s

Top conditions by `f1`:
- autoregressive_plus_shuffled_cortical_global_delta: 0.1393
- residualized_autoregressive_plus_shuffled_cortical_global_delta: 0.1294
- autoregressive: 0.1194
- autoregressive_plus_random_gaussian_cortical_global_delta: 0.1194
- residualized_autoregressive_plus_cortical_global_delta: 0.1194
- residualized_autoregressive_plus_random_gaussian_cortical_global_delta: 0.1194
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.9346, f1=0.1194

### arousal__event_peak_onset_1_3s

Top conditions by `f1`:
- autoregressive: 0.1192
- autoregressive_plus_cortical_global_delta: 0.1192
- autoregressive_plus_shuffled_cortical_global_delta: 0.1192
- autoregressive_plus_random_gaussian_cortical_global_delta: 0.1192
- residualized_autoregressive_plus_cortical_global_delta: 0.1192
- residualized_autoregressive_plus_shuffled_cortical_global_delta: 0.1192
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.9509, f1=0.1192

### arousal__event_recovery_onset_1_3s

Top conditions by `f1`:
- autoregressive_plus_cortical_global_delta: 0.1637
- autoregressive: 0.1170
- residualized_autoregressive_plus_shuffled_cortical_global_delta: 0.1170
- autoregressive_plus_shuffled_cortical_global_delta: 0.1053
- autoregressive_plus_random_gaussian_cortical_global_delta: 0.1053
- residualized_autoregressive_plus_cortical_global_delta: 0.1053
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.9489, f1=0.1170

### valence__future_state_p1s

Top conditions by `mae`:
- persistence_current_value: 0.0171
- autoregressive_plus_random_gaussian_cortical_global_delta: 0.0361
- autoregressive: 0.0365
- residualized_autoregressive_plus_random_gaussian_cortical_global_delta: 0.0366
- residualized_autoregressive_plus_cortical_global_delta: 0.0369
- residualized_autoregressive_plus_shuffled_cortical_global_delta: 0.0369
- residualized_autoregressive_plus_shuffled_combined: mae=0.0365, rmse=0.0461, pearson=0.9964, spearman=0.9954

### valence__future_change_p1s

Top conditions by `mae`:
- mean_train: 0.0171
- zero_change_or_residual: 0.0171
- time_ridge: 0.0173
- random_gaussian_cortical_global_delta: 0.0173
- shuffled_cortical_global_delta: 0.0173
- cortical_global_delta: 0.0174
- residualized_autoregressive_plus_shuffled_combined: mae=0.0366, rmse=0.0462, pearson=0.2259, spearman=0.1887

### valence__residual_future_p1s_persistence

Top conditions by `mae`:
- mean_train: 0.0171
- zero_change_or_residual: 0.0171
- shuffled_cortical_global_delta: 0.0172
- time_ridge: 0.0173
- random_gaussian_cortical_global_delta: 0.0173
- cortical_global_delta: 0.0174
- residualized_autoregressive_plus_shuffled_combined: mae=0.0366, rmse=0.0462, pearson=0.2259, spearman=0.1887

### valence__event_future_drop_1_3s

Top conditions by `f1`:
- cortical_global_delta: 0.2204
- mean_train: 0.2019
- autoregressive_plus_shuffled_cortical_global_delta: 0.2016
- residualized_autoregressive_plus_cortical_global_delta: 0.1993
- autoregressive_plus_cortical_global_delta: 0.1970
- residualized_autoregressive_plus_random_gaussian_cortical_global_delta: 0.1970
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.7662, f1=0.1899

## mode_b_blocked_temporal_gap

- Gap rows: 1040
### arousal__future_state_p1s

Top conditions by `mae`:
- persistence_current_value: 0.0166
- autoregressive_plus_cortical_global_delta: 0.0221
- autoregressive_plus_shuffled_cortical_global_delta: 0.0252
- residualized_autoregressive_plus_cortical_global_delta: 0.0260
- autoregressive_plus_random_gaussian_cortical_global_delta: 0.0263
- residualized_autoregressive_plus_shuffled_cortical_global_delta: 0.0266
- residualized_autoregressive_plus_shuffled_combined: mae=0.0267, rmse=0.0345, pearson=0.9951, spearman=0.9953

### arousal__future_change_p1s

Top conditions by `mae`:
- zero_change_or_residual: 0.0166
- time_ridge: 0.0170
- mean_train: 0.0170
- random_gaussian_cortical_global_delta: 0.0171
- cortical_global_delta: 0.0171
- shuffled_cortical_global_delta: 0.0174
- residualized_autoregressive_plus_shuffled_combined: mae=0.0261, rmse=0.0340, pearson=0.2525, spearman=0.2141

### arousal__residual_future_p1s_persistence

Top conditions by `mae`:
- zero_change_or_residual: 0.0166
- time_ridge: 0.0170
- mean_train: 0.0170
- random_gaussian_cortical_global_delta: 0.0170
- cortical_global_delta: 0.0171
- shuffled_cortical_global_delta: 0.0172
- residualized_autoregressive_plus_shuffled_combined: mae=0.0261, rmse=0.0340, pearson=0.2525, spearman=0.2141

### arousal__event_future_spike_1_3s

Top conditions by `f1`:
- cortical_global_delta: 0.2468
- mean_train: 0.2342
- autoregressive_plus_cortical_global_delta: 0.2273
- residualized_autoregressive_plus_cortical_global_delta: 0.2251
- residualized_autoregressive_plus_random_gaussian_cortical_global_delta: 0.2208
- autoregressive_plus_shuffled_cortical_global_delta: 0.2186
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.7537, f1=0.2121

### arousal__event_future_drop_1_3s

Top conditions by `f1`:
- mean_train: 0.2079
- autoregressive_plus_random_gaussian_cortical_global_delta: 0.1886
- autoregressive_plus_shuffled_cortical_global_delta: 0.1856
- residualized_autoregressive_plus_random_gaussian_cortical_global_delta: 0.1856
- autoregressive: 0.1826
- residualized_autoregressive_plus_shuffled_cortical_global_delta: 0.1826
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.8153, f1=0.1826

### arousal__event_trend_reversal_1_3s

Top conditions by `f1`:
- autoregressive_plus_cortical_global_delta: 0.1294
- autoregressive: 0.1194
- autoregressive_plus_shuffled_cortical_global_delta: 0.1194
- residualized_autoregressive_plus_cortical_global_delta: 0.1194
- residualized_autoregressive_plus_shuffled_cortical_global_delta: 0.1194
- autoregressive_plus_random_gaussian_cortical_global_delta: 0.1095
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.9346, f1=0.1194

### arousal__event_peak_onset_1_3s

Top conditions by `f1`:
- autoregressive: 0.1325
- autoregressive_plus_random_gaussian_cortical_global_delta: 0.1325
- residualized_autoregressive_plus_random_gaussian_cortical_global_delta: 0.1325
- autoregressive_plus_cortical_global_delta: 0.1192
- autoregressive_plus_shuffled_cortical_global_delta: 0.1192
- residualized_autoregressive_plus_cortical_global_delta: 0.1192
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.9516, f1=0.1325

### arousal__event_recovery_onset_1_3s

Top conditions by `f1`:
- autoregressive_plus_cortical_global_delta: 0.0702
- autoregressive: 0.0351
- autoregressive_plus_shuffled_cortical_global_delta: 0.0351
- autoregressive_plus_random_gaussian_cortical_global_delta: 0.0351
- residualized_autoregressive_plus_cortical_global_delta: 0.0351
- residualized_autoregressive_plus_shuffled_cortical_global_delta: 0.0351
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.9442, f1=0.0351

### valence__future_state_p1s

Top conditions by `mae`:
- persistence_current_value: 0.0171
- autoregressive_plus_cortical_global_delta: 0.1273
- autoregressive_plus_shuffled_cortical_global_delta: 0.1287
- residualized_autoregressive_plus_shuffled_cortical_global_delta: 0.1317
- autoregressive: 0.1319
- residualized_autoregressive_plus_random_gaussian_cortical_global_delta: 0.1319
- residualized_autoregressive_plus_shuffled_combined: mae=0.1319, rmse=0.1514, pearson=0.9826, spearman=0.9741

### valence__future_change_p1s

Top conditions by `mae`:
- mean_train: 0.0171
- zero_change_or_residual: 0.0171
- time_ridge: 0.0172
- shuffled_cortical_global_delta: 0.0172
- random_gaussian_cortical_global_delta: 0.0174
- cortical_global_delta: 0.0175
- residualized_autoregressive_plus_shuffled_combined: mae=0.1322, rmse=0.1518, pearson=0.1353, spearman=0.1112

### valence__residual_future_p1s_persistence

Top conditions by `mae`:
- mean_train: 0.0171
- zero_change_or_residual: 0.0171
- shuffled_cortical_global_delta: 0.0172
- time_ridge: 0.0172
- random_gaussian_cortical_global_delta: 0.0173
- cortical_global_delta: 0.0175
- residualized_autoregressive_plus_shuffled_combined: mae=0.1322, rmse=0.1518, pearson=0.1353, spearman=0.1112

### valence__event_future_drop_1_3s

Top conditions by `f1`:
- residualized_autoregressive_plus_cortical_global_delta: 0.3743
- autoregressive_plus_cortical_global_delta: 0.3653
- autoregressive_plus_random_gaussian_cortical_global_delta: 0.3563
- residualized_autoregressive_plus_random_gaussian_cortical_global_delta: 0.3563
- autoregressive_plus_shuffled_cortical_global_delta: 0.3517
- residualized_autoregressive_plus_shuffled_cortical_global_delta: 0.3495
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.8041, f1=0.3472

## mode_c_leave_video_out

### arousal__future_state_p1s

Top conditions by `mae`:
- autoregressive: 0.0170
- autoregressive_plus_shuffled_cortical_global_delta: 0.0171
- residualized_autoregressive_plus_shuffled_cortical_global_delta: 0.0171
- residualized_autoregressive_plus_cortical_global_delta: 0.0171
- autoregressive_plus_cortical_global_delta: 0.0171
- autoregressive_plus_random_gaussian_cortical_global_delta: 0.0171
- residualized_autoregressive_plus_shuffled_combined: mae=0.0170, pearson=0.9937, rmse=0.0277, spearman=0.9917

### arousal__future_change_p1s

Top conditions by `mae`:
- autoregressive: 0.0170
- residualized_autoregressive_plus_cortical_global_delta: 0.0171
- residualized_autoregressive_plus_random_gaussian_cortical_global_delta: 0.0171
- autoregressive_plus_shuffled_cortical_global_delta: 0.0171
- residualized_autoregressive_plus_shuffled_cortical_global_delta: 0.0171
- autoregressive_plus_cortical_global_delta: 0.0171
- residualized_autoregressive_plus_shuffled_combined: mae=0.0170, pearson=0.3333, rmse=0.0277, spearman=0.3407

### arousal__residual_future_p1s_persistence

Top conditions by `mae`:
- autoregressive: 0.0170
- residualized_autoregressive_plus_cortical_global_delta: 0.0171
- residualized_autoregressive_plus_random_gaussian_cortical_global_delta: 0.0171
- autoregressive_plus_cortical_global_delta: 0.0171
- autoregressive_plus_shuffled_cortical_global_delta: 0.0171
- residualized_autoregressive_plus_shuffled_cortical_global_delta: 0.0171
- residualized_autoregressive_plus_shuffled_combined: mae=0.0170, pearson=0.3333, rmse=0.0277, spearman=0.3407

### arousal__event_future_spike_1_3s

Top conditions by `f1`:
- residualized_autoregressive_plus_cortical_global_delta: 0.3946
- autoregressive_plus_cortical_global_delta: 0.3940
- autoregressive_plus_random_gaussian_cortical_global_delta: 0.3902
- autoregressive: 0.3899
- residualized_autoregressive_plus_shuffled_cortical_global_delta: 0.3846
- autoregressive_plus_shuffled_cortical_global_delta: 0.3845
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.7994, f1=0.3899

### arousal__event_future_drop_1_3s

Top conditions by `f1`:
- autoregressive_plus_cortical_global_delta: 0.3447
- residualized_autoregressive_plus_cortical_global_delta: 0.3428
- autoregressive_plus_random_gaussian_cortical_global_delta: 0.3349
- autoregressive: 0.3320
- residualized_autoregressive_plus_random_gaussian_cortical_global_delta: 0.3274
- autoregressive_plus_shuffled_cortical_global_delta: 0.3236
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.8552, f1=0.3320

### arousal__event_trend_reversal_1_3s

Top conditions by `f1`:
- residualized_autoregressive_plus_cortical_global_delta: 0.1869
- autoregressive_plus_cortical_global_delta: 0.1847
- autoregressive: 0.1818
- autoregressive_plus_random_gaussian_cortical_global_delta: 0.1794
- residualized_autoregressive_plus_random_gaussian_cortical_global_delta: 0.1716
- autoregressive_plus_shuffled_cortical_global_delta: 0.1692
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.9342, f1=0.1818

### arousal__event_peak_onset_1_3s

Top conditions by `f1`:
- autoregressive_plus_shuffled_cortical_global_delta: 0.1705
- residualized_autoregressive_plus_shuffled_cortical_global_delta: 0.1705
- autoregressive: 0.1701
- autoregressive_plus_random_gaussian_cortical_global_delta: 0.1697
- autoregressive_plus_cortical_global_delta: 0.1670
- residualized_autoregressive_plus_cortical_global_delta: 0.1670
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.9508, f1=0.1701

### arousal__event_recovery_onset_1_3s

Top conditions by `f1`:
- autoregressive_plus_cortical_global_delta: 0.2023
- autoregressive_plus_shuffled_cortical_global_delta: 0.1998
- residualized_autoregressive_plus_shuffled_cortical_global_delta: 0.1998
- autoregressive: 0.1964
- residualized_autoregressive_plus_cortical_global_delta: 0.1963
- residualized_autoregressive_plus_random_gaussian_cortical_global_delta: 0.1871
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.9510, f1=0.1964

### valence__future_state_p1s

Top conditions by `mae`:
- autoregressive: 0.0182
- residualized_autoregressive_plus_random_gaussian_cortical_global_delta: 0.0183
- residualized_autoregressive_plus_shuffled_cortical_global_delta: 0.0183
- autoregressive_plus_shuffled_cortical_global_delta: 0.0183
- autoregressive_plus_random_gaussian_cortical_global_delta: 0.0183
- residualized_autoregressive_plus_cortical_global_delta: 0.0183
- residualized_autoregressive_plus_shuffled_combined: mae=0.0182, pearson=0.9948, rmse=0.0335, spearman=0.9936

### valence__future_change_p1s

Top conditions by `mae`:
- autoregressive: 0.0182
- residualized_autoregressive_plus_shuffled_cortical_global_delta: 0.0183
- autoregressive_plus_shuffled_cortical_global_delta: 0.0183
- residualized_autoregressive_plus_cortical_global_delta: 0.0183
- autoregressive_plus_random_gaussian_cortical_global_delta: 0.0184
- autoregressive_plus_cortical_global_delta: 0.0184
- residualized_autoregressive_plus_shuffled_combined: mae=0.0182, pearson=0.3068, rmse=0.0335, spearman=0.3656

### valence__residual_future_p1s_persistence

Top conditions by `mae`:
- autoregressive: 0.0182
- residualized_autoregressive_plus_shuffled_cortical_global_delta: 0.0183
- autoregressive_plus_shuffled_cortical_global_delta: 0.0183
- residualized_autoregressive_plus_cortical_global_delta: 0.0183
- autoregressive_plus_random_gaussian_cortical_global_delta: 0.0184
- residualized_autoregressive_plus_random_gaussian_cortical_global_delta: 0.0184
- residualized_autoregressive_plus_shuffled_combined: mae=0.0182, pearson=0.3068, rmse=0.0335, spearman=0.3656

### valence__event_future_drop_1_3s

Top conditions by `f1`:
- autoregressive: 0.4411
- autoregressive_plus_cortical_global_delta: 0.4410
- residualized_autoregressive_plus_cortical_global_delta: 0.4410
- autoregressive_plus_shuffled_cortical_global_delta: 0.4400
- residualized_autoregressive_plus_shuffled_cortical_global_delta: 0.4381
- autoregressive_plus_random_gaussian_cortical_global_delta: 0.4317
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.8235, f1=0.4411

## Lead / Lag Diagnostics

### arousal__future_change_p1s
- cortical_global_delta: best_offset=2s, max_abs_pearson=0.1363, best_feature=rollmean3_cortical_p95_abs, best_pearson=0.1363

### arousal__future_change_p2s
- cortical_global_delta: best_offset=2s, max_abs_pearson=0.1634, best_feature=rollmean3_cortical_p95_abs, best_pearson=0.1634

### arousal__future_change_p3s
- cortical_global_delta: best_offset=3s, max_abs_pearson=0.1782, best_feature=rollmean3_cortical_p95_abs, best_pearson=0.1782

### valence__future_change_p1s
- cortical_global_delta: best_offset=-3s, max_abs_pearson=0.0508, best_feature=rollmean3_cortical_mean, best_pearson=-0.0508

### valence__future_change_p2s
- cortical_global_delta: best_offset=-3s, max_abs_pearson=0.0615, best_feature=rollmean3_cortical_mean, best_pearson=-0.0615

### valence__future_change_p3s
- cortical_global_delta: best_offset=-3s, max_abs_pearson=0.0697, best_feature=rollmean3_cortical_mean, best_pearson=-0.0697

## Feature / Permutation Importance Diagnostics

### mode_a_official_veatic_70_30

#### arousal__event_future_spike_1_3s
- autoregressive_plus_cortical_global: metric=f1, base=0.3384, top_perm=ar_history_min (0.1332), group_importance={'autoregressive': 0.7052401746724892, 'cortical_global': 0.1659388646288209}

#### valence__event_future_drop_1_3s
- autoregressive_plus_cortical_global: metric=f1, base=0.1993, top_perm=ar_history_std (0.1055), group_importance={'autoregressive': 0.46424384525205176, 'cortical_global': 0.03985932004689338}

#### arousal__residual_future_p1s_persistence
- autoregressive_plus_cortical_global: metric=mae, base=0.0167, top_perm=ar_history_max (0.1210), group_importance={'autoregressive': 0.2103043444423339, 'cortical_global': 0.005491813556315256}

#### valence__residual_future_p1s_persistence
- autoregressive_plus_cortical_global: metric=mae, base=0.0364, top_perm=ar_history_min (0.0762), group_importance={'autoregressive': 0.14718080270292985, 'cortical_global': 0.001052442796217086}

### mode_b_blocked_temporal_gap

#### arousal__event_future_spike_1_3s
- autoregressive_plus_cortical_global: metric=f1, base=0.2121, top_perm=ar_history_std (0.0736), group_importance={'autoregressive': 0.22727272727272702, 'cortical_global': 0.045454545454545386}

#### valence__event_future_drop_1_3s
- autoregressive_plus_cortical_global: metric=f1, base=0.3653, top_perm=ar_lag_3s (0.2277), group_importance={'autoregressive': 1.269447576099211, 'cortical_global': 0.15783540022547926}

#### arousal__residual_future_p1s_persistence
- autoregressive_plus_cortical_global: metric=mae, base=0.0271, top_perm=ar_history_max (0.1184), group_importance={'autoregressive': 0.18155288947915935, 'cortical_global': 0.0026464796005009443}

#### valence__residual_future_p1s_persistence
- autoregressive_plus_cortical_global: metric=mae, base=0.1284, top_perm=ar_history_min (0.0421), group_importance={'autoregressive': 0.08187597597246674, 'cortical_global': 6.252243656815204e-05}

## Interpretation Guardrails
- Do not claim neuro-additive value unless real neuro beats autoregressive-only and shuffled/random controls.
- Do not make anatomical emotion claims from feature importance; use feature contribution language only.
- This 20-video run is a gated diagnostic, not investor-grade proof.
