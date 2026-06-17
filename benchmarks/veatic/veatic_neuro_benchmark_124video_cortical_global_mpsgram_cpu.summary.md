# VEATIC Temporal Dynamics Benchmark

## How To Read This
This benchmark predicts VEATIC human valence/arousal annotations from cached TRIBE features and controls.
Raw state prediction is expected to be dominated by temporal persistence. The stronger test is whether real neuro features add value beyond autoregressive affect-history baselines and beyond shuffled/random controls.

## Dataset / Cache
- Accepted videos: 124
- Accepted rows: 10357
- Rejected cache entries: 0
- Feature sets: {'cortical_global': 6}
- Run mode: cortical_fast_default
- Feature mode: cortical_global
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
- residualized_autoregressive_plus_shuffled_cortical_global: 0.0160
- autoregressive_plus_shuffled_cortical_global: 0.0160
- residualized_autoregressive_plus_random_gaussian_cortical_global: 0.0160
- autoregressive_plus_random_gaussian_cortical_global: 0.0160
- residualized_autoregressive_plus_cortical_global: 0.0162
- residualized_autoregressive_plus_shuffled_combined: mae=0.0166, rmse=0.0265, pearson=0.9955, spearman=0.9957

### arousal__future_change_p1s

Top conditions by `mae`:
- autoregressive_plus_random_gaussian_cortical_global: 0.0161
- autoregressive_plus_shuffled_cortical_global: 0.0161
- residualized_autoregressive_plus_shuffled_cortical_global: 0.0161
- autoregressive: 0.0161
- residualized_autoregressive_plus_random_gaussian_cortical_global: 0.0161
- residualized_autoregressive_plus_cortical_global: 0.0163
- residualized_autoregressive_plus_shuffled_combined: mae=0.0167, rmse=0.0266, pearson=0.3065, spearman=0.3003

### arousal__residual_future_p1s_persistence

Top conditions by `mae`:
- residualized_autoregressive_plus_random_gaussian_cortical_global: 0.0161
- autoregressive: 0.0161
- autoregressive_plus_shuffled_cortical_global: 0.0161
- residualized_autoregressive_plus_shuffled_cortical_global: 0.0161
- autoregressive_plus_random_gaussian_cortical_global: 0.0161
- residualized_autoregressive_plus_cortical_global: 0.0163
- residualized_autoregressive_plus_shuffled_combined: mae=0.0167, rmse=0.0266, pearson=0.3065, spearman=0.3003

### arousal__event_future_spike_1_3s

Top conditions by `f1`:
- autoregressive_plus_cortical_global: 0.3384
- residualized_autoregressive_plus_cortical_global: 0.3210
- autoregressive_plus_random_gaussian_cortical_global: 0.2991
- residualized_autoregressive_plus_random_gaussian_cortical_global: 0.2969
- autoregressive: 0.2948
- autoregressive_plus_shuffled_cortical_global: 0.2904
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.7950, f1=0.3384

### arousal__event_future_drop_1_3s

Top conditions by `f1`:
- autoregressive_plus_cortical_global: 0.3737
- residualized_autoregressive_plus_cortical_global: 0.3706
- autoregressive_plus_shuffled_cortical_global: 0.3645
- residualized_autoregressive_plus_shuffled_cortical_global: 0.3645
- residualized_autoregressive_plus_random_gaussian_cortical_global: 0.3614
- autoregressive: 0.3553
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.8616, f1=0.3737

### arousal__event_trend_reversal_1_3s

Top conditions by `f1`:
- autoregressive_plus_shuffled_cortical_global: 0.1294
- autoregressive_plus_random_gaussian_cortical_global: 0.1294
- residualized_autoregressive_plus_shuffled_cortical_global: 0.1294
- autoregressive: 0.1194
- autoregressive_plus_cortical_global: 0.1194
- residualized_autoregressive_plus_cortical_global: 0.1194
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.9346, f1=0.1194

### arousal__event_peak_onset_1_3s

Top conditions by `f1`:
- autoregressive: 0.1192
- autoregressive_plus_cortical_global: 0.1192
- autoregressive_plus_shuffled_cortical_global: 0.1192
- autoregressive_plus_random_gaussian_cortical_global: 0.1192
- residualized_autoregressive_plus_cortical_global: 0.1192
- residualized_autoregressive_plus_shuffled_cortical_global: 0.1192
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.9509, f1=0.1192

### arousal__event_recovery_onset_1_3s

Top conditions by `f1`:
- autoregressive_plus_cortical_global: 0.1287
- autoregressive_plus_random_gaussian_cortical_global: 0.1287
- autoregressive: 0.1170
- autoregressive_plus_shuffled_cortical_global: 0.1170
- residualized_autoregressive_plus_cortical_global: 0.1170
- residualized_autoregressive_plus_shuffled_cortical_global: 0.1170
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.9496, f1=0.1287

### valence__future_state_p1s

Top conditions by `mae`:
- persistence_current_value: 0.0171
- autoregressive_plus_cortical_global: 0.0364
- autoregressive_plus_random_gaussian_cortical_global: 0.0365
- autoregressive: 0.0365
- residualized_autoregressive_plus_random_gaussian_cortical_global: 0.0365
- residualized_autoregressive_plus_shuffled_cortical_global: 0.0365
- residualized_autoregressive_plus_shuffled_combined: mae=0.0364, rmse=0.0459, pearson=0.9965, spearman=0.9955

### valence__future_change_p1s

Top conditions by `mae`:
- mean_train: 0.0171
- shuffled_cortical_global: 0.0171
- zero_change_or_residual: 0.0171
- random_gaussian_cortical_global: 0.0171
- cortical_global: 0.0172
- time_ridge: 0.0173
- residualized_autoregressive_plus_shuffled_combined: mae=0.0364, rmse=0.0460, pearson=0.2292, spearman=0.1874

### valence__residual_future_p1s_persistence

Top conditions by `mae`:
- mean_train: 0.0171
- zero_change_or_residual: 0.0171
- random_gaussian_cortical_global: 0.0171
- shuffled_cortical_global: 0.0171
- cortical_global: 0.0172
- time_ridge: 0.0173
- residualized_autoregressive_plus_shuffled_combined: mae=0.0364, rmse=0.0460, pearson=0.2292, spearman=0.1874

### valence__event_future_drop_1_3s

Top conditions by `f1`:
- cortical_global: 0.2345
- mean_train: 0.2019
- autoregressive_plus_cortical_global: 0.1993
- residualized_autoregressive_plus_cortical_global: 0.1993
- autoregressive_plus_random_gaussian_cortical_global: 0.1946
- residualized_autoregressive_plus_shuffled_cortical_global: 0.1946
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.7689, f1=0.1993

## mode_b_blocked_temporal_gap

- Gap rows: 1040
### arousal__future_state_p1s

Top conditions by `mae`:
- persistence_current_value: 0.0166
- autoregressive_plus_shuffled_cortical_global: 0.0259
- residualized_autoregressive_plus_cortical_global: 0.0262
- autoregressive_plus_random_gaussian_cortical_global: 0.0266
- residualized_autoregressive_plus_shuffled_cortical_global: 0.0267
- autoregressive: 0.0267
- residualized_autoregressive_plus_shuffled_combined: mae=0.0277, rmse=0.0356, pearson=0.9950, spearman=0.9952

### arousal__future_change_p1s

Top conditions by `mae`:
- zero_change_or_residual: 0.0166
- time_ridge: 0.0170
- mean_train: 0.0170
- shuffled_cortical_global: 0.0170
- random_gaussian_cortical_global: 0.0171
- cortical_global: 0.0171
- residualized_autoregressive_plus_shuffled_combined: mae=0.0271, rmse=0.0350, pearson=0.2517, spearman=0.2186

### arousal__residual_future_p1s_persistence

Top conditions by `mae`:
- zero_change_or_residual: 0.0166
- shuffled_cortical_global: 0.0170
- time_ridge: 0.0170
- mean_train: 0.0170
- random_gaussian_cortical_global: 0.0170
- cortical_global: 0.0171
- residualized_autoregressive_plus_shuffled_combined: mae=0.0271, rmse=0.0350, pearson=0.2517, spearman=0.2186

### arousal__event_future_spike_1_3s

Top conditions by `f1`:
- mean_train: 0.2342
- cortical_global: 0.2316
- residualized_autoregressive_plus_cortical_global: 0.2251
- autoregressive_plus_shuffled_cortical_global: 0.2165
- residualized_autoregressive_plus_shuffled_cortical_global: 0.2165
- residualized_autoregressive_plus_random_gaussian_cortical_global: 0.2165
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.7537, f1=0.2121

### arousal__event_future_drop_1_3s

Top conditions by `f1`:
- mean_train: 0.2079
- residualized_autoregressive_plus_cortical_global: 0.1916
- autoregressive: 0.1826
- autoregressive_plus_cortical_global: 0.1826
- autoregressive_plus_shuffled_cortical_global: 0.1826
- autoregressive_plus_random_gaussian_cortical_global: 0.1826
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.8153, f1=0.1826

### arousal__event_trend_reversal_1_3s

Top conditions by `f1`:
- autoregressive: 0.1194
- autoregressive_plus_shuffled_cortical_global: 0.1194
- residualized_autoregressive_plus_shuffled_cortical_global: 0.1194
- residualized_autoregressive_plus_random_gaussian_cortical_global: 0.1194
- autoregressive_plus_random_gaussian_cortical_global: 0.1095
- residualized_autoregressive_plus_cortical_global: 0.1095
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.9332, f1=0.0995

### arousal__event_peak_onset_1_3s

Top conditions by `f1`:
- autoregressive: 0.1325
- autoregressive_plus_shuffled_cortical_global: 0.1325
- autoregressive_plus_random_gaussian_cortical_global: 0.1325
- residualized_autoregressive_plus_shuffled_cortical_global: 0.1325
- autoregressive_plus_cortical_global: 0.1192
- residualized_autoregressive_plus_cortical_global: 0.1192
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.9509, f1=0.1192

### arousal__event_recovery_onset_1_3s

Top conditions by `f1`:
- autoregressive: 0.0351
- random_gaussian_cortical_global: 0.0351
- autoregressive_plus_cortical_global: 0.0351
- autoregressive_plus_shuffled_cortical_global: 0.0351
- autoregressive_plus_random_gaussian_cortical_global: 0.0351
- residualized_autoregressive_plus_cortical_global: 0.0351
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.9442, f1=0.0351

### valence__future_state_p1s

Top conditions by `mae`:
- persistence_current_value: 0.0171
- autoregressive_plus_cortical_global: 0.1281
- autoregressive_plus_shuffled_cortical_global: 0.1318
- autoregressive_plus_random_gaussian_cortical_global: 0.1318
- autoregressive: 0.1319
- residualized_autoregressive_plus_random_gaussian_cortical_global: 0.1319
- residualized_autoregressive_plus_shuffled_combined: mae=0.1281, rmse=0.1472, pearson=0.9834, spearman=0.9754

### valence__future_change_p1s

Top conditions by `mae`:
- mean_train: 0.0171
- zero_change_or_residual: 0.0171
- shuffled_cortical_global: 0.0171
- random_gaussian_cortical_global: 0.0171
- time_ridge: 0.0172
- cortical_global: 0.0173
- residualized_autoregressive_plus_shuffled_combined: mae=0.1284, rmse=0.1476, pearson=0.1373, spearman=0.1119

### valence__residual_future_p1s_persistence

Top conditions by `mae`:
- mean_train: 0.0171
- zero_change_or_residual: 0.0171
- shuffled_cortical_global: 0.0171
- random_gaussian_cortical_global: 0.0172
- time_ridge: 0.0172
- cortical_global: 0.0173
- residualized_autoregressive_plus_shuffled_combined: mae=0.1284, rmse=0.1476, pearson=0.1373, spearman=0.1119

### valence__event_future_drop_1_3s

Top conditions by `f1`:
- autoregressive_plus_cortical_global: 0.3653
- residualized_autoregressive_plus_cortical_global: 0.3563
- autoregressive_plus_random_gaussian_cortical_global: 0.3540
- residualized_autoregressive_plus_shuffled_cortical_global: 0.3540
- residualized_autoregressive_plus_random_gaussian_cortical_global: 0.3540
- autoregressive_plus_shuffled_cortical_global: 0.3495
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.8095, f1=0.3653

## mode_c_leave_video_out

### arousal__future_state_p1s

Top conditions by `mae`:
- residualized_autoregressive_plus_cortical_global: 0.0170
- autoregressive: 0.0170
- autoregressive_plus_random_gaussian_cortical_global: 0.0170
- residualized_autoregressive_plus_shuffled_cortical_global: 0.0170
- autoregressive_plus_shuffled_cortical_global: 0.0170
- autoregressive_plus_cortical_global: 0.0171
- residualized_autoregressive_plus_shuffled_combined: mae=0.0171, pearson=0.9937, rmse=0.0275, spearman=0.9918

### arousal__future_change_p1s

Top conditions by `mae`:
- residualized_autoregressive_plus_cortical_global: 0.0170
- autoregressive: 0.0170
- residualized_autoregressive_plus_random_gaussian_cortical_global: 0.0170
- autoregressive_plus_shuffled_cortical_global: 0.0170
- residualized_autoregressive_plus_shuffled_cortical_global: 0.0170
- autoregressive_plus_random_gaussian_cortical_global: 0.0171
- residualized_autoregressive_plus_shuffled_combined: mae=0.0171, pearson=0.3483, rmse=0.0275, spearman=0.3458

### arousal__residual_future_p1s_persistence

Top conditions by `mae`:
- residualized_autoregressive_plus_cortical_global: 0.0170
- autoregressive: 0.0170
- residualized_autoregressive_plus_random_gaussian_cortical_global: 0.0170
- autoregressive_plus_random_gaussian_cortical_global: 0.0170
- autoregressive_plus_cortical_global: 0.0171
- autoregressive_plus_shuffled_cortical_global: 0.0171
- residualized_autoregressive_plus_shuffled_combined: mae=0.0171, pearson=0.3483, rmse=0.0275, spearman=0.3458

### arousal__event_future_spike_1_3s

Top conditions by `f1`:
- residualized_autoregressive_plus_cortical_global: 0.3996
- autoregressive_plus_cortical_global: 0.3955
- autoregressive_plus_shuffled_cortical_global: 0.3917
- residualized_autoregressive_plus_shuffled_cortical_global: 0.3917
- autoregressive_plus_random_gaussian_cortical_global: 0.3917
- residualized_autoregressive_plus_random_gaussian_cortical_global: 0.3917
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.8012, f1=0.3955

### arousal__event_future_drop_1_3s

Top conditions by `f1`:
- residualized_autoregressive_plus_cortical_global: 0.3330
- residualized_autoregressive_plus_random_gaussian_cortical_global: 0.3321
- autoregressive: 0.3320
- autoregressive_plus_random_gaussian_cortical_global: 0.3320
- autoregressive_plus_shuffled_cortical_global: 0.3319
- residualized_autoregressive_plus_shuffled_cortical_global: 0.3319
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.8546, f1=0.3294

### arousal__event_trend_reversal_1_3s

Top conditions by `f1`:
- autoregressive_plus_random_gaussian_cortical_global: 0.1842
- autoregressive: 0.1818
- residualized_autoregressive_plus_random_gaussian_cortical_global: 0.1792
- autoregressive_plus_shuffled_cortical_global: 0.1790
- residualized_autoregressive_plus_shuffled_cortical_global: 0.1790
- autoregressive_plus_cortical_global: 0.1744
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.9336, f1=0.1744

### arousal__event_peak_onset_1_3s

Top conditions by `f1`:
- autoregressive: 0.1701
- autoregressive_plus_cortical_global: 0.1701
- residualized_autoregressive_plus_cortical_global: 0.1701
- residualized_autoregressive_plus_random_gaussian_cortical_global: 0.1670
- autoregressive_plus_random_gaussian_cortical_global: 0.1666
- autoregressive_plus_shuffled_cortical_global: 0.1596
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.9508, f1=0.1701

### arousal__event_recovery_onset_1_3s

Top conditions by `f1`:
- autoregressive_plus_shuffled_cortical_global: 0.2055
- residualized_autoregressive_plus_shuffled_cortical_global: 0.2055
- residualized_autoregressive_plus_random_gaussian_cortical_global: 0.1998
- autoregressive: 0.1964
- autoregressive_plus_random_gaussian_cortical_global: 0.1901
- autoregressive_plus_cortical_global: 0.1898
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.9506, f1=0.1898

### valence__future_state_p1s

Top conditions by `mae`:
- autoregressive: 0.0182
- residualized_autoregressive_plus_shuffled_cortical_global: 0.0182
- residualized_autoregressive_plus_random_gaussian_cortical_global: 0.0182
- autoregressive_plus_shuffled_cortical_global: 0.0182
- residualized_autoregressive_plus_cortical_global: 0.0182
- autoregressive_plus_cortical_global: 0.0182
- residualized_autoregressive_plus_shuffled_combined: mae=0.0182, pearson=0.9948, rmse=0.0335, spearman=0.9936

### valence__future_change_p1s

Top conditions by `mae`:
- autoregressive: 0.0182
- residualized_autoregressive_plus_shuffled_cortical_global: 0.0182
- autoregressive_plus_shuffled_cortical_global: 0.0182
- autoregressive_plus_random_gaussian_cortical_global: 0.0182
- residualized_autoregressive_plus_random_gaussian_cortical_global: 0.0182
- residualized_autoregressive_plus_cortical_global: 0.0182
- residualized_autoregressive_plus_shuffled_combined: mae=0.0183, pearson=0.3069, rmse=0.0335, spearman=0.3578

### valence__residual_future_p1s_persistence

Top conditions by `mae`:
- autoregressive: 0.0182
- autoregressive_plus_shuffled_cortical_global: 0.0182
- residualized_autoregressive_plus_shuffled_cortical_global: 0.0182
- residualized_autoregressive_plus_random_gaussian_cortical_global: 0.0182
- autoregressive_plus_random_gaussian_cortical_global: 0.0182
- residualized_autoregressive_plus_cortical_global: 0.0182
- residualized_autoregressive_plus_shuffled_combined: mae=0.0183, pearson=0.3069, rmse=0.0335, spearman=0.3578

### valence__event_future_drop_1_3s

Top conditions by `f1`:
- residualized_autoregressive_plus_cortical_global: 0.4416
- autoregressive: 0.4411
- autoregressive_plus_cortical_global: 0.4402
- autoregressive_plus_shuffled_cortical_global: 0.4400
- residualized_autoregressive_plus_shuffled_cortical_global: 0.4400
- residualized_autoregressive_plus_random_gaussian_cortical_global: 0.4373
- residualized_autoregressive_plus_shuffled_combined: accuracy=0.8232, f1=0.4402

## Lead / Lag Diagnostics

### arousal__future_change_p1s
- cortical_global: best_offset=0s, max_abs_pearson=0.1350, best_feature=cortical_p95_abs, best_pearson=0.1350

### arousal__future_change_p2s
- cortical_global: best_offset=1s, max_abs_pearson=0.1623, best_feature=cortical_p95_abs, best_pearson=0.1623

### arousal__future_change_p3s
- cortical_global: best_offset=2s, max_abs_pearson=0.1771, best_feature=cortical_p95_abs, best_pearson=0.1771

### valence__future_change_p1s
- cortical_global: best_offset=-3s, max_abs_pearson=0.0476, best_feature=cortical_mean, best_pearson=-0.0476

### valence__future_change_p2s
- cortical_global: best_offset=-3s, max_abs_pearson=0.0589, best_feature=cortical_mean, best_pearson=-0.0589

### valence__future_change_p3s
- cortical_global: best_offset=-3s, max_abs_pearson=0.0676, best_feature=cortical_mean, best_pearson=-0.0676

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
