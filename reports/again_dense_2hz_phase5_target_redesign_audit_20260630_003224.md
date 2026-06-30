# Phase 5 Target Redesign Audit 20260630_003224

This is a no-training blocked temporal target feasibility audit. It uses only `labels_aligned_2hz.parquet`, existing row/video/time metadata, and existing frozen-AR score cache for low-confidence slice diagnostics. It does not run residual models, grouped validation, 504 confirmation, V-JEPA/TRIBE/PCA, or AR retraining, and it does not change project claims.

## Bottom Line

The current blocked targets are dominated by legal current/past arousal persistence. A target redesign should move toward onset/change/surprise definitions with an explicit washout gap and/or train-only AR-residualized continuous targets.

Top recommended binary candidate: `future_arousal_max_delta_rows_4_10_train_q90`.

Top recommended continuous candidate: `residual_future_max_delta_rows_4_10`.

These are recommendations for future target work only. Strict forward-time temporal generalization remains unproven.

## Recommended Binary Target

| Field | Value |
|---|---:|
| candidate | `future_arousal_max_delta_rows_4_10_train_q90` |
| family | `washout_gap_future_movement` |
| test prevalence | `0.1146727254` |
| test positives | `7924.0` |
| test videos with zero positives | `384.0` |
| best simple AR baseline | `previous_row_arousal` |
| best simple AR PR-AUC | `0.2006483376` |
| AR dominance score | `0.0859756122` |
| washout gap seconds | `1.500` |
| overlap/leakage | `False` / `False` |

Best simple baselines for the recommended binary target:

| baseline | pr_auc | roc_auc | top_5pct_recall | direction |
|---|---|---|---|---|
| previous_row_arousal | 0.200648 | 0.717932 | 0.105628 | -1 |
| current_arousal | 0.195630 | 0.713463 | 0.100833 | -1 |
| trailing_2s_mean | 0.195602 | 0.715479 | 0.097299 | -1 |
| per_video_train_base_rate | 0.195453 | 0.643915 | 0.119637 | 1 |
| trailing_4s_mean | 0.192003 | 0.707211 | 0.097361 | -1 |
| recent_arousal_slope_delta | 0.147393 | 0.500685 | 0.109162 | -1 |
| video_relative_time | 0.108931 | 0.471993 | 0.052499 | -1 |
| train_only_video_mean_arousal | 0.094676 | 0.428938 | 0.025240 | 1 |

## Recommended Continuous Target

| Field | Value |
|---|---:|
| candidate | `residual_future_max_delta_rows_4_10` |
| family | `ar_residualized_continuous` |
| test variance | `0.0209788569` |
| test std | `0.1448407984` |
| best simple AR baseline | `recent_arousal_slope_delta` |
| best simple absolute Spearman | `0.2332493941` |
| AR dominance score | `0.2332493941` |
| washout gap seconds | `1.500` |
| overlap/leakage | `False` / `False` |

Best simple baselines for the recommended continuous target:

| baseline | spearman | pearson | top_5pct_continuous_lift | direction |
|---|---|---|---|---|
| recent_arousal_slope_delta | 0.233249 | 0.050946 | 0.018106 | 1 |
| train_only_video_mean_arousal | -0.212577 | -0.158502 | -0.043058 | 1 |
| current_arousal | 0.128635 | -0.038969 | -0.049355 | 1 |
| previous_row_arousal | 0.113491 | -0.048414 | -0.051664 | 1 |
| trailing_2s_mean | 0.105224 | -0.051246 | -0.042569 | 1 |
| per_video_train_base_rate | -0.095594 | -0.041229 | 0.037115 | 1 |
| trailing_4s_mean | 0.081460 | -0.059372 | -0.038567 | 1 |
| video_relative_time | 0.046971 | 0.038695 | 0.012314 | 1 |

## Candidate Ranking

Lowest AR-dominance binary candidates:

| candidate | family | positive_prevalence_test | test_positive_count | test_videos_zero_positives | best_simple_ar_baseline | best_simple_ar_metric | ar_dominance_score | washout_gap_seconds |
|---|---|---|---|---|---|---|---|---|
| future_arousal_max_delta_rows_4_10_train_q90 | washout_gap_future_movement | 0.114673 | 7924.000000 | 384.000000 | previous_row_arousal | 0.200648 | 0.085976 | 1.500000 |
| future_arousal_max_delta_rows_6_12_train_q90 | washout_gap_future_movement | 0.118054 | 8073.000000 | 392.000000 | previous_row_arousal | 0.218075 | 0.100021 | 2.500000 |
| future_onset_spike_rows_4_10_train_q90_current_q70 | onset_surprise_spike | 0.048118 | 3325.000000 | 571.000000 | current_arousal | 0.176798 | 0.128680 | 1.500000 |
| future_onset_spike_rows_4_10_train_q90_current_q60 | onset_surprise_spike | 0.048031 | 3319.000000 | 571.000000 | current_arousal | 0.176894 | 0.128863 | 1.500000 |
| future_arousal_abs_delta_rows_4_10_train_q90 | washout_gap_abs_movement | 0.131199 | 9066.000000 | 402.000000 | per_video_train_base_rate | 0.270731 | 0.139532 | 1.500000 |
| future_onset_spike_rows_4_10_train_q90_current_q50 | onset_surprise_spike | 0.041172 | 2845.000000 | 624.000000 | current_arousal | 0.182724 | 0.141553 | 1.500000 |
| future_arousal_abs_delta_rows_6_12_train_q90 | washout_gap_abs_movement | 0.134871 | 9223.000000 | 410.000000 | per_video_train_base_rate | 0.277729 | 0.142858 | 2.500000 |
| future_onset_spike_rows_6_12_train_q90_current_q70 | onset_surprise_spike | 0.052556 | 3594.000000 | 561.000000 | current_arousal | 0.199975 | 0.147419 | 2.500000 |
| future_onset_spike_rows_6_12_train_q90_current_q60 | onset_surprise_spike | 0.052439 | 3586.000000 | 561.000000 | current_arousal | 0.200126 | 0.147687 | 2.500000 |
| future_onset_spike_rows_6_12_train_q90_current_q50 | onset_surprise_spike | 0.045084 | 3083.000000 | 617.000000 | current_arousal | 0.207390 | 0.162307 | 2.500000 |

Lowest AR-dominance continuous candidates:

| candidate | family | target_variance_test | test_distribution_std | best_simple_ar_baseline | best_simple_ar_metric | ar_dominance_score | washout_gap_seconds |
|---|---|---|---|---|---|---|---|
| future_arousal_max_delta_rows_4_10 | washout_gap_future_movement | 0.022500 | 0.150000 | train_only_video_mean_arousal | 0.231066 | 0.231066 | 1.500000 |
| residual_future_max_delta_rows_4_10 | ar_residualized_continuous | 0.020979 | 0.144841 | recent_arousal_slope_delta | 0.233249 | 0.233249 | 1.500000 |
| future_arousal_max_delta_rows_6_12 | washout_gap_future_movement | 0.029037 | 0.170404 | train_only_video_mean_arousal | 0.240352 | 0.240352 | 2.500000 |
| future_arousal_abs_delta_rows_6_12 | washout_gap_abs_movement | 0.025307 | 0.159081 | per_video_train_base_rate | 0.246749 | 0.246749 | 2.500000 |
| future_arousal_abs_delta_rows_4_10 | washout_gap_abs_movement | 0.022613 | 0.150377 | per_video_train_base_rate | 0.249862 | 0.249862 | 1.500000 |
| residual_future_max_delta_rows_2_6 | ar_residualized_continuous | 0.011942 | 0.109279 | recent_arousal_slope_delta | 0.274244 | 0.274244 | 0.500000 |
| residual_future_abs_delta_rows_4_10 | ar_residualized_continuous | 0.023154 | 0.152166 | per_video_train_base_rate | 0.331346 | 0.331346 | 1.500000 |

## Washout Gap

Washout-gap candidates use target rows `+4..+10` or `+6..+12`, leaving rows between the current/past AR history and the future target window. This removes direct target-window overlap and makes the target less immediate than `rows_2_6`. In this audit, washout gap helps mainly when paired with onset/surprise gating or residualization; simple future-movement q90 targets are still meaningfully predictable from AR/history variables.

- Washout gap helps: `True`
- Target-window overlap found: `False`
- Future leakage suspected: `False`

## Onset Targets

Onset candidates require the future movement window to exceed the train q90 threshold while current arousal, trailing 2s mean, and recent slope are not already high. This makes the target commercially interpretable as a future onset/surprise response instead of persistence of an already-high state.

- Onset targets reduce AR dominance relative to plain washout q90 targets: `False`
- Main tradeoff: prevalence and per-video positives drop as gating gets stricter.

## AR-Residualized Continuous Targets

AR-residualized continuous targets are viable as feasibility targets, not as completed benchmark claims. They subtract a train-only simple AR baseline fit inside the blocked split using current arousal, previous arousal, trailing means, recent slope, and video-relative time. This directly targets movement not explained by legal AR/history variables.

- AR-residualized targets viable: `False`
- Residualization diagnostics are in `recommended_targets.json`.

## Low-AR-Confidence Slices

Low/ambiguous frozen-AR score slices are worth testing as evaluation slices, not as the main target yet: `True`.

| slice | row_count | video_count | original_spike_prevalence | continuous_mean | continuous_std |
|---|---|---|---|---|---|
| middle_40_60pct_ar_score | 14076 | 887 | 0.097045 | 0.037391 | 0.096835 |
| middle_30_70pct_ar_score | 28143 | 946 | 0.099279 | 0.036335 | 0.099224 |
| exclude_top_bottom_20pct_ar_score | 42199 | 965 | 0.099979 | 0.035726 | 0.099205 |

## Leakage And Overlap

Every candidate uses AR/history rows `[0, -1, -2, -4]` and future target rows only. The washout gap is explicit for rows `+4..+10` and `+6..+12`. No candidate has target-window overlap with the AR/history rows, and no future leakage is suspected from this label-only audit.

Detailed rows and offsets are in `target_overlap_audit.json`.

## Exact Next Training Recommendation

Do not run grouped, 504, broad variants, or secondary targets from this audit alone. If a follow-up training run is approved later, the smallest clean matrix should be blocked-only and target-focused:

- Protocol: `blocked_temporal_70_30` only.
- Targets: `future_arousal_max_delta_rows_4_10_train_q90` and `residual_future_max_delta_rows_4_10`.
- Variant: one frozen-AR residual candidate, preferably the existing monotonic/do-no-harm residual or an even simpler residualized target head.
- Seeds: `20260625`, `20260626`, `20260627` initially.
- Controls: frozen AR only, real residual, shuffled PCA, random PCA, label permutation with permuted inner-val selection, train-only video mean/static control, diagnostics-only.
- Cap: about `42` rows for two targets x three seeds x seven controls.
- Stop before training if the recommended target is judged commercially unclear, if simple AR baselines still dominate, or if any overlap/leakage finding changes.

## Evidence Files

- `target_candidate_summary.csv`
- `simple_ar_baseline_by_candidate.csv`
- `target_overlap_audit.json`
- `recommended_targets.json`
- `README.md`
