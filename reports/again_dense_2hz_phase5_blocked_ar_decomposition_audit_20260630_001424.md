# Phase 5 Blocked AR Decomposition Audit 20260630_001424

This is a no-training audit of what frozen AR is exploiting under `blocked_temporal_70_30`. It uses existing labels, split construction, frozen-AR score caches, and completed diagnostic outputs only. It does not change claims: strict forward-time temporal generalization remains unproven.

## Bottom Line

Frozen AR is best classified as **clean/legal**, not leaky, from this audit. The AR feature block uses current and past annotation history only, with no overlap with target future rows `+2..+6`. The strength is mainly legal but overpowering temporal information: current/local arousal state and recent arousal history are highly aligned with the future-delta target construction.

The strongest single explanatory factor for frozen AR score is `previous_row_arousal` by absolute Spearman correlation. The best simple baseline by binary PR-AUC proximity is `per_video_train_base_rate_only`. The best simple baseline by top-5pct continuous-lift proximity is `recent_arousal_slope_delta`.

## What Frozen AR Uses

The `ar_only_head` feature block contains exactly these seven columns and no explicit time or PCA columns:

| column                  | row_offsets   | second_offsets   | uses_current   | uses_past   | uses_future   | overlaps_target_future_rows_2_6   |
|:------------------------|:--------------|:-----------------|:---------------|:------------|:--------------|:----------------------------------|
| arousal                 | [0]           | [0.0]            | True           | False       | False         | False                             |
| arousal_lag_1row        | [-1]          | [-0.5]           | False          | True        | False         | False                             |
| arousal_lag_2row        | [-2]          | [-1.0]           | False          | True        | False         | False                             |
| arousal_lag_4row        | [-4]          | [-2.0]           | False          | True        | False         | False                             |
| arousal_delta_prev_1row | [0, -1]       | [0.0, -0.5]      | True           | True        | False         | False                             |
| arousal_delta_prev_2row | [0, -2]       | [0.0, -1.0]      | True           | True        | False         | False                             |
| arousal_delta_prev_4row | [0, -4]       | [0.0, -2.0]      | True           | True        | False         | False                             |

The model input is train-only standardized after assembly. The AR block comes from `AR_FEATURE_COLUMNS` and `add_dense_2hz_targets_and_ar_features` in `backend/scripts/again_dense_2hz_benchmark.py`.

## Target And Window Overlap

The continuous target is `future_arousal_max_delta_rows_2_6 = max(arousal[t+2], ..., arousal[t+6]) - arousal[t]`. At 2 Hz, those future rows are +1.0s through +3.0s. The binary q90 spike target thresholds this continuous future max-delta on the training distribution.

AR/history rows are current and past only: `t`, `t-0.5s`, `t-1.0s`, and `t-2.0s`, plus deltas from current to those past rows. There is no direct overlap with future target rows `t+1.0s..t+3.0s`.

Pass/fail: target-window overlap = **no**; future leakage suspected = **no**. The important caveat is that current arousal is part of both the AR feature block and the continuous target's baseline subtraction, which makes this a very strong local-state/autocorrelation task without being future leakage.

## Simple Baseline Audit

| baseline                              |   pr_auc_spike |   spearman_continuous |   top_5pct_continuous_lift |   abs_pr_auc_gap_to_frozen |   abs_top5_lift_gap_to_frozen |
|:--------------------------------------|---------------:|----------------------:|---------------------------:|---------------------------:|------------------------------:|
| frozen_ar_score_mean_5seed            |      0.266978  |            0.210967   |                 0.096201   |                  0         |                     0         |
| per_video_train_base_rate_only        |      0.204552  |           -0.0511953  |                 0.049249   |                  0.0624264 |                     0.046952  |
| train_only_video_mean_continuous_only |      0.193903  |            0.00677882 |                 0.056305   |                  0.0730755 |                     0.039896  |
| recent_arousal_slope_delta            |      0.170586  |            0.294912   |                 0.0635868  |                  0.0963919 |                     0.0326142 |
| low_previous_row_arousal_negated      |      0.170367  |            0.12408    |                 0.0334609  |                  0.0966118 |                     0.0627402 |
| low_trailing_2s_mean_arousal_negated  |      0.16725   |            0.126694   |                 0.0277761  |                  0.099728  |                     0.0684249 |
| low_trailing_4s_mean_arousal_negated  |      0.16711   |            0.139153   |                 0.022524   |                  0.099868  |                     0.0736771 |
| low_current_arousal_negated           |      0.16311   |            0.0993019  |                 0.0300003  |                  0.103869  |                     0.0662007 |
| time_seconds_only                     |      0.126451  |            0.0152318  |                 0.012642   |                  0.140527  |                     0.083559  |
| video_relative_time_only              |      0.123403  |            0.0210859  |                 0.0078866  |                  0.143576  |                     0.0883144 |
| negative_recent_arousal_slope_delta   |      0.105636  |           -0.294912   |                 0.00943258 |                  0.161343  |                     0.0867685 |
| current_arousal                       |      0.0770177 |           -0.0993019  |                -0.0508409  |                  0.189961  |                     0.147042  |
| previous_row_arousal                  |      0.075878  |           -0.12408    |                -0.049414   |                  0.1911    |                     0.145615  |
| trailing_2s_mean_arousal              |      0.0756414 |           -0.126694   |                -0.0505672  |                  0.191337  |                     0.146768  |
| trailing_4s_mean_arousal              |      0.0755146 |           -0.139153   |                -0.0497535  |                  0.191464  |                     0.145955  |

Low-current-arousal history baselines are the closest simple non-trained predictors, matching the negative correlation between frozen AR score and current arousal. Per-video base rate is a meaningful nuisance baseline for binary PR-AUC, but it is still well below frozen AR and has weak R2 against frozen AR predictions; time-only predictors are weaker. Per-video base rate and time position alone are not enough to explain AR.

## Prevalence And Time Position

Train spike prevalence: `0.100523`; test spike prevalence: `0.111245`.

Train continuous mean/std: `0.048830` / `0.088781`. Test continuous mean/std: `0.050841` / `0.096384`.

The detailed per-video and time-quartile table is in `prevalence_time_position_audit.csv`. The highest finite per-video AR PR-AUC rows are:

| video_id                                                                           |   rows |   positive_count |   spike_prevalence |   ar_pr_auc |   ar_spearman_continuous |   ar_top5_lift |
|:-----------------------------------------------------------------------------------|-------:|-----------------:|-------------------:|------------:|-------------------------:|---------------:|
| 77F6B31E-B228-3C6D-A66F-E7E6FDED3587_gun_17143762-D629-1D6A-8564-FAEF6E87293E      |     73 |                3 |          0.0410959 |    1        |                 0.524429 |       0.159292 |
| 8322EDEF-4C09-318A-1B7E-56D62A6839AD_platform_AEF73D6C-126D-4D3B-ADCC-4C6EA38520D3 |     71 |                6 |          0.084507  |    1        |                 0.54517  |       0.313898 |
| 72B642AD-2D4A-574F-05EC-EC93C0C52335_solid_AEA0ECCB-2DBF-7DE3-E5BE-60A633CA126A    |     71 |                2 |          0.028169  |    1        |                 0.673756 |       0.173636 |
| 1FA97947-2FB7-D626-8CD3-2426BEEE97D1_apex_EE418EBE-0A75-060D-7DF6-59B1951543B9     |     71 |               24 |          0.338028  |    1        |                 0.827907 |       0.116225 |
| 10FFF83C-FD48-DCA7-E554-491C3C1E3EC6_gun_8571A046-DDFE-A0EB-3A6D-2661E6F97CB2      |     72 |               13 |          0.180556  |    1        |                 0.753464 |       0.843997 |
| AB9921A8-F143-E101-6A47-E66882F571B2_platform_C80B406C-B1A2-6DE9-C6DF-ADADFA7F9EDA |     71 |               11 |          0.15493   |    0.992424 |                 0.687495 |       0.467835 |
| BC7EE230-0AF9-3AC5-24E5-8EC4B0B432A8_endless_93672EAF-52D9-D4C3-369B-B60EA2AC26B1  |     71 |                7 |          0.0985915 |    0.968254 |                 0.421192 |       0.146253 |
| 5E491E0A-C580-29BE-4329-ED7BFF843D0D_gun_45F033B3-E3E8-6D11-6CB9-E3FDB894AD80      |     72 |               21 |          0.291667  |    0.958259 |                 0.787946 |       0.729198 |

This indicates AR performance is heterogeneous by video and event prevalence, but the explanation is not just a train-only per-video base-rate predictor. Time position contributes, but does not match current/local arousal history.

## AR Dominance Decomposition

Top frozen-AR score correlations:

| feature                          |   corr_with_frozen_ar_score |   spearman_with_frozen_ar_score |   corr_with_frozen_ar_continuous |   spearman_with_frozen_ar_continuous |
|:---------------------------------|----------------------------:|--------------------------------:|---------------------------------:|-------------------------------------:|
| previous_row_arousal             |                 -0.6849     |                       -0.718687 |                       -0.643138  |                           -0.757011  |
| trailing_2s_mean_arousal         |                 -0.682238   |                       -0.714983 |                       -0.630633  |                           -0.756795  |
| current_arousal                  |                 -0.683387   |                       -0.707448 |                       -0.625105  |                           -0.731024  |
| trailing_4s_mean_arousal         |                 -0.663149   |                       -0.690997 |                       -0.603661  |                           -0.748182  |
| per_video_train_base_rate        |                  0.17933    |                        0.242372 |                        0.21645   |                            0.200746  |
| train_only_video_mean_continuous |                  0.160581   |                        0.205716 |                        0.203928  |                            0.171892  |
| recent_arousal_slope_delta       |                 -0.00875478 |                        0.111444 |                        0.0916818 |                            0.287557  |
| time_seconds                     |                 -0.159147   |                       -0.105051 |                       -0.0817032 |                           -0.0974165 |
| video_relative_time              |                 -0.155432   |                       -0.100872 |                       -0.0728561 |                           -0.0911192 |

Linear R2 decomposition for frozen AR score:

| feature_set              |   r2_frozen_ar_score |   r2_frozen_ar_continuous |
|:-------------------------|---------------------:|--------------------------:|
| all_simple               |            0.485862  |                0.440365   |
| lag_history_only         |            0.471431  |                0.425492   |
| current_plus_lag_history |            0.471431  |                0.425492   |
| current_only             |            0.467018  |                0.390756   |
| video_base_rate_only     |            0.0321811 |                0.0475906  |
| time_only                |            0.0254603 |                0.00698387 |

The decomposition points to local arousal state/history as the dominant explanatory family. Per-video base rate and time position are not sufficient alone.

## Leakage Risk Conclusion

Classification: **clean/legal**.

Specific findings:

- AR columns do not include target future rows `+2..+6`.
- The target construction uses future rows for the label and current arousal as the delta baseline.
- The AR model sees current arousal and recent past labels, which is legal for an annotation-history baseline but very strong.
- No evidence was found that AR features are computed after target construction using future target-window values.
- Current row alignment can use nearest annotation metadata, but the benchmark labels are already aligned 2 Hz rows; this audit did not find direct target-window leakage through AR feature construction.

## Answers

- What is frozen AR using? Current arousal, recent arousal lags, and recent deltas.
- Is there target-window overlap? No direct overlap with rows `+2..+6`.
- Is there future leakage? No direct future leakage suspected from AR feature construction.
- Are simple arousal-history baselines close to frozen AR? Yes, especially current/local arousal history.
- Is per-video base rate enough? No.
- Is time position enough? No.
- Is blocked temporal task mostly autocorrelation? Yes, the blocked task is dominated by local annotation autocorrelation and current-state persistence.
- What should the next modeling target be? A target less dominated by current arousal persistence, such as onset/change residuals beyond a current/past-only baseline, event onset rank conditioned on low current AR confidence, or a blocked target with an explicit washout gap between AR history and future label.
