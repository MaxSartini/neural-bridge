# VEATIC 124 Zero/Stable-Removed Continuous MAE Summary

This is the explicit answer to the zero-change caveat: full-frame continuous MAE is dominated by stable rows, so the fairer diagnostic is `event_only` or `event_plus_pre_3s` from the event-conditioned retest.

`real_vs_zero_mae_delta` is `zero MAE - real MAE`; positive means cortical beats the zero-change baseline on that mask.

| Feature | Target | Thr | Mask | n | Event count | Real MAE | Real-vs-zero MAE | Real-vs-AR MAE | Real-vs-shuf MAE | Real-vs-rand MAE |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cortical_fast_default | `arousal__future_change_p2s` | 0.0500 | all_frames | 2832 | 411 | 0.0922 | -0.0641 | -0.0022 | -0.0018 | -0.0020 |
| cortical_global_delta | `arousal__future_change_p2s` | 0.0500 | all_frames | 2832 | 411 | 0.0794 | -0.0513 | 0.0105 | 0.0101 | 0.0114 |
| cortical_pca64_delta | `arousal__future_change_p2s` | 0.0500 | all_frames | 2832 | 411 | 0.0540 | -0.0259 | 0.0359 | 0.0375 | 0.0290 |
| cortical_pca_64 | `arousal__future_change_p2s` | 0.0500 | all_frames | 2832 | 411 | 0.0734 | -0.0454 | 0.0165 | 0.0142 | 0.0131 |
| cortical_fast_default | `arousal__future_change_p2s` | 0.0500 | event_only | 411 | 411 | 0.1061 | -0.0148 | -0.0023 | -0.0020 | -0.0021 |
| cortical_global_delta | `arousal__future_change_p2s` | 0.0500 | event_only | 411 | 411 | 0.0966 | -0.0053 | 0.0072 | 0.0066 | 0.0077 |
| cortical_pca64_delta | `arousal__future_change_p2s` | 0.0500 | event_only | 411 | 411 | 0.0800 | 0.0113 | 0.0238 | 0.0257 | 0.0194 |
| cortical_pca_64 | `arousal__future_change_p2s` | 0.0500 | event_only | 411 | 411 | 0.0920 | -0.0007 | 0.0118 | 0.0103 | 0.0093 |
| cortical_fast_default | `arousal__future_change_p2s` | 0.0500 | event_plus_pre_3s | 855 | 411 | 0.0946 | -0.0389 | -0.0022 | -0.0018 | -0.0020 |
| cortical_global_delta | `arousal__future_change_p2s` | 0.0500 | event_plus_pre_3s | 855 | 411 | 0.0838 | -0.0281 | 0.0086 | 0.0082 | 0.0092 |
| cortical_pca64_delta | `arousal__future_change_p2s` | 0.0500 | event_plus_pre_3s | 855 | 411 | 0.0633 | -0.0076 | 0.0291 | 0.0306 | 0.0233 |
| cortical_pca_64 | `arousal__future_change_p2s` | 0.0500 | event_plus_pre_3s | 855 | 411 | 0.0793 | -0.0236 | 0.0131 | 0.0112 | 0.0102 |
| cortical_fast_default | `arousal__future_change_p2s` | 0.0750 | all_frames | 2832 | 180 | 0.0922 | -0.0641 | -0.0022 | -0.0018 | -0.0020 |
| cortical_global_delta | `arousal__future_change_p2s` | 0.0750 | all_frames | 2832 | 180 | 0.0794 | -0.0513 | 0.0105 | 0.0101 | 0.0114 |
| cortical_pca64_delta | `arousal__future_change_p2s` | 0.0750 | all_frames | 2832 | 180 | 0.0540 | -0.0259 | 0.0359 | 0.0375 | 0.0290 |
| cortical_pca_64 | `arousal__future_change_p2s` | 0.0750 | all_frames | 2832 | 180 | 0.0734 | -0.0454 | 0.0165 | 0.0142 | 0.0131 |
| cortical_fast_default | `arousal__future_change_p2s` | 0.0750 | event_only | 180 | 180 | 0.1154 | 0.0144 | -0.0015 | -0.0010 | -0.0013 |
| cortical_global_delta | `arousal__future_change_p2s` | 0.0750 | event_only | 180 | 180 | 0.1105 | 0.0193 | 0.0034 | 0.0030 | 0.0037 |
| cortical_pca64_delta | `arousal__future_change_p2s` | 0.0750 | event_only | 180 | 180 | 0.1030 | 0.0268 | 0.0109 | 0.0126 | 0.0094 |
| cortical_pca_64 | `arousal__future_change_p2s` | 0.0750 | event_only | 180 | 180 | 0.1064 | 0.0233 | 0.0074 | 0.0069 | 0.0061 |
| cortical_fast_default | `arousal__future_change_p2s` | 0.0750 | event_plus_pre_3s | 403 | 180 | 0.1004 | -0.0254 | -0.0018 | -0.0014 | -0.0016 |
| cortical_global_delta | `arousal__future_change_p2s` | 0.0750 | event_plus_pre_3s | 403 | 180 | 0.0909 | -0.0159 | 0.0077 | 0.0073 | 0.0082 |
| cortical_pca64_delta | `arousal__future_change_p2s` | 0.0750 | event_plus_pre_3s | 403 | 180 | 0.0753 | -0.0003 | 0.0232 | 0.0248 | 0.0184 |
| cortical_pca_64 | `arousal__future_change_p2s` | 0.0750 | event_plus_pre_3s | 403 | 180 | 0.0880 | -0.0130 | 0.0106 | 0.0092 | 0.0082 |
| cortical_fast_default | `arousal__future_change_p3s` | 0.0500 | all_frames | 2708 | 668 | 0.1403 | -0.1016 | -0.0038 | -0.0029 | -0.0043 |
| cortical_global_delta | `arousal__future_change_p3s` | 0.0500 | all_frames | 2708 | 668 | 0.1213 | -0.0827 | 0.0151 | 0.0183 | 0.0162 |
| cortical_pca64_delta | `arousal__future_change_p3s` | 0.0500 | all_frames | 2708 | 668 | 0.0771 | -0.0385 | 0.0593 | 0.0441 | 0.0445 |
| cortical_pca_64 | `arousal__future_change_p3s` | 0.0500 | all_frames | 2708 | 668 | 0.1061 | -0.0675 | 0.0303 | 0.0339 | 0.0244 |
| cortical_fast_default | `arousal__future_change_p3s` | 0.0500 | event_only | 668 | 668 | 0.1469 | -0.0489 | -0.0042 | -0.0033 | -0.0045 |
| cortical_global_delta | `arousal__future_change_p3s` | 0.0500 | event_only | 668 | 668 | 0.1304 | -0.0324 | 0.0124 | 0.0152 | 0.0132 |
| cortical_pca64_delta | `arousal__future_change_p3s` | 0.0500 | event_only | 668 | 668 | 0.0940 | 0.0039 | 0.0487 | 0.0354 | 0.0369 |
| cortical_pca_64 | `arousal__future_change_p3s` | 0.0500 | event_only | 668 | 668 | 0.1174 | -0.0195 | 0.0253 | 0.0281 | 0.0204 |
| cortical_fast_default | `arousal__future_change_p3s` | 0.0500 | event_plus_pre_3s | 1158 | 668 | 0.1381 | -0.0712 | -0.0036 | -0.0027 | -0.0039 |
| cortical_global_delta | `arousal__future_change_p3s` | 0.0500 | event_plus_pre_3s | 1158 | 668 | 0.1210 | -0.0541 | 0.0135 | 0.0164 | 0.0144 |
| cortical_pca64_delta | `arousal__future_change_p3s` | 0.0500 | event_plus_pre_3s | 1158 | 668 | 0.0831 | -0.0162 | 0.0514 | 0.0375 | 0.0382 |
| cortical_pca_64 | `arousal__future_change_p3s` | 0.0500 | event_plus_pre_3s | 1158 | 668 | 0.1086 | -0.0418 | 0.0259 | 0.0293 | 0.0206 |
| cortical_fast_default | `arousal__future_change_p3s` | 0.0750 | all_frames | 2708 | 363 | 0.1403 | -0.1016 | -0.0038 | -0.0029 | -0.0043 |
| cortical_global_delta | `arousal__future_change_p3s` | 0.0750 | all_frames | 2708 | 363 | 0.1213 | -0.0827 | 0.0151 | 0.0183 | 0.0162 |
| cortical_pca64_delta | `arousal__future_change_p3s` | 0.0750 | all_frames | 2708 | 363 | 0.0771 | -0.0385 | 0.0593 | 0.0441 | 0.0445 |
| cortical_pca_64 | `arousal__future_change_p3s` | 0.0750 | all_frames | 2708 | 363 | 0.1061 | -0.0675 | 0.0303 | 0.0339 | 0.0244 |
| cortical_fast_default | `arousal__future_change_p3s` | 0.0750 | event_only | 363 | 363 | 0.1503 | -0.0217 | -0.0041 | -0.0034 | -0.0044 |
| cortical_global_delta | `arousal__future_change_p3s` | 0.0750 | event_only | 363 | 363 | 0.1358 | -0.0071 | 0.0104 | 0.0129 | 0.0109 |
| cortical_pca64_delta | `arousal__future_change_p3s` | 0.0750 | event_only | 363 | 363 | 0.1054 | 0.0233 | 0.0408 | 0.0287 | 0.0302 |
| cortical_pca_64 | `arousal__future_change_p3s` | 0.0750 | event_only | 363 | 363 | 0.1240 | 0.0047 | 0.0222 | 0.0250 | 0.0179 |
| cortical_fast_default | `arousal__future_change_p3s` | 0.0750 | event_plus_pre_3s | 682 | 363 | 0.1420 | -0.0567 | -0.0033 | -0.0025 | -0.0037 |
| cortical_global_delta | `arousal__future_change_p3s` | 0.0750 | event_plus_pre_3s | 682 | 363 | 0.1256 | -0.0404 | 0.0131 | 0.0158 | 0.0137 |
| cortical_pca64_delta | `arousal__future_change_p3s` | 0.0750 | event_plus_pre_3s | 682 | 363 | 0.0898 | -0.0045 | 0.0490 | 0.0358 | 0.0363 |
| cortical_pca_64 | `arousal__future_change_p3s` | 0.0750 | event_plus_pre_3s | 682 | 363 | 0.1133 | -0.0280 | 0.0254 | 0.0286 | 0.0205 |

## Readout

- On `event_only`, `cortical_pca64_delta` beats zero-change MAE for `p2s @ 0.05`, `p2s @ 0.075`, `p3s @ 0.05`, and `p3s @ 0.075`.
- On `event_only`, `cortical_pca_64` beats zero-change MAE for `p2s @ 0.075` and `p3s @ 0.075`; it is near break-even for `p2s @ 0.05` and still below zero for `p3s @ 0.05`.
- On `event_plus_pre_3s`, the zero baseline is still hard to beat because the pre-event frames include many sub-threshold/stable rows; `pca64_delta` is close to break-even for `p2s @ 0.075` and `p3s @ 0.075`, but the ranking/event metrics remain the stronger evidence.
- Full-frame MAE should not be used as the main verdict because it rewards predicting no change across the many non-event rows.