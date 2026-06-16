# VEATIC 124 Event/Spike Retest

## SECTION 1: Executive Verdict

Classification: **promising but needs alignment fix**.

Supported claim: cortical/TRIBE features show useful signal for upcoming arousal spike/event ranking under blocked validation when evaluated as events rather than exact future decimal values.

Not supported: exact continuous future arousal-change prediction as the primary claim, especially where zero-change MAE remains competitive.

Decision thresholds for classification were calibrated on train predictions only. No test-threshold tuning was used.

Leave-video-out was not run in this pass because no cheap existing implementation was present; add it as a separate confirmatory TODO after the blocked/official diagnostics.

## SECTION 2: Main Blocked Spike Result

| Feature | Thr | PR-AUC AR | PR-AUC real | PR-AUC shuf | PR-AUC rand | F1 real | BalAcc real | Prec real | Rec real | Pass controls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| cortical_fast_default | 0.0500 | 0.1969 | 0.2052 | 0.1980 | 0.2007 | 0.2338 | 0.5004 | 0.1327 | 0.9821 | False |
| cortical_fast_default | 0.0750 | 0.1151 | 0.1330 | 0.1148 | 0.1148 | 0.1465 | 0.5138 | 0.0792 | 0.9693 | True |
| cortical_global_delta | 0.0500 | 0.1969 | 0.2196 | 0.1988 | 0.1956 | 0.2362 | 0.5083 | 0.1346 | 0.9643 | True |
| cortical_global_delta | 0.0750 | 0.1151 | 0.1500 | 0.1167 | 0.1175 | 0.1548 | 0.5437 | 0.0843 | 0.9430 | True |
| cortical_pca_64 | 0.0500 | 0.1969 | 0.2455 | 0.1981 | 0.1930 | 0.2510 | 0.5452 | 0.1445 | 0.9566 | True |
| cortical_pca_64 | 0.0750 | 0.1151 | 0.1575 | 0.1141 | 0.1147 | 0.1563 | 0.5480 | 0.0856 | 0.8991 | True |
| cortical_pca64_delta | 0.0500 | 0.1969 | 0.2536 | 0.1840 | 0.1944 | 0.2677 | 0.5811 | 0.1581 | 0.8724 | True |
| cortical_pca64_delta | 0.0750 | 0.1151 | 0.1614 | 0.1190 | 0.1208 | 0.1682 | 0.5807 | 0.0932 | 0.8640 | True |

## SECTION 3: Official Split Result

| Feature | Thr | PR-AUC AR | PR-AUC real | F1 real | BalAcc real | Rec real | Pass controls |
|---|---:|---:|---:|---:|---:|---:|---|
| cortical_fast_default | 0.0500 | 0.2751 | 0.2943 | 0.2915 | 0.6172 | 0.7832 | True |
| cortical_fast_default | 0.0750 | 0.1817 | 0.1971 | 0.2293 | 0.6258 | 0.4737 | True |
| cortical_global_delta | 0.0500 | 0.2751 | 0.2944 | 0.2929 | 0.6193 | 0.7857 | True |
| cortical_global_delta | 0.0750 | 0.1817 | 0.2113 | 0.2226 | 0.6308 | 0.5351 | True |
| cortical_pca_64 | 0.0500 | 0.2751 | 0.3434 | 0.3531 | 0.6679 | 0.6378 | True |
| cortical_pca_64 | 0.0750 | 0.1817 | 0.2514 | 0.3025 | 0.6559 | 0.4298 | True |
| cortical_pca64_delta | 0.0500 | 0.2751 | 0.3327 | 0.3634 | 0.6341 | 0.3699 | True |
| cortical_pca64_delta | 0.0750 | 0.1817 | 0.2264 | 0.2600 | 0.5920 | 0.2281 | True |

## SECTION 4: Continuous Future-Change Diagnostic

Zero-baseline MAE was beaten in 5/24 continuous checks. Continuous MAE should remain diagnostic, not the primary verdict.

| Feature | Split | Target | Real MAE | Zero MAE | Zero pass | Real-vs-AR MAE delta |
|---|---|---|---:|---:|---|---:|
| cortical_fast_default | blocked | `arousal__future_change_p1s` | 0.0271 | 0.0166 | False | -0.0009 |
| cortical_fast_default | blocked | `arousal__future_change_p2s` | 0.0922 | 0.0281 | False | -0.0022 |
| cortical_fast_default | blocked | `arousal__future_change_p3s` | 0.1403 | 0.0386 | False | -0.0038 |
| cortical_global_delta | blocked | `arousal__future_change_p1s` | 0.0216 | 0.0166 | False | 0.0046 |
| cortical_global_delta | blocked | `arousal__future_change_p2s` | 0.0794 | 0.0281 | False | 0.0105 |
| cortical_global_delta | blocked | `arousal__future_change_p3s` | 0.1213 | 0.0386 | False | 0.0151 |
| cortical_pca_64 | blocked | `arousal__future_change_p1s` | 0.0281 | 0.0166 | False | -0.0020 |
| cortical_pca_64 | blocked | `arousal__future_change_p2s` | 0.0734 | 0.0281 | False | 0.0165 |
| cortical_pca_64 | blocked | `arousal__future_change_p3s` | 0.1061 | 0.0386 | False | 0.0303 |
| cortical_pca64_delta | blocked | `arousal__future_change_p1s` | 0.0219 | 0.0166 | False | 0.0042 |
| cortical_pca64_delta | blocked | `arousal__future_change_p2s` | 0.0540 | 0.0281 | False | 0.0359 |
| cortical_pca64_delta | blocked | `arousal__future_change_p3s` | 0.0771 | 0.0386 | False | 0.0593 |

## SECTION 5: Movement/Event Threshold Sweep

Best blocked movement rows by real PR-AUC:

| Feature | Target | Thr | PR-AUC real | F1 real | Recall real | Pass controls |
|---|---|---:|---:|---:|---:|---|
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0300 | 0.4551 | 0.0699 | 0.0374 | True |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0300 | 0.4549 | 0.0324 | 0.0165 | True |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0300 | 0.4465 | 0.0172 | 0.0087 | False |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0300 | 0.4451 | 0.0206 | 0.0104 | True |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0300 | 0.4036 | 0.1274 | 0.0727 | True |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0500 | 0.3858 | 0.2017 | 0.1243 | True |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0300 | 0.3818 | 0.0648 | 0.0341 | True |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0500 | 0.3762 | 0.3548 | 0.3219 | True |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0300 | 0.3670 | 0.0323 | 0.0165 | False |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0300 | 0.3652 | 0.0344 | 0.0176 | False |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0500 | 0.3458 | 0.1373 | 0.0763 | True |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0500 | 0.3448 | 0.1337 | 0.0749 | True |

## SECTION 6: Shift/Alignment Audit

Best-shift counts: negative=10, zero=3, positive=3. Non-zero best shifts are not automatically leakage; they may reflect annotation lag, smoothing lag, feature lag, or anticipatory signal.

| Feature | Target | Thr | Best offset | Best PR-AUC | 0s PR-AUC | 0s competitive |
|---|---|---:|---:|---:|---:|---|
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | 3 | 0.3284 | 0.3101 | False |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0500 | -3 | 0.3593 | 0.3448 | True |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0500 | -5 | 0.3375 | 0.2052 | False |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0750 | -2 | 0.1913 | 0.1330 | False |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | 3 | 0.3153 | 0.3040 | True |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0500 | 0 | 0.3458 | 0.3458 | True |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0500 | -2 | 0.3560 | 0.2196 | False |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0750 | -2 | 0.2326 | 0.1500 | False |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | 0 | 0.3007 | 0.3007 | True |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0500 | 0 | 0.3762 | 0.3762 | True |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0500 | -2 | 0.3593 | 0.2536 | False |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0750 | -1 | 0.2271 | 0.1614 | False |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | 2 | 0.3239 | 0.2996 | False |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0500 | -3 | 0.4112 | 0.3858 | False |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0500 | -2 | 0.3343 | 0.2455 | False |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0750 | -2 | 0.2102 | 0.1575 | False |

Anti-leakage checks for blocked spike thresholds:

| Feature | Thr | Control | PR-AUC | F1 | BalAcc | Recall |
|---|---:|---|---:|---:|---:|---:|
| cortical_fast_default | 0.0500 | `real_cortical` | 0.2052 | 0.2338 | 0.5004 | 0.9821 |
| cortical_fast_default | 0.0500 | `label_shuffle_across_videos` | 0.1394 | 0.2342 | 0.5000 | 1.0000 |
| cortical_fast_default | 0.0500 | `label_shuffle_within_video` | 0.1471 | 0.0050 | 0.5001 | 0.0026 |
| cortical_fast_default | 0.0500 | `feature_shuffle_across_videos` | 0.1946 | 0.2339 | 0.4999 | 0.9923 |
| cortical_fast_default | 0.0500 | `feature_shuffle_within_video` | 0.1967 | 0.2345 | 0.5016 | 0.9923 |
| cortical_fast_default | 0.0500 | `timestamp_only` | 0.1281 | 0.1201 | 0.5054 | 0.1020 |
| cortical_fast_default | 0.0500 | `video_id_time_only` | 0.1427 | 0.2249 | 0.5372 | 0.3801 |
| cortical_fast_default | 0.0750 | `real_cortical` | 0.1330 | 0.1465 | 0.5138 | 0.9693 |
| cortical_fast_default | 0.0750 | `label_shuffle_across_videos` | 0.0757 | 0.1409 | 0.4922 | 0.9693 |
| cortical_fast_default | 0.0750 | `label_shuffle_within_video` | 0.1093 | 0.1221 | 0.5286 | 0.0921 |
| cortical_fast_default | 0.0750 | `feature_shuffle_across_videos` | 0.1152 | 0.1428 | 0.4985 | 0.9912 |
| cortical_fast_default | 0.0750 | `feature_shuffle_within_video` | 0.1176 | 0.1434 | 0.5011 | 0.9956 |
| cortical_fast_default | 0.0750 | `timestamp_only` | 0.0735 | 0.0297 | 0.4919 | 0.0219 |
| cortical_fast_default | 0.0750 | `video_id_time_only` | 0.1027 | 0.1885 | 0.5840 | 0.4298 |
| cortical_global_delta | 0.0500 | `real_cortical` | 0.2196 | 0.2362 | 0.5083 | 0.9643 |
| cortical_global_delta | 0.0500 | `label_shuffle_across_videos` | 0.1268 | NA | 0.5000 | 0.0000 |
| cortical_global_delta | 0.0500 | `label_shuffle_within_video` | 0.1655 | 0.2378 | 0.5405 | 0.5408 |
| cortical_global_delta | 0.0500 | `feature_shuffle_across_videos` | 0.1954 | 0.2356 | 0.5046 | 0.9923 |
| cortical_global_delta | 0.0500 | `feature_shuffle_within_video` | 0.1959 | 0.2341 | 0.5003 | 0.9923 |
| cortical_global_delta | 0.0500 | `timestamp_only` | 0.1281 | 0.1201 | 0.5054 | 0.1020 |
| cortical_global_delta | 0.0500 | `video_id_time_only` | 0.1427 | 0.2249 | 0.5372 | 0.3801 |
| cortical_global_delta | 0.0750 | `real_cortical` | 0.1500 | 0.1548 | 0.5437 | 0.9430 |
| cortical_global_delta | 0.0750 | `label_shuffle_across_videos` | 0.0660 | NA | 0.4998 | 0.0000 |
| cortical_global_delta | 0.0750 | `label_shuffle_within_video` | 0.0835 | 0.0084 | 0.5007 | 0.0044 |
| cortical_global_delta | 0.0750 | `feature_shuffle_across_videos` | 0.1158 | 0.1433 | 0.5008 | 0.9868 |
| cortical_global_delta | 0.0750 | `feature_shuffle_within_video` | 0.1153 | 0.1423 | 0.4971 | 0.9781 |
| cortical_global_delta | 0.0750 | `timestamp_only` | 0.0735 | 0.0297 | 0.4919 | 0.0219 |
| cortical_global_delta | 0.0750 | `video_id_time_only` | 0.1027 | 0.1885 | 0.5840 | 0.4298 |
| cortical_pca_64 | 0.0500 | `real_cortical` | 0.2455 | 0.2510 | 0.5452 | 0.9566 |
| cortical_pca_64 | 0.0500 | `label_shuffle_across_videos` | 0.1447 | 0.1009 | 0.5070 | 0.0740 |
| cortical_pca_64 | 0.0500 | `label_shuffle_within_video` | 0.1903 | 0.2077 | 0.5422 | 0.2143 |
| cortical_pca_64 | 0.0500 | `feature_shuffle_across_videos` | 0.1949 | 0.2349 | 0.5037 | 0.9770 |
| cortical_pca_64 | 0.0500 | `feature_shuffle_within_video` | 0.2091 | 0.2414 | 0.5219 | 0.9592 |
| cortical_pca_64 | 0.0500 | `timestamp_only` | 0.1281 | 0.1201 | 0.5054 | 0.1020 |
| cortical_pca_64 | 0.0500 | `video_id_time_only` | 0.1427 | 0.2249 | 0.5372 | 0.3801 |
| cortical_pca_64 | 0.0750 | `real_cortical` | 0.1575 | 0.1563 | 0.5480 | 0.8991 |
| cortical_pca_64 | 0.0750 | `label_shuffle_across_videos` | 0.0685 | 0.1236 | 0.4816 | 0.4474 |
| cortical_pca_64 | 0.0750 | `label_shuffle_within_video` | 0.0827 | 0.1434 | 0.5007 | 1.0000 |
| cortical_pca_64 | 0.0750 | `feature_shuffle_across_videos` | 0.1152 | 0.1434 | 0.5017 | 0.9781 |
| cortical_pca_64 | 0.0750 | `feature_shuffle_within_video` | 0.1187 | 0.1448 | 0.5074 | 0.9693 |
| cortical_pca_64 | 0.0750 | `timestamp_only` | 0.0735 | 0.0297 | 0.4919 | 0.0219 |
| cortical_pca_64 | 0.0750 | `video_id_time_only` | 0.1027 | 0.1885 | 0.5840 | 0.4298 |
| cortical_pca64_delta | 0.0500 | `real_cortical` | 0.2536 | 0.2677 | 0.5811 | 0.8724 |
| cortical_pca64_delta | 0.0500 | `label_shuffle_across_videos` | 0.1426 | 0.1813 | 0.5131 | 0.2372 |
| cortical_pca64_delta | 0.0500 | `label_shuffle_within_video` | 0.1502 | 0.0098 | 0.4996 | 0.0051 |
| cortical_pca64_delta | 0.0500 | `feature_shuffle_across_videos` | 0.1953 | 0.2414 | 0.5235 | 0.9311 |
| cortical_pca64_delta | 0.0500 | `feature_shuffle_within_video` | 0.2093 | 0.2487 | 0.5407 | 0.9337 |
| cortical_pca64_delta | 0.0500 | `timestamp_only` | 0.1281 | 0.1201 | 0.5054 | 0.1020 |
| cortical_pca64_delta | 0.0500 | `video_id_time_only` | 0.1427 | 0.2249 | 0.5372 | 0.3801 |
| cortical_pca64_delta | 0.0750 | `real_cortical` | 0.1614 | 0.1682 | 0.5807 | 0.8640 |
| cortical_pca64_delta | 0.0750 | `label_shuffle_across_videos` | 0.0806 | 0.1501 | 0.5339 | 0.5263 |
| cortical_pca64_delta | 0.0750 | `label_shuffle_within_video` | 0.1036 | 0.1484 | 0.5268 | 0.7018 |
| cortical_pca64_delta | 0.0750 | `feature_shuffle_across_videos` | 0.1176 | 0.1491 | 0.5239 | 0.9474 |
| cortical_pca64_delta | 0.0750 | `feature_shuffle_within_video` | 0.1141 | 0.1499 | 0.5272 | 0.9167 |
| cortical_pca64_delta | 0.0750 | `timestamp_only` | 0.0735 | 0.0297 | 0.4919 | 0.0219 |
| cortical_pca64_delta | 0.0750 | `video_id_time_only` | 0.1027 | 0.1885 | 0.5840 | 0.4298 |

## SECTION 7: Onset-Only Evaluation

| Feature | Thr | Lead-up | PR-AUC real | F1 real | BalAcc real | Rec real | Event count |
|---|---:|---:|---:|---:|---:|---:|---:|
| cortical_pca_64 | 0.0300 | 1 | 0.1596 | 0.0462 | 0.5109 | 0.0241 | 332.0000 |
| cortical_pca64_delta | 0.0300 | 1 | 0.1576 | 0.0389 | 0.5065 | 0.0211 | 332.0000 |
| cortical_pca_64 | 0.0500 | 1 | 0.1508 | 0.2116 | 0.5866 | 0.3510 | 245.0000 |
| cortical_global_delta | 0.0300 | 1 | 0.1448 | 0.0292 | 0.5066 | 0.0151 | 332.0000 |
| cortical_fast_default | 0.0300 | 1 | 0.1419 | 0.0391 | 0.5069 | 0.0211 | 332.0000 |
| cortical_pca64_delta | 0.0500 | 1 | 0.1269 | 0.1604 | 0.5426 | 0.1388 | 245.0000 |
| cortical_pca_64 | 0.0500 | 2 | 0.1138 | 0.1693 | 0.5869 | 0.3989 | 188.0000 |
| cortical_global_delta | 0.0500 | 2 | 0.1102 | 0.1423 | 0.5400 | 0.1064 | 188.0000 |

## SECTION 8: Local-Normalized Target Evaluation

| Feature | Variant | Window | Thr | PR-AUC real | F1 real | Pass controls |
|---|---|---:|---:|---:|---:|---|
| cortical_pca64_delta | `local__future_change_local_volatility` | 3 | 0.0300 | 0.9909 | 0.7700 | False |
| cortical_pca_64 | `local__future_change_local_volatility` | 3 | 0.0300 | 0.9893 | 0.4020 | False |
| cortical_pca64_delta | `local__future_change_local_volatility` | 3 | 0.0500 | 0.9880 | 0.4374 | False |
| cortical_global_delta | `local__future_change_local_volatility` | 3 | 0.0300 | 0.9879 | 0.2855 | False |
| cortical_pca_64 | `local__future_change_local_volatility` | 3 | 0.0500 | 0.9874 | 0.0986 | False |
| cortical_global_delta | `local__future_change_local_volatility` | 3 | 0.0500 | 0.9873 | 0.0226 | False |
| cortical_fast_default | `local__future_change_local_volatility` | 3 | 0.0500 | 0.9868 | 0.0071 | False |
| cortical_fast_default | `local__future_change_local_volatility` | 3 | 0.0300 | 0.9865 | 0.1825 | False |

## SECTION 9: Per-Video Robustness

Enough-event video rows: 224. Wins versus AR, shuffled, and random by per-video PR-AUC: 89/224.

Low-event videos are explicitly flagged in the per-video CSV; do not treat their per-video F1/PR-AUC as stable.

## SECTION 10: Final Recommendation

Recommendation: **promising but needs alignment fix**.

For follow-up or replication:

1. Blocked arousal__future_spike_1_3s at thresholds 0.05 and 0.075 for all four feature modes.
2. Blocked movement threshold sweep for p2/p3 future-change at 0.05 and 0.075.
3. Shift audit at -5,-3,-2,-1,0,+1,+2,+3,+5 before making any investor-facing claim.
4. Onset-only spike detection at lead-up windows 2s, 3s, and 5s.
5. Per-video contribution audit to verify gains are not dominated by a few high-event videos.

## Output Files

- JSON: `/Users/maxsartini/MiroFish-Offline-main/benchmarks/veatic/veatic_124_retest_event_spike_core_20260616.json`
- Diagnostics CSV: `/Users/maxsartini/MiroFish-Offline-main/benchmarks/veatic/veatic_124_retest_event_spike_core_20260616.diagnostics.csv`
- Shift audit CSV: `/Users/maxsartini/MiroFish-Offline-main/benchmarks/veatic/veatic_124_retest_event_spike_core_20260616.shift_audit.csv`
- Per-video CSV: `/Users/maxsartini/MiroFish-Offline-main/benchmarks/veatic/veatic_124_retest_event_spike_core_20260616.per_video.csv`
- Onset-only CSV: `/Users/maxsartini/MiroFish-Offline-main/benchmarks/veatic/veatic_124_retest_event_spike_core_20260616.onset_only.csv`
- Local targets CSV: `/Users/maxsartini/MiroFish-Offline-main/benchmarks/veatic/veatic_124_retest_event_spike_core_20260616.local_targets.csv`
