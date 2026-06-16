# VEATIC 124 Event-Conditioned Retest

## SECTION 1: Executive Verdict

Classification: **pre-event promising**.

The full-frame benchmark remains the real-world baseline. These rows test whether stable zero/non-event frames are suppressing cortical signal in event-relevant regions.

Decision thresholds were selected on train predictions only and then applied unchanged to every filtered test subset. Filtered test subsets were not used for threshold tuning.

## SECTION 2: Full-Frame vs Event-Conditioned Comparison

| Feature | Target | Thr | Mask | Real PR-AUC | Real-vs-AR | Real-vs-shuf | Real-vs-rand |
|---|---|---:|---|---:|---:|---:|---:|
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | all_frames | 0.3101 | 0.0052 | 0.0087 | 0.0056 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | all_frames | 0.3040 | -0.0009 | 0.0025 | -0.0044 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | all_frames | 0.3007 | -0.0042 | 0.0206 | -0.0007 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | all_frames | 0.2996 | -0.0053 | -0.0109 | 0.0198 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0500 | all_frames | 0.3448 | 0.0073 | 0.0067 | 0.0097 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0500 | all_frames | 0.3458 | 0.0082 | 0.0143 | 0.0118 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0500 | all_frames | 0.3762 | 0.0387 | 0.0525 | 0.0609 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0500 | all_frames | 0.3858 | 0.0482 | 0.0519 | 0.0526 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0750 | all_frames | 0.2763 | 0.0034 | 0.0066 | 0.0052 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0750 | all_frames | 0.3008 | 0.0279 | 0.0302 | 0.0391 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0750 | all_frames | 0.2503 | -0.0226 | 0.0154 | 0.0067 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0750 | all_frames | 0.2853 | 0.0124 | 0.0340 | 0.0235 |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0500 | all_frames | 0.2052 | 0.0083 | 0.0086 | 0.0072 |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0500 | all_frames | 0.2196 | 0.0227 | 0.0233 | 0.0238 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0500 | all_frames | 0.2536 | 0.0567 | 0.0561 | 0.0404 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0500 | all_frames | 0.2455 | 0.0486 | 0.0523 | 0.0507 |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0750 | all_frames | 0.1330 | 0.0179 | 0.0167 | 0.0165 |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0750 | all_frames | 0.1500 | 0.0348 | 0.0324 | 0.0313 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0750 | all_frames | 0.1614 | 0.0462 | 0.0524 | 0.0409 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0750 | all_frames | 0.1575 | 0.0424 | 0.0447 | 0.0436 |

Event-plus-pre rows are positive-only masks, so PR-AUC is undefined there; balanced event-vs-stable sampling below is the clean event-conditioned discrimination test.

| Feature | Target | Thr | Mask | Real recall | Real top-10% recall | Count |
|---|---|---:|---|---:|---:|---:|
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | event_plus_pre_3s | 0.1064 | 0.1006 | 855.0000 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | event_plus_pre_3s | 0.0842 | 0.1006 | 855.0000 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | event_plus_pre_3s | 0.1322 | 0.1006 | 855.0000 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | event_plus_pre_3s | 0.0842 | 0.1006 | 855.0000 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0500 | event_plus_pre_3s | 0.0579 | 0.1002 | 1158.0000 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0500 | event_plus_pre_3s | 0.0553 | 0.1002 | 1158.0000 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0500 | event_plus_pre_3s | 0.2737 | 0.1002 | 1158.0000 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0500 | event_plus_pre_3s | 0.0967 | 0.1002 | 1158.0000 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0750 | event_plus_pre_3s | 0.6569 | 0.1012 | 682.0000 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0750 | event_plus_pre_3s | 0.3372 | 0.1012 | 682.0000 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0750 | event_plus_pre_3s | 0.4428 | 0.1012 | 682.0000 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0750 | event_plus_pre_3s | 0.4868 | 0.1012 | 682.0000 |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0500 | event_plus_pre_3s | 0.9778 | 0.1006 | 676.0000 |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0500 | event_plus_pre_3s | 0.9512 | 0.1006 | 676.0000 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0500 | event_plus_pre_3s | 0.8225 | 0.1006 | 676.0000 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0500 | event_plus_pre_3s | 0.9305 | 0.1006 | 676.0000 |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0750 | event_plus_pre_3s | 0.9603 | 0.1005 | 428.0000 |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0750 | event_plus_pre_3s | 0.9276 | 0.1005 | 428.0000 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0750 | event_plus_pre_3s | 0.7850 | 0.1005 | 428.0000 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0750 | event_plus_pre_3s | 0.8692 | 0.1005 | 428.0000 |

## SECTION 3: Pre-Event Detection

Pre-event rows treat the pre-event frame as the positive early-warning region while preserving train-only thresholds. PR-AUC is undefined for positive-only subsets, so recall/top-k and balanced sampling carry more weight.

| Feature | Target | Thr | Mask | Real recall | Real top-10% recall | Real-vs-AR recall | Real-vs-shuf recall | Real-vs-rand recall |
|---|---|---:|---|---:|---:|---:|---:|---:|
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_1s | 0.0640 | 0.1047 | 0.0233 | 0.0349 | 0.0233 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_1s | 0.0349 | 0.1047 | -0.0058 | 0.0058 | -0.0174 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_1s | 0.0814 | 0.1047 | 0.0407 | -0.0291 | -0.0814 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_1s | 0.0465 | 0.1047 | 0.0058 | 0.0000 | 0.0174 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_2s | 0.0531 | 0.1020 | -0.0041 | 0.0082 | -0.0041 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_2s | 0.0408 | 0.1020 | -0.0163 | -0.0082 | -0.0204 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_2s | 0.0653 | 0.1020 | 0.0082 | -0.0694 | -0.0980 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_2s | 0.0531 | 0.1020 | -0.0041 | -0.0082 | 0.0122 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_3s | 0.0618 | 0.1004 | -0.0077 | 0.0077 | -0.0077 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_3s | 0.0425 | 0.1004 | -0.0270 | -0.0154 | -0.0270 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_3s | 0.0579 | 0.1004 | -0.0116 | -0.0502 | -0.0579 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_3s | 0.0502 | 0.1004 | -0.0193 | -0.0193 | 0.0039 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_5s | 0.0787 | 0.1024 | 0.0079 | 0.0197 | 0.0079 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_5s | 0.0394 | 0.1024 | -0.0315 | -0.0079 | -0.0197 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_5s | 0.0669 | 0.1024 | -0.0039 | -0.0394 | -0.0512 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_5s | 0.0512 | 0.1024 | -0.0197 | -0.0079 | 0.0039 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0500 | pre_event_1s | 0.0361 | 0.1031 | 0.0052 | 0.0052 | 0.0052 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0500 | pre_event_1s | 0.0258 | 0.1031 | -0.0052 | -0.0103 | 0.0000 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0500 | pre_event_1s | 0.2165 | 0.1031 | 0.1856 | 0.1856 | 0.1856 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0500 | pre_event_1s | 0.0670 | 0.1031 | 0.0361 | 0.0258 | 0.0412 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0500 | pre_event_2s | 0.0441 | 0.1017 | 0.0068 | 0.0068 | 0.0034 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0500 | pre_event_2s | 0.0339 | 0.1017 | -0.0034 | -0.0068 | -0.0034 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0500 | pre_event_2s | 0.2271 | 0.1017 | 0.1898 | 0.1966 | 0.1729 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0500 | pre_event_2s | 0.0746 | 0.1017 | 0.0373 | 0.0339 | 0.0475 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0500 | pre_event_3s | 0.0469 | 0.1026 | 0.0088 | 0.0059 | 0.0000 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0500 | pre_event_3s | 0.0381 | 0.1026 | 0.0000 | -0.0059 | -0.0029 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0500 | pre_event_3s | 0.2170 | 0.1026 | 0.1789 | 0.1789 | 0.1672 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0500 | pre_event_3s | 0.0762 | 0.1026 | 0.0381 | 0.0352 | 0.0440 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0500 | pre_event_5s | 0.0368 | 0.1012 | 0.0061 | 0.0092 | 0.0092 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0500 | pre_event_5s | 0.0245 | 0.1012 | -0.0061 | -0.0061 | 0.0000 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0500 | pre_event_5s | 0.2147 | 0.1012 | 0.1840 | 0.1871 | 0.1718 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0500 | pre_event_5s | 0.0583 | 0.1012 | 0.0276 | 0.0368 | 0.0337 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0750 | pre_event_1s | 0.6230 | 0.1066 | -0.0246 | -0.1230 | -0.0164 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0750 | pre_event_1s | 0.2787 | 0.1066 | -0.3689 | -0.3361 | -0.4426 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0750 | pre_event_1s | 0.4098 | 0.1066 | -0.2377 | -0.0492 | 0.0328 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0750 | pre_event_1s | 0.4262 | 0.1066 | -0.2213 | 0.0738 | -0.1721 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0750 | pre_event_2s | 0.5979 | 0.1005 | -0.0476 | -0.1323 | -0.0370 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0750 | pre_event_2s | 0.2381 | 0.1005 | -0.4074 | -0.3598 | -0.4921 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0750 | pre_event_2s | 0.4127 | 0.1005 | -0.2328 | -0.0053 | 0.0370 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0750 | pre_event_2s | 0.4074 | 0.1005 | -0.2381 | 0.0635 | -0.1746 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0750 | pre_event_3s | 0.5591 | 0.1000 | -0.0591 | -0.1545 | -0.0591 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0750 | pre_event_3s | 0.2000 | 0.1000 | -0.4182 | -0.3864 | -0.5182 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0750 | pre_event_3s | 0.4045 | 0.1000 | -0.2136 | -0.0182 | 0.1182 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0750 | pre_event_3s | 0.4000 | 0.1000 | -0.2182 | 0.0636 | -0.1591 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0750 | pre_event_5s | 0.4781 | 0.1009 | -0.0789 | -0.2061 | -0.0921 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0750 | pre_event_5s | 0.1623 | 0.1009 | -0.3947 | -0.3860 | -0.5263 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0750 | pre_event_5s | 0.3684 | 0.1009 | -0.1886 | 0.0088 | 0.0833 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0750 | pre_event_5s | 0.3904 | 0.1009 | -0.1667 | 0.0833 | -0.0921 |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0500 | pre_event_1s | 0.9810 | 0.1048 | -0.0095 | -0.0095 | 0.0095 |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0500 | pre_event_1s | 0.9524 | 0.1048 | -0.0381 | -0.0381 | -0.0190 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0500 | pre_event_1s | 0.8000 | 0.1048 | -0.1905 | -0.1238 | -0.0571 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0500 | pre_event_1s | 0.9238 | 0.1048 | -0.0667 | -0.0667 | -0.0286 |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0500 | pre_event_2s | 0.9771 | 0.1029 | -0.0114 | -0.0057 | 0.0114 |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0500 | pre_event_2s | 0.9314 | 0.1029 | -0.0571 | -0.0457 | -0.0457 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0500 | pre_event_2s | 0.7543 | 0.1029 | -0.2343 | -0.1657 | -0.1086 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0500 | pre_event_2s | 0.9086 | 0.1029 | -0.0800 | -0.0514 | -0.0514 |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0500 | pre_event_3s | 0.9731 | 0.1031 | -0.0179 | -0.0135 | 0.0000 |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0500 | pre_event_3s | 0.9283 | 0.1031 | -0.0628 | -0.0493 | -0.0448 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0500 | pre_event_3s | 0.7309 | 0.1031 | -0.2601 | -0.1839 | -0.1076 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0500 | pre_event_3s | 0.8834 | 0.1031 | -0.1076 | -0.0807 | -0.0852 |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0500 | pre_event_5s | 0.9828 | 0.1034 | -0.0172 | -0.0086 | -0.0172 |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0500 | pre_event_5s | 0.9440 | 0.1034 | -0.0560 | -0.0517 | -0.0474 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0500 | pre_event_5s | 0.7026 | 0.1034 | -0.2974 | -0.2328 | -0.1164 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0500 | pre_event_5s | 0.8578 | 0.1034 | -0.1422 | -0.1250 | -0.1121 |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0750 | pre_event_1s | 0.9595 | 0.1081 | -0.0405 | -0.0135 | 0.0000 |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0750 | pre_event_1s | 0.9189 | 0.1081 | -0.0811 | -0.0405 | -0.0676 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0750 | pre_event_1s | 0.7703 | 0.1081 | -0.2297 | -0.2027 | -0.0946 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0750 | pre_event_1s | 0.8649 | 0.1081 | -0.1351 | -0.0811 | -0.0946 |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0750 | pre_event_2s | 0.9603 | 0.1032 | -0.0397 | -0.0238 | -0.0079 |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0750 | pre_event_2s | 0.9048 | 0.1032 | -0.0952 | -0.0794 | -0.0873 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0750 | pre_event_2s | 0.7302 | 0.1032 | -0.2698 | -0.2222 | -0.1587 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0750 | pre_event_2s | 0.8413 | 0.1032 | -0.1587 | -0.1111 | -0.1032 |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0750 | pre_event_3s | 0.9595 | 0.1014 | -0.0405 | -0.0203 | -0.0338 |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0750 | pre_event_3s | 0.9257 | 0.1014 | -0.0743 | -0.0743 | -0.0676 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0750 | pre_event_3s | 0.7027 | 0.1014 | -0.2973 | -0.2635 | -0.1757 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0750 | pre_event_3s | 0.8446 | 0.1014 | -0.1554 | -0.1081 | -0.1014 |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0750 | pre_event_5s | 0.9557 | 0.1013 | -0.0443 | -0.0380 | -0.0443 |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0750 | pre_event_5s | 0.8354 | 0.1013 | -0.1646 | -0.1582 | -0.1519 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0750 | pre_event_5s | 0.6582 | 0.1013 | -0.3418 | -0.3165 | -0.2089 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0750 | pre_event_5s | 0.7342 | 0.1013 | -0.2658 | -0.2278 | -0.2278 |

## SECTION 4: Event-Only Detection

| Feature | Target | Thr | Real recall | Real top-10% recall | Event count |
|---|---|---:|---:|---:|---:|
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | 0.1630 | 0.1022 | 411.0000 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | 0.1363 | 0.1022 | 411.0000 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | 0.1995 | 0.1022 | 411.0000 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | 0.1290 | 0.1022 | 411.0000 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0500 | 0.0749 | 0.1003 | 668.0000 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0500 | 0.0763 | 0.1003 | 668.0000 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0500 | 0.3219 | 0.1003 | 668.0000 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0500 | 0.1243 | 0.1003 | 668.0000 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0750 | 0.7273 | 0.1019 | 363.0000 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0750 | 0.4298 | 0.1019 | 363.0000 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0750 | 0.4793 | 0.1019 | 363.0000 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0750 | 0.5620 | 0.1019 | 363.0000 |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0500 | 0.9821 | 0.1020 | 392.0000 |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0500 | 0.9643 | 0.1020 | 392.0000 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0500 | 0.8724 | 0.1020 | 392.0000 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0500 | 0.9566 | 0.1020 | 392.0000 |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0750 | 0.9693 | 0.1009 | 228.0000 |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0750 | 0.9430 | 0.1009 | 228.0000 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0750 | 0.8640 | 0.1009 | 228.0000 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0750 | 0.8991 | 0.1009 | 228.0000 |

## SECTION 5: Balanced Event-vs-Stable Sampling

| Feature | Target | Thr | Ratio | Model | PR-AUC mean +/- std | F1 mean +/- std | BalAcc mean +/- std | Recall mean +/- std |
|---|---|---:|---|---|---:|---:|---:|---:|
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | 1:1 | ar | 0.6374 +/- 0.0065 | 0.1672 +/- 0.0007 | 0.5339 +/- 0.0024 | 0.0936 +/- 0.0000 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | 1:1 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | 1:1 | random | 0.6368 +/- 0.0066 | 0.1671 +/- 0.0007 | 0.5335 +/- 0.0023 | 0.0936 +/- 0.0000 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | 1:1 | real | 0.6376 +/- 0.0078 | 0.1869 +/- 0.0008 | 0.5368 +/- 0.0026 | 0.1064 +/- 0.0000 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | 1:1 | shuffled | 0.6358 +/- 0.0065 | 0.1488 +/- 0.0005 | 0.5317 +/- 0.0018 | 0.0819 +/- 0.0000 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | 1:1 | ar | 0.6374 +/- 0.0065 | 0.1672 +/- 0.0007 | 0.5339 +/- 0.0024 | 0.0936 +/- 0.0000 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | 1:1 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | 1:1 | random | 0.6352 +/- 0.0064 | 0.1763 +/- 0.0007 | 0.5356 +/- 0.0022 | 0.0994 +/- 0.0000 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | 1:1 | real | 0.6326 +/- 0.0075 | 0.1526 +/- 0.0005 | 0.5322 +/- 0.0017 | 0.0842 +/- 0.0000 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | 1:1 | shuffled | 0.6345 +/- 0.0060 | 0.1545 +/- 0.0005 | 0.5328 +/- 0.0018 | 0.0854 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | 1:1 | ar | 0.6374 +/- 0.0065 | 0.1672 +/- 0.0007 | 0.5339 +/- 0.0024 | 0.0936 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | 1:1 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | 1:1 | random | 0.6237 +/- 0.0073 | 0.3333 +/- 0.0015 | 0.5625 +/- 0.0030 | 0.2187 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | 1:1 | real | 0.6214 +/- 0.0081 | 0.2237 +/- 0.0008 | 0.5414 +/- 0.0022 | 0.1322 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | 1:1 | shuffled | 0.6105 +/- 0.0061 | 0.2516 +/- 0.0009 | 0.5476 +/- 0.0021 | 0.1520 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | 1:1 | ar | 0.6374 +/- 0.0065 | 0.1672 +/- 0.0007 | 0.5339 +/- 0.0024 | 0.0936 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | 1:1 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | 1:1 | random | 0.6203 +/- 0.0068 | 0.1370 +/- 0.0004 | 0.5285 +/- 0.0016 | 0.0749 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | 1:1 | real | 0.6312 +/- 0.0078 | 0.1519 +/- 0.0005 | 0.5299 +/- 0.0018 | 0.0842 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | 1:1 | shuffled | 0.6375 +/- 0.0067 | 0.1786 +/- 0.0006 | 0.5374 +/- 0.0018 | 0.1006 +/- 0.0000 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | 1:2 | ar | 0.4731 +/- 0.0044 | 0.1631 +/- 0.0005 | 0.5334 +/- 0.0009 | 0.0936 +/- 0.0000 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | 1:2 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | 1:2 | random | 0.4725 +/- 0.0043 | 0.1629 +/- 0.0005 | 0.5329 +/- 0.0009 | 0.0936 +/- 0.0000 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | 1:2 | real | 0.4746 +/- 0.0045 | 0.1812 +/- 0.0007 | 0.5362 +/- 0.0011 | 0.1064 +/- 0.0000 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | 1:2 | shuffled | 0.4711 +/- 0.0044 | 0.1460 +/- 0.0005 | 0.5311 +/- 0.0009 | 0.0819 +/- 0.0000 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | 1:2 | ar | 0.4731 +/- 0.0044 | 0.1631 +/- 0.0005 | 0.5334 +/- 0.0009 | 0.0936 +/- 0.0000 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | 1:2 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | 1:2 | random | 0.4718 +/- 0.0042 | 0.1716 +/- 0.0007 | 0.5349 +/- 0.0012 | 0.0994 +/- 0.0000 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | 1:2 | real | 0.4686 +/- 0.0042 | 0.1496 +/- 0.0005 | 0.5317 +/- 0.0010 | 0.0842 +/- 0.0000 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | 1:2 | shuffled | 0.4707 +/- 0.0042 | 0.1514 +/- 0.0004 | 0.5321 +/- 0.0008 | 0.0854 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | 1:2 | ar | 0.4731 +/- 0.0044 | 0.1631 +/- 0.0005 | 0.5334 +/- 0.0009 | 0.0936 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | 1:2 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | 1:2 | random | 0.4587 +/- 0.0047 | 0.3110 +/- 0.0018 | 0.5624 +/- 0.0021 | 0.2187 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | 1:2 | real | 0.4591 +/- 0.0035 | 0.2148 +/- 0.0008 | 0.5415 +/- 0.0012 | 0.1322 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | 1:2 | shuffled | 0.4449 +/- 0.0052 | 0.2396 +/- 0.0013 | 0.5467 +/- 0.0017 | 0.1520 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | 1:2 | ar | 0.4731 +/- 0.0044 | 0.1631 +/- 0.0005 | 0.5334 +/- 0.0009 | 0.0936 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | 1:2 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | 1:2 | random | 0.4537 +/- 0.0045 | 0.1345 +/- 0.0004 | 0.5279 +/- 0.0007 | 0.0749 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | 1:2 | real | 0.4680 +/- 0.0038 | 0.1484 +/- 0.0004 | 0.5295 +/- 0.0008 | 0.0842 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | 1:2 | shuffled | 0.4746 +/- 0.0048 | 0.1744 +/- 0.0006 | 0.5370 +/- 0.0010 | 0.1006 +/- 0.0000 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | 1:3 | ar | 0.3931 +/- 0.0001 | 0.1600 +/- 0.0000 | 0.5334 +/- 0.0000 | 0.0936 +/- 0.0000 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | 1:3 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | 1:3 | random | 0.3925 +/- 0.0001 | 0.1597 +/- 0.0000 | 0.5329 +/- 0.0000 | 0.0936 +/- 0.0000 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | 1:3 | real | 0.3952 +/- 0.0001 | 0.1770 +/- 0.0000 | 0.5363 +/- 0.0000 | 0.1064 +/- 0.0000 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | 1:3 | shuffled | 0.3916 +/- 0.0001 | 0.1439 +/- 0.0000 | 0.5310 +/- 0.0000 | 0.0819 +/- 0.0000 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | 1:3 | ar | 0.3931 +/- 0.0001 | 0.1600 +/- 0.0000 | 0.5334 +/- 0.0000 | 0.0936 +/- 0.0000 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | 1:3 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | 1:3 | random | 0.3922 +/- 0.0001 | 0.1682 +/- 0.0000 | 0.5350 +/- 0.0000 | 0.0994 +/- 0.0000 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | 1:3 | real | 0.3897 +/- 0.0001 | 0.1472 +/- 0.0000 | 0.5316 +/- 0.0000 | 0.0842 +/- 0.0000 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | 1:3 | shuffled | 0.3909 +/- 0.0001 | 0.1491 +/- 0.0000 | 0.5322 +/- 0.0000 | 0.0854 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | 1:3 | ar | 0.3931 +/- 0.0001 | 0.1600 +/- 0.0000 | 0.5334 +/- 0.0000 | 0.0936 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | 1:3 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | 1:3 | random | 0.3794 +/- 0.0004 | 0.2943 +/- 0.0000 | 0.5621 +/- 0.0000 | 0.2187 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | 1:3 | real | 0.3807 +/- 0.0002 | 0.2083 +/- 0.0000 | 0.5419 +/- 0.0000 | 0.1322 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | 1:3 | shuffled | 0.3659 +/- 0.0002 | 0.2307 +/- 0.0000 | 0.5467 +/- 0.0000 | 0.1520 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | 1:3 | ar | 0.3931 +/- 0.0001 | 0.1600 +/- 0.0000 | 0.5334 +/- 0.0000 | 0.0936 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | 1:3 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | 1:3 | random | 0.3744 +/- 0.0001 | 0.1326 +/- 0.0000 | 0.5279 +/- 0.0000 | 0.0749 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | 1:3 | real | 0.3894 +/- 0.0001 | 0.1457 +/- 0.0000 | 0.5295 +/- 0.0000 | 0.0842 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | 1:3 | shuffled | 0.3945 +/- 0.0001 | 0.1710 +/- 0.0000 | 0.5369 +/- 0.0000 | 0.1006 +/- 0.0000 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0500 | 1:1 | ar | 0.5901 +/- 0.0048 | 0.0928 +/- 0.0002 | 0.5190 +/- 0.0009 | 0.0492 +/- 0.0000 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0500 | 1:1 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0500 | 1:1 | random | 0.5889 +/- 0.0048 | 0.1004 +/- 0.0002 | 0.5201 +/- 0.0010 | 0.0535 +/- 0.0000 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0500 | 1:1 | real | 0.5918 +/- 0.0044 | 0.1080 +/- 0.0002 | 0.5220 +/- 0.0009 | 0.0579 +/- 0.0000 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0500 | 1:1 | shuffled | 0.5898 +/- 0.0048 | 0.0898 +/- 0.0002 | 0.5184 +/- 0.0009 | 0.0475 +/- 0.0000 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0500 | 1:1 | ar | 0.5901 +/- 0.0048 | 0.0928 +/- 0.0002 | 0.5190 +/- 0.0009 | 0.0492 +/- 0.0000 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0500 | 1:1 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0500 | 1:1 | random | 0.5887 +/- 0.0048 | 0.0837 +/- 0.0001 | 0.5176 +/- 0.0008 | 0.0440 +/- 0.0000 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0500 | 1:1 | real | 0.5907 +/- 0.0049 | 0.1037 +/- 0.0002 | 0.5221 +/- 0.0010 | 0.0553 +/- 0.0000 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0500 | 1:1 | shuffled | 0.5871 +/- 0.0047 | 0.0865 +/- 0.0001 | 0.5168 +/- 0.0008 | 0.0458 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0500 | 1:1 | ar | 0.5901 +/- 0.0048 | 0.0928 +/- 0.0002 | 0.5190 +/- 0.0009 | 0.0492 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0500 | 1:1 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0500 | 1:1 | random | 0.5776 +/- 0.0055 | 0.0860 +/- 0.0002 | 0.5139 +/- 0.0012 | 0.0458 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0500 | 1:1 | real | 0.6003 +/- 0.0067 | 0.3820 +/- 0.0021 | 0.5571 +/- 0.0040 | 0.2737 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0500 | 1:1 | shuffled | 0.5787 +/- 0.0037 | 0.0832 +/- 0.0001 | 0.5150 +/- 0.0007 | 0.0440 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0500 | 1:1 | ar | 0.5901 +/- 0.0048 | 0.0928 +/- 0.0002 | 0.5190 +/- 0.0009 | 0.0492 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0500 | 1:1 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0500 | 1:1 | random | 0.5870 +/- 0.0044 | 0.0726 +/- 0.0001 | 0.5149 +/- 0.0008 | 0.0380 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0500 | 1:1 | real | 0.6189 +/- 0.0059 | 0.1711 +/- 0.0005 | 0.5315 +/- 0.0016 | 0.0967 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0500 | 1:1 | shuffled | 0.5898 +/- 0.0049 | 0.0881 +/- 0.0002 | 0.5173 +/- 0.0010 | 0.0466 +/- 0.0000 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0500 | 1:2 | ar | 0.4488 +/- 0.0001 | 0.0919 +/- 0.0000 | 0.5185 +/- 0.0000 | 0.0492 +/- 0.0000 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0500 | 1:2 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0500 | 1:2 | random | 0.4475 +/- 0.0001 | 0.0992 +/- 0.0000 | 0.5194 +/- 0.0000 | 0.0535 +/- 0.0000 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0500 | 1:2 | real | 0.4512 +/- 0.0001 | 0.1068 +/- 0.0000 | 0.5216 +/- 0.0000 | 0.0579 +/- 0.0000 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0500 | 1:2 | shuffled | 0.4487 +/- 0.0001 | 0.0889 +/- 0.0000 | 0.5179 +/- 0.0000 | 0.0475 +/- 0.0000 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0500 | 1:2 | ar | 0.4488 +/- 0.0001 | 0.0919 +/- 0.0000 | 0.5185 +/- 0.0000 | 0.0492 +/- 0.0000 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0500 | 1:2 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0500 | 1:2 | random | 0.4472 +/- 0.0001 | 0.0830 +/- 0.0000 | 0.5171 +/- 0.0000 | 0.0440 +/- 0.0000 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0500 | 1:2 | real | 0.4507 +/- 0.0001 | 0.1027 +/- 0.0000 | 0.5218 +/- 0.0000 | 0.0553 +/- 0.0000 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0500 | 1:2 | shuffled | 0.4457 +/- 0.0001 | 0.0857 +/- 0.0000 | 0.5165 +/- 0.0000 | 0.0458 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0500 | 1:2 | ar | 0.4488 +/- 0.0001 | 0.0919 +/- 0.0000 | 0.5185 +/- 0.0000 | 0.0492 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0500 | 1:2 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0500 | 1:2 | random | 0.4345 +/- 0.0001 | 0.0848 +/- 0.0000 | 0.5133 +/- 0.0000 | 0.0458 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0500 | 1:2 | real | 0.4634 +/- 0.0001 | 0.3514 +/- 0.0000 | 0.5562 +/- 0.0000 | 0.2737 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0500 | 1:2 | shuffled | 0.4385 +/- 0.0001 | 0.0823 +/- 0.0000 | 0.5147 +/- 0.0000 | 0.0440 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0500 | 1:2 | ar | 0.4488 +/- 0.0001 | 0.0919 +/- 0.0000 | 0.5185 +/- 0.0000 | 0.0492 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0500 | 1:2 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0500 | 1:2 | random | 0.4462 +/- 0.0001 | 0.0721 +/- 0.0000 | 0.5146 +/- 0.0000 | 0.0380 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0500 | 1:2 | real | 0.4797 +/- 0.0003 | 0.1669 +/- 0.0000 | 0.5307 +/- 0.0000 | 0.0967 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0500 | 1:2 | shuffled | 0.4479 +/- 0.0001 | 0.0872 +/- 0.0000 | 0.5169 +/- 0.0000 | 0.0466 +/- 0.0000 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0500 | 1:3 | ar | 0.4488 +/- 0.0001 | 0.0919 +/- 0.0000 | 0.5185 +/- 0.0000 | 0.0492 +/- 0.0000 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0500 | 1:3 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0500 | 1:3 | random | 0.4475 +/- 0.0001 | 0.0992 +/- 0.0000 | 0.5194 +/- 0.0000 | 0.0535 +/- 0.0000 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0500 | 1:3 | real | 0.4512 +/- 0.0001 | 0.1068 +/- 0.0000 | 0.5216 +/- 0.0000 | 0.0579 +/- 0.0000 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0500 | 1:3 | shuffled | 0.4487 +/- 0.0001 | 0.0889 +/- 0.0000 | 0.5179 +/- 0.0000 | 0.0475 +/- 0.0000 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0500 | 1:3 | ar | 0.4488 +/- 0.0001 | 0.0919 +/- 0.0000 | 0.5185 +/- 0.0000 | 0.0492 +/- 0.0000 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0500 | 1:3 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0500 | 1:3 | random | 0.4472 +/- 0.0001 | 0.0830 +/- 0.0000 | 0.5171 +/- 0.0000 | 0.0440 +/- 0.0000 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0500 | 1:3 | real | 0.4507 +/- 0.0001 | 0.1027 +/- 0.0000 | 0.5218 +/- 0.0000 | 0.0553 +/- 0.0000 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0500 | 1:3 | shuffled | 0.4457 +/- 0.0001 | 0.0857 +/- 0.0000 | 0.5165 +/- 0.0000 | 0.0458 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0500 | 1:3 | ar | 0.4488 +/- 0.0001 | 0.0919 +/- 0.0000 | 0.5185 +/- 0.0000 | 0.0492 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0500 | 1:3 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0500 | 1:3 | random | 0.4345 +/- 0.0001 | 0.0848 +/- 0.0000 | 0.5133 +/- 0.0000 | 0.0458 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0500 | 1:3 | real | 0.4634 +/- 0.0001 | 0.3514 +/- 0.0000 | 0.5562 +/- 0.0000 | 0.2737 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0500 | 1:3 | shuffled | 0.4385 +/- 0.0001 | 0.0823 +/- 0.0000 | 0.5147 +/- 0.0000 | 0.0440 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0500 | 1:3 | ar | 0.4488 +/- 0.0001 | 0.0919 +/- 0.0000 | 0.5185 +/- 0.0000 | 0.0492 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0500 | 1:3 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0500 | 1:3 | random | 0.4462 +/- 0.0001 | 0.0721 +/- 0.0000 | 0.5146 +/- 0.0000 | 0.0380 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0500 | 1:3 | real | 0.4797 +/- 0.0003 | 0.1669 +/- 0.0000 | 0.5307 +/- 0.0000 | 0.0967 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0500 | 1:3 | shuffled | 0.4479 +/- 0.0001 | 0.0872 +/- 0.0000 | 0.5169 +/- 0.0000 | 0.0466 +/- 0.0000 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0750 | 1:1 | ar | 0.6225 +/- 0.0073 | 0.6184 +/- 0.0041 | 0.5729 +/- 0.0074 | 0.6921 +/- 0.0000 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0750 | 1:1 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0750 | 1:1 | random | 0.6213 +/- 0.0071 | 0.6206 +/- 0.0037 | 0.5723 +/- 0.0066 | 0.6994 +/- 0.0000 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0750 | 1:1 | real | 0.6177 +/- 0.0075 | 0.6075 +/- 0.0042 | 0.5756 +/- 0.0074 | 0.6569 +/- 0.0000 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0750 | 1:1 | shuffled | 0.6215 +/- 0.0074 | 0.6414 +/- 0.0040 | 0.5622 +/- 0.0076 | 0.7830 +/- 0.0000 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0750 | 1:1 | ar | 0.6225 +/- 0.0073 | 0.6184 +/- 0.0041 | 0.5729 +/- 0.0074 | 0.6921 +/- 0.0000 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0750 | 1:1 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0750 | 1:1 | random | 0.6097 +/- 0.0075 | 0.6413 +/- 0.0039 | 0.5595 +/- 0.0074 | 0.7874 +/- 0.0000 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0750 | 1:1 | real | 0.6325 +/- 0.0075 | 0.4445 +/- 0.0028 | 0.5785 +/- 0.0048 | 0.3372 +/- 0.0000 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0750 | 1:1 | shuffled | 0.6186 +/- 0.0070 | 0.6131 +/- 0.0035 | 0.5669 +/- 0.0065 | 0.6862 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0750 | 1:1 | ar | 0.6225 +/- 0.0073 | 0.6184 +/- 0.0041 | 0.5729 +/- 0.0074 | 0.6921 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0750 | 1:1 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0750 | 1:1 | random | 0.6060 +/- 0.0085 | 0.4713 +/- 0.0034 | 0.5607 +/- 0.0060 | 0.3915 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0750 | 1:1 | real | 0.6114 +/- 0.0089 | 0.5037 +/- 0.0041 | 0.5637 +/- 0.0071 | 0.4428 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0750 | 1:1 | shuffled | 0.5999 +/- 0.0074 | 0.5405 +/- 0.0043 | 0.5637 +/- 0.0076 | 0.5132 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0750 | 1:1 | ar | 0.6225 +/- 0.0073 | 0.6184 +/- 0.0041 | 0.5729 +/- 0.0074 | 0.6921 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0750 | 1:1 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0750 | 1:1 | random | 0.6139 +/- 0.0078 | 0.5959 +/- 0.0045 | 0.5674 +/- 0.0082 | 0.6378 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0750 | 1:1 | real | 0.6307 +/- 0.0073 | 0.5325 +/- 0.0041 | 0.5726 +/- 0.0071 | 0.4868 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0750 | 1:1 | shuffled | 0.6095 +/- 0.0071 | 0.4890 +/- 0.0045 | 0.5602 +/- 0.0080 | 0.4208 +/- 0.0000 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0750 | 1:2 | ar | 0.4598 +/- 0.0064 | 0.4959 +/- 0.0042 | 0.5713 +/- 0.0059 | 0.6921 +/- 0.0000 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0750 | 1:2 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0750 | 1:2 | random | 0.4582 +/- 0.0067 | 0.4966 +/- 0.0040 | 0.5703 +/- 0.0057 | 0.6994 +/- 0.0000 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0750 | 1:2 | real | 0.4535 +/- 0.0056 | 0.4929 +/- 0.0045 | 0.5762 +/- 0.0061 | 0.6569 +/- 0.0000 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0750 | 1:2 | shuffled | 0.4584 +/- 0.0064 | 0.5059 +/- 0.0035 | 0.5633 +/- 0.0054 | 0.7830 +/- 0.0000 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0750 | 1:2 | ar | 0.4598 +/- 0.0064 | 0.4959 +/- 0.0042 | 0.5713 +/- 0.0059 | 0.6921 +/- 0.0000 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0750 | 1:2 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0750 | 1:2 | random | 0.4459 +/- 0.0062 | 0.5049 +/- 0.0037 | 0.5608 +/- 0.0057 | 0.7874 +/- 0.0000 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0750 | 1:2 | real | 0.4719 +/- 0.0052 | 0.3959 +/- 0.0025 | 0.5770 +/- 0.0027 | 0.3372 +/- 0.0000 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0750 | 1:2 | shuffled | 0.4557 +/- 0.0065 | 0.4907 +/- 0.0038 | 0.5654 +/- 0.0054 | 0.6862 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0750 | 1:2 | ar | 0.4598 +/- 0.0064 | 0.4959 +/- 0.0042 | 0.5713 +/- 0.0059 | 0.6921 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0750 | 1:2 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0750 | 1:2 | random | 0.4402 +/- 0.0058 | 0.4055 +/- 0.0033 | 0.5609 +/- 0.0039 | 0.3915 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0750 | 1:2 | real | 0.4483 +/- 0.0059 | 0.4282 +/- 0.0046 | 0.5650 +/- 0.0055 | 0.4428 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0750 | 1:2 | shuffled | 0.4325 +/- 0.0061 | 0.4496 +/- 0.0031 | 0.5641 +/- 0.0039 | 0.5132 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0750 | 1:2 | ar | 0.4598 +/- 0.0064 | 0.4959 +/- 0.0042 | 0.5713 +/- 0.0059 | 0.6921 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0750 | 1:2 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0750 | 1:2 | random | 0.4506 +/- 0.0068 | 0.4817 +/- 0.0043 | 0.5663 +/- 0.0059 | 0.6378 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0750 | 1:2 | real | 0.4684 +/- 0.0049 | 0.4500 +/- 0.0030 | 0.5741 +/- 0.0036 | 0.4868 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0750 | 1:2 | shuffled | 0.4464 +/- 0.0062 | 0.4166 +/- 0.0031 | 0.5605 +/- 0.0038 | 0.4208 +/- 0.0000 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0750 | 1:3 | ar | 0.3649 +/- 0.0025 | 0.4145 +/- 0.0016 | 0.5714 +/- 0.0022 | 0.6921 +/- 0.0000 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0750 | 1:3 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0750 | 1:3 | random | 0.3638 +/- 0.0024 | 0.4149 +/- 0.0016 | 0.5710 +/- 0.0022 | 0.6994 +/- 0.0000 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0750 | 1:3 | real | 0.3609 +/- 0.0025 | 0.4141 +/- 0.0017 | 0.5759 +/- 0.0022 | 0.6569 +/- 0.0000 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0750 | 1:3 | shuffled | 0.3635 +/- 0.0027 | 0.4173 +/- 0.0013 | 0.5632 +/- 0.0020 | 0.7830 +/- 0.0000 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0750 | 1:3 | ar | 0.3649 +/- 0.0025 | 0.4145 +/- 0.0016 | 0.5714 +/- 0.0022 | 0.6921 +/- 0.0000 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0750 | 1:3 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0750 | 1:3 | random | 0.3522 +/- 0.0025 | 0.4158 +/- 0.0012 | 0.5603 +/- 0.0019 | 0.7874 +/- 0.0000 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0750 | 1:3 | real | 0.3794 +/- 0.0024 | 0.3584 +/- 0.0013 | 0.5778 +/- 0.0012 | 0.3372 +/- 0.0000 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0750 | 1:3 | shuffled | 0.3612 +/- 0.0024 | 0.4094 +/- 0.0015 | 0.5655 +/- 0.0021 | 0.6862 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0750 | 1:3 | ar | 0.3649 +/- 0.0025 | 0.4145 +/- 0.0016 | 0.5714 +/- 0.0022 | 0.6921 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0750 | 1:3 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0750 | 1:3 | random | 0.3463 +/- 0.0021 | 0.3548 +/- 0.0017 | 0.5599 +/- 0.0018 | 0.3915 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0750 | 1:3 | real | 0.3549 +/- 0.0026 | 0.3721 +/- 0.0011 | 0.5652 +/- 0.0012 | 0.4428 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0750 | 1:3 | shuffled | 0.3382 +/- 0.0030 | 0.3841 +/- 0.0017 | 0.5634 +/- 0.0019 | 0.5132 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0750 | 1:3 | ar | 0.3649 +/- 0.0025 | 0.4145 +/- 0.0016 | 0.5714 +/- 0.0022 | 0.6921 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0750 | 1:3 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0750 | 1:3 | random | 0.3560 +/- 0.0026 | 0.4040 +/- 0.0014 | 0.5657 +/- 0.0019 | 0.6378 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0750 | 1:3 | real | 0.3741 +/- 0.0024 | 0.3888 +/- 0.0014 | 0.5739 +/- 0.0015 | 0.4868 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0750 | 1:3 | shuffled | 0.3516 +/- 0.0026 | 0.3615 +/- 0.0016 | 0.5592 +/- 0.0017 | 0.4208 +/- 0.0000 |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0500 | 1:1 | ar | 0.5293 +/- 0.0081 | 0.6640 +/- 0.0006 | 0.4992 +/- 0.0014 | 0.9896 +/- 0.0000 |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0500 | 1:1 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0500 | 1:1 | random | 0.5311 +/- 0.0083 | 0.6624 +/- 0.0012 | 0.5002 +/- 0.0027 | 0.9808 +/- 0.0000 |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0500 | 1:1 | real | 0.5347 +/- 0.0084 | 0.6609 +/- 0.0008 | 0.4983 +/- 0.0019 | 0.9778 +/- 0.0000 |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0500 | 1:1 | shuffled | 0.5293 +/- 0.0082 | 0.6647 +/- 0.0010 | 0.5023 +/- 0.0022 | 0.9867 +/- 0.0000 |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0500 | 1:1 | ar | 0.5293 +/- 0.0081 | 0.6640 +/- 0.0006 | 0.4992 +/- 0.0014 | 0.9896 +/- 0.0000 |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0500 | 1:1 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0500 | 1:1 | random | 0.5291 +/- 0.0082 | 0.6638 +/- 0.0009 | 0.5011 +/- 0.0021 | 0.9852 +/- 0.0000 |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0500 | 1:1 | real | 0.5466 +/- 0.0083 | 0.6552 +/- 0.0013 | 0.4994 +/- 0.0029 | 0.9512 +/- 0.0000 |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0500 | 1:1 | shuffled | 0.5301 +/- 0.0083 | 0.6619 +/- 0.0007 | 0.4975 +/- 0.0016 | 0.9837 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0500 | 1:1 | ar | 0.5293 +/- 0.0081 | 0.6640 +/- 0.0006 | 0.4992 +/- 0.0014 | 0.9896 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0500 | 1:1 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0500 | 1:1 | random | 0.5420 +/- 0.0081 | 0.6490 +/- 0.0033 | 0.5192 +/- 0.0069 | 0.8891 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0500 | 1:1 | real | 0.5949 +/- 0.0087 | 0.6468 +/- 0.0033 | 0.5509 +/- 0.0064 | 0.8225 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0500 | 1:1 | shuffled | 0.5280 +/- 0.0085 | 0.6531 +/- 0.0020 | 0.5011 +/- 0.0045 | 0.9393 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0500 | 1:1 | ar | 0.5293 +/- 0.0081 | 0.6640 +/- 0.0006 | 0.4992 +/- 0.0014 | 0.9896 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0500 | 1:1 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0500 | 1:1 | random | 0.5270 +/- 0.0083 | 0.6592 +/- 0.0011 | 0.4984 +/- 0.0025 | 0.9704 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0500 | 1:1 | real | 0.5817 +/- 0.0086 | 0.6638 +/- 0.0031 | 0.5288 +/- 0.0066 | 0.9305 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0500 | 1:1 | shuffled | 0.5267 +/- 0.0079 | 0.6613 +/- 0.0010 | 0.4984 +/- 0.0022 | 0.9793 +/- 0.0000 |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0500 | 1:2 | ar | 0.3644 +/- 0.0043 | 0.4985 +/- 0.0005 | 0.4997 +/- 0.0009 | 0.9896 +/- 0.0000 |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0500 | 1:2 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0500 | 1:2 | random | 0.3662 +/- 0.0043 | 0.4978 +/- 0.0007 | 0.5006 +/- 0.0015 | 0.9808 +/- 0.0000 |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0500 | 1:2 | real | 0.3702 +/- 0.0034 | 0.4965 +/- 0.0006 | 0.4986 +/- 0.0012 | 0.9778 +/- 0.0000 |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0500 | 1:2 | shuffled | 0.3643 +/- 0.0043 | 0.4995 +/- 0.0007 | 0.5024 +/- 0.0013 | 0.9867 +/- 0.0000 |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0500 | 1:2 | ar | 0.3644 +/- 0.0043 | 0.4985 +/- 0.0005 | 0.4997 +/- 0.0009 | 0.9896 +/- 0.0000 |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0500 | 1:2 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0500 | 1:2 | random | 0.3641 +/- 0.0043 | 0.4990 +/- 0.0006 | 0.5017 +/- 0.0013 | 0.9852 +/- 0.0000 |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0500 | 1:2 | real | 0.3830 +/- 0.0035 | 0.4939 +/- 0.0008 | 0.5004 +/- 0.0015 | 0.9512 +/- 0.0000 |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0500 | 1:2 | shuffled | 0.3646 +/- 0.0042 | 0.4968 +/- 0.0004 | 0.4978 +/- 0.0008 | 0.9837 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0500 | 1:2 | ar | 0.3644 +/- 0.0043 | 0.4985 +/- 0.0005 | 0.4997 +/- 0.0009 | 0.9896 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0500 | 1:2 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0500 | 1:2 | random | 0.3782 +/- 0.0038 | 0.4967 +/- 0.0013 | 0.5219 +/- 0.0023 | 0.8891 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0500 | 1:2 | real | 0.4313 +/- 0.0046 | 0.5048 +/- 0.0014 | 0.5521 +/- 0.0022 | 0.8225 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0500 | 1:2 | shuffled | 0.3621 +/- 0.0042 | 0.4930 +/- 0.0014 | 0.5018 +/- 0.0028 | 0.9393 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0500 | 1:2 | ar | 0.3644 +/- 0.0043 | 0.4985 +/- 0.0005 | 0.4997 +/- 0.0009 | 0.9896 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0500 | 1:2 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0500 | 1:2 | random | 0.3623 +/- 0.0039 | 0.4954 +/- 0.0007 | 0.4983 +/- 0.0014 | 0.9704 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0500 | 1:2 | real | 0.4170 +/- 0.0050 | 0.5076 +/- 0.0017 | 0.5313 +/- 0.0030 | 0.9305 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0500 | 1:2 | shuffled | 0.3611 +/- 0.0043 | 0.4965 +/- 0.0005 | 0.4982 +/- 0.0011 | 0.9793 +/- 0.0000 |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0500 | 1:3 | ar | 0.2785 +/- 0.0017 | 0.3989 +/- 0.0001 | 0.4995 +/- 0.0003 | 0.9896 +/- 0.0000 |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0500 | 1:3 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0500 | 1:3 | random | 0.2801 +/- 0.0017 | 0.3987 +/- 0.0002 | 0.5006 +/- 0.0005 | 0.9808 +/- 0.0000 |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0500 | 1:3 | real | 0.2850 +/- 0.0015 | 0.3975 +/- 0.0002 | 0.4985 +/- 0.0004 | 0.9778 +/- 0.0000 |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0500 | 1:3 | shuffled | 0.2784 +/- 0.0017 | 0.4000 +/- 0.0002 | 0.5023 +/- 0.0004 | 0.9867 +/- 0.0000 |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0500 | 1:3 | ar | 0.2785 +/- 0.0017 | 0.3989 +/- 0.0001 | 0.4995 +/- 0.0003 | 0.9896 +/- 0.0000 |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0500 | 1:3 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0500 | 1:3 | random | 0.2781 +/- 0.0017 | 0.3996 +/- 0.0001 | 0.5017 +/- 0.0003 | 0.9852 +/- 0.0000 |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0500 | 1:3 | real | 0.2970 +/- 0.0016 | 0.3960 +/- 0.0003 | 0.5001 +/- 0.0007 | 0.9512 +/- 0.0000 |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0500 | 1:3 | shuffled | 0.2784 +/- 0.0017 | 0.3976 +/- 0.0002 | 0.4979 +/- 0.0004 | 0.9837 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0500 | 1:3 | ar | 0.2785 +/- 0.0017 | 0.3989 +/- 0.0001 | 0.4995 +/- 0.0003 | 0.9896 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0500 | 1:3 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0500 | 1:3 | random | 0.2918 +/- 0.0014 | 0.4010 +/- 0.0006 | 0.5204 +/- 0.0012 | 0.8891 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0500 | 1:3 | real | 0.3394 +/- 0.0023 | 0.4129 +/- 0.0008 | 0.5510 +/- 0.0012 | 0.8225 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0500 | 1:3 | shuffled | 0.2762 +/- 0.0018 | 0.3957 +/- 0.0004 | 0.5016 +/- 0.0008 | 0.9393 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0500 | 1:3 | ar | 0.2785 +/- 0.0017 | 0.3989 +/- 0.0001 | 0.4995 +/- 0.0003 | 0.9896 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0500 | 1:3 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0500 | 1:3 | random | 0.2769 +/- 0.0016 | 0.3968 +/- 0.0002 | 0.4984 +/- 0.0004 | 0.9704 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0500 | 1:3 | real | 0.3276 +/- 0.0023 | 0.4098 +/- 0.0005 | 0.5302 +/- 0.0009 | 0.9305 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0500 | 1:3 | shuffled | 0.2757 +/- 0.0019 | 0.3975 +/- 0.0002 | 0.4983 +/- 0.0004 | 0.9793 +/- 0.0000 |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0750 | 1:1 | ar | 0.5352 +/- 0.0124 | 0.6656 +/- 0.0005 | 0.5000 +/- 0.0012 | 0.9953 +/- 0.0000 |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0750 | 1:1 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0750 | 1:1 | random | 0.5366 +/- 0.0120 | 0.6618 +/- 0.0012 | 0.4986 +/- 0.0026 | 0.9813 +/- 0.0000 |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0750 | 1:1 | real | 0.5417 +/- 0.0113 | 0.6611 +/- 0.0023 | 0.5077 +/- 0.0050 | 0.9603 +/- 0.0000 |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0750 | 1:1 | shuffled | 0.5364 +/- 0.0125 | 0.6620 +/- 0.0008 | 0.4977 +/- 0.0018 | 0.9836 +/- 0.0000 |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0750 | 1:1 | ar | 0.5352 +/- 0.0124 | 0.6656 +/- 0.0005 | 0.5000 +/- 0.0012 | 0.9953 +/- 0.0000 |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0750 | 1:1 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0750 | 1:1 | random | 0.5371 +/- 0.0127 | 0.6630 +/- 0.0012 | 0.4987 +/- 0.0026 | 0.9860 +/- 0.0000 |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0750 | 1:1 | real | 0.5572 +/- 0.0114 | 0.6660 +/- 0.0038 | 0.5348 +/- 0.0081 | 0.9276 +/- 0.0000 |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0750 | 1:1 | shuffled | 0.5340 +/- 0.0121 | 0.6646 +/- 0.0014 | 0.5012 +/- 0.0030 | 0.9883 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0750 | 1:1 | ar | 0.5352 +/- 0.0124 | 0.6656 +/- 0.0005 | 0.5000 +/- 0.0012 | 0.9953 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0750 | 1:1 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0750 | 1:1 | random | 0.5406 +/- 0.0120 | 0.6493 +/- 0.0032 | 0.5142 +/- 0.0068 | 0.8995 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0750 | 1:1 | real | 0.5936 +/- 0.0101 | 0.6289 +/- 0.0058 | 0.5367 +/- 0.0115 | 0.7850 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0750 | 1:1 | shuffled | 0.5287 +/- 0.0127 | 0.6573 +/- 0.0023 | 0.5005 +/- 0.0050 | 0.9579 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0750 | 1:1 | ar | 0.5352 +/- 0.0124 | 0.6656 +/- 0.0005 | 0.5000 +/- 0.0012 | 0.9953 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0750 | 1:1 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0750 | 1:1 | random | 0.5361 +/- 0.0125 | 0.6553 +/- 0.0020 | 0.4985 +/- 0.0043 | 0.9533 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0750 | 1:1 | real | 0.5777 +/- 0.0108 | 0.6485 +/- 0.0045 | 0.5288 +/- 0.0093 | 0.8692 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0750 | 1:1 | shuffled | 0.5315 +/- 0.0121 | 0.6575 +/- 0.0019 | 0.4985 +/- 0.0042 | 0.9626 +/- 0.0000 |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0750 | 1:2 | ar | 0.3740 +/- 0.0059 | 0.4996 +/- 0.0005 | 0.5004 +/- 0.0009 | 0.9953 +/- 0.0000 |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0750 | 1:2 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0750 | 1:2 | random | 0.3757 +/- 0.0061 | 0.4975 +/- 0.0011 | 0.4996 +/- 0.0021 | 0.9813 +/- 0.0000 |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0750 | 1:2 | real | 0.3820 +/- 0.0058 | 0.5007 +/- 0.0014 | 0.5112 +/- 0.0027 | 0.9603 +/- 0.0000 |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0750 | 1:2 | shuffled | 0.3751 +/- 0.0062 | 0.4974 +/- 0.0009 | 0.4990 +/- 0.0018 | 0.9836 +/- 0.0000 |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0750 | 1:2 | ar | 0.3740 +/- 0.0059 | 0.4996 +/- 0.0005 | 0.5004 +/- 0.0009 | 0.9953 +/- 0.0000 |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0750 | 1:2 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0750 | 1:2 | random | 0.3764 +/- 0.0059 | 0.4981 +/- 0.0006 | 0.4997 +/- 0.0013 | 0.9860 +/- 0.0000 |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0750 | 1:2 | real | 0.3979 +/- 0.0059 | 0.5100 +/- 0.0026 | 0.5362 +/- 0.0046 | 0.9276 +/- 0.0000 |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0750 | 1:2 | shuffled | 0.3724 +/- 0.0059 | 0.4996 +/- 0.0010 | 0.5021 +/- 0.0019 | 0.9883 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0750 | 1:2 | ar | 0.3740 +/- 0.0059 | 0.4996 +/- 0.0005 | 0.5004 +/- 0.0009 | 0.9953 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0750 | 1:2 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0750 | 1:2 | random | 0.3804 +/- 0.0071 | 0.4950 +/- 0.0024 | 0.5161 +/- 0.0044 | 0.8995 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0750 | 1:2 | real | 0.4328 +/- 0.0082 | 0.4910 +/- 0.0030 | 0.5394 +/- 0.0049 | 0.7850 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0750 | 1:2 | shuffled | 0.3666 +/- 0.0063 | 0.4960 +/- 0.0014 | 0.5028 +/- 0.0027 | 0.9579 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0750 | 1:2 | ar | 0.3740 +/- 0.0059 | 0.4996 +/- 0.0005 | 0.5004 +/- 0.0009 | 0.9953 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0750 | 1:2 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0750 | 1:2 | random | 0.3749 +/- 0.0059 | 0.4939 +/- 0.0016 | 0.5000 +/- 0.0032 | 0.9533 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0750 | 1:2 | real | 0.4183 +/- 0.0079 | 0.5007 +/- 0.0028 | 0.5338 +/- 0.0049 | 0.8692 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0750 | 1:2 | shuffled | 0.3689 +/- 0.0058 | 0.4954 +/- 0.0014 | 0.5004 +/- 0.0028 | 0.9626 +/- 0.0000 |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0750 | 1:3 | ar | 0.2915 +/- 0.0070 | 0.3996 +/- 0.0004 | 0.5000 +/- 0.0008 | 0.9953 +/- 0.0000 |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0750 | 1:3 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0750 | 1:3 | random | 0.2928 +/- 0.0072 | 0.3981 +/- 0.0006 | 0.4992 +/- 0.0012 | 0.9813 +/- 0.0000 |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0750 | 1:3 | real | 0.2990 +/- 0.0064 | 0.4017 +/- 0.0010 | 0.5101 +/- 0.0020 | 0.9603 +/- 0.0000 |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0750 | 1:3 | shuffled | 0.2925 +/- 0.0071 | 0.3979 +/- 0.0006 | 0.4984 +/- 0.0013 | 0.9836 +/- 0.0000 |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0750 | 1:3 | ar | 0.2915 +/- 0.0070 | 0.3996 +/- 0.0004 | 0.5000 +/- 0.0008 | 0.9953 +/- 0.0000 |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0750 | 1:3 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0750 | 1:3 | random | 0.2934 +/- 0.0064 | 0.3987 +/- 0.0006 | 0.4996 +/- 0.0013 | 0.9860 +/- 0.0000 |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0750 | 1:3 | real | 0.3143 +/- 0.0067 | 0.4126 +/- 0.0018 | 0.5357 +/- 0.0033 | 0.9276 +/- 0.0000 |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0750 | 1:3 | shuffled | 0.2900 +/- 0.0068 | 0.3998 +/- 0.0007 | 0.5016 +/- 0.0013 | 0.9883 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0750 | 1:3 | ar | 0.2915 +/- 0.0070 | 0.3996 +/- 0.0004 | 0.5000 +/- 0.0008 | 0.9953 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0750 | 1:3 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0750 | 1:3 | random | 0.2955 +/- 0.0067 | 0.3991 +/- 0.0019 | 0.5151 +/- 0.0037 | 0.8995 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0750 | 1:3 | real | 0.3460 +/- 0.0076 | 0.4035 +/- 0.0020 | 0.5414 +/- 0.0031 | 0.7850 +/- 0.0000 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0750 | 1:3 | shuffled | 0.2830 +/- 0.0062 | 0.3970 +/- 0.0011 | 0.5010 +/- 0.0022 | 0.9579 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0750 | 1:3 | ar | 0.2915 +/- 0.0070 | 0.3996 +/- 0.0004 | 0.5000 +/- 0.0008 | 0.9953 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0750 | 1:3 | majority | NA +/- NA | NA +/- NA | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0750 | 1:3 | random | 0.2920 +/- 0.0070 | 0.3960 +/- 0.0012 | 0.4997 +/- 0.0024 | 0.9533 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0750 | 1:3 | real | 0.3331 +/- 0.0071 | 0.4064 +/- 0.0022 | 0.5331 +/- 0.0039 | 0.8692 +/- 0.0000 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0750 | 1:3 | shuffled | 0.2876 +/- 0.0070 | 0.3968 +/- 0.0011 | 0.4998 +/- 0.0023 | 0.9626 +/- 0.0000 |

## SECTION 6: Controls

Anti-leakage controls were run for blocked primary-focus targets and masks: label shuffle within/across videos, feature shuffle within/across videos, timestamp-only, and video-ID/time-only.

| Feature | Target | Thr | Mask | Control | PR-AUC | F1 | Recall |
|---|---|---:|---|---|---:|---:|---:|
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | all_frames | `feature_shuffle_across_videos` | 0.3045 | 0.2189 | 0.1411 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | all_frames | `feature_shuffle_within_video` | 0.3040 | 0.2180 | 0.1387 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | all_frames | `label_shuffle_across_videos` | 0.1544 | 0.2218 | 0.6423 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | all_frames | `label_shuffle_within_video` | 0.2617 | 0.2739 | 0.8735 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | all_frames | `timestamp_only` | 0.1771 | 0.2510 | 0.6083 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | all_frames | `video_id_time_only` | 0.2297 | 0.2683 | 0.6423 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | all_frames | `feature_shuffle_across_videos` | 0.2988 | 0.2059 | 0.1265 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | all_frames | `feature_shuffle_within_video` | 0.2993 | 0.1888 | 0.1144 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | all_frames | `label_shuffle_across_videos` | 0.1553 | 0.2535 | 1.0000 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | all_frames | `label_shuffle_within_video` | 0.2321 | 0.2561 | 0.9976 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | all_frames | `timestamp_only` | 0.1771 | 0.2510 | 0.6083 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | all_frames | `video_id_time_only` | 0.2297 | 0.2683 | 0.6423 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | all_frames | `feature_shuffle_across_videos` | 0.2675 | 0.1700 | 0.1022 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | all_frames | `feature_shuffle_within_video` | 0.2795 | 0.2347 | 0.1727 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | all_frames | `label_shuffle_across_videos` | 0.1420 | 0.1703 | 0.2360 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | all_frames | `label_shuffle_within_video` | 0.2160 | 0.2594 | 0.9781 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | all_frames | `timestamp_only` | 0.1771 | 0.2510 | 0.6083 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | all_frames | `video_id_time_only` | 0.2297 | 0.2683 | 0.6423 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | all_frames | `feature_shuffle_across_videos` | 0.2884 | 0.1949 | 0.1217 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | all_frames | `feature_shuffle_within_video` | 0.3002 | 0.2062 | 0.1290 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | all_frames | `label_shuffle_across_videos` | 0.1286 | 0.0174 | 0.0097 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | all_frames | `label_shuffle_within_video` | 0.2313 | 0.2562 | 0.9927 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | all_frames | `timestamp_only` | 0.1771 | 0.2510 | 0.6083 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | all_frames | `video_id_time_only` | 0.2297 | 0.2683 | 0.6423 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | event_plus_pre_3s | `feature_shuffle_across_videos` | NA | 0.1672 | 0.0912 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | event_plus_pre_3s | `feature_shuffle_within_video` | NA | 0.1633 | 0.0889 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | event_plus_pre_3s | `label_shuffle_across_videos` | NA | 0.8239 | 0.7006 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | event_plus_pre_3s | `label_shuffle_within_video` | NA | 0.9136 | 0.8409 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | event_plus_pre_3s | `timestamp_only` | NA | 0.7724 | 0.6292 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | event_plus_pre_3s | `video_id_time_only` | NA | 0.7881 | 0.6503 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | event_plus_pre_3s | `feature_shuffle_across_videos` | NA | 0.1514 | 0.0819 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | event_plus_pre_3s | `feature_shuffle_within_video` | NA | 0.1311 | 0.0702 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | event_plus_pre_3s | `label_shuffle_across_videos` | NA | 1.0000 | 1.0000 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | event_plus_pre_3s | `label_shuffle_within_video` | NA | 0.9965 | 0.9930 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | event_plus_pre_3s | `timestamp_only` | NA | 0.7724 | 0.6292 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | event_plus_pre_3s | `video_id_time_only` | NA | 0.7881 | 0.6503 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | event_plus_pre_3s | `feature_shuffle_across_videos` | NA | 0.1291 | 0.0690 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | event_plus_pre_3s | `feature_shuffle_within_video` | NA | 0.2169 | 0.1216 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | event_plus_pre_3s | `label_shuffle_across_videos` | NA | 0.3807 | 0.2351 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | event_plus_pre_3s | `label_shuffle_within_video` | NA | 0.9803 | 0.9614 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | event_plus_pre_3s | `timestamp_only` | NA | 0.7724 | 0.6292 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | event_plus_pre_3s | `video_id_time_only` | NA | 0.7881 | 0.6503 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | event_plus_pre_3s | `feature_shuffle_across_videos` | NA | 0.1553 | 0.0842 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | event_plus_pre_3s | `feature_shuffle_within_video` | NA | 0.1533 | 0.0830 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | event_plus_pre_3s | `label_shuffle_across_videos` | NA | 0.0254 | 0.0129 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | event_plus_pre_3s | `label_shuffle_within_video` | NA | 0.9941 | 0.9883 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | event_plus_pre_3s | `timestamp_only` | NA | 0.7724 | 0.6292 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | event_plus_pre_3s | `video_id_time_only` | NA | 0.7881 | 0.6503 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_1s | `feature_shuffle_across_videos` | NA | 0.0674 | 0.0349 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_1s | `feature_shuffle_within_video` | NA | 0.0889 | 0.0465 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_1s | `label_shuffle_across_videos` | NA | 0.8259 | 0.7035 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_1s | `label_shuffle_within_video` | NA | 0.8974 | 0.8140 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_1s | `timestamp_only` | NA | 0.7758 | 0.6337 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_1s | `video_id_time_only` | NA | 0.7930 | 0.6570 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_1s | `feature_shuffle_across_videos` | NA | 0.0565 | 0.0291 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_1s | `feature_shuffle_within_video` | NA | 0.0565 | 0.0291 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_1s | `label_shuffle_across_videos` | NA | 1.0000 | 1.0000 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_1s | `label_shuffle_within_video` | NA | 0.9942 | 0.9884 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_1s | `timestamp_only` | NA | 0.7758 | 0.6337 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_1s | `video_id_time_only` | NA | 0.7930 | 0.6570 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_1s | `feature_shuffle_across_videos` | NA | 0.0782 | 0.0407 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_1s | `feature_shuffle_within_video` | NA | 0.1799 | 0.0988 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_1s | `label_shuffle_across_videos` | NA | 0.4000 | 0.2500 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_1s | `label_shuffle_within_video` | NA | 0.9822 | 0.9651 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_1s | `timestamp_only` | NA | 0.7758 | 0.6337 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_1s | `video_id_time_only` | NA | 0.7930 | 0.6570 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_1s | `feature_shuffle_across_videos` | NA | 0.0782 | 0.0407 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_1s | `feature_shuffle_within_video` | NA | 0.0674 | 0.0349 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_1s | `label_shuffle_across_videos` | NA | 0.0343 | 0.0174 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_1s | `label_shuffle_within_video` | NA | 0.9942 | 0.9884 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_1s | `timestamp_only` | NA | 0.7758 | 0.6337 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_1s | `video_id_time_only` | NA | 0.7930 | 0.6570 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_2s | `feature_shuffle_across_videos` | NA | 0.0934 | 0.0490 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_2s | `feature_shuffle_within_video` | NA | 0.0784 | 0.0408 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_2s | `label_shuffle_across_videos` | NA | 0.8578 | 0.7510 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_2s | `label_shuffle_within_video` | NA | 0.9013 | 0.8204 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_2s | `timestamp_only` | NA | 0.7781 | 0.6367 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_2s | `video_id_time_only` | NA | 0.7961 | 0.6612 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_2s | `feature_shuffle_across_videos` | NA | 0.0859 | 0.0449 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_2s | `feature_shuffle_within_video` | NA | 0.0784 | 0.0408 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_2s | `label_shuffle_across_videos` | NA | 1.0000 | 1.0000 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_2s | `label_shuffle_within_video` | NA | 0.9938 | 0.9878 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_2s | `timestamp_only` | NA | 0.7781 | 0.6367 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_2s | `video_id_time_only` | NA | 0.7961 | 0.6612 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_2s | `feature_shuffle_across_videos` | NA | 0.0934 | 0.0490 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_2s | `feature_shuffle_within_video` | NA | 0.1648 | 0.0898 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_2s | `label_shuffle_across_videos` | NA | 0.3612 | 0.2204 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_2s | `label_shuffle_within_video` | NA | 0.9770 | 0.9551 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_2s | `timestamp_only` | NA | 0.7781 | 0.6367 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_2s | `video_id_time_only` | NA | 0.7961 | 0.6612 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_2s | `feature_shuffle_across_videos` | NA | 0.0934 | 0.0490 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_2s | `feature_shuffle_within_video` | NA | 0.1008 | 0.0531 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_2s | `label_shuffle_across_videos` | NA | 0.0242 | 0.0122 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_2s | `label_shuffle_within_video` | NA | 0.9938 | 0.9878 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_2s | `timestamp_only` | NA | 0.7781 | 0.6367 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_2s | `video_id_time_only` | NA | 0.7961 | 0.6612 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_3s | `feature_shuffle_across_videos` | NA | 0.1164 | 0.0618 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_3s | `feature_shuffle_within_video` | NA | 0.0956 | 0.0502 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_3s | `label_shuffle_across_videos` | NA | 0.8884 | 0.7992 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_3s | `label_shuffle_within_video` | NA | 0.9025 | 0.8224 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_3s | `timestamp_only` | NA | 0.7897 | 0.6525 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_3s | `video_id_time_only` | NA | 0.7981 | 0.6641 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_3s | `feature_shuffle_across_videos` | NA | 0.1026 | 0.0541 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_3s | `feature_shuffle_within_video` | NA | 0.0672 | 0.0347 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_3s | `label_shuffle_across_videos` | NA | 1.0000 | 1.0000 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_3s | `label_shuffle_within_video` | NA | 0.9942 | 0.9884 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_3s | `timestamp_only` | NA | 0.7897 | 0.6525 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_3s | `video_id_time_only` | NA | 0.7981 | 0.6641 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_3s | `feature_shuffle_across_videos` | NA | 0.0815 | 0.0425 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_3s | `feature_shuffle_within_video` | NA | 0.1300 | 0.0695 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_3s | `label_shuffle_across_videos` | NA | 0.3711 | 0.2278 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_3s | `label_shuffle_within_video` | NA | 0.9743 | 0.9498 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_3s | `timestamp_only` | NA | 0.7897 | 0.6525 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_3s | `video_id_time_only` | NA | 0.7981 | 0.6641 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_3s | `feature_shuffle_across_videos` | NA | 0.1232 | 0.0656 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_3s | `feature_shuffle_within_video` | NA | 0.1026 | 0.0541 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_3s | `label_shuffle_across_videos` | NA | 0.0304 | 0.0154 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_3s | `label_shuffle_within_video` | NA | 0.9903 | 0.9807 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_3s | `timestamp_only` | NA | 0.7897 | 0.6525 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_3s | `video_id_time_only` | NA | 0.7981 | 0.6641 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_5s | `feature_shuffle_across_videos` | NA | 0.1115 | 0.0591 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_5s | `feature_shuffle_within_video` | NA | 0.0974 | 0.0512 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_5s | `label_shuffle_across_videos` | NA | 0.9122 | 0.8386 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_5s | `label_shuffle_within_video` | NA | 0.8736 | 0.7756 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_5s | `timestamp_only` | NA | 0.7905 | 0.6535 |
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_5s | `video_id_time_only` | NA | 0.7991 | 0.6654 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_5s | `feature_shuffle_across_videos` | NA | 0.0974 | 0.0512 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_5s | `feature_shuffle_within_video` | NA | 0.0684 | 0.0354 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_5s | `label_shuffle_across_videos` | NA | 1.0000 | 1.0000 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_5s | `label_shuffle_within_video` | NA | 0.9921 | 0.9843 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_5s | `timestamp_only` | NA | 0.7905 | 0.6535 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_5s | `video_id_time_only` | NA | 0.7991 | 0.6654 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_5s | `feature_shuffle_across_videos` | NA | 0.0830 | 0.0433 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_5s | `feature_shuffle_within_video` | NA | 0.1661 | 0.0906 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_5s | `label_shuffle_across_videos` | NA | 0.3822 | 0.2362 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_5s | `label_shuffle_within_video` | NA | 0.9779 | 0.9567 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_5s | `timestamp_only` | NA | 0.7905 | 0.6535 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_5s | `video_id_time_only` | NA | 0.7991 | 0.6654 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_5s | `feature_shuffle_across_videos` | NA | 0.1045 | 0.0551 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_5s | `feature_shuffle_within_video` | NA | 0.1045 | 0.0551 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_5s | `label_shuffle_across_videos` | NA | 0.0386 | 0.0197 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_5s | `label_shuffle_within_video` | NA | 0.9921 | 0.9843 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_5s | `timestamp_only` | NA | 0.7905 | 0.6535 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | pre_event_5s | `video_id_time_only` | NA | 0.7991 | 0.6654 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0500 | all_frames | `feature_shuffle_across_videos` | 0.3382 | 0.0918 | 0.0494 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0500 | all_frames | `feature_shuffle_within_video` | 0.3438 | 0.1221 | 0.0674 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0500 | all_frames | `label_shuffle_across_videos` | 0.2823 | 0.3957 | 1.0000 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0500 | all_frames | `label_shuffle_within_video` | 0.3607 | 0.3622 | 0.3817 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0500 | all_frames | `timestamp_only` | 0.2741 | 0.3949 | 0.9955 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0500 | all_frames | `video_id_time_only` | 0.3405 | 0.4282 | 0.8129 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0500 | all_frames | `feature_shuffle_across_videos` | 0.3394 | 0.1148 | 0.0629 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0500 | all_frames | `feature_shuffle_within_video` | 0.3397 | 0.1180 | 0.0659 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0500 | all_frames | `label_shuffle_across_videos` | 0.2497 | 0.0579 | 0.0329 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0500 | all_frames | `label_shuffle_within_video` | 0.3199 | 0.2636 | 0.2066 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0500 | all_frames | `timestamp_only` | 0.2741 | 0.3949 | 0.9955 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0500 | all_frames | `video_id_time_only` | 0.3405 | 0.4282 | 0.8129 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0500 | all_frames | `feature_shuffle_across_videos` | 0.3200 | 0.1102 | 0.0614 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0500 | all_frames | `feature_shuffle_within_video` | 0.3047 | 0.0796 | 0.0434 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0500 | all_frames | `label_shuffle_across_videos` | 0.2735 | 0.3702 | 0.7141 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0500 | all_frames | `label_shuffle_within_video` | 0.2906 | 0.1688 | 0.1093 |
| ... | ... | ... | ... | ... | 560 additional control rows in CSV | ... | ... |

## SECTION 7: Per-Video Robustness

Enough-event video rows: 1356. Wins versus AR, shuffled, and random by per-video PR-AUC: 483/1356.

| Feature | Target | Thr | Enough videos | Triple-control wins |
|---|---|---:|---:|---:|
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | 72 | 29 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0500 | 79 | 29 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0750 | 67 | 21 |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0500 | 66 | 21 |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0750 | 55 | 10 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | 72 | 29 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0500 | 79 | 27 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0750 | 67 | 25 |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0500 | 66 | 19 |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0750 | 55 | 14 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | 72 | 16 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0500 | 79 | 32 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0750 | 67 | 22 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0500 | 66 | 32 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0750 | 55 | 19 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | 72 | 31 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0500 | 79 | 31 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0750 | 67 | 27 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0500 | 66 | 29 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0750 | 55 | 20 |

## SECTION 8: Final Recommendation

For the full 124 benchmark: keep current full-frame metrics, add event-conditioned and balanced event-vs-stable metrics, prioritize blocked pre-event spike detection, and do not scale timing claims until the alignment/shift issue is resolved.

Claim to carry forward only if replicated on 124: frame-wide continuous MAE underestimates cortical signal because most frames are stable; conditioned on upcoming arousal-change regions, cortical/TRIBE features improve early detection of emotionally meaningful events.

## Output Files

- JSON: `/Users/maxsartini/MiroFish-Offline-main/benchmarks/veatic/veatic_124_event_conditioned_retest_20260616.json`
- Event masks CSV: `/Users/maxsartini/MiroFish-Offline-main/benchmarks/veatic/veatic_124_event_conditioned_retest_20260616.event_masks.csv`
- Balanced sampling CSV: `/Users/maxsartini/MiroFish-Offline-main/benchmarks/veatic/veatic_124_event_conditioned_retest_20260616.balanced_sampling.csv`
- Per-video CSV: `/Users/maxsartini/MiroFish-Offline-main/benchmarks/veatic/veatic_124_event_conditioned_retest_20260616.per_video.csv`
