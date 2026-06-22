# AGAIN Sparse TRIBE Teacher 2000 Queue

- benchmark_mode: `again_sparse_vitg_tribe_teacher_small_pca_confirmatory`
- max actual encoded windows: `2000`
- queued rows: `2199`
- unique actual windows: `1948`
- selector video count: `50`
- subset decision: existing 100-video scout/selector cache was not available; reused the corrected 50-video selector subset and expanded sparse coverage within it.
- causal roles: `[-2.0, -1.0, 0.0]`
- selector config hash: `77c8bf3fa2d9724f146fbd2c87baf724a820ead3767f8714439625f3703e4072`
- future rows included: `false`
- output root: `outputs/again_sparse_tribe_teacher_2000_20260622_2000_small_pca_confirmatory_v3`
- external cache root: `${NEURAL_BRIDGE_EXTERNAL_ROOT}/benchmarks/again/sparse_tribe_teacher_500_20260621_234312`

## Arm Counts
- coverage_matched_random_to_hybrid: 400 unique actual windows; 408 queued causal rows; 136 candidate centers
- fixed_random_same_budget: 240 unique actual windows; 246 queued causal rows; 82 candidate centers
- hybrid_top5_selected: 849 unique actual windows; 1032 queued causal rows; 344 candidate centers
- low_salience_background: 160 unique actual windows; 168 queued causal rows; 56 candidate centers
- oracle_upper_bound: 200 unique actual windows; 246 queued causal rows; 82 candidate centers
- sparse_anchor_windows: 99 unique actual windows; 99 queued causal rows; 33 candidate centers
