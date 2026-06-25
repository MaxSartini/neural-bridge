# AGAIN Sparse TRIBE Teacher 2000 Queue

- benchmark_mode: `again_sparse_vitg_tribe_teacher_small_pca_confirmatory`
- max actual encoded windows: `2000`
- queued rows: `2001`
- unique actual windows: `1698`
- selector video count: `50`
- subset decision: existing 100-video scout/selector cache was not available; reused the corrected 50-video selector subset and expanded sparse coverage within it.
- causal roles: `[-2.0, -1.0, 0.0]`
- selector config hash: `ef1a2154798d48e3b053d372c0ab27e5d74442c9d45357a815b86f41d69be047`
- future rows included: `false`
- output root: `outputs/again_sparse_tribe_teacher_2000_true_fixed_random_same_budget_20260622_2000_true_fixed_random_same_budget`
- external cache root: `${NEURAL_BRIDGE_EXTERNAL_ROOT}/benchmarks/again/sparse_tribe_teacher_500_20260621_234312`

## Arm Counts
- fixed_random_same_budget: 849 unique actual windows; 969 queued causal rows; 323 candidate centers
- hybrid_top5_selected: 849 unique actual windows; 1032 queued causal rows; 344 candidate centers
