# AGAIN Sparse TRIBE Teacher 500 Queue

- benchmark_mode: `again_sparse_vitg_tribe_teacher_500_pca128_causal_past2s`
- max actual encoded windows: `500`
- queued rows: `522`
- unique actual windows: `480`
- causal roles: `[-2.0, -1.0, 0.0]`
- future rows included: `false`
- output root: `outputs/again_sparse_tribe_teacher_500_20260622_pca_width_reanalysis_v2`
- external cache root: `${NEURAL_BRIDGE_EXTERNAL_ROOT}/benchmarks/again/sparse_tribe_teacher_500_20260621_234312`

## Arm Counts
- coverage_matched_random_to_hybrid: 100 unique actual windows; 102 queued causal rows; 34 candidate centers
- hybrid_top5_selected: 250 unique actual windows; 270 queued causal rows; 90 candidate centers
- low_salience_background: 60 unique actual windows; 60 queued causal rows; 20 candidate centers
- oracle_upper_bound: 60 unique actual windows; 60 queued causal rows; 20 candidate centers
- sparse_anchor_windows: 30 unique actual windows; 30 queued causal rows; 10 candidate centers
