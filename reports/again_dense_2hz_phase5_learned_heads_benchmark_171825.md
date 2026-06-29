# AGAIN Dense 2Hz Phase 5 Learned Heads Benchmark

This is a learned-head benchmark over Phase 4 fold-safe PCA bridge features. It did not rerun V-JEPA, TRIBE, PCA fitting, bridge benchmarking, or dense video decoding.

## Top Summary Rows

| target | protocol | feature | head | loss | control | PR-AUC | ROC-AUC |
|---|---|---|---|---|---:|---:|---:|
| `arousal_spike_rows_2_6_train_q90` | `blocked_temporal_70_30` | `temporal_mean_2s_then_pca256` | `gated_ar_pca_mlp` | `regression_plus_binary` | `ar_only_head` | 0.26100 | 0.73602 |
| `arousal_spike_rows_2_6_train_q90` | `blocked_temporal_70_30` | `temporal_mean_2s_then_pca256` | `gated_ar_pca_mlp` | `regression` | `ar_only_head` | 0.25881 | 0.73128 |
| `arousal_spike_rows_2_6_train_q90` | `blocked_temporal_70_30` | `temporal_mean_2s_then_pca256` | `gated_ar_pca_mlp` | `binary` | `ar_only_head` | 0.25874 | 0.73399 |
| `arousal_spike_rows_2_6_train_q90` | `blocked_temporal_70_30` | `temporal_mean_2s_then_pca256` | `gated_ar_pca_mlp` | `regression_plus_binary` | `ar_plus_random_pca` | 0.22887 | 0.70028 |
| `arousal_spike_rows_2_6_train_q90` | `grouped_video` | `temporal_mean_2s_then_pca256` | `gated_ar_pca_mlp` | `regression_plus_binary` | `real_ar_pca_diag` | 0.22828 | 0.70768 |
| `arousal_spike_rows_2_6_train_q90` | `grouped_video` | `temporal_mean_2s_then_pca256` | `gated_ar_pca_mlp` | `binary` | `real_ar_pca_diag` | 0.22713 | 0.70701 |
| `arousal_spike_rows_2_6_train_q90` | `blocked_temporal_70_30` | `temporal_mean_2s_then_pca256` | `gated_ar_pca_mlp` | `binary` | `ar_plus_random_pca` | 0.22489 | 0.69752 |
| `arousal_spike_rows_2_6_train_q90` | `blocked_temporal_70_30` | `temporal_mean_2s_then_pca256` | `gated_ar_pca_mlp` | `regression_plus_binary` | `ar_plus_shuffled_pca` | 0.22262 | 0.68208 |
| `arousal_spike_rows_2_6_train_q90` | `grouped_video` | `temporal_mean_2s_then_pca256` | `gated_ar_pca_mlp` | `regression_plus_binary` | `ar_only_head` | 0.22155 | 0.70536 |
| `arousal_spike_rows_2_6_train_q90` | `grouped_video` | `temporal_mean_2s_then_pca256` | `gated_ar_pca_mlp` | `binary` | `ar_only_head` | 0.22136 | 0.70708 |
| `arousal_spike_rows_2_6_train_q90` | `blocked_temporal_70_30` | `temporal_mean_2s_then_pca256` | `gated_ar_pca_mlp` | `regression_plus_binary` | `real_ar_pca_diag` | 0.22000 | 0.68134 |
| `arousal_spike_rows_2_6_train_q90` | `blocked_temporal_70_30` | `temporal_mean_2s_then_pca256` | `gated_ar_pca_mlp` | `binary` | `ar_plus_shuffled_pca` | 0.21738 | 0.67465 |

## Controls

Control lanes include shuffled/random PCA, shuffled temporal diagnostics, time-only, quality/motion/luma-only, and label-permutation sanity rows when enabled.
