# AGAIN Dense 2Hz Phase 5 Learned Heads Benchmark

This is a learned-head benchmark over Phase 4 fold-safe PCA bridge features. It did not rerun V-JEPA, TRIBE, PCA fitting, bridge benchmarking, or dense video decoding.

## Top Summary Rows

| target | protocol | feature | head | loss | control | PR-AUC | ROC-AUC |
|---|---|---|---|---|---:|---:|---:|
| `arousal_spike_rows_2_6_train_q90` | `grouped_video` | `temporal_mean_2s_then_pca256` | `gated_ar_pca_mlp` | `regression_plus_binary` | `real` | 0.21913 | 0.69884 |
| `arousal_spike_rows_2_6_train_q90` | `blocked_temporal_70_30` | `temporal_mean_2s_then_pca256` | `gated_ar_pca_mlp` | `binary` | `random_pca` | 0.21912 | 0.69280 |
| `arousal_spike_rows_2_6_train_q90` | `blocked_temporal_70_30` | `temporal_mean_2s_then_pca256` | `gated_ar_pca_mlp` | `binary` | `ar_plus_random_pca` | 0.21606 | 0.69150 |
| `arousal_spike_rows_2_6_train_q90` | `grouped_video` | `temporal_mean_2s_then_pca256` | `gated_ar_pca_mlp` | `binary` | `real` | 0.21581 | 0.69630 |
| `arousal_spike_rows_2_6_train_q90` | `blocked_temporal_70_30` | `temporal_mean_2s_then_pca256` | `gated_ar_pca_mlp` | `binary` | `ar_plus_shuffled_pca` | 0.21320 | 0.65810 |
| `arousal_spike_rows_2_6_train_q90` | `blocked_temporal_70_30` | `temporal_mean_2s_then_pca256` | `gated_ar_pca_mlp` | `binary` | `shuffled_pca` | 0.20847 | 0.65869 |
| `arousal_spike_rows_2_6_train_q90` | `blocked_temporal_70_30` | `temporal_mean_2s_then_pca256` | `gated_ar_pca_mlp` | `regression_plus_binary` | `real` | 0.20551 | 0.65120 |
| `arousal_spike_rows_2_6_train_q90` | `grouped_video` | `temporal_mean_2s_then_pca256` | `gated_ar_pca_mlp` | `regression` | `real` | 0.19934 | 0.65560 |
| `arousal_spike_rows_2_6_train_q90` | `blocked_temporal_70_30` | `temporal_mean_2s_then_pca256` | `gated_ar_pca_mlp` | `regression` | `real` | 0.19882 | 0.63760 |
| `arousal_spike_rows_2_6_train_q90` | `blocked_temporal_70_30` | `temporal_mean_2s_then_pca256` | `gated_ar_pca_mlp` | `binary` | `real` | 0.19822 | 0.63856 |
| `arousal_spike_rows_2_6_train_q90` | `grouped_video` | `temporal_mean_2s_then_pca256` | `gated_ar_pca_mlp` | `binary` | `ar_plus_shuffled_pca` | 0.19404 | 0.66220 |
| `arousal_spike_rows_2_6_train_q90` | `grouped_video` | `temporal_mean_2s_then_pca256` | `gated_ar_pca_mlp` | `binary` | `shuffled_pca` | 0.19278 | 0.66223 |

## Controls

Control lanes include shuffled/random PCA, shuffled temporal diagnostics, time-only, quality/motion/luma-only, and label-permutation sanity rows when enabled.

The main 540-row run included shuffled/random/time/quality controls. A compact follow-up label-permutation sanity pass used the winning `gated_ar_pca_mlp` head and collapsed as expected: grouped-video label-permutation PR-AUC `0.10428`, ROC-AUC `0.51303`. The corresponding real gated binary lane in that sanity pass reached PR-AUC `0.22458`, so the permutation check supports the absence of obvious label leakage.
