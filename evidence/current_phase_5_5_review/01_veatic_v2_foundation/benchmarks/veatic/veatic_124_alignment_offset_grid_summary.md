# VEATIC 124 Alignment Offset Grid Summary

Reliable blocked rows: 20
Median best offset: -1.5
Mean best offset: -0.775
Mode best offset: -1.5

| Feature | Split | Target | Thr | Best offset | Real PR-AUC | vs AR | vs Shuf | vs Rand |
|---|---|---|---:|---:|---:|---:|---:|---:|
| cortical_fast_default | blocked | `arousal__future_change_p2s_movement` | 0.0500 | 2.5000 | 0.3300 | 0.0053 | 0.0093 | 0.0067 |
| cortical_fast_default | blocked | `arousal__future_change_p3s_movement` | 0.0500 | 8.0000 | 0.3888 | -0.0043 | -0.0029 | -0.0048 |
| cortical_fast_default | blocked | `arousal__future_change_p3s_movement` | 0.0750 | 1.0000 | 0.2923 | -0.0050 | -0.0004 | -0.0036 |
| cortical_fast_default | blocked | `arousal__future_spike_1_3s` | 0.0500 | -5.0000 | 0.3375 | 0.0029 | 0.0097 | 0.0038 |
| cortical_fast_default | blocked | `arousal__future_spike_1_3s` | 0.0750 | -7.0000 | 0.1984 | 0.0058 | 0.0103 | 0.0038 |
| cortical_global_delta | blocked | `arousal__future_change_p2s_movement` | 0.0500 | 2.5000 | 0.3155 | -0.0091 | -0.0043 | -0.0037 |
| cortical_global_delta | blocked | `arousal__future_change_p3s_movement` | 0.0500 | 8.0000 | 0.3776 | -0.0156 | -0.0013 | -0.0022 |
| cortical_global_delta | blocked | `arousal__future_change_p3s_movement` | 0.0750 | 0.0000 | 0.3008 | 0.0279 | 0.0249 | 0.0258 |
| cortical_global_delta | blocked | `arousal__future_spike_1_3s` | 0.0500 | -1.5000 | 0.3630 | 0.0144 | 0.0179 | 0.0232 |
| cortical_global_delta | blocked | `arousal__future_spike_1_3s` | 0.0750 | -1.5000 | 0.2412 | 0.0392 | 0.0412 | 0.0377 |
| cortical_pca64_delta | blocked | `arousal__future_change_p2s_movement` | 0.0500 | 0.0000 | 0.3007 | -0.0042 | 0.0467 | -0.0008 |
| cortical_pca64_delta | blocked | `arousal__future_change_p3s_movement` | 0.0500 | -6.0000 | 0.3822 | 0.0612 | 0.0621 | 0.0780 |
| cortical_pca64_delta | blocked | `arousal__future_change_p3s_movement` | 0.0750 | -8.0000 | 0.2871 | -0.0002 | 0.0226 | 0.0371 |
| cortical_pca64_delta | blocked | `arousal__future_spike_1_3s` | 0.0500 | -2.0000 | 0.3593 | 0.0108 | 0.0553 | 0.0449 |
| cortical_pca64_delta | blocked | `arousal__future_spike_1_3s` | 0.0750 | -1.0000 | 0.2271 | 0.0597 | 0.0519 | 0.0751 |
| cortical_pca_64 | blocked | `arousal__future_change_p2s_movement` | 0.0500 | 2.0000 | 0.3239 | 0.0197 | 0.0246 | 0.0381 |
| cortical_pca_64 | blocked | `arousal__future_change_p3s_movement` | 0.0500 | -2.5000 | 0.4131 | 0.0581 | 0.0624 | 0.0591 |
| cortical_pca_64 | blocked | `arousal__future_change_p3s_movement` | 0.0750 | -2.0000 | 0.3068 | 0.0504 | 0.0649 | 0.0570 |
| cortical_pca_64 | blocked | `arousal__future_spike_1_3s` | 0.0500 | -1.5000 | 0.3426 | -0.0060 | -0.0038 | 0.0128 |
| cortical_pca_64 | blocked | `arousal__future_spike_1_3s` | 0.0750 | -1.5000 | 0.2195 | 0.0175 | 0.0149 | 0.0204 |