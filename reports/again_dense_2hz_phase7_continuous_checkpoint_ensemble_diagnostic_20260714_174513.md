# AGAIN Phase 7 blocked continuous checkpoint-ensemble diagnostic

- Output: `outputs/again_dense_2hz_phase7_continuous_checkpoint_ensemble_diagnostic_20260714_174513`
- Target/head: `residual_future_max_delta_rows_4_10` / `short_temporal_conv_residual`
- Matrix: `84/84` rows
- Real / AR / best-control Spearman: `0.1226309621` / `0.1180749764` / `0.1167953465`
- Real minus AR / best-control Spearman: `+0.0045559858` / `+0.0058356156`
- Real / AR / best-control top-5% lift: `0.0828780504` / `0.0767870865` / `0.0768766425`
- Real minus AR / best-control top-5% lift: `+0.0060909639` / `+0.0060014079`
- Ranking/lift diagnostic pass: `True`
- Exact-value candidate pass: `False`
- Failed ranking gates: `[]`
- Failed exact-value gates: `['mae_mean_improvement_vs_ar_at_least_0_0005', 'mae_mean_improvement_vs_best_control_at_least_0_0005', 'rmse_mean_improvement_vs_ar_at_least_0_0005', 'rmse_mean_improvement_vs_best_control_at_least_0_0005', 'mae_positive_vs_ar_and_best_control_all_groups', 'rmse_positive_vs_ar_and_best_control_all_groups', 'absolute_bias_no_worse_than_ar']`

This bounded diagnostic cannot itself promote blocked continuous generalization or exact-value forecasting. A fresh preregistered confirmation is required for any promotion.
