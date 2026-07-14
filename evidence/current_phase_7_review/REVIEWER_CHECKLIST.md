# Phase 7 Reviewer Checklist

- [ ] Confirm `rows_actual == rows_expected == 420`.
- [ ] Confirm `grouped_continuous_ranking_lift_pass == true` and `failed_gates == []`.
- [ ] Confirm all `15/15` fold-groups are positive versus AR and best controls for Spearman and top-5% lift.
- [ ] Confirm all five fold means are positive.
- [ ] Confirm the target is `residual_future_max_delta_rows_4_10` and the head is `short_temporal_conv_residual`.
- [ ] Confirm grouped PCA, causal-context, frozen-AR identity, checkpoint-restoration, exact-scope, and MLX audits passed.
- [ ] Keep the separate blocked `4/5` verdict distinct from the grouped pass.
- [ ] Do not reinterpret descriptive MAE/RMSE as a preregistered exact-value claim.
- [ ] Keep the observed-AR benchmark distinct from future label-free client deployment.
- [ ] Read the historical Phase 5.5 dossier only after the current Phase 7 result, so history is not mistaken for the current ceiling.
