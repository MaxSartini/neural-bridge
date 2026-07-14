# Current Reviewer Checklist

- [ ] Confirm the locked deployment result completed `140/140`, used no observed arousal at inference, won `5/5` panels on all three endpoints, and failed no Tier 1 gates.
- [ ] Confirm `rows_actual == rows_expected == 420`.
- [ ] Confirm `grouped_continuous_ranking_lift_pass == true` and `failed_gates == []`.
- [ ] Confirm all `15/15` fold-groups are positive versus AR and best controls for Spearman and top-5% lift.
- [ ] Confirm all five fold means are positive.
- [ ] Confirm the target is `residual_future_max_delta_rows_4_10` and the head is `short_temporal_conv_residual`.
- [ ] Confirm grouped PCA, causal-context, frozen-AR identity, checkpoint-restoration, exact-scope, and MLX audits passed.
- [ ] Do not reinterpret descriptive MAE/RMSE as a preregistered exact-value claim.
- [ ] Keep the observed-AR Phase 7 ceiling distinct from the newer cached-feature zero-label-at-inference result and from future end-to-end raw-video validation.
- [ ] Read the historical Phase 5.5 dossier only after the current Phase 7 result, so history is not mistaken for the current ceiling.
