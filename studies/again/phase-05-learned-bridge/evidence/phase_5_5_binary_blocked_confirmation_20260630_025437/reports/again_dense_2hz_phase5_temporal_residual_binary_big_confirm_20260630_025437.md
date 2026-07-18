# Phase 5 Temporal Residual Binary Big Confirmation

Output root: `outputs/again_dense_2hz_phase5_temporal_residual_binary_big_confirm_20260630_025437`

This is a blocked-only 10-seed confirmation for the redesigned binary washout-gap target using only `short_temporal_conv_residual`. It does not run continuous, grouped, extra targets, extra architectures, V-JEPA/TRIBE/PCA, PCA refit, or claim changes.

## Scope

- Target: `future_arousal_max_delta_rows_4_10_train_q90`
- Protocol: `blocked_temporal_70_30`
- Architecture: `short_temporal_conv_residual`
- Residual/control rows: `70`
- AR baselines reused: `3` seeds, `[20260625, 20260626, 20260627]`
- AR baselines newly trained: `7` seeds, `[20260628, 20260629, 20260630, 20260631, 20260632, 20260633, 20260634]`
- Each seed uses its own frozen AR score: `True`
- All controls within a seed use identical frozen AR scores: `True`
- Shared 3-seed AR cache reused across the 10-seed confirmation: `False`
- AR-only baseline generation is reported separately from residual/control rows: `True`
- 10/10 seed confirmation valid: `True`

## Result

- Real PR-AUC: `0.2670735630`
- Frozen AR PR-AUC: `0.2602336231`
- Best control: `random_pca_residual` PR-AUC `0.2593369051`
- Delta vs frozen AR: `+0.0068399399`
- Delta vs best control: `+0.0077366579`
- Seeds positive vs AR: `9/10`
- Seeds positive vs best control: `9/10`
- Max seed contribution vs AR: `0.1537`
- Max seed contribution vs best control: `0.1486`

## Gates

- Weak confirmation: `True`
- Credible confirmation: `True`
- Strong confirmation: `True`
- `leakage_context_audit_pass`: `True`
- `frozen_ar_integrity_pass`: `True`
- `within_seed_controls_match_ar_pass`: `True`
- `checkpoint_restore_pass`: `True`
- `eval_mode_scoring_pass`: `True`
- `ar_baseline_generation_pass`: `True`
- Failed gates: `[]`
- Recommendation: `binary_big_confirmation_pass_review_before_any_grouped_or_504`

Strict forward-time temporal generalization remains unproven until any further confirmation is explicitly reviewed and promoted. This report alone does not authorize grouped.
