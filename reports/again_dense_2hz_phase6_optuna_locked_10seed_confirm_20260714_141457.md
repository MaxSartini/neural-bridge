# Phase 6 Optuna Locked-Winner 10-Seed Confirmation

Output root: `outputs/again_dense_2hz_phase6_optuna_locked_10seed_confirm_20260714_141457`

This confirmation applies one pilot-locked configuration unchanged across the
canonical blocked seeds. It performs no optimization, encoder/PCA work,
grouped evaluation, 420 rerun, 504 reconstruction, or continuous modeling.

## Result

- tuned PR-AUC: `0.2659654274`
- canonical original PR-AUC: `0.2670735630`
- frozen AR PR-AUC: `0.2602336231`
- best tuned matched control: `random_pca_residual` / `0.2592017765`
- tuned minus original, all 10 seeds: `-0.0011081356`
- tuned minus original, nine follow-up seeds: `-0.0014666488`
- tuned minus frozen AR: `+0.0057318043`
- tuned minus best control: `+0.0067636509`
- positive vs original: `7/10`
- positive vs original on follow-up seeds: `6/9`
- positive vs frozen AR / best control: `8/10` / `8/10`
- maximum positive-seed contribution vs original: `0.2754`

## Prespecified Verdict

- locked improvement pass: `False`
- failed gates: `['followup_mean_delta_at_least_0_001', 'full_mean_delta_positive']`

## Robustness And Seed 20260627 Audit

- paired median tuned-minus-original: `+0.0004281433`
- seeds positive: `7/10`
- mean excluding seed `20260627`: `+0.0007535112`
- one-sided sign-test p-value: `0.171875`
- one-sided paired Wilcoxon p-value: `0.2158203125`
- bootstrap 95% CI for the mean: `[-0.00507096, +0.00127274]`

Seed `20260627` is an empirical outlier in the original training history. Its
original inner-validation delta rose to `+0.0272760719` at epoch 14—about 73%
above the original runs' median best peak—then declined. The original held-out
PR-AUC was `0.2770178562`; locked tuned was `0.2591548994`.

An explicitly post-hoc convergence diagnostic extended tuned training to 80
epochs with patience 12. It selected epoch 38 and reproduced `0.2591548994`
exactly, so the tuned loss is not explained by premature termination at the
40-epoch ceiling. This supports an unusually favorable original seed/checkpoint,
but does not license deleting the seed or changing the prespecified verdict.

The defensible interpretation is a small typical-seed improvement without
evidence of a robust aggregate improvement. Future Optuna work should optimize
an aggregate inner-validation objective across multiple development seeds,
then lock once before any new evaluation.

This result does not promote the locked Optuna configuration and does not change
the canonical 420-row claim.
