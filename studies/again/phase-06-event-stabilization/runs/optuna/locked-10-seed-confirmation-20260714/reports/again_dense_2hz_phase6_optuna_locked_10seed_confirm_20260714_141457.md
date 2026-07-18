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

This result is exploratory until reviewed and deliberately promoted. A pass may
justify a later grouped locked-winner confirmation; it does not itself change
the canonical 420-row claim.
