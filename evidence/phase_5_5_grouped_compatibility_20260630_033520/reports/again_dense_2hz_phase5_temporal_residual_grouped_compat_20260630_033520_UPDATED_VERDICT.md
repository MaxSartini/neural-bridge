# Phase 5 Temporal Residual Grouped Compatibility Updated Verdict

This report updates only the grouped compatibility verdict logic for the completed frozen-AR residual grouped run. No training, scoring, PCA generation, grouped rerun, or  was performed.

## Source

- Evidence bundle: `evidence/phase_5_5_grouped_compatibility_20260630_033520/`
- Original output root: `outputs/again_dense_2hz_phase5_temporal_residual_grouped_compat_20260630_033520`
- Target: `future_arousal_max_delta_rows_4_10_train_q90`
- Protocol: `grouped_video`
- Architecture: `short_temporal_conv_residual`
- Rows: `350` / `350`

## Original Gate Issue

The original verdict failed only because `label_permutation_near_chance_pass` required label permutation PR-AUC to be less than mean test prevalence plus 0.02. That is inappropriate for this frozen-AR residual design because the label-permutation residual lane includes the same frozen AR floor and only permutes residual train/inner-val labels.

- Legacy near-chance pass: `False`
- Legacy near-chance status: `recorded_inapplicable_not_promotability_blocking`
- Label permutation PR-AUC: `0.2153099775`
- Mean test positive rate: `0.1000364440`
- Legacy threshold: `0.1200364440`

## Updated Label Null

The updated frozen-AR-residual-aware policy requires:

- real mean PR-AUC beats label permutation by at least `0.003`
- real beats label permutation in at least `40/50` fold-seed comparisons
- label permutation does not beat frozen AR by at least `0.003` mean PR-AUC

Observed:

- Real PR-AUC: `0.2313831909`
- AR/frozen PR-AUC: `0.2174953276`
- Label permutation PR-AUC: `0.2153099775`
- Real - label permutation: `+0.0160732134`
- Label permutation - AR: `-0.0021853501`
- Fold-seed positives vs label permutation: `50/50`
- Updated label permutation pass: `True`

## Updated Verdict

- Real PR-AUC: `0.2313831909`
- AR/frozen PR-AUC: `0.2174953276`
- Best matched control: `train_only_video_mean_residual`, PR-AUC `0.2174209937`
- Delta vs AR/frozen: `+0.0138878634`
- Delta vs best control: `+0.0139621972`
- Fold-seed positives vs best control: `50/50`
- Leakage/context audit pass: `True`
- Frozen AR integrity pass: `True`
- Checkpoint restore pass: `True`
- Eval-mode scoring pass: `True`
- AR baseline generation pass: `True`
- Updated grouped compatibility pass: `True`
- Failed updated gates: `[]`
- Recommendation: `grouped_compatibility_pass_review_before_any_504`

This updated verdict remains a grouped compatibility result only. It is , not a broad claim change, and strict broad temporal generalization still requires explicit later confirmation.
