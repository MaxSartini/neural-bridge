# Phase 5 Temporal Residual Grouped Compatibility

Output root: `outputs/again_dense_2hz_phase5_temporal_residual_grouped_compat_20260630_033520`

This is a grouped-video compatibility check for the confirmed blocked binary washout-gap short temporal conv residual. It uses the same 10 seeds as the blocked confirmation across all 5 grouped-video folds. It does not run 504, continuous targets, extra targets, extra architectures, V-JEPA/TRIBE, or claim changes.

## Scope

- Target: `future_arousal_max_delta_rows_4_10_train_q90`
- Protocol: `grouped_video`
- Architecture: `short_temporal_conv_residual`
- Fold-safe grouped PCA root: `outputs/again_dense_2hz_phase5_temporal_residual_grouped_compat_20260630_033520/foldsafe_grouped_pca`
- Rows completed / expected: `350` / `350`
- AR baselines reused: `0`
- AR baselines newly trained: `50`
- Each fold/seed uses its own frozen AR score: `True`
- All controls within each fold/seed use identical frozen AR scores: `True`
- Shared 3-seed AR cache reused across 10-seed grouped compatibility: `False`

## Result

- Real PR-AUC: `0.2313831909`
- AR/frozen baseline PR-AUC: `0.2174953276`
- Best control: `train_only_video_mean_residual` PR-AUC `0.2174209937`
- Delta vs AR/frozen baseline: `+0.0138878634`
- Delta vs best control: `+0.0139621972`
- Fold-seed positives vs AR/frozen baseline: `50/50`
- Fold-seed positives vs best control: `50/50`
- Label permutation PR-AUC: `0.2153099775`
- Mean test positive rate: `0.1000364440`

## Gates

- `grouped_compatibility_pass`: `False`
- `leakage_context_audit_pass`: `True`
- `frozen_ar_integrity_pass`: `True`
- `checkpoint_restore_pass`: `True`
- `eval_mode_scoring_pass`: `True`
- `ar_baseline_generation_pass`: `True`
- `label_permutation_near_chance_pass`: `False`
- Failed gates: `['label_permutation_not_near_chance']`
- Recommendation: `grouped_compatibility_failed_do_not_run_504`

This report is a compatibility check, not a 504 run and not a broad claim change. Strict broad temporal generalization remains subject to review and any later explicitly approved confirmation.
