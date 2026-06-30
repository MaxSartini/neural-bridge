# Phase 5 Temporal Residual Blocked Summary

Output root: `outputs/again_dense_2hz_phase5_temporal_residual_blocked_20260630_020557`

This is a bounded blocked-only temporal/event-context residual diagnostic over the redesigned targets. It uses the fold-safe redesigned PCA256 artifacts and keeps frozen AR as the baseline floor. It does not run grouped, 504, extra targets, V-JEPA/TRIBE/PCA, AR retraining, or claim changes.

## Binary Washout-Gap Target

- Target: `future_arousal_max_delta_rows_4_10_train_q90`
- Best architecture: `short_temporal_conv_residual`
- Real PR-AUC: `0.2738556307`
- Frozen AR PR-AUC: `0.2662144771`
- Best control: `random_pca_residual` PR-AUC `0.2654930636`
- Delta vs frozen AR: `+0.0076411535`
- Delta vs best control: `+0.0083625671`
- Seed positive count: `3/3`
- Binary pass: `True`

## Continuous AR-Residualized Target

- Target: `residual_future_max_delta_rows_4_10`
- Best architecture: `short_temporal_conv_residual`
- Real Spearman: `0.0642716053`
- Frozen AR Spearman: `0.0587485086`
- Spearman delta vs frozen AR: `+0.0055230967`
- Real top 5pct lift: `0.0836282345`
- Frozen AR top 5pct lift: `0.0837513031`
- Best control: `train_only_video_mean_residual` top 5pct lift `0.0836439502`
- Delta vs frozen AR: `-0.0001230687`
- Delta vs best control: `-0.0000157158`
- Seed positive count: `1/3`
- Continuous pass: `False`

## Gates

- `leakage_context_audit_pass`: `True`
- `frozen_ar_integrity_pass`: `True`
- `checkpoint_restore_pass`: `True`
- `eval_mode_scoring_pass`: `True`
- Failed gates: `['continuous_min_delta_threshold', 'continuous_seed_consistency']`
- Recommendation: `temporal_residual_blocked_failed_do_not_run_grouped_or_504`

Strict forward-time temporal generalization remains unproven. This diagnostic should not trigger grouped or 504 unless the blocked gates pass cleanly and the result is reviewed.
