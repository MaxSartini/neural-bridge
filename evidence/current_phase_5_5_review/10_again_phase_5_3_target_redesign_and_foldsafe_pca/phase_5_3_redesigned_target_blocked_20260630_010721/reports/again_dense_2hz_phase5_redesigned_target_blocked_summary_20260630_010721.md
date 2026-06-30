# Phase 5 Redesigned Target Blocked Summary

Output root: `outputs/again_dense_2hz_phase5_redesigned_target_blocked_20260630_010721`

This is a bounded blocked-only redesigned target training test. It uses two approved targets, three seeds, seven controls, the fold-safe redesigned-target PCA root, and one residual variant: `monotonic_do_no_harm_residual`. It does not run grouped, 504, broad variants, extra targets, AR retraining, V-JEPA/TRIBE, or PCA refitting.

## Binary Target

- Target: `future_arousal_max_delta_rows_4_10_train_q90`
- Real PR-AUC: `0.2663692744`
- Frozen AR PR-AUC: `0.2662144771`
- Delta vs frozen AR: `+0.0001547973`
- Best control: `train_only_video_mean_residual` PR-AUC `0.2663835748`
- Delta vs best control: `-0.0000143004`
- Seed positive count: `1/3`
- Binary pass: `False`

## Continuous Target

- Target: `residual_future_max_delta_rows_4_10`
- Real Spearman: `0.0577169274`
- Frozen AR Spearman: `0.0587485086`
- Spearman delta: `-0.0010315812`
- Real top 5pct lift: `0.0780489144`
- Frozen AR top 5pct lift: `0.0837513031`
- Top 5pct lift delta vs frozen AR: `-0.0057023888`
- Best control: `train_only_video_mean_residual` top 5pct lift `0.0839012635`
- Delta vs best control: `-0.0058523491`
- Seed positive count: `0/3`
- Continuous pass: `False`

## Gates

- `frozen_ar_integrity_pass`: `True`
- `checkpoint_restore_pass`: `True`
- `eval_mode_scoring_pass`: `True`
- Failed gates: `['binary_min_delta_threshold', 'binary_seed_consistency', 'continuous_min_delta_threshold', 'continuous_spearman_delta', 'continuous_seed_consistency', 'continuous_binary_do_no_harm']`
- Recommendation: `redesigned_target_blocked_failed_do_not_run_grouped_or_504`

Strict forward-time temporal generalization remains unproven. Do not run grouped or 504 from this result unless the gates pass cleanly and the result is reviewed.
