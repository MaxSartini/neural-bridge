# Phase 5 Blocked Residual Clean Confirmation

Output root: `outputs/again_dense_2hz_phase5_blocked_residual_clean_confirm_20260629_234349`

This is the clean monotonic-only blocked confirmation specified by `confirmation_design.json` at design commit `0d6ce16`. It runs only `blocked_temporal_70_30`, only `monotonic_do_no_harm_residual`, only `regression_plus_binary`, and five prespecified seeds. It does not run grouped, does not run 504, does not start secondary targets, and does not change claims.

## Confirmation Result

- Weak confirmation passed: `False`
- Credible threshold passed: `False`
- Recommendation: `clean_confirmation_failed_do_not_run_grouped_or_504`
- Strict forward-time temporal generalization remains unproven: `True`
- Grouped 5-fold compatibility check justified: `False`
- 504 justified: `False`

## PR-AUC

- Real residual: `0.2663039937`
- Frozen AR: `0.2659666756`
- Shuffled PCA residual: `0.2662285526`
- Random PCA residual: `0.2662272015`
- Clean label permutation residual: `0.2661883690`
- Label permutation fixed-epoch audit: `0.2662357187`
- Train-only video mean residual: `0.2663203696`
- Full-video oracle mean residual: `0.2663088239`
- Diagnostics-only residual: `0.2662495572`

## Real-Minus-Control Deltas

- Real minus frozen AR: `+0.0003373181`
- Real minus shuffled PCA: `+0.0000754412`
- Real minus random PCA: `+0.0000767922`
- Real minus clean label permutation: `+0.0001156248`
- Real minus label permutation fixed-epoch audit: `+0.0000682751`
- Real minus train-only video mean: `-0.0000163759`
- Real minus full-video oracle mean: `-0.0000048302`
- Real minus diagnostics-only: `+0.0000544366`

## Gates

- `real_gt_frozen_ar_pass`: `False`
- `real_gt_shuffled_pass`: `False`
- `real_gt_random_pass`: `False`
- `real_gt_clean_label_permutation_pass`: `False`
- `real_gt_train_only_video_mean_pass`: `False`
- `seed_consistency_pass`: `False`
- `primary_seed_positive_counts`: `{'real_minus_frozen_ar_only': 5, 'real_minus_shuffled_pca_residual': 5, 'real_minus_random_pca_residual': 5, 'real_minus_label_permutation_residual_permuted_inner_val_selection': 5, 'real_minus_train_only_video_mean_pca_residual': 1}`
- `no_single_seed_over_60pct_pass`: `False`
- `do_no_harm_blocked_pass`: `True`
- `full_video_oracle_warning`: `False`
- `frozen_ar_integrity_pass`: `True`
- `checkpoint_restore_pass`: `True`
- `eval_mode_scoring_pass`: `True`

The weak pass threshold is `+0.0010` PR-AUC on every primary blocked gate. The credible threshold is `+0.0030` PR-AUC. Full-video oracle mean is a mechanism warning, while train-only video mean is the promotability-blocking static-control gate.

## Interpretation

This report is a confirmation result for the blocked residual candidate only. It does not update the canonical grouped claim by itself. If weak confirmation fails, do not run grouped 5-fold compatibility or 504. If weak confirmation passes, grouped 5-fold compatibility may be considered next, but strict forward-time temporal generalization should remain explicitly caveated unless the credible threshold and all control gates are satisfied.
