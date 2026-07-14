# Phase 5 Blocked Residual Targeted Diagnostic

Output root: `outputs/again_dense_2hz_phase5_blocked_residual_targeted_20260629_230325`

## Scope

This is a bounded diagnostic-only tuning run, not a headline benchmark. It uses the full `blocked_temporal_70_30` split as the primary objective and `grouped_video` fold 1 only as `grouped_fold1_reference_only`.

The grouped fold 1 reference is not comparable to the prior full 5-fold grouped result and must not be reported as a canonical grouped benchmark. No full grouped residual pass is claimed from this run.

## Variants

- `blocked_delta_selected_gated_residual`
- `monotonic_do_no_harm_residual`
- `low_ar_confidence_residual`
- `rank_lift_residual`

Controls per variant: real residual, frozen AR only, shuffled PCA residual, random PCA residual, diagnostics-only residual, video-mean PCA residual, and label permutation residual.

## Blocked Primary Result

- Blocked frozen AR PR-AUC: `0.2654721820`
- Blocked best real residual: `monotonic_do_no_harm_residual` PR-AUC `0.2657861164`
- Blocked best matched control: `shuffled_pca_residual` / `monotonic_do_no_harm_residual` PR-AUC `0.2657134334`
- Blocked delta vs frozen AR: `+0.0003139344`
- Blocked delta vs best matched control: `+0.0000726830`

## Grouped Fold 1 Reference Only

- Grouped fold 1 frozen AR PR-AUC: `0.2445517115`
- Grouped fold 1 best real residual: `monotonic_do_no_harm_residual` PR-AUC `0.2445895162`
- Grouped fold 1 best matched control: `random_pca_residual` / `low_ar_confidence_residual` PR-AUC `0.2446558191`
- Grouped fold 1 delta vs frozen AR: `+0.0000378047`
- Grouped fold 1 delta vs best matched control: `-0.0000663029`

## Gates

- `blocked_residual_pass`: `True`
- `do_no_harm_blocked_pass`: `True`
- `frozen_ar_integrity_pass`: `True`
- `label_permutation_pass`: `False`
- `video_mean_static_control_pass`: `False`
- `control_failure_blocks_actionable_repair`: `True`
- `full_forward_time_pass`: `True`
- `grouped_residual_pass`: `not_evaluated_grouped_fold1_reference_only`
- `grouped_fold1_reference_sanity_pass`: `True`
- `recommendation`: `blocked_delta_positive_but_control_failures`

`full_forward_time_pass` is diagnostic-only here and is not promotable. Strict forward-time temporal generalization remains unproven. `monotonic_do_no_harm_residual` is the best candidate for a future cleaner confirmation, but run should not be started until the label-permutation and static-control failures are understood.
