# Phase 5 Blocked Residual Targeted Evidence Snapshot

This is a lightweight tracked evidence snapshot for the bounded 168-run blocked residual diagnostic.

It is diagnostic-only. It is not a headline benchmark, and it is not comparable to the prior full 5-fold grouped result.

## Key Result

- Blocked frozen AR PR-AUC: `0.2654721820`
- Blocked best real residual: `monotonic_do_no_harm_residual`, PR-AUC `0.2657861164`
- Blocked best matched control: `shuffled_pca_residual` / `monotonic_do_no_harm_residual`, PR-AUC `0.2657134334`
- Blocked delta vs frozen AR: `+0.0003139344`
- Blocked delta vs best matched control: `+0.0000726830`

## Interpretation

This diagnostic found a tiny positive blocked delta for `monotonic_do_no_harm_residual`, but failed control gates prevent any promotion or strict forward-time claim.

- `grouped_fold1_reference_only` is a sanity check only and not a canonical grouped benchmark.
- `full_forward_time_pass` is diagnostic-only here and is not promotable.
- `label_permutation_pass`: no
- `video_mean_static_control_pass`: no
- Recommendation: `blocked_delta_positive_but_control_failures`
- This diagnostic did not prove strict temporal improvement.

`monotonic_do_no_harm_residual` is the best candidate for a future cleaner confirmation, but a  run should not be started until the label-permutation and static-control failures are understood.

Current status: later target redesign and temporal/event-context residual work moved beyond this monotonic residual diagnostic. The current bounded AGAIN blocked proof is for `future_arousal_max_delta_rows_4_10_train_q90` with `short_temporal_conv_residual`.
