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
- Strict forward-time temporal generalization remains unproven.

`monotonic_do_no_harm_residual` is the best candidate for a future cleaner confirmation, but a 504-style confirmation run should not be started until the label-permutation and static-control failures are understood.
