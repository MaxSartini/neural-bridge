# Phase 5 Blocked Residual Control Failure Audit

Output root: `outputs/again_dense_2hz_phase5_blocked_residual_targeted_20260629_230325/`

Source report: `reports/again_dense_2hz_phase5_blocked_residual_targeted_summary_230325.md`

## Bottom Line

The bounded blocked residual diagnostic produced a tiny positive blocked delta, but the failed controls make it directional only and non-promotable.

- Blocked frozen AR PR-AUC: `0.2654721820`
- Best real residual: `monotonic_do_no_harm_residual`, PR-AUC `0.2657861164`
- Best matched control: `shuffled_pca_residual` / `monotonic_do_no_harm_residual`, PR-AUC `0.2657134334`
- Real delta vs frozen AR: `+0.0003139344`
- Real delta vs best matched control: `+0.0000726830`
- Recommendation: `blocked_delta_positive_but_control_failures`

Strict forward-time temporal generalization remains unproven.

## Failed Gate Reasons

`label_permutation_pass` failed because the best blocked label-permutation row exceeded the best blocked real residual.

- Best label permutation: `blocked_delta_selected_gated_residual`, PR-AUC `0.2661663455`
- Best real residual: `monotonic_do_no_harm_residual`, PR-AUC `0.2657861164`
- Label permutation minus real: `+0.0003802291`

`video_mean_static_control_pass` failed because the best blocked video-mean static/oracle row slightly exceeded the best blocked real residual.

- Best video-mean static control: `monotonic_do_no_harm_residual`, PR-AUC `0.2658151313`
- Best real residual: `monotonic_do_no_harm_residual`, PR-AUC `0.2657861164`
- Video mean minus real: `+0.0000290149`

The gates failed because the controls exceeded real. They did not fail because of a separate minimum threshold. The practical issue is that all margins are extremely small, including the real-minus-best-matched-control margin of only `+0.0000726830`.

## Control Matching And AR Integrity

The controls were present for the same blocked protocol, fold, seeds, and variants as the real residual:

- Protocol: `blocked_temporal_70_30`
- Fold: `1`
- Seeds: `20260625`, `20260626`, `20260627`
- Variants: `blocked_delta_selected_gated_residual`, `monotonic_do_no_harm_residual`, `low_ar_confidence_residual`, `rank_lift_residual`

The gate comparison is best-of-family: best real residual over variants is compared with best label-permutation and best video-mean controls over variants. This is conservative, but it makes tiny effects easy to overturn when any control variant catches model-selection noise.

The frozen AR score was shared across real and controls. The same per-seed `frozen_ar_test_checksum` values appeared for real, label permutation, video mean, shuffled PCA, and random PCA controls:

- `20260625`: `fa40aeff7baf6a11289df36b6efaaf4b`
- `20260626`: `8c7a555b24d19576e41d88cda28a8d24`
- `20260627`: `695629cbd47953323ea7a413bc945deb`

`frozen_ar_integrity_pass` remained true, with `same_ar_as_reference: true` in the integrity audit. `checkpoint_restore_pass` also remained true.

## Construction Notes

The label permutation control permutes the residual training targets, then still selects checkpoints by true inner-validation PR-AUC against the frozen AR baseline. That makes it a hard model-selection/null-control test: random residual learning can still win a small amount if inner-validation selection noise aligns with held-out scoring. This is not direct held-out leakage, but it explains why a tiny real delta is not robust enough to promote.

The video-mean PCA control is an intentional static/oracle nuisance control. The implementation computes per-video PCA means from concatenated train and test rows, then replaces each row with its video mean and disables temporal diagnostics. This is not a deployable train-only baseline; it is a stress test for static video identity/content effects. Failure against it means the blocked residual improvement is not clearly stronger than static video-level information.

## Leakage Assessment

There is no direct evidence of gross leakage from these artifacts:

- `frozen_ar_integrity_pass`: true
- `checkpoint_restore_pass`: true
- real and controls use the same frozen AR scores per seed
- controls are matched by protocol/fold/seed/variant in the matrix

The video-mean control is intentionally oracle-like and uses test rows to build static per-video means, so it is a leakage warning control by design. Its failure does not prove the real residual leaked; it says the real blocked delta is too small to separate from a static/oracle nuisance baseline.

## Claim Impact

This audit does not invalidate the prior grouped/cross-video claim because this run used only `grouped_fold1_reference_only` as a sanity check and did not evaluate the full 5-fold grouped benchmark.

This audit does not invalidate the frozen-AR do-no-harm finding. The blocked real residual delta was positive here, and `do_no_harm_blocked_pass` stayed true. The issue is promotion, not harm.

This audit does invalidate using the bounded blocked diagnostic as proof of strict forward-time temporal generalization. The result is valid as a directional diagnostic, but not valid as promotable evidence.

## Classification

- Expected because the real delta is tiny: yes.
- Threshold/gate-definition issue: partly. The gate uses strict best-real-greater-than-best-control logic without a minimum margin, and the best-of-family comparison is conservative.
- Control construction bug: no clear bug found in the artifacts. The label-permutation and video-mean controls are doing their intended stress-test roles, though the label-permutation checkpoint selection can expose model-selection noise.
- Leakage warning: weak warning only. No gross leakage evidence; video-mean is intentionally oracle/static.
- Split/prevalence artifact: plausible contributor. The blocked split uses one fold and three seeds, and PR-AUC differences are at the `1e-4` scale.
- Evidence that the diagnostic result is not promotable: yes.

## Recommended Next Action

Do not run and do not start more training until the failed controls are understood.

The next action should be a no-training analysis of the existing diagnostic artifacts:

- Recompute existing metrics under same-variant-only control gates to separate variant-selection effects from true control failures.
- Inspect label-permutation checkpoint-selection behavior, especially cases where permuted-label residuals beat frozen AR.
- Treat `monotonic_do_no_harm_residual` as the best candidate for a future cleaner confirmation only after label-permutation and static-control behavior is explained.
