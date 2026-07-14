# Phase 6 Original Three-Checkpoint Grouped Confirmation Plan

## Status

Completed and passed all locked gates on `2026-07-14`. Exactly `420/420` rows
ran on MLX: `315` member rows plus `105` ensemble rows across five grouped-video
folds. Real ensemble PR-AUC was `0.2343675680` versus AR `0.2180497906` and
best aggregate matched control `train_only_video_mean_residual` at
`0.2179716645`. Deltas were `+0.0163177774` and `+0.0163959035`; all `15/15`
fold-groups and all `5/5` fold means were positive versus both. Ensembling added
`+0.0082200727` over the 45 real-member mean and won `15/15`. Failed gates were
`[]`.

## Question

Does the already promoted original three-checkpoint ensemble retain its controlled
future-event ranking advantage when entire videos, rather than temporal blocks,
are held out?

## Locked scope

- target/head: `future_arousal_max_delta_rows_4_10_train_q90` /
  `short_temporal_conv_residual`
- protocol: five fixed grouped-video folds
- untouched seeds: `20260675` through `20260683`
- fixed checkpoint groups: `(75,76,77)`, `(78,79,80)`, `(81,82,83)`
- lanes: frozen AR, real residual, shuffled PCA, random PCA, label
  permutation, train-only video mean, and diagnostics-only residual
- original training parameters only; no Optuna search, member selection, weight
  fitting, seed deletion, or held-out adaptation
- MLX GPU execution; existing fold-safe grouped PCA may be reused only after its
  leakage manifest passes

The matrix is exactly `420` scored rows: `315` member rows
(`5 folds x 9 seeds x 7 lanes`) plus `105` ensemble rows
(`5 folds x 3 groups x 7 lanes`). Each ensemble is the unweighted mean of three
aligned eval-mode checkpoint score vectors.

## Preregistered gates

Promotion requires every gate:

1. real ensemble mean exceeds matched AR ensemble mean by at least `0.005`;
2. real ensemble mean exceeds the strongest aggregate primary matched control by
   at least `0.005`;
3. all `15/15` fold-groups are positive versus AR and their strongest matched
   control;
4. every fold's three-group mean is positive versus AR and best control;
5. paired medians versus AR and best control are positive;
6. real ensemble mean exceeds the mean of its 45 real members by at least `0.001`,
   and at least `12/15` ensembles beat their own three-member mean;
7. label-permutation ensemble minus AR is at most `0.001`;
8. no single fold-group supplies more than `25%` of total positive real-minus-best-
   control uplift;
9. exact matrix/split/provenance, frozen-AR identity, causal-context, checkpoint-
   restoration, eval-mode, and MLX audits pass.

Failure is informative and remains recorded. It does not invalidate the existing
blocked ensemble confirmation or canonical selected-head 420. Passing promotes
only grouped-video compatibility for this bounded checkpoint-ensemble method; it
does not establish exact continuous forecasting or universal prediction.
