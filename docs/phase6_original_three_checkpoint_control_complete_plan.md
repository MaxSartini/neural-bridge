# Phase 6 Original Three-Checkpoint Control-Complete Plan

## Status

Preregistered before training or reading blocked held-out scores for seeds
`20260660`–`20260674`.

## Candidate And Scope

- target/head: `future_arousal_max_delta_rows_4_10_train_q90` /
  `short_temporal_conv_residual`
- blocked protocol only; MLX `Device(gpu, 0)` required
- candidate: equal-logit average of three independently trained checkpoints
  using the literal canonical original parameters
- five fixed groups: `(60,61,62)`, `(63,64,65)`, `(66,67,68)`, `(69,70,71)`,
  `(72,73,74)` with full `202606xx` seed values
- lanes: frozen AR plus real, shuffled-PCA, random-PCA, label-permutation,
  train-only-video-mean, and diagnostics-only residuals
- every residual/control lane is independently trained for every seed using the
  same recipe, same inner-only checkpoint selection, and the same seed-specific
  frozen AR inside that seed
- no member selection, weight search, dropped group, or viewed-seed reuse

The complete matrix is 15 seeds x 7 member lanes plus five groups x 7 ensemble
lanes = exactly `140` rows.

## Locked Gates

All must pass:

1. real ensemble mean exceeds AR ensemble by `>=0.005`
2. real ensemble mean exceeds the highest aggregate primary-control ensemble by
   `>=0.005`
3. real ensemble beats AR and the per-group best primary control in `5/5`
4. median group deltas versus AR and best control are positive
5. real ensemble mean exceeds the 15 real-member mean by `>=0.001`
6. real ensemble beats its within-group real-member mean in at least `4/5`
7. label-permutation ensemble minus AR ensemble is `<=0.001`
8. no group supplies more than `50%` of positive aggregate gain over the
   per-group best primary control
9. exactly 15 untouched seeds, five disjoint groups, and 140 rows; frozen-AR
   reuse, causal context, control policy, row alignment, checkpoint restore,
   checksum, eval mode, and MLX audits pass

Failure stops before grouped evaluation. A pass promotes only blocked
control-complete evidence for this ensemble recipe and authorizes a separate
fresh grouped confirmation plan. It does not alter the canonical 420 or broaden
continuous claims.

