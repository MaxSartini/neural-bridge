# Phase 6 Trial-4 Three-Checkpoint Fresh-15 Plan

## Status

Completed `60/60` rows on MLX and stopped fail-closed. Trial-4 checkpoint
ensembling worked as a stabilization mechanism but lost to the matched original
checkpoint ensemble: `0.2687444415` versus `0.2717155074`, with `2/5` paired
wins. It gained `+0.0027137975` over the 15 Trial-4 single-checkpoint mean,
reduced Trial-4 variability by `85.89%`, and beat AR in `5/5` by `+0.0087071701`.
The original ensemble, a prespecified comparator, gained `+0.0044814318` over
its 15 member mean and beat the AR ensemble by `+0.0116782360`; this is promising
exploratory evidence, not a post-hoc promotion. Trial-4 control-complete followup
was not authorized.

## Purpose

Run the larger retraining requested by the project owner using the locked
Optuna-selected Trial 4 parameters, while testing the specific remaining
hypothesis: averaging independently trained checkpoints can suppress rare
favorable-checkpoint outliers without sacrificing the controlled bridge signal.

This is true checkpoint ensembling, not the failed within-seed 50/50 blend.

## Fixed Design

- target/head: `future_arousal_max_delta_rows_4_10_train_q90` /
  `short_temporal_conv_residual`
- protocol: `blocked_temporal_70_30`
- accelerator: MLX `Device(gpu, 0)` only
- fresh seeds: `20260645`–`20260659`
- groups: `(45,46,47)`, `(48,49,50)`, `(51,52,53)`, `(54,55,56)`,
  `(57,58,59)` using the full `202606xx` seed values
- candidate: arithmetic mean of three row-aligned Trial-4 binary logits and
  regression outputs
- matched recipe comparator: arithmetic mean of three row-aligned canonical
  original logits/outputs trained on the identical three seeds
- AR comparator: arithmetic mean of the same three seed-specific frozen-AR
  logits/outputs
- Trial 4 and original parameters remain literal-pinned to the existing Phase 6
  runners; no Optuna rerun, weight search, member selection, or dropped member
- every model trains on the full outer-train partition and selects its checkpoint
  using inner validation only before one blocked held-out score

Reported matrix: 15 seeds x 3 member lanes plus 5 groups x 3 ensemble lanes =
exactly `60` rows. Member rows are retained for variance and ensemble-uplift
audits.

## Locked Gates

All must pass:

1. Trial-4 ensemble mean PR-AUC exceeds original-ensemble mean by `>=0.0005`
2. Trial-4 ensemble median exceeds original-ensemble median
3. Trial-4 ensemble beats original ensemble in at least `4/5` groups
4. Trial-4 ensemble mean exceeds the mean of all 15 Trial-4 members by
   `>=0.0005`
5. Trial-4 ensemble beats its own within-group mean member PR-AUC in at least
   `4/5` groups
6. Trial-4 ensemble group-level PR-AUC population standard deviation is at
   least `20%` below the 15 Trial-4 members' population standard deviation
7. Trial-4 ensemble exceeds the AR ensemble by `>=0.003` on mean and in `5/5`
8. no group contributes more than `50%` of positive aggregate candidate gain
   over the original ensemble
9. exactly 15 fresh seeds, five disjoint three-member groups, and 60 rows;
   causal-context, row alignment, eval-mode checkpoint restore, checksum, and
   MLX audits all pass

## Stop And Claim Rules

Failure stops before matched semantic controls or grouped evaluation. Passing
authorizes only a separately frozen control-complete blocked confirmation. This
experiment cannot change the canonical 420 or support exact
continuous forecasting.
