# Phase 6 Fixed-Blend Fresh-Five Pilot Plan

## Status

Completed and stopped fail-closed on `2026-07-14`. All `20/20` rows ran on MLX
without weight search or reuse of viewed seed scores. The blend beat original
in `5/5`, Trial 4 in `3/5`, and AR in `5/5`, but failed the minimum improvement,
median, and variance-reduction gates. A control-complete confirmation was not
authorized.

Mean PR-AUC was `0.2655538049` for the blend, `0.2647935898` for original,
`0.2653912587` for Trial 4, and `0.2588130741` for AR. The blend was only
`+0.0001625462` above the stronger component and its seed-level standard
deviation was `7.21%` higher than the lower-variance component.

## Question

Does a fixed blend of the canonical original and locked Trial 4 models reduce
configuration/checkpoint variance and improve blocked future-event ranking on
genuinely fresh training seeds?

## Canonical Sensitivity Context

The seed-`20260627` favorable checkpoint does not invalidate the canonical 420,
but it made later candidate-versus-original comparisons unusually harsh. A
read-only leave-one-seed-out audit performed before this pilot found that the
canonical blocked result without seed `20260627` still beats frozen AR by
`+0.0064196765` and the per-seed best primary control by `+0.0065709972`, with
`8/9` positive seeds versus both. Grouped compatibility without seed
`20260627` remains positive in `45/45`, with `+0.0139177991` versus AR and
`+0.0139392692` versus the per-fold/seed best primary control. The 420 proof is
therefore not dependent on the spike; the open issue is model/checkpoint
stability and fair comparison to an occasionally lucky original recipe.

## Fixed Scope

- target: `future_arousal_max_delta_rows_4_10_train_q90`
- architecture: `short_temporal_conv_residual`
- protocol: `blocked_temporal_70_30`
- fresh seeds: `20260640`–`20260644`
- accelerator: MLX `Device(gpu, 0)`; CPU fallback is forbidden
- original parameters: the literal `ORIGINAL_PARAMS` already frozen in the
  robust Optuna runner
- Trial 4 parameters: the literal `TRIAL4_PARAMS` already frozen in the Stage
  A2 runner
- ensemble: row-aligned arithmetic mean of the two binary logits and of the two
  regression outputs, with weights exactly `0.5 / 0.5`
- no weight search, per-seed selection, checkpoint cherry-picking, Optuna, or
  use of the already-viewed seeds `20260625`–`20260639`

Each component trains on the same outer-train rows, selects its own best
checkpoint using only the existing inner-validation split, restores that
checkpoint in eval mode, and is then scored once on blocked held-out rows. The
frozen AR model is trained/selected inner-only and reused identically by both
components and the blend within each seed.

## Pilot Rows

Five seeds x four reported lanes = `20` metric rows:

1. frozen AR
2. canonical original real residual
3. Trial 4 real residual
4. fixed 50/50 original + Trial 4 blend

Matched semantic controls are deliberately deferred. This pilot asks only
whether blending improves the two already-controlled component recipes and
retains a material AR margin. A pass authorizes a separate control-complete
confirmation plan; it does not promote a model or alter the canonical 420.

## Locked Pass Gates

All gates must pass:

1. ensemble mean PR-AUC exceeds the higher component mean by at least `0.0005`
2. ensemble median PR-AUC exceeds the higher component median
3. ensemble beats original in at least `3/5` paired seeds
4. ensemble beats Trial 4 in at least `3/5` paired seeds
5. ensemble seed-level PR-AUC standard deviation is at least `5%` lower than
   the lower component standard deviation
6. ensemble mean PR-AUC exceeds frozen AR by at least `0.003`
7. ensemble beats frozen AR in at least `4/5` seeds
8. no one seed contributes more than `50%` of the blend's positive aggregate
   gain over the stronger component-by-mean
9. all five seeds and exactly 20 rows are present, row ordering/checksums align,
   checkpoints restore in eval mode, causal-only context audits pass, and MLX
   GPU execution is attested

Ties do not count as paired wins. Standard deviation uses population `ddof=0`.

## Stop Rule And Claim Boundary

- Any failed gate stops the experiment. Do not tune the weight after seeing the
  result and do not run matched controls or grouped evaluation.
- A pass means only that the fixed blend deserves a separately preregistered,
  control-complete confirmation on additional fresh seeds.
- This pilot cannot replace or weaken the canonical 420-row result, authorize
  the failed robust campaign's Stage C, revive 504, or support exact continuous
  forecasting claims.
