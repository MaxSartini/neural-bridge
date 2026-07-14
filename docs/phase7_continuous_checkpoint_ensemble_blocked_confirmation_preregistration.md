# Phase 7 fresh blocked continuous ranking/lift confirmation preregistration

Status: locked after the diagnostic pass and before any confirmation outer held-out scoring.

## Basis and claim boundary

The `84/84` diagnostic passed every preregistered ranking/lift gate: real ensemble Spearman was `0.1226309621` versus target-specific frozen AR `0.1180749764` and best matched control `0.1167953465`; real top-5% lift was `0.0828780504` versus `0.0767870865` and `0.0768766425`. All `3/3` groups were positive on both metrics versus both comparators. The separate exact-value gate failed because MAE, RMSE, and absolute bias did not improve.

This confirmation therefore tests blocked continuous future-movement ranking/lift only. Exact-value promotion is forbidden regardless of the result.

## Fixed confirmation scope

- Dataset/protocol: AGAIN, existing blocked temporal 70/30 split.
- Target/head: `residual_future_max_delta_rows_4_10` / `short_temporal_conv_residual`.
- Inputs: existing dense 2 Hz cache and fold-safe PCA256 only.
- Seeds: fresh untouched `20260693` through `20260707`.
- Groups: five consecutive fixed groups of exactly three seeds.
- Recipe, target-specific AR training/selection, controls, and equal checkpoint averaging: identical to the diagnostic.
- Hardware: MLX GPU/MPS required; no CPU fallback.
- Matrix: 105 member rows plus 35 ensemble rows, exactly `140/140`.
- No Optuna, member selection, weight search, grouped scoring, re-encoding, PCA fitting, or extra targets.

## Confirmation gate

All conditions must pass:

1. Exact `140/140` scope plus causal-context, checkpoint, eval-mode, and frozen-AR identity audits.
2. Mean real-minus-AR and real-minus-best-control Spearman each at least `+0.002`.
3. Mean real-minus-AR and real-minus-best-control top-5% lift each at least `+0.001`.
4. Mean top-1% and top-10% lift beat AR and their respective best matched controls.
5. All `5/5` groups positive versus both AR and best matched control for Spearman and top-5% lift.
6. Median group real-minus-AR gain at least `+0.002` Spearman and `+0.001` top-5% lift.
7. Ensemble uplift over the 15 real members at least `+0.001` Spearman and positive top-5% lift; ensemble Spearman standard deviation at least 20% lower than member standard deviation.
8. Real beats label permutation on Spearman and top-5% lift.
9. No single group contributes over 50% of positive real-minus-AR gain for either primary metric.

A complete pass promotes bounded blocked continuous future-movement ranking/lift for this target/head and authorizes a separately preregistered grouped confirmation. It does not prove exact continuous values or authorize 504.

Every fixed seed and failed gate must be retained and reported.
