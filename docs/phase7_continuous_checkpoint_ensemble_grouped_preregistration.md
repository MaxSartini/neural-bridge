# Phase 7 grouped continuous checkpoint-ensemble preregistration

Status: explicitly authorized by the user and locked before grouped held-out scoring.

The prior fresh blocked confirmation remains exactly recorded: strong aggregate and control evidence, with `4/5` rather than the locked `5/5` Spearman wins versus AR. This grouped run does not relabel that verdict. It asks a different protocol question: does the same continuous target/head and fixed checkpoint ensemble generalize across held-out videos?

## Fixed scope

- AGAIN grouped-video protocol, all five fixed folds.
- Target/head: `residual_future_max_delta_rows_4_10` / `short_temporal_conv_residual`.
- Existing grouped fold-safe PCA only; no refit, re-encoding, V-JEPA, or TRIBE work.
- Nine untouched seeds `20260708`–`20260716`, in three fixed consecutive groups of three.
- Target-specific continuous AR trained/selected inner-only per fold/seed and frozen across matched lanes.
- Canonical original parameters, equal three-checkpoint averaging, no Optuna, member selection, or weight search.
- Seven lanes: real, frozen AR, shuffled, random, label permutation, train-only video mean, and diagnostics-only.
- Exactly `420/420` rows: 315 member plus 105 ensemble.
- MLX GPU/MPS required; no CPU fallback.

## Gate

All conditions must pass:

1. Exact scope plus PCA leakage, causal-context, checkpoint/eval-mode, and frozen-AR identity audits.
2. Aggregate real-minus-AR and real-minus-best-control Spearman each at least `+0.002`.
3. Aggregate real-minus-AR and real-minus-best-control top-5% lift each at least `+0.001`.
4. Aggregate top-1% and top-10% lift beat AR and their respective best controls.
5. At least `12/15` fold-groups positive versus both AR and the best control for both Spearman and top-5% lift.
6. All five fold means positive for all four primary comparisons; all four paired medians positive.
7. Ensemble uplift over members at least `+0.001` Spearman and positive top-5% lift.
8. Real beats label permutation on Spearman and top-5% lift.
9. No single fold-group contributes over 25% of positive real-minus-AR gain for either primary metric.

A pass promotes grouped continuous ranking/lift for this selected Phase 7 target/head. It does not convert the prior blocked result to `5/5`, prove exact numeric values, validate raw-video-only client deployment, or authorize 504.
