# Phase 7 blocked continuous checkpoint-ensemble preregistration

Status: locked before any Phase 7 outer held-out scoring.

## Question

Does the now-proven `short_temporal_conv_residual` recipe, stabilized by fixed three-checkpoint averaging, improve blocked washout continuous future-arousal movement ranking/lift over a matched frozen AR baseline and all primary matched controls?

This first run is a diagnostic. It cannot by itself promote blocked continuous generalization or exact-value forecasting.

## Fixed scope

- Dataset: AGAIN only.
- Protocol: existing blocked temporal 70/30 split and inner-only checkpoint selection.
- Target: `residual_future_max_delta_rows_4_10`.
- Architecture: `short_temporal_conv_residual`.
- Frozen inputs: existing dense 2 Hz cache and fold-safe PCA256 only.
- Seeds: `20260684` through `20260692`, untouched at lock time.
- Fixed groups: `(20260684, 20260685, 20260686)`, `(20260687, 20260688, 20260689)`, `(20260690, 20260691, 20260692)`.
- Recipe: canonical original Phase 6 parameters, no Optuna and no held-out hyperparameter search.
- AR baseline: newly trained for each exact seed using only inner-train rows, a top-20%/top-10% weighted continuous Huber loss, and inner-only lexicographic selection by top-5% lift, Spearman, then top-10% lift. The selected AR checkpoint is frozen and reused identically across every real/control lane for that seed.
- Ensemble: equal average of exactly three aligned eval-mode checkpoints; no member selection and no weight search.
- Hardware: MLX GPU/MPS required; no CPU fallback.
- Matrix: 63 member rows plus 21 ensemble rows, exactly `84/84`.

Controls are `frozen_ar_only`, `shuffled_pca_residual`, `random_pca_residual`, `label_permutation_residual`, `train_only_video_mean_residual`, and `diagnostics_only_residual`, alongside `real_residual`. Promotion comparisons use the same primary matched-control set as the selected-head evidence: shuffled, random, label permutation, and train-only video mean.

## Ranking/lift diagnostic gate

Every condition must pass:

1. Exact `84/84` scope, unique fixed seeds/groups/lanes, causal same-video context, restored-or-suppressed eval-mode checkpoints, and within-seed frozen-AR checksum identity.
2. Mean real-minus-AR and real-minus-best-control Spearman are each at least `+0.002`.
3. Mean real-minus-AR and real-minus-best-control top-5% continuous lift are each at least `+0.001`.
4. Mean real top-1% and top-10% lift beat AR and their respective best matched controls.
5. All `3/3` groups are positive versus AR and the best matched control on both Spearman and top-5% lift.
6. The real ensemble improves over the nine real members by at least `+0.001` Spearman and by a positive amount on top-5% lift.
7. Real beats label permutation on Spearman and top-5% lift.
8. No single group contributes more than `60%` of the positive real-minus-AR gain for Spearman or top-5% lift.

A pass authorizes a new, fresh, control-complete blocked ranking/lift confirmation. It does not authorize grouped scoring automatically.

## Separate exact-value candidate gate

Exact-value candidacy is evaluated separately and only counts if the ranking/lift gate passes. Mean MAE and RMSE must each improve over AR and the best matched control by at least `0.0005`; every group must improve both errors versus both comparators; absolute bias must be no worse than AR; and peak underprediction must improve over AR.

Passing this gate authorizes a fresh exact-value confirmation. It is not itself a claim that exact continuous forecasting is solved.

## Fail-closed rules

- Retain every fixed seed and report every failed gate.
- Do not change gates, controls, groups, weights, parameters, or target after scoring.
- Do not run grouped follow-up, 504, re-encoding, PCA fitting, or additional targets from this diagnostic.
- Do not call a failed diagnostic a pass because a subset of metrics is favorable.
- If the diagnostic fails, any Optuna or loss redesign requires a new preregistration; this result may motivate but may not tune that search.
