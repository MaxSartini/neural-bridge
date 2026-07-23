# VEATIC 2.1 Endpoint and Control Amendment

## Status and timing

This amendment is registered before the q900 current-innovation control matrix is run. It
does not reinterpret any completed failed matrix: both the original causal residual and the
full temporal innovation redesign remain failed under their recorded gates.

Preregistration is an audit safeguard, not evidence that a chosen endpoint or gate is
scientifically optimal. This amendment reconciles the active VEATIC work with the complete
Neural Bridge lifecycle from Phase 0 through the zero-label-at-inference confirmation while
retaining VEATIC-owned targets, artifacts, fitted values, and numeric decisions.

## Ability-specific endpoints

### Label-assisted spike/event ranking

- Primary endpoint: raw pooled PR-AUC, directly comparable to the established Neural Bridge
  event evidence.
- Cross-prevalence companion: average-precision skill above analytic chance. This is used to
  compare target quantiles with different prevalence; it does not replace raw PR-AUC.
- Reported secondary evidence: analytic chance, top-1/5/10% event recall, Brier score,
  defined-only per-video PR-AUC, fold/seed consistency, and paired whole-video uncertainty.
- A credible development result requires positive aggregate and paired-median gains over both
  exact frozen AR and the strongest matched control, at least four of five positive fold
  means, a nonpositive label-permutation residual gain over AR, and positive paired
  video-cluster lower confidence bounds. Five of five folds is reported as strong, not made an
  automatic prerequisite for exploratory development.

### Continuous future-movement ranking

- Primary endpoints, when this separate ability is authorized: Spearman ranking and top-5%
  true-movement lift.
- Report top-1% and top-10% lift, bias, MAE, and RMSE as diagnostics. Exact-value candidacy is
  evaluated separately and cannot be inferred from ranking/lift alone.
- Continuous work remains closed until the spike ability passes its own controls and freeze.

### Zero-label-at-inference

- Zero-label means no observed arousal, response history, teacher score, or labeled warm start
  at held-out inference; training and selection remain supervised.
- Required endpoint stack: continuous Spearman, top-5% movement lift, and event PR-AUC on both
  full-video and separately reported cold-start slices.
- Required controls are adapted to VEATIC: current-row video, diagnostics-only, no-video with
  only permitted causal metadata, deterministic whole-video input-donor shuffle, whole-video
  hard-label donor permutation, and an observed-arousal-assisted teacher ceiling reported as a
  diagnostic rather than an inference input.
- Zero-label whole-video donor controls are distinct from the within-video temporal controls
  used for the current label-assisted spike experiment. They must not be conflated.

## Permanent control rule

Every learned candidate is registered and trained control-complete from its first cell. The
same targets, rows, thresholds, fold-owned transforms, frozen baseline, seeds, recipe,
checkpoint policy, and evaluation mode are shared across the real and control lanes. Only the
declared controlled factor changes. No real-only pilot may authorize a later control backfill,
stability expansion, promotion, or confirmation.

## Current q900 matrix

The q900 current-innovation matrix uses raw PR-AUC as primary, reports the complete spike
metric stack above, and applies a VEATIC-owned paired video-cluster bootstrap with `2,048`
deterministic resamples. Its matched controls are current-innovation versions of shuffled,
random, causal-prefix video mean, diagnostics-only, and label permutation, plus an uncentered
current-row ablation and exact frozen AR. Stability remains closed unless every registered
gate passes.
