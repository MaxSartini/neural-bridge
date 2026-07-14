# Phase 6 Robust Optuna 720-Row Plan

## Purpose

Build a new, larger confirmation around the already-proven selected target/head using robust multi-seed Optuna development, explicit seed-stability review, and one locked configuration. This is separate from the canonical 420-row result and cannot weaken or rewrite it.

`420` was the number of scored confirmation lanes, not the number of training examples. The existing blocked model trains on approximately `159,923` rows (`127,369` inner-train plus `32,554` inner-validation), from the frozen 243,575-row AGAIN substrate. This plan exceeds 420 through additional stochastic replications and direct original-versus-Optuna comparison, not row padding.

## Fixed Scientific Scope

- target: `future_arousal_max_delta_rows_4_10_train_q90`
- architecture: `short_temporal_conv_residual`
- feature: fold-safe `temporal_mean_2s_then_pca256`
- data: existing frozen AGAIN dense cache; no V-JEPA/TRIBE re-encoding
- blocked protocol: `blocked_temporal_70_30`
- grouped protocol: five held-out-video folds
- seeds: `20260625` through `20260639` (15 total; five new seeds)
- controls: frozen AR plus shuffled PCA, random PCA, label permutation, train-only video mean, and diagnostics-only residual controls

## Stage A — Robust Multi-Seed Optuna Development

Run 24 seeded TPE trials on MLX. Enqueue the exact canonical original configuration as trial zero.

- development seeds: `20260625` through `20260629`
- reserved inner-validation seeds: `20260630` through `20260634`
- Optuna input: inner-train and inner-validation rows only; no blocked held-out or grouped test scores
- objective: an equal blend of worst-seed and lower-quartile inner-validation delta versus each seed's exact frozen AR, with a small mean tie-break, so one favorable seed cannot dominate and one failing seed remains visible
- tunable parameters: hidden width, learning rate, weight decay, alpha initialization/cap, gate bias, binary-loss weight, maximum epochs, and patience

Lock one winner after the development-seed study. Continue only if, on the five reserved inner-validation seeds, it improves the original configuration's robust objective by at least `+0.001`, improves mean delta, and wins at least `4/5` paired seeds.

## Stage B — Blocked 15-Seed Gate

Score 15 seeds x 8 lanes = `120` rows:

1. frozen AR
2. canonical original real residual
3. locked Optuna real residual
4. locked Optuna shuffled-PCA residual
5. locked Optuna random-PCA residual
6. locked Optuna label-permutation residual
7. locked Optuna train-only-video-mean residual
8. locked Optuna diagnostics-only residual

Reuse provenance-compatible canonical rows/caches for the original 10 seeds and train the five new seed replications. Report mean, median, trimmed mean, lower quartile, paired sign consistency, dispersion, leave-one-seed-out sensitivity, and seed/checkpoint-curve audits. No seed may be silently dropped. Any outlier exclusion is diagnostic-only and cannot determine the primary verdict.

Continue to grouped evaluation only if locked Optuna beats original on mean and median, wins at least `10/15` paired seeds, retains credible deltas over AR and every matched control, and passes all leakage/provenance/runtime gates.

## Stage C — Grouped 15-Seed Gate

Score 5 folds x 15 seeds x 8 lanes = `600` rows under the same lane definitions and one locked configuration. Reuse only provenance-compatible grouped originals/AR caches; train all missing five-seed rows. Require fold-seed consistency, controlled deltas over matched fold/seed AR and controls, and no single fold/seed dominance.

## Full Matrix

- blocked: `120` rows
- grouped: `600` rows
- total: `720` rows

This is a new Phase 6 result, not an extension or reinterpretation of the canonical 420. Stage A or B failure stops the campaign early and is recorded as a valid negative result.

## Claim Boundary

Even a full pass supports only a stronger controlled selected-head result for the same target and protocols. It does not prove exact continuous arousal values, blocked continuous generalization, universal emotion prediction, or a historical 504 design.
