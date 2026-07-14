# Phase 6 Robust Optuna 720-Row Plan

## Status

Stopped fail-closed at Stage A on `2026-07-14`. The 24-trial MLX study completed in `532.38 s` without reading blocked held-out or grouped test scores. Best trial 22 stayed close to the canonical original configuration and won `4/5` reserved inner-validation seed comparisons, but its mean delta improvement was only `+0.0000082124` and its worst-seed/lower-quartile robust objective was worse by `-0.0004650065`. It therefore failed the prespecified `+0.001` robust-objective gain gate.

Stages B and C were not authorized or run by the original gate. A post-hoc development sensitivity excluding the known seed-`20260627` favorable-original outlier changed the preferred configuration to trial 4. Fixed trial 4 then beat the original on all `5/5` original reserved inner-validation seeds and improved the robust objective by `+0.0048272773`. Because its selection was post hoc, that result cannot rescue Stage A directly.

An explicit Stage A2 rescue is therefore approved: lock trial 4 unchanged and evaluate it against the original on five entirely new seeds, `20260635` through `20260639`, using inner train/validation only. Apply the same `+0.001` robust-objective, mean-improvement, and `4/5` paired-win gates. Only a Stage A2 pass authorizes Stage B. Seed `20260627` remains in the later 15-seed held-out matrix as a stress-test; it is not deleted.

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

This would have been a new Phase 6 result, not an extension or reinterpretation of the canonical 420. Stage A failed, so the campaign stopped and is recorded as a valid negative result.

## Claim Boundary

Even a full pass supports only a stronger controlled selected-head result for the same target and protocols. It does not prove exact continuous arousal values, blocked continuous generalization, universal emotion prediction, or a historical 504 design.
