# Phase 6 Robust Optuna 720-Row Plan

## Status

Completed through Stage B and stopped fail-closed before Stage C on `2026-07-14`. The original 24-trial MLX Stage A study completed in `532.38 s` without reading blocked held-out or grouped test scores. Best trial 22 stayed close to the canonical original configuration and won `4/5` reserved inner-validation seed comparisons, but its mean delta improvement was only `+0.0000082124` and its worst-seed/lower-quartile robust objective was worse by `-0.0004650065`. It therefore failed the original prespecified `+0.001` robust-objective gain gate.

Stages B and C were not authorized or run by the original gate. A post-hoc development sensitivity excluding the known seed-`20260627` favorable-original outlier changed the preferred configuration to trial 4. Fixed trial 4 then beat the original on all `5/5` original reserved inner-validation seeds and improved the robust objective by `+0.0048272773`. Because its selection was post hoc, that result cannot rescue Stage A directly.

The explicit Stage A2 rescue passed. Locked trial 4 beat original on `4/5` entirely new inner-validation seeds, improved mean delta-vs-AR by `+0.0021797542`, and improved the robust objective by `+0.0049870786`; failed gates were `[]`. No blocked held-out or grouped score was read. Stage B is authorized. Seed `20260627` remains in the later 15-seed held-out matrix as a predesignated stress-test; it is not deleted.

Stage B completed `120/120` rows in `778.51 s` on MLX and failed the preregistered fresh-seed/dominance gates. Candidate-minus-original was `+0.0000512741` on all-seed mean, `+0.0002953952` on all-seed median, and positive in `10/15`; the stable-14 panel passed with `+0.0003639768` mean and `10/14` wins. The untouched fresh-five panel had positive median but negative mean (`-0.0013756950`) and only `3/5` wins, dominated by a favorable original seed-`20260636` result. Candidate still beat AR and best matched controls in `15/15`, with mean deltas `+0.0074160357` and `+0.0085749406`. Stage B pass was false, so Stage C's `600` grouped rows were not run.

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

Reuse provenance-compatible canonical rows/caches for the original 10 seeds and train the five new seed replications. Report mean, median, trimmed mean, lower quartile, paired sign consistency, dispersion, leave-one-seed-out sensitivity, and seed/checkpoint-curve audits.

Seed `20260627` is designated before Stage B scoring as a known original-checkpoint-instability stress stratum, based on its already-stored original curve and before trial 4 sees its held-out score. It remains in the 15-seed matrix, all-15 summaries, controls, and report, but does not determine the primary stable-panel mean. Promotion requires: positive mean/median and at least `4/5` wins on the entirely fresh seeds `20260635`–`20260639`; positive mean/median and at least `9/14` wins on the stable panel excluding only the predesignated stress seed; positive all-15 median and at least `10/15` all-seed wins; plus the controlled AR/control gates. This is stratified reporting, not silent deletion.

Continue to grouped evaluation only if the fresh-five, stable-14, all-15 consistency, controlled AR/control, and leakage/provenance/runtime gates above all pass.

## Stage C — Grouped 15-Seed Gate

Score 5 folds x 15 seeds x 8 lanes = `600` rows under the same lane definitions and one locked configuration. Reuse only provenance-compatible grouped originals/AR caches; train all missing five-seed rows. Require fold-seed consistency, controlled deltas over matched fold/seed AR and controls, and no single fold/seed dominance.

## Full Matrix

- blocked: `120` rows
- grouped: `600` rows
- total: `720` rows

This would have been a new Phase 6 result, not an extension or reinterpretation of the canonical 420. Stage B failed, so the campaign stopped at `120` new blocked rows and is recorded as a valid negative result; do not claim 720.

## Claim Boundary

Even a full pass supports only a stronger controlled selected-head result for the same target and protocols. It does not prove exact continuous arousal values, blocked continuous generalization, universal emotion prediction, or a historical 504 design.
