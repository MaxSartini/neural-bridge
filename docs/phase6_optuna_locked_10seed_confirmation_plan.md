# Phase 6 Optuna Locked-Winner 10-Seed Confirmation Plan

## Status

Completed on `2026-07-14` in `515.80 s` on MLX `Device(gpu, 0)`. The locked configuration beat the canonical original in `7/10` seeds and `6/9` follow-up seeds, but mean tuned-minus-original PR-AUC was `-0.0011081356` across all seeds and `-0.0014666488` across the nine follow-up seeds. The prespecified locked-improvement verdict therefore failed. Tuned still beat frozen AR and the best tuned matched control by `+0.0057318043` and `+0.0067636509`, respectively, with `8/10` positive seeds versus both.

The failure was dominated by seed `20260627`, where the canonical original exceeded tuned by `+0.0178629568`. Its stored original training curve confirms an unusually favorable inner-validation peak: `+0.0272760719` at epoch 14, about `73%` above the original runs' median best peak, followed by decline. A post-hoc 80-epoch convergence diagnostic reproduced the tuned score exactly and selected epoch 38 again, ruling out the 40-epoch ceiling as the explanation. This diagnostic explains the outlier but cannot change the preregistered verdict.

## Purpose

Test whether the single Optuna configuration locked by the seed-`20260625` pilot improves the already-proven selected head across the canonical blocked seeds.

This is confirmation, not optimization. The run fixes:

- target: `future_arousal_max_delta_rows_4_10_train_q90`
- architecture: `short_temporal_conv_residual`
- feature: fold-safe `temporal_mean_2s_then_pca256`
- protocol: `blocked_temporal_70_30`
- seeds: `20260625` through `20260634`
- locked pilot trial: `15`
- locked parameters: hidden `64`, learning rate `0.00010528366155183298`, weight decay `0.00020452569809101856`, alpha initial logit `-3.0`, alpha cap `0.16`, gate bias `5.0`, binary-loss weight `0.8`

## Provenance Contract

- Load the locked winner verbatim from the completed pilot manifest and verify its SHA-256 checksum.
- Reuse the exact canonical seed-specific frozen-AR train/test score caches for all 10 seeds.
- Verify each reused held-out frozen-AR checksum against the canonical 70-row confirmation table.
- Reuse the canonical original real/control rows for comparison; do not retrain the original configuration.
- Train only the locked tuned real residual and its five matched residual controls.
- No per-seed tuning, Optuna study, target/head search, V-JEPA/TRIBE run, PCA fit/refit, grouped run, 420 rerun, 504 reconstruction, or continuous-model work.

## Evaluation Sets

The full 10-seed result is reported for continuity. The primary follow-up evaluation uses the nine seeds `20260626` through `20260634`, because seed `20260625` was already used for the pilot authorization decision.

## Prespecified Gates

The locked Optuna improvement passes only if all are true:

1. On the nine follow-up seeds, mean tuned-minus-original PR-AUC is at least `+0.001`.
2. On the nine follow-up seeds, tuned beats original in at least `6/9` seeds.
3. On all 10 seeds, mean tuned-minus-original PR-AUC is positive.
4. On all 10 seeds, tuned beats original in at least `7/10` seeds.
5. On all 10 seeds, tuned exceeds frozen AR and every primary matched control on mean PR-AUC.
6. On all 10 seeds, tuned beats frozen AR and the best per-seed matched control in at least `8/10` seeds.
7. No single seed contributes more than `40%` of the summed positive tuned-minus-original improvement.
8. Frozen-AR checksum, checkpoint restore/eval mode, causal-context, label-permutation, train-only-video-mean, and locked-parameter provenance audits all pass.

The result may justify a later grouped locked-winner confirmation only after review. It does not alter the canonical 420-row claim by itself.

## Failure Interpretation

A failure means the narrow Optuna improvement did not replicate strongly enough across the locked blocked-seed confirmation. It does not invalidate the existing selected-head result or the canonical 420-row confirmation.
