# Phase 6 Optuna Selected-Head Pilot Plan

## Purpose

Measure Optuna's marginal value on the exact proven AGAIN selected-head setup before approving a larger model-development campaign.

This is not a new architecture or target search. It holds fixed:

- target: `future_arousal_max_delta_rows_4_10_train_q90`
- head: `short_temporal_conv_residual`
- feature: fold-safe `temporal_mean_2s_then_pca256`
- protocol: `blocked_temporal_70_30`
- seed: `20260625`
- matched seed-specific frozen AR

## Study Contract

- Run 16 seeded TPE trials on MLX GPU.
- Enqueue the exact original configuration as trial zero.
- Tune only hidden width, learning rate, weight decay, residual alpha initialization/cap, gate bias, and binary-loss weight.
- Optimize only inner-validation PR-AUC delta versus frozen AR.
- Do not read or score held-out test arrays during the study.
- Lock the best Optuna trial before held-out scoring.

## Held-Out Comparison

After locking, score exactly once on the existing blocked held-out rows:

1. frozen AR
2. freshly reproduced original real residual
3. Optuna-tuned real residual
4. Optuna-tuned shuffled-PCA residual
5. Optuna-tuned random-PCA residual
6. Optuna-tuned label-permutation residual
7. Optuna-tuned train-only-video-mean residual
8. Optuna-tuned diagnostics-only residual

The pilot is promising only when tuned real exceeds the fresh original, frozen AR, and best matched control by at least `+0.001` PR-AUC. The result remains exploratory because it uses one seed and a previously reported held-out split.

## Boundaries

- Do not rerun V-JEPA, TRIBE, PCA, grouped compatibility, or the 420 matrix.
- Do not change the canonical claim or evidence bundle.
- A positive result authorizes only a bounded multi-seed Optuna follow-up.
- A negative result means Optuna did not add enough value in this narrow search; it does not invalidate the existing 420 result.
