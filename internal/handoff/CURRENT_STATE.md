# Canonical Current State

Updated: 2026-07-22

## Current gate

- VEATIC 2.1 has 124 videos and 20,657 dense 2 Hz rows; the quality mask leaves 19,734 usable rows after removing 923 black-screen, end-screen, and duplicate rows.
- Every video contributes to the temporal split: 13,753 first-70% discovery rows and 5,981 sealed last-30% rows.
- Event calibration supports 18 VEATIC-specific targets across 1 s, 3 s, and 5 s horizons.
- Five fold-safe 512-component cortical PCA bases are complete. Benchmark 64, 128, 256, and 512 components plus the registered variance-derived prefixes; select from VEATIC evidence.
- `prepare-stage1` passed and sealed `artifacts/preregistrations/veatic-2.1/stage1-child-plan.json` (`8068551599153c10df7442faec1ba20bfec581de36f06856fa658e5aee2cc77a`). MLX supports batch 1,024 for every registered hidden width.
- The active gate is the Stage-1 label-assisted temporal-head trainer over frozen VEATIC AR.

## Fixed constraints

- Fit every VEATIC target, PCA projection, normalization, head, threshold, window, and checkpoint from VEATIC training rows within its owning fold.
- Labels supervise targets, AR, heads, gates, and checkpoint selection. PCA fitting remains label-blind.
- Compare fresh bounded/gated residual heads with matched video-only and false-signal controls. Permit AR fallback only from inner-validation evidence.
- Keep last-30% labels sealed until the label-assisted and zero-label recipes freeze.
- Allow checkpoint 1 to win, require at least 50 training epochs before stopping, and continue while validation improves.

## Exact next action

Implement `train_stage1` in `src/neural_bridge/veatic21/stage1.py` and expose it as `python -m neural_bridge.veatic21 train-stage1`. It must consume the sealed child plan, run the registered fold/target/width/head/seed matrix, use `CheckpointSelector`, keep tail labels sealed, and apply whole-run AR fallback from inner validation. Acceptance: a fixed fold/seed smoke runs at least 50 epochs, can retain epoch 1, has no maximum epoch, and produces a verified checkpoint/result manifest; then launch the full matrix.

Detailed protocol: [`internal/active/veatic21-event-preregistration.md`](../active/veatic21-event-preregistration.md).
