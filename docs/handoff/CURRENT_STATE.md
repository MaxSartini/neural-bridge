# Current State

Authority date: 2026-07-18. This file is the compact handoff for the rebuild.

## Current result

- The original checkout remains clean. This private rebuild branch is the canonical migration candidate; existing `main` remains untouched until reviewed.
- `src/neural_bridge/again/` is the single supported AGAIN implementation for CPU, CUDA, and MLX.
- Phase 5 and both Phase 7 checkpoint replays match published rows below `6e-8`.
- The Phase 7 blocked protocol failed its scientific gate; the grouped held-out-video protocol passed. Their targets and evidence remain distinct.
- `src/neural_bridge/zero_label/` recomputes the locked 140-row verdict, validates the no-label inference audits and prediction seal, and hash-verifies the registered 95-file external run.
- The locked zero-label result passed all three declared tiers on 299 untouched videos. Stage A remains a documented failed promotion gate.
- Supported checks pass: Ruff, ty, and 11 tests. Archived reproduction snapshots are evidence, not a linted current API.

## Scientific boundary

Build VEATIC 2.1 from fresh 2.1 data using end-of-AGAIN rigor. OG VEATIC and AGAIN contribute hypotheses only: temporal change may matter more than raw state, simple heads deserve a baseline, and event signal should be established before continuous specialization. They contribute no fitted PCA, tensors, labels, models, thresholds, checkpoints, or exact recipe.

The VEATIC 2.1 programme must use fresh V-JEPA 2.1/TRIBE features, every eligible dense 2 Hz row, the current quality mask, valid negatives from zero-event videos, target-specific frozen AR, fold-safe feature fitting, and VEATIC-specific causal temporal representations. It must carry matched shuffled, random, video-mean, diagnostic, and label-permutation controls; keep discovery, candidate freeze, fresh confirmation, and outer closure separate; keep blocked-temporal and held-out-video evidence distinct; and declare checkpoint ensembles prospectively.

The old linear ridge is only a freshly retrained sanity baseline. PCA widths, windows, AGAIN heads, and Optuna selections are candidate priors, not inherited truths. Optuna starts only after the VEATIC-specific target, representation, and head family is established.

## Exact next action

Audit the current VEATIC 2.1 foundation and migrate only coherent contracts for data, quality, targets, controls, fold-safe fitting, and evaluation into `src/neural_bridge/veatic21/`. Do not launch training or tuning during that audit. Produce a small dependency map and tests that prove no AGAIN fitted artifact or locked pool crosses the boundary; then define the first fresh event discovery matrix.

Canonical detail: [`internal/migration/programme-map.md`](../../internal/migration/programme-map.md) and [`studies/again/zero-label/README.md`](../../studies/again/zero-label/README.md).
