# Current State

Authority date: 2026-07-22. This file is the compact handoff for the rebuild.

## Current result

- This repository is the canonical private Neural Bridge `main`; the pre-rebuild history is preserved at `archive/pre-rebuild-20260718`.
- `src/neural_bridge/again/` is the single supported AGAIN implementation for CPU, CUDA, and MLX.
- Phase 5 and both Phase 7 checkpoint replays match published rows below `6e-8`.
- Phase 7 delivered the claim-bearing grouped held-out-video result: `420/420` declared evaluation cells and `15/15` positive fold-checkpoint groups. The separately scoped blocked-temporal protocol did not pass its distinct gate; the two protocols remain independent evidence.
- `src/neural_bridge/zero_label/` recomputes the locked 140-row verdict, validates the no-label inference audits and prediction seal, and hash-verifies the registered 95-file external run.
- `src/neural_bridge/veatic21/` now implements the fresh 124-video/20,657-row foundation and the paper-correct first-70%/last-30% temporal benchmark inside every video. The 923 black/high-duplicate rows are excluded before splitting, leaving 13,753 train-prefix and 5,981 sealed future-tail rows. Train-prefix calibration independently produced 1/3/5-second movement milestones and 18 supported 5--20% prevalence targets. No learned model has been selected.
- The verified five-fold cortical PCA cache is complete (`337M`, manifest `f2eaf699…61739`). Its 80/90/95/99% widths are 8/20/57--59/179--189; 512 components retain 99.978--99.979%. VEATIC validation—not explained variance—will choose among fixed 64/128/256/512 and variance-derived prefixes.
- Event preregistration/calibration v12 freeze the label-assisted → zero-label sequence, fixed paired seeds, exact cache reuse, checkpoint eligibility from epoch 1, and a minimum 50 epochs before convergence stopping. No AGAIN or original-VEATIC fitted object or numeric model configuration transfers.
- The locked zero-label result passed all three declared tiers on 299 untouched videos. Stage A selected direct-supervised temporal video learning; a separate prospective plan then promoted that fixed lane to the locked pass.
- Evidence-facing documentation now presents the complete ten-stage Original VEATIC-to-AGAIN journey. Every phase closure records its question, design, decisive evidence, rejected branches, claim boundary, audit trail, and transition; front-facing results no longer reduce the programme to Phase 7 and zero-label.
- Supported checks pass: Ruff, ty, and the supported tests. Archived reproduction snapshots are evidence, not a linted current API.

## Scientific boundary

Build VEATIC 2.1 from fresh 2.1 data using end-of-AGAIN rigor. OG VEATIC, the earlier VEATIC 2.1 attempt, and AGAIN contribute hypotheses and failure lessons only. They contribute no fitted PCA, tensors, labels, models, thresholds, checkpoints, inputs, or exact recipe.

The VEATIC 2.1 programme must use fresh V-JEPA 2.1/TRIBE features, every eligible dense 2 Hz row, the current quality mask, valid negatives from zero-event videos, target-specific frozen AR, fold-safe feature fitting, and VEATIC-specific causal temporal representations. It must carry matched within-video shuffled, causal-prefix video-mean, diagnostic, and within-video circular label-permutation controls plus a separate seeded-uniform chance diagnostic; keep discovery, candidate freeze, fresh confirmation, and outer closure separate; keep blocked-temporal and held-out-video evidence distinct; and declare checkpoint ensembles prospectively.

The old linear ridge is only a freshly retrained sanity baseline. PCA widths, windows, AGAIN heads, and Optuna selections are candidate priors, not inherited truths. Stage 1 first tests whether a fresh bounded/gated video residual adds meaningful value beyond fresh frozen VEATIC AR. A fold/seed can return unchanged AR only through inner-validation selection, never by inspecting row outcomes. Stage 2 begins zero-label conversion only after Stage 1 freezes. Optuna starts only after the VEATIC-specific target, representation, and head family is established.

PCA alone is intentionally label-blind and reusable. Targets, AR, learned residuals, supervised bottlenecks, heads, and checkpoint selection use training/validation labels inside their declared ownership. Exact-decimal affect forecasting is neither possible nor the promotion objective; ranking, timing, direction, meaningful magnitude bands, tail retrieval, continuous association, and calibration are.

VEATIC's 124 videos are small relative to AGAIN's 995. Benchmark fitting uses only the usable first-70% rows from all 124 and uncertainty is clustered by video; after the sealed future-tail test freezes the recipe, production PCA, scalers, thresholds, and head are refitted from scratch on all 19,734 usable rows. That all-data refit is not benchmark evidence. Larger future film/ad datasets then add coverage to the same shared production model.

## Exact next action

Seal the child training plan from `event-spike-v1-calibration.json` and `cortical-pca-v1/manifest.json`. No neural-memory result exists yet; do not search for one—run one bounded local capacity/batch probe. Then run Stage-1 label-assisted residual/head benchmarking against frozen VEATIC AR and matched controls. Freeze the winning target, representation, and head before Stage-2 zero-label conversion. Keep every last-30% label closed until both stages freeze and are evaluated once. Refit the frozen recipe on all 19,734 usable rows, apply the discovery method—not fitted artifacts—to continuous arousal and valence, then train the fresh VEATIC+AGAIN production generalist and test unseen ads/client-style video.

Canonical detail: [`internal/active/veatic21-event-preregistration.md`](../../internal/active/veatic21-event-preregistration.md), [`internal/active/veatic21-foundation.md`](../../internal/active/veatic21-foundation.md), and [`internal/migration/programme-map.md`](../../internal/migration/programme-map.md).
