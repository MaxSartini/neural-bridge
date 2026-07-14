# Phase 5 Selected-Head 420-Row Confirmation Plan

## Status

Approved next task. The constituent blocked and grouped evaluations already exist, but the unified 420-row confirmation artifact has not yet been assembled, audited, or promoted.

## Purpose

Consolidate the already-confirmed AGAIN redesigned washout-gap binary target/head into one bounded, deterministic confirmation spanning strict blocked temporal validation and grouped held-out-video compatibility. Reuse valid scored rows and rerun only a missing or provenance-incompatible slice.

This is a selected-head confirmation, not a new model search:

- target: `future_arousal_max_delta_rows_4_10_train_q90`
- head: `short_temporal_conv_residual`
- feature path: fold-safe `temporal_mean_2s_then_pca256`
- frozen floor: matched seed- or fold/seed-specific frozen AR

## Exact Matrix

Protocols:

- `blocked_temporal_70_30`: one blocked split
- `grouped_video`: folds `1,2,3,4,5`

Seeds:

- `20260625` through `20260634` inclusive

Seven matched lanes per protocol block and seed:

1. frozen AR only
2. real residual
3. shuffled-PCA residual
4. random-PCA residual
5. label-permutation residual
6. train-only video-mean residual
7. diagnostics-only residual

Row accounting:

- blocked: `1 split x 10 seeds x 7 lanes = 70`
- grouped: `5 folds x 10 seeds x 7 lanes = 350`
- selected-head total: `420` scored rows

## Historical 504 Distinction

The earlier literal 504 design was a development-stage matrix:

- blocked: `1 x 3 seeds x 4 exploratory variants x 7 lanes = 84`
- grouped: `5 x 3 seeds x 4 exploratory variants x 7 lanes = 420`
- total: `504`

That design used the older target/design stage and four exploratory residual variants. It is not the current selected-head confirmation and its 84 blocked rows are not missing from the current matrix. Do not add obsolete variants or synthetic audit rows to make the current row count equal 504. A ten-seed recreation of that old four-variant breadth would be 1,680 rows and is not authorized by this plan.

## Canonical Inputs

- blocked report: `reports/again_dense_2hz_phase5_temporal_residual_binary_big_confirm_20260630_025437.md`
- blocked snapshot: `evidence/phase_5_5_binary_blocked_confirmation_20260630_025437/`
- grouped report: `reports/again_dense_2hz_phase5_temporal_residual_grouped_compat_20260630_033520_UPDATED_VERDICT.md`
- grouped snapshot: `evidence/phase_5_5_grouped_compatibility_20260630_033520/`
- reviewer dossier: `evidence/current_phase_5_5_review/`

## Required Procedure

1. Inventory the existing blocked and grouped scored rows and map their lane labels into one explicit schema without altering scores.
2. Validate exactly 70 unique blocked rows and 350 unique grouped rows, with no missing or duplicate protocol/fold/seed/lane keys.
3. Verify the target, head, target-window policy, split manifests, seed set, lane semantics, eval-mode scoring, checkpoint restoration, and fold-safe train-only PCA provenance.
4. Verify frozen-AR train/test checksum identity across all matched lanes within every blocked seed and grouped fold/seed.
5. Verify label-permutation, train-only video-mean, shuffled-PCA, random-PCA, and diagnostics-only policies from executable artifacts rather than names alone.
6. Preserve the updated frozen-AR-residual-aware grouped label-permutation interpretation. Do not restore the superseded raw-prevalence-near-chance gate.
7. If a row is missing or incompatible, write the discrepancy before rerunning only that exact slice. Do not silently substitute, widen the target/head matrix, or rerun valid rows.
8. Produce deterministic row, checksum, provenance, gate, failure-reason, and adversarial-verdict artifacts.

## Overall Gate

Overall selected-head confirmation passes only when:

- matrix completeness is exactly `420/420`
- all provenance and integrity checks pass
- the canonical blocked confirmation remains passing
- the updated canonical grouped compatibility verdict remains passing
- there are no unresolved failed gates or incompatible reused rows

The assembler must not change scientific thresholds after reading results. It must distinguish reused rows from newly scored rows and report all reruns.

## Deliverables

Create bounded names such as:

- runner/assembler: `backend/scripts/assemble_again_dense_2hz_phase5_selected_head_420_confirmation.py`
- deterministic contract tests for matrix completeness, uniqueness, checksums, provenance, and gate composition
- report: `reports/again_dense_2hz_phase5_selected_head_420_confirmation_<timestamp>.md`
- output root: `outputs/again_dense_2hz_phase5_selected_head_420_confirmation_<timestamp>/`
- evidence snapshot: `evidence/phase_5_5_selected_head_420_confirmation_<timestamp>/`

The evidence snapshot must include the 420-row manifest, protocol summaries, fold/seed deltas, reuse/rerun accounting, artifact checksums, gate JSON, failure-reason JSON, adversarial verdict, and a reviewer README.

## Promotion Boundary

If every gate passes, the bounded promotion wording is:

> Neural Bridge passes a full bounded 420-row selected-head confirmation for future arousal event ranking on AGAIN across strict blocked temporal validation and grouped held-out-video compatibility, using the redesigned washout-gap target, short temporal convolution residual, matched frozen AR, and matched controls.

Do not call this a 504 run. Do not claim exact continuous-value forecasting, blocked continuous generalization, broad all-target/all-dataset prediction, or universal temporal generalization.

## Stop Condition

Stop after the consolidated artifact, adversarial review, tests, canonical documentation update, commit/push, and continuity refresh. Do not begin continuous-model development or an exploratory architecture/target sweep in the same task.
