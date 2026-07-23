# VEATIC 2.1 Fresh Foundation

Decision: foundation protocol completed and frozen on 2026-07-22.

## Boundary

- Runtime code lives only in `src/neural_bridge/veatic21/` and imports no AGAIN package.
- Inputs are the registered VEATIC-only V-JEPA 2.1 and TRIBE v2 compact caches.
- The canonical cache trees were independently rehashed on 2026-07-23 and match their
  registered SHA-256 identities, file counts, and byte sizes. Runtime loading validates the
  exact artifact IDs, complete 0–123 video inventory, per-video manifests, schemas, row plans,
  shapes, and shared V-JEPA/TRIBE metadata; it does not rehash both multi-gigabyte trees on
  every process start.
- The loader enforces 124 videos, 20,657 unique `(video_id, row_index)` rows at 2 Hz,
  and the locked 923-row black-or-high-duplicate exclusion mask.
- Fitted PCA, normalization, thresholds, heads, controls, and model-selection decisions
  are always created inside their declared VEATIC ownership.

## Foundation protocol

- Split group: `video_id` only.
- Event benchmark: first 70% / last 30% of usable rows inside every one of the 124 videos.
- Candidate selection: five deterministic video-grouped folds inside the first-70% row pool.
- Event split seed: `20260722`; assignment and manifests are content-digested.
- First plumbing target: maximum absolute future arousal movement from 1 to 3 seconds;
  the 90th-percentile event threshold is refitted on each inner or outer train scope.
- Selection metric: pooled PR-AUC. Tie-break: ascending frozen candidate name.
- Required controls: target-specific frozen AR, within-video sequence-shuffled,
  causal-prefix video-mean, diagnostics-only, and within-video circular label-permutation.
  Seeded uniform random is sealed and replayed as a chance diagnostic, not a matched control.
- The sole selectable Neural Bridge representation is the VEATIC 2.1 TRIBE v2
  `cortical_prediction`; diagnostics-only remains a control and cannot be selected or
  exported.
- The 20,484-wide cortical representation requires an explicitly declared incremental-PCA
  candidate and bounded batch size; randomized dense PCA is rejected before fitting.
- Outer predictions, row order, mask coverage, split, winner, models, code, and substrate
  are sealed before outer labels open. Resume refuses any changed dependency or seal.
- Single-class per-video PR-AUC stays undefined; eligible negatives from zero-event videos
  remain in pooled PR-AUC.

These defaults establish executable ownership and leakage contracts. They are not a final
event/spike discovery preregistration. Candidate families, gates, checkpoint ensembles,
and the held-out-video versus blocked-temporal evidence plan must be frozen separately
before claim-bearing discovery starts.

Foundation promotion fails closed. Caller flags cannot make cells or recipes promotable,
and the all-124 refit API refuses before reading data or writing output. Enable it only
after a preregistered aggregate gate has a sealed, independently verifiable contract. No
canonical all-124 refit has been run.

## Downstream authority

This record fixes the completed foundation evidence; it does not declare current work. The
sole current action is in [`internal/handoff/CURRENT_STATE.md`](../handoff/CURRENT_STATE.md),
with the registered method in
[`internal/active/veatic21-event-preregistration.md`](../active/veatic21-event-preregistration.md).
