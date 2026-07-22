# VEATIC 2.1 Fresh Foundation

Status: active, non-promotable foundation work as of 2026-07-21.

## Boundary

- Runtime code lives only in `src/neural_bridge/veatic21/` and imports no AGAIN package.
- Inputs are the registered VEATIC-only V-JEPA 2.1 and TRIBE v2 compact caches.
- The loader recomputes each cache's SHA-256 tree, file count, and byte size before use;
  registry declarations alone are not accepted as payload identity.
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
- Primary candidates are runtime-limited to the three canonical V-JEPA/TRIBE
  representations; diagnostics-only remains a control and cannot be selected or exported.
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

## Verified smoke

The canonical one-target/fold/seed smoke at
`artifacts/runs/veatic-2.1/foundation-smoke/20260720T232507Z` intentionally paused after
prediction sealing, resumed, completed all seven lanes, passed 13/13 audit gates, and
passed independent model replay and metric recomputation. It scored 3,962 held-out
target-supported eligible rows, including 409 pooled negatives from four zero-event videos.
The primary did not clear the strongest matched control in this plumbing run; that is a
non-claiming diagnostic, not a selection result. The run is permanently
`promotable=false`; its candidate choice and scores cannot select a scientific recipe.

Foundation promotion fails closed. Caller flags cannot make cells or recipes promotable,
and the all-124 refit API refuses before reading data or writing output. Enable it only
after a preregistered aggregate gate has a sealed, independently verifiable contract. No
canonical all-124 refit has been run.

## Exact next action

The label-blind per-video 70/30 split, train-prefix target calibration, compact linear
diagnostic, and reusable five-fold 512-component cortical PCA cache are complete. Continue
with the two-stage programme in
[`veatic21-event-preregistration.md`](veatic21-event-preregistration.md): first prove that a
fresh VEATIC video residual adds information beyond a fresh frozen VEATIC AR model, then
freeze that discovery before training zero-label candidates. Fit every projection and head
from first-70% rows in its current inner fold and keep every last-30% label closed until both
stages freeze. Once the single benchmark opening completes, refit every fitted object from
scratch on all 19,734 usable rows without reporting that refit as benchmark evidence.
