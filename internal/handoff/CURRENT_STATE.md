# Current State — VEATIC 2.1 After Fresh Phase 00

Updated: 2026-07-28

## Mandatory authority anchors

Read these files completely, in this order, before any VEATIC scientific action:

1. `/Users/maxsartini/Neural Bridge/AGENTS.md`
2. `/Users/maxsartini/Neural Bridge/internal/active/veatic21-master-scientific-specification.md`
3. this file

Then read the sections of
`/Users/maxsartini/Neural Bridge/internal/active/veatic21-rebuild-protocol.md` named under
**Active execution contract** below. The protocol is a derived checklist; it need not be
loaded in full on every turn.

Their roles are deliberately different:

- The master scientific specification is permanent and comprehensive. It owns the durable
  input boundary, phase-by-phase method, controls, metrics, experiment-sufficiency rules,
  provenance requirements, and implementation contracts.
- The rebuild protocol is a derived execution checklist and navigation aid. It cannot add an
  independent rule or weaken or replace the master specification.
- This file is the replace-in-place live handoff. It owns current progress, result/artifact
  hashes, current authorization, blockers, and the exact next action.

Every future replacement of this file must preserve this **Mandatory authority anchors**
section and the three absolute paths above. Do not copy the master specification back into
this live handoff. Do not infer an omitted rule from this shorter file; follow the master.

If the live state appears to require a scientific-method change, stop progression. Amend the
master specification explicitly, update the rebuild protocol and this file in the same
commit, run the authority-contract tests, and push `main` before executing the changed
method. This file cannot silently override the master specification.

## Live scientific state

- Programme: VEATIC 2.1 AGAIN-method rebuild.
- Master scientific specification: version 2.0.
- Repository: `/Users/maxsartini/Neural Bridge`.
- Branch: `main` only; do not create a branch.
- Lifecycle boundary: fresh Phase 00 concluded; before Phase 01 implementation and execution.
- Phase 00 implementation: complete.
- Phase 00 execution: PASS, 27/27 mandatory controls.
- Executed modeling phases: none.
- Registered VEATIC targets, splits, AR models, projections, representations, heads,
  checkpoints, model controls, or promotion outcomes: none.
- Current promotable VEATIC result: none.
- Authorized phase: Phase 01 exact label alignment and VEATIC target-substrate construction
  only.
- Phase 02 and every AR fit, cortical benchmark, PCA, head search, washout cortical score,
  continuous, valence, and zero-label action remain unauthorized.

## Canonical live inputs

Complete TRIBE v2 per-video raw cortical-prediction root:

`/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/tribe-v2/veatic 2.1 raw cortical predictions/per_video`

Every video `0..123` has exactly one prediction payload:

`<video_id>/tribe_v2_cortical_predictions.npz`

The real representation array inside every payload is `cortical_prediction`. The phrase
"cortical predictions" means the complete collection of all 124 per-video arrays aligned to
all eligible exact 2 Hz rows. It never means one singular file, one preferred video, or an
unregistered subset.

Matching V-JEPA row/label/metadata root:

`/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/vjepa-2.1/veatic 2.1 v jepa 2.1 stuff`

Authoritative row/label/provenance file for each matching video:

`<video_id>/rows.csv`

The row identity is `(video_id, row_index, time_seconds)` on the native 2 Hz grid: exact
`0.5s` steps beginning at `0.0s`. All 20,657 rows remain present. Quality flags are metadata,
not a silent exclusion.

Allowed V-JEPA companion inputs are `rows.csv`, `manifest.json`, `preprocessing.json`,
`status.json`, `_PAYLOAD_SHA256.json`, and `_UPLOAD_COMPLETE.json` in every video directory.
Every `vjepa21_hidden_states.npz` remains absolutely forbidden: do not open, inspect, load,
memory-map, copy, or hash it. V-JEPA and TRIBE are completed upstream substrate and are not
rerun.

AGAIN is methodology-only. Do not import, execute, copy, adapt, or reuse AGAIN code, runners,
data, splits, targets, numeric choices, PCA, AR objects, heads, checkpoints, predictions,
controls, or fitted artifacts. Every VEATIC choice and fitted object is fresh.

## Concluded Phase 00 evidence

External root:

`/Volumes/onn. Drive/Neural Bridge Artifacts/runs/veatic-2.1/fresh-method-rebuild-20260728/phase-00-dense-foundation`

- Status: PASS, 27/27 mandatory controls.
- Result SHA-256: `76667bc439af70b4ed212fe114922f0453415280fb64acf2910955d688333ffb`.
- Artifact-manifest SHA-256:
  `7a5e8dab2d442536eadd8b0d23491333c43a45355e74b361402c86abb7cc7e0e`.
- Checksums-file SHA-256:
  `fea767403cf56919697aa228eb9053587d268cbe8a3f56143b7f08dac359ea8c`.
- VEATIC code SHA-256:
  `190425435530febac3723268b858a48f6325b9fcad5095e5e6d50b42aa36a879`.
- Input-identity SHA-256:
  `da59601575403b5d5becdf98c4d348adaa324c5f99c92eabefdcc49b31d569b4`.
- TRIBE per-video tree SHA-256:
  `851d55ccaac7c587495f65cdfbfbcf6bfe22a66a7ab3da2a048d0422e4087a60`.
- V-JEPA metadata-only tree SHA-256:
  `cee65f87ff1e118353acd0c6f86c7f8c925e4e612b47884caea0544f6250e1cd`.
- Audited coverage: all 124 prediction payloads and all 20,657 exact 2 Hz rows.
- Cortical layout: `[per_video_rows, 20,484]`, float16, finite in every video.
- Quality counts: 76 black, 871 duplicate/static, 24 both, 923 union, 19,734 unflagged;
  all 20,657 rows retained.
- V-JEPA hidden state loaded/hashed/copied/inspected: false/false/false/false.
- AGAIN runtime import, execution, data, or artifact use: none.
- Target/split/PCA/AR/model/head/washout operations: none.
- Compact record: `studies/veatic-2.1/phase-00-dense-foundation`.

## Active execution contract

Read these rebuild-protocol sections for the one authorized action:

- **Canonical input boundary**;
- **Phase 01 — exact label alignment and target substrate**;
- **Comprehensive experiment and search-sufficiency checklist**;
- **Execution and artifact rules**.

Implement Phase 01 from the fresh Phase 00 identity. Open supervised arousal and valence only
from all 124 matching V-JEPA `rows.csv` files. Reconstruct the complete 20,657-row table and
reconfirm exact `(video_id, row_index, time_seconds)` identity against Phase 00. Validate
finite label values and native interpolation provenance without shifting, smoothing,
extrapolating, or repairing labels.

Phase 01 must perform comprehensive VEATIC-specific label-only analysis before any cortical
performance is read. It calculates movement distributions, per-video autocorrelation and
partial-autocorrelation decay, causal trailing-history and slope predictiveness, rise time,
event duration, video-duration compatibility, target-window coverage, eligible rows/videos,
per-video positive support, and fold-owned threshold-stability expectations.

Freeze a bounded, justified registry of initial no-washout future-maximum-increase candidate
windows and a separate bounded VEATIC-only procedure for possible washout starts/window ends.
Candidate bounds and rejection rules come only from VEATIC label dynamics, duration,
coverage, and support. AGAIN row offsets, seconds, thresholds, targets, and fitted values are
not candidates by inheritance.

Store continuous future movement values and validity masks for every registered candidate.
Do not create one global binary label column: q90 event thresholds belong to later training
partitions. Do not create an outer evaluation split. Do not read cortical values to choose a
target, and do not fit AR, PCA, a learned model, or any control lane.

Phase 01 output root:

`/Volumes/onn. Drive/Neural Bridge Artifacts/runs/veatic-2.1/fresh-method-rebuild-20260728/phase-01-label-alignment`

## Progression and handoff rule

When and only when Phase 01 passes every alignment, provenance, candidate-registry,
coverage/support, leakage, and no-cortical-selection control:

1. inspect every compact and external output;
2. run all focused VEATIC tests, authority-contract tests, and the full repository test suite;
3. create the compact defensible record under
   `studies/veatic-2.1/phase-01-label-alignment`;
4. replace this file while retaining **Mandatory authority anchors**;
5. record exact code, input, result, artifact-manifest, and checksum hashes plus the single
   newly authorized action;
6. commit and push the coherent transition directly to `origin/main`;
7. begin Phase 02 only after the Phase 01 transition is present on remote `main`.

No later phase may claim success or failure without the master specification's comprehensive
VEATIC experiment registration, full result ledger, complete controls, and
search-sufficiency gate.

## Exact next action

Implement, test, execute, and review fresh Phase 01 exact label alignment and comprehensive
VEATIC target-substrate construction over all 20,657 exact 2 Hz rows. Produce continuous
candidate values/masks and label-only derivation evidence; do not create a global binary
target, outer split, AR/PCA/model, or cortical performance result.
