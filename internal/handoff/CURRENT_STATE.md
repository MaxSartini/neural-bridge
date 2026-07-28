# Current State — VEATIC 2.1 Fresh Phase 00

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
- Lifecycle boundary: fresh Phase 00, before implementation and execution.
- Implemented phases: none.
- Executed phases: none.
- Registered VEATIC results, fitted artifacts, selected targets, AR models, representations,
  heads, checkpoints, controls, or promotion outcomes: none.
- Current promotable VEATIC result: none.
- Authorized phase: Phase 00 dense-foundation audit only.
- Phase 01 and every modeling, target-selection, PCA, AR, head-search, washout, continuous,
  valence, and zero-label action remain unauthorized.

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
`0.5s` steps beginning at `0.0s`. Phase 00 preserves every row and uses label columns only in
an isolated equality audit, never to select an outcome. Supervised labels open scientifically
only after Phase 00 authorizes Phase 01.

Allowed V-JEPA companion inputs are `rows.csv`, `manifest.json`, `preprocessing.json`,
`status.json`, `_PAYLOAD_SHA256.json`, and `_UPLOAD_COMPLETE.json` in every video directory.
Every `vjepa21_hidden_states.npz` is absolutely forbidden: do not open, inspect, load,
memory-map, copy, or hash it. V-JEPA and TRIBE are completed upstream substrate and are not
rerun.

AGAIN is methodology-only. Do not import, execute, copy, adapt, or reuse AGAIN code, runners,
data, splits, targets, numeric choices, PCA, AR objects, heads, checkpoints, predictions,
controls, or fitted artifacts. Every VEATIC choice and fitted object is fresh.

## Canonical-root preflight facts

A read-only authority-change audit of the exact roots above established the input expectation
that fresh Phase 00 must independently reproduce and seal:

- videos: `124`, exact IDs `0..123` in both roots;
- prediction payloads: `124`, one in every video directory;
- aligned rows: `20,657`, all retained;
- row rate and step: `2 Hz`, exact `0.5s`;
- cortical layout: `[per_video_rows, 20,484]`, float16, finite;
- per-video key schema: uniform across all 124 payloads;
- row counts: minimum `22`, maximum `358`;
- quality flags: 76 black, 871 duplicate/static, 24 both, 923 union;
- TRIBE per-video allowlisted tree: 373 files, 866,111,964 bytes, SHA-256
  `851d55ccaac7c587495f65cdfbfbcf6bfe22a66a7ab3da2a048d0422e4087a60`;
- V-JEPA metadata-only allowlisted tree: 744 files, 7,103,590 bytes, SHA-256
  `cee65f87ff1e118353acd0c6f86c7f8c925e4e612b47884caea0544f6250e1cd`;
- V-JEPA hidden-state files opened or hashed: false.

These are input expectations, not a Phase 00 PASS. Phase 00 must fail closed if independent
recomputation differs.

## Active execution contract

Read these rebuild-protocol sections for the one authorized action:

- **Canonical input boundary**;
- **Phase 00 — dense foundation**;
- **Execution and artifact rules**.

Implement Phase 00 from scratch under the VEATIC namespace. Do not copy or adapt any existing
phase runner. Keep feature access, row/metadata access, and later supervised-label access
structurally separate. The feature path must be unable to request label arrays as model
features, and every hidden-state path must be rejected before open, load, inspection, copy,
or hash.

Phase 00 audits all 124 TRIBE prediction payloads and all 124 matching V-JEPA row/metadata
directories. It validates exact file inventory, manifests/status, recorded payload hashes,
video IDs, complete 2 Hz row identity, row-count agreement, cortical shape/dtype/finiteness,
uniform schemas, time equality, quality flags, allowlisted tree digests, and the AGAIN runtime
firewall. It must explicitly prove that every video prediction and every canonical row was
considered.

Phase 00 performs no target construction or selection, split, PCA, AR fit, learned model,
head search, washout design, or scientific comparison.

New external lifecycle root:

`/Volumes/onn. Drive/Neural Bridge Artifacts/runs/veatic-2.1/fresh-method-rebuild-20260728`

Phase 00 output root:

`/Volumes/onn. Drive/Neural Bridge Artifacts/runs/veatic-2.1/fresh-method-rebuild-20260728/phase-00-dense-foundation`

Do not resume from or write into any other VEATIC run lifecycle.

## Progression and handoff rule

When and only when fresh Phase 00 passes every mandatory control:

1. inspect every compact and external output;
2. run all focused VEATIC tests, authority-contract tests, and the full repository test suite;
3. create the compact defensible record under
   `studies/veatic-2.1/phase-00-dense-foundation`;
4. replace this file while retaining **Mandatory authority anchors**;
5. record exact code, input, result, artifact-manifest, and checksum hashes plus the single
   newly authorized action;
6. commit and push the coherent transition directly to `origin/main`;
7. begin Phase 01 only after the Phase 00 transition is present on remote `main`.

No later phase may claim success or failure without the master specification's comprehensive
VEATIC experiment registration, full result ledger, complete controls, and
search-sufficiency gate.

## Exact next action

Implement, test, execute, and review a genuinely fresh Phase 00 dense-foundation audit over
all 124 per-video cortical prediction payloads and their matching exact 2 Hz V-JEPA row and
metadata inputs. Produce only Phase 00 evidence. Do not begin Phase 01 or any modeling action.
