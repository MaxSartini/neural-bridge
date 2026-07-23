# Current State — VEATIC 2.1 AGAIN-Method Rebuild

Updated: 2026-07-23

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
  input boundary, phase-by-phase method, controls, metrics, washout design, provenance rules,
  and implementation contracts.
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
- Master scientific specification: version 1.0.
- Repository: `/Users/maxsartini/Neural Bridge`.
- Branch: `main` only; do not create a branch.
- Lifecycle boundary: clean pre-Phase-00.
- Current Phase 00 implementation: not implemented.
- Current Phase 00 execution: not run.
- Current promotable VEATIC result: none.
- Authorized phase: Phase 00 dense-foundation audit only.
- Phase 01 and all model work remain unauthorized until the Phase 00 gate passes.

## Canonical live inputs

Final TRIBE v2 predicted-cortical root:

`/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/tribe-v2/compact-20260716`

The only real Neural Bridge representation is:

`per_video/<video_id>/tribe_v2_cortical_predictions.npz:cortical_prediction`

Matching V-JEPA row/label/metadata root:

`/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/vjepa-2.1/compact-20260716`

Authoritative row/label file:

`<video_id>/rows.csv`

`vjepa21_hidden_states.npz` is absolutely forbidden. Do not open, inspect, load, memory-map,
copy, or hash it. V-JEPA and TRIBE are completed upstream substrate and are not rerun.

AGAIN is methodology-only. Do not import, execute, copy, adapt, or reuse AGAIN code, runners,
data, splits, targets, numeric choices, PCA, AR objects, heads, checkpoints, predictions,
controls, or fitted artifacts. Follow the master specification's method-only firewall.

## Active execution contract

Implement Phase 00 exactly from:

- `veatic21-master-scientific-specification.md` → **Phase 00 implementation contract**;
- `veatic21-rebuild-protocol.md` → **Phase 00 — dense foundation**.

Required lifecycle root:

`/Volumes/onn. Drive/Neural Bridge Artifacts/runs/veatic-2.1/again-method-restart-20260723`

Phase 00 must:

1. use fresh VEATIC-specific source and tests;
2. audit all 124 videos and all 20,657 canonical rows;
3. verify the exact layouts, identities, 2 Hz time grid, hashes, provenance, quality flags,
   and finite `[rows, 20,484]` cortical arrays specified by the master;
4. enforce the V-JEPA hidden-state prohibition before any open or hash;
5. enforce runtime rejection of AGAIN code, runner, data, output, and artifact paths;
6. emit the required `veatic-derivation-ledger.json`;
7. store heavy outputs only under the external lifecycle root;
8. keep all rows; quality flags are metadata, not silent exclusions;
9. perform no PCA, AR fitting, split selection, target thresholding, or model training;
10. pass every mandatory check and focused test before authorizing Phase 01.

CPU is appropriate for Phase 00 audit, parsing, hashing, orchestration, and reporting. Later
PCA/model training uses MLX with exactly one GPU worker and no artificial memory cap.

## Progression and handoff rule

When Phase 00 completes:

1. inspect every compact and external output;
2. run all focused VEATIC and authority-contract tests;
3. create the compact defensible study record under
   `studies/veatic-2.1/phase-00-dense-foundation`;
4. replace this file with the new live state while retaining **Mandatory authority anchors**;
5. record exact code/input/output hashes and the single newly authorized action;
6. commit and push the coherent transition directly to `origin/main`;
7. begin Phase 01 only after the Phase 00 gate passes and the transition is on remote
   `main`.

Do not rewrite the master specification merely because progress changed. Amend it only for an
explicitly authorized durable method change.

## Exact next action

Implement, test, execute, and review the new Phase 00 dense-foundation audit. Do not begin
Phase 01, PCA, AR benchmarking, target selection, or learned-model work until Phase 00 passes.
