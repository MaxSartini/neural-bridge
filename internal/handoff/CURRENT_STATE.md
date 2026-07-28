# Current State — VEATIC 2.1 AGAIN-Method Rebuild

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
- Lifecycle boundary: Phase 00 concluded and sealed; pre-Phase-01.
- Current Phase 00 implementation: complete.
- Current Phase 00 execution: PASS, 27/27 mandatory controls.
- Current promotable VEATIC result: none.
- Authorized phase: Phase 01 label alignment and VEATIC target substrate only.
- Phase 02, PCA, AR benchmarking, cortical target benchmarking, and all model work remain
  unauthorized until the Phase 01 gate passes.

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

## Concluded Phase 00 evidence

External root:

`/Volumes/onn. Drive/Neural Bridge Artifacts/runs/veatic-2.1/again-method-restart-20260723/phase-00-dense-foundation`

- Result SHA-256: `e2792c8c75f80239b6687680dacba77ecc9710d4806cc9dc3351cb3611655056`.
- Artifact-manifest SHA-256:
  `a7aa9913eaa1dc7b1068719bc2405701316468fd0cd94008860490bc94361d4a`.
- Checksums-file SHA-256:
  `51bfce91aab1e7212752b0767159dfb3db015199846f3fe6aa556d80dfbf7df8`.
- VEATIC code SHA-256: `87b67fe2aa6878d703f9703d741bf0cfae33442160423ac11b78bc9a2c5c3208`.
- Input-identity SHA-256:
  `9ea8b7fb0cecdcad083e48c27027746d56be396fe6eef3a0eec2b930454414f0`.
- TRIBE tree SHA-256: `0d4adc27dd9d226de87d0cfc4df92de14cb7450de6671857e0665418ad26f6dd`.
- Allowlisted V-JEPA metadata tree SHA-256:
  `cee65f87ff1e118353acd0c6f86c7f8c925e4e612b47884caea0544f6250e1cd`.
- Audited inventory: 124 videos and 20,657 rows; all source rows retained.
- Quality counts: 76 black, 871 duplicate/static, 24 both, 923 union, 19,734 unflagged.
- Forbidden hidden state loaded/hashed/copied/inspected: false/false/false/false.
- AGAIN runtime imports or execution: none.
- Focused VEATIC tests: 34 passed, 0 failed.
- Compact record: `studies/veatic-2.1/phase-00-dense-foundation`.

## Active execution contract

Implement Phase 01 exactly from:

- `veatic21-master-scientific-specification.md` → **Phase 01 exact next-stage contract**;
- `veatic21-rebuild-protocol.md` → **Phase 01 — exact label alignment and target substrate**.

Use `rows.csv` as the sole label authority. Reconstruct and reconfirm all 20,657 aligned rows,
preserve quality flags without filtering, validate finite arousal/valence and native
interpolation provenance, and calculate the VEATIC-specific label dynamics required by the
master. Freeze the bounded initial no-washout target-window selection rule and prospective
washout candidate procedure before any cortical target result is read.

Store continuous future-movement values and masks, not one global q90 binary label. Do not
create an outer 70/30 split in Phase 01. Do not fit PCA, AR, or a learned model. AGAIN row
offsets, seconds, targets, splits, fitted artifacts, numeric choices, and results remain
forbidden by inheritance. Emit the Phase 01 derivation ledger and complete alignment,
target-source, mask, and row-ownership digests before considering Phase 02.

Heavy Phase 01 artifacts remain under the same external lifecycle root. CPU is appropriate
for label parsing and deterministic Phase 01 analysis. Later PCA/model training uses MLX with
exactly one GPU worker and no artificial memory cap.

## Progression and handoff rule

When Phase 01 completes:

1. inspect every compact and external Phase 01 output;
2. run all focused VEATIC and authority-contract tests;
3. create the compact defensible study record under
   `studies/veatic-2.1/phase-01-label-alignment`;
4. replace this file with the new live state while retaining **Mandatory authority anchors**;
5. record exact code/input/output hashes and the single newly authorized action;
6. commit and push the coherent transition directly to `origin/main`;
7. begin Phase 02 only after the Phase 01 gate passes and the transition is on remote
   `main`.

Do not rewrite the master specification merely because progress changed. Amend it only for an
explicitly authorized durable method change.

## Exact next action

Implement, test, execute, and review Phase 01 label alignment and VEATIC target-substrate
construction. Do not begin Phase 02, create an outer split, fit AR or PCA, inspect cortical
target performance, or perform learned-model work until the Phase 01 gate passes.
