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
- Lifecycle boundary: Phases 00 and 01 concluded and sealed; pre-Phase-02.
- Current Phase 00 implementation: complete.
- Current Phase 00 execution: PASS, 27/27 mandatory controls.
- Current Phase 01 implementation: complete.
- Current Phase 01 execution: PASS, 20/20 mandatory controls.
- Current promotable VEATIC result: none.
- Authorized phase: Phase 02 fresh target-specific AR baseline only.
- Phase 03 cortical benchmarking, PCA, learned bridge work, and all later phases remain
  unauthorized until the Phase 02 gate passes.

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

## Concluded Phase 01 evidence

External root:

`/Volumes/onn. Drive/Neural Bridge Artifacts/runs/veatic-2.1/again-method-restart-20260723/phase-01-label-alignment`

- Result SHA-256: `31e6933c4d7a2b6ed077d9ae57b4e667c7957286aa0c4c97ff07c56801fb5539`.
- Artifact-manifest SHA-256:
  `df86ff4f83b6f55397e4532db5121059bc8b4d40a14adc324a7e9117c091f099`.
- Checksums-file SHA-256:
  `dcc551a583c042a167c902afeac07c3be73b59d44e04313c8aedd13c50c801de`.
- Target-substrate file SHA-256:
  `50dfa45bb3a063e88e9334c8cc9e57a9b2353a809d00298ce4d137cc3d8159af`.
- VEATIC code SHA-256: `c0a6c781bb3ab0cdf530708d4fd114d6dba4a93884d03c0833069791f018d639`.
- Alignment SHA-256: `349eceb1635fd50863ab9c6bb627fa6471dd3914a4035abea2392eee45bf57b7`.
- Target-source SHA-256:
  `ad8b167dff44ae6a0c1c78ef3e501cc622e6320be9a912d879c3d9fc99863a4f`.
- Mask SHA-256: `2fe43426a67e2e4d39382b09ed5a812fbe966f0ce5ddb61adf7e901a053b2f43`.
- Row-ownership SHA-256:
  `69676e189414a85433ebfd87966684f2353fa69a2ac6a1cd801015a424cf13cd`.
- Substrate-arrays SHA-256:
  `ce4acca4b2b72320bf224ac057342be34f27c4ea713f2a7f5eed97d3f0125088`.
- Selected initial target: future maximum increase over `t+1..t+6` (0.5–3.0 seconds),
  19,913 valid rows, 96.3983% complete-table coverage.
- Prospective washout candidates: `t+5..t+10` and `t+6..t+11`; inactive and unselected.
- Global binary label stored: false. Outer split created: false.
- Hidden state loaded/hashed: false/false. Cortical values loaded: false.
- Focused VEATIC tests: 45 passed, 0 failed.
- Compact record: `studies/veatic-2.1/phase-01-label-alignment`.

## Active execution contract

Implement Phase 02 exactly from:

- `veatic21-master-scientific-specification.md` → **Phase 02 through zero-label execution
  sequence**, the Phase 02 method-transfer evidence, ownership rules, and metric contract;
- `veatic21-rebuild-protocol.md` → **Phase 02 — target-specific fresh AR floor**.

Use the sealed Phase 01 continuous target `t+1..t+6` and validity mask. Build fresh
target-specific AR baselines under separate grouped-video and blocked-temporal 70/30 outer
protocols. Fit event q90 only from each outer-training partition's continuous target values.
Select every AR lag/regularization choice by inner validation owned by the corresponding
outer-training partition; no held-out row may influence a fitted choice.

Report grouped and blocked protocols separately using the spike metric contract. Freeze exact
target/protocol/fold/seed AR predictions and checksums for later matched lanes. Do not read or
load cortical values, fit PCA, benchmark a cortical representation, activate a washout, or
begin learned bridge work. AGAIN AR lags, regularization, splits, seeds, predictions, fitted
objects, numeric results, and code remain forbidden by inheritance.

All AR training is learned training and therefore uses MLX with exactly one GPU worker and no
artificial memory cap. CPU remains limited to parsing, deterministic audits, orchestration,
metrics, hashing, and report generation.

## Progression and handoff rule

When Phase 02 completes:

1. inspect every compact and external Phase 02 output;
2. run all focused VEATIC and authority-contract tests;
3. create the compact defensible study record under
   `studies/veatic-2.1/phase-02-ar-baseline`;
4. replace this file with the new live state while retaining **Mandatory authority anchors**;
5. record exact code/input/output hashes and the single newly authorized action;
6. commit and push the coherent transition directly to `origin/main`;
7. begin Phase 03 only after the Phase 02 gate passes and the transition is on remote
   `main`.

Do not rewrite the master specification merely because progress changed. Amend it only for an
explicitly authorized durable method change.

## Exact next action

Implement, test, execute, and review the Phase 02 fresh target-specific AR floor under
separate grouped-video and blocked-temporal protocols. Do not load cortical values, fit PCA,
activate a washout, or begin Phase 03 or learned bridge work until the Phase 02 gate passes.
