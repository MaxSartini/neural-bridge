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
- Lifecycle boundary: Phases 00 through 04 concluded and sealed; pre-Phase-05.
- Current Phase 00 implementation: complete.
- Current Phase 00 execution: PASS, 27/27 mandatory controls.
- Current Phase 01 implementation: complete.
- Current Phase 01 execution: PASS, 20/20 mandatory controls.
- Current Phase 02 implementation: complete.
- Current Phase 02 execution: PASS, 24/24 mandatory controls.
- Current Phase 03 implementation: complete.
- Current Phase 03 execution: PASS, 29/29 mandatory controls; direct raw-fusion claim FAIL.
- Current Phase 04 implementation: complete.
- Current Phase 04 execution: PASS, 36/36 mandatory controls; linear PCA-fusion claim FAIL.
- Current promotable VEATIC result: none.
- Authorized phase: Phase 05 learned frozen-AR bridge only.
- Phase 06 and all later phases remain unauthorized. Washout activation remains unauthorized;
  Phase 05 must first execute the sealed no-washout residual task with complete controls.

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

## Concluded Phase 02 evidence

External root:

`/Volumes/onn. Drive/Neural Bridge Artifacts/runs/veatic-2.1/again-method-restart-20260723/phase-02-ar-baseline`

- Result SHA-256: `bf7bb7dd24432af1a6baa4f846a2f84dfcfb89c822bddfefe4ba70f30d9f6ed0`.
- Artifact-manifest SHA-256:
  `67263687318aa4f08f378320121e198bd4091a2c9546aa2c99958ec9789956cb`.
- Checksums-file SHA-256:
  `a2ce6be879d8380344209dce0ffe993f9059f8f8e55d04881e7e7b9e46113617`.
- Prediction-manifest SHA-256:
  `89c7c3c6444fc93e1a30e5274f93ee4d79eddbdee92802fc561b073ef47048dc`.
- Model-manifest SHA-256:
  `6be0059028ff1d910cbd2c7f3f3067087b7615f71aae96fd750089d11d84e32d`.
- Split-manifest SHA-256:
  `ade612dd40457918561fbbfdfa6786993df2198576d77612b05ca03b39ffeb8c`.
- AR-dominance decomposition SHA-256:
  `21e4e081094df6b4b2b2c3e206deae44f05d501c875e89e0a189d95cc1739595`.
- Fold-metrics SHA-256:
  `e67a9a09bd7fa78382a2652cc7127dd7f3a5bd90b64d32ca39a523cb3ef1ef72`.
- Per-video-metrics SHA-256:
  `6ba7879342064806c4ce01a7cfd4f2b84a0ef7d0c643b12c87abbc4a570de89b`.
- Hyperparameter-search SHA-256:
  `2f28f96b3540032845e16476ecdd557b197d5890eb1c357874a59d75c3b7a8ae`.
- VEATIC code SHA-256: `48ea2c2ec687d777098882bd3f00721e715743314d080b0ab1a18fe4a8c291ef`.
- Target: sealed continuous future maximum increase over `t+1..t+6`; q90 was fitted
  separately inside every applicable training partition.
- Common causal-history mask: 19,169 rows across 124 videos.
- Fresh VEATIC-derived AR lag candidates: `0/1/2/4/6` rows; all lag and ridge choices were
  selected by nested inner-validation raw PR-AUC.
- Protocols: five digest-derived grouped-video 70/30 cells and one per-video forward
  blocked-temporal 70/30 cell, reported separately.
- Grouped held-out AR PR-AUC: median `0.315086`, range `0.278621–0.383829`.
- Blocked held-out AR PR-AUC: `0.276250`.
- AR exceeded analytic chance and the training-owned strongest simple causal-history control
  in every cell; paired whole-video bootstrap intervals and per-video defined/undefined
  metrics are sealed in the decomposition and metrics artifacts.
- Exact outer-test rows and AR/current/slope/chance predictions are frozen in six checksummed
  target/protocol/fold/seed bundles for matched downstream lanes.
- Prospective washout activated/selected: false/false. Target/history overlap: zero rows;
  control-complete development evidence remains required before any redesign.
- Training runtime: MLX on `gpu:0`, exactly one worker, no artificial memory cap.
- Hidden state loaded/hashed: false/false. Cortical values loaded: false. PCA/bridge work: none.
- Focused VEATIC tests: 55 passed, 0 failed.
- Compact record: `studies/veatic-2.1/phase-02-ar-baseline`.

## Concluded Phase 03 evidence

External root:

`/Volumes/onn. Drive/Neural Bridge Artifacts/runs/veatic-2.1/again-method-restart-20260723/phase-03-raw-cortical`

- Result SHA-256: `8c0839d8eb8ba5c20e4c13ae83367b0fe4e0e383b7e0c3b074b80f7a5cf38c16`.
- Artifact-manifest SHA-256:
  `200c05ef34c9f7926133b3ae39aeb7c6e25350d142df34ef38e46fb24a75fdbf`.
- Checksums-file SHA-256:
  `ae6add6926dd8be81d287dc514a9d261fbfb0ba921d6920c4beaf7dc4696ca51`.
- Prediction-manifest SHA-256:
  `186acd0eb6017c7764fa1fc5215567e34ef5c55cfbee0b5bc94a0da6fc8b9d91`.
- Model-manifest SHA-256:
  `720e5756f8883c90be628180ca33efd767c03bf2d1f4702951d53e13c9612f23`.
- Primary-deltas SHA-256:
  `21808b3712c79a297a22c31b64a8da58b57b80222ddc9f93541a4a42c0734ac1`.
- Summary SHA-256: `5f453b5bdc9333d11ab18324c6de8cfc1675f5c47e8113525d8e0bb23efc1b15`.
- Fold-metrics SHA-256:
  `97e6e98e043d9fbd48349fd87216fb583865eef312dc476b90753c890ccc5f18`.
- Per-video-metrics SHA-256:
  `11c0ced3e63679f8d971b843ef8680835f8f48a7e0a454238747854589d18a80`.
- Control-matrix SHA-256:
  `809208be17960eb98171409669ae8cf95c9432b855d48e7a5950d3a64706ca59`.
- Input-manifest SHA-256:
  `e47f5e28119344ec9f405c13e8fa91ddf1ee6565d90b9abb573154b8c0f3ca00`.
- VEATIC code SHA-256: `51110daaa37578ae4f73d7b7cff3146d8a943e3aeeda3ca580283870f11c0fa1`.
- Real input: all 20,484 dimensions of final TRIBE `cortical_prediction`; no PCA or width
  selection.
- Matrix: 17 matched lanes in each of five grouped-video cells and one blocked-temporal cell.
  Controls include frozen AR, raw-only/current-row, within-video shuffled, shape-matched
  random, train-only video mean, diagnostics, time/video-time, quality/motion/luma, and
  training-label permutation, with only and AR-plus variants where applicable.
- Exact Phase 02 row, target, q90, fold, seed, and frozen AR prediction identities were reused.
- Grouped median PR-AUC: frozen AR `0.315086`; real cortical-only `0.120929`; direct
  AR-plus-real `0.317626`.
- Blocked PR-AUC: frozen AR `0.276250`; real cortical-only `0.088738`; direct AR-plus-real
  `0.263731`.
- Direct raw-fusion claim: FAIL. Fusion did not beat frozen AR consistently across grouped
  folds and degraded the blocked result. Direct fusion is not promoted.
- Exact 17-lane predictions, complete spike metrics, defined/undefined per-video metrics, and
  paired whole-video bootstrap deltas are sealed for all six cells.
- Training runtime: MLX on `gpu:0`, exactly one worker, no artificial memory cap.
- Hidden state loaded/hashed: false/false. Grouped upstream feature loaded: false.
- PCA/width selection/washout/learned bridge: none/none/none/none.
- Focused VEATIC tests: 64 passed, 0 failed.
- Compact record: `studies/veatic-2.1/phase-03-raw-cortical`.

## Concluded Phase 04 evidence

External root:

`/Volumes/onn. Drive/Neural Bridge Artifacts/runs/veatic-2.1/again-method-restart-20260723/phase-04-pca-bridge`

- Result SHA-256: `922f181ded0b9125de43242558bd6e66113bcebf24316ca38d7767b1965f8da4`.
- Artifact-manifest SHA-256:
  `e9000630e2971536babc4bd7dab123507b570c78a68a1ae3ae3ec8fc20ff56a8`.
- Checksums-file SHA-256:
  `bc195a879ddc2be9c37005c285b2560f69c2d4e7831b3182d6a95739f7631924`.
- Request SHA-256: `dd6bc73ef170e7e7b69c17d0cc03d546c2738b7d7d74ced4f0da117434fe0fa9`.
- PCA-accuracy-audit SHA-256:
  `d55ae7ebd9a4c0ea15e7307edbc6e643283aa4d1be3813e1fab8d1ddf5a28a37`.
- Projection-cache-manifest SHA-256:
  `03cefe1a72d72021bc08ca2fcb2731d32a2deebc9fced5706407a3afb0f9ec4d`.
- Prediction-manifest SHA-256:
  `ee3c73f873fe11fcab88fba840ceb7b6f63b5369933c877adf20668a99969623`.
- Final-model-manifest SHA-256:
  `1286894f22cf1b12feb373d7d05357cf6b6f9912e8137505c1cb632b3aee2afb`.
- Selected-representation SHA-256:
  `e906ff541c01113998e8a4d0081a71fe92417e82137b81c0286fc2414c38adb0`.
- Summary SHA-256: `70347ca38b29f40acef6660cfc75d4983253f4150feebad0352fc89f384990c0`.
- Primary-deltas SHA-256:
  `93ff652c9e0186842fa433a301b2b97cb306ba117d126d4b47e59840e9bc6e5e`.
- Fold-metrics SHA-256:
  `0910c1f81cf22c7378fa81e4e4f79a3102f4456334d820a206948c6383aaad13`.
- Per-video-metrics SHA-256:
  `ae1b7e3ef3c20ccc37dac43b28bcf15ddb757a5131d103d008ce6649b660af08`.
- Inner-family-search SHA-256:
  `110504d45deac45add946fd4aeaf885fcbe61e34073250bed06f8a6b9cba77af`.
- Inner-candidate-search SHA-256:
  `6a3f935315a6fd09ce10169b8dff613b38b0a873942455745e8ed4a9545be540`.
- VEATIC code SHA-256: `4d1092d6bbd134c9bd633a69292667e7d10fa2b881917191b9b5c74648955b66`.
- Per outer-training cell, one maximum rank-512 basis used every owned eligible row after
  outer-training-only scaling. Nested `64/128/256/512` prefixes and causal temporal depths
  `0/4/6` were evaluated under the complete control matrix.
- Accuracy audit: six primary and six independent-seed bases passed; no width required a
  separate fallback fit. Minimum leading-subspace overlap was `0.999983`; maximum component
  orthogonality error was `0.003089`; primary reconstruction residual fractions ranged
  `0.000121–0.000365`.
- Global inner-only selection froze width `64` and temporal depth `0` rows. Its median inner
  fusion selection margin versus frozen AR and the strongest matched control was `-0.045159`.
- Matrix: 17 matched outer lanes in each of five grouped-video cells and one blocked-temporal
  cell, with 102 exact metric rows, six prediction bundles, and six final-model bundles.
- Grouped median PR-AUC: frozen AR `0.315086`; selected real PCA-only `0.128426`; selected
  AR-plus-real PCA `0.309344`.
- Blocked PR-AUC: frozen AR `0.276250`; selected real PCA-only `0.115511`; selected
  AR-plus-real PCA `0.259457`.
- Linear PCA-fusion claim: FAIL. Fusion did not consistently beat frozen AR and matched
  controls. Linear PCA fusion is not promoted, but the selected representation is frozen for
  the ordered learned residual question.
- The complete external checksum inventory independently verifies. Exact Phase 02 ownership
  and frozen AR predictions were reused; no held-out row selected the representation.
- Training runtime: MLX on `gpu:0`, exactly one worker, no artificial memory cap.
- Hidden state loaded/hashed: false/false. Grouped upstream feature loaded: false.
- Washout/learned bridge/AGAIN runtime dependency: none/none/none.
- Focused VEATIC tests: 73 passed, 0 failed; full repository tests: 93 passed, 0 failed.
- Compact record: `studies/veatic-2.1/phase-04-pca-bridge`.

## Active execution contract

Implement Phase 05 exactly from:

- `veatic21-master-scientific-specification.md` → **AGAIN Phase 05/5.5 — learned frozen-AR
  bridge and event head**, **Control matrix required from the first applicable cell**,
  **Metric contract**, **VEATIC-specific washout procedure**, and **Phase 02 through
  zero-label execution sequence**;
- `veatic21-rebuild-protocol.md` → **Phase 05 — VEATIC learned frozen-AR bridge**.

Use the exact sealed Phase 02 target/protocol/fold/seed ownership and frozen AR predictions.
Use only the sealed Phase 04 width-64 current-row PCA representation. Make the exact matching
AR output an immutable residual floor shared by real and every matched control. Train cortical
signal only as a residual correction, include a training-owned no-harm suppression/fallback
to frozen AR, and keep representation/capacity/head/model selection inside training and inner
validation ownership.

Run the complete residual real/control matrix from the first applicable cell under identical
target, row, split, fold, seed, frozen AR, sealed PCA, temporal context, capacity, selection,
and metric ownership. Interpret label permutation against the retained AR floor and preserve
train-only video means. Restore and score selected checkpoints in eval mode; freeze exact
prediction, model, checkpoint, and row-ownership identities.

Report grouped and blocked protocols separately with the complete spike metric stack, frozen
AR and strongest-control deltas, fold/video consistency, and paired whole-video uncertainty.
The no-washout task is the only active scientific task. Do not activate a washout or execute
Phase 06. AGAIN code, runners, data, widths, temporal choices, heads, checkpoints, fitted
artifacts, predictions, and numeric selections remain forbidden by inheritance; AGAIN
contributes method and rigor patterns only.

All learned training uses MLX with exactly one GPU worker and no artificial memory cap. CPU
remains limited to parsing, deterministic audits, orchestration, metrics, hashing, and report
generation.

## Progression and handoff rule

When Phase 05 completes:

1. inspect every compact and external Phase 05 output;
2. run all focused VEATIC and authority-contract tests;
3. create the compact defensible study record under
   `studies/veatic-2.1/phase-05-learned-bridge`;
4. replace this file with the new live state while retaining **Mandatory authority anchors**;
5. record exact code/input/output hashes and the single newly authorized action;
6. commit and push the coherent transition directly to `origin/main`;
7. begin Phase 06 only after the Phase 05 gate passes and the transition is on remote
   `main`.

Do not rewrite the master specification merely because progress changed. Amend it only for an
explicitly authorized durable method change.

## Exact next action

Implement, test, execute, and review the Phase 05 VEATIC learned frozen-AR residual bridge on
the sealed width-64 current-row Phase 04 representation, exact Phase 02 ownership, and exact
frozen AR floor. Start with the complete residual control matrix, training-owned no-harm
fallback, inner-only selection, and eval-mode checkpoint scoring. Do not activate a washout or
begin Phase 06.
