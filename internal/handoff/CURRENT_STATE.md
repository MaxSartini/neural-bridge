# Current State — VEATIC 2.1 Stage A Rescue Executor Backtest Registered

Updated: 2026-07-29

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
- Master scientific specification: version 2.2.
- Repository: `/Users/maxsartini/Neural Bridge`.
- Branch: `main` only; do not create a branch.
- Lifecycle boundary: fresh Phases 00 and 01 concluded; Phase 02 scientific experiment
  registration frozen and verified; the underpowered sequential Stage A attempt is sealed;
  the initial and all-uncompiled executor matrices are complete; the winning executor is
  frozen; the replacement main Stage A execution and exhaustive verification are complete;
  the exact undertrained-cell registry is independently verified and frozen; the sparse
  rescue solver/executor and representative systems-backtest matrix are implemented,
  validated, and prospectively frozen; before the registered hardware backtest.
- Phase 00 implementation: complete.
- Phase 00 execution: PASS, 27/27 mandatory controls.
- Phase 01 implementation: complete.
- Phase 01 execution and independent verification: PASS, 28/28 mandatory controls.
- Phase 02 registration implementation/execution/verification: PASS; no model or outer score
  opened.
- Phase 02 sequential Stage A attempt: user-authorized termination sealed after 6,091 inner
  work units and 1,279,110 configuration evaluations; not eligible for main-run resume.
- Executed claim-bearing outer modeling phases: none.
- Registered target substrate: 231 continuous future-maximum-increase candidates; all 21
  no-washout candidates active for Phase 02, 210 washout candidates prospective only, no
  target selected.
- Registered VEATIC Phase 02 splits/search/control/metric rules: frozen. AR models,
  projections, representations, heads, checkpoints, and promotion outcomes: none.
- Current promotable VEATIC result: none.
- Authorized action: execute the prospectively registered hardware-saturated sparse
  16×-budget rescue backtest for the frozen Phase 02 comprehensive target-specific AR
  benchmark, then freeze and push the fastest numerically equivalent rescue configuration
  and exact main-run identity before executing any rescue cell.
- Cortical benchmark, PCA, head search, washout cortical score, continuous, valence, and
  zero-label actions remain unauthorized.

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

## Concluded Phase 01 evidence

External root:

`/Volumes/onn. Drive/Neural Bridge Artifacts/runs/veatic-2.1/fresh-method-rebuild-20260728/phase-01-label-alignment`

- Status: PASS, 28/28 mandatory controls; independent output verification PASS.
- Result SHA-256: `feaae3f2f9b954786457dd816dd22f911d30c31492da42be682558ed84182710`.
- Artifact-manifest SHA-256:
  `ea7257732a6de79b67448bebfca75267242cd17b5e7cf6a8b984cebe0c6a551e`.
- Checksums-file SHA-256:
  `0abe69e14a7d156cd5b950bcbcb7f44616965320e1c4bac70979522fa8ea1348`.
- VEATIC code SHA-256:
  `cf79ad38c91e19c9c7eb207c51f30febd77c0979156e39f4b98ff1b64e30d7fe`.
- Input-identity SHA-256:
  `5ac7bb4461ba4c746325f49b67725a6b8cad9cc11298be364537a2dbd9a1b25b`.
- Row-identity SHA-256:
  `54aa33fa242e45173ec9c4a4d3ad22857b21ee23622ec5b045d3303609eee39c`.
- Label SHA-256: `03058fec314ea7d35b9d25f590888eb3880a380c23e3201fb571add6a91bff51`.
- Alignment SHA-256:
  `82ac63431ec02d307760d592fe993c64c7936f4fde2150d4f6a31e106b4e5f83`.
- Target-source SHA-256:
  `3a563f266a9e39eb647019247b073b0bf03d5f61d33dcb2fc391f5cf40c5c8c7`.
- Continuous-target SHA-256:
  `b13d9a8c0c578d81631fa0a5485d1e2cb2a1bf7f3102c4bc1c09df8a4f6e304a`.
- Validity-mask SHA-256:
  `f217a533b59e52dae63922f9d7c09d97641b4ddec51ec37b9de3dc871a370383`.
- Row-ownership SHA-256:
  `e7d37b3e5edcbee6a4a24ff7f9b186fb09bcdbed452536530163546da10057eb`.
- Quality-metadata SHA-256:
  `75afaaac71ba7ae073ae3dbef49a5fb8f18f13ae53c86ec035c5e7f8d896a9da`.
- Coverage: all 124 videos and all 20,657 exact 2 Hz rows; no filtered rows.
- Preserved metadata: native interpolation provenance plus TRIBE black/static fractions,
  flags, union, and suggested quality weight for every row.
- TRIBE access: time and quality metadata only; no cortical value loaded or scored.
- Candidate derivation: shortest video has 22 rows, so the complete supported lattice is all
  `1 <= start <= end <= 21`, totaling 231 candidates.
- Phase 02 active family: all 21 no-washout candidates `(start=1, end=1..21)`.
- Prospective-only family: 210 washout candidates `(start>1)`; inactive until the durable
  conditional washout trigger is satisfied in a later authorized phase.
- Selected target/global binary label/outer split: none/absent/absent.
- AR/PCA/cortical performance/model/head operations: none.
- Compact record: `studies/veatic-2.1/phase-01-label-alignment`.

## Frozen Phase 02 registration evidence

External root:

`/Volumes/onn. Drive/Neural Bridge Artifacts/runs/veatic-2.1/fresh-method-rebuild-20260728/phase-02-target-specific-ar/registration`

- Status: PASS; independent registration verification PASS.
- Registration SHA-256:
  `ab21c9b971fc0cf8aa18f4d77f585b8236db08bf810b1f83c79a476dfde44815`.
- Result SHA-256: `0f3064d7ed7207879b195d8ed8ddb57fd440ed62f3922ee793159faee58a016e`.
- Artifact-manifest SHA-256:
  `39d5e587f4d1c1529039f8aae59137d3e07f6e265dda484bbf5797ccc6942a7c`.
- Checksums-file SHA-256:
  `286db04959d3f8b1bd68b361115ed43772be1f3d6ae2e8a79f384a220d71754b`.
- VEATIC code SHA-256:
  `721a47a32a40d402da6e006f061cc74071ec7c18a08a3cd1f53889fcd20f000c`.
- Input-identity SHA-256:
  `2c565a07ced8946b47b6642bd1db9266fe0ae67ff4ae4252aa89e9dd1976dd19`.
- Target coverage: all 21 active no-washout candidates; 210 prospective washout candidates
  inactive.
- Grouped protocol: four independent 10-fold video partitions, 12–13 test videos per fold,
  four nested inner folds, full `0..123` test coverage once per repeat.
- Blocked protocol: four native-time blocks and two expanding forward folds; immediately
  preceding inner-validation block; target-boundary crossing rows purged from training.
- Support gate: at least 1,000 rows per evaluation cell. Observed minima: 1,219 grouped test,
  2,672 blocked test, and 2,630 blocked inner-validation rows.
- Search breadth: all 21 causal history depths, six causal feature forms, analytic controls,
  continuous ridge, L2 logistic, elastic-net logistic, MLP, and GRU families.
- Optimization breadth: training-scaled regularization, elastic ratios, widths, depths,
  activations, dropout, AdamW/SGD-Nesterov, learning rates, batches, staged update budgets,
  three calibration methods, boundary expansion, undertraining recovery, and five fresh
  VEATIC-hash-derived finalist seeds.
- Solver/metric closure: MLX ridge and accelerated convex logistic/elastic solvers,
  convergence tolerance, nonlinear optimizer constants, inner-owned calibration and F1
  decision rule, and 1,024 whole-video bootstrap replicates are explicit in the freeze.
- Threshold ownership: q90 refit inside every applicable training partition; no global binary
  target stored.
- Outer model scores/test labels/cortical values opened: false/false/false.
- AR/PCA/head/washout operations: none.
- Repository freeze:
  `internal/active/veatic21-phase02-registration/experiment-registration.json`.

## Sealed sequential Stage A attempt

- Runner: `src/neural_bridge/veatic21/phase02_stage_a.py`.
- Stage A source SHA-256:
  `6a4f368a8ebccc2bab15d3658909438cccceee3c01601e2ec9bd7ccea5ebed78`.
- Backend: one sequential MLX GPU process with largely one-core CPU orchestration; terminated
  because it materially underutilized the Mac Studio hardware.
- Complete work-unit count: 40,824.
- Complete target/regularizer inner-evaluation record count: 8,573,040.
- Per-unit scope: one protocol/outer/inner/feature-depth/model cell, all 21 active targets,
  and all 10 registered training-scaled regularization multipliers.
- Registered Stage A families: continuous ridge and event logistic-L2 across all 21 history
  depths and all six feature forms.
- Logistic convergence rule: run the data-derived base budget, escalate unresolved cells to
  the registered 4× budget, and mark any remaining unresolved cell `undertrained`, protected
  from pruning, and required to receive the registered 16× budget before disposition.
- Causal cold-start handling: earliest/current-level padding plus explicit availability masks;
  no target-valid row silently removed for unavailable past.
- Artifact root:
  `/Volumes/onn. Drive/Neural Bridge Artifacts/runs/veatic-2.1/fresh-method-rebuild-20260728/phase-02-target-specific-ar/benchmark/stage-a-linear-screen`.
- Immutable Stage A request SHA-256:
  `f54ecf511d60455e1f84193bd17d91d2a0ffc85ed3daa41fe290b520d96ff557`.
- Immutable work-unit-registry SHA-256:
  `67ab7e7f0414b2cf7f2705e589ce69f76e37409168c231457a7a4db287677d3b`.
- Terminated at: `2026-07-28T23:11:08Z`.
- Final completed work units/configurations: `6,091` / `1,279,110`.
- Final unit files/ledger lines: `6,091` / `6,091`.
- Final append-only ledger SHA-256:
  `eb2b4ccbbaf5ab50e48230bb39649cb20857307cc385b222b81817c0c5d7b6d1`.
- Termination manifest: `<artifact root>/termination-manifest.json`.
- Termination-manifest SHA-256:
  `d647b77b125b5e1d01bbbfc81ed7824b3816f62477266f965a36e4bc691a03a6`.
- Sealed run-state SHA-256:
  `b022116d04786a2b4236c374d8672103cf88b585058c793a0a4994e21132b90e`.
- Disposition: preserved for provenance and numerical backtesting; never resume or merge into
  the replacement main run.
- First paired real-cell audit: ridge converged `210/210`; logistic used the registered 4×
  escalation and converged `210/210`; full artifact/ledger/resume identity passed.
- Resumption contract: request, work-unit registry, code/registration identity, unit JSON,
  append-only ledger, and run state must agree; a source or registration mismatch fails.
- Outer test scores and cortical values remain unopened.

## Replacement executor backtest and frozen selection

The first coordinated-backtest attempt at `<Phase 02 benchmark root>/executor-backtest-20260729`
was terminated without selection after the mandatory resume gate correctly rejected a
tuple-versus-JSON-list request-identity mismatch. It contains 985 partial unit files and is
failure provenance only: never resume or merge it. Its termination-manifest SHA-256 is
`7364e0f08f4c7f59904f2df7902d6ac945dc3a0b6fb6f62379a60123be904e0e`; outer-test scores and
cortical values remained unopened.

- Repository freeze:
  `internal/active/veatic21-phase02-registration/executor-backtest-registration.json`.
- Backtest-registration SHA-256:
  `9b9025d889149e4dda8b8b61d2c75193f62fa238951b08fb70d53965e6fe8943`.
- Active canonical external backtest root:
  `<Phase 02 benchmark root>/executor-backtest-20260729-v3-uncompiled`.
- Host: Apple M2 Max Mac Studio, 12 CPU cores (8 performance, 4 efficiency), 32 GiB unified
  memory.
- Backtest uses preserved inner-only reference units; it may not open outer-test outcomes or
  cortical values.
- Numerical-equivalence coverage: both complete blocked Stage A inner cells plus one complete
  grouped Stage A inner cell, spanning all six feature forms, all 21 history depths, and both
  Stage A model families.
- The completed initial 19-candidate matrix at
  `<Phase 02 benchmark root>/executor-backtest-20260729-v2` returned `PASS` and provisionally
  selected `optimized_uncompiled_4p1s_2m`: median `10.911889600455535` work units/second,
  mean GPU utilization `98.6969696969697%`, executor SHA-256
  `80874c11358e33bbe4182edb23bacefd38a6f6acad8ab8df3ff2be0359aa1fc1`. It passed every
  equivalence, determinism, resume, ledger, access, memory, thermal, and saturation gate.
- Initial-matrix immutable hashes: request
  `a3562d1e0abcb68d9f1dfeb94a4221ad8c15531c9161704783b333b22dd33427`;
  candidate summaries
  `ae1fbd383c1ff6d8c1a36dd4e2f349813bce64c8141ae9a792d70b02a91377f7`;
  result `c7d5c702d6c5b10c4b46173731ace0c3acb7abc8c0c39a6f0b9bf25705770a88`.
- Every candidate using compiled ridge updates failed the comprehensive real-VEATIC numerical
  equivalence gate: `52,327` mismatches, maximum metric difference `0.007819989`, and maximum
  solver-diagnostic difference `0.026921`. Full ridge-plus-logistic compilation failed more
  broadly with `175,999` mismatches. Compilation is disqualified; synthetic checks cannot
  override this evidence.
- The provisional four-process choice is not frozen because the initial matrix measured the
  1/2/3/6/8/12-queue topologies only with the now-disqualified compilation path. The active
  supplement independently measures uncompiled 1/2/3/4/6/8/12-process cases plus one/two/four
  Metal-stream cases. All candidates retain cached preparation, exact fast metrics,
  deterministic sharding, atomic publication, shard ledgers, and verified canonical merge.
- GPU work already batches the complete registered `10 regularizers × 21 targets = 210`
  solver cells per unit. Unit-level queue concurrency, CPU metric parallelism, and
  ridge/logistic preparation caching are measured around that full mathematical batch.
- A real warm-up barrier excludes MLX graph compilation and worker-start skew from timed
  throughput. Timed evidence samples GPU utilization, aggregate worker CPU, operating-system
  memory headroom, summed per-process MLX peaks, swap/power/thermal state, and all failures.
- Selection uses three timed repetitions and hardware throughput only. Scientific outcome
  scores cannot select the executor.
- The complete all-uncompiled supplement returned `PASS`: all 10 candidates passed numerical
  equivalence over 756 real VEATIC units, deterministic three-repetition output identity,
  safe resume, ledger, access, memory, thermal, and saturation gates. Every candidate recorded
  zero mismatches; the maximum metric difference was `3.3306690738754696e-16` and maximum
  solver-diagnostic difference was `0.0`.
- Supplement immutable hashes: request
  `a2677c44637287dc115fe3a9d907ea1157c9dd32102986b516f07664a1ae4d32`;
  candidate summaries
  `54b015286d9405135aea3d515a34829333471bd639941e78826565d5ce575469`;
  result `d997aa0489b39de166213de38f9131b9a0bc11e3c0f8324045755e68e1a0bd72`.
- Uncompiled median work units/second by topology: `1p1s=5.8377393965`,
  `1p2s=8.7319188012`, `1p4s=10.2728130824`, `2p1s=9.5719485235`,
  `2p2s=10.8519046600`, `3p1s=10.7663967297`, `4p1s=10.9173248642`,
  `6p1s=10.8800624550`, `8p1s=10.8696094643`, and `12p1s=10.8668807219`.
  GPU utilization rose from `79.01%` at one queue to `99.80%` at 12 processes while
  throughput plateaued after four total queues.
- Frozen selected executor: `uncompiled_3p1s_2m` — three isolated MLX processes, one
  thread-local Metal stream per process, two CPU metric workers per process, pair-owned
  preparation cache, exact fast metrics, no ridge or logistic graph compilation, pipeline
  depth four. Its median is `10.766396729699405` work units/second, within `1.38%` of the raw
  four-process peak; the pre-registered within-three-percent rule selected fewer total GPU
  queues before fewer processes and memory.
- Selected executor source SHA-256:
  `be395e0b67ce1eec0bf529051fad3ab9cc979627363dc16ca4ca2a35318c5abf`.
- Selected-executor repository freeze:
  `internal/active/veatic21-phase02-registration/selected-executor.json`; SHA-256
  `ebfca6234e254f21631ef4b4a1e136449c73c7690122c31eb6f91afccc2960f1`.
- Frozen main launcher source SHA-256:
  `a60a953f322fcc44aec7ff99c32468771bef58cf25f08890a1867df8d5d18d91`.
- New canonical main Stage A root:
  `<Phase 02 benchmark root>/stage-a-linear-screen-hardware-saturated`; it is a fresh identity
  and must never merge the sealed sequential run. Complete scope is all `40,824` work units
  and `8,573,040` registered configuration evaluations. The previous measured speed implies
  approximately 63 minutes of steady-state executor time before filesystem/finalization
  overhead; this is an estimate, not a completion claim.
- Main Stage A launched from remote-synchronized `main` commit `3ac4c94` at
  `2026-07-29T02:36:08Z` under `caffeinate -dimsu`. Immutable request SHA-256:
  `7e5e06aa29654d8b0b58bcb3f6156d9a9d94cda7d13e46849ce2ab7fa4e82eff`.
  Immutable full work-unit-registry SHA-256:
  `080485ed8794547eae087332dad99bc119a305843cb687837b2e96f143185849`.
  First recorded live checkpoint: `436/40,824`, status `RUNNING`, exact selected executor
  identity present, outer-test scores unopened, and cortical values unopened.
- Stage A execution completed before verification. The first exhaustive-verifier invocation
  from commit `69f77eb` stopped before unit hashing because it incorrectly required the
  unit-level `cortical_values_opened` field to be duplicated in the ledger-entry schema. No
  verification result was written or accepted; the immutable run artifacts were not changed.
- The second invocation from commit `86d89be` passed ledger reconciliation and entered the
  eight-process unit audit, then stopped because the verifier assumed the ridge
  `relative_residual_by_cell` diagnostic key also named the registered logistic
  `relative_gradient_by_cell` diagnostic. No verification result was written or accepted;
  both frozen model-family schemas and all run artifacts remain unchanged.
- The final Stage A state is `COMPLETE`: `40,824/40,824` unique units, `8,573,040`
  configuration evaluations, `3,790.368498459` seconds, and
  `10.77045675548361` work units/second. Canonical ledger: `40,824` unique lines, SHA-256
  `95bf4d4c18b38372ca81af0ee8210a9b18da942db12f6162c8c678a0a1b9d342`.
- Immutable final hashes: run state
  `c42fa75ef13cced9177e907157d1fa2414351d5bf1d79f4118380268f059e505`;
  resource summary `1a17beb6c5920ba72290b5d9bb929743b708e8844b393d69c981cd64a076b067`;
  resource samples `a0389ea09b36a45d3505dfa9b390462f943e02590e661766cfb0cd0da13547bd`.
- Hardware evidence: mean GPU utilization `97.43604813881892%`, minimum measured memory
  headroom `20,272,245,637` bytes, zero swap, AC power, low-power mode disabled, system sleep
  disabled, and no thermal or performance warning.
- Exhaustive eight-process verification returned `PASS` after re-reading and hashing all
  `8,508,249,816` unit-result bytes and reconciling every registry, unit schema, provenance,
  configuration ID, shard ledger, canonical ledger, and resource sample. Verification
  SHA-256: `32467b1cbe223a7297cb90b4546e71ac56478c834a720ec5af90775cfc01afb4`;
  verifier source SHA-256:
  `e0e87ce48c032b40c7eccda404aa1207c65c2ae70b9540f2a6ec6c8ecc2bf3c9`.
- Convergence disposition: `8,459,648` completed/eligible records and `113,392` undertrained,
  protected records. By family: ridge `4,173,766` completed and `112,754` undertrained;
  logistic-L2 `4,285,882` completed and `638` undertrained. Aggregation/pruning is forbidden
  until the exact undertrained set receives the frozen rescue disposition.
- Rescue registry external root:
  `<Phase 02 benchmark root>/stage-a-convergence-rescue/registration`. Independent
  re-derivation returned `PASS`: `113,392` unique configuration IDs and rescue-cell identities
  across `14,465` affected units. The registry contains ridge `112,754`, logistic-L2 `638`,
  grouped `111,827`, and blocked `1,565` cells; no completed Stage A cell is present.
- External rescue-registration hashes: request
  `37857dd1a9c5bc5b145c1c99e18b9f7acb2e9691ca1e38f591915135b05f5346`;
  undertrained-cell registry
  `0d3b1e276082263d325d0cc523f4a80619f5ce6a5794016fd31bbc180ab2c791`;
  affected-unit registry
  `46bf2efd5e622423195a722bd4c7cf14fe8e58d29a474c283b0e9349e0ef107d`;
  summary `ef8176121527d56e6f479d2b5be355e87e14e6a3c9c3644fce1bb0cafa867dac`;
  verification `b668b58e455f9486850ac51ea08c9c186531f708f49fd799315e5c8c4f2f5a6d`.
- Compact frozen rescue registration:
  `internal/active/veatic21-phase02-registration/convergence-rescue-registration.json`;
  SHA-256 `003c6f4e72e9ab2cc8198d0d4ce56b669f4b36148a6efd4b6a066f38c054efb6`.
- The compact registration freezes zero initialization, exact unchanged cell ownership and
  numerical identities, total `16B` maximum budget, converged/invalid dispositions, separate
  linked ledgers, immutable Stage A artifacts, and the full hardware-backtest gates. It does
  not authorize rescue execution.
- Sparse rescue solver and executor implementation is complete. It loads only the exact
  registered undertrained cells, reconstructs and verifies the immutable Stage A rows,
  features, scaler, threshold, regularizer, tolerance, and unit hashes, restarts each cell
  from zero, checks convergence every eight updates through at most `16B`, freezes cells
  individually when converged, and publishes only linked rescue records through atomic unit
  artifacts, shard-local append-only ledgers, and an exact canonical merge. It cannot admit a
  converged Stage A cell or perform aggregation/pruning.
- A real-data four-unit executor smoke test completed and then resumed without changing any
  unit or ledger hash. It covered `19` registered cells with two concurrent Metal streams,
  passed exact unit/cell ledger coverage and the outer/cortical/aggregation access firewall.
- Prospectively frozen sparse-rescue executor backtest registration:
  `internal/active/veatic21-phase02-registration/rescue-executor-backtest-registration.json`;
  SHA-256 `7c812c677abf92b3a2a0c7a66f4a9b3f992260ec2ddf85c0459d45425f58dedd`.
  Frozen solver identity:
  `58289ef933b42b2588d92c8b549a4b7f8a9a6083651f5f837c8f0397756abf32`;
  executor identity:
  `7674b09d893bb61a5178e4741f267ca44a0671b7498d3016a38661c1e7e3891b`.
- The registered systems search uses `24` equivalence units/`175` cells and `192` timed
  units/`1,614` cells, covering both model families, both protocols, all six feature forms,
  all 21 history depths in the timed set, all five sparse-cell-count bands, all 21 targets,
  and every affected regularization index. It stages all cell batches `1,4,8,16,32,64`, 19
  safe process/stream topologies through the 12-stream host ceiling, metric workers
  `1,2,4,8`, compiled and uncompiled update blocks, and three repeated finalist timings.
  Exact dispositions/iterations and structure plus `1e-5` float equivalence, bitwise repeated
  normalized artifacts, resume, ledger, access, six-GiB headroom, thermal, and saturation
  gates are mandatory. Scientific scores do not enter executor selection.
- Only convergence-rescue registration, executor implementation, and inner-only hardware
  backtesting are now authorized. Rescue execution remains unauthorized until its registry,
  selected executor, and main identity are frozen and pushed. Outer-test scoring, cortical
  data, aggregation/pruning, Stage B, and every later phase remain unauthorized.

## Active execution contract

Read these rebuild-protocol sections for the one authorized action:

- **Canonical input boundary**;
- **Phase 02 — target-specific fresh AR floor**;
- **Comprehensive experiment and search-sufficiency checklist**;
- **Controls from the first applicable cell**;
- **Metrics and uncertainty**;
- **Execution and artifact rules**.

Derive the convergence-rescue registry only from the verified Stage A units with disposition
`protected_from_pruning_requires_16x_budget`. It must not open an outer outcome, cortical
value, or prospective washout candidate.

The rescue must restart only the exact undertrained cells from zero at a total `16B` maximum,
preserve every original row/feature/threshold/scaler/regularizer/solver/tolerance identity,
and write linked immutable rescue records without modifying Stage A. Cells still unresolved
at `16B` receive the frozen invalid-nonconverged disposition and cannot count as negative
evidence. Converged Stage A cells cannot be rerun.

Do not add, remove, or tune a target, split, history family, feature form, model family,
regularization range, optimizer, budget, calibration method, seed count, control, metric, or
support gate based on any outer result. Apply the frozen staged pruning and boundary-expansion
rules using training/inner-validation evidence only. Fit q90 event thresholds, feature
normalization, AR regularization, calibration, and decision thresholds inside each applicable
training partition. Test rows never select a threshold, feature, model, budget, or candidate.

The replacement Phase 02 main run must maintain a full append-only experiment ledger for
every attempted configuration, including failures, convergence and learning-curve evidence,
pruning reasons, seeds, fold ownership, and exact artifacts. One estimator, history depth,
regularizer, optimizer, budget, or seed cannot establish the AR floor. A search-sufficiency
gate must show that plausible candidate families received a fair test and that the selected
configurations are not merely undertrained or boundary optima.

Begin the AR-dominance/overlap decomposition on development-owned data for each active target:
history rows consumed, target rows, intervening gap, simple causal-history baselines,
AR-versus-chance uplift, and fold/video consistency. The 210 prospective washout candidates
must not be scored or activated in Phase 02.

Phase 02 output root:

`/Volumes/onn. Drive/Neural Bridge Artifacts/runs/veatic-2.1/fresh-method-rebuild-20260728/phase-02-target-specific-ar`

## Progression and handoff rule

When and only when Phase 02 satisfies the comprehensive search-sufficiency gate and all
split-ownership, leakage, calibration, convergence, repeatability, and metric controls:

1. inspect every compact and external output;
2. run all focused VEATIC tests, authority-contract tests, and the full repository test suite;
3. create the compact defensible record under
   `studies/veatic-2.1/phase-02-target-specific-ar`;
4. replace this file while retaining **Mandatory authority anchors**;
5. record exact code, input, result, artifact-manifest, and checksum hashes plus the single
   newly authorized action;
6. commit and push the coherent transition directly to `origin/main`;
7. begin Phase 03 only after the Phase 02 transition is present on remote `main`.

No later phase may claim success or failure without the master specification's comprehensive
VEATIC experiment registration, full result ledger, complete controls, and
search-sufficiency gate.

## Exact next action

Commit and push the prospective sparse-rescue executor/backtest registration, then execute it
under `caffeinate` at the canonical external backtest root. Freeze only the fastest candidate
that passes exact structure/disposition/iteration and registered float equivalence, repeated
determinism, safe resume, unique linked ledgers, access firewall, memory, thermal, and GPU
saturation-or-safe-plateau gates. Commit and push the selected executor and exact new main
rescue identity before executing the full registry. Do not execute rescue cells outside the
registered backtest selections, and do not aggregate or prune Stage A yet.

Do not resume or merge the sealed sequential attempt. Do not score outer outcomes or cortical values.
Do not fit PCA or a learned head, aggregate/prune Stage A, advance to Stage B, or open any
prospective washout candidate while convergence rescue is pending or active.
