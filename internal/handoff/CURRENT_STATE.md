# Current State — VEATIC 2.1 Phase 02 Executor Backtest

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
- Master scientific specification: version 2.1.
- Repository: `/Users/maxsartini/Neural Bridge`.
- Branch: `main` only; do not create a branch.
- Lifecycle boundary: fresh Phases 00 and 01 concluded; Phase 02 scientific experiment
  registration frozen and verified; the underpowered sequential Stage A attempt is sealed;
  before the replacement Stage A executor backtest and main execution.
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
- Authorized action: implement and backtest a hardware-saturating executor for the frozen
  Phase 02 comprehensive target-specific AR benchmark, then freeze the fastest numerically
  equivalent configuration before starting a new main Stage A identity.
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

## Replacement executor backtest registration

- Repository freeze:
  `internal/active/veatic21-phase02-registration/executor-backtest-registration.json`.
- Backtest-registration SHA-256:
  `de77372559fda0f8eed1d887d350a448a11020b9840101845ff2d9775c67e544`.
- Host: Apple M2 Max Mac Studio, 12 CPU cores (8 performance, 4 efficiency), 32 GiB unified
  memory.
- Backtest uses preserved inner-only reference units; it may not open outer-test outcomes or
  cortical values.
- Numerical-equivalence coverage: both complete blocked Stage A inner cells plus one complete
  grouped Stage A inner cell, spanning all six feature forms, all 21 history depths, and both
  Stage A model families.
- Throughput candidates: the sequential reference plus compiled/cached 1/4/8/12-metric-worker
  cases and 1/2/3/4/6/8/12 total-GPU-queue cases. GPU concurrency includes both shared-runtime
  thread-local Metal streams and isolated MLX processes, so neither a four-process cap nor an
  untested assumption that twelve processes is optimal can determine the result. Every
  candidate uses deterministic sharding, atomic publication, shard ledgers, and verified
  canonical merge.
- GPU work already batches the complete registered `10 regularizers × 21 targets = 210`
  solver cells per unit. Eligible optimized candidates compile eight-update ridge blocks while
  retaining the reference logistic update path; separately named full-compilation candidates
  test compiled logistic blocks under the same strict equivalence gate. Uncompiled optimized
  component-ablation candidates prevent compilation from being assumed necessary. Unit-level
  queue concurrency, CPU metric parallelism, and ridge/logistic preparation caching are
  measured independently around that full mathematical batch.
- A real warm-up barrier excludes MLX graph compilation and worker-start skew from timed
  throughput. Timed evidence samples GPU utilization, aggregate worker CPU, operating-system
  memory headroom, summed per-process MLX peaks, swap/power/thermal state, and all failures.
- Selection uses three timed repetitions and hardware throughput only. Scientific outcome
  scores cannot select the executor.
- The main replacement run remains unauthorized until the fastest eligible executor and its
  exact digest/configuration are frozen here and pushed to `origin/main`.

## Active execution contract

Read these rebuild-protocol sections for the one authorized action:

- **Canonical input boundary**;
- **Phase 02 — target-specific fresh AR floor**;
- **Comprehensive experiment and search-sufficiency checklist**;
- **Controls from the first applicable cell**;
- **Metrics and uncertainty**;
- **Execution and artifact rules**.

Implement and backtest only the execution machinery for Phase 02 from the sealed Phase 01
substrate and exact frozen scientific registration. The backtest may reproduce only preserved
training/inner-validation units from the terminated attempt. It must not open a new outer
outcome, cortical value, or prospective washout candidate.

The executor must use one coordinator, deterministic disjoint shards, atomic unit
publication, shard-local ledgers, verified canonical merge, cached pair-owned preparation,
parallel CPU metrics, and measured MLX concurrency. Backtest every registered executor
candidate, record failures as well as successes, and require numerical equivalence,
convergence-mask identity, deterministic repeated output, safe resume, no duplicate work,
memory headroom, and access-firewall evidence.

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

Implement and test the coordinated hardware-saturating Stage A executor. Backtest it against
the sealed reference units using
`internal/active/veatic21-phase02-registration/executor-backtest-registration.json`. Run the
complete frozen CPU-worker and 1/2/3/4/6/8/12-total-GPU-queue matrix and three timed
repetitions; do not select from one convenient pilot. Freeze the fastest configuration only
after every equivalence, determinism, convergence, ownership, resume, ledger, access,
utilization, thermal, and memory gate passes.

Then replace this file with the selected executor digest and exact run identity, run the
focused and full tests, commit and push the coherent transition to `main`, and only then start
the new main Stage A root. Do not resume or merge the sealed sequential attempt. Do not score
outer outcomes or cortical values, fit PCA or a learned head, aggregate/prune Stage A, advance
to Stage B, or open any prospective washout candidate during executor development/backtesting.
