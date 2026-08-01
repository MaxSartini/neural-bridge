# VEATIC 2.1 Neural Bridge Master Scientific Specification

Specification version: 1.2
Authority reset: 2026-08-01
Status: Phase 00 passed; Phase 01 VEATIC derivation implementation authorized

## Purpose

This document is the durable scientific authority for the VEATIC 2.1 programme. It fixes
the input boundary, method-transfer firewall, evidence order, controls, split ownership,
metrics, experiment-sufficiency rules, hardware contract, artifact rules, and product
boundary. Live progress and the one authorized next action live in
`internal/handoff/CURRENT_STATE.md`. The operational checklist in
`internal/active/veatic21-rebuild-protocol.md` is derived from this document and cannot
override it.

VEATIC 2.1 is the original 124-video VEATIC dataset encoded through the completed V-JEPA
2.1 -> TRIBE v2 pipeline. It is not a new dataset, a second modality, or an AGAIN dataset.
The scientific task is to rebuild the downstream Neural Bridge for these VEATIC-specific
2 Hz predicted-cortical rows. The immediate objective is one supervised arousal package
containing the strongest independently validated spike specialist and continuous specialist.
Valence follows after that package wins. Zero-label-at-inference begins only after the
supervised abilities are established.

## Authority and change control

1. `AGENTS.md` is the repository contract.
2. This master specification owns durable scientific method.
3. The rebuild protocol is a checklist and navigation aid only.
4. `CURRENT_STATE.md` owns live status, hashes, authorization, and the exact next action.
5. A durable method change requires the master, protocol, and handoff to change together;
   authority tests must pass and the coherent transition must reach `origin/main` before
   the changed method executes.
6. Concluded VEATIC phase records may summarize evidence but cannot redefine this method.
7. Historical material is evidence only when the user explicitly authorizes historical
   review. The user has authorized the AGAIN study and run histories as a methodological
   map for this rebuild.

## Non-negotiable protected inputs

These roots must never be deleted, rewritten, moved, normalized in place, or regenerated:

- `/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/tribe-v2/`
- `/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/vjepa-2.1/`
- `/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/neural-bridge-input/`
- `/Volumes/onn. Drive/Neural Bridge Artifacts/runs/again/`

The first two feature roots are completed upstream scientific inputs. The
`neural-bridge-input` root is the sealed consolidated downstream bundle assembled from
allowlisted files in those two sources. The AGAIN run root is the historical execution map
used to understand the mature methodology. Cleanup operations must use literal allowlisted
VEATIC run/output paths and must fail closed if a protected root is inside the requested
deletion boundary.

## Canonical VEATIC input contract

### One upstream stack, one consolidated downstream bundle

V-JEPA 2.1 and TRIBE v2 are successive stages of one encoding pipeline:

`VEATIC video -> V-JEPA 2.1 per-video cache -> TRIBE v2 -> predicted cortical rows`

They are not independent feature modalities. To make this unambiguous, all Phase 00+ code
uses one sealed canonical bundle:

`/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/neural-bridge-input/`

The bundle has `per_video/0` through `per_video/123`. Each video folder co-locates the final
TRIBE cortical payload with its matching authoritative `rows.csv` and only the allowlisted
small provenance/alignment metadata. No hidden-state file is present. After the bundle is
sealed, downstream models must not read the two staging roots directly.

### TRIBE-v2 source used to assemble the bundle

Canonical root:

`/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/tribe-v2/veatic 2.1 raw cortical predictions/per_video`

For each video ID `v` in the complete 124-video set:

`per_video/<v>/tribe_v2_cortical_predictions.npz`

is the corresponding TRIBE payload. The primary real Neural Bridge representation is:

`cortical_prediction[row_index]`

No pooled NPZ is privileged. Every video-specific payload must be discovered by explicit
numeric video ID, and every source row must be accounted for.

### Matching row and label source used to assemble the bundle

Canonical root:

`/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/vjepa-2.1/veatic 2.1 v jepa 2.1 stuff`

For each video ID `v`:

`<v>/rows.csv`

is the authoritative row/label/alignment table. The identity contract is:

`rows.csv (video_id=v, row_index=i) <-> TRIBE payload array position i`

Alignment is per video. Filesystem order and global concatenation order are never identity.
Phase 00 must independently verify row count, contiguous row index, 0.5-second cadence,
timestamp equality, source-frame/interpolation equality, and label equality for all videos.

### V-JEPA allowlist and absolute hidden-state ban

Only small metadata and alignment files needed to validate `rows.csv` may be read from the
V-JEPA root. Every file named `vjepa21_hidden_states.npz` is forbidden. It must not be
opened, memory-mapped, hashed, copied, inspected for keys, used as a feature, or included in
a whole-tree digest. V-JEPA hidden states cannot enter PCA, training, fusion, selection, or
inference.

V-JEPA and TRIBE must not be rerun. Freshness applies to downstream VEATIC targets,
transforms, models, controls, and decisions—not to the completed upstream encoding.

### Consolidated-bundle schema

Each canonical `per_video/<v>/` directory contains:

- `tribe_v2_cortical_predictions.npz`;
- `rows.csv`;
- `cortical_manifest.json` and `cortical_status.json`;
- `alignment_manifest.json`, `alignment_preprocessing.json`, and `alignment_status.json`;
- `input-manifest.json` with source paths, source/destination SHA-256 values, byte counts,
  row counts, schema checks, and the exact `(video_id,row_index)` contract.

The upstream `_PAYLOAD_SHA256.json` and `_UPLOAD_COMPLETE.json` files are not copied because
they inventory the forbidden hidden-state payload. Their transport bookkeeping is not needed
to align or interpret the final TRIBE rows. This exclusion avoids carrying a hidden-state
filename or digest into the downstream bundle while leaving the upstream records untouched.

The root contains `bundle-manifest.json`, `README.md`, and a construction/verification audit.
The assembler writes to a temporary sibling, verifies every copied byte and row mapping, and
publishes the final root atomically. It refuses an existing final root rather than mutating
it. Any later bundle change requires a new versioned root and a method amendment.

### TRIBE array roles

Phase 00 must enumerate and verify the actual schema before later work. The expected roles
are:

- `cortical_prediction`: primary real predicted-cortical representation;
- `temporal_diagnostics53`: video-derived diagnostic/control block and an explicitly named
  fusion candidate, never silently included in the real cortical lane;
- luma, motion, duplicate/black-frame, and quality arrays: audit fields and nuisance-control
  lanes; quality flags remain attached and do not silently delete primary rows;
- timestamps and source-frame/interpolation arrays: row-identity audit fields;
- copied arousal/valence arrays: equality checks only; `rows.csv` remains label authority;
- `tribe_grouped_video_feature`: upstream intermediate excluded from Neural Bridge feature
  discovery unless a future explicit method amendment authorizes a scientifically distinct
  use.

## Method-transfer firewall

The supervised spike-and-continuous structure is defined in
`internal/active/veatic21-supervised-spike-continuous-combination.md`. It was reconstructed by
tracing the winning dependency chain backward through the authorized AGAIN evidence. That
file transfers a dependency structure only.

VEATIC 2.1 must freshly compute every target, split, scaler, projection, response-history
opponent, residual head, checkpoint, control, prediction, and selection decision from the
sealed VEATIC bundle. It must not import, execute, copy, adapt in place, or load any AGAIN
runner, source module, cache row, label, fitted artifact, target array, or prediction.

No source-dataset numeric value is a VEATIC setting. Temporal geometry, response history,
event threshold, projection width, model capacity, loss balance, optimizer, learning rate,
regularization, checkpoint budget, gate limits, seed count, and ensemble size are selected
from VEATIC training ownership. Deterministic evaluation mode, restored checkpoints,
fold-owned transforms, byte-identical frozen opponents, and outcome-independent ensemble
membership are mandatory implementation properties.

## Programme order

1. Phase 00: protected VEATIC input integrity.
2. Phase 01: VEATIC-owned target, temporal-geometry, and ownership derivation.
3. Phase 02: one consolidated supervised arousal build producing independently trained and
   validated spike and continuous specialist ensembles as a single paired package.
4. Phase 03: VEATIC valence build.
5. Phase 04: genuine zero-label-at-inference build.
6. Phase 05: paper, deployment refit, and product/runtime integration.

Event and continuous targets may share row identity and a lossless future-trajectory
primitive. They retain separate target transforms, thresholds, response-history opponents,
losses, heads, checkpoint selection, predictions, metrics, and claims. The paired supervised
package contains both specialists; it does not force them into one compromised head. A
shared-head challenger may replace them only after it beats each specialist on that
specialist's locked endpoint.

## Universal split and leakage contract

### Ownership layers

Every model-bearing phase separates:

- development rows/folds for candidate exploration;
- inner validation for hyperparameter, checkpoint, and boundary decisions;
- outer grouped-video folds for held-out-video generalization;
- blocked-forward folds for within-video forward-time generalization;
- fresh-seed confirmation after a candidate is frozen.

Phase 01 derives the exact VEATIC split design from video count, duration, label/event
support, and uncertainty requirements. Video IDs, row memberships, and hashes are frozen
before model scoring.

Grouped-video and blocked-forward protocols answer different questions and remain separate.
A pass in one cannot overwrite a failure in the other. Selection must be inner-owned within
the corresponding outer protocol. A phase must store immutable outer predictions so results
can be audited without refitting.

A blocked fraction owns eligible time rows inside every sufficiently supported video; it is
not a fraction of videos. Each such video contributes an earlier outer-training segment and
a later untouched outer-test segment. Inner train/validation ownership is another strict
forward-time split inside the outer-training segment. Whole-video withholding belongs to the
grouped protocol. Every video participates across the complete grouped fold set.

### Training ownership

The following are fitted only on their declared training rows:

- target quantiles and event thresholds;
- label/history standardization;
- feature scaling;
- PCA or any learned projection;
- AR models and residualizers;
- neural heads and checkpoint selection;
- calibration transforms;
- permutation mappings and train-only video means.

Test outcomes cannot change target definitions, target windows, folds, features, controls,
model capacity, budgets, convergence rules, seeds, checkpoints, or gates.

### Causality

At prediction row `(v, i)`, a causal feature may use only row `i` and earlier rows from the
same video. Context resets at video boundaries. Missing history is represented by explicit
masks and deterministic padding; it is never filled with another video's values. Future
target rows and any washout gap must be disjoint from response-history features.

## Efficient comprehensiveness contract

Comprehensive means rebuilding every dependency required by the strongest supervised
spike-and-continuous structure, not replaying the sequence that originally discovered it and
not launching an unbounded family grid.

The dependency structure is fixed before cortical scoring. VEATIC development evidence
selects only the dataset-dependent values inside that structure: target geometry, event
threshold, response-history support, projection rank, temporal context, capacity,
optimization, checkpoint count, and ensemble size. Each numeric axis receives a compact
registered local candidate set appropriate to its feasible range plus a boundary-expansion
rule. Categorical alternatives are registered only when they represent a distinct supported
VEATIC hypothesis.

Unit and integrity tests run first. The scientific run then uses every eligible row from all
124 videos under the frozen blocked and grouped ownership. Each required head and matched
control is adequately converged. Nested transforms and lossless shared computations are
computed once per identical ownership. A result may advance only after the complete spike
specialist and continuous specialist each beat their byte-identical response-history
opponent and strongest matched control on blocked and grouped evidence with fresh
independent checkpoints.

Every registered cell receives a terminal disposition. Outer results cannot create new
candidates, alter ensemble membership, or tune weights.

## Controls required from the first applicable phase

Controls are trained and scored with the same split, target, AR floor, optimization budget,
checkpoint rule, seed, and metric code as the real lane unless the control definition
requires a documented difference.

Required families include:

- prevalence/constant and simple current/previous response baselines;
- trailing mean and slope response-history baselines;
- strongest selected target-specific AR;
- real cortical residual over the byte-identical frozen opponent;
- deterministic row-alignment shuffle that breaks representation/outcome correspondence;
- matched random projection or random feature control;
- `temporal_diagnostics53`-only control;
- time-only and mask-only control;
- luma/motion/quality nuisance control;
- train-only video-mean/base-rate control;
- current-row video ablation against causal temporal video context;
- label permutation performed inside training ownership;
- frozen-AR integrity and no-harm controls for residual heads.

For a frozen-AR residual label-permutation lane, chance is the frozen AR floor, not raw event
prevalence: only the residual training labels are permuted while the identical frozen AR
score remains. The null is interpreted as “permuted residual adds nothing beyond AR.”

## Metrics and uncertainty

### Event/spike ranking

Primary:

- pooled held-out PR-AUC/average precision;
- PR-AUC uplift over held-out prevalence;
- delta versus the exact frozen AR;
- delta versus the strongest matched control;
- directional consistency across outer folds, videos where defined, seeds, and checkpoint
  groups.

Secondary:

- ROC-AUC, Brier score for calibrated probabilities, top-1/5/10% recall and precision,
  calibration curves, and predicted-positive rate.

Valid negatives from videos with zero events remain in pooled metrics. Undefined per-video
PR-AUC is reported as undefined, never replaced by zero.

### Continuous arousal and valence

Primary ranking endpoints are Spearman and top-tail true-movement lift at preregistered
fractions. Exact-value MAE, RMSE, bias, Pearson/CCC, and calibration are reported, but exact
trajectory claims require their own preregistered confirmation gate. Event metrics derived
from continuous predictions are supporting unless prospectively declared primary.

Valence must additionally distinguish signed direction, magnitude of change, and level.
Those are separate constructs and cannot be collapsed into whichever metric looks best.

### Uncertainty

Rows within a video are serially dependent. Confidence intervals and paired comparisons
resample whole videos or use fold/video blocks, never IID rows. Final reports include effect
sizes, uncertainty intervals, fold/video/seed consistency, and the maximum contribution of
any single group. Discovery multiplicity is handled by the locked selection/confirmation
separation and a complete result ledger; confirmatory endpoints and gates are prespecified.

## Phase 00 — protected-input foundation

### Question

Can all 124 VEATIC videos be consolidated into one immutable, exact, row-addressable 2 Hz
downstream bundle linking authoritative `rows.csv` records to their matching TRIBE cortical
rows without opening a forbidden hidden-state file or altering any upstream input?

### Required work

- discover numeric video IDs independently in both protected roots and prove exact equality;
- require the complete expected set of 124 IDs;
- enumerate required per-video files and completion statuses;
- verify per-video row counts and `(video_id, row_index)` uniqueness/contiguity;
- verify 2 Hz cadence, timestamps, source-frame positions, interpolation provenance, and
  arousal/valence equality between row authority and TRIBE copies;
- verify `cortical_prediction` shape/dtype/finiteness for every row;
- inventory diagnostics, quality, luma, motion, and provenance arrays and freeze their roles;
- prove that V-JEPA hidden states were neither opened nor hashed;
- compute allowlisted tree/file digests and a complete per-video manifest;
- copy only the allowlisted payload/alignment files into the consolidated bundle, verify
  source/destination SHA-256 equality, and atomically seal the bundle;
- record protected-root deletion guards;
- benchmark only the audit/IO implementation needed for this phase; no PCA or model fitting.

### Gate

All 124 videos and every source row must be accounted for with zero mismatched identities,
nonfinite cortical values, schema deviations, or forbidden reads. Any exclusion is a failure,
not an automatic repair. Phase 00 creates no predictive claim.

## Phase 01 — VEATIC targets, geometry, and ownership

### Question

What VEATIC-specific target families, causal histories, washout gaps, threshold candidates,
and evaluation partitions are supported by the labels before cortical outcomes are read?

### Required work

- materialize an immutable supervised row table from Phase 00 identities;
- retain continuous arousal and valence, interpolation provenance, target masks, quality
  metadata, and same-video causal history availability;
- describe autocorrelation, movement distributions, event support, threshold stability,
  video duration, label dynamics, and cross-video heterogeneity;
- do not cap autocorrelation or temporal candidates at the shortest video's duration; extend
  the lag audit while reporting the declining eligible-video and eligible-pair support, then
  apply a prospectively declared minimum-support rule to candidate geometries;
- derive candidate response-history depths from VEATIC autocorrelation/partial-correlation
  decay;
- derive candidate forecast windows and washout gaps in seconds and 2 Hz rows;
- characterize max positive change, absolute movement, onset/surprise, and signed change;
  emit only the nonzero-washout maximum-positive-arousal event and residualized-continuous
  candidates required by the supervised combination specification;
- derive threshold-quantile candidates from training-side event support;
- reject targets with leakage, insufficient fold/panel support, unstable thresholds, or
  construct ambiguity;
- freeze grouped-video, blocked-forward, and inner-development ownership before Phase 02
  scores; register fresh-seed confirmation membership before outer model scoring;
- create a target-overlap ledger stating history rows, washout rows, and future target rows;
- emit the VEATIC-owned numeric candidate inputs required by the supervised combination
  specification: event and continuous geometry, history depths, threshold support, split
  memberships, and minimum panel support.

No cortical value may influence Phase 01 target or split selection.

## Phase 02 — supervised spike-and-continuous combination

Phase 02 implements
`internal/active/veatic21-supervised-spike-continuous-combination.md` as one dependency-aware
build. Its purpose is to reach the strongest supervised arousal package directly. It does
not replay discovery phases.

### Dependency A: VEATIC target tensors and ownership

For each registered VEATIC geometry, materialize one same-video future-trajectory tensor and
derive the event and continuous views from it. Thresholds, residualizers, masks, blocked
memberships, grouped folds, and inner memberships are fitted or frozen inside their declared
ownership. Event and continuous row masks may share storage only when they are exactly
identical.

### Dependency B: separate response-history opponents

Build a simple train-owned continuous residualizer where the continuous target definition
requires one. Then train two strong response-history heads:

- an event head selected by inner event ranking;
- a continuous head selected by inner continuous ranking and top-tail concentration.

Both use only legal causal response history and explicit availability masks. Their scalers,
checkpoints, and predictions are distinct. Each final opponent prediction is sealed and
reused byte-identically beneath its real residual head and every matched control.

### Dependency C: fold-owned cortical representation

From `cortical_prediction`, construct the registered causal temporal aggregation before the
projection. Fit scaling and projection on the exact owned training rows and transform inner,
outer, and control rows without refitting. Projection rank and causal context are selected
from the VEATIC local candidate set. `temporal_diagnostics53` enters only as an explicitly
named current-row fusion block and as its own control.

### Dependency D: event residual specialist

The event specialist consumes a causal sequence of projected cortical rows plus the explicit
diagnostic block. Its head produces two corrections: one for continuous movement and one for
the event logit. A learned bounded global scale and learned input gate suppress corrections
that do not improve the frozen event opponent. Training uses the registered continuous
regression term plus event-ranking/classification term. Checkpoint selection is inner-owned
and event-primary.

### Dependency E: continuous residual specialist

The continuous specialist is trained independently on its continuous target, opponent,
scaler, projection ownership, loss, checkpoint selection, and seeds. It uses the same
structural head hypothesis only if VEATIC development confirms it. Tail-weighted continuous
regression and ranking-aware inner selection optimize Spearman and top-tail true-movement
lift. Event ranking computed from its continuous prediction is reported explicitly, but
cannot replace the dedicated event specialist unless prospectively promoted as a joint
endpoint.

### Dependency F: independent-checkpoint stabilization

After a single specialist recipe passes inner selection, train fresh independent checkpoints
with membership fixed before outer scoring. Compare individual members with equal-weight
prediction averages. Ensemble size is selected from VEATIC development variability and then
locked; outer outcomes cannot select members or weights.

### Paired supervised output

The Phase 02 deliverable is one versioned package containing:

- the confirmed event specialist ensemble;
- the confirmed continuous specialist ensemble;
- their exact target, split, transform, opponent, head, seed, and checksum identities;
- a common inference interface returning both spike and continuous scores.

This paired package is the supervised spike-and-continuous result. It is not a zero-label
model. A shared-head challenger is optional only after both specialists pass and must match
or beat both locked specialist endpoints before it can replace the pair.

### Phase 02 controls and confirmation

Every real residual lane is compared with the exact frozen opponent and matched shuffled
projection, random projection, diagnostics-only, train-only video-mean, time/mask,
luma/motion/quality, current-row, and training-owned label-permutation controls. Blocked and
grouped evidence remain separate. Fresh confirmation requires positive aggregate uplift over
the opponent and strongest control, directional consistency across folds and checkpoint
groups, no single-group domination, restored-checkpoint evaluation mode, and complete
leakage/checksum audits.

## Phase 03 — valence

After Phase 02 passes, derive valence level, signed direction, movement magnitude, and
transition targets from VEATIC labels. Build valence-specific response-history opponents,
fold-owned cortical transforms, residual heads, controls, and independent-checkpoint
ensembles. Arousal parameters do not transfer automatically. Valence abilities retain
separate targets, metrics, and claims.

## Phase 04 — zero-label at inference

Phase 04 begins only after the supervised arousal pair and valence programme pass.

Training may use labels. Held-out inference receives no arousal or valence value, no
response-history feature, no teacher score, and no labelled warm start. It may receive causal
TRIBE cortical rows, permitted video-derived diagnostics, masks, and video identity for
sequence reset. Development and locked whole-video confirmation ownership are registered at
Phase 04 entry. Predictions are sealed and checksummed before confirmation labels are opened.

The primary candidate is direct supervised causal temporal video-only learning, compared
with current-row, diagnostics-only, no-video, sequence-shuffled, and label-permuted controls.
Observed-label supervised systems are report-only ceilings.

## Phase 05 — paper and product transition

The paper reports the complete candidate/disposition ledger, negative results, protocol-
specific claims, uncertainty, and exact provenance. Only confirmed abilities enter product
work.

A separately identified deployment refit may train a frozen confirmed recipe on 100% of the
available labelled VEATIC videos. That refit cannot estimate held-out accuracy and remains
distinct from every evidence model. Product benchmarking covers cold-start and steady-state
throughput, batch scaling, end-to-end latency, CPU/GPU utilization, unified-memory peak,
deterministic equivalence, raw-video integration, uncertainty, no-harm behavior, and
external client-style validation.

## Hardware-saturation and executor contract

This Mac Studio is a heterogeneous system. “Use all cores” is not a universal prescription;
the fastest valid topology is measured per workload.

Before each computationally material phase:

1. benchmark representative real cells across safe CPU process counts, MLX stream/concurrency
   settings, GPU batch sizes, and CPU/GPU pipelining;
2. include actual loading, preparation, fitting, scoring, serialization, and resume costs;
3. require numerical, convergence, split, metric, and artifact equivalence;
4. record CPU/GPU utilization, throughput, memory, swap, thermal state, and variance over
   repeated measurements;
5. select the fastest safe median configuration, using a small preregistered plateau tie only
   to prefer a less fragile topology;
6. freeze executor code and topology before the main run.

Use MLX GPU for compatible PCA, matrix, and learned-model work. Use multiprocessing and
optimized numerical libraries for CPU-eligible parsing, feature preparation, metrics,
hashing, and reporting. Pipeline CPU preparation with GPU fitting when it improves measured
throughput. Avoid GIL-bound Python loops, library oversubscription, harmful swap, artificial
memory caps, and concurrent writers to one mutable ledger.

Changing worker count, GPU lanes, batching, solver, or coordination after a main run begins
creates a new run identity; outputs from different identities cannot be merged.

## Execution, provenance, and artifacts

- Work only on `main`; create no branches.
- Heavy outputs live only under `/Volumes/onn. Drive/Neural Bridge Artifacts/runs/veatic-2.1/`.
- The lifecycle root is created by the Phase 00 implementation; only artifacts named by the
  current authority may become later-phase inputs.
- Use one coordinator, deterministic disjoint work units, atomic publication, append-only
  shard ledgers, and a verified no-gap/no-duplicate merge.
- Every fitted artifact records code, input, row, split, target, transform, seed, executor,
  configuration, metric, prediction, and checksum identity.
- Resume accepts an artifact only after schema and hash verification.
- Outer predictions are immutable. Never refit merely to recreate a missing score.
- Every phase produces a complete success/failure/disposition ledger and compact concluded
  record; negative and invalid attempts remain visible in the scientific ledger.
- Commit and push each coherent authority or phase transition to `origin/main` before the
  next phase starts.

## Phase progression gate

A phase advances only when:

- its prospective registry is complete and arithmetically reconstructable;
- all registered meaningful families and controls have dispositions;
- leakage, ownership, convergence, checkpoint, prediction, and artifact audits pass;
- fresh confirmation requirements are satisfied for a positive claim;
- a negative claim is supported by adequate optimization rather than undertraining;
- compact evidence and exact external hashes are inspected;
- focused, authority-contract, and full repository tests pass;
- `CURRENT_STATE.md` names one next action;
- the transition is committed and pushed to `origin/main`.

## Current authorization at version 1.2

Phase 00 passed for all 124 videos and 20,657 rows with bundle-manifest SHA-256
`43dca9a25422bcdf08ac440520c0d5db81d850e166d649b85b8a4b43ae419c36`.
The only new scientific phase authorized is Phase 01 registration and implementation. Phase
01 may read authoritative `rows.csv` values and the non-cortical identity, interpolation,
sampling, luma, motion, and quality audit arrays required by its contract. It may not open a
hidden-state NPZ, read `cortical_prediction` or `temporal_diagnostics53` values for target or
split decisions, fit PCA, train AR, score a cortical outcome, or begin Phase 02+.
