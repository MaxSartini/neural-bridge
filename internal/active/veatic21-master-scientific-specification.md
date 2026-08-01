# VEATIC 2.1 Neural Bridge Master Scientific Specification

Specification version: 1.1
Authority reset: 2026-08-01
Status: Phase 00 passed; Phase 01 label/dynamics/split implementation authorized

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
2 Hz predicted-cortical rows, first for arousal events, then continuous arousal, then
valence, and finally zero-label-at-inference deployment.

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

## AGAIN method-transfer firewall

### What transfers

AGAIN supplies a mature question sequence and rigor pattern:

- audit the dense row substrate before modelling;
- align labels and target masks explicitly;
- iterate a genuinely strong target-specific AR opponent;
- test raw predicted-cortical information before sophisticated modelling;
- fit PCA/scalers inside training ownership;
- compare current, difference, and causal temporal representations;
- freeze the exact AR score beneath real and matched residual controls;
- use evaluation mode and restore the selected checkpoint before scoring;
- use washout-gap target families when legal AR persistence dominates the task;
- search causally constrained temporal residual heads;
- use shuffled, random, diagnostics-only, video-mean, time/quality, and label-permutation
  controls with interpretations appropriate to the residual design;
- stabilize selected recipes with predeclared independent-checkpoint averaging;
- specialize continuous ranking rather than copying event settings unchanged;
- attempt zero-label inference only after supervised abilities are established;
- prefer direct supervised video-only temporal learning over repeating distillation and
  self-rollout branches that failed matched controls, unless VEATIC development evidence
  independently reopens those branches.

### What never transfers

VEATIC 2.1 must not import, execute, copy, adapt in place, or load any AGAIN runner, source
module, cache row, label, split, PCA/scaler, checkpoint, prediction, fitted AR model, fitted
head, target array, control output, or result artifact.

AGAIN numerical answers are hypotheses, not VEATIC settings. This includes its exact row
offsets, seconds, horizons, washout length, history lags, threshold quantile, PCA widths,
architecture dimensions, losses, learning rates, regularization, epochs, patience,
optimizer settings, seed counts, checkpoint groups, ensemble weights, and gates. A value may
appear in a VEATIC candidate registry only because a VEATIC derivation rule or a deliberately
declared comparability anchor justifies it. The final selection must be made from VEATIC
development evidence.

Known implementation mistakes are not experiments to repeat. In particular, train-mode
dropout scoring, globally fitted PCA, test-owned thresholding, single-seed promotion,
outcome-selected ensemble membership, and shared-but-nonidentical AR floors are prohibited.

## Programme ability order

1. Arousal event/spike ranking.
2. Stabilized arousal event ranking.
3. Continuous arousal movement ranking and top-tail concentration.
4. VEATIC-specific valence abilities.
5. Zero-label-at-inference arousal and valence lanes.
6. Product/runtime integration and, only after dataset-specific confirmation, a combined
   generalist across VEATIC, AGAIN, and later datasets.

No later ability may borrow an unconfirmed earlier recipe as a fixed answer. It may use the
earlier winner as a registered starting candidate and control.

## Universal split and leakage contract

### Ownership layers

Every model-bearing phase separates:

- development rows/folds for candidate exploration;
- inner validation for hyperparameter, checkpoint, and boundary decisions;
- outer grouped-video folds for held-out-video generalization;
- blocked-forward folds for within-video forward-time generalization;
- fresh-seed confirmation after a candidate is frozen;
- a prospectively reserved zero-label confirmation pool whose outcomes are not opened during
  zero-label development.

Phase 01 derives the exact VEATIC split design and confirmation-pool size from video count,
duration, label/event support, and uncertainty requirements. No AGAIN fold count or split
fraction transfers. Video IDs, row memberships, and hashes are frozen before model scoring.

Grouped-video and blocked-forward protocols answer different questions and remain separate.
A pass in one cannot overwrite a failure in the other. Selection must be inner-owned within
the corresponding outer protocol. A phase must store immutable outer predictions so results
can be audited without refitting.

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

Comprehensive does not mean a blind Cartesian product of every value. Every scientifically
distinct family must receive a fair test, boundaries must be checked, failures must be
recorded, and a negative claim requires adequate optimization. The default search ladder is:

1. **Derive:** use VEATIC-only descriptive evidence to define candidate families and safe
   numeric ranges.
2. **Screen:** compare every distinct family with matched budgets on inner-owned evidence.
3. **Expand boundaries:** extend only axes whose winner lies at a registered boundary or
   whose learning curve is demonstrably undertrained.
4. **Successive promotion:** allocate larger budgets and more seeds only to families that
   survive predeclared no-harm and control gates.
5. **Fresh confirmation:** freeze the complete recipe, then score untouched outer evidence
   once.
6. **Stabilize:** test predeclared checkpoint aggregation only after a single-recipe result
   exists.

Screening cannot eliminate a family on an unconverged curve. Promotion rules, tie-breaking,
minimum effect requirements, and fresh-seed counts must be frozen before the corresponding
scores are opened. Every attempted candidate receives a terminal disposition: promoted,
valid negative, undertrained/incomplete, invalid/leaky, duplicate, or not applicable.

The experiment ledger must permit reconstruction of candidate count from declared axes. It
must also show why known AGAIN dead ends were not rerun and why any reopened branch was
scientifically justified by VEATIC evidence.

## Controls required from the first applicable phase

Controls are trained and scored with the same split, target, AR floor, optimization budget,
checkpoint rule, seed, and metric code as the real lane unless the control definition
requires a documented difference.

Required families include:

- prevalence/constant and simple current/previous response baselines;
- trailing mean and slope response-history baselines;
- strongest selected target-specific AR;
- real cortical-only and AR-plus-real lanes;
- deterministic row-alignment shuffle that breaks representation/outcome correspondence;
- matched random projection or random feature control;
- `temporal_diagnostics53`-only control;
- time-only and mask-only control;
- luma/motion/quality nuisance control;
- train-only video-mean/base-rate control;
- current-row video ablation against causal temporal video context;
- label permutation performed inside training ownership;
- no-video control for zero-label inference;
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

## Phase 01 — alignment, dynamics, targets, and split ownership

### Question

What VEATIC-specific target families, causal histories, washout gaps, threshold candidates,
and evaluation partitions are supported by the labels before cortical outcomes are read?

### Required work

- materialize an immutable supervised row table from Phase 00 identities;
- retain continuous arousal and valence, interpolation provenance, target masks, quality
  metadata, and same-video causal history availability;
- describe autocorrelation, movement distributions, event support, threshold stability,
  video duration, label dynamics, and cross-video heterogeneity;
- derive candidate response-history depths from VEATIC autocorrelation/partial-correlation
  decay, not AGAIN lags;
- derive candidate forecast windows and washout gaps in seconds and 2 Hz rows;
- include a compact comparability anchor for the original VEATIC/AGAIN-style event question,
  but do not privilege it without VEATIC evidence;
- include max positive change, absolute movement, onset/surprise, signed change, and
  AR-residualized continuous candidate families where supported;
- derive threshold-quantile candidates from training-side event support; q90 is an anchor,
  not an inherited winner;
- reject targets with leakage, insufficient fold/panel support, unstable thresholds, or
  construct ambiguity;
- freeze grouped-video, blocked-forward, inner-development, fresh-seed, and zero-label
  confirmation ownership before Phase 02 scores;
- create a target-overlap ledger stating history rows, washout rows, and future target rows.

No cortical value may influence Phase 01 target or split selection.

## Phase 02 — strong target-specific AR opponent

### Question

How much of each viable VEATIC event/movement target is explained by legal response history,
and what is the strongest defensible AR floor that every cortical lane must beat?

### Iteration pattern

AGAIN required five Phase-2 executions before its final reference. VEATIC must inherit the
lesson—baseline construction is part of the result—without ceremonially repeating broken
runs. Phase 02 therefore uses explicit sequential iterations:

1. simple causal baselines and target-support audit;
2. regularized linear/ranking AR screen over VEATIC-derived histories;
3. boundary and convergence expansion;
4. compact nonlinear AR challenger only where inner evidence justifies added capacity;
5. fresh-seed/fold confirmation and frozen prediction seal.

These are scientific gates, not a requirement that exactly five shell commands run.

### Candidate families

- current and previous arousal/valence where allowed by the ability;
- lagged levels, first differences, trailing means, slopes, and availability masks;
- regularized ridge/ranking and logistic event heads;
- a bounded learned AR challenger such as an MLP or causal recurrent head when it can be
  fairly optimized and compared;
- continuous and event objectives trained separately;
- training-owned scaling, thresholding, calibration, and checkpoint choice.

Phase 02 is response-history-only. It intentionally does not use cortical values or
`temporal_diagnostics53`; those begin in Phase 03 as the video-information question. This
separation is what makes later cortical uplift interpretable.

### Efficiency rule

Do not create a millions-of-cells full Cartesian head search. Screen every distinct AR
family, use nested inner selection, expand active boundaries, then spend fresh seeds on
survivors. The search must still be adequate to prevent an intentionally weak AR baseline.

### Gate

Freeze exact target/split/fold/seed-specific AR predictions, checkpoints, scalers, thresholds,
and hashes. A cortical model can later claim value only against the identical relevant frozen
AR floor. No outer score may tune Phase 02.

## Phase 03 — raw predicted-cortical benchmark

### Question

Do the raw 20,484-dimensional predicted-cortical rows, or a deterministic label-free raw
summary, contain target-specific information beyond AR and nuisance controls before PCA or a
learned bridge?

### Required lanes

- frozen AR only;
- raw cortical only;
- AR plus raw cortical;
- raw cortical plus diagnostics and AR plus raw plus diagnostics as explicitly separate lanes;
- diagnostics-only;
- shuffled cortical;
- matched random features/projection;
- timestamp/video-time only;
- luma/motion/quality only.

Phase 03 must benchmark numerically valid full/raw solvers and deterministic label-free
summaries on the actual host. The chosen computational representation cannot use labels or
held-out outcomes. A negative raw result remains valuable and does not stop Phase 04.

## Phase 04 — fold-owned projection and temporal representation discovery

### Question

Which VEATIC-specific train-fold-owned compression and causal temporal representation exposes
useful cortical information without leakage?

### Search families

- PCA and any alternative linear projection justified by VEATIC matrix geometry;
- widths derived from training rank, explained variance, memory/throughput, and boundary
  behavior; AGAIN widths are not copied;
- current projected row;
- projected first difference;
- PCA-then-causal mean/std/slope;
- causal temporal aggregation-then-PCA;
- causal windows derived from Phase 01 label/video dynamics;
- PCA-only, AR-plus-PCA, frozen-AR residual, and explicit diagnostics-fusion lanes;
- the complete matched-control set.

Fit a maximum safe width once per owned training fold and slice nested smaller widths when
mathematically equivalent. Refit for every different outer training ownership. Never use one
global PCA across held-out videos.

### Gate

Freeze one or a small preregistered set of VEATIC-selected representation recipes only after
grouped and blocked evidence, controls, boundary checks, and convergence are complete. A
representation winner is not yet a learned-head claim.

## Phase 05 — learned frozen-AR bridge and event-head discovery

### Phase 05.0: evaluation-safe learned bridge screen

From the first cell, best checkpoints are restored and models are placed in deterministic
evaluation mode before validation/test scoring. Compare direct fusion and residual learning
under matched budgets. Candidate objectives include event, continuous, and carefully bounded
joint losses, but each receives its own metrics and disposition.

### Phase 05.1: frozen-AR residual mechanism

For each fold/seed, compute one frozen AR score and reuse it byte-identically under the real
and every matched residual control. The candidate predicts a correction to that score.
No-harm suppression or gating is selected on inner data only. Report both bridge uplift and
the fraction of rows on which the correction is active.

### Phase 05.2: AR-dominance decomposition

Before blaming the cortical representation, decompose any blocked failure using label/AR
evidence: autocorrelation, simple baseline dominance, window overlap, threshold stability,
and residual variance. This audit does not read new outer cortical outcomes.

### Phase 05.3: optional target redesign

If Phase 01's registered washout targets were not sufficient and the decomposition activates
this branch, evaluate only preregistered VEATIC-specific washout/onset/residual target families.
Refit all ownership-dependent PCA, AR, thresholds, and heads. Do not reuse predictions from a
different target identity.

### Phase 05.4: temporal head-family discovery

Give fair, staged tests to scientifically distinct candidates such as:

- current-row MLP residual;
- delta-feature MLP residual;
- short causal temporal convolution residual;
- gated/low-AR-confidence temporal residual;
- a causal recurrent/attention alternative only when sample size and optimization evidence
  justify it.

Search capacity, regularization, optimizer, learning rate, batch size, loss balance,
checkpoint budget, and context length through one-factor/factorial screens and successive
promotion, not an uncontrolled full product. Every head must face the same controls.

### Phase 05.5: selected event-head confirmation

Freeze target, representation, head, loss, AR recipe, controls, seeds, checkpoint rule, and
gates. Run fresh blocked-forward and grouped-video confirmation separately. The seed count is
derived from VEATIC development variability and desired uncertainty, not inherited from
AGAIN. Promotion requires positive aggregate effect, matched-control superiority, fold/seed
consistency, no single-group domination, and all leakage/integrity audits.

## Phase 06 — event stabilization

### Question

Can the selected event recipe become a repeatable procedure rather than a lucky checkpoint?

The first candidate is a predeclared equal-weight average of independently trained reference
recipe checkpoints, because that technique survived AGAIN's fresh evidence. VEATIC must still
derive its checkpoint count and confirm that averaging helps. Compare the ensemble with its
members, frozen AR, and every matched control. Do not begin with single-seed Optuna or
outcome-selected blends; AGAIN already showed those are unreliable strategies. Reopen a
hyperparameter-search branch only if VEATIC inner evidence establishes a specific need.

Run fresh blocked confirmation first, then a separately locked grouped confirmation if its
gate passes. Ensemble membership and weights are fixed before confirmation.

## Phase 07 — continuous arousal specialization

Continuous arousal is a new experiment programme, not the event head with a renamed output.

- derive VEATIC-specific continuous target/window candidates;
- fit a target-specific continuous AR with ranking-aware inner selection;
- independently select residual target, loss, context, head, and checkpoint recipe;
- use Spearman and top-tail true-movement lift as primary ranking endpoints;
- treat MAE/RMSE/bias/exact values as a separate candidate claim;
- run matched-control blocked and grouped confirmation;
- stabilize with fresh independent checkpoints only after a single-recipe candidate passes.

A grouped pass does not erase a blocked failure, and a ranking pass does not prove exact
trajectory forecasting.

## Phase 08 — VEATIC-specific valence programme

AGAIN cannot supply valence answers. VEATIC valence therefore receives its own full ladder:

1. valence label dynamics, reliability, interpolation, target families, and split support;
2. level, signed-change, absolute-change, onset/transition, and top-tail target comparison;
3. strong target-specific valence AR/history opponents;
4. raw cortical, diagnostics, nuisance, shuffled, and random controls;
5. fold-owned projection/temporal representation discovery;
6. learned direct/residual head discovery;
7. fresh blocked and grouped confirmation;
8. stabilization and calibration where justified.

Signed direction, movement magnitude, and valence level are reported separately. Valence
cannot inherit the arousal winner without an explicit challenger test.

## Phase 09 — genuine zero-label-at-inference lane

This phase begins only after event, continuous arousal, and valence have control-complete
supervised results.

### Definition

Training remains supervised. Held-out inference receives no current/past arousal or valence,
no response-history feature, no teacher score, no labeled warm start, and no held-out label
before predictions are sealed. It may receive causal TRIBE cortical rows, permitted
video-derived diagnostics, time/masks, and video identity needed only for sequence reset.

### Stage 0: prospective freeze

Freeze development and confirmation video ownership, target identities, feature allowlist,
forbidden columns, row-0 cold-start behavior, PCA ownership, prediction-before-label seal,
lanes, seeds, ensemble rule, metrics, and gates. Fit no model.

### Stage A: development

The primary mature candidate is direct supervised causal temporal video-only learning. Test
it against:

- current-row video model;
- diagnostics-only temporal model;
- no-video time/mask model;
- sequence-shuffled video model;
- label-permuted supervised model.

Distillation and self-rollout are not default candidates because AGAIN eliminated them under
matched controls. They may be reopened only by a preregistered VEATIC-specific rationale.
Observed-label AR-assisted systems are report-only ceilings, never pass thresholds.

### Stage B: locked confirmation

Train the frozen recipe on development ownership, generate and checksum all confirmation
predictions before opening confirmation labels, then score once. Require video-block
uncertainty, full-video and cold-start slices, control superiority, and panel consistency.
PCA/scalers fit on development videos only.

## Phase 10 — paper and product transition

The scientific paper reports the full candidate/disposition ledger, negative results,
protocol-specific claims, uncertainty, and exact provenance. Only confirmed abilities enter
product work.

The product stage separately benchmarks:

- cold-start and steady-state throughput;
- batch-size scaling and end-to-end latency;
- CPU/GPU utilization and unified-memory peak;
- deterministic equivalence to the scientific reference;
- raw-video -> upstream stack -> Neural Bridge integration;
- uncertainty and no-harm output behavior;
- external/cross-domain and prospective client-style validation.

Precomputed VEATIC inputs are sufficient for the paper experiments; they do not by themselves
establish end-to-end client-video runtime.

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
creates a new run identity; the old attempt cannot be silently merged.

## Execution, provenance, and artifacts

- Work only on `main`; create no branches.
- Heavy outputs live only under `/Volumes/onn. Drive/Neural Bridge Artifacts/runs/veatic-2.1/`.
- The new lifecycle root is created by the Phase 00 implementation after its registration is
  committed; no previous VEATIC run root is an input.
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

## Current authorization at version 1.1

Phase 00 passed for all 124 videos and 20,657 rows with bundle-manifest SHA-256
`43dca9a25422bcdf08ac440520c0d5db81d850e166d649b85b8a4b43ae419c36`.
The only new scientific phase authorized is Phase 01 registration and implementation. Phase
01 may read authoritative `rows.csv` values and the non-cortical identity, interpolation,
sampling, luma, motion, and quality audit arrays required by its contract. It may not open a
hidden-state NPZ, read `cortical_prediction` or `temporal_diagnostics53` values for target or
split decisions, fit PCA, train AR, score a cortical outcome, or begin Phase 02+.
