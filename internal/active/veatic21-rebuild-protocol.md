# VEATIC 2.1 AGAIN-Method Rebuild Protocol

## Authority and scope

This file is a derived operational checklist and navigation aid for the VEATIC 2.1 research
programme. It is not an independent source of scientific truth. The permanent, comprehensive
scientific authority is
`internal/active/veatic21-master-scientific-specification.md`; read it completely before using
this checklist. Live progress, current authorization, exact result hashes, and the next action
live in `internal/handoff/CURRENT_STATE.md`.

This checklist may summarize but never add, weaken, replace, or override the master
specification. If a method detail is absent here, follow the master rather than inferring
permission. If the two differ, the master wins and this checklist must be corrected before
execution. The scientific method evidence comes from the phase-local records under
`studies/again/`, not from the compact reproduction engine under
`src/neural_bridge/again/`.

AGAIN contributes the successful phase order, scientific questions, control semantics,
fold-ownership rules, evaluation discipline, and progression gates. VEATIC contributes every
dataset-specific number and every fitted artifact: target horizons, event threshold, AR lags
and regularization, PCA width, temporal context, model family, optimizer, seeds, checkpoints,
and any later target redesign.

### Method-only transfer firewall

`studies/again/` is read-only methodological evidence. It may be used to understand the
question, comparison structure, ownership rule, control meaning, metric family, and order of
operations. It is not a runtime or artifact source for VEATIC.

The VEATIC implementation must not import, execute, copy, adapt in place, or load:

- `src/neural_bridge/again/` code or an AGAIN phase runner;
- an AGAIN cache, row table, label, target array, mask, split assignment, PCA/scaler,
  AR model or prediction, residualizer, checkpoint, model/head, control output, or result;
- an AGAIN config value for a horizon, gap, row offset, width, lag, regularization,
  architecture, hidden size, optimizer, seed, checkpoint group, or numeric gate.

Each VEATIC phase is implemented separately in the VEATIC namespace and study directory.
Every fitted or numeric choice must name its VEATIC evidence source, derivation rule, owned
rows, code digest, and artifact digest in a phase-local `veatic-derivation-ledger.json`.
Runtime input manifests must reject paths in AGAIN code, study, output, and artifact roots.
The ledger and path rejection are promotion controls, not optional documentation.

Protocol constants explicitly frozen here—the native 2 Hz row grid, q90 training-fold event
rule, and separately reported grouped-video and blocked-temporal protocols—belong to this
VEATIC protocol. Every split proportion, target window, AR family, representation family,
projection width, temporal context, head, optimizer, budget, seed, checkpoint rule, and
numeric gate is derived and evaluated afresh for VEATIC 2.1. No AGAIN row membership,
configuration, fitted value, or selected result accompanies the method.

The ability order is fixed:

1. arousal spike/event ranking;
2. continuous arousal movement ranking;
3. valence;
4. zero-label at held-out inference.

No later ability begins before the preceding ability has a control-complete result.

## Canonical input boundary

Neural Bridge begins from the complete collection of 124 per-video TRIBE v2 raw cortical
prediction payloads:

`/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/tribe-v2/veatic 2.1 raw cortical predictions/per_video`

For every `<video_id>` from `0` through `123`, the real representation array is:

`<video_id>/tribe_v2_cortical_predictions.npz:cortical_prediction`

There is no single pooled or privileged "final NPZ." Every scientific real lane must use or
explicitly account for every eligible 2 Hz row from all 124 per-video payloads under the
registered split and mask. One file, a convenient video subset, or an unregistered row subset
is not the VEATIC cortical substrate.

The authoritative aligned row, label, and interpolation provenance comes from:

`/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/vjepa-2.1/veatic 2.1 v jepa 2.1 stuff/<video_id>/rows.csv`

The JSON files in each matching V-JEPA directory are allowed metadata/provenance inputs:

- `_PAYLOAD_SHA256.json`
- `_UPLOAD_COMPLETE.json`
- `manifest.json`
- `preprocessing.json`
- `status.json`

`vjepa21_hidden_states.npz` is forbidden. It must not be loaded, memory-mapped, copied,
hashed as part of an allowed-input digest, inspected for keys, or used for any feature,
quality, label, or provenance operation. V-JEPA and TRIBE model execution is complete and is
not part of Neural Bridge.

Inside every per-video TRIBE NPZ:

- `cortical_prediction` is the real Neural Bridge representation array, considered across all
  124 payloads rather than as one singular file;
- copied arousal/valence arrays are comparison copies and are not the label authority;
- `tribe_grouped_video_feature` is not a Neural Bridge representation;
- time, luma, motion, quality, and `temporal_diagnostics53` may be used only in explicitly
  named audit or nuisance-control lanes;
- no nuisance field may be silently fused into the real cortical lane.

## Phase order

### Phase 00 — dense foundation

Audit the immutable input boundary without fitting a model, PCA, AR, target threshold, or
dataset split. Validate the exact video inventory, row identity, row rate, cache layouts,
finite cortical values, per-video manifests, allowed metadata hashes, and quality flags.

Phase 00 may inspect `rows.csv` schema and row identity but does not use arousal or valence
values to make a scientific decision. This is stage separation, not a zero-label claim.
Supervised work begins in Phase 01.

All rows remain present. Quality flags are attached metadata and do not delete rows.

### Phase 01 — exact label alignment and target substrate

Open arousal and valence only from matching V-JEPA `rows.csv`. Verify exact
`(video_id, row_index, time_seconds)` equality against every matching per-video TRIBE
prediction payload and preserve the native interpolation provenance. Do not create a new
interpolation or time shift.

Create continuous future-movement values and masks for all valid rows. The spike family is
the AGAIN form:

`max(arousal[t + h_start : t + h_end]) - arousal[t]`

VEATIC must determine `h_start` and `h_end` from its own 2 Hz label dynamics and video
durations. Store continuous values and validity masks only. Do not fit one global binary
threshold. Event thresholds are training-fold q90 values fitted later.

Phase 01 must also produce the label-only substrate needed for a possible VEATIC washout:

- VEATIC autocorrelation and partial-autocorrelation decay by video;
- predictiveness of current arousal, previous rows, causal trailing means, and causal slope;
- event-duration/rise-time summaries, candidate-window coverage, eligible videos/rows, and
  per-video positive support;
- a bounded rule for deriving candidate gap starts and target-window ends from those VEATIC
  summaries.

This prepares the redesign without assuming one is needed. It does not copy AGAIN's
`rows 4..10`, `1.5s`, `2.5s`, or any other offset. The derivation rule is frozen before any
washout-target cortical score is observed.

There is no outer evaluation split in label alignment. Target construction covers the complete
aligned table; split ownership is applied when a benchmark is fitted and scored.

### Phase 02 — target-specific fresh AR floor

Benchmark the VEATIC spike candidates using fresh target-, protocol-, fold-, and seed-specific
AR models. AR features use current and causal past arousal only. Candidate lag times and
regularization are selected from VEATIC training data through inner validation.

Report grouped held-out-video and blocked forward-time protocols separately. Phase 02 must
derive and justify the split proportions, repeat count, causal history families, feature
forms, model capacities, regularization, optimizer budgets, and calibration plan from VEATIC
2.1 support and development-owned evidence before opening outer results. Binary q90
thresholds, feature normalization, AR regularization, and decision thresholds are fitted
inside the applicable outer-training partition. Test rows never select them.

Freeze the exact AR predictions and checksums for every later matched real/control cell.

Stage A linear convergence follows the master specification's frozen rescue contract. Use
`B = next_power_of_two(sqrt(minimum training-target-valid row count))` and tolerance
`1 / sqrt(training-target-valid row count)`. The complete screen gives ridge `B` updates and
logistic-L2 from `B` through at most `4B`; unresolved cells remain `undertrained`, excluded
from aggregation, and protected from pruning. After exhaustive Stage A artifact verification,
freeze the exact undertrained configuration IDs and original unit hashes in a separate rescue
registry. Restart only those cells from zero with unchanged rows, features, thresholds,
scaling, regularization, solver, precision, step rule, and tolerance at a total maximum `16B`
budget. Link every result through a separate append-only rescue ledger; never mutate the
original Stage A unit or ledger. A cell still unresolved at `16B` is
`invalid_nonconverged_after_registered_maximum_budget`, excluded from aggregation/selection,
and never treated as negative evidence. Do not rerun a converged Stage A cell.

Backtest the sparse rescue workload independently across safe CPU, MLX-lane, Metal-stream,
and compatible cell-batch configurations. Freeze the fastest numerically equivalent executor
only after determinism, resume, ledger, utilization, thermal, memory, and access gates pass.
Outer-test and cortical values remain unopened.

After verified rescue, aggregate a configuration only when every registered inner fold is
complete. Keep converged Stage A records original, use converged rescues only through their
frozen links, and exclude invalid-at-`16B` cells as incomplete rather than negative. Apply
the master specification's exact mean-inner raw-PR-AUC one-standard-error rule, including
finite-before-undefined Brier and smaller history/capacity resolution. From the `126`
form-by-history feature sets, retain exactly `12` Stage B feature sets per target/protocol/
outer fold with all six feature forms and the `1..7`, `8..14`, `15..21` history regions
represented. Register family-specific edge expansions plus the exact elastic-net and
deduplicated OFAT MLP/sequence-only-GRU cells before any Stage B execution.

Benchmark the aggregation executor on real immutable Stage A/rescue JSON across safe process
counts, require normalized identity, and freeze the measured topology before the main
aggregation. Hashing, JSON parsing, compression, and analytic metrics use the measured
parallel CPU path unless a numerically equivalent GPU path is actually demonstrated.

For the starting no-washout target, Phase 02 also begins an AR-dominance and overlap
decomposition on development-owned data: history rows actually consumed by AR, target rows,
the intervening gap, simple causal-history baselines, AR-versus-chance uplift, and fold/video
consistency. Phase 05 extends the same audit with the actual causal context of each head under
evaluation. This is the evidence used by the conditional washout procedure.

### Phase 03 — raw predicted-cortical benchmark

Run a bounded raw cortical test before PCA or learned bridge development. Every cell includes
the complete applicable control stack in the same registered matrix. Report raw cortical,
AR plus raw cortical, cortical-only, and nuisance/control lanes without promoting direct
fusion by default.

The bounded raw experiment must still be a real VEATIC benchmark: it consumes the complete
124-video prediction collection, derives its raw model/regularization/optimization candidates
from VEATIC training-owned evidence, verifies learning-curve adequacy, expands finalists
across fresh seeds, and records every attempted or invalid run. One linear fit or one budget
cannot establish that raw cortical signal fails.

### Phase 04 — fold-owned PCA bridge

Fit scaling and PCA only on rows owned by each outer-training partition. Never fit a global
PCA using held-out videos or held-out temporal rows.

Derive a broad candidate set of reduction families, PCA widths, maximum rank, scaling
variants, causal temporal depths, and aggregation operators from VEATIC training-owned sample
size, spectrum, reconstruction behavior, and development curves. The set must span low,
medium, and high capacity and justify its upper bounds; do not inherit a width grid or temporal
window from AGAIN. A single accurate maximum basis may serve registered nested prefixes, but
it does not define the candidate set. The PCA solver must use fixed fresh seeds, float32
accumulation, adequate oversampling and power iterations, and must pass orthogonality,
explained-variance, reconstruction, and independent-seed subspace-stability audits. If a
maximum-basis prefix is not stable, fit that width separately.

Width is selected by inner validation and complete controls, not by explained variance alone.
No largest prefix is a presumed winner. PCA bases and projected scores are cached and reused
wherever row ownership, target, protocol, fold, and transform identity are identical.

### Phase 05 — VEATIC learned frozen-AR bridge

Discover VEATIC-specific heads and recipes. Do not import an AGAIN head, width, temporal
window, optimizer, checkpoint, seed, or numeric choice.

Register and execute a meaningfully diverse VEATIC head search before a winner or failure can
be declared: linear and nonlinear capacity regions, multiple depths and widths, gated and
temporal candidates where compatible with the representation, residual/fusion forms,
regularization, optimizer and learning-rate schedules, batch/training budgets, checkpoint
cadence, and fresh seed expansion. Bounds and pruning rules come from VEATIC sample size and
training/inner-validation curves. One bottleneck family, one tiny grid, or unresolved
undertraining cannot close Phase 05.

Every learned lane starts from the exact matched frozen AR score. The model may add only a
learned cortical residual. Real and controls within a matched group use identical rows,
target, split, seed, AR floor, model capacity, optimizer, checkpoint policy, and evaluation
mode. Best checkpoints are selected on inner validation, restored, and scored in eval mode.

A no-harm mechanism is mandatory: if a residual cannot earn positive inner-validation value,
it is suppressed and the output falls back to the frozen AR floor.

The first spike comparison remains the clean no-washout reference. A washout is expected to
be a plausible next step, but it is not assumed to win. If the real no-washout bridge clears
its complete control and consistency gates, retain it and report that a washout was
unnecessary.

If the completed decomposition instead shows that legal short-horizon arousal persistence
dominates the task or that the target begins too close to the causal history boundary,
activate the preregistered VEATIC washout-design procedure:

1. Define the actual causal history set used by the strongest VEATIC AR/model at row `t`.
2. Generate a small bounded set of future start offsets from VEATIC label autocorrelation,
   partial autocorrelation, rise time, event duration, video duration, and coverage—not from
   AGAIN offsets or scores.
3. For a start offset `s`, declare rows `t+1..t+s-1` the washout and place the target window
   wholly at `t+s` or later. Record both rows and seconds using the verified VEATIC 2 Hz grid.
4. Reject candidates with inadequate eligible rows, video coverage, positives, or threshold
   stability using criteria frozen before cortical scoring.
5. Choose among surviving candidates only with development-owned training/inner-validation
   data and the complete matched control matrix. Do not select a gap on an outer test score.
6. Freeze the chosen target, gap, masks, splits, gates, and seeds; then refit every
   ownership-dependent AR, scaler, PCA, residualizer, head, and control from VEATIC data.
7. Treat prior held-out results used to motivate redesign as diagnostic. A redesigned target
   earns its claim only on fresh, untouched confirmation evidence under both protocols.

No-washout and washout targets use the same future-maximum-increase construct, fold-specific
q90 event rule, metrics, and controls so the effect of temporal separation is interpretable.

### Phase 06 — event stabilization

After one target/head/recipe passes the complete event controls, stabilize it using declared,
independently trained checkpoint groups and unweighted aligned prediction averaging. Groups,
seeds, lanes, and gates are frozen before confirmation scoring. There is no checkpoint-member
selection or ensemble-weight fitting on held-out results.

### Phase 07 — continuous arousal

Treat continuous movement as a separate ability with its own target-specific AR floor,
training objective, checkpoint selection, matched controls, and confirmation. Do not infer a
continuous win from a spike model's auxiliary regression output.

Primary endpoints are Spearman future-movement ranking and top-5% true-movement lift.
Top-1% and top-10% lift are supporting endpoints. Bias, MAE, and RMSE are reported as a
separate exact-value question and cannot convert a ranking failure into a pass.

### Valence

After continuous arousal, define VEATIC-specific valence questions and gates before fitting.
Valence does not inherit an arousal target formula automatically. It uses the same ownership,
control, checkpoint, and confirmation discipline.

### Zero-label at inference

Zero-label is last. Training remains supervised. Held-out inference must not consume arousal,
valence, response-history availability flags, target values, teacher scores, or labeled warm
starts. Whole-video splits, cold-start reporting, prediction sealing, and the applicable
video-only controls are frozen before locked scoring.

## Comprehensive experiment and search-sufficiency checklist

For every claim-bearing phase and subphase, freeze before outer scoring:

- the exact question, target, 2 Hz rows, ownership, and untouched confirmation boundary;
- the VEATIC evidence that generates every candidate family and numeric bound;
- a broad candidate registry covering the relevant target, AR, representation, temporal,
  head, optimizer, regularization, budget, checkpoint, and seed dimensions;
- staged-budget, pruning, convergence, undertraining, divergence, and invalid-run rules;
- the complete matched control matrix, inner selection metric, uncertainty, promotion gates,
  and search-sufficiency gate.

Search can be staged, but training/inner-validation evidence alone controls pruning and
escalation. Every registered real lane is matched to its applicable controls at the same
stage. Outer and confirmation results never add candidates, tune a setup, select a head, or
end the search early.

Maintain a full append-only experiment ledger. It includes successful, failed, pruned,
divergent, undertrained, excluded, and null runs; exact configurations; code/input/row/split
digests; runtime; learning curves; selected checkpoints; all metrics and controls; and the
predeclared disposition rule. Reports expose the complete candidate/result matrix rather than
only a winner.

The search-sufficiency gate passes only when every registered family is completed or excluded
by a frozen pre-outcome rule, finalists are adequately optimized, fresh-seed and fold/video
stability is quantified, controls are complete, and remaining uncertainty is bounded. If it
does not pass, the conclusion is incomplete. Neither a positive promotion nor a negative
family conclusion is valid merely because one convenient implementation ran.

## Controls from the first applicable cell

Every claim-bearing representation or learned matrix contains the applicable lanes below from
its first registered cell:

- matched target-specific AR or frozen-AR floor;
- real cortical representation/residual;
- cortical-only ablation;
- current-row-only or no-temporal-context ablation;
- shuffled cortical/PCA representation that preserves shape and declared grouping while
  breaking content/row alignment;
- seeded shape-matched random representation;
- train-only causal video-mean/static-content control;
- diagnostics-only control;
- time/video-time-only control;
- quality/motion/luma-only control;
- label-permutation control in which training and inner-validation labels follow the declared
  permutation while held-out labels remain true;
- no-video/architecture ablation where applicable.

For a frozen-AR residual experiment, label permutation is a residual-null over the unchanged
AR floor. It is compared against real and AR; it is not expected to fall to raw event
prevalence.

No stability or confirmation can begin until the comparison controls pass. Controls are never
backfilled after candidate selection.

## Metrics and uncertainty

Spike/event primary metric: raw PR-AUC, computed on the exact same held-out rows for every
matched lane.

Always report:

- event prevalence and analytic chance;
- raw PR-AUC and raw deltas versus AR and strongest matched control;
- average-precision skill only as a cross-prevalence companion;
- ROC-AUC, F1, precision, recall, and Brier score;
- top-1%, top-5%, and top-10% event recall/lift;
- defined-only per-video PR-AUC and the number of undefined single-class videos;
- fold/seed/group consistency counts and medians;
- paired whole-video cluster-bootstrap uncertainty for primary deltas.

Continuous primary metrics are Spearman and top-5% true-movement lift, with top-1%/10%,
Pearson, bias, MAE, and RMSE reported separately.

## Execution and artifact rules

- Work on `main` only.
- Use the measured fastest valid MLX/GPU or parallel CPU path for PCA, learned fitting,
  representation transforms, scoring, and final inference; CPU is not a learned-model
  fallback.
- Before freezing a material run, backtest representative training/inner-only cells across
  safe CPU-worker, MLX execution-lane, and compatible GPU-batch configurations on the actual
  host. Select throughput only after numerical equivalence, determinism, convergence,
  ownership, resumability, and ledger-integrity checks pass. Outer and confirmation outcomes
  remain unopened.
- Use one coordinator, deterministic disjoint work assignment, atomic publication,
  shard-local append-only ledgers, and a verified canonical merge. Never point independent
  workers at one mutable ledger.
- Pipeline CPU feature preparation, metrics, hashing, and artifact writing with MLX work.
  Use available performance cores and unified memory without oversubscribing numerical
  libraries or imposing an artificial low memory cap.
- Freeze host identity, CPU-worker count, MLX lane count, compatible GPU batch size,
  utilization evidence, peak memory, executor digest, and backtest result in the main run
  request. Changing any of them after launch requires a sealed termination and new identity.
- Repeat the saturation/equivalence exercise for deployment inference and report cold-start
  and steady-state latency, throughput, batch scaling, CPU/GPU utilization, and unified-memory
  peak before calling an inference implementation production-ready.
- Reuse VEATIC PCA bases, projected scores, frozen AR predictions, and checkpoints only when
  their complete ownership/provenance identity matches. AGAIN-fitted objects are never
  eligible for reuse.
- Store heavy artifacts only under `/Volumes/onn. Drive/Neural Bridge Artifacts`.
- A run request, split/row digest, code digest, input digest, control matrix, and output
  manifest must be written before a result can be considered resumable or reviewable.
- Replace `internal/handoff/CURRENT_STATE.md` in place whenever scientific state changes;
  preserve its `Mandatory authority anchors` section.
- Commit and push each coherent phase transition to `main` before beginning the next phase.
