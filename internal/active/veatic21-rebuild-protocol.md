# VEATIC 2.1 AGAIN-Method Rebuild Protocol

## Authority and scope

This protocol and `internal/handoff/CURRENT_STATE.md` are the active authority for the
VEATIC 2.1 research programme. The scientific method comes from the phase-local records under
`studies/again/`, not from the compact reproduction engine under `src/neural_bridge/again/`.

AGAIN contributes the successful phase order, scientific questions, control semantics,
fold-ownership rules, evaluation discipline, and progression gates. VEATIC contributes every
dataset-specific number and every fitted artifact: target horizons, event threshold, AR lags
and regularization, PCA width, temporal context, model family, optimizer, seeds, checkpoints,
and any later target redesign.

The ability order is fixed:

1. arousal spike/event ranking;
2. continuous arousal movement ranking;
3. valence;
4. zero-label at held-out inference.

No later ability begins before the preceding ability has a control-complete result.

## Canonical input boundary

Neural Bridge begins from the completed TRIBE v2 predicted-cortical cache:

`/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/tribe-v2/compact-20260716`

Per-video real representation:

`per_video/<video_id>/tribe_v2_cortical_predictions.npz:cortical_prediction`

The authoritative aligned row, label, and interpolation provenance comes from:

`/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/vjepa-2.1/compact-20260716/<video_id>/rows.csv`

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

Inside the final TRIBE NPZ:

- `cortical_prediction` is the only real Neural Bridge representation;
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
`(video_id, row_index, time_seconds)` equality against the final TRIBE cortical cache and
preserve the native interpolation provenance. Do not create a new interpolation or time shift.

Create continuous future-movement values and masks for all valid rows. The spike family is
the AGAIN form:

`max(arousal[t + h_start : t + h_end]) - arousal[t]`

VEATIC must determine `h_start` and `h_end` from its own 2 Hz label dynamics and video
durations. Store continuous values and validity masks only. Do not fit one global binary
threshold. Event thresholds are training-fold q90 values fitted later.

There is no outer 70/30 split in label alignment. Target construction covers the complete
aligned table; split ownership is applied when a benchmark is fitted and scored.

### Phase 02 — target-specific fresh AR floor

Benchmark the VEATIC spike candidates using fresh target-, protocol-, fold-, and seed-specific
AR models. AR features use current and causal past arousal only. Candidate lag times and
regularization are selected from VEATIC training data through inner validation.

Report grouped held-out-video and blocked forward-time 70/30 protocols separately. Binary
q90 thresholds, feature normalization, AR regularization, and decision thresholds are fitted
inside the applicable outer-training partition. Test rows never select them.

Freeze the exact AR predictions and checksums for every later matched real/control cell.

### Phase 03 — raw predicted-cortical benchmark

Run a bounded raw cortical test before PCA or learned bridge development. Every cell includes
the complete applicable control stack in the same registered matrix. Report raw cortical,
AR plus raw cortical, cortical-only, and nuisance/control lanes without promoting direct
fusion by default.

### Phase 04 — fold-owned PCA bridge

Fit scaling and PCA only on rows owned by each outer-training partition. Never fit a global
PCA using held-out videos or held-out temporal rows.

The default VEATIC width experiment fits one accurate maximum rank-512 basis per
outer-training partition using every owned training row, then evaluates the nested prefixes
`64`, `128`, `256`, and `512`. The PCA solver must use fixed seeds, float32 accumulation,
adequate oversampling and power iterations, and must pass orthogonality, explained-variance,
reconstruction, and independent-seed subspace-stability audits. If the maximum-basis prefixes
are not stable, fit the affected widths separately.

Width is selected by inner validation and complete controls, not by explained variance alone.
The 512 prefix is a candidate, never a presumed winner. PCA bases and projected scores are
cached and reused wherever row ownership, target, protocol, fold, and transform identity are
identical.

### Phase 05 — VEATIC learned frozen-AR bridge

Discover VEATIC-specific heads and recipes. Do not import an AGAIN head, width, temporal
window, optimizer, checkpoint, seed, or numeric choice.

Every learned lane starts from the exact matched frozen AR score. The model may add only a
learned cortical residual. Real and controls within a matched group use identical rows,
target, split, seed, AR floor, model capacity, optimizer, checkpoint policy, and evaluation
mode. Best checkpoints are selected on inner validation, restored, and scored in eval mode.

A no-harm mechanism is mandatory: if a residual cannot earn positive inner-validation value,
it is suppressed and the output falls back to the frozen AR floor.

A washout gap or other target redesign is not part of the starting spike task. It may be
introduced only after a completed controlled AR-dominance and target-overlap decomposition
demonstrates that redesign is necessary. Any redesign uses VEATIC-derived row/second values.

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
- Use exactly one MLX GPU worker for PCA and learned training. Parallel GPU runs do not help.
- Use available memory; do not impose an artificial memory cap.
- CPU is permitted for CSV/JSON audit, deterministic orchestration, metrics, and reporting.
- Reuse PCA bases, projected scores, frozen AR predictions, and checkpoints only when their
  complete ownership/provenance identity matches.
- Store heavy artifacts only under `/Volumes/onn. Drive/Neural Bridge Artifacts`.
- A run request, split/row digest, code digest, input digest, control matrix, and output
  manifest must be written before a result can be considered resumable or reviewable.
- Replace `internal/handoff/CURRENT_STATE.md` in place whenever scientific state changes.
- Commit and push each coherent phase transition to `main` before beginning the next phase.
