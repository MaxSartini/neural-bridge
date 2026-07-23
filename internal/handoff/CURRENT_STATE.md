# Current State — VEATIC 2.1 AGAIN-Method Rebuild

Updated: 2026-07-23

## Read this first

VEATIC 2.1 is at a clean pre-Phase-00 boundary. No current Phase 00 has been implemented,
executed, or promoted. The exact next action is to implement the new Phase 00 dense-foundation
audit described below, run it against the canonical caches, review its evidence, commit and
push the completed phase to `main`, and only then begin Phase 01.

This handoff is intentionally comprehensive. Do not infer scientific semantics, paths,
controls, targets, widths, models, or progression rules from files not named here.

## Authority order

1. This file.
2. `internal/active/veatic21-rebuild-protocol.md`.
3. The phase-local scientific record under `studies/again/`, used as methodological source.
4. New VEATIC phase-local study records created under `studies/veatic-2.1/` after each phase
   becomes defensible.

The compact reproduction code under `src/neural_bridge/again/` is not scientific authority for
the VEATIC rebuild. Do not use it to infer omitted design choices. Read the study-local AGAIN
README, evidence, reports, manifests, metrics, and phase-local runners described below.

AGAIN transfers method and rigor only. No AGAIN-fitted PCA, AR model, head, optimizer, target
seconds, row offsets, thresholds, widths, seeds, checkpoints, or numeric winners may be used as
a VEATIC selection.

### Hard method-only transfer firewall

The phrase **method transferred** in this handoff means that VEATIC may reproduce the
scientific question, comparison design, ownership rule, control meaning, metric family, or
order of operations. It never means reuse of an AGAIN implementation or fitted object.

The new programme must not import, execute, copy, adapt in place, or load:

- code from `src/neural_bridge/again/`;
- a runner from any AGAIN study phase;
- an AGAIN cache, feature row, row table, label, target, mask, or split membership;
- an AGAIN PCA/scaler, projected score, AR model/prediction, residualizer, checkpoint,
  model/head, control output, or result;
- an AGAIN horizon, washout, seconds, row offset, lag, regularization, width, temporal window,
  architecture, hidden size, optimizer, seed, checkpoint group, or numeric result gate.

All VEATIC source and tests are implemented separately under the VEATIC namespace and
`studies/veatic-2.1/`. Runtime manifests reject paths under AGAIN code, study, output, and
artifact roots. Every phase emits a `veatic-derivation-ledger.json` mapping every fitted and
numeric choice to its VEATIC evidence, derivation rule, owned rows, code digest, and artifact
digest. A phase cannot advance when that ledger or the AGAIN-path rejection audit is absent.

Method constants explicitly declared by this current VEATIC protocol—native 2 Hz rows,
training-fold q90 event thresholding, separate grouped and blocked 70/30 reporting, and the
declared PCA candidate set—are current VEATIC rules. They do not authorize reuse of AGAIN row
membership, arrays, fitted values, or selected results.

## Repository and execution invariants

- Repository: `/Users/maxsartini/Neural Bridge`.
- Branch: `main` only. Do not create a branch.
- Heavy-artifact root: `/Volumes/onn. Drive/Neural Bridge Artifacts`.
- All PCA and learned training uses MLX with exactly one GPU worker.
- Do not start parallel GPU training processes. There is one useful GPU execution path.
- Do not impose an artificial memory cap; the available GPU memory should be used.
- CPU is allowed for CSV/JSON parsing, deterministic audits, orchestration, metrics, hashing,
  and report generation. It is not the learned-model fallback.
- Reuse VEATIC-fitted artifacts when and only when the complete
  row/split/target/fold/transform identity matches. The method-only firewall always forbids
  reuse of an AGAIN-fitted artifact.
- Commit and push each coherent phase transition to `main` before starting the next phase.
- Replace this file in place whenever active scientific state changes. Do not append a diary.

## Fixed ability order

The programme order is:

1. arousal spike/event ranking;
2. continuous arousal movement ranking;
3. valence;
4. zero-label at held-out inference.

No later ability begins before the previous ability has a control-complete result. Zero-label
does not apply to Phase 00 through valence training. Training is supervised. Zero-label means
that the final held-out inference process consumes no response labels or observed response
history.

## Canonical VEATIC input roots

### Final TRIBE v2 predicted-cortical cache

Root:

`/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/tribe-v2/compact-20260716`

Per-video folder:

`/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/tribe-v2/compact-20260716/per_video/<video_id>`

Per-video files:

- `manifest.json`
- `status.json`
- `tribe_v2_cortical_predictions.npz`

Run status:

`/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/tribe-v2/compact-20260716/per_video/run_status.json`

The real Neural Bridge representation is exactly:

`tribe_v2_cortical_predictions.npz:cortical_prediction`

Verified TRIBE tree identity:

- SHA-256 tree: `0d4adc27dd9d226de87d0cfc4df92de14cb7450de6671857e0665418ad26f6dd`
- files: `373`
- symlinks: `0`
- size: `866,111,964` bytes

### Matching V-JEPA row, label, and metadata cache

Root:

`/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/vjepa-2.1/compact-20260716`

Per-video folder:

`/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/vjepa-2.1/compact-20260716/<video_id>`

Allowed files:

- `_PAYLOAD_SHA256.json`
- `_UPLOAD_COMPLETE.json`
- `manifest.json`
- `preprocessing.json`
- `rows.csv`
- `status.json`

The allowed-file-only metadata tree was independently audited without reading or hashing any
hidden-state NPZ:

- SHA-256 tree: `cee65f87ff1e118353acd0c6f86c7f8c925e4e612b47884caea0544f6250e1cd`
- files: `744`
- size: `7,103,590` bytes
- hidden-state files read: `false`

The digest above covers only the six allowed filenames in every video directory. Reproduce
that allowlisted digest; do not hash the entire V-JEPA root.

### Absolute forbidden-input rule

Every file named `vjepa21_hidden_states.npz` is forbidden.

Do not:

- call `np.load` on it;
- memory-map it;
- inspect its keys;
- hash it as part of an allowed-input or tree digest;
- copy it;
- read quality arrays from it;
- read labels from it;
- use its `features` or any hidden-state tensor;
- rerun V-JEPA or TRIBE.

The hidden-state cache was the upstream input used to produce the final TRIBE cortical cache.
Neural Bridge starts after that process.

## Exact role of every relevant final-TRIBE array

The final TRIBE NPZ has one uniform key set across all 124 videos:

- `time_seconds`
- `sample_frame_indices`
- `sample_time_seconds`
- `selected_state_indices`
- `source_frame_position`
- `source_floor_frame_index`
- `source_ceil_frame_index`
- `source_interp_alpha`
- `source_arousal`
- `source_valence`
- `arousal`
- `valence`
- `luma_mean`
- `luma_std`
- `frame_luma_std_mean`
- `motion_absdiff_mean`
- `black_frame_fraction`
- `duplicate_frame_fraction`
- `quality_black_frame_flag`
- `quality_duplicate_frame_flag`
- `quality_exclusion_flag`
- `quality_weight_suggested`
- `cortical_prediction`
- `tribe_grouped_video_feature`
- `temporal_diagnostics53`

Role classification:

| Array or family | Role |
| --- | --- |
| `cortical_prediction` | Sole real Neural Bridge representation |
| `time_seconds` | Row-identity cross-check |
| copied arousal/valence and source-label arrays | Equality audit only; not label authority |
| luma, motion, quality arrays | Audit or explicitly named nuisance-control lanes only |
| `temporal_diagnostics53` | Explicit diagnostics-only control if used; not silently fused into real |
| `tribe_grouped_video_feature` | Excluded from Neural Bridge representation search |
| frame/sample indices and times | Provenance/audit only |

The authoritative labels and label interpolation provenance are the matching V-JEPA
`rows.csv`, not the duplicated arrays inside the final TRIBE NPZ.

## Verified substrate facts

The research audit established all of the following without opening a hidden-state NPZ:

- TRIBE numeric video folders: exactly `0..123`.
- V-JEPA numeric video folders: exactly `0..123`.
- Cross-root video-ID equality: exact.
- Completed videos: `124/124`.
- Total aligned rows: `20,657`.
- Minimum rows per video: `22`.
- Median rows per video: `156.5`.
- Maximum rows per video: `358`.
- Mean rows per video: `166.5887096774`.
- Every video begins at `0.0s`.
- Every video advances in exact `0.5s` steps.
- Row rate: `2 Hz` for all `20,657` rows.
- Native label frame rate: `25 fps` for all rows.
- `10,357` label rows are `native_exact`.
- `10,300` label rows are `linear_native_frames`.
- All row identities are sequential and unique within video.
- Every cortical array is `[rows, 20,484]`, float16, and finite.
- Every grouped upstream feature is `[rows, 2, 1,408]`; it is not a Neural Bridge input.
- Every copied temporal diagnostic array is `[rows, 53]`, float32, and finite.
- All 124 final-TRIBE NPZ files have one identical key schema.
- TRIBE and V-JEPA manifest/status pairs agree per video.
- All allowed V-JEPA JSON/CSV byte counts and recorded SHA-256 values agree.
- Every V-JEPA upload marker agrees with its payload-manifest hash.
- Final-TRIBE copied time equals `rows.csv` time exactly.
- Final-TRIBE copied arousal/valence and source labels agree with `rows.csv` to float32
  precision; maximum absolute difference is below `3e-8`.
- Final-TRIBE source frame positions and interpolation alphas agree with `rows.csv` exactly.
- Arousal observed range: `[-0.6425546437, 0.8886752389]`.
- Valence observed range: `[-0.8771958318, 0.8212311904]`.

Video last-time distribution in seconds:

- minimum: `10.5`
- p10: `39.05`
- p25: `57.375`
- median: `77.75`
- p75: `103.625`
- p90: `133.0`
- maximum: `178.5`

## Quality-flag semantics

The `923` number is the count of unique quality-flagged rows, not the dataset row count and not
a preselected training subset.

- black-frame rows: `76`
- extreme-duplicate/static rows: `871`
- rows flagged by both: `24`
- union: `923`
- unflagged rows: `19,734`
- total rows retained: `20,657`

The black threshold is `black_frame_fraction >= 0.50`. The duplicate threshold is
`duplicate_frame_fraction >= 0.95`.

All rows remain in the canonical table and feature substrate. Phase 00 records the flags but
does not exclude rows. Target construction does not silently delete them. If a quality-filtered
sensitivity is later needed, it is an explicitly registered benchmarking analysis with matched
rows/lanes and separately reported results.

## Authoritative `rows.csv` schema

Every matching V-JEPA `rows.csv` uses these columns:

1. `video_id`
2. `video_name`
3. `video_relpath`
4. `arousal_relpath`
5. `valence_relpath`
6. `row_index`
7. `time_seconds`
8. `row_hz`
9. `clip_start_seconds`
10. `clip_end_seconds`
11. `native_label_fps`
12. `native_label_frame_count`
13. `source_frame_position`
14. `source_floor_frame_index`
15. `source_ceil_frame_index`
16. `source_interp_alpha`
17. `source_arousal`
18. `source_valence`
19. `source_match_quality`
20. `encode_policy`
21. `arousal`
22. `valence`

Observed encode policy:

`exact_2hz_native_label_support_no_extrapolation`

Do not shift, interpolate, smooth, extrapolate, or repair these labels again.

## VEATIC descriptive label dynamics already measured

These values are research context for designing Phase 01. They are not a selected target,
threshold, AR specification, or promotion result.

Pooled arousal level correlation by lag:

| Lag | Correlation |
| --- | ---: |
| 0.5s | 0.997522 |
| 1.0s | 0.993558 |
| 1.5s | 0.988242 |
| 2.0s | 0.982072 |
| 3.0s | 0.968007 |
| 4.0s | 0.952686 |
| 5.0s | 0.937028 |
| 6.0s | 0.921147 |
| 8.0s | 0.888576 |
| 10.0s | 0.856896 |

Pooled arousal absolute-movement q90:

| Horizon | q90 absolute movement |
| --- | ---: |
| 0.5s | 0.023635 |
| 1.0s | 0.039136 |
| 1.5s | 0.053729 |
| 2.0s | 0.067241 |
| 3.0s | 0.092449 |
| 4.0s | 0.115140 |
| 5.0s | 0.135978 |
| 6.0s | 0.156163 |
| 8.0s | 0.191263 |
| 10.0s | 0.222405 |

Descriptive future-maximum-increase candidates:

| Future rows | Seconds | Valid rows | Global descriptive q90 |
| --- | --- | ---: | ---: |
| 1..2 | 0.5..1.0s | 20,409 | 0.030246 |
| 1..4 | 0.5..2.0s | 20,161 | 0.053332 |
| 2..4 | 1.0..2.0s | 20,161 | 0.053286 |
| 2..6 | 1.0..3.0s | 19,913 | 0.075080 |
| 3..6 | 1.5..3.0s | 19,913 | 0.074850 |
| 4..8 | 2.0..4.0s | 19,665 | 0.094731 |
| 4..10 | 2.0..5.0s | 19,417 | 0.113669 |
| 6..12 | 3.0..6.0s | 19,169 | 0.131268 |
| 8..16 | 4.0..8.0s | 18,673 | 0.167354 |

Do not turn one of these global q90 values into the event threshold. Binary thresholds are
fitted inside each outer-training partition. Phase 01 must write and freeze the VEATIC target
window selection procedure before any cortical benchmark is read.

## Study-local AGAIN research map

Read these directories directly. The notes below capture the successful methodological spine
that must be transferred to VEATIC.

### AGAIN Phase 00 — dense foundation

Path:

`studies/again/phase-00-dense-foundation`

Key evidence:

- `README.md`
- `evidence/README.md`
- `evidence/metadata/BASELINE_READINESS.md`
- `evidence/metadata/README_OUTPUT_SCHEMA.md`
- `evidence/metadata/global_run_metadata.json`
- `evidence/metadata/summary_report.json`
- `evidence/reports/dense-encoding-audit-20260625.md`
- `runners/build_dense_tribe_postpass.py`

Method transferred:

- foundation before modeling;
- exact saved row identity and row rate;
- complete per-video manifests;
- explicit quality flags retained on source rows;
- no PCA, AR, target benchmark, or bridge training in Phase 00;
- downstream work starts from completed cortical predictions rather than rerunning upstream
  encoders.

AGAIN reference scale was `995/995` videos and `243,575` rows at true 2 Hz. This is context,
not a VEATIC numeric prior.

The Phase 00 postpass runner is the only phase-local Python runner under `studies/again/`.
Later phase authority is captured by study-local evidence, reports, plans, metrics, and
manifests rather than phase-local source files.

### AGAIN Phase 01 — label alignment

Path:

`studies/again/phase-01-label-alignment`

Key evidence:

- `README.md`
- `evidence/README.md`
- `evidence/dense_root_metadata/labels_aligned_2hz_summary.json`
- `evidence/reports/again_labels_aligned_2hz_20260625_091209.md`

Method transferred:

- saved feature timestamps are authoritative;
- retain the true 2 Hz label movement rather than collapsing to 1 Hz;
- store continuous future target values and masks;
- keep unmatched rows explicit and do not impute them;
- fit binary event thresholds inside benchmark training folds;
- calculate target windows and movement scales for the actual dataset.

### AGAIN Phase 02 — target-specific AR baseline

Path:

`studies/again/phase-02-ar-baseline`

Key evidence:

- `README.md`
- `evidence/README.md`
- `evidence/final/run_manifest.json`
- `evidence/final/summary.json`
- `evidence/final/again_dense_2hz_ar_fold_metrics.csv`
- `evidence/final/again_dense_2hz_ar_summary_metrics.csv`
- `evidence/reports/again_dense_2hz_ar_baseline_20260625_093722.md`

Method transferred:

- train a strong target-specific AR floor;
- choose AR regularization by inner validation inside each outer-training partition;
- report grouped-video and blocked-temporal protocols separately;
- fit q90 event thresholds using outer-training continuous values only;
- freeze exact target/fold/seed AR predictions for later matched lanes.

### AGAIN Phase 03 — raw cortical benchmark

Path:

`studies/again/phase-03-raw-cortical`

Key evidence:

- `README.md`
- `evidence/README.md`
- `evidence/final/run_manifest.json`
- `evidence/final/promotion_gates.json`
- `evidence/representation-metadata/raw_cortical_block_summary_b256.json`
- `evidence/representation-metadata/temporal_diagnostics_summary_features.json`
- `evidence/reports/again_dense_2hz_raw_cortical_vs_ar_20260625_094242.md`

Method transferred:

- test raw cortical signal before learned bridge complexity;
- compare cortical-only and AR-plus-cortical against AR;
- include shuffled, random, time, quality/motion/luma, and diagnostics controls in the
  representation benchmark;
- treat direct raw fusion as a baseline question, not the final bridge claim.

The AGAIN spike reference showed why the bridge was needed: grouped raw cortical PR-AUC was
`0.136579` versus AR `0.147251`, while grouped AR-plus-raw was `0.170299`; blocked raw and
AR-plus-raw remained below the stronger blocked AR floor. These numbers explain the method and
do not select a VEATIC representation.

### AGAIN Phase 04 — fold-safe PCA bridge

Path:

`studies/again/phase-04-pca-bridge`

Key evidence:

- `README.md`
- `evidence/README.md`
- `evidence/benchmark/run_manifest.json`
- `evidence/benchmark/summary.json`
- `evidence/benchmark/pca_feature_manifest.json`
- `evidence/benchmark/diagnostics/pca_fit_diagnostics.json`
- `evidence/benchmark/diagnostics/split_leakage_audit.json`
- `evidence/benchmark/diagnostics/transform_leakage_audit.json`
- `evidence/benchmark/promotion/best_lanes_by_target.json`
- `evidence/benchmark/promotion/promotion_gates.json`
- `evidence/reports/again_dense_2hz_phase4_pca_bridge_benchmark_20260625_153419.md`
- `evidence/reports/again_dense_2hz_phase4_pca_promotion_summary_20260625_153419.md`

Method transferred:

- scaling and PCA are outer-training-only;
- no global PCA may see held-out feature rows;
- feature families and widths are compared under identical splits and controls;
- target, protocol, fold, row mask, scaler, PCA, and score identities are recorded;
- width is selected by controlled held-out behavior, not explained variance alone;
- the selected PCA is locked before learned-head work.

AGAIN examined widths `64/128/192/256` and multiple temporal families. VEATIC does not inherit
those widths or the AGAIN winning 2-second/PCA256 configuration.

### AGAIN Phase 05/5.5 — learned frozen-AR bridge and event head

Path:

`studies/again/phase-05-learned-bridge`

Read:

- `README.md`
- `evidence-ladder.md`
- every nested evidence `README.md`
- `plans/selected-head-420-confirmation.md`
- all promotion gate JSONs and reports under `evidence/`

Especially important evidence:

- `evidence/phase_5_0_evalmode_rescore/`
- `evidence/phase_5_1_frozen_ar_residual/`
- `evidence/phase_5_2_blocked_ar_decomposition_20260630_001424/`
- `evidence/phase_5_3_target_redesign_20260630_003224/`
- `evidence/phase_5_3_redesigned_target_foldsafe_pca_20260630_005312/`
- `evidence/phase_5_4_temporal_residual_diagnostic_20260630_020557/`
- `evidence/phase_5_5_binary_blocked_confirmation_20260630_025437/`
- `evidence/phase_5_5_grouped_compatibility_20260630_033520/`
- `evidence/phase_5_5_selected_head_420_confirmation_20260714_124953/`

Method transferred:

- restore the best checkpoint and score in eval mode;
- make the exact target/fold/seed AR output an immutable residual floor;
- reuse that identical AR floor across real and every matched control;
- train cortical signal as a residual correction rather than allowing fusion to destroy AR;
- include a no-harm suppression/fallback to the AR floor;
- interpret label permutation correctly when the residual lane retains an AR floor;
- use train-only video means rather than full-video static controls for promotability;
- analyze AR dominance and target-window overlap before considering target redesign;
- preserve the no-washout task as the first controlled reference while preparing a bounded,
  VEATIC-derived washout procedure;
- refit fold-safe PCA whenever a redesigned target changes split ownership;
- confirm a selected target/head across seeds and both protocols before stabilization.

The final AGAIN selected-head evidence used seven matched lanes:

- real residual;
- frozen AR;
- shuffled PCA residual;
- random PCA residual;
- label-permutation residual;
- train-only video-mean residual;
- diagnostics-only residual.

Its `420/420` matrix combined `70` blocked rows and `350` grouped rows. The exact AGAIN target,
head, feature width, seconds, and scores are historical context only.

#### VEATIC-specific washout procedure

The AGAIN result establishes that a washout can be scientifically useful; it does not
establish the correct VEATIC gap. In particular, AGAIN's `rows 4..10`, `1.5s`, `2.5s`, target
width, selected head, and fitted artifacts are not candidates by inheritance.

VEATIC proceeds as follows:

1. Run the starting spike task without a washout and with the complete controls.
2. In Phase 01, compute label-only, per-video VEATIC autocorrelation/partial-autocorrelation,
   rise time, event duration, causal-history predictiveness, coverage, and positive-support
   summaries. Freeze a bounded candidate-generation rule before observing any washout-target
   cortical result.
3. In Phase 02/05 development data, decompose the no-washout result into legal AR persistence,
   simple causal-history baselines, target/history row overlap, event prevalence, and
   fold/video consistency.
4. If the no-washout real lane clears its controls and consistency gates, retain it. The
   washout is then unnecessary rather than mandatory.
5. If legal persistence dominates or the target begins too near the causal history boundary,
   instantiate a small VEATIC-derived gap family. For future start offset `s`, the washout is
   `t+1..t+s-1` and the target begins at `t+s` or later.
6. Derive candidate `s` and window ends only from VEATIC label decay, event timing, video
   duration, coverage, and training-owned threshold stability. Record both row and second
   identities on the verified VEATIC 2 Hz grid.
7. Reject inadequate coverage/support candidates by rules frozen before cortical scoring.
   Select surviving candidates only on development-owned training/inner-validation evidence
   with every matched control; never select the gap on outer-test performance.
8. Rebuild the target, masks, folds, AR floors, scalers, PCA, residualizers, heads, and controls
   under the new ownership. No previous target-dependent object is silently reused.
9. Freeze the selected washout design and score it on fresh untouched confirmation evidence
   under blocked and grouped protocols. Evidence used to motivate redesign remains diagnostic.

The no-washout and washout comparisons retain the same future-maximum-increase construct,
fold-owned q90 threshold, metric stack, and controls. This isolates the contribution of the
temporal separation instead of changing several scientific questions at once.

### AGAIN Phase 06 — event stabilization

Path:

`studies/again/phase-06-event-stabilization`

Key evidence:

- `README.md`
- `plans/phase6_original_three_checkpoint_control_complete_plan.md`
- `plans/phase6_original_three_checkpoint_grouped_confirmation_plan.md`
- `runs/checkpoint-ensembles/reference-recipe-blocked-confirmation-20260714/`
- `runs/checkpoint-ensembles/reference-recipe-grouped-confirmation-20260714/`

Method transferred:

- stabilize only an already controlled selected recipe;
- average fixed, independently trained aligned checkpoints with equal weights;
- declare checkpoint groups before confirmation;
- include every matched control and exact AR checksum identity;
- require consistency across groups/folds, not only an aggregate mean;
- keep grouped and blocked confirmations separate.

### AGAIN Phase 07 — continuous movement ranking

Path:

`studies/again/phase-07-continuous`

Key evidence:

- `README.md`
- `evidence-summary.md`
- `preregistration/diagnostic-and-blocked.md`
- `preregistration/grouped-confirmation.md`
- `grouped-confirmation/manifests/run_manifest.json`
- `grouped-confirmation/diagnostics/audit.json`
- `grouped-confirmation/metrics/result.json`
- `grouped-confirmation/metrics/fold_group_deltas.csv`
- `grouped-confirmation/reports/again_dense_2hz_phase7_continuous_checkpoint_ensemble_grouped_20260714_181440.md`

Method transferred:

- continuous ranking is a separately specialized ability;
- train a target-specific continuous AR floor;
- select checkpoints by inner-only continuous criteria;
- use Spearman and top-5% true movement lift as primary endpoints;
- report top-1%/10% lift and exact-value errors separately;
- require fold/group consistency and fixed checkpoint ensembles;
- do not claim exact trajectories from ranking/lift evidence.

### AGAIN zero-label-at-inference closure

Path:

`studies/again/zero-label`

Key evidence:

- `README.md`
- `evidence-summary.md`
- `preregistration/development.md`
- `preregistration/locked-confirmation.md`
- `stage-0/`
- `stage-a/`
- `locked-confirmation/`

Method transferred:

- freeze target identity, split, features, controls, cold-start policy, and inference firewall
  before fitting;
- training may use labels, while held-out inference must not;
- isolate teacher/AR-assisted ceilings from deployable zero-label lanes;
- seal predictions before opening held-out outcomes;
- use whole-video shuffled, no-video, diagnostics-only, label-permutation, and temporal-context
  ablation controls;
- report full-video and cold-start results separately;
- use a prospective locked whole-video pool for final confirmation.

Zero-label is the final thorn after VEATIC spike, continuous arousal, and valence.

## What transfers from AGAIN and what does not

Transfer:

- phase order;
- question asked by each phase;
- target form for spike: future maximum increase relative to current arousal;
- continuous values/masks before fold-specific q90 event thresholding;
- grouped-video and blocked-temporal protocols reported separately;
- training-fold ownership of thresholds, normalization, PCA, AR, residualizers, checkpoints,
  and video means;
- strong target-specific AR floor;
- complete controls from the first applicable cell;
- frozen-AR residual and no-harm mechanics;
- eval-mode restored-checkpoint scoring;
- control-aware label-permutation interpretation;
- fixed equal-weight checkpoint groups;
- spike and continuous metric stacks;
- fold/seed/group consistency and paired whole-video uncertainty;
- zero-label inference firewall and prediction sealing.

Do not transfer:

- AGAIN source code or phase runners;
- AGAIN caches, rows, labels, target arrays, masks, or split assignments;
- AGAIN row offsets or seconds;
- q90 numeric thresholds;
- AR lag values or regularization;
- PCA widths, components, scalers, or scores;
- temporal aggregation windows;
- washout gaps, target-window ends, or target selections;
- head/model architecture or checkpoint weights;
- hidden widths or optimizer settings;
- seeds or checkpoint groups;
- result gates expressed as AGAIN-specific numbers;
- any fitted AGAIN artifact, prediction, control output, or metric result;
- any direct runtime dependency on an AGAIN code, study, output, or artifact path.

## Control matrix required from the first applicable cell

Every claim-bearing representation or learned matrix includes the real lane and every
applicable control in the same registered matrix. Matching means identical target, valid rows,
outer split, inner split, fold, seed, AR floor, scaler/PCA ownership, temporal context,
capacity, optimizer, checkpoint-selection policy, eval mode, and metric rows. Only the declared
controlled factor changes.

### Required controls

1. **Matched target-specific AR/frozen AR** — the central persistence floor.
2. **Real cortical lane** — only `cortical_prediction` provides real representation signal.
3. **Cortical-only lane** — measures cortical signal without AR.
4. **Current-row/no-temporal-context ablation** — measures whether causal history adds value.
5. **Shuffled cortical/PCA** — preserves declared shape/grouping while breaking real
   content-to-row alignment. The exact deterministic permutation policy is frozen before
   scoring.
6. **Shape-matched random** — seeded random representation with matched dimensionality and
   processing.
7. **Train-only causal video mean/static control** — no full-video test information.
8. **Diagnostics-only** — uses only the allowed diagnostic/nuisance input.
9. **Time/video-time-only** — tests temporal-position/base-rate structure.
10. **Quality/motion/luma-only** — tests visual nuisance structure.
11. **Label permutation** — training and inner-validation labels follow the registered
    permutation; held-out labels remain true.
12. **No-video/architecture ablation** — where the model structure permits it.

For frozen-AR residuals, label permutation retains the same AR floor. Its null question is
whether the learned residual adds anything beyond AR. Do not require its total PR-AUC to equal
raw prevalence.

No real-only pilot can authorize target selection, stability, confirmation, or later control
backfill.

## Metric contract

### Spike/event ranking

Primary metric:

- raw PR-AUC / average precision on exact matched held-out rows.

Required companion reporting:

- event prevalence and analytic chance;
- raw PR-AUC delta versus matched AR;
- raw PR-AUC delta versus strongest matched control;
- average-precision skill as a cross-prevalence companion only;
- ROC-AUC;
- precision, recall, F1;
- Brier score;
- top-1%, top-5%, and top-10% event recall/lift;
- defined-only per-video PR-AUC and undefined-video count;
- fold/seed/group positive counts and paired medians;
- paired whole-video cluster-bootstrap confidence intervals for primary deltas.

Raw PR-AUC is never replaced by AP skill. A control-complete result must beat AR and the
strongest applicable matched control, not merely analytic chance.

### Continuous arousal

Primary metrics:

- Spearman future-movement ranking;
- top-5% true-future-movement lift.

Supporting metrics:

- top-1% and top-10% lift;
- Pearson correlation;
- bias;
- MAE;
- RMSE;
- fold/seed/group consistency;
- paired whole-video uncertainty.

Exact-value metrics are a separate claim boundary. Good ranking does not prove exact values,
and an error-metric improvement does not override an unmet ranking gate.

## PCA strategy and accuracy safeguards

The default VEATIC search uses one maximum rank-512 PCA per outer-training partition and
nested prefixes `64/128/256/512`.

Reason:

- PCA directions are ordered and nested when solved accurately;
- lower-width models receive only their prefix, so components beyond their width cannot affect
  them;
- one basis makes width the variable rather than width plus randomized-fit variation;
- redundant fits needlessly rediscover the same leading components;
- the 512 model remains only a candidate and is not selected unless it generalizes best.

Accuracy requirements:

- fit on every eligible row owned by the outer-training partition, not a convenience sample;
- no held-out row in scaler or PCA fit;
- fixed PCA seed;
- float32 centering/accumulation;
- generous oversampling;
- sufficient power iterations;
- finite-value audit;
- component orthogonality audit;
- monotonic cumulative explained variance;
- reconstruction-residual audit;
- independent-seed subspace stability for the leading 64/128/256 components;
- prefix score checksums;
- recorded train/transform row digests.

If a rank-512 solve does not recover stable leading prefixes, separately fit the affected
widths. Do not prefer computational convenience over representation accuracy.

Using all 512 downstream components can overfit. Width selection is therefore inner-only and
the chosen width must survive outer control-complete confirmation. Available memory does not
make higher components informative.

## Phase 00 implementation contract

### Current status

Not implemented. Not run. Nothing later is authorized.

### New external lifecycle root

Use a new lifecycle root:

`/Volumes/onn. Drive/Neural Bridge Artifacts/runs/veatic-2.1/again-method-restart-20260723`

Phase 00 output:

`/Volumes/onn. Drive/Neural Bridge Artifacts/runs/veatic-2.1/again-method-restart-20260723/phase-00-dense-foundation`

Do not resume from an artifact outside this lifecycle root.

### Suggested implementation files

- `src/neural_bridge/veatic21/contracts.py`
- `src/neural_bridge/veatic21/data.py`
- `src/neural_bridge/veatic21/phase00.py`
- `src/neural_bridge/veatic21/__main__.py`
- `tests/veatic21/test_data.py`
- `tests/veatic21/test_phase00.py`
- `tests/veatic21/test_forbidden_inputs.py`

Keep the feature loader, row/metadata loader, and supervised label loader separate. Phase 00
must be structurally unable to obtain label values through its feature access path.

### Phase 00 allowed reads

From final TRIBE:

- per-video `manifest.json` and `status.json`;
- `per_video/run_status.json`;
- exact `tribe_v2_cortical_predictions.npz` payloads;
- `cortical_prediction` for shape/dtype/finiteness;
- `time_seconds` and provenance/quality fields needed for identity/audit;
- copied labels only if an equality audit is explicitly isolated from the Phase 00 scientific
  decision; the preferred Phase 00 path does not access their values.

From matching V-JEPA directories:

- the six allowlisted metadata/CSV files only;
- `rows.csv` header and identity columns;
- allowed JSON values and their recorded hashes.

### Phase 00 mandatory checks

1. Exact numeric video IDs `0..123` in both roots.
2. Exact matching ID set across roots.
3. `run_status.json`: 124 expected, 124 complete, empty failures.
4. Exactly one final TRIBE payload, manifest, and status per video.
5. Matching V-JEPA allowed metadata files per video.
6. TRIBE manifest/status equality and complete status.
7. V-JEPA manifest/status equality and complete status.
8. Upload marker and payload-manifest hash agreement.
9. Allowed V-JEPA file size and SHA-256 agreement without hashing hidden states.
10. Per-video row-count equality across TRIBE manifest, V-JEPA manifest, and `rows.csv`.
11. Total row count `20,657`.
12. Exact sequential `row_index` from zero within every video.
13. Exact `0.5s` time step and `2 Hz` declaration.
14. Exact `video_id` match on every CSV row.
15. Required `rows.csv` columns and encode policy.
16. Final cortical shape `[row_count, 20,484]` and float16 dtype.
17. All cortical values finite.
18. Uniform final-TRIBE key set across videos.
19. Copied TRIBE time exactly matches `rows.csv` time.
20. Quality flags have correct dtype/shape and union semantics.
21. Quality counts `76/871/24/923`, with all `20,657` source rows retained.
22. Final TRIBE tree digest matches the registered digest above.
23. Allowed V-JEPA metadata-only digest matches the registered digest above.
24. Provenance report explicitly records `vjepa_hidden_states_loaded=false` and
    `vjepa_hidden_states_hashed=false`.
25. No PCA, AR, target threshold, split, or model training occurred.
26. Source/runtime dependency audit records no import or execution from
    `src/neural_bridge/again/` or an AGAIN study runner.
27. Runtime input-manifest validation rejects every AGAIN code, study, output, and artifact
    path while retaining study documents as human-readable methodological references only.

### Phase 00 tests

Tests must prove, not merely report:

- a path containing `vjepa21_hidden_states.npz` is rejected before any open/load/hash;
- the allowlist excludes the hidden-state filename;
- the Phase 00 loader cannot request label arrays as features;
- a missing or extra video fails;
- a row-count mismatch fails;
- a time-grid mismatch fails;
- a cortical width/dtype mismatch fails;
- a nonfinite cortical value fails;
- a quality-union mismatch fails;
- an allowed-file hash mismatch fails;
- label values are not used to choose a Phase 00 outcome;
- source and runtime manifests cannot depend on AGAIN code, runners, data, or artifacts;
- no output can claim Phase 01 authorization unless every Phase 00 control passes.

### Phase 00 outputs

Heavy/audit root should contain:

- `request.json`
- `allowed-input-manifest.json`
- `row-inventory.csv`
- `quality-summary.json`
- `veatic-derivation-ledger.json`
- `result.json`
- `report.md`
- `artifact-manifest.json`

Every output is checksummed. `result.json` records the code digest, input digests, video/row
counts, array layouts, control results, forbidden-input audit, and the single next-phase
authorization.

After review, commit only compact defensible evidence under:

`studies/veatic-2.1/phase-00-dense-foundation`

The study directory owns its README, compact result, report, provenance/manifest, runner or
entrypoint reference, and tests. Heavy payloads remain external.

### Phase 00 pass gate

Every mandatory check and forbidden-input test passes. No model work occurred. A clean pass
authorizes Phase 01 only.

## Phase 01 exact next-stage contract

Phase 01 opens labels from `rows.csv` and nowhere else.

Required work:

1. Reconstruct the complete aligned supervised table from all `20,657` rows.
2. Reconfirm exact row identity against Phase 00.
3. Validate finite arousal/valence and native interpolation provenance.
4. Preserve quality flags as metadata without silently filtering rows.
5. Calculate VEATIC-specific movement histograms, autocorrelation, target coverage, per-video
   event support, and video-duration compatibility.
6. Define a bounded spike-window candidate procedure using the AGAIN future-maximum-increase
   formula.
7. Calculate per-video partial-autocorrelation decay, causal trailing-history and slope
   predictiveness, rise time, event duration, and prospective washout coverage/support.
8. Freeze both the initial no-washout selection rule and a bounded VEATIC-only procedure for
   deriving possible washout starts/window ends before reading washout cortical performance.
9. Record explicitly that AGAIN row offsets and seconds are not candidates by inheritance.
10. Store continuous future movement values and validity masks.
11. Do not create one global q90 binary label column.
12. Do not create an outer 70/30 split during alignment.
13. Write alignment, target-source, mask, row-ownership, and derivation-ledger digests.
14. Produce compact evidence and authorize Phase 02 only after all alignment controls pass.

## Phase 02 through zero-label execution sequence

1. Phase 00 dense foundation.
2. Phase 01 label alignment and VEATIC target substrate.
3. Phase 02 fresh target-specific AR under grouped and blocked protocols.
4. Freeze exact AR predictions/checksums for matched lanes.
5. Phase 03 raw cortical benchmark with complete controls.
6. Phase 04 fold-owned PCA and temporal representation search with complete controls.
7. Freeze one VEATIC-selected representation.
8. Phase 05 VEATIC-specific frozen-AR residual model discovery with no-harm and complete
   controls from the first cell.
9. If controlled decomposition activates the washout branch, derive its gaps/windows from
   VEATIC label dynamics, select only on development-owned evidence, register the redesign,
   and refit every ownership-dependent artifact before fresh confirmation. If the starting
   task clears its gates, retain the no-washout target.
10. Confirm one event target/head/recipe across fresh seeds and both protocols.
11. Phase 06 fixed checkpoint-group stabilization.
12. Phase 07 separately specialized continuous arousal.
13. VEATIC-specific valence programme.
14. Zero-label-at-inference development and prospective locked confirmation.
15. Only confirmed abilities enter production software.

## Exact next action

Implement Phase 00 from the contract above. Do not begin PCA, AR, target benchmarking, model
training, or a later phase. Do not read or hash a V-JEPA hidden-state NPZ. Do not select a
target horizon, width, head, or recipe in Phase 00.

When Phase 00 passes:

1. inspect every compact and external output;
2. run the focused VEATIC tests;
3. update this file in place with the exact result and hashes;
4. create the compact study-local Phase 00 evidence;
5. commit and push the coherent Phase 00 transition to `main`;
6. begin Phase 01 only after the commit is on `origin/main`.
