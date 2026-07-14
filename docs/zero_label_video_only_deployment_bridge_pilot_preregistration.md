# Zero-label video-only deployment-bridge pilot preregistration

Status: preregistered planning-only on `2026-07-14`. Stage 0 permits manifests,
dry runs, and contract tests only. No model fitting, teacher-score generation,
held-out scoring, or claim promotion is authorized by this document.

Stage 0 execution record on `2026-07-14`: the real `995`-video / `243,575`-row
substrate passed all target, split, feature-policy, cold-start, event-support,
and matrix contracts. The exact dry-run matrices are `96/96` for Stage A and
`140/140` for Stage B; failed contracts were `[]`. Evidence is frozen in
`evidence/zero_label_video_only_deployment_stage0_20260714/`. No model was fit,
no teacher scores were generated, no held-out predictions were scored, and
Stage A remains unauthorized.

Pre-implementation amendment on `2026-07-14`: before Stage 0 implementation,
model fitting, teacher-score generation, or held-out scoring, training-q90 event
PR-AUC was promoted from report-only to a required deployment endpoint. The
amendment also makes explicit that the realistic success scale is retention of
the teacher's *incremental gain*, not equality with the teacher's absolute
score. This prospective change does not retroactively turn the stored Phase 7
event metric into a Phase 7 promotion gate; it applies only to this deployment
pilot. No model result was inspected to make this amendment.

## Question and evidence boundary

Can a model that receives only the frozen predicted cortical/fMRI video-feature
cache and causal video metadata rank both continuous future human-arousal
movement and future high-movement events on held-out full AGAIN videos, from
cold start, without observed arousal at any inference row?

The current Phase 7 AR-assisted checkpoint ensemble is the research ceiling and
teacher recipe. It is not a deployable baseline because its AR path consumes
current arousal, lag-1/2/4 arousal, and recent observed-arousal deltas. This
pilot tests whether useful ranking/lift survives after that inference-time label
dependency is removed.

The pilot tests two prespecified hypotheses:

1. **H1, primary:** a causal video-only student can distill cross-fitted Phase 7
   teacher scores and beat every prespecified zero-label control.
2. **H2, secondary:** a strictly closed-loop response-state model can replace
   observed AR context with its own predictions and beat every prespecified
   zero-label control.

H2 cannot rescue a failed H1 after scoring by being called the same method. Each
candidate is evaluated independently in Stage A. Only the locked selection rule
below may choose a Stage B candidate.

The Phase 7 ceiling has a deliberate information advantage: it receives
observed current/past arousal, while the deployment candidate starts every
unseen video without any response labels. Matching the Phase 7 absolute score
is therefore not a pass requirement and would be an exceptional result. The
locked practical target is to beat every zero-label control and retain at least
`50%` of the Phase 7 teacher's *incremental gain over the matched no-video
zero-label anchor* on every required endpoint. This is not `50%` of the raw
Phase 7 metric value.

## Stage 0 blocking semantic audit

The deployment runner must not inherit a target merely by copying its name.
Current code has a semantic hazard that must be resolved before fitting:

- `run_again_dense_2hz_phase5_temporal_residual_grouped_compat.target_spec()`
  selects `future_arousal_max_delta_rows_4_10_train_q90`, whose continuous value
  column is `future_arousal_max_delta_rows_4_10`;
- its grouped block builder fills `train_cont` and `test_cont` from that raw
  future-movement column;
- the Phase 7 grouped runner then replaces only the block's target name/type with
  `residual_future_max_delta_rows_4_10` / continuous, without replacing the
  arrays.

The existing Phase 7 grouped ranking/lift numbers remain evidence about future
movement; this finding is a target-name/value-array identity issue, not a
retroactive failed gate. The deployment pilot locks its hard outcome to the
actual like-for-like value column:

- hard outcome: `future_arousal_max_delta_rows_4_10`;
- required event endpoint label: a training-only q90 threshold on that value;
- full cold-start score mask: `label_available`, target mask true, and finite
  hard outcome;
- forbidden mask dependency: `ar_context_available`.

Stage 0 must write a target-identity manifest containing the target name, value
column, mask column, row IDs, array shape, and digest for every split and scorer.
The builder, prediction table, and scorer digests must agree exactly. Testing a
truly residualized outcome would be a different experiment and requires a new
preregistration version and a newly matched teacher ceiling.

## Fixed substrate and prospective split lock

- Dataset: AGAIN only, all 995 videos available in the current dense substrate.
- Frozen inputs: existing 2 Hz predicted cortical/fMRI response features,
  the fixed fold-safe `temporal_mean_2s` / PCA256 recipe, and the 53
  video-derived V-JEPA temporal-diagnostic values. Stage 0 confirmed that old
  fold-specific PCA score matrices occupy different coordinate systems and
  cannot be concatenated for the new split; later PCA bases must be fit only on
  the applicable outer training pool, and nested teacher PCA bases only on the
  corresponding teacher-training partition.
- No V-JEPA/TRIBE re-encoding, dense-cache mutation, VEATIC re-encode, joint
  training, additional target, or PCA-width search.
- Hardware for later fitting: MLX GPU/MPS required; no silent CPU fallback.
- Split unit: entire `video_id`; no video may cross a train/test or nested
  teacher-cross-fit boundary.

The split is generated by sorting the 995 unique video IDs by BLAKE2b digest of
`neural_bridge_zero_label_v1_20260714|<video_id>`:

- first 696: deployment-bridge development pool;
- remaining 299: deployment-pilot-locked pool;
- Stage A: three disjoint development folds of 232 test videos, training on the
  other 464 videos;
- Stage B: train on all 696 development videos and score all 299 locked videos;
  the locked videos are hash-partitioned into reporting panels of
  `60/60/60/60/59` for consistency gates.

All 995 videos participated in historical Neural Bridge work. The 299 videos are
therefore *prospectively locked for this deployment-bridge method after this
preregistration*, not historically untouched and not an external replication.

Stage A model seeds are fixed to `20260718`, `20260719`, and `20260720`. Stage B
model seeds are fresh and fixed to `20260721`, `20260722`, and `20260723`.
Every ensemble is the unweighted average of exactly those three aligned,
eval-mode member predictions. There is no member selection or weight fitting.

## Zero-label inference contract

The held-out inference process must load video features and metadata before any
label table is available. It writes and checksums a prediction table, and only a
separate scorer may later join outcomes.

Allowed held-out inputs for H1 are:

- fold-safe PCA256 current row and four causal same-video history rows;
- current 53-dimensional video-derived temporal diagnostics;
- explicit missing-history/start masks;
- `time_seconds` / training-derived `video_time_fraction`;
- `video_id` solely for split membership, chronological ordering, and state
  reset, never as a learned feature.

H2 may additionally consume only its own earlier predicted response states and
lags/deltas deterministically derived from those predictions.

Forbidden at every held-out inference row are:

- `arousal`, `valence`, any observed response value, and all availability flags
  that reveal response presence;
- ground-truth lag-1/2/4 values, recent ground-truth deltas, target values,
  target masks, event labels, and future rows;
- teacher scores, teacher hidden states, held-out target statistics, or
  teacher-forced/scheduled-sampling state;
- normalization, PCA, threshold, residualizer, or initialization statistics fit
  with locked-video rows.

Prediction begins at cached row 0 of every video. History is zero-padded with an
explicit availability mask for H1. H2 initializes all pre-video response states
to the development-training median and all deltas to zero, then resets at every
video boundary. There is no labeled burn-in or warm start. Predictions must be
finite for 100% of rows; scoring later applies only the fixed target-valid mask.
The first 30 seconds are reported and gated separately.

This is zero-label **inference**, not label-free training. Training labels may be
used only where explicitly allowed below.

## Teacher construction and firewall

The canonical output snapshot stores aggregate Phase 7 metrics but not reusable
row-level teacher score caches or checkpoints. The pilot must therefore create
new fold-local teacher artifacts later; it must never imply canonical checkpoint
reuse.

For H1 training, Phase 7 teacher scores are generated only inside each candidate
training pool by deterministic three-fold grouped-video cross-fitting. Each
training row's soft target must come from a three-checkpoint teacher ensemble
whose AR and residual members were fit without that row's video. Teacher soft
targets are standardized with cross-fitted training-pool statistics only. H1's
loss is weighted Huber loss to those soft targets; the hard outcome never enters
H1's input or loss. Rows lacking valid teacher AR context receive no soft loss.

For the ceiling, the same fixed three-checkpoint Phase 7 recipe is fit only on
the applicable training pool and scored with observed arousal on its held-out
videos. Ceiling predictions are isolated from candidate training, selection,
normalization, initialization, and inference. They are opened only after all
candidate/control prediction checksums for the stage are sealed.

Teacher-score manifests must record split/video digests, recipe and seed,
checkpoint digests, AR feature schema, score-array digests, cross-fit ownership,
and proof that no candidate-test video entered a teacher fit.

## Candidate and control lanes

The later implementation-freeze manifest must pin exact layer dimensions,
optimizer, loss coefficients, epoch limit, patience, and batching before Stage A.
The fixed starting recipe is the Phase 7 causal five-row temporal-conv shape,
hidden width 64, AdamW at `2e-4`, weight decay `1e-4`, maximum 80 epochs,
patience 12, batch size 8192, and train-only standardization. A separate direct
scalar-output student head is required; `TemporalResidualHead` cannot be reused
as H1 because it requires `ar_score` and `ar_reg`.

Stage A has eight lanes:

1. `video_distilled_temporal` — H1; causal video features, pure cross-fitted
   teacher-score distillation.
2. `video_closed_loop_rollout` — H2; a video-conditioned response-state model
   trained with current arousal as a loss target only. At training and inference,
   every recurrent response input is the model's own earlier prediction
   (`teacher_forcing_ratio = 0`). Its rolled current/lag/delta state feeds the
   frozen fold-local Phase 7 AR-plus-residual teacher head.
3. `video_supervised_temporal` — same-capacity causal video-only active control,
   trained directly on the hard training outcome.
4. `video_supervised_current_row` — same supervised head with no past video
   sequence; a temporal-context ablation.
5. `no_video_closed_loop_persistence` — H2 state/AR path with predicted cortical
   inputs zeroed; the prespecified no-video persistence anchor.
6. `sequence_shuffled_video` — whole training/test video sequences paired to
   different videos by deterministic hash and row-count proximity, preserving
   within-sequence autocorrelation while breaking content/outcome alignment.
7. `video_label_permutation` — whole-video training targets or teacher soft
   targets deterministically reassigned among training videos; held-out labels
   remain untouched.
8. `phase7_ar_assisted_teacher_ceiling` — non-deployable research ceiling, not a
   zero-label control and never eligible as a product lane.

Stage B contains the one locked Stage A winner, the five controls, and the
teacher ceiling. No losing candidate is scored on the locked pool.

## Exact scored-row matrices

A scored row is one lane/seed-or-ensemble/fold-or-panel record containing the
complete fixed metric vector. Metrics do not multiply the row count. Training
jobs, prediction rows, audits, teacher cross-fit shards, and curves are not
scored rows.

- Stage A members: `3 folds x 8 lanes x 3 seeds = 72`.
- Stage A ensembles: `3 folds x 8 lanes x 1 group = 24`.
- Stage A total: exactly `96/96`.
- Stage B members: `5 panels x 7 lanes x 3 seeds = 105`.
- Stage B ensembles: `5 panels x 7 lanes x 1 group = 35`.
- Stage B total: exactly `140/140`.

The uniqueness key is
`stage / split_digest / fold_or_panel / lane / row_type / seed_or_group /
cold_start_policy`. Every lane in a comparison must have identical row IDs and
target-valid masks.

## Metrics and retention

Three required promotion metrics are evaluated as one conjunctive gate:

1. pooled continuous Spearman ranking;
2. top-5% average-true-future-movement lift;
3. training-q90 future high-movement event PR-AUC.

The first two preserve the continuous response-movement surface. The third
preserves the commercially actionable spike/event surface. All three must pass;
strength on one endpoint cannot compensate for failure on another. The event
threshold is fit on the applicable training pool only and then frozen for its
held-out fold or panel. The metric-specific strongest prespecified zero-label
control is the comparator for each metric. Exact-value metrics never choose a
winner.

Event PR-AUC is defined only when the scored comparison slice contains at least
one positive and one negative event under the frozen training-only threshold.
If the aggregate, required fold/panel, or required first-30-second slice lacks
either class, that event gate is unevaluable and therefore fails closed. The
slice may not be skipped, pooled with a different split, or repaired by changing
the threshold after labels are opened.

Teacher retention is evaluated on the common compatibility rows where the
AR-assisted ceiling is defined. For metric `m`:

`R_m = (candidate_m - no_video_closed_loop_persistence_m) /
       (teacher_m - no_video_closed_loop_persistence_m)`.

The denominator must be finite and positive for each of `spearman`, `top5`, and
`event_pr_auc`. Absolute and relative degradation from the teacher are always
reported even if a candidate passes. The `R_m >= 0.50` gates mean that the
candidate preserves at least half of the teacher-added signal beyond the
no-video zero-label anchor; they do not require half of an absolute metric score
and do not imply that parity with the privileged teacher is expected.

Secondary/report-only metrics are top-1% and top-10% lift, event prevalence,
prespecified top-k event precision/recall, video-macro summaries, the
first-10-second slice, MAE, RMSE, bias, and calibration. The first-30-second
slice is separately gated below for all three required endpoints. MAE/RMSE
improvements cannot promote exact continuous-value forecasting, and event
PR-AUC supports ranking rather than calibrated event probability claims.

Because the dataset size is fixed, the design uses practical minimum-effect and
consistency gates rather than a post-hoc power claim. Stage B also uses a paired
video-block bootstrap with 10,000 resamples and fixed seed `20260724`; all rows
from a sampled video move together and all three required metrics are
recomputed. Because promotion requires the conjunction rather than success on
any one endpoint, no endpoint can be selected post hoc to rescue the verdict.

## Stage A continuation gate

A candidate qualifies only if every condition passes:

1. Exact `96/96` scope, unique keys, identical comparison rows, 100% finite
   prediction coverage, and every target/provenance/zero-label/cold-start audit.
2. Aggregate candidate minus strongest zero-label control is at least `+0.002`
   Spearman, `+0.001` top-5% lift, and `+0.002` event PR-AUC.
3. Candidate beats the per-fold strongest zero-label control on all three
   required metrics in all `3/3` development folds; all three paired medians are
   positive.
4. Candidate beats `sequence_shuffled_video` and `video_label_permutation` on
   all three required metrics.
5. No one fold supplies more than `60%` of total positive candidate-minus-best-
   control gain for any required metric.
6. `R_spearman >= 0.50`, `R_top5 >= 0.50`, and
   `R_event_pr_auc >= 0.50`.
7. In the first 30 seconds, candidate beats the strongest zero-label control on
   all three required metrics in aggregate and in at least `2/3` folds.
8. Its three-checkpoint ensemble improves over its three-member mean by at least
   `+0.001` Spearman and by a positive amount on both top-5% lift and event
   PR-AUC.
9. H2 additionally has zero teacher-forced state reads, zero cross-video state
   carry, no nonfinite/exploding state, and an executable dependency audit proving
   that each response lag came only from an earlier prediction.

If neither candidate passes, stop. If one passes, it is the Stage B candidate.
If both pass, select the candidate with larger
`min(R_spearman, R_top5, R_event_pr_auc)`; if those values differ by less than
`0.02`, select H1 because it is the simpler inference path. The winner's
code/config/checkpoint policy and all manifests must be signed and locked before
any Stage B label or ceiling access.

A Stage A pass authorizes only a separate request to run Stage B. It does not
promote a deployment claim.

## Stage B bounded-pilot gate

Every condition must pass:

1. Exact `140/140` scope and all Stage A audits; the 299-video split digest is
   unchanged and there was no deployment-bridge training/selection access to its
   labels before the locked prediction checksums.
2. Aggregate candidate minus strongest zero-label control is at least `+0.002`
   Spearman, `+0.001` top-5% lift, and `+0.002` event PR-AUC.
3. Candidate beats the panel-specific strongest zero-label control on all three
   required metrics in at least `4/5` panels; all three paired panel medians are
   positive.
4. No panel supplies more than `50%` of total positive gain for any required
   metric.
5. The one-sided paired video-block bootstrap lower 95% bound is above zero for
   all three required deltas against their fixed metric-specific strongest
   controls.
6. `R_spearman >= 0.50`, `R_top5 >= 0.50`, and
   `R_event_pr_auc >= 0.50`; exact teacher degradation is present in the report.
7. In the first 30 seconds, candidate beats the strongest zero-label control on
   all three required metrics in aggregate and in at least `4/5` panels.
8. Top-1% and top-10% lift each beat their strongest zero-label control, and the
   candidate beats shuffled-video and label-permutation controls on all three
   required metrics.
9. Ensemble uplift over the three-member mean is at least `+0.001` Spearman and
   positive on both top-5% lift and event PR-AUC.

## Required Stage 0 implementation contracts

Before any later fitting, add executable tests for:

- exact target-name/value-column/mask/array-digest identity;
- a dedicated video-only block with no AR arrays or label-derived input fields;
- deterministic 696/299 split and Stage A fold/Stage B panel digests;
- complete video disjointness for outer and nested teacher-cross-fit splits;
- train-only PCA, scaling, q90, initialization, and teacher statistics;
- frozen held-out event-threshold identity, event-label digest, and PR-AUC row
  identity across every candidate/control comparison;
- fail-closed undefined-event behavior for any required aggregate,
  fold/panel, or first-30-second PR-AUC slice with only one outcome class;
- positive feature allowlist and forbidden-column rejection;
- same-video causal ordering, row-0 prediction, history padding, and video reset;
- H2 teacher-forcing ratio exactly zero and prior-state provenance;
- teacher/student firewall and score-before-label-join prediction checksums;
- fixed seeds/groups/lanes and exact dry-run `96` / `140` matrix arithmetic;
- fail-closed verdict behavior for any missing row, nonfinite prediction, audit
  failure, or altered split digest.

Stage 0 should produce only a preregistration manifest, split manifests, target
identity manifest, feature-policy manifest, dry-run matrix, and passing contract
tests. It must not create checkpoints or score models.

## Fail-closed rules and claim boundary

- Stage A requires explicit later user authorization. Stage B requires a second
  explicit authorization after the Stage A winner is locked.
- Retain every video, fold/panel, seed, lane, and failed gate. No seed deletion,
  regrouping, threshold edit, rerun-until-pass, winner substitution, weight
  search, or post-hoc gate change.
- Any leakage, target-identity, matrix, cold-start, provenance, or hardware audit
  failure stops the stage regardless of favorable metrics.
- A failed Stage A or B does not weaken the canonical Phase 7 AR-assisted or
  binary evidence; it closes only this preregistered deployment candidate.
- No result here authorizes same-family Optuna, 504, VEATIC re-encoding, joint
  training, additional targets, or an architecture zoo.

If Stage B passes, the strongest permitted statement is:

> On a prospectively locked AGAIN full-video subset, the prespecified
> cached-feature candidate produced cold-start video-only future-arousal
> movement ranking/lift and high-movement event ranking without observed-arousal
> inputs at inference, beat all fixed zero-label controls, and retained at least
> half of the matched AR-assisted teacher's incremental gain over the no-video
> zero-label anchor on every required endpoint.

It would not yet prove exact trajectories, end-to-end raw-video runtime, client
readiness, external/cross-domain transfer, VEATIC performance, fully confirmed
blocked continuous generalization, label-free training, individual profiling,
medical inference, or universal emotion prediction.
