# Zero-label direct-supervised locked confirmation preregistration

Status: prospectively locked on `2026-07-15`, after the Stage A development
screen and before any deployment-bridge access to outcomes or predictions for
the 299-video locked pool. The user explicitly authorized this bounded follow-up
and explicitly replaced teacher-retention as a success requirement with the
practical requirement that the video-only model beat its matched baselines.

This is a new confirmation protocol. It does not retroactively change the
original Stage A gate or declare H1 distillation/H2 self-rollout successful.

## Why this follow-up exists

Stage A completed `96/96` development rows. Its prespecified H1/H2 candidates
did not pass, but the prespecified direct `video_supervised_temporal` active
control produced the strongest zero-label result:

- Spearman `0.1574784207` versus the no-video anchor `0.0910654370`;
- top-5% lift `0.0611083563` versus `0.0495907196`;
- event PR-AUC `0.1461871599` versus `0.1187892131`;
- positive in `3/3` development folds and `3/3` first-30-second slices on all
  three endpoints.

It retained less than the old locked `50%` of the privileged teacher-added gain.
That threshold answered a teacher-compression question, not the practical
deployment question. A cold-start model that receives no observed arousal can
be useful without approaching a teacher that receives observed arousal. The
new test therefore asks the narrower, commercially relevant question: does the
fixed video-only method reproducibly beat no-video and false-signal baselines on
the prospectively locked videos?

## Locked candidate and data ownership

- Candidate: `video_supervised_temporal` exactly as implemented for Stage A.
- Inputs at inference: frozen predicted cortical/fMRI video features, the fixed
  causal five-row `temporal_mean_2s` / PCA256 representation, 53 video-derived
  temporal diagnostics, history masks, and causal time metadata.
- Training target: `future_arousal_max_delta_rows_4_10`.
- Event endpoint: training-only q90 threshold on that target.
- Training pool: all 696 development videos.
- Confirmation pool: the existing prospectively locked 299 videos, digest
  `ded8bc2bf079fef91ae5c253b9a9ac2e`.
- Reporting panels: the already frozen `60/60/60/60/59` hash panels.
- Seeds: `20260721`, `20260722`, `20260723`.
- Ensemble: fixed equal-weight mean of all three seeds; no member selection or
  weight search.
- Hardware: MLX GPU/MPS required; CPU fallback is forbidden.

All 995 AGAIN videos were used historically elsewhere in the project. The 299
videos are prospectively unaccessed for this deployment-bridge method, not an
external or historically untouched dataset.

## Locked lanes

Exactly seven lanes are scored:

1. `video_supervised_temporal` — the primary fixed candidate.
2. `video_supervised_current_row` — same direct objective without causal video
   history; an active temporal-context ablation.
3. `diagnostics_only_supervised_temporal` — same temporal head with predicted
   cortical/fMRI PCA channels zeroed; tests contribution beyond diagnostics.
4. `no_video_supervised_temporal` — all PCA and diagnostic content zeroed while
   retaining only causal masks/time metadata; the true no-video anchor.
5. `sequence_shuffled_supervised_temporal` — same hard-target training with
   whole-video input sequences deterministically reassigned.
6. `label_permutation_supervised_temporal` — same real inputs with whole-video
   hard targets deterministically reassigned during training.
7. `phase7_ar_assisted_teacher_ceiling` — observed-arousal-assisted research
   ceiling, opened only after all zero-label predictions are sealed.

The primary matched baseline set is lanes 2–6. The teacher ceiling is never a
pass threshold. The current-row lane is an active alternative rather than a
false-signal null; its comparison determines whether temporal history itself
adds value, not whether video-only deployment signal exists.

## Exact matrix

- members: `5 panels x 7 lanes x 3 seeds = 105`;
- ensembles: `5 panels x 7 lanes x 1 fixed group = 35`;
- total: exactly `140/140` scored rows.

Predictions are generated once for all 299 locked videos and only then sliced
into the five frozen panels for consistency reporting. Metrics do not multiply
the scored-row count.

## Required endpoints

Three endpoints remain conjunctive:

1. pooled continuous Spearman ranking;
2. top-5% average-true-future-movement lift;
3. training-q90 future high-movement event PR-AUC.

MAE, RMSE, bias, calibration, top-1%, top-10%, first 10 seconds, and teacher
retention are diagnostic only. No exact-value claim can be promoted here.

## Tiered verdict

The verdict must report each tier separately rather than collapsing the entire
experiment into one binary label.

### Tier 1 — zero-label deployment signal confirmed

Tier 1 passes only if:

1. scope is exactly `140/140`; every split, target, PCA, prediction-before-label,
   cold-start, finite-coverage, and MLX audit passes;
2. the primary ensemble beats the strongest false-signal/no-video control among
   lanes 3–6 in aggregate on all three required endpoints;
3. the paired five-panel median primary-minus-strongest-control delta is
   positive on all three endpoints (at least `3/5` directional panel wins);
4. the primary beats both sequence-shuffled and label-permutation controls in
   aggregate on all three endpoints.

There is deliberately no arbitrary minimum percentage of the teacher and no
minimum absolute delta beyond being positive. Exact deltas and relative gains
must be reported.

### Tier 2 — high-consistency confirmation

Tier 2 additionally requires, for all three endpoints:

- primary beats the per-panel strongest false-signal/no-video control in at
  least `4/5` panels; and
- a one-sided paired video-block bootstrap lower 95% bound is above zero.

Tier 2 failure cannot erase a Tier 1 baseline-beating result. It limits the
strength/stability claim.

### Tier 3 — cold-start-first-30 confirmation

Tier 3 separately requires the primary to beat the strongest false-signal/
no-video control on all three endpoints in the aggregate first 30 seconds and
in at least `4/5` panels. Undefined event PR-AUC fails this tier closed but does
not erase the full-video Tier 1 verdict.

### Mechanism and efficiency findings

- Primary greater than `video_supervised_current_row` establishes added value
  from causal video history. If current-row is stronger but itself beats the
  false-signal/no-video controls, the result still supports video-only signal;
  it selects a simpler method rather than a deployment failure.
- Primary greater than `diagnostics_only_supervised_temporal` supports
  incremental value from the predicted cortical/fMRI PCA channels beyond the
  53 diagnostics.
- Ensemble uplift and teacher-retention fractions are reported, not gates.

## Firewall and fail-closed rules

- The 299-video target values, event labels, observed arousal, and teacher
  ceiling predictions remain unavailable until every zero-label prediction
  file is written and checksummed.
- PCA, standardization, q90, target permutation, and every model fit use only
  the 696 development videos.
- Prediction starts at row 0 with zero-padded causal history and explicit masks;
  there is no labeled burn-in or teacher forcing.
- Retain every video, panel, seed, lane, and failed tier. No seed deletion,
  panel regrouping, rerun-until-pass, threshold change, member selection, or
  post-hoc weight search.
- Any split, target-identity, leakage, matrix, nonfinite, prediction-seal, or
  hardware failure invalidates all tiers regardless of favorable metrics.

## Maximum claim if Tier 1 passes

> On a prospectively locked 299-video AGAIN subset, the fixed direct-supervised
> Neural Bridge video-only model ranked future arousal movement and high-movement
> events from cold start without observed-arousal inputs at inference and beat
> matched no-video and false-signal controls across all three required endpoints.

This would validate cached-feature zero-label inference on AGAIN. It would not
yet prove end-to-end raw-video runtime, external/cross-domain transfer, exact
trajectories, client-ready accuracy, label-free training, individual profiling,
medical inference, or universal emotion prediction.
