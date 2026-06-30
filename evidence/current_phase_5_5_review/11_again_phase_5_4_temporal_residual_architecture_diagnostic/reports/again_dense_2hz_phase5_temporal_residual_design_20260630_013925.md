# Phase 5 Temporal/Event-Context Residual Design

Timestamp: `20260630_013925`

This is a design report only. No training, implementation, grouped validation, 504 confirmation, V-JEPA/TRIBE rerun, PCA refit, or claim change is authorized by this report.

## Goal

The redesigned blocked-target monotonic frozen-AR residual diagnostic failed. A plausible limitation is that the residual branch only saw current-row evidence, while the redesigned targets are event/change/washout targets. The next test should keep frozen AR as the baseline floor but give the cortical residual a small causal temporal/event context.

Scope:

- Protocol: `blocked_temporal_70_30` only
- Targets:
  - `future_arousal_max_delta_rows_4_10_train_q90`
  - `residual_future_max_delta_rows_4_10`
- Feature source: fold-safe redesigned PCA256 artifacts under `outputs/again_dense_2hz_phase5_redesigned_target_foldsafe_pca_20260630_005312/`
- No grouped run
- No 504
- No V-JEPA/TRIBE/PCA rerun
- Frozen AR remains the baseline floor

## Candidate Heads

### 1. `current_row_mlp_residual`

Purpose: baseline residual with the same current-row information family as prior runs, but in the same code path as the temporal heads.

Input tensors:

- `frozen_ar_score`
- current fold-safe PCA256: `pca_t`
- current temporal diagnostics: `diag_t`
- optional scalar AR confidence terms: `frozen_ar_score`, `abs(frozen_ar_score - median_train_score)`, and binary-score entropy where applicable

Causal window rows: current row only, row `t`.

Existing artifacts sufficient: yes.

New derivation required: none beyond loading existing fold-safe PCA and diagnostics.

Leakage risks: low if row indices are verified against the fold-safe PCA manifest and diagnostics are current/past-only.

### 2. `delta_feature_mlp_residual`

Purpose: expose event/change structure without a sequence model.

Input tensors:

- `frozen_ar_score`
- current fold-safe PCA256: `pca_t`
- causal PCA deltas:
  - `pca_t - pca_t_minus_1`
  - `pca_t - mean(pca_t_minus_1 ... pca_t_minus_4)`
  - `mean(pca_t_minus_1 ... pca_t_minus_4) - mean(pca_t_minus_5 ... pca_t_minus_8)` when same-video history exists
- current diagnostics: `diag_t`
- causal diagnostic deltas using the same 1-row, 2s, and 4s history policy

Causal window rows: current row plus rows `t-1` through `t-8` within the same video. No row after `t` is allowed.

Existing artifacts sufficient: yes. Deltas are derived from row-aligned fold-safe PCA scores and existing diagnostics; PCA is not refit.

New derivation required: lightweight causal delta arrays written only under the ignored output root.

Leakage risks:

- Must hard-mask cross-video boundaries.
- Missing history rows must be zero-filled with explicit availability flags.
- No future rows may be used in rolling means or deltas.

### 3. `short_temporal_conv_residual`

Purpose: allow a tiny learned causal temporal filter over local cortical/diagnostic history.

Input tensors:

- sequence tensor `X_seq` with shape `[rows, window, channels]`
- channels:
  - fold-safe PCA256
  - a compact diagnostics subset or all diagnostics if memory is acceptable
  - optional repeated frozen AR score/history scalars
- suggested first window: 2s causal history, rows `[t-4, t-3, t-2, t-1, t]`

Architecture:

- one causal `Conv1D` layer, 16 or 32 channels, kernel size 3
- GELU
- optional second causal `Conv1D` layer only if the first version is stable
- final MLP projection to residual score
- residual scale `alpha` initialized near zero

Causal window rows: start with 5 rows at 2Hz: current row plus previous 4 rows. A 4s, 9-row window is future work only unless the 2s version is stable and still under the matrix cap.

Existing artifacts sufficient: yes. The sequence tensor can be derived from the row-aligned fold-safe PCA score file and diagnostics.

New derivation required: causal sequence tensor builder under the ignored output root.

Leakage risks:

- Must never use centered windows.
- Must hard-mask video boundaries.
- Must not allow test rows into any scaler, PCA fit, or train-only video mean.
- Must verify row order against the fold-safe PCA row-index file.

### 4. `low_ar_confidence_temporal_residual`

Purpose: keep frozen AR dominant when it is confident and let the residual act mainly on ambiguous AR cases.

Input tensors:

- same causal temporal input as `delta_feature_mlp_residual`, or the 2s sequence input if using the conv implementation
- frozen AR score and confidence features

Gating policy:

- residual output is multiplied by a train-only AR ambiguity gate
- binary target: use a smooth ambiguity function such as `4 * p * (1 - p)` after train-calibrated sigmoid/probability conversion
- continuous target: use train percentile distance from the median frozen AR prediction, highest gate near the middle confidence band
- gate may be learned only as a small monotonic or clipped scalar modulation, not as a free second head

Causal window rows: same as selected temporal input. No future rows.

Existing artifacts sufficient: yes if frozen AR score cache is available or AR-only checkpoints are re-forwarded in eval mode without retraining.

New derivation required: frozen AR confidence features and ambiguity gate audit.

Leakage risks:

- Calibration and percentile thresholds must be fit on train rows only.
- Do not tune the ambiguity band on heldout test rows.

## Recommended First Architecture

Recommended first runnable architecture: `delta_feature_mlp_residual`.

Reasoning:

- It is the smallest change from the current-row residual.
- It directly targets event/change information that the redesigned targets are meant to capture.
- It avoids the implementation and memory risk of a sequence conv while still adding causal temporal context.
- It can reuse the fold-safe PCA256 artifacts without refitting PCA.
- It should reveal whether causal cortical change features help before spending compute on a convolutional head.

The second architecture to try, only if the delta MLP is clean but weak, is `low_ar_confidence_temporal_residual` using the same delta inputs. The short temporal conv should remain a follow-up after one delta-feature pass unless the user explicitly authorizes it.

## Proposed Bounded Matrix

First runnable matrix:

- Targets: 2
- Heads: 4 candidate heads listed above
- Seeds: 3 (`20260625`, `20260626`, `20260627`)
- Controls: 7

Rows: `2 targets x 4 heads x 3 seeds x 7 controls = 168 rows`

Residual trainings excluding frozen AR rows: `2 x 4 x 3 x 6 = 144`.

This is under the requested 200-row ceiling. If compute should be minimized further, run only `delta_feature_mlp_residual` first:

- `2 targets x 1 head x 3 seeds x 7 controls = 42 rows`

Recommended execution order:

1. Run the 42-row `delta_feature_mlp_residual` smoke-confirmation matrix.
2. If it fails by a wide margin or controls dominate, stop.
3. If it is positive but below threshold, run the full 168-row comparison.
4. Do not run grouped or 504 from any result until blocked gates pass cleanly.

## Controls

Use the same controls for each target/head/seed:

- `frozen_ar_only`
- `real_residual`
- `shuffled_pca_residual`
- `random_pca_residual`
- `label_permutation_residual`
- `train_only_video_mean_residual`
- `diagnostics_only_residual`

Label permutation policy:

- permute train labels/targets
- permute inner-validation labels/targets for checkpoint selection
- score on true heldout labels/targets

Train-only video mean policy:

- compute video means from train rows only
- use global train mean fallback for videos without train rows
- do not use full-video oracle as a promotability gate

## Metrics

Binary target:

- PR-AUC
- ROC-AUC
- top 1/5/10% recall
- precision at top 1/5/10%
- delta vs frozen AR
- delta vs best matched control
- seed consistency

Continuous target:

- Spearman
- Pearson
- top 1/5/10% continuous lift
- average true value in predicted top 1/5/10%
- NDCG@1/5/10% if cheap
- delta vs frozen AR
- delta vs best matched control
- seed consistency

## Gates

Weak pass:

- primary metric delta vs frozen AR `>= +0.001`
- primary metric delta vs shuffled/random controls `>= +0.001`
- primary metric delta vs label permutation `>= +0.001`
- primary metric delta vs train-only video mean `>= +0.001`
- at least `2/3` seeds positive vs frozen AR and primary controls
- no leakage/audit failure

Credible pass:

- same gates with all primary deltas `>= +0.003`
- at least `2/3` seeds positive
- no single seed contributes more than 60% of the mean positive delta

Strong pass:

- same gates with all primary deltas `>= +0.005`
- `3/3` seeds positive preferred

Binary primary metric:

- PR-AUC for `future_arousal_max_delta_rows_4_10_train_q90`

Continuous primary metrics:

- top 5pct continuous lift
- Spearman must also be positive vs frozen AR

Do-no-harm:

- binary target: no seed worse than frozen AR by more than `-0.0005` PR-AUC
- continuous target: no seed worse than frozen AR by more than `-0.0005` top 5pct lift unless Spearman is clearly positive and the report marks the result diagnostic-only

## Required Leakage Audits

Before training:

- verify fold-safe PCA leakage audit passes
- verify no redesigned test rows were used to fit PCA/scaler
- verify row counts and row-index checksums match the fold-safe PCA manifest
- verify target-window overlap is false
- verify future leakage is false
- verify no original Phase 4 PCA artifact is used for these targets

For temporal inputs:

- confirm all context rows are `<= t`
- confirm same-video boundary masking
- confirm missing-history flags are train/test safe
- confirm train-only scalers are fit on train rows only
- confirm train-only video means do not include heldout rows
- write a per-head causal-context manifest

Stop conditions:

- fold-safe PCA audit fails
- row counts/checksums disagree with manifest
- any temporal context includes future rows
- train-only video mean uses heldout rows
- label permutation selection uses true inner-val labels
- same-variant control gates cannot be computed
- output matrix would exceed 200 rows

## Existing Artifacts Sufficiency

Existing artifacts are sufficient for the design:

- fold-safe redesigned PCA256 score files
- fold-safe row-index files
- fold-safe PCA leakage manifests
- labels and row/video/time metadata
- existing temporal diagnostics
- frozen AR score cache or AR-only checkpoints for eval-mode re-forward if cache is missing

New derived artifacts are lightweight and should live under the ignored run output root:

- causal PCA delta arrays or tensors
- causal diagnostics delta arrays or tensors
- causal context audit JSON
- train-only AR confidence/ambiguity calibration JSON

No dense cache, PCA components, V-JEPA, or TRIBE artifacts should be modified.

## Binary vs Continuous

Run both approved targets in the first matrix. The binary target tests washout-gap spike ranking; the continuous target tests whether AR-residualized movement ranking/lift is more suitable for cortical residual signal. Running only one target risks repeating the previous failure mode without learning whether the issue is target type or architecture.

Do not add secondary targets beyond the two listed here.

## Recommended Run Prompt

Use this prompt for implementation and execution:

```text
Continue in the current repo.

Task:
Implement and run the bounded blocked-only temporal/event-context residual diagnostic.

Do not run grouped.
Do not run 504.
Do not rerun V-JEPA/TRIBE/PCA.
Do not refit PCA.
Do not change claims.

Use:
- protocol: blocked_temporal_70_30 only
- fold-safe redesigned PCA256 root: outputs/again_dense_2hz_phase5_redesigned_target_foldsafe_pca_20260630_005312/
- targets:
  1. future_arousal_max_delta_rows_4_10_train_q90
  2. residual_future_max_delta_rows_4_10
- seeds: 20260625, 20260626, 20260627
- frozen AR baseline floor

First run only:
- head: delta_feature_mlp_residual
- controls:
  1. frozen_ar_only
  2. real_residual
  3. shuffled_pca_residual
  4. random_pca_residual
  5. label_permutation_residual
  6. train_only_video_mean_residual
  7. diagnostics_only_residual

Expected matrix:
2 targets x 1 head x 3 seeds x 7 controls = 42 rows.

Input policy:
- current fold-safe PCA256
- causal PCA deltas using rows t-1 through t-8 only
- current diagnostics
- causal diagnostics deltas using rows t-1 through t-8 only
- frozen AR score/logit as baseline floor
- residual alpha initialized near zero
- no future rows

Label permutation:
- permute train labels/targets
- permute inner-val labels/targets for checkpoint selection
- score on true heldout labels/targets

Video mean:
- train-only video mean only as the promotability-blocking static control

Before training:
- verify fold-safe PCA leakage audit passes
- verify no test rows were used in PCA/scaler fit
- verify row counts/checksums match the manifest
- verify target-window overlap false
- verify future leakage false
- verify temporal context uses only current/past same-video rows

Gates:
- weak threshold +0.001
- credible threshold +0.003
- strong threshold +0.005
- at least 2/3 seeds positive

Stop after summary.
Do not commit automatically.
```

## Bottom Line

Best recommended architecture: `delta_feature_mlp_residual`.

Recommended first matrix size: `42 rows`.

Maximum bounded comparison matrix size: `168 rows`.

Safe to run: yes, if the implementation passes the fold-safe PCA audit, temporal context audit, label-permutation policy audit, and train-only video-mean audit before training.
