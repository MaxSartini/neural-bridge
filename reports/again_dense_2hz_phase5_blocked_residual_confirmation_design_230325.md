# Phase 5 Blocked Residual Confirmation Design 230325

This is a design-only protocol for a future clean blocked residual confirmation. It does not report a new benchmark, does not change existing claims, and does not justify starting the now.

Current status from the pushed same-variant audit `70215b0`:

- Cleanest diagnostic variant: `monotonic_do_no_harm_residual`
- Monotonic real minus same-variant shuffled PCA: `+0.0000726830`
- Monotonic real minus same-variant random PCA: `+0.0000767418`
- Monotonic real minus same-variant label permutation: `+0.0000669005`
- Monotonic real minus same-variant video mean: `-0.0000290149`
- Label-permutation checkpoint-selection issue: yes
- Video-mean static-control issue: yes
- Implementation bug suspected: no
- Gross leakage suspected: no
- Promotable: no

The confirmation protocol below is meant to remove ambiguous null-control definitions before spending confirmation compute.

## Recommended Next Action

Recommended next action: do not run the yet. First implement a small clean confirmation runner/summarizer design for `monotonic_do_no_harm_residual` only, with corrected null controls and stricter gates. Run it only after the gate definitions below are encoded and reviewed.

The next runnable experiment, when approved, should be a small monotonic-only blocked confirmation, not a broad variant search. It should answer one question: does the monotonic do-no-harm residual produce a practically meaningful blocked-temporal gain over frozen AR and clean same-variant null controls?

## Label-Permutation Policy

For this residual setup, label permutation should mean a null residual learner that sees the same features, AR floor, optimizer, seed, split, and training budget as the real residual, but has the residual training target relationship broken. It should not get to use true inner-validation labels to select a lucky checkpoint.

Recommended policy:

- Training labels: permute binary and continuous residual training labels within the training split only.
- Inner-validation labels for label-permutation checkpoint selection: use the corresponding permuted inner-validation labels, not true inner-validation labels.
- Test scoring: always score against true heldout labels, because the null question is whether a permuted-training residual can still appear to improve real heldout ranking.
- Primary selection rule: select label-permutation checkpoint by permuted inner-val PR-AUC delta vs frozen AR, with the same early-stop logic as real.
- Secondary audit row: also report a fixed-epoch null, using the real selected epoch or a prespecified epoch, to estimate checkpoint-selection noise.

The cleanest null is permuted-train plus permuted-inner-val selection, then true-label heldout scoring. It preserves the training and selection mechanics while breaking the target relation before selection. Fixed epoch is useful as an audit, but less faithful to the real training loop. Reusing the real checkpoint epoch is useful for paired diagnostics, but it gives the null a checkpoint chosen by the real residual and should not be the primary label-permutation gate.

The current diagnostic used permuted training labels but true inner-validation labels for checkpoint selection. That made it a hard model-selection stress test. It is valid as a warning, but too harsh and too noisy to be the sole promotion gate at `1e-4` scale.

## Video-Mean Static-Control Policy

The video-mean PCA control should be treated as a mechanism diagnostic, not as a deployable blocked-temporal baseline.

Definitions:

- Full-video mean: compute each video's PCA mean using both train and test rows. Under blocked temporal validation this uses future/test information, so it is an oracle/static warning control.
- Train-only video mean: compute each video's PCA mean using training rows only, then apply that mean to test rows for the same video. This is less oracle-like, but still encodes video identity and static video-level nuisance structure.
- Test-only video mean: do not use for promotion; it directly uses heldout rows.

Recommended policy:

- Use train-only video mean as the promotability-blocking static-control gate.
- Report full-video mean separately as an oracle/static warning control.
- If full-video mean beats real but train-only video mean does not, mark `oracle_static_warning: true` but do not fail the primary promotion gate solely on full-video mean.
- If train-only video mean beats real, fail the static-control gate.
- Clearly label full-video mean as using future/test information under blocked validation.

The current diagnostic's video-mean control used full-video means from concatenated train and test rows. That is useful for detecting whether the residual is merely recovering static video nuisance information, but it should be separated from promotability gates because it is intentionally oracle-like under blocked temporal validation.

## Revised Blocked Confirmation Gates

All gates are same-variant gates. Do not compare the best real variant against best controls from other variants in the confirmation decision.

Primary protocol: `blocked_temporal_70_30`.

Variant: `monotonic_do_no_harm_residual`.

Required gates:

- `frozen_ar_integrity_pass`: all same seed/protocol/control rows use identical frozen AR train/test score checksums.
- `checkpoint_restore_pass`: best checkpoint restored or residual explicitly suppressed according to the documented rule.
- `eval_mode_scoring_pass`: dropout disabled and deterministic eval-mode scoring used for all controls.
- `real_gt_frozen_ar_pass`: mean real PR-AUC minus frozen AR PR-AUC must be at least the selected minimum practical delta.
- `real_gt_shuffled_random_pass`: mean real PR-AUC must exceed same-variant shuffled PCA and random PCA controls by at least the selected minimum practical delta.
- `real_gt_label_permutation_pass`: mean real PR-AUC must exceed the clean label-permutation null by at least the selected minimum practical delta.
- `real_gt_train_only_video_mean_pass`: mean real PR-AUC must exceed train-only video-mean static control by at least the selected minimum practical delta.
- `oracle_full_video_mean_warning`: report whether full-video mean exceeds real. This warning blocks headline language if the margin is large, but it is not the primary promotability gate.
- `seed_consistency_pass`: real minus frozen AR is positive in at least `4/5` seeds and real minus each primary control is positive in at least `4/5` seeds.
- `confidence_pass`: bootstrap or seed-level confidence interval for real minus frozen AR and real minus best primary control should be positive at the prespecified confidence level. If bootstrap is not implemented, require at least `4/5` positive seeds and no single seed driving more than `60%` of the mean delta.
- `do_no_harm_blocked_pass`: real PR-AUC must not fall below frozen AR by more than `0.0005` in any seed, and mean delta must be nonnegative.

Promotion gate:

- `blocked_residual_confirmation_pass` is true only if all primary gates pass at `weak` threshold or better.
- `strict_forward_time_temporal_generalization_proven` remains false unless blocked real beats frozen AR and all primary same-variant controls by at least the selected threshold with seed consistency and confidence support.

## Minimum Delta Thresholds

Tiny deltas such as `+0.00007` should not pass confirmation. They are useful for diagnostics but too small relative to observed control-selection noise.

Use these PR-AUC effect bands:

- Exploratory positive: `> 0.0000`
- Weak pass: `>= +0.0010`
- Credible pass: `>= +0.0030`
- Strong pass: `>= +0.0050`

Recommended confirmation threshold: `+0.0010` minimum for every primary blocked gate, with `+0.0030` required before using strong wording about blocked residual improvement. The current diagnostic's best real-vs-control margin of `+0.0000726830` is below even weak pass and should remain non-promotable.

Rationale: the same-variant audit showed label-permutation and matched-control movements on the order of `1e-4`, and the current positive blocked delta over frozen AR was only `+0.0003139344`. A `+0.0010` floor is a conservative minimum to avoid promoting numerical noise; `+0.0030` better matches a practically credible blocked improvement.

## Recommended Next Run Matrix


Smallest clean matrix, if a rerun is approved after this design is reviewed:

- Target: `arousal_spike_rows_2_6_train_q90`
- Continuous source: `future_arousal_max_delta_rows_2_6`
- Feature: `temporal_mean_2s_then_pca256`
- Protocol: `blocked_temporal_70_30` only
- Fold: full blocked split
- Variant: `monotonic_do_no_harm_residual` only
- Loss: `regression_plus_binary` only
- Seeds: `5` seeds minimum, reusing `20260625`, `20260626`, `20260627` plus two new prespecified seeds
- Controls:
  - frozen AR only
  - real residual
  - shuffled PCA residual
  - random PCA residual
  - label-permutation residual with permuted inner-val checkpoint selection
  - label-permutation fixed-epoch audit row
  - train-only video-mean PCA residual
  - full-video video-mean PCA residual as oracle warning
  - diagnostics-only residual

Exact matrix size: `1 protocol x 1 variant x 1 loss x 5 seeds x 9 controls = 45 rows`, including frozen AR rows. If counting only trainable residual rows, `5 seeds x 8 residual/control rows = 40 residual trainings`. Do not add variants, losses, secondary targets, or full grouped folds to this run.

Checkpoint selection:

- Real, shuffled, random, diagnostics-only, and train-only/full-video video-mean controls select checkpoint by true inner-val blocked PR-AUC delta vs frozen AR.
- Clean label-permutation primary null selects checkpoint by permuted inner-val PR-AUC delta vs frozen AR.
- Label-permutation fixed-epoch audit uses the real selected epoch for the same seed, or a prespecified epoch if real suppresses residual.
- Selection is always same seed, same split, same variant, same training budget.

## Grouped Evaluation Policy

Do not run full 5-fold grouped evaluation in the next blocked confirmation. The grouped claim already has a separate full grouped frozen-AR residual result, and this confirmation is specifically about blocked temporal behavior.

Grouped evaluation should be added only if the blocked confirmation passes the weak threshold and the result is being considered for broader claim updates. At that point, run the full 5-fold grouped protocol as a separate sanity/compatibility check. Do not report grouped fold 1 as canonical.

## Stop Conditions

Stop before running any confirmation if:

- label-permutation checkpoint-selection policy is not implemented exactly as specified
- train-only and full-video video-mean controls are not separated
- same-variant gates are not encoded
- minimum PR-AUC delta thresholds are missing
- frozen AR checksum integrity cannot be verified
- checkpoint restore or eval-mode scoring audit fails

Stop during or after the clean confirmation if:

- real minus frozen AR is below `+0.0010`
- real fails same-variant shuffled/random by `+0.0010`
- clean label-permutation null beats or nearly matches real inside the `+0.0010` margin
- train-only video-mean beats or nearly matches real inside the `+0.0010` margin
- fewer than `4/5` seeds are positive for real vs frozen AR or real vs the best primary control
- full-video oracle static control beats real by `>= +0.0010`; this should trigger mechanism diagnosis before promotion even if primary train-only video-mean passes

## Promotion Language If It Passes

If the clean blocked confirmation passes at weak threshold:

> In a targeted blocked-temporal confirmation, the monotonic frozen-AR residual produced a small but consistent PR-AUC gain over frozen AR and clean same-variant residual controls. This supports a targeted blocked residual improvement for the primary arousal spike ranking task, while strict forward-time temporal generalization should still be described with effect size and control caveats unless the gain reaches the credible threshold.

If it passes at credible threshold:

> The monotonic frozen-AR residual adds a blocked-temporal arousal spike ranking improvement beyond frozen AR and clean same-variant null controls, with seed-consistent gains and no do-no-harm violation. This is evidence for blocked residual improvement on the primary task, subject to grouped compatibility checks before broader claim updates.

Do not use Phase 6, promotion, or headline benchmark language until full documentation and grouped compatibility checks are complete.

## Non-Promotion Language If It Fails

If the clean confirmation fails:

> The bounded diagnostic's tiny positive blocked delta did not survive clean same-variant confirmation gates. Strict forward-time temporal generalization remains unproven. The frozen-AR residual result remains useful as a diagnostic showing reduced blocked harm and a candidate monotonic gate, but it should not be promoted as a blocked temporal improvement.

If only full-video oracle static control fails:

> The primary train-only static-control gate passed, but full-video oracle static control still matched or exceeded real residual. Treat this as a mechanism warning: the residual may be partly recovering static video-level nuisance structure, so no headline blocked claim should be made without additional mechanism diagnosis.

## Decision

Next run: small monotonic-only clean blocked confirmation.

Exact next matrix size: `45 rows` total, or `40 residual trainings` excluding frozen AR only.
