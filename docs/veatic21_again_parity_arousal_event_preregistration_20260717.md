# VEATIC 2.1 AGAIN-Parity Arousal Event Preregistration

Date: `2026-07-17`  
Status: preregistered inner-only development  
Outer-test confirmation: unauthorized

## Question

The first six-recipe VEATIC 2.1 programme ported the broad AGAIN architecture but not every performance-critical mechanic of the promoted short-temporal-convolution system. This bounded rerun asks whether a faithful dual-task frozen-AR residual implementation can beat its matched AR floor on full-dense VEATIC development videos.

This is not canonical training. It cannot authorize confirmation, use outer-test scores, change the current claim, or advance continuous/valence work by itself.

## Dataset boundary

`AGAIN-parity` refers only to reproducing a training algorithm that was proven useful during the AGAIN programme. No AGAIN video, label, row, input tensor, diagnostic, PCA artifact, AR score, threshold, checkpoint, model weight, or fitted normalizer is permitted in this run. All input arrays come from the sealed VEATIC 2.1 compact cache; all PCA parents are fitted from VEATIC training videos; and every AR and residual model is freshly trained on VEATIC ownership scopes.

## Fixed data and target

- data: all `124` VEATIC videos and `20,657` dense 2 Hz rows under existing grouped ownership;
- continuous supervision: `max(future arousal rows +4..+10 - current arousal, 0)`;
- event supervision: training-side q90 of that non-negative movement target;
- outer panels: the existing five sealed outer partitions;
- held-out inner panels: the existing three whole-video inner folds per outer partition;
- member seeds: `20260716`, `20260717`, and `20260718`;
- outer-test scores used: `false`.

## Quality, alignment, and zero-event policy

The sealed VEATIC 2.1 row contract is enforced before any fit. Every row must
retain its exact 2 Hz identity, causal clip endpoint, source annotation
bracket/interpolation position, and arousal/valence lane aliases. PCA fitting,
supervised fitting, checkpoint selection, and held-out scoring all use the same
prespecified quality mask. A row is excluded when its causal input window is at
least `50%` black or at least `95%` duplicated/static. This exclusion is based
only on the input available at that row and is applied identically to real and
AR lanes; no row may be removed because of its label, prediction, or outcome.

Event PR-AUC is computed once on all valid rows pooled across each held-out
whole-video panel. A video with no positive event is **not** assigned PR-AUC
zero, is not scored separately, and is not inserted into a video-average with
an artificial replacement value. Its valid negative rows remain in the pooled
panel so a false alarm still counts as a real error. Every panel must contain
both classes after the locked quality/target masks; otherwise the run fails
closed. The executor must record the positive-row count, event-video count,
zero-event-video count, pooled label digest, and valid-row digest for every
panel and prove those identities are equal across recipes, seeds, and ensemble
lanes.

## Faithful AGAIN mechanics

Every member receives a target-, panel-, fold-, and seed-specific dual-output AR model trained from the exact seven legal arousal-history features. The AR jointly learns continuous future movement with Huber loss and the event with BCE, selects its epoch using deeper training-side event PR-AUC, supplies whole-video-cross-fitted training predictions to the residual, and is then frozen.

The residual head:

- emits continuous and event corrections jointly;
- uses the frozen AR continuous prediction and event logit only as additive floors, not as residual-network inputs;
- trains with continuous Huber plus `0.5 ×` event BCE;
- retains the AGAIN residual-scale penalty, near-zero initialization, `0.12` global cap, and row gate;
- selects by inner-validation event PR-AUC delta versus its exact frozen AR;
- treats zero correction as an explicit candidate and returns exact AR when no positive residual checkpoint exists;
- restores and scores checkpoints in eval mode.

The corrected fair-training policy remains locked: batch size `1,024`, checkpoint eligibility from epoch `1`, no early stop before epoch `50`, patience `100`, and epoch `5,000` only as a runaway fail-safe. The old AGAIN 80-epoch cap is not restored.

## Six fixed recipes

| Recipe | PCA source | Width | Residual input |
| --- | --- | ---: | --- |
| `delta_pca64_again_clean_joint` | causal cortical delta | 64 | five PCA rows + current 53 diagnostics |
| `delta_pca64_veatic_enriched_joint` | causal cortical delta | 64 | clean input + five availability flags + two video-time features |
| `delta_pca128_again_clean_joint` | causal cortical delta | 128 | five PCA rows + current 53 diagnostics |
| `delta_pca128_veatic_enriched_joint` | causal cortical delta | 128 | clean input + five availability flags + two video-time features |
| `temporal_mean_2s_pca256_again_clean_joint` | causal four-row cortical mean | 256 | five PCA rows + current 53 diagnostics |
| `temporal_mean_2s_pca256_veatic_enriched_joint` | causal four-row cortical mean | 256 | clean input + five availability flags + two video-time features |

No recipe may consume arousal, AR score, AR confidence, future labels, or teacher outputs through its residual input matrix.

Compatible immutable PCA-256 parent artifacts may be reused only after exact identity and checksum validation. PCA-64 and PCA-128 are leading slices of their sealed PCA-256 parent. Every dual-task AR and residual checkpoint is fresh under this programme identity.

## Exact matrix and ensembles

Member matrix:

`5 outer panels × 3 held-out inner folds × 6 recipes × 3 seeds = 270`.

For every outer/fold/recipe cell, the three prespecified member logits are equally averaged. The three corresponding frozen-AR logits are averaged identically. This produces:

`5 outer panels × 3 held-out inner folds × 6 recipes = 90` ensemble rows.

No member selection, ensemble weighting, or post-hoc checkpoint choice is allowed.

## Gates

Artifact validity requires exactly `270/270` unique member rows and `90/90` unique ensemble rows, aligned held-out labels and valid-row identities, pooled zero-event-safe metric evidence, prediction seals, VEATIC-only PCA provenance, frozen-AR identities, checkpoint/eval-mode evidence, and no outer-test scores.

A fixed member recipe is a credible inner-development win only if its 45 paired real-minus-AR cells have:

1. mean PR-AUC delta at least `+0.001`;
2. paired median delta greater than zero;
3. at least `30/45` strict wins;
4. at least `4/5` positive outer-panel means.

Its fixed ensemble is credible only if its 15 paired cells also have mean delta at least `+0.001`, positive median, at least `10/15` strict wins, and at least `4/5` positive outer-panel means. Ensemble uplift is reported against the corresponding three-member mean and must be positive; it cannot rescue a failed frozen-AR comparison by relabeling.

A strong result requires `36/45` member wins, `12/15` ensemble wins, and `5/5` positive outer means for both.

Passing authorizes only the next inner-development decision. Failing keeps VEATIC arousal event ranking open and prohibits outer confirmation, continuous transfer, and valence transfer.
