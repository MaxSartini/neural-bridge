# Phase 5 Blocked Residual Same-Variant Control Audit 230325

This is a no-training audit of the bounded blocked-residual diagnostic at `outputs/again_dense_2hz_phase5_blocked_residual_targeted_20260629_230325`. It uses existing metrics, diagnostics, and runner code only. It does not change the diagnostic claim: the blocked delta is directional only, strict forward-time temporal generalization remains unproven, and the result is not promotable.

## Source Result

- Blocked frozen AR PR-AUC: `0.2654721820`
- Best real residual: `monotonic_do_no_harm_residual`, PR-AUC `0.2657861164`
- Best matched control: `shuffled_pca_residual` / `monotonic_do_no_harm_residual`, PR-AUC `0.2657134334`
- Real delta vs frozen AR: `+0.0003139344`
- Real delta vs best matched control: `+0.0000726830`
- Label permutation pass: `False`
- Video-mean static control pass: `False`
- Recommendation: `blocked_delta_positive_but_control_failures`

## Same-Variant Gates

These comparisons keep variant fixed. They do not compare the best real variant against controls selected from other variants.

| variant_name                          | real_pr_auc   | frozen_ar_pr_auc   | shuffled_pca_pr_auc   | random_pca_pr_auc   | label_permutation_pr_auc   | video_mean_pca_pr_auc   | diagnostics_only_pr_auc   | real_minus_frozen_ar   | real_minus_shuffled   | real_minus_random   | real_minus_label_permutation   | real_minus_video_mean   | real_rank_within_variant   |
|:--------------------------------------|:--------------|:-------------------|:----------------------|:--------------------|:---------------------------|:------------------------|:--------------------------|:-----------------------|:----------------------|:--------------------|:-------------------------------|:------------------------|:---------------------------|
| blocked_delta_selected_gated_residual | 0.2592608847  | 0.2654721820       | 0.2645239193          | 0.2649599350        | 0.2661663455               | 0.2621980953            | 0.2645059332              | -0.0062112973          | -0.0052630346         | -0.0056990503       | -0.0069054608                  | -0.0029372106           | 7                          |
| monotonic_do_no_harm_residual         | 0.2657861164  | 0.2654721820       | 0.2657134334          | 0.2657093746        | 0.2657192159               | 0.2658151313            | 0.2657460655              | +0.0003139344          | +0.0000726830         | +0.0000767418       | +0.0000669005                  | -0.0000290149           | 2                          |
| low_ar_confidence_residual            | 0.2642409531  | 0.2654721820       | 0.2650081025          | 0.2637569029        | 0.2660939183               | 0.2624267176            | 0.2653747976              | -0.0012312289          | -0.0007671494         | +0.0004840501       | -0.0018529653                  | +0.0018142354           | 5                          |
| rank_lift_residual                    | 0.2618293541  | 0.2654721820       | 0.2640645549          | 0.2648119114        | 0.2659142637               | 0.2622074801            | 0.2646957073              | -0.0036428278          | -0.0022352007         | -0.0029825572       | -0.0040849095                  | -0.0003781259           | 7                          |

The cleanest same-variant profile is `monotonic_do_no_harm_residual`. It is still not clean enough for promotion: the real monotonic residual beats same-variant shuffled, random, label-permutation, diagnostics-only, and frozen AR, but it loses to the same-variant video-mean oracle/static control by `-0.0000290149` PR-AUC.

For `monotonic_do_no_harm_residual`:

- Real minus shuffled PCA: `+0.0000726830`
- Real minus random PCA: `+0.0000767418`
- Real minus label permutation: `+0.0000669005`
- Real minus video mean: `-0.0000290149`
- Real minus frozen AR: `+0.0003139344`

The previous label-permutation gate failure is mostly best-of-family selection noise: the best label-permutation row came from `blocked_delta_selected_gated_residual` at `0.2661663455`, while the monotonic same-variant label-permutation control is `0.2657188017`, below monotonic real `0.2657861164`. The video-mean failure is not only best-of-family noise, because monotonic video-mean is `0.2658151313`, above monotonic real by `0.0000290149`.

## Label-Permutation Checkpoint Selection

Label-permutation residual rows permute the residual training labels, but checkpoint selection still uses true inner-validation labels. The selected checkpoint is saved when true inner-val PR-AUC delta vs frozen AR is positive, with `best_delta` initialized at zero. This makes the label permutation a hard null/model-selection stress test.

| variant_name                          | seed     | best_epoch   | best_inner_val_delta_vs_frozen_ar   | frozen_ar_inner_val_pr_auc   | pr_auc       | delta_vs_frozen_ar_pr_auc   | residual_suppressed   | checkpoint_restored   | inner_val_selection_labels   | training_labels       | selected_checkpoint_beat_frozen_ar_on_inner_val   | selected_checkpoint_beat_frozen_ar_on_test   |
|:--------------------------------------|:---------|:-------------|:------------------------------------|:-----------------------------|:-------------|:----------------------------|:----------------------|:----------------------|:-----------------------------|:----------------------|:--------------------------------------------------|:---------------------------------------------|
| blocked_delta_selected_gated_residual | 20260625 | 6            | +0.0001253889                       | 0.2320768625                 | 0.2617032245 | +0.0002078826               | False                 | True                  | true_inner_val_labels        | permuted_train_labels | True                                              | True                                         |
| blocked_delta_selected_gated_residual | 20260626 | 7            | +0.0002785123                       | 0.2394680800                 | 0.2680561783 | -0.0001566098               | False                 | True                  | true_inner_val_labels        | permuted_train_labels | True                                              | False                                        |
| blocked_delta_selected_gated_residual | 20260627 | 14           | +0.0011159304                       | 0.2387057818                 | 0.2687396338 | +0.0020312177               | False                 | True                  | true_inner_val_labels        | permuted_train_labels | True                                              | True                                         |
| low_ar_confidence_residual            | 20260625 | 1            | +0.0000886282                       | 0.2320768625                 | 0.2616754210 | +0.0001800791               | False                 | True                  | true_inner_val_labels        | permuted_train_labels | True                                              | True                                         |
| low_ar_confidence_residual            | 20260626 | 1            | +0.0001626947                       | 0.2394680800                 | 0.2685131458 | +0.0003003578               | False                 | True                  | true_inner_val_labels        | permuted_train_labels | True                                              | True                                         |
| low_ar_confidence_residual            | 20260627 | 7            | +0.0005878226                       | 0.2387057818                 | 0.2680931883 | +0.0013847722               | False                 | True                  | true_inner_val_labels        | permuted_train_labels | True                                              | True                                         |
| monotonic_do_no_harm_residual         | 20260625 | 10           | +0.0001400164                       | 0.2320768625                 | 0.2616504668 | +0.0001551249               | False                 | True                  | true_inner_val_labels        | permuted_train_labels | True                                              | True                                         |
| monotonic_do_no_harm_residual         | 20260626 | 10           | +0.0002489334                       | 0.2394680800                 | 0.2684933589 | +0.0002805709               | False                 | True                  | true_inner_val_labels        | permuted_train_labels | True                                              | True                                         |
| monotonic_do_no_harm_residual         | 20260627 | 1            | +0.0000995174                       | 0.2387057818                 | 0.2670138220 | +0.0003054059               | False                 | True                  | true_inner_val_labels        | permuted_train_labels | True                                              | True                                         |
| rank_lift_residual                    | 20260625 | 1            | +0.0001161867                       | 0.2320768625                 | 0.2617171663 | +0.0002218244               | False                 | True                  | true_inner_val_labels        | permuted_train_labels | True                                              | True                                         |
| rank_lift_residual                    | 20260626 | 5            | +0.0005609734                       | 0.2394680800                 | 0.2692932684 | +0.0010804804               | False                 | True                  | true_inner_val_labels        | permuted_train_labels | True                                              | True                                         |
| rank_lift_residual                    | 20260627 | 4            | +0.0002726710                       | 0.2387057818                 | 0.2667323564 | +0.0000239403               | False                 | True                  | true_inner_val_labels        | permuted_train_labels | True                                              | True                                         |

Every label-permutation row selected a checkpoint that beat frozen AR on true inner-val. Most also beat frozen AR on heldout test by tiny margins; one `blocked_delta_selected_gated_residual` seed was slightly negative on test. This behavior does not by itself prove leakage. It shows that, at the `1e-4` scale of this diagnostic, true-label checkpoint selection can harvest noise even when residual training labels are permuted.

A cleaner label-permutation control for a future confirmation should select by permuted inner-validation labels or use a fixed epoch/prespecified checkpoint rule. No additional training is justified until that null-control definition is settled.

## Video-Mean Static Control

The video-mean PCA control is an oracle/static nuisance control inherited from the frozen-AR residual builder: it computes per-video PCA means from concatenated train and test rows, then assigns those static video means back to train and test. It is intentionally not a deployable forward-time baseline.

The video-mean control beat real in the same monotonic variant by `-0.0000290149` PR-AUC, a tiny margin. That is enough to block promotion because the diagnostic real delta vs controls is only `1e-4` scale. It does not invalidate the directional blocked diagnostic by itself, and it does not invalidate the earlier frozen-AR do-no-harm finding. It does mean the blocked residual signal is not separated from a static/oracle nuisance control.

## Bug And Leakage Assessment

No implementation bug is currently suspected from this audit. The same frozen AR checksums are shared across controls, `frozen_ar_integrity_pass` remains `True`, controls are matched by seed/protocol/variant in the 168-run matrix, and checkpoint restore/eval-mode scoring were already audited as passing.

No gross leakage is suspected. The video-mean control deliberately uses train+test rows as an oracle/static stress control, so it is a leakage-warning control by design rather than evidence that the real residual path leaked. The label-permutation issue is better described as true-inner-val model-selection noise under a hard null.

## Conclusion

The diagnostic remains valid as directional evidence that `monotonic_do_no_harm_residual` can produce a tiny positive blocked delta over frozen AR. It remains non-promotable because the margin is practically/statistically meaningless and the static-control gate fails in the same variant.

Recommended next action: do not run the 504 confirmation and do not train new variants yet. First refine/audit the null-control gate definitions, especially label-permutation checkpoint selection and the video-mean oracle/static control, then decide whether a cleaner confirmation is warranted.
