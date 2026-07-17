# VEATIC 2.1 Arousal Event-First Six-Recipe Inner Discovery

Date: 2026-07-17  
Branch: `codex/veatic21-retraining-foundation`  
Scope: full-dense inner-validation development only; not outer-test evidence

## Plain-language verdict

The corrected run was executed properly, and **the Neural Bridge lost to the matched VEATIC AR baseline**.

The best of the six cortical recipes was `delta_pca64_short_conv`. It scored `0.3014044849` mean PR-AUC versus freshly trained matched frozen AR at `0.3082785856`: an absolute loss of `-0.0068741007`, or `-2.23%` relative. It beat AR in only `10/45` matched inner fold-seed cells, its paired median delta was `-0.0059385296`, and its mean delta was negative in all five outer development panels.

This is not a raw-cortical concatenation experiment. Every real lane is a Neural Bridge lane: a fresh VEATIC target-specific AR model is trained first and frozen, then a separate head learns a bounded residual correction from causal PCA-compressed predicted-cortical features. No fitted AGAIN PCA, AR, head, threshold, or model weight was used. The short-convolution method family was transferred as a design prior from AGAIN, but every fitted component was retrained from scratch inside the relevant VEATIC training fold.

The useful discovery is unusually clear: **cortical change + PCA64 + short temporal convolution won the six-recipe competition in all `5/5` outer panels**. The remaining problem is that its learned correction does not yet improve the now properly trained AR path. VEATIC spike/event ranking is therefore not cracked, continuous and valence development remain closed, and no outer confirmation is authorized.

## Three separate verdicts

| Question | Verdict | Meaning |
| --- | --- | --- |
| Did the executor complete validly? | **Pass** | `270/270` unique expected rows were sealed, the selection artifact verified, and a separate audit-only invocation reproduced it. |
| Did the best Neural Bridge recipe beat AR? | **Loss** | `0.3014044849` versus `0.3082785856`, or `-2.23%`; only `10/45` paired wins. |
| Is this artifact promotable? | **No, by contract** | This was inner-only discovery. `explicitly_nonpromotable: true`, `confirmation_authorized: false`, and `outer_test_scores_used: false` prevent development scores from becoming a claim. |

`canonical_gates_passed: false` in `run_summary.json` is not a numerical failure verdict. The runner derives that field only from a completed confirmation stage. This bounded discovery scope deliberately did not run confirmation, so the field defaults to `false`. Numerical performance must be read from the paired real-versus-AR scores above.

## Why there are 270 rows

The matrix is:

`5 outer partitions × 3 inner folds × 6 recipes × 3 seeds = 270`.

One row is one held-out-inner-fold score for a specific outer partition, inner fold, recipe, and random seed. The outer partition defines which grouped videos remain sealed away while that partition's recipe is selected; no outer-test score is opened. The three inner folds test whether a recipe generalizes to different held-out development videos, and the three seeds test whether the result survives random initialization. All `124` videos and all `20,657` dense 2 Hz rows participate according to their grouped fold assignment. The earlier eight-rows-per-video data was synthetic executor plumbing only and was not used here.

## Corrected executable scope and audit

- target: `future_arousal_max_delta_rows_4_10`
- endpoint: train-only-q90 future-arousal event ranking
- protocol: `privileged_binary`
- matrix: `270/270`
- run identity: `b6e914aabc280a8b3f2ee1baf6d1cfbb0040ddfe9080728586898d9d0eb6ecf1`
- selection digest: `8465b8331ffea9c49b359d7017b7cbe9bbbf5b4474009c4866d8e95d7a714660`
- outer-test scores used: `false`
- explicitly nonpromotable: `true`
- confirmation authorized: `false`
- contract amendment authorized: `false`
- fresh execution and separate `--audit-only` reproduction: passed

Compatible sealed PCA artifacts were reused from the shared derived root. AR models and cortical heads were retrained because the corrected optimizer/checkpoint settings changed the scientific identity.

## Corrected recipe result

All five outer development panels selected the same recipe:

| Outer panel | Winner | Mean inner PR-AUC | Runner-up | Winner margin |
| --- | --- | ---: | --- | ---: |
| 1 | `delta_pca64_short_conv` | 0.3074264363 | `delta_pca256_short_conv` | +0.0311310389 |
| 2 | `delta_pca64_short_conv` | 0.2954265068 | `current_pca256_current_row_mlp` | +0.0013259060 |
| 3 | `delta_pca64_short_conv` | 0.2988633666 | `current_pca256_current_row_mlp` | +0.0124362627 |
| 4 | `delta_pca64_short_conv` | 0.3045246282 | `delta_pca256_short_conv` | +0.0191400384 |
| 5 | `delta_pca64_short_conv` | 0.3007814864 | `delta_pca256_short_conv` | +0.0095947672 |

Across all `45` fold-seed cells per fixed recipe:

| Fixed recipe | Mean PR-AUC | Delta vs matched AR | Positive cells vs AR |
| --- | ---: | ---: | ---: |
| `delta_pca64_short_conv` | 0.3014044849 | -0.0068741007 | 10/45 |
| `delta_pca256_short_conv` | 0.2804274596 | -0.0278511260 | 1/45 |
| `current_pca256_current_row_mlp` | 0.2799904848 | -0.0282881008 | 11/45 |
| `temporal_mean_2s_pca64_short_conv` | 0.2551383128 | -0.0531402728 | 0/45 |
| `temporal_mean_2s_pca256_short_conv` | 0.2177499363 | -0.0905286493 | 0/45 |
| `temporal_mean_2s_pca256_flat_mlp` | 0.2034485741 | -0.1048300114 | 0/45 |

The matched AR mean was `0.3082785856`. The winning recipe's per-panel comparisons were:

| Outer panel | Real | AR | Delta | Positive cells |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.3074264363 | 0.3103330036 | -0.0029065673 | 4/9 |
| 2 | 0.2954265068 | 0.3037702681 | -0.0083437613 | 1/9 |
| 3 | 0.2988633666 | 0.2998503825 | -0.0009870160 | 4/9 |
| 4 | 0.3045246282 | 0.3122902633 | -0.0077656351 | 1/9 |
| 5 | 0.3007814864 | 0.3151490102 | -0.0143675238 | 0/9 |

This is a broad loss rather than one bad seed or one bad panel.

## Fair-training check

All six recipes received the same corrected schedule:

- batch size `1,024`;
- `5,000` epochs as a runaway fail-safe, not a target;
- minimum checkpoint-eligible epoch `50`;
- patience `100` with overfit protection;
- true BCE-with-logits training;
- binary checkpoint selection by held-out inner-validation PR-AUC.

No run hit the 5,000-epoch fail-safe. Video-head best epochs ranged from `50` to `734` with median `50`; AR best epochs ranged from `50` to `945` with median `104`. There were no epoch-1 checkpoints. This confirms that the earlier 80-epoch cap was too small and that the corrected run gave both the baseline and every recipe a materially fairer chance.

Proper training strengthened both sides, but AR gained more. Relative to the flawed-depth diagnostic, AR rose from `0.2984364723` to `0.3082785856` (`+3.30%`), while the best corrected bridge reached `0.3014044849`. The deeper run therefore exposed the real hurdle instead of manufacturing a near-tie through under-training.

## What the training dynamics say

`203/270` video checkpoints selected the first eligible epoch, while AR continued improving much longer. The winning recipe's effective residual scale averaged about `0.00310` against a hard cap of `0.12`. The correction was not literally zero, but it remained small and usually made held-out ranking worse.

More epochs alone are not the next answer: the epoch defect is fixed, no model hit the ceiling, and some cells legitimately trained for hundreds of epochs. The next bounded discovery branch should preserve the unanimously selected `delta_pca64_short_conv` representation/head while changing how the cortical correction is learned and admitted into AR.

## Next bounded discovery decision

Stay on arousal event ranking. Do not start continuous, valence, zero-label, combined-domain training, or outer confirmation.

Preregister a new inner-only residual-learning branch around `delta_pca64_short_conv`. The primary candidate should separate representation learning from correction admission: train the cortical residual against frozen-AR errors, then fit a train-only bounded correction coefficient/gate that may choose zero when the correction is harmful. Keep `delta_pca256_short_conv` and the current-row PCA256 MLP as bounded comparators, and retain matched frozen AR as the hurdle. Any checkpoint averaging should be evaluated as a declared stabilization candidate, not used post hoc to rescue a losing mechanism.

Only a fixed method that beats matched AR broadly across inner fold-seed cells should transfer into continuous arousal discovery. Outer-test videos remain sealed.

## Historical flawed-depth diagnostic

The first run at `$NEURAL_BRIDGE_EXTERNAL_ROOT/outputs/veatic21_endstate_20260717` also completed `270/270`, but it used batch size `8,192`, capped at 80 epochs, allowed epoch-1 checkpoint selection, and restored binary checkpoints by validation BCE. `116/270` heads hit the cap and `67/270` selected epoch 1. Its apparent best fixed recipe was `temporal_mean_2s_pca64_short_conv` at `0.2979001001` versus AR `0.2984364723`, and its same-selection-set per-outer panel appeared `+0.0012429328` above AR but won only `21/45` pairs. Those numbers are retained only as an under-training diagnostic and are superseded by the corrected result above.

## Artifact locations

- corrected output root: `$NEURAL_BRIDGE_EXTERNAL_ROOT/outputs/veatic21_endstate_depth_corrected_20260717`
- corrected score rows: `discovery/development_scopes/arousal_event_first/score_rows.json`
- corrected selection artifact: `discovery/development_scopes/arousal_event_first/selection_artifact.json`
- shared derived root: `$NEURAL_BRIDGE_EXTERNAL_ROOT/outputs/veatic21_endstate_shared_derived_20260717`
- historical flawed-depth output: `$NEURAL_BRIDGE_EXTERNAL_ROOT/outputs/veatic21_endstate_20260717`
