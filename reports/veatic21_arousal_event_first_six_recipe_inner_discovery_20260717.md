# VEATIC 2.1 Arousal Event-First Six-Recipe Inner Discovery

Date: 2026-07-17  
Branch: `codex/veatic21-retraining-foundation`  
Scope: inner-validation development only; not outer-test evidence

## Verdict

The first bounded six-recipe grid identified a clear architectural family but did **not** robustly beat the matched frozen AR path. All five per-outer selections chose a short temporal convolution, with temporal-mean and delta cortical families both represented. However, the globally strongest fixed recipe remained slightly below AR on mean PR-AUC, and even the optimistically selected per-outer panel was positive versus AR in only `21/45` inner fold-seed cells.

Post-run audit then found that this was not yet a proper depth-controlled test of the six recipes: the runner capped training at 80 epochs with batch size 8,192, which was only roughly two optimizer updates per epoch on a typical inner training set, and binary checkpoint restoration minimized validation BCE instead of maximizing the registered PR-AUC endpoint. Treat the performance table below as an executable under-training diagnostic, not a fair final verdict on the six recipes.

Do not open outer-test confirmation and do not transfer this recipe into continuous arousal yet. The immediate next run is a fresh inner-only rerun of the same six recipes under a high emergency ceiling, minimum warm-up, patience-based overfit protection, smaller batches, and binary checkpoint restoration by inner-validation PR-AUC. Only after that controlled rerun should the project decide whether residual gating itself needs a new branch.

## Executable scope and audit

- target: `future_arousal_max_delta_rows_4_10`
- endpoint: train-only-q90 event ranking
- protocol: `privileged_binary`
- matrix: five outer partitions × three inner folds × six recipes × three discovery seeds = `270/270`
- run identity: `38ef99d8d164930c7bf151018dff6f1205762e46147d9661ee1ec547375f93b8`
- selection digest: `aed7c3e45bb2354c7a6ca3d42b4440619514bc6193bf398d1c2f24791c897e1e`
- outer-test scores used: `false`
- explicitly nonpromotable: `true`
- confirmation authorized: `false`
- fresh execution and separate `--audit-only` reproduction: passed

The scoped scheduling identity is the same scientific identity used by later full discovery, so its sealed PCA, frozen-AR, feature, and checkpoint artifacts can be reused when their provenance matches exactly.

## Recipe selection result

Every per-outer winner was a short temporal convolution:

| Outer partition | Selected recipe | Mean inner PR-AUC | Runner-up | Margin |
| --- | --- | ---: | --- | ---: |
| 1 | `delta_pca64_short_conv` | 0.2940452293 | `temporal_mean_2s_pca64_short_conv` | +0.0008468539 |
| 2 | `temporal_mean_2s_pca64_short_conv` | 0.2901938756 | `delta_pca64_short_conv` | +0.0019471261 |
| 3 | `temporal_mean_2s_pca64_short_conv` | 0.2900038130 | `delta_pca256_short_conv` | +0.0012880172 |
| 4 | `temporal_mean_2s_pca256_short_conv` | 0.3131119695 | `temporal_mean_2s_pca64_short_conv` | +0.0080196115 |
| 5 | `delta_pca256_short_conv` | 0.3110421378 | `temporal_mean_2s_pca64_short_conv` | +0.0000300593 |

The structural read is stable even though the exact recipe is not:

- short-conv won `5/5` outer selections;
- PCA-64 won `3/5`, PCA-256 won `2/5`;
- temporal-mean won `3/5`, delta won `2/5`;
- neither MLP recipe won an outer selection;
- four of five winner margins were below `0.002` PR-AUC.

Across all `45` fold-seed cells per fixed recipe, the mean ranking was:

| Fixed recipe | Mean PR-AUC | Delta vs matched frozen AR | Positive cells vs AR |
| --- | ---: | ---: | ---: |
| `temporal_mean_2s_pca64_short_conv` | 0.2979001001 | -0.0005363722 | 19/45 |
| `delta_pca64_short_conv` | 0.2967136949 | -0.0017227773 | 17/45 |
| `delta_pca256_short_conv` | 0.2965237307 | -0.0019127416 | 16/45 |
| `temporal_mean_2s_pca256_short_conv` | 0.2964685063 | -0.0019679660 | 27/45 |
| `current_pca256_current_row_mlp` | 0.2914361427 | -0.0070003295 | 16/45 |
| `temporal_mean_2s_pca256_flat_mlp` | 0.2890635697 | -0.0093729025 | 19/45 |

The matched frozen-AR mean across the same `45` cells was `0.2984364723`.

The per-outer selected panel reached `0.2996794050` versus AR `0.2984364723`, a mean delta of `+0.0012429328`, but it was positive in only `21/45` cells and its paired median delta was effectively zero (`-0.0000006626`). Because each outer recipe was selected from those same nine inner scores, this is selection-set performance, not a fresh validation estimate.

Per-outer selected deltas versus AR were `-0.0035228963`, `+0.0019077396`, `+0.0024767898`, `+0.0068927067`, and `-0.0015396760`. The selected panel therefore improved the mean through a few favorable partitions rather than a broad paired win.

## Training-dynamics finding

The residual correction stayed close to shut off:

- `116/270` checkpoints selected epoch `80`, the training cap;
- `67/270` selected epoch `1`;
- together, `183/270` (`67.78%`) landed at one of the two boundaries;
- the learned effective residual scale averaged only about `0.00234`–`0.00243` by recipe, versus a hard cap of `0.12` and an initialization near `0.00216`.

This boundary-heavy checkpoint pattern and near-initial residual scale explain why many recipe scores were almost identical to AR while a minority of cells produced large positive or negative excursions. The immediate defect is the training-depth and checkpoint-selection protocol. It must be corrected before concluding that the residual gate or the six feature/head choices themselves failed.

## Next bounded discovery decision

Stay on arousal event ranking. Do not start continuous, valence, zero-label, combined-domain training, or outer confirmation.

The corrected inner-only rerun should:

1. rerun all six recipes so the corrected schedule does not selectively favor the apparent winners from the flawed run;
2. reuse compatible PCA and immutable feature artifacts, while retraining AR and heads because their optimization identity changed;
3. use a high runaway-only ceiling, a minimum checkpoint-eligible warm-up, patience-based early stopping, and substantially more optimizer updates per epoch;
4. select binary checkpoints by inner-validation PR-AUC while continuing to train with true BCE-with-logits;
5. require a fixed method to beat matched AR broadly across fold-seed cells before any transfer to continuous arousal;
6. branch into altered residual gating only if the corrected-depth rerun still shows a near-closed correction and no broad AR win.

The corrected schedule changes the scientific settings digest and model schema, so it must use a new output/run identity. It remains inner-only and must not reuse outer-test scores.

## Artifact locations

- output root: `$NEURAL_BRIDGE_EXTERNAL_ROOT/outputs/veatic21_endstate_20260717`
- score rows: `discovery/development_scopes/arousal_event_first/score_rows.json`
- selection artifact: `discovery/development_scopes/arousal_event_first/selection_artifact.json`
- shared derived root: `$NEURAL_BRIDGE_EXTERNAL_ROOT/outputs/veatic21_endstate_shared_derived_20260717`
