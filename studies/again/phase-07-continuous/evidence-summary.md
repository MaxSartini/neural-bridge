# Neural Bridge Phase 7 Evidence

Last updated: 2026-07-14

Phase 7 is the current headline result. It independently confirms controlled grouped held-out-video continuous future-arousal movement ranking/lift for the selected washout target and checkpoint-ensemble method.

## One-Sentence Result

Across a fresh `420/420` grouped-video confirmation matrix, Neural Bridge ranked future human-arousal movement better than a strong target-specific autoregressive baseline and every matched control in all `15/15` fold-groups, with all preregistered gates passing.

## Magnitude at a Glance

Phase 7 is the culmination of a much larger bridge effect than its final `8.22%` Spearman and `8.97%` top-5% gains over AR imply:

- original same-target grouped spike: raw cortical `0.136579` → frozen-AR residual bridge `0.2383409298`, or `+74.51%`;
- same target: learned bridge `+39.95%` over direct AR-plus-raw and `+38.85%` over the Phase 4 bridge;
- original validated continuous bridge → Phase 7: `+16.61%` Spearman, `+23.59%` top-5% lift, and `+14.52%` top-1% lift;
- original → Phase 7 top-5% margin beyond AR: `+98.92%`; and
- current confirmation strength: `420/420`, `15/15` fold-group wins versus AR and controls on both primary metrics, failed gates `[]`.

The spike percentages are same-target comparisons. The continuous percentages compare complete system generations on the same metric family. They are not cross-metric arithmetic.

## Plain-Language Explanation

Recent arousal has momentum: if arousal is rising now, a simple model can often guess that it will remain elevated for a short time. Phase 7 asked whether Neural Bridge could do more than repeat that obvious pattern.

It could. On videos excluded from training, the system was better at ordering upcoming moments from smaller to larger true response movement. Its highest-ranked 5% of moments also contained more real future movement than the strongest baseline and false-signal controls.

In practical terms, the model produces a more informative map of *where the next meaningful human-response changes are likely to occur*. It is a ranking/selection result, not a claim that every future arousal decimal is exact.

## Scientific Result

### Protocol

- dataset: AGAIN dense 2 Hz, `995` cleaned videos;
- target: `residual_future_max_delta_rows_4_10`;
- architecture: `short_temporal_conv_residual`;
- stabilization: fixed equal-weight three-checkpoint ensemble;
- evaluation: five grouped held-out-video folds;
- fresh seeds: `20260708`–`20260716`;
- fixed checkpoint groups: three;
- matrix: `315` member rows + `105` ensemble rows = `420/420`;
- acceleration: MLX on `Device(gpu, 0)`;
- result: `grouped_continuous_ranking_lift_pass: true`;
- failed gates: `[]`.

### Primary metrics

| Metric | Real ensemble | Target-specific AR | Best matched control | Real − AR | Real − best control |
| --- | ---: | ---: | ---: | ---: | ---: |
| Spearman | `0.2603011121` | `0.2405371348` | `0.2402523335` | `+0.0197639773` | `+0.0200487786` |
| Top-5% true-movement lift | `0.0975979581` | `0.0895663763` | `0.0897088493` | `+0.0080315818` | `+0.0078891089` |

Relative to AR, this is an `8.22%` improvement in Spearman and an `8.97%` improvement in top-5% lift.

These are last-mile lifts over the dominant learned persistence signal, not a measure of the project's total value. Arousal autocorrelation makes AR unusually strong; the bridge is deliberately scored only on what it adds after that easy signal is already captured.

### Improvement over the original validated continuous bridge

The original controlled grouped continuous result was the Phase 5 eval-mode `regression_plus_binary` lane. Recomputing its stored 15 grouped fold-seed rows on the same summary metrics gives:

| Metric | Original validated bridge | Phase 7 | Absolute increase | Relative increase |
| --- | ---: | ---: | ---: | ---: |
| Future-movement Spearman | `0.2232222830` | `0.2603011121` | `+0.0370788291` | `+16.61%` |
| Top-5% true-movement lift | `0.0789694843` | `0.0975979581` | `+0.0186284738` | `+23.59%` |
| Top-1% true-movement lift | `0.1359465244` | `0.1556892559` | `+0.0197427315` | `+14.52%` |

The top-5% margin over AR increased from `+0.0040375083` to `+0.0080315818`, a `+98.92%` increase. Phase 7 therefore nearly doubled the bridge-added top-5% signal beyond AR.

This is an honest system-generation comparison rather than a controlled one-component ablation. The target/window, AR construction, temporal residual design, training protocol, and checkpoint stabilization all improved between the original bridge and Phase 7. It answers the practical question “how much better is the best system now?”; it does not assign the gain to one component.

### Full AGAIN spike/event progression

The original grouped spike target provides a same-target view of how the bridge was created:

| Stage | Grouped PR-AUC | Change from early raw cortical |
| --- | ---: | ---: |
| early raw cortical only | `0.136579` | baseline |
| early trained AR only | `0.147251` | `+7.81%` |
| early AR + raw cortical | `0.170299` | `+24.69%` |
| Phase 4 fold-safe PCA bridge | `0.1716477402` | `+25.68%` |
| Phase 5 deterministic learned bridge | `0.2300639382` | `+68.45%` |
| Phase 5 frozen-AR residual bridge | `0.2383409298` | `+74.51%` |

Within that original grouped target, the frozen-AR residual bridge is `+39.95%` above direct AR-plus-raw and `+38.85%` above the canonical Phase 4 score. This is the strongest apples-to-apples numerical demonstration that the learned Neural Bridge stack—not the raw upstream representation—created the value.

The later redesigned future-event target strengthened the scientific test and then passed both blocked and grouped confirmation. The fresh Phase 6 original-recipe ensemble reached grouped PR-AUC `0.2343675680` versus AR `0.2180497906` and best control `0.2179716645`, with `15/15` positive fold-groups. Its grouped margin over AR (`+0.0163177774`) is `17.50%` larger than the earlier promoted single-model margin (`+0.0138878634`).

For a broad beginning-to-current trajectory, the Phase 6 grouped binary ensemble is `71.60%` above the earliest grouped raw-cortical PR-AUC. The later target is intentionally different, so this number describes complete system progress, while its matched AR/control comparisons establish the formal claim.

Phase 7's continuous predictions also preserve event ranking. The stored secondary `binary_pr_auc_from_continuous_prediction` is `0.2231895329` for the real ensemble versus `0.2088047413` for frozen AR and `0.2096090680` for the strongest control. That is `+6.89%` over AR and `+6.48%` over control, positive in `15/15` fold-groups. It is supporting evidence, not a retroactively added Phase 7 promotion gate.

### Why the bridge effect is larger than “8%” sounds

The early raw-representation ablation, evaluated consistently within its own blocked binary target, showed:

| Lane | PR-AUC | Relative to trained AR |
| --- | ---: | ---: |
| trained `AR_only` | `0.203622` | baseline |
| `raw_cortical_only` | `0.124315` | `-38.95%` |
| `AR_plus_raw_cortical` | `0.167731` | `-17.63%` |

The raw representation could not beat persistence, and direct concatenation actively degraded the trained AR model. Phase 7 is the opposite regime: the full residual bridge beats the already-strong target-specific AR and every matched control, in every fold-group.

Within Phase 7 itself, ensemble Spearman by lane was:

| Lane | Mean Spearman | Meaning |
| --- | ---: | --- |
| real Neural Bridge residual | `0.2603011121` | frozen AR + real fold-safe neuro-response bridge |
| frozen AR only | `0.2405371348` | trained persistence floor |
| diagnostics-only residual | `0.2402752332` | non-cortical diagnostic correction |
| train-only video-mean residual | `0.2402523335` | static/base-rate control |
| random-PCA residual | `0.2399851718` | matched random representation |
| shuffled-PCA residual | `0.2398151737` | real representation with correspondence destroyed |
| label-permutation residual | `0.2360263012` | residual-null control |

The controls cluster at or below AR; only the correctly aligned real bridge separates materially. This is the key causal pattern. It is not “a model with more parameters beat a simple baseline.” It is “the real aligned neuro-response bridge adds signal while equally trained false-signal versions do not.”

The early ablation uses PR-AUC on a different target, while Phase 7 uses continuous Spearman/top-5% lift. A single cross-task improvement percentage would be scientifically invalid. The defensible, stronger conclusion is qualitative and replicated: Neural Bridge converted raw features that had negative incremental value into a consistently positive forward-looking correction over a learned persistence ceiling.

### Baseline definitions

- `AR-only`: a standalone trained model from observed current arousal, lag-1/2/4 arousal, and recent deltas.
- `trained AR`: the target-, split-, fold-, and seed-specific AR fit, selected using training-side inner validation.
- `frozen AR`: the trained AR checkpoint's scores fixed before residual/control training and reused identically beneath all real and control lanes.
- `raw cortical only`: a simple model over a deterministic projection of upstream predicted cortical/fMRI features, without the Neural Bridge stack.
- `AR plus raw cortical`: direct feature concatenation of AR context and raw cortical projection; it tested whether brute-force addition was sufficient.
- `real residual`: the Neural Bridge correction learned over frozen AR using fold-safe neuro-response PCA and causal temporal/event context.
- `checkpoint ensemble`: the equal-weight average of three independently trained, prespecified bridge checkpoints.

### Consistency and stabilization

- Spearman wins versus AR: `15/15` fold-groups;
- Spearman wins versus best matched control: `15/15`;
- top-5% wins versus AR: `15/15`;
- top-5% wins versus best matched control: `15/15`;
- positive fold means: `5/5`;
- ensemble Spearman uplift over member mean: `+0.0077966938`;
- ensemble top-5% uplift over member mean: `+0.0025021192`.

The consistency is as important as the average delta. No single favorable fold or seed is needed to create the result: every fold-group comparison is positive.

### Controls and audits

The confirmation compared the real bridge against target-specific AR and matched shuffled-PCA, random-PCA, diagnostics-only, train-only video-mean, and label-permutation residual lanes. Grouped-PCA leakage, causal context, frozen-AR identity, checkpoint restoration, exact scope, paired medians, single-group contribution, and accelerator audits passed.

The best aggregate Spearman control was `train_only_video_mean_residual`; the best top-5% control was `random_pca_residual`. Neural Bridge beat both.

## Relationship to Earlier Phases

Phase 5.5 established the selected binary future-event head under blocked and grouped controls. Phase 6 showed that the original recipe becomes stronger and more stable when three independently trained checkpoints are averaged, then confirmed that ensemble under fresh blocked and grouped binary protocols.

Phase 7 applied that stabilization lesson to the continuous washout-gap future-movement target. It is now the strongest continuous evidence in the repository because it combines:

- the redesigned future window;
- the proven temporal residual architecture;
- target-specific AR;
- fixed checkpoint averaging;
- fresh grouped held-out-video confirmation;
- a full matched-control matrix; and
- perfect fold-group directional consistency.

The earlier deterministic Phase 5 eval-mode lane remains valid independent evidence for grouped continuous ranking/lift. Phase 7 does not erase it; it strengthens and modernizes the continuous claim.

## Cross-Domain Meaning

VEATIC-124 v2 first established controlled future arousal event ranking on edited affective media. AGAIN then reproduced and extended the effect on a much larger gaming/interactive-media benchmark using V-JEPA 2.1 / TRIBE v2 predicted cortical response features.

Together, the datasets show that Neural Bridge is not supported by one content type alone. The current Phase 7 numbers are AGAIN-specific, while VEATIC supplies the independent film/TV/documentary/home-video foundation. A future balanced VEATIC+AGAIN training experiment could test whether shared training further improves stability and domain transfer.

## What Is Proven

- controlled future human-arousal event ranking across VEATIC and AGAIN;
- bounded blocked and grouped binary event-ranking confirmation on AGAIN;
- controlled grouped held-out-video continuous future-movement ranking/lift on AGAIN;
- an independent Phase 7 grouped continuous pass for the selected washout target/head;
- signal beyond recent-arousal persistence and multiple matched controls;
- material checkpoint-ensemble uplift;
- consistent positive performance across every Phase 7 fold-group.

## What the Result Does Not Need to Be Valuable

Exact-value forecasting is not required for the result to matter. Creative evaluation, highlight selection, weak-segment detection, response-event triage, and variant comparison depend heavily on ranking: which moments are likely to matter most, and where response movement is concentrated.

Phase 7 also reported slightly better descriptive MAE and RMSE than AR, but exact-value forecasting was not preregistered as a promotion claim. The project therefore does not inflate those descriptive numbers into a solved exact-trajectory claim.

## Current Deployment Boundary

Phase 7 is the strongest observed-arousal-assisted scientific ceiling. The separately locked direct-supervised deployment bridge now proves cached-feature inference on 299 prospectively held-out AGAIN videos with no observed arousal at inference. End-to-end raw-video feature generation, external transfer, and prospective client outcomes remain the next validation layer.

## Metric Source Map

| Metric family | Canonical source |
| --- | --- |
| VEATIC `0.2536` / `0.1969` / `0.1840` / `0.1944` and balanced `0.3394` | [`benchmarks/veatic/veatic_124_confirmatory_benchmark_report_20260616.md`](../benchmarks/veatic/veatic_124_confirmatory_benchmark_report_20260616.md) and [`benchmarks/veatic/veatic_124_event_conditioned_retest_20260616.md`](../benchmarks/veatic/veatic_124_event_conditioned_retest_20260616.md) |
| Early AGAIN blocked and grouped raw/AR/AR+raw ablations | [`evidence/current_phase_5_5_review/05_again_phase_3_raw_cortical/README.md`](../evidence/current_phase_5_5_review/05_again_phase_3_raw_cortical/README.md) and its tracked [`promotion_gates.json`](../evidence/current_phase_5_5_review/05_again_phase_3_raw_cortical/outputs/again_dense_2hz_raw_cortical_benchmark_20260625_093733/promotion_gates.json) |
| Phase 4 grouped bridge reference | [`best_lanes_by_target.json`](../evidence/current_phase_5_5_review/06_again_phase_4_pca_bridge/external_root_phase4_snapshot/promotion/best_lanes_by_target.json) and the tracked [Phase 4 promotion report](../evidence/current_phase_5_5_review/06_again_phase_4_pca_bridge/reports/again_dense_2hz_phase4_pca_promotion_summary_20260625_153419.md) |
| Original Phase 5 grouped event and continuous ranking/lift | [`reports/again_dense_2hz_phase5_evalmode_rescore_summary_.md`](../reports/again_dense_2hz_phase5_evalmode_rescore_summary_.md) and tracked [`evalmode_summary_metrics.csv`](../evidence/current_phase_5_5_review/07_again_phase_5_0_evalmode_primary/evalmode_rescore_bundle/metrics/evalmode_summary_metrics.csv) |
| Phase 5 frozen-AR residual `0.2383409298` | [`reports/again_dense_2hz_phase5_frozen_ar_residual_summary_.md`](../reports/again_dense_2hz_phase5_frozen_ar_residual_summary_.md) |
| Phase 5.5 blocked and grouped washout confirmations | [`reports/again_dense_2hz_phase5_temporal_residual_binary_big_confirm_20260630_025437.md`](../reports/again_dense_2hz_phase5_temporal_residual_binary_big_confirm_20260630_025437.md) and [`reports/again_dense_2hz_phase5_temporal_residual_grouped_compat_20260630_033520_UPDATED_VERDICT.md`](../reports/again_dense_2hz_phase5_temporal_residual_grouped_compat_20260630_033520_UPDATED_VERDICT.md) |
| Phase 6 grouped binary ensemble | [`reports/again_dense_2hz_phase6_original_three_checkpoint_grouped_confirmation_20260714_163024.md`](../reports/again_dense_2hz_phase6_original_three_checkpoint_grouped_confirmation_20260714_163024.md) |
| Phase 7 grouped metrics, controls, event support, and generation comparison | [`reports/again_dense_2hz_phase7_continuous_checkpoint_ensemble_grouped_20260714_181440.md`](../reports/again_dense_2hz_phase7_continuous_checkpoint_ensemble_grouped_20260714_181440.md) and the checksum-anchored evidence snapshot below |
| Locked zero-label-at-inference deployment confirmation | [`reports/again_dense_2hz_zero_label_direct_supervised_locked_confirmation_20260715.md`](../reports/again_dense_2hz_zero_label_direct_supervised_locked_confirmation_20260715.md) |

## Canonical Evidence

- grouped report: `reports/again_dense_2hz_phase7_continuous_checkpoint_ensemble_grouped_20260714_181440.md`
- grouped evidence snapshot: `evidence/phase_7_continuous_checkpoint_ensemble_grouped_20260714_181440/README.md`
- grouped preregistration: `docs/phase7_continuous_checkpoint_ensemble_grouped_preregistration.md`
- grouped reproduction: `studies/again/phase-07-continuous/replay/grouped-fold1-seed20260708-real.json` through `python -m neural_bridge.again replay-checkpoint`
- grouped contract test: `tests/test_again_phase7_continuous_checkpoint_ensemble_grouped.py`
- diagnostic report: `reports/again_dense_2hz_phase7_continuous_checkpoint_ensemble_diagnostic_20260714_174513.md`
- machine-readable status: `docs/current_claim_status.json`
- historical discovery ladder: `docs/how_neural_bridge_was_discovered.md`

## Current Bottom Line

Phase 7 is a real, clean win: Neural Bridge consistently ranks future arousal movement on unseen videos better than strong persistence and matched false-signal baselines. The result passed a fresh `420/420` controlled matrix with `15/15` positive fold-groups and no failed gates. The newer locked deployment confirmation shows that substantial signal survives without observed response labels at inference; raw-video runtime and external transfer are now the next layer.
