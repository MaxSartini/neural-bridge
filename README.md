# Neural Bridge

Neural Bridge predicts which upcoming moments are most likely to produce the strongest changes in human arousal, using video-derived predicted cortical/fMRI response features. Its strongest result is Phase 7: a fresh, preregistered grouped-video confirmation that passed all gates at exactly `420/420` scored rows.

## From Failed Raw Representation to Phase 7

The headline is not merely that Phase 7 is `8%` better than AR. Neural Bridge turned predicted cortical/fMRI features that were weak—or actively harmful when fused naively—into a repeatable forward-looking signal:

- on the original same-target grouped AGAIN spike benchmark, the bridge moved from raw cortical PR-AUC `0.136579` to `0.2383409298`: **`+74.51%`**;
- the learned bridge was **`+39.95%`** above direct trained-AR-plus-raw fusion and **`+38.85%`** above the Phase 4 PCA bridge;
- against the original validated continuous bridge, Phase 7 is **`+16.61%`** higher on Spearman, **`+23.59%`** higher on top-5% lift, and **`+14.52%`** higher on top-1% lift;
- the useful top-5% margin added beyond AR grew by **`+98.92%`**—almost exactly double; and
- the current Phase 7 confirmation completed **`420/420`** rows and won **`15/15`** fold-groups versus both AR and matched controls on both primary metrics.

These are deliberately separated comparisons. The `+74.51%`, `+39.95%`, and `+38.85%` figures are within the original grouped spike target. The Phase 7 percentages are whole-system generation comparisons within continuous ranking/lift. No percentage is calculated across PR-AUC and Spearman or across unrelated targets.

| Development stage | Result | What changed |
| --- | --- | --- |
| Early blocked AGAIN raw ablation | raw `0.124315`, AR `0.203622`, AR+raw `0.167731` PR-AUC | raw features were `38.95%` below trained AR; naïve fusion was still `17.63%` below AR |
| Original grouped spike progression | raw `0.136579` → Phase 5 frozen-residual bridge `0.2383409298` PR-AUC | same-target bridge gain `+74.51%`; `+39.95%` over direct AR+raw |
| Original grouped continuous win | Spearman `0.2232222830`; top-1% lift `0.1359465244` | first controlled grouped continuous-ranking/lift victory |
| Phase 5.5 stricter washout confirmation | blocked `9/10` seeds; grouped `50/50` fold-seeds; failed gates `[]` | beat matched frozen AR after an explicit persistence washout gap |
| Phase 6 checkpoint stabilization | grouped binary `0.2343675680` vs AR `0.2180497906`; `15/15` | fixed three-checkpoint averaging became a validated stability mechanism |
| Phase 7 culmination | `420/420`; Spearman `0.2603011121`; top-5% lift `0.0975979581`; `15/15` | strongest current grouped continuous result, beyond AR and every matched control |

## The Result in Plain English

Give the system a video sequence and recent response context. Neural Bridge ranks the moments ahead by how much human arousal is likely to move. On videos held out from training, it ranked those future changes better than:

- a strong autoregressive baseline that predicts from recent arousal momentum;
- shuffled and random feature controls;
- label-permutation controls; and
- a train-only video-mean control.

This was not a one-fold or one-seed fluke. Phase 7 won in all `15/15` fold-groups and all `5/5` held-out-video fold means on both future-movement rank correlation and top-5% lift. Every preregistered grouped gate passed.

The simplest interpretation is: Neural Bridge identifies *where the meaningful future response movement is likely to be*, beyond merely assuming that current arousal will continue. It is closer to a reliable forward-looking response heat map than an exact second-by-second arousal meter.

## Phase 7: Strongest Current Evidence

The selected continuous target is `residual_future_max_delta_rows_4_10`, evaluated with `short_temporal_conv_residual` and fixed three-checkpoint averaging.

| Grouped held-out-video metric | Neural Bridge | AR baseline | Best matched control | Improvement over AR |
| --- | ---: | ---: | ---: | ---: |
| Future-movement Spearman | `0.2603011121` | `0.2405371348` | `0.2402523335` | `+0.0197639773` (`+8.22%`) |
| Top-5% true-movement lift | `0.0975979581` | `0.0895663763` | `0.0897088493` | `+0.0080315818` (`+8.97%`) |

Those percentages describe only the final increment over the strongest cheap predictor. They do **not** describe the total Neural Bridge transformation. AR already captures most easy short-horizon signal because human arousal is highly persistent. The harder question is whether video-derived neuro-response features add anything reliable after that advantage is removed. Phase 7 says yes in every fold-group.

### Original validated bridge → Phase 7

Compared with the original validated grouped continuous bridge from the Phase 5 eval-mode lane, Phase 7 is a major step forward:

| Grouped continuous metric | Original bridge | Phase 7 | Increase |
| --- | ---: | ---: | ---: |
| Future-movement Spearman | `0.2232222830` | `0.2603011121` | `+0.0370788291` (`+16.61%`) |
| Top-5% true-movement lift | `0.0789694843` | `0.0975979581` | `+0.0186284738` (`+23.59%`) |
| Top-1% true-movement lift | `0.1359465244` | `0.1556892559` | `+0.0197427315` (`+14.52%`) |

The top-5% advantage over AR grew from `+0.0040375083` in the original validated bridge to `+0.0080315818` in Phase 7—a `+98.92%` increase in the bridge's incremental margin. In other words, Phase 7 nearly doubled the useful top-5% signal added beyond persistence.

This is a project-generation comparison, not a single-variable ablation: Phase 7 also uses the improved washout window, target-specific AR, selected temporal residual head, newer training discipline, and checkpoint averaging. That is precisely what “Phase 7 is the king” means—the current end-to-end method is materially stronger than the original validated continuous system.

### Original AGAIN spike/event results → current bridge

The event-ranking line improved just as dramatically. On the original grouped AGAIN spike target—the cleanest like-for-like early progression—the system moved through these stages:

| Grouped spike/event stage | PR-AUC | What it showed |
| --- | ---: | --- |
| raw cortical only | `0.136579` | upstream representation alone was weak |
| trained AR only | `0.147251` | persistence beat raw cortical |
| trained AR + raw cortical | `0.170299` | direct concatenation helped grouped ranking but was not a robust bridge |
| Phase 4 fold-safe PCA bridge | `0.1716477402` | first controlled grouped bridge recovery |
| Phase 5 deterministic learned bridge | `0.2300639382` | `+68.45%` over raw cortical |
| Phase 5 frozen-AR residual bridge | `0.2383409298` | `+74.51%` over raw cortical and `+39.95%` over AR + raw |

The project then redesigned the future-event target to beat AR under strict blocked time as well as grouped video. The promoted binary line passed:

- blocked single-model confirmation: `0.2670735630` vs AR `0.2602336231`, `9/10` positive seeds;
- grouped single-model confirmation: `0.2313831909` vs AR `0.2174953276`, `50/50` positive fold-seeds versus best control;
- fresh blocked three-checkpoint ensemble: `0.2668905427` vs AR `0.2597235728`, `5/5` positive groups;
- fresh grouped three-checkpoint ensemble: `0.2343675680` vs AR `0.2180497906`, `15/15` positive fold-groups.

Compared with the earliest grouped raw-cortical spike score, the fresh grouped binary ensemble is `+71.60%` higher in PR-AUC. Because the target/window changed to make the later task scientifically stronger, that `71.60%` is a whole-project trajectory rather than a controlled same-target lift; the within-stage AR/control deltas remain the claim-bearing comparisons.

Phase 7 also retains event information inside its continuous predictions. As a supporting secondary metric, thresholding/ranking the Phase 7 continuous output against the corresponding future event gives PR-AUC `0.2231895329` versus AR `0.2088047413` and the strongest control `0.2096090680`: `+6.89%` over AR and `+6.48%` over control, positive in `15/15` fold-groups. This was not the primary Phase 7 promotion gate, but it shows the continuous win did not abandon the spike/event capability.

Protocol strength:

- exact matrix completion: `420/420` rows (`315` members + `105` ensembles);
- five held-out-video folds, nine untouched seeds, and three fixed checkpoint groups;
- positive versus AR: `15/15` fold-groups on both metrics;
- positive versus the best matched control: `15/15` on both metrics;
- positive fold means: `5/5`;
- checkpoint averaging added `+0.0077966938` Spearman and `+0.0025021192` top-5% lift over the member mean;
- leakage, causal-context, frozen-AR identity, checkpoint-restoration, scope, and MLX audits passed;
- failed gates: `[]`.

That is a clean controlled grouped continuous future-movement ranking/lift result.

The separate Phase 7 blocked-time stress test was also strongly positive on aggregate. It beat AR and the best control in mean Spearman and top-5% lift, and passed every locked gate except unanimity versus AR: `4/5` groups instead of the preregistered `5/5`. That result remains an accurately reported near-pass; it neither weakens nor replaces the separate grouped `420/420` pass.

## Scientific Claim

Neural Bridge demonstrates controlled future human-arousal event ranking across VEATIC and AGAIN, and controlled grouped held-out-video continuous future-arousal movement ranking/lift on AGAIN. The signal comes from a bridge built over frozen predicted cortical/fMRI response features generated from video by upstream models trained on brain-response data.

Scientifically, Phase 7 shows that the selected model ranks future arousal movement better than recent-response persistence and multiple matched null/control constructions on unseen videos. The result is consistent across every fold-group in the confirmation matrix, and checkpoint averaging materially improves the result over individual trained members.

This matters because AR is not a weak baseline. Human arousal is persistent, so recent arousal already predicts near-future arousal surprisingly well. Beating that baseline means the video-side predicted neuro-response representation contributes forward-looking information beyond response momentum.

## Why “8% Better” Understates the Result

The project began with a representation that was not useful by itself on the early blocked AGAIN target:

| Earlier same-target ablation | PR-AUC | Compared with trained AR |
| --- | ---: | ---: |
| Trained AR-only | `0.203622` | baseline |
| Raw cortical only | `0.124315` | `-38.95%` |
| Trained AR + raw cortical | `0.167731` | `-17.63%` |

Raw cortical features did not merely fail to beat trained AR; naïvely adding them damaged it. The bridge's achievement is the transition from *negative incremental value* to *reliable positive incremental value*.

On the current Phase 7 target, the trained/frozen AR floor already reaches `0.2405371348` Spearman. Shuffled, random, label-permuted, static-video, and diagnostics-only residual controls remain around that floor (`0.2360`–`0.2403`). The real bridge reaches `0.2603011121`, wins all `15/15` fold-groups, and gains further from checkpoint averaging.

So the correct reading is not “the entire system is only 8% better.” It is:

1. persistence already solves a large, easy part of the problem;
2. raw neuro-response features initially had negative value relative to that baseline;
3. Neural Bridge learned to isolate the small, difficult, genuinely forward-looking component;
4. that component survives every fold-group and every matched null/control construction; and
5. the resulting model improves both ranking and concentration of true future movement.

The early PR-AUC ablation and Phase 7 Spearman result use different targets and metrics, so they must not be divided into one fake cross-task percentage. Together they show something more important: the bridge converted an unusable raw representation into reproducible incremental future-response intelligence.

## What Each Baseline Actually Is

- **AR-only** is not “copy the last number.” It is a trained target-specific model using observed current arousal, lag-1/2/4 arousal, and recent arousal deltas. Hyperparameters/checkpoints are selected on training-side inner validation.
- **Trained AR** emphasizes that the persistence baseline is fitted for the exact target, split, fold, and seed. It is a serious learned competitor.
- **Frozen AR** is that trained AR model's score/prediction after its checkpoint is fixed. The identical frozen values are reused underneath the real residual and every matched control, so Neural Bridge cannot win through an easier AR fit.
- **Raw cortical only** is the upstream predicted cortical/fMRI representation projected into a simple model without the Neural Bridge stack.
- **AR + raw cortical** directly concatenates trained-AR features and the raw cortical projection. Its failure showed that more features alone were not the answer.
- **Neural Bridge residual** starts from the frozen AR prediction and learns only the correction contributed by fold-safe neuro-response PCA, causal temporal/event context, the redesigned future target, and the selected residual head.
- **Phase 7 ensemble** averages three independently trained bridge checkpoints fixed in advance. It reduces checkpoint noise and added measurable performance over the individual members.

## Why Neural Bridge Is the Breakthrough

Raw predicted cortical/fMRI features were not enough. On the earlier AGAIN blocked target:

- `raw_cortical_only`: `0.124315` PR-AUC;
- AR-only: `0.203622`;
- direct `AR_plus_raw_cortical`: `0.167731`.

Simply attaching brain-prediction features to a trained baseline made the blocked result worse. Neural Bridge made them useful through fold-safe representation construction, frozen-AR residual learning, future-target redesign, temporal/event context, matched controls, checkpoint stabilization, and strict held-out evaluation.

The project contribution is therefore not “we used V-JEPA/TRIBE.” It is the bridge that turns predicted neuro-response features into reproducible future human-response signal.

## Evidence Across Domains

VEATIC-124 v2 established the original signal on edited affective video: film, documentary, reality-TV, and home-video content. AGAIN then scaled and strengthened the evidence on `995` cleaned gameplay videos, `243,575` aligned feature rows, nine games, three genres, and more than 37 hours of annotated material.

This is meaningful cross-domain evidence: the core effect appeared in edited real-world affect content and in interactive gameplay. AGAIN is the current large-scale benchmark; VEATIC is the independent foundation that showed the effect was not born in one gaming dataset.

The binary event-ranking line is also fully established. The selected AGAIN head passed blocked temporal confirmation, grouped-video confirmation, a unified `420/420` selected-head audit, and fresh three-checkpoint ensemble confirmations under both blocked (`140/140`) and grouped (`420/420`) protocols.

## What the Metrics Mean

- `Spearman` asks whether the model puts larger true future movements ahead of smaller ones in the correct order.
- `Top-5% lift` asks whether the moments ranked in the model's top 5% actually contain more true future movement.
- `PR-AUC` measures future-event ranking when the important events are rare.
- `AR` is the deliberately strong recent/past-arousal persistence model.
- `Grouped-video` means the evaluated videos were held out as groups, testing compatibility with unseen videos.
- `Blocked temporal` means training occurs earlier in time and evaluation later, testing strict forward-time behavior.

Think of the task as predicting where the next important response movement will occur. Phase 7 proves useful ranking and concentration of those moments. It does not depend on predicting the exact decimal value of the future trajectory to count as a meaningful success.

## Product Meaning

Neural Bridge is being built as Service as Software for neuro-response video intelligence: automated first-pass response evaluation, weak-segment diagnosis, likely high-response moment ranking, variant comparison, and response-readiness reporting.

The evidence supports the core intelligence layer. The next deployment milestone is to remove the benchmark's observed-arousal input at client inference time. The current strongest residual model uses current/past arousal as its strong AR floor; raw pre-release client video will not provide those labels. The planned bridge is therefore a video-only student or cold-start/self-rollout model tested with no teacher forcing.

That distinction is a product roadmap, not a dismissal of the result: Phase 7 proves that video-derived predicted neuro-response features contain consistent incremental information about future human response. The next task is to make that intelligence available from video alone.

Longer-term, a rigorously re-encoded V-JEPA 2.1 VEATIC dataset could be combined with AGAIN in a balanced multi-domain training design. That may improve stability and transfer—especially for the video-only student—but it should follow a bounded pilot with harmonized targets, domain-balanced sampling, and leave-one-domain-out evaluation rather than an expensive blind re-encode.

## Honest Boundaries

The current evidence does not claim mind reading, individual profiling, medical inference, exact continuous-value forecasting, universal emotion prediction, or guaranteed client outcomes.

In particular:

- grouped continuous future-movement ranking/lift is proven;
- the separate blocked continuous result is a strong `4/5` near-pass, not a formal `5/5` pass;
- exact trajectory values were not the Phase 7 promotion target;
- label-free raw-video deployment still requires the video-only bridge;
- the historical `504` design was not run or promoted.

These are precise scope boundaries around a real result—not language that turns a win into a failure.

## Start Here

- [Phase 7 evidence and interpretation](docs/neural_bridge_phase7_evidence.md)
- [Current project state](docs/current_project_state.md)
- [Machine-readable claim status](docs/current_claim_status.json)
- [Phase 7 grouped report](reports/again_dense_2hz_phase7_continuous_checkpoint_ensemble_grouped_20260714_181440.md)
- [Phase 7 evidence snapshot](evidence/phase_7_continuous_checkpoint_ensemble_grouped_20260714_181440/README.md)
- [Discovery history](docs/how_neural_bridge_was_discovered.md)
- [Service-as-Software thesis](docs/neural_bridge_service_as_software.md)
- [Executable validation index](docs/executable_validation_index.md)
- [Report authority index](reports/README.md)

## Run and Validate

Apple Silicon ML work uses MLX/MPS; Phase 7 ran on `Device(gpu, 0)` and does not permit silent CPU fallback.

```bash
npm run verify
npm run audit:repo
npm run verify:research-tooling
```

Heavy datasets, caches, checkpoints, and generated output roots remain outside git. See [REQUIREMENTS.md](REQUIREMENTS.md) and [docs/external_assets_manifest.md](docs/external_assets_manifest.md).

## Current Next Step

Freeze and run a bounded video-only deployment-bridge pilot using the Phase 7 teacher signal, with cold-start grouped-video evaluation and no observed-arousal teacher forcing. Only after that should the project decide whether a full V-JEPA 2.1 VEATIC re-encode and balanced VEATIC+AGAIN joint training is worth the cost.
