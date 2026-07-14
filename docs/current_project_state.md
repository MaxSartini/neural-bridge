# Current Project State

Last updated: 2026-07-14

## Headline

Phase 7 is the strongest current Neural Bridge result. Its separately preregistered grouped held-out-video continuous confirmation passed exactly `420/420` scored rows with failed gates `[]`. Neural Bridge beat target-specific AR and the best matched controls on both future-movement Spearman and top-5% lift in all `15/15` fold-groups and all `5/5` fold means.

Plainly: on unseen videos, Neural Bridge is better than recent-response momentum and false-signal controls at identifying where the largest upcoming human-arousal movements will occur.

Compared with the original validated grouped continuous bridge, Phase 7 increases Spearman by `16.61%` (`0.2232222830` → `0.2603011121`), top-5% lift by `23.59%` (`0.0789694843` → `0.0975979581`), and top-1% lift by `14.52%` (`0.1359465244` → `0.1556892559`). Its top-5% margin over AR nearly doubled: `+98.92%` larger than the original bridge margin.

Within the original same-target grouped spike benchmark, raw cortical-only `0.136579` grew to the frozen-AR residual bridge at `0.2383409298`: `+74.51%`. That learned bridge was `+39.95%` above direct AR-plus-raw and `+38.85%` above the canonical Phase 4 score `0.1716477402`. These are the clearest apples-to-apples measures of the value created by Neural Bridge rather than the upstream representation.

The `8.22%` Spearman and `8.97%` top-5% Phase 7 lifts over AR are the difficult final increment over a trained persistence model that already captures most easy short-horizon signal. They are not the total value created. On the early blocked ablation, raw cortical-only was `38.95%` below trained AR, and directly adding raw cortical features remained `17.63%` below AR. Phase 7 turns that former negative contribution into a positive correction in every fold-group.

The later fresh grouped redesigned-target binary ensemble reached `0.2343675680` versus AR `0.2180497906`, positive `15/15`; its margin over AR is `17.50%` larger than the earlier promoted single-model margin. Phase 7's continuous predictions additionally rank the corresponding event at `0.2231895329` PR-AUC vs AR `0.2088047413`, positive `15/15`, as supporting evidence.

## Canonical Scientific Claim

Neural Bridge demonstrates controlled future human-arousal event ranking across VEATIC and AGAIN, plus controlled grouped held-out-video continuous future-arousal movement ranking/lift on AGAIN, using frozen predicted cortical/fMRI response features generated from video by upstream models trained on brain-response data.

Phase 7 independently confirms the continuous claim for `residual_future_max_delta_rows_4_10` with `short_temporal_conv_residual` and fixed three-checkpoint averaging.

This is not generic video embedding performance. Raw predicted cortical/fMRI features alone were weak on AGAIN and could damage AR when attached directly. The signal became useful through the Neural Bridge design: fold-safe representation construction, frozen-AR residual learning, future-target redesign, temporal/event context, matched controls, strict held-out evaluation, and checkpoint stabilization.

## Phase 7 Numbers

| Metric | Neural Bridge | AR | Best control | Delta vs AR | Delta vs control |
| --- | ---: | ---: | ---: | ---: | ---: |
| Spearman | `0.2603011121` | `0.2405371348` | `0.2402523335` | `+0.0197639773` | `+0.0200487786` |
| Top-5% lift | `0.0975979581` | `0.0895663763` | `0.0897088493` | `+0.0080315818` | `+0.0078891089` |

- matrix: `420/420` (`315` member + `105` ensemble);
- held-out-video folds: five;
- untouched seeds: nine;
- fixed checkpoint groups: three;
- wins versus AR: `15/15` Spearman and `15/15` top-5%;
- wins versus best matched controls: `15/15` and `15/15`;
- positive fold means: `5/5`;
- ensemble uplift: `+0.0077966938` Spearman and `+0.0025021192` top-5%;
- accelerator: MLX `Device(gpu, 0)`;
- failed gates: `[]`.

The separate Phase 7 blocked confirmation was positive on aggregate and passed every gate except the locked unanimity requirement: Spearman beat AR in `4/5` rather than `5/5` groups. It remains a strong near-pass and is not rewritten as a formal blocked pass.

Terminology: AR-only is a trained model using current/lagged arousal and recent deltas; frozen AR is its target/fold/seed-specific fixed prediction reused identically under all residual lanes; AR-plus-raw is direct feature concatenation; Neural Bridge learns a causal, fold-safe residual correction over frozen AR. This matched construction is why the Phase 7 lift represents new video-side signal rather than a weaker baseline.

## Evidence Ladder

1. VEATIC-124 v2 established controlled future arousal event/spike ranking on edited affective video.
2. AGAIN scaled the work to `995` cleaned gameplay videos and `243,575` aligned 2 Hz feature rows using official V-JEPA 2.1 ViT-G and TRIBE v2.
3. Raw cortical-feature controls showed that upstream features alone were not sufficient.
4. Frozen-AR residual learning and the washout-gap target established the successful design path.
5. Phase 5.5 confirmed the selected binary target/head under blocked and grouped protocols.
6. Phase 6 established checkpoint stabilization and passed fresh control-complete binary ensemble confirmations.
7. Phase 7 applied the stabilized method to continuous future movement and passed the fresh grouped `420/420` confirmation with perfect fold-group directional consistency.

The result now spans edited film/TV/documentary/home-video affect content through VEATIC and scaled gameplay/interactive media through AGAIN. That is cross-domain evidence, while the strongest Phase 7 metric values are specifically from AGAIN.

## Binary Evidence Remains Strong

The selected binary event-ranking head `future_arousal_max_delta_rows_4_10_train_q90` / `short_temporal_conv_residual` remains promoted:

- canonical blocked real / AR / best-control PR-AUC: `0.2670735630` / `0.2602336231` / `0.2593369051`;
- blocked deltas: `+0.0068399399` / `+0.0077366579`, with `9/10` positive seeds;
- updated grouped real / AR / best-control: `0.2313831909` / `0.2174953276` / `0.2174209937`, with `50/50` positives versus best control;
- unified selected-head audit: `420/420`, failed gates `[]`;
- fresh Phase 6 original-ensemble blocked confirmation: `140/140`, `5/5` positive groups;
- fresh Phase 6 original-ensemble grouped confirmation: `420/420`, `15/15` positive fold-groups.

Phase 7 is the headline because it advances the harder continuous future-movement ranking/lift line; it builds on rather than replaces the binary proof.

## Product Interpretation

Neural Bridge is a Service-as-Software product direction for neuro-response video intelligence: first-pass response evaluation, high-response moment ranking, weak-segment diagnosis, variant comparison, and response-readiness reporting.

The current benchmark proves that video-derived predicted neuro-response features add consistent forward-looking information beyond observed arousal persistence. The next deployment milestone is a video-only student or cold-start/self-rollout bridge, because the strongest residual benchmark currently consumes observed current/past arousal that a raw client video will not provide.

This is the difference between scientific capability and product packaging: Phase 7 proves the capability; the next experiment removes the deployment-time response-label dependency.

## Precise Boundaries

Do not claim:

- mind reading or individual profiling;
- medical or diagnostic inference;
- exact continuous trajectory values are solved;
- the blocked Phase 7 result was literal `5/5`;
- raw-video-only client deployment is already validated;
- universal emotion prediction or guaranteed campaign outcomes;
- the obsolete `504` matrix was run or promoted.

Do state plainly:

- grouped continuous future-movement ranking/lift is proven;
- Phase 7 is an independent, fresh, all-gates-passed grouped confirmation;
- every Phase 7 fold-group beat AR and matched controls;
- checkpoint averaging materially improved the result;
- the bridge, not raw predicted cortical features alone, creates the usable signal;
- VEATIC and AGAIN provide meaningful cross-domain evidence.

## Canonical Entry Points

- `README.md`
- `docs/neural_bridge_phase7_evidence.md`
- `docs/current_claim_status.json`
- `reports/again_dense_2hz_phase7_continuous_checkpoint_ensemble_grouped_20260714_181440.md`
- `evidence/phase_7_continuous_checkpoint_ensemble_grouped_20260714_181440/README.md`
- `docs/how_neural_bridge_was_discovered.md`
- `docs/neural_bridge_service_as_software.md`
- `docs/executable_validation_index.md`
- `reports/README.md`

Historical Phase 5/5.5 and Phase 6 reports remain valid evidence for the development ladder. They are not the current ceiling.

## Current Next Work

1. Follow `docs/zero_label_video_only_deployment_bridge_pilot_preregistration.md`.
2. Stage 0 is contracts/manifests/dry-run accounting only; do not fit or score a model.
3. Resolve and checksum the Phase 7 grouped target-name/value-array identity before fitting, using a dedicated video-only block with no AR arrays.
4. Stage A (`96/96`) and Stage B (`140/140`) each require separate authorization and cold-start inference with no observed-arousal teacher forcing.
5. If the bridge passes, consider a bounded V-JEPA 2.1 VEATIC re-encode pilot and balanced VEATIC+AGAIN joint-training study with harmonized targets and leave-one-domain-out evaluation.
6. Do not resume same-family Optuna, recreate 504, or launch an architecture zoo without a new locked hypothesis.

## Validation and Handoff

Run before handoff:

```bash
npm run verify
npm run audit:repo
npm run verify:research-tooling
```

ML work must use MLX/MPS and fail closed rather than silently falling back to CPU. Heavy outputs remain ignored; claim-bearing summaries and checksum anchors are tracked.
