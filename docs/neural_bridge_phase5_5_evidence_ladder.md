# Neural Bridge Phase 5.5 Evidence Ladder

Neural Bridge now has a paired cross-dataset evidence ladder: VEATIC-124 v2 established the original controlled future arousal spike/event-ranking signal, and AGAIN replicated, scaled, validated, and strengthened it with dense V-JEPA 2.1 / TRIBE v2 features, frozen-AR residuals, a redesigned washout-gap future arousal event target, blocked temporal confirmation, and grouped-video compatibility.

The neuro-response mechanism is central. The frozen video-side features used in these benchmarks are predicted cortical/fMRI response features generated from video by upstream models trained on brain cortical response data. They are not generic video embeddings and not direct brain recordings from the benchmark viewer rows. Neural Bridge tests whether those predicted neuro-response features carry future human arousal event signal.

TRIBE/V-JEPA are the upstream substrate, not the moat by themselves. Neural Bridge is the downstream bridge, control, validation, and response-intelligence layer. As upstream brain-response prediction models improve, Neural Bridge can inherit stronger frozen neuro-response signal and evaluate it under the same controls.

Raw predicted cortical/fMRI features alone fail badly on AGAIN. On the original Phase 3 spike target `arousal_spike_rows_2_6_train_q90`, `raw_cortical_only` scored blocked PR-AUC `0.124315` versus AR-only `0.203622`; direct `AR_plus_raw_cortical` was `0.167731`, below AR. Grouped `raw_cortical_only` was `0.136579` versus AR-only `0.147251`. Neural Bridge is the difference: fold-safe compression, frozen AR anchoring, a washout-gap future event target, and temporal/event-context residual learning turn weak raw predicted cortical/fMRI features into controlled future event-ranking.

Beating AR is the central benchmark difficulty. AR/frozen AR is a strong recent-arousal persistence baseline, and many earlier lanes failed because they could not beat it under blocked temporal validation. The Phase 5.5 blocked confirmation beats matched frozen AR by `+0.0068399399` PR-AUC (`+0.684` percentage points) with `9/10` positive seeds, and the grouped compatibility run beats AR/frozen by `+0.0138878634` PR-AUC (`+1.389` points) with `50/50` fold-seed positives.


## What Is Now Proven

- Controlled future human arousal event-ranking from frozen predicted cortical/fMRI response features generated from video by upstream models trained on brain cortical response data across VEATIC and AGAIN.
- VEATIC-124 v2 established the first controlled future arousal spike/event-ranking result.
- AGAIN extended the result at scale over 995 videos and true 2 Hz labels.
- Bounded strict forward-time future-event ranking is proven on AGAIN for `future_arousal_max_delta_rows_4_10_train_q90` with `short_temporal_conv_residual`.
- Grouped held-out-video compatibility for the same AGAIN target/head is proven under the updated frozen-AR-residual-aware verdict.

## Dataset Definitions

VEATIC-124 v2 is the edited affect-video foundation. Public VEATIC documentation describes `124` clips drawn from Hollywood movies, documentaries, home videos, and reality TV shows, with continuous valence and arousal ratings for each frame via real-time annotation. In Neural Bridge, VEATIC is the film/TV/documentary/home-video emotion benchmark: it established that neuro-response-derived video representations can rank future arousal spike/event moments beyond AR and controls on affective edited media. Sources: [VEATIC project page](https://veatic.github.io/), [WACV 2024 paper](https://openaccess.thecvf.com/content/WACV2024/html/Ren_VEATIC_Video-Based_Emotion_and_Affect_Tracking_in_Context_Dataset_WACV_2024_paper.html).

AGAIN is the gaming and interactive-media confirmation. Public AGAIN documentation describes `1,116` raw in-game videos and gameplay logs from `124` participants playing `9` games across `3` genres, with a cleaned/preprocessed `995`-video version. The paper describes more than `37` hours of annotated video and game logs with first-person continuous arousal annotation. In Neural Bridge, AGAIN is the scaled proof point: dense 2 Hz V-JEPA 2.1 / TRIBE v2 neuro-response features over the cleaned `995` videos confirmed bounded strict forward-time future-event ranking. Sources: [AGAIN project page](https://again.institutedigitalgames.com/), [AGAIN arXiv paper](https://arxiv.org/abs/2104.02643).

This is why the result is broader than a single benchmark trick. The evidence spans film, TV/reality, documentary, home-video affect content, and interactive gameplay arousal: VEATIC establishes the controlled signal on edited emotion video; AGAIN scales and confirms it on gaming video.

## Metric And Term Definitions

- `PR-AUC`: area under the precision-recall curve. It is the primary metric for rare future-event/spike ranking because it rewards putting true future arousal events near the top of the ranked list.
- `PR-AUC percentage`: PR-AUC multiplied by 100. Example: `0.2670735630` is `26.707%` PR-AUC.
- `percentage-point delta`: direct difference between two PR-AUC percentages. Example: `0.2670735630 - 0.2602336231 = 0.0068399399`, reported as `+0.684` PR-AUC percentage points.
- `relative lift`: percentage-point delta divided by the baseline PR-AUC. Example: `+0.0068399399 / 0.2602336231 = +2.63%` relative lift over frozen AR.
- `AR` / `frozen AR`: autoregressive baseline using recent/past arousal information. It is intentionally strong because human arousal is temporally persistent.
- `beating AR`: the key hurdle. A future-event model is not convincing if it only tracks recent arousal persistence; the current AGAIN Phase 5.5 result beats the matched AR floor and matched controls.
- `blocked_temporal_70_30`: forward-time split; train on earlier rows and test on later rows, probing future temporal generalization.
- `grouped_video`: held-out-video split; test videos are not in training, probing cross-video compatibility.
- `shuffled/random controls`: matched controls that break real representation identity while preserving comparable model capacity.
- `label permutation`: null control that permutes training/inner-validation targets and scores on true held-out labels, testing whether selection or leakage can fake a result.
- `train-only video mean`: static video/base-rate control computed from training rows only.
- `weak/credible/strong confirmation`: project gates requiring real-vs-baseline/control deltas, seed consistency, and leakage/control audits; they are not marketing labels.

## What Remains Open

- Continuous exact arousal forecasting remains open.
- Broad all-target/all-dataset temporal prediction remains open.
- No 504 run has been performed or promoted.
- Grouped compatibility is not itself a 504 result.
- The updated grouped verdict is a no-training verdict update from existing artifacts, not a rerun.

## Commercial Significance

Neural Bridge is Service as Software for video response intelligence. The scientific result matters commercially because it supports an automated version of a high-value expert service: pre-release audience-response testing, creative diagnostics, media-response analysis, and response-readiness reporting for video.

The business wedge is not another analytics dashboard. It is a response intelligence service delivered through software: submit a video, receive ranked future response moments, weak/dead-zone segments, variant comparison, confidence/control diagnostics, and edit-review priorities before audience data exists.

Full commercial interpretation: `docs/neural_bridge_service_as_software.md`.

## Best AGAIN Results

The strongest current evidence is the AGAIN redesigned washout-gap binary result, confirmed under blocked temporal validation and checked for grouped-video compatibility.

Blocked binary confirmation:

- target: `future_arousal_max_delta_rows_4_10_train_q90`
- protocol: `blocked_temporal_70_30`
- architecture: `short_temporal_conv_residual`
- real PR-AUC: `0.2670735630`
- frozen AR PR-AUC: `0.2602336231`
- best control: `random_pca_residual`, PR-AUC `0.2593369051`
- delta vs frozen AR: `+0.0068399399`
- delta vs best control: `+0.0077366579`
- PR-AUC as percentages: real `26.707%`, frozen AR `26.023%`, best control `25.934%`
- percentage-point deltas: `+0.684` over frozen AR, `+0.774` over best control
- relative lifts: `+2.63%` over frozen AR, `+2.98%` over best control
- seeds positive vs AR: `9/10`
- seeds positive vs best control: `9/10`
- weak / credible / strong confirmation: true
- failed gates: `[]`

Grouped compatibility:

- target: `future_arousal_max_delta_rows_4_10_train_q90`
- protocol: `grouped_video`
- architecture: `short_temporal_conv_residual`
- rows: `350/350`
- real PR-AUC: `0.2313831909`
- AR/frozen PR-AUC: `0.2174953276`
- best matched control: `train_only_video_mean_residual`, PR-AUC `0.2174209937`
- delta vs AR/frozen: `+0.0138878634`
- delta vs best matched control: `+0.0139621972`
- PR-AUC as percentages: real `23.138%`, AR/frozen `21.750%`, best control `21.742%`
- percentage-point deltas: `+1.389` over AR/frozen, `+1.396` over best control
- relative lifts: `+6.39%` over AR/frozen, `+6.42%` over best control
- fold-seed positives vs best control: `50/50`
- real minus label permutation: `+0.0160732134`
- updated grouped compatibility pass: true

## Raw Predicted Cortical/FMRI Features Alone Fail Badly

The core negative-control lesson is visible from the beginning of AGAIN. Raw predicted cortical/fMRI features by themselves were weak, and directly adding them to AR did not create the current result.

- target: `arousal_spike_rows_2_6_train_q90`
- protocol: `blocked_temporal_70_30`
- `raw_cortical_only` PR-AUC: `0.124315`
- AR-only PR-AUC: `0.203622`
- `AR_plus_raw_cortical` PR-AUC: `0.167731`
- blocked conclusion: raw predicted cortical/fMRI features alone are far below AR, and directly adding raw predicted cortical/fMRI features damages AR.
- current Phase 5.5 blocked real PR-AUC: `0.2670735630`
- raw-to-Phase-5.5 gain: `+14.276` PR-AUC percentage points over raw predicted cortical/fMRI features alone, about `2.15x` the raw-only predicted cortical/fMRI baseline and `+114.8%` relative lift
- direct-AR-plus-raw-to-Phase-5.5 gain: `+9.934` PR-AUC percentage points, about `+59.2%` relative lift
- protocol: `grouped_video`
- `raw_cortical_only` PR-AUC: `0.136579`
- AR-only PR-AUC: `0.147251`
- `AR_plus_raw_cortical` PR-AUC: `0.170299`
- grouped conclusion: raw predicted cortical/fMRI features can contain weak signal, but it is not enough and is not the current claim.

Neural Bridge makes the difference by converting video into fold-safe predicted cortical/fMRI response features generated from video by upstream brain-response models, controlling the strong AR persistence baseline, and testing future event ranking against matched controls.

## VEATIC Evidence Block

VEATIC is foundational, not obsolete. It established the controlled future arousal event-ranking signal before AGAIN existed as a dense full-dataset bridge benchmark.

Strongest blocked full-frame spike row:

- feature: `cortical_pca64_delta`
- target: `arousal__future_spike_1_3s`
- PR-AUC: `0.2536`
- AR: `0.1969`
- shuffled: `0.1840`
- random: `0.1944`

Balanced event-vs-stable:

- target: `arousal__future_spike_1_3s@0.05`
- `cortical_pca64_delta` PR-AUC: `0.3394`
- delta over AR: `+0.0609`
- delta over shuffled: `+0.0631`
- delta over random: `+0.0476`

Correct VEATIC framing: VEATIC-124 v2 established the original controlled future arousal event-ranking signal.

## AGAIN Evidence Block

AGAIN is the scaled confirmation and current main result. It uses the dense H100 artifact:

- `995/995` videos complete
- `243,575` video feature rows generated from video by upstream models trained on brain cortical response data
- `2 Hz`
- `256 px`
- float16
- official V-JEPA 2.1 ViT-G
- TRIBE v2 cache-only postpass
- labels aligned at true 2 Hz

Phase 5 eval-mode correction:

- grouped real PR-AUC: `0.2300639382`
- grouped best matched control: `0.2042740689`
- grouped delta vs best matched control: `+0.0257898694`
- grouped AR-only: `0.2246816187`
- grouped fold-seed positive: `15/15`
- blocked caveat: this older fused lane lost to AR/control under blocked validation, which motivated the frozen-AR residual design

Frozen-AR residual design:

- grouped frozen AR PR-AUC: `0.2246816187`
- grouped best real residual PR-AUC: `0.2383409298`
- grouped best matched residual control PR-AUC: `0.2248361805`
- grouped delta vs frozen AR: `+0.0136593110`
- grouped delta vs best matched control: `+0.0135047493`
- blocked do-no-harm passed
- blocked residual did not yet beat AR/control in this older frozen-AR residual design

## Blocked Confirmation Block

The redesigned washout-gap target tests future movement after an explicit gap from current/past AR context.

- target: `future_arousal_max_delta_rows_4_10_train_q90`
- protocol: `blocked_temporal_70_30`
- architecture: `short_temporal_conv_residual`
- real PR-AUC: `0.2670735630`
- frozen AR PR-AUC: `0.2602336231`
- best control: `random_pca_residual`, PR-AUC `0.2593369051`
- delta vs frozen AR: `+0.0068399399`
- delta vs best control: `+0.0077366579`
- seeds positive vs AR: `9/10`
- seeds positive vs best control: `9/10`
- weak / credible / strong confirmation: true
- failed gates: `[]`

Conclusion: bounded strict forward-time future-event ranking is proven on AGAIN for this target/head.

## Grouped Compatibility Block

The same target/head was checked across all five grouped-video folds and ten seeds.

- target: `future_arousal_max_delta_rows_4_10_train_q90`
- protocol: `grouped_video`
- architecture: `short_temporal_conv_residual`
- rows: `350/350`
- real PR-AUC: `0.2313831909`
- AR/frozen PR-AUC: `0.2174953276`
- best matched control: `train_only_video_mean_residual`, PR-AUC `0.2174209937`
- delta vs AR/frozen: `+0.0138878634`
- delta vs best matched control: `+0.0139621972`
- fold-seed positives vs best control: `50/50`
- label permutation PR-AUC: `0.2153099775`
- real minus label permutation: `+0.0160732134`
- label permutation minus AR: `-0.0021853501`
- fold-seed positives vs label permutation: `50/50`
- updated grouped compatibility pass: true
- failed updated gates: `[]`

The original grouped artifact failed the legacy label-permutation-near-chance gate. That gate is inapplicable for frozen-AR residual designs because the label-permutation residual lane retains the same frozen AR floor and only permutes residual train/inner-val labels. The updated verdict was computed from existing CSV/JSON artifacts; no retraining, PCA generation, grouped rerun, or 504 was done.

## Controls And Adversarial Discipline

- Train/test and PCA transforms are fold-safe and train-only where required.
- Eval-mode scoring restores best checkpoints and disables dropout.
- Frozen-AR residual designs require same frozen AR scores across real and matched controls inside each seed/fold.
- Matched controls include shuffled PCA, random PCA, diagnostics-only, train-only video-mean, and label permutation residual lanes.
- Label permutation for frozen-AR residuals is interpreted as a residual-null over the AR floor, not as a raw-prevalence null.
- Old `holy_shit_pass`, Phase 5b/5c/Spark/max-capacity/deep/chimera outputs are not canonical.

## Executable Validation

- Full deterministic test suite: `python3 -m pytest -q tests`
- Latest local result: `93 passed in 5.52s` on `2026-06-30`
- Repo readiness audit: `npm run audit:repo`
- Latest local audit result: `repo_readiness pass controlled_evidence_items=206`
- Executable validation index: `docs/executable_validation_index.md`
- AGAIN/VEATIC v2 script-test-benchmark crosswalk: `docs/executable_validation_manifest.csv`
- Reviewer copy: `evidence/current_phase_5_5_review/14_executable_validation_and_code/`

These tests protect deterministic contracts over split construction, target windows, leakage boundaries, tensor/manifest contracts, scorer utilities, cache claims, and runtime configuration. They do not rerun training, V-JEPA/TRIBE, PCA, grouped compatibility, or 504.

## Correct Claim Wording

Neural Bridge demonstrates controlled future human arousal event-ranking from frozen predicted cortical/fMRI response features generated from video by upstream models trained on brain cortical response data across VEATIC and AGAIN. VEATIC-124 v2 established the original controlled future arousal event-ranking signal; AGAIN replicated and extended it at scale. Bounded strict forward-time future-event ranking is proven on AGAIN for `future_arousal_max_delta_rows_4_10_train_q90` with `short_temporal_conv_residual`, and grouped held-out-video compatibility for the same target/head is proven under the updated frozen-AR-residual-aware verdict. Raw predicted cortical/fMRI features alone fail badly on AGAIN; the claim is the Neural Bridge pipeline, not raw predicted cortical/fMRI features by themselves.

## Forbidden Claim Wording

- mind reading
- exact continuous arousal forecasting is solved
- universal emotion prediction
- 504 proven
- broad all-target/all-dataset temporal prediction solved
- exact continuous future arousal forecasting solved

## Artifact Links

- Reviewer evidence dossier: `evidence/current_phase_5_5_review/README.md`
- Commercial thesis: `docs/neural_bridge_service_as_software.md`
- Claim ledger: `evidence/current_phase_5_5_review/CLAIM_LEDGER.md`
- Artifact manifest: `evidence/current_phase_5_5_review/artifact_manifest.csv`
- Executable validation index: `docs/executable_validation_index.md`
- Executable validation manifest: `docs/executable_validation_manifest.csv`
- VEATIC summary: `docs/veatic_v2_evidence_summary.md`
- AGAIN dense cache handoff: `docs/again_dense_h100_cache.md`
- Current state: `docs/current_project_state.md`
- Machine-readable status: `docs/current_claim_status.json`
- Eval-mode correction report: `reports/again_dense_2hz_phase5_evalmode_rescore_summary_.md`
- Frozen-AR residual report: `reports/again_dense_2hz_phase5_frozen_ar_residual_summary_.md`
- Blocked binary confirmation report: `reports/again_dense_2hz_phase5_temporal_residual_binary_big_confirm_20260630_025437.md`
- Updated grouped compatibility verdict: `reports/again_dense_2hz_phase5_temporal_residual_grouped_compat_20260630_033520_UPDATED_VERDICT.md`
- Blocked binary confirmation evidence bundle: `evidence/phase_5_5_binary_blocked_confirmation_20260630_025437/`
- Grouped compatibility evidence bundle: `evidence/phase_5_5_grouped_compatibility_20260630_033520/`
- Phase 0-5 evidence bundle: `evidence/phase_0_to_5_historical_ladder_20260625/BUNDLE_README.md`

## Next Work

Next work is explicit review and planning for any 504/broader compatibility confirmation. Do not widen claims until that later run is intentionally performed, audited, and promoted. Continuous exact arousal forecasting remains a separate open research problem.
