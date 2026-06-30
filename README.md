# Neural Bridge

Neural Bridge demonstrates controlled future human arousal event-ranking from frozen video-derived predictions of brain cortical response structure across VEATIC and AGAIN.

## Best Results First

AGAIN blocked temporal binary confirmation: `future_arousal_max_delta_rows_4_10_train_q90` with `short_temporal_conv_residual` reached real PR-AUC `0.2670735630` vs frozen AR `0.2602336231` and best control `0.2593369051`, with deltas `+0.0068399399` vs AR and `+0.0077366579` vs best control. Seeds were positive `9/10` vs AR and `9/10` vs best control; weak, credible, and strong confirmation gates all passed.

AGAIN grouped-video compatibility for the same target/head reached real PR-AUC `0.2313831909` vs AR/frozen `0.2174953276` and best matched control `0.2174209937`, with delta `+0.0139621972` vs best control and `50/50` fold-seed positives. The updated frozen-AR-residual-aware verdict passed with failed gates `[]`.

Raw cortical-derived features alone fail badly on AGAIN: blocked `raw_cortical_only` was `0.124315` vs AR-only `0.203622`, and direct `AR_plus_raw_cortical` was only `0.167731`. Neural Bridge is the difference.

In percentage terms, the current confirmed AGAIN blocked result is `26.707%` PR-AUC versus `26.023%` frozen AR and `25.934%` best control: `+0.684` PR-AUC percentage points over AR and `+0.774` points over the best control. Compared with the early blocked raw-cortical-only result (`12.432%` PR-AUC), Phase 5.5 is `+14.276` PR-AUC points higher, about `2.15x` the raw-cortical-only score and `+114.8%` relative lift. Compared with direct `AR_plus_raw_cortical` (`16.773%`), Phase 5.5 is `+9.934` points higher and about `+59.2%` relative lift. Without the Neural Bridge pipeline, the raw cortical-derived signal is not commercially useful; with the bridge, it becomes a controlled future-event ranking system.

## Neuro-Response Core

The important thing is not "video analytics." Neural Bridge uses frozen video-side representations trained to predict brain cortical responses, then tests whether those neuro-response-derived features rank future human arousal events beyond AR, static video identity, shuffled/random controls, and label permutation.

That is the commercial and scientific wedge: the system is not merely recognizing objects, actions, or captions. It is using a learned proxy for human cortical response structure to produce pre-release response intelligence.

## Datasets: What Generalized

VEATIC-124 v2 is the film/TV/emotion side of the evidence ladder. Public VEATIC documentation describes 124 video clips from Hollywood movies, documentaries, home videos, and reality TV shows with continuous valence and arousal ratings for each frame via real-time annotation. In this repo, VEATIC established controlled future arousal event/spike ranking on edited, affective real-world video content. Sources: [VEATIC project page](https://veatic.github.io/), [WACV 2024 paper](https://openaccess.thecvf.com/content/WACV2024/html/Ren_VEATIC_Video-Based_Emotion_and_Affect_Tracking_in_Context_Dataset_WACV_2024_paper.html).

AGAIN is the gaming/interactive-media side of the evidence ladder. Public AGAIN documentation describes 1,116 raw in-game videos and gameplay logs from 124 participants playing 9 games across 3 genres, with a cleaned/preprocessed 995-video version. The AGAIN paper describes over 37 hours of annotated video/game logs and first-person continuous arousal annotation. In this repo, AGAIN scaled the result and confirmed bounded strict forward-time future-event ranking on the 995-video dense cache. Sources: [AGAIN project page](https://again.institutedigitalgames.com/), [AGAIN arXiv paper](https://arxiv.org/abs/2104.02643).

That means the evidence is not a single narrow demo. It spans film, TV/reality, documentary, home-video affect content, and interactive gameplay arousal. Put plainly: the Neural Bridge effect appears across edited emotion video and gaming video, with VEATIC as the foundational affect-video result and AGAIN as the scaled gaming confirmation.

## Metrics In Plain English

- `PR-AUC` means area under the precision-recall curve. For rare future arousal events, it measures whether true future response moments are ranked near the top.
- `26.707% PR-AUC` is the same number as `0.2670735630` written as a percentage.
- `+0.684 percentage points` means the direct PR-AUC difference between `26.707%` and `26.023%`.
- `+2.63% relative lift` means the `+0.684` point gain divided by the frozen AR baseline.
- `AR` or `frozen AR` is the strong autoregressive baseline from recent/past arousal.
- `blocked_temporal_70_30` tests forward-time prediction; `grouped_video` tests held-out-video compatibility.
- `shuffled`, `random`, `label permutation`, and `train-only video mean` controls test whether the apparent signal is fake, leaked, static, or selection noise.

## Commercial Thesis

Neural Bridge is Service as Software for video response intelligence. It automates the first-pass expert service of pre-release response evaluation: scoring videos, ranking future response moments, comparing variants, and producing response diagnostics before audience data exists.

The business outcome is not "upload video, get chart." It is: upload video, receive the kind of response intelligence report a specialist team would produce, but faster, cheaper, and at far greater scale. Neural Bridge converts pre-release video response evaluation from a slow human service into scalable software.

Commercial interpretation: [docs/neural_bridge_service_as_software.md](docs/neural_bridge_service_as_software.md)

## Current State

Canonical claim: Neural Bridge demonstrates controlled future human arousal event-ranking from frozen video-derived predictions of brain cortical response structure across VEATIC and AGAIN.

VEATIC-124 v2 established the first controlled future arousal spike/event-ranking result. AGAIN replicated and extended it at scale using 995 videos, 2 Hz dense V-JEPA 2.1 / TRIBE v2 features, frozen-AR residuals, a redesigned washout-gap future arousal event target, blocked temporal confirmation, and grouped-video compatibility.

The frozen video-side features are learned predictions of brain cortical response structure. They are not generic video embeddings, and they are not direct neural recordings from the benchmark viewer rows. The point is the neuro-response bridge: video is transformed into a cortical-response-informed representation, then tested for future human arousal event ranking.

Raw cortical-derived features alone fail badly on AGAIN. On the original Phase 3 spike target `arousal_spike_rows_2_6_train_q90`, `raw_cortical_only` scored blocked PR-AUC `0.124315` versus AR-only `0.203622`; grouped PR-AUC was `0.136579` versus AR-only `0.147251`. Adding raw cortical directly to AR damaged the blocked AR path (`AR_plus_raw_cortical` `0.167731` vs AR-only `0.203622`). The current result is not raw cortical alone; Neural Bridge is the fold-safe PCA, frozen-AR floor, redesigned washout-gap target, and short temporal/event-context residual stack that turns weak raw cortical-derived signals into controlled future event-ranking.

Bounded strict forward-time future-event ranking is now proven on AGAIN for `future_arousal_max_delta_rows_4_10_train_q90` with `short_temporal_conv_residual`. Grouped held-out-video compatibility for the same target/head is also proven under the updated frozen-AR-residual-aware verdict.

Continuous exact arousal forecasting remains open. Broad all-target/all-dataset temporal prediction remains open. No 504 run has been promoted.

## Canonical Evidence

Primary narrative: [docs/neural_bridge_phase5_5_evidence_ladder.md](docs/neural_bridge_phase5_5_evidence_ladder.md)

Commercial thesis: [docs/neural_bridge_service_as_software.md](docs/neural_bridge_service_as_software.md)

Reviewer evidence dossier: [evidence/current_phase_5_5_review/README.md](evidence/current_phase_5_5_review/README.md)

Claim-to-artifact ledger: [evidence/current_phase_5_5_review/CLAIM_LEDGER.md](evidence/current_phase_5_5_review/CLAIM_LEDGER.md)

Machine-readable status: [docs/current_claim_status.json](docs/current_claim_status.json)

Executable validation index: [docs/executable_validation_index.md](docs/executable_validation_index.md)

Canonical review: [docs/reviews/neural_bridge_phase5_adversarial_review_20260625.html](docs/reviews/neural_bridge_phase5_adversarial_review_20260625.html)

Canonical deterministic AGAIN rescore: [reports/again_dense_2hz_phase5_evalmode_rescore_summary_.md](reports/again_dense_2hz_phase5_evalmode_rescore_summary_.md)

AGAIN frozen-AR residual design: [reports/again_dense_2hz_phase5_frozen_ar_residual_summary_.md](reports/again_dense_2hz_phase5_frozen_ar_residual_summary_.md)

AGAIN blocked binary confirmation: [reports/again_dense_2hz_phase5_temporal_residual_binary_big_confirm_20260630_025437.md](reports/again_dense_2hz_phase5_temporal_residual_binary_big_confirm_20260630_025437.md)

AGAIN updated grouped compatibility verdict: [reports/again_dense_2hz_phase5_temporal_residual_grouped_compat_20260630_033520_UPDATED_VERDICT.md](reports/again_dense_2hz_phase5_temporal_residual_grouped_compat_20260630_033520_UPDATED_VERDICT.md)

## Detailed Evidence Numbers

AGAIN blocked temporal binary confirmation:

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

AGAIN grouped compatibility updated verdict:

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
- verdict update only: no retraining, no PCA generation, no grouped rerun, no 504

Raw cortical alone versus Neural Bridge:

- original Phase 3 target: `arousal_spike_rows_2_6_train_q90`
- blocked `raw_cortical_only` PR-AUC: `0.124315`
- blocked AR-only PR-AUC: `0.203622`
- blocked `AR_plus_raw_cortical` PR-AUC: `0.167731`
- grouped `raw_cortical_only` PR-AUC: `0.136579`
- grouped AR-only PR-AUC: `0.147251`
- grouped `AR_plus_raw_cortical` PR-AUC: `0.170299`
- conclusion: raw cortical-derived features alone are weak; the Neural Bridge pipeline is what makes the cortical-response-derived representation useful for controlled future event-ranking.

AGAIN dense substrate:

- `995/995` videos complete
- `243,575` brain-cortical-response-derived video feature rows
- `2 Hz`, `256 px`, float16
- official V-JEPA 2.1 ViT-G
- TRIBE v2 cache-only postpass
- frozen video-derived neuro-response features trained from brain cortical responses
- labels aligned at true 2 Hz

AGAIN Phase 5 eval-mode correction:

- grouped real PR-AUC: `0.2300639382`
- grouped best matched control PR-AUC: `0.2042740689`
- grouped delta vs best matched control: `+0.0257898694`
- grouped AR-only PR-AUC: `0.2246816187`
- grouped fold-seed positive: `15/15`
- blocked caveat: the old fused lane lost to AR/control under blocked validation, motivating the frozen-AR residual design

AGAIN frozen-AR residual design:

- grouped frozen AR PR-AUC: `0.2246816187`
- grouped best real residual PR-AUC: `0.2383409298`
- grouped best matched residual control PR-AUC: `0.2248361805`
- grouped delta vs frozen AR: `+0.0136593110`
- grouped delta vs best matched control: `+0.0135047493`
- blocked do-no-harm passed; this older residual design did not yet beat AR/control under blocked temporal validation

VEATIC-124 v2 foundational evidence:

- strongest blocked full-frame spike row: `cortical_pca64_delta` / `arousal__future_spike_1_3s`
- PR-AUC: `0.2536`
- AR: `0.1969`
- shuffled: `0.1840`
- random: `0.1944`
- balanced event-vs-stable target `arousal__future_spike_1_3s@0.05`: `cortical_pca64_delta` PR-AUC `0.3394`
- balanced deltas: `+0.0609` over AR, `+0.0631` over shuffled, `+0.0476` over random

## Claim Boundaries

Do not claim mind reading, continuous exact arousal forecasting is solved, universal emotion prediction, or 504-proven generalization.

Correct wording: bounded strict forward-time future-event ranking is proven on AGAIN for the redesigned washout-gap target/head; broader continuous forecasting and all-target/all-dataset temporal generalization remain open.

## Executable Validation

The current deterministic validation command is:

```bash
python3 -m pytest -q tests
```

Latest local result on `2026-06-30`: `93 passed in 5.52s`.

`npm test` now runs that full deterministic suite. The executable crosswalk for relevant AGAIN and VEATIC v2 scripts, tests, benchmark artifacts, and runtime-only tools is tracked in [docs/executable_validation_manifest.csv](docs/executable_validation_manifest.csv) and mirrored in [evidence/current_phase_5_5_review/14_executable_validation_and_code/](evidence/current_phase_5_5_review/14_executable_validation_and_code/). Runtime probes are not claim-bearing benchmark evidence.

## Canonical Artifacts

- Dense root: `.cache/h100_drive_downloads/again_tribe_v2_postpass_float16_256_2hz/`
- Phase 4 root: `$NEURAL_BRIDGE_EXTERNAL_ROOT/outputs/again_dense_2hz_phase4_pca_bridge_20260625_full/`
- Evidence bundle: `evidence/phase_0_to_5_historical_ladder_20260625/`
- Frozen-AR residual output root: `outputs/again_dense_2hz_phase5_frozen_ar_residual_/`
- Frozen-AR residual evidence snapshot: `evidence/phase_5_1_frozen_ar_residual/`
- Current reviewer dossier: `evidence/current_phase_5_5_review/`
- Blocked binary confirmation evidence snapshot: `evidence/phase_5_5_binary_blocked_confirmation_20260630_025437/`
- Grouped compatibility evidence snapshot: `evidence/phase_5_5_grouped_compatibility_20260630_033520/`

Current confirmed AGAIN lane: `future_arousal_max_delta_rows_4_10_train_q90` with `short_temporal_conv_residual` over fold-safe `temporal_mean_2s_then_pca256` and matched frozen AR.

Historical Phase 5 fused-head lane: `arousal_spike_rows_2_6_train_q90` with `gated_ar_pca_mlp` / `regression_plus_binary` / `temporal_mean_2s_then_pca256` / AR + temporal diagnostics.

## Next Work

Next work is explicit review and planning for any intentionally approved 504/broader compatibility confirmation. Keep claims bounded until such a run is performed, audited, and promoted. Continuous exact arousal forecasting and broad all-target/all-dataset temporal prediction remain open research problems, not current claims.
