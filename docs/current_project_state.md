# Current Project State

Last updated: 2026-06-30

## Current Claim

Canonical claim: Neural Bridge demonstrates controlled future human arousal event-ranking from frozen video-derived predictions generated from brain cortical response data across VEATIC and AGAIN.

VEATIC-124 v2 established the original controlled future arousal spike/event-ranking result. AGAIN replicated, scaled, validated, and strengthened it using 995 videos, 2 Hz dense V-JEPA 2.1 / TRIBE v2 features, frozen-AR residuals, a redesigned washout-gap future arousal event target, blocked temporal confirmation, and grouped-video compatibility.

The neuro-response features are frozen video-derived predictions generated from brain cortical response data. The claim is that these neuro-response-derived video features improve controlled future event ranking; they are not generic video embeddings, direct viewer neural measurements, or evidence of solved continuous forecasting.

Raw cortical-derived features alone fail badly on AGAIN. On the original Phase 3 target `arousal_spike_rows_2_6_train_q90`, blocked `raw_cortical_only` PR-AUC was `0.124315` versus AR-only `0.203622`, and direct `AR_plus_raw_cortical` was only `0.167731`. Grouped `raw_cortical_only` was `0.136579` versus AR-only `0.147251`. The current success comes from Neural Bridge, not raw cortical features by themselves.

Bounded strict forward-time future-event ranking is proven on AGAIN for `future_arousal_max_delta_rows_4_10_train_q90` with `short_temporal_conv_residual`. Grouped held-out-video compatibility for the same target/head is proven under the updated frozen-AR-residual-aware verdict.

Continuous exact arousal forecasting remains open. Broad all-target/all-dataset temporal prediction remains open. No 504 run has been promoted.

## Do Not Claim

- mind reading
- exact continuous future arousal forecasting is solved
- universal emotion prediction
- broad all-target/all-dataset temporal prediction is solved
- 504 has been run or promoted
- treat Phase 5b/5c/Spark/max-capacity/deep/chimera outputs as canonical
- use `holy_shit_pass` as a valid gate

## Evidence Summary

- VEATIC is foundational, not obsolete: it established the original controlled future arousal event/spike-ranking signal.
- AGAIN is the scaled confirmation and current main result: it replicated and extended the signal with dense V-JEPA 2.1 / TRIBE v2 features over 995 videos.
- VEATIC is the edited affect-video side: 124 Hollywood movie, documentary, reality-TV, and home-video clips with continuous valence/arousal annotation.
- AGAIN is the gaming/interactive-media side: 1,116 raw / 995 cleaned gameplay videos from 124 participants playing 9 games across 3 genres, with more than 37 hours of annotated video/logs.
- The original AGAIN fused head passed grouped eval-mode controls but failed blocked AR/control checks; that failure motivated the frozen-AR residual design.
- Frozen-AR residual design strengthened grouped evidence and reduced blocked harm by making AR the baseline floor.
- The redesigned washout-gap binary target plus short temporal conv residual passed a matched 10-seed blocked temporal confirmation.
- The same target/head passed grouped-video compatibility under the updated frozen-AR-residual-aware label permutation verdict.
- Continuous arousal movement diagnostics remain mixed/open and should not be promoted as solved exact forecasting.
- Raw cortical-derived features alone are a negative-control lesson: they are weak under blocked validation and can damage AR if bolted on directly.
- The neuro-response angle is central: Neural Bridge turns video into a learned proxy for brain cortical response data, then asks whether that proxy improves future human arousal event ranking under controls.

## Commercial Interpretation

Neural Bridge is Service as Software for video response intelligence. It converts pre-release video response evaluation from a slow human service into scalable software: submit video or variants, extract bridge features, rank future response-event moments, flag weak segments, compare cuts, and produce a response-readiness report before audience data exists.

The business model is automated expert analysis, not generic SaaS. The customer wants the service outcome a specialist team would normally deliver: evaluate this ad, diagnose this trailer, compare these cuts, find likely response moments, and decide what to test or ship. See `docs/neural_bridge_service_as_software.md`.

## Canonical Numbers

AGAIN blocked temporal binary confirmation:

- target: `future_arousal_max_delta_rows_4_10_train_q90`
- protocol: `blocked_temporal_70_30`
- architecture: `short_temporal_conv_residual`
- real PR-AUC: `0.2670735630`
- frozen AR PR-AUC: `0.2602336231`
- best control: `random_pca_residual`, PR-AUC `0.2593369051`
- delta vs frozen AR: `+0.0068399399`
- delta vs best control: `+0.0077366579`
- as percentages: real `26.707%`, frozen AR `26.023%`, best control `25.934%`
- percentage-point deltas: `+0.684` over frozen AR, `+0.774` over best control
- relative lifts: `+2.63%` over frozen AR, `+2.98%` over best control
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

AGAIN raw cortical alone:

- original Phase 3 target: `arousal_spike_rows_2_6_train_q90`
- blocked `raw_cortical_only` PR-AUC: `0.124315`
- blocked AR-only PR-AUC: `0.203622`
- blocked `AR_plus_raw_cortical` PR-AUC: `0.167731`
- current Phase 5.5 blocked real PR-AUC: `0.2670735630`
- raw-to-Phase-5.5 gain: `+14.276` PR-AUC percentage points, about `2.15x` raw cortical and `+114.8%` relative lift
- direct-AR-plus-raw-to-Phase-5.5 gain: `+9.934` PR-AUC percentage points, about `+59.2%` relative lift
- grouped `raw_cortical_only` PR-AUC: `0.136579`
- grouped AR-only PR-AUC: `0.147251`
- grouped `AR_plus_raw_cortical` PR-AUC: `0.170299`
- conclusion: raw cortical-derived features alone are not the result; Neural Bridge makes the brain-cortical-response-data-generated signal usable.

VEATIC-124 v2:

- strongest blocked full-frame spike row: `cortical_pca64_delta` / `arousal__future_spike_1_3s`
- PR-AUC: `0.2536`
- AR: `0.1969`
- shuffled: `0.1840`
- random: `0.1944`
- balanced event-vs-stable target `arousal__future_spike_1_3s@0.05`: `cortical_pca64_delta` PR-AUC `0.3394`
- balanced deltas: `+0.0609` over AR, `+0.0631` over shuffled, `+0.0476` over random

AGAIN dense cache:

- `995/995` videos complete
- `243,575` brain-cortical-response-data-generated video feature rows
- frozen video-derived neuro-response features generated from brain cortical response data
- `2 Hz`, `256 px`, float16
- official V-JEPA 2.1 ViT-G
- TRIBE v2 cache-only postpass
- labels aligned at true 2 Hz

AGAIN Phase 5 eval-mode correction:

- grouped real PR-AUC: `0.2300639382`
- grouped best matched control: `0.2042740689`
- grouped delta vs best matched control: `+0.0257898694`
- grouped AR-only: `0.2246816187`
- grouped fold-seed positive: `15/15`
- blocked caveat: old fused lane lost to AR/control under blocked validation

AGAIN frozen-AR residual design:

- grouped frozen AR PR-AUC: `0.2246816187`
- grouped best real residual PR-AUC: `0.2383409298`
- grouped best matched residual control PR-AUC: `0.2248361805`
- grouped delta vs frozen AR: `+0.0136593110`
- grouped delta vs best matched control: `+0.0135047493`
- blocked do-no-harm passed, but the older frozen-AR residual design did not yet beat AR/control under blocked temporal validation

## Canonical Artifacts

- Phase 5.5 evidence ladder: `docs/neural_bridge_phase5_5_evidence_ladder.md`
- Service-as-Software commercial thesis: `docs/neural_bridge_service_as_software.md`
- Machine-readable status: `docs/current_claim_status.json`
- Executable validation index: `docs/executable_validation_index.md`
- Executable validation manifest: `docs/executable_validation_manifest.csv`, `docs/executable_validation_manifest.json`
- Latest deterministic test-suite result: `docs/test_suite_result_20260630.json`
- Reviewer evidence dossier: `evidence/current_phase_5_5_review/README.md`
- Claim ledger and artifact manifest: `evidence/current_phase_5_5_review/CLAIM_LEDGER.md`, `evidence/current_phase_5_5_review/artifact_manifest.csv`
- VEATIC summary: `docs/veatic_v2_evidence_summary.md`
- AGAIN dense cache handoff: `docs/again_dense_h100_cache.md`
- Canonical adversarial review: `docs/reviews/neural_bridge_phase5_adversarial_review_20260625.html`
- Eval-mode correction report: `reports/again_dense_2hz_phase5_evalmode_rescore_summary_.md`
- Frozen-AR residual report: `reports/again_dense_2hz_phase5_frozen_ar_residual_summary_.md`
- Blocked binary confirmation report: `reports/again_dense_2hz_phase5_temporal_residual_binary_big_confirm_20260630_025437.md`
- Updated grouped compatibility verdict: `reports/again_dense_2hz_phase5_temporal_residual_grouped_compat_20260630_033520_UPDATED_VERDICT.md`
- Dense root: `.cache/h100_drive_downloads/again_tribe_v2_postpass_float16_256_2hz/`
- Phase 4 root: `$NEURAL_BRIDGE_EXTERNAL_ROOT/outputs/again_dense_2hz_phase4_pca_bridge_20260625_full/`

Do not touch dense cache files, Phase 4 outputs, original Phase 5 output roots, or evidence bundle contents unless explicitly asked. Heavy output roots remain ignored.

## Executable Validation

- Full deterministic suite: `python3 -m pytest -q tests`
- Latest local result: `93 passed in 5.52s` on `2026-06-30`
- `npm test` runs the full suite.
- Executable crosswalk: `evidence/current_phase_5_5_review/14_executable_validation_and_code/executable_validation_manifest.csv`
- Repo audit: `npm run audit:repo`

## Next Work

Next work is review and planning for any explicit 504/broader compatibility confirmation. Keep claims bounded until such a run is intentionally performed and promoted. Continuous exact arousal forecasting and broad universal temporal prediction remain research problems, not current claims.
