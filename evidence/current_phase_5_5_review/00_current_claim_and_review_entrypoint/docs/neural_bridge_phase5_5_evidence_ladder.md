# Neural Bridge Phase 5.5 Evidence Ladder

Neural Bridge now has a paired cross-dataset evidence ladder: VEATIC-124 v2 established the original controlled future arousal spike/event-ranking signal, and AGAIN replicated, scaled, validated, and strengthened it with dense V-JEPA 2.1 / TRIBE v2 features, frozen-AR residuals, a redesigned washout-gap future arousal event target, blocked temporal confirmation, and grouped-video compatibility.

The neuro-response mechanism is central. The frozen video-side features used in these benchmarks are predicted cortical/fMRI response features generated from video by upstream models trained on brain cortical response data, not generic video embeddings and not direct brain recordings from the benchmark viewer rows.

Raw predicted cortical/fMRI features alone fail badly on AGAIN. On the original Phase 3 spike target `arousal_spike_rows_2_6_train_q90`, `raw_cortical_only` scored blocked PR-AUC `0.124315` versus AR-only `0.203622`; direct `AR_plus_raw_cortical` was `0.167731`, below AR. Grouped `raw_cortical_only` was `0.136579` versus AR-only `0.147251`. Neural Bridge is the difference: fold-safe compression, frozen AR anchoring, a washout-gap future event target, and temporal/event-context residual learning turn weak raw predicted cortical/fMRI features into controlled future event-ranking.

Beating AR is the central benchmark difficulty. AR/frozen AR is a strong recent-arousal persistence baseline, and many earlier lanes failed because they could not beat it under blocked temporal validation. The Phase 5.5 blocked confirmation beats the matched seed-specific frozen AR floor by `+0.0068399399` PR-AUC (`+2.63%` relative lift) and the best matched control by `+0.0077366579` PR-AUC (`+2.98%` relative lift), with `9/10` positive seeds vs both. The grouped compatibility run beats the matched fold/seed-specific AR/frozen floor by `+0.0138878634` PR-AUC (`+6.39%` relative lift) and the best matched control by `+0.0139621972` PR-AUC (`+6.42%` relative lift), with `50/50` fold-seed positives vs the best matched control.


## What Is Now Proven

- Controlled future human arousal event-ranking from frozen predicted cortical/fMRI response features generated from video by upstream models trained on brain cortical response data across VEATIC and AGAIN.
- VEATIC-124 v2 established the first controlled future arousal spike/event-ranking result.
- AGAIN extended the result at scale over 995 videos and true 2 Hz labels.
- Bounded strict forward-time future-event ranking is proven on AGAIN for `future_arousal_max_delta_rows_4_10_train_q90` with `short_temporal_conv_residual`.
- Grouped held-out-video compatibility for the same AGAIN target/head is proven under the updated frozen-AR-residual-aware verdict.

## What Remains Open

- Continuous exact arousal forecasting remains open.
- Broad all-target/all-dataset temporal prediction remains open.
- No 504 run has been performed or promoted.
- Grouped compatibility is not itself a 504 result.
- The updated grouped verdict is a no-training verdict update from existing artifacts, not a rerun.

## Best AGAIN Results

The strongest current evidence is the AGAIN redesigned washout-gap binary result, confirmed under blocked temporal validation and checked for grouped-video compatibility.

Blocked binary confirmation:

- target: `future_arousal_max_delta_rows_4_10_train_q90`
- protocol: `blocked_temporal_70_30`
- architecture: `short_temporal_conv_residual`
- real PR-AUC: `0.2670735630`
- matched seed-specific frozen AR PR-AUC: `0.2602336231`
- best matched control: `random_pca_residual`, PR-AUC `0.2593369051`
- delta vs frozen AR: `+0.0068399399`
- delta vs best matched control: `+0.0077366579`
- seeds positive vs frozen AR: `9/10`
- seeds positive vs best matched control: `9/10`
- weak / credible / strong confirmation: true
- failed gates: `[]`

Grouped compatibility:

- target: `future_arousal_max_delta_rows_4_10_train_q90`
- protocol: `grouped_video`
- architecture: `short_temporal_conv_residual`
- rows: `350/350`
- real PR-AUC: `0.2313831909`
- matched fold/seed-specific AR/frozen PR-AUC: `0.2174953276`
- best matched control: `train_only_video_mean_residual`, PR-AUC `0.2174209937`
- delta vs AR/frozen: `+0.0138878634`
- delta vs best matched control: `+0.0139621972`
- fold-seed positives vs best matched control: `50/50`
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
- protocol: `grouped_video`
- `raw_cortical_only` PR-AUC: `0.136579`
- AR-only PR-AUC: `0.147251`
- `AR_plus_raw_cortical` PR-AUC: `0.170299`
- grouped conclusion: raw predicted cortical/fMRI features can contain weak signal, but it is not enough and is not the current claim.

Neural Bridge makes the difference by converting predicted cortical/fMRI response representations generated from video by upstream models trained on brain cortical response data into fold-safe bridge features, controlling the strong AR persistence baseline, and testing future event ranking against matched controls.

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
- matched seed-specific frozen AR PR-AUC: `0.2602336231`
- best matched control: `random_pca_residual`, PR-AUC `0.2593369051`
- delta vs frozen AR: `+0.0068399399`
- delta vs best matched control: `+0.0077366579`
- seeds positive vs frozen AR: `9/10`
- seeds positive vs best matched control: `9/10`
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
- matched fold/seed-specific AR/frozen PR-AUC: `0.2174953276`
- best matched control: `train_only_video_mean_residual`, PR-AUC `0.2174209937`
- delta vs AR/frozen: `+0.0138878634`
- delta vs best matched control: `+0.0139621972`
- fold-seed positives vs best matched control: `50/50`
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
