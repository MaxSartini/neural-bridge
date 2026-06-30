# Claim Ledger

Each current-facing claim below is paired with the numbers and evidence files that support it.

## C1: canonical_current_claim

Neural Bridge demonstrates controlled future human arousal event-ranking from frozen video-derived bridge representations derived from brain cortical responses across VEATIC and AGAIN.

Key numbers: VEATIC blocked PR-AUC 0.2536 vs AR 0.1969; AGAIN blocked real PR-AUC 0.2670735630 vs frozen AR 0.2602336231; AGAIN grouped real PR-AUC 0.2313831909 vs AR 0.2174953276; raw cortical alone blocked PR-AUC 0.124315 vs AR-only 0.203622.

Primary evidence: `README.md; docs/neural_bridge_phase5_5_evidence_ladder.md; reports/again_dense_2hz_phase5_temporal_residual_binary_big_confirm_20260630_025437.md; reports/again_dense_2hz_phase5_temporal_residual_grouped_compat_20260630_033520_UPDATED_VERDICT.md`

Caveats: Event ranking, not mind reading or exact continuous forecasting; no 504 run has been promoted.

## C2: foundational_proven

VEATIC-124 v2 established the original controlled future arousal spike/event-ranking signal.

Key numbers: cortical_pca64_delta / arousal__future_spike_1_3s PR-AUC 0.2536 vs AR 0.1969, shuffled 0.1840, random 0.1944; balanced event-vs-stable PR-AUC 0.3394 with +0.0609 over AR.

Primary evidence: `benchmarks/veatic/veatic_124_retest_event_spike_core_20260616.*; benchmarks/veatic/veatic_124_event_conditioned_retest_20260616.*; docs/veatic_v2_evidence_summary.md`

Caveats: Foundational controlled signal, not obsolete and not the only current evidence.

## C3: substrate_complete

AGAIN scaled the evidence ladder to 995 dense videos with 2 Hz V-JEPA 2.1 / TRIBE v2 features and aligned labels.

Key numbers: 995/995 videos; 243,575 cortical-response-derived bridge rows; 2 Hz; 256 px; float16; official V-JEPA 2.1 ViT-G; TRIBE v2 cache-only postpass.

Primary evidence: `docs/again_dense_h100_cache.md; evidence/phase_0_to_5_historical_ladder_20260625/reports/again_dense_h100_local_audit_20260625.md; labels and metadata summaries.`

Caveats: Dense cache and tensors remain outside the review dossier; metadata/manifests document them.

## C4: negative_control_context

Raw cortical-derived features alone fail badly on AGAIN and are not the Neural Bridge result.

Key numbers: arousal_spike_rows_2_6_train_q90; blocked raw_cortical_only 0.124315 vs AR-only 0.203622; blocked AR_plus_raw_cortical 0.167731; grouped raw_cortical_only 0.136579 vs AR-only 0.147251; grouped AR_plus_raw_cortical 0.170299.

Primary evidence: `evidence/current_phase_5_5_review/05_again_phase_3_raw_cortical/reports/again_dense_2hz_raw_cortical_vs_ar_20260625_094242.md; evidence/current_phase_5_5_review/05_again_phase_3_raw_cortical/outputs/summary_metrics.csv`

Caveats: This is the early dense raw-cortical diagnostic. It shows why the bridge stack is needed; it is not the promoted target/head.

## C5: historical_correction_result

AGAIN Phase 5 eval-mode correction showed grouped-video signal but exposed blocked AR/control caveats.

Key numbers: grouped real PR-AUC 0.2300639382; best matched control 0.2042740689; delta +0.0257898694; grouped AR-only 0.2246816187; 15/15 fold-seeds positive.

Primary evidence: `reports/again_dense_2hz_phase5_evalmode_rescore_summary_.md; evidence/phase_5_0_evalmode_primary_20260629_171825/`

Caveats: Old fused blocked lane lost to AR/control; this motivated frozen-AR residual work.

## C6: historical_design_result

Frozen-AR residual design strengthened grouped residual evidence while preventing the earlier severe blocked damage.

Key numbers: grouped frozen AR 0.2246816187; best real residual 0.2383409298; best matched residual control 0.2248361805; delta vs AR +0.0136593110; delta vs control +0.0135047493.

Primary evidence: `reports/again_dense_2hz_phase5_frozen_ar_residual_summary_.md; evidence/phase_5_1_frozen_ar_residual/`

Caveats: This older frozen-AR residual design passed blocked do-no-harm but did not yet beat AR/control under blocked validation.

## C7: proven_bounded_blocked_confirmation

Bounded strict forward-time future-event ranking is proven on AGAIN for the redesigned washout-gap binary target/head.

Key numbers: target future_arousal_max_delta_rows_4_10_train_q90; short_temporal_conv_residual; real PR-AUC 0.2670735630; frozen AR 0.2602336231; best control 0.2593369051; deltas +0.0068399399 and +0.0077366579; 9/10 positive seeds; weak/credible/strong true; failed gates [].

Primary evidence: `reports/again_dense_2hz_phase5_temporal_residual_binary_big_confirm_20260630_025437.md; evidence/phase_5_5_binary_blocked_confirmation_20260630_025437/`

Caveats: Blocked-only binary washout-gap event ranking. Continuous exact forecasting remains open.

## C8: proven_grouped_compatibility

The same AGAIN target/head is compatible with grouped held-out-video generalization under the updated frozen-AR residual-aware verdict.

Key numbers: 350/350 rows; real PR-AUC 0.2313831909; AR/frozen 0.2174953276; best control 0.2174209937; deltas +0.0138878634 and +0.0139621972; 50/50 positives vs best control; label permutation 0.2153099775; real-label +0.0160732134; label-AR -0.0021853501; 50/50 positives vs label; updated pass true; failed gates [].

Primary evidence: `reports/again_dense_2hz_phase5_temporal_residual_grouped_compat_20260630_033520_UPDATED_VERDICT.md; evidence/phase_5_5_grouped_compatibility_20260630_033520/`

Caveats: Verdict update used existing artifacts. It was not a rerun, not PCA generation, and not 504.

## C9: open_not_proven

Continuous exact arousal forecasting remains open.

Key numbers: continuous blocked residual diagnostic real Spearman 0.2484145880 vs frozen AR 0.2695371538; top 5pct lift delta -0.0029861629; continuous_residual_pass false.

Primary evidence: `reports/again_dense_2hz_phase5_continuous_residual_blocked_summary_20260630_000219.md; evidence/phase_5_2_continuous_residual_blocked_20260630_000219/`

Caveats: Failed diagnostic rejects this exact blocked monotonic continuous setup, not all future continuous work.
