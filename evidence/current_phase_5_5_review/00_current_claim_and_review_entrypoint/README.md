# Neural Bridge

Neural Bridge demonstrates controlled future human arousal event-ranking from frozen video-derived bridge/cortical representations across VEATIC and AGAIN.

## Current State

Canonical claim: Neural Bridge demonstrates controlled future human arousal event-ranking from frozen video-derived bridge/cortical representations across VEATIC and AGAIN.

VEATIC-124 v2 established the first controlled future arousal spike/event-ranking result. AGAIN replicated and extended it at scale using 995 videos, 2 Hz dense V-JEPA 2.1 / TRIBE v2 features, frozen-AR residuals, a redesigned washout-gap future arousal event target, blocked temporal confirmation, and grouped-video compatibility.

Bounded strict forward-time future-event ranking is now proven on AGAIN for `future_arousal_max_delta_rows_4_10_train_q90` with `short_temporal_conv_residual`. Grouped held-out-video compatibility for the same target/head is also proven under the repaired frozen-AR-residual-aware verdict.

Continuous exact arousal forecasting remains open. Broad all-target/all-dataset temporal prediction remains open. No 504 run has been promoted.

## Canonical Evidence

Primary narrative: [docs/neural_bridge_phase5_5_evidence_ladder.md](docs/neural_bridge_phase5_5_evidence_ladder.md)

Reviewer evidence dossier: [evidence/current_phase_5_5_review/README.md](evidence/current_phase_5_5_review/README.md)

Claim-to-artifact ledger: [evidence/current_phase_5_5_review/CLAIM_LEDGER.md](evidence/current_phase_5_5_review/CLAIM_LEDGER.md)

Machine-readable status: [docs/current_claim_status.json](docs/current_claim_status.json)

Canonical review: [docs/reviews/neural_bridge_phase5_adversarial_review_20260625.html](docs/reviews/neural_bridge_phase5_adversarial_review_20260625.html)

Canonical deterministic AGAIN rescore: [reports/again_dense_2hz_phase5_evalmode_rescore_summary_.md](reports/again_dense_2hz_phase5_evalmode_rescore_summary_.md)

AGAIN frozen-AR residual repair: [reports/again_dense_2hz_phase5_frozen_ar_residual_summary_.md](reports/again_dense_2hz_phase5_frozen_ar_residual_summary_.md)

AGAIN blocked binary confirmation: [reports/again_dense_2hz_phase5_temporal_residual_binary_big_confirm_20260630_025437.md](reports/again_dense_2hz_phase5_temporal_residual_binary_big_confirm_20260630_025437.md)

AGAIN repaired grouped compatibility verdict: [reports/again_dense_2hz_phase5_temporal_residual_grouped_compat_20260630_033520_REPAIRED_VERDICT.md](reports/again_dense_2hz_phase5_temporal_residual_grouped_compat_20260630_033520_REPAIRED_VERDICT.md)

## Best Results First

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

AGAIN grouped compatibility repaired verdict:

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
- repaired grouped compatibility pass: true
- failed repaired gates: `[]`
- verdict repair only: no retraining, no PCA generation, no grouped rerun, no 504

AGAIN dense substrate:

- `995/995` videos complete
- `243,575` row-level cortical rows
- `2 Hz`, `256 px`, float16
- official V-JEPA 2.1 ViT-G
- TRIBE v2 cache-only postpass
- labels aligned at true 2 Hz

AGAIN Phase 5 eval-mode repair:

- grouped real PR-AUC: `0.2300639382`
- grouped best matched control PR-AUC: `0.2042740689`
- grouped delta vs best matched control: `+0.0257898694`
- grouped AR-only PR-AUC: `0.2246816187`
- grouped fold-seed positive: `15/15`
- blocked caveat: the old fused lane lost to AR/control under blocked validation, motivating frozen-AR residual repair

AGAIN frozen-AR residual repair:

- grouped frozen AR PR-AUC: `0.2246816187`
- grouped best real residual PR-AUC: `0.2383409298`
- grouped best matched residual control PR-AUC: `0.2248361805`
- grouped delta vs frozen AR: `+0.0136593110`
- grouped delta vs best matched control: `+0.0135047493`
- blocked do-no-harm passed; this older repair did not yet beat AR/control under blocked temporal validation

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

## Canonical Artifacts

- Dense root: `.cache/h100_drive_downloads/again_tribe_v2_postpass_float16_256_2hz/`
- Phase 4 root: `$NEURAL_BRIDGE_EXTERNAL_ROOT/outputs/again_dense_2hz_phase4_pca_bridge_20260625_full/`
- Evidence bundle: `evidence/phase_0_to_5_historical_ladder_20260625/`
- Primary repair checkpoint root: `outputs/again_dense_2hz_phase5_adversarial_repair_fixplus_20260629_171825/`
- Eval-mode rescore root: `outputs/again_dense_2hz_phase5_adversarial_repair_fixplus_evalmode_rescore_/`
- Frozen-AR residual output root: `outputs/again_dense_2hz_phase5_frozen_ar_residual_/`
- Frozen-AR residual evidence snapshot: `evidence/phase_5_1_frozen_ar_residual/`
- Current reviewer dossier: `evidence/current_phase_5_5_review/`
- Blocked binary confirmation evidence snapshot: `evidence/phase_5_5_binary_blocked_confirmation_20260630_025437/`
- Grouped compatibility evidence snapshot: `evidence/phase_5_5_grouped_compatibility_20260630_033520/`

Current confirmed AGAIN lane: `future_arousal_max_delta_rows_4_10_train_q90` with `short_temporal_conv_residual` over fold-safe `temporal_mean_2s_then_pca256` and matched frozen AR.

Historical Phase 5 fused-head lane: `arousal_spike_rows_2_6_train_q90` with `gated_ar_pca_mlp` / `regression_plus_binary` / `temporal_mean_2s_then_pca256` / AR + temporal diagnostics.

## Next Work

Next work is explicit review and planning for any intentionally approved 504/broader compatibility confirmation. Keep claims bounded until such a run is performed, audited, and promoted. Continuous exact arousal forecasting and broad all-target/all-dataset temporal prediction remain open research problems, not current claims.
