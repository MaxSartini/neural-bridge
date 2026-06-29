# Neural Bridge

Neural Bridge turns video-derived cortical representations into ranked human-response intelligence under strict controls.

## Current State

Defensible claim: cross-video future arousal spike / emotional moment ranking.

Canonical review: [docs/reviews/neural_bridge_phase5_adversarial_review_20260625.html](docs/reviews/neural_bridge_phase5_adversarial_review_20260625.html)

Canonical deterministic rescore: [reports/again_dense_2hz_phase5_evalmode_rescore_summary_.md](reports/again_dense_2hz_phase5_evalmode_rescore_summary_.md)

Key eval-mode Phase 5 primary repair numbers:

- grouped real `regression_plus_binary` PR-AUC: `0.2300639382`
- grouped best matched control `ar_plus_shuffled_pca` PR-AUC: `0.2042740689`
- grouped real-minus-control delta: `+0.0257898694`
- grouped AR-only PR-AUC: `0.2246816187`
- grouped fold-seed delta: positive in `15/15`
- blocked real PR-AUC: `0.2218656156`
- blocked best matched control `ar_plus_random_pca` PR-AUC: `0.2311845051`
- blocked real-minus-control delta: `-0.0093188895`
- blocked AR-only PR-AUC: `0.2654721820`

The Phase 5 primary repair result is a grouped-video ranking result, not proof of strict temporal prediction. Label permutation stays near chance, video-mean PCA does not explain the grouped signal, and blocked-temporal matched controls plus AR-only beat real. The fused gated head appears to let real PCA interfere with the AR/time path under blocked validation.

Do not claim continuous arousal forecasting is solved, strict temporal prediction is proven, Phase 5b/5c/Spark outputs are canonical, or `holy_shit_pass` is a valid gate.

## Canonical Artifacts

- Dense root: `.cache/h100_drive_downloads/again_tribe_v2_postpass_float16_256_2hz/`
- Phase 4 root: `$NEURAL_BRIDGE_EXTERNAL_ROOT/outputs/again_dense_2hz_phase4_pca_bridge_20260625_full/`
- Evidence bundle: `evidence_bundle_phase0_to_phase5_20260625/`
- Primary repair checkpoint root: `outputs/again_dense_2hz_phase5_adversarial_repair_fixplus_20260629_171825/`
- Eval-mode rescore root: `outputs/again_dense_2hz_phase5_adversarial_repair_fixplus_evalmode_rescore_/`

Primary Phase 5 lane: `arousal_spike_rows_2_6_train_q90` with `gated_ar_pca_mlp` / `regression_plus_binary` / `temporal_mean_2s_then_pca256` / AR + temporal diagnostics.

## Next Work

Run the frozen-AR residual-over-AR repair before starting secondary heads. Freeze or anchor the AR score/logit as the baseline floor, train cortical PCA/diagnostics only as a residual correction, and combine `final_score = frozen_ar_score + alpha * residual_score` with `alpha` initialized near zero. The residual must beat frozen AR and matched residual controls, or learn to do no harm. Do not claim strict temporal generalization unless the frozen residual beats AR and matched controls under blocked temporal validation. Do not rerun the full 702 matrix unless a targeted diagnostic shows it is necessary.
