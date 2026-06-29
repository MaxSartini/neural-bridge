# Current Project State

Last updated: 2026-06-29

## Current Claim

Defensible claim: cross-video future arousal spike / emotional moment ranking from video-derived cortical bridge features.

Canonical review: `docs/reviews/neural_bridge_phase5_adversarial_review_20260625.html`

Canonical deterministic Phase 5 repair report: `reports/again_dense_2hz_phase5_evalmode_rescore_summary_.md`

Do not claim:

- exact continuous future arousal forecasting is solved
- strict forward-time temporal generalization is proven
- Phase 5b/5c/Spark/max-capacity/deep/chimera outputs are canonical
- `holy_shit_pass` is a valid gate

## Evidence Summary

- Phase 5 is not fraud and not grossly leaky.
- Label permutation supports no gross leakage and remains near chance under eval-mode rescore.
- Grouped-video cross-video spike/event ranking signal is real, fold-robust, and survives deterministic eval-mode checkpoint rescoring.
- The canonical eval-mode matched-control grouped cortical edge is `+0.0257898694` PR-AUC.
- Blocked-temporal matched controls and AR-only beat real, so strict forward-time temporal generalization remains unproven.
- The fused gated head appears to let real PCA interfere with the AR/time path under blocked validation.
- Continuous arousal movement is not a solved claim; score it with ranking/lift metrics and controls.

## Current Numbers

Canonical deterministic eval-mode primary repair:

- grouped real `regression_plus_binary` PR-AUC: `0.2300639382`
- grouped best matched control `ar_plus_shuffled_pca` PR-AUC: `0.2042740689`
- grouped real-minus-control delta: `+0.0257898694`
- grouped AR-only PR-AUC: `0.2246816187`
- grouped fold-seed delta: positive in `15/15`
- grouped label permutation PR-AUC: `0.1058053218`
- grouped video-mean PCA diagnostic PR-AUC: `0.1054810779`
- blocked real PR-AUC: `0.2218656156`
- blocked best matched control `ar_plus_random_pca` PR-AUC: `0.2311845051`
- blocked real-minus-control delta: `-0.0093188895`
- blocked AR-only PR-AUC: `0.2654721820`
- blocked label permutation PR-AUC: `0.1101291638`
- blocked video-mean PCA diagnostic PR-AUC: `0.1955273615`

Primary lane:

- target: `arousal_spike_rows_2_6_train_q90`
- training source: `future_arousal_max_delta_rows_2_6`
- model/loss: `gated_ar_pca_mlp` / `regression_plus_binary`
- input: AR + `temporal_mean_2s_then_pca256` + temporal diagnostics

## Canonical Artifacts

- Dense root: `.cache/h100_drive_downloads/again_tribe_v2_postpass_float16_256_2hz/`
- Phase 4 root: `$NEURAL_BRIDGE_EXTERNAL_ROOT/outputs/again_dense_2hz_phase4_pca_bridge_20260625_full/`
- Evidence bundle: `evidence_bundle_phase0_to_phase5_20260625/`
- Primary repair checkpoint root: `outputs/again_dense_2hz_phase5_adversarial_repair_fixplus_20260629_171825/`
- Eval-mode rescore root: `outputs/again_dense_2hz_phase5_adversarial_repair_fixplus_evalmode_rescore_/`

Do not touch dense cache files, Phase 4 outputs, original Phase 5 output roots, or evidence bundle contents unless explicitly asked. Heavy output roots remain ignored.

## Current Repair Status

The primary repair matrix trained correctly and saved best checkpoints. The eval-mode rescore loaded all `702/702` saved best checkpoints, disabled dropout, and scored only the original held-out rows. These eval-mode metrics are the canonical deterministic Phase 5 primary repair numbers.

Next task: frozen-AR residual-over-AR repair before starting secondary heads. Freeze or anchor the AR score/logit as the baseline floor, train cortical PCA/diagnostics only as a residual correction, and combine `final_score = frozen_ar_score + alpha * residual_score` with `alpha` initialized near zero. The residual must beat frozen AR and matched residual controls, or learn to do no harm if cortical residual signal is useless. Do not claim strict temporal prediction unless frozen residual beats AR and matched controls under blocked temporal validation. Do not use Phase 5b/5c/Spark outputs, do not use the old `holy_shit_pass`, and do not rerun the full 702 matrix unless a targeted diagnostic requires it.
