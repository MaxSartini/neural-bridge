# Neural Bridge

Neural Bridge turns video-derived cortical representations into ranked human-response intelligence under strict controls.

## Current State

Defensible claim: cross-video future arousal spike / emotional moment ranking.

Canonical review: [docs/reviews/neural_bridge_phase5_adversarial_review_20260625.html](docs/reviews/neural_bridge_phase5_adversarial_review_20260625.html)

Canonical deterministic rescore: [reports/again_dense_2hz_phase5_evalmode_rescore_summary_.md](reports/again_dense_2hz_phase5_evalmode_rescore_summary_.md)

Frozen-AR residual repair: [reports/again_dense_2hz_phase5_frozen_ar_residual_summary_.md](reports/again_dense_2hz_phase5_frozen_ar_residual_summary_.md)

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

Frozen-AR residual repair result:

- grouped frozen AR PR-AUC: `0.2246816187`
- grouped best real residual PR-AUC: `0.2383409298`
- grouped best matched residual control PR-AUC: `0.2248361805`
- grouped delta vs frozen AR: `+0.0136593110`
- grouped delta vs best matched control: `+0.0135047493`
- blocked frozen AR PR-AUC: `0.2654721820`
- blocked best real residual PR-AUC: `0.2635930904`
- blocked delta vs frozen AR: `-0.0018790916`
- blocked delta vs best matched control: `-0.0017473477`
- do_no_harm_blocked_pass: yes
- full_forward_time_pass: no
- recommendation: `exploratory_grouped_only`

The current canonical state includes the deterministic eval-mode Phase 5 repair plus the frozen-AR residual experiment. The frozen-AR residual result strengthens the cross-video future arousal spike / emotional moment ranking claim by showing real cortical residual improves grouped beyond frozen AR and matched residual controls. Blocked strict forward-time temporal generalization remains unproven. Frozen-AR residual reduced blocked harm: old fused real was far below blocked AR, while frozen residual is within do-no-harm tolerance.

Do not claim continuous arousal forecasting is solved, strict temporal prediction is proven, Phase 5b/5c/Spark outputs are canonical, or `holy_shit_pass` is a valid gate.

## Canonical Artifacts

- Dense root: `.cache/h100_drive_downloads/again_tribe_v2_postpass_float16_256_2hz/`
- Phase 4 root: `$NEURAL_BRIDGE_EXTERNAL_ROOT/outputs/again_dense_2hz_phase4_pca_bridge_20260625_full/`
- Evidence bundle: `evidence_bundle_phase0_to_phase5_20260625/`
- Primary repair checkpoint root: `outputs/again_dense_2hz_phase5_adversarial_repair_fixplus_20260629_171825/`
- Eval-mode rescore root: `outputs/again_dense_2hz_phase5_adversarial_repair_fixplus_evalmode_rescore_/`
- Frozen-AR residual output root: `outputs/again_dense_2hz_phase5_frozen_ar_residual_/`
- Frozen-AR residual evidence snapshot: `evidence_bundle_phase5_frozen_ar_residual_/`

Primary Phase 5 lane: `arousal_spike_rows_2_6_train_q90` with `gated_ar_pca_mlp` / `regression_plus_binary` / `temporal_mean_2s_then_pca256` / AR + temporal diagnostics.

## Next Work

Targeted blocked residual improvement is next; do not widen to broad secondary heads yet. Candidate targeted repairs include stronger residual alpha regularization, blocked-only inner-val delta selection, rank/lift auxiliary loss, monotonic/do-no-harm residual gating, or training residual branches only where AR confidence is low. Do not claim strict temporal generalization unless the frozen residual beats AR and matched controls under blocked temporal validation. Do not rerun the full 702 matrix unless a targeted diagnostic shows it is necessary.
