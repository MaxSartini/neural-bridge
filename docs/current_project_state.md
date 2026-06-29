# Current Project State

Last updated: 2026-06-30

## Current Claim

Defensible claim: cross-video future arousal spike / emotional moment ranking from video-derived cortical bridge features.

Canonical review: `docs/reviews/neural_bridge_phase5_adversarial_review_20260625.html`

Canonical deterministic Phase 5 repair report: `reports/again_dense_2hz_phase5_evalmode_rescore_summary_.md`

Canonical frozen-AR residual report: `reports/again_dense_2hz_phase5_frozen_ar_residual_summary_.md`

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
- Frozen-AR residual repair strengthens the cross-video ranking claim by showing real cortical residual improves grouped beyond frozen AR and matched residual controls.
- Blocked-temporal matched controls and AR-only still prevent a strict forward-time temporal generalization claim.
- Frozen-AR residual reduced blocked harm: old fused real was far below blocked AR, while frozen residual is within do-no-harm tolerance.
- The fused gated head appeared to let real PCA interfere with the AR/time path under blocked validation; frozen AR made the AR path the baseline floor.
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

Frozen-AR residual repair:

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
- Frozen-AR residual output root: `outputs/again_dense_2hz_phase5_frozen_ar_residual_/`
- Frozen-AR residual evidence snapshot: `evidence_bundle_phase5_frozen_ar_residual_/`

Do not touch dense cache files, Phase 4 outputs, original Phase 5 output roots, or evidence bundle contents unless explicitly asked. Heavy output roots remain ignored.

## Current Repair Status

The primary repair matrix trained correctly and saved best checkpoints. The eval-mode rescore loaded all `702/702` saved best checkpoints, disabled dropout, and scored only the original held-out rows. These eval-mode metrics are the canonical deterministic Phase 5 primary repair numbers.

The frozen-AR residual experiment re-forwarded AR-only best checkpoints in eval mode, avoided AR retraining, and trained cortical residual corrections over the frozen AR baseline. It produced a grouped residual pass and a blocked do-no-harm pass, but not a blocked residual pass or full forward-time pass.

Next task: targeted blocked residual improvement, not broad secondary heads. Candidate repairs include stronger residual alpha regularization, blocked-only inner-val delta selection, rank/lift auxiliary loss, monotonic/do-no-harm residual gating, or training residual branches only where AR confidence is low. Do not claim strict temporal prediction unless frozen residual beats AR and matched controls under blocked temporal validation. Do not use Phase 5b/5c/Spark outputs, do not use the old `holy_shit_pass`, and do not rerun the full 702 matrix unless a targeted diagnostic requires it.
