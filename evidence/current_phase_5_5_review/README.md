# Neural Bridge Phase 5.5 Review Evidence Dossier

This folder is the current reviewer-facing evidence pack. It exists so an external reviewer can inspect the current Neural Bridge claim without hunting through random output roots.

## Best Results First

### AGAIN Blocked Binary Confirmation

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

Primary folder: `12_again_phase_5_5_binary_blocked_confirmation/`

### AGAIN Grouped Compatibility

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

Primary folder: `13_again_phase_5_5_grouped_compatibility/`

### Raw Cortical Alone Fails Badly

- target: `arousal_spike_rows_2_6_train_q90`
- blocked `raw_cortical_only` PR-AUC: `0.124315`
- blocked AR-only PR-AUC: `0.203622`
- blocked `AR_plus_raw_cortical` PR-AUC: `0.167731`
- grouped `raw_cortical_only` PR-AUC: `0.136579`
- grouped AR-only PR-AUC: `0.147251`
- grouped `AR_plus_raw_cortical` PR-AUC: `0.170299`

This is the key negative-control context: raw cortical-derived features alone are weak. Neural Bridge is the controlled bridge pipeline that makes video-derived neuro-response representations generated from brain cortical response data useful for future event ranking.

Primary folder: `05_again_phase_3_raw_cortical/`

### VEATIC-124 v2 Foundation

- strongest blocked full-frame spike row: `cortical_pca64_delta` / `arousal__future_spike_1_3s`
- PR-AUC: `0.2536`
- AR: `0.1969`
- shuffled: `0.1840`
- random: `0.1944`
- balanced event-vs-stable target `arousal__future_spike_1_3s@0.05`: `cortical_pca64_delta` PR-AUC `0.3394`
- balanced deltas: `+0.0609` over AR, `+0.0631` over shuffled, `+0.0476` over random

Primary folder: `01_veatic_v2_foundation/`

## How To Use This Dossier

- `CLAIM_LEDGER.md` maps each claim to exact numbers, source files, and caveats.
- `artifact_manifest.csv` lists every copied evidence file, original source path, destination path, byte size, role, and SHA-256 checksum.
- `DEFINITIONS_AND_PROCESS.md` defines targets, protocols, controls, leakage discipline, and promotion boundaries.
- `REVIEWER_CHECKLIST.md` gives a compact validation path.
- `14_executable_validation_and_code/` indexes relevant AGAIN and VEATIC v2 scripts, tests, benchmark artifacts, runtime-only tools, and the latest deterministic test-suite result.

Historical benchmark reports inside milestone folders preserve their original wording. If a historical report says an earlier result was exploratory or unproven, read it in that milestone context; the current claim boundary is the one in `CLAIM_LEDGER.md` and the repo root `README.md`.

## Phase Map

- `00_current_claim_and_review_entrypoint/` - current-facing repo orientation and claim boundary.
- `01_veatic_v2_foundation/` - VEATIC-124 v2 foundation evidence.
- `02_again_phase_0_inventory_dense_cache/` - AGAIN dense-cache inventory and substrate evidence.
- `03_again_phase_1_label_alignment/` - true 2 Hz label alignment evidence.
- `04_again_phase_2_ar_baseline/` - AR baseline evidence.
- `05_again_phase_3_raw_cortical/` - raw cortical baseline evidence.
- `06_again_phase_4_pca_bridge/` - fold-safe PCA bridge evidence.
- `07_again_phase_5_0_evalmode_primary/` - deterministic eval-mode primary correction evidence.
- `08_again_phase_5_1_frozen_ar_residual/` - frozen-AR residual design evidence.
- `09_again_phase_5_2_blocked_residual_and_ar_audits/` - blocked residual/control audits, AR decomposition, and failed continuous diagnostic.
- `10_again_phase_5_3_target_redesign_and_foldsafe_pca/` - target redesign, fold-safe PCA, and small redesigned-target diagnostic.
- `11_again_phase_5_4_temporal_residual_architecture_diagnostic/` - temporal/event-context residual architecture diagnostic.
- `12_again_phase_5_5_binary_blocked_confirmation/` - matched 10-seed blocked binary confirmation.
- `13_again_phase_5_5_grouped_compatibility/` - updated grouped-video compatibility evidence.
- `14_executable_validation_and_code/` - tests, suites, scripts, benchmark artifact locations, and latest validation result.

## Scope Boundaries

This dossier is tracked evidence only. It does not include checkpoints, tensors, `.npy`, `.npz`, dense caches, V-JEPA/TRIBE assets, or full heavy output roots. No training, PCA generation, 504 run, or benchmark rerun was performed to create this documentation pack.

Correct claim: Neural Bridge demonstrates controlled future human arousal event-ranking from frozen video-derived predictions generated from brain cortical response data across VEATIC and AGAIN. The frozen video-side features are video-derived predictions generated from brain cortical response data, not generic video embeddings. Bounded strict forward-time future-event ranking is proven on AGAIN for `future_arousal_max_delta_rows_4_10_train_q90` with `short_temporal_conv_residual`; continuous exact arousal forecasting and broad all-target/all-dataset temporal prediction remain open.
