# Project Memory

Current project state is maintained in:

- `README.md`
- `docs/neural_bridge_phase7_evidence.md`
- `docs/current_claim_status.json`
- `docs/current_project_state.md`
- `docs/handoff/CURRENT_STATE.md`
- `docs/phase7_continuous_checkpoint_ensemble_grouped_preregistration.md`
- `docs/how_neural_bridge_was_discovered.md`
- `docs/veatic_v2_evidence_summary.md`
- `docs/veatic_v2_evidence_freeze.md`
- `AGENTS.md`

Current post-v2 tensor state is maintained in:

- `docs/veatic_raw_representation_audit.md`
- `outputs/veatic_124_raw_representation_tensor_export_v1/tensor_export_report.md`
- `outputs/veatic_124_raw_representation_tensor_export_v1/tensor_export_summary.json`
- external tensor root `${NEURAL_BRIDGE_EXTERNAL_ROOT}/tensors/veatic_124_raw_representation_v1`

Current post-v2 implementation state is maintained in:

- `backend/scripts/run_veatic_frozen_tensor_trained_heads_benchmark.py`
- `backend/scripts/veatic_frozen_tensor_trained_heads.py`
- `backend/app/services/mlx_vjepa21_cortical.py`

Current dense AGAIN H100 state is maintained in:

- `docs/again_dense_h100_cache.md`
- `reports/again_labels_aligned_2hz_20260625_091209.md`
- `reports/again_dense_2hz_phase5_evalmode_rescore_summary_.md`
- `reports/again_dense_2hz_phase5_frozen_ar_residual_summary_.md`
- `reports/again_dense_2hz_phase5_temporal_residual_binary_big_confirm_20260630_025437.md`
- `reports/again_dense_2hz_phase5_temporal_residual_grouped_compat_20260630_033520_UPDATED_VERDICT.md`
- `reports/again_dense_2hz_phase6_original_three_checkpoint_control_complete_20260714_160001.md`
- `reports/again_dense_2hz_phase6_original_three_checkpoint_grouped_confirmation_20260714_163024.md`
- `reports/again_dense_2hz_phase7_continuous_checkpoint_ensemble_diagnostic_20260714_174513.md`
- `reports/again_dense_2hz_phase7_continuous_checkpoint_ensemble_grouped_20260714_181440.md`
- `reports/README.md`
- `tools/run_h100_tribe_postpass.py`
- Google Drive `NeuralBridge_H100_AGAIN_tribe_v2_postpass_float16_256_2hz`
- local pull target `.cache/h100_drive_downloads/again_tribe_v2_postpass_float16_256_2hz/`

Future handoff entries should summarize only current state or newly completed work.

Routine continuity uses the compact fast path: update and validate `docs/handoff/CURRENT_STATE.md`, mine only `docs/handoff/` into the `neural_bridge` MemPalace wing, and prove it with a top-3 recall query. Do not remine the full project or the historical evidence tree after ordinary result or documentation changes. Full corpus mining is reserved for an explicitly requested structural evidence/archive rebuild.

For the current claim, VEATIC and AGAIN are a paired evidence ladder. VEATIC-124 v2 established the original controlled future arousal event-ranking signal. AGAIN is the scaled confirmation/current main benchmark. Phase 7 is the current performance headline: its fresh grouped washout-target checkpoint ensemble passed `420/420`, beat AR and best matched controls on Spearman and top-5% lift in all `15/15` fold-groups and `5/5` fold means, and failed no gates.

Lead with total bridge magnitude, not only the last-mile AR delta. On the original same-target grouped spike benchmark, raw cortical `0.136579` became `0.2383409298` (`+74.51%`) and finished `+39.95%` above direct AR-plus-raw and `+38.85%` above the canonical Phase 4 score `0.1716477402`. Relative to the original validated continuous bridge, Phase 7 is `+16.61%` Spearman, `+23.59%` top-5% lift, and `+14.52%` top-1% lift; the top-5% margin beyond AR grew `+98.92%`. Keep same-target and whole-generation comparisons explicitly labeled.

Raw predicted cortical/fMRI features alone fail badly on AGAIN: on the original Phase 3 spike target, blocked `raw_cortical_only` PR-AUC was `0.124315` vs AR-only `0.203622`, and direct `AR_plus_raw_cortical` was `0.167731`. Current AGAIN bounded proof: `future_arousal_max_delta_rows_4_10_train_q90` with `short_temporal_conv_residual` passes blocked temporal confirmation with real PR-AUC `0.2670735630` vs frozen AR `0.2602336231` and best control `0.2593369051`; updated grouped compatibility passes with real PR-AUC `0.2313831909` vs AR/frozen `0.2174953276` and best control `0.2174209937`.

Do not bury the Phase 7 win under Phase 5.5 history. State first that its grouped continuous confirmation passed cleanly, then preserve the exact scope: ranking/lift rather than calibrated exact values. The locked direct-supervised result now proves cached-feature zero-label-at-inference operation; end-to-end raw-video runtime and external transfer remain next.

The binary selected-head 420 audit and original three-checkpoint blocked/grouped confirmations remain promoted foundations. The locked direct-supervised video-only confirmation now passes on the held-out 299-video pool. A harmonized V-JEPA 2.1 VEATIC+AGAIN pilot is a later cross-domain option, not an automatic full re-encode. Do not resume same-family Optuna or launch a broad architecture sweep.

Run `npm run audit:repo` before handing the repo to a fresh Codex session or teammate.
Run `npm run evidence:verify` before relying on the frozen v2 evidence bundle.
Use the external v1 tensor root and implemented trained-head runner when extending learned heads unless the tensor contract is intentionally refreshed.
