# Project Memory

Current project state is maintained in:

- `docs/neural_bridge_phase5_5_evidence_ladder.md`
- `docs/current_claim_status.json`
- `docs/current_project_state.md`
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
- `reports/again_dense_2hz_phase5_temporal_residual_grouped_compat_20260630_033520_REPAIRED_VERDICT.md`
- `tools/run_h100_tribe_postpass.py`
- Google Drive `NeuralBridge_H100_AGAIN_tribe_v2_postpass_float16_256_2hz`
- local pull target `.cache/h100_drive_downloads/again_tribe_v2_postpass_float16_256_2hz/`

Future handoff entries should summarize only current state or newly completed work.

For the current claim, VEATIC and AGAIN are a paired evidence ladder. VEATIC-124 v2 established the original controlled future arousal event-ranking signal. AGAIN is now the scaled confirmation/current main result: dense 995-video H100 V-JEPA 2.1 / TRIBE v2 bundle, true 2Hz `labels_aligned_2hz.parquet`, Phase 5 eval-mode repair, frozen-AR residual repair, blocked washout-gap binary confirmation, and repaired grouped-video compatibility.

Current AGAIN bounded proof: `future_arousal_max_delta_rows_4_10_train_q90` with `short_temporal_conv_residual` passes blocked temporal confirmation with real PR-AUC `0.2670735630` vs frozen AR `0.2602336231` and best control `0.2593369051`; repaired grouped compatibility passes with real PR-AUC `0.2313831909` vs AR/frozen `0.2174953276` and best control `0.2174209937`. Continuous exact arousal forecasting and broad all-target/all-dataset temporal prediction remain open; no 504 run has been promoted.

Run `npm run audit:repo` before handing the repo to a fresh Codex session or teammate.
Run `npm run evidence:verify` before relying on the frozen v2 evidence bundle.
Use the external v1 tensor root and implemented trained-head runner when extending learned heads unless the tensor contract is intentionally refreshed.
