# Project Memory

Current project state is maintained in:

- `docs/neural_bridge_phase5_5_evidence_ladder.md`
- `docs/current_claim_status.json`
- `docs/current_project_state.md`
- `docs/phase5_selected_head_420_confirmation_plan.md`
- `docs/phase6_original_three_checkpoint_grouped_confirmation_plan.md`
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
- `reports/README.md`
- `tools/run_h100_tribe_postpass.py`
- Google Drive `NeuralBridge_H100_AGAIN_tribe_v2_postpass_float16_256_2hz`
- local pull target `.cache/h100_drive_downloads/again_tribe_v2_postpass_float16_256_2hz/`

Future handoff entries should summarize only current state or newly completed work.

For the current claim, VEATIC and AGAIN are a paired evidence ladder. VEATIC-124 v2 established the original controlled future arousal event-ranking signal. AGAIN is now the scaled confirmation/current main result: dense 995-video H100 V-JEPA 2.1 / TRIBE v2 bundle, true 2Hz `labels_aligned_2hz.parquet`, Phase 5 eval-mode grouped event and continuous future-movement ranking/lift, frozen-AR residual design, blocked washout-gap binary confirmation, and updated grouped-video compatibility.

Raw predicted cortical/fMRI features alone fail badly on AGAIN: on the original Phase 3 spike target, blocked `raw_cortical_only` PR-AUC was `0.124315` vs AR-only `0.203622`, and direct `AR_plus_raw_cortical` was `0.167731`. Current AGAIN bounded proof: `future_arousal_max_delta_rows_4_10_train_q90` with `short_temporal_conv_residual` passes blocked temporal confirmation with real PR-AUC `0.2670735630` vs frozen AR `0.2602336231` and best control `0.2593369051`; updated grouped compatibility passes with real PR-AUC `0.2313831909` vs AR/frozen `0.2174953276` and best control `0.2174209937`.

Do not erase the continuous result when stating the claim boundary. The deterministic Phase 5 eval-mode `regression_plus_binary` lane passed grouped continuous future-movement ranking/lift across 15 fold-seed evaluations: real future-movement Spearman `0.2232222830` beat AR-only `0.1982207591`, shuffled `0.1938183619`, and random `0.1931781163`; real top-1% average-true-movement lift `0.1359465244` beat `0.1115815364`, `0.1125842464`, and `0.1136304212`. Exact continuous values and blocked continuous generalization remain open. The later washout continuous diagnostic improved Spearman but failed its full top-5%/seed-consistency gate. Broad all-target/all-dataset temporal prediction remains open; no 504 run has been promoted.

The selected-head 420 audit is complete and canonical. The original three-checkpoint ensemble is additionally promoted under two separate fresh control-complete confirmations: blocked `140/140` with real/AR/best-control PR-AUC `0.2668905` / `0.2597236` / `0.2589302`, and grouped-video `420/420` with `0.2343676` / `0.2180498` / `0.2179717`. The grouped run was positive versus AR and the per-fold-group best control in `15/15`, positive by fold mean in `5/5`, added `+0.0082201` over the 45 real-member mean, and failed no gates. No follow-on modeling sweep is authorized without a new explicit plan. The historical literal 504 matrix was an older three-seed/four-variant development design; do not recreate it, resume same-family Optuna, or widen continuous claims.

Run `npm run audit:repo` before handing the repo to a fresh Codex session or teammate.
Run `npm run evidence:verify` before relying on the frozen v2 evidence bundle.
Use the external v1 tensor root and implemented trained-head runner when extending learned heads unless the tensor contract is intentionally refreshed.
