# Project Memory

Current project state is maintained in:

- `docs/current_project_state.md`
- `docs/veatic_v2_evidence_summary.md`
- `docs/veatic_v2_evidence_freeze.md`
- `ROADMAP.md`
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
- `tools/run_h100_tribe_postpass.py`
- Google Drive `NeuralBridge_H100_AGAIN_tribe_v2_postpass_float16_256_2hz`
- local pull target `.cache/h100_drive_downloads/again_tribe_v2_postpass_float16_256_2hz/`

Future handoff entries should summarize only current state or newly completed work.

For AGAIN, the current full-dataset substrate is the dense 995-video H100 V-JEPA 2.1 / TRIBE v2 bundle. The next work is local completeness audit and grouped/control benchmarking over that bundle.

Run `npm run audit:repo` before handing the repo to a fresh Codex session or teammate.
Run `npm run evidence:verify` before relying on the frozen v2 evidence bundle.
Use the external v1 tensor root and implemented trained-head runner when extending learned heads unless the tensor contract is intentionally refreshed.
