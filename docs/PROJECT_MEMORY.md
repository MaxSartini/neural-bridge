# Project Memory

Current project state is maintained in:

- `docs/current_project_state.md`
- `docs/veatic_v2_evidence_summary.md`
- `docs/veatic_v2_evidence_freeze.md`
- `docs/superseded_artifacts.md`
- `ROADMAP.md`
- `AGENTS.md`

Current post-v2 tensor state is maintained in:

- `docs/veatic_raw_representation_audit.md`
- `outputs/veatic_124_raw_representation_tensor_export_v1/tensor_export_report.md`
- `outputs/veatic_124_raw_representation_tensor_export_v1/tensor_export_summary.json`
- external tensor root `${NEURAL_BRIDGE_EXTERNAL_ROOT}/tensors/veatic_124_raw_representation_v1`

Older pre-v2 VEATIC 5/20/50-video handoff notes, implementation inventories, and acceleration audits were removed from `docs/` because they contradicted the VEATIC-124 v2 baseline. Use git history if that old context is needed for archaeology.

Future handoff entries should summarize only current state or newly completed work. Do not reintroduce superseded pre-v2 benchmark status as active project memory.

Run `npm run audit:repo` before handing the repo to a fresh Codex session or teammate.
Run `npm run evidence:verify` before relying on the frozen v2 evidence bundle.
Use the external v1 tensor root, not a rerun, when training the next learned heads unless the tensor contract is intentionally refreshed.
