# Neural Bridge Current-State Handoff

Updated: 2026-07-14
Branch: `main`

## Current Result

Phase 7 is the current performance headline. Its fresh grouped continuous checkpoint ensemble passed `420/420` with failed gates `[]`, `15/15` fold-group wins, and `5/5` positive fold means versus target-specific AR and matched controls on Spearman and top-5% lift.

- Spearman real / AR / best control: `0.2603011121` / `0.2405371348` / `0.2402523335`.
- Top-5% lift real / AR / best control: `0.0975979581` / `0.0895663763` / `0.0897088493`.
- Original same-target grouped spike raw cortical `0.136579` → Phase 5 frozen-AR residual `0.2383409298`: `+74.51%`.
- Same target: `+39.95%` over direct AR-plus-raw and `+38.85%` over canonical Phase 4 `0.1716477402`.
- Phase 7 versus the original validated continuous bridge: `+16.61%` Spearman, `+23.59%` top-5%, and `+14.52%` top-1%.
- Top-5% margin beyond AR versus the original continuous bridge: `+98.92%`.
- Supporting Phase 7 event PR-AUC: real `0.2231895329`, AR `0.2088047413`, control `0.2096090680`, positive `15/15`; secondary evidence, not the primary gate.

The separate Phase 7 blocked continuous result remains a strong `4/5` Spearman-vs-AR near-pass under its literal `5/5` gate. It is not a formal blocked pass and does not weaken the independent grouped `420/420` pass.

## Authority

Current files and executable evidence override historical memory:

- `README.md`
- `AGENTS.md`
- `docs/neural_bridge_phase7_evidence.md`
- `docs/current_project_state.md`
- `docs/current_claim_status.json`
- `reports/again_dense_2hz_phase7_continuous_checkpoint_ensemble_grouped_20260714_181440.md`

Validation is green: fully provisioned `150`-test suite, `npm run verify`, repository audit, and real Optuna/Polars/MLflow/SHAP research-tooling verification on MLX `Device(gpu, 0)` / MPS.

Both codebase-memory projects are ready: internal repo and external SSD heavy-artifact workspace.

## Next Task

Run a bounded zero-label video-only deployment-bridge pilot using the Phase 7 teacher signal. Evaluation must forbid observed-arousal teacher forcing. A harmonized V-JEPA 2.1 VEATIC + AGAIN pilot is the later cross-domain option, not an automatic full re-encode.

## Fast Handoff Rule

For ordinary updates, edit this file after canonical evidence is committed and validated, then mine only `docs/handoff/` into the `neural_bridge` MemPalace wing and prove it with a top-3 recall query. Never remine the full project or historical evidence tree for routine handoff.
