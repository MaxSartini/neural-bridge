# Neural Bridge Current-State Handoff

Updated: 2026-07-15
Branch: `main`

## Current Result

The newest result is the locked zero-label deployment win. The fixed direct-supervised temporal model completed `140/140` rows on the prospectively locked 299-video AGAIN pool with audit pass `true` and failed Tier 1 gates `[]`. It used no observed arousal at inference and reached `0.1785132961` Spearman, `0.0766079674` top-5% lift, and `0.1710622218` event PR-AUC. Gains over the strongest matched false-signal/no-video controls were `+77.65%`, `+70.80%`, and `+26.50%`; every full-video endpoint won `5/5` panels, every bootstrap lower bound was positive, and the first-30-second tier passed.

Phase 7 remains the observed-arousal-assisted research ceiling. Its fresh grouped continuous checkpoint ensemble passed `420/420` with failed gates `[]`, `15/15` fold-group wins, and `5/5` positive fold means versus target-specific AR and matched controls on Spearman and top-5% lift.

- Spearman real / AR / best control: `0.2603011121` / `0.2405371348` / `0.2402523335`.
- Top-5% lift real / AR / best control: `0.0975979581` / `0.0895663763` / `0.0897088493`.
- Original same-target grouped spike raw cortical `0.136579` → Phase 5 frozen-AR residual `0.2383409298`: `+74.51%`.
- Same target: `+39.95%` over direct AR-plus-raw and `+38.85%` over canonical Phase 4 `0.1716477402`.
- Phase 7 versus the original validated continuous bridge: `+16.61%` Spearman, `+23.59%` top-5%, and `+14.52%` top-1%.
- Top-5% margin beyond AR versus the original continuous bridge: `+98.92%`.
- Supporting Phase 7 event PR-AUC: real `0.2231895329`, AR `0.2088047413`, control `0.2096090680`, positive `15/15`; secondary evidence, not the primary gate.

## Authority

Current files and executable evidence override historical memory:

- `README.md`
- `AGENTS.md`
- `docs/neural_bridge_zero_label_deployment_evidence.md`
- `reports/again_dense_2hz_zero_label_direct_supervised_locked_confirmation_20260715.md`
- `evidence/zero_label_video_only_direct_supervised_locked_confirmation_20260715/README.md`
- `docs/neural_bridge_phase7_evidence.md`
- `docs/current_project_state.md`
- `docs/current_claim_status.json`
- `reports/again_dense_2hz_phase7_continuous_checkpoint_ensemble_grouped_20260714_181440.md`

Validation is green: `196 passed, 1 skipped` under `npm run verify`, repository audit, strict-benchmark dry run, production build, and real Optuna/Polars/MLflow/SHAP research-tooling verification on MLX `Device(gpu, 0)` / MPS.

Both codebase-memory projects are ready: internal repo and external SSD heavy-artifact workspace.

## Ownership And Asset Boundary

Neural Bridge is sole-founder work. The founder conceived and created the Neural Bridge ideas, target and washout design, bridge architecture, implementation, experiments, controls, training/evaluation procedures, trained heads, derived caches, evidence, reports, and product code across the VEATIC and AGAIN programmes. V-JEPA, TRIBE, and the VEATIC/AGAIN source datasets are third-party upstream dependencies or research inputs; they are replaceable, are not the Neural Bridge commercial output, and the source datasets are not shipped with the product.

Keep the two AGAIN H100 cache layers distinct:

- approximately `1 TB` in the founder's Google Drive workspace is the dense V-JEPA 2.1 cache generated from all `995` AGAIN videos;
- approximately `38.7 GiB` / `40 GB` locally is the downstream TRIBE/predicted-cortical postpass used by Neural Bridge.

Generated caches and downstream artifacts created by the project are Neural Bridge project assets, subject to applicable upstream and dataset terms. Private valuation material is not canonical project state and must not be committed or mined into MemPalace.

## Next Task

The bounded AGAIN deployment campaign is complete and passed all three locked tiers at `140/140`. Keep the direct-supervised method frozen and move outward: integrate raw-video feature generation, validate latency and output packaging, and preregister an external/cross-domain zero-label confirmation. A bounded V-JEPA 2.1 VEATIC re-encode plus harmonized VEATIC+AGAIN pilot is justified, but must retain domain balance and leave-one-domain-out evaluation. Do not reopen the 299 pool for tuning. Development trials belong in `docs/how_neural_bridge_was_discovered.md` and the Stage A report.

## Fast Handoff Rule

For ordinary updates, edit this file after canonical evidence is committed and validated, then mine only `docs/handoff/` into the `neural_bridge` MemPalace wing and prove it with a top-3 recall query. Never remine the full project or historical evidence tree for routine handoff.
