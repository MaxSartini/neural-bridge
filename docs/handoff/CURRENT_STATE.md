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

The bounded zero-label protocol is now preregistered in `docs/zero_label_video_only_deployment_bridge_pilot_preregistration.md`. This planning turn did not train or score a model. Stage 0 is limited to target/split/feature-policy manifests, dry-run matrix accounting, and executable leakage/cold-start contracts; Stage A still requires explicit authorization.

The preregistration was amended before implementation or scoring so spike/event capability cannot be lost behind a continuous-only pass. Continuous Spearman, top-5% lift, and training-q90 event PR-AUC are now three conjunctive required deployment endpoints. Every endpoint must beat its strongest zero-label control and retain at least `50%` of the matched Phase 7 teacher's incremental gain over the no-video zero-label anchor. This means half of the *teacher-added gain*, not half of its raw score. Absolute Phase 7 parity is not expected from a cold-start unseen-video model that receives none of the teacher's observed-arousal context; reaching parity would be exceptional.

The Stage 0 target-identity audit is mandatory. The current grouped Phase 7 runner relabels a block as `residual_future_max_delta_rows_4_10` while retaining continuous arrays built from `future_arousal_max_delta_rows_4_10`. The deployment pilot therefore locks the actual like-for-like raw future-movement value column and must checksum builder/scorer identity before fitting. This is a semantic contract finding, not a retroactive change to the Phase 7 grouped pass.

The preregistered path is a `96/96` development screen of teacher distillation and strict self-rollout, followed only after a locked-winner authorization by a `140/140` prospectively locked deployment-pilot confirmation. All 995 AGAIN videos have historical benchmark exposure; the 299-video subset is locked only prospectively for this new method. A harmonized V-JEPA 2.1 VEATIC + AGAIN pilot remains the later cross-domain option, not an automatic full re-encode.

## Fast Handoff Rule

For ordinary updates, edit this file after canonical evidence is committed and validated, then mine only `docs/handoff/` into the `neural_bridge` MemPalace wing and prove it with a top-3 recall query. Never remine the full project or historical evidence tree for routine handoff.
