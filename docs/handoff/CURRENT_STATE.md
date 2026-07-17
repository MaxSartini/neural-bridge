# Neural Bridge Current-State Handoff

Updated: 2026-07-17
Branch: `codex/veatic21-retraining-foundation`

## Current Result

The newest result is the locked zero-label deployment win. The fixed direct-supervised temporal model completed `140/140` rows on the prospectively locked 299-video AGAIN pool with audit pass `true` and failed Tier 1 gates `[]`. It used no observed arousal at inference and reached `0.1785132961` Spearman, `0.0766079674` top-5% lift, and `0.1710622218` event PR-AUC. Gains over the strongest matched false-signal/no-video controls were `+77.65%`, `+70.80%`, and `+26.50%`; every full-video endpoint won `5/5` panels, every bootstrap lower bound was positive, and the first-30-second tier passed.

Phase 7 remains the observed-arousal-assisted research ceiling. Its fresh grouped continuous checkpoint ensemble passed `420/420` with failed gates `[]`, `15/15` fold-group wins, and `5/5` positive fold means versus target-specific AR and matched controls on Spearman and top-5% lift.

No VEATIC 2.1 retraining result is promoted yet. The new VEATIC work described below is a validated execution foundation and real-cache dry plan, not a completed discovery or confirmation result.

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

Current change validation is green except for an unrelated pre-existing repository-readiness audit: `381 passed, 1 skipped`, Python compilation passed, the strict VEATIC benchmark dry run passed, and the frontend production build passed. `npm run verify` stops before those stages because three unchanged tools still contain machine-specific SSD defaults: `tools/benchmark_veatic_vjepa21_only_one_video.py`, `tools/compare_again_vjepa21_compact_one_video.py`, and `tools/run_single_video_neural_bridge_mlx.py`.

Both codebase-memory projects are ready: internal repo and external SSD heavy-artifact workspace.

## Ownership And Asset Boundary

Neural Bridge is sole-founder work. The founder conceived and created the Neural Bridge ideas, target and washout design, bridge architecture, implementation, experiments, controls, training/evaluation procedures, trained heads, derived caches, evidence, reports, and product code across the VEATIC and AGAIN programmes. V-JEPA, TRIBE, and the VEATIC/AGAIN source datasets are third-party upstream dependencies or research inputs; they are replaceable, are not the Neural Bridge commercial output, and the source datasets are not shipped with the product.

Keep the two AGAIN H100 cache layers distinct:

- approximately `1 TB` in the founder's Google Drive workspace is the dense V-JEPA 2.1 cache generated from all `995` AGAIN videos;
- approximately `38.7 GiB` / `40 GB` locally is the downstream TRIBE/predicted-cortical postpass used by Neural Bridge.

Generated caches and downstream artifacts created by the project are Neural Bridge project assets, subject to applicable upstream and dataset terms. Private valuation material is not canonical project state and must not be committed or mined into MemPalace.

## VEATIC 2.1 Retraining State

The bounded H100 encoding run is complete and the Vast instance has been terminated. The authoritative compact local substrates are:

- V-JEPA 2.1: `/Volumes/onn. Drive/Neural Bridge/cache/veatic_h100_vjepa21_compact_20260716`, `124/124` videos, approximately `1.0 GiB`.
- TRIBE v2: `/Volumes/onn. Drive/Neural Bridge/cache/veatic_h100_tribe_v2_mlx_compact_20260716`, `124/124` videos, approximately `827 MiB`.
- Exact dense contract: `20,657` rows at `2 Hz`, prediction width `20,484`, model SHA-256 `ded8a1375bf118a74230ba6f2baef924e2cdbd508870fcddc7dd950293ba156a`, row-plan SHA-256 `81a7491ab7653eb15dafc93ea9f31cd80a336bab614e6bec182b465f51e803b1`, and full dataset seal `c010f6a7b0438163bdf747c08e96688c8108c866e9882a0177627c5195240fe5`.

The new VEATIC 2.1 code is a full fresh-retraining foundation. It treats old VEATIC as historical identity only and re-fits every PCA, normalizer, target-specific neural AR, train-q90 threshold, temporal head, binary head, matched control, and final model from the compact 2 Hz substrate. It transfers the strongest AGAIN methods as bounded priors without reusing fitted AGAIN artifacts or blindly fixing AGAIN's winning configuration.

Implemented and tested contracts include:

- four primary `rows_4_10` continuous targets: arousal maximum rise, valence rise magnitude, valence drop magnitude, and valence absolute movement;
- exact train-only-q90 event counterparts, plus valence direction derived from paired rise/drop predictions;
- all `124` videos in five grouped outer folds with no internal reserve, plus three grouped inner folds for leakage-safe recipe selection;
- six prespecified VEATIC discovery recipes spanning temporal-mean/current/delta cortical families, PCA `64/256`, and short-conv/flat/current-row heads;
- fresh MLX-backed fold-safe randomized PCA with leading-component slicing, response-free five-row causal video features, exact neural AR7, gated/capped frozen-AR residual correction, weighted Huber continuous heads, and true BCE-with-logits binary heads;
- deterministic whole-video shuffled, matched-random, train-only-video-mean, diagnostics-only, no-video, and label-permutation controls;
- grouped continuous/event gates, contribution caps, ensemble uplift, first-30-second zero-label checks, whole-video bootstrap, and valence direction evaluation;
- exact restart-safe end-state accounting: `3,240` nested discovery score rows, `60` reusable PCA fits, `1,680` privileged continuous rows, `1,680` privileged binary rows, `560` zero-label rows, and final refits on all `124` videos.

The full-checksum real-cache dry run passed on 2026-07-17. It reported `promotable: false`, `canonical_gates_passed: false`, no outer-test selection use, exactly `3,920` planned confirmation rows, and `12` target/protocol global selections for final refit. This is the correct dry-run state, not evidence of model performance.

### Immediate blocker and next execution

The numerical discovery/confirmation/final executor wiring is not complete. `backend/scripts/run_veatic21_endstate.py` currently builds and audits the full plan but still expects three missing numerical entry points from `veatic21_modeling`: `execute_veatic21_nested_discovery`, `execute_veatic21_confirmation_cell`, and `execute_veatic21_all124_refit`. Starting the canonical run before those connections exist would stop at the executor boundary.

Next actions, in order:

1. Connect the end-state runner directly to the validated dense substrate, PCA, feature, control, neural-AR, continuous/BCE, prediction-sealing, and reporting modules. Do not persist full per-cell control matrices; persist compact mappings/parameters/digests and regenerate controls deterministically.
2. Run one target/fold/seed end-to-end smoke through discovery, confirmation, prediction sealing, resume, and audit. It is explicitly non-promotable.
3. Run the full nested discovery and freeze winners independently for each target/protocol using inner-validation data only.
4. Run the exact `3,920`-row confirmation, valence-direction evaluation, and whole-video bootstrap. Promote only gates supported by completed executable evidence.
5. Derive fixed epochs from the completed folds and retrain/export the selected privileged and zero-label ensembles from scratch on all `124` videos. Do not report in-sample final-fit metrics as confirmation.
6. Only after VEATIC 2.1 is complete, start the domain-balanced VEATIC+AGAIN combined programme and leave-one-domain-out evaluation. Keep the locked AGAIN 299-video pool closed to tuning.

## Fast Handoff Rule

For ordinary updates, edit this file after canonical evidence is committed and validated, then mine only `docs/handoff/` into the `neural_bridge` MemPalace wing and prove it with a top-3 recall query. Never remine the full project or historical evidence tree for routine handoff.
