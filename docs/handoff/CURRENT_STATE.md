# Neural Bridge Current-State Handoff

Updated: 2026-07-17
Branch: `codex/veatic21-retraining-foundation`

## Current Result

The newest result is the locked zero-label deployment win. The fixed direct-supervised temporal model completed `140/140` rows on the prospectively locked 299-video AGAIN pool with audit pass `true` and failed Tier 1 gates `[]`. It used no observed arousal at inference and reached `0.1785132961` Spearman, `0.0766079674` top-5% lift, and `0.1710622218` event PR-AUC. Gains over the strongest matched false-signal/no-video controls were `+77.65%`, `+70.80%`, and `+26.50%`; every full-video endpoint won `5/5` panels, every bootstrap lower bound was positive, and the first-30-second tier passed.

Phase 7 remains the observed-arousal-assisted research ceiling. Its fresh grouped continuous checkpoint ensemble passed `420/420` with failed gates `[]`, `15/15` fold-group wins, and `5/5` positive fold means versus target-specific AR and matched controls on Spearman and top-5% lift.

No VEATIC 2.1 retraining result is promoted yet. The first bounded arousal-event inner-discovery tranche is now complete, but it is development evidence only and did not robustly crack AR. Outer-test confirmation remains closed.

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
- `reports/veatic21_arousal_event_first_six_recipe_inner_discovery_20260717.md`

Current change validation is green. `npm run verify` passed the readiness audit with `526` controlled evidence items, Python compilation, `390` tests with one skip, the strict VEATIC benchmark dry run, and the frontend production build.

Both codebase-memory projects are ready: internal repo and external SSD heavy-artifact workspace.

## Ownership And Asset Boundary

Neural Bridge is sole-founder work. The founder conceived and created the Neural Bridge ideas, target and washout design, bridge architecture, implementation, experiments, controls, training/evaluation procedures, trained heads, derived caches, evidence, reports, and product code across the VEATIC and AGAIN programmes. V-JEPA, TRIBE, and the VEATIC/AGAIN source datasets are third-party upstream dependencies or research inputs; they are replaceable, are not the Neural Bridge commercial output, and the source datasets are not shipped with the product.

Keep the two AGAIN H100 cache layers distinct:

- approximately `1 TB` in the founder's Google Drive workspace is the dense V-JEPA 2.1 cache generated from all `995` AGAIN videos;
- approximately `38.7 GiB` / `40 GB` locally is the downstream TRIBE/predicted-cortical postpass used by Neural Bridge.

Generated caches and downstream artifacts created by the project are Neural Bridge project assets, subject to applicable upstream and dataset terms. Private valuation material is not canonical project state and must not be committed or mined into MemPalace.

## VEATIC 2.1 Retraining State

The bounded H100 encoding run is complete and the Vast instance has been terminated. The authoritative compact local substrates are:

- V-JEPA 2.1: `$NEURAL_BRIDGE_EXTERNAL_ROOT/cache/veatic_h100_vjepa21_compact_20260716`, `124/124` videos, approximately `1.0 GiB`.
- TRIBE v2: `$NEURAL_BRIDGE_EXTERNAL_ROOT/cache/veatic_h100_tribe_v2_mlx_compact_20260716`, `124/124` videos, approximately `827 MiB`.
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

### Numerical executors and smoke status

The three numerical executor integrations are now implemented. `backend/scripts/veatic21_modeling.py` exposes `execute_veatic21_nested_discovery`, `execute_veatic21_confirmation_cell`, and `execute_veatic21_all124_refit`; their numerical implementation lives in `backend/scripts/veatic21_execution.py`. The end-state runner now materializes the validated dense substrate once, passes it into each executor, seals predictions and best-epoch provenance, and resumes only after checksum/identity verification. Control artifacts retain compact mappings, parameters, and digests; full per-cell control matrices are not persisted.

The full-checksum, real-cache, one-target/one-outer-fold/one-seed smoke completed on 2026-07-17 at `$NEURAL_BRIDGE_EXTERNAL_ROOT/outputs/veatic21_executor_smoke_20260717` with run identity `80c27436f592b5b49d64adb0ef4d2d24be3a79ee83a291c83dd6bfd95fa71357`:

- discovery completed exactly `54/54` measured inner-validation rows: one target × one outer fold × three protocols × six recipes × three inner folds × one discovery seed;
- confirmation completed and sealed exactly `42/42` rows with `0` missing and `0` invalid: all continuous residual, true-BCE event, zero-label, matched-control, and fixed-ensemble lanes for the smoke slice;
- a second confirmation invocation resumed without retraining, and a separate `--audit-only` invocation passed against the same matrix and best-epoch digests;
- smoke selection is stored only as `smoke_selection_artifact.json`, is marked `explicitly_nonpromotable: true`, and cannot expand a canonical matrix; no canonical `selection_artifact.json` was created;
- `--stage all --smoke` stops after discovery and confirmation. It cannot run the all-124 refit;
- the final executor separately passed bounded synthetic contract smokes for both zero-label and privileged refit paths, including fresh neural-AR stacking, checkpoint export/resume, artifact indexing, and `in_sample_metrics_reported: false`.

These smoke scores and one-epoch recipe choices are engineering checks only. They are not VEATIC performance evidence, do not freeze a canonical recipe, and do not authorize any claim update. Canonical training has not started.

### Arousal event-first inner discovery

The first real-cache event-first discovery tranche completed and independently resumed/audited on 2026-07-17 at `$NEURAL_BRIDGE_EXTERNAL_ROOT/outputs/veatic21_endstate_20260717`. It used the shared scientific run identity `38ef99d8d164930c7bf151018dff6f1205762e46147d9661ee1ec547375f93b8`, scored exactly `270/270` inner-validation rows, used no outer-test scores, is explicitly nonpromotable, and cannot authorize confirmation. Its reusable PCA and other derived artifacts live at `$NEURAL_BRIDGE_EXTERNAL_ROOT/outputs/veatic21_endstate_shared_derived_20260717`.

All `5/5` per-outer selections chose a short temporal convolution. Temporal-mean won `3/5`, delta won `2/5`, PCA-64 won `3/5`, and PCA-256 won `2/5`; neither MLP won. The global fixed leader was `temporal_mean_2s_pca64_short_conv` at `0.2979001001` mean PR-AUC versus matched frozen AR `0.2984364723`, a delta of `-0.0005363722`, positive only `19/45`. The optimistically per-outer-selected panel reached `0.2996794050`, `+0.0012429328` over AR, but was positive only `21/45` with a paired median delta effectively zero. It is selection-set performance, not a fresh validation estimate.

Training dynamics expose a protocol defect: `116/270` checkpoints selected the epoch-80 cap, `67/270` selected epoch 1, and effective residual scale stayed near initialization at roughly `0.00234`–`0.00243` against a `0.12` cap. With batch size 8,192, 80 epochs supplied only roughly 160 optimizer updates on a typical inner training set, and binary checkpoints were selected by validation BCE rather than the registered PR-AUC endpoint. The first scores are therefore a useful under-training diagnostic, not a fair final verdict on the six recipes. Do not open outer confirmation or transfer into continuous yet. The exact report is `reports/veatic21_arousal_event_first_six_recipe_inner_discovery_20260717.md`.

Repository verification after the scoped artifact integration passed on 2026-07-17: readiness passed with `526` controlled evidence items, the deterministic suite passed `390` tests with one skip, the strict-benchmark dry run passed, and the frontend production build passed.

Next actions, in order:

1. Rerun all six recipes inner-only under the corrected depth protocol: 5,000-epoch runaway-only ceiling, minimum 50 checkpoint-eligible epochs, patience 100, batch size 1,024, and binary checkpoint selection by inner-validation PR-AUC. Reuse compatible PCA/features, but retrain AR and heads under the new settings/model identity.
2. Require a fixed arousal-event method to beat matched frozen AR broadly across inner fold-seed cells before transferring it. If corrected depth still leaves the residual scale nearly closed, preregister a separate gating/optimization branch. Treat all such work as bounded development evidence—not an outer-test win.
3. Transfer the event-stage mechanism into continuous arousal development and determine on inner folds whether the same method carries over or a separate continuous recipe is better. Do not force a shared winner.
4. Adapt the strongest proven mechanism to valence rise, valence drop, absolute movement, and derived direction. A shared or multitask candidate is allowed only as an explicit inner-discovery recipe and must beat endpoint-specific alternatives without weakening them.
5. Freeze target/protocol-specific methods, then run and report privileged outer-test confirmation with exact grouped continuous/event gates, valence direction, contribution caps, ensemble uplift, and whole-video bootstrap. Promote only completed executable evidence.
6. After privileged VEATIC is settled, complete the response-free zero-label tier, derive fixed epochs, and retrain/export selected models from scratch on all `124` videos. Do not report in-sample final-fit metrics as confirmation.
7. Only after VEATIC spike/event, continuous arousal, and valence are as strong as the locked protocol can support, start the domain-balanced VEATIC+AGAIN combined programme and leave-one-domain-out evaluation. Keep the locked AGAIN 299-video pool closed to tuning.

## Fast Handoff Rule

For ordinary updates, edit this file after canonical evidence is committed and validated, then mine only `docs/handoff/` into the `neural_bridge` MemPalace wing and prove it with a top-3 recall query. Never remine the full project or historical evidence tree for routine handoff.
