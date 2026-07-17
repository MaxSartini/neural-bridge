# Neural Bridge Current-State Handoff

Updated: 2026-07-17
Branch: `codex/veatic21-retraining-foundation`

## Current Result

The newest result is the locked zero-label deployment win. The fixed direct-supervised temporal model completed `140/140` rows on the prospectively locked 299-video AGAIN pool with audit pass `true` and failed Tier 1 gates `[]`. It used no observed arousal at inference and reached `0.1785132961` Spearman, `0.0766079674` top-5% lift, and `0.1710622218` event PR-AUC. Gains over the strongest matched false-signal/no-video controls were `+77.65%`, `+70.80%`, and `+26.50%`; every full-video endpoint won `5/5` panels, every bootstrap lower bound was positive, and the first-30-second tier passed.

Phase 7 remains the observed-arousal-assisted research ceiling. Its fresh grouped continuous checkpoint ensemble passed `420/420` with failed gates `[]`, `15/15` fold-group wins, and `5/5` positive fold means versus target-specific AR and matched controls on Spearman and top-5% lift.

No VEATIC 2.1 retraining result is promoted yet. The corrected full-dense arousal-event inner-discovery tranche is complete and identifies `delta_pca64_short_conv` as the unanimous `5/5` recipe winner, but the Neural Bridge lost to matched AR by `-2.23%` (`0.3014044849` versus `0.3082785856`) and won only `10/45` paired cells. This is development evidence only; outer-test confirmation remains closed.

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

Current change validation is green. `npm run verify` passed the readiness audit with `526` controlled evidence items, Python compilation, `391` tests with one skip, the strict VEATIC benchmark dry run, and the frontend production build.

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

The authoritative corrected-depth event-first discovery completed and independently resumed/audited on 2026-07-17 at `$NEURAL_BRIDGE_EXTERNAL_ROOT/outputs/veatic21_endstate_depth_corrected_20260717`. It used run identity `b6e914aabc280a8b3f2ee1baf6d1cfbb0040ddfe9080728586898d9d0eb6ecf1`, scored exactly `270/270` full-dense inner-validation rows, used no outer-test scores, is explicitly nonpromotable, and cannot authorize confirmation. The `270` rows are five outer partitions × three inner folds × six recipes × three seeds. All `124` videos and all `20,657` 2 Hz rows participate according to grouped fold ownership; the eight-row/video synthetic executor fixture was never used for scientific training.

Keep three verdicts separate. Matrix execution/audit passed. Promotion is contractually unavailable because this is inner-only discovery. Numerically, the best Neural Bridge recipe lost to AR. `canonical_gates_passed: false` is mechanically expected because the runner only sets that field from a completed confirmation stage, which this scope deliberately did not run; it is not the source of the numerical verdict.

All `5/5` outer selections unanimously chose `delta_pca64_short_conv`. Across its `45` fold-seed cells, it scored `0.3014044849` mean PR-AUC versus matched freshly trained frozen AR `0.3082785856`: `-0.0068741007` absolute and `-2.23%` relative, paired median `-0.0059385296`, positive only `10/45`. Its mean delta was negative in all five outer panels. The other five recipes were worse. This is a clear inner-discovery loss to AR, not a tie and not a hidden contract pass.

This was Neural Bridge rather than direct raw-cortical concatenation: every lane used a fresh VEATIC AR baseline plus a separate bounded residual correction from causal PCA-compressed predicted-cortical features. No fitted AGAIN PCA, AR, head, threshold, or weight was reused. The proven AGAIN short-convolution family entered only as an architecture prior and was retrained from scratch on VEATIC folds.

The corrected schedule gave every recipe the same fair chance: batch size `1,024`, minimum checkpoint-eligible epoch `50`, patience `100`, a 5,000-epoch runaway fail-safe, true BCE-with-logits training, and checkpoint selection by held-out inner-validation PR-AUC. No run hit the ceiling. Video best epochs ranged `50`–`734`; AR ranged `50`–`945` with median `104`. Proper training improved AR from the flawed diagnostic's `0.2984364723` to `0.3082785856` (`+3.30%`), exposing the true hurdle. The exact report is `reports/veatic21_arousal_event_first_six_recipe_inner_discovery_20260717.md`.

The earlier run at `$NEURAL_BRIDGE_EXTERNAL_ROOT/outputs/veatic21_endstate_20260717` remains only a historical under-training diagnostic. Its batch-8,192/epoch-80/BCE-selection scores are superseded by the corrected run.

Repository verification after the corrected-depth integration passed on 2026-07-17: readiness passed with `526` controlled evidence items, the deterministic suite passed `391` tests with one skip, the strict-benchmark dry run passed, and the frontend production build passed.

Next actions, in order:

1. Stay on arousal event ranking and preregister a new inner-only residual-learning branch around the unanimous `delta_pca64_short_conv` representation/head. Separate cortical residual learning from correction admission: learn against frozen-AR errors, then fit a train-only bounded correction coefficient/gate that may choose zero when harmful.
2. Keep matched frozen AR as the hurdle and retain `delta_pca256_short_conv` plus the current-row PCA256 MLP as bounded comparators. Treat checkpoint averaging only as a declared stabilization candidate, never a post-hoc rescue.
3. Require a fixed method to beat AR broadly across inner fold-seed cells before transferring it. All work remains development evidence; do not open outer-test videos.
4. Transfer the proven event-stage mechanism into continuous arousal development and determine on inner folds whether the same method carries over or a separate continuous recipe is better. Do not force a shared winner.
5. Adapt the strongest proven mechanism to valence rise, valence drop, absolute movement, and derived direction. A shared or multitask candidate is allowed only as an explicit inner-discovery recipe and must beat endpoint-specific alternatives without weakening them.
6. Freeze target/protocol-specific methods, then run and report privileged outer-test confirmation with exact grouped continuous/event gates, valence direction, contribution caps, ensemble uplift, and whole-video bootstrap. Promote only completed executable evidence.
7. After privileged VEATIC is settled, complete the response-free zero-label tier, derive fixed epochs, and retrain/export selected models from scratch on all `124` videos. Do not report in-sample final-fit metrics as confirmation.
8. Only after VEATIC spike/event, continuous arousal, and valence are as strong as the locked protocol can support, start the domain-balanced VEATIC+AGAIN combined programme and leave-one-domain-out evaluation. Keep the locked AGAIN 299-video pool closed to tuning.

## Fast Handoff Rule

For ordinary updates, edit this file after canonical evidence is committed and validated, then mine only `docs/handoff/` into the `neural_bridge` MemPalace wing and prove it with a top-3 recall query. Never remine the full project or historical evidence tree for routine handoff.
