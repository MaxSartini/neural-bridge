# Executable Validation Index

This file maps the claim-bearing Neural Bridge code to the tests, benchmark artifacts, and review evidence that validate it. It is intentionally not a smoke-test list: only deterministic contract tests, claim-bearing runners, evidence builders, and bounded diagnostics are listed as current validation surfaces.

## Best Validation First

- Full deterministic test suite: `python3 -m pytest -q tests`
- Fully provisioned result: `150 passed in 34.06s` on `2026-07-14`
- Default `npm run verify` result: `164 passed, 1 skipped in 33.60s` on `2026-07-14`
- Repo evidence/orientation audit: `npm run audit:repo`
- Latest local result: `repo_readiness pass controlled_evidence_items=206` on `2026-07-14`
- Full `npm run verify`, VEATIC frozen-evidence verification, strict-benchmark dry run, and frontend production build: pass on `2026-07-14`
- Default npm test now runs the full deterministic suite: `npm test`

## Current Claim-Bearing Runners

| Phase | Script | Role |
| --- | --- | --- |
| AGAIN deployment Stage 0 | `backend/scripts/run_again_dense_2hz_zero_label_deployment_stage0.py` | Planning-only contract freeze for the raw target, `696/299` prospective split, nested teacher ownership, video-only feature policy, and exact future `96 + 140` matrices; performs no fitting or scoring. |
| AGAIN Phase 7 grouped continuous checkpoint ensemble | `backend/scripts/run_again_dense_2hz_phase7_continuous_checkpoint_ensemble_grouped.py` | Claim-bearing fresh `420/420` grouped-video confirmation of continuous future-movement ranking/lift; all 15 fold-groups beat AR and matched controls on Spearman and top-5% lift. |
| AGAIN Phase 7 blocked continuous checkpoint ensemble | `backend/scripts/run_again_dense_2hz_phase7_continuous_checkpoint_ensemble_blocked_confirm.py` | Fresh `140/140` blocked confirmation with strong aggregate lift but a literal 4/5 Spearman-vs-AR group gate; remains an unpromoted near-pass. |
| AGAIN Phase 7 continuous checkpoint ensemble diagnostic | `backend/scripts/run_again_dense_2hz_phase7_continuous_checkpoint_ensemble_diagnostic.py` | Initial `84/84` bounded diagnostic that passed ranking/lift gates and kept exact-value gates separate. |
| AGAIN Phase 6 grouped checkpoint ensemble confirmation | `backend/scripts/run_again_dense_2hz_phase6_original_three_checkpoint_grouped_confirmation.py` | Claim-bearing fresh `420/420` grouped-video confirmation of the original three-checkpoint ensemble against fold/seed-matched AR and all controls. |
| AGAIN Phase 6 checkpoint ensemble confirmation | `backend/scripts/run_again_dense_2hz_phase6_original_three_checkpoint_control_complete.py` | Claim-bearing fresh `140/140` blocked confirmation of the original three-checkpoint ensemble against AR and all matched controls. |
| AGAIN Phase 6 checkpoint ensemble | `backend/scripts/run_again_dense_2hz_phase6_trial4_three_checkpoint_fresh15.py` | Larger fresh-15 retraining with five fixed three-checkpoint groups; Trial 4 failed against matched original ensembles. |
| AGAIN Phase 6 stabilization | `backend/scripts/run_again_dense_2hz_phase6_fixed_blend_fresh5.py` | Fixed-weight five-fresh-seed blocked pilot; failed locked gain/stability gates and stopped before controls. |
| AGAIN Phase 6 robust optimization | `backend/scripts/run_again_dense_2hz_phase6_robust_multiseed_optuna.py` | Multi-seed, inner-validation-only Stage A; no blocked/grouped held-out read. |
| AGAIN Phase 6 fresh-seed validation | `backend/scripts/run_again_dense_2hz_phase6_trial4_fresh_seed_validation.py` | Tests sensitivity-selected trial 4 on five fresh inner-validation seeds before held-out scoring. |
| AGAIN Phase 6 blocked confirmation | `backend/scripts/run_again_dense_2hz_phase6_trial4_blocked_15seed.py` | Locked 15-seed x 8-lane Stage B; failed fresh-five and dominance gates, so grouped Stage C stayed closed. |
| AGAIN Phase 6 confirmation | `backend/scripts/run_again_dense_2hz_phase6_optuna_locked_10seed_confirm.py` | Applies one checksum-pinned winner unchanged across the canonical 10 blocked seeds; exploratory and not promoted. |
| AGAIN Phase 6 diagnostic | `backend/scripts/run_again_dense_2hz_phase6_seed27_convergence_diagnostic.py` | Explicitly post-hoc convergence audit for the seed-20260627 outlier; cannot change the locked verdict. |
| AGAIN Phase 6 pilot | `backend/scripts/run_again_dense_2hz_phase6_optuna_selected_head_pilot.py` | Exploratory one-seed Optuna calibration around the exact selected target/head; MLX-required and winner locked before held-out scoring. |
| AGAIN Phase 5.5 | `backend/scripts/assemble_again_dense_2hz_phase5_selected_head_420_confirmation.py` | No-training/no-scoring assembler and fail-closed provenance audit for the full bounded 420-row selected-head confirmation. |
| AGAIN Phase 5.5 | `backend/scripts/run_again_dense_2hz_phase5_temporal_residual_binary_big_confirm.py` | 10-seed blocked binary confirmation for `future_arousal_max_delta_rows_4_10_train_q90` and `short_temporal_conv_residual`. |
| AGAIN Phase 5.5 | `backend/scripts/run_again_dense_2hz_phase5_temporal_residual_grouped_compat.py` | 5-fold x 10-seed grouped-video compatibility run for the confirmed target/head. |
| AGAIN Phase 5.5 | `backend/scripts/update_again_dense_2hz_phase5_temporal_residual_grouped_compat_verdict.py` | No-training updated verdict logic for frozen-AR residual label-permutation nulls. |
| AGAIN Phase 5.4 | `backend/scripts/run_again_dense_2hz_phase5_temporal_residual_blocked.py` | Bounded temporal/event-context residual architecture diagnostic that selected `short_temporal_conv_residual`. |
| AGAIN Phase 5.3 | `backend/scripts/run_again_dense_2hz_phase5_redesigned_target_blocked.py` | Small redesigned-target blocked diagnostic using fold-safe PCA artifacts. |
| AGAIN Phase 5.1 | `backend/scripts/run_again_dense_2hz_phase5_frozen_ar_residual.py` | Frozen-AR residual design over the original Phase 5 primary lane. |
| AGAIN Phase 5.0 | `backend/scripts/rescore_again_dense_2hz_phase5_evalmode.py` | Deterministic eval-mode rescore of the primary Phase 5 correction. |
| AGAIN Phase 5.0 | `backend/scripts/run_again_dense_2hz_phase5_adversarial_correction_fixplus.py` | Primary Phase 5 adversarial correction runner. |
| AGAIN Phase 4 | `backend/scripts/again_dense_2hz_phase4_pca_bridge.py` | Fold-safe PCA bridge benchmark builder/runner. |
| VEATIC v2 | `backend/scripts/run_veatic_strict_benchmark.py` | VEATIC strict benchmark contract and dry-run entrypoint. |
| VEATIC v2 | `backend/scripts/run_veatic_neuro_benchmark.py` | VEATIC neuro benchmark runner for cortical feature comparisons. |
| VEATIC v2 | `backend/scripts/run_veatic_temporal_context_v2.py` | VEATIC temporal-context v2 benchmark runner. |

## Deterministic Tests

The tests under `tests/` are synthetic or contract-level checks. They do not train claim-bearing models, rerun V-JEPA/TRIBE, refit PCA, or require ignored heavy output roots.

Key coverage:

- Zero-label deployment Stage 0 contracts: `test_again_zero_label_deployment_stage0.py` protects the prospective split digests, nested teacher exclusion, raw target identity, incompatible-PCA reuse ban, video-only inference block, causal row-0 cold start, rollout provenance, prediction-before-label seal, event support, and exact `96 + 140` dry-run matrices.
- Split and target leakage contracts: `test_grouped_video_split.py`, `test_again_dense_2hz_benchmark.py`, `test_again_native_temporal_alignment.py`
- AGAIN dense/Phase 4 contracts: `test_again_dense_2hz_phase4_pca_bridge.py`, `test_again_boundary_manifest.py`, `test_again_full_ar_context.py`
- AGAIN selected-head confirmation contracts: `test_again_selected_head_420_confirmation.py` protects the exact 420-key matrix, lane normalization, semantic controls, frozen checksum policy, and overall gate composition.
- AGAIN Optuna pilot contracts: `test_again_phase6_optuna_selected_head_pilot.py` protects fixed scope, held-out exclusion, baseline enqueueing, dry-run schema, and the follow-up gate.
- AGAIN locked Optuna confirmation contracts: `test_again_phase6_optuna_locked_10seed_confirm.py` protects checksum pins, exact 10 x 7 scope, preregistered gates, canonical reuse, and the post-hoc diagnostic boundary.
- AGAIN robust Optuna contracts: `test_again_phase6_robust_multiseed_optuna.py` protects held-out exclusion, fresh-seed separation, MLX enforcement, explicit retention of seed `20260627`, Stage B gates, and fail-closed Stage C authorization.
- AGAIN fixed-blend contracts: `test_again_phase6_fixed_blend_fresh5.py` protects untouched seeds, literal 50/50 weights, no viewed-score reuse or weight search, locked stability gates, and fail-closed follow-up authorization.
- AGAIN checkpoint-ensemble contracts: `test_again_phase6_trial4_three_checkpoint_fresh15.py` protects 15 untouched seeds, disjoint fixed groups, exactly three aligned members, no member selection/weight search, and fail-closed audits.
- AGAIN control-complete ensemble contracts: `test_again_phase6_original_three_checkpoint_control_complete.py` protects untouched seeds, fixed groups, full control scope, exact 140 rows, and fail-closed grouped authorization.
- AGAIN grouped ensemble contracts: `test_again_phase6_original_three_checkpoint_grouped_confirmation.py` protects the five-fold, nine-seed, three-group, seven-lane, exact 420-row grouped scope and full matched-control set.
- AGAIN Phase 7 continuous diagnostic contracts: `test_again_phase7_continuous_checkpoint_ensemble_diagnostic.py` protects the target-specific continuous AR baseline, fixed checkpoint groups, MLX enforcement, and separate ranking/lift versus exact-value gates.
- AGAIN Phase 7 blocked continuous contracts: `test_again_phase7_continuous_checkpoint_ensemble_blocked_confirm.py` protects untouched seeds, exact `140`-row scope, full controls, and the fail-closed 5/5 blocked Spearman gate.
- AGAIN Phase 7 grouped continuous contracts: `test_again_phase7_continuous_checkpoint_ensemble_grouped.py` protects the five-fold, nine-seed, exact `420`-row scope, audited fold-safe PCA reuse, all matched controls, and the exact-value claim boundary.
- VEATIC v2 benchmark contracts: `test_veatic_strict_benchmark_contract.py`, `test_veatic_raw_representation_contract.py`
- Frozen tensor/trained-head contracts: `test_veatic_frozen_tensor_adapter.py`, `test_veatic_frozen_tensor_trained_heads.py`
- Runtime/cache contracts: `test_veatic_tribe_cache_runtime.py`, `test_mlx_vjepa21_cortical.py`
- Sparse/scout historical pipeline contracts: `test_again_scout_sparse_pipeline.py`, `test_again_sparse_tribe_teacher_500.py`, `test_again_real_scout_selector_validation.py`

## Benchmark Artifacts

- `benchmarks/veatic/` is a tracked VEATIC v2 benchmark evidence mirror.
- Current AGAIN benchmark evidence lives in `reports/`, `evidence/phase_*`, and the Phase 7-first `evidence/current_phase_7_review/`; `evidence/current_phase_5_5_review/` remains the detailed historical dossier.
- Current consolidated selected-head evidence: `evidence/phase_5_5_selected_head_420_confirmation_20260714_124953/`.
- Exploratory Optuna pilot evidence: `evidence/phase_6_optuna_selected_head_pilot_20260714_135902/`.
- Exploratory locked-winner confirmation evidence: `evidence/phase_6_optuna_locked_10seed_confirmation_20260714_141457/`.
- Exploratory robust multi-seed Stage A evidence: `evidence/phase_6_robust_multiseed_optuna_20260714_143646/`.
- Exploratory fresh-seed Stage A2 evidence: `evidence/phase_6_trial4_fresh_seed_validation_20260714_145120/`.
- Exploratory blocked Stage B evidence: `evidence/phase_6_trial4_blocked_15seed_20260714_145637/`.
- Exploratory fixed-blend fresh-five evidence: `evidence/phase_6_fixed_blend_fresh5_20260714_153441/`.
- Exploratory three-checkpoint fresh-15 evidence: `evidence/phase_6_trial4_three_checkpoint_fresh15_20260714_154602/`.
- Promoted blocked three-checkpoint evidence: `evidence/phase_6_original_three_checkpoint_control_complete_20260714_160001/`.
- Promoted grouped three-checkpoint evidence: `evidence/phase_6_original_three_checkpoint_grouped_confirmation_20260714_163024/`.
- Phase 7 continuous diagnostic report: `reports/again_dense_2hz_phase7_continuous_checkpoint_ensemble_diagnostic_20260714_174513.md`.
- Phase 7 blocked continuous near-pass report: `reports/again_dense_2hz_phase7_continuous_checkpoint_ensemble_blocked_confirm_20260714_175653.md`.
- Promoted Phase 7 grouped continuous evidence: `evidence/phase_7_continuous_checkpoint_ensemble_grouped_20260714_181440/`.
- Zero-label deployment Stage 0 contract snapshot: `evidence/zero_label_video_only_deployment_stage0_20260714/`.
- Heavy output roots under `outputs/`, dense cache files, checkpoints, tensors, `.npy`, `.npz`, and model assets remain outside git unless explicitly documented as tiny metadata.

## Non-Claim Runtime Probes

Runtime probes under `tools/` are for environment, encoder, and throughput checks. They are not claim-bearing benchmark suites and should not be cited as evidence for Neural Bridge performance. The one-video V-JEPA 2.1 real-video probe is now named as a runtime probe, not as benchmark evidence.

## Reviewer Entry Points

- Current reviewer entrypoint: `evidence/current_phase_7_review/README.md`
- Historical reviewer executable manifest: `evidence/current_phase_5_5_review/14_executable_validation_and_code/executable_validation_manifest.csv`
- Machine-readable executable manifest: `docs/executable_validation_manifest.json`
- Current validation result: `docs/test_suite_result_20260714.json`
- Frozen reviewer-dossier test record: `evidence/current_phase_5_5_review/14_executable_validation_and_code/test_suite_result_20260630.json`
