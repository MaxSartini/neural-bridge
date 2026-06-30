# Executable Validation Index

This file maps the claim-bearing Neural Bridge code to the tests, benchmark artifacts, and review evidence that validate it. It is intentionally not a smoke-test list: only deterministic contract tests, claim-bearing runners, evidence builders, and bounded diagnostics are listed as current validation surfaces.

## Best Validation First

- Full deterministic test suite: `python3 -m pytest -q tests`
- Latest local result: `93 passed in 6.42s` on `2026-06-30`
- Repo evidence/orientation audit: `npm run audit:repo`
- Latest local result: `repo_readiness pass controlled_evidence_items=206` on `2026-06-30`
- Default npm test now runs the full deterministic suite: `npm test`

## Current Claim-Bearing Runners

| Phase | Script | Role |
| --- | --- | --- |
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

- Split and target leakage contracts: `test_grouped_video_split.py`, `test_again_dense_2hz_benchmark.py`, `test_again_native_temporal_alignment.py`
- AGAIN dense/Phase 4 contracts: `test_again_dense_2hz_phase4_pca_bridge.py`, `test_again_boundary_manifest.py`, `test_again_full_ar_context.py`
- VEATIC v2 benchmark contracts: `test_veatic_strict_benchmark_contract.py`, `test_veatic_raw_representation_contract.py`
- Frozen tensor/trained-head contracts: `test_veatic_frozen_tensor_adapter.py`, `test_veatic_frozen_tensor_trained_heads.py`
- Runtime/cache contracts: `test_veatic_tribe_cache_runtime.py`, `test_mlx_vjepa21_cortical.py`
- Sparse/scout historical pipeline contracts: `test_again_scout_sparse_pipeline.py`, `test_again_sparse_tribe_teacher_500.py`, `test_again_real_scout_selector_validation.py`

## Benchmark Artifacts

- `benchmarks/veatic/` is a tracked VEATIC v2 benchmark evidence mirror.
- Current AGAIN benchmark evidence lives in `reports/`, `evidence/phase_*`, and `evidence/current_phase_5_5_review/`.
- Heavy output roots under `outputs/`, dense cache files, checkpoints, tensors, `.npy`, `.npz`, and model assets remain outside git unless explicitly documented as tiny metadata.

## Non-Claim Runtime Probes

Runtime probes under `tools/` are for environment, encoder, and throughput checks. They are not claim-bearing benchmark suites and should not be cited as evidence for Neural Bridge performance. The one-video V-JEPA 2.1 real-video probe is now named as a runtime probe, not as benchmark evidence.

## Reviewer Entry Points

- Reviewer executable manifest: `evidence/current_phase_5_5_review/14_executable_validation_and_code/executable_validation_manifest.csv`
- Machine-readable executable manifest: `docs/executable_validation_manifest.json`
- Test result record: `evidence/current_phase_5_5_review/14_executable_validation_and_code/test_suite_result_20260630.json`
