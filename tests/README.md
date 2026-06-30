# Tests

The root deterministic test suite is the real validation suite for code contracts in this repository:

```bash
python3 -m pytest -q tests
```

Latest local run on `2026-06-30`: `93 passed in 6.42s`.

## Scope

These tests are contract and unit tests over synthetic/minimal fixtures. They do not train claim-bearing models, rerun V-JEPA/TRIBE, refit PCA, mutate dense caches, or read full ignored output roots.

## Coverage

- `test_again_dense_2hz_benchmark.py` - dense 2 Hz target construction, split contracts, AR thresholds, lane evaluation, and control guardrails.
- `test_again_dense_2hz_phase4_pca_bridge.py` - Phase 4 fold-safe PCA bridge contracts and required output metadata.
- `test_again_native_temporal_alignment.py` - native time-grid alignment, second-based future windows, and feasibility boundaries.
- `test_grouped_video_split.py` - grouped-video fold disjointness.
- `test_veatic_strict_benchmark_contract.py` - VEATIC v2 strict benchmark plan and modality/timing contracts.
- `test_veatic_raw_representation_contract.py` - raw representation audit plans, tensor contracts, leakage checks, and stable job keys.
- `test_veatic_frozen_tensor_adapter.py` - frozen tensor contract loading, target metadata, and no-future-feature rules.
- `test_veatic_frozen_tensor_trained_heads.py` - trained-head wrapper and lane contracts over frozen tensors.
- `test_mlx_vjepa21_cortical.py` - MLX V-JEPA 2.1 / TRIBE adapter configuration and preprocessing contracts.
- `test_veatic_tribe_cache_runtime.py` - cache identity, claim locking, and worker restart contracts.
- `test_again_boundary_manifest.py` - AGAIN annotation-boundary policy.
- `test_again_full_ar_context.py` - AR-only context benchmark guardrails.
- `test_again_scout_sparse_pipeline.py` - historical sparse scout/teacher queue contracts.
- `test_again_sparse_tribe_teacher_500.py` - sparse-teacher queue, cache, PCA, and runtime validation contracts.
- `test_again_real_scout_selector_validation.py` - selector-label validation and non-oracle controls.

## Policy

Do not add low-value placeholder checks. New tests should protect a real split, target, leakage, control, manifest, scorer, checkpoint, or claim-boundary contract.
