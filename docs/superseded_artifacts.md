# Superseded Artifact Policy

This repo keeps the current Neural Bridge baseline easy to find. Older artifacts are either removed, explicitly retained as non-authoritative context, or frozen as part of the v2 evidence bundle.

## Deleted

- Pre-v2 VEATIC transition handoffs and old acceleration audits were removed from `docs/`.
- Old smoke-test and transition scripts not needed for the consolidated strict suite were removed.
- Stale generated caches, logs, and Python bytecode are cleaned with `backend/scripts/cleanup_generated_artifacts.py`.
- `reports/current_artifact_port_audit_20260617.md` was removed after its useful asset-boundary notes were absorbed into `docs/external_assets_manifest.md` and this policy.

Reason: these files contradicted or distracted from the VEATIC-124 v2 baseline.

## Retained As Authoritative v2 Evidence

- `benchmarks/veatic/veatic_manifest_124_complete_20260616.jsonl`
- `benchmarks/veatic/veatic_manifest_124_complete_20260616.report.json`
- `benchmarks/veatic/veatic_124_*`
- `benchmarks/veatic/veatic_neuro_benchmark_124video_*`
- `outputs/veatic_124_temporal_context_v2_20260616_1557/*`
- current evidence docs listed in `docs/veatic_v2_evidence_freeze.md`

Reason: these files document or reproduce the current v2 claim.

## Retained As Current Post-v2 Tensor Contract

- External: `${NEURAL_BRIDGE_EXTERNAL_ROOT}/tensors/veatic_124_raw_representation_v1`
- Tracked: `outputs/veatic_124_raw_representation_tensor_export_v1/*`
- Code: `tools/export_veatic_raw_representation_tensors.py`
- Report: `docs/veatic_raw_representation_audit.md`

Reason: these files freeze the next model-head input contract without replacing the v2 evidence baseline. The heavy `.npy` tensors remain external; lightweight summaries, manifests, and samples are safe to retain in git.

## Retained As Current Post-v2 Implementations

- Code: `backend/scripts/veatic_frozen_tensor_adapter.py`
- Code: `backend/scripts/veatic_frozen_tensor_trained_heads.py`
- Code: `backend/scripts/run_veatic_frozen_tensor_trained_heads_benchmark.py`
- Code: `backend/app/services/mlx_vjepa21_cortical.py`
- Code: `backend/scripts/again_*`
- Code: `tools/run_again_*`, `tools/probe_*vjepa21*`, and `tools/audit_again_*`
- Reports: `reports/again_*20260622_005732.md`, `reports/again_sparse_tribe_teacher_500_*_20260622_pca_width_reanalysis_v2.md`, and `reports/again_full_ar_context_20260622_005713.md`

Reason: these are current source and lightweight summaries for the implemented trained-head and AGAIN/V-JEPA 2.1 scaling paths. They are not stale transition artifacts.

## Retained As Non-Authoritative External Context

These may exist under the external assets root, but they are not the frozen v2 baseline:

- `benchmarks/veatic/tribe_cache_mlx`: MLX hotswap/parity archaeology.
- `benchmarks/veatic/tribe_cache_multimodal_pilot`: guarded `83,84` multimodal pilot context.
- `benchmarks/veatic/tribe_smoke`: local smoke-test residue if present.
- older timestamped AGAIN pilot reports not tracked in git after the `20260622_005732` and `20260622_pca_width_reanalysis_v2` summaries were retained.

Reason: they may help debug future work, but they must not be used as headline evidence.

## Rule For Future Cleanup

If a future artifact is not part of the current v2 bundle, a new validated baseline, or a needed local dependency, delete it or document why it is retained. Do not leave unlabeled historical runs where a fresh Codex session could mistake them for current evidence.
