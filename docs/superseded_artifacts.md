# Superseded Artifact Policy

This repo keeps the current Neural Bridge baseline easy to find. Older artifacts are either removed, explicitly retained as non-authoritative context, or frozen as part of the v2 evidence bundle.

## Deleted

- Pre-v2 VEATIC transition handoffs and old acceleration audits were removed from `docs/`.
- Old smoke-test and transition scripts not needed for the consolidated strict suite were removed.
- Stale generated caches, logs, and Python bytecode are cleaned with `backend/scripts/cleanup_generated_artifacts.py`.

Reason: these files contradicted or distracted from the VEATIC-124 v2 baseline.

## Retained As Authoritative v2 Evidence

- `benchmarks/veatic/veatic_manifest_124_complete_20260616.jsonl`
- `benchmarks/veatic/veatic_manifest_124_complete_20260616.report.json`
- `benchmarks/veatic/veatic_124_*`
- `benchmarks/veatic/veatic_neuro_benchmark_124video_*`
- `outputs/veatic_124_temporal_context_v2_20260616_1557/*`
- current evidence docs listed in `docs/veatic_v2_evidence_freeze.md`

Reason: these files document or reproduce the current v2 claim.

## Retained As Non-Authoritative External Context

These may exist under the external assets root, but they are not the frozen v2 baseline:

- `benchmarks/veatic/tribe_cache_mlx`: MLX hotswap/parity archaeology.
- `benchmarks/veatic/tribe_cache_multimodal_pilot`: guarded `83,84` multimodal pilot context.
- `benchmarks/veatic/tribe_smoke`: local smoke-test residue if present.

Reason: they may help debug future work, but they must not be used as headline evidence.

## Rule For Future Cleanup

If a future artifact is not part of the current v2 bundle, a new validated baseline, or a needed local dependency, delete it or document why it is retained. Do not leave unlabeled historical runs where a fresh Codex session could mistake them for current evidence.
