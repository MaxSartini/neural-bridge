# AGAIN Source Migration Map

This map governs code extraction after the scientific phase packages were settled. Current files under `/Users/maxsartini/Neural Bridge/backend/scripts` outrank copies inside historical evidence bundles.

## Boundary

- `src/neural_bridge/again/` will contain the canonical AGAIN implementation.
- `src/neural_bridge/science/` stays empty until independently implemented systems prove identical semantics and provenance requirements.
- VEATIC 2.1 receives no model, head, target, PCA, threshold, window, checkpoint, or training recipe from this extraction.
- V-JEPA/TRIBE encoders and application services are feature-generation infrastructure, not part of the AGAIN modeling library.

## Current dependency spine

The retained study packages contain 46 Python entrypoints. Static and dynamic import inspection identifies 24 current intra-AGAIN dependency modules. Reuse is concentrated in the corrected Phase 5 chain:

- `run_again_dense_2hz_phase5_frozen_ar_residual.py`: used by 24 retained entrypoints;
- `run_again_dense_2hz_phase5_learned_heads.py`: used by 20;
- `run_again_dense_2hz_phase5_temporal_residual_blocked.py`: used by 17;
- `run_again_dense_2hz_phase5_redesigned_target_blocked.py`: used by 6.

`again_dense_2hz_benchmark.py` and `again_dense_2hz_phase4_pca_bridge.py` provide the dense-table, target, split, AR, metric, control, and fold-safe PCA substrate. Phase 6, Phase 7, and zero-label build on the corrected Phase 5 implementation.

## Monolith boundary

The dense benchmark imported three helpers from older sparse/encoder modules: `BOUNDARY_POLICY`, `assert_again_only_output_path`, and the exact `threshold_from_train`/`top_recall` metric helpers. Those definitions now live in AGAIN-owned modules. The sparse pipeline, encoder services, hardware runtime code, and application backend are not dependencies of the new engine.

## Canonical implementation

The two direct benchmark copies were rejected: preserving 3,335 lines of phase-coupled code would recreate the dependency problem. `src/neural_bridge/again/` now has one public engine over small modules for contracts, dense targets/splits/causal representations, metrics, frozen AR and residual models. Historical scripts are evidence, not imports.

Learned-head execution has one shared configuration and three explicit backends:

- PyTorch CPU;
- PyTorch CUDA;
- MLX on Apple silicon.

Backend, seed, checkpoint, frozen-AR digest, target threshold, lane, and control policy are recorded. CPU and MLX smoke contracts pass; CUDA uses the same PyTorch path and still requires execution on a CUDA host. Backend parity is metric-tolerance evidence, not a claim of bit-identical weights.

Pure-function golden checks against the current source match exactly for all 13 retained target/AR columns, all four redesigned 4–10-row target columns, and all four synthetic outer folds across grouped-video and blocked-temporal protocols. The evidence adapter recomputes the Phase 5 `420/420`, Phase 7 blocked `140/140`, and Phase 7 grouped `420/420` score tables. Portable NumPy replay of sealed real checkpoints matches published rows within `3.2e-8` for Phase 5, `5.4e-8` for Phase 7 blocked, and `5.9e-8` for Phase 7 grouped; no MLX runtime is required.

The linear ridge residual remains a freshly fitted sanity lane only. The proposed family is the bounded temporal residual head. Checkpoint averaging occurs only for seed groups listed in `checkpoint_ensembles` before training.

## Current boundary

Phase 1–7 no longer carries copied Python runners or `backend.scripts` imports. Compact study evidence points to the canonical engine and replay specs instead. Zero-label now has a small evidence package that recomputes the locked 140-row verdict, validates the no-label inference audits and seals, and optionally verifies the full external tree.

The five phase-coupled zero-label training files remain provenance-only in source history. They are not presented as a supported current API. No locked labels, pool membership, gates, or confirmation status changed during extraction.

The remaining code work is limited to high-value invariant, parity, closure, and backend smoke tests.

## Portable execution

`rtk` is not a project dependency. Reviewers use standard tools:

```bash
uv sync --extra training
uv run pytest
uv run python -m neural_bridge.again verify-evidence phase7-grouped --root <evidence-directory>
uv run python -m neural_bridge.again replay-checkpoint --spec <spec.json> --dense-root <dense> --pca-root <pca> --run-root <run>
```

Repository-root assumptions and `backend.scripts` imports remain forbidden. Snapshot code is never an import fallback.
