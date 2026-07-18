# AGAIN Zero-Label Closure

This study asks whether video-only features can predict future arousal movement without labels or AR history at inference.

The development sequence is retained because it changed the method: distillation and self-rollout failed; direct-supervised temporal video features became the locked candidate. Stage A did not promote a deployment claim, but the direct temporal lane beat no-video controls on all three endpoints in `3/3` folds.

Fresh locked confirmation then passed every Tier-1 gate. Against the strongest matched control, the video-only temporal candidate reached:

- event PR-AUC `0.1711`, delta `+0.0358`, panel wins `5/5`;
- Spearman `0.1785`, delta `+0.0780`, panel wins `5/5`;
- top-5% lift `0.0766`, delta `+0.0318`, panel wins `5/5`.

`stage-0/`, `stage-a/`, and `locked-confirmation/` contain compact closure evidence. Full folds, predictions, fitted PCA, and models remain in the registered external runs. The current repository intentionally does not expose the old phase-coupled training runners as a live API.

Verify the locked result on CPU, CUDA, or MLX hosts without reopening labels or retraining:

```bash
uv run python -m neural_bridge.zero_label studies/again/zero-label/locked-confirmation
```

Add `--external-root` and `--registry` to verify the complete registered artifact tree as well.
