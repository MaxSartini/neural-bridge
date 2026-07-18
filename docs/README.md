# Methods and Reproducibility

## What the inputs are

TRIBE/cortical features in this repository are frozen predictions generated from video by upstream models trained on cortical-response data. They are not measurements from the viewers represented by VEATIC or AGAIN.

## Evaluation rules

- Use every eligible dense 2 Hz row and retain valid negatives from zero-event videos.
- Apply the declared black/static quality mask consistently.
- Fit PCA, scalers, thresholds, AR models, and heads inside their allowed training folds only.
- Train `AR-only` separately. Reuse the exact fold/seed frozen AR unchanged in real and matched-control residual lanes.
- Pool event PR-AUC over valid rows; never invent per-video zero scores.
- Keep blocked-temporal and held-out-video evidence distinct.
- Separate exploration, candidate freeze, fresh confirmation, and outer closure.
- Treat shuffled, random, video-mean, diagnostic-only, and label-permutation lanes as scientific controls, not decoration.
- Declare checkpoint ensembles before confirmation; never select ensemble members on the locked result.

## Portable verification

The tracked closure evidence can be checked on ordinary CPU hardware:

```bash
uv sync --group dev
uv run ruff check src tests
uv run ty check
uv run pytest -q

uv run python -m neural_bridge.again verify-evidence phase5-selected \
  --root studies/again/phase-05-learned-bridge/evidence/phase_5_5_selected_head_420_confirmation_20260714_124953
uv run python -m neural_bridge.again verify-evidence phase7-blocked \
  --root studies/again/phase-07-continuous/blocked-confirmation
uv run python -m neural_bridge.again verify-evidence phase7-grouped \
  --root studies/again/phase-07-continuous/grouped-confirmation
uv run python -m neural_bridge.zero_label \
  studies/again/zero-label/locked-confirmation
```

Historical member checkpoints can also be replayed through `python -m neural_bridge.again replay-checkpoint` when the registered external dense, PCA, and run roots are available.

## Hardware support

The shared learned-head implementation supports PyTorch on CPU or CUDA and MLX on Apple silicon. Evidence verification and historical MLX checkpoint replay do not require MLX hardware. Project commands use standard `uv`, Python, and pytest; Rust Token Killer is not a project dependency.

## Heavy artifacts

Large caches, fitted PCA, checkpoints, score arrays, and model weights live outside Git under the external artifact root. [`registry/artifacts/`](../registry/artifacts/) records their canonical relative paths, file counts, sizes, and tree hashes. The ignored local `artifacts` path may point to that root.

## Claim discipline

Only prospectively declared confirmation promotes canonical claims. Grouped held-out-video and blocked-temporal protocols answer different generalization questions, so each stands on its own evidence. Exact-value, external-generalization, medical, mind-reading, and production claims require separate confirmation.
