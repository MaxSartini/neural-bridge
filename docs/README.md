# Methods and Reproducibility

## What the inputs are

TRIBE/cortical features in this repository are frozen predictions generated from video by upstream models trained on cortical-response data. They are not measurements from the viewers represented by VEATIC or AGAIN.

The AGAIN feature foundation uses the frozen [V-JEPA 2.1](https://arxiv.org/abs/2603.14482) ViT-G target encoder and [TRIBE v2](https://arxiv.org/abs/2605.04326). The primary affect sources are the [AGAIN dataset](https://doi.org/10.1109/TAFFC.2022.3188851) and [VEATIC](https://openaccess.thecvf.com/content/WACV2024/html/Ren_VEATIC_Video-Based_Emotion_and_Affect_Tracking_in_Context_Dataset_WACV_2024_paper.html). AGAIN provides first-person continuous arousal annotations; VEATIC provides continuous ratings of a selected character's perceived affect. Results are therefore reported as a cross-dataset evidence ladder, not as a single transferred model or identical label construct.

## Formal prediction target

At the 2 Hz row rate, the Phase 7 future-movement quantity is

```math
y_t = \max_{k \in \{4,\ldots,10\}} \left(a_{t+k}-a_t\right),
\qquad f_s=2\,\mathrm{Hz},
```

so the forecast window is 2–5 seconds ahead. Phase 7 predicts

```math
\widetilde{y}_t = y_t - \widehat{f}_{\mathrm{AR}}\!\left(x_t^{\mathrm{AR}}\right)
```

after fixing the autoregressive residualizer from its declared training ownership. Event labels use

```math
e_t = \mathbf{1}\!\left[T(y_t) \ge Q_q^{\mathrm{train}}\!\left(T(y)\right)\right],
```

so test labels never choose their own threshold.

- **Spearman** is rank correlation between the true and predicted continuous target.
- **Top-5% lift** is $\mathbb{E}[y_t \mid s_t \in \operatorname{Top}_{0.05}(s)]-\mathbb{E}[y_t]$ over valid held-out rows.
- **Event PR-AUC** is average precision pooled over valid held-out rows, retaining valid negatives from zero-event videos.

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

“Causal temporal” means that inputs are restricted to information available at prediction time. It does not mean the study identifies a causal effect of video content on arousal.

## Evaluation unit and uncertainty

The dense 2 Hz rows within a video are serially dependent and are not presented as independent experimental replicates.

- Phase 7 uses five outer folds grouped by `video_id`, nine fresh seeds, and three prespecified checkpoint groups. Its `420/420` count is an evaluation matrix (`315` member + `105` ensemble cells); the consistency result is `15/15` positive fold-checkpoint groups.
- The locked video-only study uses a prospectively untouched 299-video pool. Its paired uncertainty procedure resamples whole videos for `2,000` bootstrap replicates; all three one-sided 95% lower bounds for the gain over the strongest control are positive.
- Grouped-video evaluation holds out video IDs, not necessarily participant identities. No participant-exclusive or external-dataset generalization claim is made.

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
