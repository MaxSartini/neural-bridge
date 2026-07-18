# AGAIN Zero-Label-at-Inference Closure

## Outcome

The zero-label programme established that substantial event and continuous signal survives when held-out inference receives video-derived features and causal metadata—but no observed arousal, response history, teacher score, or labeled warm start.

“Zero-label” describes inference inputs. Training and model selection remain supervised.

## Research question

After the AR-assisted event and continuous wins, can a video-only temporal bridge retain useful future-response structure under a prospectively locked confirmation design?

## Development and selection

| Stage | Question | Outcome |
| --- | --- | --- |
| Stage 0 | Can distillation or self-rollout remove label/history dependence without becoming a false-signal model? | both approaches were eliminated by matched control testing |
| Stage A | Does direct-supervised temporal video learning beat no-video controls? | passed all three endpoints in `3/3` development folds |
| Prospective freeze | Which system enters confirmation? | architecture, inputs, training recipe, controls, and gates frozen before the locked pool opened |
| Locked confirmation | Does the frozen system pass untouched evidence? | all three Tier-1 endpoints passed on 299 videos |

Selection and training used `696` development videos. Only after that work was complete were the chosen system and evaluation contract frozen. The untouched `299`-video confirmation pool was then opened for the first and only locked evaluation.

## Locked result

| Endpoint | Neural Bridge | Strongest control | Absolute gain | Relative gain | Panel wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| Spearman | **`0.178513`** | `0.100488` | `+0.078025` | **`+77.65%`** | **`5/5`** |
| Top-5% movement lift | **`0.076608`** | `0.044852` | `+0.031756` | **`+70.80%`** | **`5/5`** |
| Event PR-AUC | **`0.171062`** | `0.135230` | `+0.035832` | **`+26.50%`** | **`5/5`** |

Using `2,000` whole-video bootstrap replicates, the one-sided 95% lower bounds for the paired gains were `+0.060679` Spearman, `+0.018774` top-5% lift, and `+0.023546` event PR-AUC. All were above zero. The first-30-second cold-start tier also passed.

## Why the rejected branches matter

Distillation and self-rollout were plausible routes to label-free inference, but their apparent value did not survive the control structure. Direct temporal learning earned promotion by beating no-video alternatives across all development folds; it was not chosen after inspecting the locked pool.

## Claim boundary

This is supervised learning with zero-label **inference**. It supports held-out AGAIN video ranking, top-tail lift, and event detection. It does not establish unsupervised training, individual-level neural inference, exact trajectory prediction, or arbitrary-video production validity.

## Audit and verification

[`stage-0/`](stage-0/), [`stage-a/`](stage-a/), and [`locked-confirmation/`](locked-confirmation/) contain compact closure evidence. Full folds, predictions, fitted PCA, and models remain in registered external runs. Verify the closure on CPU, CUDA, or MLX without reopening labels or retraining:

```bash
uv run python -m neural_bridge.zero_label studies/again/zero-label/locked-confirmation
```

Add `--external-root` and `--registry` to verify the complete registered artifact tree.

[Return to the complete study journey](../../README.md) · [Open the concluded scorecard](../../../results/README.md)
