# AGAIN Phase 7: Continuous Movement Ranking

## Outcome

Phase 7 cracked a separate problem from event/spike detection: ranking the magnitude of continuous future arousal movement and concentrating the strongest true movement in the model's highest-scored moments.

## Research question

Can the event-first temporal machinery be re-specialized—rather than copied unchanged—to add continuous ranking and top-tail signal beyond a target-specific frozen AR and matched controls on unseen videos?

## Design

- Target: maximum future arousal increase from 2 to 5 seconds ahead at 2 Hz.
- Model target: train-owned residual after the separately trained frozen AR.
- Evaluation: grouped held-out-video folds, nine seeds, seven real/control lanes, and three prespecified checkpoint groups.
- Primary endpoints: Spearman ranking and top-5% true-movement lift.
- Controls: current-row video, no-video, shuffled, random, diagnostics, video mean, and label permutation as declared by lane.

## Claim-bearing grouped confirmation

| Endpoint | Neural Bridge | Frozen AR | Best matched control | Gain over AR | Positive groups |
| --- | ---: | ---: | ---: | ---: | ---: |
| Spearman | **`0.260301`** | `0.240537` | `0.240252` | `+0.019764` (**`+8.22%`**) | **`15/15`** |
| Top-5% movement lift | **`0.097598`** | `0.089566` | `0.089709` | `+0.008032` (**`+8.97%`**) | **`15/15`** |

The closure completed **`420/420`** declared comparison cells: `315` member cells plus `105` prespecified ensemble cells. These are controlled evaluation cells, not independent observations. Every held-out-video fold mean was positive and every grouped gate passed.

Checkpoint averaging added `+0.007797` Spearman and `+0.002502` top-5% lift over the member mean. Relative to the original validated continuous bridge, Phase 7 improved Spearman by **`+16.61%`**, top-5% lift by **`+23.59%`**, top-1% lift by **`+14.52%`**, and the useful top-5% margin beyond AR by **`+98.92%`**—almost exactly doubling it.

## Protocol discipline

The diagnostic and blocked-temporal branches remain separately documented because they ask different generalization questions. The blocked branch did not meet its own distinct gate; it was not used to tune, veto, or retrospectively redefine the prospectively declared grouped held-out-video claim. The grouped confirmation stands on its own completed design.

## Claim boundary

This phase confirms future-movement **ranking** and top-tail concentration on held-out AGAIN videos. It does not claim exact trajectory prediction, participant-exclusive generalization, causal identification, or cross-dataset continuous transfer.

## Audit trail

[`grouped-confirmation/`](grouped-confirmation/) contains the claim-bearing audit, manifest, compact metrics, rows, and report. [`blocked-confirmation/`](blocked-confirmation/) and earlier diagnostics preserve their separate evidence. Full checkpoints, fold matrices, curves, and predictions remain in registered external runs. The canonical engine and checksum-locked replay replace the historical phase entrypoints.

## Transition

Phase 7 proved continuous value with AR assistance. The zero-label programme then asked whether the video-derived temporal representation retained useful signal when observed arousal, response history, teacher score, and labeled warm start were all removed at inference.

[Continue to zero-label-at-inference](../zero-label/README.md) · [Return to the journey](../../README.md)
