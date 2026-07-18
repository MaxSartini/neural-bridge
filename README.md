# Neural Bridge

## Predict the moments that will move an audience—before the video ships.

**Neural Bridge turns video into a forward-looking map of likely human-arousal movement: the peaks, weak moments, and sections most worth testing.**

| Proof point | Result | Why it matters |
| --- | --- | --- |
| Works without response labels at inference | **299 untouched videos**, `5/5` panels, gains of **`+77.65%`**, **`+70.80%`**, and **`+26.50%`** over the strongest controls | the bridge can operate from video-derived features alone on held-out inference |
| Continuous response ranking is confirmed | **`420/420`** rows and **`15/15`** positive fold-groups versus strong AR and matched controls | the result is repeatable across unseen videos, seeds, and checkpoint groups |
| Raw neuro-response features became useful | grouped event PR-AUC `0.136579` → **`0.238341`** | **`+74.51%`** on the same target; the value comes from the bridge, not raw TRIBE output |
| Evidence spans two domains | original VEATIC + large-scale AGAIN | the event signal appears in edited affective video and gameplay, not one convenient dataset |

> **The short version:** raw predicted cortical features initially lost badly to a strong persistence model. Neural Bridge turned them into signal that beat persistence and false-signal controls—then preserved substantial signal with response history removed entirely from inference.

[See the strongest concluded evidence →](results/README.md)

### What the metrics mean—in 30 seconds

| Metric | Plain-English question | Better means |
| --- | --- | --- |
| **Spearman** | Did the model put the moments with larger true future movement ahead of the smaller ones? | closer to `1`; ordering is more correct |
| **Top-5% lift** | Do the moments the model ranks in its top 5% actually contain more true future movement? | more real movement is concentrated in the predicted peaks |
| **PR-AUC** | Can the model find rare future response events without being rewarded for guessing the common non-event class? | rare events are ranked more precisely and completely |

### What the controls mean—in 30 seconds

| Control | What it tests |
| --- | --- |
| **Frozen AR** | a strong learned persistence model using recent arousal; the exact same frozen AR sits underneath real and control residual lanes |
| **Current-row video** | whether the current frame alone explains the result, without causal temporal history |
| **No-video** | whether masks and time metadata can produce the apparent win without video content |
| **Shuffled / random features** | whether realistic-looking but misaligned or random features work just as well |
| **Diagnostics / video mean** | whether generic quality, motion, brightness, time, or static video identity explains the signal |
| **Label permutation** | whether the model still “wins” after the true training relationship is deliberately broken |

Neural Bridge is claim-bearing only when the real lane beats the appropriate strong baseline **and** these matched alternative explanations.

### How the system fits together

```mermaid
flowchart LR
    A["Raw video"] --> B["Frozen video-to-cortical model"]
    B --> C["Predicted cortical response"]
    C --> D["Neural Bridge"]
    D --> E["Future response heat map"]
    E --> F["Peaks · weak moments · comparisons"]
```

Today, you usually learn how a video affects people **after** they watch it: panels, surveys, biometrics, expensive studies, and slow feedback. Neural Bridge is building a path toward useful response intelligence before that process is complete.

This is not a generic engagement score and it is not a prettier wrapper around a video embedding. Neural Bridge was built because the raw predicted neuro-response features were **not useful enough on their own**.

The breakthrough is the bridge that made them useful.

## The headline result

The video-only candidate was trained on 696 development videos, frozen, and then evaluated once on a prospectively locked pool of 299 videos.

At inference it received:

- cached features generated from the video;
- causal video timing and quality metadata;
- **no observed arousal**;
- **no response history**;
- **no teacher score**; and
- **no labeled warm start**.

| Locked 299-video endpoint | Neural Bridge | Strongest false-signal/no-video control | Gain | Frozen-panel wins |
| --- | ---: | ---: | ---: | ---: |
| Future-movement ranking | `0.178513` | `0.100488` | **`+77.65%`** | **`5/5`** |
| Top-5% true-movement lift | `0.076608` | `0.044852` | **`+70.80%`** | **`5/5`** |
| Future-event PR-AUC | `0.171062` | `0.135230` | **`+26.50%`** | **`5/5`** |

All three paired whole-video bootstrap lower bounds were positive. The first-30-second cold-start tier passed. The temporal model beat current-frame video, diagnostics-only, no-video, sequence-shuffled, and hard-label-permutation controls.

**Plain English:** the signal was not just “this video tends to be exciting.” The model learned useful moment-by-moment temporal structure that survived when response labels were removed from held-out inference.

[See the locked evidence, controls, and exact boundaries →](studies/again/zero-label/evidence-summary.md)

## Why this is a real breakthrough

The upstream system produces a very rich predicted cortical/fMRI representation from video. Rich does not automatically mean useful.

On the early blocked AGAIN benchmark:

| Starting point | PR-AUC | Compared with trained arousal persistence |
| --- | ---: | ---: |
| Strong trained AR baseline | `0.203622` | baseline |
| Raw predicted cortical features | `0.124315` | **`-38.95%`** |
| AR + raw cortical features | `0.167731` | **`-17.63%`** |

The raw features did not merely lose. **Naïvely adding them made a strong predictor worse.**

Neural Bridge reversed that.

On the original same-target grouped event benchmark, the system progressed from:

| System generation | PR-AUC | Change from raw cortical |
| --- | ---: | ---: |
| Raw cortical only | `0.136579` | baseline |
| Trained AR only | `0.147251` | `+0.010672` (`+7.81%`) |
| Direct AR + raw cortical | `0.170299` | `+0.033720` (`+24.69%`) |
| Fold-safe PCA bridge | `0.171648` | `+0.035069` (`+25.68%`) |
| Deterministic learned bridge | `0.230064` | `+0.093485` (**`+68.45%`**) |
| Frozen-AR residual bridge | **`0.238341`** | `+0.101762` (**`+74.51%`**) |

That final same-target bridge is:

- **`+74.51%` above raw cortical**;
- **`+39.95%` above direct AR + raw fusion**; and
- **`+38.85%` above the earlier PCA bridge**.

This is what Neural Bridge contributes. Not “we used a neuroscience model.” Not “we added more features.” It is the scientifically controlled machinery that converts a difficult, high-dimensional predicted neuro-response representation into repeatable forward-looking signal.

## From event spikes to continuous response intelligence

The programme deliberately solved the hard problem in stages.

First: **can the system find rare future response spikes?**

Then: **can it rank continuous future movement, not only binary events?**

Finally: **does useful signal survive when response history disappears at inference?**

Phase 7 answered the continuous question on grouped held-out videos:

| Phase 7 endpoint | Neural Bridge | Frozen AR | Gain over AR | Best matched control | Fold-group wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| Future-movement Spearman | **`0.260301`** | `0.240537` | `+0.019764` (**`+8.22%`**) | `0.240252` | **`15/15`** |
| Top-5% movement lift | **`0.097598`** | `0.089566` | `+0.008032` (**`+8.97%`**) | `0.089709` | **`15/15`** |

The full confirmation completed `420/420` declared rows. Every held-out-video fold mean was positive. Every preregistered grouped gate passed. Checkpoint averaging itself added `+0.007797` Spearman and `+0.002502` top-5% lift over the member mean.

Compared with the original validated continuous bridge, Phase 7 improved:

- Spearman by **`+16.61%`**;
- top-5% lift by **`+23.59%`**;
- top-1% lift by **`+14.52%`**; and
- the useful top-5% margin beyond AR by **`+98.92%`**—almost exactly double.

The claim-bearing Phase 7 result is grouped held-out-video future-movement ranking and lift: unseen videos, `420/420` declared rows, and `15/15` positive fold-groups. Blocked-temporal and grouped-video protocols remain documented separately because they test different forms of generalization.

[Open the full Phase 7 evidence →](studies/again/phase-07-continuous/evidence-summary.md)

## What does a “future response heat map” mean?

Imagine a 90-second trailer.

Neural Bridge is not trying to claim that a future viewer’s arousal will equal exactly `0.617` at second 43. That would be false precision.

It is trying to answer the questions that actually matter:

- Which upcoming moments are likely to produce the strongest response movement?
- Are the true peaks concentrated where the model says they are?
- Where does the experience lose momentum?
- Does one edit create stronger likely response moments than another?
- Which sections deserve human testing first?

That is why ranking, top-tail lift, and rare-event PR-AUC matter. They measure whether the system finds and prioritizes the moments worth attention.

## Why the baseline is so hard to beat

Human arousal is persistent. If arousal is high now, it often remains high a moment later. A model using recent response history can therefore look surprisingly good without understanding the video at all.

Neural Bridge does not compare itself with a toy baseline.

- **AR-only** is trained for the exact target, split, fold, and seed.
- **Frozen AR** is reused unchanged under the real residual and every matched control.
- **Shuffled and random controls** preserve realistic feature shapes while destroying real alignment.
- **Video-mean and diagnostics-only controls** test whether static identity or generic video statistics explain the result.
- **Label permutation** tests whether the training relationship is real.

Phase 7 beat the AR floor and the strongest matched control in every fold-group. The locked video-only candidate beat every false-signal/no-video family on every aggregate endpoint.

That is why an `8.22%` gain over AR is more meaningful than it first appears: AR already owns most of the easy persistence signal. Neural Bridge isolates the smaller, harder, genuinely forward-looking component.

## Evidence across two very different worlds

The effect did not originate in one convenient gaming dataset.

### Original VEATIC

`124` edited affective videos spanning film, documentary, reality TV, and home video.

- strongest blocked future-event PR-AUC: **`0.2536`**;
- trained AR: `0.1969` — Neural Bridge is `+0.0567` (**`+28.80%`**);
- shuffled: `0.1840` — Neural Bridge is `+0.0696` (**`+37.83%`**);
- random: `0.1944` — Neural Bridge is `+0.0592` (**`+30.45%`**); and
- balanced event-vs-stable PR-AUC: **`0.3394`**.

VEATIC established the event signal and the importance of short causal temporal context. Its confirmed scope was future-event ranking; continuous specialization was solved later on AGAIN with richer data and stronger controls.

### AGAIN

`995` cleaned gameplay videos, `243,575` aligned 2 Hz rows, nine games, three genres, and more than 37 hours of annotated material.

AGAIN introduced stronger AR controls, fold-safe representations, residual heads, strict time separation, checkpoint stabilization, held-out-video continuous confirmation, and the locked video-only study.

Together, the two datasets support a real cross-domain event-ranking story. The continuous and zero-label results remain AGAIN-specific until independently confirmed elsewhere.

[See the full scientific journey and decisive design lessons →](studies/README.md)

## For investors and product partners: the commercial thesis

**Neural Bridge is the intelligence layer between a video and the expensive process of learning how people respond to it.**

The product wedge is direct: upload a video and receive a response heat map showing likely peaks, weak moments, and the segments most worth revising, comparing, or validating with people. The aim is not to replace human testing. It is to make that testing faster, better targeted, and more valuable.

| Investment question | Evidence-backed answer |
| --- | --- |
| **What was technically unlocked?** | Raw cortical predictions scored `0.136579` event PR-AUC; Neural Bridge raised the same-target result to **`0.238341` (`+74.51%`)**. The conversion layer—not access to a fashionable embedding—is the core invention. |
| **Can it work before audience-response data exists?** | The frozen video-only system passed on **299 untouched videos**, beating the strongest controls by **`+77.65%`** in ranking, **`+70.80%`** in top-5% lift, and **`+26.50%`** in event PR-AUC. |
| **Is this a one-benchmark trick?** | Event signal was established on edited affective VEATIC video and then rebuilt at much greater scale on AGAIN. AGAIN additionally confirmed continuous ranking and zero-label-at-inference operation. |
| **Where is the defensibility?** | In the accumulated data contracts, causal temporal representations, target-specific baselines, matched controls, fold-safe training, and evidence system that repeatedly turned difficult raw features into validated signal. |
| **What unlocks commercial deployment?** | End-to-end raw-video execution, latency and cost work, external transfer, calibration, and prospective customer studies—the next proof points, not substitutes for the research already completed. |

The opportunity is a scalable **response-intelligence layer** for trailers, advertising, entertainment, games, and other video workflows: heat maps, edit comparisons, peak detection, cold-start diagnostics, confidence estimates, and intelligent prioritization of costly audience testing.

[Inspect the concluded scorecard and exact values →](results/README.md)

## For professors and scientific reviewers: the actual claim

The scientifically interesting result is not that a large video representation correlates with arousal. **Raw predicted cortical features were initially weaker than a strong autoregressive model. Neural Bridge extracted additional future-response signal and kept beating matched alternative explanations.**

| Review question | Claim-bearing evidence |
| --- | --- |
| **Does the bridge add signal beyond response persistence?** | Phase 7 grouped Spearman rose from frozen AR `0.240537` to **`0.260301` (`+8.22%`)**; top-5% lift rose from `0.089566` to **`0.097598` (`+8.97%`)**. |
| **Does that hold across unseen videos and retraining variation?** | Yes: the declared matrix completed **`420/420`** rows and the bridge was positive in **`15/15`** held-out-video fold-groups. |
| **Does useful signal survive without observed arousal at inference?** | Yes: one frozen candidate passed once on 299 locked videos with no observed arousal, response history, teacher score, or labeled warm start at inference. Training was supervised. |
| **Could static video identity, timing, quality, or accidental alignment explain it?** | The real lane beat current-row video, no-video, diagnostics-only, shuffled, random, video-mean, and label-permutation controls under their declared comparisons. |
| **Can the claims be audited rather than merely trusted?** | Compact closures recompute from tracked CSV/JSON; representative checkpoints replay published rows; large artifacts are hash-registered; the phase record preserves every decisive selection and control result. |

The evaluation discipline is deliberately strict:

- target-specific AR is trained separately, then frozen identically beneath real and residual-control lanes;
- PCA, scalers, thresholds, AR, and heads are fitted only within their declared fold ownership;
- blocked-temporal and held-out-video protocols remain separate because they test different generalization questions;
- candidates and any checkpoint ensembles are declared before confirmation; and
- discovery, candidate selection, confirmation, and locked closure cannot silently exchange roles.

[Read the methods and reproduce the evidence →](docs/README.md) · [Audit the complete study journey →](studies/README.md)

## Honest boundaries

Neural Bridge does **not** currently claim:

- mind reading;
- individual profiling;
- medical or diagnostic inference;
- universal emotion recognition;
- exact second-by-second future trajectories;
- guaranteed audience or commercial outcomes; or
- production validity from arbitrary raw client video.

The locked result is zero-label **at inference**, not label-free training. The upstream features are video-generated predictions, not direct neural recordings. Phase 7’s confirmed claim is grouped held-out-video ranking and top-tail lift on AGAIN.

Those boundaries make the result defensible. They do not make it small.

## Evidence map

| Go directly to | What is there |
| --- | --- |
| [Concluded results](results/README.md) | the compact scorecard with actual values and percentage gains |
| [Methods and reproducibility](docs/README.md) | data ownership, controls, fitting rules, verification, and hardware support |
| [Complete study journey](studies/README.md) | the evidence chain from dense data foundation through locked zero-label confirmation |
| [Phase 7 grouped closure](studies/again/phase-07-continuous/grouped-confirmation/) | the `420/420`, `15/15` continuous-ranking report and machine evidence |
| [Locked zero-label closure](studies/again/zero-label/locked-confirmation/) | the prospectively locked 299-video report, audits, controls, and machine verdict |
| [`src/neural_bridge/`](src/neural_bridge/) | the single current CPU/CUDA/MLX-capable implementation |
| [`registry/artifacts/`](registry/artifacts/) | hashes and provenance for heavy external artifacts |

## Run the tracked evidence

```bash
uv sync --group dev
uv run ruff check src tests
uv run ty check
uv run pytest -q

uv run python -m neural_bridge.again verify-evidence phase7-grouped \
  --root studies/again/phase-07-continuous/grouped-confirmation
uv run python -m neural_bridge.zero_label \
  studies/again/zero-label/locked-confirmation
```

These evidence checks run on ordinary CPU hardware. Shared model training supports PyTorch on CPU/CUDA and MLX on Apple silicon. No Rust Token Killer installation is required.
