# Original VEATIC v2 Closure

## Outcome

Original VEATIC delivered Neural Bridge's first controlled future-event/spike result. It established the central hypothesis that short, causally constrained temporal change in video-derived predicted cortical activity can be more useful than raw current state.

This was not a disposable toy benchmark. VEATIC was published at WACV 2024 as a large contextual-affect dataset with `124` videos, `257,601` frames, `192` annotators, about `60` annotators per video, and continuous frame-level valence and arousal ratings. Its stimuli span Hollywood films, documentaries, and home videos, and its task explicitly combines character and surrounding context.

## Research question

Can temporal cortical-response structure rank rare near-future arousal events beyond learned response persistence and false-signal controls?

## Design

- Historical target: `arousal__future_spike_1_3s@0.05`.
- Evaluation: blocked future-event ranking plus a balanced event-vs-stable diagnostic.
- Comparators: trained AR, shuffled features, and random features.
- Scope: `124` VEATIC videos; no cross-dataset model transfer is claimed.

## Decisive evidence

| Endpoint | Neural Bridge | Comparator | Relative gain |
| --- | ---: | ---: | ---: |
| Blocked event PR-AUC | **`0.2536`** | AR `0.1969` | **`+28.80%`** |
| Blocked event PR-AUC | **`0.2536`** | shuffled `0.1840` | **`+37.83%`** |
| Blocked event PR-AUC | **`0.2536`** | random `0.1944` | **`+30.45%`** |
| Balanced event-vs-stable PR-AUC | **`0.3394`** | — | — |

## What this changed

The result justified an event-first programme and made temporal change—not raw cortical state—the working hypothesis. It did **not** establish continuous response ranking, zero-label inference, or universal transfer. Those harder questions were rebuilt on AGAIN with denser 2 Hz data, target-specific AR, fold-safe fitting, residual heads, matched controls, and prospective confirmation.

## Audit trail

- [`results/`](results/) contains the finalized benchmark, controls, alignment audits, manifest, and compact CSV/JSON evidence.
- [`report.md`](report.md) preserves the contemporary scientific summary and claim boundary.
- [`reproduction/`](reproduction/) contains the final strict runner, validated helpers, cache builders, freezer, audit runner, and the single strict contract test.
- The sealed evidence snapshot and heavy cache are hash-registered externally; repair runs, device archaeology, smoke scripts, and fitted artifacts are not promoted here.

## Boundary for VEATIC 2.1

This closure contributes hypotheses only. VEATIC 2.1 must not inherit fitted PCA, tensors, labels, models, thresholds, checkpoints, or an exact historical recipe. It starts from fresh dense 2 Hz data and current controls.

[Return to the complete study journey](../../README.md) · [Read the WACV 2024 paper](https://openaccess.thecvf.com/content/WACV2024/html/Ren_VEATIC_Video-Based_Emotion_and_Affect_Tracking_in_Context_Dataset_WACV_2024_paper.html)
