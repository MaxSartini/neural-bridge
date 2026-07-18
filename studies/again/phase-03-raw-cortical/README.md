# AGAIN Phase 3: Raw Cortical Benchmark

## Outcome

Phase 3 produced the programme's most important early negative result: a rich 20,484-vertex predicted cortical representation was **not automatically useful**. Raw summaries were target-dependent and often weaker than learned arousal persistence.

## Research question

Before building a sophisticated bridge, can raw cortical summaries—or their direct fusion with AR—already beat the target-specific frozen AR baseline?

## Grouped held-out-video evidence

| Target | Raw cortical | AR only | AR + raw | Verdict |
| --- | ---: | ---: | ---: | --- |
| Future spike | `0.1366` | `0.1473` | **`0.1703`** | raw loses; direct fusion helps |
| Short delta | `0.1326` | **`0.2084`** | `0.2019` | raw and direct fusion both lose |
| Absolute delta | **`0.1265`** | `0.1182` | — | raw contains target-specific signal |

Shuffled, random, timestamp, quality, motion, and luma controls were retained to test whether feature shape, acquisition order, or generic video diagnostics could explain apparent gains.

## Scientific interpretation

This is not a universal claim that predicted cortical features fail. The absolute-delta result and spike fusion show that they contain information. The failure is more specific and more useful: fixed summaries do not consistently expose that information, and naïve fusion can make a strong predictor worse.

That finding defines Neural Bridge's contribution. The project is not “run TRIBE and read out emotion”; it is the machinery that turns a difficult high-dimensional representation into aligned, controlled, future-response signal.

## Audit trail

[`evidence/`](evidence/) preserves preliminary and final compact runs, the final report, controls, and representation metadata. Dense features and raw matrices remain externally registered. Current control and evaluation contracts live in [`src/neural_bridge/again/`](../../../src/neural_bridge/again/).

## Transition

Phase 4 tested the smallest disciplined response to this failure: train-fold-fitted compression plus short causal temporal aggregation, with no test-fitted PCA and no inherited full-data projection.

[Continue to Phase 4 — fold-safe PCA bridge](../phase-04-pca-bridge/README.md) · [Return to the journey](../../README.md)
