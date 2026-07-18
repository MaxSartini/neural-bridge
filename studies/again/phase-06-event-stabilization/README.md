# AGAIN Phase 6: Event Stabilization

## Outcome

Phase 6 turned the selected event head into a repeatable, prospectively declared procedure. The winner was not a larger hyperparameter search: it was a simple average of independently trained checkpoints from the already validated reference recipe.

## Research question

Which stabilization strategy survives fresh blocked and grouped confirmation rather than looking strong on one seed, one blend, or one development view?

## Controlled selection sequence

| Candidate strategy | Fresh evidence | Decision |
| --- | --- | --- |
| Optuna-selected single-seed configuration | promising pilot, insufficient robust confirmation | rejected |
| Fixed blend 1 | no repeatable improvement | rejected |
| Fixed blend 2 | no repeatable improvement | rejected |
| Declared reference-checkpoint average | passed fresh blocked and grouped gates | promoted |

These failures narrowed the scientific answer: reuse the validated head recipe, vary independent training realization, and average only the checkpoints declared before confirmation.

## Final grouped event result

| Endpoint | Neural Bridge | Frozen AR | Best matched control | Gain over AR | Positive groups |
| --- | ---: | ---: | ---: | ---: | ---: |
| Event PR-AUC | **`0.234368`** | `0.218050` | `0.217972` | `+0.016318` (**`+7.48%`**) | **`15/15`** |

The prospectively declared ensemble also passed fresh blocked confirmation. Every declared gate passed; the event gain was therefore both controlled and stable across five folds and three checkpoint groups.

## Scientific interpretation

Phase 6 is evidence against a “lucky checkpoint” explanation. It also demonstrates why Optuna is not automatically the most rigorous or effective tool: a promising single-seed optimum failed the robustness requirement, whereas a simpler prospectively declared ensemble generalized.

## Audit trail

[`plans/`](plans/) preserves every prospective decision. [`runs/`](runs/) retains each branch's result, report, manifest, and decisive audit where available. Checkpoints, scores, predictions, databases, and fold material remain in registered external runs; current execution lives in the canonical engine.

## Transition

With event/spike ranking both solved and stabilized, Phase 7 could specialize the proven machinery for a distinct target: continuous future-movement ranking and top-tail concentration.

[Continue to Phase 7 — continuous ranking](../phase-07-continuous/README.md) · [Return to the journey](../../README.md)
