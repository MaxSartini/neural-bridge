# AGAIN Phase 5/5.5: Learned Bridge and Event Head

## Outcome

Phase 5 is the central model-development chapter. It moved from fixed representations to a learned frozen-AR residual bridge, exposed and corrected an evaluation problem, rejected an event design that failed its controls, and ultimately confirmed a redesigned causal temporal event head under both blocked-temporal and held-out-video protocols.

## Research question

Can a learned temporal head extract event/spike information that raw features and fixed PCA fail to expose, while adding signal beyond the exact same frozen target-specific AR and matched false-signal controls?

## The experimental sequence

1. **Evaluation correction.** An eval-mode rescore corrected the initial learned-head evaluation before promotion.
2. **Learned residual breakthrough.** A frozen-AR residual bridge reached grouped PR-AUC `0.238341` on the original spike target.
3. **Continuous lead.** Continuous ranking also improved, creating the hypothesis later specialized and confirmed in Phase 7.
4. **Controlled rejection.** Matched controls eliminated the first redesigned event target before confirmation resources were committed.
5. **Head-family pivot.** Causally constrained temporal heads changed the result; `short_temporal_conv_residual` became the selected architecture.
6. **Dual-protocol confirmation.** The selected head passed blocked-temporal confirmation and grouped held-out-video compatibility.

## Representation breakthrough

| Original same-target grouped system | PR-AUC | Change vs raw cortical |
| --- | ---: | ---: |
| Raw cortical only | `0.136579` | baseline |
| Direct AR + raw cortical | `0.170299` | `+24.69%` |
| Phase 4 PCA bridge | `0.171648` | `+25.68%` |
| Deterministic learned bridge | `0.230064` | `+68.45%` |
| Frozen-AR residual bridge | **`0.238341`** | **`+74.51%`** |

The residual bridge finished **`+39.95%`** above direct fusion and **`+38.85%`** above the Phase 4 bridge. This is the clearest apples-to-apples evidence that Neural Bridge—not raw TRIBE output—created the useful representation.

## Selected event-head confirmation

| Protocol | Neural Bridge | Frozen AR | Absolute gain | Relative gain |
| --- | ---: | ---: | ---: | ---: |
| Blocked temporal | **`0.267074`** | `0.260234` | `+0.006840` | **`+2.63%`** |
| Grouped held-out video | **`0.231383`** | `0.217495` | `+0.013888` | **`+6.39%`** |

The grouped result was positive in **`50/50`** fold-seeds. The final `420`-cell assembly combines already completed blocked and grouped evidence without rerunning models; those are comparison cells, not 420 independent observations. Its corrected verdict applies the residual-appropriate null while preserving the original record for auditability.

## What failed—and why it mattered

- The initial evaluation behavior was corrected rather than silently overwritten.
- The first redesigned event target was rejected because its matched controls did not support promotion.
- Fixed PCA improved only marginally over direct fusion; learned temporal residual structure produced the large step change.
- Continuous improvement was treated as a new hypothesis, not claimed as confirmed from event-head discovery.

## Audit trail

[`evidence/`](evidence/) contains compact CSV, JSON, reports, manifests, audits, and promotion records for every transition. Heavy fitted PCA row exports remain in the registered external Phase 5 collection. The canonical engine, evidence verifier, and checksum-locked replay replace 22 phase-coupled script copies; old all-phase snapshots are not authorities.

## Transition

Phase 5 solved the event head. Phase 6 asked the next rigorous question: could that win be stabilized prospectively across fresh checkpoints, or was it dependent on a favourable training realization?

[Continue to Phase 6 — event stabilization](../phase-06-event-stabilization/README.md) · [Return to the journey](../../README.md)
