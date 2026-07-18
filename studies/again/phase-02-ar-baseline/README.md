# AGAIN Phase 2: Target-Specific AR Baseline

## Outcome

Phase 2 established the strong learned persistence floor used by every later comparison. Neural Bridge would have to add signal beyond this model—not merely beat a last-value or constant baseline.

## Research question

How much future event and movement structure is already predictable from current and recent arousal history, and how does that floor change between blocked-temporal and held-out-video protocols?

## Design

- Current and lagged arousal/history features, trained separately for each target.
- Train-split event thresholds and train-only inner validation for ridge selection.
- Blocked-temporal and grouped held-out-video protocols reported separately.
- The exact fold/seed AR is frozen and reused unchanged beneath later real and matched-control residual lanes.

## Final reference

| Target | Blocked PR-AUC | Grouped PR-AUC |
| --- | ---: | ---: |
| Future spike | **`0.2036`** | **`0.1473`** |
| Short delta | **`0.2619`** | **`0.2084`** |
| Absolute delta | **`0.1160`** | **`0.1182`** |

## Development lesson

Four earlier revisions produced materially different grouped spike scores. Rather than hiding that instability, Phase 2 retains those runs under [`evidence/development/`](evidence/development/) and promotes only [`evidence/final/`](evidence/final/) as the later reference. The lesson was foundational: baseline construction is part of the scientific result, and changing it can change the apparent value of cortical features.

## Claim boundary

AR captures response persistence, not necessarily video understanding. Later phases therefore train it separately and freeze it identically for the real bridge and every residual control; no control receives a weaker AR floor.

## Transition

With the hard baseline fixed, Phase 3 could test the simplest possible neuro-video proposition: do raw summaries of the 20,484 predicted cortical vertices add useful target-specific signal?

[Continue to Phase 3 — raw cortical benchmark](../phase-03-raw-cortical/README.md) · [Return to the journey](../../README.md)
