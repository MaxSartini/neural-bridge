# AGAIN Phase 1: Dense 2 Hz Label Alignment

## Outcome

Phase 1 established the supervised table contract for all later AGAIN experiments. It is alignment evidence, not a predictive-signal claim.

## Research question

Can continuous arousal labels, future-movement targets, event eligibility, and AR-history eligibility be aligned to the dense saved feature rows without interpolation shortcuts, hidden row loss, or test-label leakage?

## Design contract

- Saved dense-cache timestamps are authoritative; there is no 1 Hz fallback.
- Continuous future-movement values and eligibility masks remain row-addressable.
- Rows that cannot be matched or lack sufficient AR history remain explicit.
- Binary q90 event thresholds are learned inside each training fold, never from held-out labels.

## Decisive evidence

| Alignment audit | Result |
| --- | ---: |
| Videos represented | **`995/995`** |
| Dense 2 Hz rows | **`243,575`** |
| Rows with labels | **`243,441`** |
| Explicit unmatched rows | `134` across `38` videos |
| Rows without sufficient AR history | `4,153` |

The unmatched and insufficient-history rows were not silently imputed into eligibility. Continuous targets, masks, and provenance remain in the registered aligned parquet.

## Why this phase mattered

Phase 1 prevented target leakage and timing ambiguity from contaminating every later comparison. It also preserved valid zero-event negatives: event eligibility is a row-level scientific contract, not a requirement that every video contain a positive event.

## Audit trail

[`evidence/`](evidence/) contains the compact contract, summary, and report. The row-level parquet remains in the registered external derived collection. Current target and alignment logic lives in [`src/neural_bridge/again/`](../../../src/neural_bridge/again/); the historical phase snapshot remains provenance only.

## Transition

Once labels and eligibility were sealed, Phase 2 could answer the question every later bridge had to face: how much of the future target can a strong, separately trained persistence model already explain?

[Continue to Phase 2 — target-specific AR](../phase-02-ar-baseline/README.md) · [Return to the journey](../../README.md)
