# AGAIN Phase 1: Dense 2 Hz Label Alignment

Phase 1 establishes the supervised table contract; it does not claim predictive signal.

The aligned table contains `243,575` dense rows across all `995` videos. `243,441` rows have labels; `134` unmatched rows across `38` videos remain explicit, as do `4,153` rows without sufficient AR history. Saved dense-cache timestamps are authoritative—this is not a 1 Hz fallback.

Continuous future-movement values and eligibility masks are stored in the external aligned parquet. Binary q90 thresholds are selected inside each training fold, never from test labels.

`evidence/` contains the compact contract, summary, and report. The row-level parquet remains in the registered external derived collection. Current target and row-alignment logic lives in `src/neural_bridge/again/`; the phase-specific script snapshot is provenance only and is not duplicated here.
