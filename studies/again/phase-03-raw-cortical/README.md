# AGAIN Phase 3: Raw Cortical Benchmark

Phase 3 established a target-specific problem, not a universal raw-feature failure.

On grouped spike ranking, raw cortical scored PR-AUC `0.1366` versus AR `0.1473`, while direct AR-plus-raw improved to `0.1703`. On short-delta ranking, raw (`0.1326`) and AR-plus-raw (`0.2019`) remained below AR (`0.2084`). On absolute-delta ranking, raw beat AR: `0.1265` versus `0.1182`.

The evidence therefore says that fixed summaries of the 20,484-vertex representation were inconsistent and often weaker than learned persistence, while still containing useful target-dependent signal. Shuffled, random, timestamp, quality, motion, and luma controls remain part of the comparison.

`evidence/` preserves both preliminary and final compact runs, the final report, and representation metadata. Dense features and raw matrices remain external. Current controls and evaluation live in `src/neural_bridge/again/`; the superseded phase entrypoint is not duplicated.
