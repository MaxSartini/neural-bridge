# AGAIN Phase 2: Target-Specific AR Baseline

Phase 2 established the strong learned persistence floor used by every later comparison. This was not a naïve last-value baseline: it used current and lagged arousal/history features, train-split target thresholds, and train-only inner validation for ridge selection.

The final benchmark kept blocked-temporal and grouped held-out-video protocols distinct. Final blocked PR-AUC was `0.2036` for spike, `0.2619` for short delta, and `0.1160` for absolute delta. Grouped PR-AUC was `0.1473`, `0.2084`, and `0.1182`, respectively.

Four earlier revisions produced materially different grouped spike scores. They remain under `evidence/development/` as non-canonical development evidence; only `evidence/final/` is the Phase 2 reference used by later stages. Current AR contracts live in `src/neural_bridge/again/`; the historical phase entrypoint is not part of the current API.
