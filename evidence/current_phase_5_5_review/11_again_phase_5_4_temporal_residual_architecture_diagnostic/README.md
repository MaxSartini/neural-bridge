# AGAIN Phase 5.4: Temporal Residual Architecture Diagnostic

Purpose: compare small causal temporal/event-context residual heads on the redesigned blocked targets.

Contents:
- temporal residual design report
- 168-row blocked diagnostic evidence
- architecture comparison, seed deltas, controls, gates, leakage/context audit, label permutation audit, and train-only video mean audit

Key result:
- `short_temporal_conv_residual` was the best binary architecture.
- binary diagnostic passed.
- continuous diagnostic did not pass.

This phase selected the architecture for Phase 5.5 confirmation; it is diagnostic, not the final confirmation.
