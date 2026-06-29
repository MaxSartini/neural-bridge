# Phase 5 Primary Adversarial Repair Evidence Snapshot

This is a lightweight tracked evidence snapshot, not the full output root.

The full heavy output root remains ignored under:

`outputs/again_dense_2hz_phase5_adversarial_repair_fixplus_20260629_171825/`

This snapshot preserves only small JSON and report artifacts needed to document the completed primary repair checkpoint.

## Result

- Primary repair trainings completed: `702/702`.
- Best checkpoints were restored before scoring.
- Model: `gated_ar_pca_mlp`.
- Target: `arousal_spike_rows_2_6_train_q90`.
- Feature: `temporal_mean_2s_then_pca256`.
- Grouped matched-control delta improved to `+0.0261165893` PR-AUC.
- Blocked matched-control delta remains negative at `-0.0088708442` PR-AUC.
- Strict forward-time temporal generalization remains unproven.
- No secondary model tier or secondary target was run.

## Included Artifacts

- `promotion/corrected_promotion_gates.json`
- `promotion/adversarial_verdict.json`
- `promotion/failure_reasons.json`
- `diagnostics/training_headroom_audit.json`
- `diagnostics/adversarial_repair_artifact_completeness_audit.json`
- `reports/again_dense_2hz_phase5_primary_repair_matrix_summary_20260629_171825.md`

## Excluded Artifacts

This snapshot intentionally excludes checkpoints, full CSV matrices, parquet files, `.npy`/`.npz` files, tensors, caches, model weights, dense cache files, and the full ignored output root.
