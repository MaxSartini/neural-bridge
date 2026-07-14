# Backend Scripts

This directory contains claim-bearing benchmark runners, no-training verdict/audit scripts, summarizers, and supporting data builders. The current scripts are kept in place because imports and historical reports reference these paths.

## Current Claim-Bearing AGAIN Scripts

- `run_again_dense_2hz_phase7_continuous_checkpoint_ensemble_grouped.py` - current headline: fresh `420/420` grouped continuous future-movement ranking/lift confirmation, `15/15` positive fold-groups, failed gates `[]`.
- `run_again_dense_2hz_phase7_continuous_checkpoint_ensemble_blocked_confirm.py` - separate fresh `140/140` blocked near-pass; strong aggregate result with a literal `4/5` under its locked `5/5` gate.
- `run_again_dense_2hz_phase7_continuous_checkpoint_ensemble_diagnostic.py` - bounded precursor that separated ranking/lift from exact-value gates.
- `run_again_dense_2hz_phase6_original_three_checkpoint_control_complete.py` - fresh blocked binary checkpoint-ensemble confirmation.
- `run_again_dense_2hz_phase6_original_three_checkpoint_grouped_confirmation.py` - fresh grouped binary checkpoint-ensemble confirmation.
- `run_again_dense_2hz_phase5_temporal_residual_binary_big_confirm.py` - Phase 5.5 matched seed-specific 10-seed blocked binary confirmation.
- `summarize_again_dense_2hz_phase5_temporal_residual_binary_big_confirm.py` - summary helper for the blocked binary confirmation.
- `run_again_dense_2hz_phase5_temporal_residual_grouped_compat.py` - Phase 5.5 grouped-video compatibility run.
- `summarize_again_dense_2hz_phase5_temporal_residual_grouped_compat.py` - summary helper for grouped compatibility.
- `update_again_dense_2hz_phase5_temporal_residual_grouped_compat_verdict.py` - no-training updated verdict logic for frozen-AR residual label-permutation nulls.

## AGAIN Phase 0-5.4 Support

- Dense/cache/label substrate: `again_dense_2hz_benchmark.py`, `build_again_labels_aligned_2hz.py`, `run_again_dense_2hz_ar_baseline.py`, `run_again_dense_2hz_raw_cortical_benchmark.py`
- Phase 4 PCA bridge: `again_dense_2hz_phase4_pca_bridge.py`, `summarize_again_dense_2hz_phase4_pca_bridge.py`
- Phase 5 primary/eval-mode/frozen-AR path: `run_again_dense_2hz_phase5_adversarial_correction_fixplus.py`, `summarize_again_dense_2hz_phase5_adversarial_correction_fixplus.py`, `rescore_again_dense_2hz_phase5_evalmode.py`, `run_again_dense_2hz_phase5_frozen_ar_residual.py`, `summarize_again_dense_2hz_phase5_frozen_ar_residual.py`
- Phase 5.2/5.3/5.4 diagnostics: blocked residual, clean confirmation, continuous residual, redesigned targets, and temporal residual diagnostic runners plus summarizers.

## VEATIC Foundation Scripts

- `run_veatic_strict_benchmark.py` - canonical VEATIC strict benchmark contract/dry-run.
- `run_veatic_neuro_benchmark.py` - VEATIC cortical feature benchmark.
- `run_veatic_event_spike_retest.py`, `run_veatic_event_conditioned_retest.py`, `run_veatic_temporal_fairness_benchmark.py`, `run_veatic_temporal_context_v2.py`, `run_veatic_alignment_lag_audit.py` - VEATIC v2 evidence ladder runners.
- `run_veatic_raw_representation_audit.py`, `veatic_representation_builders.py`, `veatic_frozen_tensor_adapter.py`, `veatic_frozen_tensor_trained_heads.py`, `run_veatic_frozen_tensor_trained_heads_benchmark.py` - post-v2 tensor and trained-head layer.
- `freeze_veatic_v2_evidence.py` - evidence freeze/verification tooling.

## Maintenance And Runtime Support

`audit_repo_readiness.py`, `cleanup_generated_artifacts.py`, `check_tribe_encoder_assets.py`, `mlx_runtime_config.py`, and setup/conversion scripts support reproducibility and repository hygiene. They are not benchmark evidence by themselves.

## Policy

Do not run a benchmark script just to satisfy documentation. Claims must be backed by completed reports and evidence bundles. Heavy outputs, checkpoints, arrays, dense caches, and model assets stay outside git unless a user explicitly approves a small metadata snapshot.

Current interpretation lives in `../../docs/neural_bridge_phase7_evidence.md`; Phase 5.5 scripts remain claim-bearing foundations, not the present performance ceiling.
