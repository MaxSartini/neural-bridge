# AGAIN zero-label direct-supervised locked confirmation

Output root: `/Volumes/onn. Drive/Neural Bridge/outputs/again_dense_2hz_zero_label_direct_supervised_locked_confirm_20260715`

- exact matrix: `140/140`
- Tier 1 baseline-beating deployment signal: **true**
- Tier 2 high-consistency confirmation: **true**
- Tier 3 first-30-second confirmation: **true**
- failed Tier 1 gates: `[]`
- teacher retention was report-only and Phase 7 was a ceiling, not a pass threshold.

## Required endpoints

### `pooled_continuous_spearman`

- primary / strongest control: `0.1785132961` / `0.1004882655`
- strongest control: `diagnostics_only_supervised_temporal`
- aggregate delta: `+0.0780250306`
- panel wins: `5/5`
- one-sided bootstrap lower 95%: `+0.0606787212`
- first-30 panel wins: `5/5`

### `top_5pct_true_future_movement_lift`

- primary / strongest control: `0.0766079674` / `0.0448520122`
- strongest control: `no_video_supervised_temporal`
- aggregate delta: `+0.0317559552`
- panel wins: `5/5`
- one-sided bootstrap lower 95%: `+0.0187740072`
- first-30 panel wins: `5/5`

### `training_q90_future_event_pr_auc`

- primary / strongest control: `0.1710622218` / `0.1352295369`
- strongest control: `diagnostics_only_supervised_temporal`
- aggregate delta: `+0.0358326849`
- panel wins: `5/5`
- one-sided bootstrap lower 95%: `+0.0235455194`
- first-30 panel wins: `4/5`
