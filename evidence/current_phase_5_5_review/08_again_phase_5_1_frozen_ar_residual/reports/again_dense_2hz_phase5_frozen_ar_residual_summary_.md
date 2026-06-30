# Phase 5 Frozen-AR Residual Repair Summary

Output root: `outputs/again_dense_2hz_phase5_frozen_ar_residual_`

## Why Frozen AR

The deterministic eval-mode Phase 5 repair rescore showed a grouped-video cortical ranking signal, but it also showed blocked-temporal AR-only dominance. In blocked validation, the fused real PCA head scored below the AR-only path, which indicates the fused gated head could let real PCA interfere with the AR/time baseline.

The frozen-AR residual repair makes the eval-mode AR-only score/logit the baseline floor. Cortical PCA and temporal diagnostics can only contribute as a residual correction through `final_score = frozen_ar_score + alpha * residual_score`, with alpha initialized near zero and gated so the residual must earn improvement or do no harm.

## Reused Inputs

- Reused the existing Phase 5 primary repair artifacts.
- Reused the existing split and feature reconstruction path.
- Re-forwarded AR-only best checkpoints in eval mode because per-row AR score files were not already available.
- No AR retraining was performed.
- No V-JEPA/TRIBE reruns, PCA refits, secondary targets, secondary heads, dense-cache writes, or original Phase 4/5 output edits were performed.

## Grouped Result

- Grouped frozen AR PR-AUC: `0.2246816187`
- Grouped best real residual: `frozen_ar_plus_mlp_residual`, PR-AUC `0.2383409298`
- Grouped best matched residual control: `shuffled_pca_frozen_ar_residual` / `frozen_ar_plus_gated_residual`, PR-AUC `0.2248361805`
- Grouped delta vs frozen AR: `+0.0136593110`
- Grouped delta vs best matched control: `+0.0135047493`
- `grouped_residual_pass`: yes

## Blocked Result

- Blocked frozen AR PR-AUC: `0.2654721820`
- Blocked best real residual: `frozen_ar_plus_linear_residual`, PR-AUC `0.2635930904`
- Blocked best matched residual control: `shuffled_pca_frozen_ar_residual` / `frozen_ar_plus_gated_residual`, PR-AUC `0.2653404381`
- Blocked delta vs frozen AR: `-0.0018790916`
- Blocked delta vs best matched control: `-0.0017473477`
- `blocked_residual_pass`: no
- `do_no_harm_blocked_pass`: yes

## Gates

- `eval_mode_scoring_pass`: true
- `checkpoint_restore_pass`: true
- `frozen_ar_integrity_pass`: true
- `grouped_residual_pass`: true
- `blocked_residual_pass`: false
- `do_no_harm_blocked_pass`: true
- `full_forward_time_pass`: false
- `strict_forward_time_temporal_generalization_proven`: false
- `recommendation`: `exploratory_grouped_only`

## Interpretation

The frozen-AR residual design fixed the worst old behavior: real PCA no longer badly damages the blocked AR baseline. Cross-video/grouped residual signal is strengthened and cleaner because the real cortical residual improves beyond frozen AR and beyond matched residual controls.

Strict forward-time temporal generalization is still not proven because the blocked real residual does not beat frozen AR or the best matched residual control.

Corrected claim: Frozen-AR residual bridge adds cross-video future arousal spike / emotional moment ranking signal beyond AR and matched residual controls, while preserving blocked AR performance within do-no-harm tolerance. Strict forward-time residual improvement is not yet proven.

## Next Work

Do not widen to secondary targets yet. Blocked residual improvement remains the main open problem.

Possible next targeted repairs:

- Stronger residual alpha regularization.
- Blocked-only inner-val delta selection.
- Rank/lift auxiliary loss.
- Monotonic/do-no-harm residual gating.
- Residual branch trained only on windows where AR confidence is low.

Do not rerun the full 702 matrix unless a targeted diagnostic requires it.
