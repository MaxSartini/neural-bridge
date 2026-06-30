# Phase 5 Frozen-AR Residual Response to Eval-Mode Repair

Output root: `outputs/again_dense_2hz_phase5_frozen_ar_residual_`

## Eval-Mode Problem Addressed

The deterministic eval-mode Phase 5 primary repair remains the canonical score pass. It established grouped-video ranking signal but did not prove strict forward-time temporal generalization: blocked real PR-AUC was `0.2218656156`, blocked best matched control PR-AUC was `0.2311845051`, and blocked AR-only PR-AUC was `0.2654721820`.

That result implied the fused real PCA head could damage the blocked AR/time path. The frozen-AR residual repair directly tests whether cortical PCA/diagnostics can add residual value over an anchored AR baseline without being allowed to pull the final score far below AR.

## Reuse and Scoring

- Reused the existing Phase 5 primary repair artifacts and split/feature reconstruction.
- Re-forwarded AR-only best checkpoints in eval mode because per-row AR score files were not already available.
- Did not retrain AR-only checkpoints.
- Trained only frozen-AR residual heads and matched residual controls.
- Used deterministic eval-mode scoring, checkpoint restore audits, frozen-AR integrity audits, residual alpha gates, and do-no-harm gates.

## Artifact-Backed Result

Grouped:

- Frozen AR PR-AUC: `0.2246816187`
- Best real residual: `frozen_ar_plus_mlp_residual`, PR-AUC `0.2383409298`
- Best matched residual control: `shuffled_pca_frozen_ar_residual` / `frozen_ar_plus_gated_residual`, PR-AUC `0.2248361805`
- Delta vs frozen AR: `+0.0136593110`
- Delta vs best matched control: `+0.0135047493`
- `grouped_residual_pass`: yes

Blocked:

- Frozen AR PR-AUC: `0.2654721820`
- Best real residual: `frozen_ar_plus_linear_residual`, PR-AUC `0.2635930904`
- Best matched residual control: `shuffled_pca_frozen_ar_residual` / `frozen_ar_plus_gated_residual`, PR-AUC `0.2653404381`
- Delta vs frozen AR: `-0.0018790916`
- Delta vs best matched control: `-0.0017473477`
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

## Corrected Claim

Frozen-AR residual bridge adds cross-video future arousal spike / emotional moment ranking signal beyond AR and matched residual controls, while preserving blocked AR performance within do-no-harm tolerance. Strict forward-time residual improvement is not yet proven.

## Next Work

Do not start broad secondary targets or additional heads yet. The next task is targeted blocked residual improvement.

Candidate targeted repairs:

- Stronger residual alpha regularization.
- Blocked-only inner-val delta selection.
- Rank/lift auxiliary loss.
- Monotonic/do-no-harm residual gating.
- Residual branch trained only on windows where AR confidence is low.

No full 702 rerun should be started unless a targeted diagnostic requires it.
