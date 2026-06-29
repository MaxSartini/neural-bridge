# Phase 5 Frozen-AR Residual Evidence Snapshot

This is a lightweight tracked evidence snapshot, not the full output root.

The full heavy output root remains ignored under `outputs/again_dense_2hz_phase5_frozen_ar_residual_/`.

The frozen-AR residual experiment used deterministic eval-mode scoring. AR-only best checkpoints were reused and re-forwarded in eval mode; AR-only checkpoints were not retrained.

## Verdict

- Grouped residual pass: yes
- Blocked residual pass: no
- Do-no-harm blocked pass: yes
- Strict forward-time temporal generalization remains unproven.
- Recommendation: `exploratory_grouped_only`

## Included Artifacts

- `promotion/frozen_ar_residual_gates.json`
- `promotion/frozen_ar_residual_adversarial_verdict.json`
- `promotion/frozen_ar_residual_failure_reasons.json`
- `promotion/frozen_ar_residual_matched_control_comparison.csv`
- `promotion/frozen_ar_residual_vs_evalmode_baseline.csv`
- `diagnostics/frozen_ar_integrity_audit.json`
- `diagnostics/checkpoint_restore_audit.json`
- `diagnostics/eval_mode_scoring_audit.json`
- `diagnostics/do_no_harm_audit.json`
- `diagnostics/residual_alpha_gate_audit.json`
- `reports/again_dense_2hz_phase5_frozen_ar_residual_summary_.md`
- `reports/again_dense_2hz_phase5_frozen_ar_residual_response_to_evalmode_.md`
