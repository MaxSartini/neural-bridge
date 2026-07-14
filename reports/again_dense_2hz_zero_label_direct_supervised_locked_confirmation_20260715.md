# AGAIN zero-label direct-supervised locked confirmation

The fixed direct-supervised temporal video-only Neural Bridge passed its
prospectively locked 299-video confirmation on `2026-07-15`.

- exact scored matrix: `140/140`;
- MLX accelerator: `Device(gpu, 0)`;
- all split, target, train-only PCA, inference firewall, prediction-before-label,
  finite-coverage, and hardware audits: pass;
- Tier 1 baseline-beating deployment signal: true;
- Tier 2 high-consistency confirmation: true;
- Tier 3 first-30-second confirmation: true;
- failed Tier 1 gates: `[]`.

## Primary result

| Endpoint | Real video-only bridge | Strongest matched control | Delta | Relative gain | Panel wins | Bootstrap lower 95% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Spearman | `0.1785132961` | `0.1004882655` diagnostics-only | `+0.0780250306` | `+77.65%` | `5/5` | `+0.0606787212` |
| Top-5% lift | `0.0766079674` | `0.0448520122` no-video | `+0.0317559552` | `+70.80%` | `5/5` | `+0.0187740072` |
| Event PR-AUC | `0.1710622218` | `0.1352295369` diagnostics-only | `+0.0358326849` | `+26.50%` | `5/5` | `+0.0235455194` |

All one-sided paired whole-video bootstrap lower bounds are above zero. The
candidate also beats the current-row video model, sequence-shuffled video,
hard-label permutation, diagnostics-only, and no-video lanes on all three
aggregate endpoints.

The first-30-second tier passed with aggregate deltas `+0.0803990394`,
`+0.0350573783`, and `+0.0310250147`; directional panels were `5/5`, `5/5`,
and `4/5`.

## Interpretation

This is the first locked Neural Bridge confirmation in which held-out inference
uses no observed current/past arousal. It confirms that cached predicted
cortical/fMRI video features can support useful cold-start future-response
ranking on AGAIN beyond matched no-video and false-signal controls.

It does not overwrite Phase 7. Phase 7 remains the stronger observed-arousal-
assisted research ceiling. The locked ceiling here was `0.2992983823` Spearman,
`0.1140998111` top-5% lift, and `0.2355919160` event PR-AUC. The new result
solves a different problem: translating a substantial portion of that signal
into a deployable zero-label-at-inference lane.

The result uses cached upstream features. End-to-end raw-video feature
generation/runtime, external cross-domain confirmation, prospective client
outcomes, exact trajectories, and label-free training remain open.

Canonical snapshot:
`evidence/zero_label_video_only_direct_supervised_locked_confirmation_20260715/`.

Full runtime root:
`$NEURAL_BRIDGE_EXTERNAL_ROOT/outputs/again_dense_2hz_zero_label_direct_supervised_locked_confirm_20260715/`.
