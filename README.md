# Neural Bridge

Neural Bridge turns video-derived cortical representations into ranked human-response intelligence under strict controls.

## Current State

Defensible claim: cross-video future arousal spike / emotional moment ranking.

Canonical review: [docs/reviews/neural_bridge_phase5_adversarial_review_20260625.html](docs/reviews/neural_bridge_phase5_adversarial_review_20260625.html)

Key numbers:

- AR-only grouped spike PR-AUC: about `0.14725`
- Phase 4 grouped spike PR-AUC: about `0.17165`
- Phase 5 grouped spike PR-AUC: about `0.21913`
- Honest matched-control grouped cortical edge: about `+0.02` PR-AUC

The Phase 5 result is not fraud and not grossly leaky. Label permutation supports no gross leakage, and grouped-video signal is real and fold-robust. The old headline was overpromoted: blocked-temporal matched controls beat real, so strict forward-time temporal generalization remains under repair.

Do not claim continuous arousal forecasting is solved, strict temporal prediction is proven, Phase 5b/5c/Spark outputs are canonical, or `holy_shit_pass` is a valid gate.

## Canonical Artifacts

- Dense root: `.cache/h100_drive_downloads/again_tribe_v2_postpass_float16_256_2hz/`
- Phase 4 root: `$NEURAL_BRIDGE_EXTERNAL_ROOT/outputs/again_dense_2hz_phase4_pca_bridge_20260625_full/`
- Phase 5 main: `outputs/again_dense_2hz_phase5_learned_heads_20260625_182423/`
- Phase 5 sanity: `outputs/again_dense_2hz_phase5_learned_heads_20260625_185338/`
- Phase 5 runner: `backend/scripts/run_again_dense_2hz_phase5_learned_heads.py`
- Evidence bundle: `evidence_bundle_phase0_to_phase5_20260625/`

Primary Phase 5 lane: `arousal_spike_rows_2_6_train_q90` with `gated_ar_pca_mlp` / `regression_plus_binary` / `temporal_mean_2s_then_pca256` / AR + temporal diagnostics.

## Next Work

Next task: original Phase 5 adversarial repair. Start by rescoring or rerunning the primary lane with matched controls, corrected blocked gate, blocked split audit, video-mean/static PCA diagnostic, within-video ranking, top-percent metrics, paired fold deltas/CI, and real-minus-matched-control as the headline effect.
