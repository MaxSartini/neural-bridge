# VEATIC-124 v2 Evidence Summary

Generated from the current v2 reports imported into the cleaned Neural Bridge repo.

## Headline

VEATIC-124 v2 proves specific Neural Bridge hypotheses for arousal event/spike ranking using a video-dominant cortical/TRIBE cache. It shows that visual/video-driven cortical/TRIBE PCA feature modes can improve future arousal spike/event ranking over autoregressive, shuffled, random, timestamp, and video/time controls under blocked and grouped-video validation.

The claim remains bounded: this is event/spike ranking and temporal-context evidence from a mostly visual/video cache, not exact continuous arousal-value prediction, a finished downstream product model, or proof that full text+audio+video TRIBE has been evaluated.

After the v2 evidence freeze, a raw cortical representation audit, tensor export, and MPS trained-head benchmark were completed without re-encoding videos. These do not replace the v2 claim; they define and test the post-v2 model-head layer on top of it.

## Proven Or Supported Hypotheses

1. Real video-dominant cortical/TRIBE features carry stimulus-specific signal for future arousal spike ranking.
2. PCA feature modes are materially stronger than the 6-feature global baseline for spike/event ranking.
3. Balanced event-vs-stable evaluation exposes signal that full-frame continuous MAE can hide.
4. Short causal temporal context can improve selected spike-ranking rows over current-only evaluation.
5. Single-frame 0s evaluation can underfeed the bridge head for spike/event tasks.
6. Timing/alignment policy is resolved: current 0s alignment stays primary; offset-grid and train-selected timing checks are diagnostics.

## Key Numbers

- Strongest blocked full-frame spike row: `cortical_pca64_delta`, `arousal__future_spike_1_3s`, threshold `0.05`, PR-AUC `0.2536`.
- Same row controls: AR `0.1969`, shuffled `0.1840`, random `0.1944`.
- Official split spike rows pass controls across current feature families.
- Grouped-video aggregate spike F1 improves over AR for PCA modes: `cortical_pca_64` `+0.0256`, `cortical_pca64_delta` `+0.0177`.
- Balanced event-vs-stable `arousal__future_spike_1_3s@0.05`: `cortical_pca64_delta` PR-AUC `0.3394`, `+0.0609` over AR, `+0.0631` over shuffled, `+0.0476` over random.
- Temporal context v2: 4/4 focused feature-target rows improved over current-only by more than `0.005` PR-AUC; best focused windows were `causal_past_2s`.
- Alignment repair: best offsets vary by target/mode, so no global lag correction was selected; final policy is `keep_current_0s_as_primary_plus_report_offset_diagnostics`.
- Modality audit: `122/124` current cache entries are video-only and `2/124` contain text+audio+video, so the v2 result should not be described as a full multimodal TRIBE result.

## Post-v2 Raw Representation And Tensor Export

- Frozen baseline retained: `cortical_pca64_delta`.
- Primary trained-head input: `pca_sequence_128_causal_past_2s_mean`.
- Important side candidate: `roi_parcel_features`.
- Supervised/cautionary candidate: `topk_vertices_512`.
- Raw uncompressed ridge: valid, but not the best next build target.
- Raw representation audit output: `${NEURAL_BRIDGE_EXTERNAL_ROOT}/outputs/veatic_124_raw_representation_audit_primary_20260620_152411`.
- Tensor export root: `${NEURAL_BRIDGE_EXTERNAL_ROOT}/tensors/veatic_124_raw_representation_v1`.
- Tracked tensor summary: `outputs/veatic_124_raw_representation_tensor_export_v1`.
- Exported tensor contracts: `84` representation/split/target folders and `420` external `.npy` tensor files.
- Verification status: `pass`.
- PCA cache reuse: `14` reused, `0` rebuilt.
- Video `83` was included in all-video tensor contracts; exclude-video-83 tensor sensitivity was skipped.

## Post-v2 Trained Heads

- Implemented runner: `backend/scripts/run_veatic_frozen_tensor_trained_heads_benchmark.py`.
- Core implementation: `backend/scripts/veatic_frozen_tensor_adapter.py` and `backend/scripts/veatic_frozen_tensor_trained_heads.py`.
- Backend policy: MPS required; CPU sklearn fallback is refused.
- Freshness policy: same-row AR and controls are recomputed in-run; prior benchmark result rows are not reused.
- Completed run handle: `outputs/veatic_124_frozen_tensor_trained_heads_mps_20260620_full_lightweight.zip`.
- Result framing: `AR_plus_PCA128` and `residualized_AR_plus_PCA128` passed grouped spike incremental gates against AR, controls, and PCA64-delta incremental lanes; `PCA128_only` did not stably beat AR.

## Boundaries

- Continuous future-change MAE remains diagnostic only.
- Zero-change baselines still beat real cortical features in most continuous checks.
- Offset diagnostics should not be promoted into final scores unless a future train-only policy survives controls and grouped validation.
- Legacy validation branches and retired secondary model expansion are not active validation requirements for the v2 claim.
- Downstream product-model work is outside the current v2 evidence bundle.
- Full multimodal TRIBE remains a high-priority pilot, not part of the frozen v2 claim yet. The guarded `83,84` pilot reaches audio/text event preparation but is blocked until the gated `meta-llama/Llama-3.2-3B` text encoder is locally available or authorized.
- Do not use a full VEATIC-124 re-encode as the next multimodal step: only videos `83` and `84` contain audio streams.
- Dense AGAIN V-JEPA 2.1 / TRIBE v2 data generation has since completed on H100 for all `995` videos at `2Hz`, `256px`, float16. True 2Hz labels and a first raw/diagnostic AR-vs-cortical baseline now exist, but this is still not part of the VEATIC v2 evidence claim and is not dense PCA/bridge proof.

## Source Reports

- `benchmarks/veatic/veatic_124_confirmatory_benchmark_report_20260616.md`
- `benchmarks/veatic/veatic_124_retest_event_spike_core_20260616.md`
- `benchmarks/veatic/veatic_124_event_conditioned_retest_20260616.md`
- `outputs/veatic_124_temporal_context_v2_20260616_1557/veatic_124_temporal_context_v2_report.md`
- `outputs/veatic_124_temporal_fairness_20260616_1509/veatic_124_temporal_fairness_report.md`
- `benchmarks/veatic/veatic_124_alignment_lag_repair_20260616.md`
- `benchmarks/veatic/veatic_124_alignment_candidate_fixes.md`
- `benchmarks/veatic/veatic_124_alignment_causal_window_audit.md`
- `docs/veatic_raw_representation_audit.md`
- `outputs/veatic_124_raw_representation_tensor_export_v1/tensor_export_report.md`
- `backend/scripts/run_veatic_frozen_tensor_trained_heads_benchmark.py`
- `docs/again_dense_h100_cache.md`
