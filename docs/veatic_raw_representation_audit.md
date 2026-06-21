# VEATIC-124 Raw Representation Audit

Date: 2026-06-20

Run:

```bash
cd <repo-root>/backend
uv run python scripts/run_veatic_raw_representation_audit.py \
  --primary-audit \
  --skip-sensitivity \
  --output-dir "${NEURAL_BRIDGE_EXTERNAL_ROOT}/outputs/veatic_124_raw_representation_audit_primary_20260620_152411" \
  --tracked-output-dir "<repo-root>/outputs/veatic_124_raw_representation_audit_primary_20260620_152411"
```

This audit used existing cached TRIBE v2 raw cortical predictions only. No video re-encoding was performed and the frozen VEATIC v2 evidence bundle was not replaced.

Post-audit status: the tensor export and first MPS trained-head benchmark have both been implemented. Do not treat this document as saying learned heads still need to be created from scratch; use `backend/scripts/run_veatic_frozen_tensor_trained_heads_benchmark.py` for post-v2 trained-head work.

## Run Status

- Mode: `primary-audit`
- Scope: `all_videos`, 124/124 videos
- Video 83 sensitivity: skipped by design; it is opt-in with `--with-sensitivity`
- Elapsed time: 5458 seconds
- Raw cache inventory: 124/124 raw outputs present
- Alignment: 123 exact, 1 linear-resampled video (`83`)
- Modality reality: 122 video-only cache entries, 2 text+audio+video cache entries
- Leakage audit: pass
- Preflight: pass, including `nilearn`, `nibabel`, ROI atlas load, and MPS availability
- Checkpoint: finalized, 294/294 jobs complete, 0 skipped jobs

External output directory:

```text
${NEURAL_BRIDGE_EXTERNAL_ROOT}/outputs/veatic_124_raw_representation_audit_primary_20260620_152411
```

Lightweight repo copy:

```text
<repo-root>/outputs/veatic_124_raw_representation_audit_primary_20260620_152411
```

## Resumability and Reuse

The run is resumable from:

```text
${NEURAL_BRIDGE_EXTERNAL_ROOT}/outputs/veatic_124_raw_representation_audit_primary_20260620_152411/_checkpoint/state.json
```

Resume command:

```bash
cd <repo-root>/backend
uv run python scripts/run_veatic_raw_representation_audit.py \
  --primary-audit \
  --skip-sensitivity \
  --resume \
  --output-dir "${NEURAL_BRIDGE_EXTERNAL_ROOT}/outputs/veatic_124_raw_representation_audit_primary_20260620_152411" \
  --tracked-output-dir "<repo-root>/outputs/veatic_124_raw_representation_audit_primary_20260620_152411"
```

The audit also avoids repeated work within a run:

- Completed jobs are skipped on resume.
- Split-level PCA projections are cached in `_checkpoint/fit_cache`.
- The final run wrote 22 fit-cache files, 124 MB total.
- Row-derived AR/time/video-time matrices are reused for matching retained row sets.
- PCA projections are reused across candidates that share the same train split and component count.

## Main Results

Grouped-video validation is the promotion gate. Values below are grouped fold means.

| Candidate | Target | Threshold | PR-AUC | Baseline PR-AUC | Gain | Control pass | Positive folds vs AR |
|---|---|---:|---:|---:|---:|---|---:|
| `pca_sequence_128_causal_past_2s_mean` | `arousal__future_spike_1_3s` | 0.075 | 0.3185 | 0.2777 | +0.0408 | yes | 3/5 |
| `pca_sequence_128_causal_past_2s_mean` | `arousal__future_spike_1_3s` | 0.050 | 0.4251 | 0.3859 | +0.0393 | yes | 3/5 |
| `roi_parcel_features` | `arousal__future_change_p3s_movement` | 0.050 | 0.4621 | 0.4221 | +0.0400 | yes | 3/5 |
| `topk_vertices_512` | `arousal__future_spike_1_3s` | 0.075 | 0.3001 | 0.2777 | +0.0225 | yes | 4/5 |
| `roi_parcel_features` | `arousal__future_spike_1_3s` | 0.050 | 0.4039 | 0.3859 | +0.0180 | yes | 5/5 |
| `topk_vertices_512` | `arousal__future_spike_1_3s` | 0.050 | 0.4027 | 0.3859 | +0.0169 | yes | 5/5 |

The audit labels these as promoted by its grouped-validation rule, but they should still be treated as exploratory until confirmed in a locked follow-up run because this was a broad candidate search.

## Required Answers

1. Did raw uncompressed cortical ridge beat PCA64-delta?

No. `raw_current_ridge` did not beat same-run grouped `cortical_pca64_delta` on any primary target:

| Target | Threshold | Raw PR-AUC | PCA64-delta PR-AUC | Raw gain |
|---|---:|---:|---:|---:|
| `arousal__future_spike_1_3s` | 0.050 | 0.3856 | 0.3859 | -0.0003 |
| `arousal__future_spike_1_3s` | 0.075 | 0.2730 | 0.2777 | -0.0046 |
| `arousal__future_change_p3s_movement` | 0.050 | 0.4170 | 0.4221 | -0.0051 |

2. If raw wins, does it still beat shuffled/random high-dimensional controls?

Raw did not win. It passed controls for the two event-spike targets but failed the future-change control gate and stayed weaker than PCA64-delta.

3. Which compression is best?

For event/spike targets, `pca_sequence_128_causal_past_2s_mean` is the best candidate to build on. For future-change, `roi_parcel_features` is the best guarded compression. `topk_vertices_512` is useful but supervised and should remain exploratory until locked confirmation.

4. Is 64 PCA underfeeding compared with 128/256?

Not in a simple current-row PCA sense. `cortical_pca_64` remained stronger than `pca_current_128` and `pca_current_256` on grouped event targets. The gain came from causal 128-component sequence mean and ROI compression, not from simply increasing current-row PCA width.

5. Does 256 overfit or genuinely improve?

256-wide variants did not justify promotion. `pca_current_256`, `pca_delta_256`, and `pca_sequence_256_causal_past_2s_mean` were weaker than the best 128 sequence candidate and often failed control or AR-stability checks. Treat 256 as not yet useful.

6. Does causal PCA sequence beat single-row PCA64-delta?

Yes on grouped event targets. `pca_sequence_128_causal_past_2s_mean` beat `cortical_pca64_delta` by +0.0393 PR-AUC at threshold 0.05 and +0.0408 at threshold 0.075. Matched-row context gains over current-only features were small and mixed, so the result supports the representation as a next input contract, not a large standalone temporal-context claim.

7. Which representation should become the next tensor contract?

Retain `cortical_pca64_delta` as the frozen v2 baseline comparator. The post-audit v1 tensor export materializes `pca_sequence_128_causal_past_2s_mean` as the primary trained-head input, plus `roi_parcel_features` as a compact future-change branch and `topk_vertices_512` as supervised/cautionary.

8. Should the next learned head build on raw current, PCA128 sequence, PCA256 sequence, PLS, top-k, or PCA64-delta?

Use `pca_sequence_128_causal_past_2s_mean` as the primary learned-head input. Keep `cortical_pca64_delta` as the baseline comparator. Include `roi_parcel_features` for future-change tasks. Keep `topk_vertices_512` as exploratory. Do not prioritize raw current, PCA256 sequence, or PLS from this run. The first MPS trained-head benchmark for this tensor contract is already implemented; extend that runner rather than creating a parallel benchmark path.

9. Does excluding video 83 change the conclusion?

Not tested in this run. The full 124/124 audit intentionally kept video 83 and skipped the sensitivity branch. Excluding video 83 remains opt-in with `--with-sensitivity`; do not run it by default because it mostly repeats the full audit for a narrow sensitivity question.

10. Are gains explainable by row filtering, time/video identity, or high-dimensional nuisance?

The promoted event sequence and ROI results beat timestamp and video-time controls. The row-matched causal-window checks show only small or mixed context gains over current-only rows, so do not overclaim temporal intelligence. High-dimensional raw and several 256/top-k future-change results show nuisance warnings, which is why they are not the main recommendation.

## Recommendation

- Best frozen baseline retained: `cortical_pca64_delta`
- Best candidate for learned heads: `pca_sequence_128_causal_past_2s_mean`
- Best compact future-change branch: `roi_parcel_features`
- Best compression width to build on: 128 for causal PCA sequence
- Raw uncompressed useful: no, not as the next primary path
- Current benchmark underfed signal: yes for event ranking, but through causal 128 sequence/ROI structure rather than raw width
- Re-encoding needed yet: no
- Trained-head runner implemented: yes, under `backend/scripts/run_veatic_frozen_tensor_trained_heads_benchmark.py`

## Key Artifacts

- Full report: `${NEURAL_BRIDGE_EXTERNAL_ROOT}/outputs/veatic_124_raw_representation_audit_primary_20260620_152411/representation_audit_report.md`
- Leaderboard: `${NEURAL_BRIDGE_EXTERNAL_ROOT}/outputs/veatic_124_raw_representation_audit_primary_20260620_152411/raw_vs_compressed_leaderboard.csv`
- Promotion summary: `${NEURAL_BRIDGE_EXTERNAL_ROOT}/outputs/veatic_124_raw_representation_audit_primary_20260620_152411/candidate_promotion_summary.json`
- Leakage audit: `${NEURAL_BRIDGE_EXTERNAL_ROOT}/outputs/veatic_124_raw_representation_audit_primary_20260620_152411/leakage_audit.json`
- Run manifest: `${NEURAL_BRIDGE_EXTERNAL_ROOT}/outputs/veatic_124_raw_representation_audit_primary_20260620_152411/run_manifest.json`
