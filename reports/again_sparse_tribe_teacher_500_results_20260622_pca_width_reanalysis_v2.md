# AGAIN Sparse TRIBE Teacher 500 Results

## Scope

- This sparse teacher pilot uses `50` selected AGAIN videos out of `995` total videos.
- That is `5.0%` of the dataset.
- AR-only in this report is computed only on the same sparse pilot center rows as the sparse TRIBE/PCA rows.
- It is not the full-AGAIN AR baseline and must not be read as a 995-video comparison.
- AR + sparse PCA128 cannot be tested against all 995 videos until matching sparse PCA rows exist for that full scope.

## Executive Verdict
- Completed sparse ViT-G/TRIBE windows: `480`
- Hybrid AR-only PR-AUC: `40.19%`
- Hybrid AR + raw sparse current PR-AUC: `38.79%`
- Hybrid AR + raw sparse causal mean PR-AUC: `36.53%`
- Hybrid AR + sparse PCA128 causal PR-AUC: `24.90%`
- Hybrid AR + train-selected sparse PCA causal PR-AUC: `43.84%`
- Train-selected PCA widths by grouped outer fold: `32,16,16,8,32`
- Mean inner-validation PR-AUC for selected-width lane: `50.15%`
- Hybrid AR + telemetry + V-JEPA-B + sparse PCA128 causal PR-AUC: `24.90%`
- Coverage-random AR + sparse PCA128 causal PR-AUC: `52.98%`
- Oracle+background AR + sparse PCA128 causal PR-AUC: `57.45%`

## Smaller PCA Width Re-analysis

- This section is cache-only: it reuses existing sparse TRIBE window features and fits PCA on train rows only.
- Candidate widths are `8`, `16`, `32`, and `64`; the selected-width lane uses grouped train/inner validation only.
- PCA8: PR-AUC `46.42%`, mean actual width `8.0`
- PCA16: PR-AUC `48.24%`, mean actual width `16.0`
- PCA32: PR-AUC `52.48%`, mean actual width `32.0`
- PCA64: PR-AUC `39.51%`, mean actual width `64.0`

## Gate Summary
- sparse PCA128 vs AR-only: fail (delta -15.29 pp)
- sparse PCA128 vs AR + telemetry + V-JEPA-B: fail (delta -15.29 pp)
- sparse PCA128 vs raw sparse current: fail (delta -13.90 pp)
- train-selected small PCA vs AR-only: pass (delta 3.65 pp)
- train-selected small PCA vs raw sparse current: pass (delta 5.05 pp)
- train-selected small PCA vs raw sparse causal mean: pass (delta 7.31 pp)
- train-selected small PCA vs PCA64-delta analogue: pass (delta 7.62 pp)
- train-selected small PCA vs shuffled control: pass (delta 9.20 pp)
- train-selected small PCA vs random control: pass (delta 21.75 pp)
- train-selected small PCA vs coverage-random selected small PCA: pass (delta 19.99 pp)
- hybrid sparse vs coverage-random sparse: fail (delta -28.08 pp)

## Decision Rule
- This is a sparse teacher pilot only, not final AGAIN proof.
- PCA128 remains a negative sparse-sample lane here; do not scale it as the next sparse teacher representation.
- Treat the train-selected small PCA lane as the current follow-up candidate only if it beats AR, raw sparse current/causal mean, PCA64-delta, and shuffled/random controls.
- The selected small PCA lane passes the local sparse controls and its same-lane coverage-random control; larger sparse-teacher runs still need fresh grouped validation before promotion.
