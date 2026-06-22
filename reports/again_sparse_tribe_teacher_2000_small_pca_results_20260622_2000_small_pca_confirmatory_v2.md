# AGAIN Sparse TRIBE Teacher 2000 Small PCA Results

## Scope

- This sparse teacher pilot uses `50` selected AGAIN videos out of `995` total videos.
- That is `5.0%` of the dataset.
- Actual expensive-window budget: `2000`.
- Existing 100-video selector cache was not available, so this run expands sparse coverage on the corrected 50-video selector subset.
- AR-only in this report is computed only on the same sparse pilot center rows as the sparse TRIBE/PCA rows.
- It is not the full-AGAIN AR baseline and must not be read as a 995-video comparison.
- AR + sparse PCA128 cannot be tested against all 995 videos until matching sparse PCA rows exist for that full scope.

## Executive Verdict
- Completed sparse ViT-G/TRIBE windows: `1948`
- Hybrid AR-only PR-AUC: `21.37%`
- Hybrid AR + raw sparse current PR-AUC: `28.02%`
- Hybrid AR + raw sparse causal mean PR-AUC: `27.95%`
- Hybrid AR + sparse PCA128 causal PR-AUC: `27.30%`
- Hybrid AR + locked sparse PCA32 causal PR-AUC: `27.65%`
- Hybrid AR + train-selected sparse PCA causal PR-AUC: `21.95%`
- Train-selected PCA widths by grouped outer fold: `8,8,8,16,64`
- Mean inner-validation PR-AUC for selected-width lane: `30.35%`
- Hybrid AR + telemetry + V-JEPA-B + locked sparse PCA32 causal PR-AUC: `26.29%`
- Hybrid AR + telemetry + V-JEPA-B + train-selected sparse PCA causal PR-AUC: `21.18%`
- Hybrid AR + telemetry + V-JEPA-B + sparse PCA128 causal PR-AUC: `27.31%`
- Coverage-random AR + sparse PCA128 causal PR-AUC: `37.03%`
- Oracle+background AR + sparse PCA128 causal PR-AUC: `67.28%`

## Smaller PCA Width Re-analysis

- This section is cache-only: it reuses existing sparse TRIBE window features and fits PCA on train rows only.
- Candidate widths are `8`, `16`, `32`, and `64`; the selected-width lane uses grouped train/inner validation only.
- PCA8: PR-AUC `25.63%`, mean actual width `8.0`
- PCA16: PR-AUC `23.64%`, mean actual width `16.0`
- PCA32: PR-AUC `27.65%`, mean actual width `32.0`
- PCA64: PR-AUC `32.99%`, mean actual width `64.0`

## Gate Summary
- sparse PCA128 vs AR-only: pass (delta 5.93 pp)
- sparse PCA128 vs AR + telemetry + V-JEPA-B: pass (delta 9.40 pp)
- sparse PCA128 vs raw sparse current: fail (delta -0.72 pp)
- locked PCA32 vs AR-only: pass (delta 6.28 pp)
- locked PCA32 vs raw sparse current: fail (delta -0.36 pp)
- locked PCA32 vs raw sparse causal mean: fail (delta -0.30 pp)
- locked PCA32 vs AR + telemetry + V-JEPA-B: pass (delta 9.74 pp)
- AR + telemetry + V-JEPA-B + locked PCA32 vs AR + telemetry + V-JEPA-B: pass (delta 8.38 pp)
- train-selected small PCA vs AR-only: pass (delta 0.58 pp)
- train-selected small PCA vs AR + telemetry + V-JEPA-B: pass (delta 4.04 pp)
- AR + telemetry + V-JEPA-B + train-selected small PCA vs AR + telemetry + V-JEPA-B: pass (delta 3.26 pp)
- train-selected small PCA vs raw sparse current: fail (delta -6.06 pp)
- train-selected small PCA vs raw sparse causal mean: fail (delta -6.00 pp)
- train-selected small PCA vs PCA64-delta analogue: fail (delta -1.92 pp)
- train-selected small PCA vs shuffled control: fail (delta -6.93 pp)
- train-selected small PCA vs random control: fail (delta -5.41 pp)
- train-selected small PCA vs coverage-random selected small PCA: fail (delta -15.59 pp)
- train-selected small PCA vs fixed-random same-budget selected small PCA: fail (delta -28.98 pp)
- locked PCA32 vs coverage-random PCA32: fail (delta -8.62 pp)
- locked PCA32 vs fixed-random same-budget PCA32: fail (delta -25.56 pp)
- hybrid sparse vs coverage-random sparse: fail (delta -9.73 pp)

## Consistency

- Locked PCA32 beat AR-only in `4/5` grouped-video folds, but failed against raw sparse current, raw sparse causal mean, coverage-matched random PCA32, and fixed-random same-budget PCA32.
- Train-selected small PCA beat AR-only in only `2/5` grouped-video folds and beat split-local shuffled selected PCA in `0/5` folds.
- The corrected cheap-fusion baseline (`AR + telemetry + V-JEPA-B`) scored `17.91%`, below AR-only at `21.37%`; adding locked PCA32 reached `26.29%`, still below raw sparse current and matched-random controls.
- Hybrid arm coverage used `50` videos across `9` games, with `344` hybrid candidate centers and `72` hybrid spike events at the center row.

## Decision Rule
- This is a sparse teacher pilot only, not final AGAIN proof.
- Promote nothing unless the sparse lane beats AR, raw sparse current/causal mean, cheap AR+telemetry+V-JEPA-B, shuffled/random nuisance controls, and matched-random sparse controls.
- If a lane beats AR but loses to raw sparse, shuffled/random, or matched-random controls, treat it as non-confirmed sparse-sample signal.
- Do not approve full AGAIN scaling from this sparse pilot alone.

## Recommendation

- Do not approve broader AGAIN scaling from this run.
- Do not use train-selected small PCA as the next locked feature lane; it failed nuisance and matched-random controls.
- PCA32 remains a useful diagnostic compressed bridge, but it did not confirm as a promotion lane because raw sparse and matched-random sparse controls were stronger.
