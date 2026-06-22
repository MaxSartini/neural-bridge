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
- Reported PCA-width rows are AR + sparse PCA lanes, not PCA-only lanes.
- Candidate widths are `8`, `16`, `32`, and `64`; the selected-width lane uses grouped train/inner validation only.
- AR + sparse PCA8: PR-AUC `25.63%`, mean actual width `8.0`
- AR + sparse PCA16: PR-AUC `23.64%`, mean actual width `16.0`
- AR + sparse PCA32: PR-AUC `27.65%`, mean actual width `32.0`
- AR + sparse PCA64: PR-AUC `32.99%`, mean actual width `64.0`

## Gate Summary
- sparse PCA128 vs AR-only: pass (delta 5.93 pp)
- sparse PCA128 vs AR + telemetry + V-JEPA-B: pass (delta 9.40 pp)
- sparse PCA128 vs raw sparse current: fail (delta -0.72 pp)
- locked PCA32 vs AR-only: pass (delta 6.28 pp)
- locked PCA32 vs raw sparse current: fail (delta -0.36 pp)
- locked PCA32 vs raw sparse causal mean: fail (delta -0.30 pp)
- locked PCA32 vs AR + telemetry + V-JEPA-B: pass (delta 9.74 pp)
- AR + telemetry + V-JEPA-B + locked PCA32 vs AR + telemetry + V-JEPA-B: pass (delta 8.38 pp)
- AR + locked PCA32 vs AR + shuffled sparse PCA32 control: fail (delta -1.41 pp)
- AR + locked PCA32 vs AR + random Gaussian PCA32 control: pass (delta 5.40 pp)
- AR + train-selected small PCA vs AR-only: pass (delta 0.58 pp)
- AR + train-selected small PCA vs AR + telemetry + V-JEPA-B: pass (delta 4.04 pp)
- AR + telemetry + V-JEPA-B + train-selected small PCA vs AR + telemetry + V-JEPA-B: pass (delta 3.26 pp)
- AR + train-selected small PCA vs AR + raw sparse current: fail (delta -6.06 pp)
- AR + train-selected small PCA vs AR + raw sparse causal mean: fail (delta -6.00 pp)
- AR + train-selected small PCA vs AR + PCA64-delta analogue: fail (delta -1.92 pp)
- AR + train-selected small PCA vs AR + shuffled sparse control: fail (delta -3.96 pp)
- AR + train-selected small PCA vs AR + random Gaussian sparse control: fail (delta -0.91 pp)
- AR + train-selected small PCA vs fixed coverage-random AR + selected small PCA: fail (delta -15.59 pp)
- AR + train-selected small PCA vs true same-budget fixed-random AR + selected small PCA: fail (delta -28.98 pp)
- locked PCA32 vs coverage-random PCA32: fail (delta -8.62 pp)
- AR + locked PCA32 vs true same-budget fixed-random AR + PCA32: fail (delta -25.56 pp)
- hybrid sparse vs coverage-random sparse: fail (delta -9.73 pp)

## Delta-over-AR Matched-Arm Checks

- These checks compare each arm's improvement over its own AR baseline, avoiding unfair absolute PR-AUC comparisons across selector distributions.
- AR + locked PCA32 hybrid delta-over-AR vs coverage-random delta-over-AR: pass (delta 14.78 pp)
- AR + locked PCA32 hybrid delta-over-AR vs true fixed-random delta-over-AR: fail (delta -0.16 pp)
- AR + train-selected small PCA hybrid delta-over-AR vs coverage-random delta-over-AR: pass (delta 7.81 pp)
- AR + train-selected small PCA hybrid delta-over-AR vs true fixed-random delta-over-AR: fail (delta -3.58 pp)

## Methodology Correction

- V-JEPA 2.1 is the video-window encoding engine for TRIBE v2.
- The sparse feature being PCA-compressed is the frozen TRIBE v2 cortical prediction vector, not raw V-JEPA tokens and not a PCA-only model.
- All PCA/scalers are fit on grouped-train rows only and then applied to test rows.
- Absolute PR-AUC comparisons across selector arms are diagnostic only because each selector arm has a different AR baseline and event mix.
- The fairer matched-arm check is delta-over-AR: compare each sparse lane against its own arm-local AR baseline, then compare those improvements across arms.
- Under delta-over-AR, AR + locked PCA32 beats coverage-random in this full-arm run, but the old undersized fixed-random arm remains an unsuitable promotion gate.
- AR + locked PCA32 still fails the same-width shuffled PCA32 nuisance control and remains slightly below AR + raw sparse current/causal mean, so it is not clean promotion evidence.

## Decision Rule
- This is a sparse teacher pilot only, not final AGAIN proof.
- Promote nothing unless the sparse lane beats AR, raw sparse current/causal mean, cheap AR+telemetry+V-JEPA-B, shuffled/random nuisance controls, and matched-random sparse controls.
- If a lane beats AR but loses to raw sparse, shuffled/random, or matched-random controls, treat it as non-confirmed sparse-sample signal.
- Do not approve full AGAIN scaling from this sparse pilot alone.
