# VEATIC 2.1 AGAIN-Method Parity Inner Discovery

Date: `2026-07-17`  
Run identity: `cab3b6d7f67a348e680798012ced523cb498cb2c551b2f6388ca833fb0376732`  
Output root: `/Volumes/onn. Drive/Neural Bridge/outputs/veatic21_again_parity_inner_discovery_20260717`  
Status: complete inner-only development; no outer-test scores used; no recipe promoted

## Plain-language verdict

The run found a meaningful event-ranking lead, but not yet a robust standalone
recipe.

The fixed three-checkpoint `temporal_mean_2s_pca256_again_clean_joint`
ensemble reached PR-AUC `0.3183913271` versus its equally ensembled frozen AR
at `0.3127116235`. The absolute gain was `+0.0056797035`, or `+1.82%`
relative to the already-strong AR floor. It won `11/15` whole-video inner
panels, had a positive median delta of `+0.0060286143`, and had positive mean
deltas in `4/5` outer partitions. It therefore passed the preregistered
ensemble-specific credibility gate.

The individual checkpoints did not generalize reliably. Their mean PR-AUC was
`0.3005580415` versus member AR `0.3087399646`, a loss of `-0.0081819231`
(`-2.65%` relative), with `21/45` strict wins, `4/45` exact AR fallbacks, and
only `1/5` positive outer means. The three-checkpoint ensemble added
`+0.0178332856` PR-AUC over the member mean, but the preregistration required
both member and ensemble credibility. Consequently:

- the fixed ensemble is a credible inner-development event lead;
- the full recipe did not pass its combined promotion gate;
- `any_credible_recipe` is `false`;
- outer confirmation and canonical training remain unauthorized.

This is an honest partial win: the causal video-side signal survives after
averaging, but the current training/checkpoint process is too unstable to call
the mechanism cracked.

## Exact event results

| Recipe | Member delta vs AR | Member wins | Ensemble delta vs AR | Relative ensemble lift | Ensemble wins | Positive outer means | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `delta_pca64_again_clean_joint` | `-0.0031242042` | `21/45` | `+0.0018407640` | `+0.59%` | `10/15` | `3/5` | near miss |
| `delta_pca64_veatic_enriched_joint` | `-0.0035637946` | `24/45` | `-0.0007736835` | `-0.25%` | `11/15` | `2/5` | fail |
| `delta_pca128_again_clean_joint` | `-0.0024788752` | `20/45` | `+0.0007365163` | `+0.24%` | `11/15` | `3/5` | below gain/fold gates |
| `delta_pca128_veatic_enriched_joint` | `-0.0020094968` | `18/45` | `+0.0009208358` | `+0.29%` | `11/15` | `3/5` | below gain/fold gates |
| `temporal_mean_2s_pca256_again_clean_joint` | `-0.0081819231` | `21/45` | `+0.0056797035` | `+1.82%` | `11/15` | `4/5` | ensemble credible; members fail |
| `temporal_mean_2s_pca256_veatic_enriched_joint` | `-0.0102267942` | `19/45` | `-0.0011036945` | `-0.35%` | `5/15` | `3/5` | fail |

The delta-PCA64 clean ensemble was the runner-up. It cleared the minimum mean
gain, median, and `10/15` win gates, but only `3/5` outer means were positive.
PCA128 did not beat PCA64 decisively. Adding VEATIC availability/time features
generally hurt the event result; the clean input was better for both leading
ensembles.

The selected temporal-clean ensemble's five outer mean deltas were
`+0.004759`, `+0.001545`, `-0.007754`, `+0.023110`, and `+0.006739`. The main
weakness was outer partition 3, especially inner panel 2 at `-0.032924`.

## Why the combined gate failed

The result is dominated by checkpoint variance, not an epoch ceiling or an
artificial contract failure.

- residual best epochs extended as high as `613` and residual curves as high
  as `713` epochs;
- AR best epochs extended as high as `646` and AR curves as high as `746`;
- no model reached the `5,000` runaway fail-safe;
- only `18/270` residual members selected exact zero correction;
- the selected temporal-clean members included losses as low as `-0.099640`
  and wins as high as `+0.042307`;
- equal averaging turned those unstable members into the best result and added
  `+0.0178332856` over their mean.

The no-harm selector protects a cell when its deeper training-side validation
finds no positive correction. It cannot guarantee that a selected positive
checkpoint will generalize to the separate whole-video inner panel. That
remaining selection variance is the next technical problem.

## Exploratory continuous readout

The event preregistration did not use continuous metrics as selection or
promotion gates. The following numbers were computed after completion from the
already sealed continuous predictions and are exploratory diagnostics only.

The event-winning temporal-clean ensemble did not transfer directly to
continuous ranking:

- Spearman: `0.2951604916` versus AR `0.3075361674`, delta `-0.0123756758`,
  positive `7/15` panels;
- top-5% lift: `0.0857538998` versus AR `0.0871881783`, delta
  `-0.0014342785`, positive `9/15` panels.

The temporal-enriched ensemble, which failed the event gate, produced a small
but inconsistent continuous hint:

- Spearman delta `+0.0016688840`, positive `7/15` panels;
- top-5% lift delta `+0.0011831303`, positive `10/15` panels.

These diagnostics support sequential target-specific work. The event winner
should not be declared a continuous winner, and one shared recipe should not be
forced across spike, continuous arousal, and valence.

## Data, quality, and integrity audit

The completed audit passed every executable check:

- matrix completeness: `270/270` members and `90/90` ensembles;
- unique member and ensemble identities: pass;
- no outer-test scores used: pass;
- VEATIC 2.1-only PCA identity/checksums: pass;
- no AGAIN data, PCA, scores, normalizers, checkpoints, or weights reused:
  pass;
- held-out label and valid-row digests aligned across recipes/seeds/ensembles:
  pass;
- pooled zero-event-safe metric policy: pass;
- finite metrics and eval-mode scoring: pass.

The source remains exactly `124` unique VEATIC videos and `20,657` dense rows.
The quality preflight excluded `923` unusable causal-window rows: `76`
black-window rows and `871` highly duplicated/static rows, with overlap. The
same mask governed PCA fitting, training, checkpoint selection, AR scoring, and
bridge scoring. Across the 15 held-out panels, 95 of 496 repeated video-panel
appearances contained no positive event. None received a fabricated PR-AUC of
zero; their valid negative rows remained in pooled panel PR-AUC so false alarms
still counted.

No new V-JEPA or TRIBE cache run is needed.

## Recommended next inner-only stage

Do not open outer tests and do not run broad Optuna yet. The strongest evidence
points to checkpoint stabilization around the clean temporal family rather
than same-family hyperparameter search.

The next bounded study should use fresh seeds and treat the fixed ensemble as
the actual candidate model, with gates locked before execution. Recommended
structural candidates are:

1. `temporal_mean_2s_pca256` clean with a fixed five-checkpoint ensemble;
2. `temporal_mean_2s_pca128` clean to test whether lower width stabilizes the
   temporal family;
3. `delta_pca64` clean as the current runner-up comparator;
4. `delta_pca256` clean to complete the most relevant missing width cross.

Each candidate should use fresh member seeds, an equally ensembled AR floor,
the same pooled quality/zero-event policy, and strict panel/fold consistency
gates. Continuous movement should be stored as a preregistered secondary
diagnostic, not allowed to alter the event verdict. If a fresh stabilized event
ensemble wins, then a narrow inner-only Optuna search may tune that proven
family. If it does not, structural discovery should continue before Optuna.

