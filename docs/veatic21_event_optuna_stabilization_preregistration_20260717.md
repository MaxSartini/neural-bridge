# VEATIC 2.1 Event Optuna Stabilization Preregistration

Date: 2026-07-17  
Status: preregistered before Optuna or showdown scoring  
Scope: exploratory, inner-only, explicitly non-promotable

## Question

Can residual-learning parameters stabilize the clean
`temporal_mean_2s_pca256_again_clean_joint` event model beyond both its untuned
configuration and the exact frozen dual-task AR baseline?

No outer-test score may be read. Continuous and valence metrics are outside this
study.

## Locked substrate

- VEATIC 2.1 only: 124 videos and 20,657 dense 2 Hz rows.
- Target: `future_arousal_max_delta_rows_4_10`.
- Event label: training-scope q90 of non-negative future movement.
- Quality mask, zero-event-video policy, pooled event PR-AUC, fold-safe PCA-256,
  clean temporal input, dual-task AR, full-dense training, and zero-correction
  candidate remain fixed.
- Valid negative rows from zero-event videos remain in pooled PR-AUC. No fake
  per-video zero score is permitted.
- Batch size 1,024, minimum training epoch 50, patience 100, and maximum epoch
  5,000 remain fixed. Epoch 5,000 is a runaway fail-safe, not a target.
- Training and checkpoint selection begin at epoch 1. Early stopping is
  forbidden before epoch 50. Epoch 50 is minimum training depth, not a warm-up
  that discards earlier best checkpoints.
- AR is trained once per panel and seed, then reused unchanged by every residual
  lane in that cell.

## Optuna development scope

Exactly 50 completed trials use these five prespecified inner panels, one from
each outer partition:

`(1,1), (2,2), (3,3), (4,1), (5,2)`.

Development seeds are `20260716`, `20260717`, and `20260718`. Their compatible
frozen AR score bundles may be read from the sealed 2026-07-17 parity run only
after identity, ownership, array-digest, and checksum validation. No residual
checkpoint or score is reused.

Original parameters are enqueued as trial 0. Remaining trials use seeded
multivariate TPE with 10 startup trials. No pruning, parallel GPU competition,
trial deletion, or post-hoc trial substitution is allowed. SQLite storage and
per-trial artifacts make the run resumable; resume must stop at 50 completed
trials rather than add 50 more.

Search space:

- hidden width: `48, 64, 96, 128`;
- learning rate: log-uniform `[5e-5, 5e-4]`;
- weight decay: log-uniform `[1e-5, 1e-3]`;
- event BCE weight: `0.35, 0.50, 0.65, 0.80, 1.00`;
- residual alpha initial logit: `-6, -5, -4, -3`;
- residual alpha cap: `0.04, 0.06, 0.08, 0.12, 0.16`;
- row-gate bias: `3, 4, 5, 6`.

Batch size, epoch policy, patience, alpha penalty, architecture family, temporal
window, PCA width, input columns, labels, AR, and ensemble weights are not tuned.

For each trial and panel, logits from all three development seeds are averaged
before ensemble PR-AUC. Let `d` be the five ensemble deltas versus equally
averaged frozen AR, `m` the mean member delta versus AR, and `u` the mean
ensemble uplift over member mean. Maximize:

`0.30 mean(d) + 0.20 median(d) + 0.15 q25(d) + 0.15 min(d) - 0.10 std(d) + 0.05 m + 0.05 u + 0.002 win_rate(d)`.

This rewards average gain, broad wins, worst-panel behavior, stable members, and
ensemble benefit. It does not optimize the best single panel or checkpoint.

After exactly 50 completed trials, freeze exactly one best trial. Freeze payload
must seal parameters, objective, trial number, completed-study digest,
preregistration digest, and code identity before any showdown score is produced.

## Apples-to-apples showdown

Fresh seeds are `20260719` through `20260723`. Every one of the 15 inner panels
is scored with fixed equal five-checkpoint averaging. Ten panels never used by
Optuna are the primary estimate; all 15 panels are a secondary precision view.

Each panel/seed trains one epoch-50-eligible frozen AR. Three residual lanes use
identical data, AR predictions, seeds, epoch ceiling, patience, batch size,
metric, and ensemble rule:

1. `tuned`: frozen Optuna parameters under the corrected current training system;
2. `original`: original parameters under the identical corrected current system.

Required comparisons:

- Optuna uplift: `tuned - original`;
- every lane versus the identical frozen AR ensemble.

The old 2026-07-17 three-seed parity result may appear only as a historical,
non-paired reference.

## Success gate

Primary held-back ten-panel gate requires:

- tuned ensemble mean and median exceed `original` and AR;
- tuned wins at least 7/10 panels versus each;
- at least 4/5 outer-fold mean deltas are positive versus each;
- tuned ensemble variability is no worse than `original`;
- tuned mean ensemble uplift over its member mean is positive;
- tuned member mean exceeds both `original` and AR, with at least 30/50
  member wins versus each;
- every schedule, provenance, PCA, quality, zero-event, label, checksum,
  eval-mode, frozen-AR, and outer-closure audit passes.

Failure retains the original ensemble and triggers structural comparison of
clean temporal PCA128, delta PCA256, and delta PCA64 before another parameter
search. Success remains inner-only evidence and does not authorize outer-test
confirmation.
