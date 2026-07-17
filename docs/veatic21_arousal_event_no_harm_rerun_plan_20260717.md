# VEATIC 2.1 Arousal Event No-Harm Rerun Plan

Date: 2026-07-17  
Status: preregistered inner-only development rerun  
Outer-test confirmation: unauthorized

## Why this rerun exists

The corrected-depth six-recipe run validly completed `270/270`, selected `delta_pca64_short_conv` in all `5/5` outer development panels, and lost to matched frozen AR by `-0.0068741007` PR-AUC (`-2.23%`), with only `10/45` paired wins.

Comparison with the promoted AGAIN implementation found one material porting omission. AGAIN treats the untouched frozen-AR prediction as an explicit zero-correction checkpoint. A residual checkpoint is restored only if its inner-validation metric beats frozen AR; otherwise the residual is suppressed and inference returns AR exactly. The VEATIC port froze AR but forced selection of a residual checkpoint even when every candidate checkpoint was worse than AR.

This rerun repairs that omission without weakening the corrected training-depth schedule.

## Locked scope

- target: `future_arousal_max_delta_rows_4_10`
- endpoint: train-only-q90 binary event ranking
- protocol: `privileged_binary`
- data: all `124` VEATIC videos and all `20,657` dense 2 Hz rows under grouped ownership
- matrix: five outer partitions × three inner folds × six recipes × three seeds = `270` score rows
- outer-test scores used: false
- explicitly nonpromotable: true
- confirmation authorized: false

All six original recipes rerun. The repair may change which architecture benefits, so the previous winner is not given exclusive access to the corrected policy.

## Locked training and selection policy

- batch size: `1,024`
- maximum epochs: `5,000`, runaway fail-safe only
- minimum training duration before early stopping: `50` epochs
- patience after the minimum-training boundary: `100` stale validation epochs
- checkpoint eligibility: epoch `1`
- objective: true BCE with logits
- deterministic restored-checkpoint eval-mode scoring: required
- residual global cap and row gate: retained
- selection metric for a residual: inner-validation PR-AUC delta versus its exact matched frozen-AR offset
- zero correction: explicit candidate with selection delta `0.0`
- residual suppression: required when no checkpoint exceeds frozen AR by the locked selection minimum delta

An early checkpoint may be restored only after the model received the full minimum-duration and patience-controlled opportunity to improve. A restored epoch of 1 therefore means “the earliest state generalized best after a proper training search,” not “training stopped after one epoch.” A suppressed result uses best epoch `0` solely as a manifest sentinel for untouched AR; it is not a zero-epoch neural training run.

## Reuse boundary

Existing sealed PCA and immutable feature artifacts may be reused only when their provenance identity and checksums match. AR and residual model artifacts use a new scientific run identity because the registered checkpoint-selection policy changed. No fitted AGAIN PCA, AR, threshold, head, or weight is imported; only its proven algorithmic safeguard is ported.

## Fixed interpretation gates

The artifact/matrix audit is independent of model performance and must pass `270/270`.

For a fixed recipe to count as a credible inner-development win over AR, all must hold:

1. mean real-minus-AR PR-AUC at least `+0.001` across its `45` cells;
2. paired median delta greater than `0`;
3. at least `30/45` strictly positive cells;
4. at least `4/5` outer-panel mean deltas positive.

A strong inner-development win additionally requires at least `36/45` strictly positive cells and `5/5` positive outer-panel means. Exact AR fallbacks count as ties, not wins.

Failing these gates keeps arousal event ranking open and prohibits continuous, valence, or outer confirmation. Passing them authorizes only the next inner-development decision; it does not itself create promoted evidence.

## Next branch if this repair still loses

Keep the best representation/head and preregister a separate two-stage residual learner: learn the cortical residual against frozen-AR errors with an open learning path, then fit a train-only bounded correction coefficient that may choose zero. Do not add that method to this rerun post hoc.
