# VEATIC 2.1 Supervised Spike + Continuous Combination

Status: current Phase 02 structural specification

## Objective

Build one versioned supervised arousal package from the sealed VEATIC 2.1 rows. The package
contains an independently trained spike specialist ensemble and continuous specialist
ensemble behind one inference interface. This is the strongest supervised configuration,
not a zero-label configuration and not a minimal-capacity baseline.

Every fitted value is VEATIC-owned. The structure below is fixed; dataset-dependent numbers
are derived or selected from VEATIC training ownership.

## Sole input path

Every computation reads only:

`/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/neural-bridge-input/`

It must enumerate the exact numeric directory set `per_video/0` through `per_video/123`.
For every video `v`, labels and row identity come from `per_video/v/rows.csv`, and the primary
representation comes from that same directory's
`tribe_v2_cortical_predictions.npz:cortical_prediction`. No pooled payload, staging-root
read, partial video subset, filesystem-order join, or cross-video row join is permitted.

## Dependency graph

The build order is mandatory because every downstream object must name and checksum its
upstream dependencies:

1. sealed `(video_id, row_index)` table and label/audit columns;
2. exact blocked, inner, and grouped ownership;
3. same-video future-trajectory tensors and eligible-row masks;
4. event target and continuous target definitions;
5. continuous-target residualizer, when selected by the target definition;
6. event response-history opponent and continuous response-history opponent;
7. causal cortical temporal aggregates;
8. fold-owned scaler and projection for each distinct target mask/ownership;
9. causal projected sequences plus the explicit diagnostic block;
10. event residual checkpoints and matched controls;
11. continuous residual checkpoints and matched controls;
12. fixed equal-weight checkpoint ensembles;
13. the paired spike + continuous package and immutable outer predictions.

No downstream artifact may be constructed until every required parent has passed row,
ownership, leakage, and checksum verification.

## Computation 1: targets and ownership

For each Phase 01 arousal geometry `(washout_rows, horizon_rows)` and label row `i`, compute
the future label trajectory once from later rows in the same video. The supervised arousal
structural target is maximum positive arousal movement after a nonzero washout. Phase 01 may
audit other views to characterize VEATIC, but they do not silently become Phase 02 target
candidates. The event target is a training-owned quantile threshold over maximum positive
movement. The continuous target is the residualized maximum-positive-movement value and may
select a different registered geometry from the event specialist.

Phase 01 determines the VEATIC washout, horizon, and threshold candidates from label
dynamics, target support, and threshold stability. Every event threshold remains bound to
its exact geometry and must retain broad event-video support in every grouped fold. No
numeric window or quantile is preselected.

Every target artifact records:

- source label and exact mathematical definition;
- history, washout, and future row intervals;
- eligible and excluded row counts per video;
- target threshold and the rows that owned its fit;
- blocked/inner/grouped membership hashes;
- event prevalence, zero-event videos, and duration-panel support.

## Computation 2: continuous-target residualizer

The continuous structural hypothesis ranks movement unexplained by a simple legal history
model. For each ownership, fit a standardized regularized linear residualizer using only
causal arousal-history summaries available at prediction time. Candidate history summaries
come from Phase 01 and include current/lagged levels, trailing means, slopes or differences,
time fraction, and availability masks where supported.

The residual target equals `future_movement - residualizer_prediction`. Its scaler,
regularization, coefficients, predictions, and row mask belong to that ownership. If the
selected event target and continuous residual target have different valid rows, every later
projection and head is fitted separately for them.

## Computation 3: strong response-history opponents

Train two separate learned opponents from causal response history only.

The event opponent has a continuous-output branch and event-logit branch and is selected on
inner event ranking. The continuous opponent is trained for the continuous target with
tail-aware regression and selected lexicographically on inner top-tail lift, Spearman, and a
secondary registered tail metric. Both opponents receive adequate convergence budgets and
VEATIC-selected capacity, dropout, learning rate, weight decay, and patience.

For every split and seed, restore the selected checkpoint in deterministic evaluation mode,
write train/inner/outer predictions once, and seal their hashes. The identical event or
continuous opponent prediction is then used under the corresponding real residual lane and
all controls.

## Computation 4: fold-owned cortical projection

Start only from each matching per-video `cortical_prediction`. For every distinct target
mask and outer ownership:

1. build a same-video causal trailing cortical aggregate using the Phase 01 temporal-scale
   candidate set;
2. fit per-feature centering and scaling on outer-training rows only;
3. fit the maximum registered projection basis on those rows only;
4. transform inner and outer rows with the frozen fit;
5. expose nested legal widths without refitting when slicing is mathematically identical.

The local VEATIC projection candidate set is derived from training rank, explained variance,
memory, throughput, and active-boundary behaviour. Projection metadata records components,
mean, scale, explained variance, source rows, transformed rows, algorithm, seed, and hashes.

## Computation 5: causal specialist input

For row `i`, assemble a same-video causal sequence of projected rows ending at `i`. Missing
history at video starts is deterministically zero padded with explicit availability. Append
the current `temporal_diagnostics53` row as a separately named block. Standardize the final
head input on owned training rows only.

The short causal temporal-convolution hypothesis applies a shared causal kernel over the
projected sequence, uses the representation at the current sequence position, concatenates
the diagnostic block, and applies a post-projection hidden layer. Kernel width, sequence
length, projected width, and hidden width are VEATIC-selected local values.

## Computation 6: event residual head

For input `x`, frozen event-opponent logit `a`, and frozen event-opponent continuous output
`r`, the event head computes hidden state `h(x)`, two raw corrections `(c_reg, c_event)`, an
input gate `g(x)`, and a bounded learned scale `s`:

`event_continuous = r + s * g(x) * c_reg`

`event_logit = a + s * g(x) * c_event`

The loss combines robust continuous regression, weighted event classification/ranking, and
a scale penalty. Checkpoint selection is event-primary and inner-owned. If no checkpoint
beats the frozen opponent under the registered no-harm rule, the residual is suppressed and
the output is exactly the opponent.

## Computation 7: continuous residual head

Train a separate instance with the continuous target, continuous opponent, target-specific
projection, target-specific optimizer state, and independent seeds. Its primary output is:

`continuous_score = continuous_opponent + s * g(x) * c_continuous`

Training uses robust tail-weighted continuous regression, an opponent-anchor penalty, and a
bounded-scale penalty. Checkpoint selection is inner top-tail lift first, Spearman second,
and the registered secondary tail endpoint third. Event PR-AUC and top-k event recall derived
from `continuous_score` are recorded as explicit joint evidence.

## Computation 8: matched controls

For each specialist, train and score these lanes with the same rows, frozen opponent, budget,
checkpoint rule, seed, and metric code:

- real projected cortical residual;
- shuffled projected cortical residual;
- matched random projected residual;
- diagnostics-only residual;
- train-only video-mean residual;
- time/mask residual;
- luma/motion/quality residual;
- current-row cortical ablation;
- training-owned label-permutation residual;
- frozen opponent only.

Label permutation changes only residual-training targets. The opponent remains byte-identical,
so its null is no improvement beyond the opponent.

## Computation 9: independent-checkpoint ensembles

After each specialist recipe is frozen, train fresh independent checkpoints. Candidate
ensemble counts are derived from VEATIC seed variance and registered before outer scoring.
Average aligned continuous predictions and logits with equal weights. Never select members
or weights from outer outcomes. Compare each ensemble with every member, its frozen-opponent
ensemble, and matched-control ensembles.

## Development and confirmation

Unit and integrity tests precede computation. The development run uses every eligible row
from all 124 videos under blocked-forward and grouped-video ownership. Dataset-dependent
values use compact local candidate sets with boundary expansion only when the winner lies on
an active edge or remains undertrained.

Freeze the complete event and continuous recipes before fresh confirmation. Blocked and
grouped confirmations are separate. Each specialist must beat its exact opponent and the
strongest matched control, show positive fold/checkpoint-group consistency, avoid single-
group domination, and pass projection, row, checkpoint, evaluation-mode, and prediction-hash
audits.

## Final package

The final package exposes:

- `spike_score` from the event specialist ensemble;
- `continuous_score` from the continuous specialist ensemble;
- optional event ranking derived from `continuous_score` as a separately identified output;
- uncertainty and provenance for both outputs.

A later shared-head challenger may replace the pair only if it matches or beats the locked
event specialist on event endpoints and the locked continuous specialist on continuous
endpoints. Zero-label inference is outside this specification.
