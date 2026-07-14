# AGAIN zero-label video-only deployment bridge — Stage A

The explicitly authorized Stage A development screen completed `96/96` scored
rows (`72` member plus `24` ensemble) on MLX GPU/MPS. All target, split,
train-only PCA, nested teacher ownership, prediction-before-label, cold-start,
finite coverage, and rollout audits passed. The 299-video prospectively locked
pool was not accessed.

## Preregistered verdict

- H1 `video_distilled_temporal`: did not qualify.
- H2 `video_closed_loop_rollout`: did not qualify.
- Locked winner: none.
- Stage A continuation pass: false.
- Stage B authorized: false.
- Phase 7 AR-assisted evidence: unchanged.

The candidates failed because they did not beat the strongest zero-label
control across all three required endpoints and did not retain the required half
of teacher-added gain. H1/H2 ensembling itself worked—the three-member ensemble
improved all three metrics for both candidates—but could not repair the weaker
underlying methods.

## Strongest zero-label development lane

The fixed direct `video_supervised_temporal` active control was strongest:

- Spearman: `0.1574784207` versus no-video `0.0910654370`
  (`+0.0664129837`, `+72.93%`).
- Top-5% true-movement lift: `0.0611083563` versus `0.0495907196`
  (`+0.0115176367`, `+23.23%`).
- Training-q90 event PR-AUC: `0.1461871599` versus `0.1187892131`
  (`+0.0273979467`, `+23.06%`).
- Positive against the no-video anchor: `3/3` folds on every endpoint.
- Positive in the first 30 seconds: `3/3` folds on every endpoint.

On common teacher-compatible rows, it retained `35.59%` of the privileged
teacher's added Spearman gain, `21.25%` of top-5% gain, and `27.01%` of event
PR-AUC gain. These are real positive gains but below the preregistered `50%`
requirement.

Because this lane was an active control rather than an eligible candidate, it
is not retroactively promoted. The result prospectively identifies it as the
next deployment candidate. Any locked 299-video confirmation requires a new
plan and explicit authorization.

Canonical snapshot:
`evidence/zero_label_video_only_deployment_stage_a_20260714/`.
