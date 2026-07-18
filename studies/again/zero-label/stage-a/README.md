# Zero-label video-only deployment bridge — Stage A

Stage A completed the explicitly authorized development screen on `2026-07-14`.
The exact `96/96` scored-row matrix passed its scope, target-identity, split,
PCA, inference-firewall, prediction-seal, checkpoint, finite-coverage, and
rollout audits. Fitting used MLX GPU/MPS; all 12 fold-local PCA fits recorded the
`mlx_gpu` backend. The prospectively locked 299-video pool was not accessed.

The preregistered H1 distillation and H2 closed-loop candidates did **not** pass
the continuation gate. Neither beat the strongest zero-label control on all
three required endpoints, so there is no locked winner and Stage B remains
unauthorized. This result does not weaken Phase 7 or the earlier binary bridge
evidence; it rejects these two deployment translations under this recipe.

## The important development result

The fixed `video_supervised_temporal` active control was the strongest
zero-label lane:

| Development mean | Direct supervised temporal | No-video anchor | Absolute gain | Relative gain |
|---|---:|---:|---:|---:|
| Spearman future-movement ranking | 0.1574784207 | 0.0910654370 | +0.0664129837 | +72.93% |
| Top-5% true-movement lift | 0.0611083563 | 0.0495907196 | +0.0115176367 | +23.23% |
| Training-q90 event PR-AUC | 0.1461871599 | 0.1187892131 | +0.0273979467 | +23.06% |

It beat the no-video anchor on all three endpoints in all `3/3` development
folds, and did the same on the separately gated first-30-second slices. It also
beat the sequence-shuffled and whole-video label-permutation controls. This is
strong development evidence that cached predicted neuro-response video features
can support cold-start, zero-label-at-inference future-response ranking when the
model is trained directly on the future outcome.

This lane was a prespecified active control, not an eligible H1/H2 candidate.
It therefore cannot be retroactively declared the Stage A winner. On common
teacher-compatible rows it retained `35.59%` of teacher-added Spearman gain,
`21.25%` of teacher-added top-5% gain, and `27.01%` of teacher-added event
PR-AUC gain—useful but below the locked `50%` requirement on every endpoint.

## Canonical machine summary

- [stage_a_summary.json](stage_a_summary.json)
- Full runtime root:
  `/Volumes/onn. Drive/Neural Bridge/outputs/again_dense_2hz_zero_label_deployment_stage_a_20260714/`
- Full scored matrix:
  `metrics/stage_a_rows.csv` under that runtime root
- Full verdict:
  `metrics/stage_a_result.json` under that runtime root
- Signed implementation commit: `ace7a75`

## Next bounded hypothesis

The development evidence selects direct supervised temporal video-only learning,
not teacher-score distillation or self-rollout, as the next deployment candidate.
A new prospective confirmation plan may lock that already-fixed lane and the
existing untouched 299-video pool, but it requires a new explicit authorization.
The original Stage B is not authorized.
