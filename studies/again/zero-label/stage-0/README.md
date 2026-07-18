# Zero-label video-only deployment bridge — Stage 0

Status: **passed, planning/contracts only** on `2026-07-14`.

Stage 0 froze the target, prospective video splits, feature policy, cold-start
contracts, and exact dry-run matrices for the preregistered deployment bridge.
It did **not** fit a model, generate teacher scores, score held-out predictions,
authorize Stage A, or promote a deployment claim.

## Locked scope

- substrate: `995` AGAIN videos and `243,575` aligned 2 Hz rows;
- development / prospectively locked videos: `696` / `299`;
- Stage A dry-run matrix: `96/96` rows (`72` members + `24` ensembles);
- Stage B dry-run matrix: `140/140` rows (`105` members + `35` ensembles);
- hard outcome: `future_arousal_max_delta_rows_4_10`;
- valid target rows: `233,124`;
- required endpoints: continuous Spearman, top-5% movement lift, and
  training-q90 event PR-AUC;
- target identity digest: `446906dff30be33f204de0f973207975`;
- development split digest: `cf65a766cd827e6201544dd753049cb4`;
- locked split digest: `ded8bc2bf079fef91ae5c253b9a9ac2e`;
- failed contracts: `[]`.

Every full and first-30-second event slice in the locked folds/panels contains
both event classes, so the required PR-AUC gate is defined without changing the
training-only threshold or regrouping videos.

## Feature-policy resolution

The Phase 7 fold-specific PCA score matrices cannot be concatenated into the
new `696/299` split because they occupy different fold-fitted coordinate
systems. Stage 0 therefore locks the existing frozen predicted cortical/fMRI
row substrate and the existing `temporal_mean_2s` / PCA256 recipe, but requires
the PCA basis to be fitted anew inside each applicable outer training pool.
Nested teacher cross-fitting must likewise fit its PCA basis inside the
teacher-training partition. No V-JEPA/TRIBE re-encoding, PCA-width search,
locked-video PCA fitting, or CPU fallback is allowed.

## Artifacts

- `stage0_result.json` — pass/fail status and artifact checksums;
- `split_manifest.json` — exact video ownership, Stage A folds, Stage B panels,
  and nested teacher cross-fit ownership;
- `target_identity_manifest.json` — raw value/mask/row digests, train-only q90
  thresholds, event support, and builder/scorer source identities;
- `feature_policy_manifest.json` — positive input allowlist, forbidden response
  fields, PCA/diagnostic policy, and MLX-only later-training boundary;
- `dry_run_matrix.csv` — the exact future `96 + 140` scored-row keys.

## Executable validation

```bash
python3 -m pytest -q tests/test_again_zero_label_deployment_stage0.py
```

Stage A remains closed until a separate explicit authorization.
