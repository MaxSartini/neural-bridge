# Current Project State - 2026-06-16

This is the short operating snapshot for the cleaned Neural Bridge repo.

## Repo

- Active repo: `/Users/maxsartini/Neural Bridge`
- Archived source checkout: `/Users/maxsartini/MiroFish-Offline-main`
- External asset root: `/Volumes/onn. Drive/Neural Bridge`
- Compatibility symlink: `/Volumes/onn. Drive/MiroFish -> /Volumes/onn. Drive/Neural Bridge`

The repo should stay lightweight. Heavy research assets belong on the external drive, not in git.

## Active Scientific Direction

Neural Bridge is testing whether predicted neural response trajectories can improve human-response and simulation forecasts under controlled baselines.

Current evidence is strongest for VEATIC cortical feature experiments. The project should continue to describe this as a promising scale candidate, not a finished proof of end-to-end simulation accuracy.

## Current Benchmark Assets

- Complete VEATIC manifest: `benchmarks/veatic/veatic_manifest_124_complete_20260616.jsonl`
- Manifest rows: 10,357 at 1 Hz
- Complete cortical cache: `/Volumes/onn. Drive/Neural Bridge/benchmarks/veatic/tribe_cache`
- Cache shape contract: per-video `tribe_raw_output.npz` with required key `predictions`
- Main targets: `valence`, `arousal`

Current feature families:

- `cortical_global`
- `cortical_global_delta`
- `cortical_pca_64`
- `cortical_pca64_delta`
- raw cortical trajectories for future loader work

## Benchmark Rules

- Full-frame VEATIC rows remain the main baseline.
- Event-conditioned rows are diagnostics unless balanced against stable controls.
- Positive-only pre-event and event masks should report recall/top-k style diagnostics, not PR-AUC as the main claim.
- Thresholds must be fit on train data only.
- PCA and other transforms must be fit on train data only.
- CPU/MPS device consistency should be checked before mixing thresholded results.

## Known Open Issues

1. No protected immutable snapshot of the current 124-video baseline exists yet.
2. Video `83` has a prediction/manifest length mismatch and is currently resampled.
3. The production training tensor loader contract is not formalized.
4. The main 124-video cache is cortical; subcortical artifacts are smoke/test only unless separately extracted and frozen.
5. Old pre-124 runs and legacy docs should be deleted or archived only after the current baseline snapshot records what still matters.

## Next Safe Move

Freeze the current 124-video benchmark baseline before new model work, recursive-head experiments, or subcortical expansion.
