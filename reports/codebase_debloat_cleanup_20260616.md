# Codebase Debloat Cleanup - 2026-06-16

## Summary

Performed a conservative codebase-wide cleanup focused on stale generated artifacts, old benchmark outputs, and user-facing legacy references.

This checkout is not a Git repository, so the cleanup was intentionally scoped to files that were either generated/rebuildable or superseded by current 124-video VEATIC artifacts.

## Deleted

Generated/runtime artifacts:

- `.DS_Store`
- `.pytest_cache`
- non-venv `__pycache__` and `.pyc` files
- frontend build/dependency outputs: `frontend/dist`, `frontend/node_modules`
- local log files under `log/`, `logs/`, `backend/log/`, and `backend/logs/`
- frontend local log files

Superseded benchmark artifacts:

- VEATIC smoke current-cache benchmark JSON/summary files.
- VEATIC 5-video, 20-video, 50-video benchmark JSON/summary files.
- VEATIC 50-video gated/cache-coverage artifacts superseded by the complete 124-video manifest/results.
- VEATIC 89-video benchmark, event-conditioned, event/spike, posthoc, and 89-vs-50 scale-validation artifacts superseded by current 124-video outputs.
- Duplicate/old VEATIC manifests: `veatic_manifest_1hz.*` and `veatic_manifest_89_complete_20260615.*`.
- Large full 89-device `metric_diff.csv`; retained the smaller focused/material/prediction/threshold device-consistency files.
- Old OpenLAV first20/first50/smoke benchmark result JSONs and regen logs.

Source/docs:

- Removed `legacy_twitter` and `legacy_dual` from `SimulationManager.get_run_instructions()` output while leaving underlying compatibility scripts intact.
- Removed stale `retry` utility references from `README.md` and `PROJECT_REQUIREMENTS.md`.
- Reframed simulation script docs around the local Neural Bridge loop plus compatibility adapters.

## Preserved

- Current 124-video VEATIC manifest and benchmark outputs.
- Current 124-video event-conditioned and event/spike outputs.
- Current 124-video temporal fairness and temporal context v2 output directories.
- 124 alignment diagnostics.
- Compact 89-device consistency diagnostic files: JSON/MD, focused/material metric diffs, prediction diff, threshold diff.
- `models/`, `backend/.venv/`, `external_models/` weights/source, and external `/Volumes/onn. Drive/Neural Bridge` caches.
- Runtime upload data under `backend/uploads/`, because it may contain user/project state.
- Legacy Twitter/Reddit simulation scripts, because active runtime paths still reference them.

## Size Change

Approximate observed directory sizes:

- Repository root: `6.5G` before, `6.2G` after.
- `benchmarks/`: `307M` before, `73M` after.
- `frontend/`: `95M` before, `1.5M` after.
- `logs/`: `3.1M` before, `12K` after.
- `log/`: `36K` before, `0B` after.

The explicit superseded benchmark deletion removed 85 files totaling 244,500,697 bytes. The generated-artifact cleanup removed 231 paths on the first pass, plus a few regenerated cache/metadata files after verification.

## Verification

- `python3 -m py_compile backend/app/services/simulation_manager.py`
- Stale reference scan for removed `retry` utility and removed run-instruction keys returned no matches in the patched files.
- Confirmed `frontend/node_modules`, `frontend/dist`, and `frontend/build` are absent.
- Confirmed no `.DS_Store`, `.pytest_cache`, `__pycache__`, or `.pyc` files remain outside `backend/.venv` in the checked depth scan.
- `git status` still fails because this workspace is not a Git repository.

## Remaining Cleanup Candidates

- `backend/.venv` is about 2.5 GB under `backend/`, but it was preserved because deleting it would break the local Python runtime until recreated.
- `models/` is about 3.5 GB and is mostly expected local model state; delete only with a model-cache migration plan.
- `backend/uploads/` contains old reports/simulation databases/smoke outputs; it is runtime/user state and should be cleaned only after deciding it is not needed.
- Legacy Twitter/Reddit scripts are still used by `simulation_runner.py` and API script download validation. A deeper product simplification pass should migrate those runtime paths before deleting the scripts.
