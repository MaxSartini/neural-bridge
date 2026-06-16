# Neural Bridge Rebrand Cleanup - 2026-06-16

## Scope

Rebranded active project surfaces from MiroFish-era naming to Neural Bridge while preserving runtime paths and compatibility aliases that would otherwise break local cache or saved API requests.

## Changed

- Renamed package metadata:
  - root package: `neural-bridge`
  - backend package: `neural-bridge-backend`
- Rebranded active README, roadmap, requirements, repo map, frontend HTML metadata, frontend UI labels, backend service names, log namespaces, graph defaults, health-check labels, and generated handoff defaults.
- Changed the public single-channel simulation platform from `mirofish` to `neural_bridge`.
- Preserved legacy platform alias support:
  - `neural_bridge` and `mirofish` both map to the internal Reddit-shaped OASIS adapter.
- Added Neo4j auth fallback:
  - fresh config defaults to `neural_bridge`
  - existing local databases using the old password can still connect through `NEO4J_LEGACY_PASSWORD`
- Renamed user-facing assets:
  - `static/image/neural-bridge-banner.png`
  - `static/image/neural-bridge-screenshot.jpg`
  - `static/image/neural-bridge-logo.jpeg`
  - `static/image/neural-bridge-logo-compressed.jpeg`
  - `frontend/src/assets/logo/neural-bridge-logo-left.jpeg`
  - `frontend/src/assets/logo/neural-bridge-logo-compressed.jpeg`
- Renamed architecture review:
  - `docs/tribe_neural_bridge_architecture_review.md`

## Deleted As Junk

- `.claude/settings.local.json`
  - This was a stale local command allowlist with historical absolute paths and no runtime role.

## Compatibility Kept Intentionally

The remaining `MiroFish` text is expected and should not be bulk-replaced without moving data:

- Physical workspace path: `/Users/maxsartini/Neural Bridge`
- External cache/model/data path: `/Volumes/onn. Drive/Neural Bridge`
- Legacy env var fallbacks:
  - `NEURAL_BRIDGE_DATASET_ROOT`
  - `NEURAL_BRIDGE_EXTERNAL_ASSET_ROOT`
- Legacy Neo4j password fallback:
  - `NEO4J_LEGACY_PASSWORD`
- Legacy API platform alias:
  - `mirofish`

## Verification

- `python3 -m compileall -q backend/app backend/scripts` passed.
- `python3 -m py_compile backend/app/config.py backend/app/storage/neo4j_storage.py backend/app/services/simulation_runner.py backend/app/api/simulation.py backend/scripts/download_behavior_components.py backend/scripts/download_initial_benchmark_batch.py backend/app/services/sequential_stage_runner.py` passed.
- JSON metadata parse passed for root and frontend package files.
- Removed generated `__pycache__` and `.pyc` files after verification.
- No files remain with `mirofish` or `mirofosh` in their filename under the active checkout scan depth.

## Current Size Snapshot

- Repository root: `6.1G`
- `backend`: `2.4G`
- `benchmarks`: `73M`
- `outputs`: `49M`
- `reports`: `52K`
- `docs`: `224K`
- `frontend`: `1.5M`
- `static`: `6.5M`
