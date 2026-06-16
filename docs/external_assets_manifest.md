# External Assets Manifest

Neural Bridge keeps source code and lightweight metadata in this repo. Large model weights, datasets, feature caches, benchmark caches, temporary files, and downloaded research assets live outside the repo.

## Primary External Root

`/Volumes/onn. Drive/Neural Bridge`

## Compatibility Symlink

`/Volumes/onn. Drive/MiroFish` is a symlink to `/Volumes/onn. Drive/Neural Bridge`.

This keeps older commands and historical reports readable while active config defaults use the Neural Bridge path.

## Repo-Tracked Assets

- `backend/app/`
- `backend/scripts/`
- `backend/neuro_core/`
- `external_models/tribev2-apple-silicon/`
- `external_models/tribev2-official/`
- `models/behavior_component_registry.json`
- `models/neuro_atlases/`
- `frontend/src/`
- `static/`
- selected docs and reports

## External Heavy Assets

- `/Volumes/onn. Drive/Neural Bridge/models/`
- `/Volumes/onn. Drive/Neural Bridge/cache/`
- `/Volumes/onn. Drive/Neural Bridge/benchmarks/`
- `/Volumes/onn. Drive/Neural Bridge/datasets/`
- `/Volumes/onn. Drive/Neural Bridge/sources/`
- `/Volumes/onn. Drive/Neural Bridge/runtimes/`
- `/Volumes/onn. Drive/Neural Bridge/tmp/`

## Current Size Snapshot

- Repo: about `26M`
- External assets: about `102G`

## Policy

Do not copy heavyweight model weights, raw videos, generated benchmark outputs, or cache directories into the repo. Add a manifest entry and config default instead.
