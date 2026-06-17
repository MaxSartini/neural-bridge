# External Assets Manifest

Neural Bridge keeps source code and lightweight metadata in this repo. Large model weights, datasets, feature caches, benchmark caches, temporary files, and downloaded research assets live outside the repo.

## Primary External Root

Configured per workstation:

```bash
NEURAL_BRIDGE_EXTERNAL_ROOT=/path/to/neural-bridge-assets
```

## Repo-Tracked Assets

- `backend/app/`
- `backend/scripts/`
- `external_models/tribev2-apple-silicon/`
- `models/neuro_atlases/`
- `frontend/src/`
- selected docs and reports

## External Heavy Assets

- `<external-assets-root>/models/`
- `<external-assets-root>/cache/`
- `<external-assets-root>/benchmarks/`
- `<external-assets-root>/datasets/`
- `<external-assets-root>/sources/`
- `<external-assets-root>/runtimes/`
- `<external-assets-root>/tmp/`

## Current Size Snapshot

- Repo: about `26M`
- External assets: about `102G`

## Policy

Do not copy heavyweight model weights, raw videos, generated benchmark outputs, or cache directories into the repo. Add a manifest entry and config default instead.
