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

- Repo: lightweight source and small evidence artifacts only.
- External assets after cleanup plus Wav2Vec-BERT download: about `25G` on the local reference SSD.

## Current External Asset Families

- `<external-assets-root>/benchmarks/veatic/` - VEATIC cache roots and small logs.
- `<external-assets-root>/datasets/veatic/` - raw VEATIC videos and 1 Hz target traces.
- `<external-assets-root>/cache/tribev2/` - TRIBE/neuralset feature cache.
- `<external-assets-root>/cache/huggingface/` - active Hugging Face cache for TRIBE/V-JEPA/Llama/Wav2Vec/Whisper assets.
- `<external-assets-root>/models/tribe/` - official TRIBE checkpoint.
- `<external-assets-root>/models/tribe-mlx/` - TRIBE-MLX head.
- `<external-assets-root>/models/cortical-upstream/` - V-JEPA2 video encoder weights.
- `<external-assets-root>/models/upstream-encoders/facebook-w2v-bert-2.0/` - downloaded audio encoder for multimodal pilots.
- `<external-assets-root>/models/upstream-encoders/meta-llama-Llama-3.2-3B/` - expected gated text encoder path; currently a placeholder unless populated with authorized Llama assets.
- `<external-assets-root>/models/upstream-encoders-mlx/` - MLX upstream encoder weights.
- `<external-assets-root>/models/transcription/` - MLX Whisper transcription weights.

## Policy

Do not copy heavyweight model weights, raw videos, generated benchmark outputs, or cache directories into the repo. Add a manifest entry and config default instead.
