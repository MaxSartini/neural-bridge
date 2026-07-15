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
- External assets after cleanup plus Wav2Vec-BERT download were about `25G` on the local reference SSD before tensor export.
- The raw-representation tensor export adds about `1.7G` under `<external-assets-root>/tensors/veatic_124_raw_representation_v1/`.
- The dense H100 V-JEPA 2.1 cache generated from the `995` AGAIN videos is approximately `1 TB` in the founder's Google Drive workspace.
- The downstream TRIBE/predicted-cortical postpass retained locally is about `38.7 GiB` (approximately `40 GB`) and contains row-level cortical outputs plus compact diagnostics/manifests.

## Current External Asset Families

- `<external-assets-root>/benchmarks/veatic/` - VEATIC cache roots and small logs.
- `<external-assets-root>/evidence_snapshots/veatic_124_v2_20260616/` - protected frozen v2 evidence snapshot.
- `<external-assets-root>/outputs/veatic_124_raw_representation_audit_primary_20260620_152411/` - raw cortical representation audit output and checkpoint/PCA fit cache.
- `<external-assets-root>/tensors/veatic_124_raw_representation_v1/` - frozen model-ready tensors for the next learned-head work; 84 contracts and 420 `.npy` payloads with verification metadata.
- `<external-assets-root>/datasets/veatic/` - raw VEATIC videos and 1 Hz target traces.
- `<external-assets-root>/cache/tribev2/` - TRIBE/neuralset feature cache.
- `<external-assets-root>/cache/huggingface/` - active Hugging Face cache for TRIBE/V-JEPA/Llama/Wav2Vec/Whisper assets.
- `<external-assets-root>/models/tribe/` - official TRIBE checkpoint.
- `<external-assets-root>/models/tribe-mlx/` - TRIBE-MLX head.
- `<external-assets-root>/models/cortical-upstream/` - V-JEPA2 video encoder weights.
- `<external-assets-root>/models/vjepa21_mlx/vitg/` - converted V-JEPA 2.1 ViT-g MLX weights used by the implemented TRIBE adapter path.
- `<external-assets-root>/datasets/again/` or `<external-assets-root>/data/external/AGAIN/cleaned/` - cleaned AGAIN source videos and annotations for scaling pilots, depending on local layout.
- `<external-assets-root>/benchmarks/again/` - AGAIN benchmark caches and manifests.
- Founder Google Drive H100 workspace - approximately `1 TB` of dense V-JEPA 2.1 cache generated from all `995` AGAIN videos.
- `.cache/h100_drive_downloads/again_tribe_v2_postpass_float16_256_2hz/` - approximately `38.7 GiB` / `40 GB` local downstream TRIBE v2 predicted-cortical postpass derived from that V-JEPA cache.
- `<external-assets-root>/models/upstream-encoders/facebook-w2v-bert-2.0/` - downloaded audio encoder for multimodal pilots.
- `<external-assets-root>/models/upstream-encoders/meta-llama-Llama-3.2-3B/` - expected gated text encoder path; currently a placeholder unless populated with authorized Llama assets.
- Local LM Studio MLX text-model directory - optional workstation-local text-model candidate; may pass the repo's MLX text-model directory check but must not be hardcoded into portable configs.
- `<external-assets-root>/models/upstream-encoders-mlx/` - MLX upstream encoder weights.
- `<external-assets-root>/models/transcription/` - MLX Whisper transcription weights.

## Policy

Do not copy heavyweight model weights, raw videos, generated scratch outputs, tensor payloads, or cache directories into the repo. Keep curated lightweight v2 reports, tensor-export summaries, trained-head reports, and bounded AGAIN pilot reports in git; keep heavy raw/cache/tensor artifacts external and add manifest/verifier coverage instead.

Both H100 AGAIN cache layers should stay outside git. The approximately `1 TB` Drive asset is the dense V-JEPA 2.1 cache produced during the 995-video H100 encode. The approximately `40 GB` local postpass is the downstream TRIBE/predicted-cortical working bundle for PCA, bridge, and baseline work: it includes row-level cortical predictions, timestamps, grouped V-JEPA adapter features, compact temporal diagnostics, quality signals, row indexes, split manifests, and schema/readiness docs. It intentionally does not duplicate raw videos or model weights.

The current local cleaned AGAIN mirrors should not be treated as audio-bearing inputs. The 2026-06-22 embedded-stream inventory checked `1,095` AGAIN video containers across the internal scratch and external SSD roots and found `0` audio streams. Keep Wav2Vec-BERT available, but only use it for AGAIN after auditing a separate/original audio-bearing media source.
