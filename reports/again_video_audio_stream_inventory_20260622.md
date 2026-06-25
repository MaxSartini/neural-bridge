# AGAIN Video Audio Stream Inventory - 2026-06-22

## Purpose

Check whether the local AGAIN video containers available to Neural Bridge contain embedded audio streams that can feed the Wav2Vec-BERT audio encoder.

## Scope

- Internal scratch AGAIN root: `/Users/maxsartini/neural_bridge_scratch/external_root/data/external/AGAIN`
- External SSD AGAIN root: `/Volumes/onn. Drive/Neural Bridge/data/external/AGAIN`
- File types checked: `.webm`, `.mp4`, `.mkv`, `.mov`
- Probe tool: `ffprobe`

## Result

- Video containers checked: `1,095`
- Containers with embedded audio streams: `0`
- Probe errors: `0`
- Standalone audio files found in those AGAIN roots: `0`

## Interpretation

TRIBE and the Neural Bridge adapter can support multimodal video/audio/text paths, and `facebook/w2v-bert-2.0` is present locally. However, the current cleaned AGAIN media mirrors available to this repo are video-only containers. AGAIN runs using these files should be described as video plus annotations/telemetry, not audio-video or full multimodal.

Do not add Wav2Vec-BERT features to AGAIN from these cleaned video mirrors. If an original/audio-bearing AGAIN source is later found, run a fresh embedded-stream inventory and record the source root before using audio features.

## Related Asset Check

- External Wav2Vec-BERT path exists and is recognized as an encoder: `/Volumes/onn. Drive/Neural Bridge/models/upstream-encoders/facebook-w2v-bert-2.0`
- External `meta-llama-Llama-3.2-3B` path is a placeholder only unless populated with authorized assets.
- Local LM Studio MLX Llama candidate exists and passes the repo's MLX text-model directory check: `/Users/maxsartini/.lmstudio/models/mlx-community/Llama-3.2-3B-Instruct-4bit`
