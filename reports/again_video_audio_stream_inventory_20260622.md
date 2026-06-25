# AGAIN Video Audio Stream Inventory - 2026-06-22

## Purpose

Check whether the local AGAIN video containers available to Neural Bridge contain embedded audio streams that can feed the Wav2Vec-BERT audio encoder.

## Scope

- Internal scratch AGAIN root: local workstation scratch external root.
- External SSD AGAIN root: configured `NEURAL_BRIDGE_EXTERNAL_ROOT` external drive mirror.
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

- External Wav2Vec-BERT path exists under the configured external root and is recognized as an encoder: `models/upstream-encoders/facebook-w2v-bert-2.0`
- External `meta-llama-Llama-3.2-3B` path is a placeholder only unless populated with authorized assets.
- A local LM Studio MLX Llama candidate may pass the repo's MLX text-model directory check when configured on a workstation.
