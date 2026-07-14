# Neural Bridge Requirements

This document is the current requirements audit for the cleaned Neural Bridge repo. It describes what a fresh Codex or developer session needs before trying to run the app, inspect the VEATIC + AGAIN evidence ladder, or continue model work.

## System Requirements

- Apple Silicon macOS is the primary development target for local MLX/MPS acceleration. Linux CPU/GPU environments may work for non-MLX paths but are not the reference setup.
- Python 3.12 or newer.
- Node.js 20.19.0 or newer on the Node 20 line, or Node.js 22.12.0 or newer.
- Git.
- External assets root configured through `.env`, for example:

```bash
NEURAL_BRIDGE_EXTERNAL_ROOT=/path/to/neural-bridge-assets
```

Full benchmark work assumes that external root contains model weights, Hugging Face caches, datasets, VEATIC/TRIBE caches, generated benchmark outputs, and temporary extraction files described in `docs/external_assets_manifest.md`.

The dense AGAIN H100 TRIBE postpass bundle is external-only and currently staged through Google Drive, not the repo:

```text
NeuralBridge_H100_AGAIN_tribe_v2_postpass_float16_256_2hz
```

The reference local pull target is:

```text
.cache/h100_drive_downloads/again_tribe_v2_postpass_float16_256_2hz/
```

The frozen raw-representation tensor export also expects this external root. Model-ready tensors are external-only under:

```text
${NEURAL_BRIDGE_EXTERNAL_ROOT}/tensors/veatic_124_raw_representation_v1/
```

Tracked summaries and row-metadata samples live under `outputs/veatic_124_raw_representation_tensor_export_v1/`.

The current VEATIC-124 v2 evidence cache is video-dominant, not a full text+audio+video multimodal cache. Verify modality coverage before reporting cache scope:

```bash
python3 backend/scripts/run_veatic_strict_benchmark.py --modality-audit-only
```

## Local Services

The active repo only needs the Flask API, Vue viewer, local Python/Node tooling, and the configured external TRIBE assets.

Required for full neural workflows:

- TRIBE/MLX assets under the configured external assets root.
- V-JEPA2/encoder assets under the configured external model paths.
- Converted V-JEPA 2.1 MLX assets under the configured external root when using the `MlxVjepa21Video` TRIBE path.
- Apple Metal/MPS-capable PyTorch for Torch-based encoder code.

## Python Dependencies

Canonical backend dependency manifests:

- `backend/requirements.txt`
- `backend/pyproject.toml`

The dependency audit was run from active imports under:

- `backend/app`
- `backend/scripts`
- `tests`

Main dependency groups:

- Web/API: `flask`, `flask-cors`.
- HTTP utilities: `requests`.
- Data/benchmarking: `numpy`, `pandas`, `scipy`, `scikit-learn`.
- Parquet support for dense AGAIN row indexes and label manifests: `pyarrow`.
- Neuro/ML runtime: `torch`, `transformers`, `safetensors`, `huggingface-hub`, `nibabel`, `nilearn`, `tqdm`.
- Apple Silicon acceleration: `mlx`, `mlx-lm`.
- TRIBE extractor support imported by current code: `neuralset`, `neuraltrain`, `exca`, `einops`, `lightning`, `mne`, `torchmetrics`, `PyYAML`.
- The TRIBE/neuralset/exca trio is pinned in `backend/requirements.txt` because the released TRIBE config is schema-sensitive.
- Uncached multimodal pilots also need `moviepy`, `soundfile`, `mlx-whisper`, and `x-transformers`.
- Raw representation audit/tensor export helpers additionally rely on `nibabel`, `nilearn`, `scipy`, and `scikit-learn` for ROI atlas loading, PCA metadata, and leakage-safe representation checks.
- Frozen tensor trained-head helpers require `torch` with MPS available and refuse CPU fallback.
- AGAIN utilities use `ffmpeg`/`ffprobe` and OpenCV (`cv2`) for dataset and boundary audits.

Install path:

```bash
cd <repo-root>/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Root helper path, if `uv` is installed:

```bash
cd <repo-root>
npm run setup:backend
```

## JavaScript Dependencies

JavaScript manifests:

- `package.json` for root helper scripts only.
- `frontend/package.json` and `frontend/package-lock.json` for the Vue/Vite app.

The root `package.json` has no runtime dependencies. It only provides helper scripts for setup, backend launch, frontend launch, build, tests, and verification.

Frontend:

- `vue`
- `vue-router`
- `axios`
- `three`
- `vite`
- `@vitejs/plugin-vue`

Install path:

```bash
cd <repo-root>
npm install
npm run setup:frontend
```

## Environment Configuration

Use `.env.example` as the template and create a local `.env`. The `.env` file is intentionally ignored by git.

Important values for current work:

- `NEURAL_BRIDGE_EXTERNAL_ROOT`
- `TRIBE_CACHE_DIR`
- `TRIBE_MLX_DIR`
- `TRIBE_VIDEO_ENCODER_MLX_DIR`
- `TRIBE_VJEPA21_IMAGE_SIZE`
- `TRIBE_VIDEO_FRAME_SAMPLER`
- `TRIBE_VIDEO_WINDOW_BATCH_SIZE`
- `TRIBE_MLX_CLEAR_CACHE_EACH_WINDOW`
- `TRIBE_MLX_CLEAR_CACHE_EACH_VIDEO`
- `TRIBE_APPLE_SILICON_SOURCE_DIR`
- `HF_HOME`
- `TMPDIR`

For VEATIC/TRIBE benchmark work, prefer external-drive paths so the repo stays lightweight.

Full multimodal pilots additionally need:

- `TRIBE_AUDIO_ENCODER_LOCAL_DIR` present or resolvable from Hugging Face.
- `TRIBE_TEXT_ENCODER_LOCAL_DIR` populated with the gated `meta-llama/Llama-3.2-3B` assets, or Hugging Face credentials with access to that gated model.
- Working transcription backend dependencies.
- A separate pilot cache root passed to `backend/scripts/run_veatic_tribe_cache.py --require-multimodal`.

Current pilot status:

- `facebook/w2v-bert-2.0` is present under the external SSD upstream encoder path.
- The guarded VEATIC `83,84` pilot reaches audio extraction, word extraction, Text/Sentence creation, and text feature preparation.
- It is currently blocked at text encoder loading because `meta-llama/Llama-3.2-3B` is gated and the local SSD directory is only a placeholder.
- Do not re-encode all 124 VEATIC videos for multimodal coverage: only videos `83` and `84` contain audio streams.

V-JEPA 2.1 and AGAIN status:

- `MlxVjepa21Video` is implemented and selected when `TRIBE_VIDEO_ENCODER_MLX_DIR/config.json` declares `tensor_layout=vjepa2_1_mlx_port`.
- `backend/scripts/run_veatic_tribe_cache.py` includes worker claims, resume status, per-window checkpoints, ffmpeg frame sampling, and protected-cache write refusal for MLX/V-JEPA outputs.
- Dense full-AGAIN data generation is complete externally: H100 V-JEPA 2.1 ViT-G encoded `995` videos at `2Hz` rows / `2Hz` sampling / `256px` / float16, and cache-only TRIBE v2 completed `995/995` videos with `0` failures and `243,575` row-level cortical predictions. Use `tools/run_h100_tribe_postpass.py` only for cache-only postpass/retry work; it must not decode videos or rerun V-JEPA.
- Dense AGAIN true-2Hz supervised alignment is implemented in `backend/scripts/again_dense_2hz_benchmark.py` and writes `labels_aligned_2hz.parquet` under the local H100 pull target. Downstream Phase 5/5.5 work now includes eval-mode repair, frozen-AR residual repair, blocked washout-gap binary confirmation, and repaired grouped-video compatibility.

## Tracked Versus External Assets

Tracked:

- Source code under `backend/`, `frontend/`, `tests/`.
- Lightweight docs and evidence summaries.
- Neuro atlas files under `models/neuro_atlases/`.
- Source snapshots under `external_models/`.

External only:

- Model weights.
- Hugging Face caches.
- Raw videos and datasets.
- TRIBE/VEATIC cache directories.
- Generated large benchmark outputs.
- Temporary extraction files.

## Verification Commands

Syntax/import smoke:

```bash
python3 -m compileall -q backend/app backend/scripts tests
```

Current real test suite:

```bash
python3 -m pytest -q tests/test_veatic_raw_representation_contract.py tests/test_veatic_strict_benchmark_contract.py tests/test_grouped_video_split.py
```

Focused implemented-path tests:

```bash
python3 -m pytest -q tests/test_mlx_vjepa21_cortical.py tests/test_veatic_tribe_cache_runtime.py tests/test_veatic_frozen_tensor_adapter.py tests/test_veatic_frozen_tensor_trained_heads.py tests/test_again_boundary_manifest.py tests/test_again_full_ar_context.py tests/test_again_native_temporal_alignment.py
```

Frontend build:

```bash
npm run build
```

Full local verification:

```bash
npm run verify
```

Frozen v2 evidence verification, no video re-encoding:

```bash
npm run evidence:verify
```

Tensor export verification is recorded in:

```bash
outputs/veatic_124_raw_representation_tensor_export_v1/tensor_export_verification.json
```

Regenerate tensors only when intentionally refreshing the external tensor contract:

```bash
python3 tools/export_veatic_raw_representation_tensors.py
```

That command reads existing TRIBE raw cortical cache and PCA fit-cache payloads; it does not re-encode videos or rerun the full representation audit.

Run trained heads only when intentionally refreshing the post-v2 trained-head benchmark:

```bash
python3 backend/scripts/run_veatic_frozen_tensor_trained_heads_benchmark.py
```

Run dense AGAIN benchmarking from the audited H100 TRIBE bundle and `labels_aligned_2hz.parquet` only when intentionally extending the current Phase 5.5 evidence ladder. The H100 postpass itself did not run PCA, bridge training, spike/delta benchmarking, or promotion gates; the current promoted evidence comes from the later tracked Phase 5/5.5 reports.

## Current Evidence Entry Points

Fresh sessions should read these first:

- `README.md`
- `ROADMAP.md`
- `docs/current_project_state.md`
- `docs/neural_bridge_phase5_5_evidence_ladder.md`
- `docs/current_claim_status.json`
- `reports/again_dense_2hz_phase5_temporal_residual_binary_big_confirm_20260630_025437.md`
- `reports/again_dense_2hz_phase5_temporal_residual_grouped_compat_20260630_033520_UPDATED_VERDICT.md`
- `docs/veatic_v2_evidence_summary.md`
- `docs/veatic_v2_evidence_freeze.md`
- `benchmarks/veatic/veatic_124_confirmatory_benchmark_report_20260616.md`
- `benchmarks/veatic/veatic_124_alignment_lag_repair_20260616.md`
- `docs/veatic_raw_representation_audit.md`
- `outputs/veatic_124_raw_representation_tensor_export_v1/tensor_export_report.md`
- `docs/again_dense_h100_cache.md`
- `evidence/README.md`

Use current docs and the dense H100 handoff for AGAIN work. Do not treat historical bundle copies as current claim authority unless the current docs point to them.
