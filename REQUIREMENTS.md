# Neural Bridge Requirements

This document is the current requirements audit for the cleaned Neural Bridge repo. It describes what a fresh Codex or developer session needs before trying to run the app, reproduce the VEATIC-124 v2 evidence, or continue model work.

## System Requirements

- Apple Silicon macOS is the primary development target for local MLX/MPS acceleration. Linux CPU/GPU environments may work for non-MLX paths but are not the reference setup.
- Python 3.12 or newer.
- Node.js 18 or newer.
- Git.
- External assets root configured through `.env`, for example:

```bash
NEURAL_BRIDGE_EXTERNAL_ROOT=/path/to/neural-bridge-assets
```

Full benchmark work assumes that external root contains model weights, Hugging Face caches, datasets, VEATIC/TRIBE caches, generated benchmark outputs, and temporary extraction files described in `docs/external_assets_manifest.md`.

## Local Services

The active repo only needs the Flask API, Vue viewer, local Python/Node tooling, and the configured external TRIBE assets.

Required for full neural workflows:

- TRIBE/MLX assets under the configured external assets root.
- V-JEPA2/encoder assets under the configured external model paths.
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
- Neuro/ML runtime: `torch`, `transformers`, `safetensors`, `huggingface-hub`, `nibabel`, `nilearn`, `tqdm`.
- Apple Silicon acceleration: `mlx`, `mlx-lm`.
- TRIBE extractor support imported by current code: `neuralset`, `neuraltrain`, `exca`, `einops`, `lightning`, `mne`, `torchmetrics`, `PyYAML`.

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
- `TRIBE_APPLE_SILICON_SOURCE_DIR`
- `HF_HOME`
- `TMPDIR`

For VEATIC/TRIBE benchmark work, prefer external-drive paths so the repo stays lightweight.

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
python3 -m pytest -q tests/test_veatic_strict_benchmark_contract.py tests/test_grouped_video_split.py
```

Frontend build:

```bash
npm run build
```

Full local verification:

```bash
npm run verify
```

## Current Evidence Entry Points

Fresh sessions should read these first:

- `README.md`
- `ROADMAP.md`
- `docs/current_project_state.md`
- `docs/veatic_v2_evidence_summary.md`
- `benchmarks/veatic/veatic_124_confirmatory_benchmark_report_20260616.md`
- `benchmarks/veatic/veatic_124_alignment_lag_repair_20260616.md`

Do not use removed historical docs or deleted benchmark scaffolding as active requirements.
