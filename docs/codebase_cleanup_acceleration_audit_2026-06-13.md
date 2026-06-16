# Codebase Cleanup And Apple Silicon Acceleration Audit

Generated: 2026-06-13, Africa/Johannesburg
Workspace: `/Users/maxsartini/Neural Bridge`

## Scope

Static audit of source layout, stale/orphaned code, generated artifacts, CPU-bound benchmark paths, and places where MLX/MPS acceleration is realistic without reducing scientific accuracy.

This workspace is not currently a Git repository, so cleanup should be done with a manifest and reversible moves before deletion.

## Current Architecture Read

The codebase has three active layers:

1. Flask product app
   - `backend/app/__init__.py`
   - `backend/app/api/*`
   - `backend/app/services/*`
   - `backend/app/storage/*`

2. Neuro/TRIBE benchmark and extraction scripts
   - `backend/scripts/run_veatic_tribe_cache.py`
   - `backend/scripts/run_veatic_neuro_benchmark.py`
   - `backend/scripts/run_veatic_gated_pipeline.py`
   - OpenLAV/EmoFilm scripts in `backend/scripts/`

3. Local model/data/runtime artifacts
   - `models/`
   - `external_models/`
   - `benchmarks/`
   - `backend/uploads/`
   - logs under `backend/logs/`, `log/`, and `logs/`

The current high-value performance surface is the VEATIC/OpenLAV TRIBE path, not the Flask CRUD/API layer.

## Immediate Cleanup Candidates

### Safe Generated/Runtime Artifacts

These should not live in source control and can be removed or moved after confirming they are not the only copy:

- `frontend/node_modules/` (~91 MB)
- `frontend/dist/` (~1.6 MB)
- `backend/.venv/` (~2.8 GB)
- `backend/.pytest_cache/`
- `**/__pycache__/`
- `*.pyc`
- `.DS_Store`
- `frontend/frontend.log`
- `frontend/frontend-neuro-viewer.log`
- `backend/backend.log`
- old empty OASIS logs under `log/` and `backend/log/`
- old app logs under `backend/logs/`

The root `.gitignore` already ignores most of these, but the working tree contains them because this is an unpacked/non-git workspace. A cleanup script should remove generated artifacts from the source bundle and leave external benchmark/model caches where they are.

### Heavy Artifacts To Keep Out Of Repo Bundles

These are expected local assets, but they should be documented as external state, not app source:

- `models/` (~1.6 GB)
- `external_models/` (~9.2 MB)
- `benchmarks/` (~87 MB)
- `backend/uploads/` (~26 MB)
- `/Volumes/onn. Drive/Neural Bridge/...` external cache paths

Do not delete `models/tribe-mlx/...` or `models/tribe/facebook-tribev2/best.ckpt` without a separate model-cache migration plan.

## Stale / Legacy Code Findings

### 1. Zep-Era Text Still Present

Resolved in first cleanup pass: `backend/app/api/graph.py:492` now reports `"Creating local graph..."` instead of the stale Zep-era progress message.

Resolved in first cleanup pass: `OntologyGenerator.generate_python_code()` was removed after verifying no active callers. It existed solely to emit deprecated Zep-format Pydantic code.

Several docstrings still say "Replaces Zep..." in storage/services. That is acceptable as migration history, but product-facing messages should be cleaned.

### 2. Unused Retry Helper

Resolved in first cleanup pass: `backend/app/utils/retry.py` was removed after verifying no active callers. `Neo4jStorage` implements its own `_call_with_retry()` and `LLMClient`/`EmbeddingService` each implement direct retry behavior.

### 3. Compatibility Wrapper Duplication

`backend/neuro_core/*` mostly re-exports classes from `backend/app/services/*`. This is not harmful and is useful for standalone imports, but it is a compatibility layer, not independent implementation. Keep it, but document it as public API shims.

### 4. Legacy Simulation Entry Points

`backend/app/services/simulation_manager.py:814-816` still exposes `legacy_twitter` and `legacy_dual` commands. The scripts exist, so this is not orphaned, but they should be classified:

- Keep if the old Twitter/dual adapter path is supported.
- Otherwise move under `backend/scripts/legacy/` and remove from product run instructions.

### 5. Market Data Legacy Aliases

`backend/app/services/market_data_consolidator.py:107`, `190-193`, and `776-780` retain backward-compat names and size-matching fallback behavior. This is low risk but contributes to stale surface area. Keep only if older upload metadata still appears in real projects.

## Orphan / Reachability Notes

The static import graph initially marked Flask API modules as unreferenced, but they are registered indirectly through `backend/app/api/__init__.py`; do not delete:

- `backend/app/api/graph.py`
- `backend/app/api/simulation.py`
- `backend/app/api/report.py`
- `backend/app/api/neuro_viewer.py`
- `backend/app/api/scrape.py`

Likewise, most `backend/scripts/run_*`, `build_*`, `audit_*`, `check_*`, and `test_*` files are standalone command entrypoints. They are not imported by other modules by design.

## CPU / Slow Path Audit

### Already Correctly Accelerated

`backend/scripts/run_veatic_neuro_benchmark.py` has the right structure for exact Apple Silicon acceleration:

- `pca_fit_transform_mps_gram()` uses MPS for large cortical matrix products and CPU only for the smaller row-space eigensolve.
- `pca_fit_transform_mps_power()` exists but is approximate, so it should not be used for evidence unless parity-tested.
- `ridge_fit_predict_mps()` uses MPS solve only when feature count is high enough.
- CPU pseudo-inverse remains default for compact/PCA-sized ridge because benchmarks showed MPS overhead is slower there.

Recommended: keep `mps_gram` as the evidence default. Do not chase "pure MLX/MPS SVD" because current PyTorch/MLX kernels do not support exact GPU SVD/eigh for this workload.

### CPU Work That Should Remain CPU

These are CPU by design and should not be moved to MPS/MLX unless the model changes:

- `CatBoostRegressor` benchmarks in OpenLAV/EmoFilm/calibration scripts. CatBoost GPU is CUDA-oriented and not a useful Apple Silicon/MPS target.
- Low-dimensional ridge/compact feature regressions. Existing audit shows CPU is faster for small matrices.
- Pandas CSV parsing and file ingestion.

### CPU Work Worth Improving Without Accuracy Loss

1. `backend/app/services/tribe_adapter.py:170-174`
   - The MLX cortical path still instantiates `TribeModel.from_pretrained(..., device="cpu")`.
   - This is likely only used for NeuralSet event/feature extraction while the converted cortical head runs in MLX.
   - Main speed cost may still be upstream text/audio/video feature extraction, especially video.
   - Recommended action: isolate timing for event extraction, upstream features, and `MlxTribeEncoder.predict()`. Only move components that are confirmed CPU-bound and exact-equivalent.

2. `backend/app/services/tribe_adapter.py:354-391`
   - `data.num_workers = 0`, `data.batch_size = 1`, and unsafe ViT-G MPS is forced to CPU unless `TRIBE_ALLOW_UNSAFE_VITG_MPS=true`.
   - This is a stability tradeoff. For default evidence runs, keep the conservative path.
   - Acceleration route: prefer smaller/compatible V-JEPA exact MPS path, chunked attention, and cache reuse rather than enabling unsafe ViT-G MPS.

3. `backend/app/services/mlx_text_extractor.py:44-96`
   - Uses PyTorch `DataLoader` only for iteration and MLX for model hidden states.
   - Potential cleanup: replace the DataLoader with direct batching to reduce Torch dependency in the MLX text path. This is a speed/complexity cleanup, not likely a major runtime win.

4. `backend/app/services/market_data_consolidator.py:453`
   - Uses `df.iterrows()` to build records.
   - Replace with `df.where(pd.notna(df), None).to_dict("records")` plus type normalization. This is pure CPU/pandas cleanup and preserves accuracy.

5. `backend/scripts/run_veatic_annotation_baseline.py:103`
   - Uses NumPy pseudo-inverse for a lightweight annotation baseline.
   - Not worth MPS unless run at much larger scale; keep CPU.

## Benchmark Script Cleanup

The benchmark scripts are large and growing:

- `backend/scripts/run_veatic_neuro_benchmark.py` is ~1842 lines.
- `backend/scripts/run_openlav_benchmark.py` is ~922 lines.
- `backend/scripts/run_parallel_simulation.py` is ~2220 lines.

Recommended refactor boundaries:

- Move shared metrics/rank/correlation/split helpers into `backend/app/benchmarks/metrics.py`.
- Move ridge/PCA backends into `backend/app/benchmarks/linear_backends.py`.
- Move VEATIC target construction into `backend/app/benchmarks/veatic_targets.py`.
- Keep CLI wrappers in `backend/scripts/` thin.

This will make it easier to test MPS/CPU parity without rerunning full benchmarks.

## Dependency / Packaging Cleanup

The backend has both:

- `backend/requirements.txt`
- `backend/pyproject.toml`
- `backend/uv.lock`

This is acceptable, but they should be treated as generated from one source of truth. Right now the comments differ and `uv.lock` contains many transitive packages not obvious from requirements. Recommended:

- Make `pyproject.toml` canonical.
- Generate `requirements.txt` only for Docker/legacy install if needed.
- Add a short note in README.

## Recommended Cleanup Sequence

1. Create a reversible cleanup script or checklist:
   - remove `__pycache__`, `.pyc`, `.DS_Store`, `.pytest_cache`
   - remove local logs
   - exclude `frontend/node_modules`, `frontend/dist`, `backend/.venv`

2. Fix remaining stale user-visible strings:
   - conda-era run instructions if no longer valid

3. Delete or consolidate clearly unused helpers:
   - first-pass unused helper cleanup is complete; continue with another import-graph pass after benchmark refactors

4. Refactor benchmark utilities out of monolithic scripts.

5. Add targeted parity tests:
   - CPU SVD vs MPS-Gram PCA on fixed synthetic data
   - CPU pinv vs MPS ridge for high-dimensional raw cortical path
   - no-reencode feature modes preserve train-only PCA fitting

6. Only then make performance changes to TRIBE extraction internals.

## High-Confidence Acceleration Position

The current MPS-Gram PCA patch is the right kind of Apple Silicon acceleration: exact, measurable, and scientifically defensible. The next safe improvements are around cache reuse, timing instrumentation, and benchmark refactoring, not blindly moving every CPU operation to MLX/MPS.

## First Cleanup Pass Completed

Completed after the initial audit:

- Replaced stale product progress text `"Creating Zep graph..."` with `"Creating local graph..."`.
- Removed unused `backend/app/utils/retry.py`; no internal imports referenced `retry_with_backoff` or `RetryableAPIClient`.
- Removed the deprecated `OntologyGenerator.generate_python_code()` method that emitted Zep-era Pydantic code.
- Added `backend/scripts/cleanup_generated_artifacts.py`, a dry-run-first cleanup utility that hard-protects models, benchmark data, video/TRIBE caches, uploads, and external `/Volumes/onn. Drive/Neural Bridge` cache paths.
- Ran the cleanup utility with default safe settings. It removed `.DS_Store`, `.pytest_cache`, `__pycache__`, and `.pyc` artifacts only. It did not remove logs, frontend build outputs, node modules, models, uploads, benchmarks, or video caches.
- Replaced conda-era simulation run instructions with project-local `backend/.venv/bin/python` instructions when that interpreter exists.
- Replaced a slow `pandas.DataFrame.iterrows()` CSV-record extraction loop in `market_data_consolidator.py` with `to_dict("records")` plus the same native-value normalization.

## Second Cleanup / Correctness Pass Completed

Completed in the follow-up audit:

- Optimized `backend/scripts/run_veatic_neuro_benchmark.py` persistence baselines by replacing per-row full-history scans with sorted-history `bisect` lookups. A focused parity check confirmed identical predictions for synthetic rows, including duplicate frame indices.
- Optimized `backend/app/services/entity_reader.py` edge enrichment by indexing all edges once per graph. This changes the hot path from rescanning every edge for every entity to a single edge index lookup per entity, while preserving related-edge output order. A fake-storage parity check covered enriched and non-enriched paths plus entity-type filters.
- Optimized `backend/app/utils/file_parser.py` chunk splitting by reusing the current text window while searching separator candidates, avoiding repeated slicing.
- Optimized `backend/app/storage/search_service.py` Lucene escaping by using a module-level `frozenset` for special-character membership.
- Fixed a broken numeric dataframe summary f-string in `backend/app/utils/file_parser.py`. The old expression was caught by a broad exception and emitted `"summary unavailable"` for valid numeric columns; numeric summaries now emit the intended latest/previous/range/mean/trend text.
- Centralized embedding vector dimensions in `Config.EMBEDDING_DIMENSIONS` and reused it in both the LM Studio empty-text zero vector fallback and Neo4j vector-index Cypher. Default behavior remains 768 dimensions.
- Fixed nondeterministic ticker ordering in `backend/app/storage/ner_extractor.py` by replacing `list(set(...))` with first-seen-order de-duplication. This avoids process-dependent entity/relation ordering from the heuristic financial extractor.
- Tightened `backend/scripts/cleanup_generated_artifacts.py` protection logic so top-level `models`/`benchmarks` remain protected without accidentally skipping source packages named `models`, while `.venv`, `uploads`, `tribe_cache`, and `video_windows` are protected wherever they appear.

Verification performed:

- `python -m compileall -q backend/app backend/neuro_core backend/scripts`
- Focused parity check for VEATIC persistence baselines.
- Focused parity check for file chunking and Lucene escaping.
- Focused parity check for entity-reader edge enrichment.
- Focused numeric dataframe summary check.
- Focused embedding dimension config/schema check.
- Focused NER ticker-order check.

The active VEATIC/TRIBE cache extraction process was observed still running during this pass and was not interrupted. Video caches, TRIBE caches, benchmark manifests, model directories, uploads, and `/Volumes/onn. Drive/Neural Bridge` paths were not deleted.

## Remaining Correctness / Waste Findings

These are not patched yet because they require broader behavior review or benchmark timing:

- `backend/app/services/tribe_adapter.py` still has an intentionally conservative `device="cpu"` TRIBE model instantiation for the MLX cortical path. This may be correct for event/feature extraction stability, but it should be instrumented by stage before deciding what can move to MPS/MLX without accuracy or host-stability loss.
- `backend/app/services/tribe_adapter.py` keeps `data.batch_size = 1` and `data.num_workers = 0`. This avoids concurrency/MPS memory surprises but may leave throughput on the table for safe feature extraction stages.
- `backend/app/services/mlx_text_extractor.py` still uses a PyTorch `DataLoader` around an MLX model. Direct batching could reduce dependency overhead if profiling shows text extraction is material.
- Broad `except Exception` fallbacks remain common in API/report/simulation paths. Some are appropriate for user-facing resilience, but they can hide coding mistakes like the dataframe-summary bug above. High-value paths should narrow exception handling once tests cover expected failure modes.
- `backend/scripts/run_veatic_neuro_benchmark.py`, `backend/scripts/run_openlav_benchmark.py`, and `backend/scripts/run_parallel_simulation.py` remain large monolithic scripts. Refactoring shared metrics, split construction, PCA/ridge backends, and manifest loading into tested modules would reduce duplicate logic and make future MPS/CPU parity safer.

## Third Pass: Cortical V-JEPA2 MLX Search

The slow cortical bottleneck is `facebook/vjepa2-vitg-fpc64-256`, not the
subcortical ViT-L path.

Search result:

- No published exact Hugging Face drop-in was found for
  `facebook/vjepa2-vitg-fpc64-256` with `library_name=mlx`.
- Published MLX V-JEPA2 HF repos are currently ViT-L or action-conditioned
  ViT-G:
  - `mlx-community/V-JEPA2-vitl-fpc64-256`: useful MLX ViT-L port, not cortical
    TRIBE-compatible because it is hidden 1024 instead of hidden 1408.
  - `mlx-community/V-JEPA2-AC-vitg`: ViT-G MLX artifact, but action-conditioned
    robotics/world-model variant rather than the exact cortical HF checkpoint.
  - `dgrauet/vjepa-2.0-vitl-mlx` and `dgrauet/vjepa-2.1-vitl-mlx`: ViT-L ports,
    not cortical drop-ins.
- GitHub search found reusable MLX code:
  - `xocialize/vjepa2-mlx`: HF-style split Q/K/V MLX model; scripts include a
    ViT-G config example with hidden 1408, 40 layers, 22 heads.
  - `dgrauet/vjepa2-mlx`: broader V-JEPA 2.0/2.1 MLX port; includes
    `vit_giant_rope`, but some constructors use 16 heads and require careful
    config/key validation before use.

Local checkpoint inspection:

- `/Volumes/onn. Drive/Neural Bridge/models/cortical-upstream/facebook-vjepa2-vitg-fpc64-256`
  has the exact expected config: hidden 1408, 40 layers, 22 heads, 64 frames,
  256px, tubelet size 2.
- Its `model.safetensors` has 843 keys and HF split Q/K/V layout:
  `encoder.layer.*.attention.{query,key,value}.*`.
- Required probe shapes matched, including patch embedding
  `(1408, 3, 2, 16, 16)` and final layer MLP `(1408, 6144)`.

Added scripts:

- `backend/scripts/probe_mlx_vjepa2_cortical.py`
  - Queries Hugging Face and GitHub for MLX V-JEPA2 candidates.
  - Classifies exact, near-miss, and non-drop-in candidates.
  - Inspects community ports for ViT-G support.
  - Inspects the local cortical HF checkpoint without loading the full model.
- `backend/scripts/convert_vjepa2_vitg_hf_to_mlx.py`
  - Dry-run-first converter for the exact local cortical HF checkpoint.
  - Converts only the Conv3D patch embedding layout from
    `(out, in, kt, kh, kw)` to `(out, kt, kh, kw, in)`.
  - Defaults to fp16 output under
    `models/upstream-encoders-mlx/facebook-vjepa2-vitg-fpc64-256`.
  - Requires `--apply` before writing a multi-GB converted checkpoint.

Current implementation:

- The active VEATIC Torch/MPS extraction process was paused with SIGSTOP, not
  killed. Its caches and process state remain intact.
- `convert_vjepa2_vitg_hf_to_mlx.py --apply` produced an fp16 MLX checkpoint at
  `models/upstream-encoders-mlx/facebook-vjepa2-vitg-fpc64-256`.
- The converted checkpoint has 843 tensors and is about 1.9 GiB. Direct
  safetensors inspection confirmed the MLX Conv3D patch embedding shape is
  `(1408, 2, 16, 16, 3)` and the key ViT-G linear/norm shapes remain intact.
- Added `backend/app/services/mlx_vjepa2_cortical.py`, a local MLX V-JEPA2
  ViT-G encoder and Neuralset-compatible `MlxVjepa2Video` extractor.
- Added config flags:
  - `TRIBE_VIDEO_ENCODER_BACKEND=auto|mlx|torch`
  - `TRIBE_VIDEO_ENCODER_MLX_DIR`
- `TribeAdapter._config_update()` now selects `MlxVjepa2Video` by default for
  cortical V-JEPA2 when the converted MLX checkpoint exists. The old
  Torch/Transformers path remains available with `TRIBE_VIDEO_ENCODER_BACKEND=torch`
  or `--video-encoder-backend torch`.
- `run_veatic_tribe_cache.py` now has `--video-encoder-backend auto|mlx|torch`.
  Its default is `mlx`, and the MLX cache root is separated as
  `/Volumes/onn. Drive/Neural Bridge/benchmarks/veatic/tribe_cache_mlx` so the
  existing Torch/MPS benchmark cache is not overwritten.

Verification performed:

- Syntax checks passed for the new MLX extractor, adapter, config, and VEATIC
  cache script.
- TRIBE instantiation confirmed `model.data.video_feature` resolves to
  `MlxVjepa2Video` with layers `[0.5, 0.75, 1.0]` and `cache_n_layers=20`,
  preserving the trained cortical feature contract.
- Bounded MLX smoke test loaded the converted checkpoint and emitted finite
  hidden-state means with shape `(1, 3, 1, 1408)`.
- Full 64-frame synthetic MLX smoke test emitted finite selected hidden states
  with shape `(1, 20, 1, 1408)` in about 5 seconds.
- VEATIC dry-run confirmed the new contract records
  `video_encoder_backend: mlx` and `attention: mlx_vjepa2_sdpa`.
- One real 10.56s VEATIC video completed end-to-end through the opt-in MLX path
  into `tribe_cache_mlx/52`, producing `predictions` with shape `(11, 20484)`
  plus the existing missing-modality and segment-retention safeguard arrays.
- The first uncached MLX pass for that video encoded 21 V-JEPA windows in about
  108s, around 5.19s/window. Existing old-cache uncached-like runs have median
  5.85s/window, so current evidence suggests only about 1.13x encoder speedup,
  not enough to hotswap by default without numerical parity and broader timing.

Remaining validation:

The MLX path is wired and shape-valid, but it still needs a numerical parity
comparison against the Transformers V-JEPA2 path on the same real 64-frame
window before old and new benchmark scores should be mixed. Keep the old
`tribe_cache` and new `tribe_cache_mlx` result roots separate until that parity
check is complete.

## Fourth Pass: Hotswap + Current VEATIC Continuation State

Update time: 2026-06-13 18:10 SAST.

The user decided the observed speedup is enough for bulk encoding and requested
the MLX V-JEPA2 path be dropped in as the default. Current implementation:

- `Config.TRIBE_VIDEO_ENCODER_BACKEND` defaults to `mlx`.
- `backend/scripts/run_veatic_tribe_cache.py --video-encoder-backend` defaults
  to `mlx`.
- The old Torch/Transformers path is still available with
  `--video-encoder-backend torch` or `TRIBE_VIDEO_ENCODER_BACKEND=torch`.
- `backend/app/services/tribe_adapter.py` resolves the MLX extractor by default
  when the converted checkpoint exists.
- `backend/scripts/compare_veatic_tribe_caches.py` was added for old-vs-MLX
  cache parity checks.

Observed parity:

- Video `52` old-vs-MLX:
  - predictions shape identical: `(11, 20484)`
  - prediction correlation: `0.9999996`
  - mean absolute difference: about `7e-05`
  - `modality_missing_flags` identical
  - `segment_retention_features` identical

Observed speed:

- Old uncached-like median: about `5.85s/window`.
- MLX observed: about `5.15-5.2s/window`.
- Estimated bulk encoding speedup: about `1.13x`.

Important cache policy after user correction:

- Do not discard the first 47 old-cache videos. The user wants to reuse them.
- Complete only the missing tail videos into the original cache root:
  `/Volumes/onn. Drive/Neural Bridge/benchmarks/veatic/tribe_cache`.
- Do not switch the 50-video benchmark to the separate
  `tribe_cache_mlx` root for this run.

Current live process when checked:

```text
PID 16415
command: backend/scripts/run_veatic_tribe_cache.py --run-mode cortical_fast_default --limit 50 --cache-dir "/Volumes/onn. Drive/Neural Bridge/benchmarks/veatic/tribe_cache"
```

Current first-50 cache status from `cache_status.json` at 18:10:

```text
45/50 marked complete with raw present
95 and 19 have raw files but their status files were clobbered to complete=false by interrupted MLX reruns
70, 10, 7 still need raw completion
```

Continuation note:

- Verify whether PID `16415` is still running.
- If it is running, let it finish unless it is clearly stuck.
- If not running, run only `70,10,7` with MLX into the original cache root.
- Repair `95` and `19` status files if their `tribe_raw_output.npz` files have
  valid `predictions` arrays.
- Then run the cached 50-video benchmark gates.

The fully updated continuation handover is:

```text
docs/handoffs/20260613_veatic_50_mps_pca_handover.md
```
