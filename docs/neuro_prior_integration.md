# Neuro-Prior Integration

See `docs/local_first_prediction_pipeline.md` for the product-level thesis, local execution model, and scientific caveats.

Neural Bridge can now generate an optional stimulus-level `neuro_prior` before persona and simulation config generation.

Runtime order:

1. Use TRIBE v2 when the configured backend is installed and inference succeeds.
2. Summarize TRIBE cortical output with the Nilearn Destrieux fsaverage5 surface atlas when predictions have 20,484 vertices.
3. Map ROI proxy axes into conservative behavioural modifiers.
4. Feed the resulting prior and modifiers into Qwen-powered persona and simulation configuration prompts.
5. Fall back to Qwen/LM Studio proxy JSON when TRIBE is unavailable and fallback is enabled.

Accuracy attribution:

- TRIBE's predicted cortical response is not currently the weakest assumed
  component.
- `NeuroRoiCalibrator` performs a transparent but heuristic conversion from
  cortical parcels to behavioural axes.
- `NeuroPriorMapper` applies fixed, hand-authored formulas to produce simulation
  modifiers.
- The prior is then injected into persona generation, simulation configuration,
  and per-round agent context. Repeated prompt injection can amplify,
  homogenize, or distort the original signal.
- Low end-to-end accuracy must therefore trigger mapping and injection
  ablations before changing or rejecting TRIBE.

See `docs/benchmarking_accuracy.md` for the required shuffled-prior,
neutral-prior, inverted-prior, oracle-prior, dose-response, and direct-modifier
tests.

The app does not import TRIBE at Flask startup. TRIBE failures, gated Hugging Face access, missing feature extractors, or missing MLX assets should not prevent normal simulation preparation unless `NEURO_PRIOR_STRICT=true`.

Apple Silicon status:

- Torch is installed with MPS support. On Apple Silicon, the production
  multimodal path uses MPS for V-JEPA2 and other supported Torch extractors,
  MLX for transcription and the TRIBE cortical head, and CPU only where the
  dependency has no safe Metal implementation.
- The official `facebook/tribev2` checkpoint is expected at `models/tribe/facebook-tribev2`.
- Text inference requires the exact TRIBE text extractor model `meta-llama/Llama-3.2-3B` as a Hugging Face Transformers model. If Hugging Face access is gated, place an independently downloaded HF-format copy under `models/upstream-encoders/meta-llama-Llama-3.2-3B` or set `TRIBE_TEXT_ENCODER_LOCAL_DIR`.
- LM Studio can run LLaMA/Qwen in parallel for proxy LLM calls, but a GGUF model served through LM Studio is not the same as TRIBE's extractor input.
- MLX-format LLaMA folders from LM Studio are supported through the `MlxText` NeuralSet adapter when `TRIBE_TEXT_ENCODER_MLX_DIR` points to a complete folder with `config.json`, tokenizer files, and final `.safetensors` weights. Partial `downloading_*.part` files are rejected.
- The MLX adapter reads hidden states directly from `mlx-lm`; it is an Apple Silicon compatibility bridge, not the exact original HF extractor path.
- `TRIBE_TEXT_EVENTS_DIRECT=true` skips the expensive text-to-speech plus transcription path for text stimuli.
- On Apple Silicon, use `TRIBE_TRANSCRIPTION_BACKEND=mlx` for Metal-accelerated transcription with word timestamps. WhisperX/CTranslate2 does not support MPS and remains available through `TRIBE_TRANSCRIPTION_BACKEND=whisperx` as a slower CPU forced-alignment fallback.
- On the 32 GB Mac Studio, the official 64-frame cortical V-JEPA2 ViT-G
  contract runs on MPS through exact query-chunked scaled dot-product
  attention. Selective hidden-state capture retains only the layers consumed
  by Neuralset and averages tokens inside the forward hooks. Local parity
  tests were bit-identical to the unoptimized selected outputs while reducing
  isolated cortical-window time by about 41% and driver memory from about
  5.53 GB to 3.51 GB. Keep the per-process MPS memory cap enabled and run
  stages sequentially; uncapped attention can exhaust unified memory.
- The current extraction contract is
  `official_64_frame_exact_chunked_attention`. Do not mix earlier 32-frame
  adaptation outputs into its benchmark manifest.

Generated files live in each simulation directory:

- `neuro_prior.json`
- `neuro_prior_modifiers.json`
- `tribe_raw_output.npz` when real TRIBE output is saved
- `tribe_segments.json` when available
- `tribe_summary.json` when available, including `roi_summary`, `behavioural_axes`, and `calibration_trace`

API:

- `POST /api/simulation/create` accepts `enable_neuro_priors`, `stimulus_text`, and `stimulus_type`.
- `POST /api/simulation/prepare` can override the same fields before preparation starts.
- `GET /api/simulation/<simulation_id>/neuro-prior` returns the saved prior and modifiers.

Smoke tests:

- `PYTHONPATH=backend backend/.venv/bin/python backend/scripts/test_neuro_prior.py`
- `PYTHONPATH=backend backend/.venv/bin/python backend/scripts/test_tribe_adapter.py`
- `PYTHONPATH=backend backend/.venv/bin/python backend/scripts/test_tribe_model_load.py`
- `PYTHONPATH=backend backend/.venv/bin/python backend/scripts/check_tribe_encoder_assets.py`
- `PYTHONPATH=backend backend/.venv/bin/python backend/scripts/check_neuro_calibration_assets.py`
- `PYTHONPATH=backend backend/.venv/bin/python backend/scripts/test_neuro_roi_calibrator.py`
- `PYTHONPATH=backend backend/.venv/bin/python backend/scripts/test_mlx_text_extractor.py`
- `PYTHONPATH=backend backend/.venv/bin/python backend/scripts/test_tribe_text_end_to_end.py`
# Apple Silicon MLX Acceleration

The local default uses `zimengxiong/tribev2-mlx` for the TRIBE brain encoder:

- Converted output parity was verified locally against the released PyTorch
  checkpoint with maximum absolute difference below `1e-6`.
- Warm synthetic encoder inference improved from approximately `76 ms` with
  CPU PyTorch to `8.3 ms` with MLX.
- Raw-text smoke inference completed successfully in approximately `7.3 s`.
- LLaMA text, Wav2Vec-BERT audio, and V-JEPA2 video feature extraction remain
  separate upstream costs. MLX accelerates transcription and the brain-encoder
  stage; exact chunked MPS attention and selective hidden-state capture
  accelerate the dominant V-JEPA2 video stage.

Configuration:

```dotenv
TRIBE_MLX_ENABLED=true
NEURO_PRIOR_MODE=tribe_mlx
NEURO_PRIOR_BACKEND_PRIORITY=tribe_mlx,apple_silicon_tribe,proxy,official_tribe,disabled
```

Run the parity check:

```bash
PYTHONPATH=backend backend/.venv/bin/python backend/scripts/test_tribe_mlx_encoder.py
```
