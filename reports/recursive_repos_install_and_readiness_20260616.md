# Recursive Repos Install And Readiness - 2026-06-16

## 1. Executive Summary

This pass was a clean upstream readiness check only. No Neuro Bridge model files, wrappers, recursive heads, training scripts, benchmark scripts, production configs, or data preparation code were created or modified.

Cloned upstream repos:

- `external_repos/tinyrecursivemodels`
  - URL: `https://github.com/samsungsailmontreal/tinyrecursivemodels`
  - Branch: `main`
  - Commit: `c01103738605ba39d1430519b1ee0c62f4c707f8`
  - License: MIT
- `external_repos/HRM`
  - URL: `https://github.com/sapientinc/HRM`
  - Branch: `main`
  - Commit: `ac15626f8db096a63c775b84c9dc868776a6feda`
  - License: Apache-2.0

Installed in this pass: nothing.

Nothing was installed because both repos document CUDA-oriented training paths. TRM's pinned requirements include `torch==2.7.0+cu126` and `triton==3.3.0`; HRM requires CUDA extensions and FlashAttention. Installing those into this Mac/MPS project environment would violate the safety rule against CUDA-only packages and main-environment dependency changes.

Readiness status:

| Repo | Readable | Basic imports | Basic run check | CUDA assumption | Mac/MPS safety |
| --- | --- | --- | --- | --- | --- |
| TinyRecursiveModels | Yes | Partial: model modules import; `puzzle_dataset` needs `argdantic`; `pretrain.py --help` needs `coolname` and later `adam_atan2` | Tiny synthetic CPU forward pass of `TinyRecursiveReasoningModel_ACTV1` succeeded | Yes in README and training script; `pretrain.py` hard-codes CUDA moves | Source reading and tiny CPU model probe are safe; upstream training path is not safe as-is |
| HRM | Yes | Partial: `puzzle_dataset` imports; `models.hrm.hrm_act_v1` fails because `models/layers.py` imports `flash_attn` | Not run; model import blocked by FlashAttention | Yes in README, FlashAttention requirement, CUDA install docs, and `.cuda()` calls | Not safe for full import/training on this Mac without upstream changes or CUDA hardware |

Current Python observed: `Python 3.13.9`. Installed ambient packages included `torch 2.9.1`, `pydantic 2.12.4`, `hydra 1.3.2`, `omegaconf 2.3.0`, `einops 0.8.1`, and `wandb 0.23.1`. Missing ambient packages included `argdantic`, `adam_atan2`, and `triton`.

No long training, dataset download, checkpoint download, submodule initialization, or full dataset preparation was run.

## 2. Current Project Residue Check

`git status` result:

```text
fatal: not a git repository (or any of the parent directories): .git
```

Residue search was intentionally stopped after the user directed to move on. Captured findings before cloning upstream repos:

| Reference | Location | Classification | Notes |
| --- | --- | --- | --- |
| `tests/test_neuro_recursive_shapes.py::test_nb_hrm_shapes_and_q_contract` | `.pytest_cache/v/cache/nodeids:3` | harmless documentation | Pytest cache only; not active source. |
| `tests/test_neuro_recursive_shapes.py::test_nb_trm_shapes_and_q_contract` | `.pytest_cache/v/cache/nodeids:4` | harmless documentation | Pytest cache only; not active source. |
| `tests/test_official_numeric_recursive_heads.py::test_official_numeric_hrm_forward_grad_and_q_init` | `.pytest_cache/v/cache/nodeids:7` | harmless documentation | Pytest cache only; likely a deleted-test cache entry. |
| `tests/test_official_numeric_recursive_heads.py::test_official_numeric_trm_can_overfit_tiny_batch` | `.pytest_cache/v/cache/nodeids:8` | harmless documentation | Pytest cache only; likely a deleted-test cache entry. |
| `tests/test_official_numeric_recursive_heads.py::test_official_numeric_trm_forward_grad_and_q_init` | `.pytest_cache/v/cache/nodeids:9` | harmless documentation | Pytest cache only; likely a deleted-test cache entry. |
| `tests/test_official_numeric_recursive_heads.py::test_ptrm_uses_official_numeric_trm_rollouts_and_noise_changes_scores` | `.pytest_cache/v/cache/nodeids:10` | harmless documentation | Pytest cache only; likely a deleted-test cache entry. |
| `tests/test_ptrm_rollout_shapes.py::test_ptrm_rollout_shapes` | `.pytest_cache/v/cache/nodeids:11` | harmless documentation | Pytest cache only; likely a deleted-test cache entry. |
| `GhRM` substring inside package integrity hash | `frontend/package-lock.json:360` | unrelated | Random base64-like package-lock integrity text, not HRM code. |

Recommended cleanup actions, not performed:

- Do not touch production code based on these findings.
- Optionally clear `.pytest_cache` later if a clean test-cache state is desired.
- If a fuller residue audit is needed later, rerun a targeted search excluding `.venv`, `node_modules`, notebooks, large generated benchmark JSON, and upstream clones.

## 3. TinyRecursiveModels Repo Notes

Path: `external_repos/tinyrecursivemodels`

Commit hash: `c01103738605ba39d1430519b1ee0c62f4c707f8`

License: MIT, copyright Samsung Electronics Co., Ltd.

Submodules: none reported by `git submodule status`.

Dependency files:

- `requirements.txt`
- `specific_requirements.txt`

Dependency notes:

- README expectation: Python 3.10 or similar.
- README expectation: CUDA 12.6 or similar.
- `requirements.txt` includes `torch`, `adam-atan2`, `einops`, `tqdm`, `coolname`, `pydantic`, `argdantic`, `wandb`, `omegaconf`, `hydra-core`, `huggingface_hub`, `packaging`, `ninja`, `wheel`, `setuptools`, `setuptools-scm`, `pydantic-core`, `numba`, and `triton`.
- `specific_requirements.txt` pins `torch==2.7.0+cu126` and `triton==3.3.0`.
- Likely unsafe on this Mac/MPS setup: `torch==2.7.0+cu126`, `triton`, and any CUDA-specific install index. `adam-atan2` is an optimizer dependency that should be tested only in an isolated venv because it may involve compiled components.
- Likely safe pure-Python dependencies: `einops`, `tqdm`, `coolname`, `pydantic`, `argdantic`, `omegaconf`, `hydra-core`, `huggingface_hub`, `packaging`, `wandb`.
- Separate venv needed before attempting any fuller import/help/training check.
- Proposed isolated install command, not executed:

```bash
python3.10 -m venv .venvs/trm_repo
source .venvs/trm_repo/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install torch torchvision torchaudio
python -m pip install einops tqdm coolname pydantic argdantic wandb omegaconf hydra-core huggingface_hub packaging ninja setuptools-scm numba
# Do not install torch+cu126 or triton on this Mac.
# Test adam-atan2 separately inside the venv before using pretrain.py.
```

README training instructions:

- Dataset preparation entrypoints:
  - `python -m dataset.build_arc_dataset ...`
  - `python dataset/build_sudoku_dataset.py ...`
  - `python dataset/build_maze_dataset.py`
- Main training entrypoint:
  - `python pretrain.py ...` for single process.
  - `torchrun --nproc-per-node ... pretrain.py ...` for multi-GPU experiments.
- README examples assume L40S or H100 CUDA GPUs and long runtimes from hours to days.

Main model files:

- `models/recursive_reasoning/trm.py`
- `models/recursive_reasoning/hrm.py`
- `models/recursive_reasoning/trm_hier6.py`
- `models/recursive_reasoning/trm_singlez.py`
- `models/recursive_reasoning/transformers_baseline.py`
- Shared modules: `models/layers.py`, `models/losses.py`, `models/sparse_embedding.py`, `models/common.py`, `models/ema.py`

Main training files:

- `pretrain.py`
- `puzzle_dataset.py`
- `utils/functions.py`

Main evaluation files:

- `evaluators/arc.py`
- Evaluation also occurs inside `pretrain.py` via `evaluate(...)`; there is no separate top-level `evaluate.py` in TRM.

Config files:

- `config/cfg_pretrain.yaml`
- `config/arch/trm.yaml`
- `config/arch/hrm.yaml`
- `config/arch/trm_hier6.yaml`
- `config/arch/trm_singlez.yaml`
- `config/arch/transformers_baseline.yaml`

Dataset files:

- `dataset/build_arc_dataset.py`
- `dataset/build_sudoku_dataset.py`
- `dataset/build_maze_dataset.py`
- `dataset/common.py`
- Bundled ARC JSONs under `kaggle/combined/`

Expected input format:

- `PuzzleDataset` expects generated dataset folders with split subfolders such as `train/` and `test/`.
- Each split has `dataset.json`.
- Per set, it memory maps NumPy arrays named like `<set>__inputs.npy`, `<set>__labels.npy`, `<set>__puzzle_identifiers.npy`, `<set>__puzzle_indices.npy`, and `<set>__group_indices.npy`.
- Runtime batch object keys are `inputs`, `labels`, and `puzzle_identifiers`.
- `inputs`: integer token tensor shaped `[batch, seq_len]`.
- `labels`: integer token tensor shaped `[batch, seq_len]`, with ignored positions converted to `-100`.
- `puzzle_identifiers`: integer tensor shaped `[batch]`.
- Metadata includes `pad_id`, `ignore_label_id`, `blank_identifier_id`, `vocab_size`, `seq_len`, `num_puzzle_identifiers`, `total_groups`, `mean_puzzle_examples`, `total_puzzles`, and `sets`.

Expected output format:

- Model outputs include:
  - `logits`: `[batch, seq_len, vocab_size]`
  - `q_halt_logits`: `[batch]`
  - `q_continue_logits`: `[batch]`
  - optionally `target_q_continue`
- Loss head adds `preds` by argmax over logits and computes exact/sequence accuracy plus Q-head losses.
- Checkpoints are `torch.save(model.state_dict(), checkpoint_path/step_<step>)`; predictions can be saved as `step_<step>_all_preds.<rank>`.

Training mechanics:

- Training entrypoint: `pretrain.py` with Hydra config `config/cfg_pretrain.yaml`.
- Required config fields include `arch`, `data_paths`, `global_batch_size`, `epochs`, `lr`, `lr_min_ratio`, `lr_warmup_steps`, `weight_decay`, `beta1`, `beta2`, `puzzle_emb_lr`, and `puzzle_emb_weight_decay`.
- Device selection is CUDA-specific: model creation uses `with torch.device("cuda")`; batches use `{k: v.cuda() for k, v in batch.items()}`; checkpoint load uses `map_location="cuda"`; distributed backend is `nccl`.
- Loss is computed by `models/losses.py::ACTLossHead`, combining token cross-entropy (`stablemax_cross_entropy` by default) with Q-halt loss and optional Q-continue loss.
- Recurrence happens in `TinyRecursiveReasoningModel_ACTV1_Inner.forward` in `models/recursive_reasoning/trm.py`: repeated `H_cycles` and `L_cycles` update `z_L` and `z_H`.
- TRM hierarchy: default TRM uses one `L_level` module and two latent states `z_H`, `z_L`; it is simpler than HRM. TRM variants add or remove hierarchy in separate files.

Basic import/run status:

- `models.common`: OK
- `models.layers`: OK
- `models.losses`: OK
- `models.recursive_reasoning.trm`: OK
- `models.recursive_reasoning.hrm`: OK
- `puzzle_dataset`: blocked by missing `argdantic`
- `python3 pretrain.py --help`: blocked first by missing `coolname`
- Tiny synthetic CPU forward pass of `TinyRecursiveReasoningModel_ACTV1`: OK
  - Output `logits`: `(2, 5, 11)`, `torch.float32`
  - Output `q_halt_logits`: `(2,)`, `torch.float32`
  - Output `q_continue_logits`: `(2,)`, `torch.float32`
  - Carry `halted`: `(2,)`, `torch.bool`

Errors encountered:

- `ModuleNotFoundError: No module named 'argdantic'`
- `ModuleNotFoundError: No module named 'coolname'`

## 4. HRM Repo Notes

Path: `external_repos/HRM`

Commit hash: `ac15626f8db096a63c775b84c9dc868776a6feda`

License: Apache License 2.0.

Submodules:

- `dataset/raw-data/ARC-AGI`
- `dataset/raw-data/ARC-AGI-2`
- `dataset/raw-data/ConceptARC`

Submodules were not initialized because they are dataset sources and not required for source inspection.

Dependency files:

- `requirements.txt`

Dependency notes:

- README expects PyTorch and CUDA.
- README says CUDA extensions must be built.
- README documents CUDA 12.6 installation and CUDA PyTorch wheel installation.
- README requires FlashAttention 3 for Hopper or FlashAttention 2 for Ampere/earlier.
- `requirements.txt` includes `torch`, `adam-atan2`, `einops`, `tqdm`, `coolname`, `pydantic`, `argdantic`, `wandb`, `omegaconf`, `hydra-core`, and `huggingface_hub`.
- Likely unsafe on this Mac/MPS setup: CUDA toolkit, CUDA PyTorch wheels, `flash-attn`, FlashAttention source builds, and any `nccl`-based distributed assumptions.
- Likely safe pure-Python dependencies: `einops`, `tqdm`, `coolname`, `pydantic`, `argdantic`, `wandb`, `omegaconf`, `hydra-core`, `huggingface_hub`.
- Separate venv is required before any fuller check, but full HRM import is not expected to work on this Mac without replacing or bypassing FlashAttention.
- Proposed isolated install command for Mac source inspection only, not executed:

```bash
python3.10 -m venv .venvs/hrm_repo
source .venvs/hrm_repo/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install torch torchvision torchaudio
python -m pip install einops tqdm coolname pydantic argdantic wandb omegaconf hydra-core huggingface_hub
# Do not install CUDA toolkit, CUDA PyTorch wheels, flash-attn, or NCCL-only paths on this Mac.
```

README training instructions:

- Dataset preparation:
  - `git submodule update --init --recursive`
  - `python dataset/build_arc_dataset.py`
  - `python dataset/build_arc_dataset.py --dataset-dirs dataset/raw-data/ARC-AGI-2/data --output-dir data/arc-2-aug-1000`
  - `python dataset/build_sudoku_dataset.py ...`
  - `python dataset/build_maze_dataset.py`
- Main training entrypoint:
  - `python pretrain.py ...` for single process examples.
  - `torchrun --nproc-per-node 8 pretrain.py ...` for full-scale experiments.
- Evaluation entrypoint:
  - `torchrun --nproc-per-node 8 evaluate.py checkpoint=<CHECKPOINT_PATH>`
  - ARC finalization uses `arc_eval.ipynb`.

Main model files:

- `models/hrm/hrm_act_v1.py`
- Shared modules: `models/layers.py`, `models/losses.py`, `models/sparse_embedding.py`, `models/common.py`

Main training files:

- `pretrain.py`
- `puzzle_dataset.py`
- `utils/functions.py`

Main evaluation files:

- `evaluate.py`
- `arc_eval.ipynb`

Config files:

- `config/cfg_pretrain.yaml`
- `config/arch/hrm_v1.yaml`

Dataset files:

- `dataset/build_arc_dataset.py`
- `dataset/build_sudoku_dataset.py`
- `dataset/build_maze_dataset.py`
- `dataset/common.py`
- Dataset submodule placeholders under `dataset/raw-data/`

Expected input format:

- `PuzzleDataset` expects one generated dataset root in `data_path`.
- Each split has `dataset.json`.
- Per set, it memory maps `<set>__inputs.npy`, `<set>__labels.npy`, `<set>__puzzle_identifiers.npy`, `<set>__puzzle_indices.npy`, and `<set>__group_indices.npy`.
- Runtime batch object keys are `inputs`, `labels`, and `puzzle_identifiers`.
- `inputs`: integer token tensor shaped `[batch, seq_len]`.
- `labels`: integer token tensor shaped `[batch, seq_len]`, with ignored positions converted to `-100`.
- `puzzle_identifiers`: integer tensor shaped `[batch]`.
- Metadata includes `pad_id`, `ignore_label_id`, `blank_identifier_id`, `vocab_size`, `seq_len`, `num_puzzle_identifiers`, `total_groups`, `mean_puzzle_examples`, and `sets`.

Expected output format:

- Model outputs include:
  - `logits`: `[batch, seq_len, vocab_size]`
  - `q_halt_logits`: `[batch]`
  - `q_continue_logits`: `[batch]`
  - optionally `target_q_continue`
- `evaluate.py` defaults to saving `inputs`, `labels`, `puzzle_identifiers`, `logits`, `q_halt_logits`, and `q_continue_logits`.
- Checkpoints are `torch.save(model.state_dict(), checkpoint_path/step_<step>)`; evaluation loads with `map_location="cuda"`.

Training mechanics:

- Training entrypoint: `pretrain.py` with Hydra config `config/cfg_pretrain.yaml`.
- Required config fields include `arch`, `data_path`, `global_batch_size`, `epochs`, `lr`, `lr_min_ratio`, `lr_warmup_steps`, `weight_decay`, `beta1`, `beta2`, `puzzle_emb_lr`, and `puzzle_emb_weight_decay`.
- Device selection is CUDA-specific: model creation uses `with torch.device("cuda")`; batches use `{k: v.cuda() for k, v in batch.items()}`; evaluation tensors are allocated on CUDA; distributed backend is `nccl`.
- Loss is computed by `models/losses.py::ACTLossHead`, combining token cross-entropy (`stablemax_cross_entropy` by default) with Q-halt loss and optional Q-continue loss.
- Recurrence and hierarchy happen in `models/hrm/hrm_act_v1.py`.
- Hierarchy is explicit:
  - `H_level`: high-level recurrent module.
  - `L_level`: low-level recurrent module.
  - latent state contains `z_H` and `z_L`.
  - `HierarchicalReasoningModel_ACTV1_Inner.forward` loops over `H_cycles` and `L_cycles`; low-level updates consume `z_H + input_embeddings`, then high-level updates consume `z_L`.

Basic import/run status:

- `models.common`: OK
- `models.losses`: OK
- `puzzle_dataset`: OK
- `models.layers`: blocked by missing `flash_attn`
- `models.hrm.hrm_act_v1`: blocked by missing `flash_attn`
- `python3 pretrain.py --help`: blocked first by missing `coolname`
- No model forward run was attempted because model import is blocked by FlashAttention.

Errors encountered:

- `ModuleNotFoundError: No module named 'flash_attn'`
- `ModuleNotFoundError: No module named 'coolname'`

## 5. TRIBE / Neuro Bridge Readiness Notes

Relevant existing Neuro Bridge artifact types:

- TRIBE cortical feature or prediction tensors, especially `tribe_raw_output.npz` style artifacts containing `predictions` and sometimes `subcortical_predictions`.
- Temporal windows and lagged/context-window outputs from VEATIC temporal fairness/context runs.
- Continuous arousal/valence or affect target files.
- Event masks and spike masks from VEATIC event-conditioned retests.
- Benchmark split files and manifests under `benchmarks/veatic/`.
- Current baseline prediction/result artifacts under `benchmarks/veatic/` and `outputs/veatic_*`.

What each upstream repo expects:

- Discrete puzzle dataset folders, not arbitrary numeric feature archives.
- Integer token sequences `inputs` and integer token targets `labels`, both shaped `[batch, seq_len]`.
- A `puzzle_identifiers` vector shaped `[batch]`.
- A finite `vocab_size` and token-classification loss.
- Dataset index arrays for grouping and batching puzzles.

What existing TRIBE/Neuro Bridge artifacts appear to provide:

- Continuous numeric arrays over time, such as TRIBE cortical/subcortical predictions and derived cortical features.
- Time-aligned target traces such as arousal and valence.
- Event/spike masks and split metadata.
- Baseline predictions and benchmark metrics, not tokenized puzzle grids.

Exact mismatch:

- Upstream HRM/TRM are sequence token classifiers over a finite vocabulary; Neuro Bridge artifacts are continuous time-series regression/classification inputs and targets.
- Upstream batches are puzzle-centric and grouped by puzzle identifiers; Neuro Bridge data is video/time-window-centric and split by video/time/fold policies.
- Upstream loss predicts a token per sequence position; Neuro Bridge targets are continuous affect values, event/spike labels, or benchmark-specific target transforms.
- Upstream training code hard-codes CUDA and puzzle dataset layout; Neuro Bridge currently has cached scientific benchmark artifacts and Apple Silicon constraints.

Files to read later, read-only:

- `backend/app/services/tribe_adapter.py`
- `backend/scripts/run_veatic_neuro_benchmark.py`
- `benchmarks/veatic/*.json`
- `benchmarks/veatic/*.csv`
- `outputs/veatic_124_temporal_context_v2_20260616_1557/reused_artifacts_manifest.csv`
- Any existing `tribe_raw_output.npz` files under model/cache or benchmark cache roots.
- Existing VEATIC split/manifest files before any training proposal.

Files that must not be touched yet:

- Neuro Bridge model files.
- Benchmark scripts.
- Production configs.
- Data preparation scripts.
- Existing VEATIC/TRIBE cache files.
- Existing baseline prediction artifacts.

Minimum information needed before any coding step:

- Exact paths and shapes for cached TRIBE/cortical feature tensors.
- Exact target arrays and shapes for arousal, valence, affect, event masks, and spike masks.
- Exact split definitions and whether splits are video-level, temporal, or fold-based.
- Sampling rate and alignment policy for every tensor source.
- Whether future objective is regression, binary event prediction, multi-target affect prediction, or tokenized proxy prediction.
- Frozen baseline outputs and evaluation metrics to protect before any experimental model work.

## 6. Next Step Recommendation

Next operational step only: create a read-only data inventory of existing TRIBE/Neuro Bridge benchmark artifacts and document exact tensor shapes, target shapes, split files, sampling rates, and alignment policy.

Do not write model code yet. Do not write connectors yet. Do not change benchmark scripts yet. Do not modify Neuro Bridge behavior yet.
