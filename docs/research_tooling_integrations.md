# Research Tooling Integrations

Neural Bridge uses the real upstream Optuna, Polars, MLflow, and SHAP packages. The integration layer adds project-specific split, provenance, claim-boundary, and Apple-accelerator contracts; it does not reimplement those projects.

## Install And Verify

From `backend/`:

```bash
uv sync --extra research-tooling
uv run python scripts/verify_research_tooling_integrations.py
```

The verifier performs real work through all four libraries and fails if MLX GPU or PyTorch MPS is unavailable. These tools are infrastructure for future experiments. Their output is exploratory and cannot promote or overwrite canonical benchmark evidence.

## Device Contract

| Integration | Official upstream role | Neural Bridge Apple-silicon path |
|---|---|---|
| Optuna | Study orchestration and parameter selection | Every objective must return an explicit matching `mlx` or `mps` attestation; the adapter probes that accelerator first and rejects CPU objectives. |
| Polars | Rust-native lazy/streaming table execution | Polars performs the table query, then numeric results transfer directly to MLX without a pandas intermediary. Official Polars GPU execution is NVIDIA RAPIDS-only, so the Rust query itself is not falsely described as Metal/MPS. |
| MLflow | Experiment metadata, metrics, and artifacts | Every run records a successfully probed MLX/MPS backend plus dataset, split, feature, target, architecture, protocol, seed, and Git provenance. New runs are always tagged exploratory and noncanonical. |
| SHAP | Model-behavior attribution | Official model-agnostic SHAP owns masking and explanation logic; every expensive model prediction batch executes through the supplied MLX model callback. SHAP's host-side masking is not falsely described as fully Metal-native. |

There is no credible upstream Metal execution engine for Polars or general SHAP masking as of 2026-07-14. This is not a reason to discard either library: the hybrid boundaries keep their intended orchestration/data work intact while moving Neural Bridge model and tensor compute onto MLX/MPS.

## Scientific Safety Contracts

- Optuna accepts only disjoint inner-training and inner-validation row indices. Held-out test rows are not part of its API.
- SHAP background data must be explicitly attested as inner-training-only.
- SHAP explanations describe model behavior; they are not causal neural evidence, individual profiling, or a benchmark promotion gate.
- MLflow records exploratory runs. Canonical promotion still requires the existing deterministic evidence workflow and claim audits.
- Polars-to-MLX accepts numeric columns only and verifies the MLX GPU before transfer.

## API Location

The reusable adapters live in `backend/integrations/`. The real-package contract tests are in `tests/test_research_tooling_integrations.py`.
