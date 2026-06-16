# VEATIC Benchmark CPU/MPS Audit

Generated: 2026-06-13  
Scope: `/Users/maxsartini/Neural Bridge/backend/scripts/run_veatic_neuro_benchmark.py`

## Summary

The benchmark script now has practical Apple Silicon acceleration paths for the main expensive no-reencode feature work:

- PCA feature extraction now defaults to exact MPS-Gram PCA.
- Approximate MPS power-iteration PCA is available as an explicit fallback/experiment.
- Ridge regression now has a PyTorch MPS solve backend for very high-dimensional raw cortical features.
- Compact and PCA ridge defaults remain CPU because measured MPS overhead is slower for those small/medium design matrices.

This is not a PyTorch kernel patch. PyTorch and MLX still do not expose GPU-native SVD/eigh/QR/solve coverage for all required operations. The implementation works around that at the application layer using supported MPS matrix products and small CPU solves where unavoidable.

## Operation Support Audit

### PyTorch MPS

Tested available / usable:

- `matmul`: supported on MPS.
- `sum`: supported on MPS.
- `sqrt` / `rsqrt`: supported on MPS.
- `torch.linalg.norm`: supported on MPS.
- `torch.linalg.solve`: supported on MPS.
- `torch.linalg.inv`: supported on MPS.
- `torch.linalg.pinv`: callable, but warns/falls back through SVD-related paths.
- `torch.linalg.cholesky`: supported on MPS.

Missing or fallback:

- `torch.linalg.svd`: falls back to CPU.
- `torch.linalg.qr`: not implemented on MPS.
- `torch.linalg.eigh`: not implemented on MPS.
- `torch.linalg.lstsq`: not implemented on MPS.

### MLX

Installed version: `0.31.2`.

MLX exposes:

- `mx.linalg.svd`
- `mx.linalg.eigh`
- `mx.linalg.qr`
- `mx.linalg.solve`
- `mx.linalg.inv`
- `mx.linalg.cholesky`

But GPU stream support is not sufficient for this benchmark:

- `mx.linalg.svd(..., stream=mx.gpu)`: not supported on GPU.
- `mx.linalg.eigh(..., stream=mx.gpu)`: not supported on GPU.
- `mx.linalg.solve(..., stream=mx.gpu)`: not supported on GPU.
- `mx.linalg.inv(..., stream=mx.gpu)`: not supported on GPU.
- `mx.linalg.cholesky(..., stream=mx.gpu)`: not supported on GPU.

MLX CPU versions work, but do not solve the GPU acceleration problem.

## Implemented Patches

File changed:

```text
/Users/maxsartini/Neural Bridge/backend/scripts/run_veatic_neuro_benchmark.py
```

### PCA Backends

New/updated options:

```bash
--pca-backend auto
--pca-backend mps_gram
--pca-backend mps_power
--pca-backend cpu_svd
```

Current default:

```text
auto
```

Current `auto` order:

```text
1. mps_gram
2. mps_power
3. cpu_svd
```

### `mps_gram`

Purpose: exact PCA while moving the largest products to MPS.

Method:

```text
train_z @ train_z.T        -> MPS
row-space eigensolve       -> CPU
basis = train_z.T @ eigvec -> MPS
apply_z @ basis            -> MPS
```

Why CPU remains:

```text
The eigensolve is small relative to cortical vertex dimension, and MPS/MLX do not support GPU eigensolve here.
```

Scientific status:

```text
Exact train-fit PCA semantics.
Safe for benchmark evidence.
PCA is still fit only on train rows inside each split/fold.
```

### `mps_power`

Purpose: approximate GPU-native PCA fallback/experiment.

Method:

```text
Subspace iteration with MPS matmul and MPS modified Gram-Schmidt.
Tiny Rayleigh-Ritz eigensolve on CPU.
```

Scientific status:

```text
Approximate PCA.
Do not treat as equivalent to exact PCA unless a parity test shows benchmark-level equivalence.
Useful as an experimental speed path, not default evidence.
```

### `cpu_svd`

Purpose: original exact NumPy SVD fallback.

Scientific status:

```text
Exact but CPU-bound.
```

## Ridge Regression Backends

New/updated options:

```bash
--ridge-backend auto
--ridge-backend mps_solve
--ridge-backend cpu_pinv
```

Current default:

```text
auto
```

Current behavior:

```text
Use CPU pseudo-inverse for compact/PCA feature sets.
Use MPS solve only when feature count is very high.
Default MPS threshold: 2048 features.
```

Reason:

Microbenchmarks showed MPS solve is slower for compact/PCA-sized designs because transfer and solve overhead dominates.

Measured examples:

```text
500 rows x 20 features:
  cpu_pinv: 0.0004s
  mps_solve: 1.0313s

500 rows x 80 features:
  cpu_pinv: 0.0011s
  mps_solve: 0.0375s

1500 rows x 400 features:
  cpu_pinv: 0.0241s
  mps_solve: 0.4351s

1500 rows x 1024 features:
  cpu_pinv: 0.2253s
  mps_solve: 0.5192s
```

For raw cortical feature dimensions, the MPS path uses dual-form ridge:

```text
beta = X.T @ solve(X @ X.T + alpha I, y)
```

This avoids a huge `20485 x 20485` CPU pseudo-inverse.

## PCA Speed Checks

Representative synthetic shape:

```text
train rows: 1000
vertices: 20484
components: 64
```

Measured:

```text
mps_gram: 0.845s, exact, explained variance 0.0904
mps_power: 0.826s, approximate, explained variance 0.0794
cpu_svd: 2.281s, exact, explained variance 0.0904
```

Conclusion:

```text
mps_gram is the default because it is exact and about 2.7x faster than CPU SVD on representative cortical matrices.
mps_power is not default because it is approximate and did not preserve explained variance as well.
```

## Remaining CPU Sections

These remain CPU by design:

- Metrics (`MAE`, `RMSE`, `Pearson`, `Spearman`).
- Rank calculation for Spearman.
- JSON serialization/report generation.
- Small compact-feature transforms.
- Small ridge solves where CPU is measurably faster.
- The small eigensolve inside MPS-Gram PCA.

These are not currently worth moving to MPS:

- Metrics/reporting are not the bottleneck.
- Small ridge solves are faster on CPU.
- MLX GPU does not support the needed solve/eigh/SVD kernels.

## Current Cache Status During Audit

At last check:

```text
VEATIC 50 target videos complete: 43/50
failed: 0
current incomplete: 58
not started/missing: 64, 1, 97, 70, 10, 7
```

## Recommended Benchmark Runtime Settings

For exact PCA acceleration:

```bash
--pca-backend auto
```

or explicitly:

```bash
--pca-backend mps_gram
```

For compact/global/PCA benchmarks:

```bash
--ridge-backend auto
```

For experimental raw cortical ridge:

```bash
--feature-mode cortical_raw_ridge --ridge-backend mps_solve
```

## Guardrail

Do not claim a pure native-MPS SVD/eigh patch. The accurate claim is:

```text
The benchmark now uses MPS for the expensive cortical matrix products and high-dimensional ridge operations, while retaining exact train-fit PCA semantics through a small CPU eigensolve where Apple/PyTorch/MLX do not currently expose GPU kernels.
```

