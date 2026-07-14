"""Official Polars lazy I/O with direct MLX handoff for numeric compute."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from ._optional import require_upstream
from .acceleration import require_accelerator


def scan_table(path: str | Path, *, columns: Iterable[str] | None = None) -> Any:
    pl = require_upstream("polars")
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        lazy = pl.scan_parquet(source)
    elif suffix in {".csv", ".tsv"}:
        lazy = pl.scan_csv(source, separator="\t" if suffix == ".tsv" else ",")
    elif suffix in {".arrow", ".ipc", ".feather"}:
        lazy = pl.scan_ipc(source)
    else:
        raise ValueError(f"Unsupported table format: {source.suffix}")
    selected = list(columns) if columns is not None else None
    return lazy.select(selected) if selected is not None else lazy


def collect_pandas(
    path: str | Path,
    *,
    columns: Iterable[str] | None = None,
    streaming: bool = True,
) -> Any:
    lazy = scan_table(path, columns=columns)
    frame = lazy.collect(engine="streaming" if streaming else "auto")
    return frame.to_pandas(use_pyarrow_extension_array=False)


def collect_mlx(
    path: str | Path,
    *,
    columns: Iterable[str],
    dtype: str = "float32",
    streaming: bool = True,
) -> Any:
    """Use Polars for native query execution, then transfer directly to MLX GPU."""

    require_accelerator("mlx")
    mx = require_upstream("mlx.core")
    lazy = scan_table(path, columns=columns)
    frame = lazy.collect(engine="streaming" if streaming else "auto")
    numpy_values = frame.to_numpy()
    if numpy_values.dtype.kind not in {"b", "i", "u", "f"}:
        raise TypeError("collect_mlx requires numeric Polars columns")
    mlx_dtype = getattr(mx, dtype, None)
    if mlx_dtype is None:
        raise ValueError(f"Unknown MLX dtype: {dtype}")
    values = mx.asarray(numpy_values, dtype=mlx_dtype)
    mx.eval(values)
    return values


def write_table(data: Any, path: str | Path) -> Path:
    pl = require_upstream("polars")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = data if isinstance(data, pl.DataFrame) else pl.from_pandas(data)
    suffix = destination.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        frame.write_parquet(destination)
    elif suffix == ".csv":
        frame.write_csv(destination)
    elif suffix == ".tsv":
        frame.write_csv(destination, separator="\t")
    elif suffix in {".arrow", ".ipc", ".feather"}:
        frame.write_ipc(destination)
    else:
        raise ValueError(f"Unsupported table format: {destination.suffix}")
    return destination
