#!/usr/bin/env python3
"""Audit local Apple-Silicon V-JEPA 2.1 runtime dependencies.

This tool is intentionally read-only. It does not install packages. Use it to
record what the current environment can support before promoting optional
decode/profiling dependencies into the main environment.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def package_version(name: str, import_name: str | None = None) -> dict[str, Any]:
    import_name = import_name or name
    try:
        module = importlib.import_module(import_name)
    except Exception as exc:  # noqa: BLE001 - dependency audit should report failures
        return {
            "dependency": name,
            "import_name": import_name,
            "installed": False,
            "version": "",
            "status": "import_failed",
            "detail": str(exc),
        }
    version = getattr(module, "__version__", "")
    if not version:
        try:
            from importlib.metadata import version as metadata_version

            version = metadata_version(name)
        except Exception:  # noqa: BLE001 - optional metadata
            version = ""
    return {
        "dependency": name,
        "import_name": import_name,
        "installed": True,
        "version": str(version),
        "status": "ok",
        "detail": "",
    }


def run_command(command: list[str], timeout: int = 30) -> dict[str, Any]:
    if shutil.which(command[0]) is None:
        return {"command": " ".join(command), "ok": False, "stdout": "", "stderr": f"{command[0]} not found"}
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - audit only
        return {"command": " ".join(command), "ok": False, "stdout": "", "stderr": str(exc)}
    return {
        "command": " ".join(command),
        "ok": completed.returncode == 0,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "returncode": completed.returncode,
    }


def mlx_capabilities() -> dict[str, Any]:
    row = package_version("mlx", "mlx.core")
    if not row["installed"]:
        return row
    import mlx.core as mx

    metal = getattr(mx, "metal", None)
    row.update(
        {
            "mx_compile": callable(getattr(mx, "compile", None)),
            "mx_fast_scaled_dot_product_attention": callable(getattr(getattr(mx, "fast", None), "scaled_dot_product_attention", None)),
            "mx_metal_available": metal is not None,
            "mx_active_memory_api": callable(getattr(metal, "get_active_memory", None)) if metal is not None else False,
            "mx_peak_memory_api": callable(getattr(metal, "get_peak_memory", None)) if metal is not None else False,
            "mx_cache_memory_api": callable(getattr(metal, "get_cache_memory", None)) if metal is not None else False,
        }
    )
    device_info = {}
    for attr in ("device_info", "get_device"):
        func = getattr(metal, attr, None) if metal is not None else None
        if callable(func):
            try:
                device_info[attr] = str(func())
            except Exception as exc:  # noqa: BLE001
                device_info[attr] = f"error: {exc}"
    row["mlx_device_info"] = json.dumps(device_info, sort_keys=True)
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, Any]], ffmpeg_rows: list[dict[str, Any]], output_root: Path) -> None:
    lines = [
        "# V-JEPA 2.1 Runtime Dependency Audit",
        "",
        f"Created: `{datetime.now().isoformat(timespec='seconds')}`",
        f"Platform: `{platform.platform()}`",
        f"Python: `{sys.version.split()[0]}`",
        "",
        "## Dependency Decisions",
        "",
        "| Dependency | Installed | Version | Status | Recommendation |",
        "|---|---:|---|---|---|",
    ]
    for row in rows:
        recommendation = "promote/keep" if row.get("installed") and row.get("dependency") in {"mlx", "numpy", "safetensors", "huggingface_hub", "psutil", "tqdm"} else "optional/benchmark"
        if not row.get("installed"):
            recommendation = "do not promote until isolated benchmark wins"
        lines.append(
            f"| {row.get('dependency')} | {row.get('installed')} | {row.get('version', '')} | {row.get('status', '')} | {recommendation} |"
        )
    lines.extend(
        [
            "",
            "## FFmpeg / VideoToolbox",
            "",
            "The audit below only proves local capability advertisement. It does not prove a given AGAIN file uses hardware decode; use the benchmark tool on a sample video for that.",
            "",
            "| Command | OK | Notes |",
            "|---|---:|---|",
        ]
    )
    for row in ffmpeg_rows:
        text = (row.get("stdout") or row.get("stderr") or "").replace("\n", "<br>")[:500]
        lines.append(f"| `{row['command']}` | {row.get('ok')} | {text} |")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- CSV: `{output_root / 'dependency_audit.csv'}`",
            f"- FFmpeg JSON: `{output_root / 'ffmpeg_audit.json'}`",
            "",
            "## Production Rule",
            "",
            "Do not promote optional video loaders from this audit alone. Promote only after a no-cache ready-window benchmark shows correct frames and lower end-to-end latency.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = args.output_root or ROOT / "outputs" / f"vjepa21_runtime_dependency_audit_{stamp}"
    output_root.mkdir(parents=True, exist_ok=True)
    deps = [
        mlx_capabilities(),
        package_version("numpy"),
        package_version("safetensors"),
        package_version("huggingface_hub"),
        package_version("av"),
        package_version("eva-decord", "decord"),
        package_version("decord2"),
        package_version("opencv-python", "cv2"),
        package_version("xxhash"),
        package_version("orjson"),
        package_version("zstandard", "zstandard"),
        package_version("lz4"),
        package_version("pyarrow"),
        package_version("psutil"),
        package_version("rich"),
        package_version("tqdm"),
        package_version("pytest"),
        package_version("pytest-benchmark", "pytest_benchmark"),
        package_version("py-spy", "py_spy"),
        package_version("scalene"),
    ]
    ffmpeg_rows = [
        run_command(["ffmpeg", "-hide_banner", "-hwaccels"]),
        run_command(["ffmpeg", "-hide_banner", "-decoders"]),
        run_command(["ffmpeg", "-hide_banner", "-encoders"]),
    ]
    write_csv(output_root / "dependency_audit.csv", deps)
    (output_root / "ffmpeg_audit.json").write_text(json.dumps(ffmpeg_rows, indent=2) + "\n", encoding="utf-8")
    write_report(output_root / "dependency_audit_report.md", deps, ffmpeg_rows, output_root)
    print(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
