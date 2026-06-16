"""Create a cumulative project handoff and long-term memory entry.

This script is intentionally local and file-based. It creates a timestamped
handoff Markdown file for a new chat, then appends a compact entry to the
cumulative project memory file.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HANDOFF_DIR = ROOT / "docs" / "handoffs"
DEFAULT_MEMORY_PATH = ROOT / "docs" / "PROJECT_MEMORY.md"
DEFAULT_STATE_PATH = ROOT / "docs" / "current_project_state.md"
VEATIC_CACHE = Path("/Volumes/onn. Drive/Neural Bridge/benchmarks/veatic/tribe_cache")


def read_text(path: Path, max_chars: int = 12000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[:max_chars]


def run_command(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return (completed.stdout + completed.stderr).strip()
    except Exception as exc:  # pragma: no cover - defensive local utility
        return f"command failed: {exc}"


def veatic_cache_status() -> dict[str, Any]:
    complete: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []
    if not VEATIC_CACHE.exists():
        return {"available": False}
    for status_path in VEATIC_CACHE.glob("*/cache_status.json"):
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception as exc:
            status = {
                "video_id": status_path.parent.name,
                "complete": False,
                "error": str(exc),
            }
        if status.get("complete"):
            complete.append(status)
        elif status.get("error"):
            failed.append(status)
        else:
            incomplete.append(status)
    latest = sorted(
        [*complete, *failed, *incomplete],
        key=lambda item: (
            VEATIC_CACHE / str(item.get("video_id", "")) / "cache_status.json"
        ).stat().st_mtime
        if (VEATIC_CACHE / str(item.get("video_id", "")) / "cache_status.json").exists()
        else 0,
        reverse=True,
    )[:8]
    return {
        "available": True,
        "complete": len(complete),
        "failed": len(failed),
        "incomplete": [
            {
                "video_id": item.get("video_id"),
                "duration_seconds": item.get("duration_seconds"),
                "error": item.get("error"),
            }
            for item in incomplete[:8]
        ],
        "latest": [
            {
                "video_id": item.get("video_id"),
                "complete": item.get("complete"),
                "error": item.get("error"),
                "timings_seconds": item.get("timings_seconds"),
                "duration_seconds": item.get("duration_seconds"),
            }
            for item in latest
        ],
    }


def latest_benchmark_files() -> list[str]:
    roots = [ROOT / "benchmarks" / "veatic", ROOT / "benchmarks" / "openlav"]
    files: list[Path] = []
    for root in roots:
        if root.exists():
            files.extend(root.glob("*.json"))
            files.extend(root.glob("*.summary.md"))
    return [
        str(path.relative_to(ROOT))
        for path in sorted(files, key=lambda item: item.stat().st_mtime, reverse=True)[:16]
    ]


def slugify(value: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "-" for char in value).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug[:80] or "handoff"


def build_handoff(args: argparse.Namespace) -> tuple[Path, str]:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    title = args.title.strip() or "Neural Bridge Project Handoff"
    cache_status = veatic_cache_status()
    process_status = run_command(
        [
            "bash",
            "-lc",
            "ps -axo pid,etime,pcpu,pmem,command | rg 'run_veatic_tribe_cache' | rg -v 'rg|node_repl' || true",
        ]
    )
    state_excerpt = read_text(Path(args.state_file), max_chars=args.state_chars)
    memory_excerpt = read_text(Path(args.memory), max_chars=args.memory_chars)
    benchmark_files = latest_benchmark_files()

    lines = [
        f"# {title}",
        "",
        f"Created: {now}",
        f"Workspace: `{ROOT}`",
        "",
        "## New Chat Instruction",
        "",
        "Use this file as the starting context for the next Codex chat. Continue from the current task state, preserve the scientific guardrails, and update `docs/PROJECT_MEMORY.md` plus a new handoff file before the chat gets too long.",
        "",
        "## Current User Notes",
        "",
        args.notes.strip() or "No extra notes supplied.",
        "",
        "## Current Live Process Status",
        "",
        "```text",
        process_status or "No matching long-running benchmark/extraction process found.",
        "```",
        "",
        "## VEATIC Cache Status",
        "",
        "```json",
        json.dumps(cache_status, indent=2),
        "```",
        "",
        "## Latest Benchmark / Report Files",
        "",
        *[f"- `{path}`" for path in benchmark_files],
        "",
        "## Current Project State Snapshot",
        "",
        state_excerpt or "No project state document found.",
        "",
        "## Cumulative Memory Excerpt",
        "",
        memory_excerpt or "No cumulative memory file exists yet.",
        "",
        "## Required Next-Step Discipline",
        "",
        "- Do not claim neuro-additive value unless real cortical beats autoregressive-only and shuffled/random controls.",
        "- Keep subcortical disabled by default; use explicit ablation/research mode only.",
        "- Do not score incomplete or failed cache entries.",
        "- Run benchmark gates before scaling beyond the current target.",
        "- Keep handoffs cumulative and file-backed.",
        "",
    ]
    handoff_dir = Path(args.output_dir)
    handoff_dir.mkdir(parents=True, exist_ok=True)
    output = handoff_dir / f"{stamp}_{slugify(title)}.md"
    text = "\n".join(lines)
    output.write_text(text, encoding="utf-8")
    return output, text


def append_memory(args: argparse.Namespace, handoff_path: Path) -> None:
    memory_path = Path(args.memory)
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cache_status = veatic_cache_status()
    entry = [
        "",
        f"## {now} - {args.title.strip() or 'Project Handoff'}",
        "",
        f"- Handoff file: `{handoff_path.relative_to(ROOT)}`",
        f"- Notes: {args.notes.strip() or 'No extra notes supplied.'}",
        f"- VEATIC cache: complete={cache_status.get('complete')}, failed={cache_status.get('failed')}, incomplete={cache_status.get('incomplete')}",
        "- Standing guardrail: real cortical must beat autoregressive, shuffled cortical, and random Gaussian controls before scale-up claims.",
        "",
    ]
    if not memory_path.exists():
        memory_path.write_text("# Neural Bridge Project Memory\n", encoding="utf-8")
    with memory_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(entry))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default="Neural Bridge Project Handoff")
    parser.add_argument("--notes", default="")
    parser.add_argument("--output-dir", default=str(DEFAULT_HANDOFF_DIR))
    parser.add_argument("--memory", default=str(DEFAULT_MEMORY_PATH))
    parser.add_argument("--state-file", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--state-chars", type=int, default=12000)
    parser.add_argument("--memory-chars", type=int, default=8000)
    args = parser.parse_args()

    output, _ = build_handoff(args)
    append_memory(args, output)
    print(json.dumps({"handoff": str(output), "memory": args.memory}, indent=2))


if __name__ == "__main__":
    main()
