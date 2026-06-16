"""Download the initial human-response benchmark batch to external storage.

Every component is independently resumable. Re-running this script updates Git
repositories and asks OpenNeuro to verify and resume existing files.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


DEFAULT_ROOT = Path(
    os.environ.get("NEURAL_BRIDGE_DATASET_ROOT")
    or os.environ.get("MIROFISH_DATASET_ROOT", "/Volumes/onn. Drive/Neural Bridge/datasets")
)

GIT_DATASETS = {
    "pvp": "https://github.com/holi-lab/PVP_Personalized_Visual_Persuasion.git",
    "openlav_tools": "https://gitlab.lrz.de/nicebread/openlav.git",
}

OPENNEURO_DATASETS = {
    "openfmri_affective_videos": "ds000205",
    "emofilm_annotations": "ds004872",
}
OPENNEURO_CONCURRENCY = {
    "openfmri_affective_videos": 3,
    # This annotation-only dataset contains many tiny files; higher concurrency
    # avoids spending most of the runtime on per-request latency.
    "emofilm_annotations": 12,
}

OPENLAV_PAGE = (
    "https://www.psycharchives.org/en/item/18779e98-c04b-4299-8311-dc442dc89bcd"
)
ALL_DATASETS = [*GIT_DATASETS, *OPENNEURO_DATASETS, "openlav_videos"]


def run(command: Sequence[str], log_path: Path, cwd: Path | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{datetime.now(timezone.utc).isoformat()}] {' '.join(command)}\n")
        log.flush()
        subprocess.run(
            list(command),
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def git_download(name: str, url: str, root: Path, log_dir: Path) -> None:
    target = root / name
    log = log_dir / f"{name}.log"
    if (target / ".git").is_dir():
        run(["git", "fetch", "--all", "--prune"], log, target)
        run(["git", "pull", "--ff-only"], log, target)
    else:
        run(["git", "clone", "--recurse-submodules", url, str(target)], log)
    run(["git", "lfs", "pull"], log, target)


def openneuro_download(name: str, dataset_id: str, root: Path, log_dir: Path) -> None:
    executable = Path(sys.executable).parent / "openneuro-py"
    if not executable.exists():
        raise RuntimeError(
            f"{executable} is missing. Install openneuro-py into the active environment."
        )
    run(
        [
            str(executable),
            "download",
            "--dataset",
            dataset_id,
            "--target-dir",
            str(root / name),
            "--max-concurrent-downloads",
            str(OPENNEURO_CONCURRENCY.get(name, 3)),
            "--max-retries",
            "12",
            "--verify-hash",
            "--verify-size",
        ],
        log_dir / f"{name}.log",
    )


def _openlav_manifest() -> list[dict[str, Any]]:
    request = urllib.request.Request(
        OPENLAV_PAGE,
        headers={"User-Agent": "Mozilla/5.0 (Neural Bridge benchmark downloader)"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        html = response.read().decode("utf-8")
    match = re.search(
        r'<script[^>]+type="application/ld\+json"[^>]*>(.*?)</script>',
        html,
        re.DOTALL,
    )
    if not match:
        raise RuntimeError("OpenLAV PsychArchives page has no JSON-LD manifest")
    document = json.loads(match.group(1))
    parts = document.get("hasPart", [])
    manifest = [
        {"name": part["name"], "url": part["url"], "encoding_format": part.get("encodingFormat")}
        for part in parts
        if isinstance(part, dict) and part.get("name") and part.get("url")
    ]
    if len([item for item in manifest if item["name"].endswith(".webm")]) < 180:
        raise RuntimeError("OpenLAV manifest is unexpectedly missing video files")
    return manifest


def _download_resumable_file(item: dict[str, Any], target_dir: Path, log_dir: Path) -> None:
    target = target_dir / item["name"]
    partial = target.with_name(f"{target.name}.part")
    if target.exists() and target.stat().st_size > 0:
        return
    try:
        run(
            [
                "curl",
                "--location",
                "--fail",
                "--retry",
                "12",
                "--retry-all-errors",
                "--continue-at",
                "-",
                "--user-agent",
                "Mozilla/5.0 (Neural Bridge benchmark downloader)",
                "--output",
                str(partial),
                item["url"],
            ],
            log_dir / "openlav_videos.log",
        )
    except subprocess.CalledProcessError:
        request = urllib.request.Request(
            item["url"],
            method="HEAD",
            headers={"User-Agent": "Mozilla/5.0 (Neural Bridge benchmark downloader)"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            expected_size = int(response.headers.get("Content-Length", "0"))
        if not partial.exists() or expected_size <= 0 or partial.stat().st_size != expected_size:
            raise
    partial.replace(target)


def openlav_download(root: Path, log_dir: Path) -> None:
    target = root / "openlav_videos"
    target.mkdir(parents=True, exist_ok=True)
    manifest = _openlav_manifest()
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(_download_resumable_file, item, target, log_dir)
            for item in manifest
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def write_status(root: Path) -> None:
    records = {}
    for name in ALL_DATASETS:
        target = root / name
        size = 0
        files = 0
        if target.exists():
            for path in target.rglob("*"):
                if path.is_file():
                    files += 1
                    size += path.stat().st_size
        records[name] = {
            "path": str(target),
            "exists": target.exists(),
            "files": files,
            "size_bytes": size,
        }
    status_path = root / "initial_batch_status.json"
    temporary_path = root / f".initial_batch_status.{os.getpid()}.tmp"
    temporary_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "datasets": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary_path.replace(status_path)


def main() -> None:
    choices = ALL_DATASETS
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", choices=["all", "status", *choices])
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    log_dir = root / "_download_logs"
    if args.datasets == ["status"]:
        write_status(root)
        return
    if "status" in args.datasets:
        parser.error("status must be used by itself")
    selected = choices if "all" in args.datasets else list(dict.fromkeys(args.datasets))

    failures = {}
    for name in selected:
        try:
            if name in GIT_DATASETS:
                git_download(name, GIT_DATASETS[name], root, log_dir)
            elif name == "openlav_videos":
                openlav_download(root, log_dir)
            else:
                openneuro_download(name, OPENNEURO_DATASETS[name], root, log_dir)
        except Exception as exc:
            failures[name] = str(exc)
        finally:
            write_status(root)

    if failures:
        raise SystemExit(f"Some downloads failed and can be resumed: {failures}")


if __name__ == "__main__":
    main()
