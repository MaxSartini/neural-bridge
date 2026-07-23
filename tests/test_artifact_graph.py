from __future__ import annotations

import json
from pathlib import Path

from neural_bridge.artifact_graph import REPOSITORY_EXCLUDES, build_artifact_graph


def test_artifact_catalog_indexes_history_without_indexes_or_quarantine(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    active = root / "runs" / "current" / "result.json"
    historical = root / "archive" / "old.json"
    scratch = root / "scratch" / "experiment.json"
    recursive_index = root / "indexes" / "graphify" / "graph.json"
    quarantined = root / "quarantine" / "unsafe.json"
    for path in (active, historical, scratch, recursive_index, quarantined):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    linked = root / "runs" / "latest"
    linked.symlink_to(active.parent, target_is_directory=True)
    output = tmp_path / "index" / "graph.json"

    summary = build_artifact_graph(root, output)
    graph = json.loads(output.read_text())
    nodes = {node["label"]: node for node in graph["nodes"]}

    assert summary["artifact_count"] == 4
    assert nodes["runs/latest"]["link_target"] == str(active.parent)
    assert nodes["runs/current/result.json"]["source_file"] == str(active)
    assert nodes["archive/old.json"]["source_file"] == str(historical)
    assert nodes["scratch/experiment.json"]["source_file"] == str(scratch)
    assert "indexes/graphify/graph.json" not in nodes
    assert "quarantine/unsafe.json" not in nodes


def test_repository_catalog_excludes_only_generated_workspace_state(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    source = root / "src" / "package" / "model.py"
    historical = root / "archive" / "legacy.py"
    cached = root / ".pytest_cache" / "state"
    for path in (source, historical, cached):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("value = 1\n")
    output = tmp_path / "index" / "graph.json"

    build_artifact_graph(
        root,
        output,
        REPOSITORY_EXCLUDES,
        namespace="repository_file",
    )
    graph = json.loads(output.read_text())
    labels = {node["label"] for node in graph["nodes"]}

    assert "src/package/model.py" in labels
    assert "archive/legacy.py" in labels
    assert ".pytest_cache/state" not in labels
