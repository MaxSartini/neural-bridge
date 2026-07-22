from __future__ import annotations

import json
from pathlib import Path

import pytest

from neural_bridge.artifact_graph import (
    _watch_filter,
    build_artifact_graph,
    exact_node_payload,
)


def test_build_artifact_graph_is_exact_and_excludes_inactive_roots(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    active = root / "runs" / "veatic-2.1" / "result.json"
    inactive = root / "archive" / "old.json"
    active.parent.mkdir(parents=True)
    inactive.parent.mkdir(parents=True)
    active.write_text("{}")
    inactive.write_text("{}")
    linked = root / "runs" / "latest"
    linked.symlink_to(active.parent, target_is_directory=True)
    output = tmp_path / "index" / "graph.json"

    summary = build_artifact_graph(root, output)
    graph = json.loads(output.read_text())

    assert summary["artifact_count"] == 2
    assert graph["edges"] == []
    nodes = {node["label"]: node for node in graph["nodes"]}
    assert nodes["runs/latest"]["link_target"] == str(active.parent)
    assert nodes["runs/veatic-2.1/result.json"]["source_file"] == str(active)


def test_exact_node_payload_delivers_the_indexed_function(monkeypatch, tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    source = repository / "model.py"
    source.parent.mkdir()
    source.write_text("x = 1\n\ndef train():\n    return x\n\ny = 2\n")
    graph = tmp_path / "graph.json"
    graph.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": "repository::train",
                        "label": "train()",
                        "file_type": "code",
                        "source_file": "model.py",
                        "source_location": "L3",
                    }
                ]
            }
        )
    )
    monkeypatch.setattr("neural_bridge.artifact_graph.DEFAULT_REPOSITORY", repository)
    monkeypatch.setattr("neural_bridge.artifact_graph.DEFAULT_ARTIFACTS", tmp_path / "artifacts")
    (tmp_path / "artifacts").mkdir()

    payload = exact_node_payload("train", graph)

    assert payload["path"] == str(source)
    assert payload["start_line"] == 3
    assert payload["end_line"] == 4
    assert payload["content"] == "def train():\n    return x"


def test_exact_node_payload_rejects_ambiguous_labels(monkeypatch, tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    graph = tmp_path / "graph.json"
    graph.write_text(
        json.dumps(
            {
                "nodes": [
                    {"id": "one", "label": "main()", "source_file": "one.py"},
                    {"id": "two", "label": "main()", "source_file": "two.py"},
                ]
            }
        )
    )
    monkeypatch.setattr("neural_bridge.artifact_graph.DEFAULT_REPOSITORY", repository)

    with pytest.raises(LookupError, match="found 2"):
        exact_node_payload("main", graph)


def test_watch_filter_covers_both_roots_without_index_feedback() -> None:
    assert _watch_filter(None, "/Users/maxsartini/Neural Bridge/src/model.py")
    assert _watch_filter(None, "/Volumes/onn. Drive/Neural Bridge Artifacts/runs/result.json")
    assert not _watch_filter(None, "/Users/maxsartini/Neural Bridge/.git/index")
    assert not _watch_filter(
        None,
        "/Volumes/onn. Drive/Neural Bridge Artifacts/indexes/graphify/merged/graph.json",
    )
