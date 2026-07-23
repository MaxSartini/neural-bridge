from __future__ import annotations

import json
from pathlib import Path

import pytest

from neural_bridge.artifact_graph import (
    _watch_filter,
    build_artifact_graph,
    exact_node_payload,
)


def test_build_artifact_graph_indexes_history_but_excludes_its_own_indexes(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    active = root / "runs" / "veatic-2.1" / "result.json"
    inactive = root / "archive" / "old.json"
    nested_inactive = root / "runs" / "scratch" / "temporary.json"
    recursive_index = root / "indexes" / "graphify" / "graph.json"
    quarantined = root / "quarantine" / "unsafe.json"
    active.parent.mkdir(parents=True)
    inactive.parent.mkdir(parents=True)
    nested_inactive.parent.mkdir(parents=True)
    recursive_index.parent.mkdir(parents=True)
    quarantined.parent.mkdir(parents=True)
    active.write_text("{}")
    inactive.write_text("{}")
    nested_inactive.write_text("{}")
    recursive_index.write_text("{}")
    quarantined.write_text("{}")
    linked = root / "runs" / "latest"
    linked.symlink_to(active.parent, target_is_directory=True)
    output = tmp_path / "index" / "graph.json"

    summary = build_artifact_graph(root, output)
    graph = json.loads(output.read_text())

    assert summary["artifact_count"] == 4
    assert graph["edges"] == []
    nodes = {node["label"]: node for node in graph["nodes"]}
    assert nodes["runs/latest"]["link_target"] == str(active.parent)
    assert nodes["runs/veatic-2.1/result.json"]["source_file"] == str(active)
    assert nodes["archive/old.json"]["source_file"] == str(inactive)
    assert nodes["runs/scratch/temporary.json"]["source_file"] == str(nested_inactive)
    assert "indexes/graphify/graph.json" not in nodes
    assert "quarantine/unsafe.json" not in nodes


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


def test_qualified_symbol_resolves_without_internal_graphify_spelling(
    monkeypatch, tmp_path: Path
) -> None:
    repository = tmp_path / "repository"
    source = repository / "src" / "package" / "model.py"
    source.parent.mkdir(parents=True)
    source.write_text("def train():\n    return 1\n")
    graph = tmp_path / "graph.json"
    graph.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": "repository::train",
                        "label": "train()",
                        "file_type": "code",
                        "source_file": "src/package/model.py",
                        "source_location": "L1",
                    },
                    {
                        "id": "repository::model_train",
                        "label": "model_train()",
                        "file_type": "code",
                        "source_file": "src/package/model.py",
                        "source_location": "L1",
                    },
                ]
            }
        )
    )
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    monkeypatch.setattr("neural_bridge.artifact_graph.DEFAULT_REPOSITORY", repository)
    monkeypatch.setattr("neural_bridge.artifact_graph.DEFAULT_ARTIFACTS", artifacts)

    payload = exact_node_payload("package.model.train", graph)

    assert payload["content"] == "def train():\n    return 1"


def test_unique_filename_resolves_to_file_node(monkeypatch, tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    source = repository / "internal" / "handoff" / "CURRENT_STATE.md"
    source.parent.mkdir(parents=True)
    source.write_text("# Current\n")
    graph = tmp_path / "graph.json"
    graph.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": "repository_file::state",
                        "label": "internal/handoff/CURRENT_STATE.md",
                        "file_type": "artifact",
                        "source_file": str(source),
                        "_origin": "repository_file_catalog",
                    }
                ]
            }
        )
    )
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    monkeypatch.setattr("neural_bridge.artifact_graph.DEFAULT_REPOSITORY", repository)
    monkeypatch.setattr("neural_bridge.artifact_graph.DEFAULT_ARTIFACTS", artifacts)

    payload = exact_node_payload("CURRENT_STATE.md", graph)

    assert payload["id"] == "repository_file::state"
    assert payload["content"] == "# Current\n"


def test_provenance_context_disambiguates_repeated_names(monkeypatch, tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    current = repository / "programme-a" / "active" / "runner.py"
    historical = repository / "programme-b" / "archive" / "runner.py"
    current.parent.mkdir(parents=True)
    historical.parent.mkdir(parents=True)
    current.write_text("CURRENT = True\n")
    historical.write_text("HISTORICAL = True\n")
    graph = tmp_path / "graph.json"
    graph.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": "current",
                        "label": "programme-a/active/runner.py",
                        "file_type": "artifact",
                        "source_file": str(current),
                        "_origin": "repository_file_catalog",
                    },
                    {
                        "id": "historical",
                        "label": "programme-b/archive/runner.py",
                        "file_type": "artifact",
                        "source_file": str(historical),
                        "_origin": "repository_file_catalog",
                    },
                ]
            }
        )
    )
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    monkeypatch.setattr("neural_bridge.artifact_graph.DEFAULT_REPOSITORY", repository)
    monkeypatch.setattr("neural_bridge.artifact_graph.DEFAULT_ARTIFACTS", artifacts)

    with pytest.raises(LookupError, match="found 2"):
        exact_node_payload("runner.py", graph)

    payload = exact_node_payload("runner.py", graph, context="programme-b archive")

    assert payload["id"] == "historical"
    assert payload["content"] == "HISTORICAL = True\n"


def test_exact_path_prefers_file_node_and_delivers_text(monkeypatch, tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    source = repository / "internal" / "handoff" / "CURRENT_STATE.md"
    source.parent.mkdir(parents=True)
    source.write_text("# Current\n\nRun the executor.\n")
    graph = tmp_path / "graph.json"
    graph.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": "symbol",
                        "label": "current_state()",
                        "file_type": "code",
                        "source_file": str(source),
                        "source_location": "L1",
                    },
                    {
                        "id": "repository_file::state",
                        "label": "internal/handoff/CURRENT_STATE.md",
                        "file_type": "artifact",
                        "source_file": str(source),
                        "_origin": "repository_file_catalog",
                    },
                ]
            }
        )
    )
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    monkeypatch.setattr("neural_bridge.artifact_graph.DEFAULT_REPOSITORY", repository)
    monkeypatch.setattr("neural_bridge.artifact_graph.DEFAULT_ARTIFACTS", artifacts)

    payload = exact_node_payload(str(source), graph)

    assert payload["id"] == "repository_file::state"
    assert payload["delivery"] == "inline_content"
    assert payload["content"] == "# Current\n\nRun the executor.\n"


def test_exact_heavy_artifact_delivers_direct_consumer_path(
    monkeypatch, tmp_path: Path
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    artifacts = tmp_path / "artifacts"
    payload_path = artifacts / "features" / "cache.npy"
    payload_path.parent.mkdir(parents=True)
    payload_path.write_bytes(b"binary")
    graph = tmp_path / "graph.json"
    graph.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": "artifact::cache",
                        "label": "features/cache.npy",
                        "file_type": "artifact",
                        "source_file": str(payload_path),
                        "_origin": "artifact_catalog",
                    }
                ]
            }
        )
    )
    monkeypatch.setattr("neural_bridge.artifact_graph.DEFAULT_REPOSITORY", repository)
    monkeypatch.setattr("neural_bridge.artifact_graph.DEFAULT_ARTIFACTS", artifacts)

    payload = exact_node_payload(str(payload_path), graph)

    assert payload["delivery"] == "direct_consumer_path"
    assert payload["path"] == str(payload_path)


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
