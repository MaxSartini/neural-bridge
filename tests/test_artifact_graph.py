from __future__ import annotations

import json
from pathlib import Path

from neural_bridge.artifact_graph import build_artifact_graph, serve_mcp


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


def test_serve_mcp_refreshes_before_exec(monkeypatch) -> None:
    monkeypatch.setattr("neural_bridge.artifact_graph.shutil.which", lambda _: "/bin/mcp")
    monkeypatch.setattr(
        "neural_bridge.artifact_graph.refresh_index",
        lambda *_: {"merged_graph": "/graph.json"},
    )
    executed = []
    monkeypatch.setattr(
        "neural_bridge.artifact_graph.os.execv",
        lambda executable, arguments: executed.append((executable, arguments)),
    )

    serve_mcp()

    assert executed == [("/bin/mcp", ["/bin/mcp", "--graph", "/graph.json"])]
