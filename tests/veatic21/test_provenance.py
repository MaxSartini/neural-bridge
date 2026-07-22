from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from neural_bridge.provenance import tree_digest, verify_tree_digest


def test_tree_digest_matches_canonical_migration_algorithm(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    nested = root / "nested"
    ignored = root / "__pycache__"
    nested.mkdir(parents=True)
    ignored.mkdir()
    alpha = b"alpha"
    binary = b"\x00\xff"
    (root / "a.txt").write_bytes(alpha)
    (nested / "b.bin").write_bytes(binary)
    (root / "alias").symlink_to("a.txt")
    (root / ".DS_Store").write_bytes(b"ignored")
    (ignored / "payload.pyc").write_bytes(b"ignored")

    expected = hashlib.sha256()
    expected.update(
        f"F\0a.txt\0{len(alpha)}\0{hashlib.sha256(alpha).hexdigest()}\n".encode()
    )
    expected.update(b"L\0alias\0a.txt\n")
    expected.update(
        f"F\0nested/b.bin\0{len(binary)}\0{hashlib.sha256(binary).hexdigest()}\n".encode()
    )

    assert tree_digest(root) == {
        "path": str(root.resolve()),
        "sha256_tree": expected.hexdigest(),
        "files": 2,
        "symlinks": 1,
        "size_bytes": 7,
    }


def test_registered_tree_verification_rejects_byte_tampering(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    root.mkdir()
    payload = root / "payload.bin"
    payload.write_bytes(b"alpha")
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(tree_digest(root)), encoding="utf-8")
    registered = json.loads(registry_path.read_text(encoding="utf-8"))

    assert verify_tree_digest(root, registered, source=registry_path)["files"] == 1

    payload.write_bytes(b"omega")
    with pytest.raises(ValueError, match="sha256_tree"):
        verify_tree_digest(root, registered, source=registry_path)


@pytest.mark.parametrize(("field", "wrong_value"), (("files", 2), ("size_bytes", 6)))
def test_registered_tree_verification_rejects_wrong_inventory(
    tmp_path: Path, field: str, wrong_value: int
) -> None:
    root = tmp_path / "cache"
    root.mkdir()
    (root / "payload.bin").write_bytes(b"alpha")
    registered = dict(tree_digest(root))
    registered[field] = wrong_value

    with pytest.raises(ValueError, match=field):
        verify_tree_digest(root, registered, source="fixture registry")
