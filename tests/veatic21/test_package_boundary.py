from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "neural_bridge" / "veatic21"
FORBIDDEN_PACKAGE = "neural_bridge.again"


def _import_targets(
    node: ast.Import | ast.ImportFrom, package: tuple[str, ...]
) -> set[str]:
    if isinstance(node, ast.Import):
        return {alias.name for alias in node.names}

    if node.level:
        retained = len(package) - node.level + 1
        base = package[: max(retained, 0)]
    else:
        base = ()
    module = tuple(node.module.split(".")) if node.module else ()
    prefix = (*base, *module)
    targets = {".".join(prefix)} if prefix else set()
    targets.update(".".join((*prefix, alias.name)) for alias in node.names)
    return targets


def test_veatic21_package_does_not_import_again() -> None:
    violations: list[str] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        relative = path.relative_to(PACKAGE)
        package = ("neural_bridge", "veatic21", *relative.parent.parts)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            forbidden = sorted(
                target
                for target in _import_targets(node, package)
                if target == FORBIDDEN_PACKAGE or target.startswith(f"{FORBIDDEN_PACKAGE}.")
            )
            if forbidden:
                statement = ast.get_source_segment(source, node) or forbidden[0]
                violations.append(f"{relative}:{node.lineno}: {statement}")

    assert not violations, "VEATIC 2.1 imported AGAIN:\n" + "\n".join(violations)
