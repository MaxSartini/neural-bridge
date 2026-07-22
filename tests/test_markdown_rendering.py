from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIPPED_PARTS = {".git", ".venv", "artifacts", "node_modules"}
MARKDOWN_IMAGE_RE = re.compile(r"!\[[^]]*]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
INLINE_MATH_RE = re.compile(r"(?<!\\)(?<!\$)\$([^\n$]+?)\$(?!\$)")


def markdown_files() -> list[Path]:
    return [
        path
        for path in ROOT.rglob("*.md")
        if not any(part in SKIPPED_PARTS for part in path.relative_to(ROOT).parts)
    ]


def prose_without_fenced_code(text: str) -> str:
    prose: list[str] = []
    fence_marker: str | None = None
    for line in text.splitlines():
        match = re.match(r"^\s*([\x60~]{3,})", line)
        if match:
            marker = match.group(1)[0]
            fence_marker = None if fence_marker == marker else marker
            continue
        if fence_marker is None:
            prose.append(line)
    return "\n".join(prose)


def local_asset(markdown: Path, value: str) -> Path:
    return (markdown.parent / value).resolve()


def assert_mobile_safe_svg(path: Path) -> None:
    assert path.is_file(), f"missing rendered Markdown asset: {path}"
    root = ET.parse(path).getroot()
    assert root.tag.endswith("svg"), f"not an SVG document: {path}"
    assert root.get("viewBox"), f"SVG has no responsive viewBox: {path}"
    assert not any(
        element.tag.endswith("foreignObject") for element in root.iter()
    ), f"SVG embeds HTML that GitHub mobile may not render: {path}"
    if "equations" in path.parts:
        backgrounds = [
            element
            for element in root
            if element.tag.endswith("rect") and element.get("fill") in {"#fff", "#ffffff"}
        ]
        assert backgrounds, f"equation SVG has no theme-independent background: {path}"


def test_markdown_uses_mobile_safe_math_and_diagrams() -> None:
    failures: list[str] = []
    for path in markdown_files():
        text = path.read_text(encoding="utf-8")
        if re.search(r"^\s*```(?:math|tex|latex|mermaid)\s*$", text, re.MULTILINE):
            failures.append(f"{path.relative_to(ROOT)} contains a renderer-dependent fence")
        if re.search(r"<(?:picture|source)\b", text):
            failures.append(f"{path.relative_to(ROOT)} contains fragile responsive-image HTML")
        if INLINE_MATH_RE.search(prose_without_fenced_code(text)):
            failures.append(f"{path.relative_to(ROOT)} contains renderer-dependent inline math")
    assert not failures, "\n".join(failures)


def test_readme_render_assets_resolve_and_are_mobile_safe() -> None:
    expected_counts = {ROOT / "README.md": 5, ROOT / "docs/README.md": 3}
    for markdown, expected_count in expected_counts.items():
        images = MARKDOWN_IMAGE_RE.findall(markdown.read_text(encoding="utf-8"))
        assert len(images) == expected_count

    for markdown in markdown_files():
        images = MARKDOWN_IMAGE_RE.findall(markdown.read_text(encoding="utf-8"))
        for image in images:
            if image.startswith(("https://", "http://", "data:")):
                continue
            asset = local_asset(markdown, image)
            assert asset.is_file(), f"missing rendered Markdown asset: {asset}"
            if asset.suffix.lower() == ".svg":
                assert_mobile_safe_svg(asset)
