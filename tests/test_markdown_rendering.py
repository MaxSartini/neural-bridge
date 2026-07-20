from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIPPED_PARTS = {".git", ".codegraph", ".venv", "artifacts", "node_modules"}
PICTURE_RE = re.compile(r"<picture>.*?</picture>", re.DOTALL)
SOURCE_RE = re.compile(r'<source\s+[^>]*srcset="([^"]+)"')
IMAGE_RE = re.compile(r'<img\s+[^>]*src="([^"]+)"')
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


def assert_svg(path: Path) -> None:
    assert path.is_file(), f"missing rendered Markdown asset: {path}"
    root = ET.parse(path).getroot()
    assert root.tag.endswith("svg"), f"not an SVG document: {path}"
    assert root.get("viewBox"), f"SVG has no responsive viewBox: {path}"


def test_markdown_uses_mobile_safe_math_and_diagrams() -> None:
    failures: list[str] = []
    for path in markdown_files():
        text = path.read_text(encoding="utf-8")
        if re.search(r"^\s*```(?:math|mermaid)\s*$", text, re.MULTILINE):
            failures.append(f"{path.relative_to(ROOT)} contains a renderer-dependent fence")
        if INLINE_MATH_RE.search(prose_without_fenced_code(text)):
            failures.append(f"{path.relative_to(ROOT)} contains renderer-dependent inline math")
    assert not failures, "\n".join(failures)


def test_readme_picture_fallbacks_resolve_and_parse() -> None:
    expected_counts = {ROOT / "README.md": 5, ROOT / "docs/README.md": 3}
    for markdown, expected_count in expected_counts.items():
        pictures = PICTURE_RE.findall(markdown.read_text(encoding="utf-8"))
        assert len(pictures) == expected_count
        for picture in pictures:
            dark = SOURCE_RE.search(picture)
            light = IMAGE_RE.search(picture)
            assert dark and light, f"incomplete light/dark picture fallback in {markdown}"
            assert_svg(local_asset(markdown, dark.group(1)))
            assert_svg(local_asset(markdown, light.group(1)))
