"""The docs stay navigable: the README stays short, and every relative link and anchor in the markdown resolves."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCS = [ROOT / "README.md", ROOT / "reproducibility" / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")


def slug(heading: str) -> str:
    """GitHub's anchor for a heading."""
    text = re.sub(r"[`*_]", "", heading.strip().lower())
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def anchors(path: Path) -> set:
    found, fence = set(), False
    for line in path.read_text().splitlines():
        if line.startswith("```"):
            fence = not fence
        elif not fence and re.match(r"#{1,6} ", line):
            found.add(slug(line.lstrip("#")))
    return found


def test_the_readme_stays_a_summary():
    assert len((ROOT / "README.md").read_text().splitlines()) <= 160


def test_the_four_docs_exist_and_are_linked_from_the_readme():
    readme = (ROOT / "README.md").read_text()
    for name in ("evaluation", "training", "experiments", "operations"):
        assert (ROOT / "docs" / f"{name}.md").exists()
        assert f"docs/{name}.md" in readme


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_relative_links_and_anchors_resolve(doc):
    broken = []
    for target in LINK.findall(doc.read_text()):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        path_part, _, anchor = target.partition("#")
        path = (doc.parent / path_part).resolve() if path_part else doc
        if not path.exists():
            broken.append(f"{target}: no such file")
        elif anchor and path.suffix == ".md" and anchor not in anchors(path):
            broken.append(f"{target}: no heading '{anchor}' in {path.name}")
    assert not broken, broken
