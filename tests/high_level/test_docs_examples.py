"""Directly parse and execute markdown fenced code blocks from documentation guides.

Ensures that code examples in docs_site/docs/guide/ are directly extracted
from the Markdown files and execute cleanly without any syntax, import,
or runtime errors.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _extract_python_blocks(md_path: Path) -> list[str]:
    content = md_path.read_text(encoding="utf-8")
    pattern = re.compile(r"```python\n(.*?)\n```", re.DOTALL)
    return pattern.findall(content)


def test_source_diagnostics_guide_markdown_execution() -> None:
    """Extract and execute all Python snippets from source-diagnostics.md."""
    md_file = (
        REPO_ROOT / "docs_site" / "docs" / "guide" / "source-diagnostics.md"
    )
    assert md_file.exists(), f"{md_file} does not exist"

    snippets = _extract_python_blocks(md_file)
    assert len(snippets) >= 4, f"Expected >= 4 snippets, got {len(snippets)}"

    # Shared execution namespace so variables defined in setup are reused
    namespace: dict[str, object] = {}

    for idx, snippet in enumerate(snippets, start=1):
        try:
            exec(snippet, namespace)
        except Exception as exc:
            raise RuntimeError(
                f"Failed executing snippet {idx} from {md_file.name}:\n{snippet}"
            ) from exc


def test_water_newtonian_noise_guide_markdown_execution() -> None:
    """Extract and execute all Python snippets from water-newtonian-noise.md."""
    md_file = (
        REPO_ROOT / "docs_site" / "docs" / "guide" / "water-newtonian-noise.md"
    )
    assert md_file.exists(), f"{md_file} does not exist"

    snippets = _extract_python_blocks(md_file)
    assert len(snippets) >= 1, f"Expected >= 1 snippet, got {len(snippets)}"

    namespace: dict[str, object] = {}
    for idx, snippet in enumerate(snippets, start=1):
        try:
            exec(snippet, namespace)
        except Exception as exc:
            raise RuntimeError(
                f"Failed executing snippet {idx} from {md_file.name}:\n{snippet}"
            ) from exc
