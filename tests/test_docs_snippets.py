"""Executable-example checks for the documentation site.

The site's promise is that every Python example shown to a reader is the exact
text that the test suite ran. That promise is only worth something if it is
impossible to break by accident, so this module closes the four ways it
otherwise leaks:

* **Includes cannot reach outside the allowed roots.** Every ``--8<--`` target
  must resolve under ``docs_site/`` or ``examples/`` -- the same two roots
  ``pymdownx.snippets`` is configured with.
* **Snippets actually run**, in a temporary working directory, writing
  nothing, importing only NumPy, gragra, and Astropy units.
* **Their assertions have teeth.** NumPy's default ``atol=1e-8`` is larger
  than gragra's entire output scale: ``np.allclose`` at defaults reports a
  gravitational acceleration as equal to zero, and equal to its own negation.
  Every ``np.allclose`` / ``np.isclose`` in a snippet must pass an explicit
  ``atol``.

The import restriction keeps the small documentation examples runnable in the
core environment. Examples that genuinely need an optional extra live in
``examples/`` and are linked from the page as explicitly-labelled material.

The parsing helpers are exercised against synthetic fixtures at the bottom of
this file, so these checks are tested by something other than "it passed
today".
"""

from __future__ import annotations

import ast
import runpy
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DOCS_DIR: Path = REPO_ROOT / "docs_site" / "docs"
SNIPPETS_DIR: Path = REPO_ROOT / "docs_site" / "snippets"

# The roots `pymdownx.snippets.base_path` is set to in mkdocs.yml. Kept in
# sync with the local documentation configuration.
INCLUDE_ROOTS: tuple[str, ...] = ("docs_site", "examples")

# Pages whose Python blocks are read outside MkDocs carry the snippet text
# inline and must reproduce it verbatim.
MIRRORED_SOURCES: tuple[str, ...] = ("README.md", "examples/README.md")

PYTHON_LANGUAGES: frozenset[str] = frozenset({"python", "py", "python3"})

ALLOWED_SNIPPET_IMPORT_ROOTS: frozenset[str] = frozenset({"numpy", "gragra"})
ALLOWED_ASTROPY_IMPORT_MODULES: frozenset[str] = frozenset({"astropy.units"})

TOLERANCE_FUNCTIONS: frozenset[str] = frozenset({"allclose", "isclose"})


# ---------------------------------------------------------------------------
# Markdown fenced-block parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FencedBlock:
    """One fenced code block found in a markdown file."""

    source: str  # repository-relative path of the containing file
    line: int  # 1-based line number of the opening fence
    language: str  # first word of the info string, lowercased ("" if none)
    body: str  # text between the fences, verbatim

    @property
    def is_python(self) -> bool:
        return self.language in PYTHON_LANGUAGES


def iter_fenced_blocks(source: str, text: str) -> list[FencedBlock]:
    """Extract fenced code blocks from markdown text.

    Handles both backtick and tilde fences and the CommonMark rule that a
    closing fence must use the same character and be at least as long as the
    opening one -- which is what makes a nested fence inside a
    ``pymdownx.superfences`` block parse correctly rather than terminating its
    parent early.
    """
    blocks: list[FencedBlock] = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        if indent > 3 or stripped[:3] not in ("```", "~~~"):
            index += 1
            continue

        fence_char = stripped[0]
        fence_len = len(stripped) - len(stripped.lstrip(fence_char))
        info = stripped[fence_len:].strip()
        language = info.split()[0].lower().lstrip("{.") if info else ""

        start_line = index + 1
        body_lines: list[str] = []
        index += 1
        while index < len(lines):
            candidate = lines[index].lstrip(" ")
            if candidate and candidate[0] == fence_char:
                closing_len = len(candidate) - len(candidate.lstrip(fence_char))
                if closing_len >= fence_len and not candidate[closing_len:].strip():
                    break
            body_lines.append(lines[index])
            index += 1
        index += 1  # step over the closing fence

        body = "".join(f"{body_line}\n" for body_line in body_lines)
        blocks.append(
            FencedBlock(source=source, line=start_line, language=language, body=body)
        )
    return blocks


def include_targets(body: str) -> list[str]:
    """Return the target paths of every ``--8<--`` include line in ``body``."""
    targets: list[str] = []
    for raw in body.splitlines():
        line = raw.strip()
        if not line.startswith("--8<--"):
            continue
        remainder = line[len("--8<--") :].strip()
        quoted = (
            len(remainder) >= 2
            and remainder[0] == remainder[-1]
            and remainder[0] in "\"'"
        )
        if quoted:
            targets.append(remainder[1:-1])
    return targets


def is_include_only(body: str) -> bool:
    """True when the block contains nothing but include directives."""
    meaningful = [line for line in body.splitlines() if line.strip()]
    if not meaningful:
        return False
    return all(line.strip().startswith("--8<--") for line in meaningful)


def resolve_include(target: str) -> Path | None:
    """Resolve an include target under the allowed roots, or return ``None``.

    ``None`` covers both "no such file" and "escapes the allowed roots"; the
    caller reports them with the same message because both mean the same
    thing operationally -- the include does not point at documentation or
    executable material.
    """
    for root_name in INCLUDE_ROOTS:
        root = (REPO_ROOT / root_name).resolve()
        candidate = (root / target).resolve()
        if not candidate.is_relative_to(root):
            continue
        if candidate.is_file():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Repository inventory
# ---------------------------------------------------------------------------


def _scanned_markdown_files() -> list[Path]:
    files = sorted(DOCS_DIR.rglob("*.md"))
    for relpath in MIRRORED_SOURCES:
        path = REPO_ROOT / relpath
        if path.is_file():
            files.append(path)
    return files


def _all_blocks() -> list[FencedBlock]:
    blocks: list[FencedBlock] = []
    for path in _scanned_markdown_files():
        relpath = path.relative_to(REPO_ROOT).as_posix()
        blocks.extend(iter_fenced_blocks(relpath, path.read_text(encoding="utf-8")))
    return blocks


def _snippet_files() -> list[Path]:
    if not SNIPPETS_DIR.is_dir():
        return []
    return sorted(SNIPPETS_DIR.rglob("*.py"))


# ---------------------------------------------------------------------------
# Static analysis of snippet source
# ---------------------------------------------------------------------------


def disallowed_imports(tree: ast.Module) -> list[str]:
    """Return snippet imports outside NumPy, gragra, and Astropy units."""
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in ALLOWED_SNIPPET_IMPORT_ROOTS:
                    continue
                if alias.name in ALLOWED_ASTROPY_IMPORT_MODULES:
                    continue
                if alias.name.startswith("astropy.units."):
                    continue
                offenders.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                offenders.append("." * node.level + (node.module or ""))
                continue
            module = node.module or ""
            root = module.split(".")[0]
            if root in ALLOWED_SNIPPET_IMPORT_ROOTS:
                continue
            if module == "astropy.units" or module.startswith("astropy.units."):
                continue
            if (
                module == "astropy"
                and node.names
                and all(alias.name == "units" for alias in node.names)
            ):
                continue
            offenders.append(module)
    return offenders


def tolerance_calls_without_atol(tree: ast.Module) -> list[int]:
    """Return the line numbers of ``allclose``/``isclose`` calls lacking ``atol``.

    Both ``np.allclose(...)`` and a from-imported bare ``allclose(...)`` are
    recognised. The check is on the *keyword*, not on its value: passing
    ``atol=0`` is the intended way to say "compare by relative tolerance
    only", and is exactly what a non-zero fixed point should use.
    """
    offenders: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            name = func.attr
        elif isinstance(func, ast.Name):
            name = func.id
        else:
            continue
        if name not in TOLERANCE_FUNCTIONS:
            continue
        if not any(keyword.arg == "atol" for keyword in node.keywords):
            offenders.append(node.lineno)
    return offenders


# ---------------------------------------------------------------------------
# G2: mirrored examples use the same snippet text
# ---------------------------------------------------------------------------


def _snippet_body_map() -> dict[str, Path]:
    """Map each snippet file's exact text to its path.

    A MIRRORED_SOURCES block whose body equals a snippet's text is displaying
    that snippet just as surely as a `--8<--` include displays it. The orphan
    check uses the same definition of "shown".
    """
    return {path.read_text(encoding="utf-8"): path for path in _snippet_files()}


# ---------------------------------------------------------------------------
# G3: includes resolve, and snippets are not orphaned
# ---------------------------------------------------------------------------


def test_every_include_resolves_under_an_allowed_root() -> None:
    broken: list[str] = []
    for block in _all_blocks():
        for target in include_targets(block.body):
            if resolve_include(target) is None:
                broken.append(f"{block.source}:{block.line} -> {target!r}")
    assert not broken, (
        "these `--8<--` includes do not resolve to a file under "
        f"{list(INCLUDE_ROOTS)}:\n  " + "\n  ".join(broken)
    )


def test_no_snippet_is_orphaned() -> None:
    """Every executable snippet is shown on at least one page.

    The reverse direction of the include check. An orphaned snippet is a file
    the suite keeps running and nobody reads -- and, worse, a plausible-looking
    place to put code that no page ever had to justify.
    """
    included: set[Path] = set()
    for block in _all_blocks():
        for target in include_targets(block.body):
            resolved = resolve_include(target)
            if resolved is not None:
                included.add(resolved)

    snippet_bodies = _snippet_body_map()
    for block in _all_blocks():
        if block.source in MIRRORED_SOURCES and block.body in snippet_bodies:
            included.add(snippet_bodies[block.body].resolve())

    orphaned = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in _snippet_files()
        if path.resolve() not in included
    ]
    assert not orphaned, (
        "these snippets are executed by the test suite but embedded in no "
        "page:\n  " + "\n  ".join(orphaned)
    )


# ---------------------------------------------------------------------------
# G4 / G4b: snippet source is restricted, and its assertions are meaningful
# ---------------------------------------------------------------------------


def test_snippets_use_only_core_imports_and_astropy_units() -> None:
    offenders: list[str] = []
    for path in _snippet_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for name in disallowed_imports(tree):
            offenders.append(f"{path.relative_to(REPO_ROOT).as_posix()}: {name}")
    assert not offenders, (
        "snippets may import only numpy, gragra, and astropy.units:\n  "
        + "\n  ".join(offenders)
        + "\n\nAn optional extra would force this test to skip where the extra "
        "is absent in the core environment. Put "
        "such an example in examples/ and link to it from the page instead."
    )


def test_snippet_tolerance_assertions_state_an_explicit_atol() -> None:
    offenders: list[str] = []
    for path in _snippet_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for lineno in tolerance_calls_without_atol(tree):
            offenders.append(f"{path.relative_to(REPO_ROOT).as_posix()}:{lineno}")
    assert not offenders, (
        "these allclose/isclose calls rely on NumPy's default atol=1e-8:\n  "
        + "\n  ".join(offenders)
        + "\n\nAt gragra's output scale that default is larger than the "
        "quantities being compared, so the assertion passes for the zero array "
        "and for the sign-flipped array alike -- it demonstrates nothing. Pass "
        "atol=0 for a non-zero fixed point (relative comparison only), or, for "
        "a quantity whose true value is zero, normalise the residual by the "
        "largest contributing term instead of comparing against zero at all."
    )


def test_snippets_run_and_leave_no_files_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Execute every snippet in a temporary working directory."""
    monkeypatch.chdir(tmp_path)
    for path in _snippet_files():
        before = set(tmp_path.iterdir())
        runpy.run_path(str(path), run_name="__main__")
        new_files = set(tmp_path.iterdir()) - before
        assert not new_files, (
            f"snippet {path.relative_to(REPO_ROOT).as_posix()} wrote files: "
            + ", ".join(sorted(f.name for f in new_files))
        )


# ---------------------------------------------------------------------------
# Self-tests of the parsing helpers
#
# The checks above are only as good as their parser, and while the snippet
# directory is still empty almost all of it passes vacuously. These fixtures
# check the parsing machinery directly, including its boundary cases.
# ---------------------------------------------------------------------------


_FIXTURE_MARKDOWN = """\
# Title

```python
import numpy as np
```

```bash
pip install -e ".[dev]"
```

```python
--8<-- "snippets/example.py"
```

````markdown
```python
this nested block belongs to the outer markdown block
```
````

~~~python
tilde_fenced = True
~~~
"""


def test_fence_parser_finds_every_block_with_its_language() -> None:
    blocks = iter_fenced_blocks("fixture.md", _FIXTURE_MARKDOWN)
    assert [b.language for b in blocks] == [
        "python",
        "bash",
        "python",
        "markdown",
        "python",
    ], "a nested fence must not terminate its parent block early"
    assert blocks[0].body == "import numpy as np\n"
    assert blocks[4].body == "tilde_fenced = True\n"


def test_include_only_and_target_extraction() -> None:
    assert is_include_only('--8<-- "snippets/example.py"\n')
    assert not is_include_only('import numpy\n--8<-- "snippets/example.py"\n')
    assert not is_include_only("\n")
    assert include_targets("--8<-- \"snippets/a.py\"\n--8<-- 'examples/b.py'\n") == [
        "snippets/a.py",
        "examples/b.py",
    ]


def test_include_resolution_is_confined_to_the_allowed_roots() -> None:
    assert resolve_include("snippets/README.md") is not None
    assert resolve_include("README.md") is not None  # examples/README.md
    # A traversal out of the allowed roots must not resolve, even though the
    # file plainly exists -- this is the assertion that keeps an internal
    # document from being treated as an example include.
    assert resolve_include("../README.md") is None
    assert resolve_include("../pyproject.toml") is None
    assert resolve_include("snippets/does-not-exist.py") is None


def test_import_check_distinguishes_allowed_from_disallowed() -> None:
    allowed = ast.parse(
        "import numpy as np\nimport gragra\nfrom gragra import PointMass\n"
        "from numpy.linalg import norm\nfrom astropy import units as u\n"
        "import astropy.units as units\nfrom astropy.units import Quantity\n"
    )
    assert disallowed_imports(allowed) == []

    disallowed = ast.parse(
        "import h5py\nfrom scipy import special\nfrom . import x\n"
        "import astropy\nfrom astropy import constants\n"
        "from astropy.visualization import quantity_support\n"
    )
    assert disallowed_imports(disallowed) == [
        "h5py",
        "scipy",
        ".",
        "astropy",
        "astropy",
        "astropy.visualization",
    ]


def test_atol_check_accepts_explicit_zero_and_rejects_the_default() -> None:
    explicit = ast.parse(
        "import numpy as np\n"
        "np.allclose(a, b, rtol=1e-12, atol=0.0)\n"
        "np.isclose(a, b, atol=1e-30)\n"
    )
    assert tolerance_calls_without_atol(explicit) == []

    implicit = ast.parse(
        "import numpy as np\n"
        "np.allclose(a, b)\n"
        "np.allclose(a, b, rtol=1e-12)\n"
        "from numpy import isclose\n"
        "isclose(a, b)\n"
    )
    assert tolerance_calls_without_atol(implicit) == [2, 3, 5]
