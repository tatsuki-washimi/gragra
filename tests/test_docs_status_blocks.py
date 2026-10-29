"""Verification Block checks for the documentation site.

A Verification Block is the fixed-schema admonition that states, for one
public symbol, what is guaranteed and what is not:

    ### `point_acceleration` { #status-point-acceleration }

    !!! abstract "Status & verification"

        |  |  |
        |---|---|
        | **Status** | Stable |
        | **Verified against** | closed-form analytic solution |
        | **Agreement** | exact |
        | **Test evidence** | `tests/test_observables.py::test_...` |
        | **Conditions** | ... |
        | **Not verified** | ... |
        | **Known limits** | ... |
        | **Scope** | none |

The schema exists because prose guarantees decay silently: "verified against
the closed form" survives the change that stopped verifying it. This module
makes the block itself checkable -- the labels, their order, the closed
vocabularies, and above all the *test evidence*, which is resolved against the
actual test tree rather than taken at face value.

Two of the assertions here are RED by construction while the content pages are
still skeletons: the pages do not have blocks yet, and the
``<!-- BLOCK: symbol -->`` placeholders that mark where blocks belong have not
been replaced. That is the intended reading of "the work is not done", and it
is deliberately not softened into a skip or an xfail -- a gate that reports
success on an empty page is the failure mode this whole file exists to
prevent. Everything that *can* hold from day one (every assigned symbol has a
reserved anchor; every block that does exist is well-formed) is asserted
separately, so those assertions keep working as content arrives.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DOCS_DIR: Path = REPO_ROOT / "docs_site" / "docs"
MATRIX_PAGE = "verification/matrix.md"

# Top-level public symbol -> the page that owns its Verification Block.
#
# The mapping is pinned here, not derived, because a symbol that belongs to no
# page is exactly the failure this table prevents: it would surface only when
# the verification matrix is assembled, by which time the pages that should
# have covered it are finished. Subpackage symbols are assigned within their
# own page by the lane that writes it and are not listed here.
BLOCK_ASSIGNMENTS: dict[str, tuple[str, ...]] = {
    "concepts/conventions.md": ("G_SI",),
    "guide/point-sources.md": (
        "WeightedPointSource",
        "PointMass",
        "PointMassCloud",
        "TargetPoints",
    ),
    "guide/observables.md": (
        "point_potential",
        "point_acceleration",
        "point_gravity_gradient",
    ),
    "guide/coherent.md": (
        "coherent_point_acceleration",
        "coherent_point_gravity_gradient",
        "coherent_gravitoelastic_tensor",
        "coherent_acceleration_csd",
    ),
    "guide/fields.md": (
        "displacement_transfer_acceleration",
        "displacement_field_acceleration",
    ),
}

PAGES_REQUIRING_STATUS: tuple[str, ...] = tuple(BLOCK_ASSIGNMENTS)

FULL_SCHEMA_LABELS: tuple[str, ...] = (
    "Status",
    "Verified against",
    "Agreement",
    "Test evidence",
    "Conditions",
    "Not verified",
    "Known limits",
    "Scope",
)

# A CODATA constant and a raw dot-product primitive have no oracle to have
# been compared against; forcing the full schema on them would produce eight
# rows of boilerplate and teach readers to skim the block everywhere else.
REDUCED_SCHEMA_LABELS: tuple[str, ...] = ("Status", "Source")
REDUCED_SCHEMA_SYMBOLS: frozenset[str] = frozenset({"G_SI"})

STATUS_VOCABULARY: frozenset[str] = frozenset(
    {"stable", "experimental", "diagnostic only", "not implemented"}
)

# Prefixes, not exact strings: two of these carry a parenthesised locator.
VERIFIED_AGAINST_PREFIXES: tuple[str, ...] = (
    "closed-form analytic solution",
    "independent numerical oracle",
    "invariance property",
    "convergence sweep",
    "backend equivalence",
    "published reference",
    "none",
)

DOI_RE = re.compile(r"10\.\d{4,9}/\S+")
NOT_APPLICABLE_RE = re.compile(r"^(n/a|not applicable)\b", re.IGNORECASE)
PLACEHOLDER_RE = re.compile(r"<!--\s*BLOCK:\s*(?P<symbol>[A-Za-z_][A-Za-z0-9_]*)\s*-->")
ANCHOR_HEADING_RE = re.compile(
    r"^#{2,6}\s+(?P<title>.*?)\{\s*#(?P<anchor>[\w.:-]+)\s*\}\s*$"
)
PLAIN_HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<title>.+?)\s*$")
ADMONITION_RE = re.compile(
    r'^(?P<marker>!!!|\?\?\?\+?)\s+\w+\s+"Status & verification"\s*$'
)
TABLE_ROW_RE = re.compile(r"^\|\s*\*\*(?P<label>[^*]+)\*\*\s*\|(?P<value>.*)\|\s*$")
NODE_ID_RE = re.compile(r"tests/[\w./]+\.py::[\w:\[\]\-. ]+")
MARKDOWN_LINK_RE = re.compile(r"\]\((?P<target>[^)\s]+)\)")


def anchor_for(symbol: str) -> str:
    """`point_acceleration` -> `status-point-acceleration`."""
    return "status-" + symbol.replace("_", "-").lower()


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerificationBlock:
    page: str
    line: int  # 1-based line of the admonition marker
    anchor: str | None  # anchor of the heading immediately above, if any
    heading_symbol: str | None  # backticked name in that heading, if any
    cells: tuple[tuple[str, str], ...]  # (label, value) in document order


def _preceding_heading(lines: list[str], index: int) -> tuple[str | None, str | None]:
    """Return (anchor, backticked symbol) of the nearest heading above ``index``."""
    for cursor in range(index - 1, -1, -1):
        line = lines[cursor].rstrip()
        if not line.strip():
            continue
        match = ANCHOR_HEADING_RE.match(line)
        if match:
            title = match.group("title")
            symbol = re.search(r"`([A-Za-z_][A-Za-z0-9_.]*)`", title)
            return match.group("anchor"), (symbol.group(1) if symbol else None)
        if PLAIN_HEADING_RE.match(line):
            return None, None
        return None, None
    return None, None


def parse_blocks(page: str, text: str) -> list[VerificationBlock]:
    """Extract every Verification Block from a page."""
    lines = text.splitlines()
    blocks: list[VerificationBlock] = []
    for index, line in enumerate(lines):
        if not ADMONITION_RE.match(line.strip()):
            continue
        anchor, heading_symbol = _preceding_heading(lines, index)
        cells: list[tuple[str, str]] = []
        cursor = index + 1
        while cursor < len(lines):
            body = lines[cursor]
            if body.strip() and not body.startswith(("    ", "\t")):
                break
            row = TABLE_ROW_RE.match(body.strip())
            if row:
                cells.append(
                    (row.group("label").strip(), row.group("value").strip(" |"))
                )
            cursor += 1
        blocks.append(
            VerificationBlock(
                page=page,
                line=index + 1,
                anchor=anchor,
                heading_symbol=heading_symbol,
                cells=tuple(cells),
            )
        )
    return blocks


def _page_text(page: str) -> str:
    return (DOCS_DIR / page).read_text(encoding="utf-8")


def _all_pages() -> list[str]:
    return sorted(p.relative_to(DOCS_DIR).as_posix() for p in DOCS_DIR.rglob("*.md"))


def _all_blocks() -> list[VerificationBlock]:
    blocks: list[VerificationBlock] = []
    for page in _all_pages():
        blocks.extend(parse_blocks(page, _page_text(page)))
    return blocks


def _placeholders(page: str) -> list[str]:
    return PLACEHOLDER_RE.findall(_page_text(page))


def page_anchors(text: str) -> set[str]:
    """Anchors a page offers: explicit ``{ #id }`` ids plus slugified headings."""
    anchors: set[str] = set()
    for line in text.splitlines():
        stripped = line.rstrip()
        explicit = ANCHOR_HEADING_RE.match(stripped)
        if explicit:
            anchors.add(explicit.group("anchor"))
            continue
        plain = PLAIN_HEADING_RE.match(stripped)
        if plain:
            slug = re.sub(r"[^\w\s-]", "", plain.group("title").lower())
            anchors.add(re.sub(r"[\s]+", "-", slug.strip()))
    return anchors


# ---------------------------------------------------------------------------
# Structural invariants that hold from day one
# ---------------------------------------------------------------------------


def test_every_assigned_symbol_has_a_reserved_anchor_and_a_slot() -> None:
    """The frozen skeleton keeps its side of the bargain.

    Each assigned symbol must have an anchored heading on its page, and below
    it either the placeholder (work pending) or a real block (work done).
    Asserting both states as acceptable is what lets this invariant hold
    continuously while content is written, instead of only at the end.
    """
    problems: list[str] = []
    for page, symbols in BLOCK_ASSIGNMENTS.items():
        path = DOCS_DIR / page
        if not path.is_file():
            problems.append(f"{page}: page does not exist")
            continue
        text = path.read_text(encoding="utf-8")
        anchors = page_anchors(text)
        pending = set(_placeholders(page))
        documented = {block.heading_symbol for block in parse_blocks(page, text)}
        for symbol in symbols:
            expected_anchor = anchor_for(symbol)
            if expected_anchor not in anchors:
                problems.append(
                    f"{page}: no heading carries the anchor "
                    f"{{ #{expected_anchor} }} for {symbol!r}"
                )
            if symbol not in pending and symbol not in documented:
                problems.append(
                    f"{page}: {symbol!r} has neither a "
                    f"<!-- BLOCK: {symbol} --> placeholder nor a block"
                )
    assert not problems, "\n  ".join(["assignment table is not honoured:", *problems])


def test_placeholders_only_name_assigned_symbols() -> None:
    """No stray placeholder for a symbol nobody agreed to document here."""
    stray: list[str] = []
    for page in _all_pages():
        assigned = set(BLOCK_ASSIGNMENTS.get(page, ()))
        for symbol in _placeholders(page):
            if symbol not in assigned:
                stray.append(f"{page}: <!-- BLOCK: {symbol} -->")
    assert not stray, (
        "these placeholders name a symbol that the assignment table does not "
        "put on that page:\n  " + "\n  ".join(stray)
    )


# ---------------------------------------------------------------------------
# Block well-formedness (vacuous until blocks exist, then load-bearing)
# ---------------------------------------------------------------------------


def _expected_labels(block: VerificationBlock) -> tuple[str, ...]:
    if block.heading_symbol in REDUCED_SCHEMA_SYMBOLS:
        return REDUCED_SCHEMA_LABELS
    return FULL_SCHEMA_LABELS


def test_blocks_carry_the_full_schema_in_order_with_no_empty_cell() -> None:
    problems: list[str] = []
    for block in _all_blocks():
        expected = _expected_labels(block)
        actual = tuple(label for label, _ in block.cells)
        if actual != expected:
            problems.append(
                f"{block.page}:{block.line}: labels are {list(actual)}, expected "
                f"{list(expected)} in that order"
            )
            continue
        for label, value in block.cells:
            if not value.strip():
                problems.append(
                    f"{block.page}:{block.line}: '{label}' is empty -- write "
                    "'n/a -- <reason>' if it genuinely does not apply"
                )
            elif NOT_APPLICABLE_RE.match(value.strip()) and not re.search(
                r"(--|—|-)\s*\S", value.strip()[3:]
            ):
                problems.append(
                    f"{block.page}:{block.line}: '{label}' says it does not "
                    "apply but gives no reason"
                )
    assert not problems, "\n  ".join(["malformed Verification Blocks:", *problems])


def _vocabulary_parts(value: str) -> list[str]:
    return [part.strip() for part in value.split(";") if part.strip()]


def test_status_and_verified_against_use_the_closed_vocabularies() -> None:
    problems: list[str] = []
    for block in _all_blocks():
        cells = dict(block.cells)

        status = cells.get("Status", "").strip().strip("*").strip()
        if status and status.lower() not in STATUS_VOCABULARY:
            problems.append(
                f"{block.page}:{block.line}: Status {status!r} is not one of "
                f"{sorted(STATUS_VOCABULARY)}"
            )

        verified = cells.get("Verified against", "").strip()
        if not verified or NOT_APPLICABLE_RE.match(verified):
            continue
        for part in _vocabulary_parts(verified):
            lowered = part.lower().lstrip("*_` ")
            if not lowered.startswith(VERIFIED_AGAINST_PREFIXES):
                problems.append(
                    f"{block.page}:{block.line}: 'Verified against' entry "
                    f"{part!r} is outside the closed vocabulary "
                    f"{list(VERIFIED_AGAINST_PREFIXES)} (separate several "
                    "entries with ';')"
                )
            elif lowered.startswith("published reference") and not DOI_RE.search(part):
                problems.append(
                    f"{block.page}:{block.line}: 'published reference' entry "
                    f"{part!r} carries no DOI. The DOI is the whole point of "
                    "this vocabulary entry: it is what keeps an unpublished "
                    "talk or thesis from being cited as verification evidence."
                )
    assert not problems, "\n  ".join(["vocabulary violations:", *problems])


# ---------------------------------------------------------------------------
# Test evidence resolves against the real test tree
# ---------------------------------------------------------------------------


def _defined_names(path: Path) -> set[str]:
    """Every function/class name defined anywhere in a test module.

    Parsed with ``ast`` rather than matched with a regex, for the same reason
    the CI drift check does: a name that appears only inside a docstring or a
    comment is not a test that exists.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def test_test_evidence_node_ids_resolve() -> None:
    problems: list[str] = []
    for block in _all_blocks():
        evidence = dict(block.cells).get("Test evidence", "").strip()
        if not evidence or NOT_APPLICABLE_RE.match(evidence):
            continue
        node_ids = NODE_ID_RE.findall(evidence)
        if not node_ids:
            problems.append(
                f"{block.page}:{block.line}: 'Test evidence' contains no "
                f"tests/<file>.py::<name> reference ({evidence!r})"
            )
            continue
        for node_id in node_ids:
            relpath, _, remainder = node_id.partition("::")
            path = REPO_ROOT / relpath
            if not path.is_file():
                problems.append(f"{block.page}:{block.line}: {node_id} -- no such file")
                continue
            defined = _defined_names(path)
            wanted = [
                re.sub(r"\[.*\]$", "", part).strip()
                for part in remainder.split("::")
                if part.strip()
            ]
            missing = [name for name in wanted if name not in defined]
            if missing:
                problems.append(
                    f"{block.page}:{block.line}: {node_id} -- {missing} not "
                    f"defined in {relpath}"
                )
    assert not problems, "\n  ".join(
        ["Verification Blocks cite tests that do not exist:", *problems]
    )


def test_matrix_rows_link_to_anchors_that_exist() -> None:
    """The matrix must link into the blocks, not merely name them.

    Requiring the link to resolve (rather than requiring the two sets of
    symbols to match) is what makes the matrix stay usable: a row whose target
    heading was renamed still names the right symbol, and would pass a
    set-equality check while sending the reader nowhere.
    """
    path = DOCS_DIR / MATRIX_PAGE
    assert path.is_file(), f"{MATRIX_PAGE} is missing"
    problems: list[str] = []
    for target in MARKDOWN_LINK_RE.findall(path.read_text(encoding="utf-8")):
        if "://" in target or target.startswith("mailto:"):
            continue
        page_part, _, anchor = target.partition("#")
        if not anchor:
            continue
        base = (DOCS_DIR / MATRIX_PAGE).parent
        resolved = (base / page_part if page_part else path).resolve()
        if not resolved.is_file():
            problems.append(f"{target} -- target page does not exist")
            continue
        if anchor not in page_anchors(resolved.read_text(encoding="utf-8")):
            problems.append(f"{target} -- target page has no such anchor")
    assert not problems, "\n  ".join(
        [f"{MATRIX_PAGE} links that do not resolve:", *problems]
    )


def test_every_block_is_preceded_by_an_anchored_heading() -> None:
    """Without an anchor the matrix cannot link to the block.

    A Material admonition carries no ``id`` attribute of its own, so the
    anchor has to live on the heading above it.
    """
    problems = [
        f"{block.page}:{block.line}" for block in _all_blocks() if block.anchor is None
    ]
    assert not problems, (
        "these Verification Blocks are not preceded by a heading carrying an "
        "explicit { #anchor }:\n  " + "\n  ".join(problems)
    )


# ---------------------------------------------------------------------------
# Progress gates -- RED until the content pages are written
# ---------------------------------------------------------------------------


def test_no_block_placeholder_remains() -> None:
    remaining = [
        f"{page}: {symbol}" for page in _all_pages() for symbol in _placeholders(page)
    ]
    assert not remaining, (
        "these Verification Block placeholders have not been replaced by real "
        "blocks:\n  " + "\n  ".join(remaining) + "\n\n"
        "This assertion is RED by design while the documentation pages are "
        "still skeletons; it turns green when the last block is written. It is "
        "not an error in the gate."
    )


def test_pages_requiring_status_have_at_least_one_block() -> None:
    missing = [
        page
        for page in PAGES_REQUIRING_STATUS
        if not parse_blocks(page, _page_text(page))
    ]
    assert not missing, (
        "these pages own Verification Blocks in the assignment table but "
        "contain none:\n  " + "\n  ".join(missing) + "\n\n"
        "RED by design until those pages are written."
    )


# ---------------------------------------------------------------------------
# Self-tests of the parser
# ---------------------------------------------------------------------------


_FIXTURE_PAGE = """\
## Status and verification

### `point_acceleration` { #status-point-acceleration }

!!! abstract "Status & verification"

    |  |  |
    |---|---|
    | **Status** | Stable |
    | **Verified against** | closed-form analytic solution |
    | **Agreement** | exact |
    | **Test evidence** | `tests/test_observables.py::test_x` |
    | **Conditions** | softening_m == 0 |
    | **Not verified** | softened near field |
    | **Known limits** | Newtonian only |
    | **Scope** | none |

Ordinary prose after the block.
"""


def test_parser_reads_the_anchor_symbol_and_all_eight_cells() -> None:
    blocks = parse_blocks("fixture.md", _FIXTURE_PAGE)
    assert len(blocks) == 1
    block = blocks[0]
    assert block.anchor == "status-point-acceleration"
    assert block.heading_symbol == "point_acceleration"
    assert tuple(label for label, _ in block.cells) == FULL_SCHEMA_LABELS
    assert dict(block.cells)["Agreement"] == "exact"


def test_parser_stops_at_the_end_of_the_admonition() -> None:
    """Prose after the block must not be swallowed into it."""
    block = parse_blocks("fixture.md", _FIXTURE_PAGE)[0]
    assert all("prose" not in value for _, value in block.cells)


def test_anchor_derivation_matches_the_convention() -> None:
    assert anchor_for("point_acceleration") == "status-point-acceleration"
    assert anchor_for("G_SI") == "status-g-si"
    assert anchor_for("WeightedPointSource") == "status-weightedpointsource"


def test_page_anchors_include_explicit_ids_and_slugified_headings() -> None:
    anchors = page_anchors("# A Title\n\n### `x` { #status-x }\n\n## Another Heading\n")
    assert "status-x" in anchors
    assert "a-title" in anchors
    assert "another-heading" in anchors
