"""Machine checks for the cancellation-sensitive atol registry (§8.5).

This file deliberately does not import cupy and does not probe for a CUDA
device: docs/design/cupy_backend.md §8.5 (the registry of record for
cancellation-sensitive C-1/C-2b/C-3 tests) is plain text, and the
completeness of that text should be checked in every CI run regardless of
GPU availability. `tests/test_cupy_backend_available.py` and friends have a
module-level cupy/GPU skip and would silently skip a doc-only check like
this one if it were added there instead -- this file exists specifically to
avoid that (round-3 review, PR #57 Major 4).
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CUPY_BACKEND_DOC = REPO_ROOT / "docs" / "design" / "cupy_backend.md"

# Registry IDs required by docs/design/cupy_backend.md §8.5, and the number
# of distinct rows each is expected to occupy there (>1 only for the
# chunk-suffixed prefixes, where each chunk_size gets its own row).
REQUIRED_PREFIXES = {
    "C1-dipole-source-cancellation-": 2,
    "C2b-parametric-source-cancellation-": 2,
}
# C3-planewave-phase-only has no chunk_size suffix (consolidated to one row
# since chunk_size is a no-op on this path -- see §8.4/§8.5).
STANDALONE_ID = "C3-planewave-phase-only"

# Test files carrying `# registry-id: <ID>` cross-reference comments.
REGISTRY_ID_TEST_FILES = [
    REPO_ROOT / "tests" / "test_cupy_dipole.py",
    REPO_ROOT / "tests" / "test_cupy_parametric.py",
    REPO_ROOT / "tests" / "test_cupy_planewave.py",
]


def _section_8_5_text() -> str:
    """Extract the §8.5 section body (between the §8.5 and §8.6 headings)."""
    text = CUPY_BACKEND_DOC.read_text(encoding="utf-8")
    start = text.index("### 8.5")
    end = text.index("### 8.6", start)
    return text[start:end]


def _active_registry_text() -> str:
    """Exclude retired measurements retained below the current registry."""
    marker = "#### Historical C-3 volume measurements (retired)"
    return _section_8_5_text().split(marker, 1)[0]


def _all_required_ids() -> list[str]:
    """Full list of registry IDs required in §8.5 and in test cross-references."""
    all_suffixes = ["chunk2", "chunk5", "chunkNone"]
    ids = [STANDALONE_ID]
    for prefix, count in REQUIRED_PREFIXES.items():
        ids.extend(f"{prefix}{suffix}" for suffix in all_suffixes[:count])
    return ids


def test_registry_doc_exists():
    assert CUPY_BACKEND_DOC.is_file(), f"missing {CUPY_BACKEND_DOC}"


def test_registry_no_placeholder_text():
    """§8.5 must not contain the stale C-0 placeholder/unimplemented text."""
    section = _section_8_5_text()
    assert "populated by" not in section, (
        "§8.5 still contains the unpopulated placeholder row text"
    )
    assert "未実装" not in section, (
        "§8.5 still contains C-0-era 'not yet implemented' text"
    )


def test_registry_ids_present_with_expected_row_counts():
    """Every required registry ID must appear exactly once, at a line start.

    Uses a line-start match (`| <ID> |`) rather than a bare substring count:
    a bare substring count would also match the ID's appearance in prose
    (e.g. "C1-dipole-source-cancellation" appears in explanatory text
    elsewhere in §8.5, not just in table rows), producing false positives.
    """
    section = _active_registry_text()
    lines = section.splitlines()

    # Standalone ID: exactly one row, no chunk suffix.
    standalone_rows = [
        line for line in lines if line.startswith(f"| `{STANDALONE_ID}` |")
    ]
    assert len(standalone_rows) == 1, (
        f"expected exactly 1 row for {STANDALONE_ID!r}, found {len(standalone_rows)}"
    )
    # Guard against the standalone ID being accidentally matched as a
    # substring prefix of a chunk-suffixed row (it isn't one of the
    # REQUIRED_PREFIXES, so this should always hold, but assert explicitly).
    assert not any(line.startswith(f"| `{STANDALONE_ID}-chunk") for line in lines), (
        f"{STANDALONE_ID} unexpectedly has chunk-suffixed rows"
    )

    # Chunk-suffixed prefixes: exactly `count` rows per prefix.
    for prefix, expected_count in REQUIRED_PREFIXES.items():
        rows = [line for line in lines if line.startswith(f"| `{prefix}")]
        assert len(rows) == expected_count, (
            f"expected exactly {expected_count} rows with prefix {prefix!r}, "
            f"found {len(rows)}: {rows}"
        )
        # Each individual chunk-suffixed ID must appear exactly once.
        for row in rows:
            row_id = row.split("|")[1].strip().strip("`")
            matching = [r for r in rows if r.split("|")[1].strip().strip("`") == row_id]
            assert len(matching) == 1, (
                f"registry ID {row_id!r} appears in {len(matching)} rows, expected 1"
            )


def test_registry_ids_cross_referenced_exactly_once_in_tests():
    """Every registry ID must have exactly one `# registry-id: <ID>` comment.

    This checks the docs-side registry (§8.5) and the test-side
    measurements stay in sync: a row with no corresponding test comment (or
    a test comment with no corresponding row) would indicate drift.
    """
    combined_test_text = "\n".join(
        path.read_text(encoding="utf-8") for path in REGISTRY_ID_TEST_FILES
    )
    comment_ids = [
        line.split("# registry-id:", 1)[1].strip()
        for line in combined_test_text.splitlines()
        if "# registry-id:" in line
    ]

    required_ids = _all_required_ids()
    for registry_id in required_ids:
        occurrences = [rid for rid in comment_ids if rid == registry_id]
        assert len(occurrences) == 1, (
            f"registry ID {registry_id!r} has {len(occurrences)} "
            f"`# registry-id:` comments across test files, expected exactly 1"
        )

    # No stray comments referencing an ID that isn't in the required set
    # (would indicate a typo or an orphaned/renamed registry entry).
    for comment_id in comment_ids:
        assert comment_id in required_ids, (
            f"test file has `# registry-id: {comment_id}` with no matching "
            f"required registry ID"
        )


def test_retired_volume_measurements_are_historical_only():
    """The removed volume adapter's measurements must not be active registry rows."""
    section = _section_8_5_text()
    marker = "#### Historical C-3 volume measurements (retired)"
    assert marker in section
    active, historical = section.split(marker, 1)
    assert "C3-volume-direction-reduction" not in active
    for suffix in ("chunk2", "chunk5", "chunkNone"):
        assert f"| `C3-volume-direction-reduction-{suffix}` |" in historical
