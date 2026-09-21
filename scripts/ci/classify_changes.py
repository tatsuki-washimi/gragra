"""Classify a complete git name-status diff for public CI selection."""

from __future__ import annotations

import sys
from pathlib import PurePosixPath

_DOCUMENTATION_FILES = frozenset(
    {
        "AGENTS.md",
        "CONTRIBUTING.md",
        "README.md",
    }
)


def _is_documentation_path(path: str) -> bool:
    normalized = path.removeprefix("./")
    if normalized in _DOCUMENTATION_FILES:
        return True
    if not normalized.startswith("docs_site/"):
        return False
    return PurePosixPath(normalized).suffix.lower() in {".md", ".mdx"}


def classify_paths(records: str) -> tuple[bool, str]:
    """Return whether *records* are docs-only and explain the decision."""
    if not records:
        return False, "full: empty diff"

    for line_number, record in enumerate(records.splitlines(), start=1):
        fields = record.split("\t")
        if len(fields) != 2 or fields[0] not in {"A", "M"} or not fields[1]:
            return (
                False,
                f"full: malformed or unsupported status record on line {line_number}",
            )
        if not _is_documentation_path(fields[1]):
            return False, f"full: non-documentation path {fields[1]!r}"

    return True, "docs-only: all records are added or modified documentation"


def main() -> int:
    docs_only, reason = classify_paths(sys.stdin.read())
    print(f"docs_only={'true' if docs_only else 'false'}")
    print(f"reason={reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
