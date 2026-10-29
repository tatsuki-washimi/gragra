"""The continuous Fourier API is documented where users look for it.

A targeted check for the new surface only. The repository-wide public-symbol
inventory in ``test_docs_api_coverage.py`` is tracked separately.
"""

from __future__ import annotations

import inspect
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GUIDE = "guide/rigid-motion-fourier.md"


def _read(*parts: str) -> str:
    return REPO_ROOT.joinpath(*parts).read_text(encoding="utf-8")


def test_api_reference_renders_rigid_motion_and_links_the_guide():
    core = _read("docs_site", "docs", "api", "core.md")
    assert "::: gragra.gravity_sources.rigid_motion" in core
    assert "../guide/rigid-motion-fourier.md" in core


def test_guide_is_in_the_navigation_and_exists():
    assert GUIDE in _read("mkdocs.yml")
    assert (REPO_ROOT / "docs_site" / "docs" / GUIDE).is_file()


def test_guide_documents_both_operations_and_their_applicability():
    guide = _read("docs_site", "docs", GUIDE)
    for needle in (
        "fourier_series",
        "fourier_transform",
        "tref",
        "tau_min",
        "strictly increasing",
        "output_frame",
    ):
        assert needle in guide, needle


def test_public_api_inventory_lists_the_continuous_form():
    inventory = _read("docs", "architecture", "public_api.md")
    assert "fourier_series" in inventory
    assert "fourier_transform" in inventory
    assert "epoch=" in inventory


def test_signature_and_docstring_match_the_documented_form():
    from gragra.gravity_sources import rigid_motion

    signature = inspect.signature(rigid_motion)
    assert {"frame", "sampling", "epoch"} <= set(signature.parameters)
    doc = inspect.getdoc(rigid_motion) or ""
    assert "fourier_series" in doc
    assert "fourier_transform" in doc
