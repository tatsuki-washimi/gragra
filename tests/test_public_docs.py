"""Regression checks for the 0.2.0rc1 public documentation boundary."""

import inspect
import re
from pathlib import Path

from gragra import G_SI, PointMass, point_acceleration
from gragra.adapters.specfem.reader import SpecfemGLLDataset, read_specfem_gll_hdf5
from gragra.observables import displacement_field_acceleration

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs_site" / "docs"
VERSION = "0.2.0rc1"


def test_release_metadata_identifies_rc_and_author() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    codemeta = (ROOT / "codemeta.json").read_text(encoding="utf-8")

    assert f'version = "{VERSION}"' in pyproject
    assert f"version: {VERSION}" in citation
    assert "family-names: Washimi" in citation
    assert f'"version": "{VERSION}"' in codemeta
    assert '"familyName": "Washimi"' in codemeta


def test_mkdocs_material_site_is_configured() -> None:
    config = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert "theme:" in config
    assert "plugins:" in config
    assert "mkdocstrings" in config
    assert (DOCS / "index.md").is_file()
    assert len(list(DOCS.rglob("*.md"))) >= 20


def test_docs_avoid_private_repository_links() -> None:
    text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in DOCS.rglob("*.md")
    )
    assert "gragra-dev" not in text
    assert "chen_taup2019" not in text.lower()


def test_internal_markdown_links_resolve_under_docs_site() -> None:
    for document in sorted(DOCS.rglob("*.md")):
        for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", document.read_text()):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            relative_target = target.split("#", maxsplit=1)[0]
            if not relative_target:
                continue
            assert (document.parent / relative_target).is_file(), (document, target)


def test_workflow_has_no_publication_or_deployment_step() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    forbidden = ("deploy", "pages", "pypi", "release", "upload-artifact", "tag")
    assert not any(item in workflow.lower() for item in forbidden)


def test_manual_api_inventory_matches_public_signatures() -> None:
    reader_signature = inspect.signature(read_specfem_gll_hdf5)
    assert tuple(reader_signature.parameters) == ("path", "profile_density_range")
    assert (
        reader_signature.parameters["profile_density_range"].default is inspect._empty
    )
    signature = inspect.signature(displacement_field_acceleration)
    assert "dataset" in signature.parameters
    assert "targets" in signature.parameters
    block_signature = inspect.signature(SpecfemGLLDataset.iter_blocks)
    assert "element_block_size" in block_signature.parameters
    point_mass_signature = inspect.signature(PointMass)
    assert tuple(point_mass_signature.parameters) == ("mass_kg", "position_m")
    point_signature = inspect.signature(point_acceleration)
    assert "source" in point_signature.parameters
    assert "targets" in point_signature.parameters
    result = point_acceleration(PointMass(1.0, [-1.0, 0.0, 0.0]), [[0.0, 0.0, 0.0]])
    assert result.tolist() == [[-G_SI, 0.0, 0.0]]
    assert result.shape == (1, 3)
    assert not result.flags.writeable
