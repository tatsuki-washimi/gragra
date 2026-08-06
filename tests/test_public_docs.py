"""Regression checks for the rc2 public documentation boundary."""

import inspect
import re
from pathlib import Path

from gragra import G_SI, PointMass, point_acceleration
from gragra.adapters.specfem.reader import SpecfemGLLDataset, read_specfem_gll_hdf5
from gragra.observables import displacement_field_acceleration

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs_site" / "docs"

EXPECTED_PAGES = {
    "index.md",
    "getting-started.md",
    "contracts.md",
    "specfem3d-cartesian.md",
    "backends.md",
    "rc2-scope.md",
    "verification.md",
}


def test_mkdocs_nav_is_the_rc2_allowlist() -> None:
    config = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    for page in EXPECTED_PAGES:
        assert config.count(page) == 1
    assert "plugins:" not in config
    assert "theme:" not in config


def test_rc2_pages_are_present_and_linked_from_readme() -> None:
    assert {path.name for path in DOCS.glob("*.md")} == EXPECTED_PAGES
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for page in EXPECTED_PAGES - {"index.md"}:
        assert f"docs_site/docs/{page}" in readme


def test_internal_markdown_links_resolve() -> None:
    documents = [ROOT / "README.md", *sorted(DOCS.glob("*.md"))]
    for document in documents:
        for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", document.read_text()):
            if target.startswith(("http://", "https://", "#")):
                continue
            relative_target = target.split("#", maxsplit=1)[0]
            assert (document.parent / relative_target).is_file(), (document, target)


def test_docs_state_the_rc2_public_contract() -> None:
    text = "\n".join(
        [
            (ROOT / "README.md").read_text(encoding="utf-8"),
            *(path.read_text(encoding="utf-8") for path in DOCS.glob("*.md")),
        ]
    )
    for required in (
        "profile_density_range=None",
        "PML",
        "not merged",
        "rtol=1e-13",
        "not a streaming reader",
        "GitHub Release",
        "PyPI",
        "sha256sum --check -",
        "interprets the GLL axis",
        "first-order perturbation",
        "caller or exporter",
        "field_units.displacement",
    ):
        assert required in text
    assert "pip install gragra" not in text
    assert "certified" not in text.lower()
    assert "solver validated" not in text.lower()
    assert "removes PML" not in text
    assert "verifies `NGNOD=8`" not in text


def test_release_metadata_identifies_rc2_and_its_author() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    codemeta = (ROOT / "codemeta.json").read_text(encoding="utf-8")

    assert 'version = "0.1.0rc2"' in pyproject
    assert "version: 0.1.0rc2" in citation
    assert "family-names: Washimi" in citation
    assert '"version": "0.1.0rc2"' in codemeta
    assert '"familyName": "Washimi"' in codemeta


def test_python_fences_compile() -> None:
    for path in [ROOT / "README.md", *sorted(DOCS.glob("*.md"))]:
        for block in re.findall(r"```python\n(.*?)```", path.read_text(), re.DOTALL):
            compile(block, str(path), "exec")


def test_readme_smoke_fence_executes() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    smoke = re.findall(r"```python\n(.*?)```", readme, re.DOTALL)[0]
    namespace: dict[str, object] = {}
    exec(compile(smoke, "README.md", "exec"), namespace)


def test_workflow_has_no_publication_or_deployment_step() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    forbidden = ("deploy", "pages", "pypi", "release", "upload-artifact", "tag")
    assert not any(item in workflow.lower() for item in forbidden)


def test_manual_api_inventory_matches_public_signatures() -> None:
    reader_signature = inspect.signature(read_specfem_gll_hdf5)
    assert tuple(reader_signature.parameters) == ("path", "profile_density_range")
    assert (
        reader_signature.parameters["profile_density_range"].default is inspect._empty
    )
    assert (
        reader_signature.parameters["profile_density_range"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    signature = inspect.signature(displacement_field_acceleration)
    assert tuple(signature.parameters) == (
        "dataset",
        "targets",
        "chunk_size",
        "backend",
    )
    assert signature.parameters["chunk_size"].default is None
    assert signature.parameters["backend"].default == "numpy"
    assert signature.parameters["chunk_size"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["backend"].kind is inspect.Parameter.KEYWORD_ONLY
    block_signature = inspect.signature(SpecfemGLLDataset.iter_blocks)
    assert tuple(block_signature.parameters) == ("self", "element_block_size")
    assert (
        block_signature.parameters["element_block_size"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    point_mass_signature = inspect.signature(PointMass)
    assert tuple(point_mass_signature.parameters) == ("mass_kg", "position_m")
    point_signature = inspect.signature(point_acceleration)
    assert tuple(point_signature.parameters) == (
        "source",
        "targets",
        "softening_m",
        "chunk_size",
        "backend",
    )
    for name in ("softening_m", "chunk_size", "backend"):
        assert point_signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    result = point_acceleration(PointMass(1.0, [-1.0, 0.0, 0.0]), [[0.0, 0.0, 0.0]])
    assert result.tolist() == [[-G_SI, 0.0, 0.0]]
    assert result.shape == (1, 3)
    assert not result.flags.writeable
