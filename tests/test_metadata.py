"""Consistency checks for the research package metadata and license markers."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
CITATION = REPO_ROOT / "CITATION.cff"
LICENSE_FILE = REPO_ROOT / "LICENSE"
PY_TYPED = REPO_ROOT / "src" / "gragra" / "py.typed"
CODEMETA = REPO_ROOT / "codemeta.json"

_CFF_REQUIRED_FIELDS = (
    "cff-version",
    "title",
    "authors",
    "version",
    "date-released",
)


def _read_pyproject() -> dict:
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)


def _read_codemeta() -> dict:
    with CODEMETA.open(encoding="utf-8") as fh:
        return json.load(fh)


def _parse_cff_scalar(text: str, key: str) -> str | None:
    match = re.search(r"^" + re.escape(key) + r"\s*:\s*(.+)$", text, re.MULTILINE)
    return match.group(1).strip().strip('"').strip("'") if match else None


def _parse_cff_nested(text: str, key: str) -> str | None:
    match = re.search(
        r"^\s+(?:-\s+)?" + re.escape(key) + r"\s*:\s*(.+)$",
        text,
        re.MULTILINE,
    )
    return match.group(1).strip().strip('"').strip("'") if match else None


def _cff_has_field(text: str, key: str) -> bool:
    if key == "authors":
        return bool(re.search(r"^authors\s*:\s*\n(?:\s*\n)*\s*-", text, re.MULTILINE))
    return _parse_cff_scalar(text, key) is not None


def test_version_consistent() -> None:
    project_version = _read_pyproject()["project"]["version"]
    cff_version = _parse_cff_scalar(CITATION.read_text(encoding="utf-8"), "version")
    assert cff_version is not None
    assert project_version == cff_version


def test_license_exists_and_is_mit() -> None:
    assert LICENSE_FILE.is_file()
    license_value = _read_pyproject()["project"].get("license", {})
    license_text = (
        license_value.get("text", "")
        if isinstance(license_value, dict)
        else str(license_value)
    )
    assert "MIT" in license_text


def test_citation_required_fields() -> None:
    cff_text = CITATION.read_text(encoding="utf-8")
    missing = [
        field for field in _CFF_REQUIRED_FIELDS if not _cff_has_field(cff_text, field)
    ]
    assert not missing, f"CITATION.cff is missing required field(s): {missing}"


def test_py_typed_exists_and_is_packaged() -> None:
    assert PY_TYPED.is_file()
    data = _read_pyproject()
    build_backend = data.get("build-system", {}).get("build-backend", "")
    if "scikit_build_core" in build_backend:
        packages = (
            data.get("tool", {})
            .get("scikit-build", {})
            .get("wheel", {})
            .get("packages", [])
        )
        assert "src/gragra" in packages
    else:
        package_data = (
            data.get("tool", {}).get("setuptools", {}).get("package-data", {})
        )
        assert "py.typed" in package_data.get("gragra", [])


def test_codemeta_is_valid_json() -> None:
    data = _read_codemeta()
    assert data["@context"] == "https://doi.org/10.5063/schema/codemeta-2.0"
    assert data["@type"] == "SoftwareSourceCode"


def test_codemeta_matches_project_metadata() -> None:
    codemeta = _read_codemeta()
    project = _read_pyproject()["project"]
    assert codemeta["name"] == project["name"]
    assert codemeta["description"] == project["description"]
    assert codemeta["softwareRequirements"] == project["dependencies"]
    cff_text = CITATION.read_text(encoding="utf-8")
    cff_version = _parse_cff_scalar(cff_text, "version")
    cff_date = _parse_cff_scalar(cff_text, "date-released")
    assert codemeta["version"] == project["version"] == cff_version
    assert codemeta["datePublished"] == cff_date
    assert codemeta["license"] == "https://spdx.org/licenses/MIT"


def test_codemeta_author_and_repository_match_citation() -> None:
    codemeta = _read_codemeta()
    cff_text = CITATION.read_text(encoding="utf-8")
    authors = codemeta["author"]
    assert isinstance(authors, list) and authors
    assert authors[0]["familyName"] == _parse_cff_nested(cff_text, "family-names")
    assert authors[0]["givenName"] == _parse_cff_nested(cff_text, "given-names")
    assert codemeta["codeRepository"] == _parse_cff_scalar(cff_text, "url")


def test_codemeta_runtime_matches_requires_python() -> None:
    codemeta = _read_codemeta()
    requires_python = _read_pyproject()["project"]["requires-python"]
    assert codemeta["runtimePlatform"] == f"Python {requires_python}"
