"""Tests for the notebook runner's interpreter and artifact boundary."""

from __future__ import annotations

import sys

import nbformat
import pytest

from scripts.run_uniform_sphere_notebook import _read_and_validate_provenance


def _notebook_with_provenance(executable: str, package: str) -> nbformat.NotebookNode:
    notebook = nbformat.v4.new_notebook()
    cell = nbformat.v4.new_code_cell("provenance")
    cell.outputs = [
        nbformat.v4.new_output(
            "display_data",
            data={
                "python_executable": executable,
                "gragra_import_path": package,
                "python_prefix": sys.prefix,
                "python_version": "3.11.test",
            },
            metadata={},
        )
    ]
    notebook.cells = [cell]
    return notebook


def test_runner_rejects_kernel_from_another_interpreter(monkeypatch) -> None:
    import scripts.run_uniform_sphere_notebook as runner

    monkeypatch.setattr(runner.sys, "executable", "/different/python")
    notebook = _notebook_with_provenance(
        "/actual/python", str(runner.REPOSITORY / "src" / "gragra")
    )
    with pytest.raises(RuntimeError, match="kernel executable mismatch"):
        _read_and_validate_provenance(notebook)


def test_runner_rejects_import_from_another_checkout() -> None:

    notebook = _notebook_with_provenance(sys_executable(), "/other/gragra")
    with pytest.raises(RuntimeError, match="imported gragra outside"):
        _read_and_validate_provenance(notebook)


def sys_executable() -> str:
    return sys.executable
