"""Regression tests for the single source of truth of required C++ symbols.

These guard against drift between :mod:`gragra.kernels._cpp_required` and the
per-wrapper guards / CI sanity checks. The static checks run without the compiled
backend; the symbol-presence check is skipped when the backend is absent.
"""

import importlib.util
import pathlib

import pytest

from gragra.kernels import _cpp_required


def test_all_required_is_concatenation_without_duplicates():
    parts = (
        _cpp_required.PLANEWAVE_REQUIRED
        + _cpp_required.INVERSE_SQUARE_REQUIRED
        + _cpp_required.PARAMETRIC_REQUIRED
    )
    assert parts == _cpp_required.ALL_REQUIRED
    # 1 (planewave) + 8 (inverse-square) + 6 (parametric) = 15 unique symbols.
    assert len(_cpp_required.ALL_REQUIRED) == 15
    assert len(set(_cpp_required.ALL_REQUIRED)) == len(_cpp_required.ALL_REQUIRED)


def test_subgroups_are_disjoint():
    groups = (
        set(_cpp_required.PLANEWAVE_REQUIRED),
        set(_cpp_required.INVERSE_SQUARE_REQUIRED),
        set(_cpp_required.PARAMETRIC_REQUIRED),
    )
    union = set().union(*groups)
    assert sum(len(g) for g in groups) == len(union)


@pytest.mark.parametrize(
    ("module_name", "attr"),
    [
        ("gragra.kernels._cpp_planewave", "PLANEWAVE_REQUIRED"),
        ("gragra.kernels._cpp_inverse_square", "INVERSE_SQUARE_REQUIRED"),
        ("gragra.kernels._cpp_parametric", "PARAMETRIC_REQUIRED"),
    ],
)
def test_wrappers_reference_single_source(module_name, attr):
    # Static source check (find_spec does not execute the module), so it runs
    # even without the compiled backend, which the wrappers import at top level.
    spec = importlib.util.find_spec(module_name)
    assert spec is not None and spec.origin
    source = pathlib.Path(spec.origin).read_text(encoding="utf-8")
    assert f"from gragra.kernels._cpp_required import {attr}" in source


def test_backend_exports_all_required_when_built():
    cpp = pytest.importorskip("gragra._cpp_backend")
    missing = [s for s in _cpp_required.ALL_REQUIRED if not hasattr(cpp, s)]
    assert not missing, f"compiled backend missing symbols: {missing}"
