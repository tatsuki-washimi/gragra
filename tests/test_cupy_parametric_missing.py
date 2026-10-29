"""Tests for the coherent/parametric cupy backend under missing-cupy conditions.

Covers the validate-before-import contract for the coherent path
(``observables/coherent.py``): host-side input validation
(``softening_m``, ``chunk_size``, etc.) must precede the lazy cupy import
performed by ``_resolve_coherent_parametric_contract`` /
``_resolve_coherent_acceleration_contract``, so a malformed input raises
``ValueError`` without ever touching the ``ImportError`` hint path. This
mirrors the round-2 dipole fix in ``integrators/point_direct.py`` (see
``tests/test_cupy_dipole_missing.py`` for the reference pattern this file
is modeled on).

The ``hide_cupy`` fixture below is a local copy of the missing-cupy
simulation pattern, extended to also drop the parametric cupy wrapper
module (``gragra.kernels._cupy_parametric``) and its inverse-square
dependency (``gragra.kernels._cupy_inverse_square``) from the import
cache, so a prior test in the same pytest session cannot leave stale
cached modules that would defeat the missing-cupy simulation.
"""

import sys

import pytest

from gragra.observables.coherent import (
    coherent_point_acceleration,
    coherent_point_gravity_gradient,
    coherent_projected_acceleration,
)

_CUPY_MODULE_NAMES = (
    "gragra.kernels._cupy_parametric",
    "gragra.kernels._cupy_inverse_square",
)


@pytest.fixture
def hide_cupy(monkeypatch):
    """Simulate an environment where cupy is not installed.

    Order-independent: also drops the parametric cupy wrapper module and
    its inverse-square dependency from the import cache and from the
    ``gragra.kernels`` parent-package attribute cache, so a wrapper
    imported by an earlier test in the same session cannot leak through
    (mirrors ``tests/test_cupy_dipole_missing.py``'s ``hide_cupy``
    fixture, extended for the parametric backend modules).
    """
    monkeypatch.setitem(sys.modules, "cupy", None)
    kernels_pkg = sys.modules.get("gragra.kernels")
    for mod_name in _CUPY_MODULE_NAMES:
        monkeypatch.delitem(sys.modules, mod_name, raising=False)
        attr = mod_name.rsplit(".", 1)[-1]
        if kernels_pkg is not None and hasattr(kernels_pkg, attr):
            monkeypatch.delattr(kernels_pkg, attr)


# Small, physically-meaningless-but-shape-valid inputs shared by all cases.
_SOURCES_XYZ = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
_WEIGHTS_KG = [1.0, 2.0, 3.0]
_TARGETS_M = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
_DIRECTION = [1.0, 0.0, 0.0]


def test_coherent_point_acceleration_negative_softening_raises_value_error(
    hide_cupy,  # noqa: ARG001
):
    """softening_m<0 must raise ValueError, not ImportError, with cupy hidden."""
    with pytest.raises(ValueError, match="softening_m") as excinfo:
        coherent_point_acceleration(
            _SOURCES_XYZ,
            _WEIGHTS_KG,
            _TARGETS_M,
            softening_m=-1.0,
            backend="cupy",
        )
    assert not isinstance(excinfo.value, ImportError)


def test_coherent_point_acceleration_bad_chunk_size_raises_value_error(
    hide_cupy,  # noqa: ARG001
):
    """chunk_size=0 must raise ValueError, not ImportError, with cupy hidden."""
    with pytest.raises(ValueError, match="chunk_size") as excinfo:
        coherent_point_acceleration(
            _SOURCES_XYZ,
            _WEIGHTS_KG,
            _TARGETS_M,
            chunk_size=0,
            backend="cupy",
        )
    assert not isinstance(excinfo.value, ImportError)


def test_coherent_point_acceleration_missing_cupy_raises_import_error(
    hide_cupy,  # noqa: ARG001
):
    """Valid inputs with cupy hidden reach the hinted ImportError."""
    with pytest.raises(ImportError) as excinfo:
        coherent_point_acceleration(
            _SOURCES_XYZ,
            _WEIGHTS_KG,
            _TARGETS_M,
            backend="cupy",
        )
    assert "cupy is not installed" in str(excinfo.value)


def test_coherent_point_gravity_gradient_negative_softening_raises_value_error(
    hide_cupy,  # noqa: ARG001
):
    """softening_m<0 must raise ValueError, not ImportError, with cupy hidden."""
    with pytest.raises(ValueError, match="softening_m") as excinfo:
        coherent_point_gravity_gradient(
            _SOURCES_XYZ,
            _WEIGHTS_KG,
            _TARGETS_M,
            softening_m=-1.0,
            backend="cupy",
        )
    assert not isinstance(excinfo.value, ImportError)


def test_coherent_point_gravity_gradient_bad_chunk_size_raises_value_error(
    hide_cupy,  # noqa: ARG001
):
    """chunk_size=0 must raise ValueError, not ImportError, with cupy hidden."""
    with pytest.raises(ValueError, match="chunk_size") as excinfo:
        coherent_point_gravity_gradient(
            _SOURCES_XYZ,
            _WEIGHTS_KG,
            _TARGETS_M,
            chunk_size=0,
            backend="cupy",
        )
    assert not isinstance(excinfo.value, ImportError)


def test_coherent_point_gravity_gradient_missing_cupy_raises_import_error(
    hide_cupy,  # noqa: ARG001
):
    """Valid inputs with cupy hidden reach the hinted ImportError."""
    with pytest.raises(ImportError) as excinfo:
        coherent_point_gravity_gradient(
            _SOURCES_XYZ,
            _WEIGHTS_KG,
            _TARGETS_M,
            backend="cupy",
        )
    assert "cupy is not installed" in str(excinfo.value)


def test_coherent_projected_acceleration_negative_softening_raises_value_error(
    hide_cupy,  # noqa: ARG001
):
    """Fix propagates through coherent_projected_acceleration's internal call

    to coherent_point_acceleration: softening_m<0 must raise ValueError, not
    ImportError, with cupy hidden.
    """
    with pytest.raises(ValueError, match="softening_m") as excinfo:
        coherent_projected_acceleration(
            _SOURCES_XYZ,
            _WEIGHTS_KG,
            _TARGETS_M,
            _DIRECTION,
            softening_m=-1.0,
            backend="cupy",
        )
    assert not isinstance(excinfo.value, ImportError)
