"""Tests simulating an environment without Numba.

Ensures that using backend='numba' raises a clean ImportError when
Numba is missing.
"""

import sys

import numpy as np
import pytest

from gragra.gravity_sources import PointMass
from gragra.integrators import direct_point_dipole_sum
from gragra.integrators.point_direct import direct_point_displacement_sum
from gragra.observables import point_potential
from gragra.observables.coherent import coherent_point_acceleration


def test_numba_backend_missing_raises_importerror(monkeypatch):
    """Verify that backend='numba' raises ImportError when numba is missing."""
    # Mock sys.modules to simulate numba and our JIT kernels are missing
    monkeypatch.setitem(sys.modules, "numba", None)
    monkeypatch.setitem(sys.modules, "gragra.kernels._numba_inverse_square", None)

    source = PointMass(1.0, [0.0, 0.0, 0.0])
    target = [[1.0, 0.0, 0.0]]

    # This should trigger the lazy-import check in direct_point_sum
    with pytest.raises(
        ImportError,
        match="numba is not installed in the current environment",
    ):
        point_potential(source, target, backend="numba")


def test_coherent_numba_backend_missing_raises_importerror(monkeypatch):
    """The coherent path emits the same ImportError as the point-direct path
    when numba is unavailable (consistent backend-availability contract).
    """
    monkeypatch.setitem(sys.modules, "numba", None)
    monkeypatch.setitem(sys.modules, "gragra.kernels._numba_parametric", None)

    sources = np.array([[0.0, 0.0, 0.0]])
    weights = np.ones((1, 2))
    targets = np.array([[1.0, 0.0, 0.0]])

    with pytest.raises(
        ImportError,
        match="numba is not installed in the current environment",
    ):
        coherent_point_acceleration(sources, weights, targets, backend="numba")


def test_numba_dipole_backend_missing_raises_importerror(monkeypatch):
    """The dipole/total (vector-weight) path emits the same ImportError as
    the scalar-weight point-direct path when numba is unavailable (Phase
    14-c, consistent backend-availability contract).
    """
    monkeypatch.setitem(sys.modules, "numba", None)
    monkeypatch.setitem(sys.modules, "gragra.kernels._numba_inverse_square", None)

    with pytest.raises(
        ImportError,
        match="numba is not installed in the current environment",
    ):
        direct_point_dipole_sum(
            [1.0],
            [[1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            backend="numba",
        )


@pytest.mark.parametrize("observable", ["potential", "gradient"])
def test_numba_displacement_backend_missing_raises_importerror(monkeypatch, observable):
    """The new displacement derivatives retain the lazy dependency contract."""
    monkeypatch.setitem(sys.modules, "numba", None)
    monkeypatch.setitem(sys.modules, "gragra.kernels._numba_inverse_square", None)

    with pytest.raises(
        ImportError,
        match="numba is not installed in the current environment",
    ):
        direct_point_displacement_sum(
            [1.0],
            [[1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            observable=observable,
            backend="numba",
        )


@pytest.mark.parametrize("observable", ["potential", "gradient"])
def test_numba_displacement_invalid_inputs_are_checked_before_import(
    monkeypatch, observable
):
    """Malformed new-observable inputs must not be masked by missing Numba."""
    monkeypatch.setitem(sys.modules, "numba", None)
    monkeypatch.setitem(sys.modules, "gragra.kernels._numba_inverse_square", None)

    with pytest.raises(ValueError, match="mass_kg must be non-negative"):
        direct_point_displacement_sum(
            [-1.0],
            [[1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            observable=observable,
            backend="numba",
        )
