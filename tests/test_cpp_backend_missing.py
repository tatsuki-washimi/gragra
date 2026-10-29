"""Tests for the C++ backend under missing/unbuilt conditions."""

import sys
import types

import numpy as np
import pytest

from gragra.coherent.weights import plane_wave_mass_weights
from gragra.gravity_sources import PointMass
from gragra.integrators import direct_point_dipole_sum
from gragra.observables import point_potential
from gragra.observables.coherent import coherent_point_acceleration


def test_cpp_backend_missing_raises_import_error(monkeypatch):
    """Verify that requesting cpp backend raises clean ImportError when C++
    extension is unbuilt.
    """
    # Hide the compiled backend from sys.modules
    monkeypatch.setitem(sys.modules, "gragra._cpp_backend", None)
    # Also invalidate the cached module if it was loaded
    if "gragra.kernels._cpp_inverse_square" in sys.modules:
        monkeypatch.delitem(sys.modules, "gragra.kernels._cpp_inverse_square")

    source = PointMass(1.0, [0.0, 0.0, 0.0])
    target = [[1.0, 0.0, 0.0]]

    with pytest.raises(ImportError) as excinfo:
        point_potential(source, target, backend="cpp")

    assert "cmake.define.GRAGRA_BUILD_CPP=ON" in str(excinfo.value)


def test_coherent_cpp_backend_missing_raises_import_error(monkeypatch):
    """The coherent path emits the same build-hint ImportError as the
    point-direct path when the C++ extension is unbuilt.
    """
    monkeypatch.setitem(sys.modules, "gragra._cpp_backend", None)
    if "gragra.kernels._cpp_parametric" in sys.modules:
        monkeypatch.delitem(sys.modules, "gragra.kernels._cpp_parametric")

    sources = np.array([[0.0, 0.0, 0.0]])
    weights = np.ones((1, 2))
    targets = np.array([[1.0, 0.0, 0.0]])

    with pytest.raises(ImportError) as excinfo:
        coherent_point_acceleration(sources, weights, targets, backend="cpp")

    assert "cmake.define.GRAGRA_BUILD_CPP=ON" in str(excinfo.value)


def test_cpp_backend_numba_independence(monkeypatch):
    """Verify that backend='cpp' can run when numba is not installed.

    This ensures no transitive dependency from the C++ backend to Numba.
    """
    # Verify C++ backend is actually compiled before running this test
    try:
        import gragra._cpp_backend  # noqa: F401
    except ImportError:
        pytest.skip("C++ backend not built; cannot verify numba independence")

    # Invalidate JIT modules and hide numba
    monkeypatch.setitem(sys.modules, "numba", None)
    if "gragra.kernels._numba_inverse_square" in sys.modules:
        monkeypatch.delitem(sys.modules, "gragra.kernels._numba_inverse_square")

    source = PointMass(1.0, [0.0, 0.0, 0.0])
    target = [[1.0, 0.0, 0.0]]

    # This should succeed even though numba is unavailable
    val = point_potential(source, target, backend="cpp")
    assert val is not None


@pytest.fixture
def mock_stale_cpp_backend():
    # Backup original modules and parent attributes to prevent interference
    original_modules = {}
    original_attrs = []

    target_modules = [
        "gragra._cpp_backend",
        "gragra.kernels._cpp_planewave",
        "gragra.kernels._cpp_parametric",
        "gragra.kernels._cpp_inverse_square",
    ]
    for mod_name in target_modules:
        if mod_name in sys.modules:
            original_modules[mod_name] = sys.modules[mod_name]
            del sys.modules[mod_name]

    parent_child_pairs = [
        ("gragra", "_cpp_backend"),
        ("gragra.kernels", "_cpp_planewave"),
        ("gragra.kernels", "_cpp_parametric"),
        ("gragra.kernels", "_cpp_inverse_square"),
    ]
    for parent_name, attr_name in parent_child_pairs:
        parent_mod = sys.modules.get(parent_name)
        if parent_mod and hasattr(parent_mod, attr_name):
            original_attrs.append(
                (parent_mod, attr_name, getattr(parent_mod, attr_name))
            )
            delattr(parent_mod, attr_name)

    # Mock _cpp_backend as a stale/incomplete module with no required symbols
    sys.modules["gragra._cpp_backend"] = types.SimpleNamespace()

    yield

    # Restore original modules and parent attributes
    for mod_name in target_modules:
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        if mod_name in original_modules:
            sys.modules[mod_name] = original_modules[mod_name]

    for parent_mod, attr_name, attr_val in original_attrs:
        setattr(parent_mod, attr_name, attr_val)


def test_stale_backend_point_potential(mock_stale_cpp_backend):  # noqa: ARG001
    with pytest.raises(ImportError) as exc_info:
        point_potential(
            PointMass(1.0, [0.0, 0.0, 0.0]),
            [[1.0, 0.0, 0.0]],
            backend="cpp",
        )

    msg = str(exc_info.value)
    assert "stale/incomplete" in msg
    assert "potential_contract_real" in msg
    assert "cmake.define.GRAGRA_BUILD_CPP=ON" in msg


def test_stale_backend_coherent_point_acceleration(
    mock_stale_cpp_backend,  # noqa: ARG001
):
    with pytest.raises(ImportError) as exc_info:
        coherent_point_acceleration(
            [[0.0, 0.0, 0.0]],
            [[1.0, 1.0]],
            [[1.0, 0.0, 0.0]],
            backend="cpp",
        )

    msg = str(exc_info.value)
    assert "stale/incomplete" in msg
    assert "acceleration_contract_parametric_real" in msg
    assert "cmake.define.GRAGRA_BUILD_CPP=ON" in msg


def test_stale_backend_plane_wave_mass_weights(
    mock_stale_cpp_backend,  # noqa: ARG001
):
    with pytest.raises(ImportError) as exc_info:
        plane_wave_mass_weights(
            [1.0],
            [[0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            [1.0],
            backend="cpp",
        )

    msg = str(exc_info.value)
    assert "stale/incomplete" in msg
    assert "plane_wave_weight_factor" in msg
    assert "cmake.define.GRAGRA_BUILD_CPP=ON" in msg


def test_stale_backend_dipole_contract(mock_stale_cpp_backend):  # noqa: ARG001
    """Phase 14-c: the dipole/total (vector-weight) path detects a stale/
    incomplete C++ backend the same way as the scalar-weight contracts.
    """
    with pytest.raises(ImportError) as exc_info:
        direct_point_dipole_sum(
            [1.0],
            [[1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            backend="cpp",
        )

    msg = str(exc_info.value)
    assert "stale/incomplete" in msg
    assert "dipole_contract_real" in msg
    assert "cmake.define.GRAGRA_BUILD_CPP=ON" in msg
