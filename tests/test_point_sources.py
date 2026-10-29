"""Tests for the representation of sources and target points."""

import numpy as np
import pytest

from gragra.gravity_sources import PointMass, PointMassCloud, WeightedPointSource
from gragra.targets.points import TargetPoints


def test_point_mass_valid():
    """Test valid PointMass instantiation."""
    pm = PointMass(mass_kg=10.0, position_m=[1.0, 2.0, 3.0])
    assert pm.mass_kg == 10.0
    assert np.allclose(pm.position_m, [1.0, 2.0, 3.0])
    assert not pm.position_m.flags.writeable


def test_point_mass_invalid():
    """Test PointMass handles invalid inputs correctly."""
    # Negative mass
    with pytest.raises(ValueError, match="positive"):
        PointMass(mass_kg=-5.0, position_m=[0, 0, 0])

    # Zero mass
    with pytest.raises(ValueError, match="positive"):
        PointMass(mass_kg=0.0, position_m=[0, 0, 0])

    # Non-finite mass
    with pytest.raises(ValueError, match="finite"):
        PointMass(mass_kg=float("nan"), position_m=[0, 0, 0])

    # Invalid numeric type
    with pytest.raises(TypeError, match="numeric"):
        PointMass(mass_kg="ten", position_m=[0, 0, 0])


def test_point_mass_conversion():
    """Test conversion of PointMass to WeightedPointSource."""
    pm = PointMass(mass_kg=1.5, position_m=[1.0, 0.0, -1.0])
    wps = pm.as_weighted_source()
    assert isinstance(wps, WeightedPointSource)
    assert wps.positions_m.shape == (1, 3)
    assert wps.weights_kg.shape == (1,)
    assert wps.weights_kg[0] == 1.5
    assert np.allclose(wps.positions_m[0], [1.0, 0.0, -1.0])


def test_point_mass_cloud_valid():
    """Test valid PointMassCloud instantiation."""
    pmc = PointMassCloud(masses_kg=[1.0, 2.0], positions_m=[[0, 0, 0], [1, 1, 1]])
    assert np.allclose(pmc.masses_kg, [1.0, 2.0])
    assert np.allclose(pmc.positions_m, [[0, 0, 0], [1, 1, 1]])
    assert not pmc.masses_kg.flags.writeable
    assert not pmc.positions_m.flags.writeable


def test_point_mass_cloud_invalid():
    """Test PointMassCloud handles invalid inputs correctly."""
    # Negative mass in cloud
    with pytest.raises(ValueError, match="positive"):
        PointMassCloud(masses_kg=[1.0, -2.0], positions_m=[[0, 0, 0], [1, 1, 1]])

    # Complex masses in cloud
    with pytest.raises(TypeError, match="real-valued"):
        PointMassCloud(masses_kg=[1.0, 2.0 + 1j], positions_m=[[0, 0, 0], [1, 1, 1]])

    # Length mismatch
    with pytest.raises(ValueError, match="match"):
        PointMassCloud(masses_kg=[1.0], positions_m=[[0, 0, 0], [1, 1, 1]])


def test_point_mass_cloud_conversion():
    """Test conversion of PointMassCloud to WeightedPointSource."""
    pmc = PointMassCloud(masses_kg=[1.0, 2.0], positions_m=[[0, 0, 0], [1, 1, 1]])
    wps = pmc.as_weighted_source()
    assert isinstance(wps, WeightedPointSource)
    assert np.allclose(wps.positions_m, pmc.positions_m)
    assert np.allclose(wps.weights_kg, pmc.masses_kg)


def test_weighted_point_source_valid_real():
    """Test WeightedPointSource with signed real weights."""
    wps = WeightedPointSource(
        positions_m=[[0, 0, 0], [1, 0, 0]], weights_kg=[-10.0, 5.0]
    )
    assert wps.weights_kg.dtype == np.float64
    assert np.allclose(wps.weights_kg, [-10.0, 5.0])
    assert not wps.positions_m.flags.writeable
    assert not wps.weights_kg.flags.writeable


def test_weighted_point_source_valid_complex():
    """Test WeightedPointSource with complex weights."""
    wps = WeightedPointSource(
        positions_m=[[0, 0, 0], [1, 0, 0]], weights_kg=[1j, 5.0 - 2j]
    )
    assert wps.weights_kg.dtype == np.complex128
    assert np.allclose(wps.weights_kg, [1j, 5.0 - 2j])


def test_weighted_point_source_invalid():
    """Test WeightedPointSource length mismatch validation."""
    with pytest.raises(ValueError, match="match"):
        WeightedPointSource(positions_m=[[0, 0, 0]], weights_kg=[1.0, 2.0])


def test_target_points_valid():
    """Test TargetPoints validation and from_single classmethod."""
    tp = TargetPoints(positions_m=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert tp.positions_m.shape == (2, 3)
    assert not tp.positions_m.flags.writeable

    # Single target point creation
    tp_single = TargetPoints.from_single([0.0, 0.0, 0.0])
    assert tp_single.positions_m.shape == (1, 3)
    assert np.allclose(tp_single.positions_m[0], [0.0, 0.0, 0.0])


def test_target_points_rejects_1d():
    """Verify TargetPoints raises ValueError for raw 1D (3,) input."""
    with pytest.raises(ValueError):
        TargetPoints(positions_m=[1.0, 0.0, 0.0])


def test_public_api_promotes_single_target():
    """Verify public API accepts array-like (3,) as a single target point.

    point_acceleration(source, [1,0,0]) must return shape (1, 3).
    """
    from gragra.observables import point_acceleration

    source = PointMass(1.0, [0.0, 0.0, 0.0])
    a = point_acceleration(source, [1.0, 0.0, 0.0])
    assert a.shape == (1, 3)


def test_stored_arrays_read_only_and_defensive_copy():
    """Verify that stored arrays are read-only and defensively copied.

    Mutating the original input array after construction must not affect
    the stored values.
    """
    # --- WeightedPointSource ---
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    weights = np.array([2.0, -3.0])
    wps = WeightedPointSource(positions_m=positions, weights_kg=weights)

    assert not wps.positions_m.flags.writeable
    assert not wps.weights_kg.flags.writeable

    # Mutate originals; stored values must be unchanged
    positions[0, 0] = 999.0
    weights[0] = 999.0
    assert wps.positions_m[0, 0] == 0.0
    assert wps.weights_kg[0] == 2.0

    with pytest.raises((ValueError, TypeError)):
        wps.positions_m[0, 0] = 42.0

    # --- TargetPoints ---
    tp_pos = np.array([[5.0, 6.0, 7.0]])
    tp = TargetPoints(positions_m=tp_pos)

    assert not tp.positions_m.flags.writeable

    tp_pos[0, 0] = 999.0
    assert tp.positions_m[0, 0] == 5.0

    with pytest.raises((ValueError, TypeError)):
        tp.positions_m[0, 0] = 42.0


def test_unsupported_source_type_raises():
    """Verify public API raises TypeError for an unsupported source type."""
    from gragra.observables import point_acceleration

    with pytest.raises(TypeError):
        point_acceleration(object(), [[1.0, 0.0, 0.0]])


def test_empty_source_numpy_returns_zeros():
    """N=0 (empty source) must return zeros, not raise range() step-zero.

    Regression: the numpy backend previously raised ValueError because the
    effective chunk size was 0. It is now max(n_sources, 1), matching the
    numba/cpp backends.
    """
    from gragra.observables import (
        point_acceleration,
        point_gravity_gradient,
        point_potential,
    )

    src = WeightedPointSource(positions_m=np.zeros((0, 3)), weights_kg=[])
    targets = [[1.0, 2.0, 3.0]]

    v = point_potential(src, targets)
    assert v.shape == (1,)
    assert np.all(v == 0.0)

    a = point_acceleration(src, targets)
    assert a.shape == (1, 3)
    assert np.all(a == 0.0)

    g = point_gravity_gradient(src, targets)
    assert g.shape == (1, 3, 3)
    assert np.all(g == 0.0)

    # M=0 (empty targets) keeps the leading axis at zero length.
    src_nonempty = WeightedPointSource(positions_m=[[0.0, 0.0, 0.0]], weights_kg=[10.0])
    assert point_potential(src_nonempty, np.zeros((0, 3))).shape == (0,)


def test_empty_source_numba_matches_numpy():
    """The numba backend must agree with numpy on empty sources (zeros)."""
    pytest.importorskip("numba")
    from gragra.observables import (
        point_acceleration,
        point_gravity_gradient,
        point_potential,
    )

    src = WeightedPointSource(positions_m=np.zeros((0, 3)), weights_kg=[])
    targets = [[1.0, 2.0, 3.0]]

    for fn, shape in (
        (point_potential, (1,)),
        (point_acceleration, (1, 3)),
        (point_gravity_gradient, (1, 3, 3)),
    ):
        out = fn(src, targets, backend="numba")
        assert out.shape == shape
        assert np.all(out == 0.0)
