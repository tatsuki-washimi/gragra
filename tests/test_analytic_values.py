"""Tests for GFreeGravityValues, GravityObservables, and SI conversion."""

import numpy as np
import pytest

from gragra.analytic._validation import (
    validate_coordinate_input,
    validate_nonnegative_scalar_input,
    validate_real_xyz_table,
    validate_real_xyz_vector,
    validate_scalar_input,
    validate_strict_nonnegative_array,
)
from gragra.analytic.values import (
    GFreeGravityValues,
    GFreeSphericalShellPComponents,
)
from gragra.analytic.wave456_reference import GFreeObservables
from gragra.constants import G_SI
from gragra.observables.analytic import (
    GravityObservables,
    SphericalShellPComponents,
    to_gravity_observables,
    to_spherical_shell_p_components,
)


def test_gfree_gravity_values_static_shapes():
    p = np.array([1.0, 2.0])
    a = np.ones((2, 3))
    g = np.ones((2, 3, 3))

    val = GFreeGravityValues(p, a, g)
    assert not val.is_series
    assert val.potential.shape == (2,)
    assert val.acceleration.shape == (2, 3)
    assert val.gradient.shape == (2, 3, 3)
    assert val.potential.dtype == np.float64
    assert not val.potential.flags.writeable
    assert not val.acceleration.flags.writeable
    assert not val.gradient.flags.writeable


def test_gfree_gravity_values_series_shapes():
    p = np.ones((5, 2))
    a = np.ones((5, 2, 3))
    g = np.ones((5, 2, 3, 3))

    val = GFreeGravityValues(p, a, g)
    assert val.is_series
    assert val.potential.shape == (5, 2)
    assert val.acceleration.shape == (5, 2, 3)
    assert val.gradient.shape == (5, 2, 3, 3)


def test_gfree_gravity_values_defensive_copy():
    p = np.array([1.0, 2.0])
    a = np.ones((2, 3))
    g = np.ones((2, 3, 3))

    val = GFreeGravityValues(p, a, g)
    p[0] = 999.0
    assert val.potential[0] == 1.0


def test_gfree_gravity_values_complex_dtype():
    p = np.array([1.0 + 2.0j])
    a = np.ones((1, 3))
    g = np.ones((1, 3, 3))

    val = GFreeGravityValues(p, a, g)
    assert val.potential.dtype == np.complex128
    assert val.acceleration.dtype == np.complex128
    assert val.gradient.dtype == np.complex128


def test_gfree_gravity_values_invalid_shapes():
    with pytest.raises(ValueError, match="potential must have 1 or 2 dimensions"):
        GFreeGravityValues(np.ones((2, 3, 4)), np.ones((2, 3, 4, 3)), np.ones((2, 3, 4, 3, 3)))

    with pytest.raises(ValueError, match="acceleration shape"):
        GFreeGravityValues(np.ones(2), np.ones((2, 4)), np.ones((2, 3, 3)))

    with pytest.raises(ValueError, match="gradient shape"):
        GFreeGravityValues(np.ones(2), np.ones((2, 3)), np.ones((2, 3, 4)))


def test_gfree_gravity_values_non_finite():
    with pytest.raises(ValueError, match="potential must contain only finite"):
        GFreeGravityValues([np.nan], np.ones((1, 3)), np.ones((1, 3, 3)))


def test_gravity_observables_basics():
    p = np.array([10.0])
    a = np.ones((1, 3))
    g = np.ones((1, 3, 3))

    obs = GravityObservables(p, a, g)
    assert not obs.is_series
    assert not obs.potential.flags.writeable
    assert not obs.acceleration.flags.writeable
    assert not obs.gradient.flags.writeable


def test_to_gravity_observables_conversion():
    p = np.array([1.0, 2.0])
    a = np.ones((2, 3)) * 3.0
    g = np.ones((2, 3, 3)) * 4.0

    gfree = GFreeGravityValues(p, a, g)
    si_obs = to_gravity_observables(gfree)

    assert isinstance(si_obs, GravityObservables)
    assert not isinstance(si_obs, GFreeGravityValues)
    np.testing.assert_allclose(si_obs.potential, G_SI * p)
    np.testing.assert_allclose(si_obs.acceleration, G_SI * a)
    np.testing.assert_allclose(si_obs.gradient, G_SI * g)


def test_to_gravity_observables_rejects_si_and_legacy():
    p = np.array([1.0])
    a = np.ones((1, 3))
    g = np.ones((1, 3, 3))

    si_obs = GravityObservables(p, a, g)
    with pytest.raises(TypeError, match="already GravityObservables"):
        to_gravity_observables(si_obs)

    legacy = GFreeObservables(p, a, g)
    with pytest.raises(TypeError, match="Expected GFreeGravityValues"):
        to_gravity_observables(legacy)

    with pytest.raises(TypeError, match="Expected GFreeGravityValues"):
        to_gravity_observables("not-a-value")

    # Reject shell components
    gfree_comp = GFreeSphericalShellPComponents(
        np.ones((1, 3)), np.ones((1, 3)), np.ones((1, 3)), np.ones((1, 3)) * 3
    )
    with pytest.raises(TypeError, match="Expected GFreeGravityValues, got GFreeSphericalShellPComponents"):
        to_gravity_observables(gfree_comp)


def test_gfree_and_si_spherical_shell_p_components():
    b = np.ones((1, 3))
    iw = np.ones((1, 3)) * 2
    ob = np.ones((1, 3)) * 3
    tot = np.ones((1, 3)) * 6

    gfree_comp = GFreeSphericalShellPComponents(bulk=b, inner_wall=iw, outer_boundary=ob, total=tot)
    assert gfree_comp.unit_system == "g_free"
    assert gfree_comp.bulk.shape == (1, 3)
    assert not gfree_comp.bulk.flags.writeable

    # SI conversion
    si_comp = to_spherical_shell_p_components(gfree_comp)
    assert isinstance(si_comp, SphericalShellPComponents)
    assert si_comp.unit_system == "si"
    np.testing.assert_allclose(si_comp.bulk, G_SI * b)
    np.testing.assert_allclose(si_comp.inner_wall, G_SI * iw)
    np.testing.assert_allclose(si_comp.outer_boundary, G_SI * ob)
    np.testing.assert_allclose(si_comp.total, G_SI * tot)
    assert not si_comp.bulk.flags.writeable

    # Rejection of invalid inputs to to_spherical_shell_p_components
    with pytest.raises(TypeError, match="already SphericalShellPComponents"):
        to_spherical_shell_p_components(si_comp)
    with pytest.raises(TypeError, match="Expected GFreeSphericalShellPComponents"):
        to_spherical_shell_p_components("not-a-comp")


def test_validation_helpers_reject_bool_and_complex():
    # Booleans must be rejected before numeric cast in strict validators
    with pytest.raises(TypeError, match="must not be boolean"):
        validate_scalar_input(True, "test_param")
    with pytest.raises(TypeError, match="must not be boolean"):
        validate_nonnegative_scalar_input(False, "test_param")
    with pytest.raises(TypeError, match="must not be boolean"):
        validate_real_xyz_vector([True, False, True], "center")
    with pytest.raises(TypeError, match="must not be boolean"):
        validate_real_xyz_table([[True, False, True]], "targets")
    with pytest.raises(TypeError, match="must not be boolean"):
        validate_strict_nonnegative_array([True, False], "k")

    # Complex coordinates/vectors must be rejected with informative error in strict validators
    with pytest.raises(TypeError, match="must be real"):
        validate_scalar_input(1.0 + 2.0j, "test_param")
    with pytest.raises(TypeError, match="must be real"):
        validate_strict_nonnegative_array(1.0 + 2.0j, "k")
    with pytest.raises(TypeError, match="must be real"):
        validate_real_xyz_vector([1.0, 2.0 + 1j, 3.0], "center")
    with pytest.raises(TypeError, match="must be real"):
        validate_real_xyz_table([[1.0, 2.0 + 1j, 3.0]], "targets")

    # Legacy permissive validators reject complex via standard numeric cast failure
    with pytest.raises(TypeError):
        validate_coordinate_input(1.0 + 2.0j, "z_m")
