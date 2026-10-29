"""Tests for gravitational field observables."""

import numpy as np
import pytest

from gragra.constants import G_SI
from gragra.gravity_sources import PointMass, PointMassCloud, WeightedPointSource
from gragra.observables import (
    point_acceleration,
    point_gravity_gradient,
    point_potential,
)
from gragra.observables.projection import (
    point_projected_acceleration,
    project_vectors,
)


def test_tensor_sign_convention():
    """Verify the sign convention of the gravity gradient tensor.

    For a 1kg point mass at the origin, and target at (1, 0, 0):
    T = G_SI * diag(2, -1, -1).
    """
    source = PointMass(1.0, [0.0, 0.0, 0.0])
    target = [[1.0, 0.0, 0.0]]

    t_tensor = point_gravity_gradient(source, target, softening_m=0.0)

    assert t_tensor.shape == (1, 3, 3)
    expected = G_SI * np.diag([2.0, -1.0, -1.0])
    np.testing.assert_allclose(t_tensor[0], expected, rtol=1e-14, atol=0.0)


def test_trace_free_gradient():
    """Verify that tr(T) = 0 for softening = 0, but tr(T) != 0 for softening > 0."""
    source = PointMass(10.0, [0.1, -0.2, 0.3])
    target = [[2.0, 1.5, -1.0]]

    # With zero softening, trace is zero (within numerical precision)
    t_zero = point_gravity_gradient(source, target, softening_m=0.0)
    trace_zero = np.trace(t_zero[0])
    # The analytical trace is zero. Allow only roundoff relative to the tensor
    # scale, rather than NumPy's default absolute tolerance (much larger than G).
    roundoff = 32 * np.finfo(float).eps * np.max(np.abs(t_zero[0]))
    assert abs(trace_zero) <= roundoff

    # With softening > 0, trace is NOT zero
    t_soft = point_gravity_gradient(source, target, softening_m=0.5)
    trace_soft = np.trace(t_soft[0])
    displacement = np.array([2.0, 1.5, -1.0]) - np.array([0.1, -0.2, 0.3])
    distance_sq = np.dot(displacement, displacement)
    softening_sq = 0.5**2
    expected_trace = (
        -3 * G_SI * 10.0 * softening_sq / (distance_sq + softening_sq) ** 2.5
    )
    np.testing.assert_allclose(trace_soft, expected_trace, rtol=1e-13, atol=0.0)
    assert np.isfinite(trace_soft)


def test_projection_observables():
    """Verify projected acceleration and raw dot product behaviors."""
    source = PointMass(5.0, [0.0, 0.0, 0.0])
    target = [[3.0, 4.0, 0.0]]
    direction = [1.0, 1.0, 0.0]

    # Expected: point_projected_acceleration = dot(acceleration, unit_direction)
    acc = point_acceleration(source, target)[0]
    unit_dir = np.array(direction) / np.linalg.norm(direction)
    expected_proj = np.dot(acc, unit_dir)

    proj_acc = point_projected_acceleration(source, target, direction)
    assert proj_acc.shape == (1,)
    np.testing.assert_allclose(proj_acc[0], expected_proj, rtol=1e-13, atol=0.0)

    # Verify project_vectors is a raw dot product (without normalization of direction)
    vecs = np.array([[2.0, 3.0, 4.0]])
    raw_proj = project_vectors(vecs, [2.0, 0.0, 0.0])
    assert raw_proj.shape == (1,)
    # expected: 2 * 2 + 3 * 0 + 4 * 0 = 4.0 (if normalized it would be 2.0)
    np.testing.assert_allclose(raw_proj[0], 4.0, rtol=1e-14, atol=0.0)

    # Zero/non-finite direction raises ValueError
    with pytest.raises(ValueError, match="zero"):
        point_projected_acceleration(source, target, [0, 0, 0])

    with pytest.raises(ValueError, match="finite"):
        point_projected_acceleration(source, target, [1.0, np.nan, 0.0])


def test_acceleration_analytic_value():
    """Verify point_acceleration against analytic inverse-square law.

    For a 1 kg mass at the origin, target at (1, 0, 0):
      a = G_SI * 1 / 1^2 toward origin = G_SI * (-1, 0, 0)
    """
    source = PointMass(1.0, [0.0, 0.0, 0.0])
    a = point_acceleration(source, [[1.0, 0.0, 0.0]])
    assert a.shape == (1, 3)
    np.testing.assert_allclose(a[0, 0], -G_SI, rtol=1e-10, atol=0.0)
    # On the source-target x-axis, the transverse components are structural zeros.
    np.testing.assert_array_equal(a[0, 1:], [0.0, 0.0])


def test_potential_analytic_value():
    """Verify point_potential analytic value.

    For a 1 kg mass at origin, target at (1, 0, 0):
      Phi = -G_SI * 1 / 1 = -G_SI
    """
    source = PointMass(1.0, [0.0, 0.0, 0.0])
    phi = point_potential(source, [[1.0, 0.0, 0.0]])
    assert phi.shape == (1,)
    np.testing.assert_allclose(phi[0], -G_SI, rtol=1e-10, atol=0.0)


def test_inverse_square_scaling():
    """Verify that |a| scales as 1/r^2 (inverse-square law)."""
    source = PointMass(1.0, [0.0, 0.0, 0.0])
    a_r1 = point_acceleration(source, [[1.0, 0.0, 0.0]])
    a_r2 = point_acceleration(source, [[2.0, 0.0, 0.0]])
    ratio = np.linalg.norm(a_r1[0]) / np.linalg.norm(a_r2[0])
    # This is a dimensionless inverse-square scaling check, not a G-scale value.
    np.testing.assert_allclose(ratio, 4.0, rtol=1e-10, atol=0.0)


def test_superposition_symmetric_sources():
    """Verify that two symmetric sources cancel transverse components.

    Two equal masses at (0, +d, 0) and (0, -d, 0), target at (r, 0, 0):
    y-components cancel; x-component doubles.
    """
    d = 2.0
    r = 5.0
    source = PointMassCloud(
        masses_kg=[1.0, 1.0],
        positions_m=[[0.0, d, 0.0], [0.0, -d, 0.0]],
    )
    a = point_acceleration(source, [[r, 0.0, 0.0]])
    assert a.shape == (1, 3)
    # Symmetry cancels y; bound only floating-point accumulation relative to
    # the field scale. The z component is an exact geometric zero.
    symmetry_roundoff = 32 * np.finfo(float).eps * np.max(np.abs(a[0]))
    assert abs(a[0, 1]) <= symmetry_roundoff
    assert a[0, 2] == 0.0


def test_point_mass_cloud_equals_sum():
    """Verify PointMassCloud result equals the sum of individual PointMass results."""
    m1, m2 = 3.0, 7.0
    p1, p2 = [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]
    target = [[0.5, 1.0, 3.0]]

    cloud = PointMassCloud(masses_kg=[m1, m2], positions_m=[p1, p2])
    a_cloud = point_acceleration(cloud, target)

    a1 = point_acceleration(PointMass(m1, p1), target)
    a2 = point_acceleration(PointMass(m2, p2), target)
    # Cross-path superposition/parity check, independent of the analytic oracle.
    np.testing.assert_allclose(a_cloud, a1 + a2, rtol=1e-12, atol=0.0)


def test_weighted_source_sign_flip():
    """Verify the physical direction and weight-sign response of acceleration."""
    source_pos = WeightedPointSource(positions_m=[[1.0, 0.0, 0.0]], weights_kg=[5.0])
    source_neg = WeightedPointSource(positions_m=[[1.0, 0.0, 0.0]], weights_kg=[-5.0])
    target = [[3.0, 0.0, 0.0]]
    a_pos = point_acceleration(source_pos, target)
    a_neg = point_acceleration(source_neg, target)
    expected_magnitude = G_SI * 5.0 / 2.0**2
    np.testing.assert_allclose(a_pos[0, 0], -expected_magnitude, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(a_neg[0, 0], expected_magnitude, rtol=1e-12, atol=0.0)
    # This remains a cross-path consistency check; the direction checks above
    # supply the independent physical oracle.
    np.testing.assert_allclose(a_pos, -a_neg, rtol=1e-12, atol=0.0)


def test_weighted_source_complex_output_dtype():
    """Verify that complex weights produce complex128 output."""
    source = WeightedPointSource(positions_m=[[0.0, 0.0, 0.0]], weights_kg=[1.0 + 2.0j])
    a = point_acceleration(source, [[1.0, 0.0, 0.0]])
    assert a.dtype == np.complex128


def test_projected_acceleration_direction_normalization():
    """Verify point_projected_acceleration normalizes direction internally.

    direction=[1,0,0] and direction=[2,0,0] must give the same result.
    """
    source = PointMass(1.0, [0.0, 0.0, 0.0])
    target = [[3.0, 4.0, 0.0]]
    proj_unit = point_projected_acceleration(source, target, [1.0, 0.0, 0.0])
    proj_double = point_projected_acceleration(source, target, [2.0, 0.0, 0.0])
    np.testing.assert_allclose(proj_unit, proj_double, rtol=1e-12, atol=0.0)


def test_project_vectors_raw_dot_not_normalized():
    """Verify project_vectors is a raw dot product (no direction normalization).

    direction=[2,0,0] should give twice the result of direction=[1,0,0].
    """
    vecs = np.array([[3.0, 4.0, 5.0]])
    raw_x1 = project_vectors(vecs, [1.0, 0.0, 0.0])
    raw_x2 = project_vectors(vecs, [2.0, 0.0, 0.0])
    np.testing.assert_allclose(raw_x2, 2.0 * raw_x1, rtol=1e-12, atol=0.0)


def test_project_vectors_wrong_component_axis():
    """Verify project_vectors raises ValueError when the last axis is not size 3."""
    vecs = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError):
        project_vectors(vecs, [1.0, 0.0, 0.0])


def test_numerical_derivatives():
    """Verify that a ≈ -grad(Phi) and T ≈ grad(a) via finite differences.

    This uses a target far enough from the source to prevent near-singularity.
    """
    source = PointMass(1.0, [0.0, 0.0, 0.0])
    target_pos = np.array([4.0, 3.0, 12.0])  # r = 13.0

    # Calculate analytical acceleration and gradient at target
    a_ana = point_acceleration(source, [target_pos])[0]
    t_ana = point_gravity_gradient(source, [target_pos])[0]

    # Finite difference step
    h = 1e-6

    # 1. Verify a ≈ -grad(Phi)
    grad_phi = np.zeros(3)
    for i in range(3):
        h_vec = np.zeros(3)
        h_vec[i] = h
        phi_plus = point_potential(source, [target_pos + h_vec])[0]
        phi_minus = point_potential(source, [target_pos - h_vec])[0]
        grad_phi[i] = (phi_plus - phi_minus) / (2 * h)

    # Expected: a = -grad_phi
    np.testing.assert_allclose(a_ana, -grad_phi, rtol=1e-5, atol=0.0)

    # 2. Verify T_ij ≈ ∂a_i / ∂x_j
    # Note: gradient is with respect to target coordinates
    grad_a = np.zeros((3, 3))
    for j in range(3):
        h_vec = np.zeros(3)
        h_vec[j] = h
        a_plus = point_acceleration(source, [target_pos + h_vec])[0]
        a_minus = point_acceleration(source, [target_pos - h_vec])[0]
        grad_a[:, j] = (a_plus - a_minus) / (2 * h)

    np.testing.assert_allclose(t_ana, grad_a, rtol=1e-5, atol=0.0)


def test_fixed_point_assertions_reject_zero_and_sign_flip_mutants(monkeypatch):
    """Keep the physical G-scale oracles sensitive to zero and sign mutations."""
    cases = (
        ("point_potential", test_potential_analytic_value, "zero"),
        ("point_potential", test_potential_analytic_value, "sign_flip"),
        ("point_acceleration", test_acceleration_analytic_value, "zero"),
        ("point_acceleration", test_acceleration_analytic_value, "sign_flip"),
        ("point_gravity_gradient", test_tensor_sign_convention, "zero"),
        ("point_gravity_gradient", test_tensor_sign_convention, "sign_flip"),
        ("point_acceleration", test_weighted_source_sign_flip, "zero"),
        ("point_acceleration", test_weighted_source_sign_flip, "sign_flip"),
    )
    for name, physical_test, mutation in cases:
        original = globals()[name]

        def mutated(*args, _original=original, _mutation=mutation, **kwargs):
            value = _original(*args, **kwargs)
            if _mutation == "zero":
                return np.zeros_like(value)
            return -value

        with monkeypatch.context() as patcher:
            patcher.setitem(globals(), name, mutated)
            with pytest.raises(AssertionError):
                physical_test()


def test_observable_returns_read_only():
    """Verify that all public observable return arrays are read-only."""
    source = PointMass(1.0, [0.0, 0.0, 0.0])
    target = [[1.0, 0.0, 0.0]]
    direction = [1.0, 0.0, 0.0]

    phi = point_potential(source, target)
    assert phi.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        phi[0] = 1.0

    a = point_acceleration(source, target)
    assert a.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        a[0, 0] = 1.0

    proj = point_projected_acceleration(source, target, direction)
    assert proj.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        proj[0] = 1.0

    t = point_gravity_gradient(source, target)
    assert t.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        t[0, 0, 0] = 1.0
