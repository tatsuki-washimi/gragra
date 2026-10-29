"""Contract tests for the opt-in general-point finite-cylinder observable."""

import numpy as np
import pytest

import gragra.analytic.cavity_cylinder as cylinder_module
from gragra.analytic import (
    cylinder_axis_observables_g_free,
    cylinder_observables_g_free,
)
from gragra.analytic.wave456_reference import solid_cylinder_observables_g_free
from gragra.constants import G_SI
from gragra.observables import cylinder_observables

RHO = 1800.0
RADIUS = 2.0
HEIGHT = 3.0
CENTER = np.array([0.5, -0.25, 0.3])


def test_analytic_cylinder_acceleration_gradient_against_independent_boundary():
    points = np.array([[0.2, 0.3, 0.1], [2.6, -1.05, 2.1], [3.0, 0.2, 0.0]])
    expected = _independent_boundary_reference(points)
    assert hasattr(cylinder_module, "cylinder_acceleration_gradient_g_free")
    acc, grad = cylinder_module.cylinder_acceleration_gradient_g_free(
        CENTER, RADIUS, HEIGHT, RHO, CENTER + points
    )
    for i in range(len(points)):
        assert np.linalg.norm(acc[i] - expected[1][i]) <= 1e-10 * (
            RHO * RADIUS + np.linalg.norm(expected[1][i])
        )
        assert np.linalg.norm(grad[i] - expected[2][i]) <= 1e-9 * (
            RHO + np.linalg.norm(expected[2][i])
        )
    np.testing.assert_allclose(grad, grad.swapaxes(1, 2), atol=1e-12)
    assert not acc.flags.writeable and not grad.flags.writeable


def test_analytic_near_axis_retains_radial_field_and_full_gradient():
    points = np.array([[1e-5, -2e-5, 0.7], [1e-6, 0.0, 0.0], [0.001, 0.0, 0.7]])
    expected = _independent_boundary_reference(points)
    acc, grad = cylinder_module.cylinder_acceleration_gradient_g_free(
        CENTER, RADIUS, HEIGHT, RHO, CENTER + points
    )
    for i in range(len(points)):
        np.testing.assert_allclose(
            acc[i], expected[1][i], rtol=1e-10, atol=1e-10 * RHO * RADIUS
        )
        np.testing.assert_allclose(grad[i], expected[2][i], rtol=1e-9, atol=1e-9 * RHO)
    assert np.all(acc[:, 0] != 0)


def test_analytic_rotation_and_si_use_world_components():
    import gragra.analytic as analytic
    import gragra.observables as observables

    angle = 0.71
    q = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    points = np.array([[0.2, 0.3, 0.1], [2.6, -1.05, 2.1]])
    assert hasattr(analytic, "cylinder_acceleration_gradient_g_free")
    assert hasattr(observables, "cylinder_acceleration_gradient")
    actual = analytic.cylinder_acceleration_gradient_g_free(
        CENTER, RADIUS, HEIGHT, RHO, CENTER + points @ q.T, rotation_matrix=q
    )
    sources, normals, weights = _independent_boundary_nodes(96)
    delta = (sources @ q.T)[None, :, :] - (points @ q.T)[:, None, :]
    normals = normals @ q.T
    inv = 1 / np.linalg.norm(delta, axis=2)
    expected_acc = -RHO * np.einsum("n,ni,mn->mi", weights, normals, inv)
    expected_grad = -RHO * np.einsum(
        "n,mni,nj,mn->mij", weights, delta, normals, inv**3
    )
    np.testing.assert_allclose(
        actual[0], expected_acc, rtol=1e-10, atol=1e-10 * RHO * RADIUS
    )
    np.testing.assert_allclose(actual[1], expected_grad, rtol=1e-9, atol=1e-9 * RHO)
    si = observables.cylinder_acceleration_gradient(
        CENTER, RADIUS, HEIGHT, RHO, CENTER + points @ q.T, rotation_matrix=q
    )
    for value, scaled in zip(actual, si, strict=True):
        np.testing.assert_array_equal(scaled, G_SI * value)
        assert not scaled.flags.writeable


@pytest.mark.parametrize(
    "radius,height,point",
    [
        (2.0, 3.0, [2.0, 0.0, 3.0]),
        (2.0, 3.0, [3.0, 0.0, 1.5]),
        (2.0, 3.0, [200.0, 50.0, 100.0]),
        (2.0, 0.02, [0.4, 0.2, 0.001]),
        (2.0, 200.0, [0.4, 0.2, 0.1]),
        *[
            (2.0, 3.0, [2.0 + sign * distance, 0.0, 0.4])
            for distance in (0.2, 0.02, 0.002)
            for sign in (-1, 1)
        ],
        *[
            (2.0, 3.0, [0.4, 0.0, 1.5 + sign * distance])
            for distance in (0.2, 0.02, 0.002)
            for sign in (-1, 1)
        ],
    ],
)
def test_analytic_extensions_aspects_and_boundary_approach(radius, height, point):
    pytest.importorskip("scipy")
    from oracles.cylinder_surface_reference import surface_reference

    low = surface_reference(radius, height, RHO, point, tolerance=1e-8)
    high = surface_reference(radius, height, RHO, point, tolerance=1e-11)
    actual = cylinder_module.cylinder_acceleration_gradient_g_free(
        [0.0, 0.0, 0.0], radius, height, RHO, [point]
    )
    for index, scale in enumerate((RHO * radius, RHO)):
        # Each target and observable has its own unit-scaled acceptance.
        budget = (1e-10 if index == 0 else 1e-9) * (scale + np.linalg.norm(high[index]))
        assert np.linalg.norm(low[index] - high[index]) <= budget / 10
        assert np.linalg.norm(actual[index][0] - high[index]) <= budget


@pytest.mark.parametrize(
    "q",
    [
        np.zeros((3, 3)),
        np.diag([1.0, 1.0, -1.0]),
        np.eye(3) * 1.001,
        np.full((3, 3), np.nan),
    ],
)
def test_analytic_rejects_invalid_rotation(q):
    with pytest.raises(ValueError, match="rotation_matrix"):
        cylinder_module.cylinder_acceleration_gradient_g_free(
            CENTER, RADIUS, HEIGHT, RHO, [CENTER], rotation_matrix=q
        )


@pytest.mark.parametrize("point", [[2.0, 0.0, 0.0], [0.4, 0.0, 1.5], [2.0, 0.0, 1.5]])
def test_analytic_rejects_material_boundary(point):
    with pytest.raises(ValueError, match="physical cylinder boundary"):
        cylinder_module.cylinder_acceleration_gradient_g_free(
            [0.0, 0.0, 0.0], RADIUS, HEIGHT, RHO, [point]
        )


def test_analytic_far_field_retains_point_mass_scaling():
    direction = np.array([2.0, 0.5, 1.0])
    direction /= np.linalg.norm(direction)
    distance = 1e6 * RADIUS
    point = distance * direction
    mass = RHO * np.pi * RADIUS**2 * HEIGHT
    acc, grad = cylinder_module.cylinder_acceleration_gradient_g_free(
        [0.0, 0.0, 0.0], RADIUS, HEIGHT, RHO, [point]
    )
    expected_acc = -mass * direction / distance**2
    expected_grad = (
        mass * (3 * np.outer(direction, direction) - np.eye(3)) / distance**3
    )
    np.testing.assert_allclose(acc[0], expected_acc, rtol=1e-10, atol=0)
    np.testing.assert_allclose(grad[0], expected_grad, rtol=1e-9, atol=0)


def test_analytic_axis_exact_delegation_zero_density_and_poisson_trace():
    points = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.7], [0.0, 0.0, 4.0], [0.0, 0.0, 1000.0]]
    )
    axis = cylinder_axis_observables_g_free(
        CENTER, RADIUS, HEIGHT, RHO, CENTER + points
    )
    acc, grad = cylinder_module.cylinder_acceleration_gradient_g_free(
        CENTER, RADIUS, HEIGHT, RHO, CENTER + points
    )
    np.testing.assert_array_equal(acc, axis.acceleration)
    np.testing.assert_array_equal(grad, axis.gradient)
    offaxis = np.array([[0.2, 0.3, 0.1], [2.6, -1.05, 2.1]])
    acc, grad = cylinder_module.cylinder_acceleration_gradient_g_free(
        CENTER, RADIUS, HEIGHT, RHO, CENTER + offaxis
    )
    np.testing.assert_allclose(
        np.trace(grad, axis1=1, axis2=2), [-4 * np.pi * RHO, 0.0], atol=1e-10
    )
    zero = cylinder_module.cylinder_acceleration_gradient_g_free(
        CENTER, RADIUS, HEIGHT, 0.0, CENTER + offaxis
    )
    assert all(np.count_nonzero(value) == 0 for value in zero)


def test_analytic_production_does_not_use_spatial_quadrature(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("analytic route called spatial quadrature")

    monkeypatch.setattr(cylinder_module, "_cylinder_boundary_quadrature", forbidden)
    monkeypatch.setattr(np.polynomial.legendre, "leggauss", forbidden)
    cylinder_module.cylinder_acceleration_gradient_g_free(
        CENTER,
        RADIUS,
        HEIGHT,
        RHO,
        CENTER + np.array([[0.2, 0.3, 0.1], [2.0, 0.0, 3.0]]),
    )


def test_analytic_external_volume_reference_converges():
    targets = np.array([[5.0, 3.0, 4.0]])
    actual = cylinder_module.cylinder_acceleration_gradient_g_free(
        [0.0, 0.0, 0.0], RADIUS, HEIGHT, RHO, targets
    )
    refs = [
        solid_cylinder_observables_g_free(
            np.zeros((1, 3)),
            RADIUS,
            HEIGHT,
            RHO,
            targets,
            radial_order=order,
            azimuthal_order=4 * order,
            axial_order=order,
        )
        for order in (20, 32)
    ]
    for i, name in enumerate(("acceleration", "gradient")):
        high = getattr(refs[1], name)
        low = getattr(refs[0], name)
        scale = RHO * RADIUS if i == 0 else RHO
        budget = (1e-10 if i == 0 else 1e-9) * (scale + np.linalg.norm(high))
        assert np.linalg.norm(high - low) < budget / 10
        assert np.linalg.norm(actual[i] - high) < budget


def _independent_boundary_nodes(order):
    """Build an independent cylinder boundary rule for regression checks."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    azimuthal_order = 4 * order
    azimuth = 2.0 * np.pi * np.arange(azimuthal_order) / azimuthal_order
    azimuth_weights = 2.0 * np.pi / azimuthal_order

    z = 0.5 * HEIGHT * nodes
    zz, tt = np.meshgrid(z, azimuth, indexing="ij")
    side = np.stack([RADIUS * np.cos(tt), RADIUS * np.sin(tt), zz], axis=-1).reshape(
        -1, 3
    )
    side_normals = np.stack(
        [np.cos(tt), np.sin(tt), np.zeros_like(tt)], axis=-1
    ).reshape(-1, 3)
    side_weights = np.broadcast_to(
        (0.5 * HEIGHT * weights)[:, None] * RADIUS * azimuth_weights,
        zz.shape,
    ).reshape(-1)

    radial = 0.5 * RADIUS * (nodes + 1.0)
    rr, tt = np.meshgrid(radial, azimuth, indexing="ij")
    cap = np.stack(
        [rr * np.cos(tt), rr * np.sin(tt), np.full_like(rr, HEIGHT / 2.0)],
        axis=-1,
    ).reshape(-1, 3)
    bottom = cap.copy()
    bottom[:, 2] = -HEIGHT / 2.0
    cap_normals = np.zeros_like(cap)
    cap_normals[:, 2] = 1.0
    bottom_normals = -cap_normals
    cap_weights = np.broadcast_to(
        (0.5 * RADIUS * weights * radial)[:, None] * azimuth_weights,
        rr.shape,
    ).reshape(-1)
    return (
        np.concatenate([side, cap, bottom]),
        np.concatenate([side_normals, cap_normals, bottom_normals]),
        np.concatenate([side_weights, cap_weights, cap_weights]),
    )


def _independent_boundary_reference(relative_targets, order=96):
    """Evaluate the divergence-theorem boundary reference without production code."""
    sources, normals, weights = _independent_boundary_nodes(order)
    delta = sources[None, :, :] - relative_targets[:, None, :]
    inverse_distance = 1.0 / np.linalg.norm(delta, axis=2)
    delta_dot_normal = np.einsum("mni,ni->mn", delta, normals)
    potential = (
        -0.5 * RHO * np.einsum("mn,n->m", delta_dot_normal * inverse_distance, weights)
    )
    acceleration = -RHO * np.einsum("n,ni,mn->mi", weights, normals, inverse_distance)
    gradient = -RHO * np.einsum(
        "n,mni,nj,mn->mij", weights, delta, normals, inverse_distance**3
    )
    return potential, acceleration, gradient


def test_general_cylinder_matches_independent_boundary_reference_inside_and_near_axis():
    relative_targets = np.array(
        [
            [0.2, 0.3, 0.1],
            [1.0e-5, -2.0e-5, 0.7],
            [1.0e-6, 0.0, 0.0],
            [2.6, -1.05, 2.1],
        ]
    )
    targets = CENTER + relative_targets
    expected = _independent_boundary_reference(relative_targets)
    actual = cylinder_observables_g_free(
        CENTER, RADIUS, HEIGHT, RHO, targets, quadrature_order=48
    )

    np.testing.assert_allclose(actual.potential, expected[0], rtol=2e-7, atol=1e-7)
    np.testing.assert_allclose(actual.acceleration, expected[1], rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(actual.gradient, expected[2], rtol=2e-6, atol=2e-5)


def test_independent_boundary_reference_converges_at_all_regression_targets():
    relative_targets = np.array(
        [
            [0.2, 0.3, 0.1],
            [1.0e-5, -2.0e-5, 0.7],
            [1.0e-6, 0.0, 0.0],
            [2.6, -1.05, 2.1],
        ]
    )
    low = _independent_boundary_reference(relative_targets, order=48)
    high = _independent_boundary_reference(relative_targets, order=96)
    np.testing.assert_allclose(low[0], high[0], rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(low[1], high[1], rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(low[2], high[2], rtol=1e-7, atol=1e-6)


def test_general_cylinder_matches_independent_cylindrical_reference():
    targets = np.array(
        [
            [3.1, -1.3, 2.4],
            [0.5, -0.25, 5.0],
            [3.0, 0.5, -1.0],
        ]
    )

    actual = cylinder_observables_g_free(
        CENTER,
        RADIUS,
        HEIGHT,
        RHO,
        targets,
        quadrature_order=48,
    )
    reference = solid_cylinder_observables_g_free(
        CENTER[None, :],
        RADIUS,
        HEIGHT,
        RHO,
        targets,
        radial_order=72,
        azimuthal_order=288,
        axial_order=72,
    )

    np.testing.assert_allclose(actual.potential, reference.potential, rtol=3e-4)
    np.testing.assert_allclose(
        actual.acceleration, reference.acceleration, rtol=3e-3, atol=1e-9
    )
    np.testing.assert_allclose(
        actual.gradient, reference.gradient, rtol=2e-2, atol=1e-8
    )


def test_axis_targets_use_the_existing_axis_closed_form():
    targets = CENTER + np.array([[0.0, 0.0, -2.0], [0.0, 0.0, 0.4], [0.0, 0.0, 4.0]])

    general = cylinder_observables_g_free(CENTER, RADIUS, HEIGHT, RHO, targets)
    axis = cylinder_axis_observables_g_free(CENTER, RADIUS, HEIGHT, RHO, targets)

    np.testing.assert_array_equal(general.potential, axis.potential)
    np.testing.assert_array_equal(general.acceleration, axis.acceleration)
    np.testing.assert_array_equal(general.gradient, axis.gradient)


def test_general_cylinder_gradient_trace_uses_local_density():
    targets = CENTER + np.array([[0.2, 0.3, 0.1], [3.0, 0.4, 0.2], [0.0, 0.0, 2.5]])
    values = cylinder_observables_g_free(CENTER, RADIUS, HEIGHT, RHO, targets)

    np.testing.assert_allclose(np.trace(values.gradient[0]), -4.0 * np.pi * RHO)
    np.testing.assert_allclose(
        np.trace(values.gradient[1:], axis1=1, axis2=2), 0.0, atol=1e-8 * RHO
    )


def test_general_cylinder_near_axis_is_finite_and_matches_axis_limit():
    target = CENTER + np.array([[1.0e-5, -2.0e-5, 0.7]])
    actual = cylinder_observables_g_free(
        CENTER, RADIUS, HEIGHT, RHO, target, quadrature_order=48
    )
    axis_target = CENTER + np.array([[0.0, 0.0, 0.7]])
    axis = cylinder_axis_observables_g_free(CENTER, RADIUS, HEIGHT, RHO, axis_target)

    assert np.all(np.isfinite(actual.potential))
    assert np.all(np.isfinite(actual.acceleration))
    assert np.all(np.isfinite(actual.gradient))
    np.testing.assert_allclose(actual.potential, axis.potential, rtol=1e-3)


def test_general_cylinder_small_radius_uses_the_same_axis_tolerance():
    radius = 1.0e-6
    center = np.zeros(3)
    target = np.array([[5.0e-19, -3.0e-19, 0.7]])
    actual = cylinder_observables_g_free(
        center, radius, HEIGHT, RHO, target, quadrature_order=16
    )
    axis = cylinder_axis_observables_g_free(
        center, radius, HEIGHT, RHO, np.array([[0.0, 0.0, 0.7]])
    )

    np.testing.assert_array_equal(actual.potential, axis.potential)
    np.testing.assert_array_equal(actual.acceleration, axis.acceleration)
    np.testing.assert_array_equal(actual.gradient, axis.gradient)


def test_general_cylinder_si_entrypoint_is_g_free_scaled():
    targets = CENTER + np.array([[2.4, -1.1, 0.8]])
    g_free = cylinder_observables_g_free(CENTER, RADIUS, HEIGHT, RHO, targets)
    si = cylinder_observables(CENTER, RADIUS, HEIGHT, RHO, targets)

    np.testing.assert_allclose(si.potential, G_SI * g_free.potential)
    np.testing.assert_allclose(si.acceleration, G_SI * g_free.acceleration)
    np.testing.assert_allclose(si.gradient, G_SI * g_free.gradient)


def test_general_cylinder_rejects_physical_boundary_and_bad_order():
    side = CENTER + np.array([[RADIUS, 0.0, 0.0]])
    with pytest.raises(ValueError, match="physical cylinder boundary"):
        cylinder_observables_g_free(CENTER, RADIUS, HEIGHT, RHO, side)

    with pytest.raises(ValueError, match="quadrature_order"):
        cylinder_observables_g_free(
            CENTER, RADIUS, HEIGHT, RHO, [[4.0, 0.0, 0.0]], quadrature_order=2
        )


def test_offaxis_cylinder_rejects_mixed_nested_boolean_rotation():
    with pytest.raises(TypeError, match="rotation_matrix"):
        cylinder_module.cylinder_acceleration_gradient_g_free(
            CENTER,
            RADIUS,
            HEIGHT,
            RHO,
            [[4.0, 0.0, 0.0]],
            rotation_matrix=[[True, 0, 0], [0, 1, 0], [0, 0, 1]],
        )
