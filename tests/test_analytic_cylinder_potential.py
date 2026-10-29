"""Analytic potential against independent Newton boundary/volume references."""

import numpy as np
import pytest
from oracles.cylinder_axis_potential_check import run_isolated_reference_case
from test_analytic_cylinder_offaxis import _independent_boundary_nodes


def _axis_volume_reference(radius, height, density, z, *, dps=100):
    """Normalized Newton volume integral with radial integration done exactly."""
    import mpmath as mp

    with mp.workdps(dps):
        radius, height, density, z = map(mp.mpf, (radius, height, density, z))
        distance = abs(z)
        sign = mp.sign(z)
        radial_ratio = radius / distance
        height_ratio = height / distance

        def axial(w):
            return 1 - height_ratio * w

        def radial_distance(w):
            return mp.sqrt(axial(w) ** 2 + radial_ratio**2)

        scale = 2 * mp.pi * density * radius**2 * height
        potential = (
            -scale
            / distance
            * mp.quad(
                lambda w: 1 / (radial_distance(w) + axial(w)),
                [-mp.mpf("0.5"), mp.mpf("0.5")],
            )
        )
        acceleration = (
            -sign
            * scale
            / distance**2
            * mp.quad(
                lambda w: 1 / (radial_distance(w) * (radial_distance(w) + axial(w))),
                [-mp.mpf("0.5"), mp.mpf("0.5")],
            )
        )
        gradient_zz = (
            scale
            / distance**3
            * mp.quad(
                lambda w: 1 / (axial(w) ** 2 + radial_ratio**2) ** mp.mpf("1.5"),
                [-mp.mpf("0.5"), mp.mpf("0.5")],
            )
        )
        return tuple(float(value) for value in (potential, acceleration, gradient_zz))


def _boundary_potential(targets, order):
    points, normals, weights = _independent_boundary_nodes(order)
    result = []
    for target in targets:
        delta = points - target
        result.append(
            -900
            * np.sum(
                weights
                * np.sum(delta * normals, axis=1)
                / np.linalg.norm(delta, axis=1)
            )
        )
    return np.asarray(result)


def test_analytic_potential_matches_independent_boundary():
    from gragra.analytic.cylinder_potential import cylinder_analytic_observables_g_free

    points = np.array([[0.2, 0.3, 0.1], [2.6, -1.05, 2.1], [0.3, 0.2, -2.5]])
    low = _boundary_potential(points, 64)
    high = _boundary_potential(points, 96)
    scale = 1800 * 2**2
    np.testing.assert_allclose(low, high, atol=1e-11 * scale, rtol=0)
    actual = cylinder_analytic_observables_g_free([0, 0, 0], 2, 3, 1800, points)
    np.testing.assert_allclose(actual.potential, high, atol=1e-10 * scale, rtol=0)


@pytest.mark.parametrize("point", [[2e6, 1e6, -3e6], [0, 0, 4e6]])
def test_far_potential_has_infinity_gauge_without_cancellation(point):
    from gragra.analytic.cylinder_potential import cylinder_analytic_observables_g_free

    actual = cylinder_analytic_observables_g_free([0, 0, 0], 2, 3, 1800, [point])
    monopole = -1800 * np.pi * 2**2 * 3 / np.linalg.norm(point)
    np.testing.assert_allclose(actual.potential[0], monopole, rtol=1e-10, atol=0)


@pytest.mark.parametrize("z", [1e100, -1e100, 9e307, -9e307])
def test_extreme_axis_observables_match_independent_normalized_volume_integral(z):
    from gragra.analytic import (
        cylinder_axis_observables_g_free,
        cylinder_filled_axis_field,
        cylinder_gradient_axis_g_free,
        cylinder_potential_axis_g_free,
    )

    expected_potential, expected_acceleration, expected_gradient = (
        _axis_volume_reference(1, 1, 1800, z)
    )
    targets = [[0, 0, z]]
    values = cylinder_axis_observables_g_free([0, 0, 0], 1, 1, 1800, targets)
    potential = cylinder_potential_axis_g_free([0, 0, 0], 1, 1, 1800, targets)
    gradient = cylinder_gradient_axis_g_free([0, 0, 0], 1, 1, 1800, targets)
    acceleration = cylinder_filled_axis_field(1, 1, z, 1800)

    np.testing.assert_allclose(
        values.potential[0], expected_potential, rtol=2e-14, atol=0
    )
    np.testing.assert_allclose(potential[0], expected_potential, rtol=2e-14, atol=0)
    np.testing.assert_allclose(
        values.acceleration[0, 2], expected_acceleration, rtol=2e-14, atol=0
    )
    np.testing.assert_allclose(acceleration, expected_acceleration, rtol=2e-14, atol=0)
    np.testing.assert_allclose(
        values.gradient[0, 2, 2], expected_gradient, rtol=2e-14, atol=0
    )
    np.testing.assert_allclose(gradient[0, 2, 2], expected_gradient, rtol=2e-14, atol=0)
    for result in (
        values.potential,
        values.acceleration,
        values.gradient,
        potential,
        gradient,
    ):
        assert np.all(np.isfinite(result))


def test_extreme_off_axis_analytic_aggregate_retains_representable_potential():
    from gragra.analytic import cylinder_observables_g_free
    from gragra.analytic.cylinder_potential import cylinder_analytic_observables_g_free

    target = [[1, 0, 9e307]]
    with np.errstate(over="raise", invalid="raise"):
        results = [
            cylinder_analytic_observables_g_free([0, 0, 0], 1, 1, 1800, target),
            cylinder_observables_g_free([0, 0, 0], 1, 1, 1800, target),
        ]
    expected_monopole = -np.pi * 1800 / 9e307
    for result in results:
        assert np.isfinite(result.potential[0])
        np.testing.assert_allclose(
            result.potential[0], expected_monopole, rtol=2e-14, atol=0
        )
        assert np.all(np.isfinite(result.acceleration))
        assert np.all(np.isfinite(result.gradient))


def test_largest_finite_off_axis_coordinates_do_not_overflow_boundary_checks():
    from gragra.analytic import (
        cylinder_acceleration_gradient_g_free,
        cylinder_observables_g_free,
    )
    from gragra.analytic.cylinder_potential import cylinder_analytic_observables_g_free

    target = [[1.7e308, 1.7e308, 1.7e308]]
    expected_monopole = -1800 * np.pi / 1.7e308 / np.sqrt(3)
    with np.errstate(over="raise", invalid="raise"):
        analytic = cylinder_analytic_observables_g_free([0, 0, 0], 1, 1, 1800, target)
        numeric = cylinder_observables_g_free([0, 0, 0], 1, 1, 1800, target)
        acceleration, gradient = cylinder_acceleration_gradient_g_free(
            [0, 0, 0], 1, 1, 1800, target
        )

    for result in (analytic, numeric):
        np.testing.assert_allclose(
            result.potential[0], expected_monopole, rtol=2e-14, atol=0
        )
        assert np.all(np.isfinite(result.acceleration))
        assert np.all(np.isfinite(result.gradient))
    assert np.all(np.isfinite(acceleration))
    assert np.all(np.isfinite(gradient))


def test_axis_center_target_difference_can_exceed_float64_range():
    from gragra.analytic import (
        cylinder_axis_observables_g_free,
        cylinder_gradient_axis_g_free,
        cylinder_potential_axis_g_free,
    )

    center = [0, 0, -9e307]
    target = [[0, 0, 9e307]]
    expected_monopole = -np.pi * 1800 / 9e307 / 2
    with np.errstate(over="raise", invalid="raise"):
        result = cylinder_axis_observables_g_free(center, 1, 1, 1800, target)
        potential = cylinder_potential_axis_g_free(center, 1, 1, 1800, target)
        gradient = cylinder_gradient_axis_g_free(center, 1, 1, 1800, target)

    np.testing.assert_allclose(
        result.potential[0], expected_monopole, rtol=2e-14, atol=0
    )
    np.testing.assert_allclose(potential[0], expected_monopole, rtol=2e-14, atol=0)
    assert np.all(np.isfinite(result.acceleration))
    assert np.all(np.isfinite(result.gradient))
    assert np.all(np.isfinite(gradient))


def test_far_target_on_cap_plane_is_not_boundary_when_radially_outside():
    from gragra.analytic import (
        cylinder_acceleration_gradient_g_free,
        cylinder_observables_g_free,
    )
    from gragra.analytic.cylinder_potential import cylinder_analytic_observables_g_free

    center = [-9e307, 0, 0]
    target = [[9e307, 0, 0.5]]
    expected_monopole = -np.pi * 1800 / 9e307 / 2
    with np.errstate(over="raise", invalid="raise"):
        analytic = cylinder_analytic_observables_g_free(center, 1, 1, 1800, target)
        numeric = cylinder_observables_g_free(center, 1, 1, 1800, target)
        acceleration, gradient = cylinder_acceleration_gradient_g_free(
            center, 1, 1, 1800, target
        )

    for result in (analytic, numeric):
        np.testing.assert_allclose(
            result.potential[0], expected_monopole, rtol=2e-14, atol=0
        )
        assert np.all(np.isfinite(result.acceleration))
        assert np.all(np.isfinite(result.gradient))
    assert np.all(np.isfinite(acceleration))
    assert np.all(np.isfinite(gradient))


def test_scaled_fallback_preserves_finite_small_component_difference():
    import mpmath as mp

    from gragra.analytic.cylinder_potential import cylinder_analytic_observables_g_free

    center = [9e307, 0, -9e307]
    adjacent_x = np.nextafter(center[0], np.inf)
    target = [[adjacent_x, 0, 9e307]]
    radius = height = 1e300
    density = 1e-308
    with mp.workdps(100):
        dx = mp.mpf(float(adjacent_x)) - mp.mpf(float(center[0]))
        dz = mp.mpf(float(target[0][2])) - mp.mpf(float(center[2]))
        distance = mp.sqrt(dx**2 + dz**2)
        mass = mp.pi * mp.mpf(density) * mp.mpf(radius) ** 2 * mp.mpf(height)
        expected_ax = float(-mass * dx / distance**3)

    with np.errstate(over="raise", invalid="raise"):
        result = cylinder_analytic_observables_g_free(
            center, radius, height, density, target
        )

    assert np.isfinite(result.acceleration[0, 0])
    np.testing.assert_allclose(
        result.acceleration[0, 0], expected_ax, rtol=2e-13, atol=0
    )


def test_rotated_extreme_target_can_have_unrepresentable_local_component():
    from gragra.analytic.cylinder_potential import cylinder_analytic_observables_g_free

    root2 = np.sqrt(2.0)
    root6 = np.sqrt(6.0)
    root3 = np.sqrt(3.0)
    rotation = np.array(
        [
            [1 / root2, 1 / root6, 1 / root3],
            [-1 / root2, 1 / root6, 1 / root3],
            [0.0, -2 / root6, 1 / root3],
        ]
    )
    target = [[1.7e308, 1.7e308, 1.7e308]]
    expected_monopole = -1800 * np.pi / 1.7e308 / root3
    with np.errstate(over="raise", invalid="raise"):
        result = cylinder_analytic_observables_g_free(
            [0, 0, 0], 1, 1, 1800, target, rotation_matrix=rotation
        )

    np.testing.assert_allclose(
        result.potential[0], expected_monopole, rtol=2e-14, atol=0
    )
    assert np.all(np.isfinite(result.acceleration))
    assert np.all(np.isfinite(result.gradient))


def test_far_field_log_distance_sweep_stays_finite_and_symmetric():
    from gragra.analytic import cylinder_axis_observables_g_free

    distances = [1e3, 1e6, 1e100, 1e154, 9e307]
    targets = [[0, 0, sign * distance] for distance in distances for sign in (1, -1)]
    with np.errstate(over="raise", invalid="raise"):
        result = cylinder_axis_observables_g_free([0, 0, 0], 1, 1, 1800, targets)

    assert np.all(np.isfinite(result.potential))
    assert np.all(np.isfinite(result.acceleration))
    assert np.all(np.isfinite(result.gradient))
    assert np.all(result.potential < 0)
    np.testing.assert_allclose(
        result.potential[0::2], result.potential[1::2], rtol=2e-14
    )
    np.testing.assert_allclose(
        result.acceleration[0::2, 2], -result.acceleration[1::2, 2], rtol=2e-14, atol=0
    )
    np.testing.assert_allclose(
        result.gradient[0::2, 2, 2], result.gradient[1::2, 2, 2], rtol=2e-14, atol=0
    )


def test_far_field_binary64_neighbors_preserve_monotone_potential():
    from gragra.analytic import cylinder_potential_axis_g_free

    center = 9e307
    distances = [np.nextafter(center, 0.0), center, np.nextafter(center, np.inf)]
    values = [
        cylinder_potential_axis_g_free([0, 0, 0], 1, 1, 1800, [[0, 0, z]])[0]
        for z in distances
    ]
    expected = [_axis_volume_reference(1, 1, 1800, z)[0] for z in distances]
    np.testing.assert_allclose(values, expected, rtol=2e-14, atol=0)
    assert values[0] < values[1] < values[2]


def test_far_field_series_transition_is_continuous_at_documented_bound():
    from gragra.analytic import cylinder_axis_observables_g_free

    radius, height, density = 2.0, 3.0, 1800.0
    boundary = 64 * np.hypot(radius, height / 2)
    z_values = [boundary * (1 - 1e-12), boundary * (1 + 1e-12)]
    results = [
        cylinder_axis_observables_g_free(
            [0, 0, 0], radius, height, density, [[0, 0, z]]
        )
        for z in z_values
    ]
    for z, result in zip(z_values, results, strict=True):
        expected = _axis_volume_reference(radius, height, density, z)
        np.testing.assert_allclose(result.potential[0], expected[0], rtol=2e-12, atol=0)
        np.testing.assert_allclose(
            result.acceleration[0, 2], expected[1], rtol=2e-12, atol=0
        )
        np.testing.assert_allclose(
            result.gradient[0, 2, 2], expected[2], rtol=2e-12, atol=0
        )


def test_public_triple_rotation_si_and_readonly():
    from gragra.analytic import cylinder_analytic_observables_g_free
    from gragra.constants import G_SI
    from gragra.observables import cylinder_analytic_observables

    q = np.array([[0.6, 0, 0.8], [0, 1, 0], [-0.8, 0, 0.6]])
    center = np.array([0.5, -0.3, 0.2])
    points = np.array([[0.2, 0.3, 0.1], [2.6, -1.05, 2.1]])
    local = cylinder_analytic_observables_g_free([0, 0, 0], 2, 3, 1800, points)
    world = cylinder_analytic_observables_g_free(
        center, 2, 3, 1800, center + points @ q.T, rotation_matrix=q
    )
    si = cylinder_analytic_observables(
        center, 2, 3, 1800, center + points @ q.T, rotation_matrix=q
    )
    np.testing.assert_allclose(world.potential, local.potential, rtol=1e-13)
    np.testing.assert_allclose(world.acceleration, local.acceleration @ q.T, rtol=1e-12)
    np.testing.assert_allclose(
        world.gradient, np.einsum("ij,mjk,lk->mil", q, local.gradient, q), atol=1e-10
    )
    for name in ("potential", "acceleration", "gradient"):
        np.testing.assert_array_equal(getattr(si, name), G_SI * getattr(world, name))
        assert not getattr(world, name).flags.writeable
        assert not getattr(si, name).flags.writeable


@pytest.mark.parametrize(
    "radius,height,point",
    [
        (2, 3, [0, 0, 0.7]),
        (2, 3, [1e-5, -2e-5, 0.7]),
        (2, 3, [1e-12, 0, 0]),
        (2, 3, [2, 0, 3]),
        (2, 3, [3, 0, 1.5]),
        (2, 0.02, [0.4, 0.2, 0.001]),
        (2, 200, [0.4, 0.2, 0.1]),
        *[
            (2, 3, [2 + sign * gap, 0, 0.4])
            for gap in (0.2, 0.02, 0.002)
            for sign in (-1, 1)
        ],
        *[
            (2, 3, [0.4, 0, 1.5 + sign * gap])
            for gap in (0.2, 0.02, 0.002)
            for sign in (-1, 1)
        ],
        (2, 3, [2 - 1e-6, 0, 1.5 - 1e-6]),
        (2, 3, [2 + 1e-6, 0, 1.5 + 1e-6]),
        (2, 3, [150, 0, 0]),
        (2, 3, [170, 0, 0]),
    ],
)
def test_axis_boundary_extensions_aspect_ratios_and_far_transition(
    radius, height, point
):
    pytest.importorskip("scipy")
    from oracles.cylinder_potential_reference import boundary_potential

    from gragra.analytic import cylinder_analytic_observables_g_free

    low, _ = boundary_potential(radius, height, 1800, point, tolerance=1e-8)
    high, _ = boundary_potential(radius, height, 1800, point, tolerance=1e-11)
    budget = 1e-10 * (1800 * radius**2 + abs(high))
    assert abs(low - high) <= budget / 10
    actual = cylinder_analytic_observables_g_free(
        [0, 0, 0], radius, height, 1800, [point]
    )
    assert abs(actual.potential[0] - high) <= budget


@pytest.mark.parametrize("point", [[0.2, 0.3, 0.1], [2.6, -1.05, 2.1]])
def test_independent_volume_reference(point):
    pytest.importorskip("scipy")
    from oracles.cylinder_potential_reference import volume_potential

    from gragra.analytic import cylinder_analytic_observables_g_free

    low, _ = volume_potential(2, 3, 1800, point, tolerance=1e-8)
    high, _ = volume_potential(2, 3, 1800, point, tolerance=1e-11)
    budget = 1e-10 * (1800 * 2**2 + abs(high))
    assert abs(low - high) < budget / 10
    actual = cylinder_analytic_observables_g_free([0, 0, 0], 2, 3, 1800, [point])
    assert abs(actual.potential[0] - high) < budget


@pytest.mark.parametrize("point", [[2, 0, 0], [0.3, 0.2, 1.5], [2, 0, -1.5]])
def test_material_boundary_rejected(point):
    from gragra.analytic import cylinder_analytic_observables_g_free

    with pytest.raises(ValueError, match="boundary"):
        cylinder_analytic_observables_g_free([0, 0, 0], 2, 3, 1800, [point])


def test_axis_limit_zero_density_and_numerical_api_preserved():
    from gragra.analytic import (
        cylinder_analytic_observables_g_free,
        cylinder_axis_observables_g_free,
        cylinder_observables_g_free,
    )

    points = [[0, 0, 0.1], [0, 0, -2.1]]
    expected = cylinder_axis_observables_g_free([0, 0, 0], 2, 3, 1800, points)
    actual = cylinder_analytic_observables_g_free([0, 0, 0], 2, 3, 1800, points)
    numeric = cylinder_observables_g_free([0, 0, 0], 2, 3, 1800, points)
    for name in ("potential", "acceleration", "gradient"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
        np.testing.assert_array_equal(getattr(numeric, name), getattr(expected, name))
    zero = cylinder_analytic_observables_g_free([0, 0, 0], 2, 3, 0, [[0.2, 0.3, 0.4]])
    for name in ("potential", "acceleration", "gradient"):
        assert np.all(getattr(zero, name) == 0)


def test_thin_center_potential_shared_by_public_observables():
    if not run_isolated_reference_case("shared-observables"):
        pytest.skip("mpmath is required for the independent cylinder oracle")


@pytest.mark.parametrize("point", [[0.2, 0.3, 0.1], [2.6, -1.05, 2.1]])
def test_auxiliary_target_differences_match_c_acceleration_and_gradient(point):
    """Finite differences are auxiliary only; exact identities have a CAS artifact."""
    from gragra.analytic import cylinder_analytic_observables_g_free

    point = np.array(point)
    result = cylinder_analytic_observables_g_free([0, 0, 0], 2, 3, 1800, [point])
    first = np.array([-1, 9, -45, 0, 45, -9, 1]) / 60
    second = np.array([2, -27, 270, -490, 270, -27, 2]) / 180
    for step in (0.02, 0.01):
        for axis in range(3):
            offsets = np.outer(np.arange(-3, 4), np.eye(3)[axis]) * step
            phi = cylinder_analytic_observables_g_free(
                [0, 0, 0], 2, 3, 1800, point + offsets
            ).potential
            np.testing.assert_allclose(
                -first @ phi / step, result.acceleration[0, axis], atol=2e-6, rtol=0
            )
            np.testing.assert_allclose(
                -second @ phi / step**2,
                result.gradient[0, axis, axis],
                atol=2e-5,
                rtol=0,
            )
        for a, b in ((0, 1), (0, 2), (1, 2)):
            offsets = np.array(
                [
                    step * (i * np.eye(3)[a] + j * np.eye(3)[b])
                    for i in range(-3, 4)
                    for j in range(-3, 4)
                ]
            )
            phi = cylinder_analytic_observables_g_free(
                [0, 0, 0], 2, 3, 1800, point + offsets
            ).potential.reshape(7, 7)
            np.testing.assert_allclose(
                -first @ phi @ first / step**2,
                result.gradient[0, a, b],
                atol=2e-5,
                rtol=0,
            )


@pytest.mark.parametrize(
    "rotation",
    [
        np.diag([1, 1, -1]),
        np.zeros((3, 3)),
        np.eye(2),
        np.eye(3) * 2,
        np.eye(3) * np.nan,
    ],
)
def test_triple_rejects_invalid_rotations(rotation):
    from gragra.analytic import cylinder_analytic_observables_g_free

    with pytest.raises(ValueError, match="rotation_matrix"):
        cylinder_analytic_observables_g_free(
            [0, 0, 0], 2, 3, 1800, [[0.2, 0.3, 0.1]], rotation_matrix=rotation
        )
