import json
from pathlib import Path

import numpy as np
import pytest

import gragra
from gragra.analytic.wave456_reference import (
    GFreeObservables,
    _rayleigh_root,
    acoustic_finite_box_observables_g_free,
    acoustic_infinite_halfspace_observables_g_free,
    adaptive_box_observables_g_free,
    adaptive_displacement_box_observables_g_free,
    host_inclusion_observables_g_free,
    integrate_box_observables_g_free,
    integrate_displacement_box_observables_g_free,
    multipole_observables_g_free,
    opposed_rotor_leading_phasor_g_free,
    opposed_rotor_observables_g_free,
    passing_cuboid_observables_g_free,
    point_mass_observables_g_free,
    rayleigh_halfspace_observables_g_free,
    solid_cylinder_observables_g_free,
)

WORKTREE = Path(__file__).resolve().parents[1]


def test_tests_import_this_worktree_source():
    assert Path(gragra.__file__).resolve().is_relative_to(WORKTREE / "src")


def test_observable_container_enforces_static_or_series_shapes_and_copies():
    potential = np.arange(6.0).reshape(2, 3)
    acceleration = np.zeros((2, 3, 3))
    gradient = np.zeros((2, 3, 3, 3))
    result = GFreeObservables(potential, acceleration, gradient)
    potential[0, 0] = -99.0
    assert result.potential[0, 0] == 0.0
    assert result.potential.dtype == np.float64
    assert not result.potential.flags.writeable
    assert not result.acceleration.flags.writeable
    assert not result.gradient.flags.writeable

    with pytest.raises(ValueError, match="static .* or series"):
        GFreeObservables(
            np.zeros((2, 3, 1)), np.zeros((2, 3, 3)), np.zeros((2, 3, 3, 3))
        )
    with pytest.raises(ValueError, match="shared sample/target prefix"):
        GFreeObservables(np.zeros((2, 3)), np.zeros((4, 3, 3)), np.zeros((2, 3, 3, 3)))


def test_point_reference_matches_literal_goldens_and_preserves_complex():
    targets = np.array([[2.0, 0.0, 0.0], [0.0, -4.0, 0.0]])
    got = point_mass_observables_g_free([[0.0, 0.0, 0.0]], [2.0j], targets)
    np.testing.assert_allclose(got.potential, [-1j, -0.5j])
    np.testing.assert_allclose(got.acceleration[0], [-0.5j, 0.0, 0.0])
    np.testing.assert_allclose(got.gradient[0], np.diag([0.5j, -0.25j, -0.25j]))
    assert got.potential.dtype == np.complex128


@pytest.mark.parametrize(
    ("function", "args", "match"),
    [
        (
            point_mass_observables_g_free,
            (np.array([[1.0 + 1.0j, 0.0, 0.0]]), [1.0], [[2.0, 0.0, 0.0]]),
            "real",
        ),
        (
            point_mass_observables_g_free,
            ([[True, 0.0, 0.0]], [1.0], [[2.0, 0.0, 0.0]]),
            "boolean",
        ),
        (
            passing_cuboid_observables_g_free,
            (np.array([0.0 + 1.0j]), [[0.0, 0.0, 0.0]]),
            "real",
        ),
        (
            opposed_rotor_observables_g_free,
            (np.array([True]), [[2.0, 0.0, 0.0]]),
            "boolean",
        ),
    ],
)
def test_geometry_and_sample_axes_reject_complex_and_boolean(function, args, match):
    with pytest.raises((TypeError, ValueError), match=match):
        function(*args)


def test_zero_second_moment_multipole_is_exact_point_mass():
    targets = [[3.0, -1.0, 2.0], [-2.0, 4.0, 1.0]]
    point = point_mass_observables_g_free([[0.3, 0.2, -0.1]], [7.0], targets)
    multi = multipole_observables_g_free(
        [0.3, 0.2, -0.1], 7.0, np.zeros((3, 3)), targets
    )
    for actual, expected in zip(multi, point, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=1e-16)


def test_multipole_gradient_is_target_derivative_of_acceleration():
    target = np.array([[4.0, -2.0, 1.5]])
    moment = np.array([[3.0, 0.2, -0.1], [0.2, 2.0, 0.4], [-0.1, 0.4, 1.0]])
    got = multipole_observables_g_free([0.1, 0.2, -0.3], 8.0, moment, target)
    step = 2e-5
    numerical = np.empty((3, 3))
    for axis in range(3):
        offset = np.zeros_like(target)
        offset[0, axis] = step
        plus = multipole_observables_g_free(
            [0.1, 0.2, -0.3], 8.0, moment, target + offset
        )
        minus = multipole_observables_g_free(
            [0.1, 0.2, -0.3], 8.0, moment, target - offset
        )
        numerical[:, axis] = (plus.acceleration[0] - minus.acceleration[0]) / (2 * step)
    np.testing.assert_allclose(got.gradient[0], numerical, rtol=2e-9, atol=2e-11)


def test_constant_box_respects_center_symmetry_and_far_mass_limit():
    bounds = [[-1.0, 1.0], [-2.0, 2.0], [-0.5, 0.5]]
    symmetric = integrate_box_observables_g_free(
        bounds, 3.0, [[3.0, 0.0, 0.0], [-3.0, 0.0, 0.0]], order=12
    )
    np.testing.assert_allclose(symmetric.potential[0], symmetric.potential[1])
    np.testing.assert_allclose(
        symmetric.acceleration[0], -symmetric.acceleration[1], atol=3e-16
    )
    np.testing.assert_allclose(symmetric.gradient[0], symmetric.gradient[1], atol=3e-16)
    np.testing.assert_allclose(np.trace(symmetric.gradient[0]), 0.0, atol=2e-14)

    target = np.array([[1000.0, 0.0, 0.0]])
    box = integrate_box_observables_g_free(bounds, 3.0, target, order=12)
    point = point_mass_observables_g_free([[0.0, 0.0, 0.0]], [24.0], target)
    np.testing.assert_allclose(box.potential, point.potential, rtol=2e-6)
    np.testing.assert_allclose(
        box.acceleration, point.acceleration, rtol=5e-6, atol=1e-22
    )


@pytest.mark.parametrize("target", [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
def test_box_references_reject_interior_and_boundary_material_targets(target):
    bounds = [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]
    with pytest.raises(ValueError, match="exterior"):
        integrate_box_observables_g_free(bounds, 2.0, [target], order=4)
    with pytest.raises(ValueError, match="exterior"):
        integrate_displacement_box_observables_g_free(
            bounds,
            2.0,
            lambda points: np.ones_like(points),
            [target],
            order=4,
        )


def test_explicit_zero_density_box_is_omitted_even_for_interior_target():
    zero = integrate_box_observables_g_free(
        [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
        0.0,
        [[0.0, 0.0, 0.0]],
        order=4,
    )
    for value in zero:
        np.testing.assert_array_equal(value, 0.0)


def test_host_inclusion_partitions_material_interface_without_double_counting():
    host = [[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0]]
    inner = [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]
    targets = [[5.0, 1.0, 0.5]]
    got, evidence = host_inclusion_observables_g_free(
        host, inner, 2.0, 5.0, targets, low_order=6, high_order=8, rtol=1e-9
    )
    base = integrate_box_observables_g_free(host, 2.0, targets, order=24)
    contrast = integrate_box_observables_g_free(inner, 3.0, targets, order=24)
    for actual, expected_base, expected_delta in zip(got, base, contrast, strict=True):
        np.testing.assert_allclose(actual, expected_base + expected_delta, rtol=1e-9)
    assert evidence.status == "CONVERGED"


def test_affine_displacement_matches_centered_finite_difference():
    bounds = [[-0.5, 0.5], [-0.4, 0.4], [-0.3, 0.3]]
    targets = [[2.0, -1.5, 1.2]]
    shift = np.array([0.2, -0.1, 0.05])
    linear = integrate_displacement_box_observables_g_free(
        bounds,
        4.0,
        lambda points: np.broadcast_to(shift, points.shape),
        targets,
        order=16,
    )
    eps = 1e-5
    plus = np.asarray(bounds) + eps * shift[:, None]
    minus = np.asarray(bounds) - eps * shift[:, None]
    q_plus = integrate_box_observables_g_free(plus, 4.0, targets, order=20)
    q_minus = integrate_box_observables_g_free(minus, 4.0, targets, order=20)
    for actual, hi, lo in zip(linear, q_plus, q_minus, strict=True):
        np.testing.assert_allclose(actual, (hi - lo) / (2 * eps), rtol=3e-9, atol=2e-11)


def test_adaptive_displacement_reports_independent_order_convergence():
    bounds = [[-0.5, 0.5], [-0.4, 0.4], [-0.3, 0.3]]
    targets = [[2.0, -1.5, 1.2]]
    result, evidence = adaptive_displacement_box_observables_g_free(
        [(bounds, 4.0, lambda points: 0.01 * points + [0.2, -0.1, 0.05])],
        targets,
        low_order=6,
        high_order=8,
        rtol=1e-10,
    )
    assert result.potential.shape == (1,)
    assert evidence.status == "CONVERGED"
    assert evidence.normalized_difference <= 1e-10


def test_adaptive_budget_includes_initial_and_escalation_evaluations():
    regions = [
        ([[-1.0, 0.0], [-1.0, 1.0], [-1.0, 1.0]], 1.0),
        ([[0.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]], 1.0),
    ]
    value, evidence = adaptive_box_observables_g_free(
        regions,
        [[3.0, 0.2, 0.1]],
        low_order=2,
        high_order=3,
        rtol=1e-30,
        max_evaluations=2,
    )
    assert value.potential.shape == (1,)
    assert evidence.status == "ORACLE_UNRESOLVED"
    assert evidence.evaluations == 2

    _, escalation_blocked = adaptive_box_observables_g_free(
        [regions[0]],
        [[3.0, 0.2, 0.1]],
        low_order=2,
        high_order=3,
        rtol=1e-30,
        max_evaluations=2,
    )
    assert escalation_blocked.status == "ORACLE_UNRESOLVED"
    assert escalation_blocked.evaluations == 2
    assert (escalation_blocked.low_order, escalation_blocked.high_order) == (2, 3)


def test_series_references_keep_explicit_sample_axis():
    angles = np.array([0.0, 0.3, 1.1])
    targets = [[2.0, 0.3, 0.2], [4.0, 1.0, 0.5]]
    rotor = opposed_rotor_observables_g_free(
        angles, targets, radial_order=4, azimuthal_order=24, axial_order=4
    )
    passing = passing_cuboid_observables_g_free([-0.2, 0.0, 0.2], targets, order=8)
    for result in (rotor, passing):
        assert result.potential.shape == (3, 2)
        assert result.acceleration.shape == (3, 2, 3)
        assert result.gradient.shape == (3, 2, 3, 3)


def test_cylinder_quadrature_includes_periodic_weight_once():
    target = [[1000.0, 0.0, 0.0]]
    expected_mass = 7800.0 * np.pi * 0.1**2 * 0.2
    point = point_mass_observables_g_free([[0.0, 0.0, 0.0]], [expected_mass], target)
    for angular_order in (16, 40):
        cylinder = solid_cylinder_observables_g_free(
            [[0.0, 0.0, 0.0]],
            0.1,
            0.2,
            7800.0,
            target,
            radial_order=6,
            azimuthal_order=angular_order,
            axial_order=6,
        )
        np.testing.assert_allclose(cylinder.potential, point.potential, rtol=2e-8)


def test_opposed_rotor_leading_phasor_matches_analytic_multipole_dft():
    targets = [[8.0, -1.5, 1.0]]
    angles = 2 * np.pi * np.arange(128) / 128
    series = []
    mass = 2 * 7800.0 * np.pi * 0.1**2 * 0.2
    radial_second = mass * (0.5**2 + 0.1**2 / 4.0)
    axial_second = mass * 0.2**2 / 12.0
    for angle in angles:
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        moment = (
            rotation
            @ np.diag([radial_second, mass * 0.1**2 / 4.0, axial_second])
            @ rotation.T
        )
        series.append(multipole_observables_g_free([0, 0, 0], mass, moment, targets))
    sampled = np.stack([value.potential for value in series])
    coefficient = 2.0 * np.mean(sampled * np.exp(2j * angles)[:, None], axis=0)
    analytic = opposed_rotor_leading_phasor_g_free(targets)
    np.testing.assert_allclose(analytic.potential, coefficient, rtol=2e-13, atol=1e-15)


def test_acoustic_infinite_halfspace_obeys_fourier_derivatives():
    wavelength = 34.3
    targets = np.array(
        [[0.0, 0.0, -0.25 * wavelength], [0.125 * wavelength, 0.0, -0.5 * wavelength]]
    )
    got = acoustic_infinite_halfspace_observables_g_free(targets)
    k = 2 * np.pi / wavelength
    d = np.array([1j * k, 0.0, k])
    np.testing.assert_allclose(got.acceleration, -got.potential[:, None] * d)
    np.testing.assert_allclose(
        got.gradient, -got.potential[:, None, None] * np.outer(d, d)
    )


def test_acoustic_finite_box_requires_air_above_and_targets_below_ground():
    with pytest.raises(ValueError, match="air box"):
        acoustic_finite_box_observables_g_free(
            [[-1.0, 1.0], [-1.0, 1.0], [-0.1, 1.0]], [[0.0, 0.0, -1.0]]
        )
    with pytest.raises(ValueError, match="below"):
        acoustic_finite_box_observables_g_free(
            [[-1.0, 1.0], [-1.0, 1.0], [0.0, 1.0]], [[0.0, 0.0, 0.1]]
        )


def test_rayleigh_reference_has_expected_phase_trace_and_fixed_parameters():
    targets = np.array([[0.0, 0.0, 10.0], [4.0, -3.0, 25.0]])
    got = rayleigh_halfspace_observables_g_free(targets)
    assert got.potential.dtype == np.complex128
    np.testing.assert_allclose(
        np.trace(got.gradient, axis1=-2, axis2=-1), 0.0, atol=1e-25
    )
    assert got.acceleration[0, 0].imag > 0.0
    assert got.acceleration[0, 2].real < 0.0
    with pytest.raises(ValueError, match="above"):
        rayleigh_halfspace_observables_g_free([[0.0, 0.0, 0.0]])


def test_rayleigh_root_requires_stable_isotropic_domain_and_checks_residual():
    with pytest.raises(ValueError, match="stable isotropic"):
        rayleigh_halfspace_observables_g_free(
            [[0.0, 0.0, 1.0]], v_p_m_s=1.01, v_s_m_s=1.0
        )
    speed, _, _ = _rayleigh_root(2800.0, 1500.0)
    residual = (2.0 - (speed / 1500.0) ** 2) ** 2 - 4.0 * np.sqrt(
        1.0 - (speed / 2800.0) ** 2
    ) * np.sqrt(1.0 - (speed / 1500.0) ** 2)
    assert abs(residual) < 1e-13


def test_rayleigh_old_anna_integral_is_mapped_explicitly_to_z_up_phasor():
    fixture = json.loads((WORKTREE / "tests/oracles/data/rayleigh_v1.json").read_text())
    point = next(
        item
        for item in fixture["points"]
        if np.isclose(item["vp_vs_ratio"], np.sqrt(3.0)) and item["kH"] == 1.0
    )
    c_over_vs = np.sqrt(2.0 - 2.0 / np.sqrt(3.0))
    got = rayleigh_halfspace_observables_g_free(
        [[0.0, 0.0, 1.0]],
        density_kg_m3=1.0,
        v_p_m_s=np.sqrt(3.0),
        v_s_m_s=1.0,
        frequency_hz=c_over_vs / (2.0 * np.pi),
        vertical_surface_amplitude_m=1.0,
    )
    # Legacy JSON stores real ANNA-style coefficients: ax multiplies +i e_x,
    # while az is the signed coefficient along z.  This explicit map avoids
    # treating the old two-scalar convention as a gragra xyz phasor.
    mapped = np.array(
        [
            1j * point["full"]["ax_extrapolated_x_cut_inf"],
            0.0,
            point["full"]["az_extrapolated_x_cut_inf"],
        ]
    )
    np.testing.assert_allclose(got.acceleration[0], mapped, rtol=5e-6, atol=8e-6)
