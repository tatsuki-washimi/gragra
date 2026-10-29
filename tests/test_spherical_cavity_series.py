"""Independent checks of the fixed-cavity harmonic series."""

import numpy as np
import pytest

from gragra.analytic.cavity_sphere import (
    spherical_cavity_total_p,
    spherical_cavity_wall,
)
from gragra.analytic.spherical_cavity_series import (
    _spherical_bessel_sequence,
    spherical_cavity_observables_g_free,
)
from scripts.research_wave6_spherical_cavity_general_point import (
    _cavity_observables_at_order,
)


def parameters(wave, k):
    return {
        "radius_m": 1.0,
        "density_kg_m3": 2700.0,
        "wavenumber_rad_m": k,
        "displacement_m": 1e-6 * (1 + 0.4j),
        "propagation_direction": [0, 0, 1],
        "polarization": [0, 0, 1] if wave == "P" else [1, 0, 0],
        "wave_type": wave,
    }


@pytest.mark.parametrize("k", [0.01, 0.8, 5.0])
@pytest.mark.parametrize("wave", ["P", "S"])
def test_independent_integrals_and_center(k, wave):
    targets = np.array([[0, 0, 0], [0, 0, 0.3], [0.2, -0.1, 0.1], [1.4, 0.2, -0.1]])
    kw = parameters(wave, k)
    result = spherical_cavity_observables_g_free(targets, **kw)
    refs = [_cavity_observables_at_order(targets, **kw, order=n) for n in (48, 72)]
    scales = [2700 * abs(kw["displacement_m"])] * 3
    for part in ("bulk", "wall", "total"):
        for name, scale in zip(
            ("potential", "acceleration", "gradient"), scales, strict=True
        ):
            a, b = (ref[part][name] for ref in refs)
            for i in range(len(targets)):
                rtol = 1e-9 if name == "gradient" else 1e-10
                budget = scale * 1e-11 + rtol * np.linalg.norm(b[i])
                assert np.linalg.norm(a[i] - b[i]) < budget / 10
                assert np.linalg.norm(getattr(result[part], name)[i] - b[i]) < budget
    expected = (
        spherical_cavity_total_p(1, k, 2700, 1e-6)
        if wave == "P"
        else spherical_cavity_wall(1, k, 2700, 1e-6, wave_type="S")
    )
    np.testing.assert_allclose(
        result["total"].acceleration[0],
        expected * (1 + 0.4j) * np.array(kw["polarization"]),
        atol=1e-14,
    )


@pytest.mark.parametrize("k", [0.01, 0.8, 5.0])
@pytest.mark.parametrize("wave", ["P", "S"])
def test_near_wall_rotation_phase_and_trace(k, wave):
    targets = np.array([[0, 0, 0], [1e-12, 0, 0.4], [0.6, 0, 0.8]])
    targets = np.concatenate(
        [targets[:2], targets[2:] * (1 - 1e-9), targets[2:] * (1 + 1e-9), [[12, -4, 3]]]
    )
    kw = parameters(wave, k)
    before = targets.copy()
    values = spherical_cavity_observables_g_free(targets, **kw)
    q, _ = np.linalg.qr(np.array([[1.0, 2, 3], [-2, 1, 2], [2, 3, -1]]))
    center = np.array([2, -3, 0.7])
    rotated = dict(
        kw,
        propagation_direction=q @ kw["propagation_direction"],
        polarization=q @ kw["polarization"],
        center_m=center,
        phase_origin_m=center,
    )
    transformed = spherical_cavity_observables_g_free(targets @ q.T + center, **rotated)
    shifted = spherical_cavity_observables_g_free(
        targets, **kw, phase_origin_m=[0, 0, 0.7]
    )
    for part in values:
        value, other = values[part], transformed[part]
        np.testing.assert_allclose(other.potential, value.potential, atol=1e-13)
        np.testing.assert_allclose(
            other.acceleration, value.acceleration @ q.T, atol=1e-13
        )
        np.testing.assert_allclose(other.gradient, q @ value.gradient @ q.T, atol=1e-12)
        for a, b in zip(value, shifted[part], strict=True):
            assert not a.flags.writeable
            np.testing.assert_allclose(b, a * np.exp(-1j * k * 0.7), atol=1e-13)
        np.testing.assert_allclose(
            value.gradient, value.gradient.swapaxes(-1, -2), atol=1e-14
        )
    expected = np.zeros(len(targets), complex)
    if wave == "P":
        exterior = np.linalg.norm(targets, axis=1) > 1
        expected[exterior] = (
            4j
            * np.pi
            * 2700
            * k
            * kw["displacement_m"]
            * np.exp(1j * k * targets[exterior, 2])
        )
    np.testing.assert_allclose(
        np.trace(values["total"].gradient, axis1=-2, axis2=-1), expected, atol=2e-12
    )
    np.testing.assert_array_equal(targets, before)


def test_rejection_and_convergence():
    kw = parameters("P", 5)
    with pytest.raises(ValueError, match="wall"):
        spherical_cavity_observables_g_free([[0, 0, 1]], **kw)
    with pytest.raises(ValueError, match="converg"):
        spherical_cavity_observables_g_free(
            [[0, 0, 0], [0.6, 0.2, 0.7]], **kw, max_degree=3
        )
    spherical_cavity_observables_g_free([[0, 0, 0]], **kw, max_degree=15)
    with pytest.raises(ValueError, match="target 1"):
        spherical_cavity_observables_g_free(
            [[0, 0, 0], [0.6, 0.2, 0.7]], **kw, max_degree=15
        )
    for field, value in [
        ("polarization", [1, 0, 0]),
        ("wavenumber_rad_m", 0),
        ("density_kg_m3", np.nan),
        ("rtol", -1),
        ("max_degree", 1.5),
    ]:
        with pytest.raises((TypeError, ValueError)):
            spherical_cavity_observables_g_free(
                [[0, 0, 0.1]], **dict(kw, **{field: value})
            )


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("rtol", True),
        ("rtol", np.array([1e-3])),
        ("atol", np.array([1e-13 + 1j, 1e-13, 1e-13])),
        ("atol", (True, False, False)),
    ],
)
def test_rejects_non_real_scalar_and_non_real_or_boolean_tolerances(keyword, value):
    """Tolerance validation must inspect the raw input before float casting."""
    with pytest.raises(TypeError):
        spherical_cavity_observables_g_free(
            [[0, 0, 0]], **(parameters("P", 5) | {keyword: value})
        )


@pytest.mark.parametrize(
    ("targets_m", "overrides"),
    [
        ([[0, 0, 0.2]], {"wavenumber_rad_m": [True, 0.8]}),
        (
            [[0, 0, 0.2]],
            {"wavenumber_rad_m": [0.8, 0.9], "displacement_m": [True, 1e-6]},
        ),
        ([[True, 0, 0.2]], {}),
        (
            [[0, 0, 0.2]],
            {"propagation_direction": [True, 0, 0], "polarization": [1, 0, 0]},
        ),
        (
            [[0, 0, 0.2]],
            {"propagation_direction": [1, 0, 0], "polarization": [True, 0, 0]},
        ),
        ([[0, 0, 0.2]], {"center_m": [True, 0, 0]}),
        ([[0, 0, 0.2]], {"phase_origin_m": [True, 0, 0]}),
    ],
)
def test_rejects_mixed_boolean_array_inputs_before_numeric_cast(targets_m, overrides):
    with pytest.raises(TypeError, match="must not contain bool"):
        spherical_cavity_observables_g_free(
            targets_m, **(parameters("P", 0.8) | overrides)
        )


@pytest.mark.parametrize("q", [1e-200, 1e-100, 1e-10, 0.01, 0.8, 5.0, 30.0, 100.0])
def test_bessel_against_independent_special_function(q):
    special = pytest.importorskip("scipy.special")
    actual = _spherical_bessel_sequence(q, 160)
    np.testing.assert_allclose(
        actual, special.spherical_jn(np.arange(161), q), rtol=2e-12, atol=2e-16
    )


def test_si_wrapper():
    from gragra.constants import G_SI
    from gragra.observables.analytic import spherical_cavity_observables

    kw = parameters("S", 0.8)
    gf = spherical_cavity_observables_g_free([[0.2, -0.1, 0.1]], **kw)
    si = spherical_cavity_observables([[0.2, -0.1, 0.1]], **kw)
    for part in gf:
        for a, b in zip(gf[part], si[part], strict=True):
            np.testing.assert_array_equal(b, G_SI * a)
            assert not b.flags.writeable


def test_convergence_norm_does_not_overflow_for_finite_fields():
    kw = dict(parameters("P", 5), density_kg_m3=1, displacement_m=1e190)
    with pytest.raises(ValueError, match="target 0"):
        spherical_cavity_observables_g_free([[0.6, 0.2, 0.7]], **kw, max_degree=15)


def test_frequency_axis_and_zero_contract():
    kw = parameters("P", 0.8)
    frequencies = [0.01, 0.8, 5.0]
    displacements = np.array([1, 2j, -1]) * 1e-6
    combined = spherical_cavity_observables_g_free(
        [[0, 0, 0], [0.2, 0.1, 0.3]],
        **dict(kw, wavenumber_rad_m=frequencies, displacement_m=displacements),
    )
    for i, k in enumerate(frequencies):
        single = spherical_cavity_observables_g_free(
            [[0, 0, 0], [0.2, 0.1, 0.3]],
            **dict(kw, wavenumber_rad_m=k, displacement_m=displacements[i]),
        )
        for part in single:
            for a, b in zip(combined[part], single[part], strict=True):
                np.testing.assert_array_equal(a[i], b)
    zero = spherical_cavity_observables_g_free(
        [[0, 0, 0.2]], **dict(kw, density_kg_m3=0)
    )
    assert all(np.all(a == 0) for field in zero.values() for a in field)
    for field in ["radius_m", "density_kg_m3", "wavenumber_rad_m", "displacement_m"]:
        with pytest.raises((TypeError, ValueError)):
            spherical_cavity_observables_g_free(
                [[0, 0, 0.2]], **dict(kw, **{field: True})
            )


@pytest.mark.parametrize("wave", ["P", "S"])
@pytest.mark.parametrize("k", [0.01, 0.8, 5.0])
@pytest.mark.parametrize("side", [-1, 1])
def test_near_wall_high_precision_newton_reference(wave, k, side):
    pytest.importorskip("mpmath")
    from scripts.validate_spherical_cavity_axis import axis_reference

    r = 1 + side * 1e-9
    low, high = (axis_reference(r, k, wave, dps) for dps in (35, 50))
    actual = spherical_cavity_observables_g_free(
        [[0, 0, r]], **dict(parameters(wave, k), density_kg_m3=1, displacement_m=1)
    )
    for part in actual:
        for name, sl in zip(
            ("potential", "acceleration", "gradient"),
            (slice(0, 1), slice(1, 4), slice(4, 13)),
            strict=True,
        ):
            budget = 1e-7 * (4 * np.pi + np.linalg.norm(high[part][sl]))
            assert np.linalg.norm(low[part][sl] - high[part][sl]) <= budget / 10
            assert (
                np.linalg.norm(getattr(actual[part], name).ravel() - high[part][sl])
                <= budget
            )
