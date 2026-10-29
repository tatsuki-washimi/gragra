"""Tests for the literature-sourced single-mode and reflected-P body-wave
half-space gravity (LIT-B).

Primary source: H19 (Harms, Living Reviews in Relativity 22, 6 (2019)),
Eq. (86)-(90). See ``docs/design/literature-b.md`` and
``docs/validation/literature-b.md`` for the full equation mapping and
independent-verification report.

This file covers: exact analytic identities (Poisson trace, divergence-free
SV, surface mass-sheet jump -- derived independently from the wave's own
displacement field, not copied from production), the theta=0 gauge limits,
a finite-difference spot check of a=-grad(Phi)/Gamma=grad(a), and the
input-validation / shape / dtype contract. The independent boundary-value
re-derivation of Eq. (90) itself lives in
``tests/test_bodywave_literature_oracle.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from gragra.analytic.bodywave_literature import (
    halfspace_reflected_p_above_observables_g_free,
    halfspace_single_bodywave_observables_g_free,
)

POISSON = {"name": "poisson", "v_p_m_s": 1500.0 * np.sqrt(3.0), "v_s_m_s": 1500.0}
HARD_ROCK = {"name": "hard_rock", "v_p_m_s": 5800.0, "v_s_m_s": 3200.0}
MATERIALS = [POISSON, HARD_ROCK]
RHO = 2700.0
FREQ = 5.0


def _p_polarization(theta: float, azimuth: float) -> np.ndarray:
    h = np.array([np.cos(azimuth), np.sin(azimuth), 0.0])
    return np.sin(theta) * h + np.cos(theta) * np.array([0.0, 0.0, 1.0])


def _sv_polarization(theta: float, azimuth: float) -> np.ndarray:
    h = np.array([np.cos(azimuth), np.sin(azimuth), 0.0])
    return np.cos(theta) * h - np.sin(theta) * np.array([0.0, 0.0, 1.0])


def _delta_rho_p(material, theta, azimuth, xi0_p, target) -> complex:
    """Independent delta_rho = -rho * div(xi_P), xi_P = xi0_p * khat * exp(i k.r).

    div(xi_P) = i * xi0_p * kP * exp(i k.r) since k . khat = |k| = kP for a P
    wave (polarization parallel to propagation) -- derived from the wave
    definition, not from the production module's internal formulas.
    """
    k_p = 2.0 * np.pi * FREQ / material["v_p_m_s"]
    khat = _p_polarization(theta, azimuth)
    x, y, z = target
    phase = np.exp(1j * k_p * (khat[0] * x + khat[1] * y + khat[2] * z))
    div_xi = 1j * xi0_p * k_p * phase
    return -RHO * div_xi


@pytest.mark.parametrize("material", MATERIALS, ids=lambda m: m["name"])
def test_p_underground_poisson_trace_identity(material):
    theta, azimuth = np.deg2rad(37.0), np.deg2rad(20.0)
    xi0_p = 1e-9 + 0.4e-9j
    target = [17.0, -8.0, -4.0]
    r = halfspace_single_bodywave_observables_g_free(
        [target],
        wave_type="P",
        density_kg_m3=RHO,
        v_p_m_s=material["v_p_m_s"],
        v_s_m_s=material["v_s_m_s"],
        frequency_hz=FREQ,
        incidence_angle_rad=theta,
        azimuth_rad=azimuth,
        displacement_m=xi0_p,
    )
    trace = np.trace(r.gradient[0])
    expected = -4.0 * np.pi * _delta_rho_p(material, theta, azimuth, xi0_p, target)
    scale = RHO * abs(xi0_p) * (2.0 * np.pi * FREQ / material["v_p_m_s"])
    assert abs(trace - expected) <= 1e-10 * scale + 1e-10 * abs(expected)


@pytest.mark.parametrize("material", MATERIALS, ids=lambda m: m["name"])
@pytest.mark.parametrize("wave_type", ["P", "SV"])
def test_ground_vacuum_trace_is_zero(material, wave_type):
    theta, azimuth = np.deg2rad(37.0), np.deg2rad(20.0)
    r = halfspace_single_bodywave_observables_g_free(
        [[17.0, -8.0, 4.0]],
        wave_type=wave_type,
        density_kg_m3=RHO,
        v_p_m_s=material["v_p_m_s"],
        v_s_m_s=material["v_s_m_s"],
        frequency_hz=FREQ,
        incidence_angle_rad=theta,
        azimuth_rad=azimuth,
        displacement_m=1e-9,
    )
    scale = RHO * 1e-9 * (2.0 * np.pi * FREQ / material["v_s_m_s"]) ** 2
    assert abs(np.trace(r.gradient[0])) <= 1e-10 * scale


@pytest.mark.parametrize("material", MATERIALS, ids=lambda m: m["name"])
@pytest.mark.parametrize("z_sign", [1.0, -1.0])
def test_sv_divergence_free_trace_is_zero(material, z_sign):
    """SV has zero divergence everywhere (shear wave produces no bulk density)."""
    theta, azimuth = np.deg2rad(37.0), np.deg2rad(20.0)
    r = halfspace_single_bodywave_observables_g_free(
        [[17.0, -8.0, z_sign * 4.0]],
        wave_type="SV",
        density_kg_m3=RHO,
        v_p_m_s=material["v_p_m_s"],
        v_s_m_s=material["v_s_m_s"],
        frequency_hz=FREQ,
        incidence_angle_rad=theta,
        azimuth_rad=azimuth,
        displacement_m=1e-9 - 0.2e-9j,
    )
    scale = RHO * 1e-9 * (2.0 * np.pi * FREQ / material["v_s_m_s"]) ** 2
    assert abs(np.trace(r.gradient[0])) <= 1e-10 * scale


@pytest.mark.parametrize("material", MATERIALS, ids=lambda m: m["name"])
def test_p_surface_mass_sheet_jump(material):
    theta, azimuth = np.deg2rad(37.0), np.deg2rad(20.0)
    xi0_p = 1e-9 + 0.4e-9j
    eps = 1e-6
    x, y = 5.0, 3.0
    above = halfspace_single_bodywave_observables_g_free(
        [[x, y, eps]],
        wave_type="P",
        density_kg_m3=RHO,
        v_p_m_s=material["v_p_m_s"],
        v_s_m_s=material["v_s_m_s"],
        frequency_hz=FREQ,
        incidence_angle_rad=theta,
        azimuth_rad=azimuth,
        displacement_m=xi0_p,
    )
    below = halfspace_single_bodywave_observables_g_free(
        [[x, y, -eps]],
        wave_type="P",
        density_kg_m3=RHO,
        v_p_m_s=material["v_p_m_s"],
        v_s_m_s=material["v_s_m_s"],
        frequency_hz=FREQ,
        incidence_angle_rad=theta,
        azimuth_rad=azimuth,
        displacement_m=xi0_p,
    )
    az_jump = above.acceleration[0, 2] - below.acceleration[0, 2]
    k_p = 2.0 * np.pi * FREQ / material["v_p_m_s"]
    k_rho = k_p * np.sin(theta)
    phase0 = np.exp(1j * k_rho * (x * np.cos(azimuth) + y * np.sin(azimuth)))
    sigma = RHO * xi0_p * np.cos(theta) * phase0  # P polarization z-component at z=0
    expected = -4.0 * np.pi * sigma
    scale = RHO * abs(xi0_p) * k_rho
    assert abs(az_jump - expected) <= 10.0 * eps * (k_p + k_rho) * (
        scale + abs(expected)
    )


@pytest.mark.parametrize("material", MATERIALS, ids=lambda m: m["name"])
def test_sv_surface_mass_sheet_jump(material):
    theta, azimuth = np.deg2rad(37.0), np.deg2rad(20.0)
    xi0_sv = 1e-9 - 0.2e-9j
    eps = 1e-6
    x, y = 5.0, 3.0
    above = halfspace_single_bodywave_observables_g_free(
        [[x, y, eps]],
        wave_type="SV",
        density_kg_m3=RHO,
        v_p_m_s=material["v_p_m_s"],
        v_s_m_s=material["v_s_m_s"],
        frequency_hz=FREQ,
        incidence_angle_rad=theta,
        azimuth_rad=azimuth,
        displacement_m=xi0_sv,
    )
    below = halfspace_single_bodywave_observables_g_free(
        [[x, y, -eps]],
        wave_type="SV",
        density_kg_m3=RHO,
        v_p_m_s=material["v_p_m_s"],
        v_s_m_s=material["v_s_m_s"],
        frequency_hz=FREQ,
        incidence_angle_rad=theta,
        azimuth_rad=azimuth,
        displacement_m=xi0_sv,
    )
    az_jump = above.acceleration[0, 2] - below.acceleration[0, 2]
    k_s = 2.0 * np.pi * FREQ / material["v_s_m_s"]
    k_rho = k_s * np.sin(theta)
    phase0 = np.exp(1j * k_rho * (x * np.cos(azimuth) + y * np.sin(azimuth)))
    sigma = -RHO * xi0_sv * np.sin(theta) * phase0  # SV polarization z-component at z=0
    expected = -4.0 * np.pi * sigma
    scale = RHO * abs(xi0_sv) * k_rho
    assert abs(az_jump - expected) <= 10.0 * eps * k_s * (scale + abs(expected))


@pytest.mark.parametrize("wave_type", ["P", "SV"])
def test_single_mode_theta_zero_gauge(wave_type):
    """At exact normal incidence, a and Gamma vanish but Phi stays a nonzero
    spatial constant for SV (the design doc's gauge convention) -- and both
    of these fall out of literally evaluating Eq. (86)-(88), no special-cased
    branch in the implementation."""
    kwargs = {
        "density_kg_m3": RHO,
        "v_p_m_s": HARD_ROCK["v_p_m_s"],
        "v_s_m_s": HARD_ROCK["v_s_m_s"],
        "frequency_hz": FREQ,
        "incidence_angle_rad": 0.0,
        "azimuth_rad": 0.3,
        "displacement_m": 1e-9,
    }
    r1 = halfspace_single_bodywave_observables_g_free(
        [[1.0, 2.0, 3.0]], wave_type=wave_type, **kwargs
    )
    r2 = halfspace_single_bodywave_observables_g_free(
        [[100.0, -50.0, 3.0]], wave_type=wave_type, **kwargs
    )
    assert np.all(r1.acceleration == 0.0)
    assert np.all(r1.gradient == 0.0)
    if wave_type == "SV":
        assert r1.potential[0] != 0.0
        assert r1.potential[0] == r2.potential[0]  # spatially constant in x,y


def test_reflected_p_theta_zero_vanishes():
    """H19: 'the total P-wave contribution ... vanishes' at theta=0 (Eq. 90's
    delta(nu, 0) = 0 exactly, evaluated literally -- not special-cased)."""
    r = halfspace_reflected_p_above_observables_g_free(
        [[1.0, 2.0, 5.0]],
        density_kg_m3=RHO,
        v_p_m_s=HARD_ROCK["v_p_m_s"],
        v_s_m_s=HARD_ROCK["v_s_m_s"],
        frequency_hz=FREQ,
        incidence_angle_rad=0.0,
        azimuth_rad=0.3,
        incident_displacement_m=1e-9 + 0.3e-9j,
    )
    assert r.potential[0] == 0.0
    assert np.all(r.acceleration == 0.0)
    assert np.all(r.gradient == 0.0)


def test_reflected_p_phase_origin_z_component_rereferences_amplitude():
    theta, azimuth = 0.6, 0.2
    xi0 = 1e-9 + 0.4e-9j
    k_p = 2.0 * np.pi * FREQ / HARD_ROCK["v_p_m_s"]
    k_zp = k_p * np.cos(theta)
    z0 = 12.0
    target = [[5.0, 3.0, 8.0]]
    kwargs = {
        "density_kg_m3": RHO,
        "v_p_m_s": HARD_ROCK["v_p_m_s"],
        "v_s_m_s": HARD_ROCK["v_s_m_s"],
        "frequency_hz": FREQ,
        "incidence_angle_rad": theta,
        "azimuth_rad": azimuth,
    }
    baseline = halfspace_reflected_p_above_observables_g_free(
        target, incident_displacement_m=xi0, **kwargs
    )
    shifted = halfspace_reflected_p_above_observables_g_free(
        target,
        incident_displacement_m=xi0 * np.exp(1j * k_zp * z0),
        phase_origin_m=(0.0, 0.0, z0),
        **kwargs,
    )
    np.testing.assert_allclose(shifted.potential, baseline.potential, rtol=1e-13)
    np.testing.assert_allclose(shifted.acceleration, baseline.acceleration, rtol=1e-13)
    np.testing.assert_allclose(shifted.gradient, baseline.gradient, rtol=1e-13)


def test_reflected_p_finite_difference_acceleration_and_gradient():
    theta, azimuth = np.deg2rad(45.0), np.deg2rad(20.0)
    kwargs = {
        "density_kg_m3": RHO,
        "v_p_m_s": HARD_ROCK["v_p_m_s"],
        "v_s_m_s": HARD_ROCK["v_s_m_s"],
        "frequency_hz": FREQ,
        "incidence_angle_rad": theta,
        "azimuth_rad": azimuth,
        "incident_displacement_m": 1e-9 + 0.3e-9j,
    }
    base = np.array([10.0, 3.0, 5.0])
    eps = 1e-3

    def phi_at(pt):
        return halfspace_reflected_p_above_observables_g_free(
            [pt.tolist()], **kwargs
        ).potential[0]

    def a_at(pt):
        return halfspace_reflected_p_above_observables_g_free(
            [pt.tolist()], **kwargs
        ).acceleration[0]

    a_fd = np.array(
        [
            -(phi_at(base + eps * e) - phi_at(base - eps * e)) / (2 * eps)
            for e in np.eye(3)
        ]
    )
    gamma_fd = np.array(
        [(a_at(base + eps * e) - a_at(base - eps * e)) / (2 * eps) for e in np.eye(3)]
    ).T  # Gamma[i,j] = d(a_i)/d(x_j) -> rows from a(pt+/-e_j), transpose to (i,j)

    r = halfspace_reflected_p_above_observables_g_free([base.tolist()], **kwargs)
    assert (
        np.abs(a_fd - r.acceleration[0]).max() <= 1e-6 * np.abs(r.acceleration[0]).max()
    )
    assert np.abs(gamma_fd - r.gradient[0]).max() <= 1e-4 * np.abs(r.gradient[0]).max()


def test_frequency_series_matches_scalar_evaluation():
    freqs = np.array([1.0, 5.0, 12.0])
    r_series = halfspace_single_bodywave_observables_g_free(
        [[1.0, 2.0, 3.0], [4.0, -5.0, -6.0]],
        wave_type="P",
        density_kg_m3=RHO,
        v_p_m_s=HARD_ROCK["v_p_m_s"],
        v_s_m_s=HARD_ROCK["v_s_m_s"],
        frequency_hz=freqs,
        incidence_angle_rad=0.5,
        azimuth_rad=0.2,
        displacement_m=1e-9 + 0.1e-9j,
    )
    for i, f in enumerate(freqs):
        r_scalar = halfspace_single_bodywave_observables_g_free(
            [[1.0, 2.0, 3.0], [4.0, -5.0, -6.0]],
            wave_type="P",
            density_kg_m3=RHO,
            v_p_m_s=HARD_ROCK["v_p_m_s"],
            v_s_m_s=HARD_ROCK["v_s_m_s"],
            frequency_hz=float(f),
            incidence_angle_rad=0.5,
            azimuth_rad=0.2,
            displacement_m=1e-9 + 0.1e-9j,
        )
        np.testing.assert_allclose(r_series.potential[i], r_scalar.potential)
        np.testing.assert_allclose(r_series.acceleration[i], r_scalar.acceleration)
        np.testing.assert_allclose(r_series.gradient[i], r_scalar.gradient)


def test_amplitude_array_matches_per_scalar_calls():
    freqs = np.array([2.0, 6.0])
    amps = np.array([1e-9, 2e-9 + 1e-9j])
    r_series = halfspace_reflected_p_above_observables_g_free(
        [[1.0, 2.0, 3.0]],
        density_kg_m3=RHO,
        v_p_m_s=HARD_ROCK["v_p_m_s"],
        v_s_m_s=HARD_ROCK["v_s_m_s"],
        frequency_hz=freqs,
        incidence_angle_rad=0.5,
        azimuth_rad=0.1,
        incident_displacement_m=amps,
    )
    for i in range(2):
        r_scalar = halfspace_reflected_p_above_observables_g_free(
            [[1.0, 2.0, 3.0]],
            density_kg_m3=RHO,
            v_p_m_s=HARD_ROCK["v_p_m_s"],
            v_s_m_s=HARD_ROCK["v_s_m_s"],
            frequency_hz=float(freqs[i]),
            incidence_angle_rad=0.5,
            azimuth_rad=0.1,
            incident_displacement_m=complex(amps[i]),
        )
        np.testing.assert_allclose(r_series.potential[i], r_scalar.potential)


def test_complex_amplitude_linearity():
    kwargs = {
        "density_kg_m3": RHO,
        "v_p_m_s": HARD_ROCK["v_p_m_s"],
        "v_s_m_s": HARD_ROCK["v_s_m_s"],
        "frequency_hz": FREQ,
        "incidence_angle_rad": 0.5,
        "azimuth_rad": 0.1,
    }
    r1 = halfspace_single_bodywave_observables_g_free(
        [[1.0, 2.0, 3.0]], wave_type="P", displacement_m=1e-9, **kwargs
    )
    r2 = halfspace_single_bodywave_observables_g_free(
        [[1.0, 2.0, 3.0]], wave_type="P", displacement_m=(2.0 + 3.0j) * 1e-9, **kwargs
    )
    np.testing.assert_allclose(r2.potential, (2.0 + 3.0j) * r1.potential)
    np.testing.assert_allclose(r2.acceleration, (2.0 + 3.0j) * r1.acceleration)
    np.testing.assert_allclose(r2.gradient, (2.0 + 3.0j) * r1.gradient)


def test_zero_amplitude_and_density_give_zero_field():
    r_amp = halfspace_single_bodywave_observables_g_free(
        [[1.0, 2.0, 3.0]],
        wave_type="SV",
        density_kg_m3=RHO,
        v_p_m_s=HARD_ROCK["v_p_m_s"],
        v_s_m_s=HARD_ROCK["v_s_m_s"],
        frequency_hz=FREQ,
        incidence_angle_rad=0.5,
        azimuth_rad=0.1,
        displacement_m=0.0,
    )
    assert np.all(r_amp.potential == 0.0)
    r_rho = halfspace_single_bodywave_observables_g_free(
        [[1.0, 2.0, 3.0]],
        wave_type="P",
        density_kg_m3=0.0,
        v_p_m_s=HARD_ROCK["v_p_m_s"],
        v_s_m_s=HARD_ROCK["v_s_m_s"],
        frequency_hz=FREQ,
        incidence_angle_rad=0.5,
        azimuth_rad=0.1,
        displacement_m=1e-9,
    )
    assert np.all(r_rho.potential == 0.0)


@pytest.mark.parametrize("wave_type", ["P", "SV"])
def test_phase_origin_z_component_rereferences_amplitude(wave_type):
    """phase_origin_m[2] must re-reference displacement_m to z=0 (via the
    incident wave's own k_z), not silently no-op -- an amplitude given at a
    shifted phase origin z0, pre-multiplied by exp(+i*k_z*z0), must produce
    the same field as the same physical amplitude quoted at z=0."""
    theta, azimuth = 0.5, 0.2
    xi0 = 1e-9 + 0.4e-9j
    v = HARD_ROCK["v_p_m_s"] if wave_type == "P" else HARD_ROCK["v_s_m_s"]
    k_total = 2.0 * np.pi * FREQ / v
    k_z = k_total * np.cos(theta)
    z0 = 12.0
    target = [[5.0, 3.0, 8.0]]

    baseline = halfspace_single_bodywave_observables_g_free(
        target,
        wave_type=wave_type,
        density_kg_m3=RHO,
        v_p_m_s=HARD_ROCK["v_p_m_s"],
        v_s_m_s=HARD_ROCK["v_s_m_s"],
        frequency_hz=FREQ,
        incidence_angle_rad=theta,
        azimuth_rad=azimuth,
        displacement_m=xi0,
    )
    shifted = halfspace_single_bodywave_observables_g_free(
        target,
        wave_type=wave_type,
        density_kg_m3=RHO,
        v_p_m_s=HARD_ROCK["v_p_m_s"],
        v_s_m_s=HARD_ROCK["v_s_m_s"],
        frequency_hz=FREQ,
        incidence_angle_rad=theta,
        azimuth_rad=azimuth,
        displacement_m=xi0 * np.exp(1j * k_z * z0),
        phase_origin_m=(0.0, 0.0, z0),
    )
    np.testing.assert_allclose(shifted.potential, baseline.potential, rtol=1e-13)
    np.testing.assert_allclose(shifted.acceleration, baseline.acceleration, rtol=1e-13)
    np.testing.assert_allclose(shifted.gradient, baseline.gradient, rtol=1e-13)

    # z0=0 is a strict no-op (exp(-i*k_z*0) == 1 exactly).
    explicit_zero = halfspace_single_bodywave_observables_g_free(
        target,
        wave_type=wave_type,
        density_kg_m3=RHO,
        v_p_m_s=HARD_ROCK["v_p_m_s"],
        v_s_m_s=HARD_ROCK["v_s_m_s"],
        frequency_hz=FREQ,
        incidence_angle_rad=theta,
        azimuth_rad=azimuth,
        displacement_m=xi0,
        phase_origin_m=(0.0, 0.0, 0.0),
    )
    assert np.array_equal(explicit_zero.potential, baseline.potential)


def test_readonly_output_arrays():
    r = halfspace_single_bodywave_observables_g_free(
        [[1.0, 2.0, 3.0]],
        wave_type="P",
        density_kg_m3=RHO,
        v_p_m_s=HARD_ROCK["v_p_m_s"],
        v_s_m_s=HARD_ROCK["v_s_m_s"],
        frequency_hz=FREQ,
        incidence_angle_rad=0.5,
        azimuth_rad=0.1,
        displacement_m=1e-9,
    )
    for arr in (r.potential, r.acceleration, r.gradient):
        assert arr.flags.writeable is False


_BASE_SINGLE_KWARGS = {
    "targets_m": [[1.0, 2.0, 3.0]],
    "wave_type": "P",
    "density_kg_m3": RHO,
    "v_p_m_s": HARD_ROCK["v_p_m_s"],
    "v_s_m_s": HARD_ROCK["v_s_m_s"],
    "frequency_hz": FREQ,
    "incidence_angle_rad": 0.5,
    "azimuth_rad": 0.1,
    "displacement_m": 1e-9,
}


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"wave_type": "S"}, "wave_type"),
        ({"targets_m": [[1.0, 2.0, 0.0]]}, "z == 0"),
        ({"density_kg_m3": -1.0}, "density_kg_m3"),
        ({"v_p_m_s": 1000.0, "v_s_m_s": 2000.0}, "v_s_m_s"),
        ({"v_p_m_s": 1000.0, "v_s_m_s": 900.0}, "4/3"),
        ({"frequency_hz": 0.0}, "frequency_hz"),
        ({"frequency_hz": -1.0}, "frequency_hz"),
        ({"frequency_hz": []}, "frequency_hz"),
        ({"incidence_angle_rad": -0.1}, "incidence_angle_rad"),
        ({"incidence_angle_rad": np.pi / 2.0}, "incidence_angle_rad"),
        ({"incidence_angle_rad": np.pi}, "incidence_angle_rad"),
        ({"displacement_m": [1e-9, 2e-9]}, "displacement_m"),
    ],
)
def test_invalid_single_mode_inputs_raise_value_error(overrides, match):
    kwargs = {**_BASE_SINGLE_KWARGS, **overrides}
    with pytest.raises(ValueError, match=match):
        halfspace_single_bodywave_observables_g_free(**kwargs)


@pytest.mark.parametrize(
    "field", ["targets_m", "incidence_angle_rad", "azimuth_rad", "phase_origin_m"]
)
def test_complex_input_rejected_for_real_only_fields(field):
    kwargs = dict(_BASE_SINGLE_KWARGS)
    if field == "targets_m":
        kwargs["targets_m"] = [[1.0 + 1.0j, 2.0, 3.0]]
    elif field == "phase_origin_m":
        kwargs["phase_origin_m"] = [1.0j, 0.0, 0.0]
    else:
        kwargs[field] = 0.5 + 0.1j
    with pytest.raises(TypeError):
        halfspace_single_bodywave_observables_g_free(**kwargs)


@pytest.mark.parametrize(
    "field", ["incidence_angle_rad", "azimuth_rad", "frequency_hz"]
)
def test_bool_input_rejected_for_real_only_fields(field):
    kwargs = dict(_BASE_SINGLE_KWARGS)
    kwargs[field] = True
    with pytest.raises(TypeError):
        halfspace_single_bodywave_observables_g_free(**kwargs)


def test_reflected_p_rejects_underground_and_surface_targets():
    kwargs = {
        "density_kg_m3": RHO,
        "v_p_m_s": HARD_ROCK["v_p_m_s"],
        "v_s_m_s": HARD_ROCK["v_s_m_s"],
        "frequency_hz": FREQ,
        "incidence_angle_rad": 0.5,
        "azimuth_rad": 0.1,
        "incident_displacement_m": 1e-9,
    }
    with pytest.raises(ValueError, match="z > 0"):
        halfspace_reflected_p_above_observables_g_free([[1.0, 2.0, -3.0]], **kwargs)
    with pytest.raises(ValueError, match="z == 0"):
        halfspace_reflected_p_above_observables_g_free([[1.0, 2.0, 0.0]], **kwargs)


def test_reflected_p_rejects_grazing_incidence():
    with pytest.raises(ValueError, match="incidence_angle_rad"):
        halfspace_reflected_p_above_observables_g_free(
            [[1.0, 2.0, 3.0]],
            density_kg_m3=RHO,
            v_p_m_s=HARD_ROCK["v_p_m_s"],
            v_s_m_s=HARD_ROCK["v_s_m_s"],
            frequency_hz=FREQ,
            incidence_angle_rad=np.pi / 2.0,
            azimuth_rad=0.1,
            incident_displacement_m=1e-9,
        )
