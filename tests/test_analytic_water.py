"""Tests for analytical water Newtonian noise models (R2 Water NN).

Covers:
- Model 2: Frozen advection channel (transfer, observables, ASD, strain, DC limits)
- Waterfall model: Vertical waterfall (transfer, observables, ASD, strain, DC limits)
- Model 1: Uncorrelated surface cell model (coupling factor, ASD)
- Edge effect: Channel edge single-event deterministic Fourier spectrum
- Small-q asymptotic precision against mpmath, independent tensor quadrature,
  velocity reversal symmetry, DC contract, shape contracts, and G_SI scaling.
"""

from __future__ import annotations

import numpy as np
import pytest

from gragra.analytic.values import GFreeGravityValues
from gragra.analytic.water import (
    water_channel_advection_ax_transfer_g_free,
    water_channel_advection_dc_limit_g_free,
    water_channel_advection_observables_g_free,
    water_channel_advection_transfer_g_free,
    water_channel_uncorrelated_cell_coupling_g_free,
    water_waterfall_transfer_g_free,
)
from gragra.constants import G_SI
from gragra.observables.analytic import GravityObservables
from gragra.observables.water import (
    water_channel_advection_asd,
    water_channel_advection_dc_limit,
    water_channel_advection_observables,
    water_channel_advection_strain_asd,
    water_channel_kolmogorov_surface_asd,
    water_channel_uncorrelated_asd,
    water_edge_step_fourier_amplitude,
    water_edge_step_strain_fourier_amplitude,
    water_waterfall_asd,
    water_waterfall_strain_asd,
)

# ---------------------------------------------------------------------------
# Model 2: Frozen Advection Channel
# ---------------------------------------------------------------------------


def test_channel_advection_dc_limits_and_explicit_functions():
    """Verify DC limit behavior and dedicated DC limit functions."""
    v = 2.0
    d = 3.0
    w = 0.5
    rho = 1000.0

    # Dedicated G-free DC limit function
    acc_dc_g, grad_dc_g = water_channel_advection_dc_limit_g_free(
        distance_m=d, width_m=w, density_kg_m3=rho, eta_m=1.0
    )
    assert acc_dc_g[0] == 0.0
    assert acc_dc_g[1] == 0.0
    expected_az_g = -2.0 * rho * w / d
    np.testing.assert_allclose(acc_dc_g[2], expected_az_g, rtol=1e-14)

    np.testing.assert_allclose(grad_dc_g[0, 0], 0.0, atol=1e-14)
    np.testing.assert_allclose(grad_dc_g[1, 1], -2.0 * rho * w / (d**2), rtol=1e-14)
    np.testing.assert_allclose(grad_dc_g[2, 2], 2.0 * rho * w / (d**2), rtol=1e-14)
    np.testing.assert_allclose(np.trace(grad_dc_g), 0.0, atol=1e-14)

    # Dedicated SI DC limit function
    acc_dc_si, grad_dc_si = water_channel_advection_dc_limit(
        distance_m=d, width_m=w, density_kg_m3=rho, eta_m=1.0
    )
    np.testing.assert_allclose(acc_dc_si, G_SI * acc_dc_g, rtol=1e-14)
    np.testing.assert_allclose(grad_dc_si, G_SI * grad_dc_g, rtol=1e-14)

    # Transfer function with allow_dc=True returns potential=-inf
    pot, acc, grad = water_channel_advection_transfer_g_free(
        frequency_hz=0.0,
        velocity_mps=v,
        distance_m=d,
        width_m=w,
        density_kg_m3=rho,
        allow_dc=True,
    )
    assert np.isneginf(pot.real)
    np.testing.assert_allclose(acc[2].real, expected_az_g, rtol=1e-14)
    np.testing.assert_allclose(np.trace(grad).real, 0.0, atol=1e-14)

    # Default transfer function rejects DC without allow_dc=True
    with pytest.raises(ValueError, match="diverges logarithmically at DC"):
        water_channel_advection_transfer_g_free(
            frequency_hz=0.0,
            velocity_mps=v,
            distance_m=d,
            width_m=w,
            density_kg_m3=rho,
        )


def test_channel_advection_dc_and_zero_eta_contract():
    """Verify value container contract: reject DC, eta=0 gives zeros without NaN."""
    # Calling observables with frequency_hz=0 must raise ValueError
    with pytest.raises(ValueError, match="frequency_hz must be strictly positive"):
        water_channel_advection_observables(
            frequency_hz=0.0,
            velocity_mps=2.0,
            distance_m=3.0,
            width_m=0.5,
        )

    # Calling observables with eta_m=0.0 at f > 0 must return finite zeros, NOT NaN
    obs_zero = water_channel_advection_observables(
        frequency_hz=1.0,
        velocity_mps=2.0,
        distance_m=3.0,
        width_m=0.5,
        eta_m=0.0,
    )
    assert np.all(np.isfinite(obs_zero.potential))
    assert np.all(np.isfinite(obs_zero.acceleration))
    assert np.all(np.isfinite(obs_zero.gradient))
    np.testing.assert_allclose(obs_zero.potential, 0.0, atol=1e-15)
    np.testing.assert_allclose(obs_zero.acceleration, 0.0, atol=1e-15)
    np.testing.assert_allclose(obs_zero.gradient, 0.0, atol=1e-15)


def test_channel_advection_positive_frequencies_trace_and_symmetry():
    """Verify Trace(Gamma) == 0 and tensor symmetry for positive frequencies."""
    frequencies = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    v = 1.5
    d = 2.5
    w = 0.4
    rho = 1000.0

    pot, acc, grad = water_channel_advection_transfer_g_free(
        frequency_hz=frequencies,
        velocity_mps=v,
        distance_m=d,
        width_m=w,
        density_kg_m3=rho,
    )

    assert pot.shape == (len(frequencies),)
    assert acc.shape == (len(frequencies), 3)
    assert grad.shape == (len(frequencies), 3, 3)

    assert pot.flags.writeable is False
    assert acc.flags.writeable is False
    assert grad.flags.writeable is False

    for i in range(len(frequencies)):
        g = grad[i]
        tr = np.trace(g)
        norm_g = np.linalg.norm(g)
        np.testing.assert_allclose(np.abs(tr), 0.0, atol=1e-15 * norm_g)

        # Symmetry: Gamma_xz == Gamma_zx
        np.testing.assert_allclose(g[0, 2], g[2, 0], rtol=1e-15)

        # Off-diagonals with y are 0 by reflection symmetry
        assert g[0, 1] == 0.0
        assert g[1, 0] == 0.0
        assert g[1, 2] == 0.0
        assert g[2, 1] == 0.0

        # Acceleration phases:
        # a_x is purely imaginary (+i for v > 0, omega > 0)
        assert acc[i, 0].real == 0.0
        assert acc[i, 0].imag > 0.0

        # a_y is zero
        assert acc[i, 1] == 0.0

        # a_z is purely real and negative (pull towards channel at z = -D)
        assert acc[i, 2].imag == 0.0
        assert acc[i, 2].real < 0.0


def test_channel_advection_velocity_reversal():
    """Verify velocity reversal v -> -v for all acceleration and gradient components."""
    f = 1.0
    d = 2.0
    w = 0.4
    rho = 1000.0

    _, acc_pos, grad_pos = water_channel_advection_transfer_g_free(f, 1.5, d, w, rho)
    _, acc_neg, grad_neg = water_channel_advection_transfer_g_free(f, -1.5, d, w, rho)

    # a_x flips sign (odd in v)
    np.testing.assert_allclose(acc_neg[0], -acc_pos[0], rtol=1e-14)
    # a_z is unchanged (even in v)
    np.testing.assert_allclose(acc_neg[2], acc_pos[2], rtol=1e-14)
    # Gamma_xz flips sign (odd in v)
    np.testing.assert_allclose(grad_neg[0, 2], -grad_pos[0, 2], rtol=1e-14)
    np.testing.assert_allclose(grad_neg[2, 0], -grad_pos[2, 0], rtol=1e-14)
    # Diagonal components unchanged (even in v)
    np.testing.assert_allclose(grad_neg[0, 0], grad_pos[0, 0], rtol=1e-14)
    np.testing.assert_allclose(grad_neg[1, 1], grad_pos[1, 1], rtol=1e-14)
    np.testing.assert_allclose(grad_neg[2, 2], grad_pos[2, 2], rtol=1e-14)


def test_channel_advection_exponential_suppression():
    """Verify exponential suppression K_0(q) ~ exp(-q) at large q = Omega * D / v."""
    v = 1.0
    d = 5.0
    w = 0.4
    rho = 1000.0

    _, acc1, _ = water_channel_advection_transfer_g_free(1.0, v, d, w, rho)
    _, acc2, _ = water_channel_advection_transfer_g_free(2.0, v, d, w, rho)

    mag1 = np.abs(acc1[0])
    mag2 = np.abs(acc2[0])

    ratio = mag2 / mag1
    expected_ratio_order = np.exp(-2.0 * np.pi * 1.0 * d / v)
    assert ratio < 1e-10
    np.testing.assert_allclose(np.log(ratio), np.log(expected_ratio_order), rtol=0.1)


def test_channel_advection_small_q_asymptotic_precision():
    """Verify high-precision small-q asymptotics against mpmath arbitrary precision."""
    import mpmath as mp

    v = 2.0
    d = 2.5
    w = 0.4
    rho = 1000.0

    for q_target in [1e-16, 1e-18, 1e-25]:
        f = q_target * v / (2.0 * np.pi * d)
        pot, acc, grad = water_channel_advection_transfer_g_free(
            frequency_hz=f,
            velocity_mps=v,
            distance_m=d,
            width_m=w,
            density_kg_m3=rho,
        )

        with mp.workdps(50):
            mp_q = mp.mpf(str(q_target))
            mp_d = mp.mpf(str(d))
            mp_c0 = mp.mpf(str(2.0 * rho * w))
            mp_k0 = mp.besselk(0, mp_q)
            mp_k1 = mp.besselk(1, mp_q)

            ref_pot = float(-mp_c0 * mp_k0)
            ref_ax_imag = float(mp_c0 * (mp_q / mp_d) * mp_k0)
            ref_az_real = float(-mp_c0 * (mp_q * mp_k1 / mp_d))
            ref_gxx = float(-mp_c0 * ((mp_q / mp_d) ** 2) * mp_k0)
            ref_gyy = float(-mp_c0 * (mp_q * mp_k1 / (mp_d**2)))
            ref_gzz = float(
                mp_c0 * (((mp_q / mp_d) ** 2) * mp_k0 + (mp_q * mp_k1 / (mp_d**2)))
            )
            ref_gxz_imag = float(-mp_c0 * (mp_q * (mp_q * mp_k1) / (mp_d**2)))

        np.testing.assert_allclose(pot.real, ref_pot, rtol=1e-12)
        assert acc[0].imag != 0.0
        np.testing.assert_allclose(acc[0].imag, ref_ax_imag, rtol=1e-12)
        np.testing.assert_allclose(acc[2].real, ref_az_real, rtol=1e-12)
        np.testing.assert_allclose(grad[0, 0].real, ref_gxx, rtol=1e-12)
        np.testing.assert_allclose(grad[1, 1].real, ref_gyy, rtol=1e-12)
        np.testing.assert_allclose(grad[2, 2].real, ref_gzz, rtol=1e-12)
        assert grad[0, 2].imag != 0.0
        np.testing.assert_allclose(grad[0, 2].imag, ref_gxz_imag, rtol=1e-12)


def test_subnormal_positive_bessel_argument_uses_finite_asymptotics():
    """A positive subnormal q must not form q/2 == 0 in the log expansion."""
    f = 1e-323
    d = 0.1

    channel_result = water_channel_advection_transfer_g_free(f, 1.0, d, 1.0)
    waterfall_result = water_waterfall_transfer_g_free(f, 1.0, d)
    assert all(np.all(np.isfinite(component)) for component in channel_result)
    assert all(np.all(np.isfinite(component)) for component in waterfall_result)
    assert channel_result[1][0] != 0.0
    assert channel_result[2][0, 2] != 0.0

    q = np.nextafter(0.0, 1.0)
    expected_k0 = -np.log(q) + np.log(2.0) - 0.5772156649015329
    pot, _, _ = water_channel_advection_transfer_g_free(f, 1.0, d, 1.0)
    np.testing.assert_allclose(pot.real, -2.0 * 1000.0 * expected_k0, rtol=1e-14)


def test_positive_frequency_argument_underflow_is_rejected_not_treated_as_dc():
    """A positive f whose dimensionless argument underflows must raise explicitly."""
    with pytest.raises(ValueError, match="underflow"):
        water_channel_advection_transfer_g_free(1e-300, 1.0, 1e-100, 1.0)

    with pytest.raises(ValueError, match="underflow"):
        water_channel_advection_ax_transfer_g_free(1e-300, 1.0, 1e-100, 1.0)

    with pytest.raises(ValueError, match="underflow"):
        water_waterfall_transfer_g_free(1e-300, 1.0, 1e-100)


def test_unsupported_large_bessel_argument_raises_without_nan():
    """Arguments beyond the documented float64 range are rejected explicitly."""
    with pytest.raises(ValueError, match="supported range"):
        water_channel_advection_transfer_g_free(1e300, 1.0, 3.0, 1.0)

    with pytest.raises(ValueError, match="supported range"):
        water_channel_advection_ax_transfer_g_free(1e300, 1.0, 3.0, 1.0)

    with pytest.raises(ValueError, match="supported range"):
        water_waterfall_transfer_g_free(1e300, 1.0, 3.0)


def test_largest_supported_bessel_argument_returns_finite_transfers():
    """The q=700 boundary remains supported by the unscaled SciPy functions."""
    frequency_hz = 700.0 / (2.0 * np.pi)
    channel = water_channel_advection_transfer_g_free(frequency_hz, 1.0, 1.0, 1.0)
    waterfall = water_waterfall_transfer_g_free(frequency_hz, 1.0, 1.0)

    assert all(np.all(np.isfinite(component)) for component in channel)
    assert all(np.all(np.isfinite(component)) for component in waterfall)


def test_dc_limit_rejects_unrepresentable_physical_outputs():
    """Extreme finite geometry must raise instead of returning infinite DC values."""
    with pytest.raises(ValueError, match="finite float64"):
        water_channel_advection_dc_limit_g_free(
            distance_m=1e-200,
            width_m=1.0,
            density_kg_m3=1000.0,
        )


def test_channel_advection_observables_and_g_scaling():
    """Verify SI observables match G_SI * G-free values."""
    frequencies = np.array([1.0, 2.0])
    v = 2.0
    d = 2.0
    w = 0.5
    rho = 1000.0
    eta = 0.01 + 0.005j

    g_free_vals = water_channel_advection_observables_g_free(
        frequencies, v, d, w, rho, eta_m=eta
    )
    assert isinstance(g_free_vals, GFreeGravityValues)

    si_obs = water_channel_advection_observables(frequencies, v, d, w, rho, eta_m=eta)
    assert isinstance(si_obs, GravityObservables)

    np.testing.assert_allclose(
        si_obs.potential, G_SI * g_free_vals.potential, rtol=1e-15
    )
    np.testing.assert_allclose(
        si_obs.acceleration, G_SI * g_free_vals.acceleration, rtol=1e-15
    )
    np.testing.assert_allclose(si_obs.gradient, G_SI * g_free_vals.gradient, rtol=1e-15)


def test_channel_advection_eta_shape_contract():
    """Verify eta_m shape contract: scalar or matching 1D array."""
    f = np.array([1.0, 2.0, 3.0])
    v = 2.0
    d = 2.0
    w = 0.4

    # Scalar eta works
    obs_scalar = water_channel_advection_observables(f, v, d, w, eta_m=0.01)
    assert obs_scalar.potential.shape == (3,)

    # 1D array matching f shape works
    obs_arr = water_channel_advection_observables(
        f, v, d, w, eta_m=np.array([0.01, 0.02, 0.03])
    )
    assert obs_arr.potential.shape == (3,)

    # Mismatched shape raises ValueError
    with pytest.raises(ValueError, match="must match frequency_hz shape"):
        water_channel_advection_observables(f, v, d, w, eta_m=np.array([0.01, 0.02]))


def test_channel_advection_asd_and_strain_asd():
    """Verify ASD and strain ASD calculation."""
    f = np.array([1.0, 2.0, 3.0])
    v = 2.0
    d = 2.0
    w = 0.4
    surface_asd = 0.01  # 1 cm / sqrt(Hz)

    acc_asd_x = water_channel_advection_asd(f, v, d, w, surface_asd, component="x")
    acc_asd_z = water_channel_advection_asd(f, v, d, w, surface_asd, component="z")
    acc_asd_all = water_channel_advection_asd(f, v, d, w, surface_asd, component="all")

    assert acc_asd_x.shape == (3,)
    assert acc_asd_z.shape == (3,)
    assert acc_asd_all.shape == (3, 3)
    np.testing.assert_allclose(acc_asd_all[:, 0], acc_asd_x, rtol=1e-15)
    np.testing.assert_allclose(acc_asd_all[:, 2], acc_asd_z, rtol=1e-15)

    strain_asd = water_channel_advection_strain_asd(
        f, v, d, w, surface_asd, arm_length_m=3000.0
    )
    omega = 2.0 * np.pi * f
    expected_strain = acc_asd_x / ((omega**2) * 3000.0)
    np.testing.assert_allclose(strain_asd, expected_strain, rtol=1e-15)


def test_channel_kolmogorov_surface_asd():
    """Verify Kolmogorov water surface ASD formula from Nishizawa 2021 Eq. (10)."""
    f = 1.0
    v = 2.0
    ell = 0.4
    dv = 0.3

    sb_asd = water_channel_kolmogorov_surface_asd(f, v, ell, dv)
    expected = (
        (0.3 / (2.0 * np.pi)) * ((0.4 / 2.0) ** (2.0 / 3.0)) * (1.0 ** (-5.0 / 6.0))
    )
    np.testing.assert_allclose(sb_asd, expected, rtol=1e-14)
    np.testing.assert_allclose(sb_asd, 0.016327, atol=1e-4)


# ---------------------------------------------------------------------------
# Waterfall Model (Nishizawa 2021)
# ---------------------------------------------------------------------------


def test_waterfall_dc_and_trace():
    """Verify Waterfall DC limit and Trace(Gamma) == 0."""
    v = 3.0
    d = 2.0
    rho = 1000.0

    pot0, acc0, grad0 = water_waterfall_transfer_g_free(0.0, v, d, rho, allow_dc=True)
    np.testing.assert_allclose(acc0[0].real, -2.0 * rho / d, rtol=1e-14)
    assert acc0[2] == 0.0 + 0.0j
    np.testing.assert_allclose(np.trace(grad0).real, 0.0, atol=1e-14)

    # Positive frequency
    f = 1.5
    pot, acc, grad = water_waterfall_transfer_g_free(f, v, d, rho)
    np.testing.assert_allclose(np.trace(grad), 0.0, atol=1e-14 * np.linalg.norm(grad))
    assert acc[0].imag == 0.0  # a_x is real
    assert acc[2].real == 0.0  # a_z is imaginary


def test_waterfall_asd_and_strain():
    """Verify Waterfall ASD and strain ASD."""
    f = 2.0
    v = 2.0
    d = 2.0
    area_asd = 54e-4  # m^2 / sqrt(Hz)

    acc_asd = water_waterfall_asd(f, v, d, area_asd)
    strain_asd = water_waterfall_strain_asd(f, v, d, area_asd, arm_length_m=3000.0)

    omega = 2.0 * np.pi * f
    expected_strain = acc_asd / ((omega**2) * 3000.0)
    np.testing.assert_allclose(strain_asd, expected_strain, rtol=1e-15)


# ---------------------------------------------------------------------------
# Edge Effect Model: Single-Event Deterministic Fourier Spectrum
# ---------------------------------------------------------------------------


def test_edge_step_deterministic_fourier_amplitude():
    """Verify single-event deterministic Fourier amplitude [m/s] scales as 1/f."""
    x0 = 5.0
    d = 2.0
    volume = 0.4 * 0.005 * 0.01  # w * b0 * Delta_x
    f = 5.0
    rho = 1000.0

    amp_acc = water_edge_step_fourier_amplitude(x0, d, volume, f, density_kg_m3=rho)
    amp_strain = water_edge_step_strain_fourier_amplitude(
        x0, d, volume, f, arm_length_m=3000.0, density_kg_m3=rho
    )

    omega = 2.0 * np.pi * f
    r_cubed = (d**2 + x0**2) ** 1.5
    # Nishizawa Eq. 6: single event Fourier transform magnitude
    expected_amp_acc = G_SI * rho * volume * x0 / (omega * r_cubed)
    np.testing.assert_allclose(amp_acc, expected_amp_acc, rtol=1e-15)

    expected_amp_strain = expected_amp_acc / ((omega**2) * 3000.0)
    np.testing.assert_allclose(amp_strain, expected_amp_strain, rtol=1e-15)


# ---------------------------------------------------------------------------
# Cross-Validation against Independent Science Oracle
# ---------------------------------------------------------------------------


def test_independent_oracle_verification_suite():
    """Verify that the independent science oracle itself passes its internal checks."""
    from tests.oracles.water_analytic_reference import verify_water_analytic_reference

    results = verify_water_analytic_reference()
    assert results["all_passed"] is True


def test_production_vs_oracle_model2_individual_gradient_components():
    """Cross-validate EVERY tensor component against spatial quadrature."""
    from tests.oracles.water_analytic_reference import model2_spatial_quadrature

    v = 2.0
    d = 2.0
    w = 0.4
    rho = 1000.0
    f = 1.0
    k = 2.0 * np.pi * f / v

    _, acc_prod, grad_prod = water_channel_advection_transfer_g_free(
        frequency_hz=f,
        velocity_mps=v,
        distance_m=d,
        width_m=w,
        density_kg_m3=rho,
    )

    # Cross-check individual components against spatial quadrature (g_const=1.0)
    quad_ax = model2_spatial_quadrature("a_x", k, d, rho=rho, w=w, g_const=1.0)
    quad_az = model2_spatial_quadrature("a_z", k, d, rho=rho, w=w, g_const=1.0)
    quad_gxx = model2_spatial_quadrature("gamma_xx", k, d, rho=rho, w=w, g_const=1.0)
    quad_gyy = model2_spatial_quadrature("gamma_yy", k, d, rho=rho, w=w, g_const=1.0)
    quad_gzz = model2_spatial_quadrature("gamma_zz", k, d, rho=rho, w=w, g_const=1.0)
    quad_gxz = model2_spatial_quadrature("gamma_xz", k, d, rho=rho, w=w, g_const=1.0)

    # Relative tolerance 1e-4 between quadrature and analytic closed form
    np.testing.assert_allclose(acc_prod[0], quad_ax, rtol=1e-4)
    np.testing.assert_allclose(acc_prod[2], quad_az, rtol=1e-4)
    np.testing.assert_allclose(grad_prod[0, 0], quad_gxx, rtol=1e-4)
    np.testing.assert_allclose(grad_prod[1, 1], quad_gyy, rtol=1e-4)
    np.testing.assert_allclose(grad_prod[2, 2], quad_gzz, rtol=1e-4)
    np.testing.assert_allclose(grad_prod[0, 2], quad_gxz, rtol=1e-4)


def test_production_vs_oracle_model2_bessel_closed_forms():
    """Compare production G-free transfer against independent mpmath Bessel oracle."""
    from tests.oracles.water_analytic_reference import model2_bessel_reference

    v = 2.0
    d = 2.5
    w = 0.4
    rho = 1000.0

    for f in [0.2, 0.5, 1.0, 2.5, 5.0]:
        k = 2.0 * np.pi * f / v
        pot_prod, acc_prod, grad_prod = water_channel_advection_transfer_g_free(
            frequency_hz=f,
            velocity_mps=v,
            distance_m=d,
            width_m=w,
            density_kg_m3=rho,
        )

        ref_phi = model2_bessel_reference(
            "potential", k, d, rho=rho, w=w, g_const=1.0, use_mpmath=True
        )
        ref_ax = model2_bessel_reference(
            "a_x", k, d, rho=rho, w=w, g_const=1.0, use_mpmath=True
        )
        ref_az = model2_bessel_reference(
            "a_z", k, d, rho=rho, w=w, g_const=1.0, use_mpmath=True
        )
        ref_gxx = model2_bessel_reference(
            "gamma_xx", k, d, rho=rho, w=w, g_const=1.0, use_mpmath=True
        )
        ref_gyy = model2_bessel_reference(
            "gamma_yy", k, d, rho=rho, w=w, g_const=1.0, use_mpmath=True
        )
        ref_gzz = model2_bessel_reference(
            "gamma_zz", k, d, rho=rho, w=w, g_const=1.0, use_mpmath=True
        )
        ref_gxz = model2_bessel_reference(
            "gamma_xz", k, d, rho=rho, w=w, g_const=1.0, use_mpmath=True
        )

        np.testing.assert_allclose(pot_prod, ref_phi, rtol=1e-12)
        np.testing.assert_allclose(acc_prod[0], ref_ax, rtol=1e-12)
        np.testing.assert_allclose(acc_prod[2], ref_az, rtol=1e-12)
        np.testing.assert_allclose(grad_prod[0, 0], ref_gxx, rtol=1e-12)
        np.testing.assert_allclose(grad_prod[1, 1], ref_gyy, rtol=1e-12)
        np.testing.assert_allclose(grad_prod[2, 2], ref_gzz, rtol=1e-12)
        np.testing.assert_allclose(grad_prod[0, 2], ref_gxz, rtol=1e-12)


def test_production_vs_oracle_waterfall():
    """Compare production Waterfall transfer against independent oracle."""
    from tests.oracles.water_analytic_reference import waterfall_bessel_reference

    v = 3.0
    d = 2.0
    rho = 1000.0

    for f in [0.5, 1.0, 3.0]:
        k = 2.0 * np.pi * f / v
        _, acc_prod, _ = water_waterfall_transfer_g_free(f, v, d, rho)
        ref_ax, ref_az = waterfall_bessel_reference(
            k, d, rho=rho, g_const=1.0, use_mpmath=True
        )

        np.testing.assert_allclose(acc_prod[0], ref_ax, rtol=1e-12)
        np.testing.assert_allclose(acc_prod[2], ref_az, rtol=1e-12)


def test_production_vs_oracle_model1_uncorrelated_multi_width():
    """Verify Model 1 PSD and ASD match independent oracle across multiple widths."""
    from tests.oracles.water_analytic_reference import (
        model1_cell_quadrature,
        model1_closed_form,
    )

    rho = 1000.0
    a_cell = 0.04  # 0.2 x 0.2 m^2
    surface_asd = 0.005  # 5 mm / rtHz

    for h in [1.0, 2.5, 5.0]:
        for w in [0.1, 0.4, 1.0, 2.5]:
            # Production SI ASD
            asd_prod = water_channel_uncorrelated_asd(
                distance_m=h,
                width_m=w,
                correlation_area_m2=a_cell,
                surface_asd_m_per_rt_hz=surface_asd,
                density_kg_m3=rho,
            )
            psd_prod = asd_prod**2

            # Oracle closed form PSD: (G rho)^2 * a_cell * w * S_eta * (pi / (8 H^3))
            psd_oracle_closed = model1_closed_form(
                depth_h=h,
                w=w,
                a_cell=a_cell,
                rho=rho,
                g_const=G_SI,
                s_eta=surface_asd**2,
            )
            # Oracle numerical quadrature PSD
            psd_oracle_quad = model1_cell_quadrature(
                depth_h=h,
                w=w,
                a_cell=a_cell,
                rho=rho,
                g_const=G_SI,
                s_eta=surface_asd**2,
            )

            np.testing.assert_allclose(psd_prod, psd_oracle_closed, rtol=1e-14)
            np.testing.assert_allclose(psd_prod, psd_oracle_quad, rtol=1e-9)
            np.testing.assert_allclose(asd_prod, np.sqrt(psd_oracle_closed), rtol=1e-14)


# ---------------------------------------------------------------------------
# Adversarial Input Validation Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "invalid_kwargs",
    [
        {"velocity_mps": 0.0},
        {"distance_m": 0.0},
        {"distance_m": -1.0},
        {"width_m": 0.0},
        {"density_kg_m3": 0.0},
        {"frequency_hz": -0.1},
        {"frequency_hz": np.nan},
        {"frequency_hz": np.inf},
    ],
)
def test_channel_advection_invalid_inputs(invalid_kwargs):
    kwargs = {
        "frequency_hz": 1.0,
        "velocity_mps": 2.0,
        "distance_m": 2.0,
        "width_m": 0.4,
    }
    kwargs.update(invalid_kwargs)
    with pytest.raises(ValueError):
        water_channel_advection_transfer_g_free(**kwargs)


def test_channel_advection_type_errors():
    with pytest.raises(TypeError):
        water_channel_advection_transfer_g_free(1.0, True, 2.0, 0.4)
    with pytest.raises(TypeError):
        water_channel_advection_transfer_g_free(1.0, 2.0, "invalid", 0.4)


def test_adversarial_asd_inputs():
    """Verify ASD functions reject negative, NaN, Inf, and invalid parameters."""
    with pytest.raises(ValueError, match="non-negative"):
        water_channel_advection_asd(
            frequency_hz=1.0,
            velocity_mps=2.0,
            distance_m=2.0,
            width_m=0.4,
            surface_asd_m_per_rt_hz=-0.01,
        )

    with pytest.raises(ValueError, match="finite"):
        water_channel_advection_asd(
            frequency_hz=1.0,
            velocity_mps=2.0,
            distance_m=2.0,
            width_m=0.4,
            surface_asd_m_per_rt_hz=np.nan,
        )

    with pytest.raises(ValueError, match="strictly positive"):
        water_channel_advection_strain_asd(
            frequency_hz=1.0,
            velocity_mps=2.0,
            distance_m=2.0,
            width_m=0.4,
            surface_asd_m_per_rt_hz=0.01,
            arm_length_m=-100.0,
        )

    with pytest.raises(ValueError, match="strictly positive"):
        water_channel_kolmogorov_surface_asd(
            frequency_hz=1.0,
            velocity_mps=-2.0,
            largest_turb_scale_m=0.4,
            turb_velocity_fluc_mps=0.3,
        )

    with pytest.raises(ValueError, match="finite"):
        water_edge_step_fourier_amplitude(
            x0_m=np.nan,
            distance_m=2.0,
            volume_m3=0.01,
            frequency_hz=1.0,
        )
