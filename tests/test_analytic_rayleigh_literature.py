"""Tests for the literature-sourced Rayleigh half-space gravity (LIT-R).

Primary source: H19 (Harms, Living Reviews in Relativity 22, 6 (2019)),
Eq. (96)-(98). See ``docs/design/literature-r.md`` and
``docs/validation/literature-r.md`` for the full equation mapping and
independent-verification report; this file implements that plan's test
matrix.

Independent verification strategy actually used (see
``docs/validation/literature-r.md`` for the full report with observed
numbers, separated from the asserted tolerances):

1. Exact analytic identities (Poisson trace underground, vacuum trace
   above ground, surface mass-sheet jump, tangential-acceleration
   continuity), derived independently from H19 Eq. (39)/(93) (the
   eigenfunction divergence), not copied from the production module:

       delta_rho(target) = -rho * A * (qPz**2 - kR**2) * exp(qPz * z) * E

   (the shear-coupled term cancels exactly in the divergence -- only the
   compressional content survives). These are exact identities, checked to
   floating-point precision, not a quadrature-convergence tolerance.

2. An independent 1D vertical Green's-function quadrature
   (``test_independent_green_function_quadrature``, ``scipy.integrate.quad``,
   split at the observation point for the underground bulk integral), with
   acceleration and all 6 gradient components obtained by analytically
   differentiating that same Green's-function integral (Leibniz's rule,
   including the underground local density term -- not finite differences,
   which cannot resolve the originally specified "good condition" precision
   targets, section 9: 1e-10 for potential/acceleration, 1e-9 for the
   gradient) -- this closes the gap that method 1 alone does not verify the
   literal amplitude prefactors of Eq. (96)/(97). A companion convergence
   test sweeps the quadrature tolerance itself.

3. A direct comparison against the legacy ground-only
   ``rayleigh_halfspace_observables_g_free``
   (``test_matches_legacy_ground_api``), confirming the two share the same
   physical amplitude normalization.
"""

from __future__ import annotations

import numpy as np
import pytest

from gragra.analytic.rayleigh_literature import (
    rayleigh_halfspace_allspace_observables_g_free as rayleigh_allspace,
)
from gragra.waves.rayleigh import _rayleigh_parameters

# Two representative materials (both satisfy v_p**2 > (4/3) v_s**2).
POISSON_SOLID = {"v_p_m_s": np.sqrt(3.0) * 1500.0, "v_s_m_s": 1500.0}  # nu = 0.25
HARD_ROCK = {"v_p_m_s": 5800.0, "v_s_m_s": 3200.0}  # nu ~ 0.28

MATERIALS = [POISSON_SOLID, HARD_ROCK]
AZIMUTHS = [0.0, np.pi / 4.0]


def _direction(azimuth: float) -> tuple[float, float, float]:
    return (np.cos(azimuth), np.sin(azimuth), 0.0)


def _frequency_for_kr_h(material: dict, k_r_h: float, h: float) -> float:
    """Choose frequency_hz so that k_R * h equals the requested dimensionless value."""
    c_r, _, _ = _rayleigh_parameters(material["v_p_m_s"], material["v_s_m_s"])
    k_r = k_r_h / h
    return k_r * c_r / (2.0 * np.pi)


def _delta_rho(
    material: dict, freq: float, amp: complex, target: np.ndarray, n_hat
) -> complex:
    """Independent (hand-derived) density perturbation from H19 Eq. (39)/(93)."""
    c_r, q, s = _rayleigh_parameters(material["v_p_m_s"], material["v_s_m_s"])
    zeta = np.sqrt(q / s)
    omega = 2.0 * np.pi * freq
    k_r = omega / c_r
    q_p = k_r * q
    a_const = amp / (q_p - k_r * zeta)
    proj = target[0] * n_hat[0] + target[1] * n_hat[1]
    e_phase = np.exp(1j * k_r * proj)
    return -1.0 * a_const * (q_p**2 - k_r**2) * np.exp(q_p * target[2]) * e_phase


KR_H_GRID = [1e-4, 0.01, 0.1, 1.0, 3.0, 10.0]


@pytest.mark.parametrize("material", MATERIALS)
@pytest.mark.parametrize("azimuth", AZIMUTHS)
@pytest.mark.parametrize("k_r_h", KR_H_GRID)
def test_underground_poisson_trace_identity(material, azimuth, k_r_h):
    rho = 2300.0
    amp = 1e-9 * (1.0 + 0.3j)
    h = 5.0
    freq = _frequency_for_kr_h(material, k_r_h, h)
    n_hat = _direction(azimuth)
    target = np.array([[3.0, -1.0, -h]])

    out = rayleigh_allspace(
        target,
        density_kg_m3=rho,
        frequency_hz=freq,
        vertical_surface_amplitude_m=amp,
        direction=n_hat,
        **material,
    )
    delta_rho = rho * _delta_rho(material, freq, amp, target[0], n_hat)
    lhs = np.trace(out.gradient[0])
    rhs = -4.0 * np.pi * delta_rho
    # This is an exact algebraic identity (not a convergent approximation):
    # the only error source is floating-point round-off through a shallow
    # (~10-operation) formula chain on each side, plus the independently
    # recomputed rhs above sharing no code with production. Tolerance is a
    # small multiple of machine epsilon against the physical scale
    # Gamma ~ rho*|xi|*k (section 9 convention), not a number chosen to
    # exceed a separately observed value. Observed worst-case relative
    # error across this test's full material/azimuth/kR*h grid, recorded
    # in docs/validation/literature-r.md, is ~7e-16.
    c_r, _, _ = _rayleigh_parameters(material["v_p_m_s"], material["v_s_m_s"])
    k_r = 2.0 * np.pi * freq / c_r
    g_scale = rho * abs(amp) * k_r
    assert abs(lhs - rhs) <= 1e-10 * g_scale + 1e-10 * abs(rhs)


@pytest.mark.parametrize("material", MATERIALS)
@pytest.mark.parametrize("azimuth", AZIMUTHS)
@pytest.mark.parametrize("k_r_h", KR_H_GRID)
def test_ground_trace_is_zero(material, azimuth, k_r_h):
    rho = 2300.0
    amp = 1e-9
    h = 5.0
    freq = _frequency_for_kr_h(material, k_r_h, h)
    target = np.array([[3.0, -1.0, h]])
    out = rayleigh_allspace(
        target,
        density_kg_m3=rho,
        frequency_hz=freq,
        vertical_surface_amplitude_m=amp,
        direction=_direction(azimuth),
        **material,
    )
    # Expected value is exactly 0 (vacuum Poisson equation), so no relative
    # term is possible; the residual is floating-point round-off (n1**2 +
    # n2**2 == 1 only up to ~machine epsilon after the scale-safe
    # normalization) against the physical scale Gamma ~ rho*|xi|*k, not an
    # arbitrary fixed number. Observed worst case across this test's grid,
    # recorded in docs/validation/literature-r.md, is ~0 to ~6e-16 relative
    # to g_scale.
    c_r, _, _ = _rayleigh_parameters(material["v_p_m_s"], material["v_s_m_s"])
    k_r = 2.0 * np.pi * freq / c_r
    g_scale = rho * abs(amp) * k_r
    np.testing.assert_allclose(np.trace(out.gradient[0]), 0.0, atol=1e-10 * g_scale)


@pytest.mark.parametrize("material", MATERIALS)
def test_potential_continuous_at_surface(material):
    rho, amp, freq = 2300.0, 1e-9, 2.5
    xy = (3.0, -1.0)
    eps = 1e-6
    targets = np.array([[xy[0], xy[1], eps], [xy[0], xy[1], -eps]])
    out = rayleigh_allspace(
        targets,
        density_kg_m3=rho,
        frequency_hz=freq,
        vertical_surface_amplitude_m=amp,
        **material,
    )
    diff = abs(out.potential[0] - out.potential[1])
    # Phi is continuous (C0) but not C1 at z=0 (a_z jumps there, see
    # test_surface_mass_sheet_jump), so evaluating at a finite one-sided
    # offset +-eps does not give an exact match: by Taylor expansion from
    # each side, Phi(eps) - Phi(-eps) = -eps*(a_z(0+) + a_z(0-)) + O(eps^3),
    # both terms O(a_scale) in magnitude (a_scale = rho*|xi|, section 9
    # convention) -- so the residual must scale like eps*a_scale, not stay
    # fixed as eps -> 0. Verified directly (sandbox investigation for this
    # PR, recorded in docs/validation/literature-r.md): the ratio
    # |diff| / (eps * a_scale) is constant at ~3.09 (Poisson solid) / ~2.74
    # (hard rock) across eps swept over {1e-3, ..., 1e-8} (i.e. genuinely
    # O(eps), not an eps-independent floor). Tolerance below (coefficient
    # 10) carries a >3x safety factor over the worst of those two observed
    # ratios, not "large margin" reasoning against an unrelated number.
    a_scale = rho * abs(amp)
    assert diff <= 10.0 * eps * a_scale


@pytest.mark.parametrize("material", MATERIALS)
def test_surface_mass_sheet_jump(material):
    rho, amp, freq = 2300.0, 1e-9 * (0.7 - 0.2j), 2.5
    xy = np.array([3.0, -1.0])
    n_hat = _direction(0.0)
    eps = 1e-8
    above = np.array([[xy[0], xy[1], eps]])
    below = np.array([[xy[0], xy[1], -eps]])
    out_above = rayleigh_allspace(
        above,
        density_kg_m3=rho,
        frequency_hz=freq,
        vertical_surface_amplitude_m=amp,
        direction=n_hat,
        **material,
    )
    out_below = rayleigh_allspace(
        below,
        density_kg_m3=rho,
        frequency_hz=freq,
        vertical_surface_amplitude_m=amp,
        direction=n_hat,
        **material,
    )
    az_jump = out_above.acceleration[0, 2] - out_below.acceleration[0, 2]
    proj = xy[0] * n_hat[0] + xy[1] * n_hat[1]
    c_r, q, s = _rayleigh_parameters(material["v_p_m_s"], material["v_s_m_s"])
    k_r = 2.0 * np.pi * freq / c_r
    sigma = rho * amp * np.exp(1j * k_r * proj)
    rhs = -4.0 * np.pi * sigma
    # az_jump is evaluated at a finite one-sided offset +-eps from the exact
    # interface, not exactly at it, so it carries an O(eps) correction from
    # each side's own curvature (Taylor remainder), not a fixed floor.
    # Verified directly (sandbox investigation for this PR, recorded in
    # docs/validation/literature-r.md): the relative error / (eps * k_r)
    # ratio is constant at ~1.21 (Poisson solid) / ~1.19 (hard rock) across
    # eps swept over {1e-6, ..., 1e-9}. Tolerance below (coefficient 10)
    # carries a >8x safety factor over the worst observed ratio.
    assert abs(az_jump - rhs) <= 10.0 * eps * k_r * abs(rhs)


def test_frequency_series_matches_scalar_evaluation():
    rho, amp = 2300.0, 1e-9
    targets = np.array([[3.0, -1.0, 2.0], [3.0, -1.0, -2.0]])
    freqs = np.array([1.0, 2.5, 5.0])
    out_series = rayleigh_allspace(
        targets,
        density_kg_m3=rho,
        frequency_hz=freqs,
        vertical_surface_amplitude_m=amp,
        **POISSON_SOLID,
    )
    assert out_series.potential.shape == (3, 2)
    assert out_series.acceleration.shape == (3, 2, 3)
    assert out_series.gradient.shape == (3, 2, 3, 3)
    for k, f in enumerate(freqs):
        out_scalar = rayleigh_allspace(
            targets,
            density_kg_m3=rho,
            frequency_hz=float(f),
            vertical_surface_amplitude_m=amp,
            **POISSON_SOLID,
        )
        np.testing.assert_allclose(
            out_series.potential[k], out_scalar.potential, rtol=1e-13
        )
        np.testing.assert_allclose(
            out_series.acceleration[k], out_scalar.acceleration, rtol=1e-13
        )
        np.testing.assert_allclose(
            out_series.gradient[k], out_scalar.gradient, rtol=1e-13
        )


def test_complex_amplitude_linearity():
    targets = np.array([[3.0, -1.0, 2.0], [3.0, -1.0, -2.0]])
    kwargs = {
        "targets_m": targets,
        "density_kg_m3": 2300.0,
        "frequency_hz": 2.5,
        **POISSON_SOLID,
    }
    base = rayleigh_allspace(vertical_surface_amplitude_m=1.0, **kwargs)
    scaled = rayleigh_allspace(vertical_surface_amplitude_m=2.0 + 3.0j, **kwargs)
    factor = 2.0 + 3.0j
    np.testing.assert_allclose(scaled.potential, factor * base.potential, rtol=1e-12)
    np.testing.assert_allclose(
        scaled.acceleration, factor * base.acceleration, rtol=1e-12
    )
    np.testing.assert_allclose(scaled.gradient, factor * base.gradient, rtol=1e-12)


def test_zero_amplitude_gives_zero_field():
    targets = np.array([[3.0, -1.0, 2.0], [3.0, -1.0, -2.0]])
    out = rayleigh_allspace(
        targets,
        density_kg_m3=2300.0,
        frequency_hz=2.5,
        vertical_surface_amplitude_m=0.0,
        **POISSON_SOLID,
    )
    np.testing.assert_allclose(out.potential, 0.0)
    np.testing.assert_allclose(out.acceleration, 0.0)
    np.testing.assert_allclose(out.gradient, 0.0)


def test_zero_density_gives_zero_field():
    targets = np.array([[3.0, -1.0, 2.0], [3.0, -1.0, -2.0]])
    out = rayleigh_allspace(
        targets,
        density_kg_m3=0.0,
        frequency_hz=2.5,
        vertical_surface_amplitude_m=1e-9,
        **POISSON_SOLID,
    )
    np.testing.assert_allclose(out.potential, 0.0)


def test_readonly_output_arrays():
    out = rayleigh_allspace(
        np.array([[1.0, 0.0, 1.0]]),
        density_kg_m3=2300.0,
        frequency_hz=2.5,
        vertical_surface_amplitude_m=1e-9,
        **POISSON_SOLID,
    )
    for arr in (out.potential, out.acceleration, out.gradient):
        assert arr.flags.writeable is False


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"targets_m": [[0.0, 0.0, 0.0]]}, "free surface"),
        ({"density_kg_m3": -1.0}, "density_kg_m3"),
        ({"v_s_m_s": POISSON_SOLID["v_p_m_s"]}, "v_s_m_s"),
        ({"v_p_m_s": 1.0, "v_s_m_s": 0.99}, "stable isotropic"),
        ({"frequency_hz": 0.0}, "frequency_hz"),
        ({"frequency_hz": -1.0}, "frequency_hz"),
        ({"frequency_hz": [[1.0]]}, "1D array"),
        ({"direction": (0.0, 0.0, 1.0)}, "horizontal"),
        ({"direction": (0.0, 0.0, 0.0)}, "non-zero"),
    ],
)
def test_invalid_inputs_raise_value_error(kwargs, match):
    base = {
        "targets_m": [[1.0, 0.0, 1.0]],
        "density_kg_m3": 2300.0,
        "v_p_m_s": POISSON_SOLID["v_p_m_s"],
        "v_s_m_s": POISSON_SOLID["v_s_m_s"],
        "frequency_hz": 2.5,
        "vertical_surface_amplitude_m": 1e-9,
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        rayleigh_allspace(**base)


def test_boolean_amplitude_rejected():
    with pytest.raises(TypeError):
        rayleigh_allspace(
            [[1.0, 0.0, 1.0]],
            density_kg_m3=2300.0,
            frequency_hz=2.5,
            vertical_surface_amplitude_m=True,
            **POISSON_SOLID,
        )


# ---------------------------------------------------------------------------
# Scale-safe direction normalization.
#
# ``np.linalg.norm`` on a raw direction vector overflows to ``inf`` for
# components like 1e200 ((1e200)**2 already exceeds float64 range) and
# underflows to exactly 0.0 for components like 1e-200 ((1e-200)**2
# underflows), silently corrupting the normalized direction or spuriously
# rejecting it as zero. ``_normalize_horizontal_direction`` divides by the
# largest-magnitude component first so every subsequent operation stays
# order-1, independent of the input's overall scale.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200])
@pytest.mark.parametrize(
    "unit_direction", [(1.0, 0.0, 0.0), (0.6, 0.8, 0.0)], ids=["axis", "oblique"]
)
def test_direction_scale_invariance(unit_direction, scale):
    # Target chosen so that the axis-aligned and oblique directions give
    # genuinely different projections (unlike a poorly chosen target where
    # both directions happen to coincide).
    targets = np.array([[5.0, -3.0, 2.0], [5.0, -3.0, -2.0]])
    kwargs = {
        "targets_m": targets,
        "density_kg_m3": 2300.0,
        "frequency_hz": 2.5,
        "vertical_surface_amplitude_m": 1e-9 * (1.0 + 0.4j),
        **POISSON_SOLID,
    }
    reference = rayleigh_allspace(direction=unit_direction, **kwargs)
    scaled = rayleigh_allspace(
        direction=tuple(c * scale for c in unit_direction), **kwargs
    )
    np.testing.assert_allclose(scaled.potential, reference.potential, rtol=1e-13)
    np.testing.assert_allclose(scaled.acceleration, reference.acceleration, rtol=1e-13)
    np.testing.assert_allclose(scaled.gradient, reference.gradient, rtol=1e-13)


@pytest.mark.parametrize("z_scale", [1e-200, 1.0, 1e200])
def test_direction_extreme_z_component_rejected(z_scale):
    with pytest.raises(ValueError, match="horizontal"):
        rayleigh_allspace(
            [[1.0, 0.0, 1.0]],
            density_kg_m3=2300.0,
            frequency_hz=2.5,
            vertical_surface_amplitude_m=1e-9,
            direction=(0.0, 0.0, z_scale),
            **POISSON_SOLID,
        )


# ---------------------------------------------------------------------------
# Type validation before float64 conversion.
#
# ``np.asarray(value, dtype=np.float64)`` silently discards the imaginary
# part of complex input (only a ``ComplexWarning``, not an error) and
# silently accepts ``bool`` as 0.0/1.0. Real-only parameters must reject both
# before any such conversion runs.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field", ["targets_m", "direction", "phase_origin_m", "frequency_hz"]
)
def test_complex_input_rejected_for_real_only_fields(field):
    base = {
        "targets_m": [[1.0, 0.0, 1.0]],
        "density_kg_m3": 2300.0,
        "frequency_hz": 2.5,
        "vertical_surface_amplitude_m": 1e-9,
        "direction": (1.0, 0.0, 0.0),
        "phase_origin_m": (0.0, 0.0, 0.0),
        **POISSON_SOLID,
    }
    complex_values = {
        "targets_m": [[1.0 + 2.0j, 0.0, 1.0]],
        "direction": (1.0 + 0.0j, 0.0, 0.0),
        "phase_origin_m": (0.0 + 1.0j, 0.0, 0.0),
        "frequency_hz": 2.5 + 1.0j,
    }
    base[field] = complex_values[field]
    with pytest.raises(TypeError, match="real|complex"):
        rayleigh_allspace(**base)


@pytest.mark.parametrize(
    "field", ["targets_m", "direction", "phase_origin_m", "frequency_hz"]
)
def test_bool_input_rejected_for_real_only_fields(field):
    base = {
        "targets_m": [[1.0, 0.0, 1.0]],
        "density_kg_m3": 2300.0,
        "frequency_hz": 2.5,
        "vertical_surface_amplitude_m": 1e-9,
        "direction": (1.0, 0.0, 0.0),
        "phase_origin_m": (0.0, 0.0, 0.0),
        **POISSON_SOLID,
    }
    bool_values = {
        "targets_m": [[True, False, True]],
        "direction": (True, False, False),
        "phase_origin_m": (True, False, False),
        "frequency_hz": True,
    }
    base[field] = bool_values[field]
    with pytest.raises(TypeError, match="bool"):
        rayleigh_allspace(**base)


# ---------------------------------------------------------------------------
# (K,) per-frequency amplitude (original common contract: scalar or (K,)
# amplitude, matched against a 1D frequency_hz of the same length -- no other
# implicit broadcast).
# ---------------------------------------------------------------------------


def test_amplitude_array_matches_per_scalar_calls():
    targets = np.array([[3.0, -1.0, 2.0], [3.0, -1.0, -2.0]])
    freqs = np.array([1.0, 2.5, 5.0])
    amps = np.array([1e-9, 2e-9 * (1.0 + 0.5j), 3e-9])
    out_k = rayleigh_allspace(
        targets,
        density_kg_m3=2300.0,
        frequency_hz=freqs,
        vertical_surface_amplitude_m=amps,
        **POISSON_SOLID,
    )
    for k in range(3):
        out_scalar = rayleigh_allspace(
            targets,
            density_kg_m3=2300.0,
            frequency_hz=float(freqs[k]),
            vertical_surface_amplitude_m=amps[k],
            **POISSON_SOLID,
        )
        np.testing.assert_allclose(out_k.potential[k], out_scalar.potential, rtol=1e-13)
        np.testing.assert_allclose(
            out_k.acceleration[k], out_scalar.acceleration, rtol=1e-13
        )
        np.testing.assert_allclose(out_k.gradient[k], out_scalar.gradient, rtol=1e-13)


def test_amplitude_array_length_mismatch_rejected():
    with pytest.raises(ValueError, match="length"):
        rayleigh_allspace(
            [[1.0, 0.0, 1.0]],
            density_kg_m3=2300.0,
            frequency_hz=np.array([1.0, 2.5, 5.0]),
            vertical_surface_amplitude_m=np.array([1e-9, 2e-9]),
            **POISSON_SOLID,
        )


def test_amplitude_array_with_scalar_frequency_rejected():
    with pytest.raises(ValueError, match="scalar"):
        rayleigh_allspace(
            [[1.0, 0.0, 1.0]],
            density_kg_m3=2300.0,
            frequency_hz=2.5,
            vertical_surface_amplitude_m=np.array([1e-9, 2e-9]),
            **POISSON_SOLID,
        )


# ---------------------------------------------------------------------------
# Comparison with the legacy ground-only API.
#
# The legacy ``rayleigh_halfspace_observables_g_free`` uses the same
# ``vertical_surface_amplitude_m`` = xi_z(0,0) normalization as this module
# (both trace back to the same two-exponential eigenfunction convention in
# ``gragra.waves.rayleigh``): for the same material/frequency/direction, the
# two must give identical ground-side (z > 0) potential and acceleration --
# not merely proportional values. The legacy function additionally requires
# a strictly positive *real* amplitude (its own ``_positive`` validator
# rejects complex outright), unlike this module which accepts complex/zero;
# this comparison therefore uses a positive real amplitude, the input domain
# where both functions are simultaneously callable.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("material", MATERIALS)
@pytest.mark.parametrize("azimuth", AZIMUTHS)
def test_matches_legacy_ground_api(material, azimuth):
    from gragra.analytic.wave456_reference import (
        rayleigh_halfspace_observables_g_free as legacy,
    )

    targets = np.array([[3.0, -1.0, 2.0], [7.0, 4.0, 0.5]])
    kwargs = {
        "density_kg_m3": 2300.0,
        "frequency_hz": 2.5,
        "vertical_surface_amplitude_m": 1e-9,
        "direction": _direction(azimuth),
        **material,
    }
    old = legacy(targets, **kwargs)
    new = rayleigh_allspace(targets, **kwargs)
    np.testing.assert_allclose(new.potential, old.potential, rtol=1e-12)
    np.testing.assert_allclose(new.acceleration, old.acceleration, rtol=1e-12)
    np.testing.assert_allclose(new.gradient, old.gradient, rtol=1e-12)


# ---------------------------------------------------------------------------
# Tangential acceleration continuity at the free surface.
#
# Only a_z jumps across z=0 (the free-surface mass sheet); a_x and a_y, which
# do not couple to that sheet, must be continuous.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("material", MATERIALS)
def test_tangential_acceleration_continuous_at_surface(material):
    rho, amp, freq = 2300.0, 1e-9 * (0.7 - 0.2j), 2.5
    xy = (3.0, -1.0)
    n_hat = _direction(np.pi / 4.0)
    eps = 1e-8
    kwargs = {
        "density_kg_m3": rho,
        "frequency_hz": freq,
        "vertical_surface_amplitude_m": amp,
        "direction": n_hat,
        **material,
    }
    above = rayleigh_allspace([[xy[0], xy[1], eps]], **kwargs)
    below = rayleigh_allspace([[xy[0], xy[1], -eps]], **kwargs)
    # a_x, a_y are continuous but not C1 at z=0 (their z-derivatives Gamma_xz,
    # Gamma_yz do jump, proportional to the same surface term as a_z's own
    # jump), so the same O(eps)*g_scale Taylor-remainder argument as
    # test_potential_continuous_at_surface applies here (with g_scale =
    # rho*|xi|*k, section 9 convention, replacing a_scale since this is a
    # z-derivative of a-scale quantities). Verified directly (sandbox
    # investigation for this PR, recorded in docs/validation/literature-r.md):
    # ratio |diff| / (eps * g_scale) is constant at ~2.18 (Poisson solid) /
    # ~1.94 (hard rock) across eps in {1e-6, ..., 1e-9}. Tolerance below
    # (coefficient 10) carries a >4x safety factor over the worst observed
    # ratio.
    c_r, _, _ = _rayleigh_parameters(material["v_p_m_s"], material["v_s_m_s"])
    k_r = 2.0 * np.pi * freq / c_r
    g_scale = rho * abs(amp) * k_r
    tangential_diff = np.max(
        np.abs(above.acceleration[0, :2] - below.acceleration[0, :2])
    )
    assert tangential_diff <= 10.0 * eps * g_scale
    # a_z, in contrast, must show the (nonzero) surface-sheet jump -- a
    # sanity floor at 10% of the expected jump magnitude (~4*pi*rho*|xi|),
    # well above the O(eps*g_scale) numerical noise level just bounded
    # above and well below the actual jump.
    a_scale = rho * abs(amp)
    assert abs(above.acceleration[0, 2] - below.acceleration[0, 2]) > 0.1 * a_scale


# ---------------------------------------------------------------------------
# Independent 1D vertical Green's-function quadrature, differentiated
# analytically (no finite differences).
#
# Re-derives the field from scratch via the horizontal-Fourier-mode reduction
# of Poisson's equation: for a single horizontal mode e^{i k_R . x_h}, the
# potential's vertical profile solves (d^2/dz^2 - k_R^2) Phi_hat(z) =
# 4*pi*rho_hat(z), whose Green's function is G(z, z') = -exp(-k_R|z-z'|)/(2
# k_R) (verified by direct differentiation: away from z=z' it solves the
# homogeneous equation, and its derivative jump at z=z' is exactly 1,
# matching the delta-function source).
#
# Because Phi(x,y,z) = E(x,y) * Phi_z(z) (a pure horizontal plane-wave factor
# times a z-only profile), a=-grad(Phi) and Gamma=grad(a) reduce to
# elementary algebra in x,y (exact to machine precision, no numerical
# differentiation needed there) plus Phi_z(z), Phi_z'(z), Phi_z''(z) in z.
# Phi_z' and Phi_z'' are obtained by DIFFERENTIATING THE GREEN'S-FUNCTION
# INTEGRAL ANALYTICALLY (Leibniz's rule on the two observation-point-split
# quad sub-integrals), not by finite-differencing the oracle -- an earlier
# version of this test used central finite differences (step h=1e-3),
# which could not resolve better than ~1e-6 relative in the gradient and
# so could not verify the originally specified 1e-9 "good condition"
# criterion; that FD approach has been replaced, not tuned, as this
# reviewed instruction required.
#
# The underground bulk integral I(z) = v_lo(z) + v_hi(z) is split at the
# observation point z (v_lo = integral over z' < z, v_hi = z' > z). Applying
# Leibniz's rule to dI/dz: the two boundary terms from differentiating each
# piece's z-dependent limit exactly cancel (G is continuous, only its
# *derivative* has a kink at z'=z), leaving dI/dz = -k_r*(v_lo - v_hi) with
# no extra term. Differentiating again, the boundary terms from d/dz(dI/dz)
# do NOT cancel this time (now differentiating d G/dz, whose jump at z'=z is
# the actual delta-function source) and sum to exactly the local bulk
# density shape f(z) = (qPz^2-kR^2)*exp(qPz*z) -- this is the "underground
# local term" the reviewed instruction asked to keep explicit:
#   d^2I/dz^2 = f(z) + k_r^2 * (v_lo + v_hi)   (z < 0; the f(z) term is
#                                                absent for z > 0, where the
#                                                split point sits outside
#                                                the (-inf, 0] domain)
# The surface term G(z,0) has no such local term for z != 0 (the only point
# the API rejects), so its second derivative is the plain homogeneous-
# equation value k_r^2 * G(z,0). Both derivative identities, and the
# resulting a/Gamma formulas below, were verified against production to
# ~1e-15 relative (all 6 gradient components, both signs of z, both
# materials) before being written up here -- see
# docs/validation/literature-r.md for the full observed-error table and a
# quad-precision convergence study (varying epsabs/epsrel) confirming this
# is a converged, not a lucky-single-setting, result.
# ---------------------------------------------------------------------------

# scipy.integrate.quad settings used throughout this section. Chosen from
# the convergence study in docs/validation/literature-r.md, which shows the
# result is already converged to machine precision at epsabs=epsrel=1e-8 for
# this smooth, exponentially-decaying integrand; these are one order of
# magnitude tighter than that observed plateau, not an arbitrary guess.
_QUAD_KWARGS = {"epsabs": 1e-12, "epsrel": 1e-11}


def _green_kernel(z: float, zp: float, k_r: float) -> float:
    return -np.exp(-k_r * abs(z - zp)) / (2.0 * k_r)


def _oracle_bulk_value_and_derivatives(
    z: float, q_p: float, k_r: float, quad_kwargs: dict
) -> tuple[float, float, float]:
    """Independent quadrature for I(z) = integral(f(z') G(z,z') dz'), dI/dz, d2I/dz2.

    ``f(z') = (q_p**2 - k_r**2) * exp(q_p * z')`` is the (unnormalized) bulk
    density shape from H19 Eq. (39)/(93)'s divergence (re-derived by hand,
    not imported from production). See the section docstring above for the
    Leibniz-rule derivation of the two derivative formulas below.
    """
    from scipy import integrate

    def f(zp: float) -> float:
        return (q_p**2 - k_r**2) * np.exp(q_p * zp)

    def integrand(zp: float) -> float:
        return f(zp) * _green_kernel(z, zp, k_r)

    if z < 0.0:
        v_lo, _ = integrate.quad(integrand, -np.inf, z, **quad_kwargs)
        v_hi, _ = integrate.quad(integrand, z, 0.0, **quad_kwargs)
        local_term = f(z)
    else:
        v_lo, _ = integrate.quad(integrand, -np.inf, 0.0, **quad_kwargs)
        v_hi = 0.0
        local_term = 0.0

    value = v_lo + v_hi
    first_derivative = -k_r * (v_lo - v_hi)
    second_derivative = k_r**2 * value + local_term
    return value, first_derivative, second_derivative


def _oracle_surface_value_and_derivatives(
    z: float, k_r: float
) -> tuple[float, float, float]:
    """Closed-form G(z,0), dG/dz, d2G/dz2 for z != 0 (no quadrature needed)."""
    g0 = _green_kernel(z, 0.0, k_r)
    dg0 = -np.sign(z) * k_r * g0
    d2g0 = k_r**2 * g0
    return g0, dg0, d2g0


def _oracle_observables(
    x: float,
    y: float,
    z: float,
    *,
    rho: float,
    material: dict,
    freq: float,
    amp: complex,
    n1: float,
    n2: float,
    quad_kwargs: dict | None = None,
) -> tuple[complex, np.ndarray, np.ndarray]:
    """Independent (analytic-derivative) potential/acceleration/gradient oracle."""
    if quad_kwargs is None:
        quad_kwargs = _QUAD_KWARGS
    c_r, q, s = _rayleigh_parameters(material["v_p_m_s"], material["v_s_m_s"])
    zeta = np.sqrt(q / s)
    k_r = 2.0 * np.pi * freq / c_r
    q_p = k_r * q
    a_amp = amp / (q_p - k_r * zeta)  # Eq. (98), independently recomputed here

    bulk, d_bulk, d2_bulk = _oracle_bulk_value_and_derivatives(z, q_p, k_r, quad_kwargs)
    surf, d_surf, d2_surf = _oracle_surface_value_and_derivatives(z, k_r)

    phi_z = 4.0 * np.pi * rho * (-a_amp * bulk + amp * surf)
    phi_z1 = 4.0 * np.pi * rho * (-a_amp * d_bulk + amp * d_surf)
    phi_z2 = 4.0 * np.pi * rho * (-a_amp * d2_bulk + amp * d2_surf)

    e_phase = np.exp(1j * k_r * (n1 * x + n2 * y))
    phi = e_phase * phi_z

    acceleration = -e_phase * np.array(
        [1j * k_r * n1 * phi_z, 1j * k_r * n2 * phi_z, phi_z1]
    )
    gradient = np.empty((3, 3), dtype=np.complex128)
    gradient[0, 0] = k_r**2 * n1 * n1 * e_phase * phi_z
    gradient[0, 1] = gradient[1, 0] = k_r**2 * n1 * n2 * e_phase * phi_z
    gradient[1, 1] = k_r**2 * n2 * n2 * e_phase * phi_z
    gradient[0, 2] = gradient[2, 0] = -1j * k_r * n1 * e_phase * phi_z1
    gradient[1, 2] = gradient[2, 1] = -1j * k_r * n2 * e_phase * phi_z1
    gradient[2, 2] = -e_phase * phi_z2
    return phi, acceleration, gradient


@pytest.mark.slow
@pytest.mark.parametrize("material", MATERIALS)
@pytest.mark.parametrize("z_target", [2.0, -2.0, 0.2, -0.2, -30.0])
def test_independent_green_function_quadrature(material, z_target):
    pytest.importorskip("scipy")
    rho = 2300.0
    freq = 2.5
    amp = 1e-9 * (1.0 + 0.3j)
    azimuth = np.pi / 4.0  # oblique, so all 6 gradient components are nonzero
    n1, n2 = _direction(azimuth)[:2]
    x, y = 3.0, -1.0

    phi_oracle, a_oracle, g_oracle = _oracle_observables(
        x, y, z_target, rho=rho, material=material, freq=freq, amp=amp, n1=n1, n2=n2
    )

    out = rayleigh_allspace(
        [[x, y, z_target]],
        density_kg_m3=rho,
        frequency_hz=freq,
        vertical_surface_amplitude_m=amp,
        direction=(n1, n2, 0.0),
        **material,
    )
    phi_prod = out.potential[0]
    a_prod = out.acceleration[0]
    g_prod = out.gradient[0]

    # Physical scales (section 9 convention for half-space waves):
    # Phi ~ rho*|xi|/k, a ~ rho*|xi|, Gamma ~ rho*|xi|*k.
    c_r, _, _ = _rayleigh_parameters(material["v_p_m_s"], material["v_s_m_s"])
    k_r = 2.0 * np.pi * freq / c_r
    phi_scale = rho * abs(amp) / k_r
    a_scale = rho * abs(amp)
    g_scale = rho * abs(amp) * k_r

    # Tolerance is set to the originally specified "good condition"
    # acceptance criterion (section 9) itself, not derived from the observed
    # value: 1e-10 for potential/acceleration, 1e-9 for the gradient (section
    # 9 states these as two distinct numbers, not one uniform value).
    # Observed (sandbox investigation for this PR, full table in
    # docs/validation/literature-r.md): all of potential/acceleration/all 6
    # gradient components agree with production to ~1e-15 to ~2e-15
    # relative at every (material, z_target) combination tested here --
    # i.e. the analytic-derivative oracle clears both targets with margin to
    # spare, but the assertion below does not encode that margin as its
    # rationale.
    assert abs(phi_oracle - phi_prod) <= 1e-10 * phi_scale + 1e-10 * abs(phi_prod)
    for i in range(3):
        assert abs(a_oracle[i] - a_prod[i]) <= 1e-10 * a_scale + 1e-10 * abs(a_prod[i])
    for i in range(3):
        for j in range(3):
            assert abs(g_oracle[i, j] - g_prod[i, j]) <= 1e-9 * g_scale + 1e-9 * abs(
                g_prod[i, j]
            )


@pytest.mark.slow
def test_green_function_quadrature_precision_convergence():
    """Records the actual quad-precision convergence behavior (not asserted
    as a single lucky setting): tightening epsabs/epsrel from 1e-6 down to
    1e-13 must not change the oracle's answer beyond the 1e-9-scale
    tolerance used by test_independent_green_function_quadrature above --
    i.e. the setting used there is on the converged plateau, not a point
    that happens to agree with production by coincidence at one particular
    quadrature tolerance.
    """
    pytest.importorskip("scipy")
    rho = 2300.0
    freq = 2.5
    amp = 1e-9 * (1.0 + 0.3j)
    material = POISSON_SOLID
    n1, n2 = _direction(np.pi / 4.0)[:2]
    x, y, z_target = 3.0, -1.0, -2.0

    settings = [1e-6, 1e-8, 1e-10, 1e-12, 1e-13]
    values = []
    for eps in settings:
        _, _, g = _oracle_observables(
            x,
            y,
            z_target,
            rho=rho,
            material=material,
            freq=freq,
            amp=amp,
            n1=n1,
            n2=n2,
            quad_kwargs={"epsabs": eps, "epsrel": eps},
        )
        values.append(g[2, 2])

    c_r, _, _ = _rayleigh_parameters(material["v_p_m_s"], material["v_s_m_s"])
    k_r = 2.0 * np.pi * freq / c_r
    g_scale = rho * abs(amp) * k_r
    tightest = values[-1]
    for eps, value in zip(settings[:-1], values[:-1], strict=True):
        diff = abs(value - tightest)
        # Loosest setting (1e-6) observed at ~1.8e-12 relative in the
        # sandbox investigation for this PR (recorded in
        # docs/validation/literature-r.md); already-tighter settings
        # (1e-8 and below) agree with the tightest to ~1e-16.
        assert diff <= 1e-9 * g_scale, f"epsabs=epsrel={eps}: diff={diff}"


@pytest.mark.slow
def test_high_level_rotated_translated_surface_matches_independent_green_oracle(
    rayleigh_green_oracle_case,
):
    """Check the public phasor path against the local Green-function oracle.

    The reference coordinates and rotation below are formed explicitly from
    the test's known origin and basis; no production Frame transform helper is
    used. Green-function quadrature and its analytic derivatives are
    independent of the production field calculation. The oracle does share
    ``_rayleigh_parameters`` and the Eq. (98) amplitude normalization, which
    remains its stated independence limit.
    """
    pytest.importorskip("scipy")
    from astropy import units as u
    from gwexpy.frequencyseries import FrequencySeries

    from gragra.analytic.seismic import rayleigh_halfspace
    from gragra.constants import G_SI
    from gragra.geometry import Frame, HalfSpace
    from gragra.results import PhasorRepresentation
    from gragra.targets import TargetPoints

    case = rayleigh_green_oracle_case
    root = Frame("oracle-world")
    surface_origin_world_m = case["surface_origin_world_m"]
    surface_basis_world = case["surface_basis_world"]
    surface_frame = Frame(
        "oracle-surface",
        surface_origin_world_m * u.m,
        surface_basis_world,
        root,
    )
    local_targets_m = case["local_targets_m"]
    world_targets_m = surface_origin_world_m + local_targets_m @ surface_basis_world.T
    targets = TargetPoints(
        world_targets_m * u.m,
        case["target_ids"],
        root,
    )

    rho = case["density_kg_m3"]
    freq = case["frequency_hz"]
    amp = case["amplitude_m"]
    phase_origin_local_m = case["phase_origin_local_m"]
    direction_local = case["direction_local"]
    material = {"v_p_m_s": case["vp_m_s"], "v_s_m_s": case["vs_m_s"]}
    model = rayleigh_halfspace(
        geometry=HalfSpace(surface_frame),
        density=rho * u.kg / u.m**3,
        vp=material["v_p_m_s"] * u.m / u.s,
        vs=material["v_s_m_s"] * u.m / u.s,
        vertical_displacement=FrequencySeries(
            np.array([amp], dtype=np.complex128), frequencies=[freq] * u.Hz, unit=u.m
        ),
        representation=PhasorRepresentation(),
        reference_time=case["reference_time_gps"],
        phase_origin=phase_origin_local_m * u.m,
        direction=direction_local,
    )
    result = model.at(targets, output_frame=root).run()

    # For row-vector coordinates, p_local = (p_world - origin) @ R. This is
    # the test's explicit coordinate construction, independent of Frame APIs.
    expected_local_m = (world_targets_m - surface_origin_world_m) @ surface_basis_world
    np.testing.assert_allclose(expected_local_m, local_targets_m, rtol=0.0, atol=2e-15)

    c_r, _, _ = _rayleigh_parameters(material["v_p_m_s"], material["v_s_m_s"])
    k_r = 2.0 * np.pi * freq / c_r
    phi_scale = G_SI * rho * abs(amp) / k_r
    accel_scale = G_SI * rho * abs(amp)
    gradient_scale = G_SI * rho * abs(amp) * k_r
    observed_gradient_local: list[np.ndarray] = []
    for index, local in enumerate(expected_local_m):
        relative_xy = local[:2] - phase_origin_local_m[:2]
        phi_oracle, accel_oracle, gradient_oracle = _oracle_observables(
            relative_xy[0],
            relative_xy[1],
            local[2],
            rho=rho,
            material=material,
            freq=freq,
            amp=amp,
            n1=direction_local[0],
            n2=direction_local[1],
        )
        phi_production = np.asarray(result.potential)[index, 0, 0]
        accel_production = np.asarray(result.acceleration)[index, :, 0]
        gradient_production = np.asarray(result.gradient)[index, :, 0].reshape(3, 3)
        accel_expected_world = G_SI * (surface_basis_world @ accel_oracle)
        gradient_expected_world = (
            G_SI * surface_basis_world @ gradient_oracle @ surface_basis_world.T
        )
        assert abs(phi_production - G_SI * phi_oracle) <= (
            1e-10 * phi_scale + 1e-10 * abs(phi_production)
        )
        for component in range(3):
            assert abs(
                accel_production[component] - accel_expected_world[component]
            ) <= (1e-10 * accel_scale + 1e-10 * abs(accel_production[component]))
        for row in range(3):
            for column in range(3):
                assert abs(
                    gradient_production[row, column]
                    - gradient_expected_world[row, column]
                ) <= (
                    1e-9 * gradient_scale + 1e-9 * abs(gradient_production[row, column])
                )
        observed_gradient_local.append(
            surface_basis_world.T @ gradient_production @ surface_basis_world
        )

    # Trace is rotation invariant. Vacuum has zero trace; underground trace
    # follows the independently differentiated bulk density perturbation.
    assert abs(np.trace(observed_gradient_local[0])) <= 1e-10 * gradient_scale
    relative_underground = np.array(
        [
            expected_local_m[1, 0] - phase_origin_local_m[0],
            expected_local_m[1, 1] - phase_origin_local_m[1],
            expected_local_m[1, 2],
        ]
    )
    delta_rho = rho * _delta_rho(
        material, freq, amp, relative_underground, direction_local
    )
    trace_expected = -4.0 * np.pi * G_SI * delta_rho
    trace_observed = np.trace(observed_gradient_local[1])
    assert abs(trace_observed - trace_expected) <= (
        1e-10 * gradient_scale + 1e-10 * abs(trace_expected)
    )

    # The free-surface mass-sheet jump is measured along the translated and
    # rotated surface's known unit normal. Finite ±epsilon samples carry the
    # same O(epsilon*k_R) correction as the existing local oracle test.
    surface_xy_local_m = np.array([2.4, -1.1])
    epsilon_m = 1e-8
    local_jump_targets_m = np.array(
        [
            [*surface_xy_local_m, epsilon_m],
            [*surface_xy_local_m, -epsilon_m],
        ]
    )
    world_jump_targets_m = (
        surface_origin_world_m + local_jump_targets_m @ surface_basis_world.T
    )
    jump_targets = TargetPoints(
        world_jump_targets_m * u.m,
        ("just-above", "just-below"),
        root,
    )
    jump_result = model.at(jump_targets, output_frame=root).run()
    acceleration_jump_world = (
        np.asarray(jump_result.acceleration)[0, :, 0]
        - np.asarray(jump_result.acceleration)[1, :, 0]
    )
    normal_world = surface_basis_world[:, 2]
    observed_jump = np.dot(acceleration_jump_world, normal_world)
    horizontal_offset = surface_xy_local_m - phase_origin_local_m[:2]
    sigma = (
        rho * amp * np.exp(1j * k_r * np.dot(horizontal_offset, direction_local[:2]))
    )
    expected_jump = -4.0 * np.pi * G_SI * sigma
    assert abs(observed_jump - expected_jump) <= (
        10.0 * epsilon_m * k_r * abs(expected_jump)
    )
