"""G-free half-space closed-form benchmarks.

Primary conventions source: ``docs/architecture/halfspace_conventions.md`` §5 —
the vertical-incidence P-wave transfer functions are defined through

    delta_a_z(H) = -4 pi G rho_0 xi_inc T(kH),   kH = omega H / v_P

where ``xi_inc`` is the *incident* displacement amplitude (before the factor-2
free-surface doubling) and ``H`` the observation depth. This module returns the
dimensionless ``T`` only; it is **G-free** (no ``G_SI`` import or
multiplication — the G-free acceleration amplitude ``-4 pi rho_0 xi_inc T``
has dimension kg/m^2 = acceleration/G, and multiplying by ``G_SI`` is the
caller's responsibility). ``(4 pi G rho)^2 x PSD`` assembly stays in nn-sus
(division of labor, conventions §7).

Identities pinned by the tests (conventions §5): T_full - T_bulk = -1 for all
kH, T_bulk(0) = -1, T_full(0) = -2, nulls at kH = pi/3 (T_bulk) and pi/2
(T_full).

The Rayleigh depth-reduction factor is the *kinematic displacement-magnitude*
reduction |u(h)| / |u(0)| derived from the exact two-exponential eigenfunctions
(``gragra.waves.rayleigh``) — NOT the single-exponential gamma = 0.8
approximation (nn-sus ``calc_acc_rayleigh_analytic``), which satisfies neither
the elastic equations nor the traction-free boundary and is kept only as an
external comparison reference. It is also not an acceleration-transfer claim:
the Newtonian-noise depth dependence couples the whole field and is evaluated
separately via field integrals.

Provenance: transfer functions ported from nn-sus
``src/nnsus/newtonian/halfspace.py`` (commit 9a5e426, 2026-06-23), Sections
10.2/11.2 of its design note ``free_surface_reflection.md``; re-derived and
verified independently on 2026-07-04 (conventions §8).
"""

from __future__ import annotations

import numpy as np

# Private-helper contract: this module is the only external consumer of these
# waves.rayleigh internals (the depth profiles / (c_R, q, s) triple / scalar
# validators). Any signature change there must update this import site and the
# tests of both modules together — see test_wave_rayleigh.py and
# test_analytic_halfspace.py.
from gragra.waves.rayleigh import (
    _depth_profiles,
    _rayleigh_parameters,
    _validate_positive_scalar,
    _validate_speeds,
)


def _validate_non_negative_grid(name: str, value) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return arr


def halfspace_transfer_bulk(k_h) -> np.ndarray:
    """Dimensionless bulk transfer T_bulk(kH) = 1 - 2 cos(kH).

    Bulk density-perturbation contribution only (no surface mass sheet), for a
    vertically incident P wave in a homogeneous half-space, observed at depth
    H below the free surface; kH = omega H / v_P >= 0. The physical
    (G-carrying) relation is delta_a_z(H) = -4 pi G rho_0 xi_inc T_bulk(kH)
    with xi_inc the incident amplitude before surface doubling
    (halfspace_conventions.md §5) — this function is G-free and returns T only.

    Fixed points: T_bulk(0) = -1; null at kH = pi/3 (f = v_P / 6H).

    Returns
    -------
    numpy.ndarray
        T_bulk, dimensionless float64, same shape as ``k_h`` (read-only).
    """
    k_h_arr = _validate_non_negative_grid("k_h", k_h)
    out = 1.0 - 2.0 * np.cos(k_h_arr)
    out = np.asarray(out, dtype=np.float64).copy()
    out.setflags(write=False)
    return out


def halfspace_transfer_full(k_h) -> np.ndarray:
    """Dimensionless full transfer T_full(kH) = -2 cos(kH) = T_bulk + T_sheet.

    Adds the free-surface mass-sheet term T_sheet = -1 to the bulk term, so
    T_full - T_bulk = -1 identically. Same definition relation and G-free
    convention as :func:`halfspace_transfer_bulk`.

    Fixed points: T_full(0) = -2; null at kH = pi/2 (f = v_P / 4H). Note that
    T_full(0) = -2 is the limit approached from *inside* the medium (H -> 0+);
    for an observer above the surface the sheet term flips sign, so this value
    does not apply to surface-mounted detectors.

    Returns
    -------
    numpy.ndarray
        T_full, dimensionless float64, same shape as ``k_h`` (read-only).
    """
    k_h_arr = _validate_non_negative_grid("k_h", k_h)
    out = -2.0 * np.cos(k_h_arr)
    out = np.asarray(out, dtype=np.float64).copy()
    out.setflags(write=False)
    return out


def rayleigh_depth_reduction(depth_m, wavenumber_rad_m, v_p_m_s, v_s_m_s) -> np.ndarray:
    """Kinematic Rayleigh displacement-magnitude depth reduction R(h).

    R(h) = |u(h)| / |u(0)| with |u| = sqrt(|u_x|^2 + |u_z|^2) evaluated from
    the exact two-exponential eigenfunctions (gragra.waves.rayleigh). R(0) = 1,
    and it is NOT a single exponential — for a Poisson solid R(kh=1) = 0.8374,
    far above e^{-1} = 0.3679, which is what a naive exp(-kh) model would give.

    Monotonic decrease with depth holds only for v_P/v_S below ~2.7 (it covers
    the Poisson solid and most hard-rock sites). For larger ratios the vertical
    eigenfunction develops a subsurface maximum and R can exceed 1 (e.g.
    v_P/v_S = 5 gives max R = 1.031 near kh = 0.81) — real physics, relevant
    for water-saturated sediments where v_P/v_S reaches 3-15, so "reduction"
    must not be read as R <= 1 unconditionally.

    This is a G-free, dimensionless *kinematic* benchmark of the displacement
    field only; the depth dependence of the Newtonian-noise *acceleration*
    couples the whole field and must be computed via field integrals (it is
    not this ratio).

    Parameters
    ----------
    depth_m : float or array-like
        Depth(s) below the free surface in meters (h >= 0).
    wavenumber_rad_m : float
        Horizontal Rayleigh wavenumber k > 0 in rad/m.
    v_p_m_s, v_s_m_s : float
        Elastic wave speeds in m/s (0 < v_s < v_p).

    Returns
    -------
    numpy.ndarray
        R(h), dimensionless float64, same shape as ``depth_m`` (read-only).
    """
    v_p, v_s = _validate_speeds(v_p_m_s, v_s_m_s)
    k = _validate_positive_scalar("wavenumber_rad_m", wavenumber_rad_m)
    depth = _validate_non_negative_grid("depth_m", depth_m)

    _, q, s = _rayleigh_parameters(v_p, v_s)
    f_x, f_z = _depth_profiles(k * depth, q, s)
    f_x0, f_z0 = _depth_profiles(np.float64(0.0), q, s)
    magnitude = np.hypot(f_x, q * f_z)
    magnitude_surface = float(np.hypot(f_x0, q * f_z0))
    out = magnitude / magnitude_surface
    out = np.asarray(out, dtype=np.float64).copy()
    out.setflags(write=False)
    return out
