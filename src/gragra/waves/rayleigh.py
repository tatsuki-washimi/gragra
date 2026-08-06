"""Rayleigh-wave kinematic basis for a homogeneous elastic half-space.

Primary conventions source: ``docs/architecture/halfspace_conventions.md`` —
z-up, free surface at z = 0, elastic medium occupying z <= 0, depth h = -z >= 0,
fields proportional to e^{i(k x - omega t)} with the physical field Re[.].

The eigenfunctions implemented here are the exact two-exponential coupling
(e^{-q k h}, e^{-s k h}; q = sqrt(1 - c^2/v_P^2), s = sqrt(1 - c^2/v_S^2))
with the traction-free amplitude ratio B = -2iq/(1+s^2) A, i.e. they satisfy
both the elastic equations of motion and sigma_zz = sigma_xz = 0 at z = 0.
A single exponential with a fixed polarization satisfies neither, which is
why it is not used here.

Frame note: the sign of B depends on the potential-decomposition frame. The
value B = -2iq/(1+s^2) refers to the depth-coordinate decomposition
u_x = d(phi)/dx - d(psi)/dh, u_h = d(phi)/dh + d(psi)/dx (Aki & Richards
z-down style, then u_z = -u_h); the z-up curl form u = grad(phi) +
curl(psi y_hat) gives B = +2iq/(1+s^2). The displacement closed forms below
are frame-independent.

This module is G-free pure kinematics (no ``G_SI`` import or multiplication —
the same placement rationale as the witness-displacement helpers used by the
coherent cross-statistics path).
Complex outputs represent linear complex amplitudes or transfer-function-like
responses, not directly real-valued time-domain displacement fields.

Overflow guard: depth-dependence is restricted to the decaying exponentials
e^{-q k h}, e^{-s k h} with h >= 0. Inputs that would produce a growing
exponential — evaluation points above the free surface (z > 0), non-positive
wavenumbers — are rejected with ValueError. Attenuation beyond this evanescent
decay (anelastic Q models) is out of scope for this module.

Float64 coordinates / complex128 amplitudes (array-conventions).
"""

from __future__ import annotations

import numpy as np

from gragra._arrays import as_xyz_table

# Bisection width for rayleigh_speed, as a fraction of v_S. Chosen so that the
# propagated secular residual |dR/d(c/v_S)| * delta stays below the 1e-12 test
# gate: the slope at the root is ~3.3 for a Poisson solid, so a width of 1e-13
# leaves an order-of-magnitude margin.
_BISECTION_REL_TOL = 1e-13

# Primary root bracket for c_R/v_S. Physical media (0 <= Poisson ratio < 0.5)
# give c_R/v_S in ~[0.87, 0.96]; the lower edge 0.5 is a generous safety
# margin, and the upper edge stays strictly below the v_S branch point where
# the secular function's square root vanishes.
_PRIMARY_BRACKET = (0.5, 1.0 - 1e-9)


def _validate_speeds(v_p_m_s, v_s_m_s) -> tuple[float, float]:
    """Validate 0 < v_s < v_p (finite, non-bool scalars)."""
    values = []
    for name, value in (("v_p_m_s", v_p_m_s), ("v_s_m_s", v_s_m_s)):
        if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"{name} must be a positive finite number")
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a positive finite number")
        values.append(value)
    v_p, v_s = values
    if not v_s < v_p:
        raise ValueError(f"requires 0 < v_s_m_s < v_p_m_s, got v_s={v_s} >= v_p={v_p}")
    return v_p, v_s


def _validate_positive_scalar(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a positive finite number")
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def rayleigh_secular(c_m_s, v_p_m_s, v_s_m_s) -> np.ndarray:
    """Rayleigh secular function R(c) (Aki & Richards standard form).

    R(c) = (2 - c^2/v_S^2)^2 - 4 sqrt(1 - c^2/v_P^2) sqrt(1 - c^2/v_S^2)

    The Rayleigh phase speed c_R is the root R(c_R) = 0 on 0 < c < v_S.
    No extra c^2/v_S^2 factor multiplies the second term (a historically
    common misprint — see halfspace_conventions.md §3).

    Parameters
    ----------
    c_m_s : float or array-like
        Trial phase speed(s) in m/s. Must lie strictly inside (0, v_S) —
        the real-branch domain of the secular function; values outside
        raise ValueError.
    v_p_m_s, v_s_m_s : float
        P- and S-wave speeds in m/s (0 < v_s < v_p).

    Returns
    -------
    numpy.ndarray
        R(c), dimensionless, float64, same shape as ``c_m_s`` (read-only).
    """
    v_p, v_s = _validate_speeds(v_p_m_s, v_s_m_s)
    c = np.asarray(c_m_s, dtype=np.float64)
    if not np.all(np.isfinite(c)):
        raise ValueError("c_m_s must contain only finite values")
    if np.any(c <= 0.0) or np.any(c >= v_s):
        raise ValueError(
            "c_m_s must lie strictly inside (0, v_s_m_s) — the real branch "
            "of the Rayleigh secular function"
        )
    s2 = (c / v_s) ** 2
    p2 = (c / v_p) ** 2
    out = (2.0 - s2) ** 2 - 4.0 * np.sqrt(1.0 - p2) * np.sqrt(1.0 - s2)
    out = np.asarray(out, dtype=np.float64).copy()
    out.setflags(write=False)
    return out


def _secular_scaled(x: np.ndarray, ratio2: float) -> np.ndarray:
    """R as a function of x = c/v_S with ratio2 = (v_S/v_P)^2 (no validation)."""
    s2 = x * x
    p2 = s2 * ratio2
    return (2.0 - s2) ** 2 - 4.0 * np.sqrt(1.0 - p2) * np.sqrt(1.0 - s2)


def rayleigh_speed(v_p_m_s, v_s_m_s) -> float:
    """Rayleigh phase speed c_R from the secular equation (numpy bisection).

    Solves R(c_R) = 0 on the primary bracket [0.5 v_S, (1 - 1e-9) v_S]
    (physical media give c_R/v_S in ~0.87–0.96; 0.5 is a safety margin).
    If the primary bracket shows no sign change, scans c/v_S in
    [0.01, 0.999999] for the first sign-change interval; if none exists,
    raises ValueError carrying v_P/v_S and the sampled extrema of R
    (fail-closed with diagnostics). Converges to a width of 1e-13 v_S
    (an order of magnitude below the 1e-12 secular-residual test gate).

    For a Poisson solid (v_P/v_S = sqrt(3)) the exact closed form is
    c_R/v_S = sqrt(2 - 2/sqrt(3)) = 0.9194016868.

    Returns
    -------
    float
        c_R in m/s, guaranteed 0 < c_R < v_S.
    """
    v_p, v_s = _validate_speeds(v_p_m_s, v_s_m_s)
    ratio2 = (v_s / v_p) ** 2

    lo, hi = _PRIMARY_BRACKET
    f_lo = float(_secular_scaled(np.float64(lo), ratio2))
    f_hi = float(_secular_scaled(np.float64(hi), ratio2))
    if f_lo * f_hi > 0.0:
        # Fallback: scan for the first sign-change interval.
        grid = np.linspace(0.01, 0.999999, 4096)
        values = _secular_scaled(grid, ratio2)
        sign_change = np.nonzero(values[:-1] * values[1:] <= 0.0)[0]
        if sign_change.size == 0:
            raise ValueError(
                "rayleigh_speed: no sign change of the secular function found "
                f"(v_p/v_s={v_p / v_s:.10g}, R(0.5 v_s)={f_lo:.6g}, "
                f"R((1-1e-9) v_s)={f_hi:.6g}, scanned min={values.min():.6g}, "
                f"max={values.max():.6g})"
            )
        i = int(sign_change[0])
        lo, hi = float(grid[i]), float(grid[i + 1])
        f_lo = float(values[i])

    while hi - lo > _BISECTION_REL_TOL:
        mid = 0.5 * (lo + hi)
        f_mid = float(_secular_scaled(np.float64(mid), ratio2))
        if f_lo * f_mid <= 0.0:
            hi = mid
        else:
            lo = mid
            f_lo = f_mid
    return 0.5 * (lo + hi) * v_s


def _rayleigh_parameters(v_p: float, v_s: float) -> tuple[float, float, float]:
    """Return (c_R, q, s) for validated speeds (no re-validation)."""
    c_r = rayleigh_speed(v_p, v_s)
    q = float(np.sqrt(1.0 - (c_r / v_p) ** 2))
    s = float(np.sqrt(1.0 - (c_r / v_s) ** 2))
    return c_r, q, s


def _depth_profiles(zeta: np.ndarray, q: float, s: float) -> tuple:
    """Two-exponential depth profiles f_x, f_z at zeta = k*h >= 0.

    f_x(zeta) = e^{-q zeta} - (2 q s / (1+s^2)) e^{-s zeta}   (horizontal)
    f_z(zeta) = e^{-q zeta} - (2 / (1+s^2))     e^{-s zeta}   (vertical)

    f_z < 0 for all zeta >= 0 (the coefficient 2/(1+s^2) > 1 multiplies the
    slower-decaying exponential), so the ellipticity denominator never
    vanishes; f_x crosses zero once (the horizontal-displacement node).
    """
    coef_x = 2.0 * q * s / (1.0 + s * s)
    coef_z = 2.0 / (1.0 + s * s)
    exp_q = np.exp(-q * zeta)
    exp_s = np.exp(-s * zeta)
    return exp_q - coef_x * exp_s, exp_q - coef_z * exp_s


def rayleigh_displacement(
    positions_m: list | tuple | np.ndarray,
    direction: list | tuple | np.ndarray,
    wavenumber_rad_m: float | int,
    v_p_m_s: float | int,
    v_s_m_s: float | int,
    *,
    amplitude_m: float | int = 1.0,
) -> np.ndarray:
    """Complex Rayleigh-wave displacement field u(r) in the half-space.

    Complex outputs represent linear complex amplitudes or transfer-function-
    like responses, not directly real-valued time-domain displacement fields.

    Normalization and phase reference: the vertical surface displacement at
    the horizontal phase origin (k_hat . r_xy = 0, z = 0) is ``+amplitude_m``
    (real). The horizontal component then carries the 90-degree quadrature
    phase (factor i) required by the eigenfunction, giving retrograde
    particle motion at the surface — Im(u_x conj(u_z)) < 0 there, with a
    sign flip below the horizontal-displacement node at k h = 1.209
    (Poisson solid; see halfspace_conventions.md §3.2). The overall particle-
    motion sense is normalization-independent.

    G-free pure kinematics (no G_SI). Evaluation points with z > 0 (above the
    free surface) raise ValueError — the exponential growth direction is
    structurally rejected (overflow guard).

    Parameters
    ----------
    positions_m : array-like of shape (N, 3)
        Evaluation points in meters, z-up, free surface at z = 0, medium
        z <= 0.
    direction : array-like of shape (3,)
        Horizontal propagation direction (must lie in the xy plane; the z
        component must vanish). Normalized internally.
    wavenumber_rad_m : float
        Horizontal wavenumber k > 0 in rad/m.
    v_p_m_s, v_s_m_s : float
        Elastic wave speeds in m/s (0 < v_s < v_p).
    amplitude_m : float, optional
        Vertical surface displacement amplitude in meters (positive finite).

    Returns
    -------
    numpy.ndarray
        Complex displacement of shape (N, 3), complex128, component axis
        trailing, read-only.
    """
    positions = as_xyz_table("positions_m", positions_m)
    v_p, v_s = _validate_speeds(v_p_m_s, v_s_m_s)
    k = _validate_positive_scalar("wavenumber_rad_m", wavenumber_rad_m)
    amplitude = _validate_positive_scalar("amplitude_m", amplitude_m)

    dir_arr = np.asarray(direction, dtype=np.float64)
    if dir_arr.shape != (3,):
        raise ValueError(f"direction must have shape (3,), got {dir_arr.shape}")
    if not np.all(np.isfinite(dir_arr)):
        raise ValueError("direction must contain only finite values")
    norm = float(np.linalg.norm(dir_arr))
    if norm == 0.0:
        raise ValueError("direction must be non-zero")
    if abs(dir_arr[2]) > 1e-12 * norm:
        raise ValueError(
            "direction must be horizontal (z component must vanish) — "
            "Rayleigh waves propagate along the free surface"
        )
    k_hat = dir_arr / norm

    z = positions[:, 2]
    above = np.nonzero(z > 0.0)[0]
    if above.size:
        raise ValueError(
            f"positions_m must satisfy z <= 0 (medium side of the free "
            f"surface); first offending index: {int(above[0])} "
            f"(z={z[above[0]]!r}) — evaluation above the surface would be "
            "the exponential growth direction"
        )

    _, q, s = _rayleigh_parameters(v_p, v_s)
    zeta = k * (-z)  # k h >= 0
    f_x, f_z = _depth_profiles(zeta, q, s)

    # Scale so that u_z(surface, phase origin) = +amplitude (f_z(0) < 0, so
    # alpha is negative real; the particle-motion sense is invariant under
    # any complex rescale since Im(u_x conj(u_z)) picks up |alpha|^2 only).
    f_z0 = float(_depth_profiles(np.float64(0.0), q, s)[1])
    alpha = amplitude / (q * k * f_z0)

    phase = np.exp(1j * k * (positions[:, 0] * k_hat[0] + positions[:, 1] * k_hat[1]))
    u_x = alpha * 1j * k * f_x * phase
    u_z = alpha * q * k * f_z * phase

    out = np.empty((len(positions), 3), dtype=np.complex128)
    out[:, 0] = u_x * k_hat[0]
    out[:, 1] = u_x * k_hat[1]
    out[:, 2] = u_z
    out.setflags(write=False)
    return out


def rayleigh_ellipticity(depth_m, wavenumber_rad_m, v_p_m_s, v_s_m_s) -> np.ndarray:
    """Depth-dependent Rayleigh H/V ellipticity |u_x(h)| / |u_z(h)|.

    Derived from the two-exponential eigenfunctions (not a fixed surface
    value): H/V = |f_x(k h)| / (q |f_z(k h)|). The denominator never vanishes
    (f_z < 0 for all depths); the ratio crosses zero at the horizontal-
    displacement node (k h = 1.209 for a Poisson solid, where the surface
    value is H/V = 0.68125). Evaluated with the slower-decaying exp(-s k h)
    factored out, so the ratio stays finite at arbitrarily large depths (both
    raw profiles underflow beyond k h ~ 700/s and would give 0/0 = NaN); the
    deep limit is H/V -> s.

    G-free, dimensionless. ``depth_m`` is depth below the surface (h >= 0;
    negative depths raise ValueError).

    Returns
    -------
    numpy.ndarray
        H/V, float64, same shape as ``depth_m`` (read-only).
    """
    v_p, v_s = _validate_speeds(v_p_m_s, v_s_m_s)
    k = _validate_positive_scalar("wavenumber_rad_m", wavenumber_rad_m)
    depth = np.asarray(depth_m, dtype=np.float64)
    if not np.all(np.isfinite(depth)):
        raise ValueError("depth_m must contain only finite values")
    if np.any(depth < 0.0):
        raise ValueError("depth_m must be non-negative (depth below surface)")

    _, q, s = _rayleigh_parameters(v_p, v_s)
    # Factor out exp(-s*zeta) (the slower decay, q > s) from both profiles:
    # g = f / exp(-s*zeta). The ratio is unchanged where f is representable,
    # but stays finite at any depth — the raw profiles underflow to 0 for
    # zeta beyond ~700/s and 0/0 would return NaN. decay = exp(-(q-s)*zeta)
    # lies in (0, 1] and its own underflow to 0 is harmless (deep limit
    # H/V -> coef_x / (q * coef_z) = s).
    zeta = k * depth
    coef_x = 2.0 * q * s / (1.0 + s * s)
    coef_z = 2.0 / (1.0 + s * s)
    decay = np.exp(-(q - s) * zeta)
    g_x = decay - coef_x
    g_z = decay - coef_z  # < 0 for all zeta (coef_z > 1 >= decay)
    out = np.abs(g_x) / (q * np.abs(g_z))
    out = np.asarray(out, dtype=np.float64).copy()
    out.setflags(write=False)
    return out
