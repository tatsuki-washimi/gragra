"""Literature-sourced single-mode and reflected-P body-wave half-space gravity.

Primary source: J. Harms, "Terrestrial gravity fluctuations", Living Reviews
in Relativity 22, 6 (2019), DOI 10.1007/s41114-019-0022-2 (2019 edition; do
not mix Eq. numbers with the 2015 edition), Section 3.4.1, pp. 42-43,
Eq. (86)-(90). Design mapping, evidence classification (A/B/C), sign/gauge
conventions, and the independent boundary-value re-derivation of Eq. (90)'s
``delta(nu, k_rho)`` are recorded in full in ``docs/design/literature-b.md``
and ``docs/validation/literature-b.md``; this docstring only summarizes the
equation-to-code mapping.

- Eq. (86) (single P, ground, ``z > 0``): evidence A.
- Eq. (87) (single P, underground, ``z < 0``): evidence A.
- Eq. (88) (single SV, both sides, one formula): evidence A.
- Eq. (90) (reflected-P ground composite, closed ``delta(nu, k_rho)``):
  evidence A for the literal literature coefficient. Eq. (89) is the
  un-combined bracket form. The private
  ``_harms2019_reflected_p_delta`` helper preserves Eq. (90) exactly,
  including its ``-1j`` term; production uses the separate
  ``_physical_reflected_p_delta`` coefficient obtained from the same
  traction-free displacement field and its actual surface mass sheet. The
  two coefficients are intentionally kept distinct; this documents the
  physical API contract without claiming that Eq. (90) is a typographical
  error. Both routes are independently checked in
  ``tests/test_bodywave_literature_oracle.py``.
- Acceleration and gradient are this module's own analytic derivative
  (``a = -grad(Phi)``, ``Gamma = grad(a)``) of the potentials above via the
  ``Phi(x,y,z) = E(x,y) * F(z)`` factorization (evidence B; same pattern as
  ``gragra.analytic.rayleigh_literature``).

Sign/gauge conventions fixed by this module (see the design doc for the
rationale): free surface at ``z=0``, medium at ``z<0``, z-axis up; the
incident wave (single-P, single-SV, and the P part of the reflected
composite) propagates upward (``+z``, arriving at the surface from below);
``theta=incidence_angle_rad`` is always measured against that mode's own
wave speed (``v_p_m_s`` for "P", ``v_s_m_s`` for "SV" and for the reflected
composite's incident/reflected P legs); exact grazing incidence
(``theta = pi/2``) is rejected by this API even though the closed forms
themselves stay finite there.

G-free (no ``G_SI`` import or multiplication) -- see
``gragra.observables.bodywave_literature`` for the SI entry point.
Complex outputs represent linear complex amplitudes / transfer-function-like
responses, not directly real-valued time-domain gravitational fields.
Float64 coordinates / complex128 amplitudes throughout (array-conventions).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from gragra._arrays import _check_no_bool, as_xyz_table, as_xyz_vector
from gragra.analytic.values import GFreeGravityValues
from gragra.waves.rayleigh import _validate_speeds

ArrayLike = list | tuple | np.ndarray

_HALF_PI = np.pi / 2.0


def _reject_non_real_dtype(name: str, value) -> None:
    """Reject bool/object/complex inputs before any float64 conversion.

    Same rationale as ``rayleigh_literature._reject_non_real_dtype``: a plain
    ``np.asarray(value, dtype=np.float64)`` silently drops a complex input's
    imaginary part and silently accepts ``bool`` as 0.0/1.0, so this guards
    real-only parameters against both before that conversion runs.
    """
    _check_no_bool(value, name)
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must not be boolean")
    raw = np.asarray(value)
    if raw.dtype.kind == "b":
        raise TypeError(f"{name} must not contain boolean values")
    if raw.dtype.kind == "O":
        raise TypeError(f"{name} must not contain object dtype")
    if raw.dtype.kind == "c":
        raise TypeError(f"{name} must contain real values (got a complex dtype)")


def _validate_non_negative_scalar(name: str, value) -> float:
    _reject_non_real_dtype(name, value)
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a non-negative finite number")
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a non-negative finite number")
    return value


def _validate_incidence_angle(name: str, value) -> float:
    """Validate a single scalar incidence angle in ``[0, pi/2)``.

    Exact grazing incidence (``pi/2``) is rejected by this API's contract
    (docs/design/literature-b.md), independent of the closed forms
    themselves remaining finite in that limit.
    """
    _reject_non_real_dtype(name, value)
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a real finite scalar")
    theta = float(value)
    if not np.isfinite(theta):
        raise ValueError(f"{name} must be finite")
    if theta < 0.0 or theta >= _HALF_PI:
        raise ValueError(
            f"{name} must satisfy 0 <= {name} < pi/2 (exact grazing rejected)"
        )
    return theta


def _validate_azimuth(name: str, value) -> float:
    _reject_non_real_dtype(name, value)
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a real finite scalar")
    azimuth = float(value)
    if not np.isfinite(azimuth):
        raise ValueError(f"{name} must be finite")
    return azimuth


def _validate_frequency(name: str, value) -> np.ndarray:
    _reject_non_real_dtype(name, value)
    freq = np.asarray(value, dtype=np.float64)
    if freq.ndim not in (0, 1):
        raise ValueError(f"{name} must be a scalar or a 1D array")
    if freq.ndim == 1 and freq.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty 1D array")
    if not np.all(np.isfinite(freq)) or np.any(freq <= 0.0):
        raise ValueError(f"{name} must contain only positive finite values")
    return freq


def _validate_amplitude(name: str, value, *, k_len: int | None) -> np.ndarray:
    """Validate a scalar or (K,)-shaped complex amplitude.

    Same "scalar or (K,), no other implicit broadcast" contract as
    ``rayleigh_literature._validate_amplitude``.
    """
    _check_no_bool(value, name)
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must not be boolean")
    raw = np.asarray(value)
    if raw.dtype.kind == "b":
        raise TypeError(f"{name} must not contain boolean values")
    if raw.dtype.kind == "O":
        raise TypeError(f"{name} must not contain object dtype")
    if raw.ndim not in (0, 1):
        raise ValueError(f"{name} must be a scalar or a 1D array")
    arr = np.asarray(raw, dtype=np.complex128)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")
    if arr.ndim == 1:
        if k_len is None:
            raise ValueError(f"{name} must be a scalar when frequency_hz is a scalar")
        if arr.shape[0] != k_len:
            raise ValueError(
                f"{name} array length ({arr.shape[0]}) must match "
                f"frequency_hz length ({k_len})"
            )
    return arr


def _validate_targets_and_split(
    targets_m: ArrayLike, *, allow_underground: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Validate targets_m, reject z==0 (and z<=0 when underground is disallowed).

    Returns (targets, z) with z = targets[:, 2].
    """
    _reject_non_real_dtype("targets_m", targets_m)
    targets = as_xyz_table("targets_m", targets_m)
    z = targets[:, 2]
    if np.any(z == 0.0):
        offending = int(np.nonzero(z == 0.0)[0][0])
        raise ValueError(
            f"targets_m must not lie exactly on the free surface z == 0 "
            f"(first offending index: {offending})"
        )
    if not allow_underground and np.any(z < 0.0):
        offending = int(np.nonzero(z < 0.0)[0][0])
        raise ValueError(
            f"targets_m must satisfy z > 0 (ground only; first offending "
            f"index: {offending}) -- the reflected-P composite is only "
            "defined above the free surface"
        )
    return targets, z


def _horizontal_phase(targets: np.ndarray, origin: np.ndarray, azimuth: float, k_rho):
    """Return (qx, qy, phase), the horizontal wavevector components
    ``k_rho * (cos(azimuth), sin(azimuth))`` and
    ``phase = exp(i * k_rho * ((x-x0)*cos(azimuth) + (y-y0)*sin(azimuth)))``,
    shape (K,M) or (M,).

    Returning the *scaled* ``qx, qy`` (not the bare unit-direction cosines)
    matters at exact normal incidence (``k_rho == 0``): ``a_x = -i*qx*Phi``
    etc. must vanish there regardless of ``azimuth``, which only holds if
    ``qx, qy`` themselves carry the ``k_rho`` factor.
    """
    n1 = np.cos(azimuth)
    n2 = np.sin(azimuth)
    proj = (targets[:, 0] - origin[0]) * n1 + (targets[:, 1] - origin[1]) * n2
    phase = np.exp(1j * k_rho * proj)
    return k_rho * n1, k_rho * n2, phase


def _assemble_observables(
    phase: np.ndarray, phi_total: np.ndarray, s1: np.ndarray, s2: np.ndarray, qx, qy
) -> GFreeGravityValues:
    """Build (potential, acceleration, gradient) from the F/F'/F'' triple.

    phi_total = E*F, s1 = E*F' (so a_z = -s1), s2 = E*F'' (so Gamma_zz = -s2),
    following docs/design/literature-b.md's factorization.
    """
    acceleration = np.empty(phase.shape + (3,), dtype=np.complex128)
    acceleration[..., 0] = -1j * qx * phi_total
    acceleration[..., 1] = -1j * qy * phi_total
    acceleration[..., 2] = -s1

    gradient = np.empty(phase.shape + (3, 3), dtype=np.complex128)
    gradient[..., 0, 0] = qx * qx * phi_total
    gradient[..., 0, 1] = qx * qy * phi_total
    gradient[..., 1, 0] = gradient[..., 0, 1]
    gradient[..., 1, 1] = qy * qy * phi_total
    gradient[..., 0, 2] = -1j * qx * s1
    gradient[..., 2, 0] = gradient[..., 0, 2]
    gradient[..., 1, 2] = -1j * qy * s1
    gradient[..., 2, 1] = gradient[..., 1, 2]
    gradient[..., 2, 2] = -s2

    return GFreeGravityValues(
        potential=phi_total, acceleration=acceleration, gradient=gradient
    )


def _harms2019_reflected_p_delta(k_rho, k_zp, k_zs, k_s):
    """Return the literal ``delta(nu, k_rho)`` from H19 Eq. (90).

    This helper intentionally preserves the published coefficient, including
    its ``-1j`` term.  The production reflected-P observable uses the separate
    physical-source coefficient below; keeping both expressions private and
    distinct makes the literature transcription independently testable.
    """
    r_rho = k_rho / k_s
    r_zp = k_zp / k_s
    r_zs = k_zs / k_s
    d = 1.0 - 2.0 * r_rho**2
    denominator = d**2 + 4.0 * r_rho**2 * r_zp * r_zs
    numerator = 8.0 * r_rho**2 * r_zp * r_zs - 1j * 4.0 * r_rho * r_zp * d
    return numerator / denominator


def _physical_reflected_p_delta(k_rho, k_zp, k_zs, k_s):
    """Return the coefficient for the same traction-free displacement field.

    The independent bulk-plus-surface Green-function reconstruction uses the
    actual total surface displacement.  With the displacement-potential
    reflection ratios ``B`` and ``C`` this coefficient is ``(1+B) + i*C``;
    simplifying the boundary-value solution gives the expression below.  It
    differs from the literal Eq. (90) transcription only in the sign of the
    imaginary converted-SV term.
    """
    r_rho = k_rho / k_s
    r_zp = k_zp / k_s
    r_zs = k_zs / k_s
    d = 1.0 - 2.0 * r_rho**2
    denominator = d**2 + 4.0 * r_rho**2 * r_zp * r_zs
    numerator = 8.0 * r_rho**2 * r_zp * r_zs + 1j * 4.0 * r_rho * r_zp * d
    return numerator / denominator


def halfspace_single_bodywave_observables_g_free(
    targets_m: ArrayLike,
    *,
    wave_type: Literal["P", "SV"],
    density_kg_m3: float,
    v_p_m_s: float,
    v_s_m_s: float,
    frequency_hz: float | ArrayLike,
    incidence_angle_rad: float,
    azimuth_rad: float,
    displacement_m: float | complex | ArrayLike,
    phase_origin_m: ArrayLike = (0.0, 0.0, 0.0),
) -> GFreeGravityValues:
    """G-free gravity of a single upward-propagating plane P or SV wave.

    Implements H19 Eq. (86) (P, ``z>0``), Eq. (87) (P, ``z<0``), and Eq. (88)
    (SV, one formula valid on both sides), **without** compositing a
    free-surface reflection (that is
    :func:`halfspace_reflected_p_above_observables_g_free`). Acceleration
    and gradient are this module's own analytic derivative of those closed
    forms (evidence B; see ``docs/design/literature-b.md``).

    Parameters
    ----------
    targets_m : array-like of shape (M, 3)
        Evaluation points in meters (z-up; free surface at z=0). Points with
        ``z == 0`` are rejected; ``z > 0`` and ``z < 0`` targets may be
        mixed in one call.
    wave_type : {"P", "SV"}
        Which single mode to evaluate.
    density_kg_m3 : float
        Background mass density in kg/m^3 (non-negative; zero accepted).
    v_p_m_s, v_s_m_s : float
        P- and S-wave speeds in m/s. Requires ``0 < v_s_m_s < v_p_m_s`` and
        ``v_p_m_s**2 > (4/3) * v_s_m_s**2`` (stable isotropic half-space).
    frequency_hz : float or array-like of shape (K,)
        Positive frequency/frequencies in Hz.
    incidence_angle_rad : float
        Incidence angle ``theta`` measured from the surface normal, against
        this mode's own speed (``v_p_m_s`` for "P", ``v_s_m_s`` for "SV").
        Must satisfy ``0 <= theta < pi/2`` (exact grazing rejected). One
        scalar per call -- no per-frequency angle array.
    azimuth_rad : float
        Horizontal propagation azimuth in radians. One scalar per call.
    displacement_m : float, complex, or array-like of shape (K,)
        Complex incident-wave displacement amplitude along its own
        polarization direction (H19's ``xi_0^P`` / ``xi_0^SV``; P
        polarization ``sin(theta) h + cos(theta) e_z``, SV polarization
        ``cos(theta) h - sin(theta) e_z``, both for the upward-propagating
        convention -- the polarization *vector* itself does not enter the
        potential formulas, which already reduce to this scalar amplitude).
        Zero accepted. A 1D array of length K (one amplitude per frequency)
        is only accepted together with a 1D ``frequency_hz`` of the same
        length K.
    phase_origin_m : array-like of shape (3,), optional
        Fixed 3D phase-reference point in meters. ``displacement_m`` is the
        incident wave's own complex amplitude evaluated AT this point, not
        necessarily at ``z=0``; a nonzero ``phase_origin_m[2]`` re-references
        that amplitude to ``z=0`` via the incident wave's own vertical
        wavenumber (``exp(-i*k_z*z0)``) without moving the free surface
        itself. Default the coordinate origin.

    Returns
    -------
    GFreeGravityValues
        G-free potential (kg/m), acceleration (kg/m^2), and gradient
        (kg/m^3), complex128, read-only.
    """
    if wave_type not in ("P", "SV"):
        raise ValueError(f"wave_type must be 'P' or 'SV', got {wave_type!r}")

    targets, z = _validate_targets_and_split(targets_m, allow_underground=True)
    rho = _validate_non_negative_scalar("density_kg_m3", density_kg_m3)
    v_p, v_s = _validate_speeds(v_p_m_s, v_s_m_s)
    if not v_p * v_p > (4.0 / 3.0) * v_s * v_s:
        raise ValueError(
            "requires v_p_m_s**2 > (4/3) * v_s_m_s**2 for a stable "
            "isotropic half-space (Poisson ratio < 0.5)"
        )
    theta = _validate_incidence_angle("incidence_angle_rad", incidence_angle_rad)
    azimuth = _validate_azimuth("azimuth_rad", azimuth_rad)

    freq_raw = _validate_frequency("frequency_hz", frequency_hz)
    is_series = freq_raw.ndim == 1
    k_len = freq_raw.shape[0] if is_series else None

    amp_raw = _validate_amplitude("displacement_m", displacement_m, k_len=k_len)
    amplitude = amp_raw[:, None] if amp_raw.ndim == 1 else amp_raw

    _reject_non_real_dtype("phase_origin_m", phase_origin_m)
    origin = as_xyz_vector("phase_origin_m", phase_origin_m)

    v = v_p if wave_type == "P" else v_s
    k_flat = 2.0 * np.pi * freq_raw / v  # total wavenumber, scalar or (K,)
    k_total = k_flat[:, None] if is_series else k_flat  # (K,1) or scalar
    k_rho = k_total * np.sin(theta)
    k_z = k_total * np.cos(theta)

    # phase_origin_m[2]: re-reference the incident wave's amplitude to z=0
    # (the free surface itself is never moved -- see docs/design/literature-b.md
    # "Sign of kP.r in Eq.(87)"). displacement_m is the incident wave's own
    # complex amplitude evaluated AT phase_origin_m; converting it to the
    # z=0-referenced value the closed forms below assume uses the incident
    # wave's own vertical wavenumber k_z, exp(-i*k_z*z0).
    amplitude = amplitude * np.exp(-1j * k_z * origin[2])

    qx, qy, phase = _horizontal_phase(targets, origin, azimuth, k_rho)

    above = z > 0.0
    below = z < 0.0
    z_above = z[above]
    z_below = z[below]
    phase_above = phase[..., above]
    phase_below = phase[..., below]

    phi_total = np.empty(phase.shape, dtype=np.complex128)
    s1 = np.empty(phase.shape, dtype=np.complex128)
    s2 = np.empty(phase.shape, dtype=np.complex128)

    if wave_type == "P":
        c1 = -rho * amplitude / (1j * k_total)  # Eq. (86)/(87) prefactor, G-free
        f_ground = 2.0 * np.pi * c1 * np.exp(-k_rho * z_above)
        phi_total[..., above] = phase_above * f_ground
        s1[..., above] = phase_above * (-k_rho * f_ground)
        s2[..., above] = phase_above * (k_rho**2 * f_ground)

        f_under = (
            2.0
            * np.pi
            * c1
            * (2.0 * np.exp(1j * k_z * z_below) - np.exp(k_rho * z_below))
        )
        f_under_p1 = (
            2.0
            * np.pi
            * c1
            * (
                2.0j * k_z * np.exp(1j * k_z * z_below)
                - k_rho * np.exp(k_rho * z_below)
            )
        )
        f_under_p2 = (
            2.0
            * np.pi
            * c1
            * (
                -2.0 * k_z**2 * np.exp(1j * k_z * z_below)
                - k_rho**2 * np.exp(k_rho * z_below)
            )
        )
        phi_total[..., below] = phase_below * f_under
        s1[..., below] = phase_below * f_under_p1
        s2[..., below] = phase_below * f_under_p2
    else:  # SV, Eq. (88): one closed form, distance |z| to the surface
        c2 = rho * amplitude / k_total  # Eq. (88) prefactor, G-free
        f_ground = 2.0 * np.pi * c2 * np.exp(-k_rho * z_above)
        phi_total[..., above] = phase_above * f_ground
        s1[..., above] = phase_above * (-k_rho * f_ground)
        s2[..., above] = phase_above * (k_rho**2 * f_ground)

        f_under = 2.0 * np.pi * c2 * np.exp(k_rho * z_below)
        phi_total[..., below] = phase_below * f_under
        s1[..., below] = phase_below * (k_rho * f_under)
        s2[..., below] = phase_below * (k_rho**2 * f_under)

    return _assemble_observables(phase, phi_total, s1, s2, qx, qy)


def halfspace_reflected_p_above_observables_g_free(
    targets_m: ArrayLike,
    *,
    density_kg_m3: float,
    v_p_m_s: float,
    v_s_m_s: float,
    frequency_hz: float | ArrayLike,
    incidence_angle_rad: float,
    azimuth_rad: float,
    incident_displacement_m: float | complex | ArrayLike,
    phase_origin_m: ArrayLike = (0.0, 0.0, 0.0),
) -> GFreeGravityValues:
    """G-free gravity above ground of an incident P + reflected P + converted SV.

    Uses the coefficient consistent with the independently solved
    traction-free displacement field and its actual surface mass sheet. The
    literal H19 Eq. (90) coefficient, including its ``-1j`` term, remains
    available through the private ``_harms2019_reflected_p_delta`` helper;
    production deliberately uses the separate physical ``+1j`` coefficient
    rather than relabeling it as Eq. (90). This distinction and the full
    potential/acceleration/gradient reconstruction are checked in
    ``docs/design/literature-b.md`` and
    ``tests/test_bodywave_literature_oracle.py``. Acceleration and gradient
    are this module's own analytic derivative (evidence B).

    Only the ground side (``z > 0``) is defined by this entry point -- the
    paper only gives the above-surface composite (Eq. 89/90); the
    underground field of the full reflected/converted composite is out of
    scope for this PR (docs/design/literature-b.md).

    Parameters
    ----------
    targets_m : array-like of shape (M, 3)
        Evaluation points in meters (z-up). ``z <= 0`` is rejected.
    density_kg_m3 : float
        Background mass density in kg/m^3 (non-negative; zero accepted).
    v_p_m_s, v_s_m_s : float
        P- and S-wave speeds in m/s. Requires ``0 < v_s_m_s < v_p_m_s`` and
        ``v_p_m_s**2 > (4/3) * v_s_m_s**2``.
    frequency_hz : float or array-like of shape (K,)
        Positive frequency/frequencies in Hz.
    incidence_angle_rad : float
        Incident P-wave angle from the surface normal, against ``v_p_m_s``.
        Must satisfy ``0 <= theta < pi/2``. At ``theta=0`` the closed form
        evaluates to exactly zero (every term of Eq. (90)'s numerator
        carries a ``k_rho`` factor that vanishes there), matching H19's
        "vanishes for vertically propagating incident P-waves" -- this is
        not special-cased, it falls out of evaluating Eq. (90) literally.
    azimuth_rad : float
        Horizontal propagation azimuth in radians.
    incident_displacement_m : float, complex, or array-like of shape (K,)
        Complex incident P-wave displacement amplitude (H19's ``xi_0^P``).
        Zero accepted. A 1D array of length K only with a matching 1D
        ``frequency_hz``.
    phase_origin_m : array-like of shape (3,), optional
        Fixed 3D phase-reference point in meters. ``incident_displacement_m``
        is the incident P wave's own complex amplitude evaluated AT this
        point; a nonzero ``phase_origin_m[2]`` re-references it to ``z=0``
        via the incident leg's own vertical wavenumber ``k_zp``
        (``exp(-i*k_zp*z0)``), applied once and shared by the
        reflected/converted legs rather than separately per leg (see
        ``docs/design/literature-b.md``).

    Returns
    -------
    GFreeGravityValues
        G-free potential (kg/m), acceleration (kg/m^2), and gradient
        (kg/m^3), complex128, read-only.
    """
    targets, z = _validate_targets_and_split(targets_m, allow_underground=False)
    rho = _validate_non_negative_scalar("density_kg_m3", density_kg_m3)
    v_p, v_s = _validate_speeds(v_p_m_s, v_s_m_s)
    if not v_p * v_p > (4.0 / 3.0) * v_s * v_s:
        raise ValueError(
            "requires v_p_m_s**2 > (4/3) * v_s_m_s**2 for a stable "
            "isotropic half-space (Poisson ratio < 0.5)"
        )
    theta = _validate_incidence_angle("incidence_angle_rad", incidence_angle_rad)
    azimuth = _validate_azimuth("azimuth_rad", azimuth_rad)

    freq_raw = _validate_frequency("frequency_hz", frequency_hz)
    is_series = freq_raw.ndim == 1
    k_len = freq_raw.shape[0] if is_series else None

    amp_raw = _validate_amplitude(
        "incident_displacement_m", incident_displacement_m, k_len=k_len
    )
    amplitude = amp_raw[:, None] if amp_raw.ndim == 1 else amp_raw

    _reject_non_real_dtype("phase_origin_m", phase_origin_m)
    origin = as_xyz_vector("phase_origin_m", phase_origin_m)

    k_p_flat = 2.0 * np.pi * freq_raw / v_p
    k_s_flat = 2.0 * np.pi * freq_raw / v_s
    k_p = k_p_flat[:, None] if is_series else k_p_flat  # (K,1) or scalar
    k_s = k_s_flat[:, None] if is_series else k_s_flat

    k_rho = k_p * np.sin(theta)
    k_zp = k_p * np.cos(theta)
    k_zs = np.sqrt(k_s**2 - k_rho**2)  # real and positive: v_s<v_p => k_s>k_p>=k_rho

    # phase_origin_m[2]: same z=0 re-referencing as the single-mode entry
    # point, using the incident P leg's own vertical wavenumber k_zp -- the
    # SAME converted amplitude is then used for the reflected/converted legs
    # too (not applied separately per leg), matching docs/design/literature-b.md.
    amplitude = amplitude * np.exp(-1j * k_zp * origin[2])

    # Keep the literal H19 Eq. (90) coefficient available through its private
    # helper, but use the coefficient obtained from the same displacement
    # field and its actual surface mass sheet for the physical observable.
    delta = _physical_reflected_p_delta(k_rho, k_zp, k_zs, k_s)

    qx, qy, phase = _horizontal_phase(targets, origin, azimuth, k_rho)

    c3 = -rho * amplitude * delta / (1j * k_p)  # Eq. (90) prefactor, G-free
    f_ground = 2.0 * np.pi * c3 * np.exp(-k_rho * z)
    phi_total = phase * f_ground
    s1 = phase * (-k_rho * f_ground)
    s2 = phase * (k_rho**2 * f_ground)

    return _assemble_observables(phase, phi_total, s1, s2, qx, qy)
