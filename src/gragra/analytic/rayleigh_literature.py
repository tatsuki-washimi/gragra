"""Literature-sourced Rayleigh half-space gravity: ground and underground.

Primary source: J. Harms, "Terrestrial gravity fluctuations", Living Reviews
in Relativity 22, 6 (2019), DOI 10.1007/s41114-019-0022-2 (2019 edition; do
not mix Eq. numbers with the 2015 edition). Design mapping, evidence
classification (A/B/C), and independent verification are recorded in full in
``docs/design/literature-r.md`` and ``docs/validation/literature-r.md``; this
docstring only summarizes the equation-to-code mapping.

- Eq. (96) (ground, z > 0): combined surface + bulk potential, evidence A.
- Eq. (97) (underground, z < 0): potential including the compressional bulk
  contribution, evidence A.
- Eq. (98): amplitude A expressed via the (possibly complex) vertical
  surface displacement amplitude, evidence A.
- Acceleration and gradient below are this module's own analytic derivative
  of Eq. (96)/(97) (evidence B); the paper's own Eq. (99) is reproduced only
  as the ground-side special case of this general derivative, not
  transcribed directly.

Independent cross-check performed while implementing this module: the
paper's Eq. (92) (surface term alone) is printed without the ``1/k_rho``
factor that a 2D Fourier transform of the ``1/r`` kernel carries. Combining
Eq. (94)/(95) (bulk term, which does carry that factor) with a hypothetical
Eq. (92) that also carries it reproduces Eq. (96) exactly by hand algebra;
the literally printed Eq. (92) does not. Eq. (96)-(98) -- what this module
implements -- are the already-combined result and are unaffected by that
printed-equation issue; this note only records the cross-check performed,
not a claim that Eq. (92) itself has been corrected.

``zeta(k_rho)`` (Eq. 96-99) is expressed here through the existing
``(q, s) = (sqrt(1-(c_R/v_P)^2), sqrt(1-(c_R/v_S)^2))`` pair in
``gragra.waves.rayleigh`` via ``zeta = sqrt(q / s)``: since the paper's
``q_z^P = k_rho * q`` and ``q_z^S = k_rho * s``, ``zeta = sqrt(q_z^P/q_z^S)``
reduces to ``sqrt(q/s)``, a pure material ratio independent of frequency.
Because ``0 < q, s < 1`` strictly whenever the Rayleigh secular equation has
a root (``0 < c_R < v_S < v_P``), ``q - zeta`` (the amplitude denominator's
frequency-independent factor) cannot vanish, so no additional near-zero
guard is required beyond the existing speed validation.

This is a private-helper reuse of the same contract already used by
``gragra.analytic.halfspace`` (see that module's docstring): this module's
tests and ``gragra.waves.rayleigh``'s tests must be updated together if
that helper's signature changes.

This module returns the ground-specific ``rayleigh_halfspace_observables_g_free``
in ``gragra.analytic.wave456_reference`` unchanged; that function stays the
legacy ground-only entry point (legacy ``GFreeObservables`` return type
instead of ``GFreeGravityValues``). For the same
``vertical_surface_amplitude_m``, material, frequency, and direction, the
legacy function's ground-side (``z > 0``) potential and acceleration are
numerically identical to this module's ground branch (checked to
floating-point precision in
``tests/test_analytic_rayleigh_literature.py::test_matches_legacy_ground_api``)
-- **the physical amplitude normalization is the same**; the two functions
differ only in return type, in the legacy function's amplitude domain (its
own validator requires a strictly positive *real* amplitude, unlike this
module which accepts complex and zero), and in this module's added
underground (``z < 0``) branch. This module is a new, independent entry
point covering both ground and underground with ``GFreeGravityValues``.

G-free (no ``G_SI`` import or multiplication) -- see
``gragra.observables.rayleigh_literature`` for the SI entry point.
Complex outputs represent linear complex amplitudes / transfer-function-like
responses, not directly real-valued time-domain gravitational fields.
Float64 coordinates / complex128 amplitudes throughout (array-conventions).
"""

from __future__ import annotations

import numpy as np

from gragra._arrays import _check_no_bool, as_xyz_table, as_xyz_vector
from gragra.analytic.values import GFreeGravityValues

# Private-helper contract: this module is (together with gragra.analytic.halfspace)
# an external consumer of these gragra.waves.rayleigh internals. Any signature
# change there must update this import site and the tests of both modules
# together -- see test_wave_rayleigh.py, test_analytic_halfspace.py, and
# test_analytic_rayleigh_literature.py.
from gragra.waves.rayleigh import _rayleigh_parameters, _validate_speeds

ArrayLike = list | tuple | np.ndarray


def _validate_non_negative_scalar(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a non-negative finite number")
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a non-negative finite number")
    return value


def _reject_non_real_dtype(name: str, value) -> None:
    """Reject bool/object/complex inputs before any float64 conversion.

    ``np.asarray(value, dtype=np.float64)`` silently discards the imaginary
    part of complex input (emitting only a ``ComplexWarning``, not an error)
    and silently accepts ``bool`` as 0.0/1.0 -- this guards real-only
    parameters (``targets_m``, ``direction``, ``phase_origin_m``,
    ``frequency_hz``) against both before that conversion runs.

    A plain ``bool`` mixed with numbers inside a Python list/tuple (e.g.
    ``[1.0, True, 2.0]``) does not surface as a ``bool`` dtype once
    ``np.asarray`` runs: numpy's type promotion silently upcasts it to
    ``float64`` first, so a dtype check alone (below) cannot see it.
    ``gragra._arrays._check_no_bool`` walks the raw, not-yet-arrayified
    Python structure instead, which is the shared helper already used for
    exactly this purpose elsewhere in the package (e.g. ``as_index_table``).
    It does not, and is not asked to, detect a ``bool`` that already went
    through an explicit user-side conversion to a float array before
    reaching here -- by that point the information is genuinely gone, not
    silently dropped by this function.
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


def _normalize_horizontal_direction(name: str, value) -> tuple[float, float]:
    """Scale-safely normalize a horizontal direction vector to (n1, n2).

    Dividing by ``np.linalg.norm`` directly overflows to ``inf`` for
    components like ``1e200`` (``(1e200)**2`` already exceeds float64 range)
    and underflows to exactly ``0.0`` for components like ``1e-200``
    (``(1e-200)**2`` underflows), silently corrupting or spuriously
    rejecting the direction. Dividing by the largest-magnitude component
    first keeps every subsequent operation of order 1, independent of the
    input's overall scale.
    """
    _reject_non_real_dtype(name, value)
    direction_arr = as_xyz_vector(name, value)
    scale = float(np.max(np.abs(direction_arr)))
    if scale == 0.0:
        raise ValueError(f"{name} must be non-zero")
    scaled = direction_arr / scale
    scaled_norm = float(np.linalg.norm(scaled))
    if abs(scaled[2]) > 1e-12 * scaled_norm:
        raise ValueError(
            f"{name} must be horizontal (z component must vanish) -- "
            "Rayleigh waves propagate along the free surface"
        )
    return float(scaled[0] / scaled_norm), float(scaled[1] / scaled_norm)


def _validate_amplitude(name: str, value, *, k_len: int | None) -> np.ndarray:
    """Validate a scalar or (K,)-shaped complex amplitude (Eq. 98's xi_z(0,0)).

    ``k_len`` is the length of a 1D ``frequency_hz`` (``None`` for a scalar
    frequency). A 1D amplitude is only accepted when ``k_len`` is not
    ``None`` and its length matches -- this is the (K,) amplitude case from
    the original common contract (scalar or (K,) amplitude, no other
    implicit broadcast), not an added convenience.

    ``_check_no_bool`` catches a ``bool`` mixed with numbers inside a
    Python list (e.g. ``[1e-9, True, 2e-9]``) before ``np.asarray``'s type
    promotion silently upcasts it to ``complex128`` -- see
    ``_reject_non_real_dtype``'s docstring for why a post-array dtype check
    alone cannot see this. It does not reject complex values themselves
    (those are intentionally allowed here).
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
    arr = np.asarray(raw, dtype=np.complex128)  # complex is intentionally allowed here
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


def rayleigh_halfspace_allspace_observables_g_free(
    targets_m: ArrayLike,
    *,
    density_kg_m3: float,
    v_p_m_s: float,
    v_s_m_s: float,
    frequency_hz: float | ArrayLike,
    vertical_surface_amplitude_m: float | complex | ArrayLike,
    direction: ArrayLike = (1.0, 0.0, 0.0),
    phase_origin_m: ArrayLike = (0.0, 0.0, 0.0),
) -> GFreeGravityValues:
    """G-free Rayleigh-wave gravity above and below a homogeneous half-space.

    Implements H19 Eq. (96) (``z > 0``) and Eq. (97) (``z < 0``), with the
    amplitude ``A`` from Eq. (98) expressed via the fixed-phase-origin,
    ``t = 0`` complex vertical surface displacement amplitude
    ``vertical_surface_amplitude_m``. Acceleration and gradient are this
    module's own analytic derivative of those closed forms (evidence B; see
    the module docstring and ``docs/design/literature-r.md``).

    Parameters
    ----------
    targets_m : array-like of shape (M, 3)
        Evaluation points in meters (z-up; free surface at z = 0; medium
        occupies z < 0). Points with ``z == 0`` are rejected: the potential
        and the tangential acceleration components are continuous across
        the free surface, but the normal acceleration and the gradient are
        not (see "Surface behavior, precisely stated" in
        ``docs/design/literature-r.md``), so the ground and underground
        closed forms are only defined on their respective open
        half-spaces.
    density_kg_m3 : float
        Background mass density in kg/m^3 (non-negative; a zero density is
        accepted and does not skip any other geometry validation).
    v_p_m_s, v_s_m_s : float
        P- and S-wave speeds in m/s. Requires ``0 < v_s_m_s < v_p_m_s`` and
        ``v_p_m_s**2 > (4/3) * v_s_m_s**2`` (stable isotropic half-space,
        Poisson ratio < 0.5) -- the condition under which the Rayleigh
        secular equation has a root with ``0 < q, s < 1``.
    frequency_hz : float or array-like of shape (K,)
        Positive frequency/frequencies in Hz. A scalar gives
        ``potential: (M,)``, ``acceleration: (M, 3)``, ``gradient: (M, 3, 3)``;
        a 1D array of length K gives ``potential: (K, M)``,
        ``acceleration: (K, M, 3)``, ``gradient: (K, M, 3, 3)``.
    vertical_surface_amplitude_m : float, complex, or array-like of shape (K,)
        Fixed-phase-origin, ``t = 0`` complex vertical surface displacement
        amplitude in meters (Eq. 98's ``xi_z(0, 0)``). Zero is accepted. A
        1D array of length K (one amplitude per frequency) is only accepted
        together with a 1D ``frequency_hz`` of the same length K; no other
        implicit broadcast between the two is allowed.
    direction : array-like of shape (3,), optional
        Horizontal Rayleigh propagation direction (z component must vanish).
        Normalized scale-safely (dividing by the largest-magnitude component
        first), so ``(1e-200, 0, 0)`` and ``(1e200, 0, 0)`` both normalize to
        the same direction as ``(1, 0, 0)`` rather than underflowing to
        "zero" or overflowing the norm to ``inf``. Default ``(1, 0, 0)``.
    phase_origin_m : array-like of shape (3,), optional
        Fixed horizontal phase-reference point in meters (not moved per
        target). Default the coordinate origin.

    Returns
    -------
    GFreeGravityValues
        G-free potential (kg/m), acceleration (kg/m^2), and gradient
        (kg/m^3), complex128, read-only.
    """
    _reject_non_real_dtype("targets_m", targets_m)
    targets = as_xyz_table("targets_m", targets_m)
    z = targets[:, 2]
    if np.any(z == 0.0):
        offending = int(np.nonzero(z == 0.0)[0][0])
        raise ValueError(
            f"targets_m must not lie exactly on the free surface z == 0 "
            f"(first offending index: {offending}) -- the ground and "
            "underground closed forms are only defined on their respective "
            "open half-spaces (a_z and the gradient are discontinuous there, "
            "so no single value exists for those observables)"
        )

    rho = _validate_non_negative_scalar("density_kg_m3", density_kg_m3)
    v_p, v_s = _validate_speeds(v_p_m_s, v_s_m_s)
    if not v_p * v_p > (4.0 / 3.0) * v_s * v_s:
        raise ValueError(
            "requires v_p_m_s**2 > (4/3) * v_s_m_s**2 for a stable "
            "isotropic half-space (Poisson ratio < 0.5)"
        )

    _reject_non_real_dtype("frequency_hz", frequency_hz)
    freq_raw = np.asarray(frequency_hz, dtype=np.float64)
    if freq_raw.ndim not in (0, 1):
        raise ValueError("frequency_hz must be a scalar or a 1D array")
    if not np.all(np.isfinite(freq_raw)) or np.any(freq_raw <= 0.0):
        raise ValueError("frequency_hz must contain only positive finite values")
    is_series = freq_raw.ndim == 1
    k_len = freq_raw.shape[0] if is_series else None

    amp_raw = _validate_amplitude(
        "vertical_surface_amplitude_m", vertical_surface_amplitude_m, k_len=k_len
    )
    # (K,1) for a per-frequency (K,) amplitude, else a 0-d scalar; both
    # broadcast correctly against the (K, M)/(M,) arrays built below.
    amplitude = amp_raw[:, None] if amp_raw.ndim == 1 else amp_raw

    n1, n2 = _normalize_horizontal_direction("direction", direction)

    _reject_non_real_dtype("phase_origin_m", phase_origin_m)
    origin = as_xyz_vector("phase_origin_m", phase_origin_m)

    c_r, q, s = _rayleigh_parameters(v_p, v_s)
    zeta = float(np.sqrt(q / s))

    k_r_flat = 2.0 * np.pi * freq_raw / c_r  # scalar or (K,)
    k_r = k_r_flat[:, None] if is_series else k_r_flat  # (K,1) or scalar
    q_p = k_r * q

    proj = (targets[:, 0] - origin[0]) * n1 + (targets[:, 1] - origin[1]) * n2  # (M,)
    phase = np.exp(1j * k_r * proj)  # (K, M) or (M,)

    a_const = amplitude / (q_p - k_r * zeta)  # Eq. (98); scalar or (K, 1)

    above = z > 0.0
    below = z < 0.0

    phi_total = np.empty(phase.shape, dtype=np.complex128)
    s1 = np.empty(phase.shape, dtype=np.complex128)
    s2 = np.empty(phase.shape, dtype=np.complex128)

    z_above = z[above]
    phase_above = phase[..., above]
    phi_ground = (
        -2.0
        * np.pi
        * rho
        * a_const
        * (1.0 - zeta)
        * np.exp(-k_r * z_above)
        * phase_above
    )  # Eq. (96)
    phi_total[..., above] = phi_ground
    s1[..., above] = -k_r * phi_ground
    s2[..., above] = k_r**2 * phi_ground

    z_below = z[below]
    phase_below = phase[..., below]
    phi1 = (
        -4.0 * np.pi * rho * a_const * np.exp(q_p * z_below) * phase_below
    )  # Eq. (97), first (compressional bulk) term
    phi2 = (
        2.0 * np.pi * rho * a_const * (1.0 + zeta) * np.exp(k_r * z_below) * phase_below
    )  # Eq. (97), second (surface + shear-coupled) term
    phi_total[..., below] = phi1 + phi2
    s1[..., below] = q_p * phi1 + k_r * phi2
    s2[..., below] = q_p**2 * phi1 + k_r**2 * phi2

    acceleration = np.empty(phase.shape + (3,), dtype=np.complex128)
    acceleration[..., 0] = -1j * k_r * n1 * phi_total
    acceleration[..., 1] = -1j * k_r * n2 * phi_total
    acceleration[..., 2] = -s1

    gradient = np.empty(phase.shape + (3, 3), dtype=np.complex128)
    gradient[..., 0, 0] = k_r**2 * n1 * n1 * phi_total
    gradient[..., 0, 1] = k_r**2 * n1 * n2 * phi_total
    gradient[..., 1, 0] = gradient[..., 0, 1]
    gradient[..., 1, 1] = k_r**2 * n2 * n2 * phi_total
    gradient[..., 0, 2] = -1j * k_r * n1 * s1
    gradient[..., 2, 0] = gradient[..., 0, 2]
    gradient[..., 1, 2] = -1j * k_r * n2 * s1
    gradient[..., 2, 1] = gradient[..., 1, 2]
    gradient[..., 2, 2] = -s2

    return GFreeGravityValues(
        potential=phi_total, acceleration=acceleration, gradient=gradient
    )
