"""Analytic Newtonian noise models for water flow (R2 Water NN).

Conventions and Formulation Notes
---------------------------------
1. G-free Convention:
   All functions in this module are strictly G-free. Values returned have
   Newton's gravitational constant G divided out:
   - Potential transfer: [Phi / (G * eta)] = kg / m^2
   - Acceleration transfer: [a / (G * eta)] = kg / m^3
   - Gradient transfer: [Gamma / (G * eta)] = kg / m^4
   - Edge step response: [Delta a_x(Omega) / G] has unit kg * s / m^2
     (single-event deterministic acceleration Fourier transform).
     Dimensional derivation:
     [Delta a_x(Omega)] = (m / s^2) * s = m / s.
     Divided by G [m^3 / (kg * s^2)], the G-free spectrum has unit:
     (m / s) / (m^3 / (kg * s^2)) = kg * s / m^2.
   - Uncorrelated cell coupling: [C_a / G] has unit kg / m^3.
     Dimension breakdown: rho [kg/m^3] * sqrt(A_cell [m^2] * w [m]) [m^1.5]
     * sqrt(pi / (8 * H^3)) [m^-1.5] = kg / m^3.
     When multiplied by G [m^3 / (kg * s^2)], the transfer ratio has unit s^-2.
   Multiplying by G_SI (6.6743e-11 m^3 / (kg * s^2)) is the responsibility of
   the caller or the high-level observables in `gragra.observables.water`.

2. Coordinate Geometry:
   - Water Channel (Model 2):
     The channel runs along the x-axis with width w in the y-direction.
     The mean water surface lies in the z = 0 plane.
     The test mass is located at (0, 0, D), i.e., perpendicular offset D > 0
     along the z-axis.
     The flow moves with velocity v in the +x direction.
     Frozen advection: eta(x, t) = eta_0 * exp(-i * Omega * (t - x / v)).
   - Waterfall Model:
     The waterfall flows vertically along the z-axis with velocity v.
     The test mass is located at (D, 0, 0) horizontally.
     Cross-sectional area perturbation: A(z, t) = A_0 * exp(-i * Omega * (t - z / v)).

3. Provenance and Analytical Closed Forms:
   - Model 2 (Frozen advection channel):
     Analytical transfer function from surface displacement eta(0, t) to
     gravitational acceleration a_x formulated via modified Bessel functions
     (Gradshteyn & Ryzhik 3.754.2 / DLMF 10.32.9):
       T_{a_x eta}(Omega) = 2i * G * rho * w * (Omega / v) * K_0(|Omega| * D / |v|)
     Full 3-observable multipole structure derived analytically in gragra:
       T_{Phi eta} = -2 * G * rho * w * K_0(q)
       T_{a_x eta} = 2i * G * rho * w * (Omega / v) * K_0(q)
       T_{a_z eta} = -2 * G * rho * w * (|Omega| / |v|) * K_1(q)
       T_{Gamma_xx eta} = -2 * G * rho * w * (Omega / v)^2 * K_0(q)
       T_{Gamma_yy eta} = -2 * G * rho * w * (q / D^2) * K_1(q)
       T_{Gamma_zz eta} = 2 * G * rho * w * [
           (Omega / v)^2 * K_0(q) + (q / D^2) * K_1(q)
       ]
       T_{Gamma_xz eta} = -2i * G * rho * w * (Omega * |Omega| / (v * |v|)) * K_1(q)
     Trace(Gamma) = Gamma_xx + Gamma_yy + Gamma_zz = 0 (Laplace vacuum condition).
   - Waterfall Model:
     Formulated for vertical cylindrical cross-sectional area perturbations:
       T_{a_x A} = -2 * G * rho * (|Omega| / |v|) * K_1(|Omega| * D / |v|)
       T_{a_z A} = 2i * G * rho * (Omega / v) * K_0(|Omega| * D / |v|)
   - Model 1 (Uncorrelated surface fluctuations):
     The gragra narrow-channel uncorrelated cell model is an INDEPENDENT_DERIVATION:
       int_{-inf}^inf x^2 / (H^2 + x^2)^3 dx = pi / (8 * H^3)
     Primary source notes from external conference presentations remain
     unverified (ORIGINAL_SOURCE_UNVERIFIED).
   - Edge effect:
     Represents the deterministic Fourier spectrum of a single transient event
     at channel inflow boundary.

4. Float64 Numerical Domain:
   Positive-frequency Bessel evaluations require a representable argument
   0 < q <= 700. A positive frequency whose q underflows, q above this bound,
   an unrepresentable wavenumber or coefficient, or a non-finite result raises
   ValueError.
   Exact DC is identified from f == 0, independently of the computed q.
"""

from __future__ import annotations

import numpy as np

from gragra.analytic.values import GFreeGravityValues

# Euler-Mascheroni constant for high-precision small-q asymptotics of K0 and K1
EULER_GAMMA: float = 0.5772156649015328606065120900824
MAX_SUPPORTED_BESSEL_ARGUMENT: float = 700.0
_LOG_TWO_PI: float = float(np.log(2.0 * np.pi))


def _frequency_wavenumber_and_q(
    frequency_hz: np.ndarray, distance_m: float, velocity_mps: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return signed k and q without overflowing intermediate products.

    Positive-frequency entries are evaluated from logarithms so that forming
    ``2*pi*f*D/v`` cannot silently overflow or underflow before classification.
    The public numerical domain is 0 < q <= 700; exact f=0 remains q=0.
    """
    k = np.zeros_like(frequency_hz, dtype=np.float64)
    q = np.zeros_like(frequency_hz, dtype=np.float64)
    positive = frequency_hz > 0.0
    if not np.any(positive):
        return k, q

    f_pos = frequency_hz[positive]
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        log_abs_k = _LOG_TWO_PI + np.log(f_pos) - np.log(abs(velocity_mps))
        log_q = log_abs_k + np.log(distance_m)
        abs_k = np.exp(log_abs_k)
        q_pos = np.exp(log_q)

    if np.any(abs_k == 0.0):
        raise ValueError(
            "positive frequency is outside the float64 wavenumber range; "
            "angular wavenumber underflowed"
        )
    if not np.all(np.isfinite(abs_k)):
        raise ValueError(
            "positive frequency is outside the float64 wavenumber range; "
            "angular wavenumber overflowed"
        )
    if np.any(q_pos == 0.0):
        raise ValueError(
            "positive frequency and geometry produce a dimensionless Bessel "
            "argument that underflows float64"
        )
    if not np.all(np.isfinite(q_pos)):
        raise ValueError(
            "dimensionless Bessel argument is outside the supported range "
            f"0 < q <= {MAX_SUPPORTED_BESSEL_ARGUMENT:g}"
        )
    if np.any(q_pos > MAX_SUPPORTED_BESSEL_ARGUMENT):
        raise ValueError(
            "dimensionless Bessel argument is outside the supported range "
            f"0 < q <= {MAX_SUPPORTED_BESSEL_ARGUMENT:g}"
        )

    k[positive] = np.copysign(abs_k, velocity_mps)
    q[positive] = q_pos
    return k, q


def _require_finite_result(name: str, values: np.ndarray) -> None:
    """Reject finite-input combinations whose float64 outputs are not finite."""
    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"{name} is outside the finite float64 result range for these inputs"
        )


def _validate_positive_scalar(name: str, value: float) -> float:
    """Validate that a parameter is a strictly positive real scalar."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must not be boolean")
    try:
        val = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number") from error
    if not np.isfinite(val) or val <= 0.0:
        raise ValueError(f"{name} must be strictly positive (> 0); got {val}")
    return val


def _validate_finite_scalar(name: str, value: float) -> float:
    """Validate that a parameter is a finite real scalar."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must not be boolean")
    try:
        val = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number") from error
    if not np.isfinite(val):
        raise ValueError(f"{name} must be finite; got {val}")
    return val


def _validate_non_zero_scalar(name: str, value: float) -> float:
    """Validate that a parameter is a non-zero real scalar."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must not be boolean")
    try:
        val = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number") from error
    if not np.isfinite(val) or val == 0.0:
        raise ValueError(f"{name} must be non-zero; got {val}")
    return val


def _validate_frequencies(
    frequency_hz: float | list | tuple | np.ndarray,
    allow_zero: bool = True,
) -> tuple[np.ndarray, bool]:
    """Validate frequency array, ensuring non-negative values.

    Returns (arr, is_scalar).
    """
    if isinstance(frequency_hz, (bool, np.bool_)):
        raise TypeError("frequency_hz must not be boolean")
    is_scalar = np.ndim(frequency_hz) == 0
    arr = np.asarray(frequency_hz, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("frequency_hz must contain only finite values")
    if allow_zero:
        if np.any(arr < 0.0):
            raise ValueError("frequency_hz must be non-negative (>= 0)")
    else:
        if np.any(arr <= 0.0):
            raise ValueError("frequency_hz must be strictly positive (> 0)")
    return arr, is_scalar


def water_channel_advection_transfer_g_free(
    frequency_hz: float | list | tuple | np.ndarray,
    velocity_mps: float,
    distance_m: float,
    width_m: float,
    density_kg_m3: float = 1000.0,
    allow_dc: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute G-free transfer per unit displacement for advected water channel.

    Model 2 (Nishizawa-Chen frozen advection).

    Parameters
    ----------
    frequency_hz : float or array_like
        Frequency f in Hz (f > 0 by default; f = 0 allowed only when allow_dc=True).
    velocity_mps : float
        Flow velocity v in m/s (non-zero).
    distance_m : float
        Perpendicular distance D from channel to mirror in meters (D > 0).
    width_m : float
        Channel width w in meters (w > 0).
    density_kg_m3 : float, default=1000.0
        Fluid density perturbation rho in kg/m^3 (rho > 0).
    allow_dc : bool, default=False
        Whether to allow f = 0 (DC). If True, potential at f = 0 is set to -inf
        due to the logarithmic divergence of an infinite line mass. If False and
        f = 0 is passed, raises ValueError. For finite DC acceleration and
        gravity gradient without potential, use
        `water_channel_advection_dc_limit_g_free`.

    Numerical domain
    ----------------
    Positive-frequency samples require a representable dimensionless argument
    ``0 < q <= 700`` with representable float64 wavenumbers and coefficients.
    Positive-frequency q underflow, q above 700, unrepresentable intermediate
    coefficients, or non-finite transfer values raise ``ValueError``.

    Returns
    -------
    potential_transfer : np.ndarray
        G-free potential transfer [Phi / (G * eta)], shape (F,) or (),
        complex128/float64.
    acceleration_transfer : np.ndarray
        G-free acceleration transfer [a / (G * eta)], shape (F, 3) or (3,),
        complex128. Components are (a_x, a_y, a_z).
    gradient_transfer : np.ndarray
        G-free gravity gradient transfer [Gamma / (G * eta)], shape (F, 3, 3)
        or (3, 3), complex128. Trace is identically zero for all frequencies.
    """
    v = _validate_non_zero_scalar("velocity_mps", velocity_mps)
    d = _validate_positive_scalar("distance_m", distance_m)
    w = _validate_positive_scalar("width_m", width_m)
    rho = _validate_positive_scalar("density_kg_m3", density_kg_m3)

    f_arr, is_scalar = _validate_frequencies(frequency_hz, allow_zero=True)
    if not allow_dc and np.any(f_arr == 0.0):
        raise ValueError(
            "frequency_hz must be strictly positive (> 0) because the gravitational "
            "potential of an infinite channel diverges logarithmically at DC (f=0). "
            "To evaluate transfers with DC potential=-inf, pass allow_dc=True; "
            "to evaluate finite DC acceleration and gradient, use "
            "water_channel_advection_dc_limit_g_free."
        )

    orig_shape = f_arr.shape
    f_1d = f_arr.reshape(-1)
    k_all, q = _frequency_wavenumber_and_q(f_1d, d, v)

    # Initialize output arrays (1D flattened)
    pot_trans_1d = np.empty(len(f_1d), dtype=np.complex128)
    acc_trans_1d = np.zeros((len(f_1d), 3), dtype=np.complex128)
    grad_trans_1d = np.zeros((len(f_1d), 3, 3), dtype=np.complex128)

    # Determine DC from the actual input frequency, never from rounded q.
    zero_mask = f_1d == 0.0
    pos_mask = f_1d > 0.0

    # Factor 2 * rho * w
    with np.errstate(over="ignore", invalid="ignore"):
        c0 = 2.0 * rho * w
    if not np.isfinite(c0):
        raise ValueError("density_kg_m3 * width_m exceeds the finite float64 range")

    if np.any(zero_mask):
        # Potential at DC: infinite line mass potential diverges logarithmically
        pot_trans_1d[zero_mask] = -np.inf
        with np.errstate(
            over="ignore", under="ignore", divide="ignore", invalid="ignore"
        ):
            # a_x at DC is exactly 0
            acc_trans_1d[zero_mask, 0] = 0.0
            # a_z at DC: lim_{q->0} -c0 * (q/D) * K_1(q) = -c0 / D
            acc_trans_1d[zero_mask, 2] = -c0 / d
            # Gamma_xx = 0, Gamma_yy = -c0 / D^2, Gamma_zz = +c0 / D^2
            grad_trans_1d[zero_mask, 1, 1] = -c0 / (d * d)
            grad_trans_1d[zero_mask, 2, 2] = c0 / (d * d)

    if np.any(pos_mask):
        from scipy.special import k0, k1

        q_pos = q[pos_mask]
        k_pos = k_all[pos_mask]
        abs_k_pos = np.abs(k_pos)

        # Distinguish standard vs tiny argument for numerical precision
        tiny_mask = q_pos < 1e-15
        normal_mask = ~tiny_mask

        pot_pos = np.empty_like(q_pos, dtype=np.complex128)
        acc_pos = np.zeros((len(q_pos), 3), dtype=np.complex128)
        grad_pos = np.zeros((len(q_pos), 3, 3), dtype=np.complex128)

        if np.any(normal_mask):
            q_n = q_pos[normal_mask]
            k_n = k_pos[normal_mask]
            abs_k_n = abs_k_pos[normal_mask]

            with np.errstate(
                over="ignore", under="ignore", divide="ignore", invalid="ignore"
            ):
                k0_n = k0(q_n)
                k1_n = k1(q_n)

                # Potential: -2 * rho * w * K0(q)
                pot_pos[normal_mask] = -c0 * k0_n
                # a_x = 2i * rho * w * (omega / v) * K0(q)
                acc_pos[normal_mask, 0] = 1j * c0 * k_n * k0_n
                acc_pos[normal_mask, 1] = 0.0
                # a_z = -2 * rho * w * (|omega| / |v|) * K1(q)
                acc_pos[normal_mask, 2] = -c0 * abs_k_n * k1_n

                # Gravity Gradient components
                grad_pos[normal_mask, 0, 0] = -c0 * (k_n**2) * k0_n
                grad_pos[normal_mask, 1, 1] = -c0 * (abs_k_n / d) * k1_n
                grad_pos[normal_mask, 2, 2] = c0 * (
                    (k_n**2) * k0_n + (abs_k_n / d) * k1_n
                )
                gxz_n = -1j * c0 * (k_n * abs_k_n) * k1_n
                grad_pos[normal_mask, 0, 2] = gxz_n
                grad_pos[normal_mask, 2, 0] = gxz_n

        if np.any(tiny_mask):
            q_t = q_pos[tiny_mask]

            # High-precision small-q expansions:
            # K_0(q) = -ln(q/2) - gamma + (q^2 / 4) * (-ln(q/2) - gamma + 1) + ...
            # q * K_1(q) = 1 + (q^2 / 2) * (ln(q/2) + gamma - 0.5) + ...
            with np.errstate(
                over="ignore", under="ignore", divide="ignore", invalid="ignore"
            ):
                # Equivalent to -log(q/2), without underflowing q/2 to zero.
                log_term = -np.log(q_t) + np.log(2.0) - EULER_GAMMA
                k0_t = log_term + 0.25 * (q_t**2) * (log_term + 1.0)
                qk1_t = 1.0 + 0.5 * (q_t**2) * (-log_term - 0.5)
                k_t = k_pos[tiny_mask]

                pot_pos[tiny_mask] = -c0 * k0_t
                acc_pos[tiny_mask, 0] = 1j * c0 * k_t * k0_t
                acc_pos[tiny_mask, 1] = 0.0
                acc_pos[tiny_mask, 2] = -c0 * (qk1_t / d)

                grad_pos[tiny_mask, 0, 0] = -c0 * (k_t**2) * k0_t
                grad_pos[tiny_mask, 1, 1] = -c0 * (qk1_t / (d * d))
                grad_pos[tiny_mask, 2, 2] = c0 * ((k_t**2) * k0_t + (qk1_t / (d * d)))
                gxz_t = -1j * c0 * (k_t * qk1_t / d)
                grad_pos[tiny_mask, 0, 2] = gxz_t
                grad_pos[tiny_mask, 2, 0] = gxz_t

        pot_trans_1d[pos_mask] = pot_pos
        acc_trans_1d[pos_mask] = acc_pos
        grad_trans_1d[pos_mask] = grad_pos

    _require_finite_result("acceleration transfer", acc_trans_1d)
    _require_finite_result("gravity-gradient transfer", grad_trans_1d)
    _require_finite_result(
        "positive-frequency potential transfer", pot_trans_1d[pos_mask]
    )

    pot_trans = pot_trans_1d.reshape(orig_shape)
    acc_trans = acc_trans_1d.reshape((*orig_shape, 3))
    grad_trans = grad_trans_1d.reshape((*orig_shape, 3, 3))

    pot_trans.setflags(write=False)
    acc_trans.setflags(write=False)
    grad_trans.setflags(write=False)

    if is_scalar:
        return pot_trans.squeeze(), acc_trans.squeeze(), grad_trans.squeeze()
    return pot_trans, acc_trans, grad_trans


def water_channel_advection_dc_limit_g_free(
    distance_m: float,
    width_m: float,
    density_kg_m3: float = 1000.0,
    eta_m: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute finite G-free DC limit (f -> 0) of acceleration and gradient.

    For an infinite narrow water channel, the gravitational potential diverges
    logarithmically as -inf at DC. However, the acceleration and gravity
    gradient are strictly finite:
    - a_x = 0
    - a_y = 0
    - a_z = -2 * rho * w * eta / D (exact Newtonian line mass attraction)
    - Gamma_xx = 0
    - Gamma_yy = -2 * rho * w * eta / D^2
    - Gamma_zz = +2 * rho * w * eta / D^2
    - Gamma_xz = 0
    - Tr(Gamma) = 0

    Parameters
    ----------
    distance_m : float
        Perpendicular distance D from channel to mirror in meters (D > 0).
    width_m : float
        Channel width w in meters (w > 0).
    density_kg_m3 : float, default=1000.0
        Fluid density rho in kg/m^3 (rho > 0).
    eta_m : float, default=1.0
        DC water surface displacement in meters (finite).

    Returns
    -------
    acc_dc : np.ndarray, shape (3,)
        Acceleration in G-free units (kg / m^2).
    grad_dc : np.ndarray, shape (3, 3)
        Gravity gradient tensor in G-free units (kg / m^3).

    Raises
    ------
    ValueError
        If the finite inputs produce a non-finite float64 DC result.
    """
    d = _validate_positive_scalar("distance_m", distance_m)
    w = _validate_positive_scalar("width_m", width_m)
    rho = _validate_positive_scalar("density_kg_m3", density_kg_m3)
    eta = _validate_finite_scalar("eta_m", eta_m)

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        c0 = np.float64(2.0) * np.float64(rho) * np.float64(w) * np.float64(eta)
    if not np.isfinite(c0):
        raise ValueError("DC coupling exceeds the finite float64 coefficient range")
    if eta != 0.0 and c0 == 0.0:
        raise ValueError("DC coupling underflows the finite float64 result range")

    if eta == 0.0:
        acc_dc = np.zeros(3, dtype=np.float64)
        grad_dc = np.zeros((3, 3), dtype=np.float64)
    else:
        with np.errstate(
            over="ignore", under="ignore", divide="ignore", invalid="ignore"
        ):
            acc_dc = np.array([0.0, 0.0, np.divide(-c0, d)], dtype=np.float64)
            grad_dc = np.zeros((3, 3), dtype=np.float64)
            grad_diagonal = np.divide(c0, np.multiply(np.float64(d), np.float64(d)))
            grad_dc[1, 1] = -grad_diagonal
            grad_dc[2, 2] = grad_diagonal

    _require_finite_result("DC acceleration", acc_dc)
    _require_finite_result("DC gravity-gradient", grad_dc)

    acc_dc.setflags(write=False)
    grad_dc.setflags(write=False)
    return acc_dc, grad_dc


def water_channel_advection_ax_transfer_g_free(
    frequency_hz: float | list | tuple | np.ndarray,
    velocity_mps: float,
    distance_m: float,
    width_m: float,
    density_kg_m3: float = 1000.0,
) -> np.ndarray:
    """Compute along-flow acceleration transfer T_{a_x eta}(Omega) / G.

    Parameters
    ----------
    frequency_hz : float or array_like
        Frequency f in Hz (f >= 0).
    velocity_mps : float
        Flow velocity v in m/s (non-zero).
    distance_m : float
        Distance D in meters (D > 0).
    width_m : float
        Width w in meters (w > 0).
    density_kg_m3 : float, default=1000.0
        Density rho in kg/m^3 (rho > 0).

    Notes
    -----
    Positive-frequency samples require ``0 < q <= 700`` and a finite
    float64 wavenumbers, coefficients, and result. Argument underflow or values
    outside that range raise ``ValueError``. Exact DC returns zero for this
    along-flow component.

    Returns
    -------
    np.ndarray
        G-free acceleration transfer, shape matching frequency_hz.
    """
    v = _validate_non_zero_scalar("velocity_mps", velocity_mps)
    d = _validate_positive_scalar("distance_m", distance_m)
    w = _validate_positive_scalar("width_m", width_m)
    rho = _validate_positive_scalar("density_kg_m3", density_kg_m3)

    f_arr, is_scalar = _validate_frequencies(frequency_hz, allow_zero=True)
    orig_shape = f_arr.shape
    f_1d = f_arr.reshape(-1)
    k_all, q = _frequency_wavenumber_and_q(f_1d, d, v)

    with np.errstate(over="ignore", invalid="ignore"):
        c0 = 2.0 * rho * w
    if not np.isfinite(c0):
        raise ValueError("density_kg_m3 * width_m exceeds the finite float64 range")
    res_1d = np.zeros(len(f_1d), dtype=np.complex128)

    pos_mask = f_1d > 0.0
    if np.any(pos_mask):
        from scipy.special import k0

        k_pos = k_all[pos_mask]
        q_pos = q[pos_mask]

        tiny_mask = q_pos < 1e-15
        normal_mask = ~tiny_mask

        res_pos = np.empty_like(q_pos, dtype=np.complex128)
        if np.any(normal_mask):
            k_n = k_pos[normal_mask]
            q_n = q_pos[normal_mask]
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                res_pos[normal_mask] = 1j * c0 * k_n * k0(q_n)
        if np.any(tiny_mask):
            q_t = q_pos[tiny_mask]
            k_t = k_pos[tiny_mask]
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                log_term = -np.log(q_t) + np.log(2.0) - EULER_GAMMA
                k0_t = log_term + 0.25 * (q_t**2) * (log_term + 1.0)
                res_pos[tiny_mask] = 1j * c0 * k_t * k0_t
        res_1d[pos_mask] = res_pos

    _require_finite_result("along-flow acceleration transfer", res_1d)

    res = res_1d.reshape(orig_shape)
    res.setflags(write=False)
    if is_scalar:
        return res.squeeze()
    return res


def water_channel_advection_observables_g_free(
    frequency_hz: float | list | tuple | np.ndarray,
    velocity_mps: float,
    distance_m: float,
    width_m: float,
    density_kg_m3: float = 1000.0,
    eta_m: complex | float | np.ndarray = 1.0,
) -> GFreeGravityValues:
    """Compute G-free observables for advected channel with given amplitude.

    Returns GFreeGravityValues containing potential, acceleration, and gradient.

    Note: frequency_hz must be strictly positive (> 0) because GFreeGravityValues
    requires finite potential values, whereas the potential of an infinite line
    mass diverges logarithmically at DC (f = 0). For finite DC acceleration and
    gravity gradient, use `water_channel_advection_dc_limit_g_free`.
    """
    f_arr, _ = _validate_frequencies(frequency_hz, allow_zero=False)
    if np.any(f_arr <= 0.0):
        raise ValueError(
            "frequency_hz must be strictly positive (> 0) because GFreeGravityValues "
            "requires a finite potential, and the potential of an infinite channel "
            "diverges logarithmically at DC (f=0). For finite DC acceleration and "
            "gravity gradient, use water_channel_advection_dc_limit_g_free."
        )

    pot_t, acc_t, grad_t = water_channel_advection_transfer_g_free(
        frequency_hz=f_arr,
        velocity_mps=velocity_mps,
        distance_m=distance_m,
        width_m=width_m,
        density_kg_m3=density_kg_m3,
        allow_dc=False,
    )

    eta = np.asarray(eta_m)
    if not np.all(np.isfinite(eta)):
        raise ValueError("eta_m must contain only finite values")

    # Enforce shape contract: eta_m must be scalar or match frequency_hz shape
    if eta.ndim > 0 and eta.shape != f_arr.shape:
        raise ValueError(
            f"eta_m shape {eta.shape} must match frequency_hz shape {f_arr.shape}"
        )

    pot = pot_t * eta
    acc = acc_t * np.expand_dims(eta, axis=-1)
    grad = grad_t * np.expand_dims(eta, axis=(-2, -1))

    if pot.ndim == 0:
        pot = np.expand_dims(pot, axis=0)
        acc = np.expand_dims(acc, axis=0)
        grad = np.expand_dims(grad, axis=0)

    return GFreeGravityValues(potential=pot, acceleration=acc, gradient=grad)


def water_waterfall_transfer_g_free(
    frequency_hz: float | list | tuple | np.ndarray,
    velocity_mps: float,
    distance_m: float,
    density_kg_m3: float = 1000.0,
    allow_dc: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute G-free transfer per unit cross-sectional area for vertical waterfall.

    Nishizawa (2021) waterfall model.

    Parameters
    ----------
    frequency_hz : float or array_like
        Frequency f in Hz (f > 0 by default; f = 0 allowed only when allow_dc=True).
    velocity_mps : float
        Fall velocity v in m/s (non-zero).
    distance_m : float
        Horizontal distance D from waterfall to mirror in meters (D > 0).
    density_kg_m3 : float, default=1000.0
        Fluid density rho in kg/m^3 (rho > 0).
    allow_dc : bool, default=False
        Whether to allow f = 0.

    Numerical domain
    ----------------
    Positive-frequency samples require a representable dimensionless argument
    ``0 < q <= 700`` with representable float64 wavenumbers and coefficients.
    Positive-frequency q underflow, q above 700, unrepresentable intermediate
    coefficients, or non-finite transfer values raise ``ValueError``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (potential_transfer, acceleration_transfer, gradient_transfer)
    """
    v = _validate_non_zero_scalar("velocity_mps", velocity_mps)
    d = _validate_positive_scalar("distance_m", distance_m)
    rho = _validate_positive_scalar("density_kg_m3", density_kg_m3)

    f_arr, is_scalar = _validate_frequencies(frequency_hz, allow_zero=True)
    if not allow_dc and np.any(f_arr == 0.0):
        raise ValueError(
            "frequency_hz must be strictly positive (> 0) because the gravitational "
            "potential of an infinite vertical waterfall diverges logarithmically "
            "at DC."
        )

    orig_shape = f_arr.shape
    f_1d = f_arr.reshape(-1)
    k_all, q = _frequency_wavenumber_and_q(f_1d, d, v)

    pot_trans_1d = np.empty(len(f_1d), dtype=np.complex128)
    acc_trans_1d = np.zeros((len(f_1d), 3), dtype=np.complex128)
    grad_trans_1d = np.zeros((len(f_1d), 3, 3), dtype=np.complex128)

    zero_mask = f_1d == 0.0
    pos_mask = f_1d > 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        c0 = 2.0 * rho
    if not np.isfinite(c0):
        raise ValueError("density_kg_m3 exceeds the finite float64 coefficient range")

    if np.any(zero_mask):
        pot_trans_1d[zero_mask] = -np.inf
        with np.errstate(
            over="ignore", under="ignore", divide="ignore", invalid="ignore"
        ):
            # a_x at DC: lim_{q->0} -c0 * (q/D) * K_1(q) = -c0 / D
            acc_trans_1d[zero_mask, 0] = -c0 / d
            acc_trans_1d[zero_mask, 2] = 0.0
            grad_trans_1d[zero_mask, 0, 0] = c0 / (d * d)
            grad_trans_1d[zero_mask, 1, 1] = -c0 / (d * d)
            grad_trans_1d[zero_mask, 2, 2] = 0.0

    if np.any(pos_mask):
        from scipy.special import k0, k1

        q_pos = q[pos_mask]
        k_pos = k_all[pos_mask]
        abs_k_pos = np.abs(k_pos)

        tiny_mask = q_pos < 1e-15
        normal_mask = ~tiny_mask

        pot_pos = np.empty_like(q_pos, dtype=np.complex128)
        acc_pos = np.zeros((len(q_pos), 3), dtype=np.complex128)
        grad_pos = np.zeros((len(q_pos), 3, 3), dtype=np.complex128)

        if np.any(normal_mask):
            q_n = q_pos[normal_mask]
            k_n = k_pos[normal_mask]
            abs_k_n = abs_k_pos[normal_mask]

            with np.errstate(
                over="ignore", under="ignore", divide="ignore", invalid="ignore"
            ):
                k0_n = k0(q_n)
                k1_n = k1(q_n)

                pot_pos[normal_mask] = -c0 * k0_n
                # a_x: -2 * rho * (|omega| / |v|) * K1(q) (horizontal pull)
                # a_z: 2i * rho * (omega / v) * K0(q) (along-fall pull)
                acc_pos[normal_mask, 0] = -c0 * abs_k_n * k1_n
                acc_pos[normal_mask, 1] = 0.0
                acc_pos[normal_mask, 2] = 1j * c0 * k_n * k0_n

                # Gradient:
                # Gamma_zz = -2 * rho * (omega / v)^2 * K0(q)
                # Gamma_yy = -2 * rho * (abs_k / D) * K1(q)
                # Gamma_xx = 2 * rho * [ (omega / v)^2 * K0(q) + (abs_k / D) * K1(q) ]
                # Gamma_xz = -2i * rho * (k * |k|) * K1(q)
                grad_pos[normal_mask, 0, 0] = c0 * (
                    (k_n**2) * k0_n + (abs_k_n / d) * k1_n
                )
                grad_pos[normal_mask, 1, 1] = -c0 * (abs_k_n / d) * k1_n
                grad_pos[normal_mask, 2, 2] = -c0 * (k_n**2) * k0_n
                gxz_n = -1j * c0 * (k_n * abs_k_n) * k1_n
                grad_pos[normal_mask, 0, 2] = gxz_n
                grad_pos[normal_mask, 2, 0] = gxz_n

        if np.any(tiny_mask):
            q_t = q_pos[tiny_mask]
            k_t = k_pos[tiny_mask]
            with np.errstate(
                over="ignore", under="ignore", divide="ignore", invalid="ignore"
            ):
                log_term = -np.log(q_t) + np.log(2.0) - EULER_GAMMA
                k0_t = log_term + 0.25 * (q_t**2) * (log_term + 1.0)
                qk1_t = 1.0 + 0.5 * (q_t**2) * (-log_term - 0.5)

                pot_pos[tiny_mask] = -c0 * k0_t
                acc_pos[tiny_mask, 0] = -c0 * (qk1_t / d)
                acc_pos[tiny_mask, 1] = 0.0
                acc_pos[tiny_mask, 2] = 1j * c0 * k_t * k0_t

                grad_pos[tiny_mask, 0, 0] = c0 * ((k_t**2) * k0_t + (qk1_t / (d * d)))
                grad_pos[tiny_mask, 1, 1] = -c0 * (qk1_t / (d * d))
                grad_pos[tiny_mask, 2, 2] = -c0 * (k_t**2) * k0_t
                gxz_t = -1j * c0 * (k_t * qk1_t / d)
                grad_pos[tiny_mask, 0, 2] = gxz_t
                grad_pos[tiny_mask, 2, 0] = gxz_t

        pot_trans_1d[pos_mask] = pot_pos
        acc_trans_1d[pos_mask] = acc_pos
        grad_trans_1d[pos_mask] = grad_pos

    _require_finite_result("acceleration transfer", acc_trans_1d)
    _require_finite_result("gravity-gradient transfer", grad_trans_1d)
    _require_finite_result(
        "positive-frequency potential transfer", pot_trans_1d[pos_mask]
    )

    pot_trans = pot_trans_1d.reshape(orig_shape)
    acc_trans = acc_trans_1d.reshape((*orig_shape, 3))
    grad_trans = grad_trans_1d.reshape((*orig_shape, 3, 3))

    pot_trans.setflags(write=False)
    acc_trans.setflags(write=False)
    grad_trans.setflags(write=False)

    if is_scalar:
        return pot_trans.squeeze(), acc_trans.squeeze(), grad_trans.squeeze()
    return pot_trans, acc_trans, grad_trans


def water_channel_uncorrelated_cell_coupling_g_free(
    distance_m: float,
    width_m: float,
    correlation_area_m2: float,
    density_kg_m3: float = 1000.0,
) -> float:
    """Compute G-free spatial coupling factor for uncorrelated surface cells.

    Model 1 (narrow channel, Chen uncorrelated surface model).
    Integration of uncorrelated cells along x in [-inf, inf]:
      int_{-inf}^inf x^2 / (H^2 + x^2)^3 dx = pi / (8 * H^3)

    Spatial coupling factor:
      C_{a_x} / G = rho * sqrt(A_cell * w) * sqrt(pi / (8 * H^3))

    Dimensional derivation:
      [rho] = kg / m^3
      [sqrt(A_cell * w)] = sqrt(m^2 * m) = m^1.5
      [sqrt(pi / (8 * H^3))] = sqrt(m^-3) = m^-1.5
      [C_{a_x} / G] = (kg / m^3) * m^1.5 * m^-1.5 = kg / m^3.

    When multiplied by G [m^3 / (kg * s^2)], the transfer ratio has unit s^-2:
      [C_{a_x}] = [G] * [C_{a_x} / G] = (m^3 / (kg * s^2)) * (kg / m^3) = s^-2.
    Then ASD_a(f) = C_{a_x} * ASD_eta(f) has unit (s^-2) * (m / Hz^0.5)
                  = m / (s^2 * Hz^0.5).

    Parameters
    ----------
    distance_m : float
        Perpendicular distance H from channel to mirror in meters (H > 0).
    width_m : float
        Channel width w in meters (w > 0).
    correlation_area_m2 : float
        Correlation cell area A_cell in m^2 (A_cell > 0).
    density_kg_m3 : float, default=1000.0
        Fluid density rho in kg/m^3 (rho > 0).

    Returns
    -------
    float
        G-free spatial coupling factor C_{a_x} / G.
        Unit: kg / m^3.
    """
    h = _validate_positive_scalar("distance_m", distance_m)
    w = _validate_positive_scalar("width_m", width_m)
    a_cell = _validate_positive_scalar("correlation_area_m2", correlation_area_m2)
    rho = _validate_positive_scalar("density_kg_m3", density_kg_m3)

    spatial_integral_sqrt = np.sqrt(np.pi / (8.0 * (h**3)))
    return float(rho * np.sqrt(a_cell * w) * spatial_integral_sqrt)


def water_edge_step_transfer_g_free(
    x0_m: float,
    distance_m: float,
    volume_m3: float,
    frequency_hz: float | list | tuple | np.ndarray,
    density_kg_m3: float = 1000.0,
) -> np.ndarray:
    """Compute G-free deterministic acceleration Fourier spectrum for edge step.

    Evaluates the Fourier transform of gravitational acceleration induced by a
    single Heaviside step displacement of water volume Delta V at coordinate x0:
      Delta a_x(t) = G * rho * Delta V * (x0 / (D^2 + x0^2)^(3/2)) * Theta(t)
    with Fourier transform:
      Delta a_x(Omega) / G = rho * Delta V * (x0 / (D^2 + x0^2)^(3/2)) * (i / Omega).

    Physical Dimension:
      The Fourier transform of a single transient event has units of
      acceleration * time = (m / s^2) * s = m / (s^2 * Hz) = m / s.
      In G-free units (divided by G [m^3 / (kg * s^2)]):
      [Delta a_x(Omega) / G] = (m / s) / (m^3 / (kg * s^2)) = kg * s / m^2.
      Note: This is a DETERMINISTIC transient Fourier spectrum per single event,
      NOT a stationary power/amplitude spectral density (ASD).

    Parameters
    ----------
    x0_m : float
        Horizontal distance along channel to edge element in meters (finite).
    distance_m : float
        Perpendicular distance D from channel to mirror in meters (D > 0).
    volume_m3 : float
        Displaced volume Delta V in m^3 (Delta V > 0).
    frequency_hz : float or array_like
        Frequency f in Hz (f > 0).
    density_kg_m3 : float, default=1000.0
        Fluid density rho in kg/m^3 (rho > 0).

    Returns
    -------
    acc_transfer : np.ndarray
        G-free acceleration Fourier transform, shape matching frequency_hz.
    """
    x0 = _validate_finite_scalar("x0_m", x0_m)
    d = _validate_positive_scalar("distance_m", distance_m)
    dv = _validate_positive_scalar("volume_m3", volume_m3)
    rho = _validate_positive_scalar("density_kg_m3", density_kg_m3)

    f_arr, is_scalar = _validate_frequencies(frequency_hz, allow_zero=False)
    omega = 2.0 * np.pi * f_arr
    r_cubed = (d * d + x0 * x0) ** 1.5

    # Delta a_x(Omega) / G = rho * Delta V * (x0 / (D^2 + x0^2)^(3/2)) * (i / Omega)
    res = np.asarray(1j * rho * dv * (x0 / r_cubed) / omega, dtype=np.complex128)
    res.setflags(write=False)

    if is_scalar:
        return res.squeeze()
    return res
