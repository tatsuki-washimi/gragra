"""G-free stochastic acoustic acceleration spectra for Gen1.

Selected models:
- A1: isotropic rigid-surface atmospheric infrasound;
- A2: the same field evaluated at underground depth;
- A3: the published centered half-spherical-cavern acoustic model.

All pressure inputs are one-sided PSDs in Pa^2/Hz.  A1/A2 use total pressure
at the rigid reflecting surface.  A3 uses the cavern-field pressure PSD.
Outputs are G-free acceleration CSD tensors; the SI boundary multiplies by
G_SI**2.

Public scope notes for model A2 (underground isotropic acoustic):

- **Observable tier**: Values reproducible from this package with published
  mathematics and the public pytest suite (including mpmath cross-checks where
  documented in the analytic formula ledger).
- **Development-diagnostic tier**: Additional checks exist in the private
  development repository that are not part of formal public acceptance.
- **Unverified tier**: Order estimates and surplus upper bounds at finite
  depth parameter x are not proven as rigorous surplus bounds (see ledger).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import iv, modstruve

from gragra._arrays import _check_no_bool

ArrayLike = list | tuple | np.ndarray


def _real_array(name: str, value: object) -> np.ndarray:
    _check_no_bool(value, name)
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must not be boolean")
    raw = np.asarray(value)
    if raw.dtype.kind == "b":
        raise TypeError(f"{name} must not contain boolean values")
    if raw.dtype.kind == "O":
        raise TypeError(f"{name} must not contain object dtype")
    if raw.dtype.kind == "c":
        raise TypeError(f"{name} must contain real values")
    try:
        array = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must contain real numbers") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _positive_scalar(name: str, value: object) -> float:
    array = _real_array(name, value)
    if array.shape != ():
        raise ValueError(f"{name} must be a scalar")
    result = float(array)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_scalar(name: str, value: object) -> float:
    array = _real_array(name, value)
    if array.shape != ():
        raise ValueError(f"{name} must be a scalar")
    result = float(array)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _frequencies(value: object) -> tuple[np.ndarray, bool]:
    array = _real_array("frequency_hz", value)
    if array.ndim not in (0, 1):
        raise ValueError("frequency_hz must be a scalar or a 1D array")
    if array.ndim == 1 and len(array) == 0:
        raise ValueError("frequency_hz must be nonempty")
    if np.any(array <= 0.0):
        raise ValueError("frequency_hz must contain only positive values")
    return np.atleast_1d(array), array.ndim == 0


def _psd_values(name: str, value: object, count: int) -> np.ndarray:
    array = _real_array(name, value)
    if array.ndim == 0:
        values = np.full(count, float(array), dtype=np.float64)
    elif array.ndim == 1 and array.shape == (count,):
        values = array.copy()
    else:
        raise ValueError(f"{name} must be a scalar or have shape ({count},)")
    if np.any(values < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return values


def _finish(result: np.ndarray, scalar_frequency: bool) -> np.ndarray:
    if scalar_frequency:
        result = result[0].copy()
    diagonal = np.diagonal(result, axis1=-2, axis2=-1)
    if not np.isfinite(result).all() or np.any(diagonal < 0.0):
        raise ValueError("derived acoustic acceleration CSD is not finite/non-negative")
    result.setflags(write=False)
    return result


def _isotropic_depth_factor_small(x: float) -> float:
    """Power series of I_iso(x)=1/2 int sin^3(theta)e^-x sin(theta) dtheta."""
    # General coefficient:
    # 0.5 * (-x)^n/n! * integral_0^(pi/2) sin^(n+3)(theta) dtheta.
    # Recurrence for J_m = int sin^m(theta)dtheta:
    # J_m = (m-1)/m * J_(m-2), with J_0=pi/2, J_1=1.
    total = 0.0
    term_power = 1.0
    factorial = 1.0
    j_even = math.pi / 2.0
    j_odd = 1.0
    previous_even_m = 0
    previous_odd_m = 1
    for n in range(48):
        m = n + 3
        if m % 2 == 0:
            while previous_even_m < m:
                next_m = previous_even_m + 2
                j_even *= (next_m - 1) / next_m
                previous_even_m = next_m
            integral = j_even
        else:
            while previous_odd_m < m:
                next_m = previous_odd_m + 2
                j_odd *= (next_m - 1) / next_m
                previous_odd_m = next_m
            integral = j_odd
        term = 0.5 * term_power * integral / factorial
        total += term
        if n > 8 and abs(term) <= 2.0e-16 * max(1.0, abs(total)):
            break
        term_power *= -x
        factorial *= n + 1.0
    return total


_WATSON_COEFFS: tuple[float, ...] = (
    3.0,
    30.0,
    945.0,
    56700.0,
    5457375.0,
    766215450.0,
    147496474125.0,
    37253395179000.0,
)

# Gauss-Legendre rule in theta on [0, pi/2]: for nodes z_j and weights w_j
# on [-1, 1], theta_j=pi*(1+z_j)/4, s_j=sin(theta_j), and
# A_j=pi*w_j*sin(theta_j)^3/8. Rounded once from a 120-digit reconstruction.
_A_COEFFS: np.ndarray = np.array(
    (
        1.1136505121020204e-12, 3.7753837001627286e-10, 8.735914906181037e-09,
        7.536619368666965e-08, 3.8750908259559705e-07, 1.4464305089146808e-06,
        4.338813252563138e-06, 1.1097555132028024e-05, 2.5127763205878658e-05,
        5.165327720254605e-05, 9.811994353903096e-05, 0.00017447727485759163,
        0.00029325360206929903, 0.000469344674441806, 0.000719454241367461,
        0.001061158041231992, 0.0015116079752194977, 0.002085946450516076,
        0.0027955548017006408, 0.003646305531082393, 0.004637016800421167,
        0.005758311662094402, 0.006992059768775698, 0.008311526228598485,
        0.009682276489592332, 0.011063797798589434, 0.012411709958877939,
        0.013680364431672396, 0.014825582868116397, 0.01580727122325934,
        0.016591665365281474, 0.01715301442684997, 0.017474580139363413,
        0.017548912261918778, 0.01737743964579977, 0.01696948282226157,
        0.01634083994489369, 0.015512120247748925, 0.014506998704483057,
        0.013350546300831725, 0.012067758226002744, 0.010682363919994034,
        0.009215964314468097, 0.007687507555162011, 0.006113088716648969,
        0.004506048537038546, 0.002877412241016162, 0.0012383143684018816,
    ),
    dtype=np.float64,
)

_S_COEFFS: np.ndarray = np.array(
    (
        0.0009652484968854935, 0.005081368952164619, 0.012468199721909521,
        0.023095255576735578, 0.03691498234370307, 0.053862688734287276,
        0.07385504659420666, 0.09678834933156796, 0.12253674791702363,
        0.1509506274654814, 0.18185528402373896, 0.21505005791729484,
        0.25030806987455395, 0.2873766877237417, 0.3259788246491025,
        0.36581513542513194, 0.4065671359767717, 0.4479012258848773,
        0.48947354543064675, 0.5309355511997736, 0.5719401501062751,
        0.6121481938960484, 0.6512351074471067, 0.6888974066998751,
        0.7248588573247808, 0.75887603390037, 0.790743061120239,
        0.8202953521119155, 0.847412202220497, 0.872018146785692,
        0.8940830452764577, 0.9136209082237803, 0.9306875344034067,
        0.9453770707397073, 0.957817644101909, 0.9681662409708756,
        0.976603027125028, 0.9833253051115105, 0.9885413031673372,
        0.992463976882748, 0.9953049861392589, 0.9972689868298159,
        0.998548351739821, 0.9993184097566521, 0.9997332690122127,
        0.9999222689767913, 0.9999870897614489, 0.9999995341475612,
    ),
    dtype=np.float64,
)


def _isotropic_depth_factor_large(x: float) -> float:
    """Large-x Watson asymptotic expansion for x > 100.

    Watson's lemma applied to I_iso(x) = 1/2 int_0^1 s^3 (1-s^2)^(-1/2) e^(-s x) ds
    yields the asymptotic expansion in even inverse powers of x:
        I_iso(x) ~ 3/x^4 + 30/x^6 + 945/x^8 + 56700/x^10 + ...
    Coefficient n is (1/2)_n (2n+3)! / (2 n!). The first omitted formal term has
    order x^-20 (about 1.2e-24 at x = 100, i.e. ~4e-17 relative to I_iso); this
    order estimate is not a finite-x remainder bound.
    """
    inv2 = 1.0 / (x * x)
    val = 0.0
    term = inv2 * inv2
    for c in _WATSON_COEFFS:
        val += c * term
        term *= inv2
    return val


def isotropic_acoustic_depth_factor(x: float | ArrayLike) -> np.ndarray | float:
    """Return the analytic A2 isotropic depth factor I_iso(x).

    x = 4*pi*f*h/c_s >= 0.

    The mathematical definition is:
        I_iso(x) = 1/2 int_0^(pi/2) sin^3(theta) exp(-x sin(theta)) dtheta
                 = 1/2 int_0^1 (s^3 / sqrt(1 - s^2)) exp(-s x) ds

    Evaluation uses three complementary analytic domains, checked in tests against
    independent 60-digit mpmath quadrature of the defining integral (measured
    errors are recorded in ``tests/test_stochastic_a2_continuity.py``; they are
    not a guarantee over the whole parameter domain):

    1. Small x (0 <= x < 0.75):
       Taylor series about x = 0, with at most 48 terms.
       Exact value at origin: I_iso(0) = 1/3.
    2. Intermediate x (0.75 <= x <= 100.0):
       Precomputed 48-node Gauss-Legendre exponential representation:
           I_iso(x) approx sum_{j=1}^48 A_j exp(-s_j x)
       All weights A_j > 0 and nodes s_j in (0, 1), guaranteeing strictly
       positive terms and eliminating catastrophic cancellation between
       modified Bessel and Struve functions.
    3. Large x (x > 100.0):
       8-term Watson asymptotic expansion in powers of 1/x^2 with
       c_n = (1/2)_n (2n+3)! / (2 n!).

    No runtime quadrature is used.
    """
    raw = _real_array("x", x)
    if np.any(raw < 0.0):
        raise ValueError("x must be non-negative")
    values = np.atleast_1d(raw)
    out = np.empty_like(values)
    for index, number in enumerate(values):
        number = float(number)
        if number == 0.0:
            out[index] = 1.0 / 3.0
        elif number < 0.75:
            out[index] = _isotropic_depth_factor_small(number)
        elif number <= 100.0:
            out[index] = float(np.sum(_A_COEFFS * np.exp(-_S_COEFFS * number)))
        else:
            out[index] = _isotropic_depth_factor_large(number)
    if not np.isfinite(out).all() or np.any(out < 0.0):
        raise ValueError("derived isotropic acoustic depth factor is invalid")
    if raw.ndim == 0:
        return float(out[0])
    out.setflags(write=False)
    return out



def _one_minus_sinc(x: np.ndarray) -> np.ndarray:
    """Stable 1-sin(x)/x for finite real x."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    small = np.abs(x) < 1.0e-3
    x2 = x[small] * x[small]
    out[small] = (
        x2 / 6.0
        - x2 * x2 / 120.0
        + x2 * x2 * x2 / 5040.0
        - x2 * x2 * x2 * x2 / 362880.0
    )
    out[~small] = 1.0 - np.sin(x[~small]) / x[~small]
    return out


def surface_isotropic_acoustic_acceleration_csd_g_free(
    *,
    sound_speed_m_s: float,
    frequency_hz: float | ArrayLike,
    surface_pressure_psd_pa2_hz: float | ArrayLike,
) -> np.ndarray:
    """Return A1 G-free acceleration CSD for isotropic surface infrasound."""
    c_s = _positive_scalar("sound_speed_m_s", sound_speed_m_s)
    frequencies, scalar = _frequencies(frequency_hz)
    pressure_psd = _psd_values(
        "surface_pressure_psd_pa2_hz", surface_pressure_psd_pa2_hz, len(frequencies)
    )
    base = pressure_psd / (c_s * frequencies) ** 2
    result = np.zeros((len(frequencies), 3, 3), dtype=np.float64)
    result[:, 0, 0] = base / 3.0
    result[:, 1, 1] = base / 3.0
    result[:, 2, 2] = 2.0 * base / 3.0
    return _finish(result, scalar)


def underground_isotropic_acoustic_acceleration_csd_g_free(
    *,
    depth_m: float,
    sound_speed_m_s: float,
    frequency_hz: float | ArrayLike,
    surface_pressure_psd_pa2_hz: float | ArrayLike,
) -> np.ndarray:
    """Return A2 G-free acceleration CSD at underground depth h>0.

    Public acceptance is limited to the observable tier described in the
    module docstring. Development-only diagnostics are not formal acceptance.
    """
    depth = _nonnegative_scalar("depth_m", depth_m)
    c_s = _positive_scalar("sound_speed_m_s", sound_speed_m_s)
    frequencies, scalar = _frequencies(frequency_hz)
    pressure_psd = _psd_values(
        "surface_pressure_psd_pa2_hz", surface_pressure_psd_pa2_hz, len(frequencies)
    )
    x = 4.0 * np.pi * frequencies * depth / c_s
    factor = np.asarray(isotropic_acoustic_depth_factor(x), dtype=np.float64)
    base = pressure_psd * factor / (c_s * frequencies) ** 2
    result = np.zeros((len(frequencies), 3, 3), dtype=np.float64)
    result[:, 0, 0] = base
    result[:, 1, 1] = base
    result[:, 2, 2] = 2.0 * base
    return _finish(result, scalar)


def centered_cavern_isotropic_acoustic_acceleration_csd_g_free(
    *,
    radius_m: float,
    sound_speed_m_s: float,
    frequency_hz: float | ArrayLike,
    cavern_pressure_psd_pa2_hz: float | ArrayLike,
) -> np.ndarray:
    """Return A3 G-free CSD, including its zero-radius whole-tensor limit."""
    radius = _nonnegative_scalar("radius_m", radius_m)
    c_s = _positive_scalar("sound_speed_m_s", sound_speed_m_s)
    frequencies, scalar = _frequencies(frequency_hz)
    pressure_psd = _psd_values(
        "cavern_pressure_psd_pa2_hz", cavern_pressure_psd_pa2_hz, len(frequencies)
    )
    kr = 2.0 * np.pi * frequencies * radius / c_s
    one_minus_sinc = _one_minus_sinc(kr)
    component = (
        (2.0 / (c_s * frequencies)) ** 2
        * pressure_psd
        * (one_minus_sinc * one_minus_sinc)
        / 3.0
    )
    result = np.zeros((len(frequencies), 3, 3), dtype=np.float64)
    result[:, 0, 0] = component
    result[:, 1, 1] = component
    result[:, 2, 2] = component
    return _finish(result, scalar)


__all__ = [
    "isotropic_acoustic_depth_factor",
    "surface_isotropic_acoustic_acceleration_csd_g_free",
    "underground_isotropic_acoustic_acceleration_csd_g_free",
    "centered_cavern_isotropic_acoustic_acceleration_csd_g_free",
]
