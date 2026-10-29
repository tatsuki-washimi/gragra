"""G-free uniform-advection stochastic temperature gravity for Gen1.

Implements the selected T1 projected acceleration PSD for homogeneous/isotropic
temperature turbulence with Taylor frozen flow and uniform straight advection.

The statistical input is the structure-function coefficient c_T in
D_T(r)=c_T^2 r^p, not an arbitrary local temperature PSD. The output is a
one-sided projected acceleration PSD. The SI wrapper multiplies by G_SI**2.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gamma, kve

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


def _scaled_geometry(values: np.ndarray, theta: float) -> np.ndarray:
    """Return exp(2x) B without restoring the exponential attenuation."""
    k0e = kve(0, values)
    k1e = kve(1, values)
    k2e = kve(2, values)
    parallel_scaled = k1e * k1e - k0e * k0e
    perpendicular_scaled = k0e * k2e - k1e * k1e
    angular_scaled = (
        np.cos(theta) ** 2 * parallel_scaled
        + 0.5 * np.sin(theta) ** 2 * perpendicular_scaled
    )
    if not np.isfinite(angular_scaled).all() or np.any(angular_scaled < 0.0):
        raise ValueError("derived uniform-advection geometry factor is invalid")
    return angular_scaled


def uniform_advection_geometry_factor(
    x: float | ArrayLike,
    wind_axis_angle_rad: float,
) -> np.ndarray | float:
    """Return B(x,theta) from the Gen1 T1 uniform-flow specification.

    x=omega*r_min/v must be positive. theta is the angle between wind and the
    requested horizontal acceleration axis. Exponentially scaled K functions
    avoid underflow and reduce large-x cancellation; exp(-2x) is restored only
    after forming the algebraic differences.
    """
    raw = _real_array("x", x)
    if np.any(raw <= 0.0):
        raise ValueError("x must be positive")
    theta_raw = _real_array("wind_axis_angle_rad", wind_axis_angle_rad)
    if theta_raw.shape != ():
        raise ValueError("wind_axis_angle_rad must be a scalar")
    theta = float(theta_raw)

    values = np.atleast_1d(raw)
    angular_scaled = _scaled_geometry(values, theta)
    with np.errstate(under="ignore", invalid="ignore"):
        result = np.exp(-2.0 * values) * angular_scaled

    if not np.isfinite(result).all() or np.any(result < 0.0):
        raise ValueError("derived uniform-advection geometry factor is invalid")
    if raw.ndim == 0:
        return float(result[0])
    result.setflags(write=False)
    return result


def uniform_advection_temperature_acceleration_psd_g_free(
    *,
    density_kg_m3: float,
    temperature_k: float,
    structure_coefficient_k_m_neg_p_over_2: float,
    structure_exponent: float,
    wind_speed_m_s: float,
    minimum_approach_m: float,
    wind_axis_angle_rad: float,
    frequency_hz: float | ArrayLike,
) -> np.ndarray | float:
    """Return the T1 G-free one-sided projected acceleration PSD.

    structure_coefficient_k_m_neg_p_over_2 is c_T in D_T(r)=c_T^2 r^p,
    with units K*m^(-p/2). The model requires 0<p<2, positive wind speed
    and minimum approach distance, and a positive mean temperature.

    The result is in G-free units kg^2/m^4/Hz. A scalar frequency returns a
    float; a 1D frequency input returns a read-only float64 array.
    """
    rho = _nonnegative_scalar("density_kg_m3", density_kg_m3)
    temp = _positive_scalar("temperature_k", temperature_k)
    c_t = _nonnegative_scalar(
        "structure_coefficient_k_m_neg_p_over_2",
        structure_coefficient_k_m_neg_p_over_2,
    )
    p_raw = _real_array("structure_exponent", structure_exponent)
    if p_raw.shape != ():
        raise ValueError("structure_exponent must be a scalar")
    p = float(p_raw)
    if not 0.0 < p < 2.0:
        raise ValueError("structure_exponent must satisfy 0 < p < 2")
    wind = _positive_scalar("wind_speed_m_s", wind_speed_m_s)
    r_min = _positive_scalar("minimum_approach_m", minimum_approach_m)
    theta_raw = _real_array("wind_axis_angle_rad", wind_axis_angle_rad)
    if theta_raw.shape != ():
        raise ValueError("wind_axis_angle_rad must be a scalar")
    theta = float(theta_raw)
    frequencies, scalar = _frequencies(frequency_hz)

    omega = 2.0 * np.pi * frequencies
    x = omega * r_min / wind
    if not np.isfinite(x).all() or np.any(x <= 0.0):
        raise ValueError("derived uniform-advection x must be positive and finite")
    geometry = np.asarray(uniform_advection_geometry_factor(x, theta), dtype=np.float64)
    # Restoring subnormal B or overflowing the standalone prefactor can lose
    # a final PSD that is representable after all factors are combined.
    tail = geometry < np.finfo(np.float64).tiny
    result = np.empty_like(omega)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        if rho == 0.0 or c_t == 0.0:
            result.fill(0.0)
        else:
            if np.any(~tail):
                try:
                    coefficient = (
                        8.0
                        * np.pi**2
                        * gamma(p + 2.0)
                        * np.sin(np.pi * p / 2.0)
                        * (rho * r_min * c_t / temp) ** 2
                        * wind**p
                    )
                except OverflowError:
                    coefficient = np.inf
                if np.isfinite(coefficient) and coefficient > 0.0:
                    indices = np.flatnonzero(~tail)
                    direct = (
                        coefficient * omega[indices] ** (-(p + 1.0)) * geometry[indices]
                    )
                    result[indices] = direct
                    tail[indices[(~np.isfinite(direct)) | (direct == 0.0)]] = True
                else:
                    tail[:] = True
            if np.any(tail):
                scaled = _scaled_geometry(x[tail], theta)
                log_coefficient = (
                    np.log(8.0 * np.pi**2 * gamma(p + 2.0) * np.sin(np.pi * p / 2.0))
                    + 2.0 * (np.log(rho) + np.log(r_min) + np.log(c_t) - np.log(temp))
                    + p * np.log(wind)
                )
                with np.errstate(divide="ignore"):
                    result[tail] = np.exp(
                        log_coefficient
                        - (p + 1.0) * np.log(omega[tail])
                        - 2.0 * x[tail]
                        + np.log(scaled)
                    )

    if not np.isfinite(result).all() or np.any(result < 0.0):
        raise ValueError("derived temperature acceleration PSD is invalid")
    if scalar:
        return float(result[0])
    result = result.copy()
    result.setflags(write=False)
    return result


__all__ = [
    "uniform_advection_geometry_factor",
    "uniform_advection_temperature_acceleration_psd_g_free",
]
