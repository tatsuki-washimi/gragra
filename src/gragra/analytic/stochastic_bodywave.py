"""G-free stochastic body-wave acceleration spectra for Gen1.

This module implements two selected stochastic body-wave models:

- homogeneous/isotropic P waves in an infinite homogeneous elastic bulk;
- homogeneous/isotropic mutually incoherent P/S waves evaluated at the
  center of a spherical cavity, neglecting elastic scattering.

Inputs are one-sided per-Cartesian-component displacement PSDs in m^2/Hz.
Outputs are G-free acceleration CSD tensors.  The SI boundary multiplies by
G_SI**2.
"""

from __future__ import annotations

import numpy as np
from scipy.special import spherical_jn

from gragra._arrays import _check_no_bool
from gragra.analytic._fullspace_bodywave import _bulk_gravity_transfer_g_free

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
    scalar = array.ndim == 0
    return np.atleast_1d(array), scalar


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


def _isotropic_tensor(base: np.ndarray, scalar_frequency: bool) -> np.ndarray:
    result = np.zeros((len(base), 3, 3), dtype=np.float64)
    result[:, 0, 0] = base
    result[:, 1, 1] = base
    result[:, 2, 2] = base
    if scalar_frequency:
        result = result[0].copy()
    result.setflags(write=False)
    return result


def _j1_over_x(x: np.ndarray) -> np.ndarray:
    """Stable j1(x)/x for positive finite x."""
    x = np.asarray(x, dtype=np.float64)
    result = np.empty_like(x)
    small = np.abs(x) < 1.0e-3
    x2 = x[small] * x[small]
    result[small] = (
        1.0 / 3.0
        - x2 / 30.0
        + x2 * x2 / 840.0
        - x2 * x2 * x2 / 45360.0
    )
    result[~small] = spherical_jn(1, x[~small]) / x[~small]
    return result


def fullspace_p_isotropic_acceleration_csd_g_free(
    *,
    density_kg_m3: float,
    frequency_hz: float | ArrayLike,
    p_displacement_psd_m2_hz: float | ArrayLike,
) -> np.ndarray:
    """Return G-free acceleration CSD for isotropic P waves in full space.

    The input PSD is the one-sided per-Cartesian-component P-displacement PSD.
    Harms 2019 Eqs. (55)/(60) give delta a = 4*pi*G*rho*xi_P, so the
    G-free CSD is (4*pi*rho)^2 S_xi times the identity.

    Returns a read-only float64 array with shape (3, 3) for scalar frequency
    or (K, 3, 3) for K frequencies, in G-free units kg^2/m^4/Hz.
    """
    rho = _nonnegative_scalar("density_kg_m3", density_kg_m3)
    frequencies, scalar_frequency = _frequencies(frequency_hz)
    psd = _psd_values(
        "p_displacement_psd_m2_hz",
        p_displacement_psd_m2_hz,
        len(frequencies),
    )
    del frequencies  # frequency is an alignment coordinate; transfer is local.
    with np.errstate(over="ignore", invalid="ignore"):
        base = _bulk_gravity_transfer_g_free(rho) ** 2 * psd
    if not np.isfinite(base).all() or np.any(base < 0.0):
        raise ValueError("derived full-space P acceleration PSD is not finite")
    return _isotropic_tensor(base, scalar_frequency)


def spherical_cavity_ps_isotropic_acceleration_csd_g_free(
    *,
    radius_m: float,
    density_kg_m3: float,
    v_p_m_s: float,
    v_s_m_s: float,
    frequency_hz: float | ArrayLike,
    p_displacement_psd_m2_hz: float | ArrayLike,
    s_displacement_psd_m2_hz: float | ArrayLike,
) -> np.ndarray:
    """Return centered spherical-cavity P/S acceleration CSD, G-free.

    This is Harms 2019 Eq. (65) promoted to mutually incoherent isotropic
    P/S displacement statistics. Elastic scattering is excluded.

    The P and S inputs are one-sided per-Cartesian-component displacement PSDs.
    The output is isotropic and read-only, shape (3, 3) for scalar frequency
    or (K, 3, 3) for K frequencies, in G-free units kg^2/m^4/Hz.
    """
    radius = _positive_scalar("radius_m", radius_m)
    rho = _nonnegative_scalar("density_kg_m3", density_kg_m3)
    vp = _positive_scalar("v_p_m_s", v_p_m_s)
    vs = _positive_scalar("v_s_m_s", v_s_m_s)
    if not vs < vp:
        raise ValueError("requires 0 < v_s_m_s < v_p_m_s")

    frequencies, scalar_frequency = _frequencies(frequency_hz)
    p_psd = _psd_values(
        "p_displacement_psd_m2_hz",
        p_displacement_psd_m2_hz,
        len(frequencies),
    )
    s_psd = _psd_values(
        "s_displacement_psd_m2_hz",
        s_displacement_psd_m2_hz,
        len(frequencies),
    )

    k_p = 2.0 * np.pi * frequencies / vp
    k_s = 2.0 * np.pi * frequencies / vs
    f_p = _j1_over_x(k_p * radius)
    f_s = _j1_over_x(k_s * radius)
    with np.errstate(over="ignore", invalid="ignore"):
        base = (4.0 * np.pi * rho) ** 2 * (
            4.0 * f_p * f_p * p_psd + f_s * f_s * s_psd
        )
    if not np.isfinite(base).all() or np.any(base < 0.0):
        raise ValueError("derived spherical-cavity P/S acceleration PSD is not finite")
    return _isotropic_tensor(base, scalar_frequency)


__all__ = [
    "fullspace_p_isotropic_acceleration_csd_g_free",
    "spherical_cavity_ps_isotropic_acceleration_csd_g_free",
]
