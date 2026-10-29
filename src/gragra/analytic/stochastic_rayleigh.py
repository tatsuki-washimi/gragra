"""G-free stochastic Rayleigh-wave acceleration spectra.

This module implements the single-target, homogeneous/isotropic Rayleigh
acceleration CSD selected for the Gen1 stochastic catalogue. The physical
source is J. Harms, Living Reviews in Relativity 22, 6 (2019), Eq. (112),
together with an analytic azimuth average of the ground-side vector response
from Eqs. (96), (98), and (99).

The input PSD is the one-sided PSD of vertical surface displacement in
m^2/Hz. The output is a G-free acceleration CSD tensor: multiply by G_SI**2
at the SI boundary. No stochastic realization or spectral estimation is
performed here.
"""

from __future__ import annotations

import numpy as np

from gragra._arrays import _check_no_bool, as_xyz_table
from gragra.waves.rayleigh import _rayleigh_parameters, _validate_speeds

ArrayLike = list | tuple | np.ndarray


def _real_array(name: str, value: object) -> np.ndarray:
    """Return a finite float64 array without silently accepting bool/complex."""
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


def _nonnegative_scalar(name: str, value: object) -> float:
    array = _real_array(name, value)
    if array.shape != ():
        raise ValueError(f"{name} must be a scalar")
    result = float(array)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _frequencies(name: str, value: object) -> tuple[np.ndarray, bool]:
    array = _real_array(name, value)
    if array.ndim not in (0, 1):
        raise ValueError(f"{name} must be a scalar or a 1D array")
    if array.ndim == 1 and len(array) == 0:
        raise ValueError(f"{name} must be nonempty")
    if np.any(array <= 0.0):
        raise ValueError(f"{name} must contain only positive values")
    scalar = array.ndim == 0
    return np.atleast_1d(array), scalar


def _psd_values(name: str, value: object, frequency_count: int) -> np.ndarray:
    array = _real_array(name, value)
    if array.ndim == 0:
        values = np.full(frequency_count, float(array), dtype=np.float64)
    elif array.ndim == 1 and array.shape == (frequency_count,):
        values = array.copy()
    else:
        raise ValueError(
            f"{name} must be a scalar or have shape ({frequency_count},)"
        )
    if np.any(values < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return values


def rayleigh_isotropic_acceleration_csd_g_free(
    targets_m: ArrayLike,
    *,
    density_kg_m3: float,
    v_p_m_s: float,
    v_s_m_s: float,
    frequency_hz: float | ArrayLike,
    surface_vertical_displacement_psd_m2_hz: float | ArrayLike,
) -> np.ndarray:
    """Return the G-free one-sided acceleration CSD for isotropic Rayleigh noise.

    targets_m has shape (M, 3) in the Rayleigh surface frame. The free surface
    is z=0 and this Gen1 core accepts only z>0. Horizontal coordinates do not
    affect a single-target auto-spectrum in the homogeneous/isotropic model.

    frequency_hz is a positive scalar or 1D array. The input displacement PSD
    is in m^2/Hz; a scalar applies to every frequency or a 1D array must match
    the frequency count exactly.

    The read-only float64 return value is in G-free units kg^2/m^4/Hz with
    shape (M, 3, 3) for scalar frequency or (K, M, 3, 3) for K frequencies.
    In the surface frame it is A^2 S_xi diag(1/2, 1/2, 1), where
    A = 2*pi*rho*gamma*exp(-k_R*z),
    gamma=(1-zeta)/(q-zeta), and zeta=sqrt(q/s).

    The horizontal diagonal reproduces Harms 2019 Eq. (112) after the SI
    wrapper multiplies by G_SI**2. Off-diagonal CSDs vanish under the ideal
    isotropy assumption.
    """
    _check_no_bool(targets_m, "targets_m")
    raw_targets = np.asarray(targets_m)
    if raw_targets.dtype.kind in {"b", "O", "c"}:
        raise TypeError("targets_m must contain finite real coordinates")
    targets = as_xyz_table("targets_m", targets_m)
    if len(targets) == 0:
        raise ValueError("targets_m must be nonempty")
    heights = targets[:, 2]
    if np.any(heights <= 0.0):
        offending = int(np.nonzero(heights <= 0.0)[0][0])
        raise ValueError(
            "targets_m must lie in the open ground-side half-space z>0 "
            f"(first offending index: {offending})"
        )

    rho = _nonnegative_scalar("density_kg_m3", density_kg_m3)
    v_p, v_s = _validate_speeds(v_p_m_s, v_s_m_s)
    if not v_p * v_p > (4.0 / 3.0) * v_s * v_s:
        raise ValueError(
            "requires v_p_m_s**2 > (4/3) * v_s_m_s**2 for a stable "
            "isotropic half-space"
        )

    frequencies, scalar_frequency = _frequencies("frequency_hz", frequency_hz)
    input_psd = _psd_values(
        "surface_vertical_displacement_psd_m2_hz",
        surface_vertical_displacement_psd_m2_hz,
        len(frequencies),
    )

    c_r, q, s = _rayleigh_parameters(v_p, v_s)
    zeta = float(np.sqrt(q / s))
    gamma = (1.0 - zeta) / (q - zeta)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        k_r = 2.0 * np.pi * frequencies / c_r
        transfer = (
            2.0
            * np.pi
            * rho
            * gamma
            * np.exp(-k_r[:, None] * heights[None, :])
        )
        base = transfer * transfer * input_psd[:, None]
    if not np.isfinite(base).all() or np.any(base < 0.0):
        raise ValueError(
            "derived Rayleigh acceleration PSD is not finite and non-negative"
        )

    result = np.zeros((len(frequencies), len(targets), 3, 3), dtype=np.float64)
    result[..., 0, 0] = 0.5 * base
    result[..., 1, 1] = 0.5 * base
    result[..., 2, 2] = base
    if scalar_frequency:
        result = result[0].copy()
    result.setflags(write=False)
    return result


__all__ = ["rayleigh_isotropic_acceleration_csd_g_free"]
