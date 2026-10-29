"""Linear thermodynamic conversions to density perturbations.

These helpers accept physical pressure in pascals.  They do not infer a field
relationship from a name and do not accept kinematic pressure.
"""

from __future__ import annotations

import numpy as np


def _contains_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        return value.dtype.kind == "b"
    if isinstance(value, (list, tuple)):
        return any(_contains_bool(item) for item in value)
    return False


def _background(name: str, value: object) -> np.ndarray:
    if _contains_bool(value):
        raise TypeError(f"{name} must be real, positive, and finite")
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be real, positive, and finite")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be real, positive, and finite") from error
    if not np.isfinite(result).all() or np.any(result <= 0.0):
        raise ValueError(f"{name} must be real, positive, and finite")
    return result


def _perturbation(name: str, value: object) -> np.ndarray:
    if _contains_bool(value):
        raise TypeError(f"{name} must be a finite real or complex perturbation")
    raw = np.asarray(value)
    if raw.dtype.kind not in "iufc":
        raise TypeError(f"{name} must be a finite real or complex perturbation")
    dtype = np.complex128 if np.iscomplexobj(raw) else np.float64
    try:
        result = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"{name} must be a finite real or complex perturbation"
        ) from error
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite real or complex perturbation")
    return result


def _require_scalar_or_matching_shapes(**values: np.ndarray) -> None:
    shapes = {value.shape for value in values.values() if value.ndim != 0}
    if len(shapes) > 1:
        detail = ", ".join(f"{name}={value.shape}" for name, value in values.items())
        raise ValueError(f"array inputs must have one matched shape; got {detail}")


def _readonly_or_scalar(value: np.ndarray):
    result = np.asarray(value).copy()
    if not np.isfinite(result).all():
        raise ValueError("computed density perturbation must be finite")
    if result.ndim == 0:
        return result.dtype.type(result)
    result.setflags(write=False)
    return result


def isobaric_temperature_density_perturbation(
    rho0_kg_m3: object,
    delta_temperature_k: object,
    temperature0_k: object,
):
    """Return the ideal-gas isobaric linearization ``-rho0*delta_T/T0``.

    Background density and absolute temperature are positive finite real
    scalars or arrays.  The temperature perturbation may be real or complex.
    Scalars may scale an array, but all non-scalar inputs must have exactly the
    same shape; general NumPy broadcasting is intentionally rejected.
    """
    rho0 = _background("rho0_kg_m3", rho0_kg_m3)
    delta_temperature = _perturbation("delta_temperature_k", delta_temperature_k)
    temperature0 = _background("T0 (temperature0_k)", temperature0_k)
    _require_scalar_or_matching_shapes(
        rho0_kg_m3=rho0,
        delta_temperature_k=delta_temperature,
        temperature0_k=temperature0,
    )
    with np.errstate(all="ignore"):
        result = -rho0 * delta_temperature / temperature0
    return _readonly_or_scalar(result)


def adiabatic_pressure_density_perturbation(
    delta_pressure_pa: object,
    sound_speed_m_s: object,
):
    """Return ``delta_p/c_s**2`` for a physical pressure perturbation in Pa.

    Sound speed is positive finite and real.  Pressure perturbations may be
    real or complex.  Scalars may scale an array, but non-scalar inputs must
    have exactly matching shapes.
    """
    delta_pressure = _perturbation("delta_pressure_pa", delta_pressure_pa)
    sound_speed = _background("c_s (sound_speed_m_s)", sound_speed_m_s)
    _require_scalar_or_matching_shapes(
        delta_pressure_pa=delta_pressure,
        sound_speed_m_s=sound_speed,
    )
    with np.errstate(all="ignore"):
        result = delta_pressure / sound_speed**2
    return _readonly_or_scalar(result)
