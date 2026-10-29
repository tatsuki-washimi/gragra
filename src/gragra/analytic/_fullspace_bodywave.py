"""Private infinite-medium Fourier-Poisson plane-wave fields, G free.

Spatial phase exp(+i k.r), acceleration=-grad(Phi), gradient_ij=d_j a_i.
This is the full-space particular solution without cavity or boundary sources;
Harms2019 Eqs55/60. It supplies private deterministic source connections only.
"""

from __future__ import annotations

import numpy as np

from gragra._arrays import _check_no_bool
from gragra.analytic.values import GFreeGravityValues


def _bulk_gravity_transfer_g_free(density: float) -> float:
    """Shared longitudinal acceleration transfer4*pi*rho, with G omitted."""
    return 4.0 * np.pi * density


def _real(name: str, value: object) -> np.ndarray:
    _check_no_bool(value, name)
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be real numeric")
    result = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _scalar(name: str, value: object) -> float:
    result = _real(name, value)
    if result.shape != ():
        raise ValueError(f"{name} must be scalar")
    return float(result)


def _vector(name: str, value: object) -> np.ndarray:
    result = _real(name, value)
    if result.shape != (3,):
        raise ValueError(f"{name} must have shape(3,)")
    return result


def _direction(name: str, value: object) -> np.ndarray:
    result = _vector(name, value)
    scale = np.max(np.abs(result))
    if scale == 0:
        raise ValueError(f"{name} must be nonzero")
    scaled = result / scale
    return scaled / np.linalg.norm(scaled)


def fullspace_bodywave_observables_g_free(
    targets_m,
    *,
    density_kg_m3,
    wavenumber_rad_m,
    displacement_m,
    propagation_direction,
    polarization,
    wave_type,
    phase_origin_m=(0.0, 0.0, 0.0),
) -> GFreeGravityValues:
    """Return private P/S fields in an infinite homogeneous elastic bulk.

    Inputs are raw real direction/polarization vectors, normalized separately,
    a finite complex displacement amplitude and positive scalar wavenumber.
    P requires parallel/antiparallel polarization, S requires transverse
    polarization. Physical-type recognition uses the existing cavity1e-12
    tolerance; recognized S longitudinal residue is clamped to zero.

    No scattering, cavity, free surface, or boundary-source field is included.
    Output potential, acceleration and acceleration gradient have G-free units
    kg/m,kg/m²,kg/m³; a first-order SI boundary multiplies each field by G once.
    """
    rho = _scalar("density_kg_m3", density_kg_m3)
    k = _scalar("wavenumber_rad_m", wavenumber_rad_m)
    if rho < 0 or k <= 0:
        raise ValueError("nonnegative density and positive wavenumber required")
    _check_no_bool(displacement_m, "displacement_m")
    raw_amplitude = np.asarray(displacement_m)
    if raw_amplitude.dtype.kind not in "iufc":
        raise TypeError("displacement_m must be numeric")
    if raw_amplitude.shape != ():
        raise ValueError("displacement_m must be scalar")
    amplitude = complex(raw_amplitude)
    if not np.isfinite(amplitude):
        raise ValueError("displacement_m must be finite")
    direction = _direction("propagation_direction", propagation_direction)
    polar = _direction("polarization", polarization)
    projection = float(direction @ polar)
    if wave_type == "P" and abs(abs(projection) - 1) <= 1e-12:
        projection = float(np.copysign(1.0, projection))
    elif wave_type == "S" and abs(projection) <= 1e-12:
        projection = 0.0
    else:
        raise ValueError(
            "wave_type/polarization must specify parallel P or transverse S"
        )
    origin = _vector("phase_origin_m", phase_origin_m)
    targets = _real("targets_m", targets_m)
    if targets.ndim != 2 or targets.shape[1] != 3:
        raise ValueError("targets_m must have shape(M,3)")
    # After complete source/target validation, exact physical nulls require no
    # coefficient, phase or k division. They stay zero even when a nonnull
    # transfer would overflow or a tiny positive k would cause cancellation.
    if rho == 0 or amplitude == 0 or projection == 0:
        count = len(targets)
        return GFreeGravityValues(
            np.zeros(count, dtype=np.complex128),
            np.zeros((count, 3), dtype=np.complex128),
            np.zeros((count, 3, 3), dtype=np.complex128),
        )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        phase = np.exp(1j * k * ((targets - origin) @ direction))
        longitudinal = (
            _bulk_gravity_transfer_g_free(rho) * amplitude * projection * phase
        )
        potential = 1j * longitudinal / k
        acceleration = longitudinal[:, None] * direction
        gradient = 1j * k * acceleration[:, :, None] * direction[None, None, :]
    if not all(np.isfinite(v).all() for v in (potential, acceleration, gradient)):
        raise ValueError("derived fullspace fields must be finite")
    return GFreeGravityValues(potential, acceleration, gradient)
