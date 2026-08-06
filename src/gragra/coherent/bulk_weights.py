"""Polarization-projected bulk mass-weight helper.

Physics conventions and formulation notes:
- G-free convention: this package computes in a G-free (G_SI = 1) semantic
  space. The output is the complex weight delta_m_j [kg] representing the
  mass perturbation at each point.
- Complex-output disclaimer: complex outputs represent linear complex
  amplitudes or transfer-function-like responses, not directly real-valued
  time-domain gravitational fields.
- Formulation distinction: following the pure-bulk formulation, this
  computes the weights corresponding to the medium's volume term (the
  density perturbation).
"""

from typing import Literal

import numpy as np

from gragra._arrays import as_xyz_table
from gragra.array_types import ComplexArray
from gragra.waves import (
    displacement_direction,
    normalize_directions,
    plane_wave_complex_factor,
)


def _validate_scalar(val: float, name: str, positive: bool = False) -> float:
    """Validate a numeric scalar value (finite, optionally positive)."""
    if isinstance(val, bool) or not isinstance(val, (int, float, np.number)):
        raise TypeError(f"{name} must be a numeric scalar")
    val_float = float(val)
    if not np.isfinite(val_float):
        raise ValueError(f"{name} must be finite")
    if positive and val_float <= 0.0:
        raise ValueError(f"{name} must be positive")
    return val_float


def plane_wave_bulk_mass_weights(
    volumes_m3: list | tuple | np.ndarray,
    positions_m: list | tuple | np.ndarray,
    density_kg_m3: float,
    directions: list | tuple | np.ndarray,
    wavenumber_rad_m: float | list | tuple | np.ndarray,
    *,
    wave_type: Literal["P", "S1", "S2", "SH", "SV", "custom"] = "P",
    polarization: list | tuple | np.ndarray | None = None,
    displacement_m: float = 1.0,
    phase0: float = 0.0,
) -> ComplexArray:
    """Calculate the polarization-projected bulk mass weights for a plane wave.

    Calculates the mass perturbation weights:
        δm_j = −ρ₀ * (i k) * (k̂ · ê_pol) * ξ * V_j * e^{i k k̂ · r_j + iφ₀}

    Complex outputs represent linear complex amplitudes or transfer-function-like
    responses, not directly real-valued time-domain gravitational fields.

    G-free convention:
        The output is the mass perturbation δm_j [kg] under the G-free
        (G_SI = 1) convention.
    Complex-output disclaimer:
        The output is a complex (np.complex128) array of linear complex
        amplitudes -- see the paragraph above.
    Formulation distinction:
        Corresponds to the discretized weights of the volume-integral term
        (δm = δρ * dV) in the pure-bulk formulation.

    Parameters
    ----------
    volumes_m3 : array-like
        Shape (N,) volume of each mass element in m^3 (must be positive).
    positions_m : array-like
        Shape (N, 3) coordinates of each mass element in meters.
    density_kg_m3 : float
        Medium background mass density in kg/m^3 (positive).
    directions : array-like
        Shape (Ndir, 3) propagation directions (wavenumber vectors).
    wavenumber_rad_m : float or array-like
        Shape (Nf,) wavenumbers in rad/m (non-negative).
    wave_type : str, optional
        Wave type (e.g. "P", "SH", "SV", "custom", defaults to "P"). For a
        pure shear wave the projection k̂ · ê_pol vanishes identically, so
        the returned weights are exactly zero -- that is the physically
        correct answer for a bulk term, not a failure.
    polarization : array-like, optional
        Custom polarization direction of shape (Ndir, 3) if wave_type="custom".
    displacement_m : float, optional
        Displacement amplitude of the plane wave in meters (positive).
    phase0 : float, optional
        Initial phase of the plane wave in radians.

    Returns
    -------
    ComplexArray
        Mass weight array of shape (N, Ndir, Nf) (read-only).
    """
    # 1. Validate scalar parameters using unified helper
    rho = _validate_scalar(density_kg_m3, "density_kg_m3", positive=True)
    xi = _validate_scalar(displacement_m, "displacement_m", positive=True)
    phase0_val = _validate_scalar(phase0, "phase0")

    # 2. Convert and validate positions, volumes
    pos = as_xyz_table("positions_m", positions_m)
    vol = np.asarray(volumes_m3, dtype=np.float64)

    if vol.ndim != 1:
        raise ValueError("volumes_m3 must be a 1D array")
    if not np.all(np.isfinite(vol)):
        raise ValueError("volumes_m3 must contain only finite values")
    if np.any(vol <= 0.0):
        raise ValueError("volumes_m3 must contain only positive values")
    if len(vol) != len(pos):
        raise ValueError("volumes_m3 and positions_m must have the same length")

    # 3. Handle directions and polarization
    k_hat = normalize_directions(directions)
    e_pol = displacement_direction(wave_type, k_hat, polarization=polarization)

    # 4. Compute projection k_hat · e_pol
    proj = np.sum(k_hat * e_pol, axis=-1)

    # Force projection to exactly 0 for shear/S waves to avoid round-off error
    if wave_type in ("SH", "SV", "S1", "S2"):
        proj = np.zeros_like(proj)

    # 5. Reuse plane_wave_complex_factor for the phase factors
    # (replaces self-implemented kz/phase_factor logic)
    phase_factor = plane_wave_complex_factor(
        positions_m=pos,
        k_hat=k_hat,
        wavenumber_rad_m=wavenumber_rad_m,
        phase0=phase0_val,
        sign=1,
    )

    # 6. Align wavenumber shape for broadcasting
    k_arr = np.asarray(wavenumber_rad_m, dtype=np.float64)
    k_arr_1d = k_arr[np.newaxis] if k_arr.ndim == 0 else k_arr

    # Calculate final weights: (N, Ndir, Nf)
    res = (
        -1j
        * rho
        * xi
        * k_arr_1d[None, None, :]
        * proj[None, :, None]
        * vol[:, None, None]
        * phase_factor
    )

    # Return read-only copy
    res_arr = np.asarray(res, dtype=np.complex128).copy()
    res_arr.setflags(write=False)
    return res_arr
