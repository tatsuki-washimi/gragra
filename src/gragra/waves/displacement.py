"""Plane wave displacement utility."""

import numpy as np

from gragra.waves.phase import plane_wave_complex_factor
from gragra.waves.polarization import displacement_direction


def plane_wave_displacement(
    positions_m,
    k_hat,
    wavenumber_rad_m,
    wave_type: str,
    *,
    amplitude_m: float = 1.0,
    phase0: float = 0.0,
    vertical_unit=(0.0, 0.0, 1.0),
    polarization=None,
    orthogonality_tol: float = 1e-5,
) -> np.ndarray:
    """Calculate the displacement field of a plane wave.

    Formula: amplitude_m * displacement_direction * plane_wave_complex_factor

    Parameters
    ----------
    positions_m : array_like
        Target positions where displacement is evaluated, shape (N, 3).
    k_hat : array_like
        Wave propagation direction vectors, shape (Ndir, 3).
    wavenumber_rad_m : float or array_like
        Wavenumber values, shape (Nf,).
    wave_type : str
        Wave polarization type. Must be "P", "S1", "S2", "SH", "SV", or "custom".
    amplitude_m : float, optional
        Wave amplitude. Must be a finite real number. Negative values are allowed
        and are equivalent to adding a phase shift of pi. Default is 1.0.
    phase0 : float, optional
        Initial phase offset. Default is 0.0.
    vertical_unit : array_like, optional
        Vertical unit vector defining SV/SH orientations, shape (3,).
        Default is (0.0, 0.0, 1.0).
    polarization : array_like, optional
        Custom polarization vector, only used if wave_type is "custom". Default is None.
    orthogonality_tol : float, optional
        Orthogonality tolerance for custom polarization. Default is 1e-5.

    Returns
    -------
    numpy.ndarray
        Displacement field array of shape (N, Ndir, Nf, 3) and dtype complex128.
        It is read-only.

    Notes
    -----
    - This function is placed in `gragra.waves` because it represents a purely kinematic
      quantity and is independent of the gravitational constant G (G-free).
    - The phase follows the e^{i(k.x - omega*t)} time factor convention. The sign of the
      phase is fixed to +1 internally, meaning there is no `sign` argument in this API.
      This ensures consistency with the source-side mass weights
      (`plane_wave_bulk_mass_weights`) and prevents phase conjugation mismatches in
      cross statistics.
    - When coupling this displacement with source mass weights to calculate
      cross-statistics (like gravitoelastic tensors c_ij), the source and
      witness displacement MUST share the same wave propagation parameters
      (k_hat, wavenumber_rad_m, phase0).
    - For surface-wave displacement calculations like Rayleigh waves, the phase
      convention alignment (such as defining which component is real and
      positive at the phase origin) is shared, but the API signatures are
      not unified.
    - Complex outputs represent linear complex amplitudes or
      transfer-function-like responses, not directly real-valued time-domain
      fields.
    """
    # Validate amplitude_m
    if isinstance(amplitude_m, (complex, np.complexfloating)):
        raise TypeError("amplitude_m must be real-valued")
    try:
        amp = float(amplitude_m)
    except (ValueError, TypeError) as e:
        raise TypeError(f"amplitude_m must be a real number: {e}") from e

    if not np.isfinite(amp):
        raise ValueError("amplitude_m must be a finite number")

    # Delegate input validation for wave_type, k_hat, positions_m, etc.
    # to underlying displacement_direction and plane_wave_complex_factor
    dir_vec = displacement_direction(
        wave_type=wave_type,
        k_hat=k_hat,
        vertical_unit=vertical_unit,
        polarization=polarization,
        orthogonality_tol=orthogonality_tol,
    )

    factor = plane_wave_complex_factor(
        positions_m=positions_m,
        k_hat=k_hat,
        wavenumber_rad_m=wavenumber_rad_m,
        phase0=phase0,
        sign=1,
    )

    # Compute component product
    # dir_vec shape is (Ndir, 3) -> reshape to (1, Ndir, 1, 3)
    # factor shape is (N, Ndir, Nf) -> reshape to (N, Ndir, Nf, 1)
    res = amp * dir_vec[None, :, None, :] * factor[..., None]

    res = np.asarray(res, dtype=np.complex128)
    res = res.copy()
    res.setflags(write=False)
    return res
