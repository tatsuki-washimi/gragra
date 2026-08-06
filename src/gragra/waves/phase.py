"""Wave phase utilities."""

import numpy as np

from gragra._arrays import as_xyz_table
from gragra.waves.directions import normalize_directions


def plane_wave_phase(
    positions_m: list | tuple | np.ndarray,
    k_hat: list | tuple | np.ndarray,
    wavenumber_rad_m: float | int | list | tuple | np.ndarray,
    phase0: float | int = 0.0,
) -> np.ndarray:
    """Calculate plane wave phase: k * (k_hat . position) + phase0.

    Output shape is (N, Ndir, Nf), squeeze is disabled.
    """
    positions = as_xyz_table("positions_m", positions_m)
    k_norm = normalize_directions(k_hat)

    # Validate wavenumber
    k_val = np.asarray(wavenumber_rad_m)
    if k_val.dtype.kind == "b" or isinstance(wavenumber_rad_m, bool):
        raise TypeError("wavenumber_rad_m must be numeric, not bool")

    try:
        k_val = np.asarray(wavenumber_rad_m, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise TypeError(f"wavenumber_rad_m must be numeric: {e}") from e

    if k_val.ndim == 0:
        k_val = k_val[np.newaxis]
    elif k_val.ndim == 1:
        pass
    else:
        raise ValueError(f"wavenumber_rad_m must be 0D or 1D, got shape {k_val.shape}")

    if not np.isfinite(k_val).all():
        raise ValueError("wavenumber_rad_m must contain only finite numbers")

    if (k_val < 0.0).any():
        raise ValueError("wavenumber_rad_m must be non-negative")

    # dot product: (N, Ndir)
    dot_val = np.dot(positions, k_norm.T)

    # phase: (N, Ndir, Nf)
    phase = dot_val[:, :, np.newaxis] * k_val[np.newaxis, np.newaxis, :] + phase0

    phase = np.asarray(phase, dtype=np.float64)
    phase = phase.copy()
    phase.setflags(write=False)
    return phase


def plane_wave_complex_factor(
    positions_m: list | tuple | np.ndarray,
    k_hat: list | tuple | np.ndarray,
    wavenumber_rad_m: float | int | list | tuple | np.ndarray,
    phase0: float | int = 0.0,
    *,
    sign: int = 1,
) -> np.ndarray:
    """Calculate plane wave complex factor: exp(i * sign * phase).

    Complex outputs represent linear complex amplitudes or transfer-function-like
    responses, not directly real-valued time-domain gravitational fields.

    Output shape is (N, Ndir, Nf) and complex128, squeeze is disabled.
    """
    if sign not in (1, -1):
        raise ValueError(f"sign must be +1 or -1, got {sign}")

    phase = plane_wave_phase(positions_m, k_hat, wavenumber_rad_m, phase0)
    factor = np.exp(1j * sign * phase)

    factor = np.asarray(factor, dtype=np.complex128)
    factor = factor.copy()
    factor.setflags(write=False)
    return factor


def evanescent_complex_factor(
    positions_m: list | tuple | np.ndarray,
    k_hat_horizontal: list | tuple | np.ndarray,
    wavenumber_rad_m: float | int | list | tuple | np.ndarray,
    decay_rad_m: float | int,
    *,
    phase0: float | int = 0.0,
    sign: int = 1,
) -> np.ndarray:
    """Calculate evanescent wave complex factor.

    e^{i * sign * (k * (k_hat_h . r_xy) + phase0)} * e^{-decay * h}
    where h = -z >= 0.

    Complex outputs represent linear complex amplitudes or transfer-function-like
    responses, not directly real-valued time-domain gravitational fields.

    Output shape is (N, Ndir, Nf) and complex128, squeeze is disabled.

    Positions above surface (z > 0) are rejected (ValueError).
    Negative decay values are rejected (ValueError).
    Non-horizontal propagation directions are rejected (ValueError).
    """
    if sign not in (1, -1):
        raise ValueError(f"sign must be +1 or -1, got {sign}")

    # Validate decay_rad_m
    decay_arr = np.asarray(decay_rad_m)
    if decay_arr.dtype.kind == "b" or isinstance(decay_rad_m, bool):
        raise TypeError("decay_rad_m must be numeric, not bool")

    try:
        decay_val = float(decay_arr)
    except (ValueError, TypeError) as e:
        raise TypeError(f"decay_rad_m must be numeric: {e}") from e

    if not np.isfinite(decay_val):
        raise ValueError("decay_rad_m must contain only finite numbers")
    if decay_val < 0.0:
        raise ValueError("decay_rad_m must be non-negative")

    # Validate positions and overflow guard
    positions = as_xyz_table("positions_m", positions_m)
    if (positions[:, 2] > 0.0).any():
        raise ValueError("z > 0 is rejected (overflow guard)")

    # Validate propagation direction (horizontal)
    k_norm = normalize_directions(k_hat_horizontal)
    if not np.allclose(k_norm[:, 2], 0.0, atol=1e-12):
        raise ValueError(
            "k_hat_horizontal must be horizontal (z-component must be zero)"
        )

    # Convert positions to horizontal for phase calculation
    pos_h = positions.copy()
    pos_h[:, 2] = 0.0

    # Calculate horizontal phase
    phase = plane_wave_phase(pos_h, k_norm, wavenumber_rad_m, phase0)

    # Calculate depth h = -z >= 0
    h = -positions[:, 2]

    # Calculate complex factor
    decay_term = np.exp(-decay_val * h[:, np.newaxis, np.newaxis])
    factor = np.exp(1j * sign * phase) * decay_term

    factor = np.asarray(factor, dtype=np.complex128)
    factor = factor.copy()
    factor.setflags(write=False)
    return factor
