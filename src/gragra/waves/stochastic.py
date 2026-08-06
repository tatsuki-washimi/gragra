from dataclasses import dataclass
from typing import Literal

import numpy as np

from gragra._arrays import as_xyz_table, as_xyz_vector
from gragra.array_types import FloatArray
from gragra.waves.displacement import plane_wave_displacement


def _normalize_vector_stably(value: np.ndarray, name: str) -> np.ndarray:
    """Normalize one finite vector without norm overflow or underflow."""
    scale = float(np.max(np.abs(value)))
    if scale == 0.0:
        raise ValueError(f"{name} must not be a zero vector")

    with np.errstate(over="ignore", invalid="ignore"):
        norm = float(np.linalg.norm(value))
    if np.isfinite(norm) and norm > 0.0:
        return value / norm

    scaled = value / scale
    return scaled / np.linalg.norm(scaled)


def _normalize_rows_stably(value: np.ndarray, name: str) -> np.ndarray:
    """Normalize finite row vectors, using scaled norms only where needed."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        norms = np.linalg.norm(value, axis=1, keepdims=True)
        normalized = value / norms

    zero_rows = ~np.any(value != 0.0, axis=1)
    if np.any(zero_rows):
        raise ValueError(f"{name} must not contain zero vectors")

    unstable_rows = ~np.isfinite(norms[:, 0]) | (norms[:, 0] == 0.0)
    if np.any(unstable_rows):
        selected = value[unstable_rows]
        scales = np.max(np.abs(selected), axis=1, keepdims=True)
        scaled = selected / scales
        normalized[unstable_rows] = scaled / np.linalg.norm(
            scaled, axis=1, keepdims=True
        )

    return normalized


@dataclass(frozen=True)
class StochasticPSParameters:
    k_hat: FloatArray
    vertical_unit: FloatArray
    wave_type_p: Literal["P"]
    wave_type_s: Literal["S1", "S2", "SH", "SV"]
    amplitude_p_m: float
    amplitude_s_m: float
    wavenumber_p_rad_m: float
    wavenumber_s_rad_m: float
    phase0_p: FloatArray
    phase0_s: FloatArray

    def __post_init__(self):
        # Validate wave_types
        if self.wave_type_p != "P":
            raise ValueError("wave_type_p must be 'P'")
        if self.wave_type_s not in ("S1", "S2", "SH", "SV"):
            raise ValueError("wave_type_s must be one of 'S1', 'S2', 'SH', 'SV'")

        # Validate and canonicalize floats
        for name in (
            "amplitude_p_m",
            "amplitude_s_m",
            "wavenumber_p_rad_m",
            "wavenumber_s_rad_m",
        ):
            val = getattr(self, name)
            if isinstance(val, bool):
                raise TypeError(f"{name} must not be bool")
            if not isinstance(val, (int, float, np.integer, np.floating)):
                raise TypeError(f"{name} must be a real number")
            if not np.isfinite(val):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, float(val))

        if self.wavenumber_p_rad_m <= 0.0:
            raise ValueError("wavenumber_p_rad_m must be positive")
        if self.wavenumber_s_rad_m <= 0.0:
            raise ValueError("wavenumber_s_rad_m must be positive")

        # Validate and canonicalize arrays
        # vertical_unit
        v = np.asarray(self.vertical_unit, dtype=np.float64)
        if v.shape != (3,):
            raise ValueError(f"vertical_unit must have shape (3,), got {v.shape}")
        if not np.isfinite(v).all():
            raise ValueError("vertical_unit must contain only finite numbers")
        v = _normalize_vector_stably(v, "vertical_unit")
        v.setflags(write=False)
        object.__setattr__(self, "vertical_unit", v)

        # k_hat
        k = np.asarray(self.k_hat, dtype=np.float64)
        if k.ndim != 2 or k.shape[1] != 3:
            raise ValueError(f"k_hat must have shape (R, 3), got {k.shape}")
        if not np.isfinite(k).all():
            raise ValueError("k_hat must contain only finite numbers")
        k = _normalize_rows_stably(k, "k_hat")
        k.setflags(write=False)
        object.__setattr__(self, "k_hat", k)

        # phase0_p
        p_p = np.asarray(self.phase0_p, dtype=np.float64)
        if p_p.ndim != 1 or p_p.shape[0] != k.shape[0]:
            raise ValueError(
                f"phase0_p must be a 1D array of length R, got shape {p_p.shape}"
            )
        if not np.isfinite(p_p).all():
            raise ValueError("phase0_p must contain only finite numbers")
        p_p = p_p.copy()
        p_p.setflags(write=False)
        object.__setattr__(self, "phase0_p", p_p)

        # phase0_s
        p_s = np.asarray(self.phase0_s, dtype=np.float64)
        if p_s.ndim != 1 or p_s.shape[0] != k.shape[0]:
            raise ValueError(
                f"phase0_s must be a 1D array of length R, got shape {p_s.shape}"
            )
        if not np.isfinite(p_s).all():
            raise ValueError("phase0_s must contain only finite numbers")
        p_s = p_s.copy()
        p_s.setflags(write=False)
        object.__setattr__(self, "phase0_s", p_s)


def stochastic_ps_realization(
    rng: np.random.Generator,
    n_realizations: int,
    positions_m,
    wavenumber_p_rad_m: float,
    velocity_ratio_s_over_p: float,
    s_wave_type: Literal["S1", "S2", "SH", "SV"] = "SV",
    *,
    amplitude_p_m: float = 1.0,
    amplitude_s_m: float = 1.0,
    vertical_unit=(0.0, 0.0, 1.0),
    fixed_direction=None,
    correlated_phase: bool = True,
) -> tuple[np.ndarray, np.ndarray, StochasticPSParameters]:
    """Generate a stochastic P/S correlation-preserving plane wave ensemble realization.

    This function generates stochastic realizations of coupled P-wave and S-wave
    displacement fields. The P/S correlation is modeled by sharing a common propagation
    direction k_hat (either isotropically drawn or fixed) and phase0 (if
    correlated_phase is True).

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator used to draw stochastic directions and phases.
    n_realizations : int
        Number of realizations (R) to generate. Must be a positive integer.
    positions_m : array_like
        Target positions where displacement is evaluated, shape (N, 3).
    wavenumber_p_rad_m : float
        P-wave wavenumber. Must be positive.
    velocity_ratio_s_over_p : float
        Ratio of S-wave to P-wave velocity (C_s/C_p). Must satisfy 0 < ratio < 1.
    s_wave_type : Literal["S1", "S2", "SH", "SV"], optional
        S-wave polarization type. Default is "SV".
    amplitude_p_m : float, optional
        P-wave amplitude. Default is 1.0.
    amplitude_s_m : float, optional
        S-wave amplitude. Default is 1.0.
    vertical_unit : array_like, optional
        Vertical unit vector defining SV/SH orientations, shape (3,).
        Default is (0.0, 0.0, 1.0).
    fixed_direction : array_like or None, optional
        Fixed propagation direction, shape (3,). If None, directions are drawn
        isotropically. Default is None.
    correlated_phase : bool, optional
        If True, P and S waves share the same phase0_p = phase0_s (physical model).
        If False, phase0_p and phase0_s are drawn independently.
        Default is True.

    Returns
    -------
    u_p : np.ndarray
        P-wave displacement field array of shape (N, R, 3) and dtype complex128.
        Read-only.
    u_s : np.ndarray
        S-wave displacement field array of shape (N, R, 3) and dtype complex128.
        Read-only.
    drawn_params : StochasticPSParameters
        A frozen typed object containing all generated parameters.

    Raises
    ------
    TypeError
        If rng is not a np.random.Generator, or if float/integer parameters
        are boolean or not numeric.
    ValueError
        If shapes are invalid, non-finite values are encountered,
        n_realizations <= 0, wavenumber_p_rad_m <= 0, velocity_ratio_s_over_p is
        outside (0, 1), or a zero vector is supplied for direction.

    Notes
    -----
    - This function is placed in `gragra.waves` because it represents a purely kinematic
      quantity and is independent of the gravitational constant G (G-free).
    - Complex outputs represent linear complex amplitudes or transfer-function-like
      responses, not directly real-valued time-domain fields.
    - When coupling this displacement with source mass weights, the source and witness
      must share the exact same propagation parameters (k_hat, wavenumber, phase0).
    - This function uses sign=+1 internally, preventing silent phase conjugation.
    - "P/S correlation" refers to the correlation between P and S waves propagating in
      the same direction, which is orthogonal to the "direction-uncorrelated" models.
    - correlated_phase=False is for uncorrelated comparison cases only.
    """
    # 1. Validate inputs before RNG consumption
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator instance")

    if isinstance(n_realizations, bool):
        raise TypeError("n_realizations must not be bool")
    if not isinstance(n_realizations, (int, np.integer)):
        raise TypeError("n_realizations must be an integer")
    if n_realizations <= 0:
        raise ValueError("n_realizations must be a positive integer")

    # Validate float params
    if isinstance(wavenumber_p_rad_m, bool):
        raise TypeError("wavenumber_p_rad_m must not be bool")
    if not isinstance(wavenumber_p_rad_m, (int, float, np.integer, np.floating)):
        raise TypeError("wavenumber_p_rad_m must be a real number")
    w_p = float(wavenumber_p_rad_m)
    if not np.isfinite(w_p):
        raise ValueError("wavenumber_p_rad_m must be finite")
    if w_p <= 0.0:
        raise ValueError("wavenumber_p_rad_m must be positive")

    if isinstance(velocity_ratio_s_over_p, bool):
        raise TypeError("velocity_ratio_s_over_p must not be bool")
    if not isinstance(velocity_ratio_s_over_p, (int, float, np.integer, np.floating)):
        raise TypeError("velocity_ratio_s_over_p must be a real number")
    ratio = float(velocity_ratio_s_over_p)
    if not np.isfinite(ratio):
        raise ValueError("velocity_ratio_s_over_p must be finite")
    if ratio <= 0.0 or ratio >= 1.0:
        raise ValueError("velocity_ratio_s_over_p must be in interval (0, 1)")

    # Derived wavenumber
    w_s = w_p / ratio
    if not np.isfinite(w_s):
        raise ValueError("wavenumber_s_rad_m must be finite")

    if s_wave_type not in ("S1", "S2", "SH", "SV"):
        raise ValueError("s_wave_type must be one of 'S1', 'S2', 'SH', 'SV'")

    for amp_val, name in [
        (amplitude_p_m, "amplitude_p_m"),
        (amplitude_s_m, "amplitude_s_m"),
    ]:
        if isinstance(amp_val, bool):
            raise TypeError(f"{name} must not be bool")
        if not isinstance(amp_val, (int, float, np.integer, np.floating)):
            raise TypeError(f"{name} must be a real number")
        if not np.isfinite(float(amp_val)):
            raise ValueError(f"{name} must be finite")
    amp_p = float(amplitude_p_m)
    amp_s = float(amplitude_s_m)

    if not isinstance(correlated_phase, (bool, np.bool_)):
        raise TypeError("correlated_phase must be a boolean")

    # Validate directions & units
    v_unit = as_xyz_vector("vertical_unit", vertical_unit)
    v_unit = _normalize_vector_stably(v_unit, "vertical_unit")

    if fixed_direction is not None:
        f_dir = as_xyz_vector("fixed_direction", fixed_direction)
        f_dir = _normalize_vector_stably(f_dir, "fixed_direction")
    else:
        f_dir = None

    pos = as_xyz_table("positions_m", positions_m)

    # 2. Consume RNG in strict order
    if f_dir is None:
        raw = rng.standard_normal((n_realizations, 3))
        k_hat = _normalize_rows_stably(raw, "drawn k_hat")
    else:
        k_hat = np.tile(f_dir, (n_realizations, 1))

    phase0_p = rng.uniform(0.0, 2.0 * np.pi, size=n_realizations)

    if correlated_phase:
        phase0_s = phase0_p.copy()
    else:
        phase0_s = rng.uniform(0.0, 2.0 * np.pi, size=n_realizations)

    # 3. Construct parameters object (this does copy, normalization and sets read-only)
    drawn_params = StochasticPSParameters(
        k_hat=k_hat,
        vertical_unit=v_unit,
        wave_type_p="P",
        wave_type_s=s_wave_type,
        amplitude_p_m=amp_p,
        amplitude_s_m=amp_s,
        wavenumber_p_rad_m=w_p,
        wavenumber_s_rad_m=w_s,
        phase0_p=phase0_p,
        phase0_s=phase0_s,
    )

    # 4. Generate displacement fields
    u_p_raw = plane_wave_displacement(
        positions_m=pos,
        k_hat=drawn_params.k_hat,
        wavenumber_rad_m=drawn_params.wavenumber_p_rad_m,
        wave_type="P",
        amplitude_m=drawn_params.amplitude_p_m,
        phase0=drawn_params.phase0_p[:, np.newaxis],
        vertical_unit=drawn_params.vertical_unit,
    )
    u_p = u_p_raw[:, :, 0, :]

    u_s_raw = plane_wave_displacement(
        positions_m=pos,
        k_hat=drawn_params.k_hat,
        wavenumber_rad_m=drawn_params.wavenumber_s_rad_m,
        wave_type=drawn_params.wave_type_s,
        amplitude_m=drawn_params.amplitude_s_m,
        phase0=drawn_params.phase0_s[:, np.newaxis],
        vertical_unit=drawn_params.vertical_unit,
    )
    u_s = u_s_raw[:, :, 0, :]

    # Return defensive copy of u_p and u_s as read-only complex128
    u_p = np.asarray(u_p, dtype=np.complex128).copy()
    u_p.setflags(write=False)

    u_s = np.asarray(u_s, dtype=np.complex128).copy()
    u_s.setflags(write=False)

    return u_p, u_s, drawn_params
