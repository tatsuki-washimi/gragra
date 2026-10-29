"""Phase-consistent harmonic extraction for the SPECFEM Rayleigh study."""

from __future__ import annotations

import numpy as np


def snapshot_physical_times(
    time_step: np.ndarray, *, t_init_s: float, dt_s: float
) -> np.ndarray:
    """Map research-snapshot indices to physical times.

    The Rayleigh research snapshot is written after the first Newmark
    predictor, so normalized ``time_step=0`` represents ``t_init + dt``.
    This helper is intentionally analysis-specific; it does not alter the
    standard SPECFEM reader time axis.
    """
    steps = np.asarray(time_step, dtype=np.float64)
    if steps.ndim != 1 or not np.isfinite(steps).all():
        raise ValueError("time_step must be a finite one-dimensional array")
    t_init = float(t_init_s)
    dt = float(dt_s)
    if not np.isfinite(t_init) or not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("t_init_s must be finite and dt_s must be positive and finite")
    output = t_init + (steps + 1.0) * dt
    output.setflags(write=False)
    return output


def fit_complex_amplitude(values: np.ndarray, times_s: np.ndarray, omega_rad_s: float, *, t_ref_s: float = 0.0) -> np.ndarray:
    """Fit ``values(t) = Re[C exp(-i omega (t-t_ref))]`` using all samples."""
    y = np.asarray(values, dtype=np.float64)
    t = np.asarray(times_s, dtype=np.float64)
    if y.ndim < 1 or y.shape[0] != t.size:
        raise ValueError("values axis 0 must have the same length as times_s")
    if t.size < 3 or not np.isfinite(y).all() or not np.isfinite(t).all():
        raise ValueError("values and times_s must be finite with at least 3 samples")
    omega = float(omega_rad_s)
    if not np.isfinite(omega) or omega <= 0.0:
        raise ValueError("omega_rad_s must be positive and finite")
    phase = omega * (t - float(t_ref_s))
    design = np.stack((np.cos(phase), np.sin(phase)), axis=1)
    if np.linalg.matrix_rank(design) < 2:
        raise ValueError("time samples do not identify sine and cosine amplitudes")
    coefficients, *_ = np.linalg.lstsq(design, y.reshape(t.size, -1), rcond=None)
    return np.asarray((coefficients[0] + 1j * coefficients[1]).reshape(y.shape[1:]), dtype=np.complex128)


def complex_relative_error(numerical: np.ndarray, reference: np.ndarray, *, floor: float = 1e-30) -> float:
    """Return a zero-safe relative 2-norm using one reference scale."""
    num = np.asarray(numerical, dtype=np.complex128)
    ref = np.asarray(reference, dtype=np.complex128)
    if num.shape != ref.shape:
        raise ValueError("numerical and reference must have the same shape")
    return float(np.linalg.norm((num - ref).ravel()) / max(float(np.linalg.norm(ref.ravel())), floor))
