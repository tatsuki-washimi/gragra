"""Analytic fixtures for the continuous-time contract.

Each scalar signal ``s(t)`` has a closed form for two quantities:

* the periodic Fourier coefficient ``C_n = (1/T) ∫ q(t) exp(-2πi n (Δ+t)/T) dt``
  over one period, with ``Δ = epoch - tref`` in seconds and ``t`` relative to
  the epoch;
* the finite-window transform ``X(f) = ∫_{t0}^{t1} q(t) exp(-2πi f (Δ+t)) dt``.

The response callable is time-first and returns SI Quantities:
potential ``(Ntime, Ntarget)`` in m²/s², acceleration ``(Ntime, Ntarget, 3)`` in
m/s², gradient ``(Ntime, Ntarget, 3, 3)`` in 1/s². Each target has its own
scale, and each component has a fixed pattern, so the axis order is exercised.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import mpmath as mp
import numpy as np
from astropy import units as u
from scipy.spatial.transform import Rotation

from gragra._continuous_contract import (
    normalize_relative_times,
    validate_observable,
)

TARGET_SCALES = np.array([1.0, -0.5, 2.0], dtype=np.float64)
"""Per-target scale. Its length is Ntarget = 3."""

ACCELERATION_PATTERN = np.array([0.6, -0.8, 0.0], dtype=np.float64)
"""Fixed component pattern for acceleration (3,)."""

GRADIENT_PATTERN = np.array(
    [[1.0, 0.5, 0.0], [0.0, -2.0, 0.25], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
"""Fixed component pattern for gradient (3, 3)."""

OBSERVABLE_UNITS = {
    "potential": u.m**2 / u.s**2,
    "acceleration": u.m / u.s**2,
    "gradient": 1 / u.s**2,
}


@dataclass(frozen=True)
class ScalarSignal:
    """Real scalar signal ``s(t)`` with ``t`` in seconds from the epoch."""

    name: str
    evaluate: Callable[[np.ndarray], np.ndarray]


def expand_components(scalar: np.ndarray, observable: str) -> np.ndarray:
    """Expand a leading-axis scalar array to ``(lead, Ntarget, *component)``.

    ``scalar`` has shape ``(lead,)``. Each target has its scale, and each
    component has its fixed pattern. Potential is ``(lead, Ntarget)``,
    acceleration ``(lead, Ntarget, 3)``, and gradient ``(lead, Ntarget, 3, 3)``.
    """
    observable = validate_observable(observable)
    lead = np.asarray(scalar)
    if observable == "potential":
        return lead[:, None] * TARGET_SCALES[None, :]
    if observable == "acceleration":
        return (
            lead[:, None, None]
            * TARGET_SCALES[None, :, None]
            * ACCELERATION_PATTERN[None, None, :]
        )
    return (
        lead[:, None, None, None]
        * TARGET_SCALES[None, :, None, None]
        * GRADIENT_PATTERN[None, None, :, :]
    )


def make_response(signal: ScalarSignal) -> Callable[[u.Quantity, str], u.Quantity]:
    """Return a ``ResponseFunction`` that evaluates ``signal`` time-first."""

    def response(times: u.Quantity, observable: str) -> u.Quantity:
        name = validate_observable(observable)
        seconds = normalize_relative_times(times).to_value(u.s)
        base = np.asarray(signal.evaluate(seconds), dtype=np.float64)
        return expand_components(base, name) * OBSERVABLE_UNITS[name]

    return response


def constant(value: float) -> ScalarSignal:
    """Constant signal ``s(t) = value``."""
    return ScalarSignal(
        name=f"constant({value})",
        evaluate=lambda t: np.full(t.shape, float(value)),
    )


def cosine(amplitude: float, phase: float, frequency_hz: float) -> ScalarSignal:
    """Cosine ``s(t) = a cos(2π f t + φ)``; periodic when ``f T`` is an integer."""
    return ScalarSignal(
        name=f"cosine(a={amplitude}, phi={phase}, f={frequency_hz})",
        evaluate=lambda t: amplitude * np.cos(2.0 * np.pi * frequency_hz * t + phase),
    )


def gaussian(amplitude: float, centre_s: float, sigma_s: float) -> ScalarSignal:
    """Gaussian pulse ``s(t) = A exp(-(t - t_c)² / (2σ²))``."""
    return ScalarSignal(
        name=f"gaussian(A={amplitude}, tc={centre_s}, sigma={sigma_s})",
        evaluate=lambda t: (
            amplitude * np.exp(-((t - centre_s) ** 2) / (2.0 * sigma_s**2))
        ),
    )


def _mp_integral_exponential(nu: mp.mpf, t0: mp.mpf, t1: mp.mpf) -> mp.mpc:
    """Return ``∫_{t0}^{t1} exp(-2πi ν t) dt`` in closed form (mpmath)."""
    if nu == 0:
        return mp.mpc(t1 - t0)
    return (mp.exp(-2j * mp.pi * nu * t1) - mp.exp(-2j * mp.pi * nu * t0)) / (
        -2j * mp.pi * nu
    )


def window_transform_closed_form(
    signal: ScalarSignal,
    *,
    delta_s: float,
    t0: float,
    t1: float,
    frequencies_hz: np.ndarray,
) -> np.ndarray:
    """Closed-form ``X(f) = ∫_{t0}^{t1} s(t) exp(-2πi f (Δ+t)) dt``.

    Evaluated in mpmath at 50 digits and returned as complex128. Supported
    families: ``constant``, ``cosine``, and ``gaussian``. The Gaussian uses the
    full line, so its window must contain the pulse.
    """
    freqs = np.asarray(frequencies_hz, dtype=np.float64)
    kind = signal.name.split("(", 1)[0]
    params = _parse_params(signal.name)
    out = np.empty(freqs.shape, dtype=np.complex128)
    with mp.workdps(50):
        a0, a1 = mp.mpf(t0), mp.mpf(t1)
        d = mp.mpf(delta_s)
        for i, f_value in enumerate(freqs):
            f = mp.mpf(float(f_value))
            phase = mp.exp(-2j * mp.pi * f * d)
            if kind == "constant":
                value = (
                    mp.mpf(params["value"])
                    * phase
                    * _mp_integral_exponential(f, a0, a1)
                )
            elif kind == "cosine":
                amp, phi, f0 = (
                    mp.mpf(params["a"]),
                    mp.mpf(params["phi"]),
                    mp.mpf(params["f"]),
                )
                value = (
                    phase
                    * (amp / 2)
                    * (
                        mp.exp(1j * phi) * _mp_integral_exponential(f - f0, a0, a1)
                        + mp.exp(-1j * phi) * _mp_integral_exponential(f + f0, a0, a1)
                    )
                )
            elif kind == "gaussian":
                amp, centre, sigma = (
                    mp.mpf(params["A"]),
                    mp.mpf(params["tc"]),
                    mp.mpf(params["sigma"]),
                )
                value = (
                    amp
                    * sigma
                    * mp.sqrt(2 * mp.pi)
                    * mp.exp(-2 * mp.pi**2 * f**2 * sigma**2)
                    * mp.exp(-2j * mp.pi * f * centre)
                    * phase
                )
            else:
                raise ValueError(f"no closed form for {signal.name}")
            out[i] = complex(value)
    return out


def periodic_coefficient_closed_form(
    signal: ScalarSignal,
    *,
    period_s: float,
    delta_s: float,
    orders: tuple[int, ...],
) -> np.ndarray:
    """Closed-form ``C_n`` for ``n`` in ``orders`` (``n >= 0``).

    ``C_n = (1/T) ∫ q(t) exp(-2πi n (Δ+t)/T) dt`` over one period. Supported
    families are ``constant`` (``C_0 = c``) and ``cosine`` with ``f T`` an
    integer ``h`` (``C_h = (a/2) e^{iφ} e^{-2πi h Δ/T}``).
    """
    kind = signal.name.split("(", 1)[0]
    params = _parse_params(signal.name)
    result = np.zeros(len(orders), dtype=np.complex128)
    for i, n in enumerate(orders):
        if kind == "constant":
            if n == 0:
                result[i] = params["value"]
        elif kind == "cosine":
            harmonic = params["f"] * period_s
            if abs(harmonic - round(harmonic)) > 1e-12:
                raise ValueError("cosine fixture is not periodic on this period")
            if round(harmonic) < 1:
                raise ValueError("periodic cosine closed form needs harmonic >= 1")
            if n == round(harmonic):
                phase = complex(math.cos(params["phi"]), math.sin(params["phi"]))
                result[i] = (
                    0.5
                    * params["a"]
                    * phase
                    * complex(
                        math.cos(-2 * math.pi * n * delta_s / period_s),
                        math.sin(-2 * math.pi * n * delta_s / period_s),
                    )
                )
        else:
            raise ValueError(f"no periodic closed form for {signal.name}")
    return result


def _parse_params(name: str) -> dict[str, float]:
    """Parse ``kind(key=value, ...)`` names built by this module."""
    inner = name[name.index("(") + 1 : name.rindex(")")]
    out: dict[str, float] = {}
    for item in inner.split(","):
        item = item.strip()
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        out[key.strip()] = float(value)
    if "value" not in out and name.startswith("constant("):
        out["value"] = float(inner)
    return out


# Analytic pose fixtures. Each callable has the PoseFunction signature.


def static_pose() -> Callable[[u.Quantity], tuple[Rotation, u.Quantity]]:
    """Identity rotation and zero translation."""

    def pose(times: u.Quantity) -> tuple[Rotation, u.Quantity]:
        t = normalize_relative_times(times).to_value(u.s)
        return Rotation.identity(len(t)), np.zeros((len(t), 3)) * u.m

    return pose


def constant_velocity_pose(
    velocity_m_s: tuple[float, float, float] = (1.0, 0.0, 0.0),
    origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Callable[[u.Quantity], tuple[Rotation, u.Quantity]]:
    """Identity rotation with ``c(t) = origin + v t``."""
    velocity = np.asarray(velocity_m_s, dtype=np.float64)
    origin = np.asarray(origin_m, dtype=np.float64)

    def pose(times: u.Quantity) -> tuple[Rotation, u.Quantity]:
        t = normalize_relative_times(times).to_value(u.s)
        translation = origin[None, :] + t[:, None] * velocity[None, :]
        return Rotation.identity(len(t)), translation * u.m

    return pose


def constant_rotation_pose(
    omega_rad_s: tuple[float, float, float] = (0.0, 0.0, 2.0 * np.pi),
) -> Callable[[u.Quantity], tuple[Rotation, u.Quantity]]:
    """Constant angular velocity about a fixed axis, with zero translation."""
    omega = np.asarray(omega_rad_s, dtype=np.float64)

    def pose(times: u.Quantity) -> tuple[Rotation, u.Quantity]:
        t = normalize_relative_times(times).to_value(u.s)
        rotations = Rotation.from_rotvec(t[:, None] * omega[None, :])
        return rotations, np.zeros((len(t), 3)) * u.m

    return pose


def circular_pose() -> Callable[[u.Quantity], tuple[Rotation, u.Quantity]]:
    """Circular translation with radius 1 m and period 1 s, identity rotation."""

    def pose(times: u.Quantity) -> tuple[Rotation, u.Quantity]:
        t = normalize_relative_times(times).to_value(u.s)
        angle = 2.0 * np.pi * t
        translation = np.stack(
            [np.cos(angle), np.sin(angle), np.zeros_like(angle)], axis=1
        )
        return Rotation.identity(len(t)), translation * u.m

    return pose
