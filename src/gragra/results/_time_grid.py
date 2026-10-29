"""Provisional uniform GPS grid validation (M3-B L1-a rev6).

Internal module: tolerance policy is fixed here until G-最終; not a public knob.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from gragra.results._coords import normalize_gps

_EPS = np.finfo(float).eps


@dataclass(frozen=True, slots=True)
class UniformTimeGridPolicy:
    """Immutable L1-a tolerance policy (replaceable at implementation layer only)."""

    kappa_relative_dt: float = 0.01

    def __post_init__(self) -> None:
        k = float(self.kappa_relative_dt)
        if not np.isfinite(k) or k <= 0.0:
            raise ValueError("kappa_relative_dt must be positive and finite")


DEFAULT_TIME_GRID_POLICY = UniformTimeGridPolicy()


def gps_total_ns(epoch_gps: tuple[int, int]) -> int:
    sec, nsec = epoch_gps
    return int(sec) * 1_000_000_000 + int(nsec)


def ns_to_float64_seconds(total_ns: int) -> float:
    sec, nsec = divmod(int(total_ns), 1_000_000_000)
    return float(sec) + float(nsec) * 1e-9


def expected_sample_seconds(
    epoch_gps: tuple[int, int], dt_seconds: float, k: int
) -> float:
    t0 = ns_to_float64_seconds(gps_total_ns(epoch_gps))
    return t0 + float(k) * float(dt_seconds)


def build_uniform_time_s(
    epoch_gps: tuple[int, int], dt_seconds: float, count: int
) -> np.ndarray:
    return np.array(
        [expected_sample_seconds(epoch_gps, dt_seconds, k) for k in range(count)],
        dtype=np.float64,
    )


def float64_ulp(x: float) -> float:
    x = float(x)
    if x == 0.0:
        return _EPS
    return abs(math.ulp(x))


def tolerance_rep_seconds(a: float, b: float) -> float:
    return 2.0 * max(float64_ulp(a), float64_ulp(b), _EPS)


def tolerance_sample_seconds(
    a: float, b: float, *, dt_seconds: float, policy: UniformTimeGridPolicy
) -> float:
    ulp_tol = tolerance_rep_seconds(a, b)
    rel_cap = policy.kappa_relative_dt * abs(float(dt_seconds))
    if ulp_tol > rel_cap:
        return rel_cap
    return ulp_tol


def validate_representable_uniform_grid(
    sampling_epoch_gps: tuple[int, int],
    dt_seconds: float,
    count: int,
) -> None:
    """Reject sampling that cannot be a strictly increasing float64 uniform grid."""
    if count <= 1:
        return
    canonical = build_uniform_time_s(sampling_epoch_gps, dt_seconds, count)
    if not np.all(np.isfinite(canonical)):
        raise ValueError("uniform time grid is non-finite under float64")
    if np.any(np.diff(canonical) <= 0.0):
        raise ValueError(
            "TimeSampling dt is too small to form a strictly increasing float64 "
            "uniform grid at the given GPS epoch; adjust epoch or dt"
        )


def validate_time_s_against_sampling(
    time_s: np.ndarray,
    *,
    epoch_gps: tuple[int, int],
    dt_seconds: float,
    count: int,
    policy: UniformTimeGridPolicy = DEFAULT_TIME_GRID_POLICY,
) -> None:
    """Track B: external float64 samples vs canonical TimeSampling (rev6)."""
    epoch_gps = normalize_gps(epoch_gps)
    t = np.asarray(time_s, dtype=np.float64).ravel()
    if len(t) != count:
        raise ValueError(
            f"time_s length {len(t)} does not match TimeSampling count {count}"
        )
    if count < 1 or not np.isfinite(dt_seconds) or dt_seconds <= 0.0:
        raise ValueError("invalid TimeSampling dt or count")
    if not np.all(np.isfinite(t)):
        raise ValueError("time_s must contain only finite values")

    if count == 1:
        exp_s = ns_to_float64_seconds(gps_total_ns(epoch_gps))
        err = abs(float(t[0]) - exp_s)
        if err > tolerance_rep_seconds(exp_s, float(t[0])):
            raise ValueError("time_s[0] is inconsistent with TimeSampling epoch")
        return

    if np.any(np.diff(t) <= 0.0):
        raise ValueError("time_s must be strictly increasing")

    for ki in range(count):
        exp_s = expected_sample_seconds(epoch_gps, dt_seconds, ki)
        obs = float(t[ki])
        err = abs(obs - exp_s)
        tol = tolerance_sample_seconds(
            exp_s, obs, dt_seconds=dt_seconds, policy=policy
        )
        if err > tol:
            raise ValueError(
                f"time_s sample {ki} deviates from uniform TimeSampling grid "
                f"(|Δ|={err:.3g} s > tolerance {tol:.3g} s)"
            )


__all__ = [
    "DEFAULT_TIME_GRID_POLICY",
    "UniformTimeGridPolicy",
    "build_uniform_time_s",
    "expected_sample_seconds",
    "validate_representable_uniform_grid",
    "validate_time_s_against_sampling",
]
