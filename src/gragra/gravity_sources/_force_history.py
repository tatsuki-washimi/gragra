"""Immutable, causal force histories for analytic source models."""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass

import numpy as np
from astropy import units as u
from gwpy.time import LIGOTimeGPS

from gragra._arrays import _check_no_bool
from gragra.results._coords import gps_offset_seconds, normalize_gps


def _quantity_array_si(name: str, value: object, unit: u.UnitBase) -> np.ndarray:
    if not isinstance(value, u.Quantity):
        raise TypeError(f"{name} must be an Astropy Quantity")
    _check_no_bool(value, name)
    raw = np.asarray(value.to_value(unit))
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain only real values")
    try:
        return np.asarray(raw, dtype=np.float64).copy()
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must contain real numeric values") from error


def _finite_fsum(name: str, terms: object) -> float:
    try:
        values = [float(term) for term in terms]
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} must be finite")
    try:
        result = math.fsum(values)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _shifted_polynomial(
    coefficients: tuple[float, ...], origin: float
) -> tuple[float, ...]:
    """Return coefficients of ``P(origin + x)`` in ascending powers of x."""
    shifted = [coefficients[-1]]
    for coefficient in reversed(coefficients[:-1]):
        next_coefficients = [
            _finite_fsum(
                "ForceHistory translated coefficient",
                (coefficient, origin * shifted[0]),
            )
        ]
        next_coefficients.extend(
            _finite_fsum(
                "ForceHistory translated coefficient",
                (origin * shifted[index], shifted[index - 1]),
            )
            for index in range(1, len(shifted))
        )
        next_coefficients.append(shifted[-1])
        shifted = next_coefficients
    return tuple(shifted)


def _endpoint_data(
    coefficients: np.ndarray, width_seconds: float, local_u: float
) -> tuple[float, float, float, float]:
    powers = np.power(local_u, np.arange(len(coefficients), dtype=np.float64))
    force_terms = coefficients * powers
    derivative_indices = np.arange(1, len(coefficients), dtype=np.float64)
    derivative_powers = np.power(local_u, derivative_indices - 1.0)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        derivative_terms = (
            coefficients[1:] * derivative_indices * derivative_powers / width_seconds
        )
    force = _finite_fsum("ForceHistory endpoint value", force_terms)
    force_scale = _finite_fsum("ForceHistory endpoint force scale", np.abs(force_terms))
    derivative = _finite_fsum("ForceHistory endpoint derivative", derivative_terms)
    derivative_scale = _finite_fsum(
        "ForceHistory endpoint derivative scale", np.abs(derivative_terms)
    )
    return force, derivative, force_scale, derivative_scale


def _check_c1_join(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
    label: str,
) -> None:
    eps64 = np.finfo(np.float64).eps
    for component, left_value, right_value, left_scale, right_scale in (
        ("force", left[0], right[0], left[2], right[2]),
        ("time derivative", left[1], right[1], left[3], right[3]),
    ):
        scale = _finite_fsum(f"{label} {component} scale", (left_scale, right_scale))
        with np.errstate(over="ignore", invalid="ignore"):
            residual = abs(left_value - right_value)
            tolerance = 64.0 * eps64 * scale
        if not math.isfinite(residual) or not math.isfinite(tolerance):
            raise ValueError(f"ForceHistory {label} has a non-finite {component} join")
        if residual > tolerance:
            raise ValueError(
                f"ForceHistory must be C1 continuous at {label} ({component} mismatch)"
            )


def _real_seconds(name: str, value: object) -> float:
    _check_no_bool(value, name)
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be a real scalar number of seconds")
    result = float(raw)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _scalar_quantity_si(name: str, value: object, unit: u.UnitBase) -> float:
    if not isinstance(value, u.Quantity):
        raise TypeError(f"{name} must be an Astropy Quantity")
    _check_no_bool(value, name)
    raw = np.asarray(value.to_value(unit))
    if raw.shape != () or np.iscomplexobj(raw):
        raise ValueError(f"{name} must be a real scalar")
    result = float(raw)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, eq=False, init=False)
class ForceHistory:
    """Piecewise local-polynomial force history with causal zero before onset.

    Coefficients are in ascending powers of each interval's normalized local
    coordinate. Every interval has the same coefficient width; lower-degree
    polynomials are represented by trailing zeros. Public Quantity properties
    return copies, so caller mutation cannot change this snapshot.
    """

    _knots_seconds: tuple[float, ...]
    _coefficients_newtons: tuple[tuple[float, ...], ...]
    onset_gps: tuple[int, int]
    after: str
    _final_force_newtons: float

    def __init__(
        self,
        *,
        knots: u.Quantity,
        coefficients: u.Quantity,
        onset: object,
        after: str,
    ) -> None:
        knot_values = _quantity_array_si("knots", knots, u.s)
        coefficient_values = _quantity_array_si("coefficients", coefficients, u.N)
        if knot_values.ndim != 1 or knot_values.size < 2:
            raise ValueError("knots must have shape (K+1,) with K >= 1")
        if coefficient_values.ndim != 2 or coefficient_values.shape[1] < 1:
            raise ValueError("coefficients must have shape (K, D+1) with D >= 0")
        if coefficient_values.shape[0] != knot_values.size - 1:
            raise ValueError("coefficients must have one row per knot interval")
        if not np.isfinite(knot_values).all():
            raise ValueError("knots must contain only finite times")
        if not np.isfinite(coefficient_values).all():
            raise ValueError("coefficients must contain only finite forces")
        if knot_values[0] != 0.0:
            raise ValueError("knots must start at zero seconds")
        widths = np.diff(knot_values)
        if not np.isfinite(widths).all() or np.any(widths <= 0.0):
            raise ValueError("knots must be strictly increasing")
        if not isinstance(after, str):
            raise TypeError("after must be 'zero' or 'hold'")
        if after not in ("zero", "hold"):
            raise ValueError("after must be 'zero' or 'hold'")
        onset_gps = normalize_gps(onset)

        interval_endpoints = [
            (
                _endpoint_data(coefficient_values[index], width, 0.0),
                _endpoint_data(coefficient_values[index], width, 1.0),
            )
            for index, width in enumerate(widths)
        ]
        zero = (0.0, 0.0, 0.0, 0.0)
        _check_c1_join(zero, interval_endpoints[0][0], "onset")
        for index in range(1, len(interval_endpoints)):
            _check_c1_join(
                interval_endpoints[index - 1][1],
                interval_endpoints[index][0],
                f"knot {index}",
            )
        final_endpoint = interval_endpoints[-1][1]
        if after == "zero":
            after_endpoint = zero
        else:
            after_endpoint = (final_endpoint[0], 0.0, abs(final_endpoint[0]), 0.0)
        _check_c1_join(final_endpoint, after_endpoint, "final knot")

        object.__setattr__(
            self, "_knots_seconds", tuple(float(value) for value in knot_values)
        )
        object.__setattr__(
            self,
            "_coefficients_newtons",
            tuple(tuple(float(value) for value in row) for row in coefficient_values),
        )
        object.__setattr__(self, "onset_gps", onset_gps)
        object.__setattr__(self, "after", after)
        object.__setattr__(self, "_final_force_newtons", final_endpoint[0])

    @property
    def knots(self) -> u.Quantity:
        values = np.array(self._knots_seconds, dtype=np.float64, copy=True)
        return u.Quantity(values, u.s, copy=False)

    @property
    def coefficients(self) -> u.Quantity:
        values = np.array(self._coefficients_newtons, dtype=np.float64, copy=True)
        return u.Quantity(values, u.N, copy=False)

    @property
    def onset(self) -> LIGOTimeGPS:
        return LIGOTimeGPS(*self.onset_gps)

    def _interval_value(self, index: int, local_u: float) -> float:
        coefficients = self._coefficients_newtons[index]
        powers = (local_u**power for power in range(len(coefficients)))
        return _finite_fsum(
            "ForceHistory value",
            (
                coefficient * power
                for coefficient, power in zip(coefficients, powers, strict=True)
            ),
        )

    def _interval_derivative(self, index: int, local_u: float) -> float:
        width = self._knots_seconds[index + 1] - self._knots_seconds[index]
        coefficients = self._coefficients_newtons[index]
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            terms = (
                power * coefficients[power] * local_u ** (power - 1) / width
                for power in range(1, len(coefficients))
            )
            return _finite_fsum("ForceHistory time derivative", terms)

    def _elapsed_seconds_at_gps(self, time: object) -> float:
        return gps_offset_seconds(normalize_gps(time), self.onset_gps)

    def _value_at_gps(self, time: object) -> float:
        return self._value_at_elapsed_seconds(self._elapsed_seconds_at_gps(time))

    def _derivative_at_gps(self, time: object) -> float:
        return self._derivative_at_elapsed_seconds(self._elapsed_seconds_at_gps(time))

    def _primitive_at_gps(self, time: object) -> float:
        return self._primitive_at_elapsed_seconds(self._elapsed_seconds_at_gps(time))

    def _value_at_elapsed_seconds(self, elapsed_seconds: object) -> float:
        elapsed = _real_seconds("elapsed_seconds", elapsed_seconds)
        if elapsed < 0.0:
            return 0.0
        if elapsed >= self._knots_seconds[-1]:
            return 0.0 if self.after == "zero" else self._final_force_newtons
        index = bisect_right(self._knots_seconds, elapsed) - 1
        width = self._knots_seconds[index + 1] - self._knots_seconds[index]
        local_u = (elapsed - self._knots_seconds[index]) / width
        return self._interval_value(index, local_u)

    def _derivative_at_elapsed_seconds(self, elapsed_seconds: object) -> float:
        elapsed = _real_seconds("elapsed_seconds", elapsed_seconds)
        if elapsed < 0.0 or elapsed >= self._knots_seconds[-1]:
            return 0.0
        index = bisect_right(self._knots_seconds, elapsed) - 1
        width = self._knots_seconds[index + 1] - self._knots_seconds[index]
        local_u = (elapsed - self._knots_seconds[index]) / width
        return self._interval_derivative(index, local_u)

    def _primitive_at_elapsed_seconds(self, elapsed_seconds: object) -> float:
        """Return the causal integral of force from onset through ``elapsed``."""
        elapsed = _real_seconds("elapsed_seconds", elapsed_seconds)
        if elapsed <= 0.0:
            return 0.0

        contributions: list[float] = []
        for index, coefficients in enumerate(self._coefficients_newtons):
            start = self._knots_seconds[index]
            width = self._knots_seconds[index + 1] - start
            stop = min(elapsed, self._knots_seconds[index + 1])
            if stop <= start:
                break
            upper_u = (stop - start) / width
            terms = (
                width * coefficient * upper_u ** (power + 1) / (power + 1)
                for power, coefficient in enumerate(coefficients)
            )
            contributions.append(_finite_fsum("ForceHistory interval integral", terms))
            if stop == elapsed:
                break

        final_knot = self._knots_seconds[-1]
        if elapsed > final_knot and self.after == "hold":
            with np.errstate(over="ignore", invalid="ignore"):
                tail = self._final_force_newtons * (elapsed - final_knot)
            if not math.isfinite(tail):
                raise ValueError("ForceHistory primitive is non-finite at this time")
            contributions.append(tail)
        return _finite_fsum("ForceHistory primitive", contributions)

    def _delay_moment_at_gps(self, time: object, delay_seconds: object) -> float:
        return self._delay_moment_at_elapsed_seconds(
            self._elapsed_seconds_at_gps(time), delay_seconds
        )

    def _delay_moment_at_elapsed_seconds(
        self, elapsed_seconds: object, delay_seconds: object
    ) -> float:
        """Return ``integral_0^delay tau * F(elapsed - tau) d tau``.

        Each polynomial is translated to the overlap endpoint and integrated
        in a small local delay variable. The held tail is integrated directly
        in delay coordinates, avoiding distant antiderivative subtraction.
        """
        elapsed = _real_seconds("elapsed_seconds", elapsed_seconds)
        delay = _real_seconds("delay_seconds", delay_seconds)
        if delay < 0.0:
            raise ValueError("delay_seconds must be nonnegative")
        if elapsed <= 0.0 or delay == 0.0:
            return 0.0

        contributions: list[float] = []
        for index, coefficients in enumerate(self._coefficients_newtons):
            start = self._knots_seconds[index]
            stop = self._knots_seconds[index + 1]
            width = stop - start
            if elapsed <= start:
                continue

            right_x = min(stop, elapsed)
            tau_at_right = elapsed - right_x
            if delay <= tau_at_right:
                continue
            available_from_start = elapsed - start
            overlap_seconds = (
                right_x - start
                if delay >= available_from_start
                else delay - tau_at_right
            )
            if overlap_seconds <= 0.0:
                continue

            origin_u = (right_x - start) / width
            overlap_u = overlap_seconds / width
            shifted = _shifted_polynomial(coefficients, origin_u)
            terms = []
            for power, coefficient in enumerate(shifted):
                sign = 1.0 if power % 2 == 0 else -1.0
                first = width * tau_at_right * overlap_u ** (power + 1) / (power + 1)
                second = width**2 * overlap_u ** (power + 2) / (power + 2)
                terms.append(sign * coefficient * (first + second))
            contributions.append(_finite_fsum("ForceHistory delay interval", terms))

        final_knot = self._knots_seconds[-1]
        if elapsed > final_knot and self.after == "hold":
            held_delay = min(delay, elapsed - final_knot)
            with np.errstate(over="ignore", invalid="ignore"):
                held_moment = 0.5 * self._final_force_newtons * held_delay**2
            if not math.isfinite(held_moment):
                raise ValueError("ForceHistory delay moment is non-finite at this time")
            contributions.append(held_moment)
        return _finite_fsum("ForceHistory delay moment", contributions)


def smooth_pulse(
    *, amplitude: u.Quantity, duration: u.Quantity, onset: object
) -> ForceHistory:
    """Create a zero-ended pulse ``amplitude * 16 s² (1-s)²``."""
    force = _scalar_quantity_si("amplitude", amplitude, u.N)
    span = _scalar_quantity_si("duration", duration, u.s)
    if span <= 0.0:
        raise ValueError("duration must be positive")
    return ForceHistory(
        knots=np.array([0.0, span], dtype=np.float64) * u.s,
        coefficients=np.array([[0.0, 0.0, 16.0, -32.0, 16.0]]) * force * u.N,
        onset=onset,
        after="zero",
    )


def smooth_ramp_to_hold(
    *, amplitude: u.Quantity, duration: u.Quantity, onset: object
) -> ForceHistory:
    """Create the C1 ramp ``amplitude * (3 s² - 2 s³)`` followed by a hold."""
    force = _scalar_quantity_si("amplitude", amplitude, u.N)
    span = _scalar_quantity_si("duration", duration, u.s)
    if span <= 0.0:
        raise ValueError("duration must be positive")
    return ForceHistory(
        knots=np.array([0.0, span], dtype=np.float64) * u.s,
        coefficients=np.array([[0.0, 0.0, 3.0, -2.0]]) * force * u.N,
        onset=onset,
        after="hold",
    )


__all__ = ["ForceHistory", "smooth_pulse", "smooth_ramp_to_hold"]
