"""Executable conservative Decimal interval evaluation of analytic tail bounds.

Inputs are exact represented binary floats; callers supply a certified lower
gap and upper coefficients. This standalone reference is not imported by
production. It avoids upper-gamma library error and positive-tail underflow.
"""

from __future__ import annotations

import math
from decimal import ROUND_CEILING, ROUND_FLOOR, Context, Decimal

LOW = Context(prec=80, rounding=ROUND_FLOOR)
HIGH = Context(prec=80, rounding=ROUND_CEILING)


class Interval:
    def __init__(self, value, upper=None):
        self.lower = self.exact(value)
        self.upper = self.exact(value if upper is None else upper)
        if not self.lower.is_finite() or not self.upper.is_finite():
            raise ValueError("finite endpoints required")
        if self.lower > self.upper:
            raise ValueError("interval endpoints reversed")

    @staticmethod
    def exact(value):
        return Decimal.from_float(value) if isinstance(value, float) else Decimal(value)

    def __add__(self, other):
        other = other if isinstance(other, Interval) else Interval(other)
        return Interval(
            LOW.add(self.lower, other.lower), HIGH.add(self.upper, other.upper)
        )

    def __neg__(self):
        return Interval(self.upper.copy_negate(), self.lower.copy_negate())

    def __mul__(self, other):
        other = other if isinstance(other, Interval) else Interval(other)
        lower = min(
            LOW.multiply(a, b)
            for a in (self.lower, self.upper)
            for b in (other.lower, other.upper)
        )
        upper = max(
            HIGH.multiply(a, b)
            for a in (self.lower, self.upper)
            for b in (other.lower, other.upper)
        )
        return Interval(lower, upper)

    def __truediv__(self, other):
        other = other if isinstance(other, Interval) else Interval(other)
        if other.lower <= 0:
            raise ValueError("positive denominator required")
        reciprocal = Interval(
            LOW.divide(Decimal(1), other.upper), HIGH.divide(Decimal(1), other.lower)
        )
        return self * reciprocal

    def __pow__(self, exponent):
        if exponent < 0 or int(exponent) != exponent:
            raise ValueError("nonnegative integer powers required")
        result = Interval(1)
        for _ in range(exponent):
            result = result * self
        return result

    def exp(self):
        # Decimal exp/sqrt are correctly rounded to nearest even. Widen one
        # representable neighbor regardless of the context rounding choice.
        lower = max(Decimal(0), LOW.next_minus(LOW.exp(self.lower)))
        upper = HIGH.next_plus(HIGH.exp(self.upper))
        return Interval(lower, upper)

    def sqrt(self):
        if self.lower < 0:
            raise ValueError("nonnegative square root required")
        return Interval(
            max(Decimal(0), LOW.next_minus(LOW.sqrt(self.lower))),
            HIGH.next_plus(HIGH.sqrt(self.upper)),
        )

    def upper_float(self):
        value = float(self.upper)
        if not math.isfinite(value):
            raise ArithmeticError("tail overflow: uncertainty unresolved")
        if Decimal.from_float(value) < self.upper:
            value = math.nextafter(value, math.inf)
        if not math.isfinite(value):
            raise ArithmeticError("tail overflow: uncertainty unresolved")
        return value


def gaussian_tail(power, delta_lower, cutoff):
    if delta_lower <= 0 or cutoff <= 0 or not 0 <= power <= 6:
        raise ValueError("positive gap/cutoff and power 0..6 required")
    delta, t = Interval(delta_lower), Interval(cutoff)
    exponential = (-(delta**2 * t**2)).exp()
    denominator = delta**2 * 2
    tails = [exponential / (denominator * t), exponential / denominator]
    for p in range(2, power + 1):
        tails.append(
            t ** (p - 1) * exponential / denominator
            + tails[p - 2] * (p - 1) / denominator
        )
    return tails[power]


def exponential_tail(power, delta_lower, cutoff):
    if delta_lower <= 0 or cutoff <= 0 or not 0 <= power <= 3:
        raise ValueError("positive gap/cutoff and power 0..3 required")
    delta, k = Interval(delta_lower), Interval(cutoff)
    polynomial = Interval(0)
    for j in range(power + 1):
        polynomial = polynomial + k ** (power - j) * math.factorial(
            power
        ) / math.factorial(power - j) / delta ** (j + 1)
    return (-(delta * k)).exp() * polynomial


def modified_shape_tail(order, m, radius_lower, delta_lower, cutoff):
    if radius_lower <= 0 or not 0 <= order <= 2 or m < 0:
        raise ValueError("positive radius, order 0..2 and m>=0 required")
    r, k = Interval(radius_lower), Interval(cutoff)
    angular_factor = (Interval((m + 3) ** 2) / (k * r * 2)).exp()
    root_factor = (Interval(22) / 7 / (r * k * 2)).sqrt()
    return (
        root_factor * angular_factor * exponential_tail(order + 1, delta_lower, cutoff)
    )


def main():
    import mpmath as mp

    mp.mp.dps = 100
    for delta in [0.1, 1.0, 10.0]:
        for cutoff in [0.2, 2.0, 20.0]:
            for power in range(7):
                bound = gaussian_tail(power, delta, cutoff).upper_float()
                dd, tt = mp.mpf(delta), mp.mpf(cutoff)
                reference = mp.gammainc(
                    mp.mpf(power + 1) / 2, (dd * tt) ** 2, mp.inf
                ) / (2 * dd ** (power + 1))
                assert mp.mpf(bound) >= reference
            for power in range(4):
                bound = exponential_tail(power, delta, cutoff).upper_float()
                reference = mp.gammainc(
                    power + 1, mp.mpf(delta) * cutoff, mp.inf
                ) / mp.mpf(delta) ** (power + 1)
                assert mp.mpf(bound) >= reference
    tiny = gaussian_tail(4, 10.0, 10.0).upper_float()
    assert tiny == math.nextafter(0.0, math.inf)
    assert Interval(0).upper_float() == 0.0
    for order in range(3):
        value = modified_shape_tail(order, 3, 1.0, 0.2, 40.0).upper_float()
        assert value > 0 and math.isfinite(value)
    print("Directed Decimal Gaussian/exponential tails and nonzero underflow PASS")


if __name__ == "__main__":
    main()
