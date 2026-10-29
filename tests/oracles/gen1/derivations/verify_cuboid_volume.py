"""Supplementary Cuboid 1D Gaussian-volume CAS and bounded numerical checks.

No production or frozen oracle imports. Numerical samples search for formula
counterexamples; the CAS identities and analytic inequalities support proofs.
This does not replace the frozen independent surface-field acceptance matrix.
"""

import itertools
import math
import sys
from pathlib import Path

import mpmath as mp
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from certified_tail_envelopes import Interval  # noqa: E402


def symbolic_checks():
    t, radius, cutoff, wave_number = sp.symbols("t r T k", positive=True)
    s, x, y, z, beta, shift = sp.symbols("s x y z beta shift", real=True)
    kernel = sp.exp(-(t**2) * (s - x) ** 2)
    cosine = sp.cos(beta * (s + shift))
    sine = sp.sin(beta * (s + shift))
    assert sp.simplify(sp.diff(kernel, x) + sp.diff(kernel, s)) == 0
    assert (
        sp.simplify(
            sp.diff(kernel, x) * cosine
            + sp.diff(kernel * cosine, s)
            + beta * kernel * sine
        )
        == 0
    )
    assert (
        sp.simplify(
            sp.diff(kernel, x) * sine
            + sp.diff(kernel * sine, s)
            - beta * kernel * cosine
        )
        == 0
    )
    assert (
        sp.simplify(
            sp.diff(kernel, x, 2) + 2 * t**3 * sp.diff(kernel, t) + 2 * t**2 * kernel
        )
        == 0
    )
    # Exact away-source Newton kernel and its analytic Cartesian derivatives.
    moments = [
        sp.integrate(t ** (2 * order) * sp.exp(-(t**2) * radius**2), (t, 0, sp.oo))
        for order in range(3)
    ]
    assert sp.simplify(2 * moments[0] / sp.sqrt(sp.pi) - 1 / radius) == 0
    coordinates = (x, y, z)
    for i, coordinate in enumerate(coordinates):
        assert (
            sp.simplify(
                -4 * coordinate * moments[1] / sp.sqrt(sp.pi) + coordinate / radius**3
            )
            == 0
        )
        for j, other in enumerate(coordinates):
            integrated = (
                2
                / sp.sqrt(sp.pi)
                * (
                    4 * coordinate * other * moments[2]
                    - (2 * moments[1] if i == j else 0)
                )
            )
            expected = 3 * coordinate * other / radius**5
            expected -= 1 / radius**3 if i == j else 0
            assert sp.simplify(integrated - expected) == 0
    # Product heat identity supplies the interior contact contribution once.
    functions = [sp.Function(f"F{index}")(t, coordinates[index]) for index in range(3)]
    product = sp.prod(functions)
    laplacian = sum(sp.diff(product, coordinate, 2) for coordinate in coordinates)
    replacement = {
        sp.diff(function, coordinate, 2): (
            -2 * t**3 * sp.diff(function, t) - 2 * t**2 * function
        )
        for function, coordinate in zip(functions, coordinates, strict=True)
    }
    assert (
        sp.simplify(laplacian.xreplace(replacement) + 2 * sp.diff(t**3 * product, t))
        == 0
    )
    psi = sp.prod(
        sp.cos(b * (coordinate + a))
        for b, a, coordinate in zip(
            sp.symbols("b0 b1 b2", real=True),
            sp.symbols("a0 a1 a2", real=True),
            coordinates,
            strict=True,
        )
    )
    betas = sp.symbols("b0 b1 b2", real=True)
    assert (
        sp.simplify(
            sum(sp.diff(psi, coordinate, 2) for coordinate in coordinates)
            + sum(b**2 for b in betas) * psi
        )
        == 0
    )
    full_space = (
        sp.pi ** sp.Rational(3, 2) / t**3 * sp.exp(-(wave_number**2) / (4 * t**2))
    )
    assert sp.limit(t**3 * full_space, t, sp.oo) == sp.pi ** sp.Rational(3, 2)
    bulk_tail = 2 / wave_number**2 * (1 - sp.exp(-(wave_number**2) / (4 * cutoff**2)))
    assert (
        sp.simplify(
            sp.diff(bulk_tail, cutoff)
            + sp.exp(-(wave_number**2) / (4 * cutoff**2)) / cutoff**3
        )
        == 0
    )
    assert sp.limit(bulk_tail, cutoff, sp.oo) == 0
    assert sp.limit(bulk_tail, wave_number, 0, dir="+") == 1 / (2 * cutoff**2)
    # All interval derivative products of total Cartesian order <=2 have
    # powers -3,-2,-1,0; only the three bulk factors have algebraic power -3.
    powers = {0: (-1,), 1: (-1, 0, 0), 2: (-1, 2, 2)}
    multiindices = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    multiindices += [
        tuple(int(axis == i) + int(axis == j) for axis in range(3))
        for i in range(3)
        for j in range(3)
    ]
    assert {
        sum(choice)
        for alpha in multiindices
        for choice in itertools.product(*(powers[order] for order in alpha))
    } == {-3, -2, -1, 0}
    # Gaussian derivative parity plus centered Neumann parity certifies
    # component zeros at exact centered coordinates, never a numerical floor.
    length = sp.symbols("L", positive=True)
    q = sp.symbols("q", integer=True, nonnegative=True)
    for n in (2 * q, 2 * q + 1):
        mode = sp.cos(n * sp.pi * (s + length / 2) / length)
        assert sp.trigsimp(sp.expand_trig(mode.subs(s, -s) - (-1) ** n * mode)) == 0
    for n in range(6):
        mode = sp.cos(n * sp.pi * (s + length / 2) / length)
        assert sp.trigsimp(mode.subs(s, -s) - (-1) ** n * mode) == 0
        for order in range(3):
            derivative = sp.diff(kernel, x, order).subs(x, 0)
            assert sp.simplify(derivative.subs(s, -s) - (-1) ** order * derivative) == 0
    scale = sp.symbols("scale", positive=True)
    assert (
        sp.simplify(
            kernel.subs({t: t / scale, s: scale * s, x: scale * x}, simultaneous=True)
            - kernel
        )
        == 0
    )
    for order in range(3):
        # Three spatial Jacobians, target derivatives and one inverse-length
        # t integration give length^(2-order), times G*density in s^-2.
        assert sp.prod([scale] * 3) / scale**order / scale == scale ** (2 - order)
    print("CAS Newton/Cartesian/IBP/contact/bulk-tail/parity identities PASS")


def interval_values(t, x, length, n):
    """Direct finite 1D numerical check only; not a production algorithm."""
    a, b = -length / 2, length / 2
    beta = n * mp.pi / length
    breaks = {a, b}
    for offset in (-12, -4, 0, 4, 12):
        position = x + mp.mpf(offset) / t
        if a < position < b:
            breaks.add(position)
    breaks = sorted(breaks)

    def integrand(s, order):
        distance = s - x
        gaussian = mp.exp(-(t**2) * distance**2)
        derivatives = (
            mp.mpf(1),
            2 * t**2 * distance,
            4 * t**4 * distance**2 - 2 * t**2,
        )
        return gaussian * derivatives[order] * mp.cos(beta * (s + length / 2))

    values = tuple(
        mp.quad(lambda s, order=order: integrand(s, order), breaks)
        for order in range(3)
    )
    sine = mp.quad(
        lambda s: mp.exp(-(t**2) * (s - x) ** 2) * mp.sin(beta * (s + length / 2)),
        breaks,
    )
    da, db = a - x, b - x
    ea, eb = mp.exp(-(t**2) * da**2), mp.exp(-(t**2) * db**2)
    ibp = (
        ea - (-1) ** n * eb - beta * sine,
        2 * t**2 * (da * ea - (-1) ** n * db * eb) - beta**2 * values[0],
    )
    for actual, expected in zip(values[1:], ibp, strict=True):
        assert abs(actual - expected) < mp.mpf("1e-45") * max(
            1, abs(actual), abs(expected)
        )
    gap = max(mp.mpf(0), a - x, x - b)
    base = mp.sqrt(mp.pi) / t * mp.exp(-(t**2) * gap**2)
    bounds = (
        base,
        beta * base + ea + eb,
        beta**2 * base + 2 * t**2 * (abs(da) * ea + abs(db) * eb),
    )
    for actual, bound in zip(values, bounds, strict=True):
        assert abs(actual) <= bound + mp.mpf("1e-45") * max(1, bound)
    return values


def outward_tail_term(coefficient_upper, power, gap_squared_lower, cutoff):
    """Rigorous positive monomial tail upper bound, with outward arithmetic."""
    if coefficient_upper < 0 or cutoff <= 0 or gap_squared_lower < 0:
        raise ValueError("nonnegative coefficient/gap and positive cutoff required")
    if power not in (-3, -2, -1, 0):
        raise ValueError("only the derived volume powers are admitted")
    if coefficient_upper == 0:
        return Interval(0)
    coefficient, t = Interval(coefficient_upper), Interval(cutoff)
    if gap_squared_lower == 0:
        if power != -3:
            raise ValueError("zero-gap nonbulk term is unresolved or physical boundary")
        return coefficient / (t**2 * 2)
    gap_squared = Interval(gap_squared_lower)
    exponential = (-(gap_squared * t**2)).exp()
    mills = exponential / (gap_squared * t * 2)
    return coefficient * mills / t ** (-power)


def numerical_checks():
    mp.mp.dps = 70
    samples = 0
    for length in (mp.mpf("0.1"), mp.mpf(1), mp.mpf(10)):
        for n in (0, 1, 2):
            for fraction in (mp.mpf("0.23"), mp.mpf("1.3")):
                for scaled_t in (mp.mpf("0.01"), mp.mpf(1), mp.mpf(30)):
                    interval_values(scaled_t / length, fraction * length, length, n)
                    samples += 1
    # Interior large-t trace limit independently tests automatic contact and
    # the finite-T bulk heat factor; the exterior/contact-free limit differs.
    lengths = (mp.mpf("0.1"), mp.mpf(1), mp.mpf(10))
    indices = (0, 0, 1)
    for point in (
        (mp.mpf("0.02"), mp.mpf("0.1"), mp.mpf("0.4")),
        (mp.mpf("0.08"), mp.mpf("0.1"), mp.mpf("0.4")),
    ):
        t = mp.mpf(1000)
        value = mp.fprod(
            interval_values(t, x, length, n)[0]
            for x, length, n in zip(point, lengths, indices, strict=True)
        )
        inside = all(
            abs(x) < length / 2 for x, length in zip(point, lengths, strict=True)
        )
        k_squared = sum(
            (n * mp.pi / length) ** 2
            for n, length in zip(indices, lengths, strict=True)
        )
        psi = mp.fprod(
            mp.cos(n * mp.pi * (x + length / 2) / length)
            for x, length, n in zip(point, lengths, indices, strict=True)
        )
        expected = (
            mp.pi ** mp.mpf("1.5") * mp.exp(-k_squared / (4 * t**2)) * psi
            if inside
            else mp.mpf(0)
        )
        assert abs(t**3 * value - expected) < mp.mpf("1e-45")
    # Directed numerical evaluation must dominate the exact positive integral.
    mp.mp.dps = 100
    tail_samples = 0
    for gap_squared in (0.001, 0.1, 1.0):
        for cutoff in (0.5, 2.0, 10.0):
            for power in (-3, -2, -1, 0):
                dd, tt = mp.mpf(gap_squared), mp.mpf(cutoff)
                exact = mp.gammainc(mp.mpf(power + 1) / 2, dd * tt**2, mp.inf)
                exact /= 2 * dd ** (mp.mpf(power + 1) / 2)
                bound = outward_tail_term(1, power, gap_squared, cutoff).upper_float()
                assert mp.mpf(bound) >= exact
                tail_samples += 1
    assert outward_tail_term(1, -3, 0, 2).upper_float() >= 0.125
    assert outward_tail_term(1, -3, 100, 10).upper_float() == math.nextafter(
        0.0, math.inf
    )
    assert outward_tail_term(0, -3, 100, 10).upper_float() == 0.0
    try:
        outward_tail_term(1, 0, 0, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("zero-gap boundary/nonbulk tail must not fabricate zero")
    print(
        f"Finite interval/IBP bound samples={samples}; "
        f"outward tail samples={tail_samples} PASS"
    )


if __name__ == "__main__":
    symbolic_checks()
    numerical_checks()
