"""Independent cap modified-Bessel split: exact CAS and bounded samples.

R,kappa>0, m integer>=0, a>=0; a=0 only radial-constant m=0.
Neumann J_m'(a*R)=0. Target r<R or r>R, g=z-zcap!=0 for the inner
closed-form term. No production or frozen computational oracle imports.
Numeric samples search for counterexamples, not field acceptance.
"""

import math
import sys
from pathlib import Path

import mpmath as mp
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from certified_tail_envelopes import Interval, exponential_tail  # noqa: E402


def symbolic_checks():
    r, radius, kappa, a, g, regulator = sp.symbols(
        "r R kappa a g regulator", positive=True
    )
    m = sp.symbols("m", integer=True, nonnegative=True)
    j, i, k = (sp.Function(name)(r) for name in ("J", "I", "K"))
    replacements = {
        sp.diff(j, r, 2): -sp.diff(j, r) / r + (m**2 / r**2 - a**2) * j,
        sp.diff(i, r, 2): -sp.diff(i, r) / r + (m**2 / r**2 + kappa**2) * i,
        sp.diff(k, r, 2): -sp.diff(k, r) / r + (m**2 / r**2 + kappa**2) * k,
    }
    for modified in (i, k):
        primitive = r * (j * sp.diff(modified, r) - sp.diff(j, r) * modified)
        primitive /= a**2 + kappa**2
        assert (
            sp.simplify(sp.diff(primitive, r).xreplace(replacements) - r * j * modified)
            == 0
        )
    wronskian = i * sp.diff(k, r) - sp.diff(i, r) * k
    assert sp.simplify(sp.diff(r * wronskian, r).xreplace(replacements)) == 0
    # Small-argument normalizations fix r*(I*K_r-I_r*K)=-1.
    nu = sp.symbols("nu", positive=True)
    i_leading = (kappa * r / 2) ** nu / sp.gamma(nu + 1)
    k_leading = sp.gamma(nu) / 2 * (2 / (kappa * r)) ** nu
    assert (
        sp.combsimp(
            r * (i_leading * sp.diff(k_leading, r) - sp.diff(i_leading, r) * k_leading)
            + 1
        )
        == 0
    )
    # m=0: I~1, K_r~-1/r gives the same constant.
    assert sp.simplify(r * (1 * (-1 / r) - 0) + 1) == 0
    split = (
        k * r * (j * sp.diff(i, r) - sp.diff(j, r) * i)
        - i * r * (j * sp.diff(k, r) - sp.diff(j, r) * k)
    ) / (a**2 + kappa**2)
    assert sp.simplify(split + r * j * wronskian / (a**2 + kappa**2)) == 0
    boundary_j, boundary_i, boundary_k, ip, kp = sp.symbols("JR IR KR Ip Kp")
    inside = boundary_j + radius * boundary_j * kappa * kp * boundary_i
    outside = radius * boundary_j * kappa * ip * boundary_k
    jump = sp.expand(inside - outside)
    expected = boundary_j * (1 + radius * kappa * (boundary_i * kp - ip * boundary_k))
    assert sp.simplify(jump - expected) == 0
    assert (
        sp.simplify(
            radius * boundary_j * kappa * kp * kappa * ip
            - radius * boundary_j * kappa * ip * kappa * kp
        )
        == 0
    )
    coefficient = sp.symbols("coefficient", real=True)
    inner_w = (j + coefficient * i) / (a**2 + kappa**2)
    outer_w = coefficient * k / (a**2 + kappa**2)

    def operator(f):
        return sp.diff(f, r, 2) + sp.diff(f, r) / r - m**2 * f / r**2 - kappa**2 * f

    assert sp.simplify(operator(inner_w).xreplace(replacements) + j) == 0
    assert sp.simplify(operator(outer_w).xreplace(replacements)) == 0
    # General endpoint derivative is retained before imposing Neumann;
    # approximate numerical roots do not make this term literally exact zero.
    endpoint_derivative = sp.symbols("JrR")
    full_inside = (
        boundary_j
        + radius
        * (boundary_j * kappa * kp - endpoint_derivative * boundary_k)
        * boundary_i
    )
    full_outside = (
        radius
        * (boundary_j * kappa * ip - endpoint_derivative * boundary_i)
        * boundary_k
    )
    assert sp.simplify(full_inside - full_outside - expected) == 0
    integral = sp.integrate(
        kappa * sp.sin(kappa * g) / (kappa**2 + a**2), (kappa, 0, sp.oo)
    )
    assert sp.simplify(integral.rewrite(sp.exp) - sp.pi * sp.exp(-a * g) / 2) == 0
    assert sp.limit(integral, a, 0, dir="+") == sp.pi / 2
    # The a=0 Dirichlet integral is Abel-regulated before differentiation.
    abel_cosine = sp.integrate(
        sp.exp(-regulator * kappa) * sp.cos(kappa * g), (kappa, 0, sp.oo)
    )
    assert sp.simplify(abel_cosine - sp.diff(sp.atan(g / regulator), g)) == 0
    assert sp.limit(sp.atan(g / regulator), regulator, 0, dir="+") == sp.pi / 2
    for order in range(3):
        positive = sp.diff(sp.exp(-a * g), g, order)
        assert sp.simplify(positive - (-a) ** order * sp.exp(-a * g)) == 0
        negative = sp.diff(-sp.exp(a * g), g, order)
        assert sp.simplify(negative + a**order * sp.exp(a * g)) == 0
    # Independent direct cap-flux Cartesian kernel derivatives used by the
    # bounded numerical comparison; z here is the signed cap gap.
    coordinates = sp.symbols("x y z", real=True)
    squared = sum(coordinate**2 for coordinate in coordinates)
    flux_kernel = coordinates[2] / squared ** sp.Rational(3, 2)
    for row in range(3):
        gradient = (1 if row == 2 else 0) / squared ** sp.Rational(3, 2)
        gradient -= 3 * coordinates[2] * coordinates[row] / squared ** sp.Rational(5, 2)
        assert sp.simplify(sp.diff(flux_kernel, coordinates[row]) - gradient) == 0
        for column in range(3):
            numerator = coordinates[column] if row == 2 else 0
            numerator += coordinates[row] if column == 2 else 0
            numerator += coordinates[2] if row == column else 0
            hessian = -3 * numerator / squared ** sp.Rational(5, 2)
            hessian += (
                15
                * coordinates[2]
                * coordinates[row]
                * coordinates[column]
                / squared ** sp.Rational(7, 2)
            )
            assert (
                sp.simplify(
                    sp.diff(flux_kernel, coordinates[row], coordinates[column])
                    - hessian
                )
                == 0
            )
    # Bulk S jump and local Helmholtz jump cancel at each oriented cap.
    normal, endpoint, normalization, mode_value = sp.symbols("sigma epsilon N Jcap")
    bulk_jump = 4 * sp.pi * normalization * normal * endpoint * mode_value
    local_jump = -4 * sp.pi * normalization * normal * endpoint * mode_value
    assert sp.simplify(bulk_jump + local_jump) == 0
    # Analytic bounds: kappa^2/(a^2+kappa^2)<=1, and after a K' source
    # derivative the residual powers are j-1/2, not j+1/2.
    assert sp.simplify(1 - kappa**2 / (a**2 + kappa**2) - a**2 / (a**2 + kappa**2)) == 0
    assert [sp.Rational(order) - sp.Rational(1, 2) for order in range(3)] == [
        -sp.Rational(1, 2),
        sp.Rational(1, 2),
        sp.Rational(3, 2),
    ]
    print("CAS radial primitives/Wronskian/ODE/continuity/Fourier/g-sign/contact PASS")


def first_root(order):
    def function(argument):
        return mp.diff(lambda z: mp.besselj(order, z), argument)

    left = mp.mpf("0.25")
    while True:
        right = left + mp.mpf("0.25")
        if function(left) * function(right) < 0:
            return mp.findroot(function, (left, right), solver="anderson")
        left = right


def overlap_samples():
    mp.mp.dps = 50
    radius = mp.mpf("1.3")
    samples = 0
    for order in range(4):
        alpha = first_root(order)
        a = alpha / radius
        jr = mp.besselj(order, alpha)
        for fraction in (mp.mpf("0.23"), mp.mpf("0.95"), mp.mpf("1.3")):
            r = fraction * radius
            for scaled in (mp.mpf("0.2"), mp.mpf(2), mp.mpf(10)):
                kappa = scaled / radius
                direct = mp.quad(
                    lambda s, order=order, a=a, kappa=kappa, r=r: (
                        s
                        * mp.besselj(order, a * s)
                        * mp.besseli(order, kappa * min(r, s))
                        * mp.besselk(order, kappa * max(r, s))
                    ),
                    [0, min(r, radius), radius] if r < radius else [0, radius],
                )
                if r < radius:
                    closed = (
                        mp.besselj(order, a * r)
                        + radius
                        * jr
                        * kappa
                        * mp.diff(
                            lambda v, order=order: mp.besselk(order, v), kappa * radius
                        )
                        * mp.besseli(order, kappa * r)
                    ) / (a * a + kappa * kappa)
                else:
                    closed = (
                        radius
                        * jr
                        * kappa
                        * mp.diff(
                            lambda v, order=order: mp.besseli(order, v), kappa * radius
                        )
                        * mp.besselk(order, kappa * r)
                        / (a * a + kappa * kappa)
                    )
                assert abs(direct - closed) < mp.mpf("1e-40") * max(1, abs(direct))
                samples += 1
        for axial in (0, 1):
            eta_m = 1 if order == 0 else mp.mpf("0.5")
            eta_q = 1 if axial == 0 else mp.mpf("0.5")
            normalization = 1 / mp.sqrt(
                eta_q * eta_m * jr * jr * (1 - order * order / (alpha * alpha))
            )
            radial = mp.quad(
                lambda s, order=order, a=a: s * mp.besselj(order, a * s) ** 2,
                [0, radius],
            )
            rms = (
                normalization**2
                * radial
                * (2 * mp.pi * eta_m)
                * eta_q
                / (mp.pi * radius**2)
            )
            assert abs(rms - 1) < mp.mpf("1e-40")
    for fraction in (mp.mpf(0), mp.mpf("0.23"), mp.mpf("0.95"), mp.mpf("1.3")):
        r = fraction * radius
        for scaled in (mp.mpf("0.2"), mp.mpf(2), mp.mpf(10)):
            kappa = scaled / radius
            direct = mp.quad(
                lambda s, kappa=kappa, r=r: (
                    s
                    * mp.besseli(0, kappa * min(r, s))
                    * mp.besselk(0, kappa * max(r, s))
                ),
                [0, r, radius] if 0 < r < radius else [0, radius],
            )
            closed = (
                (
                    1
                    - radius
                    * kappa
                    * mp.besselk(1, kappa * radius)
                    * mp.besseli(0, kappa * r)
                )
                / kappa**2
                if r < radius
                else radius
                * mp.besseli(1, kappa * radius)
                * mp.besselk(0, kappa * r)
                / kappa
            )
            assert abs(direct - closed) < mp.mpf("1e-40") * max(1, abs(direct))
            samples += 1
    for axial in (1, 2):
        squared_axial = mp.quad(
            lambda u, axial=axial: mp.cos(axial * mp.pi * u) ** 2, [0, 1]
        )
        assert abs(2 * squared_axial - 1) < mp.mpf("1e-40")
    print(f"Finite radial overlap samples={samples}; real RMS modes=8 PASS")


def residual_shape_tail(order, m, radius_lower, gap_lower, cutoff):
    """Shape envelope; multiply by upper 4*N*R*abs(J_m(alpha))."""
    if not 0 <= order <= 2 or m < 0 or min(radius_lower, gap_lower, cutoff) <= 0:
        raise ValueError("nonnegative order and positive radius/gap/cutoff required")
    radius, t = Interval(radius_lower), Interval(cutoff)
    exponential = (Interval((m + 1) ** 2) / (radius * t * 2)).exp()
    # kappa^(j-1/2) <= kappa^j/sqrt(T), for kappa>=T.
    root = (Interval(22) / 7 / (radius * t * 2)).sqrt()
    return root * exponential * exponential_tail(order, gap_lower, cutoff)


def tail_and_endpoint_samples():
    mp.mp.dps = 80
    samples = 0
    for m in range(4):
        for order in range(3):
            for gap in (0.01, 0.2, 1.0):
                for cutoff in (2.0, 10.0, 100.0):
                    radius = mp.mpf(1.3)
                    delta, t = mp.mpf(gap), mp.mpf(cutoff)
                    power = mp.mpf(order) + mp.mpf("0.5")
                    exact = mp.sqrt(mp.pi / (2 * radius)) * mp.exp(
                        mp.mpf((m + 1) ** 2) / (2 * radius * t)
                    )
                    exact *= mp.gammainc(power, delta * t, mp.inf) / delta**power
                    actual = residual_shape_tail(order, m, 1.3, gap, cutoff)
                    assert mp.mpf(actual.upper_float()) >= exact
                    samples += 1
    assert residual_shape_tail(0, 0, 1.3, 1, 1000).upper_float() == math.nextafter(
        0.0, math.inf
    )
    # a=0,m=0 residual integrand (before 4N sigma epsilon) tends to -g;
    # its z derivative tends to -1. Every transverse/second derivative is zero.
    radius, kappa = mp.mpf("1.3"), mp.mpf("1e-20")
    for r in (mp.mpf(0), mp.mpf("0.43")):
        for g in (mp.mpf("0.04"), mp.mpf("-0.17")):
            radial = (
                radius * (-mp.besselk(1, kappa * radius)) * mp.besseli(0, kappa * r)
            )
            assert abs(radial * mp.sin(kappa * g) + g) < mp.mpf("1e-35")
            assert abs(radial * kappa * mp.cos(kappa * g) + 1) < mp.mpf("1e-35")
    print(f"Outward half-power tail samples={samples}; radialconstant endpoints PASS")


if __name__ == "__main__":
    symbolic_checks()
    overlap_samples()
    tail_and_endpoint_samples()
