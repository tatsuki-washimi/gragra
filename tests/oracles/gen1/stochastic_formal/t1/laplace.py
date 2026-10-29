"""Positive double-Laplace parent radial integral, independent of Bessel APIs.

Radial t integration is analytic. Finite t/u/v tails are bounded separately.
This is a production-blind candidate awaiting review, not formal evidence.
"""

from functools import lru_cache

import mpmath as mp

U_POINTS = ("0", ".125", ".25", ".5", "1", "2", "4", "8")


@lru_cache(maxsize=12)
def nodes(order, dps):
    with mp.workdps(dps):
        locations, weights = mp.gauss_quadrature(order, "legendre")
        return tuple(zip(locations, weights, strict=True))


def radial(x, resolution, length=60):
    """Integrate tK_n(t)^2 on[x,x+L] via K's positive Laplace representation.

    Tensor positive Gauss16/32/64 on every fixed u/v panel; full radial
    antiderivative includes both endpoints. No geometry-only PSD claim.
    """
    if x <= 0 or resolution not in (1, 2, 4) or length <= 0:
        raise ValueError("positive x/L and resolution1/2/4 required")
    samples = []
    for lo, hi in zip(U_POINTS[:-1], U_POINTS[1:], strict=True):
        lo, hi = mp.mpf(lo), mp.mpf(hi)
        half, mid = (hi - lo) / 2, (hi + lo) / 2
        for node, weight in nodes(16 * resolution, mp.mp.dps):
            u = mid + half * node
            c = mp.cosh(u)
            at_x = half * weight * mp.exp(-x * (c - 1))
            at_r = half * weight * mp.exp(-(x + length) * (c - 1))
            samples.append((c, at_x, at_r))
    i0, i1 = mp.mpf(0), mp.mpf(0)
    cutoff = x + length
    end_factor = mp.exp(-2 * length)
    for i, (ci, wi, ri) in enumerate(samples):
        for j in range(i + 1):
            cj, wj, rj = samples[j]
            a = ci + cj
            term = wi * wj * (x / a + 1 / a**2) - end_factor * ri * rj * (
                cutoff / a + 1 / a**2
            )
            if i != j:
                term *= 2
            i0 += term
            i1 += term * ci * cj
    restore = mp.exp(-2 * x)
    # Uniform u-tail bound L(t)<=exp(-t*cosh8+8)/(t*sinh8-1).
    # Both K_n(t) and L(t) decay at least exp(-(t-x)) for t>=x.
    # K_n(x)e^x <=sqrt(2pi/x)*exp(1/(2x)) from cosh u>=1+u²/2
    # and cosh(nu)<=exp(u), n0/1. Full square omitted piece≤2KL+L².
    denominator = x * mp.sinh(8) - 1
    if denominator <= 0:
        raise ValueError("fixed Laplace-tail cutoff inadequate")
    upper_k = mp.sqrt(2 * mp.pi / x) * mp.exp(1 / (2 * x))
    upper_l = mp.exp(-x * (mp.cosh(8) - 1) + 8) / denominator
    kernel_tail = (
        restore * (2 * upper_k * upper_l + upper_l**2) * (x / 2 + mp.mpf(".25"))
    )
    radial_tail = mp.pi / 4 * cutoff / (cutoff - 1) * mp.exp(-2 * cutoff)
    return i0 * restore, i1 * restore, radial_tail + kernel_tail
