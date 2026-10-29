"""Independent formal R1: directional halfspace potential and azimuth integral.
Harms2019 DOI10.1007/s41114-019-0022-2 equations38-40,96,98,112.
No production imports. This module calculates candidates, never admits them.
"""

import re
from functools import lru_cache

import mpmath as mp


def number(value):
    if not isinstance(value, str):
        raise TypeError("exact decimal string required")
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value):
        raise ValueError("finite decimal required")
    return mp.mpf(value)


def norm(values):
    return mp.sqrt(mp.fsum(abs(v) ** 2 for v in values))


@lru_cache(maxsize=12)
def nodes(order, dps):
    with mp.workdps(dps):
        x, w = mp.gauss_quadrature(order, "legendre")
        return tuple(zip(x, w, strict=True))


def integrate(fun, left, right, order):
    mid, half = (left + right) / 2, (right - left) / 2
    return half * mp.fsum(w * fun(mid + half * x) for x, w in nodes(order, mp.mp.dps))


def decimal(value, digits=70):
    # Preserve precision already carried by root-bracket mpf endpoints.
    if not isinstance(value, mp.mpf):
        value = mp.mpf(value)
    if not mp.isfinite(value):
        raise ValueError("nonfinite result")
    if value == 0:
        return "0e+0"
    raw = mp.nstr(value, n=digits, min_fixed=0, max_fixed=0)
    mantissa, separator, exponent = raw.lower().partition("e")
    if not separator:
        exponent = "0"
    if "." in mantissa:
        mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exponent):+d}"


def upward(value, digits=70):
    if value < 0:
        raise ValueError("negative uncertainty")
    if value == 0:
        return "0e+0"
    quantum = mp.power(10, int(mp.floor(mp.log10(value))) - digits + 1)
    return decimal((mp.ceil(value / quantum) + 1) * quantum, digits)


def root(vp, vs, dps):
    with mp.workdps(dps + 20):
        cp, cs = number(vp), number(vs)
        beta = (cs / cp) ** 2 if cp > 0 else mp.inf
        if not 0 < beta <= mp.mpf("0.5"):
            raise ValueError("nonnegative Poisson ratio domain required")

        def polynomial(y):
            return y**3 - 8 * y**2 + (24 - 16 * beta) * y - 16 * (1 - beta)

        left, right = mp.mpf(0), mp.mpf(1)
        while right - left > mp.power(10, -dps - 5):
            middle = (left + right) / 2
            if polynomial(middle) < 0:
                left = middle
            else:
                right = middle
        return cs * mp.sqrt((left + right) / 2), left, right


def amplitude(inputs, speed):
    rho, cp, cs, f, h = [
        number(inputs[k])
        for k in ("density_kg_m3", "v_p_m_s", "v_s_m_s", "frequency_hz", "height_m")
    ]
    k = 2 * mp.pi * f / speed
    qp = mp.sqrt(k * k - (2 * mp.pi * f / cp) ** 2)
    qs = mp.sqrt(k * k - (2 * mp.pi * f / cs) ** 2)
    zeta = mp.sqrt(qp / qs)
    potential = -2 * mp.pi * rho * (1 - zeta) * mp.exp(-k * h) / (qp - k * zeta)
    return -k * potential


def evaluate(inputs, order, dps, frozen_root=None):
    rho, f, h, psd = [
        number(inputs[k])
        for k in (
            "density_kg_m3",
            "frequency_hz",
            "height_m",
            "surface_vertical_displacement_psd_m2_hz",
        )
    ]
    if min(rho, h, psd) < 0 or f <= 0:
        raise ValueError("invalid physical inputs")
    speed, left, right = frozen_root or root(inputs["v_p_m_s"], inputs["v_s_m_s"], dps)
    amp = amplitude(inputs, speed)

    def vector(phi):
        return (mp.j * amp * mp.cos(phi), mp.j * amp * mp.sin(phi), -amp)

    values = [
        psd
        * integrate(
            lambda phi, i=i, j=j: vector(phi)[i] * mp.conj(vector(phi)[j]),
            mp.mpf(0),
            2 * mp.pi,
            order,
        )
        / (2 * mp.pi)
        for i in range(3)
        for j in range(3)
    ]
    cp, cs = number(inputs["v_p_m_s"]), number(inputs["v_s_m_s"])
    q, s = mp.sqrt(1 - (speed / cp) ** 2), mp.sqrt(1 - (speed / cs) ** 2)
    zeta = mp.sqrt(q / s)
    horizontal = (
        2
        * (mp.pi * rho * (1 - zeta) / (q - zeta)) ** 2
        * mp.exp(-4 * mp.pi * f * h / speed)
        * psd
    )
    literal = [
        horizontal * (2 if i == 2 else 1) if i == j else mp.mpf(0)
        for i in range(3)
        for j in range(3)
    ]
    # This is the same final-root endpoint whole-CSD spread. Guard digits
    # preserve the narrow interval during subtraction; central90/60 paths
    # and their separately owned arithmetic-precision component are unchanged.
    with mp.workdps(dps + 50):
        endpoint_cs = number(inputs["v_s_m_s"])
        endpoint_psd = number(inputs["surface_vertical_displacement_psd_m2_hz"])
        endpoint_powers = [
            amplitude(inputs, endpoint_cs * mp.sqrt(y)) ** 2 * endpoint_psd
            for y in (left, right)
        ]
        root_error = (
            10 * abs(endpoint_powers[1] - endpoint_powers[0]) * mp.sqrt(mp.mpf("1.5"))
        )
    return (
        values,
        literal,
        (2 * mp.pi * rho) ** 2 * psd,
        root_error,
        {
            "rayleigh_speed_m_s": decimal(speed),
            "root_bracket": [decimal(left, 110), decimal(right, 110)],
            "horizontal_literal": decimal(horizontal),
            "surface_limit": "h=0 evaluated only as h->0+ AUX",
        },
    )


def reference(inputs, representation):
    """Complete candidate with fixed predeclared refinements, never production."""
    with mp.workdps(90):
        levels = [evaluate(inputs, n, 90) for n in (24, 48, 96)]
        high, literal, scale, root_error, metadata = levels[-1]
        fixed_root = root(inputs["v_p_m_s"], inputs["v_s_m_s"], 90)
        with mp.workdps(60):
            low = evaluate(inputs, 96, 60, fixed_root)[0]
        factor = number("6.6743e-11") ** 2 if representation == "SI" else mp.mpf(1)
        if representation not in ("Gfree", "SI"):
            raise ValueError("unknown representation")
        high = [v * factor for v in high]
        literal = [v * factor for v in literal]
        low = [v * factor for v in low]
        scale *= factor
        data = [{"real": decimal(mp.re(v)), "imag": decimal(mp.im(v))} for v in high]
        restored = [mp.mpc(number(v["real"]), number(v["imag"])) for v in data]
        drift = max(
            norm(
                [
                    a * factor - b * factor
                    for a, b in zip(levels[j][0], levels[j - 1][0], strict=True)
                ]
            )
            for j in (1, 2)
        )
        precision = norm([a - b for a, b in zip(high, low, strict=True)])
        formulation = norm([a - b for a, b in zip(high, literal, strict=True)])
        if scale and (
            drift > scale * number("1e-12")
            or precision > scale * number("1e-40")
            or formulation > scale * number("1e-12")
            or root_error * factor > scale * number("1e-50")
        ):
            raise ValueError("frozen reference convergence gate unresolved")
        components = {
            "input": (
                mp.mpf(0),
                "exact mathematical decimal fixture and conventional G",
            ),
            "root": (
                root_error * factor,
                "10x physical root bracket endpoint fullCSD spread;0 if no root",
            ),
            "endpoint": (
                mp.mpf(0),
                "compact smooth angular endpoints; no endpoint singularity",
            ),
            "quadrature": (
                10 * drift,
                "10x maximum adjacent24/48/96 highprecision fullCSD norm",
            ),
            "tail": (
                mp.mpf(0),
                "compact azimuth or B2 radial antiderivative infinite endpoint",
            ),
            "formulation": (
                mp.mpf(0),
                "source-equivalence proof; literal discrepancy is diagnostic gate",
            ),
            "precision": (10 * precision, "10x full96level60vs90digit drift, estimate"),
            "roundoff": (
                scale * number("1e-82"),
                "declared conservative90digit arithmetic floor, estimate not bound",
            ),
            "serialization": (
                norm([a - b for a, b in zip(high, restored, strict=True)])
                + 3 * scale * number("1e-69"),
                "actual fullCSD central reparse norm plus70digit padding",
            ),
        }
        uncertainty = {
            "kind": "estimate",
            "components": {
                k: {"value": upward(v), "method": m} for k, (v, m) in components.items()
            },
            "certificate": None,
        }
        if scale == 0:
            if any(v != 0 for v in high):
                raise ValueError("zero scale with nonzero field")
            uncertainty["kind"] = "exact_zero"
            uncertainty["certificate"] = (
                "whole CSD0: exact rho0 or all applicable displacement PSD0"
            )
        return {
            "quantity": {"shape": [3, 3], "data": data},
            "uncertainty": uncertainty,
            "source_scale": decimal(scale),
            "method": {
                **metadata,
                "quadrature_drift_norm": decimal(drift),
                "precision_difference_norm": decimal(precision),
                "literal_difference_norm": decimal(formulation),
                "root_error_norm": decimal(root_error * factor),
                "serialization_digits": 70,
                "precision_digits": [60, 90],
                "orders": [24, 48, 96],
                "uncertainty_classification": "estimated not theorem",
            },
        }
