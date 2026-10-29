"""Independent formal B2: exterior Newton volume and moving spherical wall.
Harms2019 DOI10.1007/s41114-019-0022-2 equations45,61,64,65.
Bulk radial antiderivative endpoint tends to0, not a numerical truncated tail.
No scattering; zero P/S cross-CSD is an input model assumption.
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


def transfer(x, order):
    if x < 0:
        raise ValueError("nonnegative ka required")
    # Integrate delta-rho=-i*k*rho*xi_P*exp(i*k*r*mu) outside r=a.
    # Angular dipole and radial antiderivative give4pi*j0(ka), endpoint j0(kL)->0.
    bulk = 4 * mp.pi * (mp.sin(x) / x if x else 1)
    wall_p = (
        -2
        * mp.pi
        * integrate(lambda mu: mu**2 * mp.cos(x * mu), mp.mpf(-1), mp.mpf(1), order)
    )
    wall_s = -mp.pi * integrate(
        lambda mu: (1 - mu**2) * mp.cos(x * mu), mp.mpf(-1), mp.mpf(1), order
    )
    return bulk + wall_p, wall_s


def literal_factor(x):
    return (
        mp.mpf(1) / 3
        if x == 0
        else mp.sqrt(mp.pi / (2 * x)) * mp.besselj(mp.mpf("1.5"), x) / x
    )


def moments(order):
    # Angular n*n and the two separate transverse polarization projectors.
    theta = (mp.sin, mp.sin, mp.cos)
    phi = (mp.cos, mp.sin, lambda _p: mp.mpf(1))
    et = (mp.cos, mp.cos, lambda t: -mp.sin(t))
    ep = (lambda p: -mp.sin(p), mp.cos, lambda _p: mp.mpf(0))

    def angular(tfun, pfun):
        return (
            integrate(lambda t: mp.sin(t) * tfun(t), mp.mpf(0), mp.pi, order)
            * integrate(pfun, mp.mpf(0), 2 * mp.pi, order)
            / (4 * mp.pi)
        )

    p, s = [], []
    for i in range(3):
        for j in range(3):
            p.append(
                angular(
                    lambda t, i=i, j=j: theta[i](t) * theta[j](t),
                    lambda a, i=i, j=j: phi[i](a) * phi[j](a),
                )
            )
            s.append(
                angular(
                    lambda t, i=i, j=j: et[i](t) * et[j](t),
                    lambda a, i=i, j=j: phi[i](a) * phi[j](a),
                )
                + angular(lambda _t: mp.mpf(1), lambda a, i=i, j=j: ep[i](a) * ep[j](a))
            )
    return p, s


def validate_model(inputs):
    """Check frozen incoherence/scattering contract without evaluating fields."""
    if (
        number(inputs.get("p_s_cross_csd", "0e+0")) != 0
        or inputs.get("elastic_scattering", "excluded") != "excluded"
    ):
        raise ValueError("coherent/scattered model excluded")


def evaluate(inputs, order, _dps):
    rho, cp, cs, f, a, p, s = [
        number(inputs[k])
        for k in (
            "density_kg_m3",
            "v_p_m_s",
            "v_s_m_s",
            "frequency_hz",
            "radius_m",
            "p_displacement_psd_m2_hz",
            "s_displacement_psd_m2_hz",
        )
    ]
    if min(rho, a, p, s) < 0 or min(cp, cs, f) <= 0 or cs >= cp:
        raise ValueError("invalid physical inputs")
    validate_model(inputs)
    xp, xs = 2 * mp.pi * f * a / cp, 2 * mp.pi * f * a / cs
    tp, _ = transfer(xp, order)
    _, ts = transfer(xs, order)
    longitudinal, transverse = moments(order)
    # Three times component PSD longitudinal;3/2 per transverse polarization.
    values = [
        rho
        * rho
        * (3 * p * tp * tp * longitudinal_entry + mp.mpf("1.5") * s * ts * ts * t)
        for longitudinal_entry, t in zip(longitudinal, transverse, strict=True)
    ]
    literal_component = (4 * mp.pi * rho) ** 2 * (
        4 * p * literal_factor(xp) ** 2 + s * literal_factor(xs) ** 2
    )
    literal = [
        literal_component if i == j else mp.mpf(0) for i in range(3) for j in range(3)
    ]
    return (
        values,
        literal,
        (4 * mp.pi * rho) ** 2 * (4 * p + s),
        mp.mpf(0),
        {
            "ka_P": decimal(xp),
            "ka_S": decimal(xs),
            "p_s_cross_csd": "exact0 model assumption",
            "polarizations": "two incoherent S modes",
            "radial_tail": "j0(kL)->0 analytically, no finite cutoff",
            "elastic_scattering": "excluded",
        },
    )


def reference(inputs, representation):
    """Complete candidate with fixed predeclared refinements, never production."""
    with mp.workdps(90):
        levels = [evaluate(inputs, n, 90) for n in (24, 48, 96)]
        high, literal, scale, root_error, metadata = levels[-1]
        with mp.workdps(60):
            low = evaluate(inputs, 96, 60)[0]
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
