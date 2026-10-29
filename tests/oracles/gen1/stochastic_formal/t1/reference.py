"""Independent conditional A7 full projected PSD; no production computation.

Creighton gr-qc/0007050v2 A1/A4/A7, pp13–14, and14/15,p6.
Model approximation error is unquantified and excluded from numerical U.
This candidate does not grant formal admission or physical atmosphere validity.
"""

import importlib.util
import re
import sys
from decimal import ROUND_CEILING, ROUND_HALF_EVEN, Decimal, localcontext
from pathlib import Path

import mpmath as mp

_SPEC = importlib.util.spec_from_file_location(
    "t1_laplace", Path(__file__).with_name("laplace.py")
)
_LAPLACE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _LAPLACE
_SPEC.loader.exec_module(_LAPLACE)

NAMES = (
    "frequency_hz",
    "density_kg_m3",
    "temperature_k",
    "structure_coefficient_k_m_neg_p_over_2",
    "structure_exponent",
    "wind_speed_m_s",
    "minimum_approach_m",
    "wind_axis_angle_rad",
)
PATTERN = re.compile(r"(?:0e\+0|-?[1-9](?:\.[0-9]*[1-9])?e[+-](?:0|[1-9][0-9]*))\Z")


def encode(value, digits=40, upward=False):
    value = mp.mpf(value)
    if not mp.isfinite(value):
        raise ValueError("nonfinite numerical value")
    if not value:
        return "0e+0"
    with localcontext() as context:
        context.prec = digits
        context.rounding = ROUND_CEILING if upward else ROUND_HALF_EVEN
        number = +Decimal(mp.nstr(value, digits + 15))
    sign, ds, exponent = number.as_tuple()
    text = "".join(map(str, ds))
    power = len(text) + exponent - 1
    text = text.rstrip("0")
    return (
        ("-" if sign else "")
        + text[0]
        + ("." + text[1:] if len(text) > 1 else "")
        + f"e{power:+d}"
    )


def parse_inputs(inputs):
    if set(inputs) != set(NAMES):
        raise ValueError("complete exact physical tuple required")
    for value in inputs.values():
        if not isinstance(value, str) or not PATTERN.fullmatch(value):
            raise ValueError("noncanonical physical input")
        if value.endswith("e-0"):
            raise ValueError("noncanonical physical spelling")
    values = dict(zip(NAMES, (mp.mpf(inputs[n]) for n in NAMES), strict=True))
    if any(
        values[n] <= 0
        for n in NAMES
        if n not in ("structure_exponent", "wind_axis_angle_rad")
    ):
        raise ValueError("positive physical coefficients required")
    if not 0 < values["structure_exponent"] < 2:
        raise ValueError("structure exponent requires0<p<2")
    return values


def structure_coefficient(p):
    """A1 one-sided factor2 and A4 covariance -DT/2, then bundle measure.

    nu=(p+1)/2; analytic continuation temporal Fourier coefficient,
    times2π times integral t^(nu+1) Knu(t) dt. The constant covariance
    contributes only DC; no printed A2/A3 normalization is inherited.
    """
    nu = (p + 1) / 2
    return (-2 * mp.sqrt(mp.pi) * 2**nu / mp.gamma(-p / 2)) * (
        2 * mp.pi * 2**nu * mp.gamma(nu + 1)
    )


def full_path(
    inputs, dps, resolution, representation="Gfree", length=60, method="laplace"
):
    """Reparse/recompute entire nonlinear physical path at each trial.

    Positive double Laplace integral for both K kernels, analytic radial
    integration with both endpoints. No production kve or Bessel differences.
    Half-plane azimuth explicitly integrated before the radial integral.
    The slow unscaled-Bessel path remains a disjoint calibration crosscheck.
    """
    if representation not in ("Gfree", "SI"):
        raise ValueError("unknown representation")
    with mp.workdps(dps):
        q = parse_inputs(inputs)
        f, rho, temp, ct, p, wind, radius, theta = [q[n] for n in NAMES]
        omega = 2 * mp.pi * f
        x = omega * radius / wind
        scale = (
            structure_coefficient(p)
            * 4
            * (rho / temp) ** 2
            * ct**2
            * wind ** (p + 2)
            * omega ** (-p - 3)
        )
        if representation == "SI":
            scale *= mp.mpf("6.6743e-11") ** 2
        weights = (
            mp.quad(lambda _psi: mp.cos(theta) ** 2, [0, mp.pi]),
            mp.quad(lambda psi: (mp.sin(theta) * mp.cos(psi)) ** 2, [0, mp.pi]),
        )
        if method == "laplace":
            a, b, tail = _LAPLACE.radial(x, resolution, length)
            return (
                scale * (weights[0] * a + weights[1] * b),
                scale * sum(weights) * tail,
                x,
            )
        if method != "bessel":
            raise ValueError("unknown independent reference method")
        base = [mp.mpf(0), min(x, mp.mpf(1))]
        while base[-1] < length:
            base.append(min(mp.mpf(length), 2 * base[-1]))
        points = [base[0]]
        for lo, hi in zip(base[:-1], base[1:], strict=True):
            points.extend(
                lo + (hi - lo) * i / resolution for i in range(1, resolution + 1)
            )
        integrals = [
            mp.quad(
                lambda y, order=n: (
                    (x + y) * (mp.exp(x) * mp.besselk(order, x + y)) ** 2
                ),
                points,
                method="gauss-legendre",
            )
            for n in (0, 1)
        ]
        value = (
            scale
            * mp.exp(-2 * x)
            * sum(w * i for w, i in zip(weights, integrals, strict=True))
        )
        cutoff = x + length
        # K0 bound≤exp(-t)sqrt(π/(2t)); K1≤exp(-t)sqrt(π/(2(t-1))).
        tail = (
            scale
            * sum(weights)
            * mp.pi
            / 4
            * cutoff
            / (cutoff - 1)
            * mp.exp(-2 * cutoff)
        )
        return value, tail, x


def evaluate(
    inputs,
    representation="Gfree",
    precisions=(50, 70),
    resolutions=(1, 2, 4),
    digits=40,
):
    if len(precisions) != 2 or len(resolutions) != 3 or digits > min(precisions) - 10:
        raise ValueError("precision/refinement/serialization contract")
    with mp.workdps(max(precisions) + 20):
        trials = [
            full_path(inputs, dps, res, representation)
            for dps in precisions
            for res in resolutions
        ]
        value = trials[-1][0]
        floor = abs(value) * mp.mpf("1e-40")
        differences = []
        for offset in (0, 3):
            first = abs(trials[offset + 1][0] - trials[offset][0])
            last = abs(trials[offset + 2][0] - trials[offset + 1][0])
            if last > first / 2 and max(first, last) > floor:
                raise ValueError("noncontracting full PSD refinements")
            differences.extend((first, last))
        precision = 10 * max(abs(trials[j + 3][0] - trials[j][0]) for j in range(3))
        if precision > floor:
            raise ValueError("full path precision disagreement")
        central = encode(value, digits)
        parts = {
            "input": mp.mpf(0),
            "root": mp.mpf(0),
            "endpoint": mp.mpf(0),
            "quadrature": 10 * sum(differences),
            "tail": trials[-1][1],
            "formulation": mp.mpf(0),
            "precision": precision,
            "roundoff": floor,
            "serialization": abs(value - mp.mpf(central))
            + abs(value) * mp.power(10, -digits),
        }
        public = {name: encode(v, digits, upward=True) for name, v in parts.items()}
        return {
            "central": central,
            "components": public,
            "total_uncertainty": encode(
                sum(mp.mpf(v) for v in public.values()), digits, upward=True
            ),
            "x": encode(trials[-1][2]),
            "uncertainty_kind": "estimate",
            "model_approximation_error": "UNQUANTIFIED_EXCLUDED_FROM_NUMERICAL_U",
            "trials": [
                {"dps": dps, "resolution": res, "value": encode(t[0])}
                for (dps, res), t in zip(
                    ((d, r) for d in precisions for r in resolutions),
                    trials,
                    strict=True,
                )
            ],
        }
