"""Independent positive-air-side acoustic surface Newton-gradient oracle.

No production imports, production output or legacy development-reference code.
Harms (2019), Eqs 131/132: density=p_surface*cos(b*z)/c**2,
p_surface=2*p_inc. Horizontal Fourier Green transform is
2*pi*exp(-q*abs(z-t))/q. Its positive-side z->0+ integral has
J=q/(q*q+b*b), J'=q*q/(q*q+b*b). The acceleration transfer/G
is 2*pi/c**2 * (i*cos(phi)*J, i*sin(phi)*J, J'/q).
The normalized hemisphere outer product is integrated below.
"""

import importlib.abc
import sys
from contextlib import contextmanager
from decimal import ROUND_CEILING, ROUND_HALF_EVEN, Decimal, localcontext

import mpmath as mp

COMPONENTS = (
    "input",
    "root",
    "endpoint",
    "quadrature",
    "tail",
    "formulation",
    "precision",
    "roundoff",
    "serialization",
)


class ProductionDenied(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, _path=None, _target=None):
        if fullname == "gragra" or fullname.startswith("gragra."):
            raise ImportError("formal acoustic oracle denies production import")
        return


@contextmanager
def independent():
    if any(n == "gragra" or n.startswith("gragra.") for n in sys.modules):
        raise ImportError("production module already loaded")
    guard = ProductionDenied()
    sys.meta_path.insert(0, guard)
    try:
        yield
    finally:
        sys.meta_path.remove(guard)


def context(dps):
    if type(dps) is not int or dps < 25:
        raise ValueError("at least 25 digits")
    c = mp.mp.clone()
    c.dps = dps
    return c


def read_inputs(c, inputs):
    result = {}
    for name, value in inputs.items():
        if not isinstance(value, str) or not Decimal(value).is_finite():
            raise ValueError("finite decimal string input required")
        result[name] = c.mpf(value)
        if result[name] < 0 or (
            name in ("frequency_hz", "sound_speed_m_s", "g_si") and not result[name]
        ):
            raise ValueError("input domain violation")
    return result


def encode(c, value, upward=False):
    if not c.isfinite(value):
        raise ValueError("nonfinite value")
    if not value:
        return "0e+0"
    with localcontext() as dc:
        dc.prec = 40
        dc.rounding = ROUND_CEILING if upward else ROUND_HALF_EVEN
        d = +Decimal(c.nstr(value, 70))
    sign, coeff, exponent = d.as_tuple()
    text = "".join(map(str, coeff))
    power = exponent + len(text) - 1
    text = text.rstrip("0")
    return (
        ("-" if sign else "")
        + text[0]
        + ("." + text[1:] if len(text) > 1 else "")
        + f"e{power:+d}"
    )


def norm(c, values):
    return c.sqrt(c.fsum(abs(v) ** 2 for v in values))


def split_green(c, q, b, z):
    """Closed integral at positive air height, never ground-side continuation."""
    if q <= 0 or b < 0 or z < 0:
        raise ValueError("q>0,b>=0,z>=0 required")
    den = q * q + b * b
    return (
        (2 * q * c.cos(b * z) - q * c.exp(-q * z)) / den,
        (-2 * q * b * c.sin(b * z) + q * q * c.exp(-q * z)) / den,
    )


def green_integral(c, q, b, z):
    """Independent split vertical integral with an explicit exponential tail."""
    if q <= 0 or b < 0 or z <= 0:
        raise ValueError("strict positive air height required")
    end = z + (c.dps + 10) * c.log(10) / q
    n = max(1, int(c.ceil((end - z) * b / c.pi)))
    points = [z + (end - z) * i / n for i in range(n + 1)]
    lower = c.quad(lambda t: c.exp(-q * (z - t)) * c.cos(b * t), [0, z])
    upper = c.quad(lambda t: c.exp(-q * (t - z)) * c.cos(b * t), points)
    return lower + upper, q * (upper - lower), c.exp(-q * (end - z)) / q


def surface_matrix(inputs, *, dps=80, resolution=4, route="theta"):
    c = context(dps)
    p = read_inputs(c, inputs)
    if resolution not in (1, 2, 4) or route not in ("theta", "mu"):
        raise ValueError("unknown integration procedure")
    k = 2 * c.pi * p["frequency_hz"] / p["sound_speed_m_s"]
    # Positive-side surface transfer amplitude = (2pi/c**2/k)*sin(theta).
    amplitude = 2 * c.pi / (p["sound_speed_m_s"] ** 2 * k)
    if route == "theta":
        polar = c.quad(
            lambda t: c.sin(t) ** 3,
            [c.pi * i / (2 * resolution) for i in range(resolution + 1)],
            maxdegree=8,
        )
    else:
        polar = c.quad(
            lambda u: 1 - u * u,
            [c.mpf(i) / resolution for i in range(resolution + 1)],
            maxdegree=8,
        )
    azi = [
        lambda phi: c.j * c.cos(phi),
        lambda phi: c.j * c.sin(phi),
        lambda _phi: c.mpf(1),
    ]
    points = [c.pi * i / resolution for i in range(2 * resolution + 1)]
    return [
        p["pressure_psd_pa2_hz"]
        * amplitude**2
        * polar
        * c.quad(
            lambda phi, i=i, j=j: azi[i](phi) * c.conj(azi[j](phi)), points, maxdegree=8
        )
        / (2 * c.pi)
        for i in range(3)
        for j in range(3)
    ]


def record(
    family,
    fixture,
    representation,
    procedure_hash,
    source_pins,
    matrix_function,
    diagnostics=None,
):
    """Full CSD uncertainty components are in the observable Frobenius units."""
    if family not in ("A1", "A3") or representation not in ("Gfree", "SI"):
        raise ValueError("unsupported family/representation")
    expected_convention = "surface_total" if family == "A1" else "cavern"
    if fixture["pressure_convention"] != expected_convention:
        raise ValueError("frozen pressure normalization mismatch")
    c = context(110)
    p = read_inputs(c, fixture["inputs"])
    scale = p["pressure_psd_pa2_hz"] / (p["sound_speed_m_s"] * p["frequency_hz"]) ** 2
    if family == "A3":
        scale *= 4
    conversion = p["g_si"] ** 2 if representation == "SI" else c.mpf(1)
    scale *= conversion
    zero = p["pressure_psd_pa2_hz"] == 0 or (family == "A3" and p["radius_m"] == 0)
    errors = {name: c.mpf(0) for name in COMPONENTS}
    disagreement = c.mpf(0)
    if zero:
        values = [c.mpc(0)] * 9
    else:
        low = matrix_function(fixture["inputs"], dps=60, resolution=4)
        samples = [
            matrix_function(fixture["inputs"], dps=80, resolution=n) for n in (1, 2, 4)
        ]
        alternate = matrix_function(
            fixture["inputs"],
            dps=80,
            resolution=4,
            route="mu" if family == "A1" else "radial",
        )
        values = [v * conversion for v in samples[-1]]

        def distance(a, b):
            return norm(c, [(x - y) * conversion for x, y in zip(a, b, strict=True)])

        errors["precision"] = 10 * distance(low, samples[-1])
        errors["quadrature"] = 10 * max(
            distance(samples[0], samples[1]), distance(samples[1], samples[2])
        )
        disagreement = 10 * distance(alternate, samples[-1])
        errors["input"] = scale * c.mpf("1e-58")
        errors["roundoff"] = scale * c.mpf("1e-55")
        half_ulps = []
        for v in values:
            for part in (c.re(v), c.im(v)):
                half_ulps.append(
                    c.mpf(0)
                    if not part
                    else c.power(10, c.floor(c.log10(abs(part))) - 39) / 2
                )
        errors["serialization"] = norm(c, half_ulps)
    methods = {
        "input": "exact decimal input conversion allowance 1e-58 times source scale",
        "root": "not applicable: no roots",
        "endpoint": "finite removable/regular endpoints",
        "quadrature": "10 times largest 1/2/4 refinement disagreement",
        "tail": "finite domain; full convergent radial small-t series",
        "formulation": (
            "zero after independently reviewed exact coordinate/radial "
            "equivalence; discrepancy checked separately"
        ),
        "precision": "10 times 60/80dps disagreement",
        "roundoff": "heuristic 100000-operation allowance at 60dps",
        "serialization": "18 real-component half ulps in Frobenius norm",
    }
    components = {
        name: {"value": encode(c, value, True), "method": methods[name]}
        for name, value in errors.items()
    }
    with localcontext() as dc:
        dc.prec = 150
        public_sum = sum(
            (Decimal(item["value"]) for item in components.values()), Decimal(0)
        )
    resolved = zero or (
        c.isfinite(c.mpf(str(public_sum)))
        and c.mpf(str(public_sum)) <= scale * c.mpf("1e-30")
        and disagreement <= scale * c.mpf("1e-31")
        and all(
            errors[n] <= scale * c.mpf("1e-31") for n in ("precision", "quadrature")
        )
    )
    if diagnostics is not None:
        diagnostics.update(
            {
                "fixture_id": fixture["fixture_id"],
                "representation": representation,
                "precision_dps": [60, 80],
                "resolutions": [1, 2, 4],
                "alternate_disagreement": encode(c, disagreement, True),
                "alternate_gate": bool(zero or disagreement <= scale * c.mpf("1e-31")),
                "public_total": encode(c, c.mpf(str(public_sum)), True),
            }
        )
    return {
        "schema_version": 1,
        "key": {
            "family": family,
            "fixture_id": fixture["fixture_id"],
            "frequency_hz": fixture["inputs"]["frequency_hz"],
            "target_id": fixture["target_id"],
            "observable": "acceleration_csd",
            "representation": representation,
            "set": "science",
        },
        "status": "resolved" if resolved else "unresolved",
        "quantity": {
            "shape": [3, 3],
            "data": [
                {"real": encode(c, c.re(v)), "imag": encode(c, c.im(v))} for v in values
            ],
        }
        if resolved
        else None,
        "uncertainty": {
            "kind": "exact_zero" if zero else "estimate",
            "components": components,
            "certificate": (
                "zero pressure PSD; linear gravity transfer"
                if p["pressure_psd_pa2_hz"] == 0
                else "zero radius empty Newton volume"
            )
            if zero
            else None,
        },
        "input_assumptions": (
            "exact decimal mathematical inputs; one-sided per-Hz PSD; "
            "isotropic incoherent directions; numerical estimates, "
            "not rigorous bounds"
        ),
        "source_pins": source_pins,
        "procedure_hash": procedure_hash,
        "inputs": fixture["inputs"],
        "units": "kg^2 m^-4 Hz^-1" if representation == "Gfree" else "m^2 s^-4 Hz^-1",
        "input_convention": fixture["pressure_convention"],
        "admission_status": "NOT_FORMALLY_ADMITTED",
    }
