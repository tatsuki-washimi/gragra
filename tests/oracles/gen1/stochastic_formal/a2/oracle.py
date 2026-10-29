"""Independent defining A2 angular integral. Never imports production computations."""

import importlib.abc
import sys
from decimal import ROUND_CEILING, Decimal, localcontext
from pathlib import Path

import mpmath as mp

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "tests/oracles/gen1"))
from stochastic_acceptance import (  # noqa: E402 - independent sibling path
    COMPONENTS,
    canonical_decimal,
    validate_record,
)


class ProductionDenied(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, _path=None, _target=None):
        if fullname == "gragra" or fullname.startswith("gragra."):
            raise ImportError("production import forbidden")


def independence():
    if any(n == "gragra" or n.startswith("gragra.") for n in sys.modules):
        raise ImportError("production already loaded")
    sys.meta_path.insert(0, ProductionDenied())


def dec(value, digits=40, upward=False):
    ctx = mp.mp.clone()
    ctx.dps = max(digits + 25, value._mpf_[3] * 30103 // 100000 + 20)
    with localcontext() as dc:
        dc.prec = digits
        if upward:
            dc.rounding = ROUND_CEILING
        return canonical_decimal(+Decimal(ctx.nstr(value, digits + 20)))


def context(dps):
    ctx = mp.mp.clone()
    ctx.dps = dps
    return ctx


def points(ctx, x, resolution):
    base = sorted(
        {
            ctx.mpf(0),
            ctx.pi / 2,
            *[ctx.asin(ctx.mpf(j) / x) for j in (1, 2, 4, 8, 16, 32, 64) if j < x],
        }
    )
    return sorted(
        {
            *base,
            *[
                a + (b - a) * i / resolution
                for a, b in zip(base, base[1:], strict=False)
                for i in range(1, resolution)
            ],
        }
    )


def integral(x, *, dps=80, resolution=4, route="theta"):
    if resolution not in (1, 2, 4) or dps not in (60, 80):
        raise ValueError("frozen dps/resolution required")
    c = context(dps)
    x = c.mpf(x)
    if not c.isfinite(x) or x < 0:
        raise ValueError("nonnegative finite x required")
    splits = points(c, x, resolution)
    if route == "theta":
        return c.quad(
            lambda t: c.sin(t) ** 3 * c.exp(-x * c.sin(t)) / 2, splits, maxdegree=8
        )
    if route == "mu":
        return c.quad(
            lambda u: (1 - u * u) * c.exp(-x * c.sqrt(1 - u * u)) / 2,
            sorted({c.cos(t) for t in splits}),
            maxdegree=8,
        )
    raise ValueError("unknown route")


def estimate(x):
    c = context(110)
    low = integral(x, dps=60)
    samples = [integral(x, resolution=n) for n in (1, 2, 4)]
    alternate_samples = [integral(x, resolution=n, route="mu") for n in (1, 2, 4)]
    low = c.mpf(low)
    samples = [c.mpf(v) for v in samples]
    alternate_samples = [c.mpf(v) for v in alternate_samples]
    alternate = alternate_samples[-1]
    value = samples[-1]
    cause = dict.fromkeys(COMPONENTS, c.mpf(0))
    cause["precision"] = 10 * abs(low - value)
    cause["quadrature"] = 10 * max(
        abs(samples[0] - samples[1]),
        abs(samples[1] - samples[2]),
        abs(alternate_samples[0] - alternate_samples[1]),
        abs(alternate_samples[1] - alternate_samples[2]),
    )
    coordinate_disagreement = 10 * abs(alternate - value)
    cause["input"] = c.mpf("1e-95") * (1 + abs(c.mpf(x)))
    cause["endpoint"] = c.mpf("1e-75") * (1 + abs(value))
    cause["roundoff"] = c.mpf("1e-55") * abs(value)
    return (
        c,
        value,
        cause,
        {
            "exact_x": x,
            "theta_low60": dec(low, 55),
            "theta_high80_resolution124": [dec(v, 65) for v in samples],
            "mu_high80_resolution124": [dec(v, 65) for v in alternate_samples],
            "coordinate_disagreement": dec(coordinate_disagreement, 100, True),
            "uncertainty_components": {
                name: dec(u, 40, True) for name, u in cause.items()
            },
            "reference_resolved": bool(
                c.fsum(cause.values()) + coordinate_disagreement
                < abs(value) * c.mpf("1e-35")
            ),
        },
    )


def records(fixture, procedure_hash, source_pins):
    c = context(110)
    f = c.mpf(fixture["inputs"]["frequency_hz"])
    cs = c.mpf(fixture["inputs"]["sound_speed_m_s"])
    h = c.mpf(fixture["inputs"]["depth_m"])
    p = c.mpf(fixture["inputs"]["pressure_psd_pa2_hz"])
    g = c.mpf(fixture["inputs"]["g_si"])
    x = dec(4 * c.pi * f * h / cs, 100)
    _, value, cause, trace = estimate(x)
    scale = p / (cs * f) ** 2
    methods = {
        "input": (
            "exact physical decimals; 100-digit x conversion allowance; |I prime|<=1/2"
        ),
        "root": "no root operation",
        "endpoint": (
            "finite exact theta domain; "
            "80-digit trig endpoint/split operation allowance"
        ),
        "quadrature": "10 max successive resolution differences over theta and mu",
        "tail": "entire finite domain included; no omitted tail",
        "formulation": "zero: independently reviewed exact theta/mu substitution",
        "precision": "10 theta 60/80 dps disagreement",
        "roundoff": (
            "heuristic 100000-operation 60-dps relative allowance; "
            "plus scale/G arithmetic"
        ),
        "serialization": (
            "sqrt6 times maximum diagonal final 40-digit rounding half-ulp"
        ),
    }
    out = []
    for representation, factor, units in [
        ("Gfree", scale, "kg^2 m^-4 Hz^-1"),
        ("SI", scale * g * g, "m^2 s^-4 Hz^-1"),
    ]:
        central = factor * value
        serialized = [dec(central), dec(central), dec(2 * central)]
        propagated = {name: c.sqrt(6) * factor * u for name, u in cause.items()}
        propagated["roundoff"] += c.sqrt(6) * abs(central) * c.mpf("1e-55")
        propagated["serialization"] = (
            c.sqrt(6) * c.mpf(10) ** (c.floor(c.log10(abs(2 * central))) - 39) / 2
        )
        key = {**fixture["key_base"], "representation": representation}
        record = {
            "schema_version": 1,
            "key": key,
            "status": "resolved" if trace["reference_resolved"] else "unresolved",
            "quantity": {
                "shape": [3, 3],
                "data": [
                    {
                        "real": serialized[i // 4] if i in (0, 4, 8) else "0e+0",
                        "imag": "0e+0",
                    }
                    for i in range(9)
                ],
            },
            "uncertainty": {
                "kind": "estimate",
                "certificate": None,
                "components": {
                    name: {"value": dec(u, 40, True), "method": methods[name]}
                    for name, u in propagated.items()
                },
            },
            "input_assumptions": (
                "Exact decimal physical model tuple; one-sided per-Hz "
                "total surface pressure PSD; ideal rigid atmospheric "
                "halfspace; numerical U is estimated."
            ),
            "source_pins": source_pins,
            "procedure_hash": procedure_hash,
            "inputs": fixture["inputs"],
            "units": units,
            "input_convention": "surface_total_one_sided_per_hz",
            "admission_status": "NOT_FORMALLY_ADMITTED",
        }
        diagnostic = c.sqrt(6) * factor * c.mpf(trace["coordinate_disagreement"])
        trace[representation + "_public_adequacy"] = public_adequacy(
            c, record, diagnostic
        )
        if not trace[representation + "_public_adequacy"]:
            record["status"] = "unresolved"
        if record["status"] == "unresolved":
            record["quantity"] = None
        validate_record(record)
        out.append(record)
    trace["actual_physical_x"] = x
    return out, trace


def scalar_record(fixture, procedure_hash, source_pins):
    """Mathematical AUX at exact nominal x, independently of physical dispatch x."""
    c, value, causes, trace = estimate(fixture["nominal_exact_x"])
    causes["serialization"] = c.mpf(10) ** (c.floor(c.log10(abs(value))) - 39) / 2
    key = {
        **fixture["key_base"],
        "target_id": "nominal_x",
        "observable": "isotropic_depth_factor",
        "set": "auxiliary",
        "representation": "Gfree",
    }
    record = {
        "schema_version": 1,
        "key": key,
        "status": "resolved" if trace["reference_resolved"] else "unresolved",
        "quantity": {"shape": [], "data": [{"real": dec(value), "imag": "0e+0"}]},
        "uncertainty": {
            "kind": "estimate",
            "certificate": None,
            "components": {
                name: {
                    "value": dec(u, 40, True),
                    "method": (
                        "direct nominal scalar angular integral; "
                        + name
                        + "; exact finite-domain root/tail zeros; "
                        "others frozen precision/resolution/endpoint allowances"
                    ),
                }
                for name, u in causes.items()
            },
        },
        "input_assumptions": (
            "Exact nominal dimensionless x mathematical AUX; "
            "Gfree tag means no G factor; no SI representation."
        ),
        "source_pins": source_pins,
        "procedure_hash": procedure_hash,
        "inputs": {
            "nominal_exact_x": fixture["nominal_exact_x"],
            "frequency_hz": fixture["inputs"]["frequency_hz"],
        },
        "units": "1",
        "input_convention": "dimensionless_exact_nominal_x",
        "admission_status": "NOT_FORMALLY_ADMITTED",
    }
    trace["scalar_public_adequacy"] = public_adequacy(
        c, record, c.mpf(trace["coordinate_disagreement"])
    )
    if not trace["scalar_public_adequacy"]:
        record["status"] = "unresolved"
    if record["status"] == "unresolved":
        record["quantity"] = None
    validate_record(record)
    return record, trace


def public_adequacy(c, record, diagnostic):
    """Strict gate on emitted observable strings, separate from public U."""
    if record["quantity"] is None:
        return False
    norm = c.sqrt(
        c.fsum(
            c.mpf(v["real"]) ** 2 + c.mpf(v["imag"]) ** 2
            for v in record["quantity"]["data"]
        )
    )
    public_u = c.fsum(
        c.mpf(v["value"]) for v in record["uncertainty"]["components"].values()
    )
    return bool(public_u + diagnostic < norm * c.mpf("1e-35"))
