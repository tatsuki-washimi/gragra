"""Frozen A1/A3 claim checks; invoked only inside reviewed comparison authority.

No production or reference generator imports. Closed source identities check
the admitted R/P claim; they never create a replacement reference or widen the
published centered half-spherical-cavern model's physical scope.
"""

from decimal import Decimal, localcontext

from .transport import GateError, key_id, strict_json, verify_artifact

CHECKS = {
    "A1": (
        "pressure normalization amplitude2/PSD4",
        "G2 transport",
        "component/total relation",
        "positive-side Green limit",
    ),
    "A3": (
        "independent spherical Newton volume prefactor",
        "radial/angular kernel agreement",
        "direction identity average",
        "G2 transport",
    ),
}
PI = Decimal(
    "3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679"
)


def _decimal(value):
    if not isinstance(value, str):
        raise GateError("exact decimal string required")
    result = Decimal(value)
    if not result.is_finite():
        raise GateError("finite decimal required")
    return result


def _encode(value):
    if not value:
        return "0e+0"
    sign, digits, exponent = value.as_tuple()
    text = "".join(map(str, digits))
    power = len(text) + exponent - 1
    text = text.rstrip("0")
    return (
        ("-" if sign else "")
        + text[0]
        + ("." + text[1:] if len(text) > 1 else "")
        + f"e{power:+d}"
    )


def _values(quantity):
    if quantity.get("shape") != [3, 3] or len(quantity.get("data", [])) != 9:
        raise GateError("nine-entry Cartesian CSD required")
    return [(_decimal(v["real"]), _decimal(v["imag"])) for v in quantity["data"]]


def _quantity(values):
    return {
        "shape": [3, 3],
        "data": [{"real": _encode(a), "imag": _encode(b)} for a, b in values],
    }


def _case(context):
    document = context["frozen_fixture"]
    rows = document.get("science", document.get("fixtures", []))
    selected = [r for r in rows if r["fixture_id"] == context["key"]["fixture_id"]]
    if len(selected) != 1:
        raise GateError("exact one frozen acoustic fixture required")
    return selected[0]


def _volume_kernel(x):
    """Entire small-x series; range-reduced sine otherwise, no sinc/Bessel.

    Decimal90, pi100 and term floor1e-85 are fixed source-only arithmetic
    choices. Actual frozen x<=46; omitted sine term is below1e-85 and pi
    representation error below1e-99, far below the frozen binary64 budget.
    """
    if not x:
        return Decimal(0)
    if x < 0 or x > 46:
        raise GateError("outside reviewed finite acoustic kernel range")
    if x < Decimal("0.5"):
        term = x * x / 6
        total = term
        for n in range(1, 100):
            term *= -x * x / ((2 * n + 2) * (2 * n + 3))
            total += term
            if abs(term) < Decimal("1e-85"):
                return total
    else:
        y = x % (2 * PI)
        if y > PI:
            y -= 2 * PI
        term = y
        total = term
        for n in range(1, 100):
            term *= -y * y / ((2 * n) * (2 * n + 1))
            total += term
            if abs(term) < Decimal("1e-85"):
                return 1 - total / x
    raise GateError("acoustic kernel arithmetic did not converge")


def _assess_residual(context, left, right, reference=None, key=None):
    """Keep exactly N_ref/U/B from the admitted reference for every gate."""
    reference = reference or context["reference"]
    key = key or context["key"]
    center = _values(reference["quantity"])
    shifted = [
        (a + p - q, b + u - v)
        for (a, b), (p, u), (q, v) in zip(center, left, right, strict=True)
    ]
    budget = context["frozen_budget"]
    result = context["kernel"].assess_numeric(
        _quantity(shifted),
        reference["quantity"],
        context["source_scales"][key_id(key)],
        budget["absolute_rate"],
        budget["relative_rate"],
        reference["uncertainty"],
    )
    return result["numeric_accepted"], result


def _g_squared(context, production):
    key = context["key"]
    other = dict(
        key, representation="SI" if key["representation"] == "Gfree" else "Gfree"
    )
    other_ref = context["all_references"][key_id(other)]
    other_values = _values(context["actual"](other))
    g2 = _decimal(_case(context)["inputs"]["g_si"]) ** 2
    if key["representation"] == "SI":
        return _assess_residual(
            context, production, [(a * g2, b * g2) for a, b in other_values]
        )
    return _assess_residual(
        context,
        other_values,
        [(a * g2, b * g2) for a, b in production],
        other_ref,
        other,
    )


def _diagnostic(context):
    pin = context["reference_artifacts"].get("method-diagnostics.json")
    if not pin:
        raise GateError("sealed acoustic method diagnostics required")
    rows = strict_json(verify_artifact(pin))
    key = context["key"]
    matches = [
        r
        for r in rows
        if r["fixture_id"] == key["fixture_id"]
        and r["representation"] == key["representation"]
    ]
    if len(matches) != 1:
        raise GateError("unique sealed acoustic diagnostic required")
    row = matches[0]
    scale = _decimal(context["source_scales"][key_id(key)])
    valid = (
        row.get("precision_dps") == [60, 80]
        and row.get("resolutions") == [1, 2, 4]
        and row.get("alternate_gate") is True
        and 0 <= _decimal(row["alternate_disagreement"]) <= scale * Decimal("1e-31")
    )
    return valid, {"artifact": pin, "diagnostic": row}


def assess(context):
    key = context["key"]
    family = key["family"]
    if (
        family not in CHECKS
        or tuple(context["frozen_procedure"]["connection_checks"]) != CHECKS[family]
    ):
        raise GateError("exact frozen acoustic connection contract required")
    with localcontext() as arithmetic:
        arithmetic.prec = 90
        case = _case(context)
        if context["reference"]["inputs"] != case["inputs"]:
            raise GateError(
                "admitted reference physical tuple differs from frozen input"
            )
        p = {n: _decimal(v) for n, v in case["inputs"].items()}
        if (
            min(p["frequency_hz"], p["sound_speed_m_s"], p["g_si"]) <= 0
            or p["pressure_psd_pa2_hz"] < 0
        ):
            raise GateError("invalid frozen acoustic physical input")
        production = _values(context["production"])
        base = (
            p["pressure_psd_pa2_hz"] / (p["sound_speed_m_s"] * p["frequency_hz"]) ** 2
        )
        if key["representation"] == "SI":
            base *= p["g_si"] ** 2
        elif key["representation"] != "Gfree":
            raise GateError("unknown acoustic representation")
        gpass, gevidence = _g_squared(context, production)
        if family == "A1":
            if case["pressure_convention"] != "surface_total":
                raise GateError("rigid surface total pressure convention required")
            target = [(Decimal(0), Decimal(0)) for _ in range(9)]
            for index, coefficient in [(0, 1), (4, 1), (8, 2)]:
                target[index] = (base * coefficient / 3, Decimal(0))
            normalization, nevidence = _assess_residual(context, production, target)
            relation = [(Decimal(0), Decimal(0)) for _ in range(9)]
            for index, coefficient in [(0, 1), (4, 1), (8, 2)]:
                relation[index] = (production[0][0] * coefficient, Decimal(0))
            components, cevidence = _assess_residual(context, production, relation)
            gates = dict(
                zip(
                    CHECKS[family],
                    [normalization, gpass, components, normalization],
                    strict=True,
                )
            )
            evidence = {
                "positive_air_side_limit": (
                    "J=q/k²,Jprime=q²/k²; hemisphere covariance diag(1,1,2)/3"
                ),
                "pressure": (
                    "p_surface=2*p_incident; given PSD is total surface, "
                    "incident PSD=surfacePSD/4"
                ),
                "normalization": nevidence,
                "component_total": cevidence,
                "G_squared": gevidence,
            }
        else:
            if case["pressure_convention"] != "cavern":
                raise GateError(
                    "published centered cavern pressure convention required"
                )
            x = 2 * PI * p["frequency_hz"] * p["radius_m"] / p["sound_speed_m_s"]
            kernel = _volume_kernel(x)
            diagonal = 4 * base * kernel**2 / 3
            target = [
                (diagonal if i in (0, 4, 8) else Decimal(0), Decimal(0))
                for i in range(9)
            ]
            prefactor, pevidence = _assess_residual(context, production, target)
            mean = sum(production[i][0] for i in (0, 4, 8)) / 3
            isotropic = [
                (mean if i in (0, 4, 8) else Decimal(0), Decimal(0)) for i in range(9)
            ]
            direction, devidence = _assess_residual(context, production, isotropic)
            alternate, aevidence = _diagnostic(context)
            gates = dict(
                zip(
                    CHECKS[family],
                    [prefactor, alternate, direction, gpass],
                    strict=True,
                )
            )
            evidence = {
                "claim": context["frozen_procedure"]["claim"],
                "kernel": _encode(kernel),
                "x": _encode(x),
                "spherical_volume_prefactor": pevidence,
                "radial_angular": aevidence,
                "direction_average": devidence,
                "G_squared": gevidence,
            }
        return {"gates": gates, "evidence": evidence}
