"""Finite frozen seismic connection checks, after controller authorization.

Source claims establish applicability only. They never replace numerical
checks, admitted references, or the controller's production authority.
"""

from decimal import Decimal, localcontext
from fractions import Fraction

from .transport import GateError, key_id, strict_json, verify_artifact

CHECKS = {
    "R1": (
        (
            "Fundamental Rayleigh stable isotropic halfspace; "
            "vertical surface displacement one-sided input"
        ),
        "Independent azimuth average diag(1/2,1/2,1)",
        "Harms horizontal PSD Eq112 independent check",
        "Positive-side h->0+ limit, exp(-2kh) attenuation",
        "Offdiagonal zero, units and G squared",
    ),
    "B2": (
        "Pure P and S plus eta_P 0,.2,.5,.8,1",
        "Small and intermediate ka with spherical j1(ka)/(ka)",
        "Small-cavity acceleration limit and both incoherent S polarizations",
        "P-S cross-CSD=0 as model declaration",
        "No elastic scattering; units and G squared",
    ),
}
PINS = {
    "family",
    "production_commit",
    "production_tree",
    "frozen_artifacts_sha256",
    "reference_seal_sha256",
    "reference_records_sha256",
    "expected_keys_sha256",
    "source_files",
}


def verify_source_claims(context, family):
    """Verify controller-bound, reviewer-issued applicability claims by bytes."""
    expected = context["production_source_pins"]
    packet = context["production_source_reviews"]
    if (
        set(expected) != PINS
        or expected["family"] != family
        or set(packet) != {"admission", "reviews"}
    ):
        raise GateError("production source binding schema mismatch")
    admission = strict_json(verify_artifact(packet["admission"]))
    if (
        set(admission) != {"schema", "family", "decision", "pins", "reviews"}
        or admission["schema"] != "gen1-stochastic-production-source-admission-v1"
        or admission["family"] != family
        or admission["decision"] != "PASS"
        or admission["pins"] != expected
        or admission["reviews"] != packet["reviews"]
        or set(packet["reviews"]) != {"science", "numerical", "provenance"}
    ):
        raise GateError("production source admission mismatch")
    identities = set()
    proofs = {}
    for role, pin in packet["reviews"].items():
        review = strict_json(verify_artifact(pin))
        if (
            set(review)
            != {
                "schema",
                "family",
                "role",
                "reviewer_id",
                "decision",
                "pins",
                "claim_attestations",
                "original_review",
            }
            or review["schema"] != "gen1-stochastic-production-source-review-v1"
            or review["family"] != family
            or review["role"] != role
            or review["decision"] != "PASS"
            or review["pins"] != expected
            or not isinstance(review["reviewer_id"], str)
            or not review["reviewer_id"]
        ):
            raise GateError("actual production source review mismatch")
        identities.add(review["reviewer_id"])
        verify_artifact(review["original_review"])
        claims = review["claim_attestations"]
        if role != "science":
            if claims != {}:
                raise GateError("non-science reviewer cannot issue model claim")
        else:
            if set(claims) != set(CHECKS[family]):
                raise GateError("exact frozen source claims required")
            for name, claim in claims.items():
                if (
                    set(claim) != {"decision", "scope", "proof_report"}
                    or claim["decision"] != "PASS"
                    or claim["scope"] != "SOURCE_MODEL_BINDING_ONLY"
                ):
                    raise GateError("source model proof is not numeric authority")
                verify_artifact(claim["proof_report"])
                proofs[name] = claim["proof_report"]
    if len(identities) != 3:
        raise GateError("three independent source reviewers required")
    return proofs


PI = Decimal(
    "3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679"
)


def _number(value):
    if not isinstance(value, str):
        raise GateError("finite decimal string required")
    result = Decimal(value)
    if not result.is_finite():
        raise GateError("finite decimal required")
    return result


def _values(quantity):
    if (
        set(quantity) != {"shape", "data"}
        or quantity["shape"] != [3, 3]
        or len(quantity["data"]) != 9
    ):
        raise GateError("full Cartesian seismic CSD required")
    values = []
    for value in quantity["data"]:
        if set(value) != {"real", "imag"}:
            raise GateError("complex decimal quantity schema required")
        values.append((_number(value["real"]), _number(value["imag"])))
    return values


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


def _quantity(values):
    return {
        "shape": [3, 3],
        "data": [{"real": _encode(a), "imag": _encode(b)} for a, b in values],
    }


def _residual(context, left, right, key=None, error=Decimal(0)):
    """Retain admitted N_ref/U/B; conservatively check both arithmetic margins."""
    key = key or context["key"]
    reference = context["all_references"][key_id(key)]
    center = _values(reference["quantity"])
    differences = [
        (a + p - q, b + u - v)
        for (a, b), (p, u), (q, v) in zip(center, left, right, strict=True)
    ]
    budget = context["frozen_budget"]
    results = []
    for sign in (-1, 1) if error else (1,):
        shifted = [(a + sign * error, b) for a, b in differences]
        results.append(
            context["kernel"].assess_numeric(
                _quantity(shifted),
                reference["quantity"],
                context["source_scales"][key_id(key)],
                budget["absolute_rate"],
                budget["relative_rate"],
                reference["uncertainty"],
            )
        )
    return all(r["numeric_accepted"] is True for r in results), results


def _diagnostic(context):
    rows = strict_json(
        verify_artifact(context["reference_artifacts"]["reference-diagnostics.json"])
    )
    matches = [r for r in rows if r["key"] == context["key"]]
    if len(matches) != 1 or matches[0]["status"] != "resolved":
        raise GateError("unique resolved sealed seismic diagnostic required")
    method = matches[0]["method"]
    if method["precision_digits"] != [60, 90] or method["orders"] != [24, 48, 96]:
        raise GateError("frozen seismic reference method mismatch")
    scale = _number(context["source_scales"][key_id(context["key"])])
    if not 0 <= _number(method["literal_difference_norm"]) <= scale * Decimal("1e-31"):
        raise GateError("independent literal reference gate failed")
    return method


def _case(context):
    key = context["key"]
    document = context["frozen_fixture"]
    if (
        key not in document["expected_keys"]
        or key["set"] != "science"
        or key["observable"] != "acceleration_csd"
    ):
        raise GateError("exact frozen seismic science key required")
    matches = [r for r in document["science"] if r["fixture_id"] == key["fixture_id"]]
    if len(matches) != 1:
        raise GateError("unique frozen seismic fixture required")
    case = matches[0]
    inputs = dict(case)
    inputs["frequency_hz"] = key["frequency_hz"]
    if (
        key["frequency_hz"] not in case["frequency_hz"]
        or key["target_id"] != case["target_id"]
        or context["reference"]["key"] != key
        or context["reference"]["inputs"] != inputs
        or case["psd_convention"] != "one-sided per-Hz"
        or case["input_assumptions"] != "exact_input"
    ):
        raise GateError("frozen physical tuple/normalization mismatch")
    return case, inputs


def _g_squared(context, production, case):
    key = context["key"]
    other = dict(
        key, representation="SI" if key["representation"] == "Gfree" else "Gfree"
    )
    if other not in context["frozen_fixture"]["expected_keys"]:
        raise GateError("frozen Gfree/SI counterpart missing")
    values = _values(context["actual"](other))
    g2 = _number(case["g_si"]) ** 2
    if key["representation"] == "SI":
        return _residual(context, production, [(a * g2, b * g2) for a, b in values])
    if key["representation"] != "Gfree":
        raise GateError("unknown seismic representation")
    return _residual(context, values, [(a * g2, b * g2) for a, b in production], other)


def _factor(x):
    """Entire j1(x)/x power series; frozen finite ka < 20, no Bessel call."""
    if x < 0 or x > 20:
        raise GateError("outside reviewed finite cavity ka")
    term = Decimal(1) / 3
    total = term
    for n in range(200):
        term *= -x * x / (2 * (n + 1) * (2 * n + 5))
        total += term
        if abs(term) < Decimal("1e-85"):
            return total
    raise GateError("finite spherical source series did not converge")


def assess(context):
    family = context["key"]["family"]
    if (
        family not in CHECKS
        or tuple(context["frozen_procedure"]["required_connections"]) != CHECKS[family]
    ):
        raise GateError("exact frozen seismic claim contract required")
    proofs = verify_source_claims(context, family)
    with localcontext() as arithmetic:
        arithmetic.prec = 100
        case, inputs = _case(context)
        method = _diagnostic(context)
        production = _values(context["production"])
        gpass, gevidence = _g_squared(context, production, case)
        zero = Decimal(0)
        mean = sum(production[i][0] for i in (0, 4, 8)) / 3
        if family == "R1":
            isotropic = [
                (
                    production[8][0] / 2
                    if i in (0, 4)
                    else production[8][0]
                    if i == 8
                    else zero,
                    zero,
                )
                for i in range(9)
            ]
            average, aevidence = _residual(context, production, isotropic)
            literal = _number(method["horizontal_literal"])
            if context["key"]["representation"] == "SI":
                literal *= _number(case["g_si"]) ** 2
            target = [
                (literal * (2 if i == 8 else 1) if i in (0, 4, 8) else zero, zero)
                for i in range(9)
            ]
            harms, hevidence = _residual(context, production, target)
            peers = []
            ignored = {"fixture_id", "height_m"}
            for row in context["frozen_fixture"]["science"]:
                if row["fixture_id"] == case["fixture_id"]:
                    continue
                if {k: v for k, v in row.items() if k not in ignored} == {
                    k: v for k, v in case.items() if k not in ignored
                }:
                    peers.append(row)
            attenuation_evidence = []
            speed = _number(method["rayleigh_speed_m_s"])
            if (
                not 0 < speed < _number(case["v_s_m_s"])
                or not 0
                < (_number(case["v_s_m_s"]) / _number(case["v_p_m_s"])) ** 2
                <= Decimal(".5")
                or _number(case["density_kg_m3"]) < 0
                or _number(case["surface_vertical_displacement_psd_m2_hz"]) < 0
            ):
                raise GateError("stable frozen Rayleigh branch required")
            q = (1 - (speed / _number(case["v_p_m_s"])) ** 2).sqrt()
            vertical_q = (1 - (speed / _number(case["v_s_m_s"])) ** 2).sqrt()
            zeta = (q / vertical_q).sqrt()
            if _number(case["height_m"]) < 0 or q == zeta:
                raise GateError("invalid positive-side Rayleigh limit")
            surface_horizontal = (
                2
                * (PI * _number(case["density_kg_m3"]) * (1 - zeta) / (q - zeta)) ** 2
                * _number(case["surface_vertical_displacement_psd_m2_hz"])
            )
            attenuation_factor = (
                -4
                * PI
                * _number(context["key"]["frequency_hz"])
                * _number(case["height_m"])
                / speed
            ).exp()
            horizontal = surface_horizontal * attenuation_factor
            if context["key"]["representation"] == "SI":
                horizontal *= _number(case["g_si"]) ** 2
            target = [
                (horizontal * (2 if i == 8 else 1) if i in (0, 4, 8) else zero, zero)
                for i in range(9)
            ]
            error = _number(context["source_scales"][key_id(context["key"])]) * Decimal(
                "1e-60"
            )
            attenuation, analytic_evidence = _residual(
                context, production, target, error=error if horizontal else zero
            )
            attenuation_evidence.append(
                {
                    "h0_positive_limit_horizontal": str(surface_horizontal),
                    "exp_minus_2kh": str(attenuation_factor),
                    "numeric": analytic_evidence,
                }
            )
            for peer in peers:
                other = dict(context["key"], fixture_id=peer["fixture_id"])
                if other not in context["frozen_fixture"]["expected_keys"]:
                    raise GateError("attenuation counterpart outside frozen keys")
                values = _values(context["actual"](other))
                exponent = (
                    -4
                    * PI
                    * _number(context["key"]["frequency_hz"])
                    * (_number(case["height_m"]) - _number(peer["height_m"]))
                    / speed
                )
                factor = exponent.exp()
                error = _number(
                    context["source_scales"][key_id(context["key"])]
                ) * Decimal("1e-80")
                accepted, evidence = _residual(
                    context,
                    production,
                    [(a * factor, b * factor) for a, b in values],
                    error=error,
                )
                attenuation &= accepted
                attenuation_evidence.append(
                    {"key": other, "factor": str(factor), "numeric": evidence}
                )
            gates = dict(
                zip(
                    CHECKS[family],
                    [True, average, harms, attenuation, average and gpass],
                    strict=True,
                )
            )
            evidence = {
                "azimuth": aevidence,
                "Harms_horizontal": hevidence,
                "height_attenuation": attenuation_evidence,
                "G_squared": gevidence,
            }
        else:
            if (
                inputs["elastic_scattering"] != "excluded"
                or Fraction(inputs["p_s_cross_csd"]) != 0
            ):
                raise GateError("frozen incoherent unscattered cavity source required")
            p = _number(inputs["p_displacement_psd_m2_hz"])
            s = _number(inputs["s_displacement_psd_m2_hz"])
            radius = _number(inputs["radius_m"])
            rho = _number(inputs["density_kg_m3"])
            frequency = _number(inputs["frequency_hz"])
            vp = _number(inputs["v_p_m_s"])
            vs = _number(inputs["v_s_m_s"])
            if min(p, s, radius, rho) < 0 or min(frequency, vp, vs) <= 0:
                raise GateError("invalid frozen cavity domain")
            diagonal = (4 * PI * rho) ** 2 * (
                4 * p * _factor(2 * PI * frequency * radius / vp) ** 2
                + s * _factor(2 * PI * frequency * radius / vs) ** 2
            )
            if context["key"]["representation"] == "SI":
                diagonal *= _number(case["g_si"]) ** 2
            target = [(diagonal if i in (0, 4, 8) else zero, zero) for i in range(9)]
            error = _number(context["source_scales"][key_id(context["key"])]) * Decimal(
                "1e-80"
            )
            sphere, sevidence = _residual(
                context, production, target, error=error if diagonal else zero
            )
            isotropic = [(mean if i in (0, 4, 8) else zero, zero) for i in range(9)]
            direction, devidence = _residual(context, production, isotropic)
            eta = Fraction(inputs["eta_P"])
            mixed_evidence = []
            mixed = True
            candidates = [
                r
                for r in context["frozen_fixture"]["science"]
                if r["radius_m"] == case["radius_m"]
                and r["density_kg_m3"] == case["density_kg_m3"]
                and r["v_p_m_s"] == case["v_p_m_s"]
                and r["v_s_m_s"] == case["v_s_m_s"]
            ]
            for endpoint in (Fraction(0), Fraction(1)):
                matches = [r for r in candidates if Fraction(r["eta_P"]) == endpoint]
                if len(matches) != 1:
                    mixed = False
                    break
                row = matches[0]
                other = dict(context["key"], fixture_id=row["fixture_id"])
                if other not in context["frozen_fixture"]["expected_keys"]:
                    mixed = False
                    break
                mixed_evidence.append(_values(context["actual"](other)))
            if {Fraction(r["eta_P"]) for r in candidates} != {
                Fraction(v) for v in ("0", ".2", ".5", ".8", "1")
            }:
                mixed = False
            mevidence = []
            if mixed:
                for row in candidates:
                    other = dict(context["key"], fixture_id=row["fixture_id"])
                    if other not in context["frozen_fixture"]["expected_keys"]:
                        mixed = False
                        break
                    eta = Fraction(row["eta_P"])
                    weight = Decimal(eta.numerator) / Decimal(eta.denominator)
                    target = [
                        ((1 - weight) * a + weight * c, (1 - weight) * b + weight * d)
                        for (a, b), (c, d) in zip(*mixed_evidence, strict=True)
                    ]
                    values = (
                        production
                        if other == context["key"]
                        else _values(context["actual"](other))
                    )
                    accepted, numeric = _residual(context, values, target, other)
                    mixed &= accepted
                    mevidence.append({"key": other, "numeric": numeric})
            gates = dict(
                zip(
                    CHECKS[family],
                    [mixed, sphere, sphere and direction, True, sphere and gpass],
                    strict=True,
                )
            )
            evidence = {
                "spherical_kernel": sevidence,
                "direction_average": devidence,
                "incoherent_eta": mevidence,
                "G_squared": gevidence,
                "small_limit": (
                    "j1(x)/x entire series starts1/3; both S polarizations "
                    "require reviewed source proof"
                ),
            }
        return {
            "gates": {
                name: bool(value and name in proofs) for name, value in gates.items()
            },
            "evidence": {"source_proofs": proofs, **evidence},
        }
