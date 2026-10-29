"""Frozen A2 connection gates, evaluated only by an authorized comparator.

This module imports neither production nor the reference generator. Quantities
arrive through the comparator's authority-bound, cached ``actual`` callback.
"""

import hashlib
import math
import sys
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

# Pure transport/arithmetic only; no production or oracle import.
from ...common.transport import GateError, key_id, strict_json, verify_artifact
from .reader_contract import (
    RUNTIME_CONNECTION_GATES,
    validate_science_binding_claim_keys,
)

GATES = RUNTIME_CONNECTION_GATES


def _decimal_exact(value):
    """Render an exact terminating rational; never round a residual downward."""
    value = Fraction(value)
    denominator = value.denominator
    twos = fives = 0
    while denominator % 2 == 0:
        denominator //= 2
        twos += 1
    while denominator % 5 == 0:
        denominator //= 5
        fives += 1
    if denominator != 1:
        raise ValueError("nonterminating residual transport")
    places = max(twos, fives)
    numerator = value.numerator * 2 ** (places - twos) * 5 ** (places - fives)
    if numerator == 0:
        return "0e+0"
    digits = str(abs(numerator))
    exponent = len(digits) - 1 - places
    rest = digits[1:].rstrip("0")
    mantissa = digits[0] + ("." + rest if rest else "")
    return ("-" if numerator < 0 else "") + mantissa + f"e{exponent:+d}"


def neighbor_check(
    kernel, plus, minus, ref_plus, ref_minus, u_plus, u_minus, b_plus, b_minus
):
    """Enclose the whole residual at 220/260 digits with explicit arithmetic U."""
    if min(u_plus, u_minus, b_plus, b_minus) < 0:
        raise ValueError("negative uncertainty or budget")
    shape = plus["shape"]
    if any(q["shape"] != shape for q in (minus, ref_plus, ref_minus)):
        raise ValueError("neighbor shape mismatch")
    quantities = [kernel.quantity_values(q) for q in (plus, minus, ref_plus, ref_minus)]
    data = []
    for p, m, r, s in zip(*quantities, strict=True):
        data.append(
            {
                "real": _decimal_exact(p[0] - m[0] - r[0] + s[0]),
                "imag": _decimal_exact(p[1] - m[1] - r[1] + s[1]),
            }
        )
    residual = {"shape": shape, "data": data}
    intervals = [kernel.norm_bounds(residual, digits=n) for n in (220, 260)]
    # Upper minus lower is the explicit decision arithmetic allowance. The
    # residual lower endpoint plus this allowance equals its outward upper.
    decisions = [
        lo + u_plus + u_minus + (hi - lo) <= b_plus + b_minus for lo, hi in intervals
    ]
    lo, hi = intervals[-1]
    return {
        "accepted": decisions[0] == decisions[1] and decisions[1],
        "residual_lower": str(lo),
        "residual_upper": str(hi),
        "decision_arithmetic": str(hi - lo),
        "uncertainty_both": str(u_plus + u_minus),
        "budget_both": str(b_plus + b_minus),
        "precision_decisions": decisions,
    }


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
MODULE = "gragra.analytic.stochastic_acoustic"
SOURCE = "src/gragra/analytic/stochastic_acoustic.py"
CONVENTION = "surface_total_one_sided_per_hz"


def _source_binding(context):
    """Actual reviewer artifact transport; source proof grants no numeric PASS."""
    source = context["production_source_pins"]
    packet = context["production_source_reviews"]
    if (
        set(source)
        != {
            "family",
            "production_commit",
            "production_tree",
            "frozen_artifacts_sha256",
            "reference_seal_sha256",
            "reference_records_sha256",
            "expected_keys_sha256",
            "source_files",
        }
        or source["family"] != "A2"
        or set(packet) != {"admission", "reviews"}
    ):
        raise GateError("A2 exact source binding packet required")
    admission = strict_json(verify_artifact(packet["admission"]))
    if (
        set(admission) != {"schema", "family", "decision", "pins", "reviews"}
        or admission["schema"] != "gen1-stochastic-production-source-admission-v1"
        or admission["family"] != "A2"
        or admission["decision"] != "PASS"
        or admission["pins"] != source
        or admission["reviews"] != packet["reviews"]
        or set(packet["reviews"]) != {"science", "numerical", "provenance"}
        or SOURCE not in source["source_files"]
    ):
        raise GateError("A2 source-model binding is missing; not numeric gate proof")
    identities = set()
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
            or review["family"] != "A2"
            or review["role"] != role
            or review["decision"] != "PASS"
            or review["pins"] != source
            or not isinstance(review["reviewer_id"], str)
            or not review["reviewer_id"]
        ):
            raise GateError("A2 actual source reviewer binding mismatch")
        identities.add(review["reviewer_id"])
        verify_artifact(review["original_review"])
        claims = review["claim_attestations"]
        if role != "science":
            if claims != {}:
                raise GateError("A2 non-science model claim")
        else:
            validate_science_binding_claim_keys(claims)
            for claim in claims.values():
                if (
                    set(claim) != {"decision", "scope", "proof_report"}
                    or claim["decision"] != "PASS"
                    or claim["scope"] != "SOURCE_MODEL_BINDING_ONLY"
                ):
                    raise GateError("A2 source claims cannot grant numeric authority")
                verify_artifact(claim["proof_report"])
    if len(identities) != 3:
        raise GateError("A2 three independent source reviewers required")


def frozen_pairs(document):
    """Six exact frozen nominal neighbor pairs; no nearest-row substitution."""
    boundaries = document["branch_boundaries"]
    epsilons = document["branch_relative_eps"]
    if list(map(Fraction, boundaries)) != [Fraction(3, 4), Fraction(18)] or (
        list(map(Fraction, epsilons))
        != [Fraction("1e-8"), Fraction("1e-6"), Fraction("1e-4")]
    ):
        raise GateError("changed frozen branch grid")
    rows = {}
    for row in document["fixtures"]:
        x = Fraction(row["nominal_exact_x"])
        if x in rows:
            raise GateError("duplicate exact nominal neighbor input")
        rows[x] = row
    pairs = []
    for boundary in map(Fraction, boundaries):
        for epsilon in map(Fraction, epsilons):
            lo, hi = boundary * (1 - epsilon), boundary * (1 + epsilon)
            if lo not in rows or hi not in rows:
                raise GateError("missing frozen neighbor")
            pairs.append((rows[hi], rows[lo]))
    return pairs


@lru_cache(maxsize=1)
def _pi_bounds():
    """Alternating rational Machin intervals, not an oracle observable."""

    def arctangent(q):
        value, n = Fraction(), 0
        while True:
            term = Fraction((-1) ** n, (2 * n + 1) * q ** (2 * n + 1))
            value += term
            n += 1
            following = Fraction((-1) ** n, (2 * n + 1) * q ** (2 * n + 1))
            if abs(following) < Fraction("1e-115"):
                return min(value, value + following), max(value, value + following)

    a, b = arctangent(5), arctangent(239)
    return 16 * a[0] - 4 * b[1], 16 * a[1] - 4 * b[0]


def _scalar_key(row):
    return {
        **row["key_base"],
        "target_id": "nominal_x",
        "observable": "isotropic_depth_factor",
        "set": "auxiliary",
        "representation": "Gfree",
    }


def _quantity(kernel, pairs, shape, digits=280):
    """Downward decimal transport plus exact L1 serialization enclosure."""
    data, error = [], Fraction()
    for real, imag in pairs:
        a = kernel.fraction_decimal(real, digits, downward=True)
        b = kernel.fraction_decimal(imag, digits, downward=True)
        error += abs(real - kernel.decimal_fraction(a))
        error += abs(imag - kernel.decimal_fraction(b))
        data.append({"real": a, "imag": b})
    return {"shape": shape, "data": data}, error


def _norm(kernel, pairs, shape, digits):
    q, rounding = _quantity(kernel, pairs, shape, digits + 20)
    lo, hi = kernel.norm_bounds(q, digits=digits)
    return max(Fraction(), lo - rounding), hi + rounding


def _uncertainty(kernel, reference):
    uncertainty = reference["uncertainty"]
    if uncertainty["kind"] != "estimate" or set(uncertainty["components"]) != set(
        COMPONENTS
    ):
        raise GateError("A2 exact nine-component estimate required")
    values = {
        n: kernel.decimal_fraction(v["value"])
        for n, v in uncertainty["components"].items()
    }
    if (
        min(values.values()) < 0
        or any(values[n] != 0 for n in ("root", "tail", "formulation"))
        or sum(values.values()) <= 0
    ):
        raise GateError("A2 nonoverlapping estimated U/zero causes mismatch")
    return sum(values.values(), Fraction())


def _bound(context, key, digits):
    kernel = context["kernel"]
    reference = context["all_references"][key_id(key)]
    lo, _ = kernel.norm_bounds(reference["quantity"], digits=digits)
    budget = context["frozen_budget"]
    return (
        kernel.decimal_fraction(budget["absolute_rate"])
        * kernel.decimal_fraction(context["source_scales"][key_id(key)])
        + kernel.decimal_fraction(budget["relative_rate"]) * lo
    )


def _relation(context, key, left, right):
    """Original N/U/B; separately outward-enclosed exact relation residual."""
    reference = context["all_references"][key_id(key)]
    kernel = context["kernel"]
    uncertainty = _uncertainty(kernel, reference)
    residual = [(a - c, b - d) for (a, b), (c, d) in zip(left, right, strict=True)]
    decisions, intervals = [], []
    for digits in (220, 260):
        lo, hi = _norm(kernel, residual, reference["quantity"]["shape"], digits)
        budget = _bound(context, key, digits)
        decisions.append(hi + uncertainty <= budget and 10 * uncertainty <= budget)
        intervals.append((lo, hi, budget))
    return decisions == [True, True], {
        "precision_decisions": decisions,
        "residual_upper": str(intervals[-1][1]),
        "decision_arithmetic": str(intervals[-1][1] - intervals[-1][0]),
        "public_uncertainty": str(uncertainty),
        "budget_lower": str(intervals[-1][2]),
    }


def _traces(context, rows):
    pin = context["reference_artifacts"].get("stderr.jsonl")
    if not pin:
        raise GateError("sealed A2 method trace artifact required")
    parsed = [
        strict_json(line) for line in verify_artifact(pin).splitlines() if line.strip()
    ]
    ids = [row["key_base"]["fixture_id"] for row in rows]
    if len(parsed) != len(ids) or {r.get("fixture_id") for r in parsed} != set(ids):
        raise GateError("exact complete A2 trace coverage required")
    result = {}
    for record in parsed:
        name = record["fixture_id"]
        if name in result:
            raise GateError("duplicate A2 method trace")
        trace = record["trace"]
        nominal = trace["nominal_exact_x_reference"]
        for part in (trace, nominal):
            if (
                len(part["theta_high80_resolution124"]) != 3
                or len(part["mu_high80_resolution124"]) != 3
                or part["reference_resolved"] is not True
            ):
                raise GateError(
                    "frozen60/80 and both resolution1/2/4 traces unresolved"
                )
            values = [
                part["theta_low60"],
                part["exact_x"],
                part["coordinate_disagreement"],
                *part["theta_high80_resolution124"],
                *part["mu_high80_resolution124"],
            ]
            if any(context["kernel"].decimal_fraction(x) < 0 for x in values):
                raise GateError("negative method trace")
        result[name] = trace
    return result, pin


def _contract(context):
    doc, procedure, budget = (
        context["frozen_fixture"],
        context["frozen_procedure"],
        context["frozen_budget"],
    )
    if (
        context["key"]["family"] != "A2"
        or doc["family"] != "A2"
        or procedure["family"] != "A2"
        or budget["family"] != "A2"
        or Fraction(budget["absolute_rate"]) != Fraction("1e-13")
        or Fraction(budget["relative_rate"]) != Fraction("1e-10")
        or procedure["input_convention"].split(":")[0] != CONVENTION
        or procedure["uncertainty_kind"] != "estimate"
    ):
        raise GateError("exact frozen A2 budget/model required")
    method = procedure["numerical_procedure"]
    if (
        method["dps"] != [60, 80]
        or method["resolution"] != [1, 2, 4]
        or method["digits"] != 40
        or method["maxdegree"] != 8
        or method["formulations"] != ["theta", "mu"]
    ):
        raise GateError("changed frozen A2 numerical procedure")
    rows = doc["fixtures"]
    if len(rows) != 28:
        raise GateError("all28 frozen A2 physical/nominal fixture rows required")
    pairs = frozen_pairs(doc)
    expected = doc["expected_keys"] + doc["expected_scalar_keys"]
    ids = [key_id(k) for k in expected]
    refs = context["all_references"]
    if (
        len(ids) != 84
        or len(set(ids)) != 84
        or set(ids) != set(refs)
        or key_id(context["key"]) not in ids
    ):
        raise GateError("exact84 A2 reference keys required")
    if (
        len(doc["expected_keys"]) != 56
        or len(doc["expected_scalar_keys"]) != 28
        or sum(k["set"] == "science" for k in expected) != 54
        or sum(k["set"] == "auxiliary" for k in expected) != 30
    ):
        raise GateError("A2 54science/30AUX partition mismatch")
    selected = [
        r for r in rows if r["key_base"]["fixture_id"] == context["key"]["fixture_id"]
    ]
    if (
        len(selected) != 1
        or context["frozen_case"] != selected[0]
        or (refs[key_id(context["key"])] != context["reference"])
    ):
        raise GateError("selected frozen A2 case/reference mismatch")
    _source_binding(context)
    traces, pin = _traces(context, rows)
    current = context["reference"]
    for row in rows:
        if (
            row["pressure_convention"] != "surface_total"
            or row["psd_convention"] != CONVENTION
            or row["input_assumptions"] != "exact_input"
            or row["frame"] != "Cartesian underground natural frame z vertical"
        ):
            raise GateError("total rigid-surface pressure/natural frame required")
        inputs = {n: Fraction(v) for n, v in row["inputs"].items()}
        if (
            set(inputs)
            != {
                "depth_m",
                "frequency_hz",
                "g_si",
                "pressure_psd_pa2_hz",
                "sound_speed_m_s",
            }
            or min(inputs["depth_m"], inputs["pressure_psd_pa2_hz"]) < 0
            or min(inputs["frequency_hz"], inputs["sound_speed_m_s"]) <= 0
            or inputs["g_si"] != Fraction("6.6743e-11")
        ):
            raise GateError("invalid complete frozen A2 physical tuple")
        keys = [{**row["key_base"], "representation": rep} for rep in ("Gfree", "SI")]
        keys.append(_scalar_key(row))
        for key in keys:
            identity = key_id(key)
            if identity not in refs or key not in expected:
                raise GateError("missing exact physical/nominal A2 key")
            ref = refs[identity]
            scalar = key["observable"] == "isotropic_depth_factor"
            inputs_expected = (
                {
                    "nominal_exact_x": row["nominal_exact_x"],
                    "frequency_hz": row["inputs"]["frequency_hz"],
                }
                if scalar
                else row["inputs"]
            )
            if (
                ref["key"] != key
                or ref["status"] != "resolved"
                or ref["quantity"] is None
                or ref["quantity"]["shape"] != ([] if scalar else [3, 3])
                or ref["inputs"] != inputs_expected
                or ref["input_convention"]
                != ("dimensionless_exact_nominal_x" if scalar else CONVENTION)
                or ref["units"]
                != (
                    "1"
                    if scalar
                    else (
                        "m^2 s^-4 Hz^-1"
                        if key["representation"] == "SI"
                        else "kg^2 m^-4 Hz^-1"
                    )
                )
                or ref["procedure_hash"] != current["procedure_hash"]
                or ref["source_pins"] != current["source_pins"]
            ):
                raise GateError("A2 exact reference/scalar construction mismatch")
            _uncertainty(context["kernel"], ref)
    return rows, pairs, traces, pin, selected[0]


def _adequacy(context, row, key, trace):
    kernel = context["kernel"]
    ref = context["all_references"][key_id(key)]
    scalar = key["observable"] == "isotropic_depth_factor"
    data = trace["nominal_exact_x_reference"] if scalar else trace
    if (
        Fraction(data["exact_x"])
        != Fraction(row["nominal_exact_x"] if scalar else trace["actual_physical_x"])
        or data[
            "scalar_public_adequacy"
            if scalar
            else key["representation"] + "_public_adequacy"
        ]
        is not True
    ):
        raise GateError("A2 nominal/physical trace/public adequacy mismatch")
    diag = kernel.decimal_fraction(data["coordinate_disagreement"])
    factor = Fraction(1)
    shape = []
    if not scalar:
        p = {n: Fraction(v) for n, v in row["inputs"].items()}
        factor = (
            p["pressure_psd_pa2_hz"] / (p["sound_speed_m_s"] * p["frequency_hz"]) ** 2
        )
        if key["representation"] == "SI":
            factor *= p["g_si"] ** 2
        shape = [3, 3]
        pairs = [
            (factor * diag * (2 if i == 8 else 1), Fraction())
            if i in (0, 4, 8)
            else (Fraction(), Fraction())
            for i in range(9)
        ]
    else:
        pairs = [(diag, Fraction())]
    u = _uncertainty(kernel, ref)
    decisions = []
    for digits in (220, 260):
        _, upper = _norm(kernel, pairs, shape, digits)
        norm, _ = kernel.norm_bounds(ref["quantity"], digits=digits)
        decisions.append(u + upper < norm * Fraction("1e-35"))
    return decisions == [True, True]


def _number(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    if isinstance(value, (bool, list, dict)):
        raise GateError("single finite production scalar expected")
    value = float(value)
    if not math.isfinite(value):
        raise GateError("nonfinite observed A2 production intermediate")
    return value


def _observed(context):
    """Real authorized fresh callback; no import or standalone production call."""
    if sys.gettrace() is not None:
        raise GateError("A2 requires exclusive trace ownership")
    captures = {"depth": [], "scalar": [], "branches": []}
    digest = context["production_source_pins"]["source_files"][SOURCE]

    def trace(frame, event, _arg):
        if frame.f_globals.get("__name__") != MODULE:
            return None
        name = frame.f_code.co_name
        if name not in (
            "isotropic_acoustic_depth_factor",
            "underground_isotropic_acoustic_acceleration_csd_g_free",
            "_isotropic_depth_factor_small",
            "_isotropic_depth_factor_large",
        ):
            return None
        if event == "return":
            path = Path(frame.f_code.co_filename)
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise GateError("observed A2 production source bytes differ")
            if name.startswith("_isotropic_depth_factor_"):
                captures["branches"].append(name.rsplit("_", 1)[1])
                return trace
            fields = (
                ("number",)
                if name == "isotropic_acoustic_depth_factor"
                else (
                    "depth",
                    "c_s",
                    "frequencies",
                    "pressure_psd",
                    "x",
                    "factor",
                    "base",
                )
            )
            captures["scalar" if len(fields) == 1 else "depth"].append(
                {n: _number(frame.f_locals[n]) for n in fields}
            )
        return trace

    sys.settrace(trace)
    try:
        observed = context["actual_fresh"](context["key"])
    finally:
        sys.settrace(None)
    scalar = context["key"]["observable"] == "isotropic_depth_factor"
    if (
        observed != context["production"]
        or len(captures["scalar"]) != 1
        or len(captures["depth"]) != int(not scalar)
    ):
        raise GateError("missing genuine A2 production path or output mismatch")
    return captures


def _captured_evidence(captures):
    """Exact binary64 decimal strings; gate arithmetic retains its floats."""
    result = {"branches": list(captures["branches"])}
    for route in ("depth", "scalar"):
        result[route] = []
        for row in captures[route]:
            encoded = {}
            for name, value in row.items():
                if not isinstance(value, float) or not math.isfinite(value):
                    raise GateError("finite binary64 observation required")
                encoded[name] = _decimal_exact(Fraction.from_float(value))
            result[route].append(encoded)
    return result


def assess(context):
    """Four actual numeric/model gates; source-only review never implies PASS."""
    if not context:
        return {
            "gates": dict.fromkeys(GATES, False),
            "evidence": {
                "status": "unresolved",
                "reason": "missing complete A2 context",
            },
        }
    rows, pairs, traces, trace_pin, selected = _contract(context)
    kernel = context["kernel"]
    value_checks = {}
    adequacy = True
    for row in rows:
        for key in [
            {**row["key_base"], "representation": r} for r in ("Gfree", "SI")
        ] + [_scalar_key(row)]:
            identity = key_id(key)
            ref = context["all_references"][identity]
            result = kernel.assess_numeric(
                context["actual"](key),
                ref["quantity"],
                context["source_scales"][identity],
                context["frozen_budget"]["absolute_rate"],
                context["frozen_budget"]["relative_rate"],
                ref["uncertainty"],
            )
            value_checks[str(identity)] = result["numeric_accepted"]
            adequacy &= _adequacy(
                context, row, key, traces[row["key_base"]["fixture_id"]]
            )
    neighbor = []
    for plus, minus in pairs:
        for rep in ("scalar", "Gfree", "SI"):
            pk = (
                _scalar_key(plus)
                if rep == "scalar"
                else {**plus["key_base"], "representation": rep}
            )
            mk = (
                _scalar_key(minus)
                if rep == "scalar"
                else {**minus["key_base"], "representation": rep}
            )
            rp, rm = (context["all_references"][key_id(k)] for k in (pk, mk))
            result = neighbor_check(
                kernel,
                context["actual"](pk),
                context["actual"](mk),
                rp["quantity"],
                rm["quantity"],
                _uncertainty(kernel, rp),
                _uncertainty(kernel, rm),
                _bound(context, pk, 260),
                _bound(context, mk, 260),
            )
            neighbor.append({"plus_key": pk, "minus_key": mk, **result})
    assert len(neighbor) == 18
    captures = _observed(context)
    scalar = context["key"]["observable"] == "isotropic_depth_factor"
    xpass = True
    observed_x = captures["scalar"][0]["number"]
    branch = (
        []
        if observed_x == 0 or 0.75 <= observed_x <= 18
        else ["small" if observed_x < 0.75 else "large"]
    )
    if scalar:
        xpass &= captures["scalar"][0]["number"] == float(
            Fraction(selected["nominal_exact_x"])
        )
        relation = True
        normalization = {}
        if Fraction(selected["nominal_exact_x"]) == 0:
            relation, normalization = _relation(
                context,
                context["key"],
                kernel.quantity_values(context["production"]),
                [(Fraction(1, 3), Fraction())],
            )
    else:
        p = {n: Fraction(v) for n, v in selected["inputs"].items()}
        fields = captures["depth"][0]
        xpass &= (
            fields["depth"] == float(p["depth_m"])
            and fields["c_s"] == float(p["sound_speed_m_s"])
            and fields["frequencies"] == float(p["frequency_hz"])
            and fields["pressure_psd"] == float(p["pressure_psd_pa2_hz"])
            and fields["x"]
            == 4.0
            * math.pi
            * float(p["frequency_hz"])
            * float(p["depth_m"])
            / float(p["sound_speed_m_s"])
            and captures["scalar"][0]["number"] == fields["x"]
        )
        pi = _pi_bounds()
        multiplier = 4 * p["frequency_hz"] * p["depth_m"] / p["sound_speed_m_s"]
        exact = Fraction(
            traces[selected["key_base"]["fixture_id"]]["actual_physical_x"]
        )
        allowance = Fraction("1e-95") * (1 + abs(exact))
        xpass &= (
            multiplier * pi[0] - allowance <= exact <= multiplier * pi[1] + allowance
        )
        base = (
            float(p["pressure_psd_pa2_hz"])
            * fields["factor"]
            / (float(p["sound_speed_m_s"]) * float(p["frequency_hz"])) ** 2
        )
        xpass &= fields["factor"] >= 0 and fields["base"] == base
        factor = (
            p["g_si"] ** 2 if context["key"]["representation"] == "SI" else Fraction(1)
        )
        expected = [
            (Fraction.from_float(base) * factor * (2 if i == 8 else 1), Fraction())
            if i in (0, 4, 8)
            else (Fraction(), Fraction())
            for i in range(9)
        ]
        relation, normalization = _relation(
            context,
            context["key"],
            kernel.quantity_values(context["production"]),
            expected,
        )
    scaling = []
    for row in rows:
        gf = {**row["key_base"], "representation": "Gfree"}
        si = {**row["key_base"], "representation": "SI"}
        g2 = Fraction(row["inputs"]["g_si"]) ** 2
        left = kernel.quantity_values(context["actual"](si))
        right = [
            (a * g2, b * g2) for a, b in kernel.quantity_values(context["actual"](gf))
        ]
        passed, evidence = _relation(context, si, left, right)
        scaling.append(
            {"fixture_id": row["key_base"]["fixture_id"], "passed": passed, **evidence}
        )
    asymptotic = []
    for x in (Fraction(30), Fraction(100), Fraction(1000)):
        matches = [row for row in rows if Fraction(row["nominal_exact_x"]) == x]
        if len(matches) != 1:
            raise GateError("missing frozen asymptotic diagnostic input")
        key = _scalar_key(matches[0])
        reference = context["all_references"][key_id(key)]
        pr = kernel.quantity_values(context["actual"](key))[0]
        rr = kernel.quantity_values(reference["quantity"])[0]
        multiplier = x**4 / 3
        asymptotic.append(
            {
                "nominal_exact_x": str(x),
                "production_x4_I_over_3": str(pr[0] * multiplier),
                "reference_x4_I_over_3": str(rr[0] * multiplier),
                "full_reference_numeric_accepted": value_checks[str(key_id(key))],
                "leading_law_is_not_oracle_or_uncertainty": True,
            }
        )
    return {
        "gates": {
            "numerical_procedure": bool(
                adequacy
                and all(value_checks.values())
                and all(n["accepted"] for n in neighbor)
                and relation
            ),
            "actual_x_transport": bool(xpass),
            "SI": bool(all(s["passed"] for s in scaling)),
            "coverage": len(value_checks) == 84
            and len(neighbor) == 18
            and len(scaling) == 28,
        },
        "evidence": {
            "status": "assessed",
            "science_records": 54,
            "CSD_records": 56,
            "scalar_AUX_records": 28,
            "surface_CSD_AUX_records": 2,
            "reference_method_trace": trace_pin,
            "reference_public_adequacy": bool(adequacy),
            "value_checks": value_checks,
            "neighbor_checks": neighbor,
            "normalization_check": normalization,
            "representation_checks": scaling,
            "asymptotic_diagnostics": asymptotic,
            "expected_observed_branch": branch,
            "observed_branch_scope": (
                "diagnostic; frozen source binding establishes dispatch"
            ),
            "observed_production_path": _captured_evidence(captures),
            "trace_restored": sys.gettrace() is None,
            "source_binding_scope": (
                "SOURCE_MODEL_BINDING_ONLY; separately ANDed with actual numeric gates"
            ),
            "derivative": "diagnostic only; never an acceptance gate",
        },
    }
