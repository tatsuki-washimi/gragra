"""T1 frozen model connections; production executes only via authorized callback.

The adapter owns numerical comparison, units/G² and the science/AUX delivery
split. This plugin attests full binary64 production intermediates and checks
the frozen complete input/purpose contract. It is not an atmospheric validator.
"""

from __future__ import annotations

import hashlib
import math
import sys
from fractions import Fraction
from pathlib import Path

from .transport import GateError, key_id, strict_json

NAMES = (
    "density_kg_m3",
    "temperature_k",
    "structure_coefficient_k_m_neg_p_over_2",
    "structure_exponent",
    "wind_speed_m_s",
    "minimum_approach_m",
    "wind_axis_angle_rad",
    "frequency_hz",
)
NORMALIZATION = (
    "one-sided per Hz; A1 factor2, covariance sigma²-DT/2; Taylor uniform "
    "horizontal flow; half-plane measure r dr dpsi; detector response excluded"
)
MODEL_ERROR = "UNQUANTIFIED_EXCLUDED_FROM_NUMERICAL_U; not atmospheric validation"
MODULE = "gragra.analytic.stochastic_temperature"
FULL = "uniform_advection_temperature_acceleration_psd_g_free"
GEOMETRY = "uniform_advection_geometry_factor"
SCALED = "_scaled_geometry"
FIELDS = {
    FULL: (
        "rho",
        "temp",
        "c_t",
        "p",
        "wind",
        "r_min",
        "theta",
        "frequencies",
        "omega",
        "x",
        "geometry",
        "result",
    ),
    GEOMETRY: ("values", "theta", "angular_scaled", "result"),
    SCALED: (
        "values",
        "theta",
        "k0e",
        "k1e",
        "k2e",
        "parallel_scaled",
        "perpendicular_scaled",
        "angular_scaled",
    ),
}
OPTIONAL_FULL = ("coefficient", "direct", "scaled", "log_coefficient")


def _scalar(quantity):
    if set(quantity) != {"shape", "data"} or quantity["shape"] != []:
        raise GateError("T1 scalar full PSD required")
    if len(quantity["data"]) != 1 or set(quantity["data"][0]) != {"real", "imag"}:
        raise GateError("T1 one complex scalar required")
    if not all(isinstance(v, str) for v in quantity["data"][0].values()):
        raise GateError("T1 finite decimal string quantity required")
    values = [Fraction(quantity["data"][0][name]) for name in ("real", "imag")]
    if values[1] or values[0] < 0:
        raise GateError("T1 real nonnegative full PSD required")
    return values[0]


def _float(value):
    # Genuine production arrays must carry binary64. A scalar float is also
    # binary64; integer/object/complex/nonfinite intermediates are refused.
    if hasattr(value, "dtype"):
        if value.dtype.kind != "f" or value.dtype.itemsize != 8:
            raise GateError("T1 binary64 intermediate required")
        value = value.tolist()
    if isinstance(value, list):
        if len(value) != 1:
            raise GateError("T1 scalar-frequency intermediate required")
        value = value[0]
    if type(value) is not float or not math.isfinite(value):
        raise GateError("T1 finite binary64 intermediate required")
    return value


def _pi_bounds():
    def atan_bounds(q):
        lower = sum(
            (Fraction((-1) ** j, (2 * j + 1) * q ** (2 * j + 1)) for j in range(16)),
            Fraction(),
        )
        return lower, lower + Fraction(1, 33 * q**33)

    a, b = atan_bounds(5), atan_bounds(239)
    return 16 * a[0] - 4 * b[1], 16 * a[1] - 4 * b[0]


def _contract(context):
    document, procedure = context["frozen_fixture"], context["frozen_procedure"]
    key = context["key"]
    if key not in document["expected_keys"] or key["family"] != "T1":
        raise GateError("unknown frozen T1 key")
    cases = [c for c in document["fixtures"] if c["fixture_id"] == key["fixture_id"]]
    if len(cases) != 1:
        raise GateError("unique frozen T1 complete fixture required")
    case = cases[0]
    if (
        case["fixture_id"] in ("t1-x400-amplified", "t1-below-domain")
        and key["set"] != "auxiliary"
    ):
        raise GateError("T1 mandatory numerical AUX cannot become physical science")
    reference = context["reference"]
    if (
        reference["key"] != key
        or reference["inputs"] != case["inputs"]
        or reference["source_pins"] != document["source_pins"]
        or reference["input_assumptions"] != "exact_input"
        or reference["input_convention"]
        != "one_sided_per_Hz_structure_function_coefficient_cT_squared"
        or case["frame"] != "horizontal uniform wind projected axis"
        or case["psd_convention"] != "one_sided_per_hz"
        or case["input_assumptions"] != "exact_input"
        or case["purpose"] != "formal-science"
        or case["set"] != key["set"]
        or case["target_id"] != key["target_id"]
        or key["target_id"] != "projection"
        or key["observable"] != "projected_acceleration_psd"
        or case["inputs"]["frequency_hz"] != key["frequency_hz"]
        or procedure["normalization"] != NORMALIZATION
        or procedure["domain"]["model_approximation_error"] != MODEL_ERROR
    ):
        raise GateError("T1 complete model/normalization contract mismatch")
    if set(case["inputs"]) != set(NAMES) | {"g_si"}:
        raise GateError("T1 exact complete physical tuple required")
    if not all(isinstance(v, str) for v in case["inputs"].values()):
        raise GateError("T1 exact decimal string inputs required")
    inputs = {name: Fraction(case["inputs"][name]) for name in (*NAMES, "g_si")}
    if (
        min(
            inputs[n]
            for n in (
                "temperature_k",
                "wind_speed_m_s",
                "minimum_approach_m",
                "frequency_hz",
            )
        )
        <= 0
        or min(
            inputs[n]
            for n in ("density_kg_m3", "structure_coefficient_k_m_neg_p_over_2")
        )
        < 0
        or not 0 < inputs["structure_exponent"] < 2
        or inputs["g_si"] != Fraction("6.6743e-11")
    ):
        raise GateError("T1 mathematical model domain mismatch")
    pin = context["reference_artifacts"]["purpose-records.json"]
    raw = Path(pin["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin["sha256"]:
        raise GateError("T1 sealed purpose bytes mismatch")
    purposes = strict_json(raw)
    for rep in ("Gfree", "SI"):
        paired_key = dict(key, representation=rep)
        peer = context["all_references"].get(key_id(paired_key))
        if (
            peer is None
            or peer["key"] != paired_key
            or peer["inputs"] != case["inputs"]
            or peer["procedure_hash"] != reference["procedure_hash"]
            or peer["source_pins"] != reference["source_pins"]
        ):
            raise GateError("T1 both frozen representations required")
        matches = [m for m in purposes if m.get("key") == paired_key]
        expected = {
            "purpose": "formal-science",
            "set": case["set"],
            "key": paired_key,
            "scientific_input": {
                "frequency_hz": key["frequency_hz"],
                "frame": case["frame"],
                "psd_convention": case["psd_convention"],
                "inputs": case["inputs"],
                "source_pins": document["source_pins"],
                "procedure_hash": reference["procedure_hash"],
            },
        }
        if matches != [expected]:
            raise GateError("T1 frozen four-field purpose/complete input mismatch")
    pi_lower, pi_upper = _pi_bounds()
    multiplier = (
        2
        * inputs["frequency_hz"]
        * inputs["minimum_approach_m"]
        / inputs["wind_speed_m_s"]
    )
    x_lower, x_upper = pi_lower * multiplier, pi_upper * multiplier
    lower, upper = map(Fraction, document["physical_x_range"])
    if lower != Fraction(procedure["domain"]["x_min"]):
        raise GateError("T1 frozen x_min mismatch")
    physical = key["set"] == "science"
    if key["set"] not in ("science", "auxiliary"):
        raise GateError("T1 purpose set required")
    domain = x_lower > 0 and (not physical or x_lower >= lower and x_upper <= upper)
    return case, inputs, domain, physical, (str(x_lower), str(x_upper))


def _mask(value):
    if hasattr(value, "dtype"):
        if value.dtype.kind != "b":
            raise GateError("T1 boolean tail branch mask required")
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if type(value) is not bool:
        raise GateError("T1 scalar-frequency tail mask required")
    return value


def _observed(actual_fresh, key):
    if sys.gettrace() is not None:
        raise GateError("T1 intermediate audit requires exclusive trace ownership")
    captures = {name: [] for name in FIELDS}

    def trace(frame, event, _arg):
        name = frame.f_code.co_name
        if frame.f_globals.get("__name__") != MODULE or name not in FIELDS:
            return None
        if event == "return":
            if not set(FIELDS[name]) <= set(frame.f_locals):
                raise GateError("missing T1 pinned production local")
            row = {n: _float(frame.f_locals[n]) for n in FIELDS[name]}
            if name == FULL:
                row["tail"] = _mask(frame.f_locals["tail"])
                for n in OPTIONAL_FULL:
                    if n in frame.f_locals:
                        row[n] = _float(frame.f_locals[n])
            captures[name].append(row)
        return trace

    sys.settrace(trace)
    try:
        result = actual_fresh(key)
    finally:
        sys.settrace(None)
    if len(captures[FULL]) != 1 or len(captures[GEOMETRY]) != 1:
        raise GateError("missing genuine T1 full production intermediates")
    full = captures[FULL][0]
    expected_helpers = 1 + int(full["tail"] and full["rho"] > 0 and full["c_t"] > 0)
    if len(captures[SCALED]) != expected_helpers:
        raise GateError("missing or ambiguous genuine T1 scaled helper paths")
    if full["rho"] == 0 or full["c_t"] == 0:
        branch = "zero"
    elif full["tail"]:
        if not {"scaled", "log_coefficient"} <= set(full):
            raise GateError("missing T1 combined logarithmic tail locals")
        branch = "log_tail"
    else:
        if not {"coefficient", "direct"} <= set(full):
            raise GateError("missing T1 direct coefficient/result locals")
        branch = "direct"
    return result, captures, branch


def assess(context):
    """Return four frozen-purpose gates; never confer P authority."""
    case, inputs, domain, physical, x_bounds = _contract(context)
    observed, captures, branch = _observed(context["actual_fresh"], context["key"])
    value = _scalar(observed)
    if value != _scalar(context["production"]):
        raise GateError("T1 fresh observed path differs from compared production")
    full, geometry = captures[FULL][0], captures[GEOMETRY][0]
    rate = float(Fraction(context["frozen_budget"]["relative_rate"]))

    def consistent(a, b):
        return (
            math.isfinite(a)
            and math.isfinite(b)
            and abs(a - b) <= rate * max(abs(a), abs(b))
        )

    input_names = ("rho", "temp", "c_t", "p", "wind", "r_min", "theta", "frequencies")
    normalization = (
        all(
            full[n] == float(inputs[k]) for n, k in zip(input_names, NAMES, strict=True)
        )
        and consistent(full["omega"], 2 * math.pi * full["frequencies"])
        and consistent(full["x"], full["omega"] * full["r_min"] / full["wind"])
    )
    whole_path = (
        consistent(full["geometry"], geometry["result"])
        and consistent(geometry["values"], full["x"])
        and consistent(geometry["theta"], full["theta"])
        and consistent(
            geometry["result"],
            math.exp(-2 * geometry["values"]) * geometry["angular_scaled"],
        )
    )
    for scaled in captures[SCALED]:
        whole_path &= (
            consistent(scaled["values"], full["x"])
            and consistent(scaled["theta"], full["theta"])
            and consistent(
                scaled["parallel_scaled"], scaled["k1e"] ** 2 - scaled["k0e"] ** 2
            )
            and consistent(
                scaled["perpendicular_scaled"],
                scaled["k0e"] * scaled["k2e"] - scaled["k1e"] ** 2,
            )
            and consistent(
                scaled["angular_scaled"],
                math.cos(scaled["theta"]) ** 2 * scaled["parallel_scaled"]
                + 0.5 * math.sin(scaled["theta"]) ** 2 * scaled["perpendicular_scaled"],
            )
        )
    combination = {}
    if branch == "direct":
        coefficient = (
            8
            * math.pi**2
            * math.gamma(full["p"] + 2)
            * math.sin(math.pi * full["p"] / 2)
            * (full["rho"] * full["r_min"] * full["c_t"] / full["temp"]) ** 2
            * full["wind"] ** full["p"]
        )
        normalization &= consistent(full["coefficient"], coefficient)
        whole_path &= consistent(
            full["direct"],
            full["coefficient"]
            * full["omega"] ** (-(full["p"] + 1))
            * full["geometry"],
        ) and consistent(full["result"], full["direct"])
        combination["coefficient"] = full["coefficient"].hex()
    elif branch == "log_tail":
        expected_log = (
            math.log(
                8
                * math.pi**2
                * math.gamma(full["p"] + 2)
                * math.sin(math.pi * full["p"] / 2)
            )
            + 2
            * (
                math.log(full["rho"])
                + math.log(full["r_min"])
                + math.log(full["c_t"])
                - math.log(full["temp"])
            )
            + full["p"] * math.log(full["wind"])
        )
        normalization &= consistent(full["log_coefficient"], expected_log)
        whole_path &= consistent(full["scaled"], captures[SCALED][-1]["angular_scaled"])
        terms = {
            "log_coefficient": full["log_coefficient"],
            "frequency_term": -(full["p"] + 1) * math.log(full["omega"]),
            "attenuation_term": -2 * full["x"],
            "log_scaled": math.log(full["scaled"]),
        }
        exponent = (
            terms["log_coefficient"]
            + terms["frequency_term"]
            + terms["attenuation_term"]
            + terms["log_scaled"]
        )
        if not all(math.isfinite(v) for v in (*terms.values(), exponent)):
            raise GateError("T1 finite combined logarithmic terms/exponent required")
        whole_path &= consistent(full["result"], math.exp(exponent))
        combination = {n: v.hex() for n, v in terms.items()}
        combination["reconstructed_combined_exponent"] = exponent.hex()
    else:
        whole_path &= full["result"] == 0
    factor = (
        float(inputs["g_si"]) ** 2 if context["key"]["representation"] == "SI" else 1
    )
    whole_path &= float(value) == full["result"] * factor
    outputs = {context["key"]["representation"]: observed["data"][0]["real"]}
    stress = case["fixture_id"] == "t1-x400-amplified"
    scaling = True
    if stress:
        for rep in ("Gfree", "SI"):
            quantity, peer_captures, peer_branch = _observed(
                context["actual_fresh"], dict(context["key"], representation=rep)
            )
            peer_value = _scalar(quantity)
            binary = float(peer_value)
            scaling &= math.isfinite(binary) and binary > 0 and peer_value > 0
            outputs[rep] = quantity["data"][0]["real"]
            whole_path &= peer_captures == captures and peer_branch == branch
            factor = float(inputs["g_si"]) ** 2 if rep == "SI" else 1
            whole_path &= float(peer_value) == full["result"] * factor
    encoded = {
        name: [
            {k: v if isinstance(v, bool) else v.hex() for k, v in row.items()}
            for row in rows
        ]
        for name, rows in captures.items()
    }
    return {
        "gates": {
            "domain": bool(domain),
            "whole_path": bool(whole_path),
            "normalization": bool(normalization),
            "scaling_contract": bool(scaling),
        },
        "evidence": {
            "scope": "conditional_A7_finite_model"
            if physical
            else "numerical_auxiliary",
            "physical_eligible": bool(physical and domain),
            "model_approximation_error": MODEL_ERROR,
            "actual_x_bounds": list(x_bounds),
            "mandatory_x400": stress,
            "outputs": outputs,
            "intermediates": encoded,
            "branch": branch,
            "combination": combination,
            "trace_restored": sys.gettrace() is None,
            "numerical_policy": "frozen relative_rate for production consistency only",
            "log_exponent_evidence": (
                "reconstructed from observed production locals; "
                "expression is inline in reviewed source"
            ),
        },
    }
