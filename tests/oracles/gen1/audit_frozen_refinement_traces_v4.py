"""Audit saved Gen1 refinement traces without rerunning any oracle fields.

Exact serialized callback results are reconciled with named pairs, trends and
frozen uncertainty components. This module does not import the recorder or any
production module. Explicit subset mode never claims the 1178-case gate.

Row auditing binds separately authenticated anchors to independent inputs and
reconciled journal evidence. Dataset/run orchestration remains a fail-closed
scaffold. No production trust provider or root is configured here.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import sys
import types
from fractions import Fraction
from pathlib import Path

import mpmath as mp
import numpy as np

V3_SOURCE_OBSERVER_SHA256 = (
    "abf463f14ba5e3a73d694452988d666d7fc7c0f063500f928d74dc19ace84ccf"
)
HISTORICAL_SCHEMAS = {
    "gragra.gen1.refinement-trace-observer.v1": "V1",
    "gragra.gen1.refinement-trace-observer.v2": "V2",
    "gragra.gen1.refinement-trace-observer.v3": "V3",
}
ZERO_SHA256 = "6c668d2a7c7c3a80ea7b5770cb795dc4a34c719cb68b33123c843badd02429c3"
POLYNOMIAL_SHA256 = "6d39a6e00a022c1ef06e86b67f7b7a5542b140bfe62dafae5df6cf9820914152"
POLYNOMIAL_SCRIPT_SHA256 = (
    "e51402fd4779aadc86fdf048592a9f71bedf0adaac69392ea29de1b84cb32a31"
)
COUNTS = {
    "sphere": 288,
    "high_sphere": 36,
    "noncavity": 67,
    "plane": 3,
    "source": 43,
    "cuboid": 165,
    "cylinder": 576,
}
SECTIONS = {
    "potential": slice(0, 1),
    "acceleration": slice(1, 4),
    "gradient": slice(4, 13),
}
ORACLE_DIRECTORY = Path(__file__).resolve().parent
PACKAGE = "_gen1_independent_trace_audit_v4"


# Share only the V4 contract. Loading by one stable name keeps adapter/dataclass
# identity stable for callers that load this auditor by its file path.
_CONTRACT_NAME = "_gen1_refinement_trace_provenance_v4"
if _CONTRACT_NAME not in sys.modules:
    _contract_spec = importlib.util.spec_from_file_location(
        _CONTRACT_NAME, ORACLE_DIRECTORY / "refinement_trace_provenance_v4.py"
    )
    _contract = importlib.util.module_from_spec(_contract_spec)
    sys.modules[_CONTRACT_NAME] = _contract
    _contract_spec.loader.exec_module(_contract)
PROVENANCE = sys.modules[_CONTRACT_NAME]
ATTEMPT_SUCCESS_SCHEMA = "gragra.gen1.refinement-trace-attempt-success.v4"
ATTEMPT_FAILURE_SCHEMA = "gragra.gen1.refinement-trace-attempt-failure.v4"


DATASETS = {
    "sphere": "sphere-three-quadrature-20261001",
    "high_sphere": "sphere-high-degree-final-20261001",
    "noncavity": "noncavity-primary-20261001",
    "plane": "plane-static-frozen-20261001",
    "source": "source-primary-20261001",
    "cuboid": "cuboid-parallel-20261001",
    "cylinder": "cylinder-parallel-20261001",
}

EXPECTED_REFERENCE_SHA256 = {
    "sphere": "a5da1f55c0c91e84c1241d00325354af1aa79acd4f13258ee9984ef857ff2783",
    "high_sphere": "dcd7f5b4444e1a965d18d5c63fb92535217a85d371e4775ce69acd5dd7ac5b56",
    "noncavity": "b61408498c5dcfa046ccdc663ac2609cd63a221b3e9cb1089bf0bceea924678b",
    "plane": "8076a91f1f7462770518b717fb3a204ee4be29e0906a618658aff132c97b063e",
    "source": "3f648bcf58566135c343f216be00dba38fa35fc6e0f70a926e91957ef862be6d",
    "cuboid": "f94a6b54b3906351e281ebcb5ff5b8aee6509974128e557085486fb00839150f",
    "cylinder": "5e2d9bbd872fc0cd591f3c475d3277b8537eca1be2f3a4e23bb9a45dd565a261",
}

# The existing root-normalization artifact is historical V1 evidence produced
# by this pinned generator/helper pair. V4 validates that artifact read-only;
# it does not rewrite its schema or pretend the original calculation ran under
# the V4 auditor.
ROOT_SUPPLEMENT_GENERATOR_SHA256 = (
    "412cedea25683f81635a75d051c1b9aa16d4542622233aeb00ef5124597eebdd"
)
ROOT_SUPPLEMENT_AUDIT_HELPER_SHA256 = (
    "560d518f9b04468a3deff15ae3c2a83bb544f6bd970c3c73006fb14dc8734130"
)

EXPECTED_ORACLE_SHA256 = {
    "acceptance.py": "0ac9ac1f01578696fb6ef1a6c1aad887b976aab1b7b873fbc78df7d388747e94",
    "cavity_fixtures.py": (
        "ba7cc6b531719b5d3e9dece63faf6f8d0039bd848822712a630539ca88e4ffd1"
    ),
    "noncavity_fixtures.py": (
        "bbebf297fd19be82c3253114d15603909e4ecf455c08d69f9f7c0f9392bf24a7"
    ),
    "noncavity_reference.py": (
        "ab4d526b2b4ec582832051b304ea7ca0017935e69f03b3eca03c26490425e882"
    ),
    "plane_static_reference.py": (
        "69623f1502aaf1491d853681b55000b11ae1357bab0cdbf00c1a71cd030c075c"
    ),
    "source_fixtures.py": (
        "8d1ff588b351367eb979ec6e4fde8c55cd4028cb06a5bade83160abed48fabe7"
    ),
    "source_reference.py": (
        "108db9dbaf0a5dc681f6ce56b982c79e3d32edc31116ad0f03b72d9610054117"
    ),
    "sphere_fixtures.py": (
        "6a3d3669535b811acaa2b11b3b9d86301720342280d8f084beeb7b6e5c8afda1"
    ),
    "sphere_reference.py": (
        "261bb88546e665fda2377597e8be52be8d88b7554bc664d0289982eec85796c8"
    ),
    "sphere_high_degree_fixtures.py": (
        "5135a289d52a681dc71c02ad8469a4c6baeb908a259f0965e9022479e45bdc0b"
    ),
    "sphere_high_degree_reference.py": (
        "79e5bef1a5fd46506fecf9abefa4eaef2b8d910d1c7cccd4e47943dd80a2f2bb"
    ),
    "surface_reference.py": (
        "fb29d75234a366c6f730f78123198e9f1ff0de8d56d811f3428f4c52c85fde3f"
    ),
    "surface_high_precision.py": (
        "efc575b52d62a3195c27fa7468e2ad786f76d65fe2a78d73873be0637f0b7ea1"
    ),
}

INDEPENDENT_FLOOR_COMPONENT_REASONS = {
    "precision_propagation": "independent multiprecision error propagation",
    "50_80_digit_precision": "fixed-order 50/80-digit precision comparison",
    "root_normalization_arithmetic_precision": (
        "independent arithmetic precision estimate for root normalization"
    ),
    "root_normalization_precision": "independent root-normalization precision estimate",
    "high_precision_roundoff_estimate": (
        "independent high-precision arithmetic roundoff estimate"
    ),
    "longdouble_roundoff_estimate": (
        "independent longdouble arithmetic roundoff estimate"
    ),
    "float_conversion": "reported float conversion error estimate",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def strict_json_loads(value, *, description):
    try:
        return json.loads(value, object_pairs_hook=_unique_json_object)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"invalid {description}: {error}") from error


def read_json(path):
    path = Path(path)
    return strict_json_loads(path.read_text(encoding="utf-8"), description=str(path))


def read_rows(path):
    path = Path(path)
    return [
        strict_json_loads(line, description=f"JSONL record {line_number} in {path}")
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        )
        if line
    ]


def production_modules():
    return sorted(
        name for name in sys.modules if name == "gragra" or name.startswith("gragra.")
    )


def oracle(name):
    """Load only pinned independent siblings, never the recorder or production."""
    if PACKAGE not in sys.modules:
        module = types.ModuleType(PACKAGE)
        module.__path__ = [str(ORACLE_DIRECTORY)]
        sys.modules[PACKAGE] = module
    return importlib.import_module(f"{PACKAGE}.{name}")


def check_oracle_pins():
    for name, digest in EXPECTED_ORACLE_SHA256.items():
        require(
            sha256(ORACLE_DIRECTORY / name) == digest, f"oracle source changed: {name}"
        )


def callback_value_precision(value):
    """Choose enough decimal precision for every captured MP tuple."""
    precision = 0
    if isinstance(value, list):
        return max((callback_value_precision(item) for item in value), default=0)
    if not isinstance(value, dict):
        return precision
    for key in ("digits", "mp_dps_at_call"):
        item = value.get(key)
        if type(item) is int and item > 0:
            precision = max(precision, item)
    parts = value.get("mpf")
    if (
        isinstance(parts, (list, tuple))
        and len(parts) == 4
        and type(parts[3]) is int
        and parts[3] > 0
    ):
        precision = max(precision, math.ceil(parts[3] * math.log10(2)) + 8)
    decimal = value.get("decimal")
    if isinstance(decimal, str):
        mantissa = decimal.lower().split("e", 1)[0].lstrip("+-").replace(".", "")
        significant = mantissa.lstrip("0").rstrip("0")
        if significant:
            precision = max(precision, len(significant) + 8)
    for item in value.values():
        precision = max(precision, callback_value_precision(item))
    return precision


def declared_callback_digits(value):
    """Read declared arithmetic precision, separate from stored MPF mantissa."""
    if isinstance(value, list):
        return max((declared_callback_digits(item) for item in value), default=0)
    if not isinstance(value, dict):
        return 0
    precision = max(
        (
            item
            for key in ("digits", "mp_dps_at_call")
            if type(item := value.get(key)) is int and item > 0
        ),
        default=0,
    )
    return max(
        precision,
        max((declared_callback_digits(item) for item in value.values()), default=0),
    )


def decode(value):
    """Preserve MP tuples under a context derived from their stored precision."""
    with mp.workdps(max(50, callback_value_precision(value))):
        return _decode_inner(value)


def decode_mpf(value):
    """Independently check exact tuple normalization and canonical nstr display.

    Presentation precision is declared separately from callback arithmetic.
    Reconstruct with stored mantissa bits; compare the display at its explicit
    precision without decimal equality tolerances or binary64 conversion.
    """
    parts = value.get("mpf")
    require(
        isinstance(parts, (list, tuple))
        and len(parts) == 4
        and all(type(part) is int for part in parts)
        and parts[0] in (0, 1)
        and parts[1] >= 0
        and parts[3] >= 0
        and (
            tuple(parts) == (0, 0, 0, 0)
            if parts[1] == 0
            else parts[1] % 2 == 1 and parts[1].bit_length() == parts[3]
        ),
        "invalid canonical MPF tuple",
    )
    display_digits = value.get("display_digits")
    require(
        type(display_digits) is int and display_digits > 0,
        "MPF display precision must be a positive integer",
    )
    with mp.workprec(max(53, parts[3] + 8)):
        result = mp.mpf(tuple(parts))
        require(mp.isfinite(result), "nonfinite MP value")
        expected_display = mp.nstr(result, n=display_digits)
    require(
        type(value.get("decimal")) is str and value["decimal"] == expected_display,
        "MPF decimal display does not match tuple",
    )
    return result


def validate_mpf_displays(value):
    """Check all stored MPF displays independently of numerical field routing."""
    if isinstance(value, dict):
        if "mpf" in value:
            decode_mpf(value)
        for item in value.values():
            validate_mpf_displays(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            validate_mpf_displays(item)


def _decode_inner(value):
    """Decode recursively inside ``decode``'s precision context."""
    if isinstance(value, list):
        return tuple(_decode_inner(item) for item in value)
    if not isinstance(value, dict):
        return value
    if all(key in value for key in ("shape", "dtype", "values")):
        result = np.asarray(_decode_inner(value["values"]), dtype=value["dtype"])
        require(list(result.shape) == value["shape"], "serialized array shape mismatch")
        require(np.all(np.isfinite(result)), "nonfinite serialized array")
        return result
    if "mpf" in value:
        return decode_mpf(value)
    if "real" in value and "imag" in value:
        if "dtype" in value:
            return np.clongdouble(_decode_inner(value["real"])) + np.clongdouble(
                1j
            ) * np.clongdouble(_decode_inner(value["imag"]))
        return mp.mpc(_decode_inner(value["real"]), _decode_inner(value["imag"]))
    if "dtype" in value and "decimal" in value:
        result = np.dtype(value["dtype"]).type(value["decimal"])
        require(np.isfinite(result), "nonfinite longdouble value")
        return result
    return {key: _decode_inner(item) for key, item in value.items()}


def _fraction_floor_log2(value):
    """Find floor(log2(value)) for a positive exact Fraction."""
    numerator = value.numerator
    denominator = value.denominator
    exponent = numerator.bit_length() - denominator.bit_length()
    if exponent >= 0:
        if numerator < denominator << exponent:
            exponent -= 1
    elif numerator << -exponent < denominator:
        exponent -= 1
    return exponent


def _longdouble_from_integer(significand, exponent, negative):
    """Construct a rounded dyadic using only exact binary longdouble steps."""
    result = np.longdouble(0)
    for bit_index in range(significand.bit_length() - 1, -1, -1):
        result = np.ldexp(result, 1)
        if significand & (1 << bit_index):
            result += np.longdouble(1)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = np.ldexp(result, exponent)
    return -result if negative else result


def _audit_mpf_to_longdouble(value):
    """Round an MPF's exact dyadic rational to host longdouble using Fractions.

    This implementation is independent from the recorder's integer quotient
    routine. The result is constructed from an already rounded binary
    significand, avoiding NumPy's decimal parser and ambient MP context.
    """
    sign, mantissa, exponent, _bitcount = value._mpf_
    negative = bool(sign)
    if mantissa == 0:
        return np.longdouble("-0" if negative else "0")

    finfo = np.finfo(np.longdouble)
    precision_bits = int(finfo.nmant) + 1
    minimum_normal_exponent = int(finfo.minexp)
    maximum_exponent = int(finfo.maxexp)
    subnormal_exponent = minimum_normal_exponent - precision_bits + 1
    leading_exponent = exponent + mantissa.bit_length() - 1
    require(leading_exponent < maximum_exponent, "nonfinite number")
    if leading_exponent < subnormal_exponent - 1:
        return np.longdouble("-0" if negative else "0")

    signed_mantissa = -mantissa if negative else mantissa
    exact = (
        Fraction(signed_mantissa << exponent, 1)
        if exponent >= 0
        else Fraction(signed_mantissa, 1 << -exponent)
    )
    magnitude = abs(exact)
    quantum_exponent = (
        subnormal_exponent
        if leading_exponent < minimum_normal_exponent
        else leading_exponent - precision_bits + 1
    )
    numerator = magnitude.numerator
    denominator = magnitude.denominator
    if quantum_exponent >= 0:
        denominator <<= quantum_exponent
    else:
        numerator <<= -quantum_exponent
    quotient, remainder = divmod(numerator, denominator)
    twice_remainder = 2 * remainder
    if twice_remainder > denominator or (
        twice_remainder == denominator and quotient & 1
    ):
        quotient += 1
    if quotient == 0:
        return np.longdouble("-0" if negative else "0")
    if quotient.bit_length() > precision_bits:
        quotient >>= 1
        quantum_exponent += 1
    rounded_exponent = quantum_exponent + quotient.bit_length() - 1
    require(rounded_exponent < maximum_exponent, "nonfinite number")

    result = _longdouble_from_integer(quotient, quantum_exponent, negative)
    require(bool(np.isfinite(result)), "nonfinite number")
    return result


def number(value, ld=False):
    if isinstance(value, dict):
        value = decode(value)
    if ld and isinstance(value, mp.mpf):
        require(mp.isfinite(value), "nonfinite number")
        result = _audit_mpf_to_longdouble(value)
    else:
        result = np.longdouble(value) if ld else mp.mpf(value)
    require(
        bool(np.isfinite(result)) if ld else mp.isfinite(result), "nonfinite number"
    )
    return result


def vector_difference(left, right, ld=False):
    require(len(left) == len(right), "difference vector length mismatch")
    if ld:
        return oracle("surface_reference").norm_ld(np.asarray(left) - np.asarray(right))
    return oracle("sphere_reference").vector_norm(
        tuple(a - b for a, b in zip(left, right, strict=True))
    )


def observed_pairs(vectors, ld=False):
    require(len(vectors) == 3, "three refinement vectors required")
    return (
        vector_difference(vectors[1], vectors[0], ld),
        vector_difference(vectors[2], vectors[1], ld),
    )


def frozen_component(reference, key):
    components = reference["uncertainty"]["components"]
    require(len(dict(components)) == len(components), "duplicate frozen component")
    return dict(components).get(key)


def audit_floor(trace, reference):
    expected = {
        key: mp.mpf(value)
        for key, value in reference["uncertainty"]["components"]
        if key in INDEPENDENT_FLOOR_COMPONENT_REASONS
    }
    field = (
        "floor_components_si"
        if "floor_components_si" in trace
        else "independent_floor_components_si"
    )
    require(field in trace, "missing independent floor components")
    actual = trace[field]
    require(set(actual) == set(expected), "floor component set differs from allowlist")
    require(
        all(number(actual[key]) == value for key, value in expected.items()),
        "floor component value mismatch",
    )
    require(
        trace.get("independent_floor_component_reasons")
        == {key: INDEPENDENT_FLOOR_COMPONENT_REASONS[key] for key in expected},
        "floor reasons mismatch",
    )
    return mp.fsum(expected.values()) if expected else None


def audit_trend(trend, pair, floor, ld=False, expected_reason=None):
    require(isinstance(trend, dict), "missing trend")
    if pair is None:
        require(
            trend.get("classification") == "not_applicable",
            "structural N/A trend mismatch",
        )
        require(
            isinstance(expected_reason, str) and trend.get("reason") == expected_reason,
            "N/A reason mismatch",
        )
        return "not_applicable"
    require(
        trend.get("classification") != "not_applicable",
        "structural N/A trend mismatch",
    )
    require(
        [number(trend[key], ld) for key in ("coarse_to_middle_si", "middle_to_fine_si")]
        == list(pair),
        "trend/named pair mismatch",
    )
    actual_floor = trend.get("independent_floor_si")
    require(
        (actual_floor is None and floor is None)
        or (
            actual_floor is not None
            and floor is not None
            and number(actual_floor) == floor
        ),
        "trend floor mismatch",
    )
    comparison_floor = (
        _audit_mpf_to_longdouble(floor)
        if ld and isinstance(floor, mp.mpf)
        else np.longdouble(floor)
        if ld and floor is not None
        else floor
    )
    first, last = pair
    classification = (
        "decreasing"
        if last < first
        else "nondecreasing_explained_by_independent_floor"
        if comparison_floor is not None
        and comparison_floor > 0
        and comparison_floor >= last
        else "unresolved_zero_change"
        if first == last == 0
        else "unresolved_nondecreasing"
    )
    require(
        trend.get("classification") == classification, "derived trend class mismatch"
    )
    require(
        type(trend.get("nondecreasing")) is bool
        and trend["nondecreasing"] == bool(last >= first),
        "trend direction mismatch",
    )
    return classification


def certified_override_na_reason(dataset, fixture):
    if dataset in {"cuboid", "cylinder"}:
        return "the frozen output is an exact symmetry/parity certificate"
    if (
        dataset == "source"
        and fixture.get("shape") != "sphere"
        and fixture.get("family") != "temperature_grid"
    ):
        return "the frozen output is an exact inversion-parity certificate"
    return "the frozen observable is an exact symmetry/parity certificate"


def audit_component(
    trace, reference, kind, component_name, pair, ld=False, applicable=True
):
    checks = trace["frozen_component_checks"]
    require(kind in checks, f"missing {kind} component-check slot")
    check = checks[kind]
    if component_name is None:
        require(check is None, "unexpected component check")
        require(
            trace.get(f"{kind}_differences_si") is None,
            "unexpected inapplicable differences",
        )
        return
    require(isinstance(check, dict), "applicable component check cannot be None")
    require(
        check.get("component_name") == component_name, "wrong original component name"
    )
    factor = np.longdouble(10) if ld else 10
    computed = None if pair is None else factor * max(pair)
    frozen = frozen_component(reference, component_name)
    require(
        check.get("formula_float_si")
        == (None if computed is None else float(computed)),
        "component formula mismatch",
    )
    require(check.get("frozen_component_si") == frozen, "frozen component mismatch")
    require(
        check.get("matches_frozen_float") is True
        and frozen == (None if computed is None else float(computed)),
        "original component reconstruction failed",
    )
    if "formula_value_si" in check:
        require(
            (computed is None and check["formula_value_si"] is None)
            or (
                computed is not None
                and number(check["formula_value_si"], ld) == computed
            ),
            "component MP formula mismatch",
        )
    named = trace.get(f"{kind}_differences_si")
    if not applicable:
        require(named is None, "structurally inapplicable named pair present")
        return
    require(
        isinstance(named, dict) and pair is not None, "missing applicable named pair"
    )
    labels = (
        ("4h_to_2h", "2h_to_h")
        if kind == "derivative"
        else ("coarse_to_middle", "middle_to_fine")
    )
    require(set(named) == {*labels, "ten_max"}, "incomplete named pair schema")
    require(
        [number(named[key], ld) for key in labels] == list(pair),
        "captured/named pair mismatch",
    )
    require(number(named["ten_max"], ld) == computed, "named ten_max mismatch")


def audit_final(trace, reference, final):
    captured = decode(trace["captured_final_values_si"])
    require(len(captured) == len(final), "captured final shape mismatch")
    require(
        all(a == b for a, b in zip(captured, final, strict=True)),
        "captured final callback mismatch",
    )
    converted = [
        [float(complex(value).real), float(complex(value).imag)] for value in final
    ]
    require(
        converted == reference["values_si"], "final callback/frozen values mismatch"
    )
    require(
        trace.get("captured_final_values_match_frozen") is True,
        "final frozen-value flag mismatch",
    )
    ld = isinstance(final, np.ndarray)
    converted_values = tuple(complex(value) for value in final)
    converted_numeric = (
        np.asarray(converted_values, dtype=np.clongdouble)
        if ld
        else tuple(mp.mpc(value) for value in converted_values)
    )
    conversion = vector_difference(final, converted_numeric, ld=ld)
    require(
        frozen_component(reference, "float_conversion") == float(conversion),
        "independent float-reporting component reconstruction failed",
    )


def mp_trace(dataset, fixture, name, trace, reference, calls):
    high = 130 if dataset == "high_sphere" else 80
    low = 90 if dataset == "high_sphere" else 50
    orders = {
        "sphere": (32, 40, 48),
        "high_sphere": (64, 80, 96),
        "noncavity": (16, 24, 32),
        "source": (16, 24, 32),
        "plane": None,
    }[dataset]
    kernel = (
        "sphere_observables"
        if dataset in {"sphere", "high_sphere"}
        else f"{dataset}_observable_values"
    )
    events = [c for c in calls if c["kernel"] == kernel]

    def digits(event):
        return event["inputs"].get("digits") or event["inputs"]["mp_dps_at_call"]

    with mp.workdps(high):
        for event in events:
            require(
                list(decode(event["inputs"]["point_m"]))
                == [mp.mpf(x) for x in fixture["target_m"]],
                "MP callback target mismatch",
            )
        final = [
            c
            for c in events
            if digits(c) == high
            and (orders is None or c["inputs"]["quadrature_order"] == orders[-1])
        ]
        require(len(final) >= 3, "missing final derivative triplet")
        final = final[-3:]
        h = decode(final[-1]["inputs"]["step_m"])
        require(
            [decode(c["inputs"]["step_m"]) for c in final] == [4 * h, 2 * h, h],
            "original derivative steps mismatch",
        )
        for c in final:
            require(
                list(decode(c["inputs"]["point_m"]))
                == [mp.mpf(x) for x in fixture["target_m"]],
                "callback target mismatch",
            )
        values = [decode(c["output"])[name] for c in final]
        derivative = observed_pairs(values)
        qpair = None
        if orders:
            by_order = {
                c["inputs"]["quadrature_order"]: c
                for c in events
                if digits(c) == high
                and c["inputs"]["quadrature_order"] in orders
                and decode(c["inputs"]["step_m"]) == h
            }
            require(
                set(by_order) == set(orders),
                "missing original fixed-h quadrature triplet",
            )
            qpair = observed_pairs(
                [decode(by_order[o]["output"])[name] for o in orders]
            )
        derivative_applicable = name != "potential"
        quadrature_applicable = (
            dataset in {"sphere", "high_sphere"}
            or dataset == "source"
            and fixture["family"] != "temperature_grid"
            or dataset == "noncavity"
            and fixture["family"] in {"gaussian", "halfspace", "point_force"}
        )
        require(
            trace["evaluation_inputs"]["high_digits"] == high
            and trace["evaluation_inputs"]["low_digits"] == low,
            "MP precision inputs mismatch",
        )
        require(
            trace["evaluation_inputs"]["quadrature_orders"]
            == (list(orders) if quadrature_applicable else None),
            "quadrature applicability/orders mismatch",
        )
        require(
            [decode(trace["evaluation_inputs"][key]) for key in ("4h_m", "2h_m", "h_m")]
            == [4 * h, 2 * h, h],
            "declared derivative steps mismatch",
        )
        lower = [
            c
            for c in events
            if digits(c) == low
            and (orders is None or c["inputs"]["quadrature_order"] == orders[-1])
            and decode(c["inputs"]["step_m"]) == h
        ]
        require(lower, "missing fixed-order precision callback")
        precision = vector_difference(values[-1], decode(lower[-1]["output"])[name])
        require(
            number(trace["fixed_order_precision_change_si"]) == precision,
            "precision pair mismatch",
        )
        precision_component = (
            "root_normalization_arithmetic_precision"
            if dataset in {"sphere", "high_sphere"}
            else "50_80_digit_precision"
            if dataset == "plane"
            else "precision_propagation"
        )
        require(
            frozen_component(reference, precision_component) == float(10 * precision),
            "independent precision component reconstruction failed",
        )
        require(
            set(trace["frozen_component_checks"]) == {"derivative", "quadrature"},
            "MP component slots incomplete",
        )
        dname = (
            "real_field_derivative_refinement"
            if dataset == "plane"
            else "derivative_refinement"
        )
        qname = (
            "quadrature_root_normalization_refinement"
            if dataset in {"sphere", "high_sphere"}
            else "quadrature_refinement"
            if dataset != "plane"
            else None
        )
        audit_component(
            trace,
            reference,
            "derivative",
            dname,
            derivative,
            applicable=derivative_applicable,
        )
        audit_component(
            trace,
            reference,
            "quadrature",
            qname,
            qpair,
            applicable=quadrature_applicable,
        )
        if qpair is not None and not quadrature_applicable:
            named = trace["observed_inapplicable_quadrature_component_si"]
            require(
                [number(named[key]) for key in ("coarse_to_middle", "middle_to_fine")]
                == list(qpair),
                "observed inapplicable pair mismatch",
            )
            require(
                number(named["ten_max"]) == 10 * max(qpair),
                "inapplicable ten_max mismatch",
            )
        floor = audit_floor(trace, reference)
        derivative_reason = (
            None
            if derivative_applicable
            else (
                "potential is evaluated directly and does not depend on "
                "target-difference step"
            )
        )
        quadrature_reason = None
        if not quadrature_applicable:
            quadrature_reason = (
                "the original route is an explicit finite mass sum with "
                "no spatial quadrature"
                if dataset == "source" and fixture.get("family") == "temperature_grid"
                else "entrypoint uses no spatial quadrature for this family"
            )
        audit_trend(
            trace["derivative_trend"],
            derivative if derivative_applicable else None,
            floor,
            expected_reason=derivative_reason,
        )
        audit_trend(
            trace["quadrature_trend"],
            qpair if quadrature_applicable else None,
            floor,
            expected_reason=quadrature_reason,
        )
        audit_final(trace, reference, values[-1])


def surface_geometry(dataset, fixture):
    """Finite-face distances in the pinned longdouble arithmetic."""
    ld = np.longdouble
    target = np.asarray(fixture["target_m"], dtype=ld)
    geometry = fixture["geometry"] if dataset == "source" else fixture
    cuboid = (
        fixture.get("shape") == "cuboid" if dataset == "source" else dataset == "cuboid"
    )
    if cuboid:
        lengths = np.asarray(geometry["lengths_m"], dtype=ld)
        gaps = []
        for axis in range(3):
            for sign in (-1, 1):
                normal_distance = abs(target[axis] - sign * lengths[axis] / 2)
                squared = normal_distance**2 + sum(
                    max(abs(target[j]) - lengths[j] / 2, ld(0)) ** 2
                    for j in range(3)
                    if j != axis
                )
                gaps.append(np.sqrt(squared))
        characteristic = max(lengths)
    else:
        radius, height = ld(geometry["radius_m"]), ld(geometry["height_m"])
        radial = np.hypot(target[0], target[1])
        gaps = [np.hypot(radial - radius, max(abs(target[2]) - height / 2, ld(0)))]
        gaps.extend(
            np.hypot(target[2] - sign * height / 2, max(radial - radius, ld(0)))
            for sign in (-1, 1)
        )
        characteristic = max(radius, height)
    gap = min(gaps)
    require(
        np.isfinite(characteristic)
        and characteristic > 0
        and np.isfinite(gap)
        and gap > 0,
        "invalid finite-surface geometry",
    )
    return characteristic, gap


def surface_payload(event, source=False, hp=False):
    output = event["output"]
    if source:
        require(
            isinstance(output, dict)
            and set(output) == {"values_and_absolute", "geometry_summary"},
            "source payload schema incomplete",
        )
        output = output["values_and_absolute"]
    require(
        isinstance(output, list) and len(output) == (3 if hp else 2),
        "surface return must include values and absolute sums",
    )
    values, absolute = decode(output[0]), decode(output[1])
    require(len(values) == len(absolute) == 13, "surface vector schema incomplete")
    if not hp:
        require(
            isinstance(values, np.ndarray)
            and values.shape == (13,)
            and values.dtype.kind in "fc"
            and isinstance(absolute, np.ndarray)
            and absolute.shape == (13,),
            "surface saved array dtype/shape mismatch",
        )
    require(
        all(np.isfinite(v) for v in values)
        if not hp
        else all(mp.isfinite(v) for v in values),
        "nonfinite surface values",
    )
    require(
        all(isinstance(v, mp.mpf) and mp.isfinite(v) and v >= 0 for v in absolute)
        if hp
        else np.asarray(absolute).dtype.kind == "f"
        and np.all(np.isfinite(absolute))
        and np.all(np.asarray(absolute) >= 0),
        "absolute sums must be finite nonnegative real values",
    )
    count = output[2] if hp else None
    if hp:
        require(type(count) is int and count > 0, "HP node count missing")
    return values, absolute, count


def high_precision_callback_pair(calls, name):
    """Independently rebuild the 50-digit fallback refinement pair."""
    with mp.workdps(max(80, declared_callback_digits(calls))):
        section = SECTIONS[name]
        events = [
            item
            for item in calls
            if item.get("kernel") == "surface_high_precision_evaluate"
        ]
        lookup = {
            (item["inputs"]["digits"], item["inputs"]["order"]): item for item in events
        }
        require(
            all((50, order) in lookup for order in (16, 24, 32)),
            "missing HP callback triplet",
        )
        payloads = [
            surface_payload(lookup[(50, order)], hp=True) for order in (16, 24, 32)
        ]
        vectors = [payload[0][section] for payload in payloads]
        return observed_pairs(vectors), vectors, lookup


def root_precision(fixture, records):
    matches = [
        r
        for r in records
        if r["key"]
        == [
            abs(fixture["order"]),
            fixture["radial_index"],
            fixture["radius_m"],
            fixture["axial_index"],
        ]
    ]
    require(len(matches) == 1, "Cylinder root-normalization evidence unavailable")
    entry = matches[0]
    high_root, high_norm = map(decode, entry["at_80_digits"])
    low_root, low_norm = map(decode, entry["at_50_digits"])
    result = (
        abs(high_root - low_root) * (1 + abs(high_root))
        + abs(high_norm - low_norm) / high_norm
    )
    require(
        number(entry["root_precision_estimate"]) == result,
        "root-only estimate reconstruction mismatch",
    )
    return result


def surface_trace(
    dataset, fixture, name, trace, reference, metadata, calls, root_records=()
):
    section = SECTIONS[name]
    hp = (
        reference["uncertainty"]["method"]
        == "high precision Newton surface / independent refinement"
    )
    callback_dps = max(80, declared_callback_digits(calls)) if hp else 15
    with mp.workdps(callback_dps):
        if hp:
            require(trace["route"] == "mpmath_fallback", "HP route mismatch")
            events = [
                c for c in calls if c["kernel"] == "surface_high_precision_evaluate"
            ]
            lookup = {(c["inputs"]["digits"], c["inputs"]["order"]): c for c in events}
            require(
                all(
                    json.dumps(decode(c["inputs"]["fixture"]), sort_keys=True)
                    == json.dumps(fixture, sort_keys=True)
                    for c in events
                ),
                "HP callback fixture mismatch",
            )
            require(
                all((50, o) in lookup for o in (16, 24, 32)) and (80, 32) in lookup,
                "missing HP callback triplet/precision pair",
            )
            pair, vectors, _lookup = high_precision_callback_pair(calls, name)
            for order in (16, 24, 32):
                surface_payload(lookup[50, order], hp=True)
            payload = surface_payload(lookup[80, 32], hp=True)
            final = payload[0][section]
            absolute, count = payload[1][section], payload[2]
            require(
                count == metadata["fallback_surface_nodes"],
                "HP metadata node count mismatch",
            )
            roundoff = (
                1000
                * count
                * mp.mpf(10) ** -49
                * oracle("sphere_reference").vector_norm(absolute)
            )
            require(
                frozen_component(reference, "high_precision_roundoff_estimate")
                == float(roundoff),
                "HP roundoff component reconstruction failed",
            )
            precision_root = (
                mp.mpf(0)
                if dataset == "cuboid"
                else root_precision(fixture, root_records)
            )
            root_error = (
                10 * precision_root * oracle("sphere_reference").vector_norm(absolute)
            )
            require(
                frozen_component(reference, "root_normalization_precision")
                == float(root_error),
                "HP root-normalization component reconstruction failed",
            )
            precision = vector_difference(final, vectors[-1])
            require(
                number(trace["fixed_order_precision_change_si"]) == precision,
                "HP precision pair mismatch",
            )
            require(
                frozen_component(reference, "50_80_digit_precision")
                == float(10 * precision),
                "HP independent precision component reconstruction failed",
            )
            require(
                trace["evaluation_inputs"]
                == {
                    "quadrature_orders_at_50_digits": [16, 24, 32],
                    "final_order_at_80_digits": 32,
                    "digits": [50, 80],
                },
                "HP orders/digits mismatch",
            )
            component = "high_precision_surface_quadrature"
        else:
            require(trace["route"] == "numpy.longdouble", "longdouble route mismatch")
            source = dataset == "source"
            orders = [16, 24, 32] if source else metadata["orders"]
            kernel = "source_surface_values" if source else "surface_newton_evaluate"
            events = [c for c in calls if c["kernel"] == kernel]
            lookup = {c["inputs"]["order"]: c for c in events}
            require(
                len(orders) == 3 and all(o in lookup for o in orders),
                "missing surface order triplet",
            )
            vectors = []
            for order in orders:
                event = lookup[order]
                if source:
                    require(
                        json.dumps(decode(event["inputs"]["fixture"]), sort_keys=True)
                        == json.dumps(fixture, sort_keys=True),
                        "source surface callback fixture mismatch",
                    )
                if not source:
                    target = decode(event["inputs"]["target_m"])
                    require(
                        list(target) == [np.longdouble(x) for x in fixture["target_m"]],
                        "surface callback target mismatch",
                    )
                payload = surface_payload(event, source=source)
                characteristic, gap = surface_geometry(dataset, fixture)
                if source:
                    geometry = event["output"]["geometry_summary"]
                    require(
                        set(geometry) == {"characteristic_m", "minimum_gap_m"}
                        and decode(geometry["characteristic_m"]) == characteristic
                        and decode(geometry["minimum_gap_m"]) == gap,
                        "source captured geometry mismatch",
                    )
                else:
                    require(
                        np.longdouble(metadata["minimum_surface_gap_m"]) == gap
                        and np.longdouble(metadata["longdouble_epsilon"])
                        == np.finfo(np.longdouble).eps,
                        "surface metadata geometry/precision mismatch",
                    )
                vectors.append(payload[0][section])
            pair = observed_pairs(vectors, ld=True)
            final = vectors[-1]
            absolute = payload[1][section]
            norm_absolute = oracle("surface_reference").norm_ld(absolute)
            roundoff = (
                256
                * np.finfo(np.longdouble).eps
                * (1 + characteristic / gap)
                * norm_absolute
            )
            require(
                frozen_component(reference, "longdouble_roundoff_estimate")
                == float(roundoff),
                "LD roundoff component reconstruction failed",
            )
            if not source:
                with mp.workdps(80):
                    precision_root = (
                        mp.mpf(0)
                        if dataset == "cuboid"
                        else root_precision(fixture, root_records)
                    )
                root_error = 10 * np.longdouble(str(precision_root)) * norm_absolute
                require(
                    frozen_component(reference, "root_normalization_precision")
                    == float(root_error),
                    "LD root-normalization component reconstruction failed",
                )
            require(
                trace["evaluation_inputs"]
                == {"quadrature_orders": orders, "backend": "numpy.longdouble"},
                "surface orders/backend mismatch",
            )
            component = "surface_quadrature_refinement"
        require(
            set(trace["frozen_component_checks"]) == {"quadrature"},
            "surface component slot mismatch",
        )
        require(
            trace.get("derivative_differences_si") is None,
            "surface derivative pair is inapplicable",
        )
        audit_component(trace, reference, "quadrature", component, pair, ld=not hp)
        floor = audit_floor(trace, reference)
        derivative_reason = (
            "finite-source surface references have no target "
            "finite-difference refinement"
            if dataset == "source"
            else "surface values use spatial quadrature, not target finite differences"
        )
        audit_trend(
            trace["derivative_trend"],
            None,
            floor,
            ld=not hp,
            expected_reason=derivative_reason,
        )
        audit_trend(trace["quadrature_trend"], pair, floor, ld=not hp)
        audit_final(trace, reference, final)


def audit_annotations(
    dataset, index, fixture, name, trace, reference, zero, polynomial
):
    raw = {
        key: trace[key]["classification"]
        for key in ("derivative_trend", "quadrature_trend")
    }
    adjudicated = set()
    annotation = trace.get("zero_certificate_adjudication")
    if annotation is not None:
        matches = [
            c
            for c in zero
            if c["dataset"] == DATASETS[dataset]
            and c["fixture"] == fixture
            and c["observable"] == name
        ]
        require(len(matches) == 1, "zero certificate identity mismatch")
        certificate = matches[0]
        uncertainty = reference["uncertainty"]
        require(
            certificate.get("original_uncertainty_kind") == "estimate"
            and certificate.get("original_used_uncertainty_si") == 0
            and certificate.get("frozen_labels_and_values_unchanged") is True,
            "zero certificate original scope invalid",
        )
        require(
            uncertainty["uncertainty_kind"] == "estimate"
            and uncertainty["used_uncertainty_si"] == 0
            and all(pair == [0.0, 0.0] for pair in reference["values_si"]),
            "zero certificate original scope mismatch",
        )
        applicable = [key for key in raw if raw[key] == "unresolved_zero_change"]
        expected = {
            "dataset": certificate["dataset"],
            "fixture": fixture,
            "observable": name,
            "certificate": certificate["certificate"],
            "supplement_path": annotation.get("supplement_path"),
            "supplement_sha256": ZERO_SHA256,
            "original_uncertainty_kind": certificate["original_uncertainty_kind"],
            "original_used_uncertainty_si": certificate["original_used_uncertainty_si"],
            "raw_trend_classifications": raw,
            "adjudicated_zero_change_trends": applicable,
            "frozen_estimate_and_values_unchanged": True,
        }
        require(annotation == expected, "zero annotation field mismatch")
        require(
            Path(annotation["supplement_path"]).name
            == "frozen-zero-estimate-certificates-integrated.json",
            "zero supplement path mismatch",
        )
        for key in applicable:
            require(
                trace[key].get("certificate_adjudication")
                == "exact_zero_certificate_for_this_frozen_observable",
                "missing zero trend annotation",
            )
        adjudicated.update(applicable)
    annotation = trace.get("polynomial_quadrature_certificate_adjudication")
    if annotation is not None:
        matches = [
            c
            for c in polynomial
            if c["dataset"] == dataset
            and c["source_row_index"] == index
            and c["fixture"] == fixture
            and name in c["covered_observables"]
        ]
        require(len(matches) == 1, "polynomial certificate identity mismatch")
        certificate = matches[0]
        scope_valid = (
            dataset == "noncavity"
            and type(index) is int
            and 19 <= index < 46
            and fixture.get("family") == "point_force"
        ) or (
            dataset == "source"
            and type(index) is int
            and 0 <= index < 6
            and fixture.get("family") == "finite_temperature"
            and fixture.get("shape") == "sphere"
        )
        expected_kind = {
            "noncavity": "retarded_cubic_pulse_polynomial_panels",
            "source": "uniform_Sphere_radial_polynomial_moments",
        }.get(dataset)
        require(
            scope_valid
            and certificate.get("certificate_kind") == expected_kind
            and certificate.get("certified_error")
            == "zero quadrature truncation in ideal real arithmetic only"
            and certificate.get("frozen_values_uncertainty_labels_unchanged") is True,
            "polynomial certificate original scope invalid",
        )
        require(
            raw["quadrature_trend"] == "unresolved_zero_change",
            "polynomial annotation requires actual 0/0",
        )
        require(
            certificate["covered_original_stencils"]
            == "all original loop steps and axis/diagonal seven-point evaluations",
            "polynomial stencil scope mismatch",
        )
        expected = {
            "dataset": dataset,
            "source_row_index": index,
            "fixture": fixture,
            "observable": name,
            "certificate_kind": certificate["certificate_kind"],
            "certified_error": certificate["certified_error"],
            "scope": "ideal polynomial quadrature truncation only",
            "temporal_quadrature_remains_applicable": True,
            "total_numerical_error_certified": False,
            "raw_quadrature_trend_classification": "unresolved_zero_change",
            "supplement_path": annotation.get("supplement_path"),
            "supplement_sha256": POLYNOMIAL_SHA256,
            "verifier_script_sha256": POLYNOMIAL_SCRIPT_SHA256,
            "frozen_values_uncertainty_labels_unchanged": True,
        }
        require(annotation == expected, "polynomial annotation field mismatch")
        require(
            Path(annotation["supplement_path"]).name
            == "polynomial-quadrature-certificates-integrated.json",
            "polynomial supplement path mismatch",
        )
        adjudicated.add("quadrature_trend")
    return raw, adjudicated


def high_sphere_scalar_zero_supplement(dataset, index, row, traces):
    """Exact y-reflection proof, limited to four post-replay scalar records.

    A real sine-order-1 spherical harmonic is odd under y -> -y. Its
    radial factor and ball support are even. At y_target=0 the Newton
    kernel is invariant, so the potential volume integral cancels in
    reflected pairs. Derivatives normal to that plane need not vanish.
    This certifies only the mathematical scalar zero, not finite-precision
    quadrature, root/normalization error, or a general total-error bound.
    """
    degrees = {8: 20, 17: 30, 26: 50, 35: 60}
    if dataset != "high_sphere" or index not in degrees:
        return None
    fixture = {
        "degree": degrees[index],
        "order": -1,
        "radial_index": 1,
        "radius_m": 1.0,
        "sound_speed_m_s": 343.0,
        "pressure_real_pa": 1.0,
        "pressure_imag_pa": 0.0,
        "target_kind": "sine_axis_inside",
        "target_m": [0.0, 0.0, 0.73],
    }
    require(row["fixture"] == fixture, "supplement full fixture identity mismatch")
    reference = row["reference"]["potential"]
    require(
        reference["values_si"] == [[0.0, 0.0]]
        and reference["uncertainty"]["uncertainty_kind"] == "estimate"
        and reference["uncertainty"]["used_uncertainty_si"] == 0.0,
        "supplement frozen scalar scope mismatch",
    )
    trace = traces["potential"]
    require(
        trace["derivative_trend"]["classification"] == "not_applicable"
        and trace["quadrature_trend"]["classification"] == "unresolved_zero_change",
        "supplement raw trend scope mismatch",
    )
    require(
        all(
            number(trace["quadrature_differences_si"][key]) == 0
            for key in ("coarse_to_middle", "middle_to_fine", "ten_max")
        ),
        "supplement requires reconciled actual zero changes",
    )
    return {
        "dataset": dataset,
        "source_row_index": index,
        "fixture": fixture,
        "observable": "potential",
        "reference_sha256": EXPECTED_REFERENCE_SHA256[dataset],
        "oracle_source_sha256": EXPECTED_ORACLE_SHA256[
            "sphere_high_degree_reference.py"
        ],
        "proof": (
            "y-reflection: sine-order-1 density odd; ball/radial factor "
            "and Newton kernel at y_target=0 even; paired integral is zero"
        ),
        "protected_mathematical_zero": True,
        "original_uncertainty_kind": "estimate",
        "original_used_uncertainty_si": 0.0,
        "raw_unresolved_trend": "potential/quadrature_trend",
        "original_record_and_labels_unchanged": True,
        "extension_of": (
            "frozen-zero-estimate-certificates-integrated.json: "
            "real sine harmonic axis scalar zeros"
        ),
        "earlier_supplement_sha256": ZERO_SHA256,
        "finite_precision_or_total_error_bound_claim": False,
        "derivative_observables_certified": False,
    }


def journal_local_sha256(value):
    """Recorder's V4 local digest format, including ASCII JSON escapes."""
    PROVENANCE.canonical_json_bytes(value)
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def journal_line_bytes(event):
    """Exact persisted V4 JSONL layout; offsets count these bytes."""
    PROVENANCE.canonical_json_bytes(event)
    return (json.dumps(event, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def audit_attempt_journal(dataset, index, row, record, profile, journal, selected_rows):
    """Reconcile saved history independently; local digests do not authenticate it.

    ``selected_rows`` is the explicit independently loaded frozen selection. State
    entries have state/source_row_index/retry_of fields; accepted results have
    attempt_id/result_sha256 fields. Retry history uses the V4 recorder's plan
    shape, but none of its checkpoint verdicts is consumed. A complete selection
    requires one accepted result per row and no active/unfinished retry plan.
    """
    require(
        isinstance(journal, PROVENANCE.AttemptJournalInputs), "journal inputs missing"
    )
    require(
        type(selected_rows) is dict and bool(selected_rows), "selected scope missing"
    )
    require(
        all(
            type(i) is int and i >= 0 and type(r) is dict
            for i, r in selected_rows.items()
        )
        and selected_rows.get(index) == row,
        "selected scope differs from independent row",
    )
    # Strict serialization detaches aliases before validation and commitment.
    saved = PROVENANCE.strict_json_object(
        PROVENANCE.canonical_json_bytes(
            {
                field: getattr(journal, field)
                for field in (
                    "events",
                    "attempt_states",
                    "retry_plan",
                    "accepted_results",
                )
            }
        )
    )
    events, states = saved["events"], saved["attempt_states"]
    plans, accepted = saved["retry_plan"], saved["accepted_results"]
    require(
        type(events) is list
        and type(states) is dict
        and type(plans) is list
        and type(accepted) is dict,
        "journal shape invalid",
    )
    seen, failures, results, terminal = set(), {}, {}, {}
    position, retried_rows = 0, set()
    for sequence, event in enumerate(events, 1):
        require(type(event) is dict, "journal event is not an object")
        i, attempt = event.get("source_row_index"), event.get("attempt_id")
        require(
            type(i) is int and i in selected_rows, "journal row outside selected scope"
        )
        require(
            type(attempt) is str and bool(attempt) and attempt not in seen,
            "duplicate or invalid journal attempt",
        )
        require(
            type(event.get("event_sequence")) is int
            and event["event_sequence"] == sequence,
            "journal order mismatch",
        )
        require(
            event.get("run_id") == profile.run_id and event.get("dataset") == dataset,
            "journal run/dataset mismatch",
        )
        require(
            event.get("selected_row_sha256") == journal_local_sha256(selected_rows[i]),
            "journal frozen row pin mismatch",
        )
        require(
            type(event.get("last_durable_trace_offset")) is int
            and event["last_durable_trace_offset"] == position,
            "journal durable offset mismatch",
        )
        body = {k: v for k, v in event.items() if k != "event_sha256"}
        require(
            event.get("event_sha256") == journal_local_sha256(body),
            "journal event digest mismatch",
        )
        require(
            i not in results and event.get("retry_of") == failures.get(i),
            "journal retry/accepted result mismatch",
        )
        if event.get("retry_of") is not None:
            retried_rows.add(i)
        worker = event.get("process_independence")
        require(
            type(worker) is dict
            and type(worker.get("pid")) is int
            and worker["pid"] > 0
            and worker.get("modules_before") == []
            and worker.get("modules_after") == []
            and worker.get("production_module_names") == []
            and worker.get("production_imported") is False
            and worker.get("independence_check_passed") is True,
            "journal worker independence invalid",
        )
        if event.get("schema") == ATTEMPT_FAILURE_SCHEMA:
            error = event.get("exception")
            require(
                "record" not in event
                and "success_commitment" not in event
                and type(error) is dict
                and all(
                    type(error.get(k)) is str and error[k]
                    for k in ("type", "traceback")
                )
                and type(error.get("message")) is str
                and type(event.get("captured_calls_before_error")) is list,
                "journal failure shape invalid",
            )
            validate_mpf_displays(event["captured_calls_before_error"])
            failures[i] = attempt
            state = "failed"
        elif event.get("schema") == ATTEMPT_SUCCESS_SCHEMA:
            result = event.get("record")
            frozen = selected_rows[i]
            require(
                type(result) is dict
                and result.get("schema") == PROVENANCE.TRACE_SCHEMA
                and result.get("dataset") == dataset
                and type(result.get("source_row_index")) is int
                and result["source_row_index"] == i
                and result.get("fixture") == frozen["fixture"]
                and result.get("frozen_reference")
                == {"reference": frozen["reference"], "metadata": frozen["metadata"]}
                and result.get("status")
                in {"reproduced", "unresolved_trend", "replay_mismatch"},
                "journal success identity mismatch",
            )
            if i == index:
                require(
                    PROVENANCE.canonical_json_bytes(result)
                    == PROVENANCE.canonical_json_bytes(record),
                    "journal accepted row differs from audited record",
                )
            results[i] = {
                "attempt_id": attempt,
                "result_sha256": PROVENANCE.canonical_sha256(result),
            }
            failures.pop(i, None)
            state = "complete"
        else:
            raise ValueError("journal event schema invalid")
        terminal[attempt] = {
            "state": state,
            "source_row_index": i,
            "retry_of": event["retry_of"],
        }
        seen.add(attempt)
        position += len(journal_line_bytes(event))
    require(
        all(states.get(k) == v for k, v in terminal.items()),
        "journal terminal states mismatch",
    )
    pending_rows = set()
    for attempt, state in states.items():
        if attempt in terminal:
            continue
        require(
            type(attempt) is str
            and bool(attempt)
            and type(state) is dict
            and set(state) == {"state", "source_row_index", "retry_of"}
            and state["state"] in {"pending", "running", "interrupted"}
            and type(state["source_row_index"]) is int
            and state["source_row_index"] in selected_rows,
            "journal pending/interrupted state invalid",
        )
        i = state["source_row_index"]
        if state["state"] == "interrupted":
            predecessor = terminal.get(state["retry_of"])
            require(
                state["retry_of"] is None
                or predecessor is not None
                and predecessor["state"] == "failed"
                and predecessor["source_row_index"] == i,
                "journal interrupted retry link mismatch",
            )
        else:
            require(
                state["retry_of"] == failures.get(i),
                "journal active retry link mismatch",
            )
        if state["state"] != "interrupted":
            require(
                i not in results and i not in pending_rows,
                "journal active row already accepted",
            )
            pending_rows.add(i)
    require(
        accepted == {f"{dataset}:{i}": value for i, value in results.items()},
        "journal accepted result digest mismatch",
    )
    plan_ids, planned, unfinished = set(), set(), False
    for plan in plans:
        require(
            type(plan) is dict
            and type(plan.get("plan_id")) is str
            and plan["plan_id"]
            and plan["plan_id"] not in plan_ids,
            "journal retry plan identity invalid",
        )
        indices, remaining = (
            plan.get("source_row_indices"),
            plan.get("remaining_source_row_indices"),
        )
        require(
            type(indices) is list
            and type(remaining) is list
            and all(type(i) is int for i in indices + remaining)
            and indices == sorted(set(indices))
            and bool(indices)
            and remaining == sorted(set(remaining))
            and set(remaining) <= set(indices) <= set(selected_rows),
            "journal retry plan scope invalid",
        )
        plan_ids.add(plan["plan_id"])
        planned.update(indices)
        plan_state = plan.get("execution_state")
        require(
            plan_state in {None, "complete", "failed", "interrupted"},
            "journal retry plan terminal state invalid",
        )
        require(
            plan_state != "complete" or not remaining,
            "journal completed retry plan has pending rows",
        )
        # Historical failed retry plans retain their unresolved rows. Only a
        # current plan lacks execution_state and blocks completeness.
        unfinished |= plan_state is None
    require(retried_rows <= planned, "journal retry plan history missing")
    return {
        "complete": not failures
        and not pending_rows
        and not unfinished
        and set(results) == set(selected_rows),
        "prior_failures": any(v["state"] == "failed" for v in terminal.values()),
        "accepted_retry_result": bool(retried_rows) and not failures,
        "inputs": PROVENANCE.AttemptJournalInputs(**saved),
    }


def audit_row_provenance(
    dataset,
    index,
    row,
    record,
    *,
    independent_inputs,
    run_profile,
    attempt_journal,
    selected_rows,
    provenance_provider,
    verifier_profile,
    anchor_lookup_key,
):
    """Expected identity comes only from independently selected typed inputs.

    No provider, trace, manifest or checkpoint field selects the trusted adapter
    or expected identity. Asset-byte checks remain the caller's pinned-input
    boundary; the auditor additionally checks the selected row/observable scope
    and independently reconciles the complete supplied durable history.
    """
    unverified = PROVENANCE.ProvenanceResult(
        "PROVENANCE_UNVERIFIED", "independent provenance inputs unavailable"
    )
    details = {
        "complete": False,
        "valid": None,
        "prior_failures": False,
        "accepted_retry_result": False,
    }
    if independent_inputs is None or run_profile is None or attempt_journal is None:
        return unverified, details
    require(
        isinstance(independent_inputs, PROVENANCE.IndependentAuditInputs),
        "independent audit inputs required",
    )
    require(
        isinstance(run_profile, PROVENANCE.TrustedRunProfile),
        "trusted run profile required",
    )
    require(
        independent_inputs.dataset == dataset
        and independent_inputs.source_row_index == index
        and independent_inputs.fixture == row["fixture"]
        and independent_inputs.reference == row["reference"]
        and set(independent_inputs.required_observables) == set(SECTIONS),
        "independent row/observable scope mismatch",
    )
    try:
        reconciled = audit_attempt_journal(
            dataset, index, row, record, run_profile, attempt_journal, selected_rows
        )
    except (ValueError, TypeError, KeyError, RecursionError) as error:
        details["valid"] = False
        return PROVENANCE.ProvenanceResult(
            "PROVENANCE_UNVERIFIED", f"journal reconciliation failed: {error}"
        ), details
    details.update(reconciled, valid=True)
    identity = PROVENANCE.derive_expected_row_identity(
        independent_inputs,
        run_profile,
        callbacks=record["original_entrypoint"]["captured_calls"],
        attempt_journal=reconciled["inputs"],
    )
    provenance = PROVENANCE.verify_external_anchor(
        PROVENANCE.commitment_document(identity),
        provider=provenance_provider,
        verifier_profile=verifier_profile,
        lookup_key=anchor_lookup_key or f"{run_profile.run_id}:{dataset}:{index}",
    )
    if provenance.status == "PROVENANCE_VERIFIED" and not reconciled["complete"]:
        provenance = PROVENANCE.ProvenanceResult(
            "PROVENANCE_UNVERIFIED",
            "journal selected scope incomplete",
            provenance.anchor_id,
        )
    return provenance, details


def audit_record(
    dataset,
    index,
    row,
    record,
    zero=(),
    polynomial=(),
    root_records=(),
    provenance_provider=None,
    provenance_binding=None,
    *,
    independent_inputs=None,
    run_profile=None,
    attempt_journal=None,
    selected_rows=None,
    verifier_profile=None,
    anchor_lookup_key=None,
    execution_state="complete",
    required_evidence_complete=True,
):
    require(
        record.get("schema") == "gragra.gen1.refinement-trace.v4",
        "record schema mismatch",
    )
    validate_mpf_displays(record)
    require(
        record.get("dataset") == dataset
        and type(record.get("source_row_index")) is int
        and record["source_row_index"] == index,
        "record dataset/index mismatch",
    )
    require(record.get("fixture") == row["fixture"], "record fixture mismatch")
    require(
        record.get("frozen_reference")
        == {"reference": row["reference"], "metadata": row["metadata"]},
        "frozen row mismatch",
    )
    require(
        record.get("zero_certificate_supplement_sha256") == ZERO_SHA256,
        "zero supplement pin mismatch",
    )
    require(
        record.get("polynomial_certificate_supplement_sha256")
        == (POLYNOMIAL_SHA256 if dataset in {"source", "noncavity"} else None),
        "polynomial supplement pin mismatch",
    )
    worker = record["process_independence"]
    require(type(worker.get("pid")) is int and worker["pid"] > 0, "worker pid missing")
    require(
        worker.get("modules_before") == []
        and worker.get("modules_after") == []
        and worker.get("production_module_names") == []
        and worker.get("production_imported") is False
        and worker.get("independence_check_passed") is True,
        "worker production independence failed",
    )
    require(
        record.get("status") != "replay_error",
        f"replay execution failure: {record.get('error')}",
    )
    original = record["original_entrypoint"]
    require(
        original["actual_reference"] == row["reference"]
        and original["actual_metadata"] == row["metadata"],
        "original entrypoint reproduction mismatch",
    )
    require(
        record["reproduction"]
        == {
            "reference_matches_full_values_uncertainty_methods_components": True,
            "metadata_matches": True,
            "passed": True,
            "failed_checks": [],
        },
        "reproduction flags mismatch",
    )
    calls = original["captured_calls"]
    require(
        type(original["capture_call_count"]) is int
        and original["capture_call_count"] == len(calls),
        "capture count mismatch",
    )
    require(
        all(
            isinstance(c, dict)
            and isinstance(c.get("inputs"), dict)
            and c.get("kernel")
            and "output" in c
            for c in calls
        ),
        "capture schema incomplete",
    )
    traces = record["observable_traces"]
    require(set(traces) == set(SECTIONS), "observable coverage incomplete")
    raw_unresolved, unresolved, routes = [], [], {}
    fixture = row["fixture"]
    whole = dataset == "high_sphere" and row["metadata"].get("exact_zero_certificate")
    require(bool(calls) or whole, "applicable captures are empty")
    for name in SECTIONS:
        trace, reference = traces[name], row["reference"][name]
        uncertainty = reference["uncertainty"]
        if whole:
            require(
                trace.get("status") == "exact_whole_case_certificate"
                and trace.get("certificate") == whole
                and not calls,
                "whole-case proof mismatch",
            )
            require(
                all(value == [0.0, 0.0] for value in reference["values_si"]),
                "whole-case field is not zero",
            )
            whole_case_reason = "no observable evaluation occurs for this certified row"
            audit_trend(
                trace["derivative_trend"],
                None,
                None,
                expected_reason=whole_case_reason,
            )
            audit_trend(
                trace["quadrature_trend"],
                None,
                None,
                expected_reason=whole_case_reason,
            )
            routes[name] = "exact_whole_case_certificate"
        elif uncertainty["uncertainty_kind"] == "rigorous_bound":
            require(
                trace.get("status") == "certified_override"
                and trace.get("certificate") == uncertainty["method"]
                and trace.get("captured_evaluations_were_used_for_this_observable")
                is False,
                "frozen exact override mismatch",
            )
            require(
                uncertainty["used_uncertainty_si"] == 0
                and all(value == [0.0, 0.0] for value in reference["values_si"]),
                "unsupported nonzero rigorous override",
            )
            override_reason = certified_override_na_reason(dataset, fixture)
            audit_trend(
                trace["derivative_trend"],
                None,
                None,
                expected_reason=override_reason,
            )
            audit_trend(
                trace["quadrature_trend"],
                None,
                None,
                expected_reason=override_reason,
            )
            routes[name] = "certified_override"
        else:
            require(trace.get("status") == "numerical", "expected numerical trace")
            surface = (
                dataset in {"cuboid", "cylinder"}
                or dataset == "source"
                and fixture.get("shape") != "sphere"
                and fixture["family"] != "temperature_grid"
            )
            if surface:
                surface_trace(
                    dataset,
                    fixture,
                    name,
                    trace,
                    reference,
                    row["metadata"],
                    calls,
                    root_records,
                )
            else:
                require(trace.get("route") == "mpmath", "MP route label mismatch")
                mp_trace(dataset, fixture, name, trace, reference, calls)
            routes[name] = trace["route"]
        raw, adjudicated = audit_annotations(
            dataset, index, fixture, name, trace, reference, zero, polynomial
        )
        for key, classification in raw.items():
            if classification.startswith("unresolved"):
                item = f"{name}/{key}"
                raw_unresolved.append(item)
                if key not in adjudicated:
                    unresolved.append(item)
    require(
        record.get("raw_unresolved_trends") == raw_unresolved,
        "raw unresolved summary mismatch",
    )
    require(
        record.get("unresolved_trends") == unresolved,
        "adjudicated unresolved summary mismatch",
    )
    status = "unresolved_trend" if unresolved else "reproduced"
    require(record.get("status") == status, "derived record status mismatch")
    # Apply only after every numerical callback, pair, component, floor and
    # original status has reconciled. The saved row/status is never rewritten.
    supplement = high_sphere_scalar_zero_supplement(dataset, index, row, traces)
    audit_unresolved = [
        item
        for item in unresolved
        if not (supplement and item == supplement["raw_unresolved_trend"])
    ]
    # Legacy mutable binding never supplies independent expectations.
    del provenance_binding
    provenance_result, journal = audit_row_provenance(
        dataset,
        index,
        row,
        record,
        independent_inputs=independent_inputs,
        run_profile=run_profile,
        attempt_journal=attempt_journal,
        selected_rows={index: row} if selected_rows is None else selected_rows,
        provenance_provider=provenance_provider,
        verifier_profile=verifier_profile,
        anchor_lookup_key=anchor_lookup_key,
    )
    provenance = {
        "status": provenance_result.status,
        "reason": provenance_result.reason,
        "anchor_id": provenance_result.anchor_id,
    }
    admission = PROVENANCE.admission_decision(
        execution_state=execution_state,
        semantic_status="SEMANTIC_VALID",
        provenance_status=provenance_result.status,
        required_evidence_complete=required_evidence_complete,
        independent_audit_valid=True,
        certificate_validation_valid=True,
        trend_validation_complete=not audit_unresolved,
        declared_scope_valid=journal["valid"] is not False,
        journal_binding_complete=journal["complete"],
        prior_failures=journal["prior_failures"],
        accepted_retry_result=journal["accepted_retry_result"],
        accepted_result_anchored=journal["complete"]
        if journal["prior_failures"]
        else True,
    )
    return {
        "dataset": dataset,
        "source_row_index": index,
        "status": status,
        "semantic_validation": "SEMANTIC_VALID",
        "provenance_validation": provenance,
        "scientific_admission": admission,
        "certificate_validation": "CERTIFICATE_SCOPE_VALID",
        "journal_binding_complete": journal["complete"],
        "routes": routes,
        "raw_unresolved_trends": raw_unresolved,
        "unresolved_trends": unresolved,
        "audit_unresolved_trends": audit_unresolved,
        "audit_certificate_supplements": [] if supplement is None else [supplement],
    }


def audit_saved_record(dataset, index, row, record, **kwargs):
    """Report known semantic invalidity without overriding the exception API.

    Missing optional audit assets are passed through required_evidence_complete;
    absent saved rows are SEMANTIC_NOT_RUN. A present malformed row is invalid
    even when other required evidence has not been supplied.
    """
    if record is not None:
        try:
            return audit_record(dataset, index, row, record, **kwargs)
        except (ValueError, TypeError, KeyError, IndexError) as error:
            semantic, reason = "SEMANTIC_INVALID", str(error)
    else:
        semantic, reason = "SEMANTIC_NOT_RUN", "saved row evidence missing"
    admission = PROVENANCE.admission_decision(
        execution_state=kwargs.get("execution_state", "complete"),
        semantic_status=semantic,
        provenance_status="PROVENANCE_UNVERIFIED",
        required_evidence_complete=False,
        independent_audit_valid=False if record is not None else None,
        certificate_validation_valid=None,
        trend_validation_complete=False,
        declared_scope_valid=None,
        journal_binding_complete=False,
        prior_failures=False,
        accepted_retry_result=False,
        accepted_result_anchored=False,
    )
    return {
        "dataset": dataset,
        "source_row_index": index,
        "semantic_validation": semantic,
        "semantic_error": reason,
        "provenance_validation": {
            "status": "PROVENANCE_UNVERIFIED",
            "reason": "semantic audit incomplete",
            "anchor_id": None,
        },
        "scientific_admission": admission,
        "certificate_validation": "CERTIFICATE_SCOPE_NOT_RUN",
        "journal_binding_complete": False,
    }


def inspect_historical_record(record):
    """Recognize a historical envelope without applying or upgrading V4 verdicts."""
    schema = record.get("schema") if isinstance(record, dict) else None
    version = HISTORICAL_SCHEMAS.get(schema)
    if version is None:
        raise ValueError("record is not a recognized historical V1/V2/V3 envelope")
    return {
        "historical_format": version,
        "schema": schema,
        "format_readable": True,
        "semantic_validation": "SEMANTIC_NOT_RUN",
        "provenance_validation": "PROVENANCE_UNVERIFIED",
        "scientific_admission": "NOT_ADMITTED",
        "historical_source_or_trace_modified": False,
        "v4_verdict_applied_retroactively": False,
    }


def certificate_inputs(l6_root):
    zero_path = l6_root / "frozen-zero-estimate-certificates-integrated.json"
    poly_path = l6_root / "polynomial-quadrature-certificates-integrated.json"
    require(sha256(zero_path) == ZERO_SHA256, "zero certificate bytes changed")
    require(
        sha256(poly_path) == POLYNOMIAL_SHA256, "polynomial certificate bytes changed"
    )
    zero, polynomial = read_json(zero_path), read_json(poly_path)
    require(
        zero["certified_records"] == 173 and zero["uncertified_zero_records"] == 0,
        "zero certificate coverage changed",
    )
    require(
        polynomial["script_sha256"] == POLYNOMIAL_SCRIPT_SHA256
        and polynomial["production_modules_observed"] == []
        and len(polynomial["certificates"]) == 33,
        "polynomial provenance changed",
    )
    return zero["certificates"], polynomial["certificates"]


def load_root_supplement(path, expected_sha256, l6_root):
    require(sha256(path) == expected_sha256, "root supplement digest mismatch")
    data = PROVENANCE.strict_json_object(Path(path).read_bytes())
    require(
        set(data)
        == {
            "schema",
            "classification",
            "method",
            "generator_sha256",
            "audit_helper_sha256",
            "oracle_sha256",
            "reference_sha256",
            "original_values_uncertainty_labels_unchanged",
            "original_d43_callback_capture",
            "finite_precision_or_total_error_bound_claim",
            "fixture_count",
            "unique_key_count",
            "entries",
            "production_modules",
            "command",
            "environment",
        },
        "root supplement has missing or extra fields",
    )
    require(
        data["schema"] == "gragra.gen1.cylinder-root-normalization-supplement.v1"
        and data["classification"] == "observed supplemental root-only reconstruction"
        and data["method"]
        == "pinned cylinder_radial_data at 50/80; estimate at 80 digits",
        "root supplement method/scope mismatch",
    )
    require(
        data["generator_sha256"]
        == ROOT_SUPPLEMENT_GENERATOR_SHA256
        == sha256(ORACLE_DIRECTORY / "record_cylinder_root_normalization.py")
        and data["audit_helper_sha256"] == ROOT_SUPPLEMENT_AUDIT_HELPER_SHA256
        and data["audit_helper_sha256"]
        == sha256(ORACLE_DIRECTORY / "audit_frozen_refinement_traces.py")
        and data["oracle_sha256"] == EXPECTED_ORACLE_SHA256
        and data["reference_sha256"] == EXPECTED_REFERENCE_SHA256["cylinder"],
        "root supplement source/reference pins mismatch",
    )
    require(
        data["original_values_uncertainty_labels_unchanged"] is True
        and data["original_d43_callback_capture"] is False
        and data["finite_precision_or_total_error_bound_claim"] is False
        and data["production_modules"] == [],
        "root supplement classification mismatch",
    )
    rows = read_rows(l6_root / DATASETS["cylinder"] / "references.jsonl")
    expected = {}
    for index, row in enumerate(rows):
        fixture = row["fixture"]
        key = (
            abs(fixture["order"]),
            fixture["radial_index"],
            fixture["radius_m"],
            fixture["axial_index"],
        )
        expected.setdefault(key, []).append(
            {"source_row_index": index, "fixture": fixture}
        )
    entries = data["entries"]
    require(
        data["fixture_count"] == 576
        and data["unique_key_count"] == len(expected) == len(entries)
        and len({tuple(r["key"]) for r in entries}) == len(entries),
        "root supplement coverage mismatch",
    )
    for entry in entries:
        key = tuple(entry["key"])
        require(
            entry["source_rows"] == expected.get(key),
            "root supplement full fixture binding mismatch",
        )
        require(
            len(entry["at_80_digits"]) == len(entry["at_50_digits"]) == 2,
            "root supplement precision pairs missing",
        )
        with mp.workdps(80):
            value = root_precision(entry["source_rows"][0]["fixture"], [entry])
            require(
                mp.isfinite(value) and value >= 0, "invalid root sensitivity estimate"
            )
            high_root, high_norm = map(decode, entry["at_80_digits"])
            for binding in entry["source_rows"]:
                metadata = rows[binding["source_row_index"]]["metadata"]
                require(
                    metadata["root"] == mp.nstr(high_root, 80)
                    and metadata["normalization"] == mp.nstr(high_norm, 80),
                    "root supplement frozen metadata mismatch",
                )
    return entries


def dataset_overrides(replay_options, source_options, sha_options):
    """Explicitly pin a replacement Cylinder run, never migrate old metadata."""

    def parse(options):
        result = {}
        for option in options:
            key, separator, value = option.partition("=")
            require(
                separator and key == "cylinder" and value and key not in result,
                "override must be a unique cylinder=value entry",
            )
            result[key] = value
        return result

    roots, sources, digests = map(parse, (replay_options, source_options, sha_options))
    require(
        set(roots) == set(sources) == set(digests),
        "Cylinder override requires replay root, source and independent SHA",
    )
    for key, path in sources.items():
        require(
            len(digests[key]) == 64 and sha256(path) == digests[key],
            "replacement observer independently pinned SHA mismatch",
        )
        require(
            Path(roots[key]).resolve() != Path(path).resolve(),
            "replacement replay root must be a directory",
        )
    return roots, sources, digests


def audit_dataset(*_args, **_kwargs):
    """The V4 run/journal/provenance audit is not implemented at this stage."""
    raise RuntimeError("V4 dataset audit is not implemented")


def main(_argv=None):
    """Fail closed until independently pinned V4 run inputs are available."""
    raise RuntimeError("V4 run audit is not implemented")


if __name__ == "__main__":
    raise SystemExit(main())
