"""Independent second-order admission arithmetic; imports no production code.

Decimal strings and binary64 promotions are exact rationals. Square roots use
outward rational brackets, checked at 120 and 180 significant digits. This
bounds decision arithmetic; it does not turn estimated oracle U into a theorem.
"""

import json
import math
import re
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN, Decimal, localcontext
from fractions import Fraction
from itertools import combinations

KEY_FIELDS = (
    "family",
    "fixture_id",
    "frequency_hz",
    "target_id",
    "observable",
    "representation",
    "set",
)
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
CSD_GATES = (
    "hermitian",
    "real_diagonal",
    "nonnegative_diagonal",
    "positive_semidefinite",
    "natural_frame",
)
SCALAR_GATES = ("real_diagonal", "nonnegative_diagonal")
CONTRACT_GATES = (
    "units",
    "shape",
    "frequency_axis",
    "input_normalization",
    "source_connections",
    "resolved_reference",
    "sealed_exact_coverage",
)
_DECIMAL = re.compile(r"(?:0e\+0|-?[1-9](?:\.[0-9]*[1-9])?e[+-](?:0|[1-9][0-9]*))\Z")
RECORD_FIELDS = {
    "schema_version",
    "key",
    "status",
    "quantity",
    "uncertainty",
    "input_assumptions",
    "source_pins",
    "procedure_hash",
    "inputs",
    "units",
    "input_convention",
    "admission_status",
}


def canonical_decimal(value):
    """Encode a finite Decimal without context-dependent normalization."""
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ValueError("finite Decimal required")
    if not value:
        return "0e+0"
    sign, digits, exponent = value.as_tuple()
    text = "".join(map(str, digits))
    adjusted = exponent + len(text) - 1
    text = text.rstrip("0")
    mantissa = text[0] + ("." + text[1:] if len(text) > 1 else "")
    return ("-" if sign else "") + mantissa + f"e{adjusted:+d}"


def decimal_fraction(value):
    """Accept canonical scientific strings only; no unknown-to-zero coercion."""
    if not isinstance(value, str) or not _DECIMAL.fullmatch(value):
        raise ValueError("noncanonical decimal string")
    if len(value) > 4096 or abs(int(value.split("e")[1])) > 10000:
        raise ValueError("decimal resource limit")
    if canonical_decimal(Decimal(value)) != value:
        raise ValueError("noncanonical decimal spelling")
    return Fraction(Decimal(value))


def promote_binary64(value):
    """Preserve every bit of an actual Python binary64 value, including subnormals."""
    if type(value) is not float or not math.isfinite(value):
        raise ValueError("finite binary64 required")
    return canonical_decimal(Decimal.from_float(value))


def fraction_decimal(value, digits=120, upward=False, downward=False):
    with localcontext() as ctx:
        ctx.prec = digits
        ctx.rounding = (
            ROUND_CEILING if upward else ROUND_FLOOR if downward else ROUND_HALF_EVEN
        )
        ctx.Emax = 999999
        ctx.Emin = -999999
        return canonical_decimal(Decimal(value.numerator) / Decimal(value.denominator))


def _no_floats(value):
    if isinstance(value, float):
        raise ValueError("JSON scientific numbers must be decimal strings")
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        for item in value.values():
            _no_floats(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _no_floats(item)


def canonical_json(value):
    _no_floats(value)
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_number(value):
    raise ValueError(f"noninteger JSON number {value}")


def read_json(raw):
    return json.loads(
        raw,
        object_pairs_hook=_pairs,
        parse_float=_reject_number,
        parse_constant=_reject_number,
    )


def _exact_fields(value, fields):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError("unknown or missing object fields")


def quantity_values(quantity):
    _exact_fields(quantity, {"shape", "data"})
    shape = quantity["shape"]
    if shape not in ([], [3, 3]) or any(type(x) is not int for x in shape):
        raise ValueError("unsupported explicit shape")
    data = quantity["data"]
    if not isinstance(data, list) or len(data) != (9 if shape else 1):
        raise ValueError("shape/data mismatch")
    values = []
    for item in data:
        _exact_fields(item, {"real", "imag"})
        values.append((decimal_fraction(item["real"]), decimal_fraction(item["imag"])))
    return values


def _sqrt_bounds(square, digits):
    if square < 0:
        raise ValueError("negative squared norm")
    n, d = math.isqrt(square.numerator), math.isqrt(square.denominator)
    if n * n == square.numerator and d * d == square.denominator:
        exact = Fraction(n, d)
        return exact, exact
    with localcontext() as ctx:
        ctx.prec = digits + 20
        ctx.Emax, ctx.Emin = 999999, -999999
        approx = (Decimal(square.numerator) / Decimal(square.denominator)).sqrt()
        ctx.prec = digits
        rounded = +approx
        lo, hi = Fraction(rounded.next_minus()), Fraction(rounded.next_plus())
    # Exact rational comparisons verify brackets; no assumptions about sqrt errors.
    if lo < 0 or lo * lo > square or hi * hi < square:
        raise ValueError("square-root bracket verification failed")
    return lo, hi


def norm_bounds(quantity, digits=120):
    return _sqrt_bounds(
        sum((a * a + b * b for a, b in quantity_values(quantity)), Fraction()), digits
    )


def _uncertainty(value):
    _exact_fields(value, {"kind", "components", "certificate"})
    if value["kind"] not in {"estimate", "rigorous_bound", "exact_zero"}:
        raise ValueError("unknown uncertainty kind")
    _exact_fields(value["components"], COMPONENTS)
    total = Fraction()
    for component in value["components"].values():
        _exact_fields(component, {"value", "method"})
        term = decimal_fraction(component["value"])
        if (
            term < 0
            or not isinstance(component["method"], str)
            or not component["method"].strip()
        ):
            raise ValueError("invalid uncertainty component")
        total += term
    if value["kind"] == "exact_zero":
        if (
            total
            or not isinstance(value["certificate"], str)
            or not value["certificate"].strip()
        ):
            raise ValueError("whole-observable zero certificate required")
    elif value["kind"] == "rigorous_bound":
        if (
            not isinstance(value["certificate"], str)
            or not value["certificate"].strip()
        ):
            raise ValueError("full bound proof citation required")
    elif value["certificate"] is not None:
        raise ValueError("estimate certificate must be null")
    return total


def assess_numeric(
    production,
    reference,
    source_scale,
    absolute_rate,
    relative_rate,
    uncertainty,
):
    """Numeric conditions only; never returns composite science ``accepted``.

    This function consumes synthetic or serialized quantities. Oracle authors
    must not call it with production output or inspect comparison results.
    """
    pv, rv = quantity_values(production), quantity_values(reference)
    if production["shape"] != reference["shape"]:
        raise ValueError("shape mismatch")
    scale, a, r = map(decimal_fraction, (source_scale, absolute_rate, relative_rate))
    if min(scale, a, r) < 0:
        raise ValueError("negative budget input")
    try:
        used = _uncertainty(uncertainty)
    except (ValueError, TypeError):
        return {
            "numeric_accepted": False,
            "status": "uncertainty_unresolved",
            "claim": None,
        }
    ref2 = sum((x * x + y * y for x, y in rv), Fraction())
    if used == 0 and uncertainty["kind"] != "exact_zero":
        return {
            "numeric_accepted": False,
            "status": "uncertainty_unresolved",
            "claim": None,
        }
    diff2 = sum(
        ((p - x) ** 2 + (q - y) ** 2 for (p, q), (x, y) in zip(pv, rv, strict=True)),
        Fraction(),
    )
    if (scale == 0 or uncertainty["kind"] == "exact_zero") and not (
        scale == 0 and ref2 == 0 and used == 0 and uncertainty["kind"] == "exact_zero"
    ):
        return {
            "numeric_accepted": False,
            "status": "zero_certificate_invalid",
            "claim": None,
        }
    checks = []
    for digits in (120, 180):
        nl, nh = _sqrt_bounds(ref2, digits)
        dl, dh = _sqrt_bounds(diff2, digits)
        bl, bh = a * scale + r * nl, a * scale + r * nh
        numeric = True if dh + used <= bl else False if dl + used > bh else None
        oracle = True if 10 * used <= bl else False if 10 * used > bh else None
        checks.append((numeric, oracle))
    ambiguous = None in checks[0] or checks[0] != checks[1]
    accepted = not ambiguous and all(checks[0])
    claim = None
    if accepted:
        claim = (
            "based_on_recorded_oracle_uncertainty_estimate"
            if uncertainty["kind"] == "estimate"
            else "based_on_recorded_whole_observable_certificate"
        )
    return {
        "numeric_accepted": accepted,
        "status": "rounding_ambiguous" if ambiguous else "assessed",
        "claim": claim,
        "difference_plus_uncertainty_condition": checks[0][0],
        "oracle_fraction_condition": checks[0][1],
        "difference_upper": fraction_decimal(dh, upward=True),
        "reference_norm_upper": fraction_decimal(nh, upward=True),
        "budget_lower": fraction_decimal(bl, downward=True),
        "budget_upper": fraction_decimal(bh, upward=True),
        "uncertainty_upper": fraction_decimal(used, upward=True),
    }


def assess(
    production,
    reference,
    source_scale,
    absolute_rate,
    relative_rate,
    uncertainty,
    structure_conditions=None,
    contract_conditions=None,
):
    """Strict conjunction: exact applicable gate sets must all explicitly pass.

    A seal and source-connection assessment remain external. This API requires
    their explicit gate results; a numeric-only assessment cannot substitute.
    """
    result = assess_numeric(
        production, reference, source_scale, absolute_rate, relative_rate, uncertainty
    )
    required = CSD_GATES if reference["shape"] == [3, 3] else SCALAR_GATES
    try:
        _exact_fields(structure_conditions, required)
        _exact_fields(contract_conditions, CONTRACT_GATES)
    except ValueError:
        return {
            **result,
            "accepted": False,
            "status": "required_gates_unresolved",
            "claim": None,
        }
    values = [*structure_conditions.values(), *contract_conditions.values()]
    if any(type(value) is not bool for value in values):
        raise ValueError("gates require explicit booleans")
    accepted = result["numeric_accepted"] and all(values)
    return {
        **result,
        "accepted": accepted,
        "claim": result["claim"] if accepted else None,
    }


def _cadd(a, b):
    return a[0] + b[0], a[1] + b[1]


def _cmul(a, b):
    return a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]


def _conj(a):
    return a[0], -a[1]


def _within(a, tolerance):
    return a[0] * a[0] + a[1] * a[1] <= tolerance * tolerance


def structure_gates(quantity, symmetry, tolerance):
    """Exact principal-minor PSD test on Hermitian part + tolerance*I.

    Every gate is separate. PSD allowance cannot cover anti-Hermitian or frame
    defects. Residual tolerance has the same units as the full observable.
    """
    values = quantity_values(quantity)
    tau = decimal_fraction(tolerance)
    if tau < 0:
        raise ValueError("negative structural tolerance")
    if quantity["shape"] == []:
        return {
            "real_diagonal": abs(values[0][1]) <= tau,
            "nonnegative_diagonal": values[0][0] >= -tau,
        }
    m = [values[i : i + 3] for i in (0, 3, 6)]
    real_diag = all(abs(m[i][i][1]) <= tau for i in range(3))
    nonnegative = all(m[i][i][0] >= -tau for i in range(3))
    hermitian = all(
        _within(_cadd(m[i][j], tuple(-v for v in _conj(m[j][i]))), tau)
        for i in range(3)
        for j in range(3)
    )
    h = [
        [tuple(v / 2 for v in _cadd(m[i][j], _conj(m[j][i]))) for j in range(3)]
        for i in range(3)
    ]
    for i in range(3):
        h[i][i] = h[i][i][0] + tau, Fraction()
    minors = [h[i][i][0] for i in range(3)]
    for i, j in combinations(range(3), 2):
        minors.append(h[i][i][0] * h[j][j][0] - sum(v * v for v in h[i][j]))
    determinant = h[0][0][0] * h[1][1][0] * h[2][2][0]
    determinant += 2 * _cmul(_cmul(h[0][1], h[1][2]), h[2][0])[0]
    determinant -= sum(
        h[i][i][0] * sum(v * v for v in h[j][k])
        for i, j, k in [(0, 1, 2), (1, 0, 2), (2, 0, 1)]
    )
    minors.append(determinant)
    if symmetry not in {"identity", "horizontal_vertical_2", "none"}:
        raise ValueError("unknown natural-frame symmetry")
    natural = True
    if symmetry != "none":
        natural = all(
            _within(m[i][j], tau) for i in range(3) for j in range(3) if i != j
        )
        natural &= abs(m[0][0][0] - m[1][1][0]) <= tau
        factor = 2 if symmetry == "horizontal_vertical_2" else 1
        natural &= abs(m[2][2][0] - factor * m[0][0][0]) <= tau
    return {
        "hermitian": hermitian,
        "real_diagonal": real_diag,
        "nonnegative_diagonal": nonnegative,
        "positive_semidefinite": min(minors) >= 0,
        "natural_frame": natural,
    }


def key_tuple(key):
    _exact_fields(key, KEY_FIELDS)
    if any(not isinstance(key[field], str) or not key[field] for field in KEY_FIELDS):
        raise ValueError("key fields require nonempty strings")
    if decimal_fraction(key["frequency_hz"]) <= 0:
        raise ValueError("positive frequency required")
    if key["set"] not in {"science", "auxiliary"}:
        raise ValueError("unknown case set")
    if key["representation"] not in {"Gfree", "SI", "dimensionless"}:
        raise ValueError("unknown representation")
    return tuple(key[field] for field in KEY_FIELDS)


def validate_record(record):
    _exact_fields(record, RECORD_FIELDS)
    if type(record["schema_version"]) is not int or record["schema_version"] != 1:
        raise ValueError("schema version mismatch")
    key_tuple(record["key"])
    if record["status"] not in {"resolved", "unresolved"}:
        raise ValueError("unknown reference status")
    if record["status"] == "resolved":
        quantity_values(record["quantity"])
        _uncertainty(record["uncertainty"])
    elif record["quantity"] is not None:
        raise ValueError("unresolved quantity must be null")
    if (
        not isinstance(record["input_assumptions"], str)
        or not record["input_assumptions"].strip()
    ):
        raise ValueError("input assumptions required")
    if record["admission_status"] != "NOT_FORMALLY_ADMITTED":
        raise ValueError("generator cannot assert formal admission")
    for field in ("units", "input_convention"):
        if not isinstance(record[field], str) or not record[field].strip():
            raise ValueError("units and input convention required")
    if not isinstance(record["inputs"], dict) or not record["inputs"]:
        raise ValueError("complete frozen physical inputs required")
    _no_floats(record["inputs"])
    if not re.fullmatch(r"[0-9a-f]{64}", record["procedure_hash"]):
        raise ValueError("procedure SHA256 required")
    if not isinstance(record["source_pins"], list) or not record["source_pins"]:
        raise ValueError("source pins required")
    for pin in record["source_pins"]:
        _exact_fields(pin, {"sha256", "equations", "pages"})
        if not re.fullmatch(r"[0-9a-f]{64}", pin["sha256"]):
            raise ValueError("source SHA256 required")
        if not pin["equations"] or not pin["pages"]:
            raise ValueError("source equations/pages required")
    return record


def admit_records(records, expected_keys):
    """Envelope/usable-key coverage only, never formal admission or sealing."""
    expected = [key_tuple(key) for key in expected_keys]
    if not expected:
        raise ValueError("nonempty frozen expected keyset required")
    if len(set(expected)) != len(expected):
        raise ValueError("duplicate frozen expected key")
    generated, unresolved = [], []
    for record in records:
        validate_record(record)
        key = key_tuple(record["key"])
        generated.append(key)
        if record["status"] != "resolved":
            unresolved.append(key)
    duplicate = len(generated) != len(set(generated))
    missing, unknown = (
        sorted(set(expected) - set(generated)),
        sorted(set(generated) - set(expected)),
    )
    return {
        "coverage_valid": not (duplicate or missing or unknown or unresolved),
        "duplicate": duplicate,
        "missing": missing,
        "unknown": unknown,
        "unresolved": unresolved,
        "expected_count": len(expected),
        "generated_count": len(generated),
    }
