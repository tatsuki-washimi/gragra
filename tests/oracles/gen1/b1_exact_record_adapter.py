"""Exact adapter for the sealed B1 science records and accepted review sidecar.

This module contains no production imports. Values from source artifacts remain
``Fraction`` objects; only the pinned arithmetic kernel boundary receives
canonical decimal strings. The accepted sidecar does not serialize per-source
uncertainty values, so the adapter transports only the exact aggregate bound.
"""

from __future__ import annotations

import json
import re
from fractions import Fraction
from typing import Any, Iterable


KEY_FIELDS = (
    "family",
    "fixture_id",
    "frequency_hz",
    "target_id",
    "observable",
    "representation",
    "set",
)
UNCERTAINTY_COMPONENTS = (
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
RAW_FIELDS = {"key", "status", "enclosure"}
ENCLOSURE_FIELDS = {"central", "digits", "lo_exact", "hi_exact", "u_up"}
SIDECAR_FIELDS = {
    "B1_nine_source_allocation_sum_equals_u_up",
    "B_ref_lower",
    "classification",
    "conservative_10_U_matrix_over_B_ref_lower",
    "exact_endpoint_and_no_narrowing",
    "key",
    "preproduction_oracle_fraction_pass",
    "source_scale",
    "status",
    "u_up",
    "uncertainty_certificate",
    "uncertainty_matrix_upper",
    "units",
    "zero_or_nonzero_semantics",
}
_RATIO = re.compile(r"(?P<n>-?(?:0|[1-9][0-9]*))/(?P<d>[1-9][0-9]*)\Z")
_DECIMAL = re.compile(
    r"(?P<sign>-?)(?P<int>0|[1-9][0-9]*)(?:\.(?P<frac>[0-9]+))?(?:[eE](?P<exp>[+-]?(?:0|[1-9][0-9]*)))?\Z"
)
_MAX_TEXT = 8192
_MAX_ABS_EXPONENT = 10000
_MAX_SIGNIFICANT_DIGITS = 1000


def _reject_json_number(value: str) -> None:
    raise ValueError(f"JSON noninteger number is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON object field: {key}")
        output[key] = value
    return output


def strict_json_loads(raw: str | bytes) -> Any:
    """Parse JSON while rejecting duplicate keys and binary-float tokens."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="strict")
    if not isinstance(raw, str) or len(raw) > 2_000_000:
        raise ValueError("JSON input must be bounded UTF-8 text")
    return json.loads(
        raw,
        object_pairs_hook=_unique_object,
        parse_float=_reject_json_number,
        parse_constant=_reject_json_number,
    )


def parse_exact_fraction(value: str) -> Fraction:
    """Parse a bounded decimal/scientific or integer-ratio string exactly."""
    if not isinstance(value, str):
        raise TypeError("exact rational input must be a string")
    if not value or len(value) > _MAX_TEXT or value.strip() != value:
        raise ValueError("exact rational text is empty, oversized, or has whitespace")

    ratio = _RATIO.fullmatch(value)
    if ratio:
        numerator = int(ratio.group("n"))
        denominator = int(ratio.group("d"))
        return Fraction(numerator, denominator)

    decimal = _DECIMAL.fullmatch(value)
    if not decimal:
        raise ValueError("malformed exact decimal/rational text")
    exponent_text = decimal.group("exp") or "0"
    exponent = int(exponent_text)
    if abs(exponent) > _MAX_ABS_EXPONENT:
        raise ValueError("exact rational exponent resource limit")

    fractional_digits = decimal.group("frac") or ""
    coefficient_text = decimal.group("int") + fractional_digits
    if len(coefficient_text) > 4096:
        raise ValueError("exact rational coefficient resource limit")
    coefficient = int(coefficient_text)
    if decimal.group("sign") == "-":
        coefficient = -coefficient
    scale = len(fractional_digits) - exponent
    if abs(scale) > _MAX_ABS_EXPONENT:
        raise ValueError("exact rational scale resource limit")
    if scale >= 0:
        return Fraction(coefficient, 10**scale)
    return Fraction(coefficient * 10 ** (-scale), 1)


def _floor_log10(value: Fraction) -> int:
    if value <= 0:
        raise ValueError("positive rational required")
    numerator_digits = len(str(value.numerator))
    denominator_digits = len(str(value.denominator))
    exponent = numerator_digits - denominator_digits
    if exponent >= 0:
        if value.numerator < value.denominator * 10**exponent:
            exponent -= 1
    elif value.numerator * 10 ** (-exponent) < value.denominator:
        exponent -= 1
    return exponent


def fraction_to_exact_text(value: Fraction) -> str:
    """Serialize a Fraction exactly as a canonical decimal or reduced ratio."""
    if not isinstance(value, Fraction):
        raise TypeError("Fraction value required")
    if value == 0:
        return "0e+0"

    denominator = value.denominator
    twos = fives = 0
    while denominator % 2 == 0:
        denominator //= 2
        twos += 1
    while denominator % 5 == 0:
        denominator //= 5
        fives += 1
    if denominator != 1:
        return f"{value.numerator}/{value.denominator}"

    scale = max(twos, fives)
    coefficient = abs(value.numerator) * 2 ** (scale - twos) * 5 ** (scale - fives)
    digits = str(coefficient)
    adjusted_exponent = len(digits) - scale - 1
    digits = digits.rstrip("0")
    mantissa = digits[0] + (f".{digits[1:]}" if len(digits) > 1 else "")
    sign = "-" if value.numerator < 0 else ""
    result = f"{sign}{mantissa}e{adjusted_exponent:+d}"
    if parse_exact_fraction(result) != value:
        raise ValueError("exact decimal serialization failed to roundtrip")
    return result


def outward_decimal_upper(value: Fraction, significant_digits: int = 180) -> str:
    """Return a canonical finite decimal >= value using integer ceiling only."""
    if not isinstance(value, Fraction) or value < 0:
        raise ValueError("nonnegative Fraction required")
    if type(significant_digits) is not int or not 1 <= significant_digits <= _MAX_SIGNIFICANT_DIGITS:
        raise ValueError("significant digit resource limit")
    if value == 0:
        return "0e+0"

    quantum_exponent = _floor_log10(value) - significant_digits + 1
    if quantum_exponent >= 0:
        divisor = value.denominator * 10**quantum_exponent
        units = (value.numerator + divisor - 1) // divisor
        rounded = Fraction(units * 10**quantum_exponent, 1)
    else:
        multiplier = 10 ** (-quantum_exponent)
        scaled_numerator = value.numerator * multiplier
        units = (scaled_numerator + value.denominator - 1) // value.denominator
        rounded = Fraction(units, multiplier)
    if rounded < value:
        raise ValueError("outward decimal rounding narrowed exact value")
    return fraction_to_exact_text(rounded)


def key_tuple(key: dict[str, Any]) -> tuple[str, ...]:
    if not isinstance(key, dict) or set(key) != set(KEY_FIELDS):
        raise ValueError("key has unknown or missing fields")
    if any(not isinstance(key[name], str) or not key[name] for name in KEY_FIELDS):
        raise ValueError("key values must be nonempty strings")
    if parse_exact_fraction(key["frequency_hz"]) <= 0:
        raise ValueError("frequency must be positive")
    if key["set"] != "science" or key["representation"] not in {"Gfree", "SI"}:
        raise ValueError("B1 science key set or representation mismatch")
    return tuple(key[field] for field in KEY_FIELDS)


def _exact_fields(value: Any, fields: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{label} has unknown or missing fields")


def _parse_jsonl(raw_jsonl: str | bytes) -> list[dict[str, Any]]:
    if isinstance(raw_jsonl, bytes):
        raw_jsonl = raw_jsonl.decode("utf-8", errors="strict")
    if not isinstance(raw_jsonl, str) or len(raw_jsonl) > 10_000_000:
        raise ValueError("raw JSONL must be bounded UTF-8 text")
    records = []
    for line_number, line in enumerate(raw_jsonl.splitlines(), start=1):
        if not line:
            raise ValueError(f"blank raw JSONL line at {line_number}")
        record = strict_json_loads(line)
        _exact_fields(record, RAW_FIELDS, "raw record")
        records.append(record)
    if not records:
        raise ValueError("raw records are empty")
    return records


def _kernel_uncertainty(
    u_matrix_exact: Fraction,
    kind: str,
    certificate: str,
) -> tuple[dict[str, Any], Fraction]:
    if not isinstance(u_matrix_exact, Fraction) or u_matrix_exact < 0:
        raise ValueError("nonnegative exact matrix uncertainty required")
    expected_kind = "exact_zero" if kind == "exact_zero" else "rigorous_bound"
    if expected_kind == "exact_zero" and u_matrix_exact != 0:
        raise ValueError("exact-zero uncertainty acquired a nonzero aggregate")

    aggregate = fraction_to_exact_text(u_matrix_exact)
    if "/" in aggregate or parse_exact_fraction(aggregate) != u_matrix_exact:
        raise ValueError("aggregate kernel uncertainty is not an exact decimal")
    components: dict[str, dict[str, str]] = {}
    for name in UNCERTAINTY_COMPONENTS:
        if name == "formulation" and expected_kind != "exact_zero":
            components[name] = {
                "value": aggregate,
                "method": "aggregate transport of U_matrix_exact=3*u_up; not source attribution",
            }
        else:
            components[name] = {
                "value": "0e+0",
                "method": "exact zero unused kernel transport slot",
            }
    total = sum(
        (parse_exact_fraction(component["value"]) for component in components.values()),
        Fraction(),
    )
    if total != u_matrix_exact:
        raise ValueError("aggregate kernel uncertainty did not roundtrip exactly")
    return (
        {
            "kind": expected_kind,
            "components": components,
            "certificate": certificate,
        },
        total,
    )


def adapt_b1_records(
    raw_jsonl: str | bytes,
    sidecar_records: Iterable[dict[str, Any]],
    expected_keys: Iterable[dict[str, Any]],
    exact_zero_keys: Iterable[dict[str, Any]],
) -> dict[tuple[str, ...], dict[str, Any]]:
    """Reconcile pinned B1 records and sidecars into exact kernel quantities.

    ``sidecar_records`` and key manifests must come from a prior
    ``strict_json_loads`` of their containing artifacts, so duplicate object
    keys cannot be hidden by a conventional JSON parser.
    """
    raw_records = _parse_jsonl(raw_jsonl)
    raw_by_key: dict[tuple[str, ...], dict[str, Any]] = {}
    for record in raw_records:
        key_id = key_tuple(record["key"])
        if key_id in raw_by_key:
            raise ValueError("duplicate raw key")
        raw_by_key[key_id] = record

    expected_sequence = [key_tuple(key) for key in expected_keys]
    if not expected_sequence:
        raise ValueError("expected keyset is empty")
    if len(expected_sequence) != len(set(expected_sequence)):
        raise ValueError("duplicate expected key")
    expected = set(expected_sequence)
    actual = set(raw_by_key)
    if actual - expected:
        raise ValueError("unknown raw key")
    if expected - actual:
        raise ValueError("missing raw key")

    zero_sequence = [key_tuple(key) for key in exact_zero_keys]
    zero_set = set(zero_sequence)
    if len(zero_sequence) != len(zero_set) or not zero_set <= expected:
        raise ValueError("invalid exact-zero key manifest")

    sidecar_by_key: dict[tuple[str, ...], dict[str, Any]] = {}
    for sidecar in sidecar_records:
        _exact_fields(sidecar, SIDECAR_FIELDS, "review sidecar")
        key_id = key_tuple(sidecar["key"])
        if key_id in sidecar_by_key:
            raise ValueError("duplicate sidecar key")
        sidecar_by_key[key_id] = sidecar
    if set(sidecar_by_key) - expected:
        raise ValueError("unknown sidecar key")
    if expected - set(sidecar_by_key):
        raise ValueError("missing sidecar key")

    adapted: dict[tuple[str, ...], dict[str, Any]] = {}
    for key_id in expected_sequence:
        raw = raw_by_key[key_id]
        sidecar = sidecar_by_key[key_id]
        _exact_fields(raw["enclosure"], ENCLOSURE_FIELDS, "raw enclosure")
        enclosure = raw["enclosure"]
        if type(enclosure["digits"]) is not int or enclosure["digits"] != 80:
            raise ValueError("raw enclosure digits mismatch")
        if raw["status"] != "resolved" or sidecar["status"] != "resolved":
            raise ValueError("unresolved B1 record")
        if sidecar["key"] != raw["key"]:
            raise ValueError("sidecar/raw key mismatch")
        if sidecar["classification"] not in {"rigorous_bound", "exact_zero"}:
            raise ValueError("unknown uncertainty classification")
        for flag in (
            "B1_nine_source_allocation_sum_equals_u_up",
            "exact_endpoint_and_no_narrowing",
            "preproduction_oracle_fraction_pass",
            "zero_or_nonzero_semantics",
        ):
            if sidecar[flag] is not True:
                raise ValueError(f"required reference review flag is false: {flag}")
        if not isinstance(sidecar["units"], str) or not sidecar["units"]:
            raise ValueError("reference units are missing")
        certificate = sidecar["uncertainty_certificate"]
        if not isinstance(certificate, str) or not certificate.strip():
            raise ValueError("uncertainty certificate is missing")

        central = parse_exact_fraction(enclosure["central"])
        lo = parse_exact_fraction(enclosure["lo_exact"])
        hi = parse_exact_fraction(enclosure["hi_exact"])
        raw_u_up = parse_exact_fraction(enclosure["u_up"])
        sidecar_u_up = parse_exact_fraction(sidecar["u_up"])
        if raw_u_up != sidecar_u_up:
            raise ValueError("raw and sidecar u_up values do not match exactly")
        u_up = raw_u_up
        source_scale = parse_exact_fraction(sidecar["source_scale"])
        budget_lower_display = parse_exact_fraction(sidecar["B_ref_lower"])
        matrix_upper_display = parse_exact_fraction(sidecar["uncertainty_matrix_upper"])
        oracle_ratio_display = parse_exact_fraction(
            sidecar["conservative_10_U_matrix_over_B_ref_lower"]
        )
        if lo > hi:
            raise ValueError("invalid enclosure endpoint ordering")
        if min(
            u_up,
            source_scale,
            budget_lower_display,
            matrix_upper_display,
            oracle_ratio_display,
        ) < 0:
            raise ValueError("negative U, source scale, or budget display")

        midpoint = (lo + hi) / 2
        radius = (hi - lo) / 2
        center_rounding = abs(central - midpoint)
        needed = max(abs(central - lo), abs(hi - central))
        upward_slack = u_up - needed
        if upward_slack < 0 or needed != radius + center_rounding:
            raise ValueError("u_up does not enclose exact endpoints")
        if not (
            abs(central - lo) <= u_up
            and abs(hi - central) <= u_up
            and central - u_up <= lo
            and hi <= central + u_up
            and central - u_up <= source_scale <= central + u_up
        ):
            raise ValueError("enclosure does not contain source scale and endpoints")
        if radius + center_rounding + upward_slack != u_up:
            raise ValueError("nine-source enclosure terms do not sum exactly to u_up")

        is_zero_key = key_id in zero_set
        if is_zero_key:
            if sidecar["classification"] != "exact_zero":
                raise ValueError("exact-zero key has nonzero uncertainty classification")
            if any(
                value != 0
                for value in (
                    central,
                    lo,
                    hi,
                    u_up,
                    source_scale,
                    budget_lower_display,
                    matrix_upper_display,
                    oracle_ratio_display,
                )
            ):
                raise ValueError("exact-zero record contains a nonzero value")
        else:
            if sidecar["classification"] != "rigorous_bound":
                raise ValueError("nonzero key has wrong uncertainty classification")
            if min(central, lo, hi, source_scale, u_up) <= 0:
                raise ValueError("nonzero record has zero or negative scale/enclosure/U")

        u_matrix_exact = 3 * u_up
        kernel_uncertainty, u_matrix_kernel = _kernel_uncertainty(
            u_matrix_exact,
            sidecar["classification"],
            certificate,
        )
        if u_matrix_kernel != u_matrix_exact:
            raise ValueError("kernel uncertainty differs from exact full-matrix bound")

        central_text = fraction_to_exact_text(central)
        if "/" in central_text:
            raise ValueError("central reference value is not a terminating decimal")
        zero_text = "0e+0"
        quantity_data = [
            {
                "real": central_text if index in {0, 4, 8} else zero_text,
                "imag": zero_text,
            }
            for index in range(9)
        ]
        adapted[key_id] = {
            "key": raw["key"],
            "central": central,
            "lo": lo,
            "hi": hi,
            "source_scale": source_scale,
            "u_up_exact": u_up,
            "budget_lower_display": budget_lower_display,
            "uncertainty_matrix_display": matrix_upper_display,
            "u_matrix_exact": u_matrix_exact,
            "u_matrix_kernel": u_matrix_kernel,
            "kernel_quantity": {"shape": [3, 3], "data": quantity_data},
            "kernel_uncertainty": kernel_uncertainty,
            "units": sidecar["units"],
            "classification": sidecar["classification"],
            "is_exact_zero": is_zero_key,
        }
    return adapted
