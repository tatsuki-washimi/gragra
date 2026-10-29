"""Replay pinned Gen1 references and record their missing refinement traces.

This additive observer imports only the independent oracle modules beside this
file.  It calls their original public reference entrypoints, temporarily wraps
the exact numerical callbacks those entrypoints use, and never changes a
frozen reference or its uncertainty estimate.

The attempt journal accounts for execution and explicit retry history. The
canonical V4 runner remains a fail-closed scaffold; no trust anchor is provided.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import copy
import dataclasses
import fcntl
import hashlib
import importlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
import traceback
import types
import uuid
from pathlib import Path

import mpmath as mp
import numpy as np

ORACLE_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY = Path(__file__).resolve().parents[3]
ORACLE_PACKAGE = "_gen1_frozen_oracle_modules_v4"
SCHEMA = "gragra.gen1.refinement-trace.v4"

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

ZERO_CERTIFICATES_FILENAME = "frozen-zero-estimate-certificates-integrated.json"
EXPECTED_ZERO_CERTIFICATES_SHA256 = (
    "6c668d2a7c7c3a80ea7b5770cb795dc4a34c719cb68b33123c843badd02429c3"
)
POLYNOMIAL_CERTIFICATES_FILENAME = "polynomial-quadrature-certificates-integrated.json"
EXPECTED_POLYNOMIAL_CERTIFICATES_SHA256 = (
    "6d39a6e00a022c1ef06e86b67f7b7a5542b140bfe62dafae5df6cf9820914152"
)
POLYNOMIAL_CERTIFICATE_SOURCE_JSON_SHA256 = (
    "0a353d9e357770afc182ca73c47d93740df9ddd7b093a19ddd6cc08f19fbaf16"
)
POLYNOMIAL_CERTIFICATE_REPORT_SHA256 = (
    "613aca3a7d9acb16f487bfea39fec9f6eb3b24947579873283486e6afaa75ca7"
)
EXPECTED_POLYNOMIAL_VERIFIER_SCRIPT_SHA256 = (
    "e51402fd4779aadc86fdf048592a9f71bedf0adaac69392ea29de1b84cb32a31"
)
POLYNOMIAL_CERTIFICATE_CERTIFIED_ERROR = (
    "zero quadrature truncation in ideal real arithmetic only"
)
POLYNOMIAL_CERTIFICATE_COVERED_STENCILS = (
    "all original loop steps and axis/diagonal seven-point evaluations"
)

# These are the untouched independent modules copied from the integration
# worktree.  Each run records and verifies these hashes before importing them.
EXPECTED_ORACLE_SHA256 = {
    "acceptance.py": (
        "0ac9ac1f01578696fb6ef1a6c1aad887b976aab1b7b873fbc78df7d388747e94"
    ),
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

OBSERVABLE_SLICES = {
    "potential": slice(0, 1),
    "acceleration": slice(1, 4),
    "gradient": slice(4, 13),
}
TREND_CLASSIFICATIONS = {
    "decreasing",
    "nondecreasing_explained_by_independent_floor",
    "not_applicable",
    "unresolved_nondecreasing",
    "unresolved_zero_change",
}


def oracle_module(name: str):
    """Load an oracle sibling under a namespace isolated from gragra imports."""
    if ORACLE_PACKAGE not in sys.modules:
        package = types.ModuleType(ORACLE_PACKAGE)
        package.__path__ = [str(ORACLE_DIRECTORY)]
        package.__package__ = ORACLE_PACKAGE
        sys.modules[ORACLE_PACKAGE] = package
    return importlib.import_module(f"{ORACLE_PACKAGE}.{name}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _strict_json_loads(value, *, description):
    try:
        return json.loads(value, object_pairs_hook=_unique_json_object)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"invalid {description}: {error}") from error


def _strict_json_file(path, *, description=None):
    path = Path(path)
    return _strict_json_loads(
        path.read_text(encoding="utf-8"),
        description=description or str(path),
    )


def _validated_selected_indices(indices, available_indices):
    requested = list(indices)
    if any(type(index) is not int for index in requested):
        raise ValueError("selected source row indices must be integers")
    if len(requested) != len(set(requested)):
        raise ValueError("selected source row indices contain duplicates")
    missing = set(requested) - set(available_indices)
    if missing:
        raise ValueError(
            "selected source row indices outside the frozen reference: "
            f"{sorted(missing)}"
        )
    return sorted(requested)


def _decode_trace_bytes(data: bytes, *, description: str):
    if not data:
        return []
    if not data.endswith(b"\n"):
        raise RuntimeError(f"{description} ends with an incomplete JSONL record")
    records = []
    for line_number, line in enumerate(data.splitlines(), 1):
        try:
            record = _strict_json_loads(
                line.decode("utf-8"),
                description=f"JSONL record {line_number} in {description}",
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise RuntimeError(
                f"invalid JSONL record {line_number} in {description}: {error}"
            ) from error
        if not isinstance(record, dict):
            raise RuntimeError(
                f"non-object JSONL record {line_number} in {description}"
            )
        records.append(record)
    return records


def _validate_polynomial_adjudication(
    adjudication,
    dataset,
    index,
    row,
    observable,
    trace,
    certificates,
    certificate_sha256,
):
    certificate = _matching_polynomial_certificate(
        dataset, index, row["fixture"], observable, certificates
    )
    if (
        certificate is None
        or trace.get("quadrature_trend", {}).get("classification")
        != "unresolved_zero_change"
    ):
        raise ValueError(
            f"trace source row {index} has an out-of-scope polynomial certificate"
        )
    expected = {
        "dataset": dataset,
        "source_row_index": index,
        "fixture": row["fixture"],
        "observable": observable,
        "certificate_kind": certificate["certificate_kind"],
        "certified_error": certificate["certified_error"],
        "scope": "ideal polynomial quadrature truncation only",
        "temporal_quadrature_remains_applicable": True,
        "total_numerical_error_certified": False,
        "raw_quadrature_trend_classification": "unresolved_zero_change",
        "supplement_path": adjudication.get("supplement_path"),
        "supplement_sha256": certificate_sha256,
        "verifier_script_sha256": EXPECTED_POLYNOMIAL_VERIFIER_SCRIPT_SHA256,
        "frozen_values_uncertainty_labels_unchanged": True,
    }
    if _canonical_json(adjudication) != _canonical_json(expected):
        raise ValueError(
            f"trace source row {index} has an invalid polynomial certificate"
        )


def _validate_process_independence(record, index, *, required):
    evidence = record.get("process_independence")
    if evidence is None:
        if required:
            raise ValueError(
                f"trace source row {index} is missing worker process evidence"
            )
        return
    if not isinstance(evidence, dict):
        raise ValueError(f"trace source row {index} has malformed worker evidence")
    pid = evidence.get("pid")
    if type(pid) is not int or pid <= 0:
        raise ValueError(f"trace source row {index} has invalid worker pid evidence")
    modules_before = evidence.get("modules_before")
    modules_after = evidence.get("modules_after")
    if any(
        not isinstance(modules, list)
        or any(not isinstance(name, str) for name in modules)
        or modules != sorted(set(modules))
        for modules in (modules_before, modules_after)
    ):
        raise ValueError(
            f"trace source row {index} has malformed worker module observations"
        )
    expected_names = sorted(set(modules_before) | set(modules_after))
    if evidence.get("production_module_names") != expected_names:
        raise ValueError(
            f"trace source row {index} has inconsistent worker module summary"
        )
    expected_imported = bool(expected_names)
    if evidence.get("production_imported") is not expected_imported:
        raise ValueError(
            f"trace source row {index} has inconsistent worker import flag"
        )
    if evidence.get("independence_check_passed") is not (not expected_names):
        raise ValueError(
            f"trace source row {index} has inconsistent worker independence flag"
        )


def _validate_captured_calls(calls, index, *, allow_empty, allow_errors=False):
    if not isinstance(calls, list):
        raise ValueError(f"trace source row {index} has malformed captured calls")
    if not calls and not allow_empty:
        raise ValueError(f"trace source row {index} has no captured evaluations")
    for call in calls:
        if (
            not isinstance(call, dict)
            or not isinstance(call.get("kernel"), str)
            or not call["kernel"]
            or not isinstance(call.get("inputs"), dict)
            or (
                "output" not in call
                and not (
                    allow_errors
                    and isinstance(call.get("_error"), str)
                    and call["_error"]
                )
            )
        ):
            raise ValueError(f"trace source row {index} has a malformed captured call")


def _validate_trend(trace, key, index, observable):
    trend = trace.get(key)
    if (
        not isinstance(trend, dict)
        or not isinstance(trend.get("classification"), str)
        or trend["classification"] not in TREND_CLASSIFICATIONS
    ):
        raise ValueError(
            f"trace source row {index} has malformed {observable}/{key} evidence"
        )
    classification = trend["classification"]
    if classification == "not_applicable":
        if not isinstance(trend.get("reason"), str) or not trend["reason"]:
            raise ValueError(
                f"trace source row {index} has no {observable}/{key} reason"
            )
        return
    coarse_text = trend.get("coarse_to_middle_si")
    middle_text = trend.get("middle_to_fine_si")
    floor_text = trend.get("independent_floor_si")
    if (
        not isinstance(coarse_text, str)
        or not isinstance(middle_text, str)
        or (floor_text is not None and not isinstance(floor_text, str))
        or type(trend.get("nondecreasing")) is not bool
    ):
        raise ValueError(
            f"trace source row {index} has incomplete {observable}/{key} differences"
        )
    try:
        if trace.get("route") == "numpy.longdouble":
            coarse = np.longdouble(coarse_text)
            middle = np.longdouble(middle_text)
            floor = None if floor_text is None else np.longdouble(floor_text)
            finite = (
                np.isfinite(coarse)
                and np.isfinite(middle)
                and (floor is None or np.isfinite(floor))
            )
        else:
            precision = max(
                100,
                len(coarse_text.replace(".", "")) + 16,
                len(middle_text.replace(".", "")) + 16,
                0 if floor_text is None else len(floor_text.replace(".", "")) + 16,
            )
            with mp.workdps(precision):
                coarse = mp.mpf(coarse_text)
                middle = mp.mpf(middle_text)
                floor = None if floor_text is None else mp.mpf(floor_text)
                finite = (
                    mp.isfinite(coarse)
                    and mp.isfinite(middle)
                    and (floor is None or mp.isfinite(floor))
                )
        if not finite:
            raise ValueError("nonfinite difference")
        if coarse < 0 or middle < 0 or (floor is not None and floor < 0):
            raise ValueError("negative norm or floor")
        if coarse < 0 or middle < 0 or (floor is not None and floor < 0):
            raise ValueError("negative norm or floor")
        expected_nondecreasing = bool(middle >= coarse)
        if middle < coarse:
            expected_classification = "decreasing"
        elif floor is not None and floor > 0 and floor >= middle:
            expected_classification = "nondecreasing_explained_by_independent_floor"
        elif middle == 0 and coarse == 0:
            expected_classification = "unresolved_zero_change"
        else:
            expected_classification = "unresolved_nondecreasing"
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"trace source row {index} has invalid {observable}/{key} differences"
        ) from error
    if (
        classification != expected_classification
        or trend["nondecreasing"] is not expected_nondecreasing
    ):
        raise ValueError(
            f"trace row {index}: inconsistent {observable}/{key} classification"
        )


def _trend_applicability(dataset, fixture, observable, key, trace):
    status = trace.get("status")
    if status in {"certified_override", "exact_whole_case_certificate"}:
        return False, None
    if status != "numerical":
        raise ValueError(f"unknown {observable} trace status for trend validation")
    route = trace.get("route")
    if route == "mpmath":
        if key == "derivative_trend":
            return observable != "potential", (
                "potential is evaluated directly and does not depend on "
                "target-difference step"
            )
        family = fixture.get("family")
        applicable = (
            dataset in {"sphere", "high_sphere"}
            or dataset == "source"
            and family != "temperature_grid"
            or dataset == "noncavity"
            and family in {"gaussian", "halfspace", "point_force"}
        )
        if applicable:
            return True, None
        if dataset == "source" and family == "temperature_grid":
            return False, (
                "the original route is an explicit finite mass sum with no "
                "spatial quadrature"
            )
        return False, "entrypoint uses no spatial quadrature for this family"
    if route in {"numpy.longdouble", "mpmath_fallback"}:
        if key == "derivative_trend":
            if dataset == "source":
                return False, (
                    "finite-source surface references have no target "
                    "finite-difference refinement"
                )
            return (
                False,
                "surface values use spatial quadrature, not target finite differences",
            )
        return True, None
    raise ValueError(f"unknown numerical route {route!r} for trend validation")


def _certified_override_na_reason(dataset, fixture):
    if dataset in {"cuboid", "cylinder"}:
        return "the frozen output is an exact symmetry/parity certificate"
    if (
        dataset == "source"
        and fixture.get("shape") != "sphere"
        and fixture.get("family") != "temperature_grid"
    ):
        return "the frozen output is an exact inversion-parity certificate"
    return "the frozen observable is an exact symmetry/parity certificate"


def _validate_trend_applicability(trace, dataset, fixture, index, observable):
    for key in ("derivative_trend", "quadrature_trend"):
        trend = trace[key]
        applicable, expected_reason = _trend_applicability(
            dataset, fixture, observable, key, trace
        )
        classification = trend["classification"]
        if applicable and classification == "not_applicable":
            raise ValueError(
                f"trace source row {index} marks applicable {observable}/{key} "
                "as not_applicable"
            )
        if not applicable:
            if classification != "not_applicable":
                raise ValueError(
                    f"trace source row {index} omits structural not_applicable "
                    f"state for {observable}/{key}"
                )
            reason = trend.get("reason")
            if trace["status"] == "certified_override":
                allowed_reasons = {_certified_override_na_reason(dataset, fixture)}
            elif trace["status"] == "exact_whole_case_certificate":
                allowed_reasons = {
                    "no observable evaluation occurs for this certified row"
                }
            else:
                allowed_reasons = {expected_reason}
            if reason not in allowed_reasons:
                raise ValueError(
                    f"trace source row {index} has an unsupported {observable}/{key} "
                    "not_applicable reason"
                )


def _captured_vector_length(value):
    if isinstance(value, dict):
        if "shape" in value:
            shape = value["shape"]
            if not isinstance(shape, list) or any(
                type(dimension) is not int or dimension < 0 for dimension in shape
            ):
                return None
            size = 1
            for dimension in shape:
                size *= dimension
            return size
        if "real" in value and "imag" in value:
            return 1
        if "decimal" in value or "mpf" in value:
            return 1
        return None
    if isinstance(value, list):
        return len(value)
    return 1 if value is not None else None


def _captured_real(value, index, observable):
    captured_dtype = None
    if isinstance(value, dict) and "decimal" in value:
        captured_dtype = value.get("dtype")
        value = value["decimal"]
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(
            f"trace source row {index} has malformed {observable} captured number"
        )
    try:
        if captured_dtype is None:
            result = float(value)
        else:
            dtype = np.dtype(captured_dtype)
            if dtype.kind != "f":
                raise ValueError("captured dtype is not a real floating type")
            result = float(dtype.type(value))
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(
            f"trace source row {index} has malformed {observable} captured number"
        ) from error
    if not math.isfinite(result):
        raise ValueError(
            f"trace source row {index} has nonfinite {observable} captured number"
        )
    return result


def _captured_complex_vector(value, index, observable):
    if isinstance(value, dict) and "shape" in value:
        value = value.get("values")
    if isinstance(value, list):
        return [
            component
            for item in value
            for component in _captured_complex_vector(item, index, observable)
        ]
    if isinstance(value, dict) and "real" in value and "imag" in value:
        return [
            [
                _captured_real(value["real"], index, observable),
                _captured_real(value["imag"], index, observable),
            ]
        ]
    if isinstance(value, dict) and ("decimal" in value or "mpf" in value):
        return [[_captured_real(value, index, observable), 0.0]]
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return [[_captured_real(value, index, observable), 0.0]]
    raise ValueError(
        f"trace source row {index} has malformed {observable} captured vector"
    )


def _callback_value_precision(value):
    """Return sufficient dps to reconstruct every serialized MP value exactly."""
    precision = 0
    if isinstance(value, list):
        return max((_callback_value_precision(item) for item in value), default=0)
    if not isinstance(value, dict):
        return precision
    for key in ("digits", "mp_dps_at_call"):
        item = value.get(key)
        if type(item) is int and item > 0:
            precision = max(precision, item)
    if "mpf" in value:
        parts = value["mpf"]
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
        precision = max(precision, _callback_value_precision(item))
    return precision


def _declared_callback_digits(value):
    """Read declared callback calculation precision, excluding MPF storage bits."""
    if isinstance(value, list):
        return max((_declared_callback_digits(item) for item in value), default=0)
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
        max((_declared_callback_digits(item) for item in value.values()), default=0),
    )


def _decode_callback_value(value, *, precision=None):
    """Restore callbacks under enough MP precision for their serialized bits."""
    dps = max(50, precision or 0, _callback_value_precision(value))
    with mp.workdps(dps):
        return _decode_callback_value_inner(value)


def _round_integer_ties_to_even(value, right_shift):
    """Round a nonnegative integer divided by a power of two, exactly."""
    if right_shift <= 0:
        return value << -right_shift
    if right_shift > value.bit_length():
        return 0
    quotient, remainder = divmod(value, 1 << right_shift)
    halfway = 1 << (right_shift - 1)
    if remainder > halfway or (remainder == halfway and quotient & 1):
        quotient += 1
    return quotient


def _integer_to_decimal(value):
    """Convert an arbitrarily large nonnegative integer without global limits."""
    if value == 0:
        return "0"
    chunks = []
    while value:
        value, chunk = divmod(value, 1_000_000_000)
        chunks.append(chunk)
    return str(chunks[-1]) + "".join(f"{chunk:09d}" for chunk in reversed(chunks[:-1]))


def _dyadic_decimal(sign, mantissa, exponent):
    """Format an exact signed ``mantissa * 2**exponent`` as decimal text."""
    if mantissa == 0:
        return "-0" if sign else "0"
    prefix = "-" if sign else ""
    if exponent >= 0:
        return prefix + _integer_to_decimal(mantissa << exponent)
    decimal_exponent = -exponent
    digits = _integer_to_decimal(mantissa * (5**decimal_exponent))
    scientific_exponent = len(digits) - 1 - decimal_exponent
    significand = digits[0]
    if len(digits) > 1:
        significand += "." + digits[1:]
    return f"{prefix}{significand}e{scientific_exponent}"


def _mpf_to_longdouble(value, *, precision=None):
    """Round an MPF's exact binary tuple to the host binary longdouble.

    ``precision`` remains in the signature for compatibility. The exact tuple,
    not the ambient mpmath context or requested display precision, determines
    the rounded result.
    """
    del precision
    if not mp.isfinite(value):
        raise ValueError("captured callback difference is nonfinite")
    sign, mantissa, exponent, _bitcount = value._mpf_
    if mantissa == 0:
        return np.longdouble("-0" if sign else "0")

    finfo = np.finfo(np.longdouble)
    precision_bits = int(finfo.nmant) + 1
    minimum_normal_exponent = int(finfo.minexp)
    maximum_exponent = int(finfo.maxexp)
    subnormal_exponent = minimum_normal_exponent - precision_bits + 1
    leading_exponent = exponent + mantissa.bit_length() - 1

    if leading_exponent >= maximum_exponent:
        raise ValueError("captured callback difference is nonfinite")

    if leading_exponent >= minimum_normal_exponent:
        rounded_exponent = leading_exponent - precision_bits + 1
        rounded_mantissa = _round_integer_ties_to_even(
            mantissa, rounded_exponent - exponent
        )
        if rounded_mantissa.bit_length() > precision_bits:
            rounded_mantissa >>= 1
            rounded_exponent += 1
        if rounded_exponent + rounded_mantissa.bit_length() - 1 >= maximum_exponent:
            raise ValueError("captured callback difference is nonfinite")
    else:
        rounded_exponent = subnormal_exponent
        rounded_mantissa = _round_integer_ties_to_even(
            mantissa, rounded_exponent - exponent
        )
        if rounded_mantissa == 0:
            return np.longdouble("-0" if sign else "0")
        if rounded_mantissa > (1 << (precision_bits - 1)):
            raise ValueError("captured callback difference is nonfinite")

    decimal = _dyadic_decimal(sign, rounded_mantissa, rounded_exponent)
    converted = np.longdouble(decimal)
    if not np.isfinite(converted):
        raise ValueError("captured callback difference is nonfinite")
    return converted


def _decode_mpf(value):
    """Check the canonical binary tuple and its deterministic decimal display.

    display_digits describes presentation precision, not callback arithmetic.
    Exact tuple preservation permits inputs retained from a higher-dps loop.
    The display is compared with nstr at the declared presentation precision;
    decimal parsing or binary64 conversion is never the source of truth.
    """
    parts = value.get("mpf")
    if (
        not isinstance(parts, (list, tuple))
        or len(parts) != 4
        or any(type(part) is not int for part in parts)
        or parts[0] not in (0, 1)
        or parts[1] < 0
        or parts[3] < 0
        or (parts[1] == 0 and tuple(parts) != (0, 0, 0, 0))
        or (parts[1] != 0 and (parts[1] % 2 != 1 or parts[1].bit_length() != parts[3]))
    ):
        raise ValueError("invalid canonical MPF tuple")
    display_digits = value.get("display_digits")
    if type(display_digits) is not int or display_digits <= 0:
        raise ValueError("MPF display precision must be a positive integer")
    with mp.workprec(max(53, parts[3] + 8)):
        number = mp.mpf(tuple(parts))
        if not mp.isfinite(number):
            raise ValueError("captured callback MP value is nonfinite")
        expected_display = mp.nstr(number, n=display_digits)
    if type(value.get("decimal")) is not str or value["decimal"] != expected_display:
        raise ValueError("MPF decimal display does not match tuple")
    return number


def _validate_mpf_displays(value):
    """Validate every MPF payload, including otherwise-unused callback fields."""
    if isinstance(value, dict):
        if "mpf" in value:
            _decode_mpf(value)
        for item in value.values():
            _validate_mpf_displays(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_mpf_displays(item)


def _decode_callback_value_inner(value):
    """Decode recursively inside a context chosen by ``_decode_callback_value``."""
    if isinstance(value, list):
        return [_decode_callback_value_inner(item) for item in value]
    if not isinstance(value, dict):
        return value
    if all(key in value for key in ("shape", "dtype", "values")):
        array = np.asarray(
            _decode_callback_value_inner(value["values"]), dtype=value["dtype"]
        )
        if list(array.shape) != value["shape"] or not np.all(np.isfinite(array)):
            raise ValueError("captured callback array shape or values are invalid")
        return array
    if "mpf" in value:
        return _decode_mpf(value)
    if "real" in value and "imag" in value:
        real = _decode_callback_value_inner(value["real"])
        imag = _decode_callback_value_inner(value["imag"])
        if "dtype" in value:
            dtype = np.dtype(value["dtype"])
            result = dtype.type(real) + dtype.type(1j) * dtype.type(imag)
            if not np.isfinite(result):
                raise ValueError("captured callback complex value is nonfinite")
            return result
        result = mp.mpc(real, imag)
        if not mp.isfinite(result):
            raise ValueError("captured callback complex value is nonfinite")
        return result
    if "dtype" in value and "decimal" in value:
        result = np.dtype(value["dtype"]).type(value["decimal"])
        if not np.isfinite(result):
            raise ValueError("captured callback floating value is nonfinite")
        return result
    return {key: _decode_callback_value_inner(item) for key, item in value.items()}


def _callback_number(value, *, longdouble, precision=None):
    if isinstance(value, dict):
        value = _decode_callback_value(value)
    if longdouble:
        number = (
            _mpf_to_longdouble(value, precision=precision)
            if isinstance(value, mp.mpf)
            else np.longdouble(value)
        )
        if not np.isfinite(number):
            raise ValueError("captured callback difference is nonfinite")
        return number
    with mp.workdps(precision or mp.mp.dps):
        number = value if isinstance(value, mp.mpf) else mp.mpf(str(value))
        if not mp.isfinite(number):
            raise ValueError("captured callback difference is nonfinite")
        return number


def _callback_difference(left, right, observable, *, longdouble):
    if longdouble:
        surface = oracle_module("surface_reference")
        return surface.norm_ld(np.asarray(left) - np.asarray(right))
    sphere = oracle_module("sphere_reference")
    return sphere.differences({observable: left}, {observable: right})[observable]


def _callback_pair(vectors, observable, *, longdouble):
    if len(vectors) != 3:
        raise ValueError("three captured refinement vectors are required")
    return (
        _callback_difference(vectors[1], vectors[0], observable, longdouble=longdouble),
        _callback_difference(vectors[2], vectors[1], observable, longdouble=longdouble),
    )


def _require_number_equal(actual, expected, *, longdouble, message, precision=None):
    try:
        parsed = _callback_number(actual, longdouble=longdouble, precision=precision)
        comparison_detail = ""
        if longdouble:
            if isinstance(expected, mp.mpf):
                expected_value = _mpf_to_longdouble(expected, precision=precision)
            else:
                expected_value = np.longdouble(expected)
            equal = parsed == expected_value
        elif isinstance(expected, float):
            # Frozen components originate as floats; this is an exact round-trip
            # comparison in the source value's native representation.
            equal = float(parsed) == expected
        else:
            with mp.workdps(precision or mp.mp.dps):
                expected_number = (
                    expected if isinstance(expected, mp.mpf) else mp.mpf(str(expected))
                )
                equal = parsed == expected_number
                comparison_detail = (
                    f"; parsed={mp.nstr(parsed, precision or 50)}, "
                    f"expected={mp.nstr(expected_number, precision or 50)}"
                )
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(message) from error
    if not equal:
        raise ValueError(
            f"{message}: saved={actual!r}, expected={expected!r}{comparison_detail}"
        )


def _require_saved_pair(
    trace, field, pair, labels, *, longdouble, index, observable, precision=None
):
    actual = trace.get(field)
    if pair is None:
        if actual is not None:
            raise ValueError(
                f"trace source row {index} has an inapplicable {observable}/{field}"
            )
        return
    if not isinstance(actual, dict) or set(actual) != {*labels, "ten_max"}:
        raise ValueError(
            f"trace source row {index} has an incomplete {observable}/{field}"
        )
    if longdouble:
        scaled_max = np.longdouble(10) * max(pair)
    else:
        with mp.workdps(precision or mp.mp.dps):
            scaled_max = 10 * max(pair)
    expected_values = (*pair, scaled_max)
    for label, expected in zip((*labels, "ten_max"), expected_values, strict=True):
        _require_number_equal(
            actual[label],
            expected,
            longdouble=longdouble,
            precision=precision,
            message=(
                f"trace source row {index} {observable}/{field} differs from "
                "captured callbacks"
            ),
        )


def _require_trend_pair(
    trend, pair, *, longdouble, index, observable, key, precision=None
):
    if not isinstance(trend, dict):
        raise ValueError(f"trace source row {index} is missing {observable}/{key}")
    for field, expected in zip(
        ("coarse_to_middle_si", "middle_to_fine_si"), pair, strict=True
    ):
        _require_number_equal(
            trend.get(field),
            expected,
            longdouble=longdouble,
            precision=precision,
            message=(
                f"trace source row {index} {observable}/{key} differs from "
                "captured callbacks"
            ),
        )


def _check_component_formula(
    trace, kind, pair, component_name, *, longdouble, index, precision=None
):
    check = trace["frozen_component_checks"][kind]
    if component_name is None:
        if check is not None:
            raise ValueError(f"trace source row {index} has an unexpected {kind} check")
        return
    if not isinstance(check, dict) or pair is None:
        raise ValueError(f"trace source row {index} has no applicable {kind} check")
    if longdouble:
        computed = np.longdouble(10) * max(pair)
    else:
        with mp.workdps(precision or mp.mp.dps):
            computed = 10 * max(pair)
    expected_float = float(computed)
    if check.get("formula_float_si") != expected_float:
        raise ValueError(
            f"trace source row {index} {kind} component formula differs from callbacks"
        )
    formula = check.get("formula_value_si")
    if "formula_value_si" in check:
        if formula is None:
            raise ValueError(
                f"trace source row {index} {kind} component formula is missing"
            )
        _require_number_equal(
            formula,
            computed,
            longdouble=longdouble,
            precision=precision,
            message=(
                f"trace source row {index} {kind} component formula "
                "differs from callbacks"
            ),
        )


def _mp_callback_pairs(calls, dataset, fixture, observable):
    high_digits = 130 if dataset == "high_sphere" else 80
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
    events = [
        {"inputs": _decode_callback_value(call["inputs"]), "output": call["output"]}
        for call in calls
        if call["kernel"] == kernel
    ]

    def output(event):
        return _decode_callback_value(event["output"])

    def digits(event):
        return event["inputs"].get("digits") or event["inputs"].get("mp_dps_at_call")

    def order(event):
        return event["inputs"].get("quadrature_order", event["inputs"].get("order"))

    with mp.workdps(high_digits):
        final = [
            event
            for event in events
            if digits(event) == high_digits
            and (orders is None or order(event) == orders[-1])
        ]
        if len(final) < 3:
            raise ValueError("missing captured derivative callback triplet")
        final = final[-3:]
        steps = [
            _callback_number(event["inputs"]["step_m"], longdouble=False)
            for event in final
        ]
        h = steps[-1]
        if steps != [4 * h, 2 * h, h]:
            raise ValueError("captured derivative callback steps are inconsistent")
        for event in final:
            point = _decode_callback_value(event["inputs"]["point_m"])
            if list(point) != [mp.mpf(value) for value in fixture["target_m"]]:
                raise ValueError("captured derivative callback target is inconsistent")
        derivative_vectors = [output(event)[observable] for event in final]
        derivative_pair = _callback_pair(
            derivative_vectors, observable, longdouble=False
        )
        quadrature_pair = None
        if orders is not None:
            by_order = {
                order(event): event
                for event in events
                if digits(event) == high_digits
                and order(event) in orders
                and _callback_number(event["inputs"]["step_m"], longdouble=False) == h
            }
            if set(by_order) != set(orders):
                raise ValueError("missing captured fixed-step quadrature triplet")
            quadrature_pair = _callback_pair(
                [output(by_order[value])[observable] for value in orders],
                observable,
                longdouble=False,
            )
    family = fixture.get("family")
    derivative_applicable = observable != "potential"
    quadrature_applicable = (
        dataset in {"sphere", "high_sphere"}
        or dataset == "source"
        and family != "temperature_grid"
        or dataset == "noncavity"
        and family in {"gaussian", "halfspace", "point_force"}
    )
    return (
        derivative_pair,
        quadrature_pair,
        derivative_applicable,
        quadrature_applicable,
    )


def _surface_callback_pair(calls, dataset, fixture, metadata, trace, observable):
    section = OBSERVABLE_SLICES[observable]
    if trace.get("route") == "mpmath_fallback":
        events = [
            call
            for call in calls
            if call["kernel"] == "surface_high_precision_evaluate"
        ]
        if not events:
            raise ValueError("missing captured high-precision surface triplet")
        decode_precision = max(
            80,
            max((_callback_value_precision(event) for event in events), default=0),
        )
        calculation_precision = max(
            80,
            max((_declared_callback_digits(event) for event in events), default=0),
        )
        with mp.workdps(decode_precision):
            decoded = [
                {
                    "inputs": _decode_callback_value(
                        call["inputs"], precision=decode_precision
                    ),
                    "output": _decode_callback_value(
                        call["output"], precision=decode_precision
                    ),
                }
                for call in events
            ]
            lookup = {
                (event["inputs"]["digits"], event["inputs"]["order"]): event
                for event in decoded
            }
            try:
                vectors = [lookup[(50, order)]["output"][0] for order in (16, 24, 32)]
            except KeyError as error:
                raise ValueError(
                    "missing captured high-precision surface triplet"
                ) from error
        with mp.workdps(calculation_precision):
            pair = _callback_pair(
                [vector[section] for vector in vectors], observable, longdouble=False
            )
        return pair, False

    source = dataset == "source"
    kernel = "source_surface_values" if source else "surface_newton_evaluate"
    events = [call for call in calls if call["kernel"] == kernel]
    decoded = [
        {
            "inputs": _decode_callback_value(call["inputs"]),
            "output": _decode_callback_value(call["output"]),
        }
        for call in events
    ]
    orders = [16, 24, 32] if source else list(metadata["orders"])
    by_order = {event["inputs"]["order"]: event for event in decoded}
    if len(orders) != 3 or any(order not in by_order for order in orders):
        raise ValueError("missing captured surface quadrature triplet")
    vectors = []
    for order in orders:
        event = by_order[order]
        if source:
            payload = event["output"]["values_and_absolute"]
        else:
            target = event["inputs"].get("target_m")
            if list(target) != [np.longdouble(value) for value in fixture["target_m"]]:
                raise ValueError("captured surface target is inconsistent")
            payload = event["output"]
        vectors.append(payload[0][section])
    return _callback_pair(vectors, observable, longdouble=True), True


def _validate_callback_trend(trace, reference, calls, dataset, row, index, observable):
    if trace["status"] != "numerical":
        return
    route = trace.get("route")
    precision = (
        max(130 if dataset == "high_sphere" else 80, _declared_callback_digits(calls))
        if route in {"mpmath", "mpmath_fallback"}
        else None
    )
    if route == "mpmath":
        (
            derivative_pair,
            quadrature_pair,
            derivative_applicable,
            quadrature_applicable,
        ) = _mp_callback_pairs(calls, dataset, row["fixture"], observable)
        _require_saved_pair(
            trace,
            "derivative_differences_si",
            derivative_pair if derivative_applicable else None,
            ("4h_to_2h", "2h_to_h"),
            longdouble=False,
            precision=precision,
            index=index,
            observable=observable,
        )
        _require_saved_pair(
            trace,
            "quadrature_differences_si",
            quadrature_pair if quadrature_applicable else None,
            ("coarse_to_middle", "middle_to_fine"),
            longdouble=False,
            precision=precision,
            index=index,
            observable=observable,
        )
        if derivative_applicable:
            _require_trend_pair(
                trace["derivative_trend"],
                derivative_pair,
                longdouble=False,
                precision=precision,
                index=index,
                observable=observable,
                key="derivative_trend",
            )
        if quadrature_applicable:
            _require_trend_pair(
                trace["quadrature_trend"],
                quadrature_pair,
                longdouble=False,
                precision=precision,
                index=index,
                observable=observable,
                key="quadrature_trend",
            )
        elif quadrature_pair is not None:
            _require_saved_pair(
                trace,
                "observed_inapplicable_quadrature_component_si",
                quadrature_pair,
                ("coarse_to_middle", "middle_to_fine"),
                longdouble=False,
                precision=precision,
                index=index,
                observable=observable,
            )
        derivative_component = (
            "real_field_derivative_refinement"
            if dataset == "plane"
            else "derivative_refinement"
        )
        quadrature_component = (
            "quadrature_root_normalization_refinement"
            if dataset in {"sphere", "high_sphere"}
            else "quadrature_refinement"
            if dataset != "plane"
            else None
        )
        _check_component_formula(
            trace,
            "derivative",
            derivative_pair,
            derivative_component,
            longdouble=False,
            precision=precision,
            index=index,
        )
        _check_component_formula(
            trace,
            "quadrature",
            quadrature_pair,
            quadrature_component,
            longdouble=False,
            precision=precision,
            index=index,
        )
    elif route in {"numpy.longdouble", "mpmath_fallback"}:
        quadrature_pair, longdouble = _surface_callback_pair(
            calls, dataset, row["fixture"], row["metadata"], trace, observable
        )
        _require_saved_pair(
            trace,
            "quadrature_differences_si",
            quadrature_pair,
            ("coarse_to_middle", "middle_to_fine"),
            longdouble=longdouble,
            precision=precision,
            index=index,
            observable=observable,
        )
        _require_trend_pair(
            trace["quadrature_trend"],
            quadrature_pair,
            longdouble=longdouble,
            precision=precision,
            index=index,
            observable=observable,
            key="quadrature_trend",
        )
        component = (
            "high_precision_surface_quadrature"
            if route == "mpmath_fallback"
            else "surface_quadrature_refinement"
        )
        _check_component_formula(
            trace,
            "quadrature",
            quadrature_pair,
            component,
            longdouble=longdouble,
            precision=precision,
            index=index,
        )
    else:
        raise ValueError(f"trace source row {index} has an unknown numerical route")

    floor_dps = (
        max(130 if dataset == "high_sphere" else 80, _declared_callback_digits(calls))
        if route in {"mpmath", "mpmath_fallback"}
        else mp.mp.dps
    )
    with mp.workdps(floor_dps):
        floors = _floor_candidates(reference["uncertainty"], observable=observable)
        expected_floor = _floor_for(floors)
    for key in ("derivative_trend", "quadrature_trend"):
        if trace[key].get("classification") == "not_applicable":
            continue
        saved_floor = trace[key].get("independent_floor_si")
        if expected_floor is None:
            if saved_floor is not None:
                raise ValueError(
                    f"trace source row {index} has an unsupported "
                    f"{observable} trend floor"
                )
        else:
            _require_number_equal(
                saved_floor,
                expected_floor,
                longdouble=trace.get("route") == "numpy.longdouble",
                precision=floor_dps,
                message=(
                    f"trace source row {index} has an invalid {observable} trend floor"
                ),
            )

    floor_field = (
        "floor_components_si"
        if "floor_components_si" in trace
        else "independent_floor_components_si"
    )
    floor_values = trace.get(floor_field)
    if not isinstance(floor_values, dict) or set(floor_values) != set(floors):
        raise ValueError(
            f"trace source row {index} has an incomplete {observable} floor"
        )
    for name, value in floors.items():
        _require_number_equal(
            floor_values[name],
            value,
            longdouble=False,
            precision=floor_dps,
            message=(
                f"trace source row {index} has an invalid {observable} floor component"
            ),
        )
    expected_reasons = {
        name: INDEPENDENT_FLOOR_COMPONENT_REASONS[name] for name in floors
    }
    if trace.get("independent_floor_component_reasons") != expected_reasons:
        raise ValueError(
            f"trace source row {index} has invalid {observable} floor reasons"
        )


def _validate_component_checks(checks, row, index, observable):
    if (
        not isinstance(checks, dict)
        or not checks
        or not set(checks).issubset({"derivative", "quadrature"})
    ):
        raise ValueError(
            f"trace source row {index} has malformed {observable} component checks"
        )
    frozen_components = dict(
        row["reference"][observable].get("uncertainty", {}).get("components", [])
    )
    for check in checks.values():
        if check is None:
            continue
        if (
            not isinstance(check, dict)
            or not isinstance(check.get("component_name"), str)
            or not check["component_name"]
            or type(check.get("matches_frozen_float")) is not bool
            or "formula_float_si" not in check
            or "frozen_component_si" not in check
        ):
            raise ValueError(
                f"trace source row {index} has malformed {observable} component check"
            )
        formula_float = check["formula_float_si"]
        formula_text = check.get("formula_value_si")
        frozen_value = check["frozen_component_si"]
        for value in (formula_float, frozen_value):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(
                    f"trace row {index}: nonfinite {observable} component check"
                )
        if "formula_value_si" in check:
            if formula_text is not None:
                if not isinstance(formula_text, str):
                    raise ValueError(
                        f"trace row {index}: malformed {observable} component formula"
                    )
                if formula_float is None:
                    raise ValueError(
                        f"trace {index}: inconsistent {observable} formula"
                    )
                try:
                    if float(formula_text) != formula_float:
                        raise ValueError(
                            f"trace {index}: inconsistent {observable} formula"
                        )
                except (OverflowError, ValueError) as error:
                    raise ValueError(
                        f"trace row {index}: malformed {observable} component formula"
                    ) from error
            elif formula_float is not None:
                raise ValueError(
                    f"trace row {index}: inconsistent {observable} component formula"
                )
        expected_frozen = frozen_components.get(check["component_name"])
        if frozen_value != expected_frozen:
            raise ValueError(
                f"trace row {index}: inconsistent frozen {observable} component"
            )
        if check["matches_frozen_float"] is not (frozen_value == formula_float):
            raise ValueError(
                f"trace source row {index} has inconsistent {observable} component flag"
            )


def _validate_observable_traces(record, row, index, dataset):
    traces = record.get("observable_traces")
    expected_names = set(OBSERVABLE_SLICES)
    if not isinstance(traces, dict) or set(traces) != expected_names:
        raise ValueError(
            f"trace source row {index} must contain exactly the frozen observable set"
        )
    for observable in OBSERVABLE_SLICES:
        trace = traces[observable]
        if not isinstance(trace, dict):
            raise ValueError(
                f"trace source row {index} has malformed {observable} trace"
            )
        trace_status = trace.get("status")
        if trace_status not in {
            "numerical",
            "certified_override",
            "exact_whole_case_certificate",
        }:
            raise ValueError(
                f"trace source row {index} has unknown {observable} trace status"
            )
        _validate_trend(trace, "derivative_trend", index, observable)
        _validate_trend(trace, "quadrature_trend", index, observable)
        _validate_trend_applicability(trace, dataset, row["fixture"], index, observable)
        if trace_status == "numerical":
            if not isinstance(trace.get("route"), str) or not trace["route"]:
                raise ValueError(
                    f"trace source row {index} has no {observable} numerical route"
                )
            if (
                not isinstance(trace.get("evaluation_inputs"), dict)
                or not trace["evaluation_inputs"]
            ):
                raise ValueError(
                    f"trace source row {index} has no {observable} evaluation inputs"
                )
            if type(trace.get("captured_final_values_match_frozen")) is not bool:
                raise ValueError(
                    f"trace source row {index} has no {observable} frozen-value result"
                )
            if "captured_final_values_si" not in trace:
                raise ValueError(
                    f"trace source row {index} has no {observable} captured values"
                )
            expected_length = len(row["reference"][observable]["values_si"])
            actual_length = _captured_vector_length(trace["captured_final_values_si"])
            if actual_length != expected_length:
                raise ValueError(
                    f"trace source row {index} has incomplete {observable} values"
                )
            checks = trace.get("frozen_component_checks")
            expected_check_slots = (
                {"derivative", "quadrature"}
                if trace["route"] == "mpmath"
                else {"quadrature"}
                if trace["route"] in {"numpy.longdouble", "mpmath_fallback"}
                else set()
            )
            if (
                not expected_check_slots
                or not isinstance(checks, dict)
                or set(checks) != expected_check_slots
            ):
                raise ValueError(
                    f"trace source row {index} has incomplete {observable} "
                    "component-check slots for its route"
                )
            _validate_component_checks(checks, row, index, observable)
            captured_matches = (
                _captured_complex_vector(
                    trace["captured_final_values_si"], index, observable
                )
                == row["reference"][observable]["values_si"]
            )
            if trace["captured_final_values_match_frozen"] is not captured_matches:
                raise ValueError(
                    f"trace row {index}: inconsistent {observable} captured-value flag"
                )
        elif trace_status == "certified_override":
            if "frozen_component_checks" in trace:
                raise ValueError(
                    f"trace source row {index} has component checks on a "
                    f"certified {observable} override"
                )
            if (
                not isinstance(trace.get("certificate"), str)
                or not trace["certificate"]
            ):
                raise ValueError(
                    f"trace source row {index} has no {observable} certificate"
                )
            uncertainty = row["reference"][observable].get("uncertainty", {})
            if uncertainty.get("uncertainty_kind") != "rigorous_bound" or trace[
                "certificate"
            ] != uncertainty.get("method"):
                raise ValueError(
                    f"trace source row {index} has an out-of-scope "
                    f"{observable} override"
                )
            if (
                trace.get("captured_evaluations_were_used_for_this_observable")
                is not False
            ):
                raise ValueError(
                    f"trace source row {index} has invalid "
                    f"{observable} override evidence"
                )
        else:
            if "frozen_component_checks" in trace:
                raise ValueError(
                    f"trace source row {index} has component checks on an "
                    f"unevaluated {observable} certificate"
                )
            if (
                not isinstance(trace.get("certificate"), str)
                or not trace["certificate"]
            ):
                raise ValueError(
                    f"trace source row {index} has no {observable} "
                    "whole-case certificate"
                )
    exact_whole_case = [
        trace
        for trace in traces.values()
        if trace["status"] == "exact_whole_case_certificate"
    ]
    if exact_whole_case:
        if dataset != "high_sphere" or len(exact_whole_case) != len(traces):
            raise ValueError(
                f"trace source row {index} has an invalid whole-case certificate scope"
            )
        expected_certificate = (
            record.get("original_entrypoint", {})
            .get("actual_metadata", {})
            .get("exact_zero_certificate")
        )
        if not expected_certificate or any(
            trace["certificate"] != expected_certificate for trace in exact_whole_case
        ):
            raise ValueError(
                f"trace source row {index} has an unpinned whole-case certificate"
            )
    return traces


def _derived_trace_status(traces, failed_checks):
    raw_unresolved = [
        f"{name}/{key}"
        for name in OBSERVABLE_SLICES
        for key in ("derivative_trend", "quadrature_trend")
        if traces[name][key]["classification"].startswith("unresolved")
    ]
    unresolved = []
    for item in raw_unresolved:
        observable, key = item.split("/", 1)
        trace = traces[observable]
        zero_adjudication = trace.get("zero_certificate_adjudication", {})
        polynomial_adjudication = trace.get(
            "polynomial_quadrature_certificate_adjudication", {}
        )
        if key in zero_adjudication.get("adjudicated_zero_change_trends", []):
            continue
        if (
            key == "quadrature_trend"
            and polynomial_adjudication.get("raw_quadrature_trend_classification")
            == "unresolved_zero_change"
        ):
            continue
        unresolved.append(item)
    status = (
        "replay_mismatch"
        if failed_checks
        else "unresolved_trend"
        if unresolved
        else "reproduced"
    )
    return raw_unresolved, unresolved, status


def _validate_trace_record(
    record,
    dataset,
    index,
    row,
    *,
    zero_certificates=(),
    zero_certificate_sha256=None,
    polynomial_certificates=(),
    polynomial_certificate_sha256=None,
    require_worker_evidence=False,
):
    if record.get("schema") != SCHEMA:
        raise ValueError(f"trace source row {index} has an unknown schema")
    _validate_mpf_displays(record)
    if record.get("dataset") != dataset:
        raise ValueError(f"trace source row {index} has the wrong dataset")
    if type(record.get("source_row_index")) is not int:
        raise ValueError(f"trace source row {index} has a non-integer index")
    if record["source_row_index"] != index:
        raise ValueError(f"trace source row index mismatch: expected {index}")
    if _canonical_json(record.get("fixture")) != _canonical_json(row["fixture"]):
        raise ValueError(f"trace fixture identity differs from frozen row {index}")
    expected_frozen = {"reference": row["reference"], "metadata": row["metadata"]}
    if _canonical_json(record.get("frozen_reference")) != _canonical_json(
        expected_frozen
    ):
        raise ValueError(f"trace frozen payload differs from frozen row {index}")
    if (
        record.get("polynomial_certificate_supplement_sha256")
        != polynomial_certificate_sha256
    ):
        raise ValueError(
            f"trace source row {index} uses a different polynomial certificate pin"
        )
    if record.get("zero_certificate_supplement_sha256") != zero_certificate_sha256:
        raise ValueError(
            f"trace source row {index} uses a different zero-certificate pin"
        )
    _validate_process_independence(record, index, required=require_worker_evidence)
    status = record.get("status")
    if status == "replay_error":
        if not isinstance(record.get("error"), str):
            raise ValueError(f"replay-error trace {index} has no error text")
        if record["error"] == "":
            raise ValueError(f"replay-error trace {index} has empty error text")
        if "observable_traces" in record or "reproduction" in record:
            raise ValueError(f"replay-error trace {index} has inconsistent status data")
        if "raw_unresolved_trends" in record or "unresolved_trends" in record:
            raise ValueError(
                f"replay-error trace {index} has trend data without traces"
            )
        captured_before_error = record.get("captured_calls_before_error")
        if (
            not isinstance(captured_before_error, list)
            or type(record.get("capture_call_count_before_error")) is not int
            or record["capture_call_count_before_error"] != len(captured_before_error)
        ):
            raise ValueError(
                f"replay-error trace {index} capture count differs from call list"
            )
        _validate_captured_calls(
            captured_before_error,
            index,
            allow_empty=True,
            allow_errors=True,
        )
        for call in record["captured_calls_before_error"]:
            if "output" not in call and not isinstance(call.get("_error"), str):
                raise ValueError(
                    f"replay-error trace {index} has an incomplete callback record"
                )
        return
    if status not in {"reproduced", "unresolved_trend", "replay_mismatch"}:
        raise ValueError(f"trace source row {index} has unknown status {status!r}")
    original = record.get("original_entrypoint")
    reproduction = record.get("reproduction")
    if not isinstance(original, dict) or not isinstance(reproduction, dict):
        raise ValueError(f"trace source row {index} is missing replay output evidence")
    if not isinstance(original.get("actual_reference"), dict) or not isinstance(
        original.get("actual_metadata"), dict
    ):
        raise ValueError(f"trace source row {index} has malformed returned output")
    if set(original["actual_reference"]) != set(OBSERVABLE_SLICES):
        raise ValueError(f"trace source row {index} has an incomplete returned output")
    captured_calls = original.get("captured_calls")
    capture_count = original.get("capture_call_count")
    if (
        not isinstance(captured_calls, list)
        or type(capture_count) is not int
        or capture_count != len(captured_calls)
    ):
        raise ValueError(
            f"trace source row {index} capture count differs from call list"
        )
    traces = _validate_observable_traces(record, row, index, dataset)
    exact_whole_case = all(
        trace["status"] == "exact_whole_case_certificate" for trace in traces.values()
    )
    _validate_captured_calls(captured_calls, index, allow_empty=exact_whole_case)
    if exact_whole_case and captured_calls:
        raise ValueError(
            f"trace source row {index} has callbacks for an unevaluated certificate"
        )
    for observable, trace in traces.items():
        _validate_callback_trend(
            trace,
            row["reference"][observable],
            captured_calls,
            dataset,
            row,
            index,
            observable,
        )
    values_match = _canonical_json(original.get("actual_reference")) == _canonical_json(
        row["reference"]
    )
    metadata_match = _canonical_json(
        original.get("actual_metadata")
    ) == _canonical_json(row["metadata"])
    if (
        reproduction.get("reference_matches_full_values_uncertainty_methods_components")
        is not values_match
    ):
        raise ValueError(f"trace source row {index} has inconsistent value-match flag")
    if reproduction.get("metadata_matches") is not metadata_match:
        raise ValueError(
            f"trace source row {index} has inconsistent metadata-match flag"
        )
    failed_checks = reproduction.get("failed_checks")
    if not isinstance(failed_checks, list) or any(
        not isinstance(item, str) for item in failed_checks
    ):
        raise ValueError(f"trace source row {index} has malformed failed checks")
    derived_failures = []
    for observable in OBSERVABLE_SLICES:
        trace = traces[observable]
        if trace["status"] != "numerical":
            continue
        if not trace["captured_final_values_match_frozen"]:
            derived_failures.append(
                f"{observable}: captured values differ from frozen values"
            )
        for check in trace["frozen_component_checks"].values():
            if check is not None and not check["matches_frozen_float"]:
                derived_failures.append(
                    f"{observable}: {check['component_name']} differs from "
                    "frozen component"
                )
    if not values_match:
        derived_failures.append(
            "actual original-entrypoint values/uncertainties differ from "
            "frozen reference"
        )
    if not metadata_match:
        derived_failures.append(
            "actual original-entrypoint metadata differ from frozen metadata"
        )
    if failed_checks != derived_failures:
        raise ValueError(
            f"trace source row {index} failed checks differ from replay evidence"
        )
    expected_passed = values_match and metadata_match and not failed_checks
    if reproduction.get("passed") is not expected_passed:
        raise ValueError(
            f"trace source row {index} has inconsistent reproduction result"
        )
    for observable, trace in traces.items():
        polynomial_adjudication = trace.get(
            "polynomial_quadrature_certificate_adjudication"
        )
        matched_polynomial_certificate = _matching_polynomial_certificate(
            dataset,
            index,
            row["fixture"],
            observable,
            polynomial_certificates,
        )
        if (
            values_match
            and metadata_match
            and matched_polynomial_certificate is not None
            and trace.get("quadrature_trend", {}).get("classification")
            == "unresolved_zero_change"
            and polynomial_adjudication is None
        ):
            raise ValueError(
                f"trace source row {index} is missing its polynomial certificate"
            )
        if polynomial_adjudication is not None:
            if not values_match or not metadata_match:
                raise ValueError(
                    f"trace source row {index} adjudicates an unreproduced "
                    "polynomial row"
                )
            _validate_polynomial_adjudication(
                polynomial_adjudication,
                dataset,
                index,
                row,
                observable,
                trace,
                polynomial_certificates,
                polynomial_certificate_sha256,
            )
        adjudication = trace.get("zero_certificate_adjudication")
        if adjudication is None:
            continue
        if not values_match or not metadata_match:
            raise ValueError(
                f"trace source row {index} adjudicates an unreproduced zero row"
            )
        if observable not in row["reference"]:
            raise ValueError(
                f"trace source row {index} certifies an unknown observable"
            )
        certificate = _matching_zero_certificate(
            dataset,
            row["fixture"],
            observable,
            row["reference"][observable],
            zero_certificates,
        )
        if certificate is None or _canonical_json(
            {
                key: adjudication.get(key)
                for key in (
                    "dataset",
                    "fixture",
                    "observable",
                    "certificate",
                    "original_uncertainty_kind",
                    "original_used_uncertainty_si",
                    "frozen_estimate_and_values_unchanged",
                )
            }
        ) != _canonical_json(
            {
                "dataset": certificate["dataset"],
                "fixture": certificate["fixture"],
                "observable": certificate["observable"],
                "certificate": certificate["certificate"],
                "original_uncertainty_kind": certificate["original_uncertainty_kind"],
                "original_used_uncertainty_si": certificate[
                    "original_used_uncertainty_si"
                ],
                "frozen_estimate_and_values_unchanged": certificate[
                    "frozen_labels_and_values_unchanged"
                ],
            }
        ):
            raise ValueError(
                f"trace source row {index} has an invalid zero certificate"
            )
        if adjudication.get("supplement_sha256") != zero_certificate_sha256:
            raise ValueError(
                f"trace source row {index} has an unpinned zero certificate"
            )
        expected_raw = {
            key: trace.get(key, {}).get("classification")
            for key in ("derivative_trend", "quadrature_trend")
        }
        if adjudication.get("raw_trend_classifications") != expected_raw:
            raise ValueError(f"trace source row {index} changed its raw trend record")
        expected_adjudicated = [
            key
            for key in ("derivative_trend", "quadrature_trend")
            if expected_raw[key] == "unresolved_zero_change"
        ]
        if adjudication.get("adjudicated_zero_change_trends") != expected_adjudicated:
            raise ValueError(
                f"trace source row {index} changed zero trend adjudication"
            )
        if adjudication.get("supplement_sha256") != zero_certificate_sha256:
            raise ValueError(
                f"trace source row {index} has an unpinned zero certificate"
            )
        expected_zero_adjudication = {
            "dataset": certificate["dataset"],
            "fixture": certificate["fixture"],
            "observable": certificate["observable"],
            "certificate": certificate["certificate"],
            "supplement_path": adjudication.get("supplement_path"),
            "supplement_sha256": zero_certificate_sha256,
            "original_uncertainty_kind": certificate["original_uncertainty_kind"],
            "original_used_uncertainty_si": certificate["original_used_uncertainty_si"],
            "raw_trend_classifications": expected_raw,
            "adjudicated_zero_change_trends": expected_adjudicated,
            "frozen_estimate_and_values_unchanged": certificate[
                "frozen_labels_and_values_unchanged"
            ],
        }
        if _canonical_json(adjudication) != _canonical_json(expected_zero_adjudication):
            raise ValueError(
                f"trace source row {index} has an invalid zero certificate"
            )
    raw_unresolved, unresolved, expected_status = _derived_trace_status(
        traces, failed_checks
    )
    if record.get("raw_unresolved_trends") != raw_unresolved:
        raise ValueError(f"trace source row {index} has inconsistent raw trends")
    if record.get("unresolved_trends") != unresolved:
        raise ValueError(f"trace source row {index} has inconsistent unresolved trends")
    if status != expected_status:
        raise ValueError(
            f"trace source row {index} status is inconsistent with replay evidence"
        )
    if "scientific_admission" in record:
        if record.get("execution_state") not in {
            "running",
            "complete",
            "failed",
            "interrupted",
        }:
            raise ValueError(f"trace source row {index} has an invalid execution state")
        expected_semantic = (
            "SEMANTIC_NOT_RUN" if status == "replay_error" else "SEMANTIC_VALID"
        )
        if record.get("semantic_validation") != expected_semantic:
            raise ValueError(
                f"trace source row {index} semantic state disagrees with its record"
            )
        if record.get("provenance_validation") not in {
            "PROVENANCE_VERIFIED",
            "PROVENANCE_UNVERIFIED",
            "PROVENANCE_MISMATCH",
            "PROVENANCE_NOT_RUN",
        }:
            raise ValueError(
                f"trace source row {index} has an invalid provenance state"
            )
        expected_certificate = (
            "CERTIFICATE_NOT_RUN"
            if status == "replay_error"
            else "CERTIFICATE_SCOPE_VALID"
        )
        if record.get("certificate_validation") != expected_certificate:
            raise ValueError(
                f"trace source row {index} has an invalid certificate state"
            )
        if record.get("scientific_admission") != "NOT_ADMITTED":
            raise ValueError(
                f"trace source row {index} is admitted before independent audit"
            )


def _verify_trace_file(
    path,
    checkpoint,
    *,
    dataset,
    selected_rows,
    selected_indices,
    zero_certificates=(),
    zero_certificate_sha256=None,
    polynomial_certificates=(),
    polynomial_certificate_sha256=None,
):
    if checkpoint.get("schema") != "gragra.gen1.refinement-trace-checkpoint.v4":
        raise RuntimeError("checkpoint schema is not supported by observer v4")
    selected_indices = _validated_selected_indices(
        selected_indices, selected_rows.keys()
    )
    checkpoint_indices = _validated_selected_indices(
        checkpoint.get("selected_source_row_indices", []), selected_rows.keys()
    )
    if checkpoint_indices != selected_indices:
        raise RuntimeError("checkpoint selected source row indices changed")
    if set(selected_rows) != set(selected_indices):
        raise RuntimeError("selected frozen row mapping is incomplete")
    status = checkpoint.get("status")
    if status not in {"running", "complete", "unresolved_trend", "replay_failure"}:
        raise RuntimeError(f"unknown checkpoint status {status!r}")

    data = Path(path).read_bytes()
    committed_bytes = checkpoint.get("committed_trace_bytes")
    committed_hash = checkpoint.get("committed_trace_sha256")
    if type(committed_bytes) is not int or committed_bytes < 0:
        raise RuntimeError("checkpoint has an invalid committed trace byte count")
    if committed_bytes > len(data):
        raise RuntimeError("trace file is shorter than committed trace prefix")
    committed_prefix = data[:committed_bytes]
    if _sha256_bytes(committed_prefix) != committed_hash:
        raise RuntimeError("committed trace prefix SHA-256 mismatch")
    if committed_prefix and not committed_prefix.endswith(b"\n"):
        raise RuntimeError("committed trace prefix ends inside a JSONL record")
    committed_records = _decode_trace_bytes(
        committed_prefix, description="committed trace prefix"
    )

    completed_checkpoint = _validated_selected_indices(
        checkpoint.get("completed_source_row_indices", []), selected_rows.keys()
    )
    committed_indices = []
    for record in committed_records:
        index = record.get("source_row_index")
        if type(index) is not int or index not in selected_rows:
            raise ValueError(f"trace source row index {index!r} is not selected")
        _validate_trace_record(
            record,
            dataset,
            index,
            selected_rows[index],
            zero_certificates=zero_certificates,
            zero_certificate_sha256=zero_certificate_sha256,
            polynomial_certificates=polynomial_certificates,
            polynomial_certificate_sha256=polynomial_certificate_sha256,
            require_worker_evidence=True,
        )
        committed_indices.append(index)
    if len(committed_indices) != len(set(committed_indices)):
        raise ValueError("duplicate trace source row index in committed prefix")
    if set(completed_checkpoint) != set(committed_indices):
        raise RuntimeError("checkpoint completed indices differ from committed records")

    tail = data[committed_bytes:]
    if status != "running":
        if committed_bytes != len(data):
            raise RuntimeError("finished checkpoint has uncommitted trace bytes")
        if checkpoint.get("trace_bytes") != len(data):
            raise RuntimeError("finished checkpoint trace byte count mismatch")
        if checkpoint.get("traces_jsonl_sha256") != _sha256_bytes(data):
            raise RuntimeError("finished trace SHA-256 mismatch")
        if checkpoint.get("record_count") != len(committed_records):
            raise RuntimeError("finished checkpoint record count mismatch")
        if set(committed_indices) != set(selected_indices):
            raise ValueError("final trace set is incomplete")
        mismatch_count = sum(
            record.get("status") in {"replay_mismatch", "replay_error"}
            for record in committed_records
        )
        unresolved_count = sum(
            bool(record.get("unresolved_trends")) for record in committed_records
        )
        if checkpoint.get("case_count") != len(selected_indices):
            raise RuntimeError("finished checkpoint case count mismatch")
        if checkpoint.get("mismatch_count") != mismatch_count:
            raise RuntimeError("finished checkpoint mismatch count is inconsistent")
        if checkpoint.get("unresolved_trend_case_count") != unresolved_count:
            raise RuntimeError("finished checkpoint unresolved count is inconsistent")
        process_names = checkpoint.get("production_module_names")
        if (
            not isinstance(process_names, list)
            or any(not isinstance(name, str) for name in process_names)
            or process_names != sorted(set(process_names))
        ):
            raise RuntimeError("finished checkpoint has malformed process evidence")
        production_imported = bool(process_names)
        if checkpoint.get(
            "production_imported"
        ) is not production_imported or checkpoint.get(
            "independence_check_passed"
        ) is not (not production_imported):
            raise RuntimeError("finished checkpoint process status is inconsistent")
        worker_names = {
            name
            for record in committed_records
            for name in record["process_independence"]["production_module_names"]
        }
        if not worker_names.issubset(set(process_names)):
            raise RuntimeError("finished checkpoint omits worker process evidence")
        expected_status = (
            "replay_failure"
            if mismatch_count or production_imported
            else "unresolved_trend"
            if unresolved_count
            else "complete"
        )
        if status != expected_status:
            raise RuntimeError(
                "finished checkpoint status is inconsistent with evidence"
            )
        records = committed_records
        completed_indices = set(committed_indices)
        recovered_indices = set()
    else:
        # Bytes beyond the checkpoint are not committed evidence. Preserve
        # them verbatim and rerun their selected rows; even a syntactically
        # valid row may have been only partially written or semantically cut.
        records = list(committed_records)
        recovered_indices = set()
        all_indices = [record["source_row_index"] for record in records]
        if len(all_indices) != len(set(all_indices)):
            raise ValueError("duplicate trace source row index")
        completed_indices = set(all_indices)
    return {
        "records": records,
        "completed_source_row_indices": completed_indices,
        "recovered_source_row_indices": recovered_indices,
        "uncommitted_tail_bytes": tail,
        "committed_trace_bytes": committed_bytes,
        "committed_trace_sha256": committed_hash,
        "trace_bytes": len(data),
        "trace_sha256": _sha256_bytes(data),
    }


def _validate_quarantine_history(target, entries):
    if not isinstance(entries, list):
        raise RuntimeError("checkpoint has malformed quarantined-tail history")
    root = Path(target).resolve()
    for entry in entries:
        if not isinstance(entry, dict):
            raise RuntimeError("checkpoint has malformed quarantined-tail record")
        relative = entry.get("path")
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise RuntimeError("checkpoint has invalid quarantined-tail path")
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise RuntimeError("checkpoint quarantined-tail file is missing")
        if (
            type(entry.get("byte_count")) is not int
            or path.stat().st_size != entry["byte_count"]
            or _sha256(path) != entry.get("sha256")
        ):
            raise RuntimeError("checkpoint quarantined-tail hash mismatch")
        if not isinstance(entry.get("invocation_id"), str):
            raise RuntimeError("checkpoint quarantined-tail has no invocation id")
        if type(entry.get("committed_trace_bytes")) is not int or not isinstance(
            entry.get("committed_trace_sha256"), str
        ):
            raise RuntimeError("checkpoint quarantined-tail has no prefix pin")
    return list(entries)


def _quarantine_uncommitted_tail(
    target, records_path, checkpoint_path, checkpoint, tail, invocation_id
):
    if not tail:
        return checkpoint, None
    target = Path(target)
    quarantine_dir = target / "quarantine"
    quarantine_dir.mkdir(exist_ok=True)
    relative_path = Path("quarantine") / f"{invocation_id}-{uuid.uuid4().hex}.bin"
    quarantine_path = target / relative_path
    with quarantine_path.open("xb") as stream:
        stream.write(tail)
        stream.flush()
        os.fsync(stream.fileno())
    entry = {
        "path": relative_path.as_posix(),
        "sha256": _sha256_bytes(tail),
        "byte_count": len(tail),
        "invocation_id": invocation_id,
        "committed_trace_bytes": checkpoint["committed_trace_bytes"],
        "committed_trace_sha256": checkpoint["committed_trace_sha256"],
    }
    history = list(checkpoint.get("quarantined_uncommitted_tails", []))
    history.append(entry)
    updated_checkpoint = dict(checkpoint)
    updated_checkpoint["quarantined_uncommitted_tails"] = history
    _atomic_json(checkpoint_path, updated_checkpoint)

    with Path(records_path).open("r+b") as stream:
        stream.truncate(checkpoint["committed_trace_bytes"])
        stream.flush()
        os.fsync(stream.fileno())
    committed_prefix = Path(records_path).read_bytes()
    if (
        len(committed_prefix) != checkpoint["committed_trace_bytes"]
        or _sha256_bytes(committed_prefix) != checkpoint["committed_trace_sha256"]
    ):
        raise RuntimeError("trace prefix changed while quarantining uncommitted tail")
    return updated_checkpoint, entry


def _load_zero_certificates(l6_root):
    path = Path(l6_root) / ZERO_CERTIFICATES_FILENAME
    digest = _sha256(path)
    if digest != EXPECTED_ZERO_CERTIFICATES_SHA256:
        raise RuntimeError(
            "pinned frozen zero-certificate supplement SHA mismatch: "
            f"expected {EXPECTED_ZERO_CERTIFICATES_SHA256}, got {digest}"
        )
    data = _strict_json_file(path)
    if (
        data.get("certified_records") != 173
        or data.get("uncertified_zero_records") != 0
    ):
        raise RuntimeError("pinned zero-certificate supplement has unexpected scope")
    for dataset in ("sphere", "noncavity"):
        reference_path, manifest_path = _dataset_paths(Path(l6_root), dataset)
        for input_path in (reference_path, manifest_path):
            expected = data.get("frozen_inputs", {}).get(str(input_path.resolve()))
            if expected != _sha256(input_path):
                raise RuntimeError(
                    f"zero-certificate supplement does not pin {input_path}"
                )
    return {"path": str(path.resolve()), "sha256": digest, "data": data}


def _load_polynomial_quadrature_certificates(l6_root, dataset):
    if dataset not in {"noncavity", "source"}:
        return None
    path = Path(l6_root).resolve() / POLYNOMIAL_CERTIFICATES_FILENAME
    digest = _sha256(path)
    if digest != EXPECTED_POLYNOMIAL_CERTIFICATES_SHA256:
        raise RuntimeError(
            "pinned polynomial-quadrature certificate SHA mismatch: "
            f"expected {EXPECTED_POLYNOMIAL_CERTIFICATES_SHA256}, got {digest}"
        )
    data = _strict_json_file(path)
    if (
        data.get("scope")
        != "additive ideal-polynomial quadrature certificate for pinned fixtures only"
        or data.get("script_sha256") != EXPECTED_POLYNOMIAL_VERIFIER_SCRIPT_SHA256
        or data.get("production_modules_observed") != []
    ):
        raise RuntimeError(
            "pinned polynomial-quadrature certificate provenance differs"
        )

    expected_oracle_inputs = {
        name: EXPECTED_ORACLE_SHA256[name]
        for name in (
            "noncavity_reference.py",
            "noncavity_fixtures.py",
            "source_reference.py",
            "source_fixtures.py",
            "sphere_reference.py",
        )
    }
    recorded_inputs = data.get("inputs")
    if not isinstance(recorded_inputs, dict):
        raise RuntimeError("polynomial-quadrature input provenance is malformed")
    for filename, expected_digest in expected_oracle_inputs.items():
        matches = [
            digest_value
            for input_path, digest_value in recorded_inputs.items()
            if Path(input_path).name == filename
        ]
        if matches != [expected_digest]:
            raise RuntimeError(
                f"polynomial-quadrature certificate does not pin {filename}"
            )
    for frozen_dataset in ("noncavity", "source"):
        reference_path, manifest_path = _dataset_paths(Path(l6_root), frozen_dataset)
        for input_path in (reference_path, manifest_path):
            digest_value = _sha256(input_path)
            matches = [
                recorded_digest
                for recorded_path, recorded_digest in recorded_inputs.items()
                if Path(recorded_path).parent.name == DATASETS[frozen_dataset]
                and Path(recorded_path).name == input_path.name
            ]
            if matches != [digest_value]:
                raise RuntimeError(
                    "polynomial-quadrature certificate does not pin "
                    f"{frozen_dataset}/{input_path.name}"
                )

    certificates = data.get("certificates")
    expected_indices = {("noncavity", index) for index in range(19, 46)} | {
        ("source", index) for index in range(6)
    }
    if not isinstance(certificates, list) or len(certificates) != 33:
        raise RuntimeError("polynomial-quadrature certificate count differs")
    actual_indices = []
    for certificate in certificates:
        dataset_name = certificate.get("dataset")
        row_index = certificate.get("source_row_index")
        actual_indices.append((dataset_name, row_index))
        expected_kind = {
            "noncavity": "retarded_cubic_pulse_polynomial_panels",
            "source": "uniform_Sphere_radial_polynomial_moments",
        }.get(dataset_name)
        expected_family_shape = {
            "noncavity": ("point_force", None),
            "source": ("finite_temperature", "sphere"),
        }.get(dataset_name)
        fixture = certificate.get("fixture")
        if (
            expected_kind is None
            or not isinstance(fixture, dict)
            or (fixture.get("family"), fixture.get("shape")) != expected_family_shape
            or certificate.get("certificate_kind") != expected_kind
            or certificate.get("covered_observables")
            != ["potential", "acceleration", "gradient"]
            or certificate.get("covered_original_stencils")
            != POLYNOMIAL_CERTIFICATE_COVERED_STENCILS
            or certificate.get("certified_error")
            != POLYNOMIAL_CERTIFICATE_CERTIFIED_ERROR
            or certificate.get("frozen_values_uncertainty_labels_unchanged") is not True
        ):
            raise RuntimeError("polynomial-quadrature certificate scope differs")
    if set(actual_indices) != expected_indices or len(actual_indices) != len(
        set(actual_indices)
    ):
        raise RuntimeError("polynomial-quadrature certificate fixture set differs")
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "source_certificate_sha256": POLYNOMIAL_CERTIFICATE_SOURCE_JSON_SHA256,
        "report_sha256": POLYNOMIAL_CERTIFICATE_REPORT_SHA256,
        "verifier_script_sha256": data["script_sha256"],
        "scope": data["scope"],
        "certificates": certificates,
    }


def _matching_polynomial_certificate(
    dataset, source_row_index, fixture, observable, certificates
):
    if dataset not in {"noncavity", "source"}:
        return None
    if type(source_row_index) is not int:
        return None
    if dataset == "noncavity" and source_row_index not in range(19, 46):
        return None
    if dataset == "source" and source_row_index not in range(6):
        return None
    if dataset == "noncavity" and fixture.get("family") != "point_force":
        return None
    if dataset == "source" and (
        fixture.get("family") != "finite_temperature"
        or fixture.get("shape") != "sphere"
    ):
        return None
    expected_kind = {
        "noncavity": "retarded_cubic_pulse_polynomial_panels",
        "source": "uniform_Sphere_radial_polynomial_moments",
    }[dataset]
    matches = [
        certificate
        for certificate in certificates
        if certificate.get("dataset") == dataset
        and certificate.get("source_row_index") == source_row_index
        and _canonical_json(certificate.get("fixture")) == _canonical_json(fixture)
        and observable in certificate.get("covered_observables", [])
        and certificate.get("certificate_kind") == expected_kind
        and certificate.get("certified_error") == POLYNOMIAL_CERTIFICATE_CERTIFIED_ERROR
        and certificate.get("covered_original_stencils")
        == POLYNOMIAL_CERTIFICATE_COVERED_STENCILS
        and certificate.get("frozen_values_uncertainty_labels_unchanged") is True
    ]
    if len(matches) > 1:
        raise RuntimeError("duplicate matching polynomial-quadrature certificates")
    return matches[0] if matches else None


def _matching_zero_certificate(
    dataset, fixture, observable, observable_record, certificates
):
    uncertainty = observable_record.get("uncertainty", {})
    values = observable_record.get("values_si")
    if (
        uncertainty.get("uncertainty_kind") != "estimate"
        or uncertainty.get("used_uncertainty_si") != 0.0
        or not isinstance(values, list)
        or not values
        or any(
            not isinstance(pair, list)
            or len(pair) != 2
            or pair[0] != 0.0
            or pair[1] != 0.0
            for pair in values
        )
    ):
        return None
    for certificate in certificates:
        if (
            certificate.get("dataset") == DATASETS[dataset]
            and _canonical_json(certificate.get("fixture")) == _canonical_json(fixture)
            and certificate.get("observable") == observable
            and certificate.get("original_uncertainty_kind") == "estimate"
            and certificate.get("original_used_uncertainty_si") == 0.0
            and certificate.get("frozen_labels_and_values_unchanged") is True
        ):
            return certificate
    return None


def _mpf_tuple(value: mp.mpf) -> list[int]:
    return [int(part) for part in value._mpf_]


def _capture_number(value, digits: int):
    if isinstance(value, mp.mpf):
        display_digits = max(50, digits + 4)
        with mp.workprec(max(53, value._mpf_[3] + 8)):
            display = mp.nstr(value, n=display_digits)
        return {
            "decimal": display,
            "display_digits": display_digits,
            "mpf": _mpf_tuple(value),
        }
    if isinstance(value, mp.mpc):
        return {
            "real": _capture_number(value.real, digits),
            "imag": _capture_number(value.imag, digits),
        }
    if isinstance(value, np.generic):
        if value.dtype.kind in "fc":
            if value.dtype.kind == "c":
                return {
                    "real": _capture_number(value.real, digits),
                    "imag": _capture_number(value.imag, digits),
                    "dtype": str(value.dtype),
                }
            return {
                "decimal": np.format_float_scientific(
                    value, unique=True, precision=21, trim="k"
                ),
                "dtype": str(value.dtype),
            }
        return int(value) if value.dtype.kind in "iu" else bool(value)
    if isinstance(value, np.ndarray):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": _capture_number(value.tolist(), digits),
        }
    if dataclasses.is_dataclass(value):
        return _capture_number(dataclasses.asdict(value), digits)
    if isinstance(value, dict):
        return {str(key): _capture_number(item, digits) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_capture_number(item, digits) for item in value]
    if isinstance(value, complex):
        return {"real": repr(value.real), "imag": repr(value.imag)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"type": type(value).__name__, "repr": repr(value)}


def _jsonable(value):
    """Match Python's JSON collector representation for original returns."""
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, complex):
        return [float(value.real), float(value.imag)]
    if isinstance(value, mp.mpc):
        return [float(value.real), float(value.imag)]
    if isinstance(value, mp.mpf):
        return float(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        if value.dtype.kind in "fc":
            return float(value)
        if value.dtype.kind in "iu":
            return int(value)
        return bool(value)
    return value


def _clone(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone(item) for item in value)
    if isinstance(value, list):
        return [_clone(item) for item in value]
    if hasattr(value, "copy") and callable(value.copy):
        try:
            return value.copy()
        except TypeError:
            pass
    return value


def _describe_mp_engine(engine, point, step):
    return {
        "engine_type": type(engine).__name__,
        "digits": getattr(engine, "digits", None),
        "quadrature_order": getattr(
            engine, "quadrature_order", getattr(engine, "order", None)
        ),
        "mp_dps_at_call": int(mp.mp.dps),
        "point_m": _capture_number(point, int(mp.mp.dps)),
        "step_m": _capture_number(step, int(mp.mp.dps)),
    }, {
        "engine": engine,
        "point": point,
        "step": step,
        "digits": getattr(engine, "digits", int(mp.mp.dps)),
        "order": getattr(engine, "quadrature_order", getattr(engine, "order", None)),
    }


def _watch(owner, attribute, kernel, describe, captures):
    original = getattr(owner, attribute)

    def wrapped(*args, **kwargs):
        public_inputs, raw_inputs = describe(args, kwargs)
        try:
            result = original(*args, **kwargs)
        except BaseException as error:
            captures.append(
                {
                    "kernel": kernel,
                    "inputs": public_inputs,
                    "_raw_inputs": raw_inputs,
                    "_error": f"{type(error).__name__}: {error}",
                }
            )
            raise
        if kernel == "source_surface_values":
            raw_result = (_clone(result[0]), _clone(result[1]))
            serial_result = {
                "values_and_absolute": _capture_number(raw_result, int(mp.mp.dps)),
                "geometry_summary": {
                    "characteristic_m": _capture_number(
                        result[2].characteristic, int(mp.mp.dps)
                    ),
                    "minimum_gap_m": _capture_number(
                        result[2].minimum_gap, int(mp.mp.dps)
                    ),
                },
            }
        else:
            raw_result = _clone(result)
            digits = int(raw_inputs.get("digits", mp.mp.dps))
            if kernel == "surface_high_precision_evaluate":
                digits = int(raw_inputs["digits"])
            serial_result = _capture_number(raw_result, digits)
        captures.append(
            {
                "kernel": kernel,
                "inputs": public_inputs,
                "_raw_inputs": raw_inputs,
                "_raw_result": raw_result,
                "output": serial_result,
            }
        )
        return result

    setattr(owner, attribute, wrapped)

    def restore():
        setattr(owner, attribute, original)

    return restore


@contextlib.contextmanager
def capture_context(dataset, captures):
    restorers = []

    def watch_observable(module, kernel):
        def describe(args, _kwargs):
            engine, point, step = args[:3]
            return _describe_mp_engine(engine, point, step)

        restorers.append(
            _watch(module, "observable_values", kernel, describe, captures)
        )

    try:
        if dataset in {"sphere", "high_sphere"}:
            module = oracle_module("sphere_reference")

            def describe(args, _kwargs):
                return _describe_mp_engine(args[0], args[1], args[2])

            restorers.append(
                _watch(
                    module.NewtonSphere,
                    "observables",
                    "sphere_observables",
                    describe,
                    captures,
                )
            )
        elif dataset == "noncavity":
            watch_observable(
                oracle_module("noncavity_reference"), "noncavity_observable_values"
            )
        elif dataset == "plane":
            watch_observable(
                oracle_module("plane_static_reference"), "plane_observable_values"
            )
        elif dataset == "source":
            module = oracle_module("source_reference")
            watch_observable(module, "source_observable_values")

            def describe_source_surface(args, _kwargs):
                fixture, order = args[:2]
                digits = int(mp.mp.dps)
                return {
                    "order": int(order),
                    "backend": "numpy.longdouble",
                    "mp_dps_at_call": digits,
                    "fixture": _capture_number(fixture, digits),
                }, {"fixture": fixture, "order": int(order), "digits": digits}

            restorers.append(
                _watch(
                    module,
                    "surface_values",
                    "source_surface_values",
                    describe_source_surface,
                    captures,
                )
            )
        elif dataset in {"cuboid", "cylinder"}:
            module = oracle_module("surface_reference")

            def describe_surface(args, kwargs):
                engine, order = args[:2]
                target = kwargs.get("target", args[2] if len(args) > 2 else None)
                if target is None:
                    target = engine.target
                return {
                    "order": int(order),
                    "backend": "numpy.longdouble",
                    "mp_dps_at_call": int(mp.mp.dps),
                    "target_m": _capture_number(target, int(mp.mp.dps)),
                }, {
                    "fixture": engine.fixture,
                    "order": int(order),
                    "target": _clone(target),
                    "digits": int(mp.mp.dps),
                }

            restorers.append(
                _watch(
                    module.SurfaceNewton,
                    "evaluate",
                    "surface_newton_evaluate",
                    describe_surface,
                    captures,
                )
            )
            high_precision = oracle_module("surface_high_precision")

            def describe_high_precision(args, _kwargs):
                fixture, order, digits, gravitational_constant = args[:4]
                return {
                    "order": int(order),
                    "digits": int(digits),
                    "backend": "mpmath",
                    "mp_dps_at_call": int(mp.mp.dps),
                    "gravitational_constant_si": float(gravitational_constant),
                    "fixture": _capture_number(fixture, int(digits)),
                }, {
                    "fixture": fixture,
                    "order": int(order),
                    "digits": int(digits),
                    "gravitational_constant": gravitational_constant,
                }

            restorers.append(
                _watch(
                    high_precision,
                    "evaluate_high_precision",
                    "surface_high_precision_evaluate",
                    describe_high_precision,
                    captures,
                )
            )
        else:
            raise ValueError(f"unsupported trace dataset: {dataset}")
        yield
    finally:
        for restore in reversed(restorers):
            restore()


def _sphere_fixture(module, fixture):
    normalized = dict(fixture)
    normalized["target_m"] = tuple(normalized["target_m"])
    return module.SphereFixture(**normalized)


def _surface_fixture(dataset, fixture):
    module = oracle_module("cavity_fixtures")
    normalized = dict(fixture)
    normalized["target_m"] = tuple(normalized["target_m"])
    if dataset == "cuboid":
        normalized["lengths_m"] = tuple(normalized["lengths_m"])
        normalized["mode"] = tuple(normalized["mode"])
        return module.CuboidFixture(**normalized)
    return module.CylinderFixture(**normalized)


def _invoke_original_entrypoint(dataset, fixture):
    if dataset == "sphere":
        module = oracle_module("sphere_reference")
        source = oracle_module("noncavity_fixtures")
        return module.sphere_reference(
            _sphere_fixture(oracle_module("sphere_fixtures"), fixture), source.G_SI
        )
    if dataset == "high_sphere":
        module = oracle_module("sphere_high_degree_reference")
        source = oracle_module("noncavity_fixtures")
        return module.sphere_high_degree_reference(
            _sphere_fixture(oracle_module("sphere_fixtures"), fixture), source.G_SI
        )
    if dataset == "noncavity":
        return oracle_module("noncavity_reference").noncavity_reference(fixture)
    if dataset == "plane":
        return oracle_module("plane_static_reference").static_plane_reference(fixture)
    if dataset == "source":
        return oracle_module("source_reference").source_reference(fixture)
    if dataset in {"cuboid", "cylinder"}:
        return oracle_module("surface_reference").surface_reference(
            _surface_fixture(dataset, fixture)
        )
    raise ValueError(f"unknown dataset {dataset!r}")


def _exact_mp_string(value):
    return mp.nstr(value, n=max(60, int(mp.mp.dps) + 6))


def _to_mp_vector(values):
    return tuple(mp.mpc(value) for value in values)


def _mp_difference(function, left, right, name):
    result = function({name: left}, {name: right})
    return result[name]


def _measure_string(value):
    if isinstance(value, np.generic) and value.dtype.kind == "f":
        return np.format_float_scientific(value, unique=True, precision=21, trim="k")
    return _exact_mp_string(value)


def _change_trend(coarse_to_middle, middle_to_fine, recorded_floor):
    if isinstance(coarse_to_middle, np.generic):
        floor_for_comparison = (
            None if recorded_floor is None else np.longdouble(str(recorded_floor))
        )
    else:
        floor_for_comparison = recorded_floor
    if middle_to_fine < coarse_to_middle:
        classification = "decreasing"
    elif (
        floor_for_comparison is not None
        and floor_for_comparison > 0
        and floor_for_comparison >= middle_to_fine
    ):
        classification = "nondecreasing_explained_by_independent_floor"
    elif middle_to_fine == 0 and coarse_to_middle == 0:
        classification = "unresolved_zero_change"
    else:
        classification = "unresolved_nondecreasing"
    return {
        "classification": classification,
        "nondecreasing": bool(middle_to_fine >= coarse_to_middle),
        "coarse_to_middle_si": _measure_string(coarse_to_middle),
        "middle_to_fine_si": _measure_string(middle_to_fine),
        "independent_floor_si": None
        if recorded_floor is None
        else _measure_string(recorded_floor),
    }


INDEPENDENT_FLOOR_COMPONENT_REASONS = {
    "precision_propagation": "independent multiprecision error propagation",
    "50_80_digit_precision": "fixed-order 50/80-digit precision comparison",
    "root_normalization_arithmetic_precision": (
        "independent arithmetic precision estimate for root normalization"
    ),
    "root_normalization_precision": (
        "independent root-normalization precision estimate"
    ),
    "high_precision_roundoff_estimate": (
        "independent high-precision arithmetic roundoff estimate"
    ),
    "longdouble_roundoff_estimate": (
        "independent longdouble arithmetic roundoff estimate"
    ),
    "float_conversion": "reported float conversion error estimate",
}


def _floor_candidates(uncertainty, *, observable):
    components = {name: mp.mpf(value) for name, value in uncertainty["components"]}
    del observable  # Crosscheck components are excluded for every observable.
    return {
        name: components[name]
        for name in INDEPENDENT_FLOOR_COMPONENT_REASONS
        if name in components
    }


def _floor_basis_metadata(components):
    return {name: INDEPENDENT_FLOOR_COMPONENT_REASONS[name] for name in components}


def _floor_for(components):
    return mp.fsum(components.values()) if components else None


def _component_check(uncertainty, component_name, computed):
    components = dict(uncertainty["components"])
    frozen = components.get(component_name)
    as_float = None if computed is None else float(computed)
    return {
        "component_name": component_name,
        "formula_value_si": None if computed is None else _measure_string(computed),
        "formula_float_si": as_float,
        "frozen_component_si": frozen,
        "matches_frozen_float": frozen == as_float,
    }


def _mp_trace_for_observable(dataset, name, captures, reference, metadata):
    high_degree = dataset in {"sphere", "high_sphere"}
    high_order = {
        "sphere": 48,
        "high_sphere": 96,
        "noncavity": 32,
        "plane": None,
        "source": 32,
    }[dataset]
    middle_order = {"sphere": 40, "high_sphere": 80, "noncavity": 24, "source": 24}.get(
        dataset
    )
    coarse_order = {"sphere": 32, "high_sphere": 64, "noncavity": 16, "source": 16}.get(
        dataset
    )
    high_dps = 130 if dataset == "high_sphere" else 80
    low_dps = 90 if dataset == "high_sphere" else 50
    kernel = (
        "sphere_observables"
        if high_degree
        else {
            "noncavity": "noncavity_observable_values",
            "plane": "plane_observable_values",
            "source": "source_observable_values",
        }[dataset]
    )
    events = [event for event in captures if event["kernel"] == kernel]
    high_events = [
        event
        for event in events
        if int(event["_raw_inputs"]["digits"]) == high_dps
        and (high_order is None or event["_raw_inputs"]["order"] == high_order)
    ]
    if not high_events:
        raise RuntimeError(
            f"no high precision evaluations captured for {dataset}/{name}"
        )
    final_event = high_events[-1]
    h = final_event["_raw_inputs"]["step"]
    if len(high_events) < 3:
        raise RuntimeError(
            f"fewer than three final-step refinements for {dataset}/{name}"
        )
    four_h_event, two_h_event, h_event = high_events[-3:]
    four_h, two_h, current = (
        event["_raw_result"][name] for event in (four_h_event, two_h_event, h_event)
    )

    sphere = oracle_module("sphere_reference")
    with mp.workdps(high_dps):
        observed_derivative_changes = (
            _mp_difference(sphere.differences, four_h, two_h, name),
            _mp_difference(sphere.differences, two_h, current, name),
        )
        derivative_applicable = name != "potential"
        derivative_changes = (
            observed_derivative_changes if derivative_applicable else None
        )
        quadrature_applicable = (
            dataset in {"sphere", "high_sphere"}
            or (
                dataset == "noncavity"
                and metadata.get("fixture_family")
                in {"gaussian", "halfspace", "point_force"}
            )
            or (
                dataset == "source"
                and metadata.get("fixture_family") != "temperature_grid"
            )
        )
        quadrature_changes = None
        observed_quadrature_changes = None
        quadrature_vectors = None
        if dataset != "plane":
            by_order = {}
            for event in events:
                raw = event["_raw_inputs"]
                event_dps = int(raw["digits"])
                event_order = raw.get("order")
                if (
                    event_dps == high_dps
                    and event_order in (middle_order, coarse_order)
                    and raw["step"] == h
                ):
                    by_order[event_order] = event["_raw_result"][name]
            if middle_order not in by_order or coarse_order not in by_order:
                raise RuntimeError(f"missing same-step order pair for {dataset}/{name}")
            quadrature_vectors = (
                by_order[coarse_order],
                by_order[middle_order],
                current,
            )
            observed_quadrature_changes = (
                _mp_difference(
                    sphere.differences,
                    quadrature_vectors[0],
                    quadrature_vectors[1],
                    name,
                ),
                _mp_difference(
                    sphere.differences,
                    quadrature_vectors[1],
                    quadrature_vectors[2],
                    name,
                ),
            )
            if quadrature_applicable:
                quadrature_changes = observed_quadrature_changes

        low_candidates = [
            event
            for event in events
            if int(event["_raw_inputs"]["digits"]) == low_dps
            and (high_order is None or event["_raw_inputs"]["order"] == high_order)
            and event["_raw_inputs"]["step"] == h
        ]
        precision_change = None
        if low_candidates:
            precision_change = _mp_difference(
                sphere.differences,
                current,
                low_candidates[-1]["_raw_result"][name],
                name,
            )

        frozen_uncertainty = reference[name]["uncertainty"]
        floors = _floor_candidates(frozen_uncertainty, observable=name)
        floor = _floor_for(floors)
        derivative_trend = (
            _change_trend(*derivative_changes, floor)
            if derivative_changes is not None
            else {
                "classification": "not_applicable",
                "nondecreasing": None,
                "reason": (
                    "potential is evaluated directly and does not depend on "
                    "target-difference step"
                ),
            }
        )
        quadrature_trend = (
            _change_trend(*quadrature_changes, floor)
            if quadrature_changes is not None
            else {
                "classification": "not_applicable",
                "nondecreasing": None,
                "reason": "entrypoint uses no spatial quadrature for this family",
            }
        )
        derivative_component_name = {
            "sphere": "derivative_refinement",
            "high_sphere": "derivative_refinement",
            "noncavity": "derivative_refinement",
            "plane": "real_field_derivative_refinement",
            "source": "derivative_refinement",
        }[dataset]
        quadrature_component_name = {
            "sphere": "quadrature_root_normalization_refinement",
            "high_sphere": "quadrature_root_normalization_refinement",
            "noncavity": "quadrature_refinement",
            "plane": None,
            "source": "quadrature_refinement",
        }[dataset]
        derivative_check = _component_check(
            frozen_uncertainty,
            derivative_component_name,
            10 * max(observed_derivative_changes),
        )
        quadrature_check = (
            _component_check(
                frozen_uncertainty,
                quadrature_component_name,
                None
                if observed_quadrature_changes is None
                else 10 * max(observed_quadrature_changes),
            )
            if quadrature_component_name
            else None
        )
        output_values = tuple(complex(value) for value in current)
        frozen_values = reference[name]["values_si"]
        value_match = [
            [value.real, value.imag] for value in output_values
        ] == frozen_values
        return {
            "status": "numerical",
            "route": "mpmath",
            "evaluation_inputs": {
                "h_m": _capture_number(h, high_dps),
                "2h_m": _capture_number(2 * h, high_dps),
                "4h_m": _capture_number(4 * h, high_dps),
                "high_digits": high_dps,
                "low_digits": low_dps if low_candidates else None,
                "quadrature_orders": [coarse_order, middle_order, high_order]
                if quadrature_applicable
                else None,
            },
            "derivative_differences_si": None
            if derivative_changes is None
            else {
                "4h_to_2h": _exact_mp_string(derivative_changes[0]),
                "2h_to_h": _exact_mp_string(derivative_changes[1]),
                "ten_max": _exact_mp_string(10 * max(derivative_changes)),
            },
            "quadrature_differences_si": None
            if quadrature_changes is None
            else {
                "coarse_to_middle": _exact_mp_string(quadrature_changes[0]),
                "middle_to_fine": _exact_mp_string(quadrature_changes[1]),
                "ten_max": _exact_mp_string(10 * max(quadrature_changes)),
            },
            "fixed_order_precision_change_si": None
            if precision_change is None
            else _exact_mp_string(precision_change),
            "observed_inapplicable_quadrature_component_si": (
                None
                if quadrature_changes is not None or observed_quadrature_changes is None
                else {
                    "coarse_to_middle": _exact_mp_string(
                        observed_quadrature_changes[0]
                    ),
                    "middle_to_fine": _exact_mp_string(observed_quadrature_changes[1]),
                    "ten_max": _exact_mp_string(10 * max(observed_quadrature_changes)),
                }
            ),
            "independent_floor_components_si": {
                key: _exact_mp_string(value) for key, value in floors.items()
            },
            "floor_basis": (
                "Only the explicitly allowlisted precision-propagation, "
                "arithmetic/root-roundoff, and float-reporting components may "
                "explain a trend. Fixed truncation bias and potential/"
                "normalization crosschecks are excluded."
            ),
            "independent_floor_component_reasons": _floor_basis_metadata(floors),
            "derivative_trend": derivative_trend,
            "quadrature_trend": quadrature_trend,
            "frozen_component_checks": {
                "derivative": derivative_check,
                "quadrature": quadrature_check,
            },
            "captured_final_values_match_frozen": value_match,
            "captured_final_values_si": _capture_number(current, high_dps),
        }


def _longdouble_norm(values, surface_module):
    return surface_module.norm_ld(np.asarray(values))


def _surface_trace_for_observable(dataset, name, captures, reference, metadata):
    method = reference[name]["uncertainty"]["method"]
    certified = reference[name]["uncertainty"]["uncertainty_kind"] == "rigorous_bound"
    if certified:
        return {
            "status": "certified_override",
            "certificate": method,
            "derivative_trend": {
                "classification": "not_applicable",
                "reason": "the frozen output is an exact symmetry/parity certificate",
            },
            "quadrature_trend": {
                "classification": "not_applicable",
                "reason": "the frozen output is an exact symmetry/parity certificate",
            },
            "captured_evaluations_were_used_for_this_observable": False,
        }
    high_precision_route = (
        method == "high precision Newton surface / independent refinement"
    )
    surface_module = oracle_module("surface_reference")
    section = OBSERVABLE_SLICES[name]
    if high_precision_route:
        events = [
            event
            for event in captures
            if event["kernel"] == "surface_high_precision_evaluate"
        ]
        by_digits_order = {
            (event["_raw_inputs"]["digits"], event["_raw_inputs"]["order"]): event[
                "_raw_result"
            ]
            for event in events
        }
        selected = [by_digits_order[(50, order)] for order in (16, 24, 32)]
        final80 = by_digits_order[(80, 32)]
        with mp.workdps(80):
            vectors = [tuple(item[0][section]) for item in selected]
            high_precision = oracle_module("surface_high_precision")
            # Keep slices as Python sequences so the original mpmath norm is
            # used at the frozen 80-digit arithmetic context.
            vectors = [tuple(item[0][section]) for item in selected]
            diff01 = high_precision.vector_norm(
                tuple(a - b for a, b in zip(vectors[1], vectors[0], strict=True))
            )
            diff12 = high_precision.vector_norm(
                tuple(a - b for a, b in zip(vectors[2], vectors[1], strict=True))
            )
            quadrature_value = 10 * max(diff01, diff12)
            final_values = tuple(final80[0][section])
            precision = high_precision.vector_norm(
                tuple(a - b for a, b in zip(final_values, vectors[2], strict=True))
            )
            floor = _floor_for(
                _floor_candidates(reference[name]["uncertainty"], observable=name)
            )
            frozen_component = dict(reference[name]["uncertainty"]["components"])[
                "high_precision_surface_quadrature"
            ]
            final_for_conversion = tuple(complex(value) for value in final_values)
            matches = [
                [value.real, value.imag] for value in final_for_conversion
            ] == reference[name]["values_si"]
            return {
                "status": "numerical",
                "route": "mpmath_fallback",
                "evaluation_inputs": {
                    "quadrature_orders_at_50_digits": [16, 24, 32],
                    "final_order_at_80_digits": 32,
                    "digits": [50, 80],
                },
                "quadrature_differences_si": {
                    "coarse_to_middle": _exact_mp_string(diff01),
                    "middle_to_fine": _exact_mp_string(diff12),
                    "ten_max": _exact_mp_string(quadrature_value),
                },
                "fixed_order_precision_change_si": _exact_mp_string(precision),
                "independent_floor_components_si": {
                    key: _exact_mp_string(value)
                    for key, value in _floor_candidates(
                        reference[name]["uncertainty"], observable=name
                    ).items()
                },
                "floor_basis": (
                    "Only the explicitly allowlisted precision-propagation, "
                    "arithmetic/root-roundoff, and float-reporting components may "
                    "explain a trend. Fixed truncation bias and potential/"
                    "normalization crosschecks are excluded."
                ),
                "independent_floor_component_reasons": _floor_basis_metadata(
                    _floor_candidates(reference[name]["uncertainty"], observable=name)
                ),
                "derivative_trend": {
                    "classification": "not_applicable",
                    "reason": (
                        "surface values use spatial quadrature, not target "
                        "finite differences"
                    ),
                },
                "quadrature_trend": _change_trend(diff01, diff12, floor),
                "frozen_component_checks": {
                    "quadrature": {
                        "component_name": "high_precision_surface_quadrature",
                        "formula_float_si": float(quadrature_value),
                        "frozen_component_si": frozen_component,
                        "matches_frozen_float": frozen_component
                        == float(quadrature_value),
                    }
                },
                "captured_final_values_match_frozen": matches,
                "captured_final_values_si": _capture_number(final_values, 80),
            }

    events = [
        event for event in captures if event["kernel"] == "surface_newton_evaluate"
    ]
    final_orders = tuple(metadata["orders"])
    by_order = {}
    for order in final_orders:
        matching = [event for event in events if event["_raw_inputs"]["order"] == order]
        if not matching:
            raise RuntimeError(
                f"missing ordinary surface order {order} for {dataset}/{name}"
            )
        by_order[order] = matching[-1]["_raw_result"]
    coarse, middle, high = (by_order[order][0] for order in final_orders)
    values = high[section]
    q0 = _longdouble_norm((middle - coarse)[section], surface_module)
    q1 = _longdouble_norm((high - middle)[section], surface_module)
    quadrature = np.longdouble(10) * max(q0, q1)
    floors = _floor_candidates(reference[name]["uncertainty"], observable=name)
    floor = _floor_for(floors)
    component_name = "surface_quadrature_refinement"
    frozen_component = dict(reference[name]["uncertainty"]["components"])[
        component_name
    ]
    converted = tuple(complex(value) for value in values)
    matches = [[value.real, value.imag] for value in converted] == reference[name][
        "values_si"
    ]
    return {
        "status": "numerical",
        "route": "numpy.longdouble",
        "evaluation_inputs": {
            "quadrature_orders": list(final_orders),
            "backend": "numpy.longdouble",
        },
        "quadrature_differences_si": {
            "coarse_to_middle": _capture_number(q0, 40),
            "middle_to_fine": _capture_number(q1, 40),
            "ten_max": _capture_number(quadrature, 40),
        },
        "derivative_differences_si": None,
        "floor_components_si": {
            key: _exact_mp_string(value) for key, value in floors.items()
        },
        "floor_basis": (
            "Only the explicitly allowlisted precision-propagation, "
            "arithmetic/root-roundoff, and float-reporting components may "
            "explain a trend. Fixed truncation bias and potential/"
            "normalization crosschecks are excluded."
        ),
        "independent_floor_component_reasons": _floor_basis_metadata(floors),
        "derivative_trend": {
            "classification": "not_applicable",
            "reason": (
                "surface values use spatial quadrature, not target finite differences"
            ),
        },
        "quadrature_trend": _change_trend(q0, q1, floor),
        "frozen_component_checks": {
            "quadrature": {
                "component_name": component_name,
                "formula_float_si": float(quadrature),
                "frozen_component_si": frozen_component,
                "matches_frozen_float": frozen_component == float(quadrature),
            }
        },
        "captured_final_values_match_frozen": matches,
        "captured_final_values_si": _capture_number(values, 40),
    }


def _source_surface_trace(name, captures, reference):
    if reference[name]["uncertainty"]["uncertainty_kind"] == "rigorous_bound":
        return {
            "status": "certified_override",
            "certificate": reference[name]["uncertainty"]["method"],
            "derivative_trend": {
                "classification": "not_applicable",
                "reason": "the frozen output is an exact inversion-parity certificate",
            },
            "quadrature_trend": {
                "classification": "not_applicable",
                "reason": "the frozen output is an exact inversion-parity certificate",
            },
            "captured_evaluations_were_used_for_this_observable": False,
        }
    events = [event for event in captures if event["kernel"] == "source_surface_values"]
    by_order = {event["_raw_inputs"]["order"]: event["_raw_result"] for event in events}
    coarse, middle, high = (by_order[order][0] for order in (16, 24, 32))
    section = OBSERVABLE_SLICES[name]
    surface_module = oracle_module("surface_reference")
    q0 = _longdouble_norm((middle - coarse)[section], surface_module)
    q1 = _longdouble_norm((high - middle)[section], surface_module)
    q = np.longdouble(10) * max(q0, q1)
    uncertainty = reference[name]["uncertainty"]
    frozen_component = dict(uncertainty["components"])["surface_quadrature_refinement"]
    values = high[section]
    converted = tuple(complex(value) for value in values)
    matches = [[item.real, item.imag] for item in converted] == reference[name][
        "values_si"
    ]
    floors = _floor_candidates(uncertainty, observable=name)
    floor = _floor_for(floors)
    return {
        "status": "numerical",
        "route": "numpy.longdouble",
        "evaluation_inputs": {
            "quadrature_orders": [16, 24, 32],
            "backend": "numpy.longdouble",
        },
        "quadrature_differences_si": {
            "coarse_to_middle": _capture_number(q0, 40),
            "middle_to_fine": _capture_number(q1, 40),
            "ten_max": _capture_number(q, 40),
        },
        "derivative_trend": {
            "classification": "not_applicable",
            "reason": (
                "finite-source surface references have no target "
                "finite-difference refinement"
            ),
        },
        "independent_floor_components_si": {
            key: _exact_mp_string(value) for key, value in floors.items()
        },
        "floor_basis": (
            "Only the explicitly allowlisted precision-propagation, "
            "arithmetic/root-roundoff, and float-reporting components may "
            "explain a trend. Fixed truncation bias and potential/"
            "normalization crosschecks are excluded."
        ),
        "independent_floor_component_reasons": _floor_basis_metadata(floors),
        "quadrature_trend": _change_trend(q0, q1, floor),
        "frozen_component_checks": {
            "quadrature": {
                "component_name": "surface_quadrature_refinement",
                "formula_float_si": float(q),
                "frozen_component_si": frozen_component,
                "matches_frozen_float": frozen_component == float(q),
            }
        },
        "captured_final_values_match_frozen": matches,
        "captured_final_values_si": _capture_number(values, 40),
    }


def _case_trace(
    dataset,
    row,
    captures,
    output,
    metadata,
    *,
    source_row_index=None,
    zero_certificates=(),
    zero_certificate_source=None,
    zero_certificate_sha256=None,
    polynomial_certificates=(),
    polynomial_certificate_source=None,
    polynomial_certificate_sha256=None,
    polynomial_verifier_script_sha256=None,
):
    actual_reference, actual_metadata = _jsonable(output), _jsonable(metadata)
    reference = row["reference"]
    frozen_metadata = row["metadata"]
    traces = {}
    family = row["fixture"].get("family") if isinstance(row["fixture"], dict) else None
    if dataset == "high_sphere" and metadata.get("exact_zero_certificate"):
        for name in OBSERVABLE_SLICES:
            traces[name] = {
                "status": "exact_whole_case_certificate",
                "certificate": metadata["exact_zero_certificate"],
                "derivative_trend": {
                    "classification": "not_applicable",
                    "reason": "no observable evaluation occurs for this certified row",
                },
                "quadrature_trend": {
                    "classification": "not_applicable",
                    "reason": "no observable evaluation occurs for this certified row",
                },
            }
    elif dataset in {"cuboid", "cylinder"}:
        traces = {
            name: _surface_trace_for_observable(
                dataset, name, captures, reference, metadata
            )
            for name in OBSERVABLE_SLICES
        }
    elif (
        dataset == "source"
        and row["fixture"].get("family") != "temperature_grid"
        and row["fixture"].get("shape") != "sphere"
    ):
        traces = {
            name: _source_surface_trace(name, captures, reference)
            for name in OBSERVABLE_SLICES
        }
    else:
        metric_metadata = dict(metadata)
        if dataset in {"source", "noncavity"}:
            metric_metadata["fixture_family"] = family
        for name in OBSERVABLE_SLICES:
            uncertainty = reference[name]["uncertainty"]
            if uncertainty.get("uncertainty_kind") == "rigorous_bound":
                traces[name] = {
                    "status": "certified_override",
                    "certificate": uncertainty["method"],
                    "derivative_trend": {
                        "classification": "not_applicable",
                        "reason": (
                            "the frozen observable is an exact "
                            "symmetry/parity certificate"
                        ),
                    },
                    "quadrature_trend": {
                        "classification": "not_applicable",
                        "reason": (
                            "the frozen observable is an exact "
                            "symmetry/parity certificate"
                        ),
                    },
                    "captured_evaluations_were_used_for_this_observable": False,
                }
            else:
                traces[name] = _mp_trace_for_observable(
                    dataset, name, captures, reference, metric_metadata
                )
        if dataset == "source" and family == "temperature_grid":
            for trace in traces.values():
                trace["quadrature_trend"] = {
                    "classification": "not_applicable",
                    "nondecreasing": None,
                    "reason": (
                        "the original route is an explicit finite mass sum "
                        "with no spatial quadrature"
                    ),
                }
                trace["quadrature_differences_si"] = None

    if actual_reference == reference and actual_metadata == frozen_metadata:
        for name, trace in traces.items():
            certificate = _matching_zero_certificate(
                dataset, row["fixture"], name, reference[name], zero_certificates
            )
            if certificate is None:
                continue
            raw_classifications = {
                key: trace.get(key, {}).get("classification")
                for key in ("derivative_trend", "quadrature_trend")
            }
            adjudicated = []
            for key in ("derivative_trend", "quadrature_trend"):
                trend = trace.get(key)
                if (
                    isinstance(trend, dict)
                    and trend.get("classification") == "unresolved_zero_change"
                ):
                    trend["certificate_adjudication"] = (
                        "exact_zero_certificate_for_this_frozen_observable"
                    )
                    adjudicated.append(key)
            trace["zero_certificate_adjudication"] = {
                "dataset": certificate["dataset"],
                "fixture": certificate["fixture"],
                "observable": certificate["observable"],
                "certificate": certificate["certificate"],
                "supplement_path": zero_certificate_source,
                "supplement_sha256": zero_certificate_sha256,
                "original_uncertainty_kind": certificate["original_uncertainty_kind"],
                "original_used_uncertainty_si": certificate[
                    "original_used_uncertainty_si"
                ],
                "raw_trend_classifications": raw_classifications,
                "adjudicated_zero_change_trends": adjudicated,
                "frozen_estimate_and_values_unchanged": certificate[
                    "frozen_labels_and_values_unchanged"
                ],
            }

        if source_row_index is not None:
            for name, trace in traces.items():
                if (
                    trace.get("quadrature_trend", {}).get("classification")
                    != "unresolved_zero_change"
                ):
                    continue
                certificate = _matching_polynomial_certificate(
                    dataset,
                    source_row_index,
                    row["fixture"],
                    name,
                    polynomial_certificates,
                )
                if certificate is None:
                    continue
                trace["polynomial_quadrature_certificate_adjudication"] = {
                    "dataset": dataset,
                    "source_row_index": source_row_index,
                    "fixture": row["fixture"],
                    "observable": name,
                    "certificate_kind": certificate["certificate_kind"],
                    "certified_error": certificate["certified_error"],
                    "scope": "ideal polynomial quadrature truncation only",
                    "temporal_quadrature_remains_applicable": True,
                    "total_numerical_error_certified": False,
                    "raw_quadrature_trend_classification": "unresolved_zero_change",
                    "supplement_path": polynomial_certificate_source,
                    "supplement_sha256": polynomial_certificate_sha256,
                    "verifier_script_sha256": polynomial_verifier_script_sha256,
                    "frozen_values_uncertainty_labels_unchanged": certificate[
                        "frozen_values_uncertainty_labels_unchanged"
                    ],
                }

    failed_checks = []
    for name, trace in traces.items():
        if trace.get("status") == "numerical":
            if not trace.get("captured_final_values_match_frozen", False):
                failed_checks.append(
                    f"{name}: captured values differ from frozen values"
                )
            for check in trace.get("frozen_component_checks", {}).values():
                if check is not None and not check["matches_frozen_float"]:
                    failed_checks.append(
                        f"{name}: {check['component_name']} differs from "
                        "frozen component"
                    )
    output_match = actual_reference == reference
    metadata_match = actual_metadata == frozen_metadata
    if not output_match:
        failed_checks.append(
            "actual original-entrypoint values/uncertainties differ from "
            "frozen reference"
        )
    if not metadata_match:
        failed_checks.append(
            "actual original-entrypoint metadata differ from frozen metadata"
        )
    raw_unresolved = [
        f"{name}/{key}"
        for name, trace in traces.items()
        for key in ("derivative_trend", "quadrature_trend")
        if isinstance(trace.get(key), dict)
        and str(trace[key].get("classification", "")).startswith("unresolved")
    ]
    unresolved = [
        item
        for item in raw_unresolved
        if not (
            item.split("/", 1)[1]
            in traces[item.split("/", 1)[0]]
            .get("zero_certificate_adjudication", {})
            .get("adjudicated_zero_change_trends", [])
            or (
                item.split("/", 1)[1] == "quadrature_trend"
                and traces[item.split("/", 1)[0]]
                .get("polynomial_quadrature_certificate_adjudication", {})
                .get("raw_quadrature_trend_classification")
                == "unresolved_zero_change"
            )
        )
    ]
    status = (
        "replay_mismatch"
        if failed_checks
        else "unresolved_trend"
        if unresolved
        else "reproduced"
    )
    return {
        "schema": SCHEMA,
        "dataset": dataset,
        "source_row_index": source_row_index,
        "fixture": row["fixture"],
        "original_entrypoint": {
            "actual_reference": actual_reference,
            "actual_metadata": actual_metadata,
            "capture_call_count": len(captures),
            "captured_calls": [
                {
                    key: event[key]
                    for key in ("kernel", "inputs", "output")
                    if key in event
                }
                for event in captures
            ],
        },
        "frozen_reference": {"reference": reference, "metadata": frozen_metadata},
        "reproduction": {
            "reference_matches_full_values_uncertainty_methods_components": (
                output_match
            ),
            "metadata_matches": metadata_match,
            "passed": output_match and metadata_match and not failed_checks,
            "failed_checks": failed_checks,
        },
        "observable_traces": traces,
        "raw_unresolved_trends": raw_unresolved,
        "unresolved_trends": unresolved,
        "zero_certificate_supplement_sha256": zero_certificate_sha256,
        "polynomial_certificate_supplement_sha256": (polynomial_certificate_sha256),
        "status": status,
    }


def record_row(
    dataset,
    row,
    *,
    source_row_index=None,
    zero_certificates=(),
    zero_certificate_source=None,
    zero_certificate_sha256=None,
    polynomial_certificates=(),
    polynomial_certificate_source=None,
    polynomial_certificate_sha256=None,
    polynomial_verifier_script_sha256=None,
):
    """Run one original reference and emit its additive trace record."""
    captures = []
    try:
        with capture_context(dataset, captures):
            output, metadata = _invoke_original_entrypoint(dataset, row["fixture"])
        return _case_trace(
            dataset,
            row,
            captures,
            output,
            metadata,
            source_row_index=source_row_index,
            zero_certificates=zero_certificates,
            zero_certificate_source=zero_certificate_source,
            zero_certificate_sha256=zero_certificate_sha256,
            polynomial_certificates=polynomial_certificates,
            polynomial_certificate_source=polynomial_certificate_source,
            polynomial_certificate_sha256=polynomial_certificate_sha256,
            polynomial_verifier_script_sha256=polynomial_verifier_script_sha256,
        )
    except Exception as error:
        return {
            "schema": SCHEMA,
            "dataset": dataset,
            "source_row_index": source_row_index,
            "fixture": row["fixture"],
            "frozen_reference": {
                "reference": row.get("reference"),
                "metadata": row.get("metadata"),
            },
            "captured_calls_before_error": [
                {
                    key: event[key]
                    for key in ("kernel", "inputs", "output", "_error")
                    if key in event
                }
                for event in captures
            ],
            "capture_call_count_before_error": len(captures),
            "status": "replay_error",
            "error": f"{type(error).__name__}: {error}",
            "exception": _exception_details(error),
            "zero_certificate_supplement_sha256": zero_certificate_sha256,
            "polynomial_certificate_supplement_sha256": (polynomial_certificate_sha256),
        }


def _dataset_paths(l6_root, dataset):
    directory = l6_root / DATASETS[dataset]
    return directory / "references.jsonl", directory / "manifest.json"


def _read_rows(path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                yield (
                    line_number - 1,
                    _strict_json_loads(
                        line,
                        description=f"frozen reference row {line_number} in {path}",
                    ),
                )


def _module_hashes():
    hashes = {name: _sha256(ORACLE_DIRECTORY / name) for name in EXPECTED_ORACLE_SHA256}
    mismatches = {
        name: {"expected": expected, "actual": hashes[name]}
        for name, expected in EXPECTED_ORACLE_SHA256.items()
        if hashes[name] != expected
    }
    if mismatches:
        raise RuntimeError(f"pinned independent oracle modules changed: {mismatches}")
    return hashes


def _loaded_production_modules():
    """Return production package modules currently loaded in this process."""
    return sorted(
        name for name in sys.modules if name == "gragra" or name.startswith("gragra.")
    )


def _process_independence(modules):
    names = sorted(set(modules))
    return {
        "production_imported": bool(names),
        "production_module_names": names,
        "independence_check_passed": not names,
    }


def _manifest_pin_sha256(manifest):
    """Hash immutable replay inputs while leaving observed process state mutable."""
    pinned = {key: value for key, value in manifest.items() if key != "process_state"}
    return _sha256_for_json(pinned)


def _worker(payload):
    dataset, index, row, *certificate_payload = payload
    if not certificate_payload:
        zero_certificates, certificate_source, certificate_sha256 = (), None, None
        polynomial_certificates = ()
        polynomial_certificate_source = None
        polynomial_certificate_sha256 = None
        polynomial_verifier_script_sha256 = None
    elif len(certificate_payload) == 3:
        zero_certificates, certificate_source, certificate_sha256 = certificate_payload
        polynomial_certificates = ()
        polynomial_certificate_source = None
        polynomial_certificate_sha256 = None
        polynomial_verifier_script_sha256 = None
    elif len(certificate_payload) == 7:
        (
            zero_certificates,
            certificate_source,
            certificate_sha256,
            polynomial_certificates,
            polynomial_certificate_source,
            polynomial_certificate_sha256,
            polynomial_verifier_script_sha256,
        ) = certificate_payload
    else:
        raise ValueError("worker received an invalid certificate payload")
    started = time.monotonic()
    modules_before = _loaded_production_modules()
    try:
        record = record_row(
            dataset,
            row,
            source_row_index=index,
            zero_certificates=zero_certificates,
            zero_certificate_source=certificate_source,
            zero_certificate_sha256=certificate_sha256,
            polynomial_certificates=polynomial_certificates,
            polynomial_certificate_source=polynomial_certificate_source,
            polynomial_certificate_sha256=polynomial_certificate_sha256,
            polynomial_verifier_script_sha256=polynomial_verifier_script_sha256,
        )
        record["source_row_index"] = index
        record["elapsed_seconds"] = time.monotonic() - started
    except Exception as error:
        record = {
            "schema": SCHEMA,
            "dataset": dataset,
            "source_row_index": index,
            "fixture": row.get("fixture"),
            "frozen_reference": {
                "reference": row.get("reference"),
                "metadata": row.get("metadata"),
            },
            "status": "replay_error",
            "error": f"{type(error).__name__}: {error}",
            "exception": _exception_details(error),
            "captured_calls_before_error": [],
            "capture_call_count_before_error": 0,
            "zero_certificate_supplement_sha256": certificate_sha256,
            "polynomial_certificate_supplement_sha256": (polynomial_certificate_sha256),
        }
    modules_after = _loaded_production_modules()
    process_state = _process_independence(modules_before + modules_after)
    record["process_independence"] = {
        "pid": os.getpid(),
        "modules_before": modules_before,
        "modules_after": modules_after,
        **process_state,
    }
    error = record.get("error") if record.get("status") == "replay_error" else None
    return index, record, error


def _dataset_manifest(
    dataset,
    reference_path,
    upstream_manifest_path,
    rows,
    selected_indices,
    observer_sha256,
    oracle_hashes,
    parent_modules,
    zero_certificate,
    polynomial_certificate,
):
    actual_reference_hash = _sha256(reference_path)
    if actual_reference_hash != EXPECTED_REFERENCE_SHA256[dataset]:
        raise RuntimeError(
            f"accepted {dataset} reference SHA mismatch: expected "
            f"{EXPECTED_REFERENCE_SHA256[dataset]}, got {actual_reference_hash}"
        )
    manifest_value = _strict_json_file(upstream_manifest_path)
    upstream = {
        "manifest": str(upstream_manifest_path),
        "manifest_sha256": _sha256(upstream_manifest_path),
        "manifest_contents": manifest_value,
    }
    process_state = _process_independence(parent_modules)
    return {
        "schema": "gragra.gen1.refinement-trace-dataset-manifest.v4",
        "dataset": dataset,
        "frozen_reference_path": str(reference_path),
        "frozen_reference_sha256": actual_reference_hash,
        "upstream_provenance": upstream,
        "observer_sha256": observer_sha256,
        "oracle_module_sha256": oracle_hashes,
        "zero_certificate_sha256": zero_certificate["sha256"],
        "zero_certificate_supplement": {
            "path": zero_certificate["path"],
            "sha256": zero_certificate["sha256"],
            "classification": zero_certificate["data"]["classification"],
        },
        "polynomial_certificate_sha256": (
            None if polynomial_certificate is None else polynomial_certificate["sha256"]
        ),
        "polynomial_quadrature_certificate_supplement": (
            None
            if polynomial_certificate is None
            else {
                key: polynomial_certificate[key]
                for key in (
                    "path",
                    "sha256",
                    "source_certificate_sha256",
                    "report_sha256",
                    "verifier_script_sha256",
                    "scope",
                )
            }
        ),
        "case_count_total": len(rows),
        "selected_source_row_indices": selected_indices,
        "selected_case_count": len(selected_indices),
        "source_tree": str(REPOSITORY),
        "process_state": {
            "scope": "parent process at dataset start",
            "parent_modules_at_start": sorted(set(parent_modules)),
            **process_state,
        },
        "frozen_reference_modified": False,
        "limits_changed": False,
        "workers_maximum": 2,
    }


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _append_jsonl(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _run_environment():
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "mpmath_version": mp.__version__,
        "numpy_version": np.__version__,
    }


def _validate_resume_pins(
    checkpoint,
    manifest,
    *,
    reference_sha256,
    observer_sha256,
    auditor_sha256,
    oracle_hashes,
    zero_certificate_sha256,
    polynomial_certificate_sha256=None,
):
    if manifest.get("schema") != "gragra.gen1.refinement-trace-dataset-manifest.v4":
        raise RuntimeError("dataset manifest schema is not supported by observer v4")
    if checkpoint.get("schema") != "gragra.gen1.refinement-trace-checkpoint.v4":
        raise RuntimeError("checkpoint schema is not supported by observer v4")
    expected = {
        "frozen_reference_sha256": reference_sha256,
        "observer_sha256": observer_sha256,
        "auditor_sha256": auditor_sha256,
        "oracle_module_sha256": oracle_hashes,
        "zero_certificate_sha256": zero_certificate_sha256,
        "polynomial_certificate_sha256": polynomial_certificate_sha256,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise RuntimeError(f"checkpoint {key} changed; use a new run id")
        if manifest.get(key) != value:
            raise RuntimeError(f"dataset manifest {key} changed; use a new run id")
    manifest_pin = _manifest_pin_sha256(manifest)
    if checkpoint.get("dataset_manifest_pin_sha256") != manifest_pin:
        raise RuntimeError("dataset manifest input pins changed; use a new run id")
    upstream = manifest["upstream_provenance"]
    if _sha256(Path(upstream["manifest"])) != upstream["manifest_sha256"]:
        raise RuntimeError("upstream accepted manifest changed; use a new run id")


def _sha256_for_json(value):
    return hashlib.sha256(
        (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        )
    ).hexdigest()


ATTEMPT_SUCCESS_SCHEMA = "gragra.gen1.refinement-trace-attempt-success.v4"
ATTEMPT_FAILURE_SCHEMA = "gragra.gen1.refinement-trace-attempt-failure.v4"
CHECKPOINT_SCHEMA = "gragra.gen1.refinement-trace-checkpoint.v4"


def _exception_details(error):
    return {
        "type": type(error).__name__,
        "module": type(error).__module__,
        "message": str(error),
        "traceback": "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        ),
    }


def _event_digest(event):
    return _sha256_bytes(
        _canonical_json(
            {key: value for key, value in event.items() if key != "event_sha256"}
        ).encode("utf-8")
    )


class AttemptPersistenceError(RuntimeError):
    """Fatal persistence error retaining the outcome that could not commit."""

    def __init__(self, event):
        self.event = event
        super().__init__(
            f"cannot durably persist attempt {event['attempt_id']}: "
            f"{event.get('exception', {})}"
        )


class AttemptJournal:
    """Single-writer execution accounting; local digests are not trust anchors.

    Terminal events are append-only. The checkpoint commits their fsynced
    prefix. Interrupted attempt IDs remain in history, never as accepted rows.
    """

    def __init__(
        self, target, *, run_id, dataset, selected_rows, manifest, resume=False
    ):
        self.target = Path(target)
        self.path = self.target / "attempts.jsonl"
        self.checkpoint_path = self.target / "checkpoint.json"
        self.manifest_path = self.target / "manifest.json"
        self.rows = dict(selected_rows)
        self.indices = _validated_selected_indices(sorted(self.rows), self.rows)
        if (
            not self.indices
            or not isinstance(run_id, str)
            or not run_id
            or dataset not in DATASETS
        ):
            raise ValueError(
                "journal requires a run ID, known dataset and selected rows"
            )
        self.run_id, self.dataset = run_id, dataset
        self.manifest = dict(manifest)
        for key, value in {
            "run_id": run_id,
            "dataset": dataset,
            "selected_source_row_indices": self.indices,
            "schema": "gragra.gen1.refinement-trace-dataset-manifest.v4",
        }.items():
            if manifest.get(key) != value:
                raise RuntimeError(f"dataset manifest {key} differs from requested run")
        self.manifest_pin = _manifest_pin_sha256(manifest)
        self.row_pins = {
            str(i): _sha256_bytes(_canonical_json(row).encode())
            for i, row in self.rows.items()
        }
        self.events = []
        self._broken = False
        self._phase_lock_held = False
        if resume:
            # Reject immutable identity changes before opening the journal.
            stored_manifest = _strict_json_file(self.manifest_path)
            checkpoint = _strict_json_file(self.checkpoint_path)
            if _manifest_pin_sha256(stored_manifest) != self.manifest_pin:
                raise RuntimeError("immutable dataset manifest changed")
            expected = {
                "schema": CHECKPOINT_SCHEMA,
                "run_id": run_id,
                "dataset": dataset,
                "dataset_manifest_pin_sha256": self.manifest_pin,
                "selected_source_row_indices": self.indices,
                "selected_row_sha256": self.row_pins,
            }
            for key, value in expected.items():
                if checkpoint.get(key) != value:
                    raise RuntimeError(f"checkpoint {key} changed")
            self.checkpoint = checkpoint
            with self._writer_lock():
                self._reconcile()
        else:
            self.target.mkdir(parents=True, exist_ok=False)
            _atomic_json(self.manifest_path, manifest)
            with self.path.open("xb") as stream:
                stream.flush()
                os.fsync(stream.fileno())
            self.checkpoint = {
                "schema": CHECKPOINT_SCHEMA,
                "run_id": run_id,
                "dataset": dataset,
                "dataset_manifest_pin_sha256": self.manifest_pin,
                "selected_source_row_indices": self.indices,
                "selected_row_sha256": self.row_pins,
                "committed_journal_bytes": 0,
                "committed_journal_sha256": _sha256_bytes(b""),
                "event_sequence": 0,
                "event_digests": [],
                "accepted_source_row_indices": [],
                "failed_source_row_indices": [],
                "active_attempts": {},
                "interrupted_attempts": [],
                "retry_plan": None,
                "retry_plan_history": [],
                "failure_drain_attempt_ids": [],
                "quarantined_uncommitted_tails": [],
                "execution_state": "running",
            }
            _atomic_json(self.checkpoint_path, self.checkpoint)

    @contextlib.contextmanager
    def _writer_lock(self):
        phase_lock = (
            contextlib.nullcontext() if self._phase_lock_held else self.worker_phase()
        )
        with phase_lock, self._checkpoint_lock():
            yield

    @contextlib.contextmanager
    def worker_phase(self):
        """Keep concurrent resume from classifying a live parent as crashed."""
        with (self.target / ".attempt-phase.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(
                    "attempt journal worker phase already has a writer"
                ) from error
            self._phase_lock_held = True
            try:
                yield
            finally:
                self._phase_lock_held = False
                fcntl.flock(lock, fcntl.LOCK_UN)

    @contextlib.contextmanager
    def _checkpoint_lock(self):
        with (self.target / ".attempt-journal.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("attempt journal already has a writer") from error
            try:
                if self._broken:
                    raise RuntimeError(
                        "attempt journal requires resume after persistence failure"
                    )
                if _strict_json_file(self.checkpoint_path) != self.checkpoint:
                    raise RuntimeError(
                        "attempt journal checkpoint advanced by another writer"
                    )
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _save(self, checkpoint):
        _atomic_json(self.checkpoint_path, checkpoint)
        self.checkpoint = checkpoint

    def _row_states(self, events=None):
        accepted, failures = set(), {}
        for event in self.events if events is None else events:
            index = event["source_row_index"]
            if event["schema"] == ATTEMPT_SUCCESS_SCHEMA:
                accepted.add(index)
                failures.pop(index, None)
            else:
                failures[index] = event["attempt_id"]
        return accepted, failures

    def _reconcile(self):
        checkpoint = self.checkpoint
        count, offset = (
            checkpoint.get("event_sequence"),
            checkpoint.get("committed_journal_bytes"),
        )
        if type(count) is not int or count < 0 or type(offset) is not int or offset < 0:
            raise RuntimeError("invalid journal sequence or byte offset")
        data = self.path.read_bytes()
        prefix = data[:offset]
        if len(prefix) != offset or _sha256_bytes(prefix) != checkpoint.get(
            "committed_journal_sha256"
        ):
            raise RuntimeError("committed journal missing or changed")
        events = _decode_trace_bytes(prefix, description="committed attempt journal")
        if count != len(events) or len(checkpoint.get("event_digests", [])) != count:
            raise RuntimeError("checkpoint event sequence mismatch")
        seen, accepted, failures, position = set(), set(), {}, 0
        for sequence, (event, line) in enumerate(
            zip(events, prefix.splitlines(keepends=True), strict=True), 1
        ):
            index, attempt_id = event.get("source_row_index"), event.get("attempt_id")
            if type(index) is not int or index not in self.rows:
                raise RuntimeError("journal has an unselected source row")
            if not isinstance(attempt_id, str) or not attempt_id or attempt_id in seen:
                raise RuntimeError("duplicate or invalid journal attempt ID")
            if (
                type(event.get("event_sequence")) is not int
                or event["event_sequence"] != sequence
            ):
                raise RuntimeError("duplicate or invalid journal event sequence")
            if (
                event.get("run_id") != self.run_id
                or event.get("dataset") != self.dataset
            ):
                raise RuntimeError("journal run identity mismatch")
            if event.get("selected_row_sha256") != self.row_pins[str(index)]:
                raise RuntimeError("journal fixture/source row changed")
            if event.get("last_durable_trace_offset") != position:
                raise RuntimeError("journal event durable offset mismatch")
            if (
                event.get("event_sha256") != _event_digest(event)
                or checkpoint["event_digests"][sequence - 1] != event["event_sha256"]
            ):
                raise RuntimeError("journal event digest mismatch")
            if index in accepted:
                raise RuntimeError("journal retries an accepted successful row")
            if event.get("retry_of") != failures.get(index):
                raise RuntimeError("journal retry does not link to the prior failure")
            self._validate_outcome(event)
            if event["schema"] == ATTEMPT_SUCCESS_SCHEMA:
                accepted.add(index)
                failures.pop(index, None)
            else:
                failures[index] = attempt_id
            seen.add(attempt_id)
            position += len(line)
        if checkpoint.get("accepted_source_row_indices") != sorted(
            accepted
        ) or checkpoint.get("failed_source_row_indices") != sorted(failures):
            raise RuntimeError("checkpoint row state differs from committed journal")
        active, interrupted = (
            checkpoint.get("active_attempts"),
            checkpoint.get("interrupted_attempts"),
        )
        if not isinstance(active, dict) or not isinstance(interrupted, list):
            raise RuntimeError("malformed active/interrupted attempts")
        if any(
            key not in self.row_pins or not isinstance(value, dict)
            for key, value in active.items()
        ):
            raise RuntimeError("malformed active attempt row")
        for attempt in interrupted + [
            dict(value, source_row_index=int(key)) for key, value in active.items()
        ]:
            if not isinstance(attempt, dict):
                raise RuntimeError("malformed interrupted attempt")
            index, attempt_id = (
                attempt.get("source_row_index"),
                attempt.get("attempt_id"),
            )
            if (
                type(index) is not int
                or index not in self.rows
                or not isinstance(attempt_id, str)
                or not attempt_id
                or attempt_id in seen
            ):
                raise RuntimeError(
                    "duplicate or invalid active/interrupted attempt identity"
                )
            seen.add(attempt_id)
        for key, attempt in active.items():
            index = int(key)
            if index in accepted or attempt.get("retry_of") != failures.get(index):
                raise RuntimeError("active attempt row state mismatch")
        plan = checkpoint.get("retry_plan")
        if plan is not None:
            if not isinstance(plan, dict):
                raise RuntimeError("malformed retry plan")
            planned, remaining = (
                plan.get("source_row_indices"),
                plan.get("remaining_source_row_indices"),
            )
            if (
                not isinstance(plan.get("plan_id"), str)
                or not plan["plan_id"]
                or not isinstance(planned, list)
                or not isinstance(remaining, list)
            ):
                raise RuntimeError("malformed retry plan")
            if (
                any(type(i) is not int for i in planned + remaining)
                or planned != sorted(set(planned))
                or remaining != sorted(set(remaining))
                or not set(remaining) <= set(planned) <= set(self.indices)
            ):
                raise RuntimeError("invalid retry plan rows")
            if (
                sorted(failures) != remaining
                or len(active) > 1
                or (active and int(next(iter(active))) != remaining[0])
            ):
                raise RuntimeError(
                    "retry plan does not cover every unresolved failure sequentially"
                )
        drain = checkpoint.get("failure_drain_attempt_ids")
        active_ids = {attempt["attempt_id"] for attempt in active.values()}
        if (
            not isinstance(drain, list)
            or len(drain) != len(set(drain))
            or not set(drain) <= active_ids
        ):
            raise RuntimeError("invalid failure drain plan")
        if failures and active and plan is None and set(drain) != active_ids:
            raise RuntimeError("unplanned failure beside active attempt")
        state = checkpoint.get("execution_state")
        if state not in {"running", "failed", "interrupted", "complete"}:
            raise RuntimeError("unknown execution state")
        if state == "complete" and (
            active or failures or accepted != set(self.indices) or plan is not None
        ):
            raise RuntimeError("complete checkpoint has unfinished attempts")
        if state == "failed" and (active or not failures):
            raise RuntimeError("failed checkpoint has active or no failed attempts")
        if state == "running" and failures and not active and plan is None:
            raise RuntimeError("running checkpoint has an unplanned terminal failure")
        self.events = events
        for key in ("retry_plan_history", "quarantined_uncommitted_tails"):
            if not isinstance(checkpoint.get(key), list) or any(
                not isinstance(item, dict) for item in checkpoint[key]
            ):
                raise RuntimeError(f"malformed checkpoint {key}")
        for entry in checkpoint["quarantined_uncommitted_tails"]:
            name = entry.get("path")
            if (
                not isinstance(name, str)
                or Path(name).name != name
                or not name.startswith("uncommitted-attempt-tail-")
            ):
                raise RuntimeError("invalid journal tail quarantine path")
            tail_path = self.target / name
            if (
                not tail_path.is_file()
                or tail_path.stat().st_size != entry.get("bytes")
                or _sha256(tail_path) != entry.get("sha256")
            ):
                raise RuntimeError("journal tail quarantine changed")
        tail = data[offset:]
        if tail:
            name = f"uncommitted-attempt-tail-{uuid.uuid4().hex}.bin"
            with (self.target / name).open("xb") as stream:
                stream.write(tail)
                stream.flush()
                os.fsync(stream.fileno())
            updated = copy.deepcopy(self.checkpoint)
            updated["quarantined_uncommitted_tails"].append(
                {
                    "path": name,
                    "bytes": len(tail),
                    "sha256": _sha256_bytes(tail),
                    "committed_journal_bytes": offset,
                    "event_sequence": count,
                }
            )
            self._save(updated)
            with self.path.open("r+b") as stream:
                stream.truncate(offset)
                stream.flush()
                os.fsync(stream.fileno())
        if active or state == "running":
            updated = copy.deepcopy(self.checkpoint)
            updated["interrupted_attempts"].extend(
                dict(value, source_row_index=int(key)) for key, value in active.items()
            )
            updated["active_attempts"] = {}
            updated["failure_drain_attempt_ids"] = []
            updated["execution_state"] = (
                "interrupted"
                if active or plan is not None or not failures
                else "failed"
            )
            self._save(updated)

    def _validate_outcome(self, event):
        index = event["source_row_index"]
        if event["schema"] == ATTEMPT_FAILURE_SCHEMA:
            if "record" in event or "success_commitment" in event:
                raise RuntimeError("failure event contains success evidence")
            error = event.get("exception")
            if (
                not isinstance(error, dict)
                or any(
                    not isinstance(error.get(key), str) or not error[key]
                    for key in ("type", "traceback")
                )
                or not isinstance(error.get("message"), str)
            ):
                raise RuntimeError("failure event omits original exception/traceback")
            if not isinstance(event.get("captured_calls_before_error"), list):
                raise RuntimeError("failure event has malformed partial callbacks")
        elif event["schema"] == ATTEMPT_SUCCESS_SCHEMA:
            record = event.get("record")
            if not isinstance(record, dict) or record.get("status") not in {
                "reproduced",
                "unresolved_trend",
                "replay_mismatch",
            }:
                raise RuntimeError("success event has no successful execution result")
            if (
                record.get("schema") != SCHEMA
                or record.get("dataset") != self.dataset
                or record.get("source_row_index") != index
            ):
                raise RuntimeError("success record identity mismatch")
            row = self.rows[index]
            if _canonical_json(record.get("fixture")) != _canonical_json(
                row["fixture"]
            ):
                raise RuntimeError(
                    "success record fixture differs from selected frozen row"
                )
            frozen = {
                "reference": row.get("reference"),
                "metadata": row.get("metadata"),
            }
            if _canonical_json(record.get("frozen_reference")) != _canonical_json(
                frozen
            ):
                raise RuntimeError(
                    "success record frozen payload differs from selected row"
                )
        else:
            raise RuntimeError("unknown attempt event schema")
        _validate_process_independence(event, index, required=True)

    def start(self, index):
        with self._writer_lock():
            accepted, failures = self._row_states()
            if type(index) is not int or index not in self.rows:
                raise ValueError("attempt row is outside the selection")
            if index in accepted:
                raise RuntimeError("cannot retry an accepted successful row")
            if str(index) in self.checkpoint["active_attempts"]:
                raise RuntimeError("row already has an active attempt")
            plan = self.checkpoint["retry_plan"]
            if failures and (
                plan is None
                or not plan["remaining_source_row_indices"]
                or index != plan["remaining_source_row_indices"][0]
                or self.checkpoint["active_attempts"]
            ):
                raise RuntimeError(
                    "failed rows require an explicit sequential retry plan"
                )
            if self.checkpoint["failure_drain_attempt_ids"]:
                raise RuntimeError("cannot dispatch while draining a failure")
            attempt = {
                "attempt_id": uuid.uuid4().hex,
                "retry_of": failures.get(index),
                "source_row_index": index,
            }
            used_ids = {event["attempt_id"] for event in self.events}
            used_ids.update(
                item["attempt_id"]
                for item in self.checkpoint["active_attempts"].values()
            )
            used_ids.update(
                item["attempt_id"] for item in self.checkpoint["interrupted_attempts"]
            )
            if attempt["attempt_id"] in used_ids:
                raise RuntimeError("attempt ID collision; no worker was dispatched")
            updated = copy.deepcopy(self.checkpoint)
            updated["active_attempts"][str(index)] = {
                key: value
                for key, value in attempt.items()
                if key != "source_row_index"
            }
            updated["execution_state"] = "running"
            self._save(updated)
            return attempt

    def finish(self, attempt, result):
        with self._writer_lock():
            existing = self.path.read_bytes()
            if (
                len(existing) != self.checkpoint["committed_journal_bytes"]
                or _sha256_bytes(existing)
                != self.checkpoint["committed_journal_sha256"]
            ):
                raise RuntimeError(
                    "journal changed or has uncommitted bytes; resume required"
                )
            index, record, error = result
            if index != attempt["source_row_index"] or self.checkpoint[
                "active_attempts"
            ].get(str(index)) != {
                key: value
                for key, value in attempt.items()
                if key != "source_row_index"
            }:
                raise RuntimeError("worker result differs from active attempt")
            failed = bool(error) or record.get("status") == "replay_error"
            event = {
                "schema": ATTEMPT_FAILURE_SCHEMA if failed else ATTEMPT_SUCCESS_SCHEMA,
                "run_id": self.run_id,
                "dataset": self.dataset,
                **attempt,
                "event_sequence": self.checkpoint["event_sequence"] + 1,
                "selected_row_sha256": self.row_pins[str(index)],
                "last_durable_trace_offset": self.checkpoint["committed_journal_bytes"],
                "process_independence": record.get("process_independence"),
            }
            if failed:
                event.update(
                    exception=record.get("exception"),
                    captured_calls_before_error=record.get(
                        "captured_calls_before_error", []
                    ),
                )
            else:
                event["record"] = record
            self._validate_outcome(event)
            event["event_sha256"] = _event_digest(event)
            updated = copy.deepcopy(self.checkpoint)
            del updated["active_attempts"][str(index)]
            updated["event_sequence"] += 1
            updated["event_digests"].append(event["event_sha256"])
            accepted, failures = self._row_states(self.events + [event])
            updated["accepted_source_row_indices"] = sorted(accepted)
            updated["failed_source_row_indices"] = sorted(failures)
            if updated["retry_plan"] is not None and not failed:
                updated["retry_plan"]["remaining_source_row_indices"].remove(index)
            updated["failure_drain_attempt_ids"] = (
                sorted(a["attempt_id"] for a in updated["active_attempts"].values())
                if failures
                else []
            )
            updated["execution_state"] = (
                "running"
                if updated["active_attempts"]
                or (updated["retry_plan"] is not None and not failed)
                else "failed"
                if failures
                else "complete"
                if accepted == set(self.indices)
                else "running"
            )
            try:
                _append_jsonl(self.path, event)  # flush + fsync precede checkpoint.
                data = self.path.read_bytes()
                updated["committed_journal_bytes"] = len(data)
                updated["committed_journal_sha256"] = _sha256_bytes(data)
                self._save(updated)
            except BaseException as persistence_error:
                self._broken = True
                raise AttemptPersistenceError(event) from persistence_error
            self.events.append(event)
            return event

    def begin_retry(self):
        with self._writer_lock():
            _, failures = self._row_states()
            if self.checkpoint["active_attempts"]:
                raise RuntimeError("retry requires a drained worker phase")
            updated = copy.deepcopy(self.checkpoint)
            if updated["retry_plan"] is None:
                if not failures:
                    return
                updated["retry_plan"] = {
                    "plan_id": uuid.uuid4().hex,
                    "source_row_indices": sorted(failures),
                    "remaining_source_row_indices": sorted(failures),
                }
            updated["execution_state"] = "running"
            self._save(updated)

    def settle(self, *, interrupted=False):
        with self._writer_lock():
            updated = copy.deepcopy(self.checkpoint)
            accepted, failures = self._row_states()
            if interrupted:
                updated["interrupted_attempts"].extend(
                    dict(value, source_row_index=int(key))
                    for key, value in updated["active_attempts"].items()
                )
                updated["active_attempts"] = {}
            elif updated["active_attempts"]:
                raise RuntimeError("cannot settle with live attempts")
            updated["failure_drain_attempt_ids"] = []
            updated["execution_state"] = (
                "interrupted"
                if interrupted
                else "failed"
                if failures
                else "complete"
                if accepted == set(self.indices)
                else "interrupted"
            )
            if updated["retry_plan"] is not None and not interrupted:
                updated["retry_plan_history"].append(
                    dict(
                        updated["retry_plan"],
                        execution_state=updated["execution_state"],
                    )
                )
                updated["retry_plan"] = None
            self._save(updated)


def _future_failure(index, error):
    modules = _loaded_production_modules()
    return (
        index,
        {
            "status": "replay_error",
            "exception": _exception_details(error),
            "captured_calls_before_error": [],
            "process_independence": {
                "pid": os.getpid(),
                "modules_before": modules,
                "modules_after": modules,
                "observation_scope": "parent process; worker outcome unavailable",
                **_process_independence(modules),
            },
        },
        str(error) or type(error).__name__,
    )


def execute_attempts(
    journal, payload_for_index, *, workers=1, retry_failed=False, worker=None
):
    with journal.worker_phase():
        return _execute_attempt_phase(
            journal,
            payload_for_index,
            workers=workers,
            retry_failed=retry_failed,
            worker=worker,
        )


def _execute_attempt_phase(
    journal, payload_for_index, *, workers=1, retry_failed=False, worker=None
):
    """Bounded dispatch; drain submitted outcomes on the first failure.

    Retry rows run sequentially before unrelated rows. Supplied workers are
    in-process embedding adapters; normal parallel execution uses processes.
    This function makes no semantic or admission verdict.
    """
    if workers not in (1, 2):
        raise ValueError("worker count must be 1 or 2")
    worker = _worker if worker is None else worker

    def invoke(index):
        try:
            return worker(payload_for_index(index))
        except Exception as error:
            return _future_failure(index, error)

    if (
        journal.checkpoint["failed_source_row_indices"]
        or journal.checkpoint["retry_plan"]
    ):
        if not retry_failed:
            return journal.checkpoint
        journal.begin_retry()
    try:
        while (
            journal.checkpoint["retry_plan"]
            and journal.checkpoint["retry_plan"]["remaining_source_row_indices"]
        ):
            index = journal.checkpoint["retry_plan"]["remaining_source_row_indices"][0]
            event = journal.finish(journal.start(index), invoke(index))
            if event["schema"] == ATTEMPT_FAILURE_SCHEMA:
                journal.settle()
                return journal.checkpoint
        if journal.checkpoint["retry_plan"]:
            journal.settle()
        pending = [
            i
            for i in journal.indices
            if i not in journal.checkpoint["accepted_source_row_indices"]
        ]
        if workers == 1 or worker is not _worker:
            for index in pending:
                event = journal.finish(journal.start(index), invoke(index))
                if event["schema"] == ATTEMPT_FAILURE_SCHEMA:
                    break
        elif pending:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
                futures, remaining, stopped = {}, iter(pending), False

                def submit_next():
                    nonlocal stopped
                    if stopped:
                        return
                    index = next(remaining, None)
                    if index is not None:
                        attempt = journal.start(index)
                        try:
                            future = pool.submit(worker, payload_for_index(index))
                        except Exception as error:
                            journal.finish(attempt, _future_failure(index, error))
                            stopped = True
                        else:
                            futures[future] = attempt

                for _ in range(workers):
                    submit_next()
                while futures:
                    done, _ = concurrent.futures.wait(
                        futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for future in done:
                        attempt = futures.pop(future)
                        try:
                            result = future.result()
                        except Exception as error:
                            result = _future_failure(attempt["source_row_index"], error)
                        if (
                            journal.finish(attempt, result)["schema"]
                            == ATTEMPT_FAILURE_SCHEMA
                        ):
                            stopped = True
                    if not stopped:
                        for _ in done:
                            submit_next()
        journal.settle()
    except AttemptPersistenceError:
        # Never accidentally commit more bytes after a persistence failure.
        raise
    except BaseException:
        journal.settle(interrupted=True)
        raise
    return journal.checkpoint


def run_dataset(
    dataset=None,
    l6_root=None,
    output_root=None,
    run_id=None,
    workers=1,
    resume=False,
    indices=None,
    *,
    retry_failed=False,
):
    """Record a dataset's execution history, without a scientific verdict."""
    if dataset is None:
        raise RuntimeError("V4 canonical run orchestration is not implemented")
    if dataset not in DATASETS or workers not in (1, 2):
        raise ValueError("dataset must be known and worker count must be 1 or 2")
    if retry_failed and not resume:
        raise ValueError("--retry-failed requires --resume")
    if (
        not isinstance(run_id, str)
        or not run_id
        or Path(run_id).name != run_id
        or run_id in {".", ".."}
    ):
        raise ValueError("run ID must be a nonempty directory name")
    l6_root, output_root = Path(l6_root), Path(output_root)
    reference_path, upstream_manifest = _dataset_paths(l6_root, dataset)
    reference_hash = _sha256(reference_path)
    if reference_hash != EXPECTED_REFERENCE_SHA256[dataset]:
        raise RuntimeError(f"accepted {dataset} reference SHA mismatch")
    rows = list(_read_rows(reference_path))
    selected_indices = _validated_selected_indices(
        [index for index, _ in rows] if indices is None else indices,
        [index for index, _ in rows],
    )
    selected_rows = {index: row for index, row in rows if index in selected_indices}
    oracle_hashes = _module_hashes()
    observer_hash = _sha256(Path(__file__).resolve())
    auditor_hash = _sha256(
        Path(__file__).with_name("audit_frozen_refinement_traces_v4.py")
    )
    zero = _load_zero_certificates(l6_root)
    polynomial = _load_polynomial_quadrature_certificates(l6_root, dataset)
    manifest = _dataset_manifest(
        dataset,
        reference_path,
        upstream_manifest,
        rows,
        selected_indices,
        observer_hash,
        oracle_hashes,
        _loaded_production_modules(),
        zero,
        polynomial,
    )
    manifest.update(
        run_id=run_id, auditor_sha256=auditor_hash, environment=_run_environment()
    )
    journal = AttemptJournal(
        output_root / run_id / dataset,
        run_id=run_id,
        dataset=dataset,
        selected_rows=selected_rows,
        manifest=manifest,
        resume=resume,
    )

    def payload(index):
        row = selected_rows[index]
        matches = [
            certificate
            for certificate in zero["data"]["certificates"]
            if certificate.get("dataset") == DATASETS[dataset]
            and _canonical_json(certificate.get("fixture"))
            == _canonical_json(row["fixture"])
        ]
        return (
            dataset,
            index,
            row,
            matches,
            zero["path"],
            zero["sha256"],
            () if polynomial is None else polynomial["certificates"],
            None if polynomial is None else polynomial["path"],
            None if polynomial is None else polynomial["sha256"],
            None if polynomial is None else polynomial["verifier_script_sha256"],
        )

    return execute_attempts(
        journal, payload, workers=workers, retry_failed=retry_failed
    )


def _argument_parser():
    """Recorder options for the upcoming canonical runner's explicit retry."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l6-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--datasets", nargs="+", choices=sorted(DATASETS), required=True
    )
    parser.add_argument("--workers", choices=(1, 2), type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true", default=False)
    return parser


def main(_argv=None):
    """Fail closed until the V4 runner and immutable run identity are available."""
    raise RuntimeError("V4 run orchestration is not implemented")


if __name__ == "__main__":
    raise SystemExit(main())
