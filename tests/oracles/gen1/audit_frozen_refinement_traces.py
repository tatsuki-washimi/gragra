"""Audit saved Gen1 refinement traces without rerunning any oracle fields.

Exact serialized callback results are reconciled with named pairs, trends and
frozen uncertainty components. This module does not import the recorder or any
production module. Explicit subset mode never claims the 1178-case gate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import sys
import types
from pathlib import Path

import mpmath as mp
import numpy as np

OBSERVER_SHA256 = "d43be29b355c1f6e4144c396d9f2b576a2972e61d8eff0ea9edd78f38855c446"
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
PACKAGE = "_gen1_independent_trace_audit"

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


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


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


def decode(value):
    """Preserve original MP tuples and platform longdouble values."""
    if isinstance(value, list):
        return tuple(decode(item) for item in value)
    if not isinstance(value, dict):
        return value
    if all(key in value for key in ("shape", "dtype", "values")):
        result = np.asarray(decode(value["values"]), dtype=value["dtype"])
        require(list(result.shape) == value["shape"], "serialized array shape mismatch")
        require(np.all(np.isfinite(result)), "nonfinite serialized array")
        return result
    if "mpf" in value:
        result = mp.mpf(tuple(value["mpf"]))
        require(mp.isfinite(result), "nonfinite MP value")
        # A low-dps callback can display an h retained from the high-dps loop.
        # Never reconstruct its computational input from the shortened display.
        require(mp.isfinite(mp.mpf(value["decimal"])), "nonfinite MP display")
        return result
    if "real" in value and "imag" in value:
        if "dtype" in value:
            return np.clongdouble(decode(value["real"])) + np.clongdouble(
                1j
            ) * np.clongdouble(decode(value["imag"]))
        return mp.mpc(decode(value["real"]), decode(value["imag"]))
    if "dtype" in value and "decimal" in value:
        result = np.dtype(value["dtype"]).type(value["decimal"])
        require(np.isfinite(result), "nonfinite longdouble value")
        return result
    return {key: decode(item) for key, item in value.items()}


def number(value, ld=False):
    if isinstance(value, dict):
        return decode(value)
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


def audit_trend(trend, pair, floor, ld=False):
    require(isinstance(trend, dict), "missing trend")
    if pair is None:
        require(
            trend.get("classification") == "not_applicable",
            "structural N/A trend mismatch",
        )
        require(
            isinstance(trend.get("reason"), str) and trend["reason"],
            "N/A reason missing",
        )
        return "not_applicable"
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
    comparison_floor = np.longdouble(str(floor)) if ld and floor is not None else floor
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
        audit_trend(
            trace["derivative_trend"],
            derivative if derivative_applicable else None,
            floor,
        )
        audit_trend(
            trace["quadrature_trend"], qpair if quadrature_applicable else None, floor
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
    with mp.workdps(80 if hp else 15):
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
            selected = [surface_payload(lookup[50, o], hp=True) for o in (16, 24, 32)]
            payload = surface_payload(lookup[80, 32], hp=True)
            vectors = [item[0][section] for item in selected]
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
            pair = observed_pairs(vectors)
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
        audit_trend(trace["derivative_trend"], None, floor, ld=not hp)
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


def audit_record(dataset, index, row, record, zero=(), polynomial=(), root_records=()):
    require(
        record.get("schema") == "gragra.gen1.refinement-trace-observer.v1",
        "record schema mismatch",
    )
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
            audit_trend(trace["derivative_trend"], None, None)
            audit_trend(trace["quadrature_trend"], None, None)
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
            audit_trend(trace["derivative_trend"], None, None)
            audit_trend(trace["quadrature_trend"], None, None)
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
    return {
        "dataset": dataset,
        "source_row_index": index,
        "status": status,
        "routes": routes,
        "raw_unresolved_trends": raw_unresolved,
        "unresolved_trends": unresolved,
        "audit_unresolved_trends": audit_unresolved,
        "audit_certificate_supplements": [] if supplement is None else [supplement],
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
    data = read_json(path)
    require(
        data["schema"] == "gragra.gen1.cylinder-root-normalization-supplement.v1"
        and data["classification"] == "observed supplemental root-only reconstruction"
        and data["method"]
        == "pinned cylinder_radial_data at 50/80; estimate at 80 digits",
        "root supplement method/scope mismatch",
    )
    require(
        data["generator_sha256"]
        == sha256(ORACLE_DIRECTORY / "record_cylinder_root_normalization.py")
        and data["audit_helper_sha256"] == sha256(__file__)
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


def audit_process_evidence(
    directory,
    manifest,
    checkpoint,
    run,
    records,
    indices,
    observer_sha256=OBSERVER_SHA256,
):
    """Reconcile observations with actual history and trace worker identities."""
    history_path = directory / "run-invocations.jsonl"
    require(
        sha256(history_path) == run["run_invocations_jsonl_sha256"],
        "invocation history digest mismatch",
    )
    history = read_rows(history_path)
    require(
        history == run["invocation_history"] and history,
        "invocation history payload mismatch",
    )
    expected, started, finished = [], {}, set()
    commands = []

    def flags(obj, modules):
        require(
            isinstance(modules, list)
            and all(isinstance(m, str) for m in modules)
            and modules == sorted(set(modules)),
            "malformed module observation",
        )
        require(
            obj.get("production_module_names") == modules
            and obj.get("production_imported") is bool(modules)
            and obj.get("independence_check_passed") is (not bool(modules)),
            "process flags do not derive from observations",
        )
        require(not modules, "production module observation detected")

    for item in history:
        identity, event = item["invocation_id"], item["event"]
        require(isinstance(identity, str) and identity, "invalid invocation identity")
        if event == "started":
            require(identity not in started, "duplicate started invocation")
            started[identity] = item
            modules = item["parent_modules_at_start"]
            require(
                isinstance(item.get("command"), list) and item["command"],
                "missing invocation command",
            )
            commands.append(item["command"])
            require(
                item["selected_source_row_indices"] == indices
                and item["observer_sha256"] == observer_sha256
                and item["oracle_module_sha256"] == EXPECTED_ORACLE_SHA256
                and item["frozen_reference_sha256"]
                == manifest["frozen_reference_sha256"],
                "started invocation input pins mismatch",
            )
        else:
            require(
                event == "finished"
                and identity in started
                and identity not in finished,
                "finished invocation has no unique preceding start",
            )
            finished.add(identity)
            modules = item["parent_modules_at_end"]
        flags(item, modules)
        expected.append(
            {
                "process": "parent",
                "phase": event,
                "invocation_id": identity,
                "modules": modules,
            }
        )
    require(
        history[-1]["event"] == "finished"
        and history[-1]["invocation_id"] == list(started)[-1],
        "latest invocation is not finished",
    )
    require(run["commands"] == commands, "recorded command history mismatch")
    last = history[-1]
    require(
        last["checkpoint_status"] == checkpoint["status"]
        and last["completed_source_row_indices"] == indices
        and last["traces_jsonl_sha256"] == checkpoint["traces_jsonl_sha256"]
        and last["exit_status"]
        == run["exit_status"]
        == (0 if checkpoint["status"] == "complete" else 1),
        "finished invocation summary mismatch",
    )
    for record in records:
        worker = record["process_independence"]
        before, after = worker["modules_before"], worker["modules_after"]
        require(
            isinstance(before, list) and isinstance(after, list),
            "worker observations missing",
        )
        flags(worker, sorted(set(before) | set(after)))
        require(type(worker["pid"]) is int and worker["pid"] > 0, "worker PID invalid")
        expected.append(
            {
                "process": "worker",
                "source_row_index": record["source_row_index"],
                "pid": worker["pid"],
                "modules": sorted(set(before) | set(after)),
            }
        )
    require(
        manifest["process_state"] == run["process_state"],
        "manifest/run process evidence differs",
    )
    process = manifest["process_state"]
    require(
        process["observations"] == expected,
        "parent/worker observation history mismatch",
    )
    observed = sorted({name for item in expected for name in item["modules"]})
    flags(process, observed)
    flags(checkpoint, observed)
    return sorted(set(started) - finished)


def audit_dataset(
    dataset,
    l6_root,
    replay_root,
    zero,
    polynomial,
    allow_selected=False,
    root_records=(),
    observer_sha256=OBSERVER_SHA256,
):
    directory = replay_root / dataset
    reference_path = l6_root / DATASETS[dataset] / "references.jsonl"
    upstream_path = reference_path.parent / "manifest.json"
    require(
        sha256(reference_path) == EXPECTED_REFERENCE_SHA256[dataset],
        "frozen reference digest changed",
    )
    rows = read_rows(reference_path)
    require(len(rows) == COUNTS[dataset], "frozen fixture count changed")
    paths = [
        directory / name
        for name in (
            "manifest.json",
            "checkpoint.json",
            "run-evidence.json",
            "traces.jsonl",
        )
    ]
    before = {
        str(path): sha256(path)
        for path in paths
        + [reference_path, upstream_path, directory / "run-invocations.jsonl"]
    }
    manifest, checkpoint, run = map(read_json, paths[:3])
    require(
        manifest.get("dataset")
        == dataset
        == checkpoint.get("dataset")
        == run.get("dataset")
        and manifest.get("frozen_reference_modified") is False
        and manifest.get("limits_changed") is False,
        "dataset identity or immutable-data flags mismatch",
    )
    manifest_pin = {
        key: value for key, value in manifest.items() if key != "process_state"
    }
    manifest_pin_digest = hashlib.sha256(
        (
            json.dumps(manifest_pin, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode()
    ).hexdigest()
    require(
        checkpoint["dataset_manifest_pin_sha256"] == manifest_pin_digest,
        "immutable dataset manifest pin mismatch",
    )
    records = read_rows(paths[3])
    indices = manifest["selected_source_row_indices"]
    require(
        indices
        and all(type(i) is int and 0 <= i < len(rows) for i in indices)
        and indices == sorted(set(indices)),
        "selection invalid",
    )
    require(
        allow_selected or indices == list(range(len(rows))),
        "full dataset selection required",
    )
    require(
        manifest["case_count_total"] == len(rows)
        and manifest["selected_case_count"] == len(indices),
        "manifest coverage mismatch",
    )
    expected_pins = {
        "observer_sha256": observer_sha256,
        "frozen_reference_sha256": EXPECTED_REFERENCE_SHA256[dataset],
        "oracle_module_sha256": EXPECTED_ORACLE_SHA256,
        "zero_certificate_sha256": ZERO_SHA256,
        "polynomial_certificate_sha256": POLYNOMIAL_SHA256
        if dataset in {"source", "noncavity"}
        else None,
    }
    for obj in (manifest, checkpoint, run):
        for key, value in expected_pins.items():
            if obj is run and key == "zero_certificate_sha256":
                continue  # This run-schema field is pinned by manifest/checkpoint.
            require(obj.get(key) == value, f"{key} provenance mismatch")
    upstream = manifest["upstream_provenance"]
    require(
        upstream["manifest_sha256"] == sha256(upstream_path)
        and upstream["manifest_contents"] == read_json(upstream_path),
        "upstream provenance changed",
    )
    trace_hash = sha256(paths[3])
    trace_size = paths[3].stat().st_size
    require(
        checkpoint["status"] in {"complete", "unresolved_trend", "replay_failure"},
        "dataset is not finished",
    )
    require(
        checkpoint["committed_trace_sha256"]
        == trace_hash
        == checkpoint["traces_jsonl_sha256"]
        == run["traces_jsonl_sha256"],
        "finished trace digest mismatch",
    )
    require(
        checkpoint["committed_trace_bytes"] == trace_size == checkpoint["trace_bytes"],
        "uncommitted bytes in final dataset",
    )
    require(
        checkpoint["selected_source_row_indices"] == indices
        and checkpoint["completed_source_row_indices"] == indices,
        "checkpoint coverage mismatch",
    )
    actual_indices = [r["source_row_index"] for r in records]
    require(sorted(actual_indices) == indices, "duplicate/missing/extra trace rows")
    results, errors = [], []
    for record in records:
        index = record["source_row_index"]
        try:
            results.append(
                audit_record(
                    dataset, index, rows[index], record, zero, polynomial, root_records
                )
            )
        except (ValueError, KeyError, TypeError, IndexError, ArithmeticError) as error:
            errors.append(
                {"source_row_index": index, "error": f"{type(error).__name__}: {error}"}
            )
    original_unresolved = sum(bool(row["unresolved_trends"]) for row in results)
    unresolved = sum(bool(row["audit_unresolved_trends"]) for row in results)
    require(
        checkpoint["case_count"] == len(indices)
        and checkpoint["record_count"] == len(records),
        "checkpoint counts mismatch",
    )
    require(
        checkpoint["unresolved_trend_case_count"]
        == sum(bool(row.get("unresolved_trends")) for row in records),
        "checkpoint unresolved count mismatch",
    )
    recorded_mismatches = sum(
        r.get("status") in {"replay_error", "replay_mismatch"} for r in records
    )
    require(
        checkpoint["mismatch_count"] == recorded_mismatches,
        "checkpoint mismatch count mismatch",
    )
    expected_status = (
        "replay_failure"
        if recorded_mismatches
        else "unresolved_trend"
        if original_unresolved
        else "complete"
    )
    if not errors:
        require(
            checkpoint["status"] == expected_status == run["checkpoint_status"],
            "dataset status inconsistent",
        )
    require(
        run["dataset_manifest_sha256"] == sha256(paths[0]),
        "run manifest digest mismatch",
    )
    for obj in (manifest["process_state"], run["process_state"], checkpoint):
        require(
            obj["production_imported"] is False
            and obj["independence_check_passed"] is True
            and obj["production_module_names"] == [],
            "dataset production imports detected",
        )
    interrupted_invocations = audit_process_evidence(
        directory, manifest, checkpoint, run, records, indices, observer_sha256
    )
    require(
        all(sha256(path) == digest for path, digest in before.items()),
        "input changed during audit",
    )
    return {
        "dataset": dataset,
        "case_count": len(records),
        "expected_case_count": len(rows),
        "full_selection": indices == list(range(len(rows))),
        "valid_records": len(results),
        "unresolved_cases": unresolved,
        "errors": errors,
        "records": results,
        "input_sha256": before,
        "original_unresolved_cases": original_unresolved,
        "checkpoint_status": checkpoint["status"],
        "prior_interrupted_invocation_ids": interrupted_invocations,
        "observer_sha256": observer_sha256,
    }


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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l6-root", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--observer-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root-supplement", type=Path)
    parser.add_argument("--root-supplement-sha256")
    parser.add_argument(
        "--dataset-replay-root", action="append", default=[], metavar="cylinder=PATH"
    )
    parser.add_argument(
        "--dataset-observer-source",
        action="append",
        default=[],
        metavar="cylinder=PATH",
    )
    parser.add_argument(
        "--dataset-observer-sha256",
        action="append",
        default=[],
        metavar="cylinder=SHA256",
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=tuple(COUNTS), default=list(COUNTS)
    )
    parser.add_argument(
        "--allow-selected",
        action="store_true",
        help="bounded checks only; never aggregate closure",
    )
    args = parser.parse_args(argv)
    require(
        not args.output.resolve().is_relative_to(Path(__file__).resolve().parents[3]),
        "audit output must remain outside Git",
    )
    require(
        sha256(args.observer_source) == OBSERVER_SHA256,
        "observer source must remain frozen d43",
    )
    check_oracle_pins()
    require(not production_modules(), "production already imported")
    zero, polynomial = certificate_inputs(args.l6_root)
    replay_overrides, source_overrides, sha_overrides = dataset_overrides(
        args.dataset_replay_root,
        args.dataset_observer_source,
        args.dataset_observer_sha256,
    )
    require(set(replay_overrides).issubset(args.datasets), "unused dataset override")
    require(
        bool(args.root_supplement) == bool(args.root_supplement_sha256),
        "supply both root supplement path and independently pinned SHA",
    )
    root_records = (
        ()
        if args.root_supplement is None
        else load_root_supplement(
            args.root_supplement, args.root_supplement_sha256, args.l6_root
        )
    )
    results, failures = [], []
    for dataset in args.datasets:
        try:
            results.append(
                audit_dataset(
                    dataset,
                    args.l6_root,
                    Path(replay_overrides.get(dataset, args.replay_root)),
                    zero,
                    polynomial,
                    args.allow_selected,
                    root_records,
                    sha_overrides.get(dataset, OBSERVER_SHA256),
                )
            )
        except (
            ValueError,
            KeyError,
            TypeError,
            IndexError,
            OSError,
            ArithmeticError,
        ) as error:
            failures.append(
                {"dataset": dataset, "error": f"{type(error).__name__}: {error}"}
            )
    failed = bool(failures or any(r["errors"] for r in results) or production_modules())
    unresolved = sum(r["unresolved_cases"] for r in results)
    count = sum(r["case_count"] for r in results)
    full_coverage = (
        set(args.datasets) == set(COUNTS)
        and count == sum(COUNTS.values())
        and all(r["full_selection"] for r in results)
        and not args.allow_selected
    )
    outcome = (
        "failed"
        if failed
        else "consistent_unresolved"
        if unresolved
        else "consistent_complete"
        if full_coverage
        else "consistent_subset"
    )
    exit_status = 2 if failed else 1 if unresolved else 0
    output = {
        "schema": "gragra.gen1.independent-refinement-trace-audit.v1",
        "status": outcome,
        "exit_status": exit_status,
        "full_gate_complete": full_coverage and not failed and not unresolved,
        "aggregate_cases": count,
        "expected_aggregate_cases": sum(COUNTS.values()),
        "original_values_uncertainty_labels_unchanged": True,
        "production_modules": production_modules(),
        "datasets": results,
        "dataset_failures": failures,
        "unresolved_cases": unresolved,
        "original_unresolved_cases": sum(
            r["original_unresolved_cases"] for r in results
        ),
        "auditor_sha256": sha256(__file__),
        "observer_sha256": sha256(args.observer_source),
        "dataset_provenance": {
            dataset: {
                "replay_root": str(
                    Path(replay_overrides.get(dataset, args.replay_root)).resolve()
                ),
                "observer_source": str(
                    Path(source_overrides.get(dataset, args.observer_source)).resolve()
                ),
                "observer_sha256": sha_overrides.get(dataset, OBSERVER_SHA256),
                "explicit_replacement_run": dataset in replay_overrides,
            }
            for dataset in args.datasets
        },
        "oracle_sha256": EXPECTED_ORACLE_SHA256,
        "certificate_sha256": {"zero": ZERO_SHA256, "polynomial": POLYNOMIAL_SHA256},
        "root_normalization_supplement": None
        if args.root_supplement is None
        else {
            "path": str(args.root_supplement.resolve()),
            "sha256": args.root_supplement_sha256,
            "classification": (
                "observed supplemental root-only reconstruction; "
                "not original d43 captures"
            ),
        },
        "command": sys.argv,
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "conda": os.environ.get("CONDA_DEFAULT_ENV"),
            "numpy": np.__version__,
            "mpmath": mp.__version__,
            "platform": platform.platform(),
            "longdouble_epsilon": str(np.finfo(np.longdouble).eps),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                key: output[key]
                for key in (
                    "status",
                    "full_gate_complete",
                    "aggregate_cases",
                    "unresolved_cases",
                    "dataset_failures",
                )
            },
            indent=2,
        )
    )
    return exit_status


if __name__ == "__main__":
    raise SystemExit(main())
