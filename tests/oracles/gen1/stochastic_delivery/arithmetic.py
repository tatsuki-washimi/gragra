"""Replay retained serialized arithmetic; never run production or regenerate R."""

from __future__ import annotations

import hashlib
import math
from decimal import Decimal
from fractions import Fraction
from types import ModuleType

from .archive import EvidenceError, decode_json
from .graph import assessment_ratios, key_id

# The unchanged frozen common decision kernel, not a caller-selected executable.
KERNEL_SHA256 = "09dbe97ceaf635b0fd0eafdf2cb3f11ac994af276e318a684c45c00a5548be68"
GENERIC_FAMILIES = {"R1", "B2", "A1", "A3", "T1"}


def load_kernel(store, pin):
    raw = store.read(pin)
    if (
        pin["sha256"] != KERNEL_SHA256
        or hashlib.sha256(raw).hexdigest() != KERNEL_SHA256
    ):
        raise EvidenceError("unknown frozen arithmetic kernel")
    kernel = ModuleType("gen1_archived_arithmetic_kernel")
    exec(compile(raw, "<exact frozen CAS arithmetic kernel>", "exec"), kernel.__dict__)
    return kernel


def jsonl(store, pin):
    raw = store.read(pin)
    if not raw or not raw.endswith(b"\n"):
        raise EvidenceError("retained records require exact LF transport")
    result = {}
    for line in raw.splitlines(keepends=True):
        row = decode_json(line)
        identity = key_id(row["key"])
        if identity in result:
            raise EvidenceError("duplicate retained record key")
        result[identity] = (row, hashlib.sha256(line).hexdigest())
    return result


def _binary64(value, kernel):
    number = float(Decimal(value))
    if not math.isfinite(number) or kernel.decimal_fraction(
        kernel.promote_binary64(number)
    ) != kernel.decimal_fraction(value):
        raise EvidenceError("retained production value is not exact finite binary64")
    return number


def replay_generic_arithmetic(store, stage, *, family):
    """Numeric/structure/unit replay only; admission and transfer stay separate."""
    if family not in GENERIC_FAMILIES:
        raise EvidenceError("unsupported arithmetic profile")
    policy, attempt = store.json(stage["policy"]), store.json(stage["attempt"])
    artifacts = policy["artifacts"]
    kernel = load_kernel(store, artifacts["decision_kernel"])
    fixture, budget, scales = [
        store.json(artifacts[name]) for name in ("fixture", "budget", "source_scales")
    ]
    expected = fixture["expected_keys"] + fixture.get("expected_auxiliary_keys", [])
    ids = [key_id(key) for key in expected]
    if len(ids) != len(set(ids)) or not ids or any(key[0] != family for key in ids):
        raise EvidenceError("invalid frozen arithmetic keyset")
    references = jsonl(store, artifacts["records"])
    comparisons = jsonl(store, attempt["artifacts"]["records"])
    scale_ids = [key_id(row["key"]) for row in scales["rows"]]
    if (
        scales["schema"] != "gen1-stochastic-source-scales-v1"
        or scales["family"] != family
        or len(scale_ids) != len(set(scale_ids))
        or set(scale_ids) != set(ids)
        or set(references) != set(ids)
        or set(comparisons) != set(ids)
    ):
        raise EvidenceError("retained arithmetic coverage differs from exact F")
    source_scales = {key_id(row["key"]): row["value"] for row in scales["rows"]}
    counts = {"science": 0, "auxiliary": 0}
    kinds = {"estimate": 0, "rigorous_bound": 0, "exact_zero": 0}
    claims = {}
    maximum = {"D_over_B": None, "U_over_B": None}
    zero_budget = 0
    x400 = []
    for identity in ids:
        reference, reference_hash = references[identity]
        row = comparisons[identity][0]
        key, quantity = row["key"], row["production_quantity"]
        if (
            row["reference_record_sha256"] != reference_hash
            or row["reference_purpose"] != "formal-science"
            or row["reference_set"] != key["set"]
            or row["status"] != "resolved"
            or row["passed"] is not True
            or reference["status"] != "resolved"
            or reference["quantity"]["shape"] != quantity["shape"]
        ):
            raise EvidenceError("retained reference/production binding differs")
        for value in quantity["data"]:
            _binary64(value["real"], kernel)
            _binary64(value["imag"], kernel)
        g = float(Decimal(reference["inputs"]["g_si"]))
        if not math.isfinite(g) or g <= 0:
            raise EvidenceError("frozen G must be finite and positive")
        gf = comparisons[key_id({**key, "representation": "Gfree"})][0]
        si = comparisons[key_id({**key, "representation": "SI"})][0]
        g_squared = True
        for p, q in zip(
            gf["production_quantity"]["data"],
            si["production_quantity"]["data"],
            strict=True,
        ):
            for part in ("real", "imag"):
                expected_si = _binary64(p[part], kernel) * g**2
                g_squared &= kernel.decimal_fraction(
                    kernel.promote_binary64(expected_si)
                ) == kernel.decimal_fraction(q[part])
        scalar = quantity["shape"] == []
        tolerance = kernel.fraction_decimal(
            kernel.decimal_fraction(budget.get("structural_rate", "0e+0"))
            * kernel.decimal_fraction(source_scales[identity]),
            180,
            downward=True,
        )
        structures = kernel.structure_gates(
            quantity,
            "none"
            if scalar
            else "identity"
            if family in {"B2", "A3"}
            else "horizontal_vertical_2",
            tolerance,
        )
        connection = row["connection_assessment"]
        if (
            set(connection) != {"gates", "evidence"}
            or set(connection["gates"]) != set(policy["connection_requirements"])
            or any(value is not True for value in connection["gates"].values())
        ):
            raise EvidenceError("retained family connection gates incomplete")
        contracts = {
            "units": reference["units"] == fixture["units"][key["representation"]]
            and g_squared,
            "shape": quantity["shape"] == reference["quantity"]["shape"],
            "frequency_axis": reference["key"]["frequency_hz"] == key["frequency_hz"],
            "input_normalization": True,
            "source_connections": all(connection["gates"].values()),
            "resolved_reference": reference["status"] == "resolved",
            "sealed_exact_coverage": True,
        }
        assessment = kernel.assess(
            quantity,
            reference["quantity"],
            source_scales[identity],
            budget["absolute_rate"],
            budget["relative_rate"],
            reference["uncertainty"],
            structures,
            contracts,
        )
        if (
            assessment != row["assessment"]
            or structures != row["structure_conditions"]
            or contracts != row["contract_conditions"]
            or assessment["accepted"] is not True
        ):
            raise EvidenceError(
                "retained arithmetic/structure/units assessment differs"
            )
        kind = reference["uncertainty"]["kind"]
        kinds[kind] += 1
        claim = assessment["claim"]
        claims[claim] = claims.get(claim, 0) + 1
        ratios = assessment_ratios(assessment, whole_zero=kind == "exact_zero")
        if ratios["reason"]:
            zero_budget += 1
        for name in maximum:
            if ratios[name] is not None:
                ratio = Fraction(ratios[name])
                if maximum[name] is None or ratio > Fraction(maximum[name]["value"]):
                    maximum[name] = {
                        "value": str(ratio),
                        "display_upper": kernel.fraction_decimal(
                            ratio, 12, upward=True
                        ),
                        "key": key,
                    }
        counts[key["set"]] += 1
        if family == "T1" and key["fixture_id"] == "t1-x400-amplified":
            if (
                key["set"] != "auxiliary"
                or not scalar
                or _binary64(quantity["data"][0]["real"], kernel) <= 0
            ):
                raise EvidenceError(
                    "mandatory T1 x400 AUX is not positive finite binary64"
                )
            x400.append(key["representation"])
    if family == "T1" and (
        counts != {"science": 12, "auxiliary": 4} or sorted(x400) != ["Gfree", "SI"]
    ):
        raise EvidenceError("T1 physical/AUX/x400 frozen partition differs")
    return {
        "status": "RETAINED_ARITHMETIC_REPLAYED",
        "family": family,
        "science_count": counts["science"],
        "auxiliary_count": counts["auxiliary"],
        "nonzero_oracle_count": len(ids) - kinds["exact_zero"],
        "whole_zero_oracle_count": kinds["exact_zero"],
        "uncertainty_kind_counts": kinds,
        "claim_basis_counts": claims,
        "max_ratios": maximum,
        "zero_budget_ratio_count": zero_budget,
        "zero_budget_ratio_reason": "CERTIFIED_WHOLE_ZERO_BUDGET"
        if zero_budget
        else None,
        "ratio_basis": "exact serialized rational upper bounds / budget lower bound",
        "T1_x400_positive_Gfree_and_SI": True if family == "T1" else None,
        "numeric_delivery_gates_pass": bool(counts["science"]),
        "independent_result_admission_verified": False,
        "final_transfer_established": False,
        "original_execution_repeated": False,
        "production_api_calls": 0,
    }
