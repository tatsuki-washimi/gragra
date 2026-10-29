"""Production API mappings and serialized arithmetic; imports no production at load."""

from __future__ import annotations

import importlib
import inspect
import math
from collections.abc import Sequence
from decimal import Decimal

from .transport import GateError, exact_keys, key_id


def validate_frozen_connection_gates(
    connection: object, requirements: Sequence[str]
) -> dict[str, bool]:
    """Validate connector output before ``source_connections`` contract use."""
    if (
        not isinstance(connection, dict)
        or set(connection) != {"gates", "evidence"}
        or set(connection["gates"]) != set(requirements)
        or any(type(v) is not bool for v in connection["gates"].values())
    ):
        raise GateError("family connector missing/unknown/implicit frozen gates")
    return connection["gates"]

_API = {
    "R1": ("rayleigh", "rayleigh_isotropic_acceleration_csd"),
    "B2": ("bodywave", "spherical_cavity_ps_isotropic_acceleration_csd"),
    "A1": ("acoustic", "surface_isotropic_acoustic_acceleration_csd"),
    "A2": ("acoustic", "underground_isotropic_acoustic_acceleration_csd"),
    "A3": ("acoustic", "centered_cavern_isotropic_acoustic_acceleration_csd"),
    "T1": ("temperature", "uniform_advection_temperature_acceleration_psd"),
}


def call_plan(key: dict, inputs: dict) -> dict:
    """Map complete frozen decimals without imports, evaluation or float coercion."""
    family, representation = key["family"], key["representation"]
    if family not in _API or representation not in {"Gfree", "SI"}:
        raise GateError("unknown family/representation mapping")
    package, symbol = _API[family]
    args, kwargs, index, shape = [], {}, [], [3, 3]
    if family == "A2" and key["observable"] == "isotropic_depth_factor":
        if representation != "Gfree" or "nominal_exact_x" not in inputs:
            raise GateError(
                "scalar A2 AUX needs exact nominal x and dimensionless Gfree key"
            )
        return {
            "module": "gragra.analytic.stochastic_acoustic",
            "symbol": "isotropic_acoustic_depth_factor",
            "args": [inputs["nominal_exact_x"]],
            "kwargs": {},
            "shape": [],
            "output_index": [],
        }
    if key["observable"] != (
        "projected_acceleration_psd" if family == "T1" else "acceleration_csd"
    ):
        raise GateError("unknown frozen family observable")
    kwargs["frequency_hz"] = key["frequency_hz"]
    if inputs.get("frequency_hz") != key["frequency_hz"]:
        raise GateError("input/key frequency mapping mismatch")
    if family in {"R1", "B2"}:
        names = ["density_kg_m3", "v_p_m_s", "v_s_m_s"]
        if family == "R1":
            names += ["surface_vertical_displacement_psd_m2_hz"]
            args, index = [[["0e+0", "0e+0", inputs["height_m"]]]], [0]
        else:
            names += [
                "radius_m",
                "p_displacement_psd_m2_hz",
                "s_displacement_psd_m2_hz",
            ]
    elif family in {"A1", "A2", "A3"}:
        names = ["sound_speed_m_s"] + (
            ["depth_m"] if family == "A2" else ["radius_m"] if family == "A3" else []
        )
        pressure = (
            "cavern_pressure_psd_pa2_hz"
            if family == "A3"
            else "surface_pressure_psd_pa2_hz"
        )
        kwargs[pressure] = inputs["pressure_psd_pa2_hz"]
    else:
        shape = []
        names = [
            "density_kg_m3",
            "temperature_k",
            "structure_coefficient_k_m_neg_p_over_2",
            "structure_exponent",
            "wind_speed_m_s",
            "minimum_approach_m",
            "wind_axis_angle_rad",
        ]
    kwargs.update({n: inputs[n] for n in names})
    return {
        "module": "gragra."
        + ("analytic" if representation == "Gfree" else "observables")
        + ".stochastic_"
        + package,
        "symbol": symbol + ("_g_free" if representation == "Gfree" else ""),
        "args": args,
        "kwargs": kwargs,
        "shape": shape,
        "output_index": index,
    }


def promote_quantity(value, shape: list, index: list, kernel) -> dict:
    """Preserve actual shape, every complex entry and every binary64 bit."""
    for position in index:
        value = value[position]
    actual_shape = list(getattr(value, "shape", ()))
    if actual_shape != shape:
        raise GateError("actual production output shape mismatch")
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        if (getattr(dtype, "kind", None), getattr(dtype, "itemsize", None)) not in {
            ("f", 8),
            ("c", 16),
        }:
            raise GateError("actual production dtype must be float64/complex128")
    elif shape or type(value) not in {float, complex}:
        raise GateError("actual production tensor must expose binary64 dtype")
    raw = value.tolist() if hasattr(value, "tolist") else value
    entries = [raw] if not shape else [entry for row in raw for entry in row]
    if len(entries) != (1 if not shape else 9):
        raise GateError("actual production tensor cardinality mismatch")
    data = []
    for value in entries:
        if type(value) not in {float, complex}:
            raise GateError("actual production scalar must be binary64/complex128")
        real, imag = float(value.real), float(value.imag)
        if not math.isfinite(real) or not math.isfinite(imag):
            raise GateError("actual production output nonfinite")
        data.append(
            {
                "real": kernel.promote_binary64(real),
                "imag": kernel.promote_binary64(imag),
            }
        )
    return {"shape": shape, "data": data}


def partition_summary(records: list[dict]) -> dict:
    if any(
        r["key"]["set"] not in {"science", "auxiliary"}
        or type(r.get("passed")) is not bool
        for r in records
    ):
        raise GateError("comparison partition or explicit result invalid")
    science = [r for r in records if r["key"]["set"] == "science"]
    auxiliary = [r for r in records if r["key"]["set"] == "auxiliary"]
    physical = bool(science) and all(r["passed"] for r in science)
    auxiliary_pass = all(r["passed"] for r in auxiliary)
    return {
        "science_count": len(science),
        "science_passed": sum(r["passed"] for r in science),
        "auxiliary_count": len(auxiliary),
        "auxiliary_passed": sum(r["passed"] for r in auxiliary),
        "physical_acceptance": physical,
        "numerical_aux_status": "PASS" if auxiliary_pass else "FAIL",
        "delivery_ready": physical and auxiliary_pass,
    }


def source_scale_table(table: dict, expected: list[dict], kernel) -> dict:
    """Consume exact reviewed scale bytes, never reconstruct from oracle output."""
    if (
        set(table) != {"schema", "family", "rows", "pins", "rounding"}
        or table["schema"] != "gen1-stochastic-source-scales-v1"
    ):
        raise GateError("source-scale table schema mismatch")
    keys, rows = [], {}
    for row in table["rows"]:
        if (
            set(row) != {"key", "value", "method", "input_sha256"}
            or not isinstance(row["method"], str)
            or not row["method"].strip()
            or kernel.decimal_fraction(row["value"]) < 0
        ):
            raise GateError("source-scale value/method invalid")
        keys.append(row["key"])
        rows[key_id(row["key"])] = row["value"]
    if exact_keys(keys) != exact_keys(expected) or any(
        k["family"] != table["family"] for k in keys
    ):
        raise GateError("source-scale full-key coverage mismatch")
    return rows


def production_evaluator(prepared: dict):
    """Lazy factory called only by the exact authorized supervised driver.

    A separately reviewed family connection assessor owns all additional frozen
    source/neighbor contracts. Missing or unknown gates can never become PASS.
    """
    kernel = prepared["kernel"]
    spec = prepared["policy"]["connection_assessor"]
    module = importlib.import_module(spec["module"])
    from .transport import artifact_pin

    if (
        artifact_pin(module.__file__, "connection_assessor")["sha256"]
        != spec["source"]["sha256"]
    ):
        raise GateError("actual connection assessor source differs")
    connector = getattr(module, spec["symbol"], None)
    if not callable(connector):
        raise GateError("actual connection assessor callable missing")
    outputs, calls = {}, {}

    def coerce(value):
        if isinstance(value, list):
            return [coerce(v) for v in value]
        if isinstance(value, dict):
            return {k: coerce(v) for k, v in value.items()}
        numeric = float(Decimal(value))
        if not math.isfinite(numeric):
            raise GateError("frozen input cannot be represented in binary64")
        return numeric

    def actual(key, fresh=False):
        identity = key_id(key)
        if identity not in prepared["plans"]:
            raise GateError("family connector requested an unfrozen production key")
        if identity not in outputs or fresh:
            plan = prepared["plans"][identity]
            api = getattr(importlib.import_module(plan["module"]), plan["symbol"], None)
            if not callable(api):
                raise GateError("actual family production callable missing")
            args, kwargs = coerce(plan["args"]), coerce(plan["kwargs"])
            inspect.signature(api).bind(*args, **kwargs)
            import sys

            from .dependencies import verify_runtime_closure

            verify_runtime_closure(prepared["closure"], sys.modules, [])
            try:
                result = api(*args, **kwargs)
            finally:
                verify_runtime_closure(prepared["closure"], sys.modules, [])
            outputs[identity] = promote_quantity(
                result, plan["shape"], plan["output_index"], kernel
            )
            calls[identity] = {
                "module": plan["module"],
                "symbol": plan["symbol"],
                "args": plan["args"],
                "kwargs": plan["kwargs"],
                "binary64_args": [str(v) for v in args],
                "binary64_kwargs": {
                    k: kernel.promote_binary64(v) for k, v in kwargs.items()
                },
            }
        return outputs[identity]

    def evaluate(key):
        identity = key_id(key)
        reference = prepared["records"][identity]
        production = actual(key)
        budget, family = prepared["budget"], key["family"]
        scalar = production["shape"] == []
        symmetry = "identity" if family in {"B2", "A3"} else "horizontal_vertical_2"
        tolerance = kernel.fraction_decimal(
            kernel.decimal_fraction(budget.get("structural_rate", "0e+0"))
            * kernel.decimal_fraction(prepared["scales"][identity]),
            180,
            downward=True,
        )
        structures = kernel.structure_gates(
            production, "none" if scalar else symmetry, tolerance
        )
        connection = connector(
            {
                "key": key,
                "reference": reference,
                "production": production,
                "actual": actual,
                "actual_fresh": lambda k: actual(k, fresh=True),
                "production_identity": {
                    "commit": prepared["pins"]["production_commit"],
                    "tree": prepared["pins"]["production_tree"],
                    "module_sources": [
                        p
                        for p in prepared["closure"]["modules"]
                        if p["package"] == "gragra"
                    ],
                },
                "production_source_reviews": prepared["source_review"]["packet"],
                "production_source_pins": prepared["source_review"]["pins"],
                "reference_artifacts": prepared["reference_artifacts"],
                "frozen_case": next(
                    r
                    for r in prepared["fixtures"].get(
                        "science", prepared["fixtures"].get("fixtures", [])
                    )
                    if r.get("fixture_id", r.get("key_base", {}).get("fixture_id"))
                    == key["fixture_id"]
                ),
                "all_references": prepared["records"],
                "frozen_fixture": prepared["fixtures"],
                "frozen_procedure": prepared["procedure"],
                "frozen_budget": budget,
                "source_scales": prepared["scales"],
                "kernel": kernel,
            }
        )
        requirements = prepared["policy"]["connection_requirements"]
        gates = validate_frozen_connection_gates(connection, requirements)
        g_squared = True
        if key["observable"] != "isotropic_depth_factor":
            gf_key, si_key = (
                {**key, "representation": "Gfree"},
                {**key, "representation": "SI"},
            )
            gf, si = actual(gf_key), actual(si_key)
            constants = importlib.import_module("gragra.constants")
            frozen_g = float(Decimal(reference["inputs"]["g_si"]))
            if getattr(constants, "G_SI", None) != frozen_g:
                raise GateError("actual family G_SI differs from frozen input")
            factor = frozen_g**2
            for p, q in zip(gf["data"], si["data"], strict=True):
                for part in ("real", "imag"):
                    expected = float(Decimal(p[part])) * factor
                    g_squared &= kernel.decimal_fraction(
                        kernel.promote_binary64(expected)
                    ) == kernel.decimal_fraction(q[part])
        contracts = {
            "units": reference["units"]
            == (
                "1"
                if key["observable"] == "isotropic_depth_factor"
                else prepared["fixtures"]["units"][key["representation"]]
            )
            and g_squared,
            "shape": production["shape"] == reference["quantity"]["shape"],
            "frequency_axis": reference["key"]["frequency_hz"] == key["frequency_hz"],
            "input_normalization": True,
            "source_connections": all(gates.values()),
            "resolved_reference": reference["status"] == "resolved",
            "sealed_exact_coverage": True,
        }
        assessment = kernel.assess(
            production,
            reference["quantity"],
            prepared["scales"][identity],
            budget["absolute_rate"],
            budget["relative_rate"],
            reference["uncertainty"],
            structures,
            contracts,
        )
        return {
            "key": key,
            "status": "resolved",
            "passed": assessment["accepted"],
            "assessment": assessment,
            "production_quantity": production,
            "structure_conditions": structures,
            "contract_conditions": contracts,
            "connection_assessment": connection,
            "actual_calls": [
                {
                    "key": dict(prepared["records"][call_id]["key"]),
                    "call": call,
                }
                for call_id, call in sorted(calls.items())
            ],
            "reference_record_sha256": prepared["record_hashes"][identity],
            "reference_purpose": "formal-science",
            "reference_set": key["set"],
        }

    return evaluate
