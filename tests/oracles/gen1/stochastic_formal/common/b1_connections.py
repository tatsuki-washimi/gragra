"""Separate frozen first-order/consistency reader; never imports an oracle kernel."""

from __future__ import annotations

import hashlib
from decimal import Decimal
from fractions import Fraction

from .transport import CONNECTION_GATES, GateError, canonical_bytes, exact_keys, key_id

SHAPES = {
    "p_acceleration_connection": [3],
    "s_potential_null": [],
    "s_acceleration_null": [3],
    "s_gradient_null": [3, 3],
    "rotation_csd_connection": [3, 3],
    "component_total_psd_connection": [3, 3],
}
CAUSES = (
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


def values(quantity: dict) -> list[tuple[Fraction, Fraction]]:
    if set(quantity) != {"shape", "real", "imag"} or quantity["shape"] not in (
        [],
        [3],
        [3, 3],
    ):
        raise GateError("first-order scalar/vector/tensor shape required")
    size = 1 if not quantity["shape"] else 3 if quantity["shape"] == [3] else 9
    if len(quantity["real"]) != size or len(quantity["imag"]) != size:
        raise GateError("quantity component count mismatch")

    def rational(value):
        if not isinstance(value, str):
            raise GateError("quantity scientific strings required")
        try:
            dec = Decimal(value)
        except Exception as exc:
            raise GateError("quantity decimal parse failed") from exc
        if not dec.is_finite():
            raise GateError("nonfinite quantity")
        return Fraction(dec)

    return [
        (rational(a), rational(b))
        for a, b in zip(quantity["real"], quantity["imag"], strict=True)
    ]


def numeric_connection(
    production: dict,
    reference: dict,
    scale: str,
    absolute_rate: str,
    relative_rate: str,
    uncertainty: str,
    kernel,
    *,
    consistency=False,
    exact_production_values=None,
) -> dict:
    """Full-field norm decisions at both frozen precisions; no second-order assess."""
    serialized_pv, rv = values(production), values(reference)
    pv = serialized_pv if exact_production_values is None else exact_production_values
    if production["shape"] != reference["shape"]:
        raise GateError("connection observable shape mismatch")
    sf, a, r, u = (
        kernel.decimal_fraction(v)
        for v in (scale, absolute_rate, relative_rate, uncertainty)
    )
    if min(sf, a, r, u) < 0 or sf <= 0:
        raise GateError("connection positive physical source scale required")
    d2 = sum(
        ((x - p) ** 2 + (y - q) ** 2 for (x, y), (p, q) in zip(pv, rv, strict=True)),
        Fraction(),
    )
    n2 = sum((x * x + y * y for x, y in rv), Fraction())
    decisions = []
    snapshots = []
    for digits in (120, 180):
        dl, du = kernel._sqrt_bounds(d2, digits)
        nl, nu = kernel._sqrt_bounds(n2, digits)
        # Exact quaternion/paired subtraction is rational. Only outward norm
        # reporting has arithmetic enclosure width, explicitly retained here.
        serialization_square = sum(
            (
                (x - a) ** 2 + (y - b) ** 2
                for (x, y), (a, b) in zip(pv, serialized_pv, strict=True)
            ),
            Fraction(),
        )
        serialization_u = kernel._sqrt_bounds(serialization_square, digits)[1]
        arithmetic_u = du - dl + serialization_u if consistency else Fraction()
        total_u = u + arithmetic_u
        bl, bu = (
            ((a + r) * sf, (a + r) * sf)
            if consistency
            else (a * sf + r * nl, a * sf + r * nu)
        )
        passed = du + total_u <= bl and 10 * total_u <= bl
        decisions.append(passed)
        snapshots.append(
            {
                "precision": digits,
                "difference_upper": kernel.fraction_decimal(du, digits, upward=True),
                "reference_norm_lower": kernel.fraction_decimal(
                    nl, digits, downward=True
                ),
                "reference_norm_upper": kernel.fraction_decimal(
                    nu, digits, upward=True
                ),
                "allowance_lower": kernel.fraction_decimal(bl, digits, downward=True),
                "allowance_upper": kernel.fraction_decimal(bu, digits, upward=True),
                "uncertainty_upper": kernel.fraction_decimal(
                    total_u, digits, upward=True
                ),
                "arithmetic_uncertainty_upper": kernel.fraction_decimal(
                    arithmetic_u, digits, upward=True
                ),
                "difference_plus_uncertainty": du + total_u <= bl,
                "oracle_fraction": 10 * total_u <= bl,
                "passed": passed,
            }
        )
    return {
        "passed": decisions[0] and decisions[1] and decisions[0] == decisions[1],
        "precision_crosscheck": decisions[0] == decisions[1],
        "conditions": snapshots,
        "evidence_class": "production_consistency_only"
        if consistency
        else "independent_first_order_reference",
    }


def rotated_difference(
    matrix: list[list[tuple[Fraction, Fraction]]], quaternion: list[str]
):
    w, x, y, z = [Fraction(Decimal(q)) for q in quaternion]
    norm2 = w * w + x * x + y * y + z * z
    if not norm2:
        raise GateError("nonzero quaternion required")
    rotation = [
        [
            (w * w + x * x - y * y - z * z) / norm2,
            2 * (x * y - w * z) / norm2,
            2 * (x * z + w * y) / norm2,
        ],
        [
            2 * (x * y + w * z) / norm2,
            (w * w - x * x + y * y - z * z) / norm2,
            2 * (y * z - w * x) / norm2,
        ],
        [
            2 * (x * z - w * y) / norm2,
            2 * (y * z + w * x) / norm2,
            (w * w - x * x - y * y + z * z) / norm2,
        ],
    ]
    return [
        tuple(
            sum(
                (
                    rotation[i][k] * rotation[j][column] * matrix[k][column][part]
                    for k in range(3)
                    for column in range(3)
                ),
                Fraction(),
            )
            - matrix[i][j][part]
            for part in (0, 1)
        )
        for i in range(3)
        for j in range(3)
    ]


def validate_endpoint_map(endpoints: dict, expected_keys: list[dict]) -> None:
    required = {k["observable"] for k in expected_keys}
    if set(endpoints) != required:
        raise GateError("production endpoint mapping incomplete")
    for name, endpoint in endpoints.items():
        if (
            name not in SHAPES
            or not isinstance(endpoint, dict)
            or set(endpoint) != {"module", "symbol", "source_sha256"}
        ):
            raise GateError("production endpoint schema mismatch")
        if not all(isinstance(v, str) and v for v in endpoint.values()):
            raise GateError("production endpoint unresolved")


def validate_auxiliary_records(
    records: list[dict],
    expected_keys: list[dict],
    f_aux_sha: str,
    fixtures: dict[str, dict],
    scale_rows: list[dict],
    kernel,
) -> dict:
    expected = exact_keys(expected_keys)
    if len(expected) != 56 or len(records) != 56:
        raise GateError("56 independent reference records required")
    scales = {key_id(row["key"]): row for row in scale_rows}
    result = {}
    fields = {
        "schema_version",
        "record_kind",
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
    for row in records:
        rid = key_id(row["key"])
        if rid not in expected or rid in result or set(row) != fields:
            raise GateError("R_aux key/record schema mismatch")
        if (
            row["record_kind"] != "first_order_connection_reference"
            or row["status"] != "resolved"
            or row["procedure_hash"] != f_aux_sha
            or row["input_assumptions"] != "exact_input"
            or row["quantity"]["shape"] != SHAPES[row["key"]["observable"]]
            or row["units"] != scales[rid]["units"]
        ):
            raise GateError("R_aux resolved procedure/shape/units mismatch")
        values(row["quantity"])
        u = row["uncertainty"]
        if set(u) != {"kind", "components", "certificate"} or set(
            u["components"]
        ) != set(CAUSES):
            raise GateError("R_aux uncertainty sources incomplete")
        total = Fraction()
        for component in u["components"].values():
            if set(component) != {"value", "method"} or not component["method"]:
                raise GateError("R_aux uncertainty method absent")
            term = kernel.decimal_fraction(component["value"])
            if term < 0:
                raise GateError("negative R_aux uncertainty")
            total += term
        fixture = fixtures[row["key"]["fixture_id"]]
        if row["inputs"] != fixture:
            raise GateError("R_aux input binding differs from frozen fixture")
        if fixture["kind"] == "S_null":
            raw = sum(
                (
                    Fraction(Decimal(x)) * Fraction(Decimal(y))
                    for x, y in zip(
                        fixture["direction_raw"],
                        fixture["polarization_raw"],
                        strict=True,
                    )
                ),
                Fraction(),
            )
            certificate = u["certificate"]
            if (
                raw != 0
                or total != 0
                or u["kind"] != "exact_zero"
                or not isinstance(certificate, dict)
                or any(x != 0 or y != 0 for x, y in values(row["quantity"]))
                or certificate.get("kind") != "whole_field_transverse_null"
                or certificate.get("domain")
                != "infinite homogeneous boundary-free medium"
                or certificate.get("source_equations") != ["55", "60"]
                or certificate.get("nonzero_direction_and_polarization") is not True
                or certificate.get("normalization_preserves_transversality") is not True
                or certificate.get("whole_field") is not True
                or certificate.get("raw_dot_product") != "0e+0"
                or certificate.get("direction_raw") != fixture["direction_raw"]
                or certificate.get("polarization_raw") != fixture["polarization_raw"]
            ):
                raise GateError("R_aux null applicability certificate mismatch")
        elif (
            total <= 0 or u["kind"] != "rigorous_bound" or u["certificate"] is not None
        ):
            raise GateError("nonzero P reference uncertainty invalid")
        result[rid] = {
            "record": row,
            "uncertainty": kernel.fraction_decimal(total, 180, upward=True),
        }
    if set(result) != expected:
        raise GateError("R_aux missing keys")
    return result


def assess_connection(
    key: dict,
    fixture: dict,
    actual: dict,
    reference: dict | None,
    scale: dict,
    connection_budget: dict,
    kernel,
    *,
    reference_pin: dict | None,
    production_pin: dict,
) -> dict:
    """Assess already-authorized actual outputs; all gates are explicit and vetoing."""
    expected_shape = SHAPES[key["observable"]]
    quantity = actual["quantity"]
    values(quantity)
    if quantity["shape"] != expected_shape:
        raise GateError("production connection shape mismatch")
    consistency = reference is None
    if consistency:
        comparison_reference = {
            "shape": expected_shape,
            "real": ["0e+0"] * len(quantity["real"]),
            "imag": ["0e+0"] * len(quantity["imag"]),
        }
        uncertainty = "0e+0"
    else:
        comparison_reference = reference["record"]["quantity"]
        uncertainty = reference["uncertainty"]
    numeric = numeric_connection(
        quantity,
        comparison_reference,
        scale["value"],
        connection_budget["absolute_rate"],
        connection_budget["relative_rate"],
        uncertainty,
        kernel,
        consistency=consistency,
        exact_production_values=actual.get("exact_values"),
    )
    call = actual["call"]
    structural = call.get("gate_results")
    if not isinstance(structural, dict) or set(structural) != set(CONNECTION_GATES) - {
        "numeric_inequalities",
        "precision_crosscheck",
    }:
        raise GateError(
            "production call must document every nonnumeric frozen connection gate"
        )
    gates = {
        **structural,
        "numeric_inequalities": numeric["passed"],
        "precision_crosscheck": numeric["precision_crosscheck"],
    }
    if any(type(v) is not bool for v in gates.values()):
        raise GateError("ambiguous connection gate")
    passed = all(gates.values())
    return {
        "key": key,
        "status": "resolved",
        "passed": passed,
        "gate_results": gates,
        "production_quantity": quantity,
        "assessment": numeric,
        "production_pin": production_pin,
        "reference_pin": reference_pin,
        "evidence_class": numeric["evidence_class"],
        "inputs": fixture,
        "actual_call": call,
        "input_sha256": hashlib.sha256(canonical_bytes(fixture)).hexdigest(),
    }


def check_frozen_g_si(constants, fixtures: list[dict]) -> float:
    """Bind the actual production constant without a scientific API call."""
    value = getattr(constants, "G_SI", None)
    if type(value) is not float or value <= 0:
        raise GateError("actual production G_SI constant unavailable")
    for fixture in fixtures:
        if fixture.get("g_si") is not None and float(Decimal(fixture["g_si"])) != value:
            raise GateError("actual runtime G differs from frozen connection input")
    return value


def production_startup_audit(endpoints: dict, fixtures: list[dict]) -> dict:
    """Import-only audit of every factory callable, field and constant binding.

    Signature.bind uses inert placeholders; no production API is invoked and
    no production result or reference difference is constructed.
    """
    import importlib
    import inspect

    from .transport import artifact_pin

    first_order = {
        name: object()
        for name in (
            "targets_m",
            "density_kg_m3",
            "wavenumber_rad_m",
            "displacement_m",
            "propagation_direction",
            "polarization",
            "wave_type",
            "phase_origin_m",
        )
    }
    second_order = {
        name: object()
        for name in ("density_kg_m3", "frequency_hz", "p_displacement_psd_m2_hz")
    }
    reports = {}
    for name, endpoint in endpoints.items():
        module = importlib.import_module(endpoint["module"])
        if (
            artifact_pin(module.__file__, "production_source")["sha256"]
            != endpoint["source_sha256"]
        ):
            raise GateError("startup endpoint source bytes mismatch")
        api = getattr(module, endpoint["symbol"], None)
        if not callable(api):
            raise GateError("startup endpoint callable missing")
        signature = inspect.signature(api)
        kwargs = (
            second_order
            if name in {"rotation_csd_connection", "component_total_psd_connection"}
            else first_order
        )
        try:
            signature.bind(**kwargs)
        except TypeError as error:
            raise GateError("startup actual production signature differs") from error
        reports[name] = {
            "module": endpoint["module"],
            "symbol": endpoint["symbol"],
            "signature": str(signature),
            "source_sha256": endpoint["source_sha256"],
        }
    si = importlib.import_module("gragra.observables.analytic")
    wrapper = getattr(si, "to_gravity_observables", None)
    si_csd = importlib.import_module("gragra.observables.stochastic_bodywave")
    api = getattr(si_csd, "fullspace_p_isotropic_acceleration_csd", None)
    if not callable(wrapper) or not callable(api):
        raise GateError("startup actual SI conversion/CSD callable missing")
    try:
        inspect.signature(wrapper).bind(object())
        inspect.signature(api).bind(**second_order)
    except TypeError as error:
        raise GateError("startup actual SI signature differs") from error
    fields = importlib.import_module("gragra.analytic.values")
    for cls in (
        getattr(fields, "GFreeGravityValues", None),
        getattr(si, "GravityObservables", None),
    ):
        if not {"potential", "acceleration", "gradient"} <= set(
            getattr(cls, "__dataclass_fields__", {})
        ):
            raise GateError("startup actual first-order observable fields missing")
    constant = check_frozen_g_si(importlib.import_module("gragra.constants"), fixtures)
    return {
        "status": "PASS_IMPORT_ONLY",
        "production_api_calls": 0,
        "endpoints": reports,
        "G_SI_binary64_hex": constant.hex(),
        "first_order_fields": ["potential", "acceleration", "gradient"],
        "SI_wrapper_signature": str(inspect.signature(wrapper)),
        "SI_CSD_signature": str(inspect.signature(api)),
    }


def production_factory(
    endpoints: dict,
    fixtures_by_id: dict,
    science_by_id: dict,
    scale_by_key: dict,
    kernel,
    references: dict,
    production_pin: dict,
    reference_pin: dict,
    connection_budget: dict,
):
    """Load reviewed actual production functions lazily, never an oracle fallback."""
    import importlib
    import math
    from pathlib import Path

    import numpy as np

    from .transport import artifact_pin

    production_startup_audit(endpoints, list(fixtures_by_id.values()))

    required_symbols = {
        "p_acceleration_connection": (
            "gragra.analytic._fullspace_bodywave",
            "fullspace_bodywave_observables_g_free",
        ),
        "s_potential_null": (
            "gragra.analytic._fullspace_bodywave",
            "fullspace_bodywave_observables_g_free",
        ),
        "s_acceleration_null": (
            "gragra.analytic._fullspace_bodywave",
            "fullspace_bodywave_observables_g_free",
        ),
        "s_gradient_null": (
            "gragra.analytic._fullspace_bodywave",
            "fullspace_bodywave_observables_g_free",
        ),
        "rotation_csd_connection": (
            "gragra.analytic.stochastic_bodywave",
            "fullspace_p_isotropic_acceleration_csd_g_free",
        ),
        "component_total_psd_connection": (
            "gragra.analytic.stochastic_bodywave",
            "fullspace_p_isotropic_acceleration_csd_g_free",
        ),
    }
    loaded = {}
    for name, endpoint in endpoints.items():
        if (endpoint["module"], endpoint["symbol"]) != required_symbols[name]:
            raise GateError("reviewed frozen production endpoint mapping differs")
        try:
            module = importlib.import_module(endpoint["module"])
        except ImportError as exc:
            raise GateError(
                f"production endpoint unavailable: {endpoint['module']}"
            ) from exc
        if (
            artifact_pin(Path(module.__file__), "production_source")["sha256"]
            != endpoint["source_sha256"]
        ):
            raise GateError("production endpoint source bytes mismatch")
        api = getattr(module, endpoint["symbol"], None)
        if not callable(api):
            raise GateError("production endpoint callable missing")
        loaded[name] = api
    si_module = importlib.import_module("gragra.observables.analytic")
    constants = importlib.import_module("gragra.constants")
    check_frozen_g_si(constants, list(fixtures_by_id.values()))

    def quantity(array, shape):
        array = np.asarray(array)
        if list(array.shape) != shape or array.dtype not in (
            np.dtype("float64"),
            np.dtype("complex128"),
        ):
            raise GateError("actual scalar/vector/tensor binary64 shape/dtype mismatch")
        if not np.isfinite(array).all():
            raise GateError("nonfinite actual connection output")
        flattened = array.reshape(-1)
        return {
            "shape": shape,
            "real": [kernel.promote_binary64(float(v.real)) for v in flattened],
            "imag": [kernel.promote_binary64(float(v.imag)) for v in flattened],
        }

    def csd_call(key, fixture):
        index = fixture["frequency_hz"].index(key["frequency_hz"])
        psd = Fraction(Decimal(fixture["displacement_psd_m2_hz"][index]))
        if fixture["input_convention"] == "total":
            psd /= 3
        kwargs = {
            "density_kg_m3": float(Decimal(fixture["density_kg_m3"])),
            "frequency_hz": float(Decimal(key["frequency_hz"])),
            "p_displacement_psd_m2_hz": float(psd),
        }
        result = loaded[key["observable"]](**kwargs)
        if key["representation"] == "SI":
            si = importlib.import_module("gragra.observables.stochastic_bodywave")
            result = si.fullspace_p_isotropic_acceleration_csd(**kwargs)
        return result, kwargs

    def evaluate(key):
        fixture = fixtures_by_id[key["fixture_id"]]
        kind = fixture["kind"]
        shape = SHAPES[key["observable"]]
        calls = []
        raw_quantities = []
        exact_values = None
        if kind in {"P_transfer", "S_null"}:
            speed = (
                fixture["p_speed_m_s"]
                if kind == "P_transfer"
                else fixture["s_speed_m_s"]
            )
            kwargs = {
                "targets_m": [[float(Decimal(v)) for v in fixture["target_m"]]],
                "density_kg_m3": float(Decimal(fixture["density_kg_m3"])),
                "wavenumber_rad_m": 2
                * math.pi
                * float(Decimal(key["frequency_hz"]))
                / float(Decimal(speed)),
                "displacement_m": complex(
                    float(Decimal(fixture["displacement_amplitude_m"]["real"])),
                    float(Decimal(fixture["displacement_amplitude_m"]["imag"])),
                ),
                "propagation_direction": [
                    float(Decimal(v)) for v in fixture["direction_raw"]
                ],
                "polarization": [
                    float(Decimal(v)) for v in fixture["polarization_raw"]
                ],
                "wave_type": "P" if kind == "P_transfer" else "S",
                "phase_origin_m": (0.0, 0.0, 0.0),
            }
            fields = loaded[key["observable"]](**kwargs)
            if key["representation"] == "SI":
                fields = si_module.to_gravity_observables(fields)
            attr = (
                "potential"
                if shape == []
                else "acceleration"
                if shape == [3]
                else "gradient"
            )
            output_quantity = quantity(getattr(fields, attr)[0], shape)
            calls.append(kwargs)
            raw_quantities.append(output_quantity)
            reference = references[key_id(key)]
            raw_dot = sum(
                (
                    Fraction(Decimal(a)) * Fraction(Decimal(b))
                    for a, b in zip(
                        fixture["direction_raw"],
                        fixture["polarization_raw"],
                        strict=True,
                    )
                ),
                Fraction(),
            )
            polarization = (
                raw_dot == 0
                if kind == "S_null"
                else fixture["direction_raw"] == fixture["polarization_raw"]
            )
            applicability = (
                fixture["frame"] == "Cartesian" and fixture["target_m"] == ["0e+0"] * 3
            )
        elif kind == "rotation":
            raw, kwargs = csd_call(key, science_by_id[fixture["science_fixture_id"]])
            calls.append(kwargs)
            raw_quantities.append(quantity(raw, [3, 3]))
            base = values(raw_quantities[-1])
            matrix = [base[i * 3 : i * 3 + 3] for i in range(3)]
            exact_values = rotated_difference(matrix, fixture["quaternion_raw"])
            output_quantity = {
                "shape": [3, 3],
                "real": [kernel.fraction_decimal(x, 180) for x, y in exact_values],
                "imag": [kernel.fraction_decimal(y, 180) for x, y in exact_values],
            }
            reference = None
            polarization = True
            applicability = fixture["quaternion_order"] == ["w", "x", "y", "z"]
        elif kind == "component_total":
            first, kwargs1 = csd_call(
                key, science_by_id[fixture["component_fixture_id"]]
            )
            second, kwargs2 = csd_call(key, science_by_id[fixture["total_fixture_id"]])
            calls.extend([kwargs1, kwargs2])
            raw_quantities.extend([quantity(first, [3, 3]), quantity(second, [3, 3])])
            p1, p2 = [values(item) for item in raw_quantities]
            exact_values = [
                (x - a, y - b) for (x, y), (a, b) in zip(p1, p2, strict=True)
            ]
            output_quantity = {
                "shape": [3, 3],
                "real": [kernel.fraction_decimal(x, 180) for x, y in exact_values],
                "imag": [kernel.fraction_decimal(y, 180) for x, y in exact_values],
            }
            reference = None
            polarization = True
            applicability = True
        else:
            raise GateError("unknown frozen connection kind")
        scale = scale_by_key[key_id(key)]
        # The two consistency kinds have no applicable polarization/phase
        # reference claim. Their gates certify the fixed relation and actual
        # frozen input mapping, with evidence_class and null reference retained.
        gates = {
            "units": (
                reference is None or reference["record"]["units"] == scale["units"]
            ),
            "shape": output_quantity["shape"] == shape,
            "frequency_axis": key["frequency_hz"] in fixture["frequency_hz"],
            "input_normalization": (
                reference is None or reference["record"]["inputs"] == fixture
            ),
            "polarization": polarization,
            "spatial_phase": (
                reference is None or fixture["spatial_phase"] == "exp(i*k dot target)"
            ),
            "frame": (reference is None or fixture["frame"] == "Cartesian"),
            "source_applicability": applicability,
            "reference_resolved": reference is None
            or reference["record"]["status"] == "resolved",
        }

        def evidence_value(value):
            if isinstance(value, complex):
                return {
                    "real": kernel.promote_binary64(value.real),
                    "imag": kernel.promote_binary64(value.imag),
                }
            if isinstance(value, float):
                return kernel.promote_binary64(value)
            if isinstance(value, (list, tuple)):
                return [evidence_value(v) for v in value]
            return value

        actual = {
            "quantity": output_quantity,
            "call": {
                "endpoint": endpoints[key["observable"]],
                "raw_production_quantities": raw_quantities,
                "kwargs": [
                    {k: evidence_value(v) for k, v in call.items()} for call in calls
                ],
                "gate_results": gates,
            },
        }
        if exact_values is not None:
            actual["exact_values"] = exact_values
        return assess_connection(
            key,
            fixture,
            actual,
            reference,
            scale,
            connection_budget,
            kernel,
            reference_pin=reference_pin if reference is not None else None,
            production_pin=production_pin,
        )

    return evaluate
