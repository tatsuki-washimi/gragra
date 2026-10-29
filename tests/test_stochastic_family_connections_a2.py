"""Synthetic transport/gate tests; no oracle or production evaluations."""

import hashlib
import importlib.util
import json
import math
from fractions import Fraction
from pathlib import Path

import pytest

MODULE = (
    Path(__file__).parent
    / "oracles/gen1/stochastic_formal/a2/v1/family_connections.py"
)


def load():
    spec = importlib.util.spec_from_file_location(
        "tests.oracles.gen1.stochastic_formal.a2.v1.family_connections", MODULE
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def kernel():
    p = Path(__file__).parent / "oracles/gen1/stochastic_acceptance.py"
    spec = importlib.util.spec_from_file_location("frozen_a2_test_kernel", p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def q(real, imag="0e+0"):
    k = kernel()
    return {
        "shape": [],
        "data": [
            {
                "real": k.fraction_decimal(Fraction(real)),
                "imag": k.fraction_decimal(Fraction(imag)),
            }
        ],
    }


def test_empty_context_fails_all_four_gates():
    result = load().assess({})
    assert result["gates"] == dict.fromkeys(
        ("numerical_procedure", "actual_x_transport", "SI", "coverage"), False
    )
    assert result["evidence"]["status"] == "unresolved"


def test_strict_neighbor_residual_includes_both_uncertainties():
    m = load()
    k = kernel()
    # Each synthetic per-value residual is <= its budget. The joint residual
    # fails once both independently recorded U values are included.
    check = m.neighbor_check(
        k,
        q("1.09"),
        q("0.91"),
        q("1"),
        q("1"),
        Fraction(2, 100),
        Fraction(2, 100),
        Fraction(1, 10),
        Fraction(1, 10),
    )
    assert check["accepted"] is False
    assert Fraction(check["residual_upper"]) >= Fraction(18, 100)


def test_neighbor_accepts_equal_residual_at_exact_budget():
    m = load()
    check = m.neighbor_check(
        kernel(),
        q("1.1"),
        q("0.9"),
        q("1"),
        q("1"),
        Fraction(),
        Fraction(),
        Fraction(1, 10),
        Fraction(1, 10),
    )
    assert check["accepted"] is True
    assert Fraction(check["decision_arithmetic"]) == 0


def test_neighbor_all_nine_complex_entries_are_used():
    m = load()
    matrix = {
        "shape": [3, 3],
        "data": [{"real": "0e+0", "imag": "0e+0"} for _ in range(9)],
    }
    shifted = {"shape": [3, 3], "data": [dict(z) for z in matrix["data"]]}
    shifted["data"][7]["imag"] = "1e-3"
    result = m.neighbor_check(
        kernel(),
        shifted,
        matrix,
        matrix,
        matrix,
        Fraction(),
        Fraction(),
        Fraction(1, 10000),
        Fraction(1, 10000),
    )
    assert result["accepted"] is False


def test_negative_uncertainty_refused():
    with pytest.raises(ValueError):
        load().neighbor_check(
            kernel(),
            q("1"),
            q("1"),
            q("1"),
            q("1"),
            Fraction(-1),
            Fraction(),
            Fraction(1),
            Fraction(1),
        )


def toy_fixture():
    rows = []
    for b in ("0.75", "18"):
        for eps in ("1e-8", "1e-6", "1e-4"):
            for sign in (-1, 1):
                x = Fraction(b) * (1 + sign * Fraction(eps))
                rows.append(
                    {
                        "nominal_exact_x": kernel().fraction_decimal(x),
                        "key_base": {"fixture_id": f"toy-{len(rows)}"},
                    }
                )
    return {
        "branch_boundaries": ["0.75", "18"],
        "branch_relative_eps": ["1e-8", "1e-6", "1e-4"],
        "fixtures": rows,
    }


def test_all_six_frozen_pairs_required():
    pairs = load().frozen_pairs(toy_fixture())
    assert len(pairs) == 6
    assert len({row["key_base"]["fixture_id"] for pair in pairs for row in pair}) == 12


def test_missing_neighbor_never_defaults_to_pass():
    f = toy_fixture()
    f["fixtures"].pop()
    with pytest.raises(ValueError, match="neighbor"):
        load().frozen_pairs(f)


def test_relaxed_or_changed_branch_grid_rejected():
    f = toy_fixture()
    f["branch_relative_eps"] = ["1e-7"]
    with pytest.raises(ValueError, match="frozen"):
        load().frozen_pairs(f)


def synthetic_context(tmp_path, selected=1, representation="Gfree", scalar=False):
    """Toy constant integrand values, never a scientific reference or API call.

    A file-backed toy function supplies genuine Python return-frame observations.
    It is not imported into gragra or used as a production endpoint.
    """
    m, k = load(), kernel()
    source = tmp_path / "toy_stochastic_acoustic.py"
    source.write_text("""import math
def _isotropic_depth_factor_small(x):
    return 0.2
def _isotropic_depth_factor_large(x):
    return 0.2
def isotropic_acoustic_depth_factor(value):
    number = float(value)
    if number == 0:
        return 1.0/3.0
    if number < .75:
        return _isotropic_depth_factor_small(number)
    if number > 18:
        return _isotropic_depth_factor_large(number)
    return 0.2
def underground_isotropic_acoustic_acceleration_csd_g_free(
    depth, c_s, frequencies, pressure_psd
):
    x = 4.0*math.pi*frequencies*depth/c_s
    factor = isotropic_acoustic_depth_factor(x)
    base = pressure_psd*factor/(c_s*frequencies)**2
    return base
""")
    namespace = {"__name__": m.MODULE}
    exec(compile(source.read_text(), str(source), "exec"), namespace)
    numbers = [
        Fraction(x)
        for x in (
            "0",
            "1e-8",
            "1e-6",
            "1e-4",
            ".01",
            ".1",
            ".5",
            ".75",
            "1",
            "2",
            "5",
            "10",
            "18",
            "30",
            "100",
            "1000",
        )
    ]
    for b in (Fraction(".75"), Fraction(18)):
        for e in map(Fraction, ("1e-8", "1e-6", "1e-4")):
            numbers += [b * (1 - e), b * (1 + e)]
    rows, expected, scalars, refs, scales, values, traces = [], [], [], {}, {}, {}, []
    pi = sum(m._pi_bounds(), Fraction()) / 2

    def quantity(pairs, shape):
        return {
            "shape": shape,
            "data": [
                {"real": k.fraction_decimal(a, 100), "imag": k.fraction_decimal(b, 100)}
                for a, b in pairs
            ],
        }

    def diagonal(base):
        return [
            (base * (2 if i == 8 else 1), Fraction())
            if i in (0, 4, 8)
            else (Fraction(), Fraction())
            for i in range(9)
        ]

    for i, x in enumerate(numbers):
        inputs = {
            "depth_m": k.fraction_decimal(
                x * Fraction(331) / (4 * pi * Fraction("1.7")), 110
            ),
            "frequency_hz": "1.7e+0",
            "g_si": "6.6743e-11",
            "pressure_psd_pa2_hz": "3.7e-3",
            "sound_speed_m_s": "3.31e+2",
        }
        key = {
            "family": "A2",
            "fixture_id": f"toy-{i}",
            "frequency_hz": "1.7e+0",
            "target_id": "underground_origin",
            "observable": "acceleration_csd",
            "set": "auxiliary" if x == 0 else "science",
        }
        row = {
            "key_base": key,
            "nominal_exact_x": k.fraction_decimal(x),
            "inputs": inputs,
            "pressure_convention": "surface_total",
            "psd_convention": m.CONVENTION,
            "input_assumptions": "exact_input",
            "frame": "Cartesian underground natural frame z vertical",
        }
        rows.append(row)
        base_scale = (
            Fraction(inputs["pressure_psd_pa2_hz"])
            / (Fraction(inputs["sound_speed_m_s"]) * Fraction(inputs["frequency_hz"]))
            ** 2
        )
        intensity = Fraction(1, 3) if x == 0 else Fraction(1, 5)
        physical_x = (
            4
            * pi
            * Fraction(inputs["frequency_hz"])
            * Fraction(inputs["depth_m"])
            / Fraction(inputs["sound_speed_m_s"])
        )
        part = {
            "theta_low60": "2e-1",
            "theta_high80_resolution124": ["2e-1"] * 3,
            "mu_high80_resolution124": ["2e-1"] * 3,
            "coordinate_disagreement": "0e+0",
            "reference_resolved": True,
        }
        trace = {
            **part,
            "exact_x": k.fraction_decimal(physical_x, 100),
            "actual_physical_x": k.fraction_decimal(physical_x, 100),
            "Gfree_public_adequacy": True,
            "SI_public_adequacy": True,
            "nominal_exact_x_reference": {
                **part,
                "exact_x": row["nominal_exact_x"],
                "scalar_public_adequacy": True,
            },
        }
        traces.append({"fixture_id": key["fixture_id"], "trace": trace})
        for rep in ("Gfree", "SI", "scalar"):
            item = (
                m._scalar_key(row)
                if rep == "scalar"
                else {**key, "representation": rep}
            )
            (scalars if rep == "scalar" else expected).append(item)
            factor = (
                Fraction(1)
                if rep == "scalar"
                else base_scale * (Fraction(inputs["g_si"]) ** 2 if rep == "SI" else 1)
            )
            shape = [] if rep == "scalar" else [3, 3]
            pairs = (
                [(intensity, Fraction())]
                if rep == "scalar"
                else diagonal(intensity * factor)
            )
            identity = m.key_id(item)
            uncertainty = {
                "kind": "estimate",
                "components": {
                    n: {
                        "value": k.fraction_decimal(factor * Fraction("1e-50"))
                        if n == "serialization"
                        else "0e+0",
                        "method": "Synthetic non-scientific test component",
                    }
                    for n in m.COMPONENTS
                },
                "certificate": None,
            }
            refs[identity] = {
                "key": item,
                "quantity": quantity(pairs, shape),
                "uncertainty": uncertainty,
                "status": "resolved",
                "inputs": {
                    "nominal_exact_x": row["nominal_exact_x"],
                    "frequency_hz": inputs["frequency_hz"],
                }
                if rep == "scalar"
                else inputs,
                "input_convention": "dimensionless_exact_nominal_x"
                if rep == "scalar"
                else m.CONVENTION,
                "units": "1"
                if rep == "scalar"
                else ("m^2 s^-4 Hz^-1" if rep == "SI" else "kg^2 m^-4 Hz^-1"),
                "procedure_hash": "0" * 64,
                "source_pins": {"toy": "not scientific evidence"},
            }
            scales[identity] = (
                k.fraction_decimal(Fraction(1, 3), 180, downward=True)
                if rep == "scalar"
                else k.fraction_decimal(
                    k.norm_bounds(quantity(diagonal(factor), shape), 180)[0],
                    180,
                    downward=True,
                )
            )
            if rep == "scalar":
                produced = [(Fraction.from_float(float(intensity)), Fraction())]
            else:
                base = (
                    float(inputs["pressure_psd_pa2_hz"])
                    * float(intensity)
                    / (float(inputs["sound_speed_m_s"]) * float(inputs["frequency_hz"]))
                    ** 2
                )
                produced = diagonal(
                    Fraction.from_float(
                        base * (float(inputs["g_si"]) ** 2 if rep == "SI" else 1)
                    )
                )
            values[identity] = quantity(produced, shape)
    artifact = tmp_path / "stderr.jsonl"
    artifact.write_text("".join(json.dumps(t) + "\n" for t in traces))
    pins = {"source_files": {m.SOURCE: hashlib.sha256(source.read_bytes()).hexdigest()}}
    item = (
        m._scalar_key(rows[selected])
        if scalar
        else {**rows[selected]["key_base"], "representation": representation}
    )
    context = {
        "key": item,
        "reference": refs[m.key_id(item)],
        "production": values[m.key_id(item)],
        "all_references": refs,
        "source_scales": scales,
        "kernel": k,
        "frozen_case": rows[selected],
        "frozen_fixture": {
            "family": "A2",
            "fixtures": rows,
            "branch_boundaries": ["0.75", "18"],
            "branch_relative_eps": ["1e-8", "1e-6", "1e-4"],
            "expected_keys": expected,
            "expected_scalar_keys": scalars,
        },
        "frozen_procedure": {
            "family": "A2",
            "input_convention": m.CONVENTION + ": toy fixture",
            "uncertainty_kind": "estimate",
            "numerical_procedure": {
                "dps": [60, 80],
                "resolution": [1, 2, 4],
                "digits": 40,
                "maxdegree": 8,
                "formulations": ["theta", "mu"],
            },
        },
        "frozen_budget": {
            "family": "A2",
            "absolute_rate": "1e-13",
            "relative_rate": "1e-10",
        },
        "reference_artifacts": {
            "stderr.jsonl": {
                "path": str(artifact),
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "kind": "toy trace",
            }
        },
        "production_source_pins": pins,
        "production_source_reviews": {
            "schema": "gen1-stochastic-production-source-admission-v1",
            "family": "A2",
            "decision": "PASS",
            "pins": pins,
            "reviews": dict.fromkeys(("science", "numerical", "provenance"), "toy"),
        },
        "actual": lambda key: values[m.key_id(key)],
    }

    def fresh(key):
        row = next(r for r in rows if r["key_base"]["fixture_id"] == key["fixture_id"])
        if key["observable"] == "isotropic_depth_factor":
            namespace["isotropic_acoustic_depth_factor"](
                float(Fraction(row["nominal_exact_x"]))
            )
        else:
            p = row["inputs"]
            namespace["underground_isotropic_acoustic_acceleration_csd_g_free"](
                float(p["depth_m"]),
                float(p["sound_speed_m_s"]),
                float(p["frequency_hz"]),
                float(p["pressure_psd_pa2_hz"]),
            )
        return values[m.key_id(key)]

    pins.update(
        {
            "family": "A2",
            "production_commit": "0" * 40,
            "production_tree": "1" * 40,
            "frozen_artifacts_sha256": {},
            "reference_seal_sha256": "2" * 64,
            "reference_records_sha256": "3" * 64,
            "expected_keys_sha256": "4" * 64,
        }
    )

    def artifact_pin(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "kind": "synthetic non-scientific source packet",
        }

    proof = artifact_pin("toy-proof.json", {"scope": "SYNTHETIC_ONLY"})
    reviews = {}
    for role in ("science", "numerical", "provenance"):
        reviews[role] = artifact_pin(
            role + ".json",
            {
                "schema": "gen1-stochastic-production-source-review-v1",
                "family": "A2",
                "role": role,
                "reviewer_id": "toy-" + role,
                "decision": "PASS",
                "pins": pins,
                "original_review": proof,
                # Source-binding claims are intentionally empty in the v2
                # reader contract; runtime gates are evaluated from evidence.
                "claim_attestations": {},
            },
        )
    context["production_source_reviews"] = {
        "reviews": reviews,
        "admission": artifact_pin(
            "admission.json",
            {
                "schema": "gen1-stochastic-production-source-admission-v1",
                "family": "A2",
                "decision": "PASS",
                "pins": pins,
                "reviews": reviews,
            },
        ),
    }
    context["actual_fresh"] = fresh
    context["_toy_namespace"] = namespace
    context["_toy_source_path"] = source
    return context, values, artifact


@pytest.mark.parametrize(
    "selected,representation,scalar",
    [(1, "Gfree", False), (0, "SI", False), (0, "Gfree", True)],
)
def test_complete_synthetic_context_all_routes(
    tmp_path, selected, representation, scalar
):
    context, _, _ = synthetic_context(tmp_path, selected, representation, scalar)
    result = load().assess(context)
    assert result["gates"] == dict.fromkeys(load().GATES, True)
    assert len(result["evidence"]["neighbor_checks"]) == 18
    assert len(result["evidence"]["representation_checks"]) == 28
    assert len(result["evidence"]["asymptotic_diagnostics"]) == 3
    from tests.oracles.gen1.stochastic_formal.common.transport import (
        canonical_bytes,
        strict_json,
    )

    assert strict_json(canonical_bytes(result)) == result
    for route in ("depth", "scalar"):
        for observed in result["evidence"]["observed_production_path"][route]:
            assert all(isinstance(value, str) for value in observed.values())


@pytest.mark.parametrize(
    "mutation", ["coverage", "inputs", "method", "uncertainty", "trace", "source"]
)
def test_metadata_mutations_fail_closed(tmp_path, mutation):
    context, _, artifact = synthetic_context(tmp_path)
    if mutation == "coverage":
        context["all_references"].pop(next(iter(context["all_references"])))
    elif mutation == "inputs":
        context["reference"]["inputs"] = {"nominal_exact_x": "1e-8"}
    elif mutation == "method":
        context["frozen_procedure"]["numerical_procedure"]["dps"] = [40, 60]
    elif mutation == "uncertainty":
        context["reference"]["uncertainty"]["components"]["formulation"]["value"] = (
            "1e-40"
        )
    elif mutation == "trace":
        artifact.write_text(artifact.read_text() + "\n")
    else:
        context["production_source_reviews"]["decision"] = "REVISE"
    with pytest.raises(ValueError):
        load().assess(context)


def test_source_binding_cannot_hide_wrong_numeric_values(tmp_path):
    context, values, _ = synthetic_context(tmp_path)
    context["production"]["data"][8]["real"] = "1e+0"
    result = load().assess(context)
    assert result["gates"]["numerical_procedure"] is False
    assert result["gates"]["SI"] is False


def test_science_review_rejects_runtime_gate_names_in_claim_attestations(tmp_path):
    context, _, _ = synthetic_context(tmp_path)
    packet = context["production_source_reviews"]
    science_pin = packet["reviews"]["science"]
    science_path = Path(science_pin["path"])
    review = json.loads(science_path.read_text())
    proof_report = review["original_review"]
    review["claim_attestations"] = {
        "numerical_procedure": {
            "decision": "PASS",
            "scope": "SOURCE_MODEL_BINDING_ONLY",
            "proof_report": proof_report,
        }
    }
    science_path.write_text(json.dumps(review))
    science_pin["sha256"] = hashlib.sha256(science_path.read_bytes()).hexdigest()
    admission_pin = packet["admission"]
    admission_path = Path(admission_pin["path"])
    admission = json.loads(admission_path.read_text())
    admission["reviews"] = packet["reviews"]
    admission_path.write_text(json.dumps(admission))
    admission_pin["sha256"] = hashlib.sha256(admission_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="runtime gates"):
        load().assess(context)


def test_fresh_callback_without_observed_path_is_refused(tmp_path):
    context, values, _ = synthetic_context(tmp_path)
    context["actual_fresh"] = lambda key: values[load().key_id(key)]
    with pytest.raises(ValueError, match="genuine"):
        load().assess(context)


def test_observed_physical_x_cannot_be_replaced_by_nominal_x(tmp_path):
    context, values, _ = synthetic_context(tmp_path)
    fn = context["_toy_namespace"][
        "underground_isotropic_acoustic_acceleration_csd_g_free"
    ]
    p = context["frozen_case"]["inputs"]

    def wrong(key):
        fn(
            2 * float(p["depth_m"]),
            float(p["sound_speed_m_s"]),
            float(p["frequency_hz"]),
            float(p["pressure_psd_pa2_hz"]),
        )
        return values[load().key_id(key)]

    context["actual_fresh"] = wrong
    assert load().assess(context)["gates"]["actual_x_transport"] is False


def test_wrong_si_power_is_rejected(tmp_path):
    context, values, _ = synthetic_context(tmp_path)
    k = context["kernel"]
    for identity, quantity in values.items():
        if identity[5] == "SI":
            for cell in quantity["data"]:
                cell["real"] = k.fraction_decimal(
                    k.decimal_fraction(cell["real"]) * Fraction("6.6743e-11") ** 2
                )
    result = load().assess(context)
    assert result["gates"]["SI"] is False
    assert result["gates"]["numerical_procedure"] is False


def test_route_disagreement_is_a_separate_public_adequacy_gate(tmp_path):
    context, _, artifact = synthetic_context(tmp_path)
    records = [json.loads(line) for line in artifact.read_text().splitlines()]
    records[1]["trace"]["coordinate_disagreement"] = "1e-20"
    artifact.write_text("".join(json.dumps(r) + "\n" for r in records))
    context["reference_artifacts"]["stderr.jsonl"]["sha256"] = hashlib.sha256(
        artifact.read_bytes()
    ).hexdigest()
    result = load().assess(context)
    assert result["gates"]["numerical_procedure"] is False
    assert result["evidence"]["reference_public_adequacy"] is False


def test_changed_observed_source_restores_trace_even_on_failure(tmp_path):
    import sys

    context, _, _ = synthetic_context(tmp_path)
    context["_toy_source_path"].write_text("changed bytes\n")
    with pytest.raises(ValueError, match="source bytes"):
        load().assess(context)
    assert sys.gettrace() is None


def test_unknown_current_key_never_defaults_to_pass(tmp_path):
    context, _, _ = synthetic_context(tmp_path, scalar=True)
    context["key"] = {**context["key"], "representation": "SI"}
    with pytest.raises(ValueError, match="84"):
        load().assess(context)


@pytest.mark.parametrize(
    "value",
    [
        0.0,
        -0.0,
        0.2,
        math.nextafter(0.0, 1.0),
        1e308,
        math.nextafter(0.75, 0.0),
        math.nextafter(0.75, 1.0),
    ],
)
def test_observed_evidence_preserves_exact_finite_binary64(value):
    m = load()
    result = m._captured_evidence(
        {"branches": [], "depth": [], "scalar": [{"number": value}]}
    )
    text = result["scalar"][0]["number"]
    assert kernel().decimal_fraction(text) == Fraction.from_float(value)


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_nonfinite_observed_evidence_cannot_be_serialized(value):
    with pytest.raises(ValueError, match="finite binary64"):
        load()._captured_evidence(
            {"branches": [], "depth": [], "scalar": [{"number": value}]}
        )
