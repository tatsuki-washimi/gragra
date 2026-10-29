"""Bounded saved-data reconciliation; no expensive oracle field generation."""

import copy
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import mpmath as mp
import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "independent_trace_audit",
    Path(__file__).with_name("audit_frozen_refinement_traces.py"),
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def capture(value):
    if isinstance(value, np.ndarray):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": capture(value.tolist()),
        }
    if isinstance(value, np.generic):
        if value.dtype.kind == "c":
            return {
                "real": capture(value.real),
                "imag": capture(value.imag),
                "dtype": str(value.dtype),
            }
        return {
            "decimal": np.format_float_scientific(
                value, unique=True, precision=21, trim="k"
            ),
            "dtype": str(value.dtype),
        }
    if isinstance(value, mp.mpc):
        return {"real": capture(value.real), "imag": capture(value.imag)}
    if isinstance(value, mp.mpf):
        return {"decimal": mp.nstr(value, 100), "mpf": list(value._mpf_)}
    if isinstance(value, (tuple, list)):
        return [capture(item) for item in value]
    return value


def surface_case(route):
    hp = route == "hp"
    dtype = mp.mpc if hp else np.clongdouble
    with mp.workdps(80 if hp else 15):
        vectors = []
        for shift in ("0", "1e-18", "1.01e-18"):
            if hp:
                with mp.workdps(50):
                    vectors.append(
                        tuple(
                            mp.mpc("1e-13", "2e-13") + mp.mpf(shift) for _ in range(13)
                        )
                    )
            else:
                vectors.append(
                    np.asarray(
                        [
                            dtype("1e-13") + dtype(1j) * dtype("2e-13") + dtype(shift)
                            for _ in range(13)
                        ],
                        dtype=dtype,
                    )
                )
        final = vectors[-1]
        if hp:
            final = tuple(value + mp.mpf("1e-24") for value in final)
        section = AUDIT.SECTIONS["acceleration"]
        pair = AUDIT.observed_pairs([vector[section] for vector in vectors], ld=not hp)
        quadrature = (10 if hp else np.longdouble(10)) * max(pair)
        component = (
            "high_precision_surface_quadrature"
            if hp
            else "surface_quadrature_refinement"
        )
        reference = {
            "values_si": [[complex(v).real, complex(v).imag] for v in final[section]],
            "uncertainty": {
                "method": "high precision Newton surface / independent refinement"
                if hp
                else "ordinary",
                "components": [
                    (component, float(quadrature)),
                    ("float_conversion", 0.0),
                ],
            },
        }
        check = {
            "component_name": component,
            "formula_float_si": float(quadrature),
            "frozen_component_si": float(quadrature),
            "matches_frozen_float": True,
        }
        trace = {
            "status": "numerical",
            "route": "mpmath_fallback" if hp else "numpy.longdouble",
            "evaluation_inputs": {
                "quadrature_orders_at_50_digits": [16, 24, 32],
                "final_order_at_80_digits": 32,
                "digits": [50, 80],
            }
            if hp
            else {"quadrature_orders": [16, 24, 32], "backend": "numpy.longdouble"},
            "quadrature_differences_si": {
                "coarse_to_middle": capture(pair[0])
                if not hp
                else mp.nstr(pair[0], 100),
                "middle_to_fine": capture(pair[1]) if not hp else mp.nstr(pair[1], 100),
                "ten_max": capture(quadrature) if not hp else mp.nstr(quadrature, 100),
            },
            "derivative_trend": {
                "classification": "not_applicable",
                "reason": "analytic kernel",
            },
            "quadrature_trend": {
                "classification": "decreasing",
                "nondecreasing": False,
                "coarse_to_middle_si": np.format_float_scientific(
                    pair[0], unique=True, precision=21, trim="k"
                )
                if not hp
                else mp.nstr(pair[0], 100),
                "middle_to_fine_si": np.format_float_scientific(
                    pair[1], unique=True, precision=21, trim="k"
                )
                if not hp
                else mp.nstr(pair[1], 100),
                "independent_floor_si": "0.0",
            },
            "independent_floor_components_si": {"float_conversion": "0.0"},
            "independent_floor_component_reasons": {
                "float_conversion": AUDIT.INDEPENDENT_FLOOR_COMPONENT_REASONS[
                    "float_conversion"
                ]
            },
            "frozen_component_checks": {"quadrature": check},
            "captured_final_values_match_frozen": True,
            "captured_final_values_si": capture(final[section]),
        }
        fixture = {"target_m": [0.0, 0.0, 1.25], "lengths_m": [2.0, 2.0, 2.0]}
        if route == "source":
            fixture = {
                "target_m": [0.0, 0.0, 1.25],
                "shape": "cuboid",
                "geometry": {"lengths_m": [2.0, 2.0, 2.0]},
            }
        calls = []
        for order, vector in zip((16, 24, 32), vectors, strict=True):
            absolute = (
                tuple(mp.mpf("1e-12") for _ in range(13))
                if hp
                else np.full(13, np.longdouble("1e-12"), dtype=np.longdouble)
            )
            output = capture((vector, absolute, 10) if hp else (vector, absolute))
            kernel = (
                "surface_high_precision_evaluate"
                if hp
                else "source_surface_values"
                if route == "source"
                else "surface_newton_evaluate"
            )
            if route == "source":
                output = {
                    "values_and_absolute": output,
                    "geometry_summary": {
                        "characteristic_m": capture(np.longdouble(2)),
                        "minimum_gap_m": capture(np.longdouble(".25")),
                    },
                }
            calls.append(
                {
                    "kernel": kernel,
                    "inputs": {
                        "order": order,
                        "digits": 50,
                        "fixture": fixture,
                        "target_m": capture(
                            np.asarray(fixture["target_m"], dtype=np.longdouble)
                        ),
                    },
                    "output": output,
                }
            )
        if hp:
            calls.append(
                {
                    "kernel": "surface_high_precision_evaluate",
                    "inputs": {"order": 32, "digits": 80, "fixture": fixture},
                    "output": capture((final, absolute, 10)),
                }
            )
            trace["fixed_order_precision_change_si"] = mp.nstr(
                AUDIT.vector_difference(final[section], vectors[-1][section]), 100
            )
        converted = tuple(complex(value) for value in final[section])
        numeric = (
            tuple(mp.mpc(value) for value in converted)
            if hp
            else np.asarray(converted, dtype=np.clongdouble)
        )
        conversion = AUDIT.vector_difference(final[section], numeric, ld=not hp)
        components = [
            (component, float(quadrature)),
            ("float_conversion", float(conversion)),
        ]
        if hp:
            precision = AUDIT.vector_difference(final[section], vectors[-1][section])
            components.append(("50_80_digit_precision", float(10 * precision)))
            components.extend(
                [
                    (
                        "high_precision_roundoff_estimate",
                        float(
                            1000
                            * 10
                            * mp.mpf(10) ** -49
                            * AUDIT.oracle("sphere_reference").vector_norm(
                                absolute[section]
                            )
                        ),
                    ),
                    ("root_normalization_precision", 0.0),
                ]
            )
        else:
            components.append(
                (
                    "longdouble_roundoff_estimate",
                    float(
                        256
                        * np.finfo(np.longdouble).eps
                        * (1 + np.longdouble(2) / np.longdouble(".25"))
                        * AUDIT.oracle("surface_reference").norm_ld(absolute[section])
                    ),
                )
            )
            if route != "source":
                components.append(("root_normalization_precision", 0.0))
        reference["uncertainty"]["components"] = components
        floors = {
            key: mp.mpf(value)
            for key, value in components
            if key in AUDIT.INDEPENDENT_FLOOR_COMPONENT_REASONS
        }
        trace["independent_floor_components_si"] = {
            key: mp.nstr(value, 100) for key, value in floors.items()
        }
        trace["independent_floor_component_reasons"] = {
            key: AUDIT.INDEPENDENT_FLOOR_COMPONENT_REASONS[key] for key in floors
        }
        trace["quadrature_trend"]["independent_floor_si"] = mp.nstr(
            mp.fsum(floors.values()), 100
        )
    metadata = {
        "orders": [16, 24, 32],
        "minimum_surface_gap_m": ".25",
        "longdouble_epsilon": str(np.finfo(np.longdouble).eps),
        "fallback_surface_nodes": 10,
    }
    return fixture, trace, reference, metadata, calls


@pytest.mark.parametrize("route", ["ordinary", "source", "hp"])
def test_surface_saved_routes_reconcile_without_field_recomputation(route):
    fixture, trace, reference, metadata, calls = surface_case(route)
    AUDIT.surface_trace(
        "source" if route == "source" else "cuboid",
        fixture,
        "acceleration",
        trace,
        reference,
        metadata,
        calls,
    )


@pytest.mark.parametrize("route", ["ordinary", "source", "hp"])
@pytest.mark.parametrize(
    "defect", ["missing_pair", "wrong_ten_max", "none_component", "wrong_floor"]
)
def test_surface_saved_routes_reject_incomplete_or_mismatched_evidence(route, defect):
    fixture, trace, reference, metadata, calls = surface_case(route)
    if defect == "missing_pair":
        trace.pop("quadrature_differences_si")
    elif defect == "wrong_ten_max":
        trace["quadrature_differences_si"]["ten_max"] = "1.0"
    elif defect == "none_component":
        trace["frozen_component_checks"]["quadrature"] = None
    else:
        trace["independent_floor_components_si"]["explicit_truncation_tail"] = "1.0"
    with pytest.raises((ValueError, KeyError)):
        AUDIT.surface_trace(
            "source" if route == "source" else "cuboid",
            fixture,
            "acceleration",
            trace,
            reference,
            metadata,
            calls,
        )


@pytest.fixture
def external_assets():
    root = os.environ.get("GRAGRA_TRACE_AUDIT_ASSETS")
    if not root:
        pytest.skip(
            "set GRAGRA_TRACE_AUDIT_ASSETS to the pinned external production run root"
        )
    return Path(root)


def saved_plane(root):
    row = AUDIT.read_rows(root / "l6/plane-static-frozen-20261001/references.jsonl")[0]
    record = AUDIT.read_rows(
        root
        / "l6/refinement-traces/observer-resume-semantic-repair-20261002-04"
        / "plane/traces.jsonl"
    )[0]
    return row, record


def test_actual_pinned_plane_trace_and_all_module_pins(external_assets):
    AUDIT.check_oracle_pins()
    row, record = saved_plane(external_assets)
    result = AUDIT.audit_record("plane", 0, row, record)
    assert result["audit_unresolved_trends"] == []


@pytest.mark.parametrize(
    "defect",
    [
        "missing_pair",
        "wrong_ten_max",
        "none_component",
        "wrong_trend",
        "wrong_floor",
        "missing_observable",
        "bad_capture_count",
        "wrong_final",
        "wrong_fixture",
    ],
)
def test_actual_mp_trace_rejects_deliberate_semantic_defects(external_assets, defect):
    row, original = saved_plane(external_assets)
    record = copy.deepcopy(original)
    trace = record["observable_traces"]["acceleration"]
    if defect == "missing_pair":
        trace.pop("derivative_differences_si")
    elif defect == "wrong_ten_max":
        trace["derivative_differences_si"]["ten_max"] = "1.0"
    elif defect == "none_component":
        trace["frozen_component_checks"]["derivative"] = None
    elif defect == "wrong_trend":
        trace["derivative_trend"]["middle_to_fine_si"] = "1.0"
    elif defect == "wrong_floor":
        trace["independent_floor_components_si"]["float_conversion"] = "1.0"
    elif defect == "missing_observable":
        record["observable_traces"].pop("gradient")
    elif defect == "bad_capture_count":
        record["original_entrypoint"]["capture_call_count"] = 0
    elif defect == "wrong_final":
        trace["captured_final_values_si"][0] = capture(mp.mpf(1))
    else:
        record["fixture"]["target_m"][0] = 1.0
    with pytest.raises((ValueError, KeyError)):
        AUDIT.audit_record("plane", 0, row, record)


def test_three_actual_zero_and_active_force_records_preserve_labels(external_assets):
    zero, polynomial = AUDIT.certificate_inputs(external_assets / "l6")
    root = external_assets / "l6/observer-semantic-scientist-review-20261002/replays"
    for run, dataset, index in [
        ("zero-sphere-gaussian-001", "sphere", 0),
        ("zero-sphere-gaussian-001", "noncavity", 0),
        ("active-force-001", "noncavity", 24),
    ]:
        row = AUDIT.read_rows(
            external_assets / "l6" / AUDIT.DATASETS[dataset] / "references.jsonl"
        )[index]
        record = AUDIT.read_rows(root / run / dataset / "traces.jsonl")[0]
        before = json.dumps(record, sort_keys=True)
        result = AUDIT.audit_record(dataset, index, row, record, zero, polynomial)
        assert result["audit_unresolved_trends"] == []
        assert json.dumps(record, sort_keys=True) == before


@pytest.mark.parametrize(
    "dataset", ["sphere", "high_sphere", "noncavity", "plane", "source"]
)
def test_completed_actual_datasets_have_exact_coverage_and_consistent_routes(
    external_assets, dataset
):
    zero, polynomial = AUDIT.certificate_inputs(external_assets / "l6")
    result = AUDIT.audit_dataset(
        dataset,
        external_assets / "l6",
        external_assets / "l6/refinement-traces/observer-full-d43-20261002",
        zero,
        polynomial,
    )
    assert result["case_count"] == AUDIT.COUNTS[dataset]
    assert result["errors"] == []
    assert result["unresolved_cases"] == 0
    if dataset == "high_sphere":
        assert result["original_unresolved_cases"] == 4
        assert (
            sum(len(r["audit_certificate_supplements"]) for r in result["records"]) == 4
        )


def test_high_degree_zero_supplement_does_not_cover_derivatives_or_other_fixture(
    external_assets,
):
    row = AUDIT.read_rows(
        external_assets / "l6/sphere-high-degree-final-20261001/references.jsonl"
    )[8]
    record = next(
        r
        for r in AUDIT.read_rows(
            external_assets
            / "l6/refinement-traces/observer-full-d43-20261002/high_sphere/traces.jsonl"
        )
        if r["source_row_index"] == 8
    )
    result = AUDIT.audit_record("high_sphere", 8, row, record)
    certificate = result["audit_certificate_supplements"][0]
    assert certificate["observable"] == "potential"
    assert certificate["derivative_observables_certified"] is False
    assert result["status"] == "unresolved_trend"
    altered = copy.deepcopy(row)
    altered["fixture"]["target_m"][1] = 1e-200
    with pytest.raises(ValueError, match="full fixture"):
        AUDIT.high_sphere_scalar_zero_supplement(
            "high_sphere", 8, altered, record["observable_traces"]
        )


def test_aggregate_count_is_frozen_1178():
    assert sum(AUDIT.COUNTS.values()) == 1178


@pytest.mark.parametrize("defect", ["missing", "duplicate"])
def test_finished_dataset_rejects_incomplete_or_duplicate_rows_even_with_new_hashes(
    external_assets, tmp_path, defect
):
    original = external_assets / (
        "l6/refinement-traces/observer-resume-semantic-repair-20261002-04/plane"
    )
    copied = tmp_path / "plane"
    shutil.copytree(original, copied)
    traces = copied / "traces.jsonl"
    data = traces.read_bytes()
    traces.write_bytes(b"" if defect == "missing" else data + data)
    digest, size = AUDIT.sha256(traces), traces.stat().st_size
    checkpoint = AUDIT.read_json(copied / "checkpoint.json")
    for key in ("committed_trace_sha256", "traces_jsonl_sha256"):
        checkpoint[key] = digest
    for key in ("committed_trace_bytes", "trace_bytes"):
        checkpoint[key] = size
    (copied / "checkpoint.json").write_text(json.dumps(checkpoint))
    run = AUDIT.read_json(copied / "run-evidence.json")
    run["traces_jsonl_sha256"] = digest
    (copied / "run-evidence.json").write_text(json.dumps(run))
    zero, polynomial = AUDIT.certificate_inputs(external_assets / "l6")
    with pytest.raises(ValueError, match="duplicate/missing/extra"):
        AUDIT.audit_dataset(
            "plane",
            external_assets / "l6",
            tmp_path,
            zero,
            polynomial,
            allow_selected=True,
        )


@pytest.mark.parametrize("route", ["ordinary", "source", "hp"])
@pytest.mark.parametrize(
    "defect", ["missing_absolute", "huge_absolute", "negative_absolute"]
)
def test_surface_absolute_evidence_is_required_and_reconciled(route, defect):
    fixture, trace, reference, metadata, calls = surface_case(route)
    event = calls[-1]
    output = (
        event["output"]["values_and_absolute"] if route == "source" else event["output"]
    )
    if defect == "missing_absolute":
        del output[1]
    else:
        value = "-1.0" if defect == "negative_absolute" else "1e99"
        absolute = (
            tuple(mp.mpf(value) for _ in range(13))
            if route == "hp"
            else np.full(13, np.longdouble(value), dtype=np.longdouble)
        )
        output[1] = capture(absolute)
    with pytest.raises((ValueError, KeyError)):
        AUDIT.surface_trace(
            "source" if route == "source" else "cuboid",
            fixture,
            "acceleration",
            trace,
            reference,
            metadata,
            calls,
        )


@pytest.mark.parametrize(
    "defect",
    ["missing_absolute", "huge_absolute", "missing_geometry", "wrong_geometry"],
)
def test_actual_source_row12_rejects_missing_or_contradictory_roundoff_evidence(
    external_assets, defect
):
    row = AUDIT.read_rows(
        external_assets / "l6/source-primary-20261001/references.jsonl"
    )[12]
    record = next(
        r
        for r in AUDIT.read_rows(
            external_assets
            / "l6/refinement-traces/observer-full-d43-20261002/source/traces.jsonl"
        )
        if r["source_row_index"] == 12
    )
    altered = copy.deepcopy(record)
    for call in altered["original_entrypoint"]["captured_calls"]:
        if call["kernel"] != "source_surface_values":
            continue
        output = call["output"]
        if defect == "missing_absolute":
            output["values_and_absolute"] = output["values_and_absolute"][:1]
        elif defect == "huge_absolute":
            output["values_and_absolute"][1] = capture(
                np.full(13, np.longdouble("1e99"))
            )
        elif defect == "missing_geometry":
            output.pop("geometry_summary")
        else:
            output["geometry_summary"]["minimum_gap_m"] = capture(np.longdouble("1e99"))
    with pytest.raises((ValueError, KeyError)):
        AUDIT.audit_record("source", 12, row, altered)


@pytest.mark.parametrize(
    "defect",
    [
        "missing_parents",
        "empty_run_observations",
        "wrong_worker_pid",
        "missing_history_phase",
    ],
)
def test_parent_and_worker_observations_reconcile_with_actual_resume_history(
    external_assets, tmp_path, defect
):
    original = (
        external_assets
        / "l6/refinement-traces/observer-resume-semantic-repair-20261002-04/plane"
    )
    directory = tmp_path / "plane"
    shutil.copytree(original, directory)
    manifest = AUDIT.read_json(directory / "manifest.json")
    checkpoint = AUDIT.read_json(directory / "checkpoint.json")
    run = AUDIT.read_json(directory / "run-evidence.json")
    records = AUDIT.read_rows(directory / "traces.jsonl")
    assert (
        AUDIT.audit_process_evidence(directory, manifest, checkpoint, run, records, [0])
        == []
    )
    if defect == "missing_parents":
        for data in (manifest, run):
            data["process_state"]["observations"] = [
                o
                for o in data["process_state"]["observations"]
                if o["process"] == "worker"
            ]
    elif defect == "empty_run_observations":
        run["process_state"]["observations"] = []
    elif defect == "wrong_worker_pid":
        for data in (manifest, run):
            for observation in data["process_state"]["observations"]:
                if observation["process"] == "worker":
                    observation["pid"] = 1
    else:
        # Keep a self-consistent amended history digest, but remove the latest
        # start so the final observation cannot be authenticated by a start.
        history = run["invocation_history"]
        history.pop(-2)
        path = directory / "run-invocations.jsonl"
        path.write_text("".join(json.dumps(item) + "\n" for item in history))
        run["run_invocations_jsonl_sha256"] = AUDIT.sha256(path)
    with pytest.raises((ValueError, KeyError)):
        AUDIT.audit_process_evidence(directory, manifest, checkpoint, run, records, [0])


def test_cube_actual_complete_matrix_reconstructs_ld_hp_uncertainty(external_assets):
    zero, polynomial = AUDIT.certificate_inputs(external_assets / "l6")
    result = AUDIT.audit_dataset(
        "cuboid",
        external_assets / "l6",
        external_assets / "l6/refinement-traces/observer-full-d43-20261002",
        zero,
        polynomial,
    )
    assert result["case_count"] == 165 and result["errors"] == []


def test_cylinder_root_precision_fails_closed_without_supplement():
    fixture = {"order": 0, "radial_index": 1, "radius_m": 1.0, "axial_index": 0}
    with pytest.raises(ValueError, match="evidence unavailable"):
        AUDIT.root_precision(fixture, ())


@pytest.mark.parametrize("defect", ["missing_pin", "wrong_pin", "other_dataset"])
def test_mixed_observer_provenance_requires_explicit_matched_cylinder_pins(
    tmp_path, defect
):
    source = tmp_path / "observer.py"
    source.write_text("independent bounded test source\n")
    digest = AUDIT.sha256(source)
    roots = ["cylinder=" + str(tmp_path / "new-run")]
    sources = ["cylinder=" + str(source)]
    digests = ["cylinder=" + digest]
    if defect == "missing_pin":
        digests = []
    elif defect == "wrong_pin":
        digests = ["cylinder=" + "0" * 64]
    else:
        roots = ["source=" + str(tmp_path)]
    with pytest.raises(ValueError):
        AUDIT.dataset_overrides(roots, sources, digests)


def test_mixed_observer_provenance_is_not_a_blanket_source_substitution(tmp_path):
    source = tmp_path / "observer.py"
    source.write_text("independent bounded test source\n")
    digest = AUDIT.sha256(source)
    roots, sources, digests = AUDIT.dataset_overrides(
        ["cylinder=" + str(tmp_path / "new-run")],
        ["cylinder=" + str(source)],
        ["cylinder=" + digest],
    )
    assert set(roots) == set(sources) == set(digests) == {"cylinder"}
    assert (
        AUDIT.OBSERVER_SHA256
        == "d43be29b355c1f6e4144c396d9f2b576a2972e61d8eff0ea9edd78f38855c446"
    )


@pytest.mark.parametrize("defect", ["absolute", "parent_observations"])
def test_cli_fails_on_rehashed_semantically_corrupt_completed_dataset(
    external_assets, tmp_path, defect
):
    dataset = "source" if defect == "absolute" else "plane"
    original = (
        external_assets / "l6/refinement-traces/observer-full-d43-20261002" / dataset
    )
    directory = tmp_path / dataset
    shutil.copytree(original, directory)
    if defect == "absolute":
        records = AUDIT.read_rows(directory / "traces.jsonl")
        record = next(r for r in records if r["source_row_index"] == 12)
        for call in record["original_entrypoint"]["captured_calls"]:
            if call["kernel"] == "source_surface_values":
                call["output"]["values_and_absolute"][1] = capture(
                    np.full(13, np.longdouble("1e99"))
                )
        traces = directory / "traces.jsonl"
        traces.write_text("".join(json.dumps(r) + "\n" for r in records))
        digest, size = AUDIT.sha256(traces), traces.stat().st_size
        checkpoint = AUDIT.read_json(directory / "checkpoint.json")
        for key in ("committed_trace_sha256", "traces_jsonl_sha256"):
            checkpoint[key] = digest
        for key in ("committed_trace_bytes", "trace_bytes"):
            checkpoint[key] = size
        (directory / "checkpoint.json").write_text(json.dumps(checkpoint))
        run = AUDIT.read_json(directory / "run-evidence.json")
        run["traces_jsonl_sha256"] = digest
        (directory / "run-evidence.json").write_text(json.dumps(run))
    else:
        manifest = AUDIT.read_json(directory / "manifest.json")
        run = AUDIT.read_json(directory / "run-evidence.json")
        for obj in (manifest, run):
            obj["process_state"]["observations"] = [
                o
                for o in obj["process_state"]["observations"]
                if o["process"] == "worker"
            ]
        (directory / "manifest.json").write_text(json.dumps(manifest))
        run["dataset_manifest_sha256"] = AUDIT.sha256(directory / "manifest.json")
        (directory / "run-evidence.json").write_text(json.dumps(run))
    output = tmp_path / "audit.json"
    # Run the actual CLI in a clean interpreter: other collected tests and the
    # source-pinned runner may already have imported production in this process.
    child_environment = os.environ.copy()
    child_environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            str(Path(AUDIT.__file__).resolve()),
            "--l6-root",
            str(external_assets / "l6"),
            "--replay-root",
            str(tmp_path),
            "--observer-source",
            os.environ.get(
                "GRAGRA_TRACE_OBSERVER_SOURCE",
                str(
                    Path(AUDIT.read_json(original / "manifest.json")["source_tree"])
                    / "tests/oracles/gen1/record_frozen_refinement_traces.py"
                ),
            ),
            "--output",
            str(output),
            "--datasets",
            dataset,
        ],
        cwd=tmp_path,
        env=child_environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2, completed.stdout + completed.stderr
    cli_result = json.loads(completed.stdout)
    assert cli_result["status"] == "failed"
    assert cli_result["full_gate_complete"] is False
    result = AUDIT.read_json(output)
    assert result["status"] == "failed" and result["full_gate_complete"] is False
    assert result["production_modules"] == []
    assert result["auditor_sha256"] == AUDIT.sha256(AUDIT.__file__)


def test_actual_root_only_supplement_has_strict_source_and_all_fixture_bindings(
    external_assets, tmp_path
):
    path = (
        external_assets
        / "l6/frozen-trace-audit-repair-20261002/cylinder-root-normalization-first.json"
    )
    entries = AUDIT.load_root_supplement(
        path, AUDIT.sha256(path), external_assets / "l6"
    )
    assert len(entries) == 10
    assert sum(len(entry["source_rows"]) for entry in entries) == 576
    data = AUDIT.read_json(path)
    data["entries"][0]["source_rows"][0]["fixture"]["target_m"][1] = 1e-200
    altered = tmp_path / "altered-root-supplement.json"
    altered.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="full fixture binding"):
        AUDIT.load_root_supplement(
            altered, AUDIT.sha256(altered), external_assets / "l6"
        )


def test_root_sensitivity_reconstruction_rejects_inconsistent_estimate(external_assets):
    path = (
        external_assets
        / "l6/frozen-trace-audit-repair-20261002/cylinder-root-normalization-first.json"
    )
    entries = AUDIT.read_json(path)["entries"]
    altered = copy.deepcopy(entries[0])
    altered["root_precision_estimate"] = "1.0"
    with mp.workdps(80), pytest.raises(ValueError, match="estimate reconstruction"):
        AUDIT.root_precision(altered["source_rows"][0]["fixture"], [altered])


def test_interrupted_prior_invocation_retains_observed_phases(
    external_assets, tmp_path
):
    original = (
        external_assets
        / "l6/refinement-traces/observer-resume-semantic-repair-20261002-04/plane"
    )
    directory = tmp_path / "plane"
    shutil.copytree(original, directory)
    manifest = AUDIT.read_json(directory / "manifest.json")
    checkpoint = AUDIT.read_json(directory / "checkpoint.json")
    run = AUDIT.read_json(directory / "run-evidence.json")
    records = AUDIT.read_rows(directory / "traces.jsonl")
    identity = run["invocation_history"][0]["invocation_id"]
    run["invocation_history"].pop(1)
    path = directory / "run-invocations.jsonl"
    path.write_text(
        "".join(json.dumps(item) + "\n" for item in run["invocation_history"])
    )
    run["run_invocations_jsonl_sha256"] = AUDIT.sha256(path)
    for obj in (manifest, run):
        obj["process_state"]["observations"] = [
            o
            for o in obj["process_state"]["observations"]
            if not (o.get("invocation_id") == identity and o.get("phase") == "finished")
        ]
    assert AUDIT.audit_process_evidence(
        directory, manifest, checkpoint, run, records, [0]
    ) == [identity]
