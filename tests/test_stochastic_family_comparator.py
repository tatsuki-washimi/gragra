"""Synthetic admission/dispatch checks; no scientific production calls."""

import importlib
import json
import os
import subprocess
import types
from pathlib import Path

import pytest

from tests.oracles.gen1.stochastic_formal.common.transport import GateError


def test_family_driver_available_without_production_imports():
    import sys

    before = {n for n in sys.modules if n == "gragra" or n.startswith("gragra.")}
    assert (
        importlib.util.find_spec(
            "tests.oracles.gen1.stochastic_formal.common.family_driver"
        )
        is not None
    )
    importlib.import_module("tests.oracles.gen1.stochastic_formal.common.family_driver")
    assert before == {
        n for n in sys.modules if n == "gragra" or n.startswith("gragra.")
    }


def adapter():
    return importlib.import_module(
        "tests.oracles.gen1.stochastic_formal.common.family_adapter"
    )


def _git_blob(blob):
    command = ["git", "-C", str(Path(__file__).parents[1]), "cat-file", "blob", blob]
    if "A2_CAMPAIGN_LEDGER" in os.environ:
        from tests.oracles.gen1.stochastic_formal.a2.budget import BoundedGit

        return BoundedGit(os.environ["A2_CAMPAIGN_LEDGER"]).check_output(command)
    return subprocess.check_output(command)


@pytest.mark.parametrize(
    "family,symbol,shape",
    [
        ("R1", "rayleigh_isotropic_acceleration_csd_g_free", [3, 3]),
        ("B2", "spherical_cavity_ps_isotropic_acceleration_csd_g_free", [3, 3]),
        ("A1", "surface_isotropic_acoustic_acceleration_csd_g_free", [3, 3]),
        ("A2", "underground_isotropic_acoustic_acceleration_csd_g_free", [3, 3]),
        ("A3", "centered_cavern_isotropic_acoustic_acceleration_csd_g_free", [3, 3]),
        ("T1", "uniform_advection_temperature_acceleration_psd_g_free", []),
    ],
)
def test_exact_family_call_mapping(family, symbol, shape):
    key = {
        "family": family,
        "fixture_id": "f",
        "frequency_hz": "2e+0",
        "target_id": "origin",
        "observable": "projected_acceleration_psd"
        if family == "T1"
        else "acceleration_csd",
        "representation": "Gfree",
        "set": "science",
    }
    p = {
        "frequency_hz": "2e+0",
        "g_si": "6.6743e-11",
        "density_kg_m3": "3e+0",
        "height_m": "1e+0",
        "radius_m": "1e+0",
        "v_p_m_s": "6e+0",
        "v_s_m_s": "3e+0",
        "surface_vertical_displacement_psd_m2_hz": "1e-3",
        "p_displacement_psd_m2_hz": "1e-3",
        "s_displacement_psd_m2_hz": "2e-3",
        "sound_speed_m_s": "3e+2",
        "pressure_psd_pa2_hz": "2e-3",
        "depth_m": "1e+0",
        "temperature_k": "3e+2",
        "structure_coefficient_k_m_neg_p_over_2": "1e+0",
        "structure_exponent": "6e-1",
        "wind_speed_m_s": "2e+1",
        "minimum_approach_m": "3e+0",
        "wind_axis_angle_rad": "0e+0",
    }
    call = adapter().call_plan(key, p)
    assert call["symbol"] == symbol and call["shape"] == shape
    assert call["kwargs"]["frequency_hz"] == "2e+0"
    if family == "A1":
        assert call["kwargs"]["surface_pressure_psd_pa2_hz"] == "2e-3"
    if family == "R1":
        assert call["args"] == [[["0e+0", "0e+0", "1e+0"]]]
        assert call["output_index"] == [0]


def test_a2_scalar_dispatch_uses_exact_nominal_not_physical_x():
    key = {
        "family": "A2",
        "observable": "isotropic_depth_factor",
        "representation": "Gfree",
        "frequency_hz": "1e+0",
    }
    call = adapter().call_plan(key, {"nominal_exact_x": "7.5e-1", "depth_m": "9e+0"})
    assert call["args"] == ["7.5e-1"] and call["shape"] == []
    assert call["symbol"] == "isotropic_acoustic_depth_factor"


def test_actual_frozen_a2_keys_include_separate_scalar_partition():
    from tests.oracles.gen1.stochastic_formal.common.family_driver import (
        frozen_expected_keys,
    )

    fixtures = json.loads(_git_blob("d4f70937b87bca7e7601c30121db9a00cce7e386"))
    keys = frozen_expected_keys(fixtures)
    assert len(keys) == 84
    assert sum(k["set"] == "science" for k in keys) == 54
    assert sum(k["observable"] == "isotropic_depth_factor" for k in keys) == 28
    assert (
        sum(
            k["set"] == "auxiliary" and k["observable"] == "acceleration_csd"
            for k in keys
        )
        == 2
    )
    assert fixtures["expected_keys"] == keys[:56]
    bad = {**fixtures, "expected_scalar_keys": fixtures["expected_scalar_keys"][:-1]}
    with pytest.raises(GateError, match="scalar AUX"):
        frozen_expected_keys(bad)
    duplicate = {
        **fixtures,
        "expected_scalar_keys": [fixtures["expected_scalar_keys"][0]] * 28,
    }
    with pytest.raises(GateError):
        frozen_expected_keys(duplicate)


def test_frozen_a2_fixture_git_is_prelaunch_metered(monkeypatch, tmp_path):
    """The actual Git PID must occupy a global slot before it is spawned."""
    from tests.oracles.gen1.stochastic_formal.a2 import budget

    ledger_path = Path(os.environ.get("A2_CAMPAIGN_LEDGER", tmp_path / "ledger.json"))
    budget.Ledger(ledger_path)
    monkeypatch.setenv("A2_CAMPAIGN_LEDGER", str(ledger_path))
    command = [
        "git",
        "-C",
        str(Path(__file__).parents[1]),
        "cat-file",
        "blob",
        "d4f70937b87bca7e7601c30121db9a00cce7e386",
    ]
    seen = {}
    real_popen = subprocess.Popen

    def check_prelaunch(argv, *args, **kwargs):
        assert argv == command
        entries = json.loads(ledger_path.read_text())["entries"]
        owned = [
            e
            for e in entries
            if e["command"] == command
            and e["parent_pid"] == os.getpid()
            and e["pid"] is None
            and "elapsed_seconds" not in e
        ]
        assert len(owned) == 1
        process = real_popen(argv, *args, **kwargs)
        seen["pid"] = process.pid
        return process

    def unmetered(*_args, **_kwargs):
        raise AssertionError("unmetered frozen A2 fixture Git")

    monkeypatch.setattr(subprocess, "Popen", check_prelaunch)
    monkeypatch.setattr(subprocess, "check_output", unmetered)
    test_actual_frozen_a2_keys_include_separate_scalar_partition()
    entries = json.loads(ledger_path.read_text())["entries"]
    actual = [e for e in entries if e["command"] == command and e["pid"] == seen["pid"]]
    assert len(actual) == 1
    assert actual[0]["status"] == "success"
    assert actual[0]["cleanup_confirmed"] is True
    assert actual[0]["elapsed_seconds"] >= 0


def test_unset_a2_ledger_preserves_git_blob_reader(monkeypatch):
    monkeypatch.delenv("A2_CAMPAIGN_LEDGER", raising=False)
    seen = []

    def original(command):
        seen.append(command)
        return b"synthetic bytes"

    monkeypatch.setattr(subprocess, "check_output", original)
    assert _git_blob("toy-blob") == b"synthetic bytes"
    assert seen == [
        ["git", "-C", str(Path(__file__).parents[1]), "cat-file", "blob", "toy-blob"]
    ]


def test_unknown_mapping_and_a2_scalar_si_fail_closed():
    with pytest.raises(GateError):
        adapter().call_plan(
            {
                "family": "B1",
                "observable": "acceleration_csd",
                "representation": "Gfree",
            },
            {},
        )
    with pytest.raises(GateError):
        adapter().call_plan(
            {
                "family": "A2",
                "observable": "isotropic_depth_factor",
                "representation": "SI",
            },
            {"nominal_exact_x": "1e+0"},
        )


def test_promotion_preserves_complex_offdiagonal_and_subnormal():
    class Tensor:
        shape = (3, 3)
        dtype = types.SimpleNamespace(kind="c", itemsize=16)

        def tolist(self):
            return [
                [1.0, complex(0.0, 0.25), 0.0],
                [complex(0.0, -0.25), 1.0, 0.0],
                [0.0, 0.0, 5e-324],
            ]

    from tests.oracles.gen1.stochastic_formal.common.b1_driver import legacy

    blob = _git_blob(legacy.KERNEL_GIT_BLOB)
    kernel = types.ModuleType("synthetic_admitted_kernel")
    exec(compile(blob, "<admitted-kernel>", "exec"), kernel.__dict__)
    q = adapter().promote_quantity(Tensor(), [3, 3], [], kernel)
    assert q["data"][1]["imag"] == "2.5e-1"
    assert kernel.decimal_fraction(q["data"][8]["real"]) > 0
    with pytest.raises(GateError):
        adapter().promote_quantity(Tensor(), [], [], kernel)


@pytest.mark.parametrize("dtype", ["float32", "complex64", "int64", "object"])
def test_production_dtype_cannot_be_hidden_by_tolist(dtype):
    import numpy as np

    with pytest.raises(GateError, match="dtype"):
        adapter().promote_quantity(np.eye(3, dtype=dtype), [3, 3], [], None)


@pytest.mark.parametrize("dtype", ["float64", "complex128"])
def test_production_binary64_dtype_retained(dtype):
    import numpy as np

    kernel = types.SimpleNamespace(promote_binary64=lambda v: repr(v))
    quantity = adapter().promote_quantity(np.eye(3, dtype=dtype), [3, 3], [], kernel)
    assert quantity["data"][0]["real"] == "1.0"


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "missing_resource_binding",
        "timeout",
        "rss",
        "cleanup",
        "runtime",
        "failed_row",
        "wrong_parent",
        "missing_gate",
    ],
)
def test_family_result_requires_actual_resource_terminal_binding(
    tmp_path, monkeypatch, mutation
):
    from tests.oracles.gen1.stochastic_formal.common import family_supervisor as module
    from tests.oracles.gen1.stochastic_formal.common.transport import (
        SEAL_SCHEMA,
        artifact_pin,
        canonical_bytes,
    )

    monkeypatch.setattr(module, "verify_authorization", lambda *_args: None)
    worker, parent = tmp_path / "worker", tmp_path / "parent"
    worker.mkdir()
    parent.mkdir()

    def put(root, name, value, kind):
        (root / name).write_bytes(canonical_bytes(value))
        return artifact_pin(root / name, kind)

    key = {
        "family": "A1",
        "fixture_id": "synthetic",
        "frequency_hz": "1e+0",
        "target_id": "origin",
        "observable": "acceleration_csd",
        "representation": "Gfree",
        "set": "science",
    }
    pins = {"policy_sha256": "p" * 64, "dependency_closure_sha256": "c" * 64}
    authority = {"sha256": "a" * 64}
    resources = {"resources": {"timeout_seconds": 10}, "rss_bytes_per_process": 100}
    procedure = {"path": "synthetic", "sha256": "f" * 64, "kind": "procedure"}
    parent_pin = put(
        parent,
        "launch.json",
        {
            "supervisor_pid": 123,
            "policy_sha256": pins["policy_sha256"],
            "mode": "FORMAL",
            "frozen_procedure": procedure,
            "timeout_seconds": 10,
            "supervisor": artifact_pin(
                Path(module.__file__).with_name("supervisor.py"), "supervisor_source"
            ),
            "cleanup_helper": artifact_pin(
                Path(module.__file__).parents[2] / "run_stochastic_local.py",
                "admitted_cleanup_helper",
            ),
        },
        "supervisor_launch",
    )
    resource_pin = put(
        parent,
        "resource-launch.json",
        {
            "schema": "gen1-stochastic-family-resource-launch-v1",
            "parent_launch": {**parent_pin, "sha256": "bad"}
            if mutation == "wrong_parent"
            else parent_pin,
            "policy_sha256": pins["policy_sha256"],
            "resources": resources,
            "supervisor_pid": 123,
            "source": artifact_pin(module.__file__, "family_supervisor_source"),
            "aggregate_budget_reservation": None,
        },
        "family_resource_launch",
    )
    run_pins = {
        **pins,
        "authorization_sha256": authority["sha256"],
        "supervisor_launch_sha256": parent_pin["sha256"],
        "resource_launch_sha256": resource_pin["sha256"],
    }
    if mutation == "missing_resource_binding":
        run_pins.pop("resource_launch_sha256")
    launch_pin = put(
        worker, "launch.json", {"pins": run_pins, "expected_keys": [key]}, "launch"
    )
    record = {
        "key": key,
        "status": "resolved",
        "passed": mutation != "failed_row",
        "assessment": {"accepted": True},
        "connection_assessment": {
            "gates": {} if mutation == "missing_gate" else {"claim": True}
        },
    }
    records_pin = put(worker, "records.jsonl", record, "records")
    module_pin = {"name": "gragra.synthetic"}
    runtime_pin = put(
        worker,
        "runtime-dependencies.json",
        {
            "status": "FAIL" if mutation == "runtime" else "MATCH",
            "planned_sha256": pins["dependency_closure_sha256"],
            "python": {},
            "observed_modules": [module_pin],
            "observed_edges": [],
            "observed_native_backends": [],
        },
        "runtime_dependencies",
    )
    attempt_pin = put(
        worker,
        "seal.json",
        {
            "schema": SEAL_SCHEMA,
            "family": "A1",
            "phase": "comparison",
            "decision": "PASS",
            "exit_code": 0,
            "pins": run_pins,
            "completed": 1,
            "expected_keys": [key],
            "artifacts": {
                "launch": launch_pin,
                "records": records_pin,
                "runtime_dependencies": runtime_pin,
            },
        },
        "attempt",
    )
    supervisor_pin = put(
        parent,
        "seal.json",
        {
            "schema": "gen1-stochastic-family-supervisor-seal-v1",
            "family": "A1",
            "policy_sha256": pins["policy_sha256"],
            "outcome": "CHILD_EXITED",
            "exit_code": 0,
            "cleanup_confirmed": mutation != "cleanup",
            "cleanup": {"status": "confirmed_stopped"},
            "frozen_resources": resources,
            "elapsed_seconds": "11" if mutation == "timeout" else "1",
            "observed_process_peak_rss_bytes": {
                "123": 101 if mutation == "rss" else 10
            },
            "artifacts": {
                name: artifact_pin(parent / name, "supervisor_evidence")
                for name in ("launch.json", "resource-launch.json")
            },
        },
        "supervisor",
    )
    for root in (worker, parent):
        for path in root.iterdir():
            path.chmod(0o444)
        root.chmod(0o555)
    prepared = {
        "pins": pins,
        "policy": {
            "family": "A1",
            "connection_requirements": ["claim"],
            "artifacts": {"procedure": procedure},
        },
        "resources": resources,
        "fixtures": {"expected_keys": [key]},
        "closure": {
            "modules": [module_pin],
            "python": {},
            "import_edges": [],
            "native_backends": [],
        },
    }
    kwargs = {
        "attempt_pin": attempt_pin,
        "supervisor_pin": supervisor_pin,
        "authorization": authority,
        "reviews": {},
    }
    if mutation is None:
        result = module.verify_family_result(prepared, **kwargs)
        assert result["summary"]["delivery_ready"] is True
        assert result["pins"]["resource_launch_sha256"] == resource_pin["sha256"]
    else:
        with pytest.raises(GateError):
            module.verify_family_result(prepared, **kwargs)


def test_science_and_auxiliary_summary_never_conflate():
    rows = [
        {"key": {"set": "science"}, "passed": True},
        {"key": {"set": "auxiliary"}, "passed": False},
    ]
    summary = adapter().partition_summary(rows)
    assert summary["physical_acceptance"] is True
    assert summary["numerical_aux_status"] == "FAIL"
    assert summary["delivery_ready"] is False


def test_family_execution_seals_verified_resource_pin(tmp_path, monkeypatch):
    import hashlib

    from tests.oracles.gen1.stochastic_formal.common import (
        family_execution as execution,
    )
    from tests.oracles.gen1.stochastic_formal.common import (
        family_supervisor as supervisor,
    )
    from tests.oracles.gen1.stochastic_formal.common.transport import (
        canonical_bytes,
        strict_json,
    )

    pins = {"production_commit": "p" * 40, "production_tree": "t" * 40}
    closure = dict(pins)
    pins["dependency_closure_sha256"] = hashlib.sha256(
        canonical_bytes(closure)
    ).hexdigest()
    resource = {"sha256": "r" * 64}
    key = {
        "family": "A1",
        "fixture_id": "synthetic",
        "frequency_hz": "1e+0",
        "target_id": "origin",
        "observable": "acceleration_csd",
        "representation": "Gfree",
        "set": "science",
    }
    prepared = {
        "pins": pins,
        "fixtures": {"expected_keys": [key]},
        "policy": {"family": "A1"},
        "closure": closure,
    }
    monkeypatch.setattr(execution, "verify_authorization", lambda *_args: None)
    monkeypatch.setattr(execution, "verify_formal_closure", lambda *_args: None)
    monkeypatch.setattr(
        execution, "verify_worker_supervision", lambda *_args: {"sha256": "s" * 64}
    )
    monkeypatch.setattr(
        supervisor, "verify_family_supervision", lambda *_args: resource
    )

    class Guard:
        def __init__(self, manifest):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def verify(self):
            return {"status": "MATCH"}

    monkeypatch.setattr(execution, "ImportGuard", Guard)
    output = tmp_path / "attempt"
    execution.run_family_phase(
        authorization={"sha256": "a" * 64},
        pins=pins,
        reviews={},
        family="A1",
        phase="comparison",
        expected_keys=[key],
        output=output,
        evaluator_factory=lambda: (
            lambda k: {"key": k, "status": "resolved", "passed": True}
        ),
        dependency_closure=closure,
        resource_launch=resource,
        prepared=prepared,
    )
    seal = strict_json((output / "seal.json").read_bytes())
    assert seal["pins"]["resource_launch_sha256"] == resource["sha256"]
    assert seal["pins"]["supervisor_launch_sha256"] == "s" * 64
    assert seal["decision"] == "PASS"
    assert not (output / "seal.json").stat().st_mode & 0o222


def test_family_execution_authority_refusal_precedes_allocation(tmp_path, monkeypatch):
    from tests.oracles.gen1.stochastic_formal.common import family_execution

    def refuse(*_args):
        raise GateError("synthetic unauthorized")

    monkeypatch.setattr(family_execution, "verify_authorization", refuse)
    output = tmp_path / "not-allocated"
    with pytest.raises(GateError, match="unauthorized"):
        family_execution.run_family_phase(
            authorization={},
            pins={},
            reviews={},
            family="A1",
            phase="comparison",
            expected_keys=[],
            output=output,
            evaluator_factory=lambda: pytest.fail("factory must remain lazy"),
            dependency_closure={},
            resource_launch={},
            prepared={},
        )
    assert not output.exists()


def test_reference_admission_schema_requires_actual_pins():
    from tests.oracles.gen1.stochastic_formal.common.family_driver import (
        require_reference_admission,
    )

    with pytest.raises(GateError):
        require_reference_admission({"decision": "PASS", "pins": {}}, {})


def test_source_scale_exact_formula_and_downward_pi_bounds():
    scales = importlib.import_module(
        "tests.oracles.gen1.stochastic_formal.common.family_scales"
    )
    from fractions import Fraction

    lo, hi = scales.pi_bounds()
    assert Fraction(314159, 100000) < lo < hi < Fraction(314160, 100000)
    assert hi - lo < Fraction(1, 10**330)
    value, method = scales.scale_bounds(
        "A1",
        {"representation": "Gfree", "observable": "acceleration_csd"},
        {
            "pressure_psd_pa2_hz": "1e-3",
            "sound_speed_m_s": "1e+1",
            "frequency_hz": "2e+0",
            "g_si": "6.6743e-11",
        },
        None,
    )
    assert value == Fraction(1, 400000)
    assert "exact rational" in method


def test_source_scale_scalar_one_third_and_cavern_whole_zero():
    scales = importlib.import_module(
        "tests.oracles.gen1.stochastic_formal.common.family_scales"
    )
    from fractions import Fraction

    assert scales.scale_bounds(
        "A2",
        {"observable": "isotropic_depth_factor", "representation": "Gfree"},
        {},
        None,
    )[0] == Fraction(1, 3)
    p = {
        "radius_m": "0e+0",
        "pressure_psd_pa2_hz": "1e-3",
        "sound_speed_m_s": "1e+1",
        "frequency_hz": "2e+0",
        "g_si": "6.6743e-11",
    }
    assert (
        scales.scale_bounds(
            "A3", {"observable": "acceleration_csd", "representation": "Gfree"}, p, None
        )[0]
        == 0
    )


def test_family_resources_normalize_existing_frozen_field_names():
    from tests.oracles.gen1.stochastic_formal.common.family_driver import (
        execution_resources,
    )

    assert (
        execution_resources(
            {"numerical_procedure": {"formal_timeout_seconds": 1800}}, "R1"
        )["resources"]["timeout_seconds"]
        == 1800
    )
    assert (
        execution_resources(
            {
                "resources": {
                    "max_attempt_seconds": 3600,
                    "rss_bytes_per_process": 4294967296,
                }
            },
            "A2",
        )["rss_bytes_per_process"]
        == 4294967296
    )


def test_direct_family_worker_cannot_bypass_resource_supervisor(monkeypatch):
    supervisor = importlib.import_module(
        "tests.oracles.gen1.stochastic_formal.common.family_supervisor"
    )
    monkeypatch.delenv(supervisor.TOKEN, raising=False)
    with pytest.raises(GateError, match="resource supervisor"):
        supervisor.verify_family_supervision({})


def synthetic_reference(tmp_path):
    from tests.oracles.gen1.stochastic_formal.common.family_driver import (
        derive_reference_pins,
    )
    from tests.oracles.gen1.stochastic_formal.common.transport import (
        artifact_pin,
        canonical_bytes,
    )

    root = tmp_path / "sealed"
    root.mkdir()

    def write(path, data, kind):
        path.write_bytes(canonical_bytes(data) + b"\n")
        path.chmod(0o444)
        return artifact_pin(path, kind)

    key = {
        "family": "T1",
        "fixture_id": "mock",
        "frequency_hz": "1e+0",
        "observable": "projected_acceleration_psd",
        "representation": "Gfree",
        "target_id": "projection",
        "set": "science",
    }
    fixture = write(
        tmp_path / "fixture.json",
        {"family": "T1", "expected_keys": [key]},
        "frozen_evidence",
    )
    budget = write(tmp_path / "budget.json", {"family": "T1"}, "frozen_evidence")
    procedure = write(tmp_path / "procedure.json", {"family": "T1"}, "frozen_evidence")
    freeze = write(
        tmp_path / "freeze.json",
        {
            "family": "T1",
            "status": "FROZEN",
            "fixture_sha256": fixture["sha256"],
            "budget_sha256": budget["sha256"],
            "procedure_sha256": procedure["sha256"],
        },
        "frozen_evidence",
    )
    record = write(root / "records.jsonl", {"key": key}, "sealed_reference")
    env = write(
        root / "environment.json",
        {"loaded_modules": {}, "production_modules": []},
        "sealed_reference",
    )
    seal = write(
        root / "seal.json",
        {
            "family": "T1",
            "mode": "formal",
            "outcome": "COMPLETE",
            "exit_code": 0,
            "coverage_valid": True,
            "cleanup_confirmed": True,
            "pins": {
                "E_commit": "1" * 40,
                "E_tree": "2" * 40,
                "fixtures_sha256": fixture["sha256"],
                "budget_sha256": budget["sha256"],
                "procedure_sha256": procedure["sha256"],
                "freeze_declaration_sha256": freeze["sha256"],
                "expected_environment_sha256": env["sha256"],
            },
            "expected_keys": [key],
            "artifact_hashes": {
                "records.jsonl": record["sha256"],
                "environment.json": env["sha256"],
            },
        },
        "sealed_reference",
    )
    artifacts = {
        "freeze": freeze,
        "fixture": fixture,
        "budget": budget,
        "family_procedure": procedure,
        "records": record,
        "seal": seal,
        "environment": env,
    }
    root.chmod(0o555)
    return artifacts, derive_reference_pins(artifacts), write


@pytest.mark.parametrize(
    "mutation",
    [
        "decision",
        "role",
        "stale",
        "duplicate_identity",
        "unknown_field",
        "original_hash",
    ],
)
def test_reference_review_companions_reject_counterfeit_bindings(tmp_path, mutation):
    from tests.oracles.gen1.stochastic_formal.common.family_driver import (
        require_reference_admission,
    )

    artifacts, pins, write = synthetic_reference(tmp_path)
    reports = {}
    for role in ["science", "numerical", "provenance"]:
        original = write(
            tmp_path / (role + "-report.json"),
            {"scope": "synthetic-only"},
            "reference_review_report",
        )
        report = {
            "schema": "gen1-stochastic-reference-review-v1",
            "family": "T1",
            "role": role,
            "reviewer_id": role,
            "decision": "PASS",
            "pins": pins,
            "original_review": original,
        }
        if role == "provenance":
            if mutation == "decision":
                report["decision"] = "REVISE"
            elif mutation == "role":
                report["role"] = "science"
            elif mutation == "stale":
                report["pins"] = {**pins, "records_sha256": "f" * 64}
            elif mutation == "duplicate_identity":
                report["reviewer_id"] = "science"
            elif mutation == "unknown_field":
                report["approved"] = True
            else:
                report["original_review"] = {**original, "sha256": "0" * 64}
        reports[role] = write(
            tmp_path / (role + "-companion.json"), report, "reference_review"
        )
    admission = {
        "schema": "gen1-stochastic-reference-admission-v1",
        "family": "T1",
        "decision": "PASS",
        "pins": pins,
        "reviews": reports,
    }
    with pytest.raises((GateError, ValueError)):
        require_reference_admission(admission, artifacts)


def test_reference_frozen_and_sealed_bytes_cannot_be_swapped(tmp_path):
    from tests.oracles.gen1.stochastic_formal.common.family_driver import (
        derive_reference_pins,
    )

    artifacts, pins, write = synthetic_reference(tmp_path)
    other = write(
        tmp_path / "other-records.jsonl", {"key": {"family": "T1"}}, "sealed_reference"
    )
    with pytest.raises(GateError):
        derive_reference_pins({**artifacts, "records": other})
    other = write(
        tmp_path / "other-budget.json",
        {"family": "T1", "extra": "unfrozen"},
        "frozen_evidence",
    )
    with pytest.raises(GateError):
        derive_reference_pins({**artifacts, "budget": other})


def test_rss_observation_preserves_per_process_cap(tmp_path):
    from tests.oracles.gen1.stochastic_formal.common.family_supervisor import (
        process_group_rss,
    )

    for pid, rss, group in [(101, 20, 101), (102, 30, 101), (103, 90, 999)]:
        p = tmp_path / str(pid)
        p.mkdir()
        (p / "stat").write_text(f"{pid} (worker space) S 1 {group} {group} 0")
        (p / "status").write_text(f"Name: worker\nVmRSS: {rss} kB\n")
    assert process_group_rss(101, tmp_path) == {101: 20480, 102: 30720}


def test_source_proof_cannot_be_fixture_declared_pass(tmp_path):
    from tests.oracles.gen1.stochastic_formal.common.family_source_review import (
        require_source_admission,
    )
    from tests.oracles.gen1.stochastic_formal.common.transport import artifact_pin

    p = tmp_path / "claims.json"
    p.write_text(json.dumps({"decision": "PASS", "claims": ["isotropic"]}))
    with pytest.raises(GateError):
        require_source_admission(
            artifact_pin(p, "source_review"),
            family="R1",
            production_repo=tmp_path,
            production_commit="1" * 40,
            production_tree="2" * 40,
            frozen_artifacts={},
            reference_pins={},
            plans={},
            closure={},
            requirements=[],
        )
