"""Mutation probes for actual managed completion and clock transport contracts."""

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from tests.oracles.gen1.stochastic_delivery.a2 import (
    LEDGER_PATH,
    RESOURCE_TOKEN,
    RUNTIME_PROFILE_SOURCES,
    operational_json,
    verify_a2_comparison,
    verify_managed_completion,
)
from tests.oracles.gen1.stochastic_delivery.archive import EvidenceError, decode_json

HISTORICAL_INTEGRATION_COMMIT = "ad096eb09a41bffc132bda48f101d99f8446102c"


def historical_source(relative):
    return subprocess.run(
        [
            "git",
            "show",
            f"{HISTORICAL_INTEGRATION_COMMIT}:{relative}",
        ],
        cwd=Path(__file__).parents[1],
        check=True,
        capture_output=True,
    ).stdout


class MemoryCAS:
    def __init__(self):
        self.objects = {}

    def add(self, name, value):
        raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        return self.add_raw(name, raw)

    def add_raw(self, name, raw):
        pin = {
            "path": "/retained/" + name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "kind": "test_metadata",
        }
        self.objects[pin["sha256"]] = raw
        return pin

    def read(self, pin):
        raw = self.objects[pin["sha256"]]
        assert hashlib.sha256(raw).hexdigest() == pin["sha256"]
        return raw

    def json(self, pin):
        return decode_json(self.read(pin))


def managed_case():
    store = MemoryCAS()
    formal = "tests/oracles/gen1/stochastic_formal"
    adapter = store.add_raw(
        "adapter.py", historical_source(formal + "/common/a2_resource_adapter.py")
    )
    budget = store.add_raw("budget.py", historical_source(formal + "/a2/budget.py"))
    assert adapter["sha256"] in RUNTIME_PROFILE_SOURCES[budget["sha256"]]
    environment = store.add("env", "actual managed environment executable identity")
    resource = {
        "managed_profile": "a2-ledger-v1",
        "controller_id": 4,
        "aggregate_budget_reservation": {
            "ledger_path": LEDGER_PATH,
            "launch_key": "f" * 32,
        },
        "managed_child_adapter": adapter,
        "normative_budget_source": budget,
        "managed_environment_executable": environment,
        "supervisor_pid": 31,
    }
    resource_pin = store.add("resource-launch.json", resource)
    token = (
        RESOURCE_TOKEN
        + "="
        + json.dumps(resource_pin, sort_keys=True, separators=(",", ":"))
    )
    worker = {
        "id": 2,
        "pid": 32,
        "parent_pid": 31,
        "kind": "repair_validation",
        "status": "success",
        "exit_code": 0,
        "cleanup_confirmed": True,
        "elapsed_seconds": 2.25,
        "command": ["/usr/bin/env", "-i", "--", token, "/python", "/worker.py"],
    }
    span = {
        "issuer": "root",
        "decision": "CONTROLLER_EXTERNAL_SPAN_ATTESTED",
        "controller_id": 4,
        "controller_argv": ["/python", "/controller.py"],
        "started_monotonic": 100.25,
        "elapsed_seconds": 3.5,
    }
    span_pin = store.add("span.json", span)
    controller = {
        "id": 4,
        "pid": 31,
        "status": "completed",
        "outcome": "success",
        "command": span["controller_argv"],
        "started_monotonic": 100.25,
        "external_span_upper_seconds": 3.5,
        "deadline_seconds": 3600,
        "wall_bound_seconds": 3700,
        "external_span_pin": {name: span_pin[name] for name in ("path", "sha256")},
    }
    terminal_pin = store.add("terminal.json", {"entries": [worker]})
    postwait_pin = store.add(
        "postwait.json", {"entries": [worker], "controllers": [controller]}
    )
    supervisor = {
        "managed_profile": "a2-ledger-v1",
        "controller_id": 4,
        "aggregate_budget_reservation": resource["aggregate_budget_reservation"],
        "managed_ledger_id": 2,
        "cleanup": {"pid": 32},
        "elapsed_seconds": "2.25",
        "artifacts": {"managed-ledger-snapshot.json": terminal_pin},
    }
    supervisor_pin = store.add("supervisor-seal.json", supervisor)
    attempt_pin = store.add("attempt-seal.json", {"decision": "PASS"})
    completion = {
        "schema": "gen1-stochastic-a2-managed-controller-completion-v1",
        "issuer": "root",
        "decision": "CONTROLLER_AND_MANAGED_CHILD_COMPLETE",
        "managed_profile": "a2-ledger-v1",
        "controller_id": 4,
        "aggregate_budget_reservation": resource["aggregate_budget_reservation"],
        "resource_launch": resource_pin,
        "supervisor_seal": supervisor_pin,
        "attempt_seal": attempt_pin,
        "postwait_ledger_snapshot": postwait_pin,
        "root_external_span": span_pin,
    }
    closure = {
        "modules": [
            {
                **adapter,
                "name": (
                    "tests.oracles.gen1.stochastic_formal.common.a2_resource_adapter"
                ),
            },
            {**budget, "name": "tests.oracles.gen1.stochastic_formal.a2.budget"},
        ]
    }
    arguments = {
        "closure": closure,
        "resource_pin": resource_pin,
        "resource": resource,
        "supervisor_pin": supervisor_pin,
        "supervisor": supervisor,
        "attempt_pin": attempt_pin,
        "completion_pin": store.add("completion.json", completion),
    }
    return store, arguments, completion


def test_actual_managed_completion_field_contract():
    store, arguments, completion = managed_case()
    assert verify_managed_completion(store, **arguments) == completion


@pytest.mark.parametrize(
    "name",
    [
        "resource_launch",
        "supervisor_seal",
        "attempt_seal",
        "controller_id",
        "managed_profile",
        "aggregate_budget_reservation",
        "issuer",
        "decision",
    ],
)
def test_completion_cannot_swap_actual_subject(name):
    store, arguments, completion = managed_case()
    completion[name] = "counterfeit"
    arguments["completion_pin"] = store.add("counterfeit.json", completion)
    with pytest.raises(EvidenceError):
        verify_managed_completion(store, **arguments)


def test_missing_managed_source_in_full_planned_closure():
    store, arguments, _ = managed_case()
    arguments["closure"]["modules"].pop()
    with pytest.raises(EvidenceError, match="absent from actual closure"):
        verify_managed_completion(store, **arguments)


def test_matching_closure_cannot_admit_an_unknown_operational_source_version():
    store, arguments, _ = managed_case()
    counterfeit = store.add_raw("other-budget.py", b"unreviewed budget source")
    arguments["resource"]["normative_budget_source"] = counterfeit
    arguments["closure"]["modules"][1].update(counterfeit)
    with pytest.raises(
        EvidenceError, match="completion/actual resource/attempt/source"
    ):
        verify_managed_completion(store, **arguments)


def managed_case_with_source_identities(monkeypatch, budget_sha, adapter_sha):
    """Isolate version admission after the archive has verified source bytes.

    The store's source reads are mocked only for this identity-policy probe;
    archive hash validation and real local source reads have separate tests.
    All resource, closure, terminal-command and completion bindings are rebuilt.
    """
    store, arguments, completion = managed_case()
    resource = arguments["resource"]
    original_sources = {
        resource[name]["path"]: dict(resource[name])
        for name in ("normative_budget_source", "managed_child_adapter")
    }
    original_read = store.read

    def read(pin):
        if pin["path"] in original_sources:
            return original_read(original_sources[pin["path"]])
        return original_read(pin)

    monkeypatch.setattr(store, "read", read)
    resource["normative_budget_source"]["sha256"] = budget_sha
    resource["managed_child_adapter"]["sha256"] = adapter_sha
    arguments["closure"]["modules"][0]["sha256"] = adapter_sha
    arguments["closure"]["modules"][1]["sha256"] = budget_sha
    arguments["resource_pin"] = store.add("versioned-resource.json", resource)
    completion["resource_launch"] = arguments["resource_pin"]
    token = (
        RESOURCE_TOKEN
        + "="
        + json.dumps(arguments["resource_pin"], sort_keys=True, separators=(",", ":"))
    )
    postwait = operational_json(store.read(completion["postwait_ledger_snapshot"]))
    postwait["entries"][0]["command"][3] = token
    arguments["supervisor"]["artifacts"]["managed-ledger-snapshot.json"] = store.add(
        "versioned-terminal.json", {"entries": postwait["entries"]}
    )
    completion["postwait_ledger_snapshot"] = store.add(
        "versioned-postwait.json", postwait
    )
    arguments["supervisor_pin"] = store.add(
        "versioned-supervisor.json", arguments["supervisor"]
    )
    completion["supervisor_seal"] = arguments["supervisor_pin"]
    arguments["completion_pin"] = store.add("versioned-completion.json", completion)
    return store, arguments, completion


@pytest.mark.parametrize(
    "budget_sha,adapter_sha",
    [
        (
            "f46f41dad65c76be2d255d1596b70ba9b43e47b164c6781dd58fd1aa788506b9",
            "bd7af6bcbbb441a5f7339d4c5b05760bad3e897d4730414d372a8acdd1e6ebad",
        ),
        (
            "a61ed66ad19291fe1a00fa87c5b266c7633e4e17d52802cb24b41709690d90a8",
            "1cf51b42dfd4c8bcd4d00aae10ba19980e5bebcca523c6ae1e734947cf39df96",
        ),
        (
            "a61ed66ad19291fe1a00fa87c5b266c7633e4e17d52802cb24b41709690d90a8",
            "e249760aaab011dba787263ec654ed1e6104633d7d2a3c778a1b136f811cc015",
        ),
    ],
)
def test_each_retained_exact_runtime_source_pair(monkeypatch, budget_sha, adapter_sha):
    store, arguments, completion = managed_case_with_source_identities(
        monkeypatch, budget_sha, adapter_sha
    )
    assert verify_managed_completion(store, **arguments) == completion


@pytest.mark.parametrize(
    "budget_sha,adapter_sha",
    [
        (
            "f46f41dad65c76be2d255d1596b70ba9b43e47b164c6781dd58fd1aa788506b9",
            "1cf51b42dfd4c8bcd4d00aae10ba19980e5bebcca523c6ae1e734947cf39df96",
        ),
        (
            "a61ed66ad19291fe1a00fa87c5b266c7633e4e17d52802cb24b41709690d90a8",
            "bd7af6bcbbb441a5f7339d4c5b05760bad3e897d4730414d372a8acdd1e6ebad",
        ),
        (
            "f46f41dad65c76be2d255d1596b70ba9b43e47b164c6781dd58fd1aa788506b9",
            "e249760aaab011dba787263ec654ed1e6104633d7d2a3c778a1b136f811cc015",
        ),
        (
            "a61ed66ad19291fe1a00fa87c5b266c7633e4e17d52802cb24b41709690d90a8",
            "8501139ed4a0d049f551f7895bd659ecf758102f3bee16f3baf4437eba346422",
        ),
        ("f" * 64, "bd7af6bcbbb441a5f7339d4c5b05760bad3e897d4730414d372a8acdd1e6ebad"),
        ("a61ed66ad19291fe1a00fa87c5b266c7633e4e17d52802cb24b41709690d90a8", "f" * 64),
    ],
)
def test_matching_closure_still_rejects_crosspaired_or_unknown_source_identities(
    monkeypatch, budget_sha, adapter_sha
):
    store, arguments, _ = managed_case_with_source_identities(
        monkeypatch, budget_sha, adapter_sha
    )
    with pytest.raises(
        EvidenceError, match="completion/actual resource/attempt/source"
    ):
        verify_managed_completion(store, **arguments)


def test_partial_worker_is_not_a_pass_completion():
    store, arguments, completion = managed_case()
    postwait = operational_json(store.read(completion["postwait_ledger_snapshot"]))
    terminal = copy.deepcopy(postwait)
    terminal["entries"][0].update(status="error", exit_code=1)
    arguments["supervisor"]["artifacts"]["managed-ledger-snapshot.json"] = store.add(
        "failed-terminal.json", terminal
    )
    completion["postwait_ledger_snapshot"] = store.add("failed-postwait.json", terminal)
    arguments["completion_pin"] = store.add("failed-completion.json", completion)
    with pytest.raises(EvidenceError, match="post-wait cleanup"):
        verify_managed_completion(store, **arguments)


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_worker",
        "wrong_pid",
        "wrong_parent",
        "wrong_command",
        "unfinished_controller",
        "span_changed",
        "unreaped_descendant",
    ],
)
def test_managed_terminal_and_postwait_must_describe_actual_owned_processes(mutation):
    store, arguments, completion = managed_case()
    postwait = operational_json(store.read(completion["postwait_ledger_snapshot"]))
    terminal = operational_json(
        store.read(arguments["supervisor"]["artifacts"]["managed-ledger-snapshot.json"])
    )
    worker = postwait["entries"][0]
    if mutation == "duplicate_worker":
        postwait["entries"].append(copy.deepcopy(worker))
    elif mutation == "wrong_pid":
        worker["pid"] = 999
        terminal["entries"][0]["pid"] = 999
    elif mutation == "wrong_parent":
        worker["parent_pid"] = 999
        terminal["entries"][0]["parent_pid"] = 999
    elif mutation == "wrong_command":
        worker["command"][3] = RESOURCE_TOKEN + "=counterfeit"
        terminal["entries"][0]["command"] = worker["command"]
    elif mutation == "unfinished_controller":
        postwait["controllers"][0]["status"] = "running"
    elif mutation == "span_changed":
        postwait["controllers"][0]["external_span_upper_seconds"] = 3.75
    else:
        postwait["entries"].append(
            {
                "id": 3,
                "pid": 33,
                "parent_pid": 31,
                "elapsed_seconds": 1,
                "cleanup_confirmed": False,
            }
        )
    arguments["supervisor"]["artifacts"]["managed-ledger-snapshot.json"] = store.add(
        "mutated-terminal.json", terminal
    )
    completion["postwait_ledger_snapshot"] = store.add(
        "mutated-postwait.json", postwait
    )
    arguments["completion_pin"] = store.add("mutated-completion.json", completion)
    with pytest.raises(EvidenceError):
        verify_managed_completion(store, **arguments)


def test_partial_lifecycle_does_not_become_bounded_admission():
    store, arguments, completion = managed_case()
    postwait = operational_json(store.read(completion["postwait_ledger_snapshot"]))
    postwait["entries"][0].update(status="error", exit_code=1)
    postwait["controllers"][0]["outcome"] = "error"
    arguments["supervisor"]["exit_code"] = 1
    arguments["supervisor"]["artifacts"]["managed-ledger-snapshot.json"] = store.add(
        "partial-terminal.json", {"entries": postwait["entries"]}
    )
    completion["postwait_ledger_snapshot"] = store.add(
        "partial-postwait.json", postwait
    )
    arguments["completion_pin"] = store.add("partial-completion.json", completion)
    assert verify_managed_completion(store, partial=True, **arguments) == completion
    with pytest.raises(EvidenceError):
        verify_managed_completion(store, **arguments)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"clock":NaN}',
        b'{"clock":Infinity}',
        b'{"clock":1e999}',
        b'{"clock":1,"clock":2}',
    ],
)
def test_operational_metadata_rejects_nonfinite_and_duplicate_fields(raw):
    with pytest.raises(EvidenceError):
        operational_json(raw)


def test_operational_clock_permission_does_not_weaken_scientific_decoder():
    assert operational_json(b'{"clock":2.25}') == {"clock": 2.25}
    with pytest.raises(EvidenceError):
        decode_json(b'{"clock":2.25}')


@pytest.mark.parametrize("profile", ["generic-v1", "b1-v1", "a2-managed-v1"])
def test_a2_never_accepts_generic_or_aliased_profile(profile):
    with pytest.raises(EvidenceError, match="native A2 managed profile"):
        verify_a2_comparison(
            None, {"profile": profile, "stages": [{"phase": "comparison"}]}
        )


def test_partial_requires_an_actual_issued_eligibility_pin():
    # A descriptive status cannot substitute for the exact issued packet.
    with pytest.raises(EvidenceError, match="native A2 managed profile"):
        verify_a2_comparison(
            None,
            {
                "profile": "a2-ledger-v1",
                "stages": [{"phase": "comparison"}],
                "status": "PARTIAL_NUMERICAL_CONDITIONING_UNRESOLVED",
            },
        )


@pytest.mark.parametrize(
    "deadline,wall,elapsed", [(3601, 3701, 1), (300, 401, 1), (3600, 3700, 3600.01)]
)
def test_managed_completion_enforces_frozen_maxima_and_declared_margin(
    deadline, wall, elapsed
):
    store, arguments, completion = managed_case()
    postwait = operational_json(store.read(completion["postwait_ledger_snapshot"]))
    postwait["controllers"][0].update(
        deadline_seconds=deadline, wall_bound_seconds=wall
    )
    postwait["entries"][0]["elapsed_seconds"] = elapsed
    arguments["supervisor"]["elapsed_seconds"] = str(elapsed)
    arguments["supervisor"]["artifacts"]["managed-ledger-snapshot.json"] = store.add(
        "over-cap-terminal.json", {"entries": postwait["entries"]}
    )
    completion["postwait_ledger_snapshot"] = store.add(
        "over-cap-postwait.json", postwait
    )
    arguments["completion_pin"] = store.add("over-cap-completion.json", completion)
    with pytest.raises(EvidenceError):
        verify_managed_completion(store, partial=True, **arguments)


def test_managed_completion_accepts_a_stricter_declared_controller_cap():
    store, arguments, completion = managed_case()
    postwait = operational_json(store.read(completion["postwait_ledger_snapshot"]))
    postwait["controllers"][0].update(deadline_seconds=300, wall_bound_seconds=400)
    completion["postwait_ledger_snapshot"] = store.add(
        "strict-cap-postwait.json", postwait
    )
    arguments["completion_pin"] = store.add("strict-cap-completion.json", completion)
    assert verify_managed_completion(store, **arguments) == completion


def test_unpaired_retained_pass_is_not_an_independently_replayed_pass(monkeypatch):
    from fractions import Fraction

    from tests.oracles.gen1.stochastic_delivery import arithmetic
    from tests.oracles.gen1.stochastic_delivery.a2 import replay_a2_arithmetic

    class UnusedKernel:
        @staticmethod
        def promote_binary64(value):
            return str(value)

        @staticmethod
        def decimal_fraction(value):
            return Fraction(value)

        @staticmethod
        def assess(*_args, **_kwargs):
            pytest.fail("unpaired quantity must not be classified as replayed")

    store = MemoryCAS()
    csd_keys = [
        {
            "family": "A2",
            "fixture_id": f"synthetic-{i}",
            "frequency_hz": "1",
            "target_id": "synthetic",
            "observable": "acceleration_csd",
            "representation": representation,
            "set": "science" if i < 27 else "auxiliary",
        }
        for i in range(28)
        for representation in ("Gfree", "SI")
    ]
    scalar_keys = [
        {
            "family": "A2",
            "fixture_id": f"scalar-{i}",
            "frequency_hz": "1",
            "target_id": "nominal_x",
            "observable": "isotropic_depth_factor",
            "representation": "Gfree",
            "set": "auxiliary",
        }
        for i in range(28)
    ]
    quantities = [{"real": "0e+0", "imag": "0e+0"}] * 9
    references = [
        {
            "key": key,
            "status": "resolved",
            "uncertainty": {"kind": "estimate"},
            "quantity": {
                "shape": [] if key in scalar_keys else [3, 3],
                "data": quantities[:1] if key in scalar_keys else quantities,
            },
            "inputs": {"g_si": "1e0"},
        }
        for key in csd_keys + scalar_keys
    ]
    lines = [
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for row in references
    ]
    row = {
        "key": csd_keys[0],
        "status": "resolved",
        "passed": True,
        "reference_record_sha256": hashlib.sha256(lines[0]).hexdigest(),
        "reference_purpose": "formal-science",
        "reference_set": "science",
        "production_quantity": {"shape": [3, 3], "data": quantities},
    }
    artifacts = {
        "fixture": store.add(
            "fixture.json",
            {"expected_keys": csd_keys, "expected_scalar_keys": scalar_keys},
        ),
        "budget": store.add("budget.json", {}),
        "source_scales": store.add(
            "scales.json",
            {
                "schema": "gen1-stochastic-source-scales-v1",
                "family": "A2",
                "rows": [
                    {"key": key, "value": "1e0"} for key in csd_keys + scalar_keys
                ],
            },
        ),
        "records": store.add_raw("reference.jsonl", b"".join(lines)),
        "decision_kernel": store.add(
            "kernel.py", "not executed in this reporting probe"
        ),
    }
    stage = {
        "policy": store.add("policy.json", {"artifacts": artifacts}),
        "attempt": store.add(
            "attempt.json",
            {
                "artifacts": {
                    "records": store.add_raw(
                        "comparison.jsonl",
                        json.dumps(row, sort_keys=True).encode() + b"\n",
                    )
                }
            },
        ),
    }
    monkeypatch.setattr(arithmetic, "load_kernel", lambda *_args: UnusedKernel())
    summary = replay_a2_arithmetic(store, stage, partial=True)
    assert summary["passed_counts"] == {"science": 0, "auxiliary": 0}
    assert summary["retained_claimed_passed_counts"] == {"science": 1, "auxiliary": 0}
    assert summary["unreplayed_resolved_count"] == 1
    assert summary["unresolved_count"] == 83
    assert summary["numeric_delivery_gates_pass"] is False
    assert summary["independent_result_admission_verified"] is False
