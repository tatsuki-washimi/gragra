"""Independent reservation ledger and strict diagnostic admission checks."""

from __future__ import annotations

import hashlib
import base64
import json
import math
import os
import re
from pathlib import Path
from typing import Any


class LedgerError(ValueError):
    """A diagnostic authorization, reservation, identity or terminal is invalid."""


PHASE_SCHEDULE = {
    "A2": [
        {
            "phase_id": "environment",
            "wall_cap_seconds": 75,
            "subtree_reserved_seconds": 75,
            "roles": [{"role": "environment_leaf", "max_seconds": 75, "kind": "leaf"}],
        },
        {
            "phase_id": "accounting",
            "wall_cap_seconds": 60,
            "subtree_reserved_seconds": 100,
            "roles": [
                {"role": "accounting_test_controller", "max_seconds": 60, "kind": "controller"},
                {"role": "accounting_timer_fixture", "max_seconds": 10, "kind": "leaf"},
                {"role": "accounting_pdeath_refusal", "max_seconds": 10, "kind": "leaf"},
                {"role": "accounting_handshake_leaf", "max_seconds": 10, "kind": "leaf"},
                {"role": "accounting_cleanup_leaf", "max_seconds": 10, "kind": "leaf"},
            ],
        },
        {
            "phase_id": "retained",
            "wall_cap_seconds": 100,
            "subtree_reserved_seconds": 100,
            "roles": [{"role": "retained_leaf", "max_seconds": 100, "kind": "leaf"}],
        },
        {
            "phase_id": "metadata",
            "wall_cap_seconds": 100,
            "subtree_reserved_seconds": 150,
            "roles": [
                {"role": "metadata_controller", "max_seconds": 100, "kind": "controller"},
                {"role": "metadata_worker", "max_seconds": 45, "kind": "leaf"},
                {"role": "metadata_helper", "max_seconds": 5, "kind": "leaf"},
            ],
        },
        {
            "phase_id": "timing",
            "wall_cap_seconds": 200,
            "subtree_reserved_seconds": 400,
            "roles": [
                {"role": "timing_controller", "max_seconds": 200, "kind": "controller"},
                {"role": "timing_worker", "max_seconds": 150, "kind": "leaf"},
                {"role": "timing_helper", "max_seconds": 50, "kind": "leaf"},
            ],
        },
        {
            "phase_id": "terminal",
            "wall_cap_seconds": 75,
            "subtree_reserved_seconds": 75,
            "roles": [{"role": "record_leaf", "max_seconds": 75, "kind": "leaf"}],
        },
    ],
    "Cylinder": [
        {
            "phase_id": "preflight",
            "wall_cap_seconds": 30,
            "subtree_reserved_seconds": 30,
            "roles": {"cleanup_capacity": 10, "environment_leaf": 20},
        },
        {
            "phase_id": "regression-1",
            "wall_cap_seconds": 120,
            "subtree_reserved_seconds": 120,
            "roles": {
                "accounting_controller_leaf": 20,
                "cleanup_capacity": 10,
                "contained_probe_helper": 10,
                "regression_worker": 80,
            },
        },
        *[
            {
                "phase_id": f"regression-{number}",
                "wall_cap_seconds": 110,
                "subtree_reserved_seconds": 120,
                "roles": {
                    "cleanup_capacity": 10,
                    "regression_worker": 100,
                    "unused_helper_capacity": 10,
                },
            }
            for number in (2, 3, 4)
        ],
        *[
            {
                "phase_id": f"row-{number}",
                "wall_cap_seconds": 50,
                "subtree_reserved_seconds": 60,
                "requires": "all-four-regressions-PASS",
                "roles": {
                    "cleanup_capacity": 10,
                    "row_worker": 40,
                    "unused_helper_capacity": 10,
                },
            }
            for number in range(1, 7)
        ],
        {
            "phase_id": "terminal",
            "wall_cap_seconds": 30,
            "subtree_reserved_seconds": 30,
            "roles": {"cleanup_capacity": 10, "record_leaf": 20},
        },
    ],
}

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
ROOT_MANIFEST_SHA256_PLACEHOLDER = "ROOT_BOUND_MANIFEST_SHA256_V1"
_SIX_DEPENDENCY_PIN = {
    "module": "six",
    "version": "1.17.0",
    "path": "/home/washimi/miniforge3/envs/gragra/lib/python3.11/site-packages/six.py",
    "sha256": "c51c91f703d3d4b3696c923cb5fec213e05e75d9215393befac7f2fa6a3904df",
}
_EXPECTED_CHILD_TIMER_TERMINAL = {
    "profile": "a2-accounting-child-timer-sigkill-v1",
    "model": "A2",
    "phase": "accounting",
    "role": "accounting_timer_fixture",
    "signal": 9,
    "source": "child_owned_posix_timer",
}


def bind_root_manifest_sha256(payload: dict[str, Any], actual_sha256: str) -> dict[str, Any]:
    """Resolve the one self-referential child request field after manifest hashing."""
    if (
        not isinstance(payload, dict)
        or payload.get("manifest_sha256") != ROOT_MANIFEST_SHA256_PLACEHOLDER
        or not isinstance(actual_sha256, str)
        or _HEX64.fullmatch(actual_sha256) is None
    ):
        raise LedgerError("worker payload lacks the exact root-bound manifest digest slot")
    bound = dict(payload)
    bound["manifest_sha256"] = actual_sha256
    return bound


def validate_dependency_inventory(raw: Any) -> list[dict[str, str]]:
    """Validate the top-level immutable A2 import pins, including the reviewed six source."""
    if not isinstance(raw, list) or not raw:
        raise LedgerError("A2 dependency inventory is missing")
    names = []
    normalized = []
    for pin in raw:
        if (
            not isinstance(pin, dict)
            or set(pin) != {"module", "version", "path", "sha256"}
            or not isinstance(pin.get("module"), str)
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", pin["module"])
            or not isinstance(pin.get("version"), str)
            or not pin["version"]
            or not isinstance(pin.get("path"), str)
            or not Path(pin["path"]).is_absolute()
            or not isinstance(pin.get("sha256"), str)
            or _HEX64.fullmatch(pin["sha256"]) is None
        ):
            raise LedgerError("A2 dependency pin fields or values are malformed")
        names.append(pin["module"])
        normalized.append({name: pin[name] for name in ("module", "version", "path", "sha256")})
    if names != sorted(set(names)):
        raise LedgerError("A2 dependency inventory is duplicated or nondeterministic")
    if _SIX_DEPENDENCY_PIN not in normalized:
        raise LedgerError("A2 dependency inventory omits the exact reviewed six pin")
    return normalized


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LedgerError(f"{name} must be a finite nonnegative number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise LedgerError(f"{name} must be a finite nonnegative number")
    return number


def _phase(model: str, phase: str) -> dict[str, Any]:
    if model not in PHASE_SCHEDULE:
        raise LedgerError(f"unsupported model profile: {model}")
    found = [item for item in PHASE_SCHEDULE[model] if item["phase_id"] == phase]
    if len(found) != 1:
        raise LedgerError(f"phase is not in the frozen model schedule: {phase}")
    return found[0]


def _required_success_roles(model: str, phase: str) -> set[str]:
    """Roles that must run for a phase to claim success; spare capacity stays optional."""
    if model == "A2":
        return {
            "environment": {"environment_leaf"},
            "accounting": {
                "accounting_test_controller",
                "accounting_timer_fixture",
                "accounting_pdeath_refusal",
                "accounting_handshake_leaf",
                "accounting_cleanup_leaf",
            },
            "retained": {"retained_leaf"},
            "metadata": {"metadata_controller", "metadata_worker"},
            "timing": {"timing_controller", "timing_worker"},
            "terminal": {"record_leaf"},
        }[phase]
    if phase == "preflight":
        return {"environment_leaf"}
    if phase == "regression-1":
        return {"regression_worker", "accounting_controller_leaf", "contained_probe_helper"}
    if phase.startswith("regression-"):
        return {"regression_worker"}
    if phase.startswith("row-"):
        return {"row_worker"}
    if phase == "terminal":
        return {"record_leaf"}
    raise LedgerError(f"phase has no required success role: {phase}")


def validate_authorization(manifest: dict[str, Any], *, expected_model: str) -> dict[str, Any]:
    """Accept the fixed A2 diagnostic envelope; formal runs are never admitted."""
    if not isinstance(manifest, dict):
        raise LedgerError("manifest must be a JSON object")
    if expected_model not in PHASE_SCHEDULE:
        raise LedgerError("model has no frozen governor schedule")
    required = {
        "schema": "gragra-diagnostic-manifest-v1",
        "model": expected_model,
        "diagnostic_only": True,
        "science_credit": 0,
        "old_credit_seconds": 0,
        "root_seconds": 900,
        "root_cleanup_seconds": 60,
        "descendant_seconds": 900,
    }
    for name, value in required.items():
        if manifest.get(name) != value or type(manifest.get(name)) is not type(value):
            raise LedgerError(f"manifest {name} differs from the diagnostic envelope")
    if not _HEX64.fullmatch(str(manifest.get("authorization_revision_sha256", ""))):
        raise LedgerError("authorization revision pin is malformed")
    if not _HEX64.fullmatch(str(manifest.get("plan_sha256", ""))):
        raise LedgerError("plan pin is malformed")
    if not _HEX64.fullmatch(str(manifest.get("procedure_sha256", ""))):
        raise LedgerError("procedure pin is malformed")
    if not _HEX40.fullmatch(str(manifest.get("candidate_commit", ""))):
        raise LedgerError("candidate commit is malformed")
    if not _HEX40.fullmatch(str(manifest.get("candidate_tree", ""))):
        raise LedgerError("candidate tree is malformed")
    forbidden = {
        "formal",
        "formal_acceptance",
        "formal_comparison",
        "delivery",
        "acceptance",
        "science_acceptance",
        "unlimited",
        "repair",
        "reference_generation",
    }
    if forbidden & set(manifest):
        raise LedgerError("manifest contains a formal, delivery, repair or unbounded mode")
    for name in ("root_seconds", "root_cleanup_seconds", "descendant_seconds", "old_credit_seconds"):
        _finite_nonnegative(manifest.get(name), name)
    if manifest.get("phase_schedule") != PHASE_SCHEDULE[expected_model]:
        raise LedgerError("phase schedule differs from the frozen diagnostic schedule")
    total_descendant = sum(
        phase["subtree_reserved_seconds"] for phase in PHASE_SCHEDULE[expected_model]
    )
    total_root = sum(phase["wall_cap_seconds"] for phase in PHASE_SCHEDULE[expected_model])
    if total_descendant != 900 or total_root > 840:
        raise LedgerError("frozen phase totals exceed their diagnostic envelope")
    return {
        "model": expected_model,
        "science_credit": 0,
        "old_credit_seconds": 0,
        "root_seconds": 900,
        "root_cleanup_seconds": 60,
        "descendant_seconds": 900,
        "phase_schedule": PHASE_SCHEDULE[expected_model],
    }


def validate_role_inventory(model: str, phase: str, roles: list[dict[str, Any]]) -> None:
    """Require the complete, exact role names and lifetime maxima for one phase."""
    definition = _phase(model, phase)
    if not isinstance(roles, list):
        raise LedgerError("role inventory must be a list")
    observed = []
    for item in roles:
        if not isinstance(item, dict):
            raise LedgerError("each role inventory item must be an object")
        if not isinstance(item.get("role"), str) or type(item.get("max_seconds")) is not int:
            raise LedgerError("role name or fixed lifetime maximum is malformed")
        _finite_nonnegative(item.get("max_seconds"), "role.max_seconds")
        observed.append({"role": item.get("role"), "max_seconds": item.get("max_seconds")})
    if isinstance(definition["roles"], dict):
        expected = [
            {"role": role, "max_seconds": maximum}
            for role, maximum in definition["roles"].items()
        ]
    else:
        expected = [
            {"role": item["role"], "max_seconds": item["max_seconds"]}
            for item in definition["roles"]
        ]
    if observed != expected:
        raise LedgerError("role inventory is missing, duplicated, reordered or over/under-reserved")


def validate_launch_binding(expected: dict[str, Any], observed: dict[str, Any]) -> None:
    """Compare every identity component that binds the gated child startup."""
    fields = (
        "pid",
        "parent_pid",
        "boot_id",
        "starttime_ticks",
        "parent_boot_id",
        "parent_starttime_ticks",
        "pdeath_signal",
        "argv",
        "launch_token",
        "phase",
        "model",
        "ledger_id",
    )
    if not isinstance(expected, dict) or not isinstance(observed, dict):
        raise LedgerError("launch binding objects are malformed")
    for field in fields:
        if field not in expected or field not in observed or observed[field] != expected[field]:
            raise LedgerError(f"child launch identity differs at {field}")
    for field in ("pid", "parent_pid", "starttime_ticks", "parent_starttime_ticks"):
        if type(observed[field]) is not int or observed[field] <= 0:
            raise LedgerError(f"launch {field} must be a positive integer")
    for field in ("boot_id", "parent_boot_id", "launch_token", "phase", "model", "ledger_id"):
        if not isinstance(observed[field], str) or not observed[field]:
            raise LedgerError(f"launch {field} must be a nonempty string")
    argv = observed["argv"]
    if (
        type(observed["pdeath_signal"]) is not int
        or observed["pdeath_signal"] != 9
        or not isinstance(argv, list)
        or not argv
        or any(not isinstance(part, str) or not part for part in argv)
        or not observed["launch_token"]
    ):
        raise LedgerError("launch must bind argv, token and PDEATHSIG=SIGKILL")


def validate_terminal_witness(expected: dict[str, Any], observed: dict[str, Any]) -> None:
    """Reject contradictory exit/signal or identity/cleanup evidence."""
    fields = ("pid", "starttime_ticks", "launch_token", "boot_id", "model", "ledger_id")
    for field in fields:
        if field not in expected or field not in observed:
            raise LedgerError(f"terminal identity field is absent: {field}")
    if type(expected["pid"]) is not int or expected["pid"] <= 0:
        raise LedgerError("terminal expected PID is invalid")
    if type(expected["starttime_ticks"]) is not int or expected["starttime_ticks"] <= 0:
        raise LedgerError("terminal expected starttime is invalid")
    if not isinstance(expected["launch_token"], str) or not expected["launch_token"]:
        raise LedgerError("terminal expected launch token is invalid")
    for field in fields:
        if field in {"pid", "starttime_ticks"}:
            if type(observed[field]) is not int or observed[field] <= 0:
                raise LedgerError(f"terminal {field} is invalid")
        elif not isinstance(observed[field], str) or not observed[field]:
            raise LedgerError(f"terminal {field} is invalid")
        if observed[field] != expected[field]:
            raise LedgerError("terminal process identity differs from its start witness")
    code, sig = observed.get("exit_code"), observed.get("signal")
    if code is not None and (type(code) is not int or code < 0):
        raise LedgerError("terminal exit code is invalid")
    if sig is not None and (type(sig) is not int or sig <= 0):
        raise LedgerError("terminal signal is invalid")
    if (code is None) == (sig is None):
        raise LedgerError("terminal witness must report exactly one of exit code or signal")
    if code != expected.get("exit_code") or sig != expected.get("signal"):
        raise LedgerError("terminal exit fields differ from the expected witness")
    if (
        type(observed.get("cleanup_confirmed")) is not bool
        or observed["cleanup_confirmed"] is not True
        or type(expected.get("cleanup_confirmed")) is not bool
        or expected["cleanup_confirmed"] is not True
    ):
        raise LedgerError("terminal witness lacks confirmed cleanup")


def validate_controller_broker_eof(
    expected_roles: list[str], requested_roles: list[str]
) -> None:
    """Accept broker EOF only after one request for every reserved child role."""
    if (
        not isinstance(expected_roles, list)
        or not isinstance(requested_roles, list)
        or any(not isinstance(role, str) or not role for role in expected_roles)
        or any(not isinstance(role, str) or not role for role in requested_roles)
        or len(expected_roles) != len(set(expected_roles))
        or len(requested_roles) != len(set(requested_roles))
        or set(expected_roles) != set(requested_roles)
    ):
        raise LedgerError(
            "controller broker closed before exactly one request for every reserved child role"
        )


def validate_controller_terminal(controller: dict[str, Any], children: list[dict[str, Any]]) -> None:
    """Require a positive parent wait and a terminal record for every reserved role."""
    roles = [child.get("role") for child in children]
    if len(roles) != len(set(roles)) or any(not isinstance(role, str) for role in roles):
        raise LedgerError("controller child roles are duplicated or malformed")
    waited = controller.get("waited_for_roles")
    if not isinstance(waited, list) or len(waited) != len(set(waited)) or set(waited) != set(roles):
        raise LedgerError("controller did not wait for every reserved child role")
    wait_seconds = _finite_nonnegative(controller.get("wait_seconds"), "controller.wait_seconds")
    if wait_seconds <= 0:
        raise LedgerError("controller wait duration must be positive")
    for child in children:
        if child.get("status") not in {"success", "failure", "not_started", "not_started_error"}:
            raise LedgerError("child terminal status is absent")
        if child.get("reaped") is not True:
            raise LedgerError("child has not been reaped")
    if controller.get("status") not in {"success", "failure"}:
        raise LedgerError("controller terminal status is absent")


def validate_cleanup_receipt(receipt: dict[str, Any]) -> None:
    """Only an empty, positively confirmed inventory closes cleanup."""
    if (
        not isinstance(receipt, dict)
        or type(receipt.get("confirmed")) is not bool
        or receipt["confirmed"] is not True
    ):
        raise LedgerError("cleanup is unconfirmed")
    for field in (
        "unknown_pids",
        "live_identities",
        "nonempty_process_groups",
        "unreaped_children",
        "unknown_reaped_children",
        "visibility_errors",
        "task_children_pids",
    ):
        value = receipt.get(field)
        if not isinstance(value, list) or value:
            raise LedgerError(f"cleanup field {field} is missing or nonempty")
    reaped = receipt.get("reaped_identities")
    if not isinstance(reaped, list):
        raise LedgerError("cleanup reaped identity evidence is missing or malformed")
    for identity in reaped:
        if (
            not isinstance(identity, dict)
            or type(identity.get("pid")) is not int
            or identity["pid"] <= 0
            or type(identity.get("starttime_ticks")) is not int
            or identity["starttime_ticks"] <= 0
            or type(identity.get("wait_status")) is not int
        ):
            raise LedgerError("cleanup reaped identity evidence is malformed")


def _validate_host_worker_admission(
    root_start: dict[str, Any], root_terminal: dict[str, Any], receipt: dict[str, Any], *, run_id: str, model: str
) -> None:
    """Replay the fixed two-slot host lease without treating it as native-exit proof."""
    directory = "/home/washimi/.local/state/gragra/diagnostic-execution/host-worker-admission-v1"
    start = root_start.get("host_worker_admission")
    terminal = root_terminal.get("host_worker_admission")
    if not isinstance(start, dict) or not isinstance(terminal, dict):
        raise LedgerError("root ledger lacks manifest-bound host admission evidence")
    if (
        start.get("schema") != "gragra.diagnostic.host-worker-admission.v1"
        or start.get("convention_id") != "gragra-host-scientific-worker-cap-2-v1"
        or start.get("capacity") != 2
        or start.get("status") not in {"acquired", "refused"}
        or not isinstance(start.get("observed_slots"), list)
        or start.get("release") is not None
    ):
        raise LedgerError("root host admission acquisition record is malformed")
    preparation = start.get("directory_preparation")
    root_external_pins = root_start.get("external_pins")
    external_preparation = (
        root_external_pins.get("host_admission_preparation")
        if isinstance(root_external_pins, dict)
        else None
    )
    if (
        not isinstance(preparation, dict)
        or preparation.get("schema") != "gragra.diagnostic.host-admission-directory-preparation.v1"
        or preparation.get("step") != "Step3"
        or preparation.get("target_directory") != directory
        or preparation.get("governor_created_directory") is not False
        or preparation.get("pin") != {
            "path": external_preparation.get("path") if isinstance(external_preparation, dict) else None,
            "sha256": external_preparation.get("sha256") if isinstance(external_preparation, dict) else None,
        }
        or type(preparation.get("directory_exists_before_admission")) is not bool
        or (start.get("status") == "acquired" and preparation.get("directory_exists_before_admission") is not True)
        or (preparation.get("directory_exists_before_admission") is False and start.get("status") != "refused")
    ):
        raise LedgerError("host admission directory preparation evidence is absent or contradictory")
    normalized_terminal = dict(terminal)
    normalized_terminal["release"] = None
    if normalized_terminal != start:
        raise LedgerError("root host admission terminal evidence differs from its acquired slot")
    release = terminal.get("release")
    if not isinstance(release, dict) or release.get("cleanup_confirmed") is not receipt.get("confirmed"):
        raise LedgerError("host admission release does not match cleanup confirmation")
    if start["status"] == "acquired":
        slot_id = start.get("slot_id")
        owner = start.get("owner")
        if (
            slot_id not in {"slot-0", "slot-1"}
            or start.get("lock_path") != f"{directory}/{slot_id}.lock"
            or not isinstance(owner, dict)
            or owner.get("schema") != "gragra.diagnostic.host-worker-slot-owner.v1"
            or owner.get("convention_id") != "gragra-host-scientific-worker-cap-2-v1"
            or owner.get("capacity") != 2
            or owner.get("slot_id") != slot_id
            or owner.get("model") != model
            or owner.get("epoch_id") != run_id
            or owner.get("pid") != root_start.get("pid")
            or owner.get("starttime_ticks") != root_start.get("starttime_ticks")
            or owner.get("boot_id") != root_start.get("boot_id")
            or owner.get("manifest_sha256") != root_start.get("manifest_sha256")
            or type(owner.get("manifest_sha256")) is not str
            or _HEX64.fullmatch(owner["manifest_sha256"]) is None
            or type(start.get("owner_record_sha256")) is not str
            or _HEX64.fullmatch(start["owner_record_sha256"]) is None
            or hashlib.sha256(_canonical_bytes(owner)).hexdigest() != start.get("owner_record_sha256")
        ):
            raise LedgerError("host admission owner identity is not bound to this root")
        allowed_releases = (
            {"released_after_confirmed_cleanup"}
            if receipt.get("confirmed") is True
            else {"stale_owner_retained_cleanup_unconfirmed", "release_unconfirmed"}
        )
        if (
            release.get("status") not in allowed_releases
            or release.get("slot_id") != slot_id
            or release.get("lock_path") != start.get("lock_path")
            or release.get("owner_record_sha256") != start.get("owner_record_sha256")
        ):
            raise LedgerError("host admission release/stale-owner status contradicts cleanup")
        if release.get("owner_record_cleared") is not (receipt.get("confirmed") is True):
            raise LedgerError("host admission owner-record clear contradicts cleanup status")
    else:
        if (
            start.get("slot_id") is not None
            or release.get("status") != "not_owned"
            or root_terminal.get("status") == "success"
        ):
            raise LedgerError("refused host admission contradicts root terminal status")
    if receipt.get("host_worker_admission") != release:
        raise LedgerError("cleanup receipt host admission release differs from ledger terminal")


def _validate_expected_timer_terminal(event: dict[str, Any]) -> None:
    """Accept only the one preauthorized child-owned SIGKILL accounting fixture."""
    try:
        from .deadline import DeadlineError, verify_new_proc_timer
    except ImportError:
        from deadline import DeadlineError, verify_new_proc_timer

    if event.get("expected_terminal") != _EXPECTED_CHILD_TIMER_TERMINAL:
        raise LedgerError("expected-negative terminal profile differs from its frozen role")
    evidence = event.get("expected_terminal_evidence")
    required = {
        "timer_raw_before_base64",
        "timer_raw_after_base64",
        "child_timer_record",
        "timer_observed_monotonic_ns",
        "terminal_monotonic_ns",
        "role_deadline_monotonic_ns",
        "role_max_deadline_monotonic_ns",
        "shutdown_allowance_ns",
        "root_soft_deadline_monotonic_ns",
        "root_deadline_monotonic_ns",
        "child_timer_owned",
        "native_wait_signal",
        "root_timeout_fired",
        "rss_limit_fired",
        "cleanup_signal_sent",
    }
    if not isinstance(evidence, dict) or set(evidence) != required:
        raise LedgerError("expected timer terminal evidence fields differ")
    for name in (
        "timer_observed_monotonic_ns",
        "terminal_monotonic_ns",
        "role_deadline_monotonic_ns",
        "role_max_deadline_monotonic_ns",
        "root_soft_deadline_monotonic_ns",
        "root_deadline_monotonic_ns",
    ):
        if type(evidence.get(name)) is not int or evidence[name] <= 0:
            raise LedgerError(f"timer terminal {name} is invalid")
    if not (
        evidence["timer_observed_monotonic_ns"]
        < evidence["terminal_monotonic_ns"]
        < evidence["role_deadline_monotonic_ns"]
        < evidence["role_max_deadline_monotonic_ns"]
        < evidence["root_soft_deadline_monotonic_ns"]
        < evidence["root_deadline_monotonic_ns"]
    ):
        raise LedgerError("child timer signal was not observed before role/root deadlines")
    if (
        type(evidence.get("shutdown_allowance_ns")) is not int
        or evidence["role_max_deadline_monotonic_ns"] - evidence["role_deadline_monotonic_ns"]
        != evidence["shutdown_allowance_ns"]
        or evidence["shutdown_allowance_ns"] <= 0
        or evidence["shutdown_allowance_ns"] > 2_000_000_000
        or evidence["terminal_monotonic_ns"] != event.get("terminal_monotonic_ns")
        or evidence["role_deadline_monotonic_ns"] != event.get("role_deadline_monotonic_ns")
        or evidence["role_max_deadline_monotonic_ns"] != event.get("role_max_deadline_monotonic_ns")
    ):
        raise LedgerError("child timer deadline fields differ from the reaped role terminal")
    if (
        evidence.get("child_timer_owned") is not True
        or evidence.get("native_wait_signal") != 9
        or evidence.get("root_timeout_fired") is not False
        or evidence.get("rss_limit_fired") is not False
        or evidence.get("cleanup_signal_sent") is not False
        or event.get("signal") != 9
        or event.get("exit_code") is not None
        or type(event.get("wait_status")) is not int
        or event["wait_status"] != 9
        or event.get("popen_returncode") != -9
        or event.get("result_status") != "PASS"
    ):
        raise LedgerError("child timer terminal contradicts native SIGKILL/reap evidence")
    try:
        before = base64.b64decode(evidence["timer_raw_before_base64"], validate=True)
        after = base64.b64decode(evidence["timer_raw_after_base64"], validate=True)
        observed_record = verify_new_proc_timer(before, after, root_pid=event["pid"])
    except (ValueError, KeyError, TypeError, DeadlineError) as exc:
        raise LedgerError("child timer procfs readback does not prove a new SIGKILL timer") from exc
    if observed_record != evidence.get("child_timer_record"):
        raise LedgerError("child timer record differs from its raw /proc readback")


def validate_event_log(events: list[dict[str, Any]], *, run_id: str, model: str) -> None:
    """Replay reservations, child lifecycle and a ROOT_TERMINAL intent awaiting native exit."""
    if model not in PHASE_SCHEDULE or not isinstance(run_id, str) or not run_id:
        raise LedgerError("event log model or run identifier is invalid")
    if not isinstance(events, list) or len(events) < 2 or any(not isinstance(row, dict) for row in events):
        raise LedgerError("event log is incomplete or contains a non-object event")
    names = [event.get("event") for event in events]
    if names[0] != "ROOT_START" or names[-1] != "ROOT_TERMINAL" or names.count("ROOT_START") != 1:
        raise LedgerError("event log has a restart, missing root start or contradictory root-terminal intent")
    allowed = {
        "ROOT_START",
        "PHASE_RESERVE",
        "CHILD_START",
        "CHILD_TERMINAL",
        "ROLE_NOT_STARTED",
        "PHASE_TERMINAL",
        "ROOT_TERMINAL",
    }
    if any(name not in allowed for name in names):
        raise LedgerError("event log contains an unknown event type")

    for event in events:
        if event.get("run_id") != run_id or event.get("model") != model:
            raise LedgerError("event log run/model identity differs")
        if "old_credit_seconds" in event and event["old_credit_seconds"] != 0:
            raise LedgerError("old credit is forbidden")
        if "refund_seconds" in event or "borrowed_seconds" in event:
            raise LedgerError("reservation refunds/borrowing are forbidden")

    root_start = events[0]
    if type(root_start.get("reserved_seconds")) is not int or root_start["reserved_seconds"] != 900:
        raise LedgerError("root start reservation differs from the immutable envelope")
    for field in ("pid", "starttime_ticks"):
        if type(root_start.get(field)) is not int or root_start[field] <= 0:
            raise LedgerError(f"root start {field} is missing or invalid")
    if not isinstance(root_start.get("boot_id"), str) or not root_start["boot_id"]:
        raise LedgerError("root start boot ID is missing")
    if (
        type(root_start.get("root_deadline_monotonic_ns")) is not int
        or type(root_start.get("soft_deadline_monotonic_ns")) is not int
        or root_start["root_deadline_monotonic_ns"] - root_start["soft_deadline_monotonic_ns"]
        != 60_000_000_000
    ):
        raise LedgerError("root monotonic deadlines are missing or contradictory")

    schedule = PHASE_SCHEDULE[model]
    schedule_names = [item["phase_id"] for item in schedule]
    next_phase_index = 0
    current_phase: str | None = None
    phase_status: dict[str, str] = {}
    reserved_roles: dict[str, dict[str, Any]] = {}
    role_state: dict[str, str] = {}
    starts_by_role: dict[str, dict[str, Any]] = {}
    starts_by_token: dict[str, dict[str, Any]] = {}
    terminal_child_markers: list[tuple[int, int]] = []
    root_terminal: dict[str, Any] | None = None

    def phase_roles(phase_id: str) -> list[dict[str, Any]]:
        definition = _phase(model, phase_id)
        roles = definition["roles"]
        if isinstance(roles, dict):
            return [{"role": role, "max_seconds": cap} for role, cap in roles.items()]
        return [
            {
                "role": item["role"],
                "max_seconds": item["max_seconds"],
                **({"kind": item["kind"]} if "kind" in item else {}),
            }
            for item in roles
        ]

    def require_current(event: dict[str, Any]) -> str:
        phase = event.get("phase")
        if not isinstance(phase, str) or phase != current_phase:
            raise LedgerError("child event is outside its currently reserved phase")
        return phase

    def require_reserved_role(phase: str, role: Any) -> str:
        if not isinstance(role, str) or role not in reserved_roles:
            raise LedgerError("child role was not reserved by its phase")
        if reserved_roles[role]["phase"] != phase:
            raise LedgerError("child role is reserved under another phase")
        return role

    for index, event in enumerate(events[1:-1], start=1):
        name = event["event"]
        if name == "PHASE_RESERVE":
            if current_phase is not None or next_phase_index >= len(schedule_names):
                raise LedgerError("phase reservation occurs before closure or after the schedule")
            phase = event.get("phase")
            if phase != schedule_names[next_phase_index]:
                raise LedgerError("phase reservation violates the immutable phase order")
            if any(value != "success" for value in phase_status.values()):
                raise LedgerError("phase reservation continues after an earlier failed phase")
            if type(event.get("reserved_seconds")) is not int:
                raise LedgerError("phase reservation seconds must be an integer")
            expected_seconds = _phase(model, phase)["subtree_reserved_seconds"]
            if event["reserved_seconds"] != expected_seconds:
                raise LedgerError("phase reservation differs from the immutable schedule")
            roles = event.get("roles")
            validate_role_inventory(model, phase, roles)
            current_phase = phase
            reserved_roles = {item["role"]: {**item, "phase": phase} for item in roles}
            role_state = {}
            starts_by_role = {}
            next_phase_index += 1
            if _phase(model, phase).get("requires") == "all-four-regressions-PASS":
                if any(phase_status.get(f"regression-{n}") != "success" for n in range(1, 5)):
                    raise LedgerError("conditional row phase lacks four successful regression terminals")
        elif name == "ROLE_NOT_STARTED":
            phase = require_current(event)
            role = require_reserved_role(phase, event.get("role"))
            if role in role_state or event.get("status") != "not_started":
                raise LedgerError("role not-started record is duplicated or malformed")
            reason = event.get("reason")
            if not isinstance(reason, str) or not reason:
                raise LedgerError("not-started role needs an explicit reason")
            role_state[role] = "not_started"
        elif name == "CHILD_START":
            phase = require_current(event)
            role = require_reserved_role(phase, event.get("role"))
            if role in role_state:
                raise LedgerError("reserved role has more than one start or disposition")
            role_start = event.get("role_start_monotonic_ns")
            role_deadline = event.get("role_deadline_monotonic_ns")
            role_max_deadline = event.get("role_max_deadline_monotonic_ns")
            shutdown_allowance = event.get("shutdown_allowance_ns")
            role_cap_ns = reserved_roles[role]["max_seconds"] * 1_000_000_000
            if (
                type(role_start) is not int
                or type(role_deadline) is not int
                or type(role_max_deadline) is not int
                or type(shutdown_allowance) is not int
                or role_start <= 0
                or role_deadline <= role_start
                or role_max_deadline <= role_deadline
                or role_max_deadline - role_start > role_cap_ns
                or shutdown_allowance != role_max_deadline - role_deadline
                or shutdown_allowance < 250_000_000
                or shutdown_allowance > 2_000_000_000
            ):
                raise LedgerError("child start deadlines exceed its fixed reserved role maximum")
            if (
                model == "Cylinder"
                and phase == "regression-1"
                and role == "contained_probe_helper"
                and role_state.get("accounting_controller_leaf") != "success"
            ):
                raise LedgerError(
                    "contained-probe helper start precedes successful accounting-controller terminal"
                )
            if model == "Cylinder" and phase == "regression-1" and role == "regression_worker":
                prerequisites = ("accounting_controller_leaf", "contained_probe_helper")
                if any(role_state.get(prerequisite) != "success" for prerequisite in prerequisites):
                    raise LedgerError(
                        "regression-1 worker start precedes exact accounting and contained-probe success"
                    )
            validate_launch_binding(event, event)
            if (
                event["pid"] == root_start["pid"]
                or event["parent_pid"] != root_start["pid"]
                or event["parent_boot_id"] != root_start["boot_id"]
                or event["parent_starttime_ticks"] != root_start["starttime_ticks"]
                or event["phase"] != phase
                or event["model"] != model
                or event["ledger_id"] != f"{model}:{run_id}"
            ):
                raise LedgerError("child start is not bound to the exact direct root launch")
            token = event["launch_token"]
            if token in starts_by_token:
                raise LedgerError("launch token was reused")
            role_state[role] = "running"
            starts_by_role[role] = event
            starts_by_token[token] = event
        elif name == "CHILD_TERMINAL":
            phase = require_current(event)
            role = require_reserved_role(phase, event.get("role"))
            start = starts_by_role.get(role)
            if role_state.get(role) != "running" or start is None:
                raise LedgerError("child terminal has no unique admitted start")
            expected_identity = {
                **start,
                "cleanup_confirmed": True,
                "exit_code": event.get("exit_code"),
                "signal": event.get("signal"),
            }
            validate_terminal_witness(expected_identity, event)
            if event.get("reaped") is not True:
                raise LedgerError("child terminal lacks its positive Popen/wait reaping result")
            if event.get("cleanup_confirmed") is not True:
                raise LedgerError("child terminal carries unconfirmed cleanup")
            role_deadline = event.get("role_deadline_monotonic_ns")
            role_max_deadline = event.get("role_max_deadline_monotonic_ns")
            role_start = event.get("role_start_monotonic_ns")
            shutdown_allowance = event.get("shutdown_allowance_ns")
            terminal_observed = event.get("terminal_monotonic_ns")
            late_by = event.get("late_by_ns")
            dispatch_late_by = event.get("dispatch_late_by_ns")
            maximum_overrun = event.get("maximum_overrun_ns")
            late_terminal = event.get("late_terminal")
            if (
                type(role_start) is not int
                or type(role_deadline) is not int
                or type(role_max_deadline) is not int
                or type(shutdown_allowance) is not int
                or type(terminal_observed) is not int
                or type(late_by) is not int
                or type(dispatch_late_by) is not int
                or type(maximum_overrun) is not int
                or type(event.get("maximum_overrun")) is not bool
                or type(event.get("cleanup_unconfirmed_at_role_maximum")) is not bool
                or type(late_terminal) is not bool
                or type(event.get("role_timeout_fired")) is not bool
                or not isinstance(event.get("terminal_observation_basis"), str)
                or role_start != start.get("role_start_monotonic_ns")
                or role_deadline != start.get("role_deadline_monotonic_ns")
                or role_max_deadline != start.get("role_max_deadline_monotonic_ns")
                or shutdown_allowance != start.get("shutdown_allowance_ns")
                or role_max_deadline - role_start
                > reserved_roles[role]["max_seconds"] * 1_000_000_000
                or role_deadline <= role_start
                or role_deadline <= 0
                or role_max_deadline < role_deadline
                or shutdown_allowance != role_max_deadline - role_deadline
                or shutdown_allowance < 250_000_000
                or shutdown_allowance > 2_000_000_000
                or terminal_observed <= 0
                or terminal_observed < role_start
                or late_by != max(0, terminal_observed - role_deadline)
                or dispatch_late_by != late_by
                or late_terminal is not (late_by > 0)
                or maximum_overrun != max(0, terminal_observed - role_max_deadline)
                or event.get("maximum_overrun") is not (maximum_overrun > 0)
            ):
                raise LedgerError("child terminal role deadline/shutdown evidence is malformed")
            if terminal_observed > role_max_deadline or maximum_overrun > 0:
                raise LedgerError("child terminal was observed beyond its reserved role maximum")
            if event.get("cleanup_unconfirmed_at_role_maximum"):
                raise LedgerError("child terminal followed an unconfirmed maximum-deadline reap")
            terminal_child_markers.append((event["pid"], event["starttime_ticks"]))
            status = event.get("status")
            if status not in {"success", "failed", "refused", "timeout"}:
                raise LedgerError("child terminal lifecycle status is invalid")
            if status == "success":
                if event.get("role_timeout_fired") or event.get("late_terminal"):
                    raise LedgerError("successful child terminal was observed after its role deadline")
                expected_terminal = event.get("expected_terminal")
                if expected_terminal is None:
                    if event.get("exit_code") != 0 or event.get("signal") is not None:
                        raise LedgerError("successful role has a nonzero or signaled process terminal")
                    if event.get("result_status") != "PASS":
                        raise LedgerError("successful role lacks its family PASS result")
                elif (
                    model != "A2"
                    or phase != "accounting"
                    or role != "accounting_timer_fixture"
                ):
                    raise LedgerError("expected-negative terminal is outside its frozen role")
                else:
                    _validate_expected_timer_terminal(event)
            elif event.get("result_status") not in {"FAIL", "REFUSED", None}:
                raise LedgerError("failed role has a contradictory family result")
            role_state[role] = status
        elif name == "PHASE_TERMINAL":
            phase = require_current(event)
            expected_role_names = [item["role"] for item in phase_roles(phase)]
            if set(role_state) != set(expected_role_names):
                raise LedgerError("phase terminal omits a reserved role disposition")
            statuses = [role_state[role] for role in expected_role_names]
            status = event.get("status")
            if status not in {"success", "failed"}:
                raise LedgerError("phase terminal status is invalid")
            if status == "success" and any(item not in {"success", "not_started"} for item in statuses):
                raise LedgerError("successful phase contains a failed or running role")
            if status == "success" and any(
                role_state.get(role) != "success" for role in _required_success_roles(model, phase)
            ):
                raise LedgerError("successful phase omitted a required execution role")
            wait_seconds = _finite_nonnegative(event.get("wait_seconds"), "phase.wait_seconds")
            if wait_seconds <= 0:
                raise LedgerError("phase close lacks a positive charged wait interval")
            waited = event.get("waited_for_roles")
            expected_waited = [role for role in expected_role_names if role != next(
                (item["role"] for item in phase_roles(phase) if item.get("kind") == "controller"), None
            )]
            if not isinstance(waited, list) or waited != expected_waited:
                raise LedgerError("phase close did not reconcile every non-controller role wait")
            if any(state == "running" for state in statuses):
                raise LedgerError("phase close contains an unresolved child start")
            phase_status[phase] = status
            current_phase = None
            reserved_roles = {}
            role_state = {}
            starts_by_role = {}
        else:
            raise LedgerError(f"event {name} is not valid before root terminal")

    root_terminal = events[-1]
    status = root_terminal.get("status")
    if status not in {"success", "failed", "unconfirmed_cleanup"}:
        raise LedgerError("root-terminal intent status is invalid")
    if (
        root_terminal.get("root_terminal_intent_schema") != "gragra-diagnostic-root-terminal-intent-v1"
        or type(root_terminal.get("root_terminal_intent_sha256")) is not str
        or _HEX64.fullmatch(root_terminal["root_terminal_intent_sha256"]) is None
        or root_terminal.get("native_root_exit_receipt_required") is not True
        or root_terminal.get("native_process_exit_confirmed") is not False
        or type(root_terminal.get("root_terminal_monotonic_ns")) is not int
        or root_terminal["root_terminal_monotonic_ns"] <= 0
    ):
        raise LedgerError("root-terminal ledger record must be an intent with external native-exit confirmation")
    if current_phase is not None:
        raise LedgerError("root terminal occurs before reserved phase roles are closed")
    for field in ("pid", "starttime_ticks", "boot_id"):
        if field not in root_terminal or root_terminal[field] != root_start.get(field):
            raise LedgerError("root terminal identity differs from root start")
    receipt = root_terminal.get("cleanup_receipt")
    if not isinstance(receipt, dict):
        raise LedgerError("root terminal lacks full cleanup receipt evidence")
    _validate_host_worker_admission(root_start, root_terminal, receipt, run_id=run_id, model=model)
    if status == "unconfirmed_cleanup":
        if type(receipt.get("confirmed")) is not bool or receipt["confirmed"] is not False:
            raise LedgerError("unconfirmed cleanup terminal contradicts its receipt")
        for field in (
            "unknown_pids",
            "live_identities",
            "nonempty_process_groups",
            "unreaped_children",
            "unknown_reaped_children",
            "visibility_errors",
            "task_children_pids",
            "reaped_identities",
        ):
            if not isinstance(receipt.get(field), list):
                raise LedgerError(f"cleanup field {field} is missing or malformed")
    else:
        validate_cleanup_receipt(receipt)
        reaped_markers = []
        for identity in receipt["reaped_identities"]:
            reaped_markers.append((identity["pid"], identity["starttime_ticks"]))
        if len(reaped_markers) != len(set(reaped_markers)):
            raise LedgerError("cleanup receipt repeats a reaped process identity")
        if set(reaped_markers) != set(terminal_child_markers):
            raise LedgerError("cleanup reaped identities do not reconcile with child terminals")
    if status == "success":
        if next_phase_index != len(schedule_names) or set(phase_status) != set(schedule_names):
            raise LedgerError("successful root terminal lacks every scheduled phase")
        if any(value != "success" for value in phase_status.values()):
            raise LedgerError("successful root terminal contains a failed phase")


def _canonical_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError) as exc:
        raise LedgerError("ledger event is not finite canonical JSON") from exc


class Ledger:
    """One-create, one-writer, fsync-on-every-event hash-chained JSONL ledger."""

    def __init__(self, path: str | Path, *, run_id: str, model: str) -> None:
        if model not in PHASE_SCHEDULE or not run_id:
            raise LedgerError("ledger model or run identity is invalid")
        self.path = Path(path)
        self.run_id = run_id
        self.model = model
        self._previous = "0" * 64
        self._sequence = 0
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise LedgerError("ledger path already exists; restart/reuse is forbidden") from exc
        except OSError as exc:
            raise LedgerError("ledger cannot be created exclusively") from exc
        self._fsync_directory(self.path.parent)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def append(self, event: str, **fields: Any) -> dict[str, Any]:
        if not isinstance(event, str) or not event:
            raise LedgerError("ledger event name is empty")
        if {"event", "run_id", "model", "sequence", "previous_sha256", "event_sha256"} & set(fields):
            raise LedgerError("ledger-owned identity fields cannot be overridden")
        for name, value in fields.items():
            if name.endswith("_seconds") or name == "reserved_seconds":
                _finite_nonnegative(value, name)
        payload = {
            "event": event,
            "run_id": self.run_id,
            "model": self.model,
            "sequence": self._sequence,
            "previous_sha256": self._previous,
            **fields,
        }
        digest = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
        record = {**payload, "event_sha256": digest}
        raw = _canonical_bytes(record)
        view = memoryview(raw)
        while view:
            written = os.write(self.fd, view)
            if written <= 0:
                raise LedgerError("short ledger write")
            view = view[written:]
        os.fsync(self.fd)
        self._previous = digest
        self._sequence += 1
        return record

    def close(self) -> None:
        os.close(self.fd)


def verify_chain(raw: bytes, *, run_id: str, model: str) -> list[dict[str, Any]]:
    """Parse and validate a saved hash chain; duplicate fields are rejected."""
    events: list[dict[str, Any]] = []
    previous = "0" * 64
    for expected_sequence, line in enumerate(raw.splitlines()):
        def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in items:
                if key in result:
                    raise LedgerError("duplicate JSON object key in ledger")
                result[key] = value
            return result

        try:
            event = json.loads(
                line.decode("utf-8"),
                object_pairs_hook=pairs,
                parse_constant=lambda value: (_ for _ in ()).throw(LedgerError(value)),
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LedgerError("ledger line is malformed") from exc
        if (
            not isinstance(event, dict)
            or event.get("run_id") != run_id
            or event.get("model") != model
            or event.get("sequence") != expected_sequence
            or event.get("previous_sha256") != previous
        ):
            raise LedgerError("ledger chain sequence or identity differs")
        claimed = event.pop("event_sha256", None)
        if not isinstance(claimed, str) or claimed != hashlib.sha256(_canonical_bytes(event)).hexdigest():
            raise LedgerError("ledger event hash differs")
        event["event_sha256"] = claimed
        previous = claimed
        events.append(event)
    validate_event_log(events, run_id=run_id, model=model)
    return events
