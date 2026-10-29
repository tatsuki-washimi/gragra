"""Frozen resource enforcement for the six independently admitted families."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from .transport import (
    SEAL_SCHEMA,
    GateError,
    _write_new,
    artifact_pin,
    canonical_bytes,
    exact_keys,
    require_external_directory,
    strict_json,
    verify_artifact,
    verify_authorization,
)

TOKEN = "GRAGRA_STOCHASTIC_FAMILY_RESOURCE_LAUNCH"


def verify_family_result(
    prepared: dict,
    *,
    attempt_pin: dict,
    supervisor_pin: dict,
    authorization: dict,
    reviews: dict,
) -> dict:
    """Admit only a complete result bound to successful actual resource cleanup.

    This is a mechanical result gate, not an independent scientific review.
    It reads sealed retained bytes and never invokes a production function.
    """
    from decimal import Decimal

    from .family_adapter import partition_summary
    from .family_driver import frozen_expected_keys

    verify_authorization(authorization, prepared["pins"], "comparison", reviews)
    attempt = strict_json(verify_artifact(attempt_pin))
    supervisor = strict_json(verify_artifact(supervisor_pin))
    for pin in (attempt_pin, supervisor_pin):
        path = Path(pin["path"])
        if (
            path.is_symlink()
            or path.stat().st_mode & 0o222
            or path.parent.stat().st_mode & 0o222
        ):
            raise GateError("family result evidence remains writable/aliased")
    family = prepared["policy"]["family"]
    resources = prepared["resources"]
    if (
        supervisor.get("schema") != "gen1-stochastic-family-supervisor-seal-v1"
        or supervisor.get("family") != family
        or supervisor.get("policy_sha256") != prepared["pins"]["policy_sha256"]
        or supervisor.get("outcome") != "CHILD_EXITED"
        or supervisor.get("exit_code") != 0
        or supervisor.get("cleanup_confirmed") is not True
        or supervisor.get("cleanup", {}).get("status") != "confirmed_stopped"
        or supervisor.get("frozen_resources") != resources
    ):
        raise GateError("terminal family resource outcome/cleanup/F mismatch")
    elapsed = Decimal(supervisor["elapsed_seconds"])
    peaks = supervisor["observed_process_peak_rss_bytes"]
    cap = resources["rss_bytes_per_process"]
    if (
        not elapsed.is_finite()
        or elapsed < 0
        or elapsed > resources["resources"]["timeout_seconds"]
        or not isinstance(peaks, dict)
        or not peaks
        or any(
            type(rss) is not int or rss < 0 or cap is not None and rss > cap
            for rss in peaks.values()
        )
    ):
        raise GateError("terminal family observed resources violate frozen caps")
    for pin in supervisor["artifacts"].values():
        verify_artifact(pin)
        path = Path(pin["path"])
        if (
            path.is_symlink()
            or path.parent.resolve() != Path(supervisor_pin["path"]).parent.resolve()
            or path.stat().st_mode & 0o222
        ):
            raise GateError("resource artifact outside readonly sealed supervisor")
    parent_pin = artifact_pin(
        supervisor["artifacts"]["launch.json"]["path"], "supervisor_launch"
    )
    resource_pin = artifact_pin(
        supervisor["artifacts"]["resource-launch.json"]["path"],
        "family_resource_launch",
    )
    parent = strict_json(verify_artifact(parent_pin))
    resource = strict_json(verify_artifact(resource_pin))
    if (
        resource.get("schema") != "gen1-stochastic-family-resource-launch-v1"
        or resource.get("parent_launch") != parent_pin
        or resource.get("policy_sha256") != prepared["pins"]["policy_sha256"]
        or resource.get("resources") != resources
        or resource.get("supervisor_pid") != parent.get("supervisor_pid")
        or parent.get("mode") != "FORMAL"
        or parent.get("policy_sha256") != prepared["pins"]["policy_sha256"]
        or parent.get("frozen_procedure")
        != prepared["policy"]["artifacts"]["procedure"]
        or parent.get("timeout_seconds") != resources["resources"]["timeout_seconds"]
    ):
        raise GateError("actual parent/resource launch binding differs")
    for pin in (resource["source"], parent["supervisor"], parent["cleanup_helper"]):
        verify_artifact(pin)
    if parent["supervisor"] != artifact_pin(
        Path(__file__).with_name("supervisor.py"), "supervisor_source"
    ) or parent["cleanup_helper"] != artifact_pin(
        Path(__file__).parents[2] / "run_stochastic_local.py", "admitted_cleanup_helper"
    ):
        raise GateError("parent supervisor/cleanup source identity differs")
    if resource["source"] != artifact_pin(__file__, "family_supervisor_source"):
        raise GateError("resource seal source differs from admitted reader")
    if family == "A2" and (
        not resource.get("aggregate_budget_reservation")
        or resource["aggregate_budget_reservation"]
        != supervisor.get("aggregate_budget_reservation")
    ):
        raise GateError("A2 terminal aggregate reservation mismatch")
    run_pins = {
        **prepared["pins"],
        "authorization_sha256": authorization["sha256"],
        "supervisor_launch_sha256": parent_pin["sha256"],
        "resource_launch_sha256": resource_pin["sha256"],
    }
    expected = exact_keys(frozen_expected_keys(prepared["fixtures"]))
    if (
        attempt.get("schema") != SEAL_SCHEMA
        or attempt.get("family") != family
        or attempt.get("phase") != "comparison"
        or attempt.get("decision") != "PASS"
        or attempt.get("exit_code") != 0
        or attempt.get("pins") != run_pins
        or attempt.get("completed") != len(expected)
        or exact_keys(attempt["expected_keys"]) != expected
    ):
        raise GateError("actual family attempt decision/coverage/supervision mismatch")
    for pin in attempt["artifacts"].values():
        verify_artifact(pin)
        path = Path(pin["path"])
        if (
            path.is_symlink()
            or path.parent.resolve() != Path(attempt_pin["path"]).parent.resolve()
            or path.stat().st_mode & 0o222
        ):
            raise GateError("attempt artifact outside readonly sealed result")
    launch = strict_json(verify_artifact(attempt["artifacts"]["launch"]))
    if launch["pins"] != run_pins or exact_keys(launch["expected_keys"]) != expected:
        raise GateError("actual family attempt launch differs")
    runtime = strict_json(verify_artifact(attempt["artifacts"]["runtime_dependencies"]))
    closure = prepared["closure"]
    modules = {p["name"]: p for p in closure["modules"]}
    if (
        runtime.get("status") != "MATCH"
        or runtime.get("planned_sha256")
        != prepared["pins"]["dependency_closure_sha256"]
        or runtime.get("python") != closure["python"]
        or not any(p["name"].startswith("gragra.") for p in runtime["observed_modules"])
        or any(modules.get(p["name"]) != p for p in runtime["observed_modules"])
        or any(e not in closure["import_edges"] for e in runtime["observed_edges"])
        or any(
            p not in closure["native_backends"]
            for p in runtime["observed_native_backends"]
        )
    ):
        raise GateError("actual family runtime does not match finite P closure")
    rows = [
        strict_json(line)
        for line in verify_artifact(attempt["artifacts"]["records"]).splitlines()
    ]
    if len(rows) != len(expected) or exact_keys([r["key"] for r in rows]) != expected:
        raise GateError("actual family records do not cover frozen keys")
    requirements = set(prepared["policy"]["connection_requirements"])
    for row in rows:
        if (
            row.get("status") != "resolved"
            or row.get("passed") is not True
            or row.get("assessment", {}).get("accepted") is not True
            or set(row["connection_assessment"]["gates"]) != requirements
            or any(
                v is not True for v in row["connection_assessment"]["gates"].values()
            )
        ):
            raise GateError(
                "actual family row is failed/unresolved or lacks frozen gates"
            )
    return {
        "decision": "PASS",
        "pins": run_pins,
        "attempt": attempt_pin,
        "resource_supervisor": supervisor_pin,
        "summary": partition_summary(rows),
    }


def process_group_rss(pgid: int, proc_root: Path = Path("/proc")) -> dict[int, int]:
    """Observe each member separately; do not reinterpret a per-process cap."""
    values = {}
    for path in proc_root.iterdir():
        if not path.name.isdigit():
            continue
        try:
            tail = (path / "stat").read_text().rsplit(")", 1)[1].split()
            if int(tail[2]) != pgid:
                continue
            rows = (path / "status").read_text().splitlines()
            rss = next((r.split()[1] for r in rows if r.startswith("VmRSS:")), None)
            if rss is not None:
                values[int(path.name)] = int(rss) * 1024
        except (FileNotFoundError, ProcessLookupError):
            continue
    return values


def verify_family_supervision(prepared: dict) -> dict:
    from .supervisor import verify_worker_supervision

    try:
        pin = strict_json(os.environ[TOKEN])
    except (KeyError, ValueError) as error:
        raise GateError(
            "formal family worker requires frozen resource supervisor"
        ) from error
    launch = strict_json(verify_artifact(pin))
    old_pin = verify_worker_supervision(prepared["pins"])
    if (
        launch.get("schema") != "gen1-stochastic-family-resource-launch-v1"
        or launch.get("policy_sha256") != prepared["pins"]["policy_sha256"]
        or launch.get("supervisor_pid") != os.getppid()
        or launch.get("resources") != prepared["resources"]
        or launch.get("parent_launch") != old_pin
    ):
        raise GateError("family resource supervisor actual parent/policy/F mismatch")
    source = launch["source"]
    verify_artifact(source)
    if source != artifact_pin(__file__, "family_supervisor_source"):
        raise GateError("family supervisor source mismatch")
    planned = {p["name"]: p for p in prepared["closure"]["modules"]}
    own = planned.get(__name__)
    if (
        own is None
        or own["path"] != source["path"]
        or own["sha256"] != source["sha256"]
    ):
        raise GateError("resource supervisor absent from reviewed dependency closure")
    if prepared["policy"]["family"] == "A2" and not launch.get(
        "aggregate_budget_reservation"
    ):
        raise GateError("A2 aggregate frozen budget reservation missing")
    return pin


def supervise_family(
    command: list[str],
    *,
    policy_pin: dict,
    evidence: Path,
    source_roots: list[Path],
    aggregate_budget=None,
) -> dict:
    """Use admitted group cleanup, fresh immutable evidence and actual frozen caps.

    A2 requires a reviewed controller ledger object with reserve/check/finish
    methods. Admission of that adapter and its source is separate from this
    generic driver; no default unlimited ledger is supplied.
    """
    from .family_driver import preflight
    from .supervisor import TOKEN as PARENT_TOKEN
    from .supervisor import _process_group_exists, _reap

    prepared = preflight(policy_pin)
    evidence = evidence.absolute()
    require_external_directory(evidence)
    roots = [p.resolve(strict=True) for p in source_roots]
    required = {
        Path(prepared["policy"][n]).resolve()
        for n in ("production_repo", "comparator_repo")
    }
    if (
        not required <= set(roots)
        or any(evidence.resolve().is_relative_to(r) for r in roots)
        or evidence.exists()
        or not command
    ):
        raise GateError(
            "fresh external evidence, actual source roots and command required"
        )
    family = prepared["policy"]["family"]
    if family == "A2" and aggregate_budget is None:
        raise GateError("A2 requires reviewed frozen aggregate budget controller")
    resources = prepared["resources"]
    timeout = resources["resources"]["timeout_seconds"]
    rss_cap = resources["rss_bytes_per_process"]
    reservation = None
    if aggregate_budget is not None:
        reservation = aggregate_budget.reserve(
            command=command, policy_sha256=policy_pin["sha256"], resources=resources
        )
        if not reservation:
            raise GateError("aggregate resource reservation refused")
    evidence.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    process = None
    reason = None
    cleanup = {"status": "not_started"}
    peaks = {}
    result = None
    try:
        helper = Path(__file__).parents[2] / "run_stochastic_local.py"
        parent = {
            "command": command,
            "timeout_seconds": timeout,
            "supervisor": artifact_pin(
                Path(__file__).with_name("supervisor.py"), "supervisor_source"
            ),
            "cleanup_helper": artifact_pin(helper, "admitted_cleanup_helper"),
            "supervisor_pid": os.getpid(),
            "mode": "FORMAL",
            "policy_sha256": policy_pin["sha256"],
            "frozen_procedure": prepared["policy"]["artifacts"]["procedure"],
        }
        _write_new(evidence / "launch.json", canonical_bytes(parent))
        parent_pin = artifact_pin(evidence / "launch.json", "supervisor_launch")
        launch = {
            "schema": "gen1-stochastic-family-resource-launch-v1",
            "policy_sha256": policy_pin["sha256"],
            "supervisor_pid": os.getpid(),
            "parent_launch": parent_pin,
            "resources": resources,
            "source": artifact_pin(__file__, "family_supervisor_source"),
            "aggregate_budget_reservation": reservation,
        }
        _write_new(evidence / "resource-launch.json", canonical_bytes(launch))
        env = dict(os.environ)
        env[PARENT_TOKEN] = json.dumps(parent_pin)
        env[TOKEN] = json.dumps(
            artifact_pin(evidence / "resource-launch.json", "family_resource_launch")
        )
        with (
            (evidence / "stdout.txt").open("xb") as stdout,
            (evidence / "stderr.txt").open("xb") as stderr,
        ):
            process = subprocess.Popen(
                command, stdout=stdout, stderr=stderr, start_new_session=True, env=env
            )
            while True:
                observed = process_group_rss(process.pid)
                for pid, rss in observed.items():
                    peaks[pid] = max(peaks.get(pid, 0), rss)
                if rss_cap is not None and any(
                    rss > rss_cap for rss in observed.values()
                ):
                    reason = "RSS_LIMIT"
                    break
                if aggregate_budget is not None:
                    aggregate_budget.check(
                        reservation, process.pid, time.monotonic() - started
                    )
                if process.poll() is not None:
                    break
                if time.monotonic() - started >= timeout:
                    reason = "TIMEOUT"
                    break
                time.sleep(0.02)
            if process.poll() is not None and _process_group_exists(process.pid):
                reason = "SURVIVING_DESCENDANTS"
    except BaseException as error:
        reason = type(error).__name__
        raise
    finally:
        if process is not None:
            cleanup = _reap(process, terminate=reason is not None)
        elapsed = time.monotonic() - started
        if aggregate_budget is not None:
            try:
                aggregate_budget.finish(
                    reservation,
                    elapsed_seconds=elapsed,
                    cleanup=cleanup,
                    outcome=reason or "CHILD_EXITED",
                )
            except BaseException as error:
                reason = "AGGREGATE_LEDGER_" + type(error).__name__
        for name in ("stdout.txt", "stderr.txt"):
            if not (evidence / name).exists():
                _write_new(evidence / name, b"")
        result = {
            "schema": "gen1-stochastic-family-supervisor-seal-v1",
            "family": family,
            "policy_sha256": policy_pin["sha256"],
            "outcome": reason or "CHILD_EXITED",
            "exit_code": process.returncode if process else None,
            "cleanup_confirmed": process is None
            or cleanup.get("status") == "confirmed_stopped",
            "cleanup": cleanup,
            "elapsed_seconds": str(elapsed),
            "frozen_resources": resources,
            "observed_process_peak_rss_bytes": {str(p): r for p, r in peaks.items()},
            "aggregate_budget_reservation": reservation,
            "artifacts": {
                p.name: artifact_pin(p, "supervisor_evidence")
                for p in evidence.iterdir()
                if p.is_file()
            },
        }
        _write_new(evidence / "seal.json", canonical_bytes(result))
        for path in evidence.iterdir():
            if path.is_file():
                path.chmod(0o444)
        evidence.chmod(0o555)
    if reason or result["exit_code"] != 0 or not result["cleanup_confirmed"]:
        raise GateError(
            "family worker failed; immutable resource/cleanup evidence retained"
        )
    return result
