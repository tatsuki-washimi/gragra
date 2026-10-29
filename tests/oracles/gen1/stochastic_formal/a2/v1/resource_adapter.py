"""A2 managed-child resource profile; no production or reference imports.

The admitted E60 ledger owns every child process. A resource launch names a
unique immutable token before the ledger index is allocated. Workers verify
the actual live PID and exact command; no ledger index is predicted.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from types import ModuleType

from ...common.transport import (
    GateError,
    artifact_pin,
    canonical_bytes,
    strict_json,
    verify_artifact,
)

PROFILE = "a2-ledger-v1"
RESOURCE_TOKEN = "GRAGRA_STOCHASTIC_FAMILY_RESOURCE_LAUNCH"
CONTEXT_TOKEN = "GRAGRA_STOCHASTIC_A2_LEDGER_CONTEXT"
BUDGET_SHA256 = "a61ed66ad19291fe1a00fa87c5b266c7633e4e17d52802cb24b41709690d90a8"
CAMPAIGN_LEDGER = Path(
    "/home/washimi/.local/state/gragra/gen1-production/20261004-pr159-final/a2/resource-ledger.json"
)
ENVIRONMENT_NAMES = frozenset(
    {
        "PATH",
        "HOME",
        "TMPDIR",
        "PYTHONPATH",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONHASHSEED",
        "LD_LIBRARY_PATH",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "GRAGRA_REQUIRE_CPP",
        "GRAGRA_REQUIRE_NUMBA",
    }
)


def pin_environment(pin: dict) -> str:
    return canonical_bytes(pin).decode().strip()


def environment_command(command: list[str], environment: dict[str, str]) -> list[str]:
    """Use env's exec behavior: the managed PID becomes the actual worker PID."""
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(v, str) or not v or "\0" in v for v in command)
        or not isinstance(environment, dict)
    ):
        raise GateError("exact nonempty managed command required")
    for name, value in environment.items():
        if (
            not isinstance(name, str)
            or name not in ENVIRONMENT_NAMES
            and not name.startswith("GRAGRA_STOCHASTIC_")
            or not isinstance(value, str)
            or "\0" in value
            or "=" in name
            or "\0" in name
        ):
            raise GateError("unapproved managed environment variable")
    return [
        "/usr/bin/env",
        "-i",
        "--",
        *[name + "=" + environment[name] for name in sorted(environment)],
        *command,
    ]


def match_managed_entry(
    entries: list[dict],
    *,
    pid: int,
    parent_pid: int,
    argv: list[str],
    resource_pin: dict,
) -> dict:
    """Require one live ledger reservation for this actual worker and launch."""
    token = RESOURCE_TOKEN + "=" + pin_environment(resource_pin)
    matches = []
    for entry in entries:
        command = entry.get("command")
        if (
            entry.get("pid") == pid
            and entry.get("parent_pid") == parent_pid
            and entry.get("status") == "running"
            and entry.get("kind") == "repair_validation"
            and "elapsed_seconds" not in entry
            and isinstance(command, list)
            and command[:3] == ["/usr/bin/env", "-i", "--"]
            and len(command) >= len(argv) + 4
            and command[-len(argv) :] == argv
            and token in command[3 : -len(argv)]
        ):
            matches.append(entry)
    if len(matches) != 1:
        raise GateError("actual worker lacks a unique live exact managed ledger entry")
    return matches[0]


def context() -> dict:
    """Verify operational sources before the first controller/worker Git child."""
    try:
        value = strict_json(os.environ[CONTEXT_TOKEN].encode())
    except (KeyError, ValueError) as error:
        raise GateError("A2 requires the actual admitted ledger context") from error
    if (
        set(value)
        != {
            "profile",
            "ledger_path",
            "adapter_source",
            "budget_source",
            "controller_id",
            "launch_key",
        }
        or value["profile"] != PROFILE
        or Path(value["ledger_path"]).resolve() != CAMPAIGN_LEDGER.resolve()
        or type(value["controller_id"]) is not int
        or value["controller_id"] < 0
        or not isinstance(value["launch_key"], str)
        or not re.fullmatch(r"[0-9a-f]{32}", value["launch_key"])
    ):
        raise GateError("A2 ledger profile/token/path fields mismatch")
    verify_artifact(value["adapter_source"])
    verify_artifact(value["budget_source"])
    if (
        value["adapter_source"] != artifact_pin(__file__, "a2_managed_child_source")
        or value["budget_source"]
        != artifact_pin(
            Path(__file__).parents[1] / "budget.py", "a2_normative_budget_source"
        )
        or value["budget_source"]["sha256"] != BUDGET_SHA256
    ):
        raise GateError("A2 actual adapter/E60 ledger bytes differ")
    if any(os.environ.get(n) for n in ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR")):
        raise GateError("ambient Git repository override forbidden for A2")
    return value


def budget_api():
    value = context()
    module = _budget_module()
    if (
        Path(module.__file__).resolve()
        != Path(value["budget_source"]["path"]).resolve()
    ):
        raise GateError("actual imported normative budget origin differs")
    require_live_context(value)
    return module


def require_live_context(value):
    """Bind controller or managed worker before any validation child launches."""
    deadline = time.monotonic() + 1
    while True:
        state = operational_json(Path(value["ledger_path"]).read_bytes())
        matches = [
            r
            for r in state.get("controllers", [])
            if r.get("id") == value["controller_id"]
        ]
        if len(matches) != 1 or matches[0].get("status") != "running":
            raise GateError("A2 context lacks an actual running controller account")
        controller = matches[0]
        if controller.get("pid") == os.getpid():
            if controller.get("command") != [sys.executable, *sys.argv]:
                raise GateError("A2 controller actual argv differs from account")
            return controller
        if controller.get("pid") != os.getppid():
            raise GateError("A2 worker actual parent differs from controller account")
        try:
            resource_pin = strict_json(os.environ[RESOURCE_TOKEN].encode())
            launch = strict_json(verify_artifact(resource_pin))
        except (KeyError, ValueError) as error:
            raise GateError(
                "A2 worker lacks an actual pinned resource launch"
            ) from error
        if (
            launch.get("managed_profile") != PROFILE
            or launch.get("aggregate_budget_reservation")
            != {"ledger_path": value["ledger_path"], "launch_key": value["launch_key"]}
            or launch.get("managed_child_adapter") != value["adapter_source"]
            or launch.get("normative_budget_source") != value["budget_source"]
        ):
            raise GateError("A2 worker launch token/source profile differs")
        try:
            match_managed_entry(
                state["entries"],
                pid=os.getpid(),
                parent_pid=os.getppid(),
                argv=actual_process_command(),
                resource_pin=resource_pin,
            )
            return controller
        except GateError:
            # Popen can start the child just before the parent durably records its
            # returned PID. Only this unique, still-live exact launch may wait.
            argv = actual_process_command()
            token = RESOURCE_TOKEN + "=" + pin_environment(resource_pin)
            initializing = [
                e
                for e in state["entries"]
                if e.get("pid") is None
                and e.get("parent_pid") == os.getppid()
                and e.get("status") == "running"
                and e.get("kind") == "repair_validation"
                and e.get("command", [])[-len(argv) :] == argv
                and token in e.get("command", [])
            ]
            if len(initializing) != 1 or time.monotonic() >= deadline:
                raise
            time.sleep(0.01)


def actual_process_command():
    """Retain interpreter flags; sys.argv omits options such as Python -I."""
    raw = Path("/proc/self/cmdline").read_bytes()
    if not raw.endswith(b"\0"):
        raise GateError("actual managed process argv is unavailable")
    argv = [part.decode() for part in raw[:-1].split(b"\0")]
    if not argv or not argv[0]:
        raise GateError("actual managed process argv is empty")
    return argv


def claim_controller(value):
    """The controller is direct root orchestration, never an outer child slot."""
    module = _budget_module()
    if (
        Path(module.__file__).resolve()
        != Path(value["budget_source"]["path"]).resolve()
    ):
        raise GateError("actual normative budget import origin differs")
    account = module.ControllerAccount(module.Ledger(value["ledger_path"]))
    deadline = account.claim(
        value["controller_id"],
        [sys.executable, *sys.argv],
        require_explicit_origin=True,
    )
    return module, account, deadline


def _budget_module():
    """Load frozen E60 bytes under the versioned A2 package identity."""
    name = __package__ + ".budget"
    source = Path(__file__).parents[1] / "budget.py"
    module = sys.modules.get(name)
    if module is not None:
        if Path(module.__file__).resolve() != source.resolve():
            raise GateError("versioned A2 budget module origin differs")
        return module
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise GateError("versioned A2 budget source cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def own_entries(state, controller_pid):
    """Finite actual-parent closure; unrelated campaign entries are excluded."""
    owners = {controller_pid}
    selected = []
    pending = list(state["entries"])
    while True:
        found = [r for r in pending if r.get("parent_pid") in owners]
        if not found:
            return selected
        for row in found:
            selected.append(row)
            pending.remove(row)
            if type(row.get("pid")) is int:
                owners.add(row["pid"])


def supervise_a2(command, *, policy_pin, evidence, source_roots, aggregate_budget=None):
    """Use the pinned E60 Ledger.run as the sole managed child process owner."""
    from ...common.supervisor import TOKEN as PARENT_TOKEN
    from ...common.transport import _write_new
    from .family_driver import preflight
    from .family_supervisor import process_group_rss

    if aggregate_budget is not None:
        raise GateError("A2 legacy reserve/check/finish objects are not admitted")
    value = context()
    module, account, deadline = claim_controller(value)
    ledger = module.Ledger(value["ledger_path"])
    evidence = Path(evidence).absolute()
    allocated = False
    terminal = None
    worker_result = None
    argv = None
    peaks = {}
    failure = None
    stop = threading.Event()
    monitor = None
    monitor_failure = []
    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.getitimer(signal.ITIMER_REAL)

    def interrupted(_signum, _frame):
        raise TimeoutError("A2 managed controller hard deadline")

    def watch():
        try:
            while not stop.wait(0.01):
                state = operational_json(Path(value["ledger_path"]).read_bytes())
                for row in own_entries(state, os.getpid()):
                    if "elapsed_seconds" in row:
                        continue
                    pid = row.get("pid")
                    if type(pid) is not int:
                        continue
                    observed = process_group_rss(pid)
                    for observed_pid, rss in observed.items():
                        peaks[observed_pid] = max(peaks.get(observed_pid, 0), rss)
                        if rss > 4294967296:
                            monitor_failure.append("RSS_LIMIT")
                            if module.live_group(pid):
                                os.killpg(pid, signal.SIGKILL)
        except BaseException as error:
            monitor_failure.append(type(error).__name__)

    try:
        signal.signal(signal.SIGALRM, interrupted)
        signal.setitimer(signal.ITIMER_REAL, max(0.001, deadline - time.monotonic()))
        prepared = preflight(policy_pin)
        require_external_directory(evidence)
        roots = [Path(p).resolve(strict=True) for p in source_roots]
        required = {
            Path(prepared["policy"][n]).resolve()
            for n in ("production_repo", "comparator_repo")
        }
        if (
            prepared["policy"]["family"] != "A2"
            or not required <= set(roots)
            or any(evidence.resolve().is_relative_to(r) for r in roots)
            or evidence.exists()
            or not command
        ):
            raise GateError("A2 fresh external evidence/source roots/command required")
        resources = prepared["resources"]
        if (
            resources["rss_bytes_per_process"] != 4294967296
            or resources["resources"]["timeout_seconds"] != 3600
        ):
            raise GateError("A2 actual frozen process cap/deadline differs")
        # Mask only the allocation-to-local-state transition. The encompassing
        # recovery guard already owns the leaf when the pending signal arrives.
        blocked = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
        try:
            evidence.mkdir(parents=True, exist_ok=False)
            allocated = True
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, blocked)
        parent = {
            "command": command,
            "timeout_seconds": 3600,
            "supervisor": artifact_pin(
                Path(__file__).parents[2] / "common/supervisor.py", "supervisor_source"
            ),
            "cleanup_helper": artifact_pin(
                Path(__file__).parents[3] / "run_stochastic_local.py",
                "admitted_cleanup_helper",
            ),
            "supervisor_pid": os.getpid(),
            "mode": "FORMAL",
            "policy_sha256": policy_pin["sha256"],
            "frozen_procedure": prepared["policy"]["artifacts"]["procedure"],
        }
        _write_new(evidence / "launch.json", canonical_bytes(parent))
        parent_pin = artifact_pin(evidence / "launch.json", "supervisor_launch")
        reservation = {
            "ledger_path": value["ledger_path"],
            "launch_key": value["launch_key"],
        }
        launch = {
            "schema": "gen1-stochastic-family-resource-launch-v1",
            "managed_profile": PROFILE,
            "policy_sha256": policy_pin["sha256"],
            "supervisor_pid": os.getpid(),
            "parent_launch": parent_pin,
            "resources": resources,
            "source": artifact_pin(
                Path(__file__).with_name("family_supervisor.py"),
                "family_supervisor_source",
            ),
            "aggregate_budget_reservation": reservation,
            "managed_child_adapter": value["adapter_source"],
            "normative_budget_source": value["budget_source"],
            "managed_environment_executable": artifact_pin(
                "/usr/bin/env", "managed_environment_executable"
            ),
            "controller_id": value["controller_id"],
            "purpose": "production-comparison",
        }
        _write_new(evidence / "resource-launch.json", canonical_bytes(launch))
        resource_pin = artifact_pin(
            evidence / "resource-launch.json", "family_resource_launch"
        )
        environment = {n: os.environ[n] for n in ENVIRONMENT_NAMES if n in os.environ}
        environment.update(
            {
                CONTEXT_TOKEN: pin_environment(value),
                PARENT_TOKEN: pin_environment(parent_pin),
                RESOURCE_TOKEN: pin_environment(resource_pin),
            }
        )
        argv = environment_command(command, environment)
        monitor = threading.Thread(target=watch, name="a2-managed-rss", daemon=False)
        monitor.start()
        worker_result = ledger.run(
            argv,
            kind="repair_validation",
            stdout=evidence / "stdout.txt",
            stderr=evidence / "stderr.txt",
            timeout=min(3600, max(0.001, deadline - time.monotonic())),
        )
        if (
            worker_result["status"] != "success"
            or worker_result["exit_code"] != 0
            or not worker_result["cleanup_confirmed"]
        ):
            failure = "MANAGED_CHILD_" + worker_result["status"]
    except BaseException as error:
        failure = type(error).__name__
    finally:
        # The 3700s already-reserved whole-controller bound covers this finite
        # recovery and final IO. It is never released based on an internal clock.
        signal.setitimer(signal.ITIMER_REAL, 0)
        stop.set()
        if monitor is not None:
            monitor.join(timeout=1)
            if monitor.is_alive():
                failure = "RSS_MONITOR_PENDING_RECOVERY"
        if monitor_failure:
            failure = "RSS_MONITOR_" + monitor_failure[0]
        state = operational_json(Path(value["ledger_path"]).read_bytes())
        owned = own_entries(state, os.getpid())
        unresolved = [
            r
            for r in owned
            if "elapsed_seconds" not in r or r.get("cleanup_confirmed") is not True
        ]
        if unresolved:
            failure = "MANAGED_CHILD_PENDING_RECOVERY"
        if allocated:
            for name in ("stdout.txt", "stderr.txt"):
                if not (evidence / name).exists():
                    _write_new(evidence / name, b"")
            _write_new(
                evidence / "managed-ledger-snapshot.json",
                (
                    json.dumps(
                        state, sort_keys=True, separators=(",", ":"), allow_nan=False
                    )
                    + "\n"
                ).encode(),
            )
            rows = [
                r
                for r in owned
                if (worker_result and r["id"] == worker_result["ledger_id"])
                or (
                    worker_result is None
                    and argv is not None
                    and r.get("command") == argv
                    and r.get("kind") == "repair_validation"
                )
            ]
            row = rows[0] if len(rows) == 1 else None
            if row is None or not peaks:
                failure = failure or "MANAGED_IDENTITY_OR_RSS_UNRESOLVED"
            elapsed = (
                str(row["elapsed_seconds"]) if row and "elapsed_seconds" in row else "0"
            )
            terminal = {
                "schema": "gen1-stochastic-family-supervisor-seal-v1",
                "managed_profile": PROFILE,
                "family": "A2",
                "policy_sha256": policy_pin["sha256"],
                "outcome": failure or "CHILD_EXITED",
                "exit_code": row.get("exit_code") if row else None,
                "cleanup_confirmed": row is not None
                and row.get("cleanup_confirmed") is True
                and not unresolved,
                "cleanup": {
                    "status": "confirmed_stopped"
                    if row and row.get("cleanup_confirmed") and not unresolved
                    else "pending_recovery",
                    "pid": row.get("pid") if row else None,
                    "leader_reaped": bool(row and row.get("cleanup_confirmed")),
                    "process_group_empty": bool(row and row.get("cleanup_confirmed")),
                },
                "elapsed_seconds": elapsed,
                "frozen_resources": prepared["resources"],
                "observed_process_peak_rss_bytes": {
                    str(p): r for p, r in peaks.items()
                },
                "aggregate_budget_reservation": {
                    "ledger_path": value["ledger_path"],
                    "launch_key": value["launch_key"],
                },
                "managed_ledger_id": row["id"] if row else None,
                "controller_id": value["controller_id"],
                "artifacts": {
                    p.name: artifact_pin(p, "supervisor_evidence")
                    for p in evidence.iterdir()
                    if p.is_file()
                },
            }
            _write_new(evidence / "seal.json", canonical_bytes(terminal))
            for p in evidence.iterdir():
                if p.is_file():
                    p.chmod(0o444)
            evidence.chmod(0o555)
        account.outcome(
            value["controller_id"],
            "pending_recovery" if unresolved else ("failure" if failure else "success"),
        )
        signal.signal(signal.SIGALRM, old_handler)
        signal.setitimer(signal.ITIMER_REAL, *old_timer)
    if failure or terminal is None:
        raise GateError(
            "A2 managed execution refused/failed; actual accounting and "
            "allocated evidence retained: " + str(failure)
        )
    return terminal


def verify_worker_profile(prepared, resource_pin, launch):
    """Actual managed PID and finite operational source closure are mandatory."""
    value = context()
    require_live_context(value)
    expected_fields = {
        "schema",
        "managed_profile",
        "policy_sha256",
        "supervisor_pid",
        "parent_launch",
        "resources",
        "source",
        "aggregate_budget_reservation",
        "managed_child_adapter",
        "normative_budget_source",
        "managed_environment_executable",
        "controller_id",
        "purpose",
    }
    if (
        set(launch) != expected_fields
        or launch.get("managed_profile") != PROFILE
        or launch.get("purpose") != "production-comparison"
        or launch.get("controller_id") != value["controller_id"]
        or launch.get("aggregate_budget_reservation")
        != {"ledger_path": value["ledger_path"], "launch_key": value["launch_key"]}
        or launch.get("managed_child_adapter") != value["adapter_source"]
        or launch.get("normative_budget_source") != value["budget_source"]
        or launch.get("managed_environment_executable")
        != artifact_pin("/usr/bin/env", "managed_environment_executable")
    ):
        raise GateError("A2 worker genuine managed resource profile differs")
    modules = {p["name"]: p for p in prepared["closure"]["modules"]}
    for name, pin in [
        (__name__, value["adapter_source"]),
        (__package__ + ".budget", value["budget_source"]),
    ]:
        verify_artifact(pin)
        actual = modules.get(name)
        if (
            actual is None
            or actual["path"] != pin["path"]
            or actual["sha256"] != pin["sha256"]
        ):
            raise GateError(
                "A2 managed worker operational source absent from finite closure"
            )
    if launch["managed_environment_executable"] not in prepared["closure"]["artifacts"]:
        raise GateError(
            "A2 exact env executable absent from finite operational closure"
        )
    state = operational_json(Path(value["ledger_path"]).read_bytes())
    return match_managed_entry(
        state["entries"],
        pid=os.getpid(),
        parent_pid=os.getppid(),
        argv=actual_process_command(),
        resource_pin=resource_pin,
    )


def verify_result_profile(
    prepared,
    *,
    resource_pin,
    resource,
    supervisor_pin,
    supervisor,
    attempt_pin,
    completion_pin,
):
    """Admit no A2 result until actual root post-wait account completion is bound."""
    if completion_pin is None:
        raise GateError(
            "A2 result requires actual managed controller completion evidence"
        )
    declared_sources = {
        item["path"]: item["sha256"]
        for item in prepared["closure"]["modules"]
        if item.get("path") is not None
    }
    for name in ("managed_child_adapter", "normative_budget_source"):
        pin = resource[name]
        if declared_sources.get(pin["path"]) != pin["sha256"]:
            raise GateError("A2 result operational source absent from finite closure")
    proof = strict_json(verify_artifact(completion_pin))
    path = Path(completion_pin["path"])
    if (
        path.is_symlink()
        or path.stat().st_mode & 0o222
        or path.parent.stat().st_mode & 0o222
    ):
        raise GateError("A2 controller completion evidence remains writable/aliased")
    fields = {
        "schema",
        "issuer",
        "decision",
        "managed_profile",
        "aggregate_budget_reservation",
        "controller_id",
        "resource_launch",
        "supervisor_seal",
        "attempt_seal",
        "postwait_ledger_snapshot",
        "root_external_span",
    }
    if (
        set(proof) != fields
        or proof["schema"] != "gen1-stochastic-a2-managed-controller-completion-v1"
        or proof["issuer"] != "root"
        or proof["decision"] != "CONTROLLER_AND_MANAGED_CHILD_COMPLETE"
        or proof["managed_profile"] != PROFILE
        or resource.get("managed_profile") != PROFILE
        or supervisor.get("managed_profile") != PROFILE
        or proof["aggregate_budget_reservation"]
        != resource["aggregate_budget_reservation"]
        or supervisor["aggregate_budget_reservation"]
        != resource["aggregate_budget_reservation"]
        or proof["controller_id"] != resource["controller_id"]
        or proof["controller_id"] != supervisor["controller_id"]
        or proof["resource_launch"] != resource_pin
        or proof["supervisor_seal"] != supervisor_pin
        or proof["attempt_seal"] != attempt_pin
        or resource["aggregate_budget_reservation"]["ledger_path"]
        != str(CAMPAIGN_LEDGER)
        or resource["normative_budget_source"]["sha256"] != BUDGET_SHA256
        or resource["managed_child_adapter"]
        != artifact_pin(__file__, "a2_managed_child_source")
    ):
        raise GateError("A2 completion/actual resource/attempt/source identity differs")
    verify_artifact(resource["normative_budget_source"])
    verify_artifact(resource["managed_environment_executable"])
    terminal = operational_json(
        verify_artifact(supervisor["artifacts"]["managed-ledger-snapshot.json"])
    )
    postwait = operational_json(verify_artifact(proof["postwait_ledger_snapshot"]))
    span = operational_json(verify_artifact(proof["root_external_span"]))
    rows = [
        r for r in terminal["entries"] if r.get("id") == supervisor["managed_ledger_id"]
    ]
    later = [
        r for r in postwait["entries"] if r.get("id") == supervisor["managed_ledger_id"]
    ]
    accounts = [
        r
        for r in postwait.get("controllers", [])
        if r.get("id") == proof["controller_id"]
    ]
    if len(rows) != 1 or later != rows or len(accounts) != 1:
        raise GateError(
            "A2 actual worker/controller terminal identities missing/changed"
        )
    row, controller = rows[0], accounts[0]
    if (
        row.get("status") != "success"
        or row.get("exit_code") != 0
        or row.get("cleanup_confirmed") is not True
        or row.get("kind") != "repair_validation"
        or row.get("parent_pid") != resource["supervisor_pid"]
        or row.get("pid") != supervisor["cleanup"]["pid"]
        or str(row.get("elapsed_seconds")) != supervisor["elapsed_seconds"]
        or controller.get("status") != "completed"
        or controller.get("outcome") != "success"
        or controller.get("pid") != resource["supervisor_pid"]
        or controller.get("external_span_pin")
        != {
            "path": proof["root_external_span"]["path"],
            "sha256": proof["root_external_span"]["sha256"],
        }
        or span.get("issuer") != "root"
        or span.get("decision") != "CONTROLLER_EXTERNAL_SPAN_ATTESTED"
        or span.get("controller_id") != proof["controller_id"]
        or span.get("controller_argv") != controller["command"]
        or span.get("started_monotonic") != controller["started_monotonic"]
        or span.get("elapsed_seconds") != controller["external_span_upper_seconds"]
        or not 0 <= span["elapsed_seconds"] <= controller["wall_bound_seconds"]
    ):
        raise GateError("A2 actual post-wait completion/cleanup/duration fails closed")
    token = RESOURCE_TOKEN + "=" + pin_environment(resource_pin)
    if (
        row["command"][:3] != ["/usr/bin/env", "-i", "--"]
        or token not in row["command"]
    ):
        raise GateError("A2 terminal exact managed command is unbound")
    owned = own_entries(postwait, resource["supervisor_pid"])
    if any(
        "elapsed_seconds" not in r or r.get("cleanup_confirmed") is not True
        for r in owned
    ):
        raise GateError("A2 controller descendants remain live/unaccounted")
    return proof


def derive_native_reference_pins(artifacts, _documents, fixtures, seal):
    """Validate native A2 evidence; never invent common-seal fields."""
    from ...common.transport import exact_keys
    from .family_driver import frozen_expected_keys

    root = Path(artifacts["seal"]["path"]).parent
    if root.is_symlink() or root.stat().st_mode & 0o222:
        raise GateError("native A2 reference root remains writable/aliased")
    hashes = seal.get("artifacts_sha256", {})
    for name, sha in hashes.items():
        path = root / name
        if (
            not path.is_file()
            or path.is_symlink()
            or path.parent.resolve() != root.resolve()
            or path.stat().st_mode & 0o222
            or hashlib.sha256(path.read_bytes()).hexdigest() != sha
        ):
            raise GateError("native A2 sealed bytes differ")
    required = {
        "authorization.json",
        "family-freeze.json",
        "policy.json",
        "records.raw.jsonl",
        "resource-ledger.json",
        "resource-result.json",
        "purpose-records.json",
        "stderr.jsonl",
        "stdout.jsonl",
    }
    if set(hashes) != required:
        raise GateError("native A2 sealed artifact closure differs")
    auth = strict_json((root / "authorization.json").read_bytes())
    execution, coverage = seal.get("execution", {}), seal.get("execution_coverage", {})
    provenance = seal.get("provenance", {})
    expected = exact_keys(frozen_expected_keys(fixtures))
    if (
        seal.get("format") != "gen1-stochastic-attempt-seal-v2"
        or execution.get("status") != "success"
        or execution.get("exit_code") != 0
        or execution.get("cleanup", {}).get("status") != "confirmed_stopped"
        or coverage.get("status") != "accepted"
        or coverage.get("record_count") != 84
        or exact_keys(coverage.get("expected_keys", [])) != expected
        or any(
            coverage.get(n)
            for n in (
                "malformed_lines",
                "partial_final_line",
                "reasons",
                "duplicate_keys",
                "duplicate_expected_keys",
                "unknown_keys",
                "unresolved_keys",
                "not_generated_keys",
            )
        )
        or provenance.get("family") != "A2"
        or provenance.get("production_imported") is not False
        or provenance.get("purpose") != "formal-science"
        or auth.get("family") != "A2"
        or auth.get("authorized_by") != "root"
        or auth.get("formal_execution") != "AUTHORIZED"
        or auth.get("E_commit") != provenance.get("E_commit")
        or auth.get("F_sha256") != provenance.get("F_sha256")
        or auth.get("F_sha256") != artifacts["freeze"]["sha256"]
        or auth.get("fixture_sha256") != provenance.get("fixture_sha256")
        or auth.get("fixture_sha256") != artifacts["fixture"]["sha256"]
        or auth.get("procedure_sha256") != artifacts["family_procedure"]["sha256"]
    ):
        raise GateError("native A2 generation identity/execution/coverage differs")
    raw = verify_artifact(artifacts["records"])
    if (
        Path(artifacts["records"]["path"]).parent != root
        or not raw.endswith(b"\n")
        or hashes["records.raw.jsonl"] != artifacts["records"]["sha256"]
        or hashes["stdout.jsonl"] != artifacts["records"]["sha256"]
        or hashes["family-freeze.json"] != artifacts["freeze"]["sha256"]
        or hashes["policy.json"] != artifacts["fixture"]["sha256"]
        or auth["reference_runtime_inventory_pin"]["path"]
        != artifacts["environment"]["path"]
        or auth["reference_runtime_inventory_pin"]["sha256"]
        != artifacts["environment"]["sha256"]
    ):
        raise GateError("native A2 records/F/actual cold environment transport differs")
    native_bytes(auth["reference_runtime_inventory_pin"])
    if Path(artifacts["environment"]["path"]).stat().st_mode & 0o222:
        raise GateError("native A2 environment inventory remains writable")
    result = strict_json((root / "resource-result.json").read_bytes())
    ledger = operational_json((root / "resource-ledger.json").read_bytes())
    entries = [e for e in ledger["entries"] if e.get("id") == result.get("ledger_id")]
    if len(entries) != 1:
        raise GateError("native A2 terminal worker ledger identity missing/duplicated")
    entry = entries[0]
    if (
        result.get("status") != "success"
        or result.get("exit_code") != 0
        or result.get("cleanup_confirmed") is not True
        or result.get("launch_state") != "started"
        or entry.get("kind") != "formal"
        or entry.get("status") != "success"
        or entry.get("exit_code") != 0
        or entry.get("cleanup_confirmed") is not True
        or type(entry.get("pid")) is not int
        or entry["pid"] <= 0
        or not 0 <= entry.get("elapsed_seconds", -1) <= 3600
        or entry["command"][-2] != "--authorization"
        or hashlib.sha256(Path(entry["command"][-1]).read_bytes()).hexdigest()
        != hashes["authorization.json"]
    ):
        raise GateError(
            "native A2 actual terminal worker duration/cleanup/argv differs"
        )
    admission = strict_json(native_bytes(auth["root_source_admission"]))
    durable = strict_json(native_bytes(auth["durable_E"]))
    integration = strict_json(native_bytes(auth["root_E_integration_admission"]))
    if (
        admission.get("decision") != "PASS_FOR_REVIEWED_REFERENCE_SOURCE_INTEGRATION"
        or integration.get("decision") != "PASS_REVIEWED_E_INTEGRATED"
        or durable.get("E_commit") != auth["E_commit"]
        or durable.get("E_tree") != auth["E_tree"]
        or integration.get("E_commit") != auth["E_commit"]
        or integration.get("E_tree") != auth["E_tree"]
        or durable.get("restored_tree") != auth["E_tree"]
        or durable.get("clean_E") is not True
        or durable.get("reviewed_source_hashes_by_group")
        != auth["reviewed_source_hashes_by_group"]
    ):
        raise GateError("native A2 actual source/durable E admission differs")
    native_bytes(durable["bundle"])
    reviewers = []
    for role in ("science", "numerical", "provenance"):
        review = strict_json(native_bytes(auth["reviews"][role]))
        if (
            review.get("role") != role
            or review.get("decision") != "PASS"
            or review.get("scope") != "reference implementation"
            or review.get("families") != ["A2"]
            or review.get("reviewed_commit") != admission["reviewed_commit"]
            or review.get("file_hashes_by_group")
            != auth["reviewed_source_hashes_by_group"]
        ):
            raise GateError("native A2 actual implementation review identity differs")
        for original in review["original_reviews"]:
            native_bytes(original)
        reviewers.append(review["reviewer_id"])
    if len(set(reviewers)) != 3:
        raise GateError("native A2 source reviewers are not distinct")
    return {
        "family": "A2",
        "E_commit": auth["E_commit"],
        "E_tree": auth["E_tree"],
        "records_sha256": artifacts["records"]["sha256"],
        "seal_sha256": artifacts["seal"]["sha256"],
        "environment_sha256": artifacts["environment"]["sha256"],
        "expected_keys_sha256": hashlib.sha256(
            canonical_bytes(frozen_expected_keys(fixtures))
        ).hexdigest(),
        "frozen_artifacts_sha256": {
            n: artifacts[n]["sha256"]
            for n in ("freeze", "fixture", "budget", "family_procedure")
        },
        "source_pin": {
            "head": auth["E_commit"],
            "tree": auth["E_tree"],
            "files_sha256": auth["file_hashes_by_group"],
        },
        "artifact_hashes": hashes,
    }


def native_bytes(pin):
    """Read the issued native pin vocabulary without changing retained bytes."""
    if (
        not isinstance(pin, dict)
        or set(pin)
        not in (
            {"path", "sha256"},
            {"path", "sha256", "kind"},
            {"path", "sha256", "decision"},
        )
        or "decision" in pin
        and pin["decision"] != "PASS"
    ):
        raise GateError("native A2 provenance pin shape differs")
    return verify_artifact(
        {
            "path": pin["path"],
            "sha256": pin["sha256"],
            "kind": pin.get("kind", "native_reference_provenance"),
        }
    )


def operational_json(raw):
    """Operational wall clocks are finite numbers, distinct from scientific JSON."""

    def pairs(items):
        result = {}
        for name, value in items:
            if name in result:
                raise GateError("duplicate operational metadata field")
            result[name] = value
        return result

    def nonfinite(_value):
        raise GateError("nonfinite operational metadata")

    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise GateError("nonfinite operational metadata")
        return result

    return json.loads(
        raw, object_pairs_hook=pairs, parse_constant=nonfinite, parse_float=finite_float
    )


_GIT_CACHE = OrderedDict()
_GIT_CACHE_BYTES = 0
_GIT_CACHE_MAX_ENTRIES = 1024
_GIT_CACHE_MAX_BYTES = 8 * 1024**2
_GIT_CACHE_LOCK = threading.RLock()


def _immutable_git_key(repo, arguments, text):
    """Only raw object queries under freshly checked lookup semantics qualify."""
    if len(arguments) != 2 or arguments[0] not in {"rev-parse", "show"}:
        return None
    expression = arguments[1]
    tree = re.fullmatch(r"[0-9a-f]{40}\^\{tree\}", expression)
    blob = re.fullmatch(r"[0-9a-f]{40}:(.+)", expression)
    if not (tree and arguments[0] == "rev-parse") and not blob:
        return None
    if blob and (
        blob[1].startswith("/")
        or any(p in {"", ".", ".."} for p in blob[1].split("/"))
        or any(c in expression for c in "\x00\n\r")
    ):
        return None
    # GIT_PAGER cannot alter these captured raw object reads. All lookup/config
    # overrides are rejected; even allowed environment values enter the key.
    if any(n.startswith("GIT_") and n != "GIT_PAGER" for n in os.environ):
        return None
    try:
        root = Path(repo).resolve(strict=True)
        entry = root / ".git"
        if entry.is_dir():
            gitdir = entry.resolve(strict=True)
        else:
            raw = entry.read_text().strip()
            if not raw.startswith("gitdir: "):
                return None
            gitdir = (root / raw[8:]).resolve(strict=True)
        commonfile = gitdir / "commondir"
        common = (
            (gitdir / commonfile.read_text().strip()).resolve(strict=True)
            if commonfile.exists()
            else gitdir
        )
        if (common / "refs/replace").exists():
            return None
        if (common / "objects/info/alternates").exists():
            return None
        packed = common / "packed-refs"
        packed_raw = packed.read_bytes() if packed.exists() else b""
        if re.search(rb"\srefs/replace/", packed_raw):
            return None
        xdg = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
        configs = []
        for path in (
            common / "config",
            gitdir / "config.worktree",
            Path("/etc/gitconfig"),
            Path.home() / ".gitconfig",
            xdg / "git/config",
        ):
            raw = path.read_bytes() if path.exists() else b""
            if (
                re.search(rb"(?mi)^\s*\[\s*include(?:if)?(?:\s|\.|\])", raw)
                or b"reftable" in raw.lower()
                or b"objectformat" in raw.lower()
            ):
                return None
            configs.append((str(path), hashlib.sha256(raw).hexdigest()))
        binary = shutil.which("git")
        if binary is None:
            return None
        binary = Path(binary).resolve(strict=True)
        return (
            str(root),
            str(gitdir),
            str(common),
            tuple(arguments),
            bool(text),
            tuple(configs),
            hashlib.sha256(packed_raw).hexdigest(),
            str(binary),
            hashlib.sha256(binary.read_bytes()).hexdigest(),
            hashlib.sha256(canonical_bytes(dict(os.environ))).hexdigest(),
        )
    except (OSError, ValueError):
        return None


def _check_git_cache_authority(module, value):
    """Hits remain subject to actual parent authority, deadline and aggregate."""
    require_live_context(value)

    def check(state):
        controller = state["controllers"][value["controller_id"]]
        if (
            controller["status"] != "running"
            or time.monotonic()
            >= controller["started_monotonic"] + controller["deadline_seconds"]
            or module.Ledger.elapsed(state) >= 21600
        ):
            raise GateError("A2 cached Git authority/deadline/aggregate exhausted")

    module.Ledger(value["ledger_path"]).transaction(check)


def git_output(repo: str | Path, *arguments: str, text=False):
    """Managed Git misses; finite immutable hits retain fresh live guards."""
    global _GIT_CACHE_BYTES
    value = context()
    module = budget_api()
    key = _immutable_git_key(repo, arguments, text)
    with _GIT_CACHE_LOCK:
        cached = _GIT_CACHE.get(key) if key is not None else None
        if cached is not None:
            _GIT_CACHE.move_to_end(key)
    if cached is not None:
        _check_git_cache_authority(module, value)
        return cached
    command = ["git", "-C", str(repo), *arguments]
    result, output, error = module.BoundedGit(value["ledger_path"]).execute(command)
    if not result["cleanup_confirmed"] or result["status"] not in {"success", "error"}:
        raise GateError(
            "A2 metered Git resource/lifecycle failure; source status unresolved"
        )
    if result["exit_code"] != 0:
        raise subprocess.CalledProcessError(
            result["exit_code"], command, output=output, stderr=error
        )
    returned = output.decode() if text else output
    if (
        key is not None
        and result["status"] == "success"
        and len(output) <= _GIT_CACHE_MAX_BYTES
        and key == _immutable_git_key(repo, arguments, text)
    ):
        with _GIT_CACHE_LOCK:
            previous = _GIT_CACHE.pop(key, None)
            if previous is not None:
                _GIT_CACHE_BYTES -= len(
                    previous.encode() if isinstance(previous, str) else previous
                )
            while _GIT_CACHE and (
                len(_GIT_CACHE) >= _GIT_CACHE_MAX_ENTRIES
                or _GIT_CACHE_BYTES + len(output) > _GIT_CACHE_MAX_BYTES
            ):
                _, previous = _GIT_CACHE.popitem(last=False)
                _GIT_CACHE_BYTES -= len(
                    previous.encode() if isinstance(previous, str) else previous
                )
            _GIT_CACHE[key] = returned
            _GIT_CACHE_BYTES += len(output)
    return returned


def verify_clean_git_pin(repo, commit, tree):
    if git_output(repo, "rev-parse", "HEAD", text=True).strip() != commit:
        raise GateError("repository commit pin mismatch")
    if git_output(repo, "rev-parse", "HEAD^{tree}", text=True).strip() != tree:
        raise GateError("repository tree pin mismatch")
    if git_output(repo, "status", "--porcelain", text=True).strip():
        raise GateError("pinned repository worktree is dirty")


def require_ancestor(repo, ancestor):
    try:
        git_output(repo, "merge-base", "--is-ancestor", ancestor, "HEAD")
    except subprocess.CalledProcessError as error:
        raise GateError(
            "sealed reference E is not durable in actual integration lineage"
        ) from error


def require_external_directory(path):
    parent = Path(path).resolve()
    while not parent.exists():
        parent = parent.parent
    if not parent.is_dir():
        parent = parent.parent
    try:
        git_output(parent, "rev-parse", "--absolute-git-dir")
    except subprocess.CalledProcessError as error:
        if error.returncode == 128 and b"not a git repository" in error.stderr:
            return
        raise GateError(
            "external Git-tree validation failed without a nonrepository result"
        ) from error
    raise GateError("formal evidence must be outside every Git repository")


def source_blob(repo, commit, relative):
    return git_output(repo, "show", commit + ":" + relative)


def load_pinned_kernel(repo, *, commit, legacy):
    blob = git_output(
        repo, "rev-parse", commit + ":" + legacy.KERNEL_PATH, text=True
    ).strip()
    source = source_blob(repo, commit, legacy.KERNEL_PATH)
    if (
        blob != legacy.KERNEL_GIT_BLOB
        or hashlib.sha256(source).hexdigest() != legacy.KERNEL_SHA256
    ):
        raise GateError("arithmetic-kernel blob or SHA-256 mismatch")
    module = ModuleType("_pinned_b1_stochastic_acceptance")
    module.__file__ = commit + ":" + legacy.KERNEL_PATH
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    for name in ("promote_binary64", "assess", "structure_gates", "canonical_json"):
        if not callable(getattr(module, name, None)):
            raise GateError("pinned arithmetic kernel lacks " + name)
    return module


def verify_formal_closure(manifest):
    """Preserve frozen closure checks while replacing only Git process ownership."""
    from ...common.dependencies import verify_runtime_closure

    for name in ("production_commit", "production_tree"):
        if not re.fullmatch(r"[0-9a-f]{40}", manifest.get(name, "")):
            raise GateError("actual P commit/tree missing from dependency closure")
    required = {
        "gragra",
        "numpy",
        "scipy",
        "math",
        "fractions",
        "decimal",
        "_decimal",
        "tests.oracles.gen1.stochastic_formal.common",
    }
    if not required <= set(manifest["protected_packages"]) or not required <= {
        p["package"] for p in manifest["modules"]
    }:
        raise GateError(
            "formal closure omits mandatory protected scientific packages/sources"
        )
    if not any(p["kind"] == "import_configuration" for p in manifest["artifacts"]):
        raise GateError("formal closure omits material import configuration")
    for pin in manifest["modules"]:
        if pin["package"] != "gragra":
            continue
        if not isinstance(pin["git_blob"], str) or not re.fullmatch(
            r"[0-9a-f]{40}", pin["git_blob"]
        ):
            raise GateError("formal gragra module lacks exact Git blob")
        repo, commit = Path(pin["root"]), manifest["production_commit"]
        if (
            git_output(repo, "rev-parse", commit + "^{tree}", text=True).strip()
            != manifest["production_tree"]
        ):
            raise GateError("P tree mismatch")
        blob = git_output(
            repo, "rev-parse", commit + ":" + pin["relative_path"], text=True
        ).strip()
        source = source_blob(repo, commit, pin["relative_path"])
        if (
            blob != pin["git_blob"]
            or hashlib.sha256(source).hexdigest() != pin["sha256"]
        ):
            raise GateError("planned gragra blob bytes differ from actual P")
    value = context()
    modules = {p["name"]: p for p in manifest["modules"]}
    for name, source in [
        (__name__, value["adapter_source"]),
        (__package__ + ".budget", value["budget_source"]),
    ]:
        declared = modules.get(name)
        if (
            declared is None
            or declared["path"] != source["path"]
            or declared["sha256"] != source["sha256"]
        ):
            raise GateError(
                "A2 resource adapter/ledger absent from finite source closure"
            )
    return verify_runtime_closure(manifest, {}, [])
