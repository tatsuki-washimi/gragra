"""A2 bounded controller; formal generation requires exact independently reviewed E."""

import argparse
import hashlib
import json
import os
import platform
import signal
import sys
import time
from pathlib import Path

import mpmath
from budget import BoundedGit, ControllerAccount, Ledger, live_group
from freeze import verify_freeze
from purpose import planned_purposes, separation
from review import verify_implementation_reviews

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "tests/oracles/gen1"))
from stochastic_acceptance import KEY_FIELDS  # noqa: E402
from stochastic_attempt import seal_attempt  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_authorization(auth, policy):
    git = BoundedGit(auth["resource_ledger_path"])
    head = git.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if git.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("clean E required")
    if auth.get("E_commit") != head or auth.get("family") != "A2":
        raise ValueError("exact E required")
    freeze = verify_freeze(
        ROOT, auth["family_freeze"], policy, auth["procedure_sha256"]
    )
    if auth.get("F_sha256") != auth["family_freeze"]["sha256"]:
        raise ValueError("actual F sidecar pin required")
    policy_bytes = Path(policy).read_bytes()
    policy_data = json.loads(policy_bytes)
    separation(policy_data, auth["procedure_sha256"])
    verify_implementation_reviews(auth, ROOT, head)
    if auth.get("fixture_sha256") != sha(policy) or auth.get("authorized_by") != "root":
        raise ValueError("root F authorization required")
    integration = Path(auth["integration_root"]).resolve()
    durable = git.check_output(
        ["git", "-C", str(integration), "rev-parse", "HEAD"], text=True
    ).strip()
    if durable != auth["durability_commit"]:
        raise ValueError("durability pin must equal actual integration HEAD")
    git.run(
        ["git", "-C", str(integration), "merge-base", "--is-ancestor", head, durable],
        check=True,
    )
    committed_pins = {
        **freeze["artifact_hashes"],
        **freeze["baseline_pins"],
        str(Path(auth["family_freeze"]["path"]).resolve().relative_to(ROOT)): auth[
            "family_freeze"
        ]["sha256"],
    }
    for path, expected in committed_pins.items():
        committed = git.check_output(["git", "show", f"{head}:{path}"], cwd=ROOT)
        if hashlib.sha256(committed).hexdigest() != expected:
            raise ValueError("F/baseline byte not committed in E")
    required_paths = {
        str(p.relative_to(ROOT)) for p in Path(__file__).parent.glob("*.py")
    }
    required_paths.update(
        {
            "tests/oracles/gen1/stochastic_attempt.py",
            "tests/oracles/gen1/stochastic_acceptance.py",
            "tests/oracles/gen1/stochastic_f_common_v2.py",
        }
    )
    if {pin["path"] for pin in auth["implementation_pins"]} != required_paths:
        raise ValueError("complete exact implementation closure required")
    environment = {
        "python_version": platform.python_version(),
        "python_executable_sha256": sha(sys.executable),
        "mpmath_version": mpmath.__version__,
        "mpmath_module_sha256": sha(mpmath.__file__),
        "python_implementation": platform.python_implementation(),
        "executable": str(Path(sys.executable).absolute()),
        "mpmath_files": {
            str(path.relative_to(Path(mpmath.__file__).parent)): sha(path)
            for path in sorted(Path(mpmath.__file__).parent.rglob("*.py"))
        },
        "sitecustomize_sha256": sha(
            Path(sys.prefix) / "lib/python3.11/site-packages/sitecustomize.py"
        ),
    }
    if auth.get("reference_environment") != environment:
        raise ValueError("reference environment mismatch")
    if {pin["sha256"] for pin in auth["source_artifacts"]} != {
        pin["sha256"] for pin in policy_data["source_pins"]
    }:
        raise ValueError("primary source exact set mismatch")
    for source in auth["source_artifacts"]:
        if sha(source["path"]) != source["sha256"]:
            raise ValueError("primary source actual bytes mismatch")
    for pin in auth["implementation_pins"]:
        committed = git.check_output(["git", "show", f"{head}:{pin['path']}"], cwd=ROOT)
        if (
            sha(ROOT / pin["path"]) != pin["sha256"]
            or hashlib.sha256(committed).hexdigest() != pin["sha256"]
        ):
            raise ValueError("implementation committed byte mismatch")
    return head


def _execute(args, p):
    if any(
        (parent / ".git").is_file() or (parent / ".git" / "HEAD").is_file()
        for parent in (args.attempt.resolve(), *args.attempt.resolve().parents)
    ):
        p.error("attempt artifacts must remain outside every Git worktree")
    policy = json.loads(args.policy.read_bytes())
    auth = None
    head = None
    if not args.calibration:
        if args.authorization is None:
            p.error("formal HOLD: root exact F/E reviewed authorization required")
        auth = json.loads(args.authorization.read_bytes())
        if args.attempt.resolve().parent != Path(auth["campaign_dir"]).resolve():
            p.error("fresh attempt must be in authorized campaign directory")
        if args.ledger.resolve() != Path(auth["resource_ledger_path"]).resolve():
            p.error("exact campaign resource ledger required")
        try:
            head = verify_authorization(auth, args.policy)
        except BaseException as exc:
            import uuid

            refusal = args.ledger.parent / (
                "preflight-refusal-" + uuid.uuid4().hex + ".json"
            )
            refusal.write_text(
                json.dumps(
                    {
                        "family": "A2",
                        "status": "PREFLIGHT_REFUSED",
                        "formal_attempt_allocated": False,
                        "error": str(exc),
                        "authorization_sha256": sha(args.authorization),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            refusal.chmod(0o444)
            raise
    command = [
        str(args.python),
        str(Path(__file__).with_name("worker.py")),
        "--proposal",
        str(args.policy),
    ]
    if args.calibration:
        command.append("--calibration")
    else:
        command.extend(["--authorization", str(args.authorization)])
    if args.attempt.exists():
        raise FileExistsError(args.attempt)
    ledger = Ledger(args.ledger)
    kind = "calibration" if args.calibration else "formal"
    # Set recovery context BEFORE the transaction. A signal after durable append
    # but before returned-id assignment is recovered by exact owned ledger row.
    args.allocation = {
        "policy": policy,
        "auth": auth,
        "head": head,
        "command": command,
        "kind": kind,
        "started": time.time(),
    }
    reservation = ledger.reserve(kind, command)
    try:
        args.attempt.mkdir(parents=True, exist_ok=False)
    except BaseException:
        ledger.recover(reservation)
        raise
    try:
        result = ledger.run(
            command,
            kind=kind,
            reservation=reservation,
            stdout=args.attempt / "stdout.jsonl",
            stderr=args.attempt / "stderr.jsonl",
        )
    except BaseException as exc:
        entry = json.loads(args.ledger.read_bytes())["entries"][reservation]
        cleanup = entry["pid"] is None or not live_group(entry["pid"])
        if cleanup and "elapsed_seconds" not in entry:
            ledger.recover(reservation)
        result = {
            "ledger_id": reservation,
            "status": "exception",
            "exit_code": entry.get("exit_code"),
            "cleanup_confirmed": cleanup,
            "controller_error": str(exc),
            "launch_state": "not_started" if entry["pid"] is None else "started",
        }
    return finish_attempt(args, policy, auth, head, result)


def recovery_artifacts(args, artifacts, result, recovery):
    if recovery:
        # A interrupted sealer may already have published immutable artifacts.
        # Preserve those exact bytes and bind new failure evidence separately.
        artifacts = {
            name: (args.attempt / name).read_bytes()
            if (args.attempt / name).exists()
            else raw
            for name, raw in artifacts.items()
        }
        artifacts["controller-recovery.json"] = (
            json.dumps(result, sort_keys=True) + "\n"
        ).encode()
    return artifacts


def finish_attempt(args, policy, auth, head, result, recovery=False):
    status = result["status"]
    if status in {"rss_limit", "pending_cleanup"}:
        status = "error"
    # The timer stops active work. Final sealing uses the reserved cleanup/I/O
    # margin; external verification must still establish the total wall bound.
    if args.controller_id is not None:
        args.begin_finalization()
    seal_attempt(
        args.attempt,
        key_fields=KEY_FIELDS,
        expected_keys=(
            [
                item["key"]
                for item in planned_purposes(policy, sha(args.policy), calibration=True)
            ]
            if args.calibration
            else policy["expected_keys"] + policy["expected_scalar_keys"]
        ),
        record_key_path=("key",),
        records_bytes=(
            (args.attempt / "stdout.jsonl").read_bytes()
            if (args.attempt / "stdout.jsonl").exists()
            else b""
        ),
        execution={
            "status": status,
            "exit_code": result["exit_code"],
            "launch_state": result.get("launch_state", "started"),
            "cleanup": {
                "status": "confirmed_stopped"
                if result.get("cleanup_confirmed", True)
                else "pending"
            },
        },
        artifacts=recovery_artifacts(
            args,
            {
                "policy.json": args.policy.read_bytes(),
                "authorization.json": b"null\n"
                if args.calibration
                else args.authorization.read_bytes(),
                "family-freeze.json": b"null\n"
                if args.calibration
                else Path(auth["family_freeze"]["path"]).read_bytes(),
                "purpose-records.json": (
                    json.dumps(
                        separation(
                            policy,
                            sha(args.policy)
                            if args.calibration
                            else auth["procedure_sha256"],
                        ),
                        sort_keys=True,
                    )
                    + "\n"
                ).encode(),
                "resource-ledger.json": args.ledger.read_bytes(),
                "resource-result.json": (
                    json.dumps(result, sort_keys=True) + "\n"
                ).encode(),
            },
            result,
            recovery,
        ),
        provenance={
            "family": "A2",
            "E_commit": head,
            "F_sha256": None if args.calibration else auth["F_sha256"],
            "purpose": "calibration" if args.calibration else "formal-science",
            "fixture_sha256": sha(args.policy),
            "production_imported": False,
        },
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "success" else 1


def recover_allocation(args, exc):
    state = getattr(args, "allocation", None)
    if state is None:
        return False
    ledger = Ledger(args.ledger)
    entries = ledger.transaction(
        lambda data: [
            dict(e)
            for e in data["entries"]
            if e["kind"] == state["kind"]
            and e["command"] == state["command"]
            and e.get("parent_pid") == os.getpid()
            and e["started"] >= state["started"]
        ]
    )
    if not entries:
        return False
    if len(entries) != 1:
        raise RuntimeError("ambiguous owned allocation; recovery HOLD")
    entry = entries[0]
    if args.controller_id is not None:
        args.begin_finalization()
    if entry["pid"] is not None and live_group(entry["pid"]):
        import contextlib

        with contextlib.suppress(ProcessLookupError):
            os.killpg(entry["pid"], signal.SIGKILL)
        deadline = time.monotonic() + 10
        while live_group(entry["pid"]) and time.monotonic() < deadline:
            with contextlib.suppress(ChildProcessError):
                os.waitpid(entry["pid"], os.WNOHANG)
            time.sleep(0.05)
    if entry["pid"] is not None and live_group(entry["pid"]):
        raise RuntimeError("owned group remains active; recovery HOLD")
    if "elapsed_seconds" not in entry:
        ledger.recover(entry["id"])
    if not args.attempt.exists():
        args.attempt.mkdir(parents=True, exist_ok=False)
    if (args.attempt / "seal.json").exists():
        return True  # Never rewrite an already published one-way seal.
    result = {
        "ledger_id": entry["id"],
        "status": "interruption" if isinstance(exc, KeyboardInterrupt) else "exception",
        "exit_code": entry.get("exit_code"),
        "cleanup_confirmed": True,
        "launch_state": "not_started" if entry["pid"] is None else "started",
        "controller_error": str(exc),
        "recovery": "exact durable owned allocation; original artifacts retained",
    }
    finish_attempt(
        args, state["policy"], state["auth"], state["head"], result, recovery=True
    )
    return True


def execute(args, p):
    try:
        return _execute(args, p)
    except BaseException as exc:
        try:
            allocated = recover_allocation(args, exc)
            if not allocated and hasattr(args, "allocation"):
                import uuid

                refusal = args.ledger.parent / (
                    "preflight-refusal-" + uuid.uuid4().hex + ".json"
                )
                refusal.write_text(
                    json.dumps(
                        {
                            "family": "A2",
                            "status": "PREFLIGHT_REFUSED",
                            "formal_attempt_allocated": False,
                            "error": str(exc),
                            "command": args.allocation["command"],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                refusal.chmod(0o444)
        except BaseException as recovery_error:
            import uuid

            path = args.ledger.parent / (
                "controller-pending-recovery-" + uuid.uuid4().hex + ".json"
            )
            with path.open("x") as stream:
                json.dump(
                    {
                        "family": "A2",
                        "status": "PENDING_RECOVERY",
                        "attempt": str(args.attempt),
                        "original_error": str(exc),
                        "recovery_error": str(recovery_error),
                    },
                    stream,
                    sort_keys=True,
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            path.chmod(0o444)
            args.pending_recovery = str(path)
            raise
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--attempt", type=Path, required=True)
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--authorization", type=Path)
    p.add_argument("--calibration", action="store_true")
    p.add_argument("--controller-id", type=int)
    p.add_argument("--controller-attestation", type=Path)
    args = p.parse_args()
    if args.controller_id is None:
        if not args.calibration:
            p.error("formal HOLD: prelaunch controller whole-wall reservation required")
        # Historical calibration under an outer charged supervisor remains
        # valid. Direct calibration can exercise the formal controller topology.
        return execute(args, p)
    if args.controller_attestation is None or args.controller_attestation.exists():
        p.error("fresh external controller attestation path required")
    account = ControllerAccount(Ledger(args.ledger))
    deadline = account.claim(
        args.controller_id,
        [sys.executable, *sys.argv],
        require_explicit_origin=not args.calibration,
    )
    previous = signal.getsignal(signal.SIGALRM)
    active = True

    def begin_finalization():
        nonlocal active
        active = False
        signal.setitimer(
            signal.ITIMER_REAL, max(0.000001, deadline + 100 - time.monotonic())
        )

    def stop_active_work(_signal, _frame):
        if not active:
            raise SystemExit("controller whole-wall bound; external recovery required")
        begin_finalization()
        raise KeyboardInterrupt("controller active deadline; cleanup margin reserved")

    args.begin_finalization = begin_finalization
    signal.signal(signal.SIGALRM, stop_active_work)
    signal.setitimer(signal.ITIMER_REAL, max(0.000001, deadline - time.monotonic()))
    outcome = "exception"
    try:
        result = execute(args, p)
        outcome = "success" if result == 0 else "error"
        return result
    except BaseException as exc:
        outcome = (
            "pending_recovery"
            if getattr(args, "pending_recovery", None)
            else "interruption"
            if isinstance(exc, KeyboardInterrupt)
            else "exception"
        )
        raise
    finally:
        begin_finalization()
        attestation = account.outcome(args.controller_id, outcome)
        # All claimed outcomes remain conservatively charged even if final I/O
        # fails; root's actual external-span attestation is still mandatory.
        with args.controller_attestation.open("x") as stream:
            json.dump(attestation, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            import os

            os.fsync(stream.fileno())
        args.controller_attestation.chmod(0o444)
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


if __name__ == "__main__":
    raise SystemExit(main())
