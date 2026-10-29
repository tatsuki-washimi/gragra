"""Mandatory reviewed worker boundary using the admitted process-group cleanup."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

from tests.oracles.gen1 import stochastic_attempt as _attempt

from .transport import (
    GateError,
    _write_new,
    artifact_pin,
    canonical_bytes,
    require_external_directory,
    strict_json,
    verify_artifact,
)

# Preserve the admitted script's historical absolute sibling import.
if (
    "stochastic_attempt" in sys.modules
    and sys.modules["stochastic_attempt"] is not _attempt
):
    raise RuntimeError("unexpected admitted supervisor sibling import")
sys.modules["stochastic_attempt"] = _attempt
_cleanup = importlib.import_module("tests.oracles.gen1.run_stochastic_local")
_process_group_exists = _cleanup._process_group_exists
_reap = _cleanup._reap

TOKEN = "GRAGRA_STOCHASTIC_SUPERVISOR_LAUNCH"


def verify_worker_supervision(pins: dict) -> dict:
    """Direct library/CLI execution cannot construct a formal evaluator."""
    try:
        pin = strict_json(os.environ[TOKEN])
    except (KeyError, ValueError) as error:
        raise GateError(
            "formal worker requires reviewed external supervisor"
        ) from error
    launch = strict_json(verify_artifact(pin))
    if (
        launch.get("mode") != "FORMAL"
        or launch.get("policy_sha256") != pins.get("policy_sha256")
        or launch.get("supervisor_pid") != os.getppid()
        or os.getpgid(0) != os.getpid()
    ):
        raise GateError("formal supervisor parent/session/policy mismatch")
    procedure = strict_json(verify_artifact(launch["frozen_procedure"]))
    if launch["timeout_seconds"] != procedure["resources"]["timeout_seconds"]:
        raise GateError("supervisor timeout differs from frozen procedure")
    verify_artifact(launch["supervisor"])
    verify_artifact(launch["cleanup_helper"])
    return pin


def supervise(
    command: list[str],
    *,
    evidence: Path,
    timeout_seconds: int,
    policy_pin: dict | None = None,
    frozen_procedure: dict | None = None,
    source_roots: list[Path] | None = None,
) -> dict:
    """Timeout is normative; observed RSS is metadata when no frozen cap exists."""
    roots = [r.resolve(strict=True) for r in source_roots or []]
    evidence = evidence.absolute()
    require_external_directory(evidence)
    if timeout_seconds <= 0 or not command or evidence.exists():
        raise GateError(
            "fresh supervisor evidence and positive frozen timeout required"
        )
    for root in roots:
        if evidence.resolve().is_relative_to(root):
            raise GateError("supervisor evidence must be outside source trees")
    formal = policy_pin is not None
    if formal:
        policy = strict_json(verify_artifact(policy_pin))
        if len(roots) < 2 or frozen_procedure is None:
            raise GateError("formal supervisor source trees/procedure required")
        required_roots = {
            Path(policy[n]).resolve() for n in ("production_repo", "comparator_repo")
        }
        if not required_roots <= set(roots):
            raise GateError("supervisor source roots differ from actual policy")
        procedure = strict_json(verify_artifact(frozen_procedure))
        if timeout_seconds != procedure["resources"]["timeout_seconds"]:
            raise GateError("supervisor resource differs from frozen procedure")
        if any(
            frozen_procedure[name] != policy["artifacts"]["procedure"][name]
            for name in ("path", "sha256")
        ):
            raise GateError("supervisor procedure differs from reviewed policy")
    evidence.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    process = None
    interrupted = None
    returncode = None
    cleanup = {"status": "not_started"}
    try:
        helper = Path(__file__).parents[2] / "run_stochastic_local.py"
        launch = {
            "command": command,
            "timeout_seconds": timeout_seconds,
            "supervisor": artifact_pin(__file__, "supervisor_source"),
            "cleanup_helper": artifact_pin(helper, "admitted_cleanup_helper"),
            "supervisor_pid": os.getpid(),
            "mode": "FORMAL" if formal else "DEVELOPMENT",
            "policy_sha256": policy_pin["sha256"] if formal else None,
            "frozen_procedure": frozen_procedure,
        }
        _write_new(evidence / "launch.json", canonical_bytes(launch))
        environment = dict(os.environ)
        environment[TOKEN] = json.dumps(
            artifact_pin(evidence / "launch.json", "supervisor_launch")
        )
        with (
            (evidence / "stdout.txt").open("xb") as stdout,
            (evidence / "stderr.txt").open("xb") as stderr,
        ):
            process = subprocess.Popen(
                command,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                env=environment,
            )
            try:
                returncode = process.wait(timeout=timeout_seconds)
                if _process_group_exists(process.pid):
                    interrupted = "SURVIVING_DESCENDANTS"
            except subprocess.TimeoutExpired:
                interrupted = "TIMEOUT"
    except BaseException as error:
        interrupted = type(error).__name__
        raise
    finally:
        if process is not None:
            cleanup = _reap(process, terminate=interrupted is not None)
        for name in ("stdout.txt", "stderr.txt"):
            path = evidence / name
            if not path.exists():
                _write_new(path, b"")
        result = {
            "schema": "gen1-stochastic-supervisor-seal-v1",
            "outcome": interrupted or "CHILD_EXITED",
            "exit_code": process.returncode if process is not None else None,
            "cleanup_confirmed": process is None
            or cleanup["status"] == "confirmed_stopped",
            "cleanup": cleanup,
            "elapsed_seconds": str(time.monotonic() - started),
            "observed_children_maxrss_kib": resource.getrusage(
                resource.RUSAGE_CHILDREN
            ).ru_maxrss,
            "rss_acceptance_cap": None,
            "rss_cap_reason": "no frozen B1 RSS acceptance cap; observation only",
            "artifacts": {
                n: artifact_pin(evidence / n, "supervisor_evidence")
                for n in ("stdout.txt", "stderr.txt")
            },
        }
        if (evidence / "launch.json").exists():
            result["artifacts"]["launch"] = artifact_pin(
                evidence / "launch.json", "supervisor_launch"
            )
        _write_new(evidence / "seal.json", canonical_bytes(result))
        for path in evidence.iterdir():
            if path.is_file():
                path.chmod(0o444)
        evidence.chmod(0o555)
    if returncode != 0 or interrupted or not result["cleanup_confirmed"]:
        raise GateError(
            "comparison child failed; immutable supervisor evidence retained"
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument("--source-root", type=Path, action="append")
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--policy-sha256")
    parser.add_argument("--frozen-procedure", type=Path)
    parser.add_argument("--frozen-procedure-sha256")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    policy = (
        {
            "path": str(args.policy.absolute()),
            "sha256": args.policy_sha256,
            "kind": "policy",
        }
        if args.policy
        else None
    )
    procedure = (
        {
            "path": str(args.frozen_procedure.absolute()),
            "sha256": args.frozen_procedure_sha256,
            "kind": "frozen_procedure",
        }
        if args.frozen_procedure
        else None
    )
    supervise(
        command,
        evidence=args.evidence,
        timeout_seconds=args.timeout_seconds,
        policy_pin=policy,
        frozen_procedure=procedure,
        source_roots=args.source_root,
    )


if __name__ == "__main__":
    main()
