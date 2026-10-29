"""Family execution additionally binds the actual frozen resource supervisor."""

from __future__ import annotations

import hashlib
from pathlib import Path

from ...common.dependencies import ImportGuard, verify_formal_closure
from ...common.supervisor import verify_worker_supervision
from ...common.transport import (
    GateError,
    SealedAttempt,
    _write_new,
    canonical_bytes,
    exact_keys,
    require_external_directory,
    verify_authorization,
)


def run_family_phase(
    *,
    authorization: dict,
    pins: dict,
    reviews: dict,
    phase: str,
    family: str,
    expected_keys: list[dict],
    output: str | Path,
    evaluator_factory,
    dependency_closure: dict,
    resource_launch: dict,
    prepared: dict,
):
    """Verify exact authority before extending actual run pins with supervision."""
    from .family_driver import frozen_expected_keys
    from .family_supervisor import verify_family_supervision

    if family not in {"R1", "B2", "A1", "A2", "A3", "T1"} or phase != "comparison":
        raise GateError("generic family execution scope mismatch")
    verify_authorization(authorization, pins, phase, reviews)
    if prepared["policy"]["family"] != family:
        raise GateError("prepared family differs from authorized worker scope")
    if (
        prepared["pins"] != pins
        or verify_family_supervision(prepared) != resource_launch
    ):
        raise GateError("actual resource launch binding mismatch")
    if exact_keys(expected_keys) != exact_keys(
        frozen_expected_keys(prepared["fixtures"])
    ):
        raise GateError("family worker expected keys differ from admitted F")
    if dependency_closure is None:
        raise GateError("finite runtime dependency closure required")
    if (
        dependency_closure != prepared["closure"]
        or hashlib.sha256(canonical_bytes(dependency_closure)).hexdigest()
        != pins["dependency_closure_sha256"]
    ):
        raise GateError("family worker closure differs from exact authorized bytes")
    if family == "A2":
        from . import resource_adapter

        resource_adapter.verify_formal_closure(dependency_closure)
    else:
        verify_formal_closure(dependency_closure)
    # Actual P must be preserved; never label the production source as a later
    # integration revision merely because its scientific bytes appear equal.
    if (
        dependency_closure["production_commit"] != pins["production_commit"]
        or dependency_closure["production_tree"] != pins["production_tree"]
    ):
        raise GateError("dependency closure does not identify actual P")
    supervisor_launch = verify_worker_supervision(pins)
    output = Path(output)
    if family == "A2":
        resource_adapter.require_external_directory(output)
    else:
        require_external_directory(output)
    if output.exists():
        raise GateError("fresh phase output directory required")
    output.mkdir(parents=True, exist_ok=False)
    run_pins = {
        **pins,
        "authorization_sha256": authorization["sha256"],
        "supervisor_launch_sha256": supervisor_launch["sha256"],
        "resource_launch_sha256": resource_launch["sha256"],
    }
    try:
        attempt = SealedAttempt(
            output,
            family=family,
            phase=phase,
            pins=run_pins,
            expected_keys=expected_keys,
        )
    except BaseException as error:
        _write_new(
            output / "initialization-failure.json",
            canonical_bytes(
                {
                    "schema": "gen1-stochastic-initialization-failure-v1",
                    "decision": "FAILED_INITIALIZATION",
                    "pins": run_pins,
                    "exception": {"type": type(error).__name__, "message": str(error)},
                }
            ),
        )
        for path in output.iterdir():
            if path.is_file():
                path.chmod(0o444)
        output.chmod(0o555)
        raise
    with (
        attempt as run,
        ImportGuard(dependency_closure) as guard,
    ):
        evaluator = evaluator_factory()
        if not callable(evaluator):
            raise GateError("evaluator factory did not return callable")
        guard.verify()
        for key in expected_keys:
            guard.verify()
            record = evaluator(key)
            if record.get("key") != key:
                raise GateError("actual evaluator key mismatch")
            guard.verify()
            run.append(record)
        _write_new(
            output / "runtime-dependencies.json", canonical_bytes(guard.verify())
        )
    return output
