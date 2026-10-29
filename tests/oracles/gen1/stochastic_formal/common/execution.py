"""Reviewed phase execution with durable failure evidence and runtime closure."""

from __future__ import annotations

from pathlib import Path

from .dependencies import ImportGuard, verify_formal_closure
from .supervisor import verify_worker_supervision
from .transport import (
    GateError,
    SealedAttempt,
    _write_new,
    canonical_bytes,
    exact_keys,
    require_external_directory,
    require_science_authority,
    verify_authorization,
)


def run_reviewed_phase(
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
    connection_seal: dict | None = None,
    connection_keys: list[dict] | None = None,
    connection_pins: dict | None = None,
):
    """Only an exact independently reviewed authority can construct an evaluator.

    The callable is family-owned and must return already assessed actual records.
    This runner never imports a reference computational helper or supplies gates.
    """
    if family == "B1" and phase == "comparison":
        require_science_authority(
            authorization,
            pins,
            reviews,
            connection_seal=connection_seal,
            connection_keys=connection_keys or [],
            connection_pins=connection_pins or {},
        )
    else:
        verify_authorization(authorization, pins, phase, reviews)
    expected = exact_keys(expected_keys)
    required = (
        70
        if family == "B1" and phase == "connection"
        else 30
        if family == "B1"
        else None
    )
    if required is not None and len(expected) != required:
        raise GateError("phase exact coverage count mismatch")
    if dependency_closure is None:
        raise GateError("finite runtime dependency closure required")
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
    require_external_directory(output)
    if output.exists():
        raise GateError("fresh phase output directory required")
    output.mkdir(parents=True, exist_ok=False)
    run_pins = {
        **pins,
        "authorization_sha256": authorization["sha256"],
        "supervisor_launch_sha256": supervisor_launch["sha256"],
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
