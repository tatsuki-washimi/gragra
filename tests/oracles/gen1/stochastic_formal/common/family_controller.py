"""Construct reviewable family policies using actual readers; never issue reviews."""

from __future__ import annotations

import subprocess
from pathlib import Path

from .transport import (
    GateError,
    _write_new,
    artifact_pin,
    canonical_bytes,
    require_external_directory,
    strict_json,
    verify_artifact,
)


def reference_admission(artifacts, reviews):
    from .family_driver import (
        REFERENCE_ADMISSION_SCHEMA,
        derive_reference_pins,
        require_reference_admission,
    )

    pins = derive_reference_pins(artifacts)
    admission = {
        "schema": REFERENCE_ADMISSION_SCHEMA,
        "family": pins["family"],
        "decision": "PASS",
        "pins": pins,
        "reviews": reviews,
    }
    require_reference_admission(admission, artifacts)
    return admission


def write_candidate(spec, output: Path):
    """All three prior R/source admissions are compulsory; P is only a candidate."""
    from .controller import authorization_candidate
    from .family_driver import POLICY_SCHEMA, preflight

    fields = {
        "family",
        "production_repo",
        "comparator_repo",
        "integration_repo",
        "artifacts",
        "connection_assessor",
    }
    if set(spec) != fields:
        raise GateError("family controller specification fields mismatch")
    artifacts = spec["artifacts"]
    for pin in artifacts.values():
        verify_artifact(pin)
    procedure = strict_json(verify_artifact(artifacts["family_procedure"]))
    requirements = procedure.get(
        "required_connections", procedure.get("connection_checks")
    )
    if requirements is None:
        requirements = (
            ["domain", "whole_path", "normalization", "scaling_contract"]
            if spec["family"] == "T1"
            else ["numerical_procedure", "actual_x_transport", "SI", "coverage"]
        )
    pins = {
        "family": spec["family"],
        **{n + "_sha256": p["sha256"] for n, p in artifacts.items()},
    }
    for prefix in ("production", "comparator"):
        repo = spec[prefix + "_repo"]
        for suffix, expression in [("commit", "HEAD"), ("tree", "HEAD^{tree}")]:
            pins[prefix + "_" + suffix] = subprocess.run(
                ["git", "-C", repo, "rev-parse", expression],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
    policy = {
        "schema": POLICY_SCHEMA,
        **spec,
        "pins": pins,
        "connection_requirements": requirements,
    }
    output = output.absolute()
    require_external_directory(output)
    if output.exists():
        raise GateError("fresh candidate directory required")
    output.mkdir(parents=True, exist_ok=False)
    try:
        _write_new(output / "policy.json", canonical_bytes(policy))
        policy_pin = artifact_pin(output / "policy.json", "policy")
        prepared = preflight(policy_pin)
        candidate = authorization_candidate(
            family=spec["family"],
            phase="comparison",
            pins=prepared["pins"],
            evidence={**artifacts, "policy": policy_pin},
        )
        _write_new(output / "authorization-candidate.json", canonical_bytes(candidate))
        if preflight(policy_pin)["pins"] != candidate["pins"]:
            raise GateError("actual preflight pins differ from candidate")
        _write_new(
            output / "preflight.json",
            canonical_bytes(
                {
                    "status": "PASS_NO_COMPUTATION",
                    "pins": candidate["pins"],
                    "production_api_calls": 0,
                }
            ),
        )
        return artifact_pin(
            output / "authorization-candidate.json", "authorization_candidate"
        )
    except BaseException as error:
        _write_new(
            output / "preflight-refusal.json",
            canonical_bytes(
                {
                    "status": "REFUSED_NO_COMPUTATION",
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "production_api_calls": 0,
                }
            ),
        )
        raise
    finally:
        for path in output.iterdir():
            if path.is_file():
                path.chmod(0o444)
        output.chmod(0o555)
