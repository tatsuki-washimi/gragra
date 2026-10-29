"""Controller builders consume existing independent reviews; never invent decisions."""

from __future__ import annotations

import argparse
from pathlib import Path

from .b1_connections import SHAPES
from .b1_driver import REFERENCE_ADMISSION_SCHEMA, auxiliary_admission_pins
from .transport import (
    AUTH_SCHEMA,
    GateError,
    artifact_pin,
    bind_receipts,
    canonical_bytes,
    content_hash,
    exact_keys,
    key_id,
    strict_json,
    verify_artifact,
    verify_authorization,
)


def reference_admission(records_pin: dict, seal_pin: dict, reviews: dict) -> dict:
    """Build a reviewable admission, checking the same reader used by preflight."""
    from .b1_driver import reviewed_auxiliary_admission

    pins = auxiliary_admission_pins(records_pin, seal_pin)
    admission = {
        "schema": REFERENCE_ADMISSION_SCHEMA,
        "family": "B1_aux",
        "decision": "PASS",
        "f_b1_aux_sha256": pins["f_b1_aux_sha256"],
        "records_sha256": records_pin["sha256"],
        "seal_sha256": seal_pin["sha256"],
        "pins": pins,
        "reviews": reviews,
    }
    reviewed_auxiliary_admission(
        admission,
        records_pin,
        seal_pin,
        pins["f_b1_aux_sha256"],
        {r: p["sha256"] for r, p in reviews.items()},
    )
    return admission


def authorization_candidate(
    *, family: str, phase: str, pins: dict, evidence: dict
) -> dict:
    """Create only the candidate for review; AUTHORIZED requires external PASS files."""
    if phase not in {"connection", "comparison"}:
        raise GateError("unknown authorization phase")
    for pin in evidence.values():
        verify_artifact(pin)
    return {
        "schema": AUTH_SCHEMA,
        "family": family,
        "phase": phase,
        "status": "REVIEW_CANDIDATE",
        "pins": pins,
        "evidence": evidence,
    }


def authorization_from_reviews(candidate_pin: dict, reviews: dict) -> dict:
    candidate = strict_json(verify_artifact(candidate_pin))
    binding = {
        "schema": AUTH_SCHEMA,
        "family": candidate["family"],
        "phase": candidate["phase"],
        "status": "AUTHORIZED",
        "pins": candidate["pins"],
        "candidate": candidate_pin,
        "reviews": reviews,
    }
    # Verify all immutable external review digests before returning authority.
    # Use an in-memory check with the same exact review contract as the reader.
    roles = {"science", "numerical", "provenance"}
    if set(reviews) != roles:
        raise GateError("three independent reviews required")
    ids = set()
    for role, pin in reviews.items():
        review = strict_json(verify_artifact(pin))
        if (
            review.get("schema") != "gen1-stochastic-review-v1"
            or review.get("role") != role
            or review.get("decision") != "PASS"
            or review.get("candidate_sha256") != candidate_pin["sha256"]
            or review.get("pins") != candidate["pins"]
            or not isinstance(review.get("reviewer_id"), str)
            or not review["reviewer_id"]
            or review["reviewer_id"] in ids
        ):
            raise GateError("independent authorization review mismatch")
        ids.add(review["reviewer_id"])
    return binding


def receipt_expectations(
    *,
    fixtures: dict,
    r_aux_pin: dict,
    fixture_pin: dict,
    endpoint_paths: dict[str, Path],
) -> dict:
    """Declare the full field identity mapping and all 30 explicit dependencies."""
    reference = exact_keys(fixtures["expected_reference_auxiliary_keys"])
    science = fixtures["expected_keys"]
    if len(science) != 30:
        raise GateError("frozen 30 science dependencies required")
    rows = []
    for key in fixtures["expected_auxiliary_keys"]:
        shape = SHAPES[key["observable"]]
        count = 1 if shape == [] else 3 if shape == [3] else 9
        independent = key_id(key) in reference
        rows.append(
            {
                "receipt_id": "B1-connection-" + content_hash(key)[:24],
                "key": key,
                "kind": "independent_reference"
                if independent
                else "production_consistency",
                "source": r_aux_pin if independent else fixture_pin,
                "destination": artifact_pin(
                    endpoint_paths[key["observable"]], "production_source"
                ),
                "component_mapping": {
                    "source_shape": shape,
                    "destination_shape": shape,
                    "indices": [[i, i] for i in range(count)],
                },
                "representation_mapping": {
                    "source": key["representation"],
                    "destination": key["representation"],
                    "G_power": 0,
                },
                "science_keys": science,
                "schema_version": 1,
                "rule_version": "B1-frozen-first-order-and-consistency-v1",
            }
        )
    return {"schema": "gen1-stochastic-receipt-expectations-v1", "receipts": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    request = strict_json(
        verify_artifact(
            {
                "path": str(args.request.absolute()),
                "sha256": args.request_sha256,
                "kind": "controller_request",
            }
        )
    )
    operation = request["operation"]
    kwargs = request["arguments"]
    if operation == "reference_admission":
        result = reference_admission(**kwargs)
    elif operation == "authorization_candidate":
        result = authorization_candidate(**kwargs)
    elif operation == "authorization_from_reviews":
        result = authorization_from_reviews(**kwargs)
    elif operation == "bind_receipts":
        result = bind_receipts(**kwargs)
    else:
        raise GateError("unknown controller operation")
    if args.output.exists():
        raise GateError("fresh controller output required")
    args.output.write_bytes(canonical_bytes(result))
    if operation == "authorization_from_reviews":
        verify_authorization(
            artifact_pin(args.output, "authorization"),
            result["pins"],
            result["phase"],
            {r: p["sha256"] for r, p in result["reviews"].items()},
        )


if __name__ == "__main__":
    main()
