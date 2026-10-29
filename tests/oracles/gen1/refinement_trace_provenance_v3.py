"""V3 provenance commitment helpers and fail-closed admission decisions.

This module does not create a trust root. A provider supplied by a caller must
fetch an anchor from a separate trust domain and verify its issuer/root before
returning ``trust_root_verified=True``. A provider loaded from trace data is
never consulted.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol

SCHEMA = "gragra.gen1.refinement-trace-provenance-commitment.v3"
EXECUTION_STATES = {"running", "complete", "failed", "interrupted"}
SEMANTIC_STATES = {"SEMANTIC_VALID", "SEMANTIC_INVALID", "SEMANTIC_NOT_RUN"}
PROVENANCE_STATES = {
    "PROVENANCE_VERIFIED",
    "PROVENANCE_UNVERIFIED",
    "PROVENANCE_MISMATCH",
    "PROVENANCE_NOT_RUN",
}
ADMISSION_STATES = {
    "ADMITTED_WITH_DECLARED_SCOPE",
    "NOT_ADMITTED",
    "INCOMPLETE_EVIDENCE",
}


class TrustedCommitmentProvider(Protocol):
    """Adapter for an independently controlled commitment store/verifier."""

    def verify_commitment(
        self, lookup_key: str, expected_document: dict[str, Any]
    ) -> dict[str, Any]:
        """Fetch and authenticate an external document for this exact lookup."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def commitment_document(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("commitment payload must be an object")
    body = {"schema": SCHEMA, "payload": payload}
    return {**body, "commitment_sha256": canonical_sha256(body)}


def callback_payload(record: dict[str, Any], binding: dict[str, Any]) -> dict[str, Any]:
    """Bind exact serialized callbacks and all declared run identities.

    The returned digest is only a commitment candidate. It gains provenance
    meaning only when an independent provider authenticates the same document.
    """
    required = {
        "run_id",
        "dataset",
        "source_row_index",
        "observer_sha256",
        "auditor_sha256",
        "oracle_module_sha256",
        "frozen_reference_sha256",
        "fixture_sha256",
        "certificate_sha256",
        "environment",
        "numeric_representation",
    }
    if not isinstance(binding, dict) or not required.issubset(binding):
        missing = sorted(required - set(binding if isinstance(binding, dict) else ()))
        raise ValueError(f"incomplete provenance binding: {missing}")
    calls = record.get("original_entrypoint", {}).get("captured_calls")
    if not isinstance(calls, list):
        raise ValueError("record has no ordered captured callback list")
    return {
        "schema": SCHEMA,
        "binding": binding,
        "record_schema": record.get("schema"),
        "record_status": record.get("status"),
        "worker_independence_evidence": record.get("process_independence"),
        "callback_count": len(calls),
        "callbacks": calls,
    }


def verify_external_commitment(
    expected_document: dict[str, Any],
    provider: TrustedCommitmentProvider | None,
    *,
    lookup_key: str,
) -> dict[str, Any]:
    """Compare against an independently authenticated external commitment.

    An absent provider or an unverified trust root is ``UNVERIFIED``. Once an
    authenticated root is established, a malformed or different document is a
    provenance mismatch. The provider implementation itself is a trust input.
    """
    if provider is None:
        return {
            "status": "PROVENANCE_UNVERIFIED",
            "reason": "no independent provenance provider is configured",
            "anchor_id": None,
        }
    try:
        result = provider.verify_commitment(lookup_key, expected_document)
    except Exception as error:  # provider errors cannot silently produce trust
        return {
            "status": "PROVENANCE_UNVERIFIED",
            "reason": f"provenance provider could not verify an anchor: {type(error).__name__}",
            "anchor_id": None,
        }
    if not isinstance(result, dict) or result.get("trust_root_verified") is not True:
        return {
            "status": "PROVENANCE_UNVERIFIED",
            "reason": "provider did not independently verify its trust root",
            "anchor_id": result.get("anchor_id") if isinstance(result, dict) else None,
        }
    actual = result.get("document")
    try:
        if not isinstance(actual, dict) or not isinstance(expected_document, dict):
            raise ValueError("commitment document is not an object")
        expected_body = {
            key: expected_document[key]
            for key in ("schema", "payload")
        }
        actual_body = {key: actual[key] for key in ("schema", "payload")}
        if (
            expected_document.get("schema") != SCHEMA
            or actual.get("schema") != SCHEMA
            or expected_document.get("commitment_sha256")
            != canonical_sha256(expected_body)
            or actual.get("commitment_sha256") != canonical_sha256(actual_body)
        ):
            raise ValueError("commitment document digest or schema is invalid")
    except (KeyError, TypeError, ValueError):
        return {
            "status": "PROVENANCE_MISMATCH",
            "reason": "authenticated commitment document is malformed",
            "anchor_id": result.get("anchor_id"),
        }
    if canonical_json_bytes(actual) != canonical_json_bytes(expected_document):
        return {
            "status": "PROVENANCE_MISMATCH",
            "reason": "authenticated callback commitment differs from the saved record",
            "anchor_id": result.get("anchor_id"),
        }
    return {
        "status": "PROVENANCE_VERIFIED",
        "reason": "authenticated external commitment matches the bound callback record",
        "anchor_id": result.get("anchor_id"),
    }


def admission_decision(
    *,
    execution_state: str,
    semantic_status: str,
    provenance_status: str,
    independent_audit_valid: bool,
    certificate_validation_valid: bool,
    trend_validation_complete: bool,
) -> str:
    """Derive admission without collapsing execution, semantics, or trust."""
    if execution_state not in EXECUTION_STATES:
        return "NOT_ADMITTED"
    if semantic_status not in SEMANTIC_STATES or provenance_status not in PROVENANCE_STATES:
        return "NOT_ADMITTED"
    if semantic_status == "SEMANTIC_INVALID" or provenance_status == "PROVENANCE_MISMATCH":
        return "NOT_ADMITTED"
    if execution_state == "failed":
        return "NOT_ADMITTED"
    if execution_state in {"running", "interrupted"}:
        return "INCOMPLETE_EVIDENCE"
    if type(trend_validation_complete) is not bool:
        return "NOT_ADMITTED"
    if semantic_status == "SEMANTIC_NOT_RUN" or provenance_status == "PROVENANCE_NOT_RUN":
        return "INCOMPLETE_EVIDENCE"
    if (
        semantic_status != "SEMANTIC_VALID"
        or provenance_status != "PROVENANCE_VERIFIED"
        or independent_audit_valid is False
        or certificate_validation_valid is False
        or not trend_validation_complete
    ):
        return "NOT_ADMITTED"
    if independent_audit_valid is not True or certificate_validation_valid is not True:
        return "INCOMPLETE_EVIDENCE"
    return "ADMITTED_WITH_DECLARED_SCOPE"
