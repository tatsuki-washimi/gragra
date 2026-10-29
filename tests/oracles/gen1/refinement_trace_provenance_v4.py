"""V4 identity, external-anchor, and admission contracts.

These helpers contain no trust root, production verifier, recorder, or numerical
oracle. Callers must construct audit inputs from independently selected and
verified assets, and select profiles outside the trace/checkpoint/manifest write
domain. A local digest or a provider's trust flag does not authenticate execution.
Mock adapters can exercise this interface but cannot establish real provenance.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeAlias

TRACE_SCHEMA = "gragra.gen1.refinement-trace.v4"
SUCCESS_EVENT_SCHEMA = "gragra.gen1.refinement-trace-success-event.v4"
FAILURE_EVENT_SCHEMA = "gragra.gen1.refinement-trace-failure-event.v4"
JOURNAL_SCHEMA = "gragra.gen1.refinement-trace-attempt-journal.v4"
CHECKPOINT_SCHEMA = "gragra.gen1.refinement-trace-checkpoint.v4"
MANIFEST_SCHEMA = "gragra.gen1.refinement-trace-run-manifest.v4"
IDENTITY_SCHEMA = "gragra.gen1.refinement-trace-row-identity.v4"
COMMITMENT_SCHEMA = "gragra.gen1.refinement-trace-provenance-commitment.v4"
REPORT_SCHEMA = "gragra.gen1.refinement-trace-report.v4"
ROOT_SUPPLEMENT_SCHEMA = "gragra.gen1.cylinder-root-supplement.v4"
V4_SCHEMA_IDENTITIES = {
    "trace": TRACE_SCHEMA,
    "success_event": SUCCESS_EVENT_SCHEMA,
    "failure_event": FAILURE_EVENT_SCHEMA,
    "journal": JOURNAL_SCHEMA,
    "checkpoint": CHECKPOINT_SCHEMA,
    "manifest": MANIFEST_SCHEMA,
    "identity": IDENTITY_SCHEMA,
    "commitment": COMMITMENT_SCHEMA,
    "report": REPORT_SCHEMA,
    "root_supplement": ROOT_SUPPLEMENT_SCHEMA,
}
REPORT_SOURCE_FIELDS = (
    "observer_sha256",
    "auditor_sha256",
    "provenance_sha256",
    "runner_sha256",
)
ExecutionState: TypeAlias = Literal[
    "not_started", "running", "complete", "failed", "interrupted"
]
SemanticState: TypeAlias = Literal[
    "SEMANTIC_VALID", "SEMANTIC_INVALID", "SEMANTIC_NOT_RUN"
]
ProvenanceState: TypeAlias = Literal[
    "PROVENANCE_VERIFIED",
    "PROVENANCE_UNVERIFIED",
    "PROVENANCE_MISMATCH",
    "PROVENANCE_NOT_RUN",
]
AdmissionState: TypeAlias = Literal[
    "ADMITTED_WITH_DECLARED_SCOPE",
    "NOT_ADMITTED",
    "INCOMPLETE_EVIDENCE",
]
EXECUTION_STATES = frozenset(
    {"not_started", "running", "complete", "failed", "interrupted"}
)
SEMANTIC_STATES = frozenset({"SEMANTIC_VALID", "SEMANTIC_INVALID", "SEMANTIC_NOT_RUN"})
PROVENANCE_STATES = frozenset(
    {
        "PROVENANCE_VERIFIED",
        "PROVENANCE_UNVERIFIED",
        "PROVENANCE_MISMATCH",
        "PROVENANCE_NOT_RUN",
    }
)
ADMISSION_STATES = frozenset(
    {"ADMITTED_WITH_DECLARED_SCOPE", "NOT_ADMITTED", "INCOMPLETE_EVIDENCE"}
)


def _strict_json_value(value: Any) -> None:
    if value is None or type(value) in {bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("canonical JSON requires finite numbers")
        return
    if type(value) is str:
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("canonical JSON requires valid Unicode") from error
        return
    if type(value) is list:
        for item in value:
            _strict_json_value(item)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError("canonical JSON object keys must be strings")
            _strict_json_value(key)
            _strict_json_value(item)
        return
    raise TypeError("canonical JSON accepts only JSON types, without coercion")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON types exactly: UTF-8, sorted keys, no NaN or coercion.

    Ordered lists retain order. High precision values must already use the
    declared string/tuple-object representation; this helper never converts
    them through binary64. Tuples, custom objects and nonstring keys are rejected.
    """
    _strict_json_value(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def strict_json_object(data: str | bytes) -> dict[str, Any]:
    """Load an object without duplicate keys, nonfinite constants or coercion."""

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    if type(data) not in {str, bytes}:
        raise TypeError("JSON input must be text or bytes")
    # Decode explicitly so json.loads cannot silently accept UTF-16/32 bytes.
    if type(data) is bytes:
        data = data.decode("utf-8")
    result = json.loads(data, object_pairs_hook=pairs, parse_constant=reject_constant)
    if type(result) is not dict:
        raise ValueError("JSON document must be an object")
    _strict_json_value(result)
    return result


def _snapshot(value: Any) -> Any:
    return json.loads(canonical_json_bytes(value))


class _JSONSnapshot:
    """Dataclass field retaining only bytes and returning fresh JSON objects.

    Descriptor fields preserve constructor, replace and asdict field names.
    Unlike a mapping proxy, no equality operation can expose retained mutable
    storage: even a hostile comparison operand can retain only a decoded copy.
    """

    def __init__(self, *, optional: bool = False) -> None:
        self.optional = optional

    def __set_name__(self, owner, name: str) -> None:
        self.name = name
        self.storage_name = f"_{name}_json_bytes"

    def __get__(self, instance, owner=None):
        if instance is None:
            if self.optional:
                return None
            # Dataclasses interpret AttributeError as a required field.
            raise AttributeError(self.name)
        retained = getattr(instance, self.storage_name)
        return None if retained is None else strict_json_object(retained)

    def __set__(self, instance, value: Any) -> None:
        if self.optional and value is None:
            retained = None
        else:
            if type(value) is not dict:
                raise ValueError(f"{self.name} must be an object")
            retained = canonical_json_bytes(value)
        object.__setattr__(instance, self.storage_name, retained)


def _text(value: Any, name: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _sha256(value: Any, name: str) -> None:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")


def _exact_keys(value: Any, keys: set[str], name: str) -> None:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{name} has missing or extra fields")


@dataclass(frozen=True)
class IndependentAuditInputs:
    """Row and pin inputs selected by the independent auditor, never a provider.

    ``dataset``, ``dataset_sha256`` and ``source_row_index`` identify the frozen
    dataset asset and the row selected by explicit audit arguments. ``fixture``
    and ``reference`` are that row's parsed values from verified frozen bytes;
    their canonical digests are computed locally. ``required_observables`` comes
    from the independently configured dataset scope. ``oracle_module_sha256``
    and ``kernel_identities`` come from the checked-in pin inventory and pinned
    oracle-to-kernel map after asset-byte verification. ``frozen_reference_sha256``
    and each ``certificate_sha256`` come from verified reference/certificate
    asset bytes, with an empty certificate tuple meaning none is applicable.
    ``root_supplement`` is None outside applicable routes, or the independently
    validated Cylinder supplement's exact schema, asset digest and fixture-scope
    digest. A caller must not fill these fields from trace/manifest claims.

    Constructing this type does not verify asset bytes or scientific scope. That
    remains the independent auditor's responsibility and admission prerequisite.
    JSON objects are retained as immutable canonical bytes. Each field access
    returns a fresh strict JSON copy, including nested objects and arrays;
    dataclasses.replace and asdict preserve the public field names.
    """

    dataset: str
    dataset_sha256: str
    source_row_index: int
    fixture: dict[str, Any] = _JSONSnapshot()
    reference: dict[str, Any] = _JSONSnapshot()
    required_observables: tuple[str, ...]
    oracle_module_sha256: str
    kernel_identities: dict[str, str] = _JSONSnapshot()
    frozen_reference_sha256: str
    certificate_sha256: tuple[str, ...]
    root_supplement: dict[str, str] | None = _JSONSnapshot(optional=True)


@dataclass(frozen=True)
class TrustedRunProfile:
    """Immutable request/configuration authority outside mutable run documents.

    ``run_id`` comes from the explicit new-run request, never the trace or anchor.
    All four V4 source hashes are allowed identities from trusted configuration:
    the recorder, independent auditor, provenance contract, and canonical runner.
    ``environment`` and ``numeric_representation`` are the pre-execution runner
    profile, independently retained for the audit. They must describe the exact
    environment and numerical representation/precision.
    This container does not turn an untrusted trace profile into a trusted one.
    Profiles retain immutable canonical JSON bytes. Each field access and
    identity derivation returns detached ordinary JSON objects and arrays.
    """

    run_id: str
    observer_sha256: str
    auditor_sha256: str
    provenance_sha256: str
    runner_sha256: str
    environment: dict[str, Any] = _JSONSnapshot()
    numeric_representation: dict[str, Any] = _JSONSnapshot()


@dataclass(frozen=True)
class AttemptJournalInputs:
    """Complete journal as independently read/reconciled by the auditor.

    ``events`` is the full ordered durable event history, including failures and
    explicit retry links. ``attempt_states`` includes pending and terminal states;
    ``retry_plan`` retains the explicit retry history; ``accepted_results`` binds
    each accepted attempt/result. None may be reduced to just a successful row.
    The digest binds all four sections, even empty ones. The independent auditor
    must verify completeness against selected row scope and durable history;
    computing this digest alone does not prove completeness or authenticate it.
    """

    events: list[dict[str, Any]]
    attempt_states: dict[str, Any]
    retry_plan: list[dict[str, Any]]
    accepted_results: dict[str, Any]


IDENTITY_FIELDS = frozenset(
    {
        "schema",
        "dataset",
        "dataset_sha256",
        "source_row_index",
        "fixture_sha256",
        "reference_values_sha256",
        "required_observables",
        "oracle_module_sha256",
        "kernel_identities",
        "frozen_reference_sha256",
        "certificate_sha256",
        "root_supplement",
        "run_id",
        "observer_sha256",
        "auditor_sha256",
        "provenance_sha256",
        "runner_sha256",
        "environment",
        "numeric_representation",
        "callback_count",
        "callbacks_sha256",
        "attempt_journal_sha256",
    }
)


def _validate_identity(value: Any) -> None:
    _exact_keys(value, set(IDENTITY_FIELDS), "V4 row identity")
    _strict_json_value(value)
    if value["schema"] != IDENTITY_SCHEMA:
        raise ValueError("wrong V4 row identity schema")
    for field in ("dataset", "run_id"):
        _text(value[field], field)
    for field in IDENTITY_FIELDS:
        if field.endswith("sha256") and field != "certificate_sha256":
            _sha256(value[field], field)
    for field in ("source_row_index", "callback_count"):
        if type(value[field]) is not int or value[field] < 0:
            raise ValueError(f"{field} must be a nonnegative integer")
    observables = value["required_observables"]
    if type(observables) is not list or not observables:
        raise ValueError("required_observables must be a nonempty ordered set")
    for observable in observables:
        _text(observable, "observable")
    if observables != sorted(set(observables)):
        raise ValueError("observable set must be unique and canonical")
    kernels = value["kernel_identities"]
    if type(kernels) is not dict or set(kernels) != set(observables):
        raise ValueError("kernel identities must cover exactly the observable set")
    for pin in kernels.values():
        _sha256(pin, "kernel identity")
    certificates = value["certificate_sha256"]
    if type(certificates) is not list:
        raise ValueError("certificate pins must be a list")
    for pin in certificates:
        _sha256(pin, "certificate pin")
    if certificates != sorted(set(certificates)):
        raise ValueError("certificate pins must be unique and canonical")
    for field in ("environment", "numeric_representation"):
        if type(value[field]) is not dict or not value[field]:
            raise ValueError(f"{field} must be a nonempty profile object")
    root = value["root_supplement"]
    if root is not None:
        _exact_keys(
            root, {"schema", "sha256", "scope_sha256"}, "root supplement identity"
        )
        if root["schema"] != ROOT_SUPPLEMENT_SCHEMA:
            raise ValueError("wrong V4 root supplement schema")
        _sha256(root["sha256"], "root supplement digest")
        _sha256(root["scope_sha256"], "root supplement scope digest")


def derive_expected_row_identity(
    audit_inputs: IndependentAuditInputs,
    run_profile: TrustedRunProfile | None,
    *,
    callbacks: list[dict[str, Any]],
    attempt_journal: AttemptJournalInputs,
) -> dict[str, Any] | None:
    """Compute the expected binding independently of any external anchor.

    Row/pin authority is documented on ``IndependentAuditInputs``; run/source
    authority is on ``TrustedRunProfile``. Ordered ``callbacks`` are independently
    read from saved callback evidence and committed exactly, including inputs and
    outputs. Journal authority is on ``AttemptJournalInputs``. Mutable trace,
    checkpoint and manifest claims never supply their own expected identity.
    Missing trusted run configuration returns None, so provenance stays unverified.
    The returned deep snapshot cannot change when the input containers change.
    """
    if run_profile is None:
        return None
    if not isinstance(audit_inputs, IndependentAuditInputs):
        raise TypeError("independent audit inputs are required")
    if not isinstance(run_profile, TrustedRunProfile):
        raise TypeError("trusted run profile is required")
    if not isinstance(attempt_journal, AttemptJournalInputs):
        raise TypeError("complete independent attempt journal inputs are required")
    if type(callbacks) is not list or any(type(call) is not dict for call in callbacks):
        raise ValueError("callbacks must be an ordered list of objects")
    if type(audit_inputs.required_observables) is not tuple:
        raise TypeError("required observables must be a tuple")
    if type(audit_inputs.certificate_sha256) is not tuple:
        raise TypeError("certificate pins must be a tuple")
    journal = {
        field: getattr(attempt_journal, field)
        for field in ("events", "attempt_states", "retry_plan", "accepted_results")
    }
    for field in ("events", "retry_plan"):
        if type(journal[field]) is not list or any(
            type(event) is not dict for event in journal[field]
        ):
            raise ValueError(f"journal {field} must be an ordered list of objects")
    for field in ("attempt_states", "accepted_results"):
        if type(journal[field]) is not dict:
            raise ValueError(f"journal {field} must be an object")
    result = {
        "schema": IDENTITY_SCHEMA,
        "dataset": audit_inputs.dataset,
        "dataset_sha256": audit_inputs.dataset_sha256,
        "source_row_index": audit_inputs.source_row_index,
        "fixture_sha256": canonical_sha256(audit_inputs.fixture),
        "reference_values_sha256": canonical_sha256(audit_inputs.reference),
        "required_observables": sorted(audit_inputs.required_observables),
        "oracle_module_sha256": audit_inputs.oracle_module_sha256,
        "kernel_identities": audit_inputs.kernel_identities,
        "frozen_reference_sha256": audit_inputs.frozen_reference_sha256,
        "certificate_sha256": sorted(audit_inputs.certificate_sha256),
        "root_supplement": audit_inputs.root_supplement,
        "run_id": run_profile.run_id,
        "observer_sha256": run_profile.observer_sha256,
        "auditor_sha256": run_profile.auditor_sha256,
        "provenance_sha256": run_profile.provenance_sha256,
        "runner_sha256": run_profile.runner_sha256,
        "environment": run_profile.environment,
        "numeric_representation": run_profile.numeric_representation,
        "callback_count": len(callbacks),
        "callbacks_sha256": canonical_sha256(callbacks),
        "attempt_journal_sha256": canonical_sha256(journal),
    }
    _validate_identity(result)
    return _snapshot(result)


def commitment_document(payload: dict[str, Any]) -> dict[str, Any]:
    """Construct a strict V4 candidate commitment; this is not authentication."""
    _validate_identity(payload)
    body = {"schema": COMMITMENT_SCHEMA, "payload": _snapshot(payload)}
    return {**body, "commitment_sha256": canonical_sha256(body)}


def _validated_commitment(document: Any) -> dict[str, Any]:
    _exact_keys(document, {"schema", "payload", "commitment_sha256"}, "V4 commitment")
    if document["schema"] != COMMITMENT_SCHEMA:
        raise ValueError("wrong V4 commitment schema")
    _validate_identity(document["payload"])
    _sha256(document["commitment_sha256"], "commitment digest")
    body = {"schema": document["schema"], "payload": document["payload"]}
    if document["commitment_sha256"] != canonical_sha256(body):
        raise ValueError("V4 commitment digest differs")
    return _snapshot(document)


class ExternalAnchorProvider(Protocol):
    """Retrieval only. Provider documents, status and root flags are untrusted."""

    def fetch_anchor(self, lookup_key: str) -> Any:
        """Fetch raw external material; expected identity is never passed here."""


@dataclass(frozen=True)
class AuthenticatedAnchor:
    """Material authenticated by a configured trusted verifier adapter.

    Only the adapter's return value is consumed. A dictionary carrying an
    equivalent boolean/status is not accepted. ``trust_root_id`` must equal the
    root selected independently in ``TrustedVerifierProfile``. This type is an
    adapter interface, not cryptographic proof by itself.
    """

    document: dict[str, Any]
    anchor_id: str
    trust_root_id: str


class ExternalAnchorVerifier(Protocol):
    """Trusted adapter configured outside all mutable evidence write domains."""

    def authenticate_anchor(
        self,
        raw_anchor: Any,
        *,
        trust_root_id: str,
    ) -> AuthenticatedAnchor | None:
        """Authenticate issuer/signature/root and return signed material or None."""


@dataclass(frozen=True)
class TrustedVerifierProfile:
    """Trusted configuration selects the adapter and independently pinned root.

    Neither field may be selected from provider/trace/checkpoint/manifest data.
    No default profile or real provider exists in this module.
    """

    trust_root_id: str
    verifier: ExternalAnchorVerifier


@dataclass(frozen=True)
class ProvenanceResult:
    status: ProvenanceState
    reason: str
    anchor_id: str | None = None


def verify_external_anchor(
    expected_document: dict[str, Any] | None,
    *,
    provider: ExternalAnchorProvider | None,
    verifier_profile: TrustedVerifierProfile | None,
    lookup_key: str,
) -> ProvenanceResult:
    """Authenticate first, then compare every canonical independently expected field.

    Missing provider/profile/expected run identity and adapter errors are
    unverified. An authenticated malformed or different document is a mismatch.
    A provider's VERIFIED/root flag is never used. The lookup key is only a
    retrieval hint; the complete signed payload must match the audit identity.
    """

    def unverified(reason):
        return ProvenanceResult("PROVENANCE_UNVERIFIED", reason)

    if expected_document is None or provider is None or verifier_profile is None:
        return unverified(
            "no independently configured anchor provider, verifier or run profile"
        )
    if not isinstance(verifier_profile, TrustedVerifierProfile):
        return unverified("trusted verifier profile is unavailable")
    try:
        expected = _validated_commitment(expected_document)
        _text(lookup_key, "anchor lookup key")
        _text(verifier_profile.trust_root_id, "configured trust root")
        raw = provider.fetch_anchor(lookup_key)
        anchor = verifier_profile.verifier.authenticate_anchor(
            raw,
            trust_root_id=verifier_profile.trust_root_id,
        )
    except Exception as error:
        return unverified(
            f"anchor retrieval/authentication unavailable: {type(error).__name__}"
        )
    if not isinstance(anchor, AuthenticatedAnchor):
        return unverified(
            "configured verifier did not return authenticated anchor material"
        )
    if anchor.trust_root_id != verifier_profile.trust_root_id:
        return unverified(
            "verifier material is not authenticated under the configured root"
        )
    if type(anchor.anchor_id) is not str or not anchor.anchor_id.strip():
        return unverified("verifier returned no authenticated anchor identity")
    try:
        actual = _validated_commitment(anchor.document)
    except (TypeError, ValueError, RecursionError):
        return ProvenanceResult(
            "PROVENANCE_MISMATCH",
            "authenticated V4 anchor is malformed",
            anchor.anchor_id,
        )
    if canonical_json_bytes(actual) != canonical_json_bytes(expected):
        return ProvenanceResult(
            "PROVENANCE_MISMATCH",
            "authenticated identity differs from independent audit inputs",
            anchor.anchor_id,
        )
    return ProvenanceResult(
        "PROVENANCE_VERIFIED",
        "authenticated exact V4 anchor matches; adapter trust is externally configured",
        anchor.anchor_id,
    )


CHECK_FIELDS = frozenset(
    {
        "required_evidence_complete",
        "independent_audit_valid",
        "certificate_validation_valid",
        "trend_validation_complete",
        "declared_scope_valid",
        "journal_binding_complete",
        "prior_failures",
        "accepted_retry_result",
        "accepted_result_anchored",
    }
)


def admission_decision(
    *,
    execution_state: ExecutionState,
    semantic_status: SemanticState,
    provenance_status: ProvenanceState,
    required_evidence_complete: bool,
    independent_audit_valid: bool | None,
    certificate_validation_valid: bool | None,
    trend_validation_complete: bool,
    declared_scope_valid: bool | None,
    journal_binding_complete: bool,
    prior_failures: bool,
    accepted_retry_result: bool,
    accepted_result_anchored: bool,
) -> AdmissionState:
    """Derive admission from independent validated states and required checks.

    Inputs are results computed by the auditor/runner, never self-asserted trace
    fields. Provenance VERIFIED must come from ``verify_external_anchor``. A
    successful explicit retry is represented by a complete execution, retained
    prior failures, one accepted retry result and full-history/result anchoring.
    A still-failed execution never becomes complete merely by setting a retry
    flag. None means a validity check was not performed; False means invalid.
    Completion/anchoring flags False mean missing evidence, not validation.
    """
    if type(execution_state) is not str or execution_state not in EXECUTION_STATES:
        return "NOT_ADMITTED"
    if type(semantic_status) is not str or semantic_status not in SEMANTIC_STATES:
        return "NOT_ADMITTED"
    if type(provenance_status) is not str or provenance_status not in PROVENANCE_STATES:
        return "NOT_ADMITTED"
    flags = (
        required_evidence_complete,
        trend_validation_complete,
        journal_binding_complete,
        prior_failures,
        accepted_retry_result,
        accepted_result_anchored,
    )
    validity = (
        independent_audit_valid,
        certificate_validation_valid,
        declared_scope_valid,
    )
    if any(type(flag) is not bool for flag in flags):
        return "NOT_ADMITTED"
    if any(flag is not None and type(flag) is not bool for flag in validity):
        return "NOT_ADMITTED"
    if (
        semantic_status == "SEMANTIC_INVALID"
        or provenance_status == "PROVENANCE_MISMATCH"
    ):
        return "NOT_ADMITTED"
    if execution_state == "failed":
        return "NOT_ADMITTED"
    if any(flag is False for flag in validity):
        return "NOT_ADMITTED"
    if (
        execution_state in {"not_started", "running", "interrupted"}
        or semantic_status == "SEMANTIC_NOT_RUN"
    ):
        return "INCOMPLETE_EVIDENCE"
    if prior_failures and not accepted_retry_result:
        return "NOT_ADMITTED"
    if (
        not required_evidence_complete
        or not trend_validation_complete
        or not journal_binding_complete
        or not accepted_result_anchored
        or any(flag is None for flag in validity)
    ):
        return "INCOMPLETE_EVIDENCE"
    if provenance_status != "PROVENANCE_VERIFIED":
        return "NOT_ADMITTED"
    return "ADMITTED_WITH_DECLARED_SCOPE"


def validate_v4_report(
    report: dict[str, Any],
    *,
    expected_source_identity: dict[str, str],
) -> dict[str, Any]:
    """Validate the structural V4 report gate against external allowed source pins.

    ``expected_source_identity`` comes from trusted source inventory verified by
    the reader, never this report. Legacy full_gate_complete booleans have no
    meaning here. Every schema/source/check field is exact and admission must
    equal the central decision. Passing this gate establishes only report shape
    and consistency; it does not authenticate a reported provenance status or
    replace the independent audit and external-anchor verification.
    """
    if type(report) is not dict or report.get("schema") != REPORT_SCHEMA:
        raise ValueError("only the exact V4 report schema is accepted")
    _exact_keys(
        report,
        {
            "schema",
            "source_identity",
            "schema_identity",
            "execution_state",
            "semantic_status",
            "provenance_status",
            "checks",
            "admission",
        },
        "V4 report",
    )
    _strict_json_value(report)
    for sources in (expected_source_identity, report["source_identity"]):
        _exact_keys(sources, set(REPORT_SOURCE_FIELDS), "V4 source identity")
        for field, pin in sources.items():
            _sha256(pin, field)
    if report["source_identity"] != expected_source_identity:
        raise ValueError(
            "V4 source identity does not match the trusted source inventory"
        )
    if report["schema_identity"] != V4_SCHEMA_IDENTITIES:
        raise ValueError("V4 schema identities differ")
    _exact_keys(report["checks"], set(CHECK_FIELDS), "V4 report checks")
    states = (
        ("execution_state", EXECUTION_STATES),
        ("semantic_status", SEMANTIC_STATES),
        ("provenance_status", PROVENANCE_STATES),
        ("admission", ADMISSION_STATES),
    )
    for field, allowed in states:
        if type(report[field]) is not str or report[field] not in allowed:
            raise ValueError(f"unknown V4 {field}")
    for field, flag in report["checks"].items():
        optional = field in {
            "independent_audit_valid",
            "certificate_validation_valid",
            "declared_scope_valid",
        }
        if type(flag) is not bool and not (optional and flag is None):
            raise ValueError(f"invalid V4 check type: {field}")
    expected_admission = admission_decision(
        execution_state=report["execution_state"],
        semantic_status=report["semantic_status"],
        provenance_status=report["provenance_status"],
        **report["checks"],
    )
    if report["admission"] != expected_admission:
        raise ValueError("V4 admission does not match the central contract")
    return _snapshot(report)
