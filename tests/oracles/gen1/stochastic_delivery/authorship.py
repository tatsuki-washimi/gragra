"""Finite legacy input provenance exception; preserve unknown historical owners."""

from .archive import EvidenceError
from .graph import ROLES, _fields


def verify_legacy_input_authorship(
    store, wrapper, *, family, source, admission_pin, admission
):
    """R1 old production source only: actual reviewers declare non-authorship.

    Complete known-author inputs retain the ordinary reviewed roster profile.
    This does not infer an identity from Git email or certify unknown history.
    """
    if family != "R1" or not source:
        raise EvidenceError("legacy unknown-author exception is R1 production only")
    bindings = {
        "family": family,
        "subject_kind": "production_source",
        "admission": admission_pin,
        "pins": admission["pins"],
    }
    fields = {
        "schema",
        "authors",
        "original_attestation",
        "historical_author_status",
        "reviewer_non_authorship",
    } | set(bindings)
    _fields(wrapper, fields, "legacy input authorship")
    if (
        wrapper["schema"] != "gen1-stochastic-input-authorship-legacy-v1"
        or any(wrapper[name] != value for name, value in bindings.items())
        or wrapper["historical_author_status"]
        != "UNKNOWN_PRESESSION_OWNERSHIP_RETAINED"
        or not isinstance(wrapper["authors"], list)
        or set(wrapper["reviewer_non_authorship"]) != ROLES
    ):
        raise EvidenceError("legacy unknown-history subject binding differs")
    authors = set()
    for author in wrapper["authors"]:
        _fields(author, {"reviewer_id", "source_artifacts"}, "known session author")
        identity = author["reviewer_id"]
        if (
            not isinstance(identity, str)
            or not identity
            or identity in authors
            or not author["source_artifacts"]
        ):
            raise EvidenceError(
                "finite known session author/source identities required"
            )
        authors.add(identity)
        for pin in author["source_artifacts"]:
            store.read(pin)
    original = store.json(wrapper["original_attestation"])
    _fields(
        original,
        fields - {"original_attestation"} | {"issuer_id", "issued_at"},
        "issued legacy authorship",
    )
    if (
        original["schema"] != "gen1-root-input-authorship-legacy-attestation-v1"
        or any(
            original[name] != wrapper[name]
            for name in fields - {"schema", "original_attestation"}
        )
        or not isinstance(original["issuer_id"], str)
        or not original["issuer_id"]
        or original["issuer_id"] in authors
        or not isinstance(original["issued_at"], str)
        or not original["issued_at"]
    ):
        raise EvidenceError("issued legacy original subjects/roster differ")
    identities = set()
    for role, pin in wrapper["reviewer_non_authorship"].items():
        declaration = store.json(pin)
        _fields(
            declaration,
            {
                "schema",
                "family",
                "subject_kind",
                "role",
                "reviewer_id",
                "admission",
                "pins",
                "declaration",
                "original_review",
                "issued_at",
            },
            "reviewer non-authorship declaration",
        )
        review = store.json(admission["reviews"][role])
        identity = declaration["reviewer_id"]
        if (
            declaration["schema"] != "gen1-stochastic-input-nonauthorship-review-v1"
            or declaration["family"] != family
            or declaration["subject_kind"] != "production_source"
            or declaration["role"] != role
            or declaration["admission"] != admission_pin
            or declaration["pins"] != admission["pins"]
            or declaration["declaration"] != "DID_NOT_AUTHOR_REVIEWED_IMPLEMENTATION"
            or identity != review["reviewer_id"]
            or declaration["original_review"] != review["original_review"]
            or not isinstance(identity, str)
            or not identity
            or identity in authors
            or identity in identities
            or not isinstance(declaration["issued_at"], str)
            or not declaration["issued_at"]
        ):
            raise EvidenceError("actual reviewer exact-subject non-authorship required")
        store.read(declaration["original_review"])
        identities.add(identity)
    if original["issuer_id"] in identities:
        raise EvidenceError(
            "root issuer cannot manufacture its own reviewer declaration"
        )
    return authors
