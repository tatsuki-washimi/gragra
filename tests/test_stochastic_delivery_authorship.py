"""Finite synthetic legacy author declarations; no original evidence mutations."""

import pytest

from tests.oracles.gen1.stochastic_delivery.archive import EvidenceError
from tests.oracles.gen1.stochastic_delivery.graph import verify_input_admission
from tests.test_stochastic_delivery_evidence import input_admission_fixture


def legacy_fixture():
    store, admission_pin, known_pin = input_admission_fixture(source=True)
    admission = store.json(admission_pin)
    wrapper = store.json(known_pin)
    wrapper.update(
        schema="gen1-stochastic-input-authorship-legacy-v1",
        authors=[],
        historical_author_status="UNKNOWN_PRESESSION_OWNERSHIP_RETAINED",
        reviewer_non_authorship={},
    )
    for role, pin in admission["reviews"].items():
        review = store.json(pin)
        wrapper["reviewer_non_authorship"][role] = store.add(
            {
                "schema": "gen1-stochastic-input-nonauthorship-review-v1",
                "family": "R1",
                "subject_kind": "production_source",
                "role": role,
                "reviewer_id": review["reviewer_id"],
                "admission": admission_pin,
                "pins": admission["pins"],
                "declaration": "DID_NOT_AUTHOR_REVIEWED_IMPLEMENTATION",
                "original_review": review["original_review"],
                "issued_at": "2026-10-04T13:00:00Z",
            }
        )
    original = {
        key: value for key, value in wrapper.items() if key != "original_attestation"
    }
    original.update(
        schema="gen1-root-input-authorship-legacy-attestation-v1",
        issuer_id="actual-controller",
        issued_at="2026-10-04T13:01:00Z",
    )
    wrapper["original_attestation"] = store.add(original)
    return store, admission_pin, wrapper


def test_unknown_history_is_retained_without_inventing_an_author():
    store, admission, wrapper = legacy_fixture()
    assert wrapper["authors"] == []
    result = verify_input_admission(
        store,
        admission,
        family="R1",
        source=True,
        expected_bindings={},
        authorship_pin=store.add(wrapper),
    )
    assert result == store.json(admission)["pins"]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_role",
        "unknown_erased",
        "different_subject",
        "different_original",
        "different_reviewer",
        "different_role",
        "not_declared",
        "issuer_review",
        "known_author",
    ],
)
def test_legacy_exception_rejects_absent_or_stale_actual_declarations(mutation):
    store, admission, wrapper = legacy_fixture()
    declaration = store.json(wrapper["reviewer_non_authorship"]["science"])
    if mutation == "missing_role":
        del wrapper["reviewer_non_authorship"]["provenance"]
    elif mutation == "unknown_erased":
        wrapper["historical_author_status"] = "NO_AUTHORS"
    elif mutation == "different_subject":
        declaration["pins"] = {}
    elif mutation == "different_original":
        declaration["original_review"] = store.add({"unrelated": True})
    elif mutation == "different_reviewer":
        declaration["reviewer_id"] = "new-reviewer"
    elif mutation == "different_role":
        declaration["role"] = "numerical"
    elif mutation == "not_declared":
        declaration["declaration"] = "UNKNOWN"
    elif mutation == "issuer_review":
        original = store.json(wrapper["original_attestation"])
        original["issuer_id"] = "science"
        wrapper["original_attestation"] = store.add(original)
    elif mutation == "known_author":
        wrapper["authors"] = [
            {
                "reviewer_id": "science",
                "source_artifacts": [store.add(b"own source", raw=True)],
            }
        ]
    if mutation not in {
        "missing_role",
        "unknown_erased",
        "issuer_review",
        "known_author",
    }:
        wrapper["reviewer_non_authorship"]["science"] = store.add(declaration)
    # Even a newly issued attestation cannot permit false or stale declarations.
    if mutation not in {"issuer_review"}:
        original = store.json(wrapper["original_attestation"])
        original.update(
            {
                name: wrapper[name]
                for name in (
                    "authors",
                    "historical_author_status",
                    "reviewer_non_authorship",
                )
            }
        )
        wrapper["original_attestation"] = store.add(original)
    with pytest.raises(EvidenceError):
        verify_input_admission(
            store,
            admission,
            family="R1",
            source=True,
            expected_bindings={},
            authorship_pin=store.add(wrapper),
        )


@pytest.mark.parametrize(
    "family,source", [("A1", True), ("B1", True), ("A2", True), ("R1", False)]
)
def test_exception_does_not_relax_other_family_or_reference_provenance(family, source):
    from tests.oracles.gen1.stochastic_delivery.authorship import (
        verify_legacy_input_authorship,
    )

    store, admission, wrapper = legacy_fixture()
    with pytest.raises(EvidenceError, match="R1 production only"):
        verify_legacy_input_authorship(
            store,
            wrapper,
            family=family,
            source=source,
            admission_pin=admission,
            admission=store.json(admission),
        )
