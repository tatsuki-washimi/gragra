"""Selected final-family verification; no execution, Git spawning or P imports."""

from __future__ import annotations

from .archive import EvidenceError
from .arithmetic import GENERIC_FAMILIES, replay_generic_arithmetic
from .closure import compare_closures
from .graph import ROLES, verify_authorship_attestation, verify_comparison
from .source import verify_catalog, verify_closure_retrieval, verify_git_catalog_origin

FAMILIES = ("B1", "R1", "B2", "A1", "A3", "T1", "A2")


def verify_selected_comparison(store, descriptor, *, family):
    """Dispatch exact reviewed profiles, then replay only retained arithmetic."""
    if family == "A2":
        from .a2 import verify_a2_comparison

        return verify_a2_comparison(store, descriptor)
    if family == "B1":
        from .b1 import verify_b1_comparison

        return verify_b1_comparison(store, descriptor)
    if family not in GENERIC_FAMILIES:
        raise EvidenceError(
            "selected family profile pending exact reviewed implementation"
        )
    results = verify_comparison(store, descriptor, family=family)
    summary = replay_generic_arithmetic(store, descriptor["stages"][-1], family=family)
    # The arithmetic helper deliberately cannot make an admission on its own.
    # The preceding exact stage reader has now verified actual issued admission.
    summary["independent_result_admission_verified"] = True
    results[-1]["summary"] = summary
    return results


def verify_transfer(store, descriptor, *, actual_p, actual_closure):
    fields = {
        "source_closure",
        "target_closure",
        "source_catalog",
        "target_catalog",
        "relocation",
        "reviews",
        "original_git_proof",
        "authorship",
    }
    if not isinstance(descriptor, dict) or set(descriptor) != fields:
        raise EvidenceError("complete typed transfer descriptor required")
    if descriptor["source_closure"] != actual_closure:
        raise EvidenceError("transfer closure is not exact accepted comparison closure")
    source, target = [
        store.json(descriptor[name]) for name in ("source_closure", "target_closure")
    ]
    if {
        "commit": source["production_commit"],
        "tree": source["production_tree"],
    } != actual_p:
        raise EvidenceError("transfer substituted immutable actual comparison P")
    result = compare_closures(source, target, descriptor["relocation"])
    omissions = verify_closure_retrieval(store, source)
    target_omissions = verify_closure_retrieval(store, target)
    if omissions != target_omissions:
        raise EvidenceError("retrieval missing provenance changed")
    catalogs = [
        store.json(descriptor[name]) for name in ("source_catalog", "target_catalog")
    ]
    for catalog, closure in zip(catalogs, (source, target), strict=True):
        if (
            catalog["commit"] != closure["production_commit"]
            or catalog["tree"] != closure["production_tree"]
        ):
            raise EvidenceError("source catalog immutable revision differs")
    if verify_catalog(store, catalogs[0]) != verify_catalog(store, catalogs[1]):
        raise EvidenceError("complete gragra Git source bytes changed")
    subjects = {
        name: descriptor[name]
        for name in fields - {"relocation", "reviews", "authorship"}
    }
    authorship = store.json(descriptor["authorship"])
    authors = verify_authorship_attestation(
        store,
        authorship,
        schema="gen1-stochastic-transfer-authorship-v1",
        original_schema="gen1-root-transfer-authorship-attestation-v1",
        bindings={"subjects": subjects},
    )
    subjects["authorship"] = descriptor["authorship"]
    identities = set()
    if set(descriptor["reviews"]) != ROLES:
        raise EvidenceError("three independent complete transfer reviews required")
    for role, pin in descriptor["reviews"].items():
        review = store.json(pin)
        if (
            set(review)
            != {
                "schema",
                "role",
                "reviewer_id",
                "decision",
                "subjects",
                "relocation",
                "original_review",
            }
            or review["schema"] != "gen1-stochastic-closure-transfer-review-v1"
            or review["role"] != role
            or review["decision"] != "PASS"
            or review["subjects"] != subjects
            or review["relocation"] != descriptor["relocation"]
            or not isinstance(review["reviewer_id"], str)
            or not review["reviewer_id"]
            or review["reviewer_id"] in identities
            or review["reviewer_id"] in authors
        ):
            raise EvidenceError("exact complete-payload transfer review required")
        identities.add(review["reviewer_id"])
        store.read(review["original_review"])
    verify_git_catalog_origin(
        store, store.json(descriptor["original_git_proof"]), catalogs
    )
    return {
        **result,
        "original_reviewed_package_missing_reasons": omissions,
        "archived_bytes_verified": True,
        "original_execution_repeated": False,
    }


def verify_selected_delivery(store, descriptor):
    """Reject pending/unknown profiles instead of upgrading them to acceptance."""
    if (
        set(descriptor) != {"schema", "families", "history", "final_identity_boundary"}
        or descriptor["schema"] != "gen1-stochastic-selected-delivery-v1"
    ):
        raise EvidenceError("selected delivery descriptor schema mismatch")
    if (
        descriptor["final_identity_boundary"]
        != "EXTERNAL_ATTESTATION_AFTER_TRACKED_COMMIT"
    ):
        raise EvidenceError("H_final/self-hash cannot be embedded in tracked payload")
    if [family["family"] for family in descriptor["families"]] != list(FAMILIES):
        raise EvidenceError("exactly seven ordered selected families required")
    for historical in descriptor["history"]:
        store.read(historical)
    results = []
    for family in descriptor["families"]:
        if set(family) != {"family", "comparison", "transfer"}:
            raise EvidenceError("unknown selected family field")
        for stage in family["comparison"]["stages"]:
            policy = store.json(stage["policy"])
            if family["family"] != "B1":
                required = {
                    "budget",
                    "decision_kernel",
                    "dependency_closure",
                    "environment",
                    "family_procedure",
                    "fixture",
                    "freeze",
                    "procedure",
                    "production_source_admission",
                    "purpose",
                    "records",
                    "reference_admission",
                    "seal",
                    "source_scales",
                }
                if set(policy["artifacts"]) != required:
                    raise EvidenceError(
                        "complete admitted generic input domains required"
                    )
        comparisons = verify_selected_comparison(
            store, family["comparison"], family=family["family"]
        )
        pins = comparisons[-1]["pins"]
        actual_p = {
            "commit": pins["production_commit"],
            "tree": pins["production_tree"],
        }
        final_policy = store.json(family["comparison"]["stages"][-1]["policy"])
        actual_closure = final_policy["artifacts"]["dependency_closure"]
        if actual_closure["sha256"] != pins["dependency_closure_sha256"]:
            raise EvidenceError("actual comparison closure raw binding differs")
        transfer = verify_transfer(
            store, family["transfer"], actual_p=actual_p, actual_closure=actual_closure
        )
        results.append(
            {
                "family": family["family"],
                "actual_comparison_P": actual_p,
                "comparison_summary": comparisons[-1]["summary"],
                "phase_summaries": [
                    {"phase": result["phase"], "summary": result["summary"]}
                    for result in comparisons
                ],
                "transfer": transfer,
            }
        )
    return {
        "status": "SELECTED_BYTE_GRAPH_VERIFIED",
        "families": results,
        "history_in_accepted_counts": False,
        "H_final_assigned": False,
        "production_api_calls": 0,
    }


def verify_selected_delivery_v2(store, descriptor):
    """Verify the additive six-family-plus-A2-closure profile explicitly."""
    from .v2 import verify_selected_delivery_v2 as verify

    return verify(store, descriptor)
