"""Synthetic byte graphs only; no scientific API calls or original evidence edits."""

import copy
import hashlib
import json

import pytest

from tests.oracles.gen1.stochastic_delivery.archive import (
    EvidenceError,
    EvidenceStore,
    archive_selected,
    materialize_checkout,
)
from tests.oracles.gen1.stochastic_delivery.closure import compare_closures
from tests.oracles.gen1.stochastic_delivery.graph import (
    assessment_ratios,
    verify_authorship_attestation,
    verify_comparison,
    verify_input_admission,
    verify_stage,
)
from tests.oracles.gen1.stochastic_delivery.reader import verify_selected_delivery
from tests.oracles.gen1.stochastic_delivery.source import git_oid, verify_catalog


def pin(path, kind="test"):
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "kind": kind,
    }


def test_archive_preserves_raw_bytes_original_identity_and_graph_role(tmp_path):
    original = tmp_path / "original.json"
    raw = b'{ "value" : 1 }\n'
    original.write_bytes(raw)
    source = pin(original)
    archive = tmp_path / "archive"
    mapping = archive_selected(
        [{"original": source, "role": "reference_seal"}], archive
    )
    original.unlink()
    store = EvidenceStore(archive, mapping)
    assert store.read(source, role="reference_seal") == raw
    assert mapping["entries"][0]["original"] == source
    assert (
        mapping["entries"][0]["archive_relative_path"] == "sha256/" + source["sha256"]
    )
    assert store.json(source, role="reference_seal") == {"value": 1}


def test_archive_missing_mutated_or_wrong_role_fails_closed(tmp_path):
    original = tmp_path / "original"
    original.write_bytes(b"original")
    source = pin(original)
    root = tmp_path / "archive"
    mapping = archive_selected([{"original": source, "role": "reference"}], root)
    store = EvidenceStore(root, mapping)
    with pytest.raises(EvidenceError, match="role"):
        store.read(source, role="comparison")
    (root / "sha256" / source["sha256"]).chmod(0o644)
    (root / "sha256" / source["sha256"]).write_bytes(b"counterfeit")
    with pytest.raises(EvidenceError, match="hash|writable"):
        store.read(source)
    with pytest.raises(EvidenceError, match="missing"):
        store.read({**source, "path": str(tmp_path / "unlisted")})


def test_archive_rejects_symlinks_path_traversal_and_duplicate_mapping(tmp_path):
    original = tmp_path / "original"
    original.write_bytes(b"x")
    alias = tmp_path / "alias"
    alias.symlink_to(original)
    with pytest.raises(EvidenceError, match="symlink"):
        archive_selected([{"original": pin(alias), "role": "source"}], tmp_path / "a")
    mapping = archive_selected(
        [{"original": pin(original), "role": "source"}], tmp_path / "b"
    )
    bad = copy.deepcopy(mapping)
    bad["entries"].append(copy.deepcopy(bad["entries"][0]))
    with pytest.raises(EvidenceError, match="duplicate"):
        EvidenceStore(tmp_path / "b", bad)
    bad = copy.deepcopy(mapping)
    bad["entries"][0]["archive_relative_path"] = "../original"
    with pytest.raises(EvidenceError, match="CAS"):
        EvidenceStore(tmp_path / "b", bad)


def test_git_checkout_requires_exact_readonly_external_materialization(tmp_path):
    original = tmp_path / "original"
    original.write_bytes(b'{ "raw": "unchanged" }\n')
    source = pin(original)
    checkout = tmp_path / "checkout"
    mapping = archive_selected([{"original": source, "role": "result"}], checkout)
    stored = checkout / "sha256" / source["sha256"]
    stored.chmod(0o644)  # A fresh Git checkout cannot retain 0444.
    original.unlink()  # Proof must not require the original host path.
    with pytest.raises(EvidenceError, match="writable"):
        EvidenceStore(checkout, mapping).read(source)
    destination = tmp_path / "external"
    result = materialize_checkout(checkout, mapping, destination)
    assert result["status"] == "EXACT_BYTES_READONLY_MATERIALIZED"
    assert result["original_execution_repeated"] is False
    assert result["objects"] == 1
    assert EvidenceStore(destination, mapping).read(source) == stored.read_bytes()
    assert stored.stat().st_mode & 0o222
    assert not (destination / "sha256" / source["sha256"]).stat().st_mode & 0o222
    with pytest.raises(EvidenceError, match="fresh"):
        materialize_checkout(checkout, mapping, destination)


def test_materialization_refuses_counterfeit_alias_and_git_destination(tmp_path):
    original = tmp_path / "original"
    original.write_bytes(b"bytes")
    source = pin(original)
    checkout = tmp_path / "checkout"
    mapping = archive_selected([{"original": source, "role": "result"}], checkout)
    git_root = tmp_path / "repository"
    git_root.mkdir()
    (git_root / ".git").write_text("gitdir: elsewhere")
    with pytest.raises(EvidenceError, match="outside Git"):
        materialize_checkout(checkout, mapping, git_root / "archive")
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(EvidenceError, match="aliased"):
        materialize_checkout(checkout, mapping, alias / "archive")
    stored = checkout / "sha256" / source["sha256"]
    stored.chmod(0o644)
    stored.write_bytes(b"counterfeit")
    destination = tmp_path / "external"
    with pytest.raises(EvidenceError, match="hash"):
        materialize_checkout(checkout, mapping, destination)
    assert not destination.exists()


def closure(root):
    artifact = {
        "path": root + "/python",
        "sha256": "1" * 64,
        "kind": "python_executable",
    }
    return {
        "schema": "gen1-stochastic-dependencies-v1",
        "role": "production",
        "production_commit": "a" * 40,
        "production_tree": "b" * 40,
        "protected_packages": ["gragra"],
        "modules": [
            {
                "name": "gragra.unloaded",
                "package": "gragra",
                "path": root + "/src/gragra/unloaded.py",
                "root": root,
                "relative_path": "src/gragra/unloaded.py",
                "git_blob": "c" * 40,
                "git_blob_missing_reason": None,
                "sha256": "2" * 64,
                "kind": "python",
            }
        ],
        "import_edges": [{"from": "gragra.unloaded", "to": "math"}],
        "packages": [
            {
                "name": "package",
                "version": "1",
                "location": root + "/site",
                "artifact": None,
                "artifact_missing_reason": "reviewed original package has no RECORD",
            }
        ],
        "native_backends": [
            {"path": root + "/blas.so", "sha256": "3" * 64, "kind": "native_backend"}
        ],
        "artifacts": [
            {
                "path": root + "/config",
                "sha256": "4" * 64,
                "kind": "import_configuration",
            }
        ],
        "python": {
            "implementation": "CPython",
            "version": "3.11",
            "executable": artifact,
        },
    }


def test_complete_typed_transfer_accepts_only_explicit_benign_root_relocation():
    source, target = closure("/original"), closure("/relocated")
    target.update(production_commit="d" * 40, production_tree="e" * 40)
    relocation = [{"from": "/original", "to": "/relocated"}]
    result = compare_closures(source, target, relocation)
    assert result["status"] == "MATCH_COMPLETE_TYPED_PAYLOAD"
    assert result["source_identity"]["production_commit"] == "a" * 40
    assert result["target_identity"]["production_commit"] == "d" * 40
    with pytest.raises(EvidenceError, match="different"):
        compare_closures(source, target, [])


@pytest.mark.parametrize(
    "domain",
    [
        "modules",
        "import_edges",
        "packages",
        "native_backends",
        "artifacts",
        "protected_packages",
    ],
)
def test_subset_closure_never_proves_equivalence(domain):
    source = closure("/source")
    target = copy.deepcopy(source)
    target[domain] = []
    with pytest.raises(EvidenceError):
        compare_closures(source, target, [])


@pytest.mark.parametrize(
    "field,value",
    [
        ("sha256", "f" * 64),
        ("git_blob", "f" * 40),
        ("relative_path", "src/gragra/different.py"),
        ("git_blob_missing_reason", "unknown provenance"),
    ],
)
def test_unloaded_module_and_provenance_changes_taint_transfer(field, value):
    source = closure("/source")
    target = copy.deepcopy(source)
    target["modules"][0][field] = value
    with pytest.raises(EvidenceError):
        compare_closures(source, target, [])


def test_unknown_material_domain_and_unknown_edge_are_rejected():
    source = closure("/source")
    target = copy.deepcopy(source)
    target["unreviewed_build"] = {"secret": True}
    with pytest.raises(EvidenceError, match="schema"):
        compare_closures(source, target, [])
    target = copy.deepcopy(source)
    target["import_edges"].append({"from": "gragra.unloaded", "to": "unknown"})
    with pytest.raises(EvidenceError):
        compare_closures(source, target, [])


def test_relocation_cannot_replace_hashes_or_scientific_relative_paths():
    source = closure("/source")
    target = closure("/target")
    with pytest.raises(EvidenceError, match="relocation"):
        compare_closures(source, target, [{"from": "2" * 64, "to": "f" * 64}])
    with pytest.raises(EvidenceError, match="relocation"):
        compare_closures(
            source,
            target,
            [{"from": "/source", "to": "/target", "relative_path": "new"}],
        )


class MemoryEvidence:
    """Raw-hashed synthetic artifacts exercise graph contracts without local APIs."""

    def __init__(self):
        self.raw = {}
        self.counter = 0

    def add(self, value, *, raw=False):
        self.counter += 1
        data = value if raw else json.dumps(value, sort_keys=True).encode() + b"\n"
        artifact = {
            "path": f"/synthetic/{self.counter}",
            "sha256": hashlib.sha256(data).hexdigest(),
            "kind": "synthetic",
        }
        self.raw[artifact["path"]] = data
        return artifact

    def read(self, artifact):
        data = self.raw[artifact["path"]]
        if hashlib.sha256(data).hexdigest() != artifact["sha256"]:
            raise EvidenceError("raw hash differs")
        return data

    def json(self, artifact):
        from tests.oracles.gen1.stochastic_delivery.archive import decode_json

        return decode_json(self.read(artifact))


def input_admission_fixture(source=False, science_id="science"):
    store = MemoryEvidence()
    subject = {"source_files": {"reference.py": "a" * 64}}
    label = "production-source" if source else "reference"
    reviews = {}
    for role in ("science", "numerical", "provenance"):
        review = {
            "schema": f"gen1-stochastic-{label}-review-v1",
            "family": "R1",
            "role": role,
            "reviewer_id": science_id if role == "science" else role,
            "decision": "PASS",
            "pins": subject,
            "original_review": store.add({"retained": role}),
        }
        if source:
            review["claim_attestations"] = {}
        reviews[role] = store.add(review)
    admission = store.add(
        {
            "schema": f"gen1-stochastic-{label}-admission-v1",
            "family": "R1",
            "decision": "PASS",
            "pins": subject,
            "reviews": reviews,
        }
    )
    bindings = {
        "family": "R1",
        "subject_kind": "production_source" if source else "reference",
        "admission": admission,
        "pins": subject,
    }
    authors = [
        {
            "reviewer_id": "actual-input-author",
            "source_artifacts": [store.add(b"source", raw=True)],
        }
    ]
    original = {
        "schema": "gen1-root-input-authorship-attestation-v1",
        **bindings,
        "authors": authors,
        "issuer_id": "actual-controller",
        "issued_at": "2026-10-04T12:00:00Z",
    }
    wrapper = {
        "schema": "gen1-stochastic-input-authorship-v1",
        **bindings,
        "authors": authors,
        "original_attestation": store.add(original),
    }
    return store, admission, store.add(wrapper)


@pytest.mark.parametrize("source", [False, True])
def test_input_admission_requires_its_own_issued_author_roster(source):
    store, admission, authorship = input_admission_fixture(source)
    assert (
        verify_input_admission(
            store,
            admission,
            family="R1",
            source=source,
            expected_bindings={},
            authorship_pin=authorship,
        )
        == store.json(admission)["pins"]
    )
    store, admission, authorship = input_admission_fixture(
        source, science_id="actual-input-author"
    )
    with pytest.raises(EvidenceError, match="independent exact input"):
        verify_input_admission(
            store,
            admission,
            family="R1",
            source=source,
            expected_bindings={},
            authorship_pin=authorship,
        )


@pytest.mark.parametrize(
    "mutation", ["roster", "subject", "empty", "issuer_author", "raw_attestation"]
)
def test_projected_author_roster_cannot_replace_issued_original(mutation):
    store, _admission, authorship = input_admission_fixture()
    wrapper = store.json(authorship)
    bindings = {
        name: wrapper[name] for name in ("family", "subject_kind", "admission", "pins")
    }
    original = store.json(wrapper["original_attestation"])
    if mutation == "roster":
        wrapper["authors"][0]["reviewer_id"] = "different-author"
    elif mutation == "subject":
        original["pins"] = {}
    elif mutation == "empty":
        wrapper["authors"] = []
    elif mutation == "issuer_author":
        original["issuer_id"] = wrapper["authors"][0]["reviewer_id"]
    elif mutation == "raw_attestation":
        original = {"decision": "PASS"}
    wrapper["original_attestation"] = store.add(original)
    with pytest.raises(EvidenceError):
        verify_authorship_attestation(
            store,
            wrapper,
            schema="gen1-stochastic-input-authorship-v1",
            original_schema="gen1-root-input-authorship-attestation-v1",
            bindings=bindings,
        )


def synthetic_stage():
    store = MemoryEvidence()
    key = {
        "family": "R1",
        "fixture_id": "finite",
        "frequency_hz": "1",
        "target_id": "height",
        "observable": "acceleration_csd",
        "representation": "Gfree",
        "set": "science",
    }
    validator = store.add(b"def verify_family_result(): pass\n", raw=True)
    module = {
        "name": "tests.oracles.gen1.stochastic_formal.common.family_supervisor",
        "path": validator["path"],
        "sha256": validator["sha256"],
    }
    closure_pin = store.add({"modules": [module]})
    policy = store.add(
        {
            "artifacts": {
                "fixture": store.add({"expected_keys": [key]}),
                "dependency_closure": closure_pin,
            }
        }
    )
    pins = {
        "policy_sha256": policy["sha256"],
        "production_commit": "a" * 40,
        "dependency_closure_sha256": closure_pin["sha256"],
    }
    candidate = store.add(
        {
            "schema": "gen1-stochastic-authorization-v2",
            "family": "R1",
            "phase": "comparison",
            "status": "REVIEW_CANDIDATE",
            "pins": pins,
            "evidence": {"policy": policy},
        }
    )
    roles = ("science", "numerical", "provenance")
    auth_reviews = {
        role: store.add(
            {
                "schema": "gen1-stochastic-review-v1",
                "role": role,
                "reviewer_id": role,
                "decision": "PASS",
                "candidate_sha256": candidate["sha256"],
                "pins": pins,
            }
        )
        for role in roles
    }
    authority = store.add(
        {
            "schema": "gen1-stochastic-authorization-v2",
            "family": "R1",
            "phase": "comparison",
            "status": "AUTHORIZED",
            "pins": pins,
            "candidate": candidate,
            "reviews": auth_reviews,
        }
    )
    parent, resource = store.add({}), store.add({})
    run_pins = {
        **pins,
        "authorization_sha256": authority["sha256"],
        "supervisor_launch_sha256": parent["sha256"],
        "resource_launch_sha256": resource["sha256"],
    }
    launch = store.add({"pins": run_pins, "expected_keys": [key]})
    rows = store.add(
        json.dumps(
            {
                "key": key,
                "passed": True,
                "status": "resolved",
                "assessment": {"accepted": True},
                "reference_purpose": "formal-science",
                "reference_set": "science",
            }
        ).encode()
        + b"\n",
        raw=True,
    )
    runtime = store.add({"status": "MATCH", "planned_sha256": closure_pin["sha256"]})
    attempt = store.add(
        {
            "schema": "gen1-stochastic-execution-seal-v1",
            "family": "R1",
            "phase": "comparison",
            "pins": run_pins,
            "decision": "PASS",
            "exit_code": 0,
            "exception": None,
            "completed": 1,
            "expected_keys": [key],
            "artifacts": {
                "launch": launch,
                "records": rows,
                "runtime_dependencies": runtime,
            },
        }
    )
    supervisor = store.add(
        {
            "schema": "gen1-stochastic-family-supervisor-seal-v1",
            "family": "R1",
            "policy_sha256": policy["sha256"],
            "outcome": "CHILD_EXITED",
            "exit_code": 0,
            "cleanup_confirmed": True,
            "artifacts": {"launch.json": parent, "resource-launch.json": resource},
        }
    )
    report = store.add({"decision": "PASS"})
    result_reviews = {
        role: store.add(
            {
                "schema": "gen1-stochastic-comparison-result-review-v1",
                "family": "R1",
                "phase": "comparison",
                "role": role,
                "reviewer_id": role,
                "decision": "PASS",
                "pins": run_pins,
                "seal_sha256": attempt["sha256"],
                "original_review": report,
            }
        )
        for role in roles
    }
    root_admission = store.add(
        {
            "schema": "gen1-root-family-production-acceptance-v1",
            "family": "R1",
            "decision": "VERIFIED_BOUNDED",
            "attempt": attempt,
            "resource_supervisor": supervisor,
            "production_comparison_pins": run_pins,
            "required_result_reviews": result_reviews,
            "excluded_reviewer_ids": ["actual-author"],
        }
    )
    admission = store.add(
        {
            "schema": "gen1-stochastic-result-admission-v1",
            "family": "R1",
            "phase": "comparison",
            "decision": "PASS",
            "pins": run_pins,
            "attempt": attempt,
            "reviews": result_reviews,
            "original_admission": root_admission,
        }
    )
    subjects = {
        "policy": policy,
        "candidate": candidate,
        "authorization": authority,
        "attempt": attempt,
        "resource_supervisor": supervisor,
    }
    script = store.add(b"pass\n", raw=True)
    original_proof = store.add(
        {
            "schema": "gen1-root-family-result-mechanical-verification-v1",
            "result": {
                "decision": "PASS",
                "pins": run_pins,
                "attempt": attempt,
                "resource_supervisor": supervisor,
            },
            "authorization": authority,
            "authorization_candidate": candidate,
            "comparator_source": validator,
            "production_api_calls_during_verification": 0,
            "script": script,
        }
    )
    proof = store.add(
        {
            "schema": "gen1-stochastic-local-validator-proof-v1",
            "family": "R1",
            "phase": "comparison",
            "profile": "generic-v1",
            "decision": "PASS",
            "subjects": subjects,
            "pins": run_pins,
            "argv": ["/original/python", script["path"]],
            "validator": {
                "source": validator,
                "qualified_function": (
                    "tests.oracles.gen1.stochastic_formal.common."
                    "family_supervisor.verify_family_result"
                ),
            },
            "original_proof": original_proof,
        }
    )
    return store, {
        "phase": "comparison",
        **subjects,
        "authorization_reviews": auth_reviews,
        "result_admission": admission,
        "result_reviews": result_reviews,
        "local_validator_proof": proof,
        "excluded_reviewer_ids": ["actual-author"],
        "input_authorship": {},
    }


def test_exact_synthetic_graph_retains_actual_p_and_raw_record_partition():
    store, stage = synthetic_stage()
    result = verify_stage(store, stage, family="R1", profile="generic-v1")
    assert result["pins"]["production_commit"] == "a" * 40
    assert result["records"][0]["key"]["set"] == "science"


@pytest.mark.parametrize(
    "subject,field,value",
    [
        ("candidate", "status", "AUTHORIZED"),
        ("authorization", "pins", {"production_commit": "f" * 40}),
        ("attempt", "decision", "BOUND"),
        ("attempt", "completed", 2),
        ("resource_supervisor", "cleanup_confirmed", False),
        ("local_validator_proof", "subjects", {}),
    ],
)
def test_wrong_graph_subject_or_terminal_evidence_rejected(subject, field, value):
    store, stage = synthetic_stage()
    document = store.json(stage[subject])
    document[field] = value
    stage[subject] = store.add(document)
    with pytest.raises(EvidenceError):
        verify_stage(store, stage, family="R1", profile="generic-v1")


def test_review_missing_role_duplicate_identity_self_approval_rejected():
    for change in ("missing", "same", "author"):
        store, stage = synthetic_stage()
        if change == "missing":
            del stage["result_reviews"]["science"]
        elif change == "author":
            stage["excluded_reviewer_ids"] = ["science"]
        else:
            review = store.json(stage["result_reviews"]["science"])
            review["reviewer_id"] = "numerical"
            stage["result_reviews"]["science"] = store.add(review)
            admission = store.json(stage["result_admission"])
            admission["reviews"] = stage["result_reviews"]
            stage["result_admission"] = store.add(admission)
        with pytest.raises(EvidenceError):
            verify_stage(store, stage, family="R1", profile="generic-v1")


def test_unknown_profile_and_b1_connection_omission_rejected():
    store, stage = synthetic_stage()
    for family, profile in (
        ("R1", "unknown"),
        ("A2", "a2-managed-v1"),
        ("B1", "b1-v1"),
    ):
        with pytest.raises(EvidenceError):
            verify_comparison(
                store, {"profile": profile, "stages": [stage]}, family=family
            )


def test_exact_ratios_and_whole_zero_budget_are_not_float_maxima():
    assert assessment_ratios(
        {"difference_upper": "0.1", "uncertainty_upper": "0.02", "budget_lower": "0.3"}
    ) == {"D_over_B": "1/3", "U_over_B": "1/15", "reason": None}
    with pytest.raises(EvidenceError, match="certified"):
        assessment_ratios(
            {"difference_upper": "0", "uncertainty_upper": "0", "budget_lower": "0"}
        )
    assert (
        assessment_ratios(
            {"difference_upper": "0", "uncertainty_upper": "0", "budget_lower": "0"},
            whole_zero=True,
        )["D_over_B"]
        is None
    )


def test_source_catalog_reconstructs_complete_git_tree_and_rejects_omission():
    store = MemoryEvidence()
    source = b"print('source')\n"
    blob = git_oid("blob", source)
    raw_tree = b"100644 module.py\0" + bytes.fromhex(blob)
    catalog = {
        "schema": "gen1-stochastic-source-catalog-v1",
        "commit": "a" * 40,
        "tree": "b" * 40,
        "subtree": "src/gragra",
        "subtree_git_tree": git_oid("tree", raw_tree),
        "files": [
            {
                "relative_path": "module.py",
                "mode": "100644",
                "git_blob": blob,
                "artifact": store.add(source, raw=True),
            }
        ],
    }
    assert verify_catalog(store, catalog) == {"module.py": ("100644", blob)}
    bad = copy.deepcopy(catalog)
    bad["files"] = []
    with pytest.raises(EvidenceError):
        verify_catalog(store, bad)
    bad = copy.deepcopy(catalog)
    bad["files"][0]["git_blob"] = "f" * 40
    with pytest.raises(EvidenceError, match="blob"):
        verify_catalog(store, bad)
    bad = copy.deepcopy(catalog)
    bad["subtree_git_tree"] = "f" * 40
    with pytest.raises(EvidenceError, match="incomplete"):
        verify_catalog(store, bad)


def test_final_h_self_reference_or_historical_recount_cannot_enter_selected_graph():
    descriptor = {
        "schema": "gen1-stochastic-selected-delivery-v1",
        "families": [],
        "history": [],
        "final_identity_boundary": "EXTERNAL_ATTESTATION_AFTER_TRACKED_COMMIT",
    }
    with pytest.raises(EvidenceError, match="seven"):
        verify_selected_delivery(MemoryEvidence(), descriptor)
    descriptor["H_final"] = "a" * 40
    with pytest.raises(EvidenceError, match="schema"):
        verify_selected_delivery(MemoryEvidence(), descriptor)


def test_writable_cas_is_rejected_even_when_content_hash_matches(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"same")
    original = pin(source)
    root = tmp_path / "cas"
    mapping = archive_selected([{"original": original, "role": "source"}], root)
    (root / "sha256" / original["sha256"]).chmod(0o644)
    with pytest.raises(EvidenceError, match="writable"):
        EvidenceStore(root, mapping).read(original)


def test_a2_generic_shaped_and_b1_incomplete_profiles_refuse_before_read():
    store, stage = synthetic_stage()
    for family, profile in (("A2", "generic-v1"), ("B1", "b1-v1")):
        with pytest.raises(EvidenceError, match="pending"):
            verify_comparison(
                store, {"profile": profile, "stages": [stage]}, family=family
            )


@pytest.mark.parametrize("field", ["attempt", "resource_supervisor"])
def test_original_validator_proof_cannot_swap_actual_result_subject(field):
    store, stage = synthetic_stage()
    proof = store.json(stage["local_validator_proof"])
    original = store.json(proof["original_proof"])
    original["result"][field] = store.add({"counterfeit": True})
    proof["original_proof"] = store.add(original)
    stage["local_validator_proof"] = store.add(proof)
    with pytest.raises(EvidenceError, match="actual subjects"):
        verify_stage(store, stage, family="R1", profile="generic-v1")


def test_author_exclusions_cannot_be_removed_from_selected_result():
    store, stage = synthetic_stage()
    stage["excluded_reviewer_ids"] = []
    with pytest.raises(EvidenceError, match="author roster"):
        verify_stage(store, stage, family="R1", profile="generic-v1")


def test_other_planned_closure_same_p_cannot_replace_actual_comparison_closure():
    from tests.oracles.gen1.stochastic_delivery.reader import verify_transfer

    store = MemoryEvidence()
    actual = store.add({"modules": ["all"]})
    smaller = store.add({"modules": []})
    descriptor = dict.fromkeys(
        (
            "target_closure",
            "source_catalog",
            "target_catalog",
            "original_git_proof",
            "authorship",
        ),
        actual,
    )
    descriptor.update(source_closure=smaller, relocation=[], reviews={})
    with pytest.raises(EvidenceError, match="exact accepted comparison closure"):
        verify_transfer(
            store,
            descriptor,
            actual_p={"commit": "a" * 40, "tree": "b" * 40},
            actual_closure=actual,
        )


def test_raw_git_commit_and_tree_prove_catalog_subtree_inclusion():
    from tests.oracles.gen1.stochastic_delivery.source import verify_git_catalog_origin

    store = MemoryEvidence()
    subtree = git_oid("tree", b"")
    src = b"40000 gragra\0" + bytes.fromhex(subtree)
    root = b"40000 src\0" + bytes.fromhex(git_oid("tree", src))
    root_oid = git_oid("tree", root)
    commit = (
        b"tree "
        + root_oid.encode()
        + b"\nauthor test <test@example.com> 0 +0000\n"
        + b"committer test <test@example.com> 0 +0000\n\nsynthetic\n"
    )
    catalog = {
        "commit": git_oid("commit", commit),
        "tree": root_oid,
        "subtree_git_tree": subtree,
    }
    entry = {
        **catalog,
        "commit_object": store.add(commit, raw=True),
        "root_tree_object": store.add(root, raw=True),
        "src_tree_object": store.add(src, raw=True),
    }
    proof = {"schema": "gen1-stochastic-git-subtree-proof-v1", "entries": [entry]}
    verify_git_catalog_origin(store, proof, [catalog])
    bad = copy.deepcopy(proof)
    bad["entries"][0]["commit_object"] = store.add(commit + b"modified", raw=True)
    with pytest.raises(EvidenceError, match="commit/root"):
        verify_git_catalog_origin(store, bad, [catalog])
    bad_catalog = {**catalog, "subtree_git_tree": "f" * 40}
    bad = copy.deepcopy(proof)
    bad["entries"][0]["subtree_git_tree"] = "f" * 40
    with pytest.raises(EvidenceError, match="linked"):
        verify_git_catalog_origin(store, bad, [bad_catalog])
