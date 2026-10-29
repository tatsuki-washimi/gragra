"""Pure synthetic B1 byte-graph regressions; no production or oracle calls."""

import hashlib
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

from tests.oracles.gen1.stochastic_delivery.archive import EvidenceError, decode_json
from tests.oracles.gen1.stochastic_delivery.b1 import (
    AUTHOR_IDS,
    CONNECTION_GATES,
    _canonical,
    _load_arithmetic,
    verify_b1_comparison,
)


class Store:
    def __init__(self):
        self.raw = {}

    def put(self, raw, kind="test_source"):
        sha = hashlib.sha256(raw).hexdigest()
        pin = {"path": "/synthetic/" + sha, "sha256": sha, "kind": kind}
        self.raw[sha] = raw
        return pin

    def read(self, pin):
        raw = self.raw[pin["sha256"]]
        assert hashlib.sha256(raw).hexdigest() == pin["sha256"]
        return raw


@pytest.mark.parametrize(
    "descriptor",
    [
        {},
        {"profile": "generic-v1", "stages": []},
        {"profile": "b1-v1", "stages": []},
        {
            "profile": "b1-v1",
            "stages": [{"phase": "comparison"}, {"phase": "connection"}],
        },
    ],
)
def test_reject_unknown_or_reversed_b1_profile(descriptor):
    with pytest.raises(EvidenceError):
        verify_b1_comparison(Store(), descriptor)


def test_counterfeit_kernel_is_never_executed():
    store = Store()
    pin = store.put(b"raise AssertionError('counterfeit code executed')\n")
    with pytest.raises(EvidenceError, match="arithmetic"):
        _load_arithmetic(store, pin, "kernel")


def test_admitted_cas_kernel_and_adapter_only():
    store = Store()
    _put_style_proofs(store)
    root = Path(__file__).parents[1] / "tests/oracles/gen1"
    kernel = _load_arithmetic(
        store, store.put((root / "stochastic_acceptance.py").read_bytes()), "kernel"
    )
    adapter = _load_arithmetic(
        store, store.put((root / "b1_exact_record_adapter.py").read_bytes()), "adapter"
    )
    assert kernel.decimal_fraction("3e-1") == adapter.parse_exact_fraction("3e-1")


def toy_graph(adapter_version="formatted"):
    """Complete disjoint synthetic 70/30 chain, using admitted arithmetic only."""
    store = Store()
    _put_style_proofs(store)
    store.json = lambda pin: decode_json(store.read(pin))
    store.j = lambda obj, kind="synthetic_metadata": store.put(_canonical(obj), kind)
    root = Path(__file__).parents[1] / "tests/oracles/gen1"
    kernel_pin = store.put((root / "stochastic_acceptance.py").read_bytes())
    if adapter_version == "original":
        adapter_bytes = _original_adapter_bytes()
    elif adapter_version == "formatted":
        adapter_bytes = _historical_formatted_adapter_bytes()
    else:
        raise ValueError(f"unknown synthetic adapter fixture: {adapter_version}")
    adapter_pin = store.put(adapter_bytes)
    transport_pin = store.put(
        (root / "stochastic_formal/common/transport.py").read_bytes()
    )
    kernel = _load_arithmetic(store, kernel_pin, "kernel")
    adapter = _load_arithmetic(store, adapter_pin, "adapter")
    authors = sorted(AUTHOR_IDS)
    production = {"commit": "a" * 40, "tree": "b" * 40}

    def key(case, f, rep, obs="acceleration_csd", group="science"):
        return {
            "family": "B1",
            "fixture_id": case,
            "frequency_hz": f,
            "target_id": "origin",
            "observable": obs,
            "representation": rep,
            "set": group,
        }

    frequencies = ["1.3e+0", "2.7e+0", "5.1e+0"]
    cases = [
        "toy_component_a",
        "toy_component_b",
        "toy_total",
        "zero_density",
        "zero_psd",
    ]
    science_keys = [
        key(case, f, rep)
        for case in cases
        for f in frequencies
        for rep in ["Gfree", "SI"]
    ]
    aux_keys = [
        key(
            "toy_aux_" + str(i),
            "1.3e+0",
            "Gfree",
            "p_acceleration_connection",
            "auxiliary",
        )
        for i in range(70)
    ]
    science_fixtures = [
        {
            "fixture_id": case,
            "frequency_hz": frequencies,
            "input_assumptions": "exact_input",
            "input_convention": "component",
            "displacement_psd_m2_hz": ["3e-9"] * 3,
        }
        for case in cases
    ]
    fixture = {
        "expected_keys": science_keys,
        "expected_auxiliary_keys": aux_keys,
        "expected_reference_auxiliary_keys": aux_keys[:56],
        "expected_consistency_auxiliary_keys": aux_keys[56:],
        "science": science_fixtures,
    }
    common = store.j({"scope": "SYNTHETIC_ONLY_COMMON"})
    faux = store.j({"scope": "SYNTHETIC_ONLY_F_AUX"})
    aux = [
        {
            "key": k,
            "status": "resolved",
            "procedure_hash": faux["sha256"],
            "record_kind": "first_order_connection_reference",
        }
        for k in aux_keys[:56]
    ]
    aux_records = store.put(b"".join(_canonical(r) for r in aux))
    aux_seal = store.j(
        {
            "records_sha256": aux_records["sha256"],
            "f_b1_aux_sha256": faux["sha256"],
            "mode": "formal",
            "exit_code": 0,
            "coverage_valid": True,
            "cleanup_confirmed": True,
        }
    )
    aux_subject = {
        "records_sha256": aux_records["sha256"],
        "seal_sha256": aux_seal["sha256"],
    }
    aux_reviews = {
        role: store.j(
            {
                "decision": "PASS",
                "role": role,
                "reviewer_id": {
                    "science": "seismic_exec_independent_B1aux",
                    "numerical": "/root/temperature_exec",
                    "provenance": "/root/acoustic_exec",
                }[role],
                "pins": aux_subject,
                (
                    "prior_science_review" if role == "science" else "original_review"
                ): store.j({"scope": "SYNTHETIC_ONLY_RAUX_REPORT", "role": role}),
            }
        )
        for role in ["science", "numerical", "provenance"]
    }
    aux_admission = store.j(
        {
            "schema": "gen1-stochastic-reference-admission-v1",
            "family": "B1_aux",
            "decision": "PASS",
            **aux_subject,
            "f_b1_aux_sha256": faux["sha256"],
            "pins": aux_subject,
            "reviews": aux_reviews,
        }
    )
    raw_refs = []
    sidecars = []
    for k in science_keys:
        zero = k["fixture_id"] in {"zero_density", "zero_psd"}
        central = Fraction() if zero else Fraction(2, 1000)
        u = Fraction() if zero else Fraction(1, 10**90)
        text = adapter.fraction_to_exact_text
        raw_refs.append(
            {
                "key": k,
                "status": "resolved",
                "enclosure": {
                    "central": text(central),
                    "digits": 80,
                    "lo_exact": text(central - u),
                    "hi_exact": text(central + u),
                    "u_up": text(u),
                },
            }
        )
        sidecars.append(
            {
                "key": k,
                "status": "resolved",
                "classification": "exact_zero" if zero else "rigorous_bound",
                "B1_nine_source_allocation_sum_equals_u_up": True,
                "B_ref_lower": "0e+0" if zero else "2e-16",
                "conservative_10_U_matrix_over_B_ref_lower": "0e+0"
                if zero
                else "1e-50",
                "exact_endpoint_and_no_narrowing": True,
                "preproduction_oracle_fraction_pass": True,
                "source_scale": text(central),
                "u_up": text(u),
                "uncertainty_certificate": "SYNTHETIC_ONLY whole observable",
                "uncertainty_matrix_upper": text(3 * u),
                "units": "kg^2 m^-4 Hz^-1"
                if k["representation"] == "Gfree"
                else "m^2 s^-4 Hz^-1",
                "zero_or_nonzero_semantics": True,
            }
        )
    rb_records = store.put(b"".join(_canonical(r) for r in raw_refs))
    rb_seal = store.j({"artifacts_sha256": {"records.raw.jsonl": rb_records["sha256"]}})
    rb_review = store.j(
        {
            "R_B1": "ACCEPTED_FROZEN_REFERENCE_SET",
            "attempt": {
                "records_sha256": rb_records["sha256"],
                "seal_sha256": rb_seal["sha256"],
            },
            "uncertainty_and_oracle_fraction": {"per_record": sidecars},
        }
    )
    src = store.put(b"# SYNTHETIC_ONLY no production\n")
    module = {**src, "name": "gragra.synthetic_unused"}
    closure = {
        "production_commit": production["commit"],
        "production_tree": production["tree"],
        "modules": [module],
        "import_edges": [],
        "native_backends": [],
        "packages": [],
        "python": {"version": "SYNTHETIC_ONLY"},
        "artifacts": [adapter_pin],
    }
    dep = store.j(closure)
    budget = store.j(
        {"absolute_rate": "1e-13", "relative_rate": "1e-12", "structural_rate": "7e-15"}
    )
    declarations = [
        {
            "key": k,
            "receipt_id": "toy-" + str(i),
            "source": aux_records if i < 56 else src,
            "destination": src,
            "kind": "independent_reference" if i < 56 else "production_consistency",
            "science_keys": science_keys,
            "rule_version": "SYNTHETIC_ONLY",
        }
        for i, k in enumerate(aux_keys)
    ]
    expectations = store.j({"receipts": declarations})
    binding_pins = {
        "production_commit": production["commit"],
        "production_tree": production["tree"],
        "f_b1_aux_sha256": faux["sha256"],
    }
    bound = []
    for row in declarations:
        row = {**row, "status": "BOUND"}
        bound.append(
            {**row, "content_sha256": hashlib.sha256(_canonical(row)).hexdigest()}
        )
    bundle = store.j(
        {
            "schema": "gen1-stochastic-bound-receipts-v1",
            "status": "BOUND",
            "pins": binding_pins,
            "receipts": bound,
        }
    )
    artifacts = {
        "fixture": store.j(fixture),
        "budget": budget,
        "procedure": store.j({"scope": "SYNTHETIC_ONLY_PROCEDURE"}),
        "f_common": common,
        "f_b1_aux": faux,
        "aux_freeze": store.j({"scope": "SYNTHETIC_ONLY_FREEZE"}),
        "decision_kernel": kernel_pin,
        "dependency_closure": dep,
        "r_b1_records": rb_records,
        "r_b1_seal": rb_seal,
        "r_b1_review": rb_review,
        "r_aux_records": aux_records,
        "r_aux_seal": aux_seal,
        "r_aux_admission": aux_admission,
        "receipt_expectations": expectations,
        "bound_receipts": bundle,
    }
    base = {
        "family": "B1",
        "production_commit": production["commit"],
        "production_tree": production["tree"],
        "comparator_commit": "c" * 40,
        "comparator_tree": "d" * 40,
    }
    for name, p in artifacts.items():
        if name != "aux_freeze":
            base[name + "_sha256"] = p["sha256"]
    reviewer_ids = {
        "science": "/root/a2_exec",
        "numerical": "/root/temperature_exec",
        "provenance": "/root/acoustic_exec",
    }
    stages = []
    original_connection = None
    connection_run = None
    for phase, keys in [("connection", aux_keys), ("comparison", science_keys)]:
        policy = {
            "schema": "gen1-stochastic-B1-comparison-policy-v2",
            "family": "B1",
            "phase": phase,
            "pins": dict(base),
            "artifacts": dict(artifacts),
            "connection_pins": {},
        }
        if phase == "comparison":
            policy["artifacts"]["connection_seal"] = stages[0]["attempt"]
            policy["connection_pins"] = connection_run
            policy["pins"]["connection_pass_sha256"] = stages[0]["attempt"]["sha256"]
        pp = store.j(policy, "policy")
        pins = {**policy["pins"], "policy_sha256": pp["sha256"]}
        candidate = store.j(
            {
                "schema": "gen1-stochastic-authorization-v2",
                "family": "B1",
                "phase": phase,
                "status": "REVIEW_CANDIDATE",
                "pins": pins,
                "evidence": {"policy": pp},
            }
        )
        authority_reviews = {
            role: store.j(
                {
                    "schema": "gen1-stochastic-review-v1",
                    "role": role,
                    "reviewer_id": "synthetic_binding_" + role,
                    "decision": "PASS",
                    "candidate_sha256": candidate["sha256"],
                    "pins": pins,
                }
            )
            for role in reviewer_ids
        }
        authority = store.j(
            {
                "schema": "gen1-stochastic-authorization-v2",
                "family": "B1",
                "phase": phase,
                "status": "AUTHORIZED",
                "pins": pins,
                "candidate": candidate,
                "reviews": authority_reviews,
            }
        )
        supervisor_launch = store.j(
            {"mode": "FORMAL", "policy_sha256": pp["sha256"], "timeout_seconds": 300}
        )
        run = {
            **pins,
            "authorization_sha256": authority["sha256"],
            "supervisor_launch_sha256": supervisor_launch["sha256"],
        }
        launch = store.j({"pins": run, "expected_keys": keys})
        if phase == "connection":
            rows = [
                {
                    "key": k,
                    "status": "resolved",
                    "passed": True,
                    "evidence_class": "independent_first_order_reference"
                    if i < 56
                    else "production_consistency_only",
                    "gate_results": dict.fromkeys(CONNECTION_GATES, True),
                    "production_pin": production,
                    "inputs": {"scope": "SYNTHETIC_ONLY"},
                    "input_sha256": hashlib.sha256(
                        _canonical({"scope": "SYNTHETIC_ONLY"})
                    ).hexdigest(),
                    "reference_pin": aux_records if i < 56 else None,
                    "assessment": {
                        "evidence_class": "independent_first_order_reference"
                        if i < 56
                        else "production_consistency_only",
                        "passed": True,
                        "precision_crosscheck": True,
                        "conditions": [
                            {
                                "precision": digits,
                                "difference_upper": "0e+0",
                                "uncertainty_upper": "0e+0",
                                "allowance_lower": "1e-20",
                                "difference_plus_uncertainty": True,
                                "oracle_fraction": True,
                                "passed": True,
                            }
                            for digits in [120, 180]
                        ],
                    },
                }
                for i, k in enumerate(keys)
            ]
        else:
            adapted = adapter.adapt_b1_records(
                store.read(rb_records),
                sidecars,
                science_keys,
                [
                    k
                    for k in science_keys
                    if k["fixture_id"] in {"zero_density", "zero_psd"}
                ],
            )
            rows = []
            b = store.json(budget)
            for k in keys:
                ref = adapted[
                    tuple(
                        k[n]
                        for n in [
                            "family",
                            "fixture_id",
                            "frequency_hz",
                            "target_id",
                            "observable",
                            "representation",
                            "set",
                        ]
                    )
                ]
                quantity = ref["kernel_quantity"]
                structure = kernel.structure_gates(
                    quantity,
                    "identity",
                    adapter.fraction_to_exact_text(
                        Fraction(7, 10**15) * ref["source_scale"]
                    ),
                )
                contract = dict.fromkeys(kernel.CONTRACT_GATES, True)
                assessment = kernel.assess(
                    quantity,
                    quantity,
                    adapter.fraction_to_exact_text(ref["source_scale"]),
                    b["absolute_rate"],
                    b["relative_rate"],
                    ref["kernel_uncertainty"],
                    structure,
                    contract,
                )
                rows.append(
                    {
                        "key": k,
                        "status": "resolved",
                        "passed": True,
                        "production_quantity": quantity,
                        "assessment": assessment,
                        "structure_conditions": structure,
                        "contract_conditions": contract,
                        "reference_central_mapping": "central*I3",
                        "reference_uncertainty_mapping": "3*u_up",
                        "exact_zero_pass": True,
                    }
                )
        records = store.put(b"".join(_canonical(r) for r in rows))
        runtime = store.j(
            {
                "status": "MATCH",
                "planned_sha256": dep["sha256"],
                "python": closure["python"],
                "observed_modules": [module],
                "observed_edges": [],
                "observed_native_backends": [],
            }
        )
        attempt = store.j(
            {
                "schema": "gen1-stochastic-execution-seal-v1",
                "family": "B1",
                "phase": phase,
                "decision": "PASS",
                "exit_code": 0,
                "exception": None,
                "pins": run,
                "completed": len(keys),
                "expected_keys": keys,
                "artifacts": {
                    "launch": launch,
                    "records": records,
                    "runtime_dependencies": runtime,
                },
            }
        )
        supervisor = store.j(
            {
                "schema": "gen1-stochastic-supervisor-seal-v1",
                "outcome": "CHILD_EXITED",
                "exit_code": 0,
                "cleanup_confirmed": True,
                "elapsed_seconds": "1e-1",
                "cleanup": {
                    "leader_reaped": True,
                    "process_group_empty": True,
                    "status": "confirmed_stopped",
                    "wait_errors": [],
                },
                "artifacts": {"launch": supervisor_launch},
            }
        )
        results = {
            role: store.j(
                {
                    "schema": "gen1-stochastic-connection-result-review-v1"
                    if phase == "connection"
                    else "gen1-stochastic-comparison-result-review-v1",
                    "family": "B1",
                    "phase": phase,
                    "role": role,
                    "reviewer_id": reviewer_ids[role],
                    "decision": "PASS",
                    "pins": run,
                    "seal_sha256": attempt["sha256"],
                    "original_review": store.j({"scope": "SYNTHETIC_ONLY " + role}),
                }
            )
            for role in reviewer_ids
        }
        if phase == "connection":
            original_connection = store.j(
                {
                    "schema": "gen1-stochastic-connection-result-admission-v1",
                    "family": "B1",
                    "decision": "PASS",
                    "pins": run,
                    "reviews": results,
                    "seal": attempt,
                    "counts": {
                        "independent_first_order_reference": 56,
                        "production_consistency_only": 14,
                    },
                    "source_connections": True,
                    "science_comparison_authorized": False,
                }
            )
            original = original_connection
            connection_run = run
        else:
            original = store.j(
                {
                    "schema": "gen1-stochastic-family-acceptance-v1",
                    "family": "B1",
                    "status": "VERIFIED_BOUNDED",
                    "pins": run,
                    "reviews": results,
                    "accepted": 30,
                    "science_records": 30,
                    "failed": 0,
                    "unresolved": 0,
                    "P_family": production["commit"],
                    "comparison_seal": attempt,
                    "connection_reference_gates": 56,
                    "connection_consistency_gates": 14,
                    "supervisor_seal": supervisor,
                    "connection_admission": original_connection,
                }
            )
        admission = store.j(
            {
                "schema": "gen1-stochastic-result-admission-v1",
                "family": "B1",
                "phase": phase,
                "decision": "PASS",
                "pins": run,
                "attempt": attempt,
                "reviews": results,
                "original_admission": original,
            }
        )
        stage = {
            "phase": phase,
            "policy": pp,
            "candidate": candidate,
            "authorization": authority,
            "authorization_reviews": authority_reviews,
            "attempt": attempt,
            "resource_supervisor": supervisor,
            "result_admission": admission,
            "result_reviews": results,
            "excluded_reviewer_ids": authors,
        }
        subjects = {
            n: stage[n]
            for n in [
                "policy",
                "candidate",
                "authorization",
                "attempt",
                "resource_supervisor",
            ]
        }
        actual = store.j(
            {
                "schema": "gen1-root-B1-result-binding-verification-v1",
                "phase": phase,
                "scope": "SUPPLEMENTAL_CURRENT_AUTHORITY_BINDING_ONLY",
                "created_at_utc": "SYNTHETIC_ONLY_TIMESTAMP",
                "subjects": subjects,
                "pins": run,
                "result": {"decision": "PASS", "pins": run},
                "production_API_calls": 0,
                "reference_generations": 0,
                "original_execution_repeated": False,
                "argv": ["synthetic_metadata_only"],
                "orchestration_source": src,
                "preflight_source": src,
            }
        )
        proof = store.j(
            {
                "schema": "gen1-stochastic-local-validator-proof-v1",
                "family": "B1",
                "phase": phase,
                "profile": "b1-v1",
                "decision": "PASS",
                "subjects": subjects,
                "pins": run,
                "validator": {
                    "source": transport_pin,
                    "qualified_function": (
                        "tests.oracles.gen1.stochastic_formal.common.transport."
                    )
                    + (
                        "validate_connection_pass"
                        if phase == "connection"
                        else "require_science_authority"
                    ),
                },
                "argv": ["synthetic_metadata_only"],
                "original_proof": actual,
            }
        )
        stage["local_validator_proof"] = proof
        stages.append(stage)
    return store, {"profile": "b1-v1", "stages": stages}


def test_complete_synthetic_chain_replays_18_bounds_and_12_zeros():
    store, descriptor = toy_graph()
    results = verify_b1_comparison(store, descriptor)
    assert [len(r["records"]) for r in results] == [70, 30]
    assert results[0]["summary"]["independent_first_order_reference"] == 56
    assert results[1]["summary"]["rigorous_bound"] == 18
    assert results[1]["summary"]["exact_whole_zero"] == 12


@pytest.mark.parametrize(
    "where",
    [
        "authorization",
        "attempt",
        "resource_supervisor",
        "result_admission",
        "local_validator_proof",
    ],
)
def test_missing_connection_graph_stops_before_science_read(where):
    store, descriptor = toy_graph()
    missing = descriptor["stages"][0][where]["sha256"]
    del store.raw[missing]
    science_policy = descriptor["stages"][1]["policy"]["sha256"]
    old_read = store.read

    def read(pin):
        assert pin["sha256"] != science_policy, "science read before complete 70 PASS"
        return old_read(pin)

    store.read = read
    with pytest.raises(EvidenceError):
        verify_b1_comparison(store, descriptor)


@pytest.mark.parametrize("bad_stage", [None, 3, [], "connection"])
def test_nonobject_stage_is_evidence_error(bad_stage):
    with pytest.raises(EvidenceError):
        verify_b1_comparison(
            Store(),
            {"profile": "b1-v1", "stages": [bad_stage, {"phase": "comparison"}]},
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "BOUND_attempt",
        "stale_authority",
        "writer_review",
        "aliased_review",
        "runtime_edge",
        "R_aux_substitution",
        "missing_unloaded_source",
        "U_mapping",
        "whole_zero",
    ],
)
def test_semantic_faults_are_rejected_after_archive_boundary(mutation):
    """Inject decoded metadata faults; Store's raw-byte check is tested separately.

    These are verifier unit probes, not fabricated admissible archives. Each
    mutation changes one contract that must still reject after byte admission.
    """
    import copy

    store, descriptor = toy_graph()
    connection, science = descriptor["stages"]
    policy = store.json(connection["policy"])
    cseal = store.json(connection["attempt"])
    sseal = store.json(science["attempt"])
    targets = {
        "BOUND_attempt": connection["attempt"],
        "stale_authority": connection["authorization"],
        "writer_review": connection["result_reviews"]["science"],
        "aliased_review": connection["result_reviews"]["science"],
        "runtime_edge": cseal["artifacts"]["runtime_dependencies"],
        "R_aux_substitution": connection["policy"],
        "missing_unloaded_source": policy["artifacts"]["dependency_closure"],
    }
    old_json = store.json
    old_read = store.read
    if mutation in {"U_mapping", "whole_zero"}:
        pin = sseal["artifacts"]["records"]
        rows = [decode_json(line) for line in old_read(pin).splitlines()]
        if mutation == "U_mapping":
            rows[0]["reference_uncertainty_mapping"] = "u_up"
        else:
            row = next(r for r in rows if r["key"]["fixture_id"] == "zero_density")
            row["production_quantity"]["data"][0]["real"] = "1e-90"
        replacement = b"".join(_canonical(r) for r in rows)
        store.read = lambda p: (
            replacement if p["sha256"] == pin["sha256"] else old_read(p)
        )
    else:
        target = targets[mutation]["sha256"]

        def decoded(pin):
            obj = old_json(pin)
            if pin["sha256"] != target:
                return obj
            obj = copy.deepcopy(obj)
            if mutation == "BOUND_attempt":
                obj["decision"] = "BOUND"
            elif mutation == "stale_authority":
                obj["pins"]["production_commit"] = "f" * 40
            elif mutation == "writer_review":
                obj["reviewer_id"] = "/root/b1_exec"
            elif mutation == "aliased_review":
                obj["reviewer_id"] = "b1_exec-independent-science"
            elif mutation == "runtime_edge":
                obj["observed_edges"].append(
                    {"caller": "undeclared", "callee": "gragra"}
                )
            elif mutation == "R_aux_substitution":
                obj["artifacts"]["r_b1_records"] = obj["artifacts"]["r_aux_records"]
            elif mutation == "missing_unloaded_source":
                obj["artifacts"].append(
                    {
                        "path": "/missing/unloaded.py",
                        "sha256": "0" * 64,
                        "kind": "unloaded_source",
                    }
                )
            return obj

        store.json = decoded
    with pytest.raises(EvidenceError):
        verify_b1_comparison(store, descriptor)


@pytest.mark.parametrize("identity", ["/root/b1_exec", "b1_exec_independent_B1aux"])
def test_raux_author_cannot_review_own_reference_even_under_alias(identity):
    store, descriptor = toy_graph()
    policy = store.json(descriptor["stages"][0]["policy"])
    admission = store.json(policy["artifacts"]["r_aux_admission"])
    target = admission["reviews"]["science"]["sha256"]
    previous = store.json

    def decoded(pin):
        obj = previous(pin)
        if pin["sha256"] == target:
            obj = dict(obj, reviewer_id=identity)
        return obj

    store.json = decoded
    with pytest.raises(EvidenceError, match="R_aux"):
        verify_b1_comparison(store, descriptor)


@pytest.mark.parametrize("role", ["science", "numerical", "provenance"])
def test_missing_raux_original_report_is_not_an_independent_review(role):
    store, descriptor = toy_graph()
    policy = store.json(descriptor["stages"][0]["policy"])
    admission = store.json(policy["artifacts"]["r_aux_admission"])
    review = store.json(admission["reviews"][role])
    field = "prior_science_review" if role == "science" else "original_review"
    del store.raw[review[field]["sha256"]]
    with pytest.raises(EvidenceError):
        verify_b1_comparison(store, descriptor)


@pytest.mark.parametrize("role", ["science", "numerical", "provenance"])
def test_historical_two_field_raux_report_pin_has_explicit_scoped_type(role):
    from tests.oracles.gen1.stochastic_delivery.archive import validate_pin

    store, descriptor = toy_graph()
    policy = store.json(descriptor["stages"][0]["policy"])
    admission = store.json(policy["artifacts"]["r_aux_admission"])
    target = admission["reviews"][role]["sha256"]
    old_json, old_read = store.json, store.read

    def strict_read(pin):
        validate_pin(pin)
        return old_read(pin)

    def decoded(pin):
        obj = old_json(pin)
        if pin["sha256"] == target:
            obj = dict(obj)
            field = "prior_science_review" if role == "science" else "original_review"
            obj[field] = {n: obj[field][n] for n in ("path", "sha256")}
        return obj

    store.read, store.json = strict_read, decoded
    results = verify_b1_comparison(store, descriptor)
    assert results[1]["summary"]["science_passed"] == 30


# Exact retained fab8 review/equivalence bytes; no local campaign is required.
_STYLE_PROOF_FIXTURE = (
    "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"
    "vvKArzzgKw/4ygO+8oDfmQf8v/8POUozmg=="
)


def _put_style_proofs(store):
    import base64
    import zlib

    payload = decode_json(zlib.decompress(base64.b64decode(_STYLE_PROOF_FIXTURE)))
    for digest, encoded in payload.items():
        raw = base64.b64decode(encoded)
        assert hashlib.sha256(raw).hexdigest() == digest
        store.raw[digest] = raw


def _original_adapter_bytes():
    from tests.test_stochastic_family_comparator import _git_blob

    raw = _git_blob("fdba6f3cd1f00c57e800940c8a967063d6b0a680")
    assert (
        hashlib.sha256(raw).hexdigest()
        == "3d13f6e509f3727cd0d2045a6e8eeab5f609987425ec5156386efd39f5006aea"
    )
    return raw


def _historical_formatted_adapter_bytes():
    commit = "ad096eb09a41bffc132bda48f101d99f8446102c"
    relative_path = "tests/oracles/gen1/b1_exact_record_adapter.py"
    raw = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=Path(__file__).parents[1],
        check=True,
        capture_output=True,
    ).stdout
    assert (
        hashlib.sha256(raw).hexdigest()
        == "16a0e035b0f502010a8e91d67d1799d617ae53097a6f465e255559caead6829e"
    )
    return raw


def test_formatted_adapter_fixture_uses_pinned_historical_bytes():
    raw = _historical_formatted_adapter_bytes()
    current = (
        Path(__file__).parents[1]
        / "tests/oracles/gen1/b1_exact_record_adapter.py"
    ).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "16a0e035b0f502010a8e91d67d1799d617ae53097a6f465e255559caead6829e"
    )
    assert raw != current


@pytest.mark.parametrize("version", ["original", "formatted"])
def test_exact_reviewed_adapter_versions_replay_complete_chain(version):
    store, descriptor = toy_graph(version)
    result = verify_b1_comparison(store, descriptor)
    assert result[1]["summary"]["science_passed"] == 30
    assert result[1]["summary"]["arithmetic_sources"]["adapter"]["version"] == version


def test_unknown_adapter_source_is_not_executed():
    store = Store()
    pin = store.put(b'raise AssertionError("unreviewed adapter ran")\n')
    with pytest.raises(EvidenceError, match="arithmetic"):
        _load_arithmetic(store, pin, "adapter")


def test_formatted_adapter_requires_actual_style_original_reports():
    store, descriptor = toy_graph()
    from tests.oracles.gen1.stochastic_delivery.b1 import ADAPTER_STYLE_REVIEWS

    review = decode_json(store.read(ADAPTER_STYLE_REVIEWS["science"]))
    del store.raw[review["original_review"]["sha256"]]
    with pytest.raises(EvidenceError):
        verify_b1_comparison(store, descriptor)
