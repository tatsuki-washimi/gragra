"""Serialized synthetic quantities only; no scientific API or oracle execution."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tests.oracles.gen1.stochastic_delivery.archive import EvidenceError
from tests.oracles.gen1.stochastic_delivery.arithmetic import (
    KERNEL_SHA256,
    load_kernel,
    replay_generic_arithmetic,
)
from tests.test_stochastic_delivery_evidence import MemoryEvidence


def sample(kind="estimate"):
    store = MemoryEvidence()
    raw = (Path(__file__).parent / "oracles/gen1/stochastic_acceptance.py").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == KERNEL_SHA256
    kernel_pin = store.add(raw, raw=True)
    kernel = load_kernel(store, kernel_pin)
    keys = [
        {
            "family": "A3",
            "fixture_id": "synthetic",
            "frequency_hz": "1e+0",
            "target_id": "center",
            "observable": "acceleration_csd",
            "representation": rep,
            "set": "science",
        }
        for rep in ("Gfree", "SI")
    ]

    def quantity(diagonal):
        return {
            "shape": [3, 3],
            "data": [
                {
                    "real": kernel.promote_binary64(diagonal if i % 4 == 0 else 0.0),
                    "imag": "0e+0",
                }
                for i in range(9)
            ],
        }

    uncertainty = {
        "kind": kind,
        "components": {
            name: {
                "value": "1e-9"
                if name == "precision" and kind != "exact_zero"
                else "0e+0",
                "method": "synthetic whole observable",
            }
            for name in kernel.COMPONENTS
        },
        "certificate": "synthetic whole zero" if kind == "exact_zero" else None,
    }
    refs, comparisons = [], []
    for key in keys:
        q = quantity(
            0.0
            if kind == "exact_zero"
            else 1.0
            if key["representation"] == "Gfree"
            else 4.0
        )
        reference = {
            "key": key,
            "quantity": q,
            "status": "resolved",
            "inputs": {"g_si": "2e+0"},
            "units": key["representation"],
            "uncertainty": uncertainty,
        }
        rawref = json.dumps(reference, sort_keys=True).encode() + b"\n"
        refs.append(rawref)
        structures = kernel.structure_gates(q, "identity", "0e+0")
        contracts = dict.fromkeys(kernel.CONTRACT_GATES, True)
        assessment = kernel.assess(
            q,
            q,
            "0e+0" if kind == "exact_zero" else "1e+0",
            "1e-4",
            "1e-4",
            uncertainty,
            structures,
            contracts,
        )
        row = {
            "key": key,
            "status": "resolved",
            "passed": True,
            "production_quantity": q,
            "reference_record_sha256": hashlib.sha256(rawref).hexdigest(),
            "reference_purpose": "formal-science",
            "reference_set": "science",
            "structure_conditions": structures,
            "contract_conditions": contracts,
            "assessment": assessment,
            "connection_assessment": {
                "gates": {"frozen connection": True},
                "evidence": {},
            },
        }
        comparisons.append(json.dumps(row, sort_keys=True).encode() + b"\n")
    artifacts = {
        "decision_kernel": kernel_pin,
        "fixture": store.add(
            {"expected_keys": keys, "units": {"Gfree": "Gfree", "SI": "SI"}}
        ),
        "budget": store.add({"absolute_rate": "1e-4", "relative_rate": "1e-4"}),
        "source_scales": store.add(
            {
                "schema": "gen1-stochastic-source-scales-v1",
                "family": "A3",
                "rows": [
                    {"key": key, "value": "0e+0" if kind == "exact_zero" else "1e+0"}
                    for key in keys
                ],
            }
        ),
        "records": store.add(b"".join(refs), raw=True),
    }
    policy = store.add(
        {"artifacts": artifacts, "connection_requirements": ["frozen connection"]}
    )
    attempt = store.add(
        {"artifacts": {"records": store.add(b"".join(comparisons), raw=True)}}
    )
    return store, {"policy": policy, "attempt": attempt}


def alter_comparisons(store, stage, mutate):
    attempt = store.json(stage["attempt"])
    rows = [
        json.loads(line)
        for line in store.read(attempt["artifacts"]["records"]).splitlines()
    ]
    mutate(rows)
    attempt["artifacts"]["records"] = store.add(
        b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in rows),
        raw=True,
    )
    return {**stage, "attempt": store.add(attempt)}


def test_replay_retains_estimate_claim_and_exact_metrics_without_adoption():
    store, stage = sample()
    result = replay_generic_arithmetic(store, stage, family="A3")
    assert result["status"] == "RETAINED_ARITHMETIC_REPLAYED"
    assert result["science_count"] == 2
    assert result["uncertainty_kind_counts"]["estimate"] == 2
    assert result["nonzero_oracle_count"] == 2
    assert result["max_ratios"]["D_over_B"]["value"] == "0"
    assert result["production_api_calls"] == 0
    assert result["independent_result_admission_verified"] is False
    assert result["final_transfer_established"] is False
    assert result["original_execution_repeated"] is False


def test_whole_zero_ratios_are_null_with_an_explicit_certificate_reason():
    store, stage = sample("exact_zero")
    result = replay_generic_arithmetic(store, stage, family="A3")
    assert result["zero_budget_ratio_count"] == 2
    assert result["whole_zero_oracle_count"] == 2
    assert result["nonzero_oracle_count"] == 0
    assert result["max_ratios"] == {"D_over_B": None, "U_over_B": None}


@pytest.mark.parametrize(
    "mutation",
    [
        "quantity",
        "assessment",
        "structure",
        "reference",
        "missing",
        "duplicate",
        "connection",
        "claim",
        "nonbinary64",
    ],
)
def test_replay_rejects_inconsistent_retained_quantities_and_gates(mutation):
    store, stage = sample()

    def mutate(rows):
        if mutation == "quantity":
            rows[0]["production_quantity"]["data"][0]["real"] = "2e+0"
        elif mutation == "assessment":
            rows[0]["assessment"]["difference_upper"] = "1e-5"
        elif mutation == "structure":
            rows[0]["structure_conditions"]["hermitian"] = False
        elif mutation == "reference":
            rows[0]["reference_record_sha256"] = "a" * 64
        elif mutation == "missing":
            rows.pop()
        elif mutation == "duplicate":
            rows.append(copy.deepcopy(rows[0]))
        elif mutation == "connection":
            rows[0]["connection_assessment"]["gates"] = {"invented gate": True}
        elif mutation == "claim":
            rows[0]["assessment"]["claim"] = "rigorous theorem"
        elif mutation == "nonbinary64":
            rows[0]["production_quantity"]["data"][0]["real"] = (
                "1.00000000000000000000000001e+0"
            )

    with pytest.raises(EvidenceError):
        replay_generic_arithmetic(
            store, alter_comparisons(store, stage, mutate), family="A3"
        )


def test_unknown_kernel_and_pending_family_profiles_are_rejected():
    store, stage = sample()
    with pytest.raises(EvidenceError, match="kernel"):
        load_kernel(
            store, store.add(b"raise RuntimeError('must never execute')", raw=True)
        )
    for family in ("B1", "A2"):
        with pytest.raises(EvidenceError, match="unsupported"):
            replay_generic_arithmetic(store, stage, family=family)
