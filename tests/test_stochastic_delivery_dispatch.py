"""Profile integration tests use only synthetic retained bytes, never P APIs."""

import pytest

from tests.oracles.gen1.stochastic_delivery import reader
from tests.oracles.gen1.stochastic_delivery.archive import EvidenceError
from tests.test_stochastic_delivery_arithmetic import sample
from tests.test_stochastic_delivery_b1 import toy_graph


def test_dispatch_replays_real_synthetic_b1_70_before_30_with_legacy_authority():
    store, descriptor = toy_graph()
    assert len(descriptor["stages"][0]) == 11
    results = reader.verify_selected_comparison(store, descriptor, family="B1")
    assert [len(result["records"]) for result in results] == [70, 30]
    assert results[0]["summary"]["independent_first_order_reference"] == 56
    assert results[0]["summary"]["production_consistency_only"] == 14
    assert results[1]["summary"]["rigorous_bound"] == 18
    assert results[1]["summary"]["exact_whole_zero"] == 12


def test_failed_b1_connection_does_not_read_science_subject():
    store, descriptor = toy_graph()
    descriptor["stages"][0]["attempt"] = store.put(b'{"decision":"BOUND"}')
    descriptor["stages"][1]["policy"] = {"must_not_be_read": True}
    with pytest.raises(EvidenceError):
        reader.verify_selected_comparison(store, descriptor, family="B1")


def test_generic_dispatch_adds_actual_admission_only_after_successful_stage_gate(
    monkeypatch,
):
    store, stage = sample()
    descriptor = {"profile": "generic-v1", "stages": [stage]}
    called = []

    def admitted(_store, _descriptor, *, family):
        called.append(family)
        return [{"phase": "comparison", "pins": {}, "records": []}]

    # Admission is isolated here; arithmetic is the real frozen-kernel replay.
    monkeypatch.setattr(reader, "verify_comparison", admitted)
    result = reader.verify_selected_comparison(store, descriptor, family="A3")[-1]
    assert called == ["A3"]
    assert result["summary"]["science_count"] == 2
    assert result["summary"]["uncertainty_kind_counts"]["estimate"] == 2
    assert result["summary"]["independent_result_admission_verified"] is True
    assert result["summary"]["final_transfer_established"] is False
    assert result["summary"]["original_execution_repeated"] is False


def test_failed_admission_cannot_replay_arithmetic_or_upgrade_its_claim(monkeypatch):
    def refused(*_args, **_kwargs):
        raise EvidenceError("actual stage admission failed")

    def prohibited(*_args, **_kwargs):
        pytest.fail("arithmetic read before actual admission")

    monkeypatch.setattr(reader, "verify_comparison", refused)
    monkeypatch.setattr(reader, "replay_generic_arithmetic", prohibited)
    with pytest.raises(EvidenceError, match="stage admission"):
        reader.verify_selected_comparison(None, {}, family="T1")


@pytest.mark.parametrize(
    ("family", "message"),
    [("A2", "native A2 managed profile"), ("unknown", "pending")],
)
def test_unknown_or_absent_profile_is_refused_before_any_artifact_read(family, message):
    with pytest.raises(EvidenceError, match=message):
        reader.verify_selected_comparison(None, {}, family=family)
