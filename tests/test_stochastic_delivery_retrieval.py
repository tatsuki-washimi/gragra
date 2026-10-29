"""Finite package aliases only; all test source bytes are synthetic."""

import copy
import json

import pytest

from tests.oracles.gen1.stochastic_delivery.archive import (
    EvidenceError,
    EvidenceStore,
    archive_selected,
    materialize_checkout,
)
from tests.oracles.gen1.stochastic_delivery.retrieval import (
    archive_with_dependency_resolutions,
)
from tests.test_stochastic_delivery_evidence import closure, pin


def case(tmp_path):
    physical = tmp_path / "physical"
    physical.mkdir()
    record = physical / "RECORD"
    record.write_bytes(b"exact package metadata\n")
    alias = tmp_path / "logical"
    alias.symlink_to(physical, target_is_directory=True)
    source = pin(alias / "RECORD", "package_record")
    planned = closure(str(tmp_path))
    planned["packages"][0].update(artifact=source, artifact_missing_reason=None)
    closure_path = tmp_path / "closure.json"
    closure_path.write_text(json.dumps(planned))
    closure_pin = pin(closure_path, "dependency_closure")
    profile = {
        "schema": "gen1-stochastic-dependency-resolution-v1",
        "dependency_closures": [closure_pin],
        "entries": [
            {
                "original": source,
                "declared_path": source["path"],
                "resolved_path": str(record),
                "symlinks": [{"path": str(alias), "target": str(physical)}],
            }
        ],
    }
    profile_path = tmp_path / "resolution.json"
    profile_path.write_text(json.dumps(profile))
    profile_pin = pin(profile_path, "package_resolution")
    selection = [
        {"original": source, "role": "planned_package"},
        {"original": closure_pin, "role": "planned_closure"},
        {"original": profile_pin, "role": "retrieval_proof"},
    ]
    return source, profile, profile_path, selection


def apply_profile(tmp_path, profile, profile_path, selection):
    profile_path.write_text(json.dumps(profile))
    profile_pin = pin(profile_path, "package_resolution")
    selection = copy.deepcopy(selection)
    selection[-1]["original"] = profile_pin
    return archive_with_dependency_resolutions(
        selection, tmp_path / "archive", profile_pin
    )


def test_exact_package_alias_preserves_logical_pin_and_offline_materialization(
    tmp_path,
):
    source, profile, profile_path, selection = case(tmp_path)
    with pytest.raises(EvidenceError, match="symlink"):
        archive_selected(selection, tmp_path / "strict")
    mapping = apply_profile(tmp_path, profile, profile_path, selection)
    root = tmp_path / "archive"
    assert mapping["entries"][0]["original"] == source
    store = EvidenceStore(root, mapping)
    assert store.read(source) == b"exact package metadata\n"
    assert not (root / "sha256" / source["sha256"]).is_symlink()
    assert not (root / "sha256" / source["sha256"]).stat().st_mode & 0o222
    (tmp_path / "logical").unlink()  # No host alias needed for offline recovery.
    for entry in mapping["entries"]:
        (root / entry["archive_relative_path"]).chmod(0o644)
    destination = tmp_path / "readonly"
    materialize_checkout(root, mapping, destination)
    assert (
        EvidenceStore(destination, mapping).read(source)
        == store.root.joinpath("sha256", source["sha256"]).read_bytes()
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_schema",
        "unknown_field",
        "duplicate_entry",
        "duplicate_closure",
        "wrong_declared",
        "wrong_resolved",
        "missing_link",
        "wrong_link_target",
        "counterfeit_hash",
        "unplanned_kind",
        "unplanned_pin",
        "missing_selection",
    ],
)
def test_package_resolution_refuses_counterfeit_or_unbounded_mapping(
    tmp_path,
    mutation,
):
    _source, profile, profile_path, selection = case(tmp_path)
    entry = profile["entries"][0]
    if mutation == "unknown_schema":
        profile["schema"] = "wildcard-v1"
    elif mutation == "unknown_field":
        profile["allow_all"] = True
    elif mutation == "duplicate_entry":
        profile["entries"].append(copy.deepcopy(entry))
    elif mutation == "duplicate_closure":
        profile["dependency_closures"] *= 2
    elif mutation == "wrong_declared":
        entry["declared_path"] += "/other"
    elif mutation == "wrong_resolved":
        entry["resolved_path"] += ".other"
    elif mutation == "missing_link":
        entry["symlinks"] = []
    elif mutation == "wrong_link_target":
        entry["symlinks"][0]["target"] += ".other"
    elif mutation == "counterfeit_hash":
        (tmp_path / "physical" / "RECORD").write_bytes(b"counterfeit")
    elif mutation == "unplanned_kind":
        entry["original"]["kind"] = "reference_seal"
    elif mutation == "unplanned_pin":
        entry["original"]["sha256"] = "0" * 64
    elif mutation == "missing_selection":
        selection = selection[1:]
    with pytest.raises(EvidenceError):
        apply_profile(tmp_path, profile, profile_path, selection)
    assert not (tmp_path / "archive").exists()


def test_profile_does_not_allow_normative_artifact_alias(tmp_path):
    _source, profile, profile_path, selection = case(tmp_path)
    result = tmp_path / "result.json"
    result.write_bytes(b"sealed result")
    alias = tmp_path / "sealed-alias.json"
    alias.symlink_to(result)
    selection.insert(0, {"original": pin(alias, "result_seal"), "role": "result"})
    with pytest.raises(EvidenceError, match="symlink"):
        apply_profile(tmp_path, profile, profile_path, selection)
    assert not (tmp_path / "archive").exists()


def test_exact_planned_import_configuration_alias_is_retrieved(tmp_path):
    source, profile, profile_path, selection = case(tmp_path)
    source["kind"] = "import_configuration"
    closure_path = tmp_path / "closure.json"
    planned = json.loads(closure_path.read_text())
    planned["packages"][0].update(artifact=None, artifact_missing_reason="original")
    planned["artifacts"].append(source)
    closure_path.write_text(json.dumps(planned))
    closure_pin = pin(closure_path, "dependency_closure")
    profile["dependency_closures"] = [closure_pin]
    selection[1]["original"] = closure_pin
    mapping = apply_profile(tmp_path, profile, profile_path, selection)
    assert EvidenceStore(tmp_path / "archive", mapping).read(source) == (
        b"exact package metadata\n"
    )
