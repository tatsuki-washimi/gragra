"""V4 canonical runner routing, scope and admission regression controls."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).with_name("run_refinement_trace_validation_v4.py")


def load_runner():
    spec = importlib.util.spec_from_file_location("gen1_v4_runner_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_canonical_runner_selects_only_v4_sources_and_schema():
    runner = load_runner()

    assert runner.RECORDER.__file__.endswith("record_frozen_refinement_traces_v4.py")
    assert runner.AUDITOR.__file__.endswith("audit_frozen_refinement_traces_v4.py")
    assert runner.VERSION == "V4"
    assert runner.RUN_REPORT_SCHEMA == "gragra.gen1.refinement-trace-run-report.v4"


@pytest.mark.parametrize(
    ("datasets", "path", "sha256"),
    [
        (["cylinder"], None, None),
        (["cylinder"], "root.json", None),
        (["cylinder"], None, "a" * 64),
        (["plane"], "root.json", "a" * 64),
        (["cylinder"], "root.json", "A" * 64),
        (["cylinder"], "root.json", "not-a-sha"),
    ],
)
def test_root_supplement_arguments_fail_closed(datasets, path, sha256):
    runner = load_runner()

    with pytest.raises(ValueError, match="root supplement"):
        runner.validate_root_supplement_arguments(datasets, path, sha256)


def test_cylinder_root_supplement_path_and_independent_digest_are_required_together(
    tmp_path,
):
    runner = load_runner()
    supplement = tmp_path / "root.json"
    supplement.write_text("{}", encoding="utf-8")
    digest = runner._sha256(supplement)

    result = runner.validate_root_supplement_arguments(
        ["cylinder"], str(supplement), digest
    )

    assert result == (supplement.resolve(), digest)


@pytest.mark.parametrize(
    ("dataset", "root_args"),
    [("cylinder", []), ("plane", [])],
)
def test_missing_required_evidence_emits_not_started_incomplete_report_without_run(
    tmp_path, capsys, dataset, root_args
):
    runner = load_runner()
    output_root = tmp_path / "runs"
    run_id = "missing-evidence"

    result = runner.main(
        [
            "--l6-root",
            str(tmp_path / "missing-l6"),
            "--output-root",
            str(output_root),
            "--run-id",
            run_id,
            "--datasets",
            dataset,
            *root_args,
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    report = payload["report"]
    assert result == 3
    assert payload["run_created"] is False
    assert report["execution_state"] == "not_started"
    assert report["semantic_status"] == "SEMANTIC_NOT_RUN"
    assert report["provenance_status"] == "PROVENANCE_NOT_RUN"
    assert report["admission"] == "INCOMPLETE_EVIDENCE"
    assert not (output_root / run_id).exists()


def test_prepared_manifest_returns_validated_root_records_and_worker_scope(
    tmp_path, monkeypatch
):
    runner = load_runner()
    selected_rows = {
        "plane": {
            0: {
                "fixture": {"source": "fixture-0"},
                "reference": {"potential": "-1"},
                "metadata": {},
            }
        }
    }
    monkeypatch.setattr(runner.AUDITOR, "certificate_inputs", lambda _root: ((), ()))
    monkeypatch.setattr(
        runner,
        "_selected_scope",
        lambda *_args: (
            {"plane": {"reference_sha256": "a" * 64}},
            selected_rows,
        ),
    )

    (
        manifest,
        rows,
        root_records,
        root_identity,
        zero_certificates,
        polynomial_certificates,
    ) = runner._prepare_run_manifest(
        l6_root=tmp_path,
        datasets=["plane"],
        run_id="prepared-test",
        indices=[0],
        root_supplement_path=None,
        root_supplement_sha256=None,
        workers=1,
    )

    assert rows == selected_rows
    assert root_records == ()
    assert root_identity is None
    assert zero_certificates == ()
    assert polynomial_certificates == ()
    assert manifest["maximum_workers"] == 1
    assert manifest["source_identity"]["runner_sha256"] == runner._sha256(SCRIPT)


def test_run_campaign_rejects_changed_manifest_before_recorder_or_journal_access(
    tmp_path, monkeypatch
):
    runner = load_runner()
    output_root = tmp_path / "runs"
    run_id = "resume-manifest-check"
    run_root = output_root / run_id
    run_root.mkdir(parents=True)
    (run_root / runner.RUN_MANIFEST_FILENAME).write_text(
        json.dumps({"schema": "old", "run_id": run_id}), encoding="utf-8"
    )
    (run_root / "plane").mkdir()
    journal = run_root / "plane" / "attempts.jsonl"
    journal.write_bytes(b"do not open")
    expected_manifest = {"schema": "new", "run_id": run_id}
    row = {
        "fixture": {"id": 0},
        "reference": {"potential": "0"},
        "metadata": {},
    }

    monkeypatch.setattr(
        runner,
        "_prepare_run_manifest",
        lambda **_kwargs: (expected_manifest, {"plane": {0: row}}, (), None, (), ()),
    )
    monkeypatch.setattr(
        runner.RECORDER,
        "run_dataset",
        lambda *_args, **_kwargs: pytest.fail("recorder ran before manifest check"),
    )
    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path):
        if path == journal:
            pytest.fail("resume opened a journal before checking the run manifest")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    with pytest.raises(ValueError, match="immutable run manifest"):
        runner.run_campaign(
            l6_root=tmp_path / "l6",
            output_root=output_root,
            run_id=run_id,
            datasets=["plane"],
            resume=True,
        )

    assert original_read_bytes(journal) == b"do not open"


def _runner_row(index):
    return {
        "fixture": {"id": f"plane-fixture-{index}"},
        "reference": {"potential": str(-index - 1)},
        "metadata": {"unit": "test"},
    }


def _runner_record(recorder, index, row):
    return {
        "schema": recorder.SCHEMA,
        "dataset": "plane",
        "source_row_index": index,
        "fixture": row["fixture"],
        "frozen_reference": {
            "reference": row["reference"],
            "metadata": row["metadata"],
        },
        "status": "reproduced",
        "process_independence": {
            "pid": 1,
            "modules_before": [],
            "modules_after": [],
            "production_module_names": [],
            "production_imported": False,
            "independence_check_passed": True,
        },
        "original_entrypoint": {"captured_calls": []},
    }


def _runner_failure(recorder, index, row, message):
    return {
        "status": "replay_error",
        "exception": {
            "type": "RuntimeError",
            "message": message,
            "traceback": f"RuntimeError: {message}",
        },
        "captured_calls_before_error": [],
        "process_independence": _runner_record(recorder, index, row)[
            "process_independence"
        ],
    }


def _new_runner_journal(runner, run_root, rows):
    recorder = runner.RECORDER
    upstream = run_root.parent / "upstream-manifest.json"
    upstream.write_text('{"schema":"test"}\n', encoding="utf-8")
    dataset_manifest = {
        "schema": "gragra.gen1.refinement-trace-dataset-manifest.v4",
        "run_id": run_root.name,
        "dataset": "plane",
        "selected_source_row_indices": sorted(rows),
        "upstream_provenance": {
            "manifest": str(upstream),
            "manifest_sha256": runner._sha256(upstream),
        },
    }
    journal = recorder.AttemptJournal(
        run_root / "plane",
        run_id=run_root.name,
        dataset="plane",
        selected_rows=rows,
        manifest=dataset_manifest,
    )
    return dataset_manifest, journal


def _write_complete_runner_journal(runner, run_root):
    recorder = runner.RECORDER
    row = _runner_row(0)
    _manifest, journal = _new_runner_journal(runner, run_root, {0: row})
    record = _runner_record(recorder, 0, row)
    journal.finish(journal.start(0), (0, record, None))
    journal.settle()
    return row, journal


def test_runner_reads_complete_v4_journal_and_rejects_checkpoint_mismatch(tmp_path):
    runner = load_runner()
    run_root = tmp_path / "run-with-journal"
    row, journal = _write_complete_runner_journal(runner, run_root)

    decoded, checkpoint, complete, error = runner._read_attempt_history(
        run_root, "plane", {0: row}
    )

    assert decoded.events == journal.events
    assert checkpoint["execution_state"] == "complete"
    assert complete is True
    assert error is None

    checkpoint["accepted_source_row_indices"] = []
    journal.checkpoint_path.write_text(
        json.dumps(checkpoint, sort_keys=True), encoding="utf-8"
    )
    _decoded, _checkpoint, complete, error = runner._read_attempt_history(
        run_root, "plane", {0: row}
    )

    assert complete is False
    assert error is None


def test_runner_and_auditor_accept_completed_explicit_retry_after_interruption(
    tmp_path,
):
    runner = load_runner()
    recorder = runner.RECORDER
    run_root = tmp_path / "run-after-interrupted-retry"
    row = _runner_row(0)
    dataset_manifest, journal = _new_runner_journal(runner, run_root, {0: row})
    failed = _runner_failure(recorder, 0, row, "initial worker failure")
    journal.finish(journal.start(0), (0, failed, "initial worker failure"))
    journal.settle()
    journal.begin_retry()
    journal.start(0)
    journal.settle(interrupted=True)

    resumed = recorder.AttemptJournal(
        run_root / "plane",
        run_id=run_root.name,
        dataset="plane",
        selected_rows={0: row},
        manifest=dataset_manifest,
        resume=True,
    )
    record = _runner_record(recorder, 0, row)
    recorder.execute_attempts(
        resumed,
        lambda index: (index,),
        retry_failed=True,
        worker=lambda payload: (payload[0], record, None),
    )

    decoded, checkpoint, runner_complete, error = runner._read_attempt_history(
        run_root, "plane", {0: row}
    )
    independent = runner.AUDITOR.audit_attempt_journal(
        "plane",
        0,
        row,
        record,
        SimpleNamespace(run_id=run_root.name),
        decoded,
        {0: row},
    )
    assert checkpoint["execution_state"] == "complete"
    assert len(checkpoint["interrupted_attempts"]) == 1
    assert runner_complete is True
    assert error is None
    assert independent["complete"] is True
    assert (
        checkpoint["interrupted_attempts"][0]["retry_of"]
        == decoded.events[0]["attempt_id"]
    )
    assert decoded.events[-1]["retry_of"] == decoded.events[0]["attempt_id"]

    aggregate = runner.aggregate_admission(
        [
            {
                "semantic_validation": "SEMANTIC_VALID",
                "provenance_validation": {"status": "PROVENANCE_VERIFIED"},
                "journal_binding_complete": independent["complete"],
                "certificate_validation": "CERTIFICATE_SCOPE_VALID",
                "audit_unresolved_trends": [],
            }
        ],
        execution_state=checkpoint["execution_state"],
        required_evidence_complete=runner_complete,
        selected_row_count=1,
        prior_failures=True,
        accepted_retry_result=independent["accepted_retry_result"],
    )
    assert aggregate["admission"] == "ADMITTED_WITH_DECLARED_SCOPE"


@pytest.mark.parametrize("selected_row_count", [0, 1])
def test_runner_aggregate_rejects_empty_selected_scope(selected_row_count):
    runner = load_runner()

    aggregate = runner.aggregate_admission(
        [],
        execution_state="complete",
        required_evidence_complete=True,
        selected_row_count=selected_row_count,
    )

    assert aggregate["semantic_status"] == "SEMANTIC_NOT_RUN"
    assert aggregate["provenance_status"] == "PROVENANCE_NOT_RUN"
    assert aggregate["checks"]["declared_scope_valid"] is False
    assert aggregate["admission"] == "NOT_ADMITTED"


def test_empty_selection_campaign_returns_nonadmitted_report(tmp_path, monkeypatch):
    runner = load_runner()
    run_id = "empty-selection"
    manifest = {
        "schema": runner.CONTRACT.MANIFEST_SCHEMA,
        "run_id": run_id,
        "datasets": ["plane"],
        "dataset_scope": {"plane": {"selected_row_indices": []}},
        "source_identity": runner._source_identity(),
        "schema_identity": dict(runner.CONTRACT.V4_SCHEMA_IDENTITIES),
        "environment": runner._environment_profile(),
        "numeric_representation": runner._numeric_profile(),
    }
    monkeypatch.setattr(
        runner,
        "_prepare_run_manifest",
        lambda **_kwargs: (manifest, {"plane": {}}, (), None, (), ()),
    )

    def reject_empty_selection(*_args, **_kwargs):
        raise ValueError("journal requires selected rows")

    monkeypatch.setattr(
        runner.RECORDER,
        "run_dataset",
        reject_empty_selection,
    )

    report, evidence = runner.run_campaign(
        l6_root=tmp_path / "l6",
        output_root=tmp_path / "runs",
        run_id=run_id,
        datasets=["plane"],
    )

    assert report["semantic_status"] == "SEMANTIC_NOT_RUN"
    assert report["provenance_status"] == "PROVENANCE_NOT_RUN"
    assert report["admission"] == "NOT_ADMITTED"
    assert evidence["admission"] == "NOT_ADMITTED"


def test_empty_scope_guard_preserves_known_semantic_and_provenance_invalidity():
    runner = load_runner()

    aggregate = runner.aggregate_admission(
        [
            {
                "semantic_validation": "SEMANTIC_INVALID",
                "provenance_validation": {"status": "PROVENANCE_MISMATCH"},
                "journal_binding_complete": False,
                "certificate_validation": "CERTIFICATE_SCOPE_INVALID",
                "audit_unresolved_trends": [],
            }
        ],
        execution_state="complete",
        required_evidence_complete=True,
        selected_row_count=0,
    )

    assert aggregate["semantic_status"] == "SEMANTIC_INVALID"
    assert aggregate["provenance_status"] == "PROVENANCE_MISMATCH"
    assert aggregate["admission"] == "NOT_ADMITTED"


def test_runner_rejects_interrupted_row_when_only_another_row_succeeded(tmp_path):
    runner = load_runner()
    recorder = runner.RECORDER
    run_root = tmp_path / "run-with-unresolved-row"
    rows = {index: _runner_row(index) for index in (0, 1)}
    _manifest, journal = _new_runner_journal(runner, run_root, rows)

    def success_record(index):
        return _runner_record(recorder, index, rows[index])

    failure = _runner_failure(recorder, 0, rows[0], "row 0 worker failure")
    attempt_0, attempt_1 = journal.start(0), journal.start(1)
    journal.finish(attempt_0, (0, failure, "row 0 worker failure"))
    journal.finish(attempt_1, (1, success_record(1), None))
    journal.settle()
    journal.begin_retry()
    journal.start(0)
    journal.settle(interrupted=True)

    decoded, checkpoint, complete, error = runner._read_attempt_history(
        run_root, "plane", rows
    )
    independent = runner.AUDITOR.audit_attempt_journal(
        "plane",
        1,
        rows[1],
        success_record(1),
        SimpleNamespace(run_id=run_root.name),
        decoded,
        rows,
    )

    assert checkpoint["accepted_source_row_indices"] == [1]
    assert checkpoint["execution_state"] == "interrupted"
    assert complete is False
    assert error is None
    assert independent["complete"] is False


def test_runner_recomputes_event_digest_after_checkpoint_digest_reconciliation(
    tmp_path,
):
    runner = load_runner()
    run_root = tmp_path / "run-with-altered-event"
    _row, _journal = _write_complete_runner_journal(runner, run_root)
    journal_path = run_root / "plane" / "attempts.jsonl"
    checkpoint_path = run_root / "plane" / "checkpoint.json"
    lines = journal_path.read_text(encoding="utf-8").splitlines()
    event = json.loads(lines[0])
    event["record"]["status"] = "unresolved_trend"
    lines[0] = json.dumps(event, sort_keys=True, allow_nan=False)
    changed_journal = ("\n".join(lines) + "\n").encode("utf-8")
    journal_path.write_bytes(changed_journal)
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["committed_journal_bytes"] = len(changed_journal)
    checkpoint["committed_journal_sha256"] = hashlib.sha256(changed_journal).hexdigest()
    checkpoint_path.write_text(json.dumps(checkpoint, sort_keys=True), encoding="utf-8")

    _decoded, _checkpoint, complete, error = runner._read_attempt_history(
        run_root,
        "plane",
        {
            0: {
                "fixture": {"id": "plane-fixture-0"},
                "reference": {"potential": "-1"},
                "metadata": {"unit": "test"},
            }
        },
    )

    assert complete is False
    assert error is not None
    assert "event digest" in error


def test_v4_root_reader_accepts_only_digest_pinned_historical_source(
    tmp_path, monkeypatch
):
    runner = load_runner()
    auditor = runner.AUDITOR
    import mpmath as mp

    l6_root = tmp_path / "l6"
    (l6_root / auditor.DATASETS["cylinder"]).mkdir(parents=True)
    fixture = {
        "order": 1,
        "radial_index": 0,
        "radius_m": 1.0,
        "axial_index": 0,
    }
    rows = [
        {"fixture": dict(fixture), "metadata": {"root": "2.0", "normalization": "3.0"}}
        for _ in range(576)
    ]
    monkeypatch.setattr(auditor, "read_rows", lambda _path: rows)
    monkeypatch.setattr(auditor, "decode", lambda item: mp.mpf(item["decimal"]))
    monkeypatch.setattr(auditor, "root_precision", lambda *_args: mp.mpf("1"))
    entry = {
        "key": [1, 0, 1.0, 0],
        "source_rows": [
            {"source_row_index": index, "fixture": dict(fixture)}
            for index in range(576)
        ],
        "at_80_digits": [{"decimal": "2.0"}, {"decimal": "3.0"}],
        "at_50_digits": [{"decimal": "2.0"}, {"decimal": "3.0"}],
        "root_precision_estimate": "1.0",
    }
    payload = {
        "schema": "gragra.gen1.cylinder-root-normalization-supplement.v1",
        "classification": "observed supplemental root-only reconstruction",
        "method": "pinned cylinder_radial_data at 50/80; estimate at 80 digits",
        "generator_sha256": auditor.sha256(
            auditor.ORACLE_DIRECTORY / "record_cylinder_root_normalization.py"
        ),
        "audit_helper_sha256": auditor.ROOT_SUPPLEMENT_AUDIT_HELPER_SHA256,
        "oracle_sha256": auditor.EXPECTED_ORACLE_SHA256,
        "reference_sha256": auditor.EXPECTED_REFERENCE_SHA256["cylinder"],
        "original_values_uncertainty_labels_unchanged": True,
        "original_d43_callback_capture": False,
        "finite_precision_or_total_error_bound_claim": False,
        "fixture_count": 576,
        "unique_key_count": 1,
        "entries": [entry],
        "production_modules": [],
        "command": [],
        "environment": {},
    }
    path = tmp_path / "root.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    loaded = auditor.load_root_supplement(path, auditor.sha256(path), l6_root)

    assert loaded == [entry]
    payload["oracle_sha256"] = {"surface_reference.py": "0" * 64}
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="source/reference pins"):
        auditor.load_root_supplement(path, auditor.sha256(path), l6_root)


def test_v4_root_reader_rejects_duplicate_json_keys_even_with_matching_file_digest(
    tmp_path,
):
    runner = load_runner()
    auditor = runner.AUDITOR
    root_path = tmp_path / "duplicate.json"
    root_path.write_text('{"schema":"first","schema":"second"}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        auditor.load_root_supplement(root_path, auditor.sha256(root_path), tmp_path)


def test_new_run_rejects_reused_directory_without_touching_contents(tmp_path):
    runner = load_runner()
    run_root = tmp_path / "run-1"
    run_root.mkdir()
    sentinel = run_root / "sentinel"
    sentinel.write_bytes(b"preserve")

    with pytest.raises(ValueError, match="new run"):
        runner.validate_run_location(tmp_path, "run-1", resume=False)

    assert sentinel.read_bytes() == b"preserve"
    assert sorted(path.name for path in run_root.iterdir()) == ["sentinel"]


def test_resume_manifest_mismatch_is_rejected_without_reading_journal(
    tmp_path, monkeypatch
):
    runner = load_runner()
    run_root = tmp_path / "existing"
    run_root.mkdir()
    (run_root / "run-manifest-v4.json").write_text(
        json.dumps({"schema": "old", "run_id": "existing"}), encoding="utf-8"
    )
    journal = run_root / "plane" / "attempts.jsonl"
    journal.parent.mkdir()
    journal.write_bytes(b"untouched journal")
    expected = {"schema": "new", "run_id": "existing"}
    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path):
        if path == journal:
            pytest.fail("journal was opened before immutable manifest comparison")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    with pytest.raises(ValueError, match="immutable run manifest"):
        runner.validate_resume_manifest(run_root, expected)

    assert original_read_bytes(journal) == b"untouched journal"


@pytest.mark.parametrize(
    ("rows", "execution", "required", "semantic", "provenance", "expected"),
    [
        (
            [
                {
                    "semantic_validation": "SEMANTIC_VALID",
                    "provenance_validation": {"status": "PROVENANCE_UNVERIFIED"},
                    "journal_binding_complete": True,
                    "certificate_validation": "CERTIFICATE_SCOPE_VALID",
                    "audit_unresolved_trends": [],
                }
            ],
            "complete",
            True,
            "SEMANTIC_VALID",
            "PROVENANCE_UNVERIFIED",
            "NOT_ADMITTED",
        ),
        (
            [
                {
                    "semantic_validation": "SEMANTIC_INVALID",
                    "provenance_validation": {"status": "PROVENANCE_UNVERIFIED"},
                }
            ],
            "complete",
            False,
            "SEMANTIC_INVALID",
            "PROVENANCE_UNVERIFIED",
            "NOT_ADMITTED",
        ),
        (
            [
                {
                    "semantic_validation": "SEMANTIC_NOT_RUN",
                    "provenance_validation": {"status": "PROVENANCE_NOT_RUN"},
                }
            ],
            "interrupted",
            False,
            "SEMANTIC_NOT_RUN",
            "PROVENANCE_NOT_RUN",
            "INCOMPLETE_EVIDENCE",
        ),
        (
            [
                {
                    "semantic_validation": "SEMANTIC_VALID",
                    "provenance_validation": {"status": "PROVENANCE_MISMATCH"},
                }
            ],
            "complete",
            True,
            "SEMANTIC_VALID",
            "PROVENANCE_MISMATCH",
            "NOT_ADMITTED",
        ),
    ],
)
def test_runner_aggregate_preserves_invalid_unverified_and_incomplete_states(
    rows, execution, required, semantic, provenance, expected
):
    runner = load_runner()

    summary = runner.aggregate_admission(
        rows,
        execution_state=execution,
        required_evidence_complete=required,
        selected_row_count=1,
    )

    assert summary["semantic_status"] == semantic
    assert summary["provenance_status"] == provenance
    assert summary["admission"] == expected


def test_legacy_full_gate_boolean_is_not_a_v4_admission_source():
    runner = load_runner()
    v1 = {
        "schema": "gragra.gen1.refinement-trace-report.v1",
        "full_gate_complete": True,
    }
    v3 = {
        "schema": "gragra.gen1.refinement-trace-report.v3",
        "full_gate_complete": True,
    }

    for report in (v1, v3):
        with pytest.raises(ValueError, match="V4"):
            runner.CONTRACT.validate_v4_report(report, expected_source_identity={})
