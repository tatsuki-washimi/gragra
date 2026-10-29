"""Canonical V4 entrypoint for new Gen1 refinement-trace campaigns.

Execution, semantic validation, provenance, and scientific admission are reported
separately. This repository has no production external-anchor provider, so a
semantic pass without a separately trusted anchor remains NOT_ADMITTED.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import platform
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ORACLE_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY = Path(__file__).resolve().parents[3]
DEFAULT_RECORDER = ORACLE_DIRECTORY / "record_frozen_refinement_traces_v4.py"
DEFAULT_AUDITOR = ORACLE_DIRECTORY / "audit_frozen_refinement_traces_v4.py"
DEFAULT_CONTRACT = ORACLE_DIRECTORY / "refinement_trace_provenance_v4.py"
VERSION = "V4"
RUN_REPORT_SCHEMA = "gragra.gen1.refinement-trace-run-report.v4"
RUN_MANIFEST_FILENAME = "run-manifest-v4.json"


class MissingEvidenceError(ValueError):
    """A required external input is absent before any run state is created."""


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pinned V4 source: {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RECORDER = _load("_gen1_v4_default_recorder", DEFAULT_RECORDER)
AUDITOR = _load("_gen1_v4_default_auditor", DEFAULT_AUDITOR)
CONTRACT = AUDITOR.PROVENANCE


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    return CONTRACT.strict_json_object(Path(path).read_bytes())


def _atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    CONTRACT.canonical_json_bytes(value)
    data = (
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_root_supplement_arguments(datasets, path, sha256):
    """Require an independently pinned root asset exactly for Cylinder scope."""
    cylinder = "cylinder" in datasets
    if cylinder and (path is None or sha256 is None):
        raise MissingEvidenceError(
            "Cylinder root supplement path and SHA256 are required"
        )
    if not cylinder and (path is not None or sha256 is not None):
        raise ValueError(
            "Cylinder root supplement path and SHA256 must be supplied together"
        )
    if not cylinder:
        return None
    if type(sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
        raise ValueError(
            "Cylinder root supplement SHA256 must be lowercase hexadecimal"
        )
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise MissingEvidenceError("Cylinder root supplement file is unavailable")
    if _sha256(resolved) != sha256:
        raise ValueError("Cylinder root supplement digest mismatch")
    return resolved, sha256


def validate_run_location(output_root, run_id, *, resume):
    output_root = Path(output_root).expanduser().resolve()
    if output_root.is_relative_to(REPOSITORY):
        raise ValueError("raw trace output must remain outside Git")
    if (
        type(run_id) is not str
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", run_id) is None
        or ".." in run_id
    ):
        raise ValueError("run ID must be a new safe identifier without path components")
    run_root = output_root / run_id
    if run_root.is_symlink():
        raise ValueError("run directory may not be a symbolic link")
    if resume:
        if not run_root.is_dir():
            raise ValueError("V4 resume requires an existing run ID directory")
    elif run_root.exists():
        raise ValueError("new run requires a new unused run ID and output path")
    return run_root


def validate_resume_manifest(run_root, expected_manifest):
    """Compare the outer immutable manifest before any dataset journal is read."""
    path = Path(run_root) / RUN_MANIFEST_FILENAME
    try:
        stored = _strict_json(path)
    except (OSError, UnicodeError, ValueError, TypeError) as error:
        raise ValueError(f"immutable run manifest unavailable: {error}") from error
    if CONTRACT.canonical_json_bytes(stored) != CONTRACT.canonical_json_bytes(
        expected_manifest
    ):
        raise ValueError("immutable run manifest changed; use a new run ID")
    return stored


def _source_identity():
    return {
        "observer_sha256": _sha256(DEFAULT_RECORDER),
        "auditor_sha256": _sha256(DEFAULT_AUDITOR),
        "provenance_sha256": _sha256(DEFAULT_CONTRACT),
        "runner_sha256": _sha256(Path(__file__).resolve()),
    }


def _environment_profile():
    # Avoid recording the user's absolute Python executable path in run metadata.
    return {
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "mpmath_version": RECORDER.mp.__version__,
        "numpy_version": RECORDER.np.__version__,
    }


def _numeric_profile():
    return {
        "mpf_encoding": "mpmath internal (sign, mantissa, exponent, bitcount) tuple",
        "decimal_display": "validated display only; not an arithmetic precision claim",
        "float_encoding": "JSON finite binary64 values retained as parsed",
        "callback_precision": "declared per serialized callback input and MPF tuple",
        "ambient_mpmath_dps_at_run_start": int(RECORDER.mp.mp.dps),
    }


def _selected_scope(l6_root, datasets, indices):
    scope, selected_rows = {}, {}
    for dataset in datasets:
        reference_path = Path(l6_root) / RECORDER.DATASETS[dataset] / "references.jsonl"
        upstream_path = Path(l6_root) / RECORDER.DATASETS[dataset] / "manifest.json"
        if _sha256(reference_path) != RECORDER.EXPECTED_REFERENCE_SHA256[dataset]:
            raise ValueError(f"accepted {dataset} reference SHA256 mismatch")
        rows = AUDITOR.read_rows(reference_path)
        all_indices = [index for index, _ in enumerate(rows)]
        chosen = RECORDER._validated_selected_indices(
            all_indices if indices is None else indices, all_indices
        )
        row_map = {index: rows[index] for index in chosen}
        upstream = _strict_json(upstream_path)
        selected_rows[dataset] = row_map
        scope[dataset] = {
            "reference_sha256": _sha256(reference_path),
            "upstream_manifest_sha256": _sha256(upstream_path),
            "upstream_manifest_schema": upstream.get("schema"),
            "total_rows": len(rows),
            "selected_row_indices": chosen,
            "selected_rows_sha256": CONTRACT.canonical_sha256(
                {str(index): row for index, row in row_map.items()}
            ),
        }
    return scope, selected_rows


def _load_root_evidence(path, expected_sha256, l6_root, selected_rows):
    if path is None:
        return (), None
    entries = AUDITOR.load_root_supplement(path, expected_sha256, Path(l6_root))
    document = CONTRACT.strict_json_object(Path(path).read_bytes())
    cylinder_rows = selected_rows["cylinder"]
    scope_document = {
        "source_schema": document["schema"],
        "reference_sha256": document["reference_sha256"],
        "selected_rows": [
            {"source_row_index": index, "fixture": row["fixture"]}
            for index, row in cylinder_rows.items()
        ],
        "validated_root_entries": entries,
    }
    identity = {
        "schema": CONTRACT.ROOT_SUPPLEMENT_SCHEMA,
        "sha256": expected_sha256,
        "scope_sha256": CONTRACT.canonical_sha256(scope_document),
    }
    return entries, identity


def _prepare_run_manifest(
    *,
    l6_root,
    datasets,
    run_id,
    indices,
    root_supplement_path,
    root_supplement_sha256,
    workers,
):
    datasets = list(datasets)
    if not datasets or len(datasets) != len(set(datasets)):
        raise ValueError("datasets must be a nonempty unique selection")
    root_pair = validate_root_supplement_arguments(
        datasets, root_supplement_path, root_supplement_sha256
    )
    if root_pair is not None:
        root_supplement_path, root_supplement_sha256 = root_pair
    zero_certificates, polynomial_certificates = AUDITOR.certificate_inputs(
        Path(l6_root)
    )
    dataset_scope, selected_rows = _selected_scope(l6_root, datasets, indices)
    root_records, root_identity = _load_root_evidence(
        root_supplement_path, root_supplement_sha256, l6_root, selected_rows
    )
    source_identity = _source_identity()
    oracle_pins = dict(AUDITOR.EXPECTED_ORACLE_SHA256)
    for name, expected in oracle_pins.items():
        source_path = ORACLE_DIRECTORY / name
        if _sha256(source_path) != expected:
            raise ValueError(f"pinned independent oracle changed: {name}")
    manifest = {
        "schema": CONTRACT.MANIFEST_SCHEMA,
        "run_id": run_id,
        "datasets": datasets,
        "dataset_scope": dataset_scope,
        "source_identity": source_identity,
        "schema_identity": dict(CONTRACT.V4_SCHEMA_IDENTITIES),
        "oracle_module_sha256": oracle_pins,
        "zero_certificate_sha256": AUDITOR.ZERO_SHA256,
        "polynomial_certificate_sha256": AUDITOR.POLYNOMIAL_SHA256,
        "root_supplement": root_identity,
        "environment": _environment_profile(),
        "numeric_representation": _numeric_profile(),
        "maximum_workers": workers,
        "retry_policy": "failed attempts require explicit --retry-failed on resume",
    }
    CONTRACT.canonical_json_bytes(manifest)
    return (
        manifest,
        selected_rows,
        root_records,
        root_identity,
        zero_certificates,
        polynomial_certificates,
    )


def _read_attempt_history(run_root, dataset, selected_rows):
    target = Path(run_root) / dataset
    journal_path = target / "attempts.jsonl"
    checkpoint_path = target / "checkpoint.json"
    manifest_path = target / "manifest.json"
    phase_lock_path = target / ".attempt-phase.lock"
    if (
        not journal_path.is_file()
        or not checkpoint_path.is_file()
        or not manifest_path.is_file()
        or not phase_lock_path.is_file()
    ):
        return None, None, False, "attempt journal or checkpoint missing"
    try:
        phase_lock = phase_lock_path.open("a")
        fcntl.flock(phase_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return None, None, False, "attempt journal has an active writer"
    try:
        return _read_attempt_history_locked(
            dataset, selected_rows, manifest_path, checkpoint_path, journal_path
        )
    finally:
        fcntl.flock(phase_lock.fileno(), fcntl.LOCK_UN)
        phase_lock.close()


def _independent_journal_completion(dataset, selected_rows, manifest, journal):
    if not selected_rows:
        return False
    index = next(iter(selected_rows))
    record = next(
        (
            event.get("record")
            for event in journal.events
            if event.get("schema") == RECORDER.ATTEMPT_SUCCESS_SCHEMA
            and event.get("source_row_index") == index
        ),
        None,
    )
    if type(record) is not dict:
        return False
    try:
        # The journal auditor reads only run_id from this value; it is not
        # provenance evidence or a trusted run profile for scientific admission.
        result = AUDITOR.audit_attempt_journal(
            dataset,
            index,
            selected_rows[index],
            record,
            SimpleNamespace(run_id=manifest.get("run_id")),
            journal,
            selected_rows,
        )
    except (KeyError, TypeError, ValueError):
        return False
    return result.get("complete") is True


def _read_attempt_history_locked(
    dataset, selected_rows, manifest_path, checkpoint_path, journal_path
):
    try:
        manifest = _strict_json(manifest_path)
        checkpoint = _strict_json(checkpoint_path)
    except (OSError, UnicodeError, ValueError, TypeError) as error:
        return None, None, False, f"dataset manifest or checkpoint invalid: {error}"
    if (
        manifest.get("schema") != "gragra.gen1.refinement-trace-dataset-manifest.v4"
        or manifest.get("dataset") != dataset
        or type(manifest.get("selected_source_row_indices")) is not list
        or manifest.get("selected_source_row_indices") != sorted(selected_rows)
    ):
        return None, checkpoint, False, "dataset manifest scope mismatch"
    upstream = manifest.get("upstream_provenance")
    if (
        type(upstream) is not dict
        or type(upstream.get("manifest")) is not str
        or not Path(upstream["manifest"]).is_file()
        or _sha256(Path(upstream["manifest"])) != upstream.get("manifest_sha256")
    ):
        return None, checkpoint, False, "upstream manifest identity mismatch"
    try:
        raw = journal_path.read_bytes()
    except OSError as error:
        return None, checkpoint, False, f"attempt journal unavailable: {error}"
    committed_bytes = checkpoint.get("committed_journal_bytes")
    if type(committed_bytes) is not int or not 0 <= committed_bytes <= len(raw):
        return None, checkpoint, False, "committed journal byte count invalid"
    committed = raw[:committed_bytes]
    if (
        hashlib.sha256(committed).hexdigest()
        != checkpoint.get("committed_journal_sha256")
        or len(raw) != committed_bytes
    ):
        return None, checkpoint, False, "committed journal bytes differ from checkpoint"
    if committed and not committed.endswith(b"\n"):
        return None, checkpoint, False, "committed journal has an incomplete JSONL line"
    try:
        events = [
            CONTRACT.strict_json_object(line) for line in committed.splitlines() if line
        ]
    except (UnicodeError, ValueError, TypeError) as error:
        return None, checkpoint, False, f"committed journal JSON invalid: {error}"
    states = {}
    accepted = {}
    accepted_rows_from_events = set()
    failed_attempts_by_row = {}
    terminal_attempts = {}
    for event in events:
        attempt = event.get("attempt_id")
        index = event.get("source_row_index")
        schema = event.get("schema")
        expected_event_digest = CONTRACT.canonical_sha256(
            {key: value for key, value in event.items() if key != "event_sha256"}
        )
        if (
            type(attempt) is not str
            or not attempt
            or attempt in states
            or type(index) is not int
            or index not in selected_rows
            or type(schema) is not str
            or schema
            not in {
                RECORDER.ATTEMPT_SUCCESS_SCHEMA,
                RECORDER.ATTEMPT_FAILURE_SCHEMA,
            }
            or event.get("event_sha256") != expected_event_digest
            or event.get("retry_of") != failed_attempts_by_row.get(index)
        ):
            return (
                None,
                checkpoint,
                False,
                "attempt event digest, identity, or retry lineage invalid",
            )
        state = "complete" if schema == RECORDER.ATTEMPT_SUCCESS_SCHEMA else "failed"
        if index in accepted_rows_from_events:
            return None, checkpoint, False, "attempt journal repeats an accepted row"
        states[attempt] = {
            "state": state,
            "source_row_index": index,
            "retry_of": event.get("retry_of"),
        }
        terminal_attempts[attempt] = states[attempt]
        if state == "complete":
            accepted_rows_from_events.add(index)
            failed_attempts_by_row.pop(index, None)
            if type(event.get("record")) is dict:
                accepted[f"{dataset}:{index}"] = {
                    "attempt_id": attempt,
                    "result_sha256": CONTRACT.canonical_sha256(event["record"]),
                }
        else:
            failed_attempts_by_row[index] = attempt

    active_attempts = checkpoint.get("active_attempts")
    interrupted_attempts = checkpoint.get("interrupted_attempts")
    if type(active_attempts) is not dict or type(interrupted_attempts) is not list:
        return None, checkpoint, False, "attempt checkpoint state invalid"
    seen_attempt_ids = set(states)
    for row_key, active_attempt in active_attempts.items():
        try:
            index = int(row_key)
        except (TypeError, ValueError):
            return None, checkpoint, False, "active attempt row identity invalid"
        if (
            type(row_key) is not str
            or str(index) != row_key
            or index not in selected_rows
            or type(active_attempt) is not dict
            or set(active_attempt) != {"attempt_id", "retry_of"}
        ):
            return None, checkpoint, False, "active attempt identity invalid"
        attempt = active_attempt.get("attempt_id")
        if (
            type(attempt) is not str
            or not attempt
            or attempt in seen_attempt_ids
            or index in accepted_rows_from_events
            or active_attempt.get("retry_of") != failed_attempts_by_row.get(index)
        ):
            return None, checkpoint, False, "active attempt lineage invalid"
        seen_attempt_ids.add(attempt)
        states[attempt] = {
            "state": "running",
            "source_row_index": index,
            "retry_of": active_attempt.get("retry_of"),
        }

    for interrupted_attempt in interrupted_attempts:
        if type(interrupted_attempt) is not dict or set(interrupted_attempt) != {
            "attempt_id",
            "retry_of",
            "source_row_index",
        }:
            return None, checkpoint, False, "interrupted attempt identity invalid"
        attempt = interrupted_attempt.get("attempt_id")
        index = interrupted_attempt.get("source_row_index")
        retry_of = interrupted_attempt.get("retry_of")
        if retry_of is not None and type(retry_of) is not str:
            return None, checkpoint, False, "interrupted attempt lineage invalid"
        predecessor = terminal_attempts.get(retry_of) if retry_of is not None else None
        if (
            type(attempt) is not str
            or not attempt
            or attempt in seen_attempt_ids
            or type(index) is not int
            or index not in selected_rows
            or retry_of is not None
            and (
                predecessor is None
                or predecessor["state"] != "failed"
                or predecessor["source_row_index"] != index
            )
        ):
            return None, checkpoint, False, "interrupted attempt lineage invalid"
        seen_attempt_ids.add(attempt)
        states[attempt] = {
            "state": "interrupted",
            "source_row_index": index,
            "retry_of": retry_of,
        }
    retry_plan_history = checkpoint.get("retry_plan_history", [])
    if type(retry_plan_history) is not list:
        return None, checkpoint, False, "retry plan history invalid"
    plans = list(retry_plan_history)
    current = checkpoint.get("retry_plan")
    if current is not None:
        plans.append(current)
    journal = CONTRACT.AttemptJournalInputs(
        events=events,
        attempt_states=states,
        retry_plan=plans,
        accepted_results=accepted,
    )
    journal_auditor_complete = _independent_journal_completion(
        dataset, selected_rows, manifest, journal
    )
    expected_manifest_pin = hashlib.sha256(
        (
            json.dumps(
                {
                    key: value
                    for key, value in manifest.items()
                    if key != "process_state"
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    expected_row_pins = {
        str(index): hashlib.sha256(
            json.dumps(
                row, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()
        for index, row in selected_rows.items()
    }
    accepted_rows, failed_rows = set(), {}
    for event in events:
        index = event.get("source_row_index")
        if event.get("schema") == RECORDER.ATTEMPT_SUCCESS_SCHEMA:
            accepted_rows.add(index)
            failed_rows.pop(index, None)
        else:
            failed_rows[index] = event.get("attempt_id")
    # Retained interrupted attempts are historical lineage, not active work.
    # Each must still belong to a selected row that now has an accepted result.
    checkpoint_complete = (
        checkpoint.get("schema") == RECORDER.CHECKPOINT_SCHEMA
        and checkpoint.get("run_id") == manifest.get("run_id")
        and checkpoint.get("dataset") == dataset
        and checkpoint.get("dataset_manifest_pin_sha256") == expected_manifest_pin
        and checkpoint.get("selected_source_row_indices") == sorted(selected_rows)
        and checkpoint.get("selected_row_sha256") == expected_row_pins
        and checkpoint.get("event_sequence") == len(events)
        and checkpoint.get("event_digests")
        == [event.get("event_sha256") for event in events]
        and journal_auditor_complete
        and checkpoint.get("execution_state") == "complete"
        and checkpoint.get("accepted_source_row_indices") == sorted(accepted_rows)
        and accepted_rows == set(selected_rows)
        and checkpoint.get("failed_source_row_indices") == sorted(failed_rows)
        and not failed_rows
        and not checkpoint.get("failed_source_row_indices")
        and not active_attempts
        and all(
            attempt["source_row_index"] in accepted_rows
            for attempt in interrupted_attempts
        )
        and checkpoint.get("retry_plan") is None
    )
    return journal, checkpoint, bool(checkpoint_complete), None


def _kernel_identities(dataset):
    # This V4 mapping intentionally commits each observable to the complete
    # independently pinned oracle module set, not to a trace-supplied kernel tag.
    pins_sha256 = CONTRACT.canonical_sha256(AUDITOR.EXPECTED_ORACLE_SHA256)
    return {
        name: CONTRACT.canonical_sha256(
            {
                "dataset": dataset,
                "observable": name,
                "oracle_module_set_sha256": pins_sha256,
            }
        )
        for name in AUDITOR.SECTIONS
    }


def _independent_inputs(dataset, index, row, root_identity):
    return CONTRACT.IndependentAuditInputs(
        dataset=dataset,
        dataset_sha256=AUDITOR.EXPECTED_REFERENCE_SHA256[dataset],
        source_row_index=index,
        fixture=row["fixture"],
        reference=row["reference"],
        required_observables=tuple(sorted(AUDITOR.SECTIONS)),
        oracle_module_sha256=CONTRACT.canonical_sha256(AUDITOR.EXPECTED_ORACLE_SHA256),
        kernel_identities=_kernel_identities(dataset),
        frozen_reference_sha256=AUDITOR.EXPECTED_REFERENCE_SHA256[dataset],
        certificate_sha256=tuple(
            sorted(
                {
                    AUDITOR.ZERO_SHA256,
                    *(
                        [AUDITOR.POLYNOMIAL_SHA256]
                        if dataset in {"source", "noncavity"}
                        else []
                    ),
                }
            )
        ),
        root_supplement=root_identity if dataset == "cylinder" else None,
    )


def aggregate_admission(
    row_results,
    *,
    execution_state,
    required_evidence_complete,
    selected_row_count,
    prior_failures=False,
    accepted_retry_result=False,
):
    """Aggregate states without allowing a legacy or partial row to promote."""
    rows = list(row_results)
    semantic_values = [item.get("semantic_validation") for item in rows]
    # Empty selection must never inherit truth from all(empty) checks below.
    invalid_scope = (
        not rows or type(selected_row_count) is not int or selected_row_count <= 0
    )
    if "SEMANTIC_INVALID" in semantic_values:
        semantic = "SEMANTIC_INVALID"
    elif invalid_scope:
        semantic = "SEMANTIC_NOT_RUN"
    elif len(rows) == selected_row_count and all(
        value == "SEMANTIC_VALID" for value in semantic_values
    ):
        semantic = "SEMANTIC_VALID"
    else:
        semantic = "SEMANTIC_NOT_RUN"
    provenance_values = [
        item.get("provenance_validation", {}).get("status") for item in rows
    ]
    if "PROVENANCE_MISMATCH" in provenance_values:
        provenance = "PROVENANCE_MISMATCH"
    elif invalid_scope:
        provenance = "PROVENANCE_NOT_RUN"
    elif (
        semantic == "SEMANTIC_VALID"
        and len(rows) == selected_row_count
        and all(value == "PROVENANCE_VERIFIED" for value in provenance_values)
    ):
        provenance = "PROVENANCE_VERIFIED"
    elif rows and any(
        value in {"PROVENANCE_UNVERIFIED", "PROVENANCE_VERIFIED"}
        for value in provenance_values
    ):
        provenance = "PROVENANCE_UNVERIFIED"
    else:
        provenance = "PROVENANCE_NOT_RUN"
    binding_complete = (
        not invalid_scope
        and len(rows) == selected_row_count
        and all(item.get("journal_binding_complete") is True for item in rows)
    )
    independent_valid = (
        False
        if semantic == "SEMANTIC_INVALID"
        else True
        if semantic == "SEMANTIC_VALID"
        else None
    )
    certificate_valid = (
        False
        if any(
            item.get("certificate_validation") == "CERTIFICATE_SCOPE_INVALID"
            for item in rows
        )
        else True
        if semantic == "SEMANTIC_VALID"
        else None
    )
    trend_complete = (
        not invalid_scope
        and semantic == "SEMANTIC_VALID"
        and all(not item.get("audit_unresolved_trends") for item in rows)
    )
    declared_scope_valid = (
        False
        if invalid_scope or semantic == "SEMANTIC_INVALID"
        else (True if semantic == "SEMANTIC_VALID" else None)
    )
    checks = {
        "required_evidence_complete": bool(required_evidence_complete),
        "independent_audit_valid": independent_valid,
        "certificate_validation_valid": certificate_valid,
        "trend_validation_complete": trend_complete,
        "declared_scope_valid": declared_scope_valid,
        "journal_binding_complete": binding_complete,
        "prior_failures": bool(prior_failures),
        "accepted_retry_result": bool(accepted_retry_result),
        "accepted_result_anchored": binding_complete if prior_failures else True,
    }
    admission = CONTRACT.admission_decision(
        execution_state=execution_state,
        semantic_status=semantic,
        provenance_status=provenance,
        **checks,
    )
    return {
        "execution_state": execution_state,
        "semantic_status": semantic,
        "provenance_status": provenance,
        "admission": admission,
        "checks": checks,
    }


def _row_record_map(journal):
    records = {}
    if journal is None:
        return records
    for event in journal.events:
        if event.get("schema") == RECORDER.ATTEMPT_SUCCESS_SCHEMA:
            records[event.get("source_row_index")] = event.get("record")
    return records


def _audit_dataset(
    *,
    dataset,
    run_root,
    selected_rows,
    zero,
    polynomial,
    root_records,
    root_identity,
    manifest,
    execution_state,
    provenance_provider=None,
    verifier_profile=None,
):
    journal, checkpoint, journal_complete, journal_error = _read_attempt_history(
        run_root, dataset, selected_rows
    )
    records = _row_record_map(journal)
    profile = CONTRACT.TrustedRunProfile(
        run_id=manifest["run_id"],
        observer_sha256=manifest["source_identity"]["observer_sha256"],
        auditor_sha256=manifest["source_identity"]["auditor_sha256"],
        provenance_sha256=manifest["source_identity"]["provenance_sha256"],
        runner_sha256=manifest["source_identity"]["runner_sha256"],
        environment=manifest["environment"],
        numeric_representation=manifest["numeric_representation"],
    )
    rows = []
    required_assets = journal is not None and journal_error is None and journal_complete
    for index, row in selected_rows.items():
        record = records.get(index)
        if dataset == "cylinder" and not root_records:
            result = {
                "dataset": dataset,
                "source_row_index": index,
                "semantic_validation": "SEMANTIC_NOT_RUN",
                "semantic_error": "required Cylinder root supplement is unavailable",
                "provenance_validation": {
                    "status": "PROVENANCE_NOT_RUN",
                    "reason": "required root evidence unavailable",
                    "anchor_id": None,
                },
                "scientific_admission": "INCOMPLETE_EVIDENCE",
                "certificate_validation": "CERTIFICATE_SCOPE_NOT_RUN",
                "journal_binding_complete": False,
            }
        elif journal is None:
            result = AUDITOR.audit_saved_record(
                dataset,
                index,
                row,
                record,
                execution_state=execution_state,
                required_evidence_complete=False,
                zero=zero,
                polynomial=polynomial,
                root_records=root_records,
            )
        else:
            independent = _independent_inputs(dataset, index, row, root_identity)
            result = AUDITOR.audit_saved_record(
                dataset,
                index,
                row,
                record,
                zero=zero,
                polynomial=polynomial,
                root_records=root_records,
                independent_inputs=independent,
                run_profile=profile,
                attempt_journal=journal,
                selected_rows=selected_rows,
                provenance_provider=provenance_provider,
                verifier_profile=verifier_profile,
                execution_state=execution_state,
                required_evidence_complete=required_assets,
            )
        if journal_error:
            result = dict(result)
            result["journal_error"] = journal_error
        rows.append(result)
    retry_history = (
        [] if checkpoint is None else checkpoint.get("retry_plan_history", [])
    )
    failed_events = (
        []
        if journal is None
        else [
            event
            for event in journal.events
            if event.get("schema") == RECORDER.ATTEMPT_FAILURE_SCHEMA
        ]
    )
    successful_retries = (
        []
        if journal is None
        else [
            event
            for event in journal.events
            if event.get("schema") == RECORDER.ATTEMPT_SUCCESS_SCHEMA
            and event.get("retry_of") is not None
        ]
    )
    return (
        rows,
        {
            "dataset": dataset,
            "execution_state": execution_state,
            "checkpoint_execution_state": None
            if checkpoint is None
            else checkpoint.get("execution_state"),
            "selected_row_count": len(selected_rows),
            "audited_row_count": sum(
                item.get("semantic_validation") == "SEMANTIC_VALID" for item in rows
            ),
            "semantic_invalid_count": sum(
                item.get("semantic_validation") == "SEMANTIC_INVALID" for item in rows
            ),
            "journal_complete": journal_complete,
            "journal_error": journal_error,
            "attempt_count": 0 if journal is None else len(journal.events),
            "failed_attempt_count": len(failed_events),
            "successful_retry_count": len(successful_retries),
            "retry_plan_history_count": len(retry_history),
        },
        bool(required_assets),
    )


def run_campaign(
    *,
    l6_root,
    output_root,
    run_id,
    datasets,
    workers=1,
    indices=None,
    resume=False,
    retry_failed=False,
    root_supplement_path=None,
    root_supplement_sha256=None,
    provenance_provider=None,
    verifier_profile=None,
):
    if retry_failed and not resume:
        raise ValueError("--retry-failed requires --resume")
    if workers not in (1, 2):
        raise ValueError("workers must be 1 or 2")
    run_root = validate_run_location(output_root, run_id, resume=resume)
    l6_root = Path(l6_root).expanduser().resolve()
    try:
        (
            manifest,
            selected,
            root_records,
            root_identity,
            zero,
            polynomial,
        ) = _prepare_run_manifest(
            l6_root=l6_root,
            datasets=datasets,
            run_id=run_id,
            indices=indices,
            root_supplement_path=root_supplement_path,
            root_supplement_sha256=root_supplement_sha256,
            workers=workers,
        )
    except FileNotFoundError as error:
        raise MissingEvidenceError("required frozen evidence is unavailable") from error
    if resume:
        validate_resume_manifest(run_root, manifest)
    else:
        run_root.mkdir(parents=True, exist_ok=False)
        _atomic_json(run_root / RUN_MANIFEST_FILENAME, manifest)

    dataset_summaries, all_rows, errors = [], [], []
    execution_states = []
    evidence_complete = True
    for dataset in datasets:
        checkpoint = None
        try:
            checkpoint = RECORDER.run_dataset(
                dataset,
                l6_root,
                Path(output_root).expanduser().resolve(),
                run_id,
                workers=workers,
                resume=resume,
                indices=indices,
                retry_failed=retry_failed,
            )
            execution_state = checkpoint.get("execution_state", "failed")
        except Exception as error:
            errors.append(
                {
                    "dataset": dataset,
                    "phase": "execution",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            checkpoint_path = run_root / dataset / "checkpoint.json"
            if checkpoint_path.is_file():
                try:
                    checkpoint = _strict_json(checkpoint_path)
                except (OSError, ValueError, TypeError):
                    checkpoint = None
            execution_state = (
                checkpoint.get("execution_state", "failed")
                if checkpoint is not None
                else "failed"
            )
        execution_states.append(execution_state)
        row_results, summary, row_evidence_complete = _audit_dataset(
            dataset=dataset,
            run_root=run_root,
            selected_rows=selected[dataset],
            zero=zero,
            polynomial=polynomial,
            root_records=root_records if dataset == "cylinder" else (),
            root_identity=root_identity if dataset == "cylinder" else None,
            manifest=manifest,
            execution_state=execution_state,
            provenance_provider=provenance_provider,
            verifier_profile=verifier_profile,
        )
        dataset_summaries.append(summary)
        all_rows.extend(row_results)
        evidence_complete = evidence_complete and row_evidence_complete

    if errors or any(state == "failed" for state in execution_states):
        aggregate_execution = "failed"
    elif any(state in {"running", "interrupted"} for state in execution_states):
        aggregate_execution = "interrupted"
    else:
        aggregate_execution = "complete" if execution_states else "failed"
    selected_count = sum(len(rows) for rows in selected.values())
    has_failures = any(item["failed_attempt_count"] for item in dataset_summaries)
    has_retries = any(item["successful_retry_count"] for item in dataset_summaries)
    overall = aggregate_admission(
        all_rows,
        execution_state=aggregate_execution,
        required_evidence_complete=evidence_complete and not errors,
        selected_row_count=selected_count,
        prior_failures=has_failures,
        accepted_retry_result=has_retries,
    )
    checks = overall["checks"]
    validation_report = {
        "schema": CONTRACT.REPORT_SCHEMA,
        "source_identity": manifest["source_identity"],
        "schema_identity": dict(CONTRACT.V4_SCHEMA_IDENTITIES),
        "execution_state": overall["execution_state"],
        "semantic_status": overall["semantic_status"],
        "provenance_status": overall["provenance_status"],
        "checks": checks,
        "admission": overall["admission"],
    }
    CONTRACT.validate_v4_report(
        validation_report, expected_source_identity=manifest["source_identity"]
    )
    evidence_report = {
        "schema": RUN_REPORT_SCHEMA,
        "run_id": run_id,
        "selected_version": VERSION,
        "run_manifest_sha256": CONTRACT.canonical_sha256(manifest),
        "source_identity": manifest["source_identity"],
        "root_supplement": root_identity,
        **overall,
        "selected_row_count": selected_count,
        "datasets": dataset_summaries,
        "rows": all_rows,
        "failures": errors,
        "trusted_provenance_provider_configured": provenance_provider is not None,
        "faithful_replay_claimed": False,
        "limitations": [
            "execution completion is not semantic validation or scientific admission",
            "no production external provenance provider or trust root is configured",
            "V1/V2/V3 reports and full_gate_complete flags are not V4 admission inputs",
            "historical evidence and seven-dataset scientific closure are unchanged",
        ],
    }
    _atomic_json(run_root / "validation-report-v4.json", validation_report)
    _atomic_json(run_root / "run-evidence-v4.json", evidence_report)
    return validation_report, evidence_report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l6-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--datasets", nargs="+", choices=tuple(RECORDER.DATASETS), required=True
    )
    parser.add_argument("--workers", type=int, default=1, choices=(1, 2))
    parser.add_argument("--indices", type=int, nargs="+")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true", default=False)
    parser.add_argument("--cylinder-root-supplement", type=Path)
    parser.add_argument("--cylinder-root-supplement-sha256")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report, evidence = run_campaign(
            l6_root=args.l6_root,
            output_root=args.output_root,
            run_id=args.run_id,
            datasets=args.datasets,
            workers=args.workers,
            indices=args.indices,
            resume=args.resume,
            retry_failed=args.retry_failed,
            root_supplement_path=args.cylinder_root_supplement,
            root_supplement_sha256=args.cylinder_root_supplement_sha256,
        )
    except MissingEvidenceError:
        source_identity = _source_identity()
        checks = {
            "required_evidence_complete": False,
            "independent_audit_valid": None,
            "certificate_validation_valid": None,
            "trend_validation_complete": False,
            "declared_scope_valid": None,
            "journal_binding_complete": False,
            "prior_failures": False,
            "accepted_retry_result": False,
            "accepted_result_anchored": False,
        }
        report = {
            "schema": CONTRACT.REPORT_SCHEMA,
            "source_identity": source_identity,
            "schema_identity": dict(CONTRACT.V4_SCHEMA_IDENTITIES),
            "execution_state": "not_started",
            "semantic_status": "SEMANTIC_NOT_RUN",
            "provenance_status": "PROVENANCE_NOT_RUN",
            "checks": checks,
            "admission": "INCOMPLETE_EVIDENCE",
        }
        CONTRACT.validate_v4_report(report, expected_source_identity=source_identity)
        print(
            json.dumps(
                {
                    "report": report,
                    "preflight_error": "required external evidence is unavailable",
                    "run_created": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 3
    except (OSError, ValueError, RuntimeError, TypeError) as error:
        parser.error(f"V4 campaign not admitted: {type(error).__name__}: {error}")
    print(
        json.dumps({"report": report, "evidence": evidence}, indent=2, sort_keys=True)
    )
    return 0 if report["admission"] == "ADMITTED_WITH_DECLARED_SCOPE" else 3


if __name__ == "__main__":
    raise SystemExit(main())
