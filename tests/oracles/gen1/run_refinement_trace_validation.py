"""Run a new Gen1 refinement-trace campaign through the V3-only route.

The recorder completes and the independent V3 auditor checks semantic
consistency. Scientific admission remains fail-closed unless a configured,
independently trusted provenance provider is available; this CLI currently has
no such production provider and therefore exits nonzero after a semantic pass.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

ORACLE_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY = Path(__file__).resolve().parents[3]
DEFAULT_RECORDER = ORACLE_DIRECTORY / "record_frozen_refinement_traces_v3.py"
DEFAULT_AUDITOR = ORACLE_DIRECTORY / "audit_frozen_refinement_traces_v3.py"
DEFAULT_SCHEMA = "gragra.gen1.refinement-trace-observer.v3"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pinned V3 source: {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RECORDER = _load("_gen1_v3_default_recorder", DEFAULT_RECORDER)
AUDITOR = _load("_gen1_v3_default_auditor", DEFAULT_AUDITOR)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv=None) -> int:
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
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", args.run_id) or ".." in args.run_id:
        parser.error("run id must be a new safe identifier without path components")
    if args.output_root.resolve().is_relative_to(REPOSITORY):
        parser.error("raw trace output must remain outside Git")
    run_root = args.output_root.resolve() / args.run_id
    if not args.resume and run_root.exists():
        parser.error("new campaigns require a new run id and empty output path")
    if args.resume and not run_root.is_dir():
        parser.error("V3 resume requires an existing V3 run id")

    observer_sha256 = _sha256(DEFAULT_RECORDER)
    auditor_sha256 = _sha256(DEFAULT_AUDITOR)
    dataset_summaries = []
    audit_summaries = []
    failures = []
    for dataset in args.datasets:
        try:
            dataset_summaries.append(
                RECORDER.run_dataset(
                    dataset,
                    args.l6_root.resolve(),
                    args.output_root.resolve(),
                    args.run_id,
                    workers=args.workers,
                    resume=args.resume,
                    indices=args.indices,
                )
            )
        except Exception as error:
            failures.append(
                {"dataset": dataset, "phase": "record", "error": f"{type(error).__name__}: {error}"}
            )

    try:
        zero_certificates, polynomial_certificates = AUDITOR.certificate_inputs(
            args.l6_root.resolve()
        )
        for dataset in args.datasets:
            if any(item["dataset"] == dataset for item in failures):
                continue
            try:
                audit_summaries.append(
                    AUDITOR.audit_dataset(
                        dataset,
                        args.l6_root.resolve(),
                        run_root,
                        zero_certificates,
                        polynomial_certificates,
                        allow_selected=args.indices is not None,
                        root_records=(),
                        observer_sha256=observer_sha256,
                    )
                )
            except Exception as error:
                failures.append(
                    {"dataset": dataset, "phase": "audit", "error": f"{type(error).__name__}: {error}"}
                )
    except Exception as error:
        failures.append(
            {"dataset": None, "phase": "certificate-inputs", "error": f"{type(error).__name__}: {error}"}
        )

    semantic_valid = bool(audit_summaries) and not failures and all(
        item.get("semantic_validation") == "SEMANTIC_VALID"
        and not item.get("errors")
        for item in audit_summaries
    )
    execution_complete = bool(dataset_summaries) and not failures and all(
        item.get("execution_state") == "complete"
        for item in dataset_summaries
    )
    unresolved_trend_cases = sum(
        int(item.get("unresolved_trend_cases", 0)) for item in dataset_summaries
    )
    # No production trust provider was found in the pinned repository. A mock
    # provider is used only in unit tests and cannot upgrade this CLI result.
    provenance_status = "PROVENANCE_UNVERIFIED"
    admission = "NOT_ADMITTED"
    if not execution_complete or not semantic_valid:
        admission = "INCOMPLETE_EVIDENCE"

    report = {
        "schema": "gragra.gen1.refinement-trace-validation-run.v3",
        "run_id": args.run_id,
        "selected_observer": "V3",
        "observer_sha256": observer_sha256,
        "auditor_sha256": auditor_sha256,
        "schema_identity": DEFAULT_SCHEMA,
        "execution_state": "complete" if execution_complete else "failed",
        "semantic_validation": "SEMANTIC_VALID" if semantic_valid else "SEMANTIC_INVALID",
        "provenance_validation": provenance_status,
        "scientific_admission": admission,
        "unresolved_trend_cases": unresolved_trend_cases,
        "trusted_provenance_provider_configured": False,
        "faithful_replay_claimed": False,
        "datasets": dataset_summaries,
        "independent_audits": audit_summaries,
        "failures": failures,
        "limitations": [
            "local trace/checkpoint commitments are not an external trust anchor",
            "historical evidence and full seven-dataset closure are not changed or claimed",
        ],
    }
    _atomic_json(run_root / "validation-report-v3.json", report)
    _atomic_json(run_root / "run-evidence-v3.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    # Semantic consistency alone is not scientific admission.
    return 0 if admission == "ADMITTED_WITH_DECLARED_SCOPE" else 3


if __name__ == "__main__":
    raise SystemExit(main())
