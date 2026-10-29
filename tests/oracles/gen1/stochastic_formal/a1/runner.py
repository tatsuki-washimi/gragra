"""Reviewed-gate-only acoustic attempt runner; every launched outcome is sealed.

CLI: python -m tests.oracles.gen1.stochastic_formal.a1.runner
 --family A1|A3 --authorization AUTH.json --attempt EXTERNAL_FRESH_DIR
Authorization is authored by the coordinator after independent reviews and F.
It pins E, durability integration commit, all dependency bytes and environment.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import resource
import signal
import subprocess
import sys
from pathlib import Path

if not __package__:
    __package__ = "tests.oracles.gen1.stochastic_formal.a1"
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import mpmath

from ... import stochastic_acceptance as sa
from ... import stochastic_attempt as _attempt

if (
    "stochastic_attempt" in sys.modules
    and sys.modules["stochastic_attempt"] is not _attempt
):
    raise RuntimeError("unexpected sealer sibling import")
sys.modules["stochastic_attempt"] = _attempt
from ...run_stochastic_local import run_frozen_attempt, supervise_attempt  # noqa: E402
from ...stochastic_f_common_v2 import (  # noqa: E402
    check_calibration_formal_separation,
    validate_purpose_record,
)
from ..a3.generator import cavern_matrix  # noqa: E402
from .generator import independent, record, surface_matrix  # noqa: E402


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment():
    version = platform.python_version()
    package = Path(mpmath.__file__).parent.resolve()
    production = sorted(
        n for n in sys.modules if n == "gragra" or n.startswith("gragra.")
    )
    if production:
        raise ValueError("production modules already loaded")
    return {
        "python": version,
        "implementation": platform.python_implementation(),
        "executable": str(Path(sys.executable).absolute()),
        "executable_sha256": digest(Path(sys.executable)),
        "mpmath": mpmath.__version__,
        "mpmath_package": str(package),
        "mpmath_files": {
            str(path.relative_to(package)): digest(path)
            for path in sorted(package.rglob("*.py"))
        },
        "production_modules": production,
        "loaded_modules": {
            name: {
                "path": str(Path(module.__file__).resolve())
                if getattr(module, "__file__", None)
                else None,
                "sha256": digest(Path(module.__file__))
                if getattr(module, "__file__", None) and Path(module.__file__).is_file()
                else None,
            }
            for name, module in sorted(sys.modules.copy().items())
        },
    }


def pinned_json(pin):
    if digest(Path(pin["path"])) != pin["sha256"]:
        raise ValueError("review pin mismatch")
    return json.loads(Path(pin["path"]).read_bytes())


def reviewed(reviews, *, family, commit, scope, hashes):
    if set(reviews) != {"science", "numerical", "provenance"}:
        raise ValueError("three independent reviews required")
    identities = set()
    for role, pin in reviews.items():
        report = pinned_json(pin)
        if (
            set(report)
            != {
                "schema_version",
                "role",
                "reviewer_id",
                "decision",
                "scope",
                "family",
                "reviewed_commit",
                "artifact_hashes",
                "original_reviews",
            }
            or not re.fullmatch(r"[0-9a-f]{40}", commit)
            or report.get("schema_version") != 1
            or report.get("role") != role
            or report.get("decision") != "PASS"
            or report.get("family") != family
            or report.get("scope") != scope
            or report.get("reviewed_commit") != commit
            or report.get("artifact_hashes") != hashes
            or not isinstance(report.get("reviewer_id"), str)
            or not report["reviewer_id"]
            or report["reviewer_id"] in identities
            or not report.get("original_reviews")
        ):
            raise ValueError("actual independent review binding mismatch")
        identities.add(report["reviewer_id"])
        for original in report["original_reviews"]:
            if digest(Path(original["path"])) != original["sha256"]:
                raise ValueError("original review byte mismatch")


def implementation_reviews(root, auth, family, head):
    groups = auth["reviewed_source_hashes_by_group"]
    if set(groups) != {"generator", "runner", "sealer_helpers"} or any(
        not group for group in groups.values()
    ):
        raise ValueError("three nonempty implementation groups required")
    flattened = {p: h for group in groups.values() for p, h in group.items()}
    if len(flattened) != sum(map(len, groups.values())):
        raise ValueError("duplicate grouped implementation source")
    required = {
        "tests/oracles/gen1/stochastic_formal/a1/generator.py",
        "tests/oracles/gen1/stochastic_formal/a1/runner.py",
        "tests/oracles/gen1/stochastic_formal/a3/generator.py",
        "tests/oracles/gen1/stochastic_acceptance.py",
        "tests/oracles/gen1/stochastic_attempt.py",
        "tests/oracles/gen1/run_stochastic_local.py",
        "tests/oracles/gen1/stochastic_f_common_v2.py",
    }
    if not required.issubset(flattened):
        raise ValueError("reviewed groups omit actual computational imports")
    candidate = auth["reviewed_commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", candidate):
        raise ValueError("exact reviewed source commit required")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", candidate, head], cwd=root, check=True
    )
    for path, sha in flattened.items():
        if auth["file_sha256"].get(path) != sha:
            raise ValueError("implementation group differs from exact E closure")
        raw = subprocess.check_output(["git", "show", f"{candidate}:{path}"], cwd=root)
        if hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError("reviewed candidate source differs from E")
    identities = set()
    for role, pin in auth["reviews"].items():
        review = pinned_json(pin)
        if (
            set(review)
            != {
                "schema_version",
                "role",
                "reviewer_id",
                "decision",
                "scope",
                "reviewed_commit",
                "families",
                "file_hashes_by_group",
                "original_reviews",
            }
            or review.get("schema_version") != 1
            or review.get("role") != role
            or review.get("decision") != "PASS"
            or review.get("scope") != "reference implementation"
            or review.get("reviewed_commit") != candidate
            or family not in review.get("families", [])
            or review.get("file_hashes_by_group") != groups
            or not isinstance(review.get("reviewer_id"), str)
            or not review["reviewer_id"]
            or review["reviewer_id"] in identities
            or not review.get("original_reviews")
        ):
            raise ValueError("actual implementation review binding mismatch")
        identities.add(review["reviewer_id"])
        for original in review["original_reviews"]:
            if digest(Path(original["path"])) != original["sha256"]:
                raise ValueError("original implementation review byte mismatch")


def frozen(declaration, *, root, family, paths, hashes, procedure):
    artifacts = {
        paths[k]: hashes[paths[k]] for k in ("fixtures", "budget", "procedure")
    }
    if (
        declaration["schema_version"] != 1
        or declaration["status"] != "FROZEN"
        or declaration["family"] != family
        or declaration["freeze_id"] != "gen1-stochastic-F_" + family + "-v1"
        or declaration["artifact_hashes"] != artifacts
        or declaration["baseline_pins"] != procedure["baseline_pins"]
    ):
        raise ValueError("actual three-file family freeze required")
    for kind, contract in (
        ("fixture", "fixtures"),
        ("budget", "budget"),
        ("procedure", "procedure"),
    ):
        if (
            declaration[kind + "_file"] != paths[contract]
            or declaration[kind + "_sha256"] != artifacts[paths[contract]]
        ):
            raise ValueError("direct/map freeze identity mismatch")
    for path, sha in declaration["baseline_pins"].items():
        if digest(root / path) != sha:
            raise ValueError("frozen baseline bytes changed")
    if (
        procedure["fixture_sha256"] != artifacts[paths["fixtures"]]
        or procedure["budget_sha256"] != artifacts[paths["budget"]]
        or procedure["fixture_file"] != paths["fixtures"]
        or procedure["budget_file"] != paths["budget"]
    ):
        raise ValueError("procedure fixture/budget identity mismatch")
    reviewed(
        declaration["reviews"],
        family=family,
        commit=declaration["reviewed_commit"],
        scope="protocol freeze",
        hashes=artifacts,
    )
    receipt = pinned_json(declaration["root_concurrence"])
    decision = "PASS_FOR_F_" + family + "_V1_FREEZE"
    if (
        declaration["root_concurrence"].get("decision") != decision
        or receipt.get("decision") != decision
        or receipt.get("family") != family
        or receipt.get("artifact_hashes") != artifacts
        or receipt.get("review_hashes")
        != {role: pin["sha256"] for role, pin in declaration["reviews"].items()}
    ):
        raise ValueError("actual root freeze concurrence mismatch")


def gate(root, auth, family):
    if auth["family"] != family or auth["status"] != "REFERENCE_GENERATION_AUTHORIZED":
        raise ValueError("family authorization required")
    reviews = auth["reviews"]
    if set(reviews) != {"science", "numerical", "provenance"}:
        raise ValueError("three independent reviews required")
    for item in reviews.values():
        pinned_json(item)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    if head != auth["e_commit"] or subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root
    ):
        raise ValueError("clean exact E required")
    integration = Path(auth["integration_root"]).resolve(strict=True)
    integration_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=integration, text=True
    ).strip()
    if integration_head != auth["durability_commit"]:
        raise ValueError("actual integration HEAD differs from durability pin")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", head, integration_head],
        cwd=integration,
        check=True,
    )
    required = {
        "tests/oracles/gen1/stochastic_formal/a1/generator.py",
        "tests/oracles/gen1/stochastic_formal/a1/runner.py",
        "tests/oracles/gen1/stochastic_formal/a3/generator.py",
        "tests/oracles/gen1/stochastic_acceptance.py",
        "tests/oracles/gen1/stochastic_attempt.py",
        "tests/oracles/gen1/run_stochastic_local.py",
        "tests/oracles/gen1/stochastic_f_common_v2.py",
    }
    if not required.issubset(auth["file_sha256"]):
        raise ValueError("incomplete oracle closure")
    for name, expected in auth["file_sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or digest(path) != expected:
            raise ValueError("dependency byte mismatch")
        committed = subprocess.check_output(["git", "show", f"{head}:{name}"], cwd=root)
        if hashlib.sha256(committed).hexdigest() != expected:
            raise ValueError("dependency not committed in exact E")
    if environment() != auth["environment"]:
        raise ValueError("environment mismatch")
    implementation_reviews(root, auth, family, head)
    paths = auth["contracts"]
    if set(paths) != {"fixtures", "budget", "procedure"}:
        raise ValueError("exact three-file contract required")
    for path in paths.values():
        if path not in auth["file_sha256"]:
            raise ValueError("unpinned contract")
    contracts = {
        name: json.loads((root / path).read_bytes()) for name, path in paths.items()
    }
    frozen(
        pinned_json(auth["freeze_declaration"]),
        root=root,
        family=family,
        paths=paths,
        hashes=auth["file_sha256"],
        procedure=contracts["procedure"],
    )
    for source in auth["source_artifacts"]:
        if digest(Path(source["path"])) != source["sha256"]:
            raise ValueError("primary source artifact mismatch")
    if {source["sha256"] for source in auth["source_artifacts"]} != {
        pin["sha256"] for pin in contracts["procedure"]["record_source_pins"]
    }:
        raise ValueError("primary source set mismatch")
    return contracts


def purpose(rec, fixture, kind="formal-science"):
    metadata = {
        "purpose": kind,
        "set": "science" if kind == "formal-science" else "calibration",
        "key": rec["key"],
        "scientific_input": {
            "frequency_hz": rec["key"]["frequency_hz"],
            "frame": fixture["frame"],
            "psd_convention": fixture["psd_convention"],
            "source_pins": rec["source_pins"],
            "procedure_hash": rec["procedure_hash"],
            "inputs": {
                **rec["inputs"],
                "pressure_convention": fixture["pressure_convention"],
            },
        },
    }
    validate_purpose_record(metadata)
    return metadata


def purpose_plan(fixtures, family, procedure_hash, pins, kind="formal-science"):
    result = []
    for fixture in fixtures:
        for rep in ("Gfree", "SI"):
            descriptor = {
                "key": {
                    "family": family,
                    "fixture_id": fixture["fixture_id"],
                    "frequency_hz": fixture["inputs"]["frequency_hz"],
                    "target_id": fixture["target_id"],
                    "observable": "acceleration_csd",
                    "representation": rep,
                    "set": "science",
                },
                "source_pins": pins,
                "procedure_hash": procedure_hash,
                "inputs": fixture["inputs"],
            }
            result.append(purpose(descriptor, fixture, kind))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["A1", "A3"], required=True)
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument(
        "--purpose", choices=["formal-science", "calibration"], default="formal-science"
    )
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--procedure", type=Path)
    parser.add_argument("--environment-inventory", type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--attempt", type=Path)
    args = parser.parse_args()
    if args.inventory:
        print(json.dumps(environment(), sort_keys=True))
        return 0
    if args.attempt is None:
        parser.error("fresh external attempt required")
    root = Path(__file__).resolve().parents[5]
    formal_metadata = None
    if args.purpose == "calibration":
        if (
            args.authorization is not None
            or args.fixtures is None
            or args.procedure is None
        ):
            parser.error(
                "calibration requires fixture/procedure files, no formal authorization"
            )
        fixtures = json.loads(args.fixtures.read_bytes())
        procedure = json.loads(args.procedure.read_bytes())
        if not fixtures.get("calibration"):
            raise ValueError("declared disjoint calibration required")
        formal_metadata = purpose_plan(
            fixtures["science"],
            args.family,
            digest(args.procedure),
            procedure["record_source_pins"],
        )
        plans = purpose_plan(
            fixtures["calibration"],
            args.family,
            digest(args.procedure),
            procedure["record_source_pins"],
            "calibration",
        )
        contracts = {
            "procedure": procedure,
            "fixtures": {
                **fixtures,
                "science": fixtures["calibration"],
                "expected_keys": [p["key"] for p in plans],
            },
        }
        auth = {
            "campaign_dir": str(args.attempt.resolve().parent),
            "contracts": {"procedure": str(args.procedure)},
            "file_sha256": {str(args.procedure): digest(args.procedure)},
            "calibration_purpose_records": [],
            "environment": environment(),
        }
        if args.worker:
            if args.environment_inventory is None:
                raise ValueError("calibration worker expected inventory required")
            expected = json.loads(args.environment_inventory.read_bytes())
            if expected != auth["environment"]:
                raise ValueError("calibration worker environment mismatch")
    else:
        if args.authorization is None:
            parser.error("formal execution requires authorization")
        auth = json.loads(args.authorization.read_bytes())
        contracts = gate(root, auth, args.family)
        names = auth["attempt_names"]
        if len(names) != 2 or len(set(names)) != 2 or args.attempt.name not in names:
            raise ValueError("frozen two-attempt resource limit")
    campaign = Path(auth["campaign_dir"]).resolve()
    if args.attempt.resolve().parent != campaign:
        raise ValueError("attempt not in authorized campaign")
    if args.attempt.resolve().is_relative_to(root):
        raise ValueError("attempt must be outside Git")
    ancestor = args.attempt.resolve().parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if (
        subprocess.run(
            ["git", "-C", str(ancestor), "rev-parse", "--show-toplevel"],
            capture_output=True,
        ).returncode
        == 0
    ):
        raise ValueError("attempt must be outside every Git working tree")
    records, purposes, diagnostics = [], [], []
    status, error = "success", None
    procedure_hash = auth["file_sha256"][auth["contracts"]["procedure"]]
    pins = contracts["procedure"]["record_source_pins"]
    function = surface_matrix if args.family == "A1" else cavern_matrix
    if not args.worker and args.purpose == "calibration":
        inventory_path = args.environment_inventory or args.attempt.with_suffix(
            ".environment.json"
        )
        with inventory_path.open("xb") as stream:
            stream.write(sa.canonical_json(auth["environment"]))
        seal = supervise_attempt(
            args.attempt,
            repository_root=root,
            command=[
                sys.executable,
                "-I",
                str(Path(__file__).resolve()),
                "--worker",
                "--purpose",
                "calibration",
                "--family",
                args.family,
                "--fixtures",
                str(args.fixtures.resolve()),
                "--procedure",
                str(args.procedure.resolve()),
                "--environment-inventory",
                str(inventory_path.resolve()),
                "--attempt",
                str(args.attempt.resolve()),
            ],
            key_fields=sa.KEY_FIELDS,
            expected_keys=contracts["fixtures"]["expected_keys"],
            record_key_path=["key"],
            timeout_seconds=1800,
            provenance={
                "purpose": "calibration",
                "formal_evidence": False,
                "fixture_sha256": digest(args.fixtures),
                "procedure_sha256": digest(args.procedure),
                "environment_inventory_sha256": digest(inventory_path),
                "source_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=root, text=True
                ).strip(),
            },
        )
        return 0 if seal["execution_coverage"]["status"] == "accepted" else 1
    if not args.worker:
        grouped = auth["file_hashes_by_group"]
        flattened = {p: h for group in grouped.values() for p, h in group.items()}
        if flattened != auth["file_sha256"]:
            raise ValueError("five-group launch closure differs from reviewed E")
        inventory_pin = auth["environment_inventory"]
        if pinned_json(inventory_pin) != auth["environment"]:
            raise ValueError("inventory artifact differs from actual environment pin")
        seal = run_frozen_attempt(
            args.attempt,
            repository_root=root,
            command=[
                sys.executable,
                "-I",
                str(Path(__file__).resolve()),
                "--worker",
                "--family",
                args.family,
                "--authorization",
                str(args.authorization.resolve()),
                "--attempt",
                str(args.attempt.resolve()),
            ],
            key_fields=sa.KEY_FIELDS,
            expected_keys=contracts["fixtures"]["expected_keys"],
            record_key_path=["key"],
            expected_head=auth["e_commit"],
            expected_tree=auth["e_tree"],
            file_hashes_by_group=grouped,
            environment_inventory_path=inventory_pin["path"],
            environment_inventory_sha256=inventory_pin["sha256"],
            timeout_seconds=1800,
        )
        return 0 if seal["execution_coverage"]["status"] == "accepted" else 1
    launch = json.loads((args.attempt / "launch.json").read_bytes())
    if launch["pid"] != os.getpid() or launch["process_group_id"] != os.getpgid(0):
        raise ValueError("worker requires actual durable supervisor launch")
    record_stream = (args.attempt / "records.raw.jsonl").open("xb")

    def alarm(_signum, _frame):
        raise TimeoutError("formal attempt timeout")

    try:
        signal.signal(signal.SIGALRM, alarm)
        signal.alarm(1800)
        resource.setrlimit(resource.RLIMIT_AS, (4294967296, 4294967296))
        with independent():
            purposes = purpose_plan(
                contracts["fixtures"]["science"],
                args.family,
                procedure_hash,
                pins,
                args.purpose,
            )
            calibration = purpose_plan(
                contracts["fixtures"].get("calibration", []),
                args.family,
                procedure_hash,
                pins,
                "calibration",
            )
            separation = check_calibration_formal_separation(
                calibration + purposes
                if args.purpose == "calibration"
                else calibration + auth["calibration_purpose_records"],
                formal_metadata if args.purpose == "calibration" else purposes,
            )
            if not separation["admitted"]:
                raise ValueError("calibration/formal overlap")
            with (args.attempt / "purpose-records.json").open("xb") as stream:
                stream.write(sa.canonical_json(purposes))
                stream.flush()
                os.fsync(stream.fileno())
            for fixture in contracts["fixtures"]["science"]:
                for rep in ("Gfree", "SI"):
                    diagnostic = {}
                    rec = record(
                        args.family,
                        fixture,
                        rep,
                        procedure_hash,
                        pins,
                        function,
                        diagnostics=diagnostic,
                    )
                    sa.validate_record(rec)
                    records.append(rec)
                    record_stream.write(sa.canonical_json(rec))
                    record_stream.flush()
                    os.fsync(record_stream.fileno())
                    diagnostics.append(diagnostic)
            if not sa.admit_records(records, contracts["fixtures"]["expected_keys"])[
                "coverage_valid"
            ]:
                raise ValueError("reference exact coverage/resolution failed")
            if environment() != auth["environment"]:
                raise ValueError(
                    "runtime scientific closure changed during computation"
                )
    except BaseException as exc:
        status = (
            "timeout"
            if isinstance(exc, TimeoutError)
            else "interruption"
            if isinstance(exc, KeyboardInterrupt)
            else "exception"
        )
        error = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        signal.alarm(0)
        record_stream.close()
        for name, value in (
            ("method-diagnostics.json", diagnostics),
            ("error.json", error),
        ):
            with (args.attempt / name).open("xb") as output:
                output.write(sa.canonical_json(value))
    return 0 if status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
