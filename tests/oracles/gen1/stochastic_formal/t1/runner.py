"""Fresh reference-only T1 attempts with external review and durable E gates.

Candidate supervisor: formal artifacts still require independent seal review.
No default/ad hoc CLI switch confers formal approval.
"""

import argparse
import contextlib
import hashlib
import importlib.abc
import json
import os
import resource
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMMON_PATHS = (
    "docs/design/gen1/stochastic/B1-budget.json",
    "docs/design/gen1/stochastic/B1-fixtures.json",
    "docs/design/gen1/stochastic/B1-procedure.json",
    "docs/design/gen1/stochastic/common-freeze.json",
    "docs/design/gen1/stochastic/common-policy.json",
    "tests/oracles/gen1/stochastic_acceptance.py",
    "tests/test_gen1_stochastic_acceptance.py",
)
REVIEW_PATHS = {
    "generator": {
        "tests/oracles/gen1/stochastic_formal/t1/" + name
        for name in ("reference.py", "laplace.py", "DERIVATION.md")
    },
    "runner": {"tests/oracles/gen1/stochastic_formal/t1/runner.py"},
    "sealer_helpers": {
        "tests/oracles/gen1/stochastic_f_common_v2.py",
        "tests/oracles/gen1/stochastic_acceptance.py",
    },
}
REQUIRED_ADMISSION_INFRASTRUCTURE = {
    "tests/oracles/gen1/stochastic_attempt.py",
    "tests/oracles/gen1/run_stochastic_local.py",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dumps(obj):
    return (
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    )


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def reject_noncanonical_number(value):
    raise ValueError("scientific JSON numbers must be finite exact strings: " + value)


def verify_protocol_reviews(freeze, artifact_hashes):
    """Check reviewer-issued companions and the retained original report bytes."""
    fields = {
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
    reviewers = set()
    for role, pin in freeze["reviews"].items():
        if (
            set(pin) != {"path", "sha256", "decision", "scope"}
            or pin["decision"] != "PASS"
            or pin["scope"] != "protocol freeze"
            or digest(pin["path"]) != pin["sha256"]
        ):
            raise ValueError("family freeze review bytes/verdict mismatch")
        actual = json.loads(
            Path(pin["path"]).read_bytes(),
            object_pairs_hook=unique_object,
            parse_float=reject_noncanonical_number,
            parse_constant=reject_noncanonical_number,
        )
        if (
            set(actual) != fields
            or type(actual["schema_version"]) is not int
            or actual["schema_version"] != 1
            or actual["role"] != role
            or actual["scope"] != "protocol freeze"
            or actual["family"] != "T1"
            or actual["decision"] != "PASS"
            or actual["reviewed_commit"] != freeze["reviewed_commit"]
            or actual["artifact_hashes"] != artifact_hashes
            or not isinstance(actual["reviewer_id"], str)
            or not actual["reviewer_id"]
            or actual["reviewer_id"] in reviewers
        ):
            raise ValueError("protocol review semantic binding mismatch")
        reviewers.add(actual["reviewer_id"])
        originals = actual["original_reviews"]
        if not isinstance(originals, list) or not originals:
            raise ValueError("original independent review bytes required")
        for original in originals:
            if (
                set(original) != {"path", "sha256"}
                or digest(original["path"]) != original["sha256"]
            ):
                raise ValueError("original review actual bytes mismatch")


def verify_implementation_reviews(auth, root, e_commit):
    """Bind independent pre-E reviews to actual candidate, E and working bytes."""
    groups = auth["reviewed_source_hashes_by_group"]
    if set(groups) != set(REVIEW_PATHS) or any(
        set(groups[group]) != paths for group, paths in REVIEW_PATHS.items()
    ):
        raise ValueError("exact T1 three-group reviewed source closure required")
    admission = auth["file_hashes_by_group"]
    if set(admission) != {"policy", "fixture", "generator", "source", "infrastructure"}:
        raise ValueError("frozen five-group admission boundary required")
    if any(not isinstance(group, dict) or not group for group in admission.values()):
        raise ValueError("nonempty frozen admission group maps required")
    if not REQUIRED_ADMISSION_INFRASTRUCTURE.issubset(admission["infrastructure"]):
        raise ValueError("frozen required infrastructure files omitted")
    admitted = {}
    for group in admission.values():
        for path, expected in group.items():
            actual_path = (root / path).resolve()
            if (
                Path(path).is_absolute()
                or ".." in Path(path).parts
                or not Path(path).parts
                or not actual_path.is_relative_to(root)
            ):
                raise ValueError("relative admitted source path required")
            if path in admitted:
                raise ValueError("duplicate admitted source path")
            admitted[path] = expected
            blob = subprocess.check_output(
                ["git", "-C", str(root), "show", e_commit + ":" + path]
            )
            if (
                digest(actual_path) != expected
                or hashlib.sha256(blob).hexdigest() != expected
            ):
                raise ValueError(
                    "admitted source differs from committed E/working bytes"
                )
    reviewed = {path: sha for group in groups.values() for path, sha in group.items()}
    if any(admitted.get(path) != sha for path, sha in reviewed.items()):
        raise ValueError("reviewed source omitted from five-group admission closure")
    fields = {
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
    if set(auth.get("reviews", {})) != {"science", "numerical", "provenance"}:
        raise ValueError("all independent reviews required")
    reviewer_ids = set()
    for role, pin in auth["reviews"].items():
        if pin.get("decision") != "PASS" or digest(pin["path"]) != pin["sha256"]:
            raise ValueError("implementation review bytes/verdict mismatch")
        actual = json.loads(
            Path(pin["path"]).read_bytes(),
            object_pairs_hook=unique_object,
            parse_float=reject_noncanonical_number,
            parse_constant=reject_noncanonical_number,
        )
        if (
            set(actual) != fields
            or type(actual["schema_version"]) is not int
            or actual["schema_version"] != 1
            or actual["role"] != role
            or actual["decision"] != "PASS"
            or actual["scope"] != "reference implementation"
            or actual["families"] != ["T1"]
            or actual["file_hashes_by_group"] != groups
            or not isinstance(actual["reviewer_id"], str)
            or not actual["reviewer_id"]
            or actual["reviewer_id"] in reviewer_ids
            or not isinstance(actual["reviewed_commit"], str)
            or len(actual["reviewed_commit"]) != 40
            or any(c not in "0123456789abcdef" for c in actual["reviewed_commit"])
        ):
            raise ValueError(
                "implementation review semantic source/role binding mismatch"
            )
        reviewer_ids.add(actual["reviewer_id"])
        candidate = actual["reviewed_commit"]
        ancestry = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "merge-base",
                "--is-ancestor",
                candidate,
                e_commit,
            ],
            capture_output=True,
        )
        if ancestry.returncode:
            raise ValueError("reviewed source candidate is not an ancestor of E")
        for path, sha in reviewed.items():
            blob = subprocess.check_output(
                ["git", "-C", str(root), "show", candidate + ":" + path]
            )
            if hashlib.sha256(blob).hexdigest() != sha:
                raise ValueError(
                    "reviewed source differs from candidate committed blob"
                )
        originals = actual["original_reviews"]
        if not isinstance(originals, list) or not originals:
            raise ValueError("original implementation reports required")
        for original in originals:
            if (
                set(original) != {"path", "sha256"}
                or digest(original["path"]) != original["sha256"]
            ):
                raise ValueError("original implementation report byte mismatch")


def protocol_helpers():
    sys.path.insert(0, str(HERE.parents[3]))
    from oracles.gen1 import stochastic_f_common_v2

    return stochastic_f_common_v2


def plan_records(fixtures, procedure_hash, mode):
    common = protocol_helpers()
    plans = {}
    for purpose, cases in (
        ("formal-science", fixtures["fixtures"]),
        ("calibration", fixtures["calibration"]),
    ):
        entries = []
        for case in cases:
            if (
                case["inputs"].get("g_si") != "6.6743e-11"
                or case["target_id"] != "projection"
                or case["psd_convention"] != "one_sided_per_hz"
            ):
                raise ValueError("unsupported frozen G/projection/PSD convention")
            for representation in ("Gfree", "SI"):
                key = {
                    "family": "T1",
                    "fixture_id": case["fixture_id"],
                    "frequency_hz": case["inputs"]["frequency_hz"],
                    "target_id": case["target_id"],
                    "observable": "projected_acceleration_psd",
                    "representation": representation,
                    "set": case["set"] if purpose == "formal-science" else "auxiliary",
                }
                metadata = {
                    "purpose": purpose,
                    "set": case["set"]
                    if purpose == "formal-science"
                    else "calibration",
                    "key": key,
                    "scientific_input": {
                        "frequency_hz": key["frequency_hz"],
                        "frame": case["frame"],
                        "psd_convention": case["psd_convention"],
                        "source_pins": fixtures["source_pins"],
                        "procedure_hash": procedure_hash,
                        "inputs": case["inputs"],
                    },
                }
                common.validate_purpose_record(metadata)
                common.canonical_scientific_input_hash(metadata["scientific_input"])
                entries.append((case, key, metadata))
        plans[purpose] = entries
    formal = [m for _, _, m in plans["formal-science"]]
    calibration = [m for _, _, m in plans["calibration"]]
    if (
        not formal
        or not calibration
        or not common.check_calibration_formal_separation(calibration, formal)[
            "admitted"
        ]
    ):
        raise ValueError("calibration/formal scientific inputs are not disjoint")
    keys = [dumps(key) for _, key, _ in plans["formal-science"]]
    if len(set(keys)) != len(keys) or set(keys) != set(
        map(dumps, fixtures["expected_keys"])
    ):
        raise ValueError("complete unique frozen expected keyset required")
    return plans["formal-science" if mode == "formal" else "calibration"]


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


class DenyProduction(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):  # noqa: ARG002
        if fullname == "gragra" or fullname.startswith("gragra."):
            raise ImportError("T1 reference denies production imports")
        return


def worker(args):
    if args.mode == "formal":
        preflight(args)
    sys.meta_path.insert(0, DenyProduction())
    sys.path.insert(0, str(HERE))
    from reference import NAMES, evaluate

    resource.setrlimit(resource.RLIMIT_AS, (4294967296, 4294967296))
    fixtures = json.loads(Path(args.fixtures).read_text())
    plans = plan_records(fixtures, digest(args.procedure), args.mode)
    observed = inventory()
    if args.mode == "formal":
        expected = json.loads(Path(args.expected_environment).read_text())
        if observed != expected:
            raise ValueError("reference runtime bytes mismatch preflight allowlist")
    purpose_records = []
    for case, key, metadata in plans:
        inputs = {name: case["inputs"][name] for name in NAMES}
        for representation in (key["representation"],):
            result = evaluate(inputs, representation)
            components = {
                name: {
                    "value": value,
                    "method": {
                        "input": (
                            "exact mathematical decimals; "
                            "measurement uncertainty excluded"
                        ),
                        "root": "not applicable: no roots",
                        "endpoint": "regular positive x; no endpoint removal",
                        "quadrature": (
                            "10 times sum full PSD resolution "
                            "differences at both precisions"
                        ),
                        "tail": (
                            "analytic radial K bound beyond x+60 plus "
                            "positive Laplace u/v tail bound beyond8"
                        ),
                        "formulation": (
                            "exact conditional parent-model integral; "
                            "unquantified model approximation excluded"
                        ),
                        "precision": (
                            "10 times maximum aligned full-path 50/70 difference"
                        ),
                        "roundoff": (
                            "heuristic full PSD relative floor1e-40; "
                            "arithmetic included once"
                        ),
                        "serialization": (
                            "actual central serialization residual "
                            "plus relative digit floor"
                        ),
                    }[name],
                }
                for name, value in result["components"].items()
            }
            record = {
                "schema_version": 1,
                "key": key,
                "status": "resolved",
                "quantity": {
                    "shape": [],
                    "data": [{"real": result["central"], "imag": "0e+0"}],
                },
                "uncertainty": {
                    "kind": "estimate",
                    "components": components,
                    "certificate": None,
                },
                "input_assumptions": "exact_input",
                "source_pins": fixtures["source_pins"],
                "procedure_hash": digest(args.procedure),
                "inputs": case["inputs"],
                "units": "kg^2 m^-4 Hz^-1"
                if representation == "Gfree"
                else "m^2 s^-4 Hz^-1",
                "input_convention": (
                    "one_sided_per_Hz_structure_function_coefficient_cT_squared"
                ),
                "admission_status": "NOT_FORMALLY_ADMITTED",
            }
            purpose_records.append(metadata)
            print(dumps(record), end="", flush=True)
    final_environment = inventory()
    if final_environment != observed:
        raise ValueError("reference imported unknown runtime edge")
    Path(args.environment).write_text(dumps(final_environment))
    Path(args.purpose_records).write_text(dumps(purpose_records))


def inventory():
    loaded = {
        name: {
            "path": getattr(module, "__file__", None),
            "sha256": digest(module.__file__)
            if getattr(module, "__file__", None) and Path(module.__file__).is_file()
            else None,
        }
        for name, module in sys.modules.copy().items()
    }
    if any(name == "gragra" or name.startswith("gragra.") for name in loaded):
        raise RuntimeError("production contamination")
    return {
        "python": sys.executable,
        "python_sha256": digest(sys.executable),
        "version": sys.version,
        "loaded_modules": loaded,
        "production_modules": [],
    }


def preflight(args):
    root = Path(args.root).resolve()
    head = git(root, "rev-parse", "HEAD")
    pins = {
        "E_commit": head,
        "E_tree": git(root, "rev-parse", "HEAD^{tree}"),
        "fixtures_sha256": digest(args.fixtures),
        "procedure_sha256": digest(args.procedure),
        "generator_sha256": digest(HERE / "reference.py"),
        "laplace_sha256": digest(HERE / "laplace.py"),
        "runner_sha256": digest(__file__),
    }
    if args.mode == "formal":
        if git(root, "status", "--porcelain"):
            raise ValueError("formal source tree must be clean")
        if not args.budget:
            raise ValueError("formal frozen budget required")
        pins["budget_sha256"] = digest(args.budget)
        if not args.freeze_declaration:
            raise ValueError("formal separate frozen declaration required")
        pins["freeze_declaration_sha256"] = digest(args.freeze_declaration)
        if not args.expected_environment:
            raise ValueError("reviewed reference environment allowlist required")
        pins["expected_environment_sha256"] = digest(args.expected_environment)
        pins["common_hashes"] = {path: digest(root / path) for path in COMMON_PATHS}
        for path in (
            args.fixtures,
            args.procedure,
            args.budget,
            args.freeze_declaration,
            HERE / "reference.py",
            HERE / "laplace.py",
            Path(__file__),
            *(root / path for path in COMMON_PATHS),
        ):
            path = Path(path).resolve()
            rel = path.relative_to(root).as_posix()
            committed = subprocess.check_output(
                ["git", "-C", str(root), "show", f"{head}:{rel}"]
            )
            if hashlib.sha256(committed).hexdigest() != digest(path):
                raise ValueError("formal inputs not in exact E")
        auth = json.loads(Path(args.admission).read_text())
        if (
            auth.get("pins") != pins
            or auth.get("decision") != "PASS"
            or auth.get("family") != "T1"
        ):
            raise ValueError("external admission mismatches exact source")
        campaign_root = Path(auth["attempt_parent"]).resolve()
        if Path(args.attempt).resolve().parent != campaign_root:
            raise ValueError("attempt path outside admitted campaign")
        previous = []
        for path in campaign_root.glob("*/manifest.json"):
            data = json.loads(path.read_text())
            if data.get("mode") == "formal" and data.get("pins", {}).get(
                "generator_sha256"
            ):
                previous.append(path)
        if len(previous) >= (5 if args.worker else 4):
            raise ValueError("formal four-attempt resource limit exhausted")
        freeze = json.loads(Path(args.freeze_declaration).read_text())
        if (
            freeze.get("status") != "FROZEN"
            or freeze.get("family") != "T1"
            or freeze.get("freeze_id") != "gen1-stochastic-F_T1-v1"
        ):
            raise ValueError("family freeze not independently declared")
        artifact_hashes = {}
        for field in ("fixture", "budget", "procedure"):
            expected = pins[("fixtures" if field == "fixture" else field) + "_sha256"]
            if freeze.get(field + "_sha256") != expected:
                raise ValueError("family freeze triplet byte mismatch")
            path = freeze[field + "_file"]
            if digest(root / path) != expected:
                raise ValueError("family freeze declared path mismatch")
            artifact_hashes[path] = expected
        if freeze.get("artifact_hashes") != artifact_hashes:
            raise ValueError("family freeze exact artifact map mismatch")
        for path, expected in freeze["baseline_pins"].items():
            if digest(root / path) != expected:
                raise ValueError("family frozen baseline changed")
        if set(freeze.get("reviews", {})) != {
            "science",
            "numerical",
            "provenance",
        }:
            raise ValueError("family freeze reviews incomplete")
        verify_protocol_reviews(freeze, artifact_hashes)
        concurrence_pin = freeze["root_concurrence"]
        if (
            concurrence_pin.get("decision") != "PASS_FOR_F_T1_V1_FREEZE"
            or digest(concurrence_pin["path"]) != concurrence_pin["sha256"]
        ):
            raise ValueError("root receipt byte mismatch")
        concurrence = json.loads(Path(concurrence_pin["path"]).read_text())
        if (
            concurrence.get("family") != "T1"
            or concurrence.get("decision") != "PASS_FOR_F_T1_V1_FREEZE"
            or concurrence.get("artifact_hashes") != artifact_hashes
            or concurrence.get("review_hashes")
            != {role: review["sha256"] for role, review in freeze["reviews"].items()}
        ):
            raise ValueError("root freeze concurrence missing")
        verify_implementation_reviews(auth, root, head)
        bundle = auth["durable_E_bundle"]
        if digest(bundle["path"]) != bundle["sha256"]:
            raise ValueError("durable E bundle bytes mismatch")
        if (
            bundle.get("restored_tree_hash") != pins["E_tree"]
            or bundle.get("expected_tree_hash") != pins["E_tree"]
        ):
            raise ValueError("durable E restoration tree certificate mismatch")
        subprocess.run(
            ["git", "-C", str(root), "bundle", "verify", bundle["path"]],
            check=True,
            capture_output=True,
        )
        refs = subprocess.check_output(
            ["git", "bundle", "list-heads", bundle["path"]], text=True
        )
        if not any(line.split()[0] == head for line in refs.splitlines()):
            raise ValueError("durable bundle lacks exact E")
        with tempfile.TemporaryDirectory(prefix="t1-bundle-restore-") as directory:
            subprocess.run(
                ["git", "init", "--bare", directory], check=True, capture_output=True
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    directory,
                    "fetch",
                    str(Path(bundle["path"]).resolve()),
                    head,
                ],
                check=True,
                capture_output=True,
            )
            restored_tree = git(directory, "rev-parse", f"{head}^{{tree}}")
            if restored_tree != pins["E_tree"]:
                raise ValueError("actual durable E restore tree mismatch")
        source = json.loads(Path(args.fixtures).read_text())["source_pins"][0]
        if digest(args.source_artifact) != source["sha256"]:
            raise ValueError("primary source artifact mismatch")
    return pins


def supervise(args):
    pins = preflight(args)
    attempt = Path(args.attempt).resolve()
    for parent in (attempt, *attempt.parents):
        if (parent / ".git").exists():
            raise ValueError("attempt must be outside every Git worktree")
    fixtures = json.loads(Path(args.fixtures).read_text())
    expected = [
        key for _, key, _ in plan_records(fixtures, digest(args.procedure), args.mode)
    ]
    attempt.mkdir(parents=True, exist_ok=False)
    (attempt / "manifest.json").write_text(
        dumps(
            {
                "mode": args.mode,
                "pins": pins,
                "admission_status": "NOT_FORMALLY_ADMITTED",
            }
        )
    )
    command = [
        sys.executable,
        "-I",
        str(Path(__file__).resolve()),
        "--worker",
        "--fixtures",
        str(Path(args.fixtures).resolve()),
        "--procedure",
        str(Path(args.procedure).resolve()),
        "--environment",
        str(attempt / "worker-environment.json"),
        "--purpose-records",
        str(attempt / "purpose-records.json"),
        "--mode",
        args.mode,
    ]
    if args.mode == "formal":
        command.extend(
            [
                "--root",
                args.root,
                "--admission",
                args.admission,
                "--source-artifact",
                args.source_artifact,
                "--budget",
                args.budget,
                "--freeze-declaration",
                args.freeze_declaration,
                "--expected-environment",
                args.expected_environment,
                "--attempt",
                str(attempt),
            ]
        )
    start = time.monotonic()
    exit_code, outcome = None, "FAILED"
    process, cleanup_confirmed, worker_error = None, True, None
    with (
        (attempt / "records.jsonl").open("wb") as output,
        (attempt / "stderr.txt").open("wb") as error,
    ):
        try:
            process = subprocess.Popen(
                command,
                stdout=output,
                stderr=error,
                start_new_session=True,
                env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
            )
            exit_code = process.wait(timeout=3600)
            outcome = "COMPLETE" if exit_code == 0 else "FAILED"
        except subprocess.TimeoutExpired:
            outcome = "TIMEOUT"
        except KeyboardInterrupt:
            outcome = "INTERRUPTED"
        except BaseException as exc:
            worker_error = str(exc)
        finally:
            if process is not None:
                try:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
                    exit_code = process.returncode
                except BaseException as exc:
                    cleanup_confirmed = False
                    worker_error = str(exc)
    if not cleanup_confirmed:
        outcome = "PENDING_CLEANUP"
    records, parse_error = [], None
    try:
        raw = (attempt / "records.jsonl").read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise ValueError("partial final JSONL line")
        for line in raw.splitlines():
            record = json.loads(
                line,
                object_pairs_hook=unique_object,
                parse_float=reject_noncanonical_number,
                parse_constant=reject_noncanonical_number,
            )
            if not isinstance(record, dict):
                raise ValueError("non-object record")
            records.append(record)
    except (ValueError, UnicodeError) as exc:
        parse_error = str(exc)
    generated = [r.get("key") for r in records]
    exp, actual = set(map(dumps, expected)), list(map(dumps, generated))
    unresolved = [r.get("key") for r in records if r.get("status") != "resolved"]
    coverage = (
        outcome == "COMPLETE"
        and cleanup_confirmed
        and bool(expected)
        and len(actual) == len(exp)
        and len(set(actual)) == len(actual)
        and set(actual) == exp
        and not unresolved
        and parse_error is None
    )
    seal = {
        "schema_version": 1,
        "family": "T1",
        "mode": args.mode,
        "pins": pins,
        "outcome": outcome,
        "exit_code": exit_code,
        "elapsed_seconds": str(time.monotonic() - start),
        "cleanup_confirmed": cleanup_confirmed,
        "worker_error": worker_error,
        "children_maxrss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "timeout_seconds": 3600,
        "expected_keys": expected,
        "generated_keys": generated,
        "unresolved_keys": unresolved,
        "missing_keys": [k for k in expected if dumps(k) not in actual],
        "unknown_keys": [k for k in generated if dumps(k) not in exp],
        "coverage_valid": coverage,
        "parse_error": parse_error,
        "admission_status": "NOT_FORMALLY_ADMITTED",
        "artifact_hashes": {
            p.name: digest(p) for p in attempt.iterdir() if p.is_file()
        },
    }
    (attempt / "seal.json").write_text(dumps(seal))
    for path in attempt.iterdir():
        path.chmod(0o444)
    attempt.chmod(0o555)
    return 0 if outcome == "COMPLETE" and coverage else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--inspect-environment", action="store_true")
    parser.add_argument("--fixtures", required=True)
    parser.add_argument("--procedure", required=True)
    parser.add_argument("--budget")
    parser.add_argument("--freeze-declaration")
    parser.add_argument("--expected-environment")
    parser.add_argument("--environment")
    parser.add_argument("--purpose-records")
    parser.add_argument("--root")
    parser.add_argument("--attempt")
    parser.add_argument(
        "--mode", choices=("calibration", "formal"), default="calibration"
    )
    parser.add_argument("--admission")
    parser.add_argument("--source-artifact")
    args = parser.parse_args()
    if args.inspect_environment:
        sys.meta_path.insert(0, DenyProduction())
        sys.path.insert(0, str(HERE))
        import reference  # noqa: F401

        protocol_helpers()

        print(dumps(inventory()), end="")
        return 0
    if args.worker:
        worker(args)
        return 0
    if (
        not args.root
        or not args.attempt
        or (args.mode == "formal" and (not args.admission or not args.source_artifact))
    ):
        parser.error(
            "root/attempt required; formal requires external admission and source bytes"
        )
    return supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())
