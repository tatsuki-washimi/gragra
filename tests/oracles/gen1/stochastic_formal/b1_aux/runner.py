"""Production-denying B1 worker and fail-closed immutable attempt supervisor."""

import argparse
import contextlib
import fractions
import hashlib
import importlib.abc
import json
import math
import os
import platform
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    )


def reject_nonfinite(value):
    raise ValueError("nonfinite JSON constant: " + value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field: " + key)
        result[key] = value
    return result


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def production_modules():
    return sorted(k for k in sys.modules if k == "gragra" or k.startswith("gragra."))


class DenyProduction(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):  # noqa: ARG002
        if fullname == "gragra" or fullname.startswith("gragra."):
            raise ImportError(
                "B1 auxiliary reference process denies production imports"
            )
        return


def inventory():
    # Imports needed by Fraction/isqrt are deliberately loaded before this snapshot.
    assert fractions.Fraction and math.isqrt
    return {
        "python_executable": sys.executable,
        "python_executable_sha256": digest(sys.executable),
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "arithmetic": "stdlib Fraction/int only",
        "production_modules": production_modules(),
        "generator_sha256": digest(HERE / "reference.py"),
        "runner_sha256": digest(__file__),
        "loaded_modules": {
            k: {
                "path": getattr(v, "__file__", None),
                "sha256": digest(v.__file__)
                if getattr(v, "__file__", None) and Path(v.__file__).is_file()
                else None,
                "missing_reason": None
                if getattr(v, "__file__", None) and Path(v.__file__).is_file()
                else "builtin/frozen/no source file",
            }
            for k, v in sys.modules.copy().items()
        },
    }


def guard_worker(expected_path=None):
    if production_modules():
        raise RuntimeError("production module already loaded before reference import")
    sys.meta_path.insert(0, DenyProduction())
    observed = inventory()
    if expected_path and observed != json.loads(Path(expected_path).read_text()):
        raise RuntimeError("reviewed worker runtime environment mismatch")
    return observed


def worker(args):
    observed = guard_worker(args.expected_environment)
    Path(args.worker_environment).write_text(encode(observed))
    from reference import make_record  # Import only after observed environment matches.

    fixtures = json.loads(Path(args.fixtures).read_text())
    procedure = json.loads(Path(args.family_procedure).read_text())
    declaration = fixtures["expected_reference_auxiliary_keys"]
    ids = [encode(k) for k in declaration]
    if len(ids) != 56 or len(set(ids)) != 56:
        raise ValueError("expected exactly 56 unique independent auxiliary keys")
    by_id = {
        f["fixture_id"]: f
        for f in fixtures["connection_fixtures"]
        if f["evidence_class"] == "independent_first_order_reference"
    }
    records = [
        make_record(
            k,
            by_id[k["fixture_id"]],
            procedure["source_pins"],
            digest(args.freeze),
            ["55", "60"],
        )
        for k in declaration
    ]
    if production_modules():
        raise RuntimeError("production module loaded")
    final_modules = inventory()["loaded_modules"]
    allowed = dict(observed["loaded_modules"])
    allowed["reference"] = {
        "path": str(HERE / "reference.py"),
        "sha256": digest(HERE / "reference.py"),
        "missing_reason": None,
    }
    if final_modules != allowed:
        raise RuntimeError("unreviewed runtime import edge or changed module bytes")
    for record in records:
        print(encode(record), end="")


def source_pins(args, root):
    head = git(root, "rev-parse", "HEAD")
    return {
        "f_b1_aux_sha256": digest(args.freeze),
        "fixtures_sha256": digest(args.fixtures),
        "family_procedure_sha256": digest(args.family_procedure),
        "generator_sha256": digest(HERE / "reference.py"),
        "runner_sha256": digest(__file__),
        "E_aux_SHA": head,
        "E_aux_tree": git(root, "rev-parse", "HEAD^{tree}"),
    }


def formal_preflight(args, root, pins):
    if git(root, "status", "--porcelain"):
        raise ValueError("formal E must be completely clean")
    if not args.integration_ref:
        raise ValueError("formal E requires exact reachable integration ref")
    integration_head = git(root, "rev-parse", args.integration_ref)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "merge-base",
            "--is-ancestor",
            pins["E_aux_SHA"],
            integration_head,
        ],
        check=True,
    )
    admission = json.loads(Path(args.admission).read_text())
    if (
        admission.get("pins") != pins
        or admission.get("decision") != "PASS"
        or admission.get("integration_head") != integration_head
    ):
        raise ValueError("external admission does not match exact E/F/integration pins")
    reviews = admission.get("reviews", {})
    if set(reviews) != {"science", "numerical", "provenance"}:
        raise ValueError("three independent source reviews required")
    for review in reviews.values():
        if review.get("source_pins") != {
            name: pins[name]
            for name in ("generator_sha256", "runner_sha256", "f_b1_aux_sha256")
        }:
            raise ValueError("review does not bind exact candidate source bytes")
        if (
            digest(review["path"]) != review["sha256"]
            or review.get("decision") != "PASS"
        ):
            raise ValueError("review bytes or verdict mismatch")
    freeze = json.loads(Path(args.freeze).read_text())
    declaration = json.loads(Path(args.freeze_declaration).read_text())
    if (
        freeze.get("protocol_id") != "gen1-stochastic-F_B1_aux-v1"
        or declaration.get("status") != "FROZEN"
        or declaration.get("f_b1_aux_sha256") != pins["f_b1_aux_sha256"]
        or declaration.get("freeze_id") != freeze.get("protocol_id")
    ):
        raise ValueError("unfrozen or unconcurred F_B1_aux declaration")
    author = declaration["author_proposal"]
    if digest(author["path"]) != author["sha256"]:
        raise ValueError("frozen author concurrence bytes mismatch")
    for artifact in declaration["reviews"].values():
        if digest(artifact["path"]) != artifact["sha256"]:
            raise ValueError("frozen protocol review bytes mismatch")
    concurrence = json.loads(
        Path(declaration["reviews"]["root_concurrence"]["path"]).read_text()
    )
    if (
        concurrence.get("decision") != "PASS_FOR_F_B1_AUX_FREEZE"
        or concurrence.get("controller") != "/root"
        or concurrence.get("f_b1_aux_sha256") != pins["f_b1_aux_sha256"]
        or concurrence.get("latest_required_review_verdicts")
        != {"science": "PASS", "numerical": "PASS", "provenance": "PASS"}
    ):
        raise ValueError("invalid root freeze concurrence")
    if digest(args.freeze_declaration) != admission.get("freeze_declaration_sha256"):
        raise ValueError("freeze declaration bytes not externally admitted")
    if digest(args.expected_environment) != admission.get(
        "expected_environment_sha256"
    ):
        raise ValueError("expected environment bytes not reviewed/admitted")
    fixtures = json.loads(Path(args.fixtures).read_text())
    if (
        freeze["coverage"]["expected_reference_auxiliary_keys"]
        != fixtures["expected_reference_auxiliary_keys"]
    ):
        raise ValueError("frozen key declaration mismatch")
    for relative, expected in freeze["baseline_pins"].items():
        if digest(root / relative) != expected:
            raise ValueError("frozen baseline input bytes mismatch")
    for path in (
        args.freeze,
        args.freeze_declaration,
        args.fixtures,
        args.family_procedure,
        str(HERE / "reference.py"),
        str(__file__),
    ):
        relative = Path(path).resolve().relative_to(root).as_posix()
        committed = subprocess.check_output(
            ["git", "-C", str(root), "show", f"{pins['E_aux_SHA']}:{relative}"]
        )
        if hashlib.sha256(committed).hexdigest() != digest(path):
            raise ValueError("formal input bytes must belong to exact E")
    if not args.source_artifact:
        raise ValueError("formal primary source bytes required")
    return {
        "integration_head": integration_head,
        "admission_sha256": digest(args.admission),
        "freeze_declaration_sha256": digest(args.freeze_declaration),
        "expected_environment_sha256": digest(args.expected_environment),
    }


def validate_attempt_location(attempt, root, args):
    if attempt == root or root in attempt.parents:
        raise ValueError("attempt must be outside the source worktree")
    for source in (
        HERE,
        Path(args.freeze).resolve().parent,
        Path(args.fixtures).resolve().parent,
    ):
        if attempt == source or source in attempt.parents:
            raise ValueError("attempt must be outside every input/source path")
    for name in ("source_artifact", "expected_environment", "freeze_declaration"):
        value = getattr(args, name, None)
        if value:
            source = Path(value).resolve().parent
            if attempt == source or source in attempt.parents:
                raise ValueError("attempt must be outside every auxiliary source path")
    # Check destination's existing ancestors for any repository/worktree, including
    # another checkout or gitdir. resolve() also rejects symlink escapes into them.
    for parent in (attempt, *attempt.parents):
        if (
            (parent / ".git").exists()
            or (parent / "HEAD").is_file()
            and (parent / "objects").is_dir()
        ):
            raise ValueError("attempt must be outside all Git directories")


def live_group_members(pgid):
    """Linux process evidence excluding reaped/dead zombies; unknown fails closed."""
    members = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
                if int(fields[2]) == pgid and fields[0] != "Z":
                    members.append(int(entry.name))
            except FileNotFoundError:
                continue
    return members


def stop_worker(process):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, sig)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            continue
        if not live_group_members(process.pid):
            return True
    return process.poll() is not None and not live_group_members(process.pid)


def supervise(args):
    root = Path(args.root).resolve()
    pins = source_pins(args, root)
    outer_pins = formal_preflight(args, root, pins) if args.mode == "formal" else {}
    primary = json.loads(Path(args.family_procedure).read_text())["source_pins"]
    if args.source_artifact and digest(args.source_artifact) != primary[0]["sha256"]:
        raise ValueError("primary source byte mismatch")
    expected = json.loads(Path(args.fixtures).read_text())[
        "expected_reference_auxiliary_keys"
    ]
    attempt = Path(args.attempt).resolve()
    validate_attempt_location(attempt, root, args)
    attempt.mkdir(parents=True, exist_ok=False)
    started, process, exit_code = time.monotonic(), None, None
    outcome, cleanup_confirmed, errors = "FAILED", True, []
    records, raw, artifact_hashes = [], b"", {}
    timeout = (
        300 if args.mode == "formal" else getattr(args, "development_timeout", 300)
    )
    try:
        (attempt / "manifest.json").write_text(
            encode(
                {
                    "mode": args.mode,
                    "pins": pins,
                    **outer_pins,
                    "admission_status": "NOT_FORMALLY_ADMITTED",
                }
            )
        )
        (attempt / "supervisor-environment.json").write_text(encode(inventory()))
        command = [
            sys.executable,
            "-I",
            str(Path(__file__).resolve()),
            "--worker",
            "--freeze",
            str(Path(args.freeze).resolve()),
            "--fixtures",
            str(Path(args.fixtures).resolve()),
            "--family-procedure",
            str(Path(args.family_procedure).resolve()),
            "--worker-environment",
            str(attempt / "worker-environment.json"),
        ]
        if args.expected_environment:
            (attempt / "expected-environment.json").write_bytes(
                Path(args.expected_environment).read_bytes()
            )
            command += [
                "--expected-environment",
                str(attempt / "expected-environment.json"),
            ]
        with (
            (attempt / "records.jsonl").open("wb") as output,
            (attempt / "stderr.txt").open("wb") as error,
        ):
            process = subprocess.Popen(
                command,
                stdout=output,
                stderr=error,
                start_new_session=True,
                env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
            )
            exit_code = process.wait(timeout=timeout)
        outcome = "COMPLETE" if exit_code == 0 else "FAILED"
        raw = (attempt / "records.jsonl").read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise ValueError("partial final JSONL line")
        records = [
            json.loads(
                line, parse_constant=reject_nonfinite, object_pairs_hook=unique_object
            )
            for line in raw.splitlines()
        ]
        if any(not isinstance(record, dict) for record in records):
            raise ValueError("non-object reference record")
    except BaseException as exc:
        outcome = (
            "TIMEOUT"
            if isinstance(exc, subprocess.TimeoutExpired)
            else (
                "INTERRUPTED"
                if isinstance(exc, (KeyboardInterrupt, SystemExit))
                else "FAILED"
            )
        )
        errors.append({"type": type(exc).__name__, "message": str(exc)})
    finally:
        # Always prove process/group cleanup before constructing the terminal seal.
        if process is not None:
            try:
                cleanup_confirmed = stop_worker(process)
                exit_code = process.returncode
            except BaseException as exc:
                cleanup_confirmed = False
                errors.append({"type": type(exc).__name__, "message": str(exc)})
        if not cleanup_confirmed:
            outcome = "PENDING_CLEANUP"
        try:
            raw = (attempt / "records.jsonl").read_bytes()
        except BaseException as exc:
            errors.append({"type": type(exc).__name__, "message": str(exc)})
        try:
            for path in attempt.iterdir():
                if path.is_file():
                    try:
                        artifact_hashes[path.name] = digest(path)
                    except BaseException as exc:
                        errors.append({"type": type(exc).__name__, "message": str(exc)})
        except BaseException as exc:
            errors.append({"type": type(exc).__name__, "message": str(exc)})
        try:
            maxrss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        except BaseException as exc:
            maxrss = None
            errors.append({"type": type(exc).__name__, "message": str(exc)})
        generated = [record.get("key") for record in records]
        generated_ids = list(map(encode, generated))
        unresolved = [r.get("key") for r in records if r.get("status") != "resolved"]
        coverage = (
            outcome == "COMPLETE"
            and cleanup_confirmed
            and not errors
            and len(generated_ids) == 56
            and len(set(generated_ids)) == 56
            and set(generated_ids) == set(map(encode, expected))
            and not unresolved
        )
        if not coverage and outcome == "COMPLETE":
            outcome = "FAILED"
        seal = {
            "schema_version": 1,
            "record_kind": "b1_aux_attempt_seal",
            **pins,
            **outer_pins,
            "mode": args.mode,
            "exit_code": exit_code,
            "outcome": outcome,
            "elapsed_seconds": str(time.monotonic() - started),
            "cleanup_confirmed": cleanup_confirmed,
            "resources": {
                "timeout_seconds": timeout,
                "children_maxrss_kib": maxrss,
            },
            "expected_keys": expected,
            "generated_keys": generated,
            "unresolved_keys": unresolved,
            "missing_keys": [k for k in expected if encode(k) not in generated_ids],
            "unknown_keys": [
                k for k in generated if encode(k) not in set(map(encode, expected))
            ],
            "coverage_valid": coverage,
            "errors": errors,
            "records_sha256": hashlib.sha256(raw).hexdigest(),
            "record_hashes": [
                hashlib.sha256(line).hexdigest()
                for line in raw.splitlines(keepends=True)
            ],
            "artifact_hashes": artifact_hashes,
            "admission_status": "NOT_FORMALLY_ADMITTED",
        }
        # Final write errors cannot be converted into a usable seal. They propagate
        # fail-closed; filesystem loss is never claimed to be a sealed outcome.
        (attempt / "seal.json").write_text(encode(seal))
        for path in attempt.iterdir():
            path.chmod(0o444)
        attempt.chmod(0o555)
    return 0 if coverage else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--inspect-environment", action="store_true")
    for name in (
        "freeze",
        "fixtures",
        "family-procedure",
        "worker-environment",
        "root",
        "attempt",
        "admission",
        "source-artifact",
        "integration-ref",
        "freeze-declaration",
        "expected-environment",
    ):
        parser.add_argument("--" + name)
    parser.add_argument(
        "--mode", choices=["development", "formal"], default="development"
    )
    args = parser.parse_args()
    if args.inspect_environment:
        if production_modules():
            raise RuntimeError("production module already loaded")
        print(encode(inventory()), end="")
        return 0
    if not all((args.freeze, args.fixtures, args.family_procedure)):
        parser.error("frozen inputs required")
    if args.worker:
        sys.path.insert(0, str(HERE))
        worker(args)
        return 0
    if (
        not args.root
        or not args.attempt
        or (
            args.mode == "formal"
            and not all(
                (
                    args.admission,
                    args.freeze_declaration,
                    args.expected_environment,
                    args.integration_ref,
                )
            )
        )
    ):
        parser.error(
            "supervisor needs root/attempt; formal needs reviewed E/F/environment/ref"
        )
    return supervise(args)


if __name__ == "__main__":
    sys.exit(main())
