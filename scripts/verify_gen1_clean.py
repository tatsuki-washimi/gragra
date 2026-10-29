#!/usr/bin/env python3
"""Clean-environment verification of the Gen 1 B1 tests (versioned wrapper).

The wrapper does only the following:

- clones the given repository and checks out the given commits by SHA into a
  temporary workspace (the candidate, and the sibling production object that
  the comparator reads);
- lays out the sibling-repository path that the B1 production comparator
  resolves from its own location (``<workspace>/stoch-integration-20261002``);
- builds a dedicated environment from the tracked requirements file, with the
  candidate installed editable from its own clone;
- runs the existing test files under an origin guard and writes a JSON report.

It does not change any comparator, binding, expected value, tolerance, or
decision rule. It does not use or modify existing worktrees or shared conda
environments. Groups that cannot run in a clean environment are reported as
ENVIRONMENT_BOUND; they are never counted as passed.

Exit status: 0 when every executed group passes and nothing is environment
bound; 1 when a group fails; 2 when every executed group passes but the run is
incomplete.
"""

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

SIBLING_NAME = "stoch-integration-20261002"
SIBLING_OBJECT = "31e1a0e3cb9bca32cc7e00722580d73fcd2241d0"
ARITHMETIC_KERNEL = "tests/oracles/gen1/stochastic_acceptance.py"
ARITHMETIC_KERNEL_BLOB = "7be0818890d55350ed5bf538f676f903c3709a5e"
REQUIREMENTS = "requirements/gen1-verify.txt"

B1_FORMAL = [
    "tests/test_gen1_b1_formal_generator.py",
    "tests/test_gen1_b1_nonzero_generator.py",
    "tests/test_stochastic_reference_test_isolation.py",
]
B1_COMPARATOR = ["tests/test_gen1_b1_production_comparator.py"]
SIX_LOCATION = "tests/test_stochastic_a2_v2_diagnostic.py"
SIX_LOCATION_REASON = (
    "tests/oracles/gen1/stochastic_formal/a2/v2/binding.py pins six.__file__ "
    "to an absolute path in the author's historical conda environment. A clean "
    "environment cannot satisfy that pin, and the binding is frozen, so the "
    "file is not run here."
)


def run(argv, cwd=None, env=None, check=True):
    result = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True)
    if check and result.returncode != 0:
        command = " ".join(argv)
        raise SystemExit(
            f"command failed ({result.returncode}): {command}\n{result.stderr[-2000:]}"
        )
    return result


def git(repo, *args, check=True):
    return run(["git", "-C", str(repo), *args], check=check)


def clone_at(source, destination, sha):
    run(["git", "clone", "--quiet", "--no-checkout", source, str(destination)])
    git(destination, "checkout", "--quiet", "--detach", sha)


def junit_totals(path):
    root = ET.parse(path).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    tests = int(suite.get("tests", 0))
    failures = int(suite.get("failures", 0))
    errors = int(suite.get("errors", 0))
    skipped = int(suite.get("skipped", 0))
    return {
        "tests": tests,
        "passed": tests - failures - errors - skipped,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
    }


def run_pytest(python, gragra, name, reports, targets, extra=()):
    junit = reports / f"{name}.xml"
    env = dict(
        os.environ,
        GRAGRA_EXPECTED_SRC=str(gragra / "src"),
        PYTHONPATH=str(gragra / "scripts"),
    )
    argv = [
        str(python),
        "-m",
        "pytest",
        "-p",
        "pytest_origin_guard",
        "-q",
        "-p",
        "no:cacheprovider",
        f"--junitxml={junit}",
        *extra,
        *targets,
    ]
    result = run(argv, cwd=gragra, env=env, check=False)
    totals = junit_totals(junit) if junit.exists() else None
    clean = totals is not None and totals["failures"] + totals["errors"] == 0
    status = "PASS" if result.returncode == 0 and clean else "FAIL"
    return {
        "name": name,
        "status": status,
        "exit_code": result.returncode,
        "totals": totals,
        "targets": list(targets),
        "extra": list(extra),
    }


def sha256_of(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", required=True, help="repository URL or local path")
    parser.add_argument("--candidate", required=True, help="candidate commit SHA")
    parser.add_argument(
        "--review-pin", default=None, help="review commit SHA that must be reachable"
    )
    parser.add_argument(
        "--sibling-object",
        default=SIBLING_OBJECT,
        help="commit holding the arithmetic kernel",
    )
    parser.add_argument(
        "--workspace", default=None, help="empty directory for the temporary workspace"
    )
    parser.add_argument(
        "--report",
        default=None,
        help="JSON report path (default: inside the workspace)",
    )
    parser.add_argument(
        "--python", default="3.11.15", help="interpreter version for uv venv"
    )
    parser.add_argument("--uv", default="uv")
    parser.add_argument(
        "--full", action="store_true", help="also run every non-slow test file"
    )
    args = parser.parse_args()

    if args.workspace:
        workspace = Path(args.workspace)
    else:
        workspace = Path(tempfile.mkdtemp(prefix="gen1-verify-"))
    workspace.mkdir(parents=True, exist_ok=True)
    if any(workspace.iterdir()):
        raise SystemExit(f"workspace must be empty: {workspace}")
    gragra = workspace / "gragra"
    sibling = workspace / SIBLING_NAME
    reports = workspace / "reports"
    reports.mkdir()
    venv = workspace / "venv"
    python = venv / "bin" / "python"

    clone_at(args.source, gragra, args.candidate)
    clone_at(args.source, sibling, args.sibling_object)
    identities = {
        "candidate_commit": git(gragra, "rev-parse", "HEAD").stdout.strip(),
        "candidate_tree": git(gragra, "rev-parse", "HEAD^{tree}").stdout.strip(),
        "candidate_clean": git(gragra, "status", "--porcelain").stdout.strip() == "",
        "sibling_commit": git(sibling, "rev-parse", "HEAD").stdout.strip(),
        "sibling_kernel_blob": git(
            sibling, "rev-parse", f"HEAD:{ARITHMETIC_KERNEL}"
        ).stdout.strip(),
        "sibling_kernel_blob_expected": ARITHMETIC_KERNEL_BLOB,
    }
    if identities["sibling_kernel_blob"] != ARITHMETIC_KERNEL_BLOB:
        raise SystemExit("sibling arithmetic kernel blob differs from the pin")
    if args.review_pin:
        identities["review_pin"] = args.review_pin
        identities["review_pin_type"] = git(
            gragra, "cat-file", "-t", args.review_pin
        ).stdout.strip()

    requirements = gragra / REQUIREMENTS
    identities["requirements_sha256"] = sha256_of(requirements)
    run([args.uv, "venv", "--quiet", "--python", args.python, str(venv)])
    run(
        [
            args.uv,
            "pip",
            "install",
            "--quiet",
            "--python",
            str(python),
            "-r",
            str(requirements),
        ]
    )
    run(
        [
            args.uv,
            "pip",
            "install",
            "--quiet",
            "--python",
            str(python),
            "--no-deps",
            "-e",
            str(gragra),
        ]
    )
    version = run([str(python), "--version"])
    environment = {"python": (version.stdout or version.stderr).strip()}

    groups = [
        run_pytest(python, gragra, "b1-formal", reports, B1_FORMAL),
        run_pytest(python, gragra, "b1-production-comparator", reports, B1_COMPARATOR),
    ]
    bound = [
        {
            "name": "a2-v2-six-location",
            "status": "ENVIRONMENT_BOUND",
            "targets": [SIX_LOCATION],
            "reason": SIX_LOCATION_REASON,
        }
    ]
    if args.full:
        groups.append(
            run_pytest(
                python,
                gragra,
                "non-slow-all",
                reports,
                ["tests"],
                extra=["-m", "not slow", f"--ignore={SIX_LOCATION}"],
            )
        )

    complete = not bound and all(group["status"] == "PASS" for group in groups)
    failed = any(group["status"] == "FAIL" for group in groups)
    now = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    report = {
        "generated_utc": now,
        "source": args.source,
        "identities": identities,
        "environment": environment,
        "groups": groups,
        "environment_bound": bound,
        "complete": complete,
    }
    report_path = Path(args.report) if args.report else workspace / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for group in groups:
        print(f"{group['status']:5s} {group['name']} {group['totals']}")
    for item in bound:
        print(f"ENVIRONMENT_BOUND {item['name']}: {item['reason']}")
    print(f"report={report_path} complete={complete}")
    if failed:
        return 1
    return 0 if complete else 2


if __name__ == "__main__":
    sys.exit(main())
