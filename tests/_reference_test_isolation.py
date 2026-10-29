"""Run unchanged production-blind test bodies in fresh Python interpreters."""

import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

CHILD_NODE = "GRAGRA_REFERENCE_TEST_CHILD_NODE"


class ReferenceChildError(AssertionError):
    """The selected child did not execute exactly one passing test."""


def run_node(repository, nodeid, output):
    """Preserve child diagnostics and reject every outcome except one real pass."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    junit = output / "junit.xml"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-o",
        "addopts=",
        "-o",
        "xfail_strict=true",
        "-q",
        "--junitxml=" + str(junit),
        nodeid,
    ]
    environment = os.environ.copy()
    environment.pop("PYTEST_ADDOPTS", None)
    environment[CHILD_NODE] = nodeid
    result = subprocess.run(
        command,
        cwd=repository,
        env=environment,
        capture_output=True,
        check=False,
    )
    (output / "stdout").write_bytes(result.stdout)
    (output / "stderr").write_bytes(result.stderr)
    raw = junit.read_bytes() if junit.is_file() else b""
    parse_error = None
    try:
        cases = ET.fromstring(raw).findall(".//testcase")
    except ET.ParseError as error:
        cases = []
        parse_error = str(error)
    counts = {
        name: sum(case.find(tag) is not None for case in cases)
        for name, tag in (
            ("failures", "failure"), ("errors", "error"), ("skipped", "skipped")
        )
    }
    receipt = {
        "nodeid": nodeid,
        "command": command,
        "exit_code": result.returncode,
        "testcases": len(cases),
        "passed": len(cases) - sum(counts.values()),
        **counts,
        "junit": str(junit),
        "junit_sha256": hashlib.sha256(raw).hexdigest(),
        "stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(result.stderr).hexdigest(),
        "parse_error": parse_error,
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if not (
        result.returncode == 0
        and len(cases) == receipt["passed"] == 1
        and all(value == 0 for value in counts.values())
        and cases[0].get("name") == nodeid.rsplit("::", 1)[-1]
    ):
        raise ReferenceChildError(
            json.dumps(receipt, indent=2) + "\n"
            + result.stdout.decode(errors="replace")
            + result.stderr.decode(errors="replace")
        )
    return receipt
