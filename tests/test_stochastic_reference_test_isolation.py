"""Production-blind reference assertions must survive a mixed pytest session."""

import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from tests._reference_test_isolation import ReferenceChildError, run_node


def test_production_loaded_parent_runs_original_reference_assertions(tmp_path):
    repository = Path(__file__).parents[1]
    node = (
        "tests/test_gen1_b1_formal_generator.py::"
        "test_build_all_unresolved_retained"
    )
    junit = tmp_path / "poisoned-parent.xml"
    arguments = ["-o", "addopts=", "-q", "--junitxml=" + str(junit), node]
    program = (
        "import gragra,pytest,sys; "
        "assert 'gragra.constants' in sys.modules; "
        "raise SystemExit(pytest.main(" + repr(arguments) + "))"
    )
    environment = os.environ.copy()
    environment.pop("PYTEST_ADDOPTS", None)
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=repository,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    cases = ET.parse(junit).getroot().findall(".//testcase")
    assert len(cases) == 1
    assert not any(
        cases[0].find(name) is not None for name in ("failure", "error", "skipped")
    )
    properties = {
        entry.get("name"): entry.get("value")
        for entry in cases[0].findall("properties/property")
    }
    child = json.loads(properties["reference_child"])
    assert child["nodeid"] == node
    assert child["exit_code"] == 0
    assert child["testcases"] == child["passed"] == 1
    assert child["failures"] == child["errors"] == child["skipped"] == 0


@pytest.mark.parametrize(
    ("source", "selector"),
    [
        (
            "def test_original():\n    assert False, 'original failure'\n",
            "::test_original",
        ),
        (
            "import pytest\ndef test_original():\n    pytest.skip('no pass')\n",
            "::test_original",
        ),
        (
            "import pytest\n@pytest.mark.xfail(reason='no pass')\n"
            "def test_original():\n    assert False\n", "::test_original",
        ),
        (
            "import pytest\n@pytest.mark.xfail(reason='unexpected pass')\n"
            "def test_original():\n    assert True\n", "::test_original",
        ),
        ("VALUE = 1\n", ""),
        (
            "import pytest\n@pytest.mark.parametrize('value', [1, 2])\n"
            "def test_original(value):\n    assert value > 0\n", "::test_original",
        ),
    ],
    ids=["failure", "skip", "xfail", "xpass", "empty", "multiple"],
)
def test_nonpassing_or_ambiguous_child_cannot_be_reported_as_pass(
    tmp_path, source, selector
):
    (tmp_path / "test_original.py").write_text(source)
    with pytest.raises(ReferenceChildError):
        run_node(tmp_path, "test_original.py" + selector, tmp_path / "diagnostics")
    receipt = json.loads((tmp_path / "diagnostics/receipt.json").read_text())
    assert receipt["nodeid"] == "test_original.py" + selector
    assert (tmp_path / "diagnostics/stdout").is_file()
    assert (tmp_path / "diagnostics/stderr").is_file()
