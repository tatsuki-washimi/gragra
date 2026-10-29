"""Exercise extension models through fresh Python interpreters."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("kind", ("static", "phasor"))
def test_external_newtonian_workflow_in_separate_process(kind: str) -> None:
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("workflow_runner.py")), kind],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(completed.stdout)
    assert report == {
        "kind": kind,
        "source": str((root / "src" / "gragra" / "__init__.py").resolve()),
        "target_count": 1,
        "frequency_count": 0 if kind == "static" else 1,
        "runs_per_evaluation": [1, 1],
        "selected_observables": ["gradient", "potential"],
        "difference": 1.0,
    }


def test_external_model_files_import_only_public_gragra_names() -> None:
    root = Path(__file__).resolve().parent
    violations: list[str] = []
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    parts = alias.name.split(".")
                    if parts[0] == "gragra" and any(
                        part.startswith("_") for part in parts[1:]
                    ):
                        violations.append(
                            f"{path.name}:{node.lineno}: import {alias.name}"
                        )
            elif isinstance(node, ast.ImportFrom) and node.module:
                parts = node.module.split(".")
                if parts[0] != "gragra":
                    continue
                private_parts = [part for part in parts[1:] if part.startswith("_")]
                private_names = [
                    alias.name for alias in node.names if alias.name.startswith("_")
                ]
                if private_parts or private_names:
                    imported = ".".join(parts + private_names)
                    violations.append(f"{path.name}:{node.lineno}: import {imported}")

    assert not violations, "private gragra imports found:\n" + "\n".join(violations)
