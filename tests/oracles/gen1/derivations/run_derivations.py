"""Run the L0 formula gate and persist exact input/environment/artifact hashes."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import mpmath
import numpy
import scipy
import sympy

SCRIPTS = (
    "verify_non_cavity.py",
    "verify_point_force.py",
    "verify_sphere.py",
    "verify_transforms.py",
    "verify_cylinder_cartesian.py",
    "check_cuboid_transform.py",
    "check_cylinder_spectral.py",
    "certified_tail_envelopes.py",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    checkout = script_dir.parents[3]
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = args.evidence_root.resolve() / stamp
    run_dir.mkdir(parents=True)
    input_dir = run_dir / "inputs"
    input_dir.mkdir()
    environment = {
        "python": sys.version,
        "executable": sys.executable,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "libraries": {
            name: {"version": module.__version__, "import_path": module.__file__}
            for name, module in (
                ("sympy", sympy),
                ("mpmath", mpmath),
                ("numpy", numpy),
                ("scipy", scipy),
            )
        },
        "product_import": None,
        "product_import_reason": "standalone derivations import no gragra package",
    }
    env_path = run_dir / "environment.json"
    env_path.write_text(json.dumps(environment, indent=2) + "\n")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
    ).strip()
    inputs = []
    for path in sorted(script_dir.glob("*.py")):
        snapshot = input_dir / path.name
        snapshot.write_bytes(path.read_bytes())
        inputs.append({"path": str(path.relative_to(checkout)), "sha256": digest(path)})
    for path in sorted((checkout / "docs/design/gen1").glob("*.md")):
        snapshot = input_dir / path.name
        snapshot.write_bytes(path.read_bytes())
        inputs.append({"path": str(path.relative_to(checkout)), "sha256": digest(path)})
    tree = hashlib.sha256()
    for item in inputs:
        tree.update(item["path"].encode())
        tree.update(item["sha256"].encode())
    records = []
    for name in SCRIPTS:
        path = script_dir / name
        command = [sys.executable, "-u", str(path)]
        logfile = run_dir / f"{path.stem}.log"
        start = dt.datetime.now(dt.UTC)
        with logfile.open("w") as stream:
            result = subprocess.run(
                command,
                cwd=checkout,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        record = {
            "command": command,
            "cwd": str(checkout),
            "exit_code": result.returncode,
            "status": "pass" if result.returncode == 0 else "fail",
            "input_sha256": digest(path),
            "log_path": str(logfile),
            "log_sha256": digest(logfile),
            "started_utc": start.isoformat(),
            "elapsed_seconds": (dt.datetime.now(dt.UTC) - start).total_seconds(),
            "classification": (
                "directed_interval_tail_evaluation_with_finite_crosschecks"
                if name == "certified_tail_envelopes.py"
                else "finite_numerical_formula_check"
                if name.startswith("check_")
                else "exact_CAS_with_separately_labelled_finite_checks"
            ),
        }
        records.append(record)
        print(f"{name}: {record['status']} exit={result.returncode}", flush=True)
    sources = []
    for name, url in (
        (
            "harms2019.html",
            "https://link.springer.com/article/10.1007/s41114-019-0022-2",
        ),
        ("dlmf-10-32.html", "https://dlmf.nist.gov/10.32"),
        ("dlmf-10-47.html", "https://dlmf.nist.gov/10.47"),
        ("dlmf-10-53.html", "https://dlmf.nist.gov/10.53"),
        ("python-decimal.html", "https://docs.python.org/3/library/decimal.html"),
    ):
        path = args.evidence_root.resolve() / name
        if path.exists():
            sources.append({"url": url, "snapshot": str(path), "sha256": digest(path)})
    guide = Path(
        "/home/washimi/.paseo/uploads/"
        "upload_168e2435-d576-4a30-b8b4-be8e04a4ac58/"
        "ChatGPT-gragra②主要公式の最終選定-20261001-1847.md"
    )
    manifest = {
        "schema_version": 1,
        "lane": "L0",
        "gate": "formula_verified"
        if all(r["exit_code"] == 0 for r in records)
        else "partial",
        "production_verified": False,
        "base_sha": "d2df70688f943492d0405222bb0aeb132f2c6cc7",
        "execution_revision": revision,
        "derivation_input_tree_sha256": tree.hexdigest(),
        "generated_utc": dt.datetime.now(dt.UTC).isoformat(),
        "environment": environment,
        "environment_artifact": {"path": str(env_path), "sha256": digest(env_path)},
        "conditions": {
            "conventions": "SI; a=-grad Phi; Gamma=-Hessian Phi; G once",
            "norms": "complex absolute/Euclidean/Frobenius all nine components",
            "CAS_domains": "positive lengths/speeds/cutoffs; integer modes; C1 force",
            "sphere_numeric": "mpmath 45 dps; l=0..4 positive derivative roots 1..3",
            "cuboid_numeric": (
                "float64, indices110/211/001, 9 targets including face extension, "
                "surface100/140"
            ),
            "cylinder_numeric": (
                "float64 R=1 H=1.4, m=0..3 q=1 sin/cos + radialconstant, "
                "3 targets, surface100/140"
            ),
            "rigorous_claim": (
                "analytic truncation component tails only; "
                "not total arithmetic/quadrature uncertainty"
            ),
            "acceptance": "no production acceptance claimed by this formula gate",
        },
        "inputs": inputs,
        "retrieval_guide": (
            {
                "path": str(guide),
                "sha256": digest(guide),
                "role": "guide_not_verification",
                "availability": "available",
            }
            if guide.exists()
            else {
                "path": str(guide),
                "sha256": None,
                "role": "guide_not_verification",
                "availability": "unavailable",
            }
        ),
        "original_sources": sources,
        "records": records,
        "native_audit_artifacts": [
            {"path": str(path), "sha256": digest(path)}
            for path in (
                args.evidence_root.resolve() / "sca-native-regression.log",
                args.evidence_root.resolve() / "sca-native-regression.json",
            )
            if path.exists()
        ],
        "historical_attempts": [
            {
                "path": str(path),
                "sha256": digest(path),
                "role": "retained_prior_attempt_not_current_gate",
            }
            for path in sorted(args.evidence_root.resolve().glob("*.log"))
        ],
    }
    raw_manifest = run_dir / "manifest.json"
    raw_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    manifest["raw_manifest"] = {
        "path": str(raw_manifest),
        "sha256": digest(raw_manifest),
    }
    path = checkout / "docs/design/gen1/evidence-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"manifest: {path}", flush=True)
    if manifest["gate"] != "formula_verified":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
