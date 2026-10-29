"""Freeze the predeclared high-degree regression independently of production."""

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import mpmath

from .sphere_high_degree_fixtures import sphere_high_degree_fixtures
from .sphere_high_degree_reference import sphere_high_degree_reference


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gravitational-constant", type=float, default=6.67430e-11)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    if output.is_relative_to(repository):
        parser.error("raw evidence must remain outside the Git tree")
    output.mkdir(parents=True, exist_ok=False)
    inputs = [
        Path(__file__).resolve(),
        *[
            Path(__file__).with_name(name)
            for name in (
                "sphere_high_degree_reference.py",
                "sphere_high_degree_fixtures.py",
                "sphere_reference.py",
                "sphere_fixtures.py",
                "acceptance.py",
            )
        ],
        repository / "docs/design/gen1/sphere-high-degree-science-protocol.md",
        repository / "docs/design/gen1/sphere-science-protocol.md",
        repository / "docs/design/gen1/acceptance-error-budget.md",
    ]
    hashes = {str(path.relative_to(repository)): digest(path) for path in inputs}
    for path in inputs:
        destination = output / "inputs" / path.relative_to(repository)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    fixtures = tuple(sphere_high_degree_fixtures())
    matrix_path = output / "predeclared-fixtures.json"
    matrix_path.write_text(
        json.dumps([dataclasses.asdict(fixture) for fixture in fixtures], indent=2)
        + "\n"
    )
    started = time.monotonic()
    counts = {"cases": 0, "resolved_observables": 0, "unresolved_observables": 0}
    with (output / "references.jsonl").open("w") as stream:
        for fixture in fixtures:
            values, metadata = sphere_high_degree_reference(
                fixture, args.gravitational_constant
            )
            serial = {}
            for name, record in values.items():
                uncertainty = record["uncertainty"]
                counts[
                    "resolved_observables"
                    if uncertainty.resolved()
                    else "unresolved_observables"
                ] += 1
                serial[name] = {
                    "values_si": [
                        [value.real, value.imag] for value in record["values_si"]
                    ],
                    "uncertainty": dataclasses.asdict(uncertainty),
                }
            counts["cases"] += 1
            stream.write(
                json.dumps(
                    {
                        "fixture": dataclasses.asdict(fixture),
                        "reference": serial,
                        "metadata": metadata,
                    },
                    allow_nan=False,
                )
                + "\n"
            )
            stream.flush()
            print(
                json.dumps({**counts, "elapsed_s": time.monotonic() - started}),
                flush=True,
            )
    manifest = {
        "protocol": "sphere-high-degree-science-protocol.md",
        "discovery_following_regression": True,
        "production_values_seen": False,
        "production_root_or_normalization_used": False,
        "gravitational_constant_si": args.gravitational_constant,
        "python": platform.python_version(),
        "mpmath": mpmath.__version__,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "inputs": hashes,
        "predeclared_fixture_sha256": digest(matrix_path),
        "reference_sha256": digest(output / "references.jsonl"),
        "counts": counts,
        "elapsed_s": time.monotonic() - started,
        "claim": "finite_references_mixed_exact_degree_bounds_and_numerical_estimates",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if counts["unresolved_observables"]:
        raise SystemExit("unresolved reference uncertainty")


if __name__ == "__main__":
    main()
