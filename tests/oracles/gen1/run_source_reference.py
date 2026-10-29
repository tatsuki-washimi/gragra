"""Freeze finite-source independent references without reading production outputs."""

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import mpmath

from .noncavity_fixtures import G_SI
from .source_fixtures import source_fixtures
from .source_reference import source_reference


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    if output.is_relative_to(repository):
        parser.error("raw evidence must remain outside Git")
    output.mkdir(parents=True, exist_ok=False)
    names = (
        "source_reference.py",
        "source_fixtures.py",
        "noncavity_reference.py",
        "noncavity_fixtures.py",
        "sphere_reference.py",
        "sphere_fixtures.py",
        "surface_reference.py",
        "surface_high_precision.py",
        "cavity_fixtures.py",
        "acceptance.py",
    )
    inputs = [
        Path(__file__).resolve(),
        *(Path(__file__).with_name(name) for name in names),
        repository / "docs/design/gen1/source-science-protocol.md",
        repository / "docs/design/gen1/acceptance-error-budget.md",
    ]
    input_hashes = {str(path.relative_to(repository)): digest(path) for path in inputs}
    for path in inputs:
        destination = output / "inputs" / path.relative_to(repository)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    counts = {"cases": 0, "resolved_observables": 0, "unresolved_observables": 0}
    started = time.monotonic()
    with (output / "references.jsonl").open("w") as stream:
        for fixture in source_fixtures():
            values, metadata = source_reference(fixture)
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
                    {"fixture": fixture, "reference": serial, "metadata": metadata},
                    allow_nan=False,
                )
                + "\n"
            )
            stream.flush()
            print(
                json.dumps(
                    {
                        **counts,
                        "family": fixture["family"],
                        "elapsed_s": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    manifest = {
        "protocol": "source-science-protocol.md",
        "production_values_seen": False,
        "gravitational_constant_si": G_SI,
        "python": platform.python_version(),
        "mpmath": mpmath.__version__,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "inputs": input_hashes,
        "reference_sha256": digest(output / "references.jsonl"),
        "counts": counts,
        "elapsed_s": time.monotonic() - started,
        "claim": "finite_references_mixed_exact_parity_bounds_and_numerical_estimates",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if counts["unresolved_observables"]:
        raise SystemExit("unresolved reference uncertainty")


if __name__ == "__main__":
    main()
