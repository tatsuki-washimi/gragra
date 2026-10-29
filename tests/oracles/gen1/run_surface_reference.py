"""Independent finite-surface evidence collector; contains no product imports."""

import argparse
import dataclasses
import hashlib
import json
import platform
import time
from pathlib import Path

import mpmath
import numpy

from .cavity_fixtures import cuboid_fixtures, cylinder_fixtures
from .surface_reference import surface_reference


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("cuboid", "cylinder"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gravitational-constant", type=float, default=6.67430e-11)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    if output.is_relative_to(repository):
        parser.error("raw evidence must remain outside the Git working tree")
    output.mkdir(parents=True, exist_ok=False)
    inputs = [
        Path(__file__).resolve(),
        Path(__file__).with_name("surface_reference.py"),
        Path(__file__).with_name("surface_high_precision.py"),
        Path(__file__).with_name("cavity_fixtures.py"),
        Path(__file__).with_name("sphere_reference.py"),
        Path(__file__).with_name("acceptance.py"),
        repository / "docs/design/gen1/surface-science-protocol.md",
        repository / "docs/design/gen1/acceptance-error-budget.md",
    ]
    counts = {"cases": 0, "resolved_observables": 0, "unresolved_observables": 0}
    input_hashes = {str(path.relative_to(repository)): digest(path) for path in inputs}
    snapshot = output / "inputs"
    for path in inputs:
        destination = snapshot / path.relative_to(repository)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    started = time.monotonic()
    fixtures = cuboid_fixtures() if args.family == "cuboid" else cylinder_fixtures()
    with (output / "references.jsonl").open("w") as stream:
        for fixture in fixtures:
            if args.limit is not None and counts["cases"] >= args.limit:
                break
            values, metadata = surface_reference(fixture, args.gravitational_constant)
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
        "protocol": "surface-science-protocol.md",
        "family": args.family,
        "production_values_seen": False,
        "gravitational_constant_si": args.gravitational_constant,
        "python": platform.python_version(),
        "mpmath": mpmath.__version__,
        "numpy": numpy.__version__,
        "inputs": input_hashes,
        "reference_sha256": digest(output / "references.jsonl"),
        "counts": counts,
        "elapsed_s": time.monotonic() - started,
        "claim": "references_with_recorded_uncertainty_estimates",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if counts["unresolved_observables"]:
        raise SystemExit("unresolved reference uncertainty")


if __name__ == "__main__":
    main()
