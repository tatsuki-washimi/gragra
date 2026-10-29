"""Freeze independent Sphere references and raw evidence outside the Git tree."""

import argparse
import dataclasses
import hashlib
import json
import platform
import time
from pathlib import Path

import mpmath

from .sphere_fixtures import sphere_fixtures
from .sphere_reference import sphere_reference


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
        Path(__file__).with_name("sphere_reference.py"),
        Path(__file__).with_name("sphere_fixtures.py"),
        Path(__file__).with_name("acceptance.py"),
        repository / "docs/design/gen1/sphere-science-protocol.md",
        repository / "docs/design/gen1/acceptance-error-budget.md",
    ]
    started = time.monotonic()
    counts = {"cases": 0, "resolved_observables": 0, "unresolved_observables": 0}
    with (output / "references.jsonl").open("w") as stream:
        for fixture in sphere_fixtures():
            if args.limit is not None and counts["cases"] >= args.limit:
                break
            values, metadata = sphere_reference(fixture, args.gravitational_constant)
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
        "protocol": "sphere-science-protocol.md",
        "production_values_seen": False,
        "gravitational_constant_si": args.gravitational_constant,
        "python": platform.python_version(),
        "mpmath": mpmath.__version__,
        "inputs": {str(path.relative_to(repository)): digest(path) for path in inputs},
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
