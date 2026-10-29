"""Bounded parallel collector with hash-verified independent reference seeds."""

import argparse
import dataclasses
import hashlib
import json
import math
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mpmath
import numpy

from .acceptance import OracleUncertainty
from .cavity_fixtures import cuboid_fixtures, cylinder_fixtures
from .surface_reference import surface_reference


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def digest(path):
    return digest_bytes(path.read_bytes())


def fixture_key(fixture):
    return json.dumps(fixture, sort_keys=True)


def calculate(fixture):
    values, metadata = surface_reference(fixture)
    serial = {}
    for name, record in values.items():
        serial[name] = {
            "values_si": [[v.real, v.imag] for v in record["values_si"]],
            "uncertainty": dataclasses.asdict(record["uncertainty"]),
        }
    return {
        "fixture": dataclasses.asdict(fixture),
        "reference": serial,
        "metadata": metadata,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("cuboid", "cylinder"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("bounded workers must be 1..4")
    repository = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    if output.is_relative_to(repository):
        parser.error("raw evidence must remain outside Git")
    output.mkdir(parents=True, exist_ok=False)
    names = (
        "acceptance.py",
        "cavity_fixtures.py",
        "surface_reference.py",
        "surface_high_precision.py",
        "sphere_reference.py",
        "sphere_fixtures.py",
    )
    science_inputs = [Path(__file__).with_name(name) for name in names]
    science_inputs += [
        repository / "docs/design/gen1/surface-science-protocol.md",
        repository / "docs/design/gen1/acceptance-error-budget.md",
    ]
    inputs = [Path(__file__).resolve(), *science_inputs]
    input_hashes = {str(path.relative_to(repository)): digest(path) for path in inputs}
    for path in inputs:
        destination = output / "inputs" / path.relative_to(repository)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    seeds = {}
    seed_evidence = None
    if args.seed:
        seed = args.seed.resolve()
        # The old collector saved the executed scientific files at run start.
        for path in science_inputs:
            snapshot = seed.parent / "inputs" / path.relative_to(repository)
            if not snapshot.exists() and path.name == "sphere_fixtures.py":
                # Old collector did not list this unused Sphere-only dependency.
                continue
            if not snapshot.exists() or digest(snapshot) != digest(path):
                raise ValueError(f"seed scientific input differs: {path}")
        data = seed.read_bytes()
        (output / "seed-references.jsonl").write_bytes(data)
        for line in data.decode().splitlines():
            row = json.loads(line)
            valid = True
            for name, record in row["reference"].items():
                uncertainty = OracleUncertainty(**record["uncertainty"])
                norm = math.hypot(*(v for pair in record["values_si"] for v in pair))
                budget = (
                    1e-12 * row["metadata"]["scales_si"][name]
                    + (1e-9 if name == "gradient" else 1e-10) * norm
                )
                valid &= (
                    uncertainty.resolved()
                    and uncertainty.used_uncertainty_si <= 0.05 * budget
                )
            if valid:
                seeds[fixture_key(row["fixture"])] = row
        seed_evidence = {
            "original_path": str(seed),
            "copied_sha256": digest_bytes(data),
            "validated_records": len(seeds),
            "scientific_input_snapshots_verified": True,
        }
    fixtures = list(
        cuboid_fixtures() if args.family == "cuboid" else cylinder_fixtures()
    )
    counts = {
        "cases": 0,
        "seeded_cases": 0,
        "resolved_observables": 0,
        "unresolved_observables": 0,
    }
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            fixture_key(dataclasses.asdict(fixture)): pool.submit(calculate, fixture)
            for fixture in fixtures
            if fixture_key(dataclasses.asdict(fixture)) not in seeds
        }
        with (output / "references.jsonl").open("w") as stream:
            for fixture in fixtures:
                key = fixture_key(dataclasses.asdict(fixture))
                if key in seeds:
                    row = seeds[key]
                    counts["seeded_cases"] += 1
                else:
                    row = futures[key].result()
                for record in row["reference"].values():
                    uncertainty = OracleUncertainty(**record["uncertainty"])
                    counts[
                        "resolved_observables"
                        if uncertainty.resolved()
                        else "unresolved_observables"
                    ] += 1
                counts["cases"] += 1
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                print(
                    json.dumps({**counts, "elapsed_s": time.monotonic() - started}),
                    flush=True,
                )
    manifest = {
        "protocol": "surface-science-protocol.md",
        "family": args.family,
        "production_values_seen": False,
        "gravitational_constant_si": 6.67430e-11,
        "python": platform.python_version(),
        "mpmath": mpmath.__version__,
        "numpy": numpy.__version__,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "workers": args.workers,
        "inputs": input_hashes,
        "seed_evidence": seed_evidence,
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
