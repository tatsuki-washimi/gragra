"""Additive bounded certificates; never alter frozen reference values or labels.

Only SymPy and frozen JSON data are inputs. No production or numerical oracle
imports. This certifies physical whole-observable zeros, not numerical estimates
in general and not the correctness of nonzero reference values.
"""

import argparse
import collections
import hashlib
import json
from pathlib import Path

import sympy as sp

x, y, z = sp.symbols("x y z", real=True)
coordinates = (x, y, z)
squared = x * x + y * y + z * z
radial = sp.Function("f")
# Normalization factors are finite nonzero constants and cannot change zeros.
# These polynomials are independently constructed from exact Legendre powers.
w = sp.symbols("w", real=True)


def harmonic(degree, signed_order):
    order = abs(signed_order)
    legendre = sp.Poly(sp.diff(sp.legendre(degree, w), w, order), w)
    remaining = sum(
        coefficient * z ** power[0] * squared ** ((degree - order - power[0]) // 2)
        for power, coefficient in legendre.terms()
    )
    azimuth = (x + sp.I * y) ** order
    azimuth = (
        sp.im(sp.expand(azimuth)) if signed_order < 0 else sp.re(sp.expand(azimuth))
    )
    return sp.expand(remaining * azimuth)


def zero_sphere(fixture, observable):
    degree, order = fixture["degree"], fixture["order"]
    h = harmonic(degree, order)
    assert sp.expand(sum(sp.diff(h, c, 2) for c in coordinates)) == 0
    assert sp.Poly(h, x, y, z).total_degree() == degree
    potential = radial(squared) * h
    values = {
        "potential": [potential],
        "acceleration": [-sp.diff(potential, c) for c in coordinates],
        "gradient": [
            -sp.diff(potential, c, d) for c in coordinates for d in coordinates
        ],
    }[observable]
    point = fixture["target_m"]
    if point == [0.0, 0.0, 0.0]:
        substitution = {x: 0, y: 0, z: 0}
        proof = (
            "regular radial factor times exact degree-l harmonic; center derivatives"
        )
    else:
        assert point[0] == point[1] == 0 and point[2] != 0
        substitution = {x: 0, y: 0}
        proof = (
            "exact polar-axis harmonic and Cartesian derivatives, "
            "arbitrary radial factor"
        )
    assert all(sp.simplify(value.subs(substitution)) == 0 for value in values)
    return proof


def zero_noncavity(fixture, observable):
    if fixture["family"] == "gaussian":
        assert fixture["target_m"] == [0.0, 0.0, 0.0] and observable == "acceleration"
        assert all(
            sp.diff(radial(squared), c).subs({x: 0, y: 0, z: 0}) == 0
            for c in coordinates
        )
        return "smooth even radial Newton potential; gradient at center vanishes"
    assert fixture["family"] == "point_force"
    assert fixture["direction"] == [1.0, 0.0, 0.0]
    point = tuple(sp.Rational(value) for value in fixture["target_m"])
    elapsed = sp.Rational(fixture["elapsed_s"])
    radius_squared = sum(c * c for c in point)
    speed = sp.Rational(fixture["vp_m_s"])
    assert radius_squared > 0 and speed > 0
    if elapsed == 0:
        return (
            "causal history F(s)=0 for s<=0; J and target derivatives vanish at onset"
        )
    terminal = sp.Rational(fixture["knots_s"][-1])
    if elapsed > terminal and radius_squared < speed**2 * (elapsed - terminal) ** 2:
        return (
            "retarded integration interval lies strictly after compact pulse; "
            "zero on open target neighborhood"
        )
    assert point[0] == 0 and observable == "potential"
    moment = sp.Function("J")
    potential = -x * moment(sp.sqrt(squared)) / squared ** sp.Rational(3, 2)
    assert potential.subs(x, 0) == 0
    return (
        "force-axis contraction mu=0; scalar dipole potential vanishes "
        "on perpendicular plane"
    )


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sphere-references", type=Path, required=True)
    parser.add_argument("--noncavity-references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    datasets = (
        (
            "sphere-three-quadrature-20261001",
            args.sphere_references.resolve(),
            zero_sphere,
        ),
        (
            "noncavity-primary-20261001",
            args.noncavity_references.resolve(),
            zero_noncavity,
        ),
    )
    target = args.output.resolve()
    if any(
        target in (data, data.with_name("manifest.json")) for _, data, _ in datasets
    ):
        raise ValueError("certificate output must not overwrite a frozen input")
    records, counts, inputs = [], {}, {}
    for dataset, data, proof in datasets:
        manifest = data.with_name("manifest.json")
        inputs[str(data)] = sha(data)
        inputs[str(manifest)] = sha(manifest)
        if json.loads(manifest.read_text())["reference_sha256"] != sha(data):
            raise ValueError(f"frozen reference hash mismatch: {data}")
        group_counts = collections.Counter()
        for row in map(json.loads, data.read_text().splitlines()):
            for observable, reference in row["reference"].items():
                uncertainty = reference["uncertainty"]
                if (
                    uncertainty["used_uncertainty_si"] != 0
                    or uncertainty["uncertainty_kind"] != "estimate"
                ):
                    continue
                assert all(value == [0.0, 0.0] for value in reference["values_si"])
                assert row["metadata"]["scales_si"][observable] > 0
                certificate = proof(row["fixture"], observable)
                group_counts[certificate] += 1
                records.append(
                    {
                        "dataset": dataset,
                        "fixture": row["fixture"],
                        "observable": observable,
                        "certificate": certificate,
                        "original_uncertainty_kind": "estimate",
                        "original_used_uncertainty_si": 0.0,
                        "frozen_labels_and_values_unchanged": True,
                    }
                )
        counts[dataset] = dict(group_counts)
    assert len(records) == 173
    result = {
        "scope": "additive exact whole-observable zero certificates only",
        "classification": (
            "exact symbolic identities under declared radial/causal domains"
        ),
        "script_sha256": sha(Path(__file__)),
        "sympy": sp.__version__,
        "frozen_inputs": inputs,
        "counts": counts,
        "certified_records": len(records),
        "uncertified_zero_records": 0,
        "certificates": records,
        "qualification": (
            "Does not relabel frozen estimates as rigorous "
            "or certify other numerical values."
        ),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "certified_records": len(records),
                "uncertified_zero_records": 0,
                "counts": counts,
                "output": str(target),
                "output_sha256": sha(target),
            }
        )
    )


if __name__ == "__main__":
    main()
