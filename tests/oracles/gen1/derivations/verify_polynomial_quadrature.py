"""Narrow real-arithmetic Gauss certificate for pinned frozen oracle routes.

Point-force: fixed cubic pulse at knots0/1/2, tau*F(t-tau), positive source
distance and speed, original seven-point Cartesian stencil. Uniform Temperature
Sphere: constant density, moments s**2 and s only. No arbitrary ForceHistory,
SCA, Grid, acoustic Sphere, total numerical bound or changed uncertainty label.
No production or independent computational oracle is imported.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path

import mpmath as mp
import sympy as sp

ORACLE_DIRECTORY = Path(__file__).resolve().parent.parent
REPOSITORY = ORACLE_DIRECTORY.parents[2]
SOURCES = {
    "noncavity_reference.py": (
        "ab4d526b2b4ec582832051b304ea7ca0017935e69f03b3eca03c26490425e882"
    ),
    "noncavity_fixtures.py": (
        "bbebf297fd19be82c3253114d15603909e4ecf455c08d69f9f7c0f9392bf24a7"
    ),
    "source_reference.py": (
        "108db9dbaf0a5dc681f6ce56b982c79e3d32edc31116ad0f03b72d9610054117"
    ),
    "source_fixtures.py": (
        "8d1ff588b351367eb979ec6e4fde8c55cd4028cb06a5bade83160abed48fabe7"
    ),
    "sphere_reference.py": (
        "261bb88546e665fda2377597e8be52be8d88b7554bc664d0289982eec85796c8"
    ),
}
REFERENCES = {
    "noncavity": "b61408498c5dcfa046ccdc663ac2609cd63a221b3e9cb1089bf0bceea924678b",
    "source": "3f648bcf58566135c343f216be00dba38fa35fc6e0f70a926e91957ef862be6d",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_module(name):
    spec = importlib.util.spec_from_file_location(name, ORACLE_DIRECTORY / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def legendre_moments(order):
    """Exact sum over all ideal roots, using rational quotient-ring arithmetic."""
    x = sp.symbols("x", real=True)
    polynomial = sp.Poly(sp.legendre(order, x), x, domain=sp.QQ)
    assert polynomial.count_roots(-1, 1) == order
    assert sp.gcd(polynomial, polynomial.diff()).degree() == 0
    # Ideal Gauss weights w_i=2/[(1-x_i**2)*P_n'(x_i)**2].
    denominator = sp.Poly((1 - x**2) * polynomial.diff().as_expr() ** 2, x)
    weights_mod = sp.rem(2 * sp.invert(denominator, polynomial), polynomial)
    coefficients = polynomial.monic().all_coeffs()
    powers = [sp.Integer(order)]
    for power in range(1, order + 4):
        if power <= order:
            value = sum(
                coefficients[index] * powers[power - index] for index in range(1, power)
            )
            value += power * coefficients[power]
        else:
            value = sum(
                coefficients[index] * powers[power - index]
                for index in range(1, order + 1)
            )
        powers.append(-value)
    moments = []
    for power in range(5):
        value = sum(
            coefficient * powers[index[0] + power]
            for index, coefficient in weights_mod.terms()
        )
        expected = sp.Rational(2, power + 1) if power % 2 == 0 else sp.Integer(0)
        assert value == expected
        moments.append(str(value))
    return moments


def symbolic_certificate():
    tau, elapsed, lower, upper, x = sp.symbols("tau t lower upper x", real=True)
    history_time = sp.symbols("s", real=True)
    rise = 3 * history_time**2 - 2 * history_time**3
    fall = 1 - 3 * (history_time - 1) ** 2 + 2 * (history_time - 1) ** 3
    for knot, left, right in (
        (0, sp.Integer(0), rise),
        (1, rise, fall),
        (2, fall, sp.Integer(0)),
    ):
        for derivative in (0, 1):
            assert sp.diff(left, history_time, derivative).subs(history_time, knot) == (
                sp.diff(right, history_time, derivative).subs(history_time, knot)
            )
    degrees = []
    for branch in (sp.Integer(0), rise, fall):
        integrand = sp.expand(tau * branch.subs(history_time, elapsed - tau))
        if integrand == 0:
            degrees.append("zero")
            continue
        polynomial = sp.Poly(integrand, tau)
        assert polynomial.degree() <= 4
        midpoint, half = (lower + upper) / 2, (upper - lower) / 2
        transformed = sp.Poly(sp.expand(integrand.subs(tau, midpoint + half * x)), x)
        assert transformed.degree() <= 4
        moments = [sp.Rational(2, n + 1) if n % 2 == 0 else 0 for n in range(5)]
        exact_rule = half * sum(
            coefficient * moments[index[0]]
            for index, coefficient in transformed.terms()
        )
        exact_integral = sp.integrate(integrand, (tau, lower, upper))
        assert sp.simplify(exact_rule - exact_integral) == 0
        degrees.append(polynomial.degree())
    # Every original difference point uses an axis or sum-of-two-axes direction,
    # offset<=3 and h<=r/12; all later h shrink by powers of2.
    assert sp.sqrt(2) / 4 < 1
    # Uniform Sphere's actual radial polynomial moments, including zero width.
    radius, limit = sp.symbols("R u", nonnegative=True)
    assert sp.integrate(tau**2, (tau, 0, limit)) == limit**3 / 3
    assert sp.integrate(tau, (tau, limit, radius)) == (radius**2 - limit**2) / 2
    return {
        "history_panel_degrees": degrees,
        "ideal_Gauss_moments_0_through_4": {
            str(order): legendre_moments(order) for order in (16, 24, 32)
        },
        "stencil_source_exclusion": "3*sqrt(2)*h <=sqrt(2)*r/4 <r",
    }


def certified_rows(path, dataset):
    if digest(path) != REFERENCES[dataset]:
        raise ValueError(f"unsupported frozen reference hash: {path}")
    manifest = path.with_name("manifest.json")
    if json.loads(manifest.read_text())["reference_sha256"] != digest(path):
        raise ValueError("frozen manifest/reference mismatch")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    module_name = "noncavity_fixtures" if dataset == "noncavity" else "source_fixtures"
    generator = getattr(fixture_module(module_name), module_name)

    def applicable(fixture):
        return (
            fixture["family"] == "point_force"
            if dataset == "noncavity"
            else fixture["family"] == "finite_temperature"
            and fixture["shape"] == "sphere"
        )

    allowed = {canonical(fixture) for fixture in generator() if applicable(fixture)}
    selected, seen = [], set()
    for index, row in enumerate(rows):
        fixture = row["fixture"]
        if not applicable(fixture):
            continue
        identity = canonical(fixture)
        if identity not in allowed or identity in seen:
            raise ValueError("unsupported or duplicated certificate fixture")
        seen.add(identity)
        if dataset == "noncavity":
            assert fixture["knots_s"] == [0.0, 1.0, 2.0]
            assert fixture["coefficients_N"] == [
                [0.0, 0.0, 3.0, -2.0],
                [1.0, 0.0, -3.0, 2.0],
            ]
            assert fixture["direction"] == [1.0, 0.0, 0.0]
            assert sp.Rational(fixture["vp_m_s"]) > 0
            assert sum(sp.Rational(value) ** 2 for value in fixture["target_m"]) > 0
            kind = "retarded_cubic_pulse_polynomial_panels"
        else:
            assert sp.Rational(fixture["geometry"]["radius_m"]) > 0
            kind = "uniform_Sphere_radial_polynomial_moments"
        selected.append(
            {
                "dataset": dataset,
                "source_row_index": index,
                "fixture": fixture,
                "certificate_kind": kind,
                "covered_observables": ["potential", "acceleration", "gradient"],
                "covered_original_stencils": (
                    "all original loop steps and axis/diagonal seven-point evaluations"
                ),
                "certified_error": (
                    "zero quadrature truncation in ideal real arithmetic only"
                ),
                "frozen_values_uncertainty_labels_unchanged": True,
            }
        )
    if seen != allowed:
        raise ValueError("incomplete frozen certificate fixture set")
    return selected, {str(path): digest(path), str(manifest): digest(manifest)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noncavity-references", type=Path, required=True)
    parser.add_argument("--source-references", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(REPOSITORY):
        parser.error("raw certificates must remain outside Git")
    if output.exists():
        parser.error("certificate output already exists; use a new output path")
    production_modules = sorted(
        name for name in sys.modules if name == "gragra" or name.startswith("gragra.")
    )
    if production_modules:
        raise RuntimeError("run certificate standalone without production imports")
    inputs = {
        str(ORACLE_DIRECTORY / name): digest(ORACLE_DIRECTORY / name)
        for name in SOURCES
    }
    if any(
        inputs[str(ORACLE_DIRECTORY / name)] != expected
        for name, expected in SOURCES.items()
    ):
        raise ValueError("unsupported independent source revision")
    symbolic = symbolic_certificate()
    records, references = certified_rows(
        args.noncavity_references.resolve(), "noncavity"
    )
    if args.source_references:
        extra, extra_inputs = certified_rows(args.source_references.resolve(), "source")
        records.extend(extra)
        references.update(extra_inputs)
    inputs.update(references)
    if str(output) in inputs:
        parser.error("output must not overwrite a frozen input")
    result = {
        "scope": (
            "additive ideal-polynomial quadrature certificate for pinned fixtures only"
        ),
        "script_sha256": digest(Path(__file__)),
        "inputs": inputs,
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "sympy": sp.__version__,
            "mpmath": mp.__version__,
        },
        "exact_symbolic_checks": symbolic,
        "certificates": records,
        "production_modules_observed": production_modules,
        "not_certified": [
            "finite-precision nodes/weights/root and panel arithmetic",
            "stencil differentiation and precision propagation",
            "conversion/reporting and independent crosscheck uncertainty",
            "total numerical error or arbitrary ForceHistory",
        ],
        "qualification": (
            "Temporal quadrature remains an applicable algorithm; "
            "only its ideal polynomial truncation is certified zero. "
            "Existing values, U components and estimate/bound labels stay unchanged."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "certificate_count": len(records),
                "exact_CAS": "PASS",
                "output": str(output),
                "output_sha256": digest(output),
            }
        )
    )


if __name__ == "__main__":
    main()
