"""Independent real-cosine target derivatives with the Static acceptance budget."""

import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path

import mpmath as mp

from .acceptance import OracleUncertainty
from .noncavity_fixtures import G_SI, noncavity_fixtures
from .noncavity_reference import observable_values
from .sphere_reference import differences, vector_norm


class StaticPlane:
    def __init__(self, fixture):
        self.fixture = fixture
        self.k = tuple(mp.mpf(value) for value in fixture["wavevector_m_inv"])
        self.squared = mp.fsum(value * value for value in self.k)
        self.length = 1 / mp.sqrt(self.squared)
        self.density = (
            -mp.mpf(fixture["rho0_kg_m3"])
            * mp.mpf(fixture["delta_temperature_K"])
            / mp.mpf(fixture["T0_K"])
        )
        amplitude = mp.mpf(G_SI) * abs(self.density)
        self.scales = {
            "potential": amplitude * self.length**2,
            "acceleration": amplitude * self.length,
            "gradient": amplitude,
        }

    def potential(self, point):
        phase = mp.fsum(
            k * (x - mp.mpf(origin))
            for k, x, origin in zip(
                self.k, point, self.fixture["phase_origin_m"], strict=True
            )
        )
        return -4 * mp.pi * mp.mpf(G_SI) * self.density * mp.cos(phase) / self.squared


def static_plane_reference(fixture):
    with mp.workdps(80):
        high = StaticPlane(fixture)
        point = tuple(mp.mpf(value) for value in fixture["target_m"])
        step = high.length / 50
        previous = prior = None
        resolved = False
        for refinement in range(13):
            current = observable_values(high, point, step)
            change = differences(current, previous) if previous else None
            if change and prior:
                derivative = {
                    name: 10 * max(change[name], prior[name]) for name in current
                }
                budgets = {
                    name: mp.mpf("1e-12") * high.scales[name]
                    + mp.mpf("1e-9" if name == "gradient" else "1e-10")
                    * vector_norm(values)
                    for name, values in current.items()
                }
                if all(derivative[name] <= budgets[name] / 40 for name in current):
                    resolved = True
                    break
            previous, prior = current, change
            if refinement < 12:
                step /= 2
        with mp.workdps(50):
            low_values = observable_values(StaticPlane(fixture), point, step)
        precision = differences(current, low_values)
        output = {}
        for name, values in current.items():
            converted = tuple(complex(value) for value in values)
            rounding = vector_norm(
                tuple(
                    value - mp.mpc(actual)
                    for value, actual in zip(values, converted, strict=True)
                )
            )
            components = (
                ("real_field_derivative_refinement", float(derivative[name])),
                ("50_80_digit_precision", float(10 * precision[name])),
                ("float_conversion", float(rounding)),
            )
            total = math.fsum(value for _, value in components)
            qualified = resolved and total <= float(budgets[name] / 20)
            output[name] = {
                "values_si": [[value.real, value.imag] for value in converted],
                "uncertainty": dataclasses.asdict(
                    OracleUncertainty(
                        total if qualified else None,
                        None,
                        total if qualified else None,
                        "estimate" if qualified else None,
                        "real cosine / controlled derivatives / Static budget",
                        components,
                    )
                ),
            }
        return output, {
            "digits": [50, 80],
            "derivative_refinement": refinement,
            "step_m": str(step),
            "scales_si": {name: float(value) for name, value in high.scales.items()},
            "budget_uses_real_reference_norm": True,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    if output.is_relative_to(repository):
        parser.error("raw evidence must remain outside Git")
    output.mkdir(parents=True, exist_ok=False)
    paths = [
        Path(__file__),
        Path(__file__).with_name("noncavity_reference.py"),
        Path(__file__).with_name("noncavity_fixtures.py"),
        Path(__file__).with_name("sphere_reference.py"),
        Path(__file__).with_name("acceptance.py"),
    ]
    inputs = {
        str(path.relative_to(repository)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }
    count = unresolved = 0
    with (output / "references.jsonl").open("w") as stream:
        for fixture in noncavity_fixtures():
            if fixture["family"] != "temperature_plane":
                continue
            reference, metadata = static_plane_reference(fixture)
            count += 1
            unresolved += sum(
                record["uncertainty"]["used_uncertainty_si"] is None
                for record in reference.values()
            )
            stream.write(
                json.dumps(
                    {"fixture": fixture, "reference": reference, "metadata": metadata},
                    allow_nan=False,
                )
                + "\n"
            )
    data_hash = hashlib.sha256((output / "references.jsonl").read_bytes()).hexdigest()
    manifest = {
        "inputs": inputs,
        "reference_sha256": data_hash,
        "gravitational_constant_si": G_SI,
        "representation": "Static",
        "production_values_used_in_refinement": False,
        "counts": {"cases": count, "unresolved_observables": unresolved},
        "claim": "real_field_references_with_recorded_uncertainty_estimates",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    if unresolved:
        raise SystemExit("unresolved Static reference")


if __name__ == "__main__":
    main()
