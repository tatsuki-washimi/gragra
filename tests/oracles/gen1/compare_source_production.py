"""Public-API comparison only; no computational oracle implementation is imported."""

import argparse
import dataclasses
import hashlib
import json
import math
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
from astropy import units as u

import gragra
from gragra.analytic.atmospheric import isobaric_temperature
from gragra.analytic.seismic import small_cavity_wall
from gragra.constants import G_SI
from gragra.fields import TemperaturePerturbationGrid
from gragra.geometry import Cuboid, Cylinder, Frame, Sphere
from gragra.gravity_sources import point_sum
from gragra.targets import TargetPoints

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--references", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
repository = Path(gragra.__file__).resolve().parents[2]
sys.path.insert(0, str(repository / "tests/oracles/gen1"))
from acceptance import OracleUncertainty, assess, norm_si  # noqa: E402


# Resolve/pin acceptance from the selected production checkout.
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


reference_path = args.references.resolve()
reference_manifest = json.loads(reference_path.with_name("manifest.json").read_text())
assert reference_manifest["reference_sha256"] == sha(reference_path)
assert reference_manifest["gravitational_constant_si"] == G_SI
acceptance_path = repository / "tests/oracles/gen1/acceptance.py"
assert (
    sha(acceptance_path)
    == reference_manifest["inputs"]["tests/oracles/gen1/acceptance.py"]
)
output = args.output.resolve()
if output.is_relative_to(repository):
    parser.error("Raw output must remain outside Git")
output.mkdir(parents=True, exist_ok=False)
source_inputs = {
    str(p.relative_to(repository)): sha(p)
    for p in sorted((repository / "src").rglob("*.py"))
}
source_inputs["pyproject.toml"] = sha(repository / "pyproject.toml")
a, b, c = 0.37, -0.53, 0.81
rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
rot = rz @ ry @ rx
world = Frame("source-reference-world")
frames = {
    "baseline": world,
    "rotated": Frame("source-reference-rotated", basis=rot, parent=world),
}
frames["rotated_world_targets"] = frames["rotated"]
shapes = {"potential": (), "acceleration": (3,), "gradient": (3, 3)}
units = {
    "potential": u.m**2 / u.s**2,
    "acceleration": u.m / u.s**2,
    "gradient": u.s**-2,
}
epoch = (1_400_000_000, 17)


def models_for(f, frame, variant):
    family = f["family"]
    rho0 = f["rho0_kg_m3"] * u.kg / u.m**3
    background_temperature = f["T0_K"] * u.K
    if family == "temperature_grid":
        grid = TemperaturePerturbationGrid(
            x_edges=np.array(f["x_edges_m"]) * u.m,
            y_edges=np.array(f["y_edges_m"]) * u.m,
            z_edges=np.array(f["z_edges_m"]) * u.m,
            delta_temperature=np.array(f["delta_temperature_K"]) * u.K,
            frame=frame,
        )
        source = grid.as_volume_element_source(rho0=rho0, T0=background_temperature)
        return [
            point_sum(
                source,
                frame=grid.frame,
                field_definition=grid.field_definition,
                provenance=grid.point_sum_provenance(
                    rho0=rho0, T0=background_temperature
                ),
            )
        ]
    g = f["geometry"]
    if f["shape"] == "sphere":
        geometry = Sphere(
            center=np.zeros(3) * u.m, radius=g["radius_m"] * u.m, frame=frame
        )
    elif f["shape"] == "cuboid":
        geometry = Cuboid(
            center=np.zeros(3) * u.m,
            lengths=np.array(g["lengths_m"]) * u.m,
            frame=frame,
        )
    elif f["shape"] == "cylinder":
        geometry = Cylinder(
            center=np.zeros(3) * u.m,
            radius=g["radius_m"] * u.m,
            height=g["height_m"] * u.m,
            frame=frame,
        )
    else:
        raise AssertionError(f["shape"])
    extra = (
        {"frequency": 17 * u.Hz, "reference_time": epoch} if variant == "phasor" else {}
    )
    if family == "finite_temperature":
        return [
            isobaric_temperature(
                geometry,
                f["delta_temperature_K"] * u.K,
                rho0=rho0,
                T0=background_temperature,
                **extra,
            )
        ]
    if family == "sca_wall":
        return [
            small_cavity_wall(
                geometry,
                np.array(f["displacement_m"]) * u.m,
                density=f["density_kg_m3"] * u.kg / u.m**3,
                **extra,
            )
        ]
    raise AssertionError(family)


def extract(result, name, variant):
    payload = getattr(result, name)
    if variant == "static":
        assert payload.unit == units[name]
        assert payload.shape == (1,) + shapes[name]
        return np.asarray(payload.to_value(units[name]))[0].reshape(shapes[name])
    assert payload.shape == (
        1,
        {"potential": 1, "acceleration": 3, "gradient": 9}[name],
        1,
    )
    assert all(payload.meta[0, i].unit == units[name] for i in range(payload.shape[1]))
    return np.asarray(payload)[0, :, 0].reshape(shapes[name])


counts = {
    "fixtures": 0,
    "evaluations": 0,
    "assessments": 0,
    "passed": 0,
    "failed": 0,
    "execution_failures": 0,
}
by_family = {}
started = time.monotonic()
with (output / "comparisons.jsonl").open("w") as stream:
    for line in reference_path.read_text().splitlines():
        case = json.loads(line)
        f = case["fixture"]
        family = f["family"]
        counts["fixtures"] += 1
        stats = by_family.setdefault(
            family + "/" + f.get("shape", "grid"),
            {
                "passed": 0,
                "failed": 0,
                "max_ratio": 0.0,
                "max_oracle_ratio": 0.0,
                "execution_failures": 0,
            },
        )
        variants = ["static"] if family == "temperature_grid" else ["static", "phasor"]
        for orientation, frame in frames.items():
            target_positions = np.array([f["target_m"]])
            target_frame = frame
            if orientation == "rotated_world_targets":
                target_positions = target_positions @ rot.T
                target_frame = world
            targets = TargetPoints(
                target_positions * u.m, ("reference-target",), target_frame
            )
            for variant in variants:
                selected_case = case
                selected_reference_path = reference_path
                try:
                    results = [
                        model.at(targets, output_frame=world).run()
                        for model in models_for(f, frame, variant)
                    ]
                except Exception as error:
                    counts["execution_failures"] += 1
                    stats["execution_failures"] += 1
                    stream.write(
                        json.dumps(
                            {
                                "fixture": f,
                                "orientation": orientation,
                                "variant": variant,
                                "execution_error": {
                                    "type": type(error).__name__,
                                    "message": str(error),
                                },
                            },
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    continue
                counts["evaluations"] += len(results)
                for result in results:
                    assert result.record.g_applications == 1
                    assert result.field_definition is result.context.field_definition
                    if variant == "phasor":
                        assert result.context.reference_time_gps == epoch
                    definition = result.field_definition
                    assert definition.contribution.coverage == (
                        "selected" if family == "sca_wall" else "total"
                    )
                    assert definition.kind.value == "field_change"
                    assert (
                        definition.change.form.value == "linearized_response"
                        and definition.change.order == 1
                    )
                    assert definition.change.reference.status.value == "defined"
                    assert definition.potential_reference.value == "zero_at_infinity"
                    assert definition.change.reference.temporal_kind.value == "state"
                    if family == "sca_wall":
                        assert definition.source_kind == "cavity_wall_perturbation"
                        assert definition.change.changed_input == "displacement"
                        assert definition.contribution.components == ("wall",)
                    else:
                        assert (
                            definition.source_kind
                            == "isobaric_temperature_perturbation"
                        )
                        assert (
                            definition.change.changed_input
                            == "temperature_perturbation"
                        )
                for name, shape in shapes.items():
                    record = selected_case["reference"][name]
                    reference = np.array(
                        [complex(*v) for v in record["values_si"]]
                    ).reshape(shape)
                    if variant == "static":
                        reference = reference.real
                    original_norm = norm_si(name, reference)
                    if orientation != "baseline":
                        if name == "acceleration":
                            reference = rot @ reference
                        elif name == "gradient":
                            reference = rot @ reference @ rot.T
                    uncertainty_record = dict(record["uncertainty"])
                    components = tuple(
                        tuple(item) for item in uncertainty_record["components"]
                    )
                    additional = (
                        64 * np.finfo(float).eps * original_norm
                        if orientation != "baseline" and name != "potential"
                        else 0.0
                    )
                    components += (
                        ("rotated_reference_arithmetic_estimate", float(additional)),
                    )
                    if additional:
                        assert uncertainty_record["uncertainty_kind"] == "estimate"
                        total = math.nextafter(
                            max(
                                math.fsum(v for _, v in components),
                                math.fsum(
                                    (
                                        uncertainty_record["total_error_estimate_si"],
                                        additional,
                                    )
                                ),
                            ),
                            math.inf,
                        )
                        uncertainty_record["total_error_estimate_si"] = total
                        uncertainty_record["used_uncertainty_si"] = total
                    uncertainty_record["components"] = components
                    uncertainty_record["evidence_refs"] = tuple(
                        uncertainty_record["evidence_refs"]
                    ) + (str(selected_reference_path),)
                    uncertainty = OracleUncertainty(**uncertainty_record)
                    actual = sum(
                        (extract(r, name, variant) for r in results),
                        start=np.zeros(shape),
                    )
                    assert (
                        selected_case["metadata"]["scales_si"][name]
                        == case["metadata"]["scales_si"][name]
                    )
                    assessment = assess(
                        name,
                        actual,
                        reference,
                        case["metadata"]["scales_si"][name],
                        uncertainty,
                    )
                    budget = assessment["budget_ref_si"]
                    ratio = (
                        assessment["difference_norm_si"]
                        + assessment["used_uncertainty_si"]
                    ) / budget
                    stats["max_ratio"] = max(stats["max_ratio"], ratio)
                    stats["max_oracle_ratio"] = max(
                        stats["max_oracle_ratio"],
                        assessment["used_uncertainty_si"] / budget,
                    )
                    assert math.isclose(
                        norm_si(name, reference),
                        original_norm,
                        rel_tol=3e-15,
                        abs_tol=0.0,
                    )
                    outcome = "passed" if assessment["accepted"] else "failed"
                    counts["assessments"] += 1
                    counts[outcome] += 1
                    stats[outcome] += 1
                    stream.write(
                        json.dumps(
                            {
                                "fixture": f,
                                "orientation": orientation,
                                "variant": variant,
                                "assessment": assessment,
                                "uncertainty": dataclasses.asdict(uncertainty),
                                "production_si": [
                                    [complex(v).real, complex(v).imag]
                                    for v in actual.flat
                                ],
                            },
                            allow_nan=False,
                        )
                        + "\n"
                    )
manifest = {
    "claim": (
        "finite_validation_set_with_numerical_estimates_and_exact_symmetry_zero_bounds"
    ),
    "counts": counts,
    "by_family": by_family,
    "source_inputs": source_inputs,
    "reference_path": str(reference_path),
    "reference_sha256": sha(reference_path),
    "reference_manifest_sha256": sha(reference_path.with_name("manifest.json")),
    "driver_sha256": sha(Path(__file__)),
    "acceptance_sha256": sha(acceptance_path),
    "comparisons_sha256": sha(output / "comparisons.jsonl"),
    "gragra_import_path": gragra.__file__,
    "python": platform.python_version(),
    "versions": {
        name: version(name) for name in ["numpy", "scipy", "astropy", "gwexpy"]
    },
    "rotation_matrix": rot.tolist(),
    "rotation_rounding_estimate": (
        "64*eps64*reference_norm for transformed vector/tensor"
    ),
    "grid_scope": (
        "exact discrete center-mass source; "
        "continuum discretization accuracy not claimed"
    ),
    "elapsed_s": time.monotonic() - started,
}
(output / "manifest.json").write_text(
    json.dumps(manifest, indent=2, allow_nan=False) + "\n"
)
print(json.dumps({k: v for k, v in manifest.items() if k != "source_inputs"}, indent=2))
if counts["failed"] or counts["execution_failures"]:
    raise SystemExit("Finite-source independent acceptance failed")
