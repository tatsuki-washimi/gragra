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
from gragra.analytic.atmospheric import acoustic_halfspace, isobaric_temperature
from gragra.analytic.objects import (
    constant_velocity,
    fixed_axis_rotation,
    translational_oscillation,
)
from gragra.analytic.seismic import point_force
from gragra.constants import G_SI
from gragra.fields import TemperatureFourierMode, TemperatureGaussian
from gragra.geometry import Frame, HalfSpace
from gragra.gravity_sources import ForceHistory, PointMass, PointMassCloud
from gragra.results import TimeSampling
from gragra.targets import TargetPoints

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--references", type=Path, required=True)
parser.add_argument("--static-plane-references", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
repository = Path(gragra.__file__).resolve().parents[2]
sys.path.insert(0, str(repository / "tests/oracles/gen1"))
from acceptance import OracleUncertainty, assess, norm_si  # noqa: E402

# Pin acceptance from the selected production checkout after resolving its path.


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
static_path = args.static_plane_references.resolve()
static_manifest = json.loads(static_path.with_name("manifest.json").read_text())
assert static_manifest["reference_sha256"] == sha(static_path)
assert static_manifest["gravitational_constant_si"] == G_SI
assert (
    sha(acceptance_path)
    == static_manifest["inputs"]["tests/oracles/gen1/acceptance.py"]
)
static_cases = {
    json.dumps(r["fixture"], sort_keys=True): r
    for r in map(json.loads, static_path.read_text().splitlines())
}
assert len(static_cases) == 3
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
world = Frame("noncavity-reference-world")
frames = {
    "baseline": world,
    "rotated": Frame("noncavity-reference-rotated", basis=rot, parent=world),
}
frames["rotated_world_targets"] = frames["rotated"]
shapes = {"potential": (), "acceleration": (3,), "gradient": (3, 3)}
units = {
    "potential": u.m**2 / u.s**2,
    "acceleration": u.m / u.s**2,
    "gradient": u.s**-2,
}
epoch = (1_400_000_000, 17)


def sampling_at(elapsed):
    nanos = round(elapsed * 1_000_000_000) + epoch[1]
    seconds, nanos = divmod(nanos, 1_000_000_000)
    return TimeSampling(epoch=(epoch[0] + seconds, nanos), dt=0.1 * u.s, count=1)


def models_for(f, frame, variant):
    family = f["family"]
    if family in ("gaussian", "temperature_plane"):
        if family == "gaussian":
            profile = TemperatureGaussian(
                center=np.zeros(3) * u.m, sigma=f["sigma_m"] * u.m, frame=frame
            )
        else:
            profile = TemperatureFourierMode(
                wavevector=np.array(f["wavevector_m_inv"]) / u.m,
                phase_origin=np.array(f["phase_origin_m"]) * u.m,
                frame=frame,
            )
        extra = (
            {"frequency": 17 * u.Hz, "reference_time": epoch}
            if variant == "phasor"
            else {}
        )
        return [
            isobaric_temperature(
                profile,
                f["delta_temperature_K"] * u.K,
                rho0=f["rho0_kg_m3"] * u.kg / u.m**3,
                T0=f["T0_K"] * u.K,
                **extra,
            )
        ]
    if family == "halfspace":
        return [
            acoustic_halfspace(
                HalfSpace(frame),
                complex(*f["pressure_pa"]) * u.Pa,
                sound_speed=f["sound_speed_m_s"] * u.m / u.s,
                frequency=f["frequency_hz"] * u.Hz,
                incidence_angle=f["incidence_rad"] * u.rad,
                azimuth=f["azimuth_rad"] * u.rad,
                phase_origin=np.array(f["phase_origin_m"]) * u.m,
                reference_time=epoch,
            )
        ]
    if family == "point_force":
        history = ForceHistory(
            knots=np.array(f["knots_s"]) * u.s,
            coefficients=np.array(f["coefficients_N"]) * u.N,
            onset=epoch,
            after="zero",
        )
        return [
            point_force(
                np.zeros(3) * u.m,
                f["direction"],
                history,
                frame=frame,
                vp=f["vp_m_s"] * u.m / u.s,
                sampling=sampling_at(f["elapsed_s"]),
            )
        ]
    cloud = PointMassCloud(f["masses_kg"], f["positions_m"])
    if family == "constant_speed":
        return [
            constant_velocity(
                PointMass(m, p),
                np.array(f["velocity_m_s"]) * u.m / u.s,
                frame=frame,
                motion_epoch=epoch,
                sampling=sampling_at(f["elapsed_s"]),
            )
            for m, p in zip(f["masses_kg"], f["positions_m"], strict=True)
        ]
    if family == "rotation":
        return [
            fixed_axis_rotation(
                cloud,
                f["rotation_axis"],
                f["angular_speed_rad_s"] * u.rad / u.s,
                frame=frame,
                origin=np.zeros(3) * u.m,
                angle0=0 * u.rad,
                angle_epoch=epoch,
                sampling=sampling_at(f["elapsed_s"]),
            )
        ]
    if family == "oscillation":
        displacement = np.array([complex(*p) for p in f["displacement_m"]]) * u.m
        return [
            translational_oscillation(
                cloud,
                displacement,
                frame=frame,
                frequency=3.7 * u.Hz,
                reference_time=epoch,
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


counts = {"fixtures": 0, "evaluations": 0, "assessments": 0, "passed": 0, "failed": 0}
by_family = {}
started = time.monotonic()
with (output / "comparisons.jsonl").open("w") as stream:
    for line in reference_path.read_text().splitlines():
        case = json.loads(line)
        f = case["fixture"]
        family = f["family"]
        counts["fixtures"] += 1
        stats = by_family.setdefault(
            family,
            {"passed": 0, "failed": 0, "max_ratio": 0.0, "max_oracle_ratio": 0.0},
        )
        variants = (
            ["phasor", "static"]
            if family == "temperature_plane"
            else ["static"]
            if family == "gaussian"
            else ["timeseries"]
            if family in ("point_force", "constant_speed", "rotation")
            else ["phasor"]
        )
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
                selected_case = (
                    static_cases[json.dumps(f, sort_keys=True)]
                    if family == "temperature_plane" and variant == "static"
                    else case
                )
                selected_reference_path = (
                    static_path
                    if family == "temperature_plane" and variant == "static"
                    else reference_path
                )
                results = [
                    model.at(targets, output_frame=world).run()
                    for model in models_for(f, frame, variant)
                ]
                counts["evaluations"] += len(results)
                for result in results:
                    assert result.record.g_applications == 1
                    assert result.field_definition is result.context.field_definition
                    if variant == "phasor":
                        assert result.context.reference_time_gps == epoch
                    if variant == "timeseries":
                        assert result.context.time_sampling == sampling_at(
                            f["elapsed_s"]
                        )
                        assert result.context.reference_time_gps is None
                    definition = result.field_definition
                    assert definition.contribution.coverage == "total"
                    if family in ("constant_speed", "rotation"):
                        assert (
                            definition.kind.value == "source_field"
                            and definition.change is None
                        )
                        assert definition.source_kind == "moving_point_masses"
                    else:
                        assert definition.kind.value == "field_change"
                        assert (
                            definition.change.form.value == "linearized_response"
                            and definition.change.order == 1
                        )
                        assert definition.change.reference.status.value == "defined"
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
    "claim": "finite_validation_set_based_on_recorded_oracle_uncertainty_estimates",
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
    "constant_speed_composition": "sum of two public PointMass evaluations",
    "elapsed_s": time.monotonic() - started,
}
manifest["static_plane_reference"] = {
    "path": str(static_path),
    "sha256": sha(static_path),
    "manifest_sha256": sha(static_path.with_name("manifest.json")),
}
(output / "manifest.json").write_text(
    json.dumps(manifest, indent=2, allow_nan=False) + "\n"
)
print(json.dumps({k: v for k, v in manifest.items() if k != "source_inputs"}, indent=2))
if counts["failed"]:
    raise SystemExit("Noncavity independent acceptance failed")
