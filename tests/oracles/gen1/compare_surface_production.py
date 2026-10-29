"""Coordinator comparison only: frozen independent references versus public API."""

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
from gragra.analytic.atmospheric import (
    CuboidAcousticMode,
    CylinderAcousticMode,
    acoustic_cavity_mode,
)
from gragra.constants import G_SI
from gragra.geometry import Cuboid, Cylinder, Frame
from gragra.targets import TargetPoints

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--family", choices=("cuboid", "cylinder"), required=True)
parser.add_argument("--references", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
repository = Path(gragra.__file__).resolve().parents[2]
sys.path.insert(0, str(repository / "tests/oracles/gen1"))
from acceptance import OracleUncertainty, assess, norm_si  # noqa: E402

# Pin acceptance from the selected production checkout after resolving its path.

reference_path = args.references.resolve()
output = args.output.resolve()
if output.is_relative_to(repository):
    parser.error("Raw comparisons must remain outside Git")
output.mkdir(parents=True, exist_ok=False)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


reference_manifest = json.loads(reference_path.with_name("manifest.json").read_text())
assert reference_manifest["reference_sha256"] == sha(reference_path)
assert reference_manifest["gravitational_constant_si"] == G_SI
acceptance_path = repository / "tests/oracles/gen1/acceptance.py"
assert (
    sha(acceptance_path)
    == reference_manifest["inputs"]["tests/oracles/gen1/acceptance.py"]
)
source_inputs = {
    str(p.relative_to(repository)): sha(p)
    for p in sorted((repository / "src").rglob("*.py"))
}
source_inputs["pyproject.toml"] = sha(repository / "pyproject.toml")
angles = (0.37, -0.53, 0.81)
a, b, c = angles
rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
rot = rz @ ry @ rx
world = Frame("surface-reference-world")
frames = {
    "baseline": world,
    "rotated": Frame("surface-reference-rotated", basis=rot, parent=world),
}
frames["rotated_world_targets"] = frames["rotated"]
models = {}
shapes = {"potential": (), "acceleration": (3,), "gradient": (3, 3)}
si_units = {
    "potential": u.m**2 / u.s**2,
    "acceleration": u.m / u.s**2,
    "gradient": u.s**-2,
}
counts = {
    "fixtures": 0,
    "evaluations": 0,
    "observables": 0,
    "passed": 0,
    "failed": 0,
    "execution_failures": 0,
}
max_ratio = dict.fromkeys(shapes, 0.0)
max_oracle_ratio = dict.fromkeys(shapes, 0.0)
started = time.monotonic()
with (output / "comparisons.jsonl").open("w") as stream:
    for line in reference_path.read_text().splitlines():
        case = json.loads(line)
        f = case["fixture"]
        counts["fixtures"] += 1
        for orientation, frame in frames.items():
            if args.family == "cuboid":
                geometry = Cuboid(
                    center=np.zeros(3) * u.m,
                    lengths=np.array(f["lengths_m"]) * u.m,
                    frame=frame,
                )
                mode = CuboidAcousticMode(*f["mode"])
                wavenumber = np.linalg.norm(
                    np.pi * np.array(f["mode"]) / np.array(f["lengths_m"])
                )
            else:
                geometry = Cylinder(
                    center=np.zeros(3) * u.m,
                    radius=f["radius_m"] * u.m,
                    height=f["height_m"] * u.m,
                    frame=frame,
                )
                mode = CylinderAcousticMode(
                    abs(f["order"]),
                    f["radial_index"],
                    f["axial_index"],
                    azimuth="sin" if f["order"] < 0 else "cos",
                )
                wavenumber = math.hypot(
                    float(case["metadata"]["root"]) / f["radius_m"],
                    np.pi * f["axial_index"] / f["height_m"],
                )
            key = (
                orientation,
                json.dumps(
                    {
                        k: v
                        for k, v in f.items()
                        if k not in ("target_m", "target_kind")
                    },
                    sort_keys=True,
                ),
            )
            if key not in models:
                models[key] = acoustic_cavity_mode(
                    geometry,
                    mode,
                    complex(f["pressure_real_pa"], f["pressure_imag_pa"]) * u.Pa,
                    sound_speed=f["sound_speed_m_s"] * u.m / u.s,
                    reference_time=(1_400_000_000, 17),
                )
            model = models[key]
            target_positions = np.array([f["target_m"]])
            target_frame = frame
            if orientation == "rotated_world_targets":
                target_positions = target_positions @ rot.T
                target_frame = world
            target = TargetPoints(
                target_positions * u.m, ("reference-target",), target_frame
            )
            try:
                result = model.at(target, output_frame=world).run()
            except Exception as error:
                counts["execution_failures"] += 1
                cause = getattr(error, "cause", None)
                diagnostics = getattr(cause, "diagnostics", ())
                stream.write(
                    json.dumps(
                        {
                            "fixture": f,
                            "orientation": orientation,
                            "execution_error": {
                                "type": type(error).__name__,
                                "message": str(error),
                                "cause_type": type(cause).__name__,
                                "cause_message": str(cause),
                                "diagnostics": [
                                    dataclasses.asdict(d) for d in diagnostics
                                ],
                            },
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
                stream.flush()
                continue
            assert result.record.g_applications == 1
            assert result.context.reference_time_gps == (1_400_000_000, 17)
            diagnostics = result.record.numerical_diagnostics
            assert len(diagnostics) == 3
            by_observable = {d.observable: d for d in diagnostics}
            assert set(by_observable) == set(shapes)
            for name, d in by_observable.items():
                assert d.target_index == 0 and d.target_id == "reference-target"
                assert (
                    d.termination == "converged"
                    and d.accuracy_policy == "gen1_cavity_v1"
                )
                assert (
                    d.norm
                    == {
                        "potential": "absolute",
                        "acceleration": "euclidean",
                        "gradient": "frobenius",
                    }[name]
                )
                assert d.scale_si > 0 and math.isclose(
                    d.scale_si, case["metadata"]["scales_si"][name], rel_tol=3e-14
                )
                assert (
                    d.error_estimate_si is not None
                    and d.rigorous_tail_bound_si is not None
                )
                assert (
                    d.error_estimate_si + d.rigorous_tail_bound_si
                    <= d.requested_tolerance_si
                )
                r = 1e-9 if name == "gradient" else 1e-10
                assert math.isclose(
                    d.requested_tolerance_si,
                    0.25 * (1e-12 * d.scale_si + r * d.value_norm_si),
                    rel_tol=5e-14,
                )
                assert d.evaluations > 0 and d.refinements >= 0
            expected_frequency = wavenumber * f["sound_speed_m_s"] / (2 * np.pi)
            assert math.isclose(
                result.context.frequencies_hz[0], expected_frequency, rel_tol=2e-13
            )
            counts["evaluations"] += 1
            for name, shape in shapes.items():
                record = case["reference"][name]
                reference = np.array(
                    [complex(*value) for value in record["values_si"]]
                ).reshape(shape)
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
                    total = max(
                        math.fsum(value for _, value in components),
                        math.fsum(
                            (uncertainty_record["total_error_estimate_si"], additional)
                        ),
                    )
                    total = math.nextafter(total, math.inf)
                    uncertainty_record["total_error_estimate_si"] = total
                    uncertainty_record["used_uncertainty_si"] = total
                uncertainty_record["components"] = components
                uncertainty_record["evidence_refs"] = tuple(
                    uncertainty_record["evidence_refs"]
                ) + (str(reference_path),)
                uncertainty = OracleUncertainty(**uncertainty_record)
                payload = getattr(result, name)
                assert all(
                    payload.meta[0, index].unit == si_units[name]
                    for index in range(payload.shape[1])
                )
                actual = np.asarray(payload)[0, :, 0].reshape(shape)
                assert getattr(result, name).shape == (
                    1,
                    {"potential": 1, "acceleration": 3, "gradient": 9}[name],
                    1,
                )
                assessment = assess(
                    name,
                    actual,
                    reference,
                    case["metadata"]["scales_si"][name],
                    uncertainty,
                )
                budget = assessment["budget_ref_si"]
                max_ratio[name] = max(
                    max_ratio[name],
                    (
                        assessment["difference_norm_si"]
                        + assessment["used_uncertainty_si"]
                    )
                    / budget,
                )
                max_oracle_ratio[name] = max(
                    max_oracle_ratio[name], assessment["used_uncertainty_si"] / budget
                )
                assert math.isclose(
                    norm_si(name, reference), original_norm, rel_tol=3e-15, abs_tol=0.0
                )
                counts["observables"] += 1
                counts["passed" if assessment["accepted"] else "failed"] += 1
                serial = {
                    "fixture": f,
                    "orientation": orientation,
                    "assessment": assessment,
                    "diagnostic": dataclasses.asdict(by_observable[name]),
                    "uncertainty": dataclasses.asdict(uncertainty),
                    "production_si": [
                        [complex(v).real, complex(v).imag] for v in actual.flat
                    ],
                }
                stream.write(json.dumps(serial, allow_nan=False) + "\n")
        if counts["fixtures"] % 24 == 0:
            print(
                json.dumps({**counts, "elapsed_s": time.monotonic() - started}),
                flush=True,
            )
manifest = {
    "claim": "finite_validation_set_with_recorded_estimates_and_exact_parity_bounds",
    "family": args.family,
    "counts": counts,
    "max_error_plus_uncertainty_over_budget": max_ratio,
    "max_oracle_uncertainty_over_budget": max_oracle_ratio,
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
    "elapsed_s": time.monotonic() - started,
}
(output / "manifest.json").write_text(
    json.dumps(manifest, indent=2, allow_nan=False) + "\n"
)
print(json.dumps({k: v for k, v in manifest.items() if k != "source_inputs"}, indent=2))
if counts["failed"] or counts["execution_failures"]:
    raise SystemExit("Surface cavity independent acceptance failed")
