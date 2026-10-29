"""Generate fixed Wave 4--6 reference values outside the repository.

The imported implementation is itself the independent NumPy-only reference,
not a production Newton kernel/integrator/wave path.  This driver records
order ladders and refuses to label a case accepted above the fixed 1e-11 gate.
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np

import gragra
from gragra.analytic.wave456_reference import (
    GFreeObservables,
    acoustic_infinite_halfspace_observables_g_free,
    adaptive_box_observables_g_free,
    adaptive_displacement_box_observables_g_free,
    host_inclusion_observables_g_free,
    opposed_rotor_leading_phasor_g_free,
    opposed_rotor_observables_g_free,
    passing_cuboid_observables_g_free,
    rayleigh_halfspace_observables_g_free,
)

GATE = 1e-11


def _encoded(array: np.ndarray):
    value = np.asarray(array)
    if np.iscomplexobj(value):
        return np.stack((value.real, value.imag), axis=-1).tolist()
    return value.tolist()


def _observables(value: GFreeObservables) -> dict:
    return {
        "potential": _encoded(value.potential),
        "acceleration": _encoded(value.acceleration),
        "gradient": _encoded(value.gradient),
    }


def _metric(value: GFreeObservables) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.abs(value.potential),
        np.linalg.norm(value.acceleration, axis=-1),
        np.linalg.norm(value.gradient, axis=(-2, -1)),
    )


def _difference(low: GFreeObservables, high: GFreeObservables) -> float:
    delta = GFreeObservables(
        high.potential - low.potential,
        high.acceleration - low.acceleration,
        high.gradient - low.gradient,
    )
    ratios = []
    for numerator, denominator in zip(_metric(delta), _metric(high), strict=True):
        scale = max(float(np.max(denominator, initial=0.0)), np.finfo(float).tiny)
        ratios.append(float(np.max(numerator, initial=0.0)) / scale)
    return max(ratios)


def _ladder(values: list[tuple[object, GFreeObservables]]) -> dict:
    trace = []
    for (low_label, low), (high_label, high) in zip(
        values[:-1], values[1:], strict=True
    ):
        trace.append(
            {
                "low": low_label,
                "high": high_label,
                "normalized_difference": _difference(low, high),
            }
        )
    final = trace[-1]["normalized_difference"]
    return {
        "status": "CONVERGED" if final <= GATE else "ORACLE_UNRESOLVED",
        "gate": GATE,
        "trace": trace,
    }


def _rotor() -> dict:
    targets = np.array(((2.0, 0.3, 0.2), (4.0, 1.0, 0.5), (8.0, -1.5, 1.0)))
    angles = np.array((0.0, np.pi / 8.0, np.pi / 4.0))
    gauss_values = []
    for order in (10, 12, 16, 24):
        value = opposed_rotor_observables_g_free(
            angles,
            targets,
            radial_order=order,
            azimuthal_order=96,
            axial_order=order,
        )
        gauss_values.append(({"radial": order, "angular": 96, "axial": order}, value))
    angular_values = []
    for angular_order in (40, 48, 64, 96):
        value = opposed_rotor_observables_g_free(
            angles,
            targets,
            radial_order=24,
            azimuthal_order=angular_order,
            axial_order=24,
        )
        angular_values.append(
            ({"radial": 24, "angular": angular_order, "axial": 24}, value)
        )
    gauss = _ladder(gauss_values)
    angular = _ladder(angular_values)
    status = (
        "CONVERGED"
        if gauss["status"] == angular["status"] == "CONVERGED"
        else "ORACLE_UNRESOLVED"
    )
    return {
        "classification": "Observed numerical reference",
        "status": status,
        "targets_m": targets.tolist(),
        "angles_rad": angles.tolist(),
        "axis_contract": "(angle,target,trailing_components)",
        "gauss_convergence": gauss,
        "angular_convergence": angular,
        "reference": _observables(gauss_values[-1][1]),
        "leading_2omega_phasor": _observables(
            opposed_rotor_leading_phasor_g_free(targets)
        ),
        "phasor_convention": "Re[q_hat exp(-i 2 omega t)]",
    }


def _passing() -> dict:
    targets = np.array(((0.0, 0.0, 0.0), (0.0, 0.0, 2.0), (3.0, -2.0, 1.0)))
    certification_times = np.array((-2.0, -1.0, 0.0, 0.3, 1.0, 2.0))
    values = [
        (
            order,
            passing_cuboid_observables_g_free(
                certification_times, targets, order=order
            ),
        )
        for order in (10, 12, 16, 24)
    ]
    convergence = _ladder(values)
    times = np.linspace(-2.0, 2.0, 401)
    reference = passing_cuboid_observables_g_free(times, targets, order=24)
    return {
        "classification": "Observed numerical reference",
        "status": convergence["status"],
        "targets_m": targets.tolist(),
        "times_s": times.tolist(),
        "axis_contract": "(time,target,trailing_components)",
        "convergence": convergence,
        "reference": _observables(reference),
    }


def _finite_bodies() -> dict:
    host = np.array(((-20.0, 20.0), (-16.0, 16.0), (-24.0, 0.0)))
    inner = np.array(((-4.0, 8.0), (-4.0, 4.0), (-16.0, -8.0)))
    outside = np.array(
        ((20.5, 3.0, -14.0), (21.0, 3.0, -14.0), (22.0, 3.0, -14.0), (24.0, 3.0, -14.0))
    )
    cavity_targets = np.array(
        (
            (2.31, -0.27, -11.81),
            (7.5, 0.37, -12.23),
            (7.0, -0.41, -11.71),
            (6.0, 0.33, -11.79),
        )
    )
    cavity, cavity_evidence = host_inclusion_observables_g_free(
        host,
        inner,
        2300.0,
        0.0,
        cavity_targets,
        rtol=GATE,
        max_evaluations=4096,
    )
    inclusion, inclusion_evidence = host_inclusion_observables_g_free(
        host,
        inner,
        2300.0,
        3200.0,
        outside,
        rtol=GATE,
        max_evaluations=4096,
    )
    center = np.array((2.0, 0.0, -12.0))
    offset = np.array((0.40, -0.25, 0.15))
    affine = np.array(
        ((0.010, 0.004, -0.003), (-0.002, -0.006, 0.001), (0.003, -0.002, 0.008))
    )

    def displacement(points: np.ndarray) -> np.ndarray:
        return offset + (points - center) @ affine.T

    # Six non-overlapping host pieces are reconstructed explicitly here so
    # the displacement integration never assigns mass to the cavity.
    regions = []
    for axis in range(3):
        for side in range(2):
            box = host.copy()
            for prior in range(axis):
                box[prior] = inner[prior]
            box[axis] = (
                (host[axis, 0], inner[axis, 0])
                if side == 0
                else (inner[axis, 1], host[axis, 1])
            )
            regions.append((box, 2300.0, displacement))
    displacement_value, displacement_evidence = (
        adaptive_displacement_box_observables_g_free(
            regions,
            cavity_targets,
            rtol=GATE,
            max_evaluations=4096,
        )
    )
    return {
        "classification": "Observed numerical reference",
        "host_bounds_m": host.tolist(),
        "internal_bounds_m": inner.tolist(),
        "cavity": {
            "targets_m": cavity_targets.tolist(),
            "convergence": asdict(cavity_evidence),
            "reference": _observables(cavity),
        },
        "inclusion": {
            "targets_m": outside.tolist(),
            "convergence": asdict(inclusion_evidence),
            "reference": _observables(inclusion),
        },
        "affine_displacement": {
            "b_m": offset.tolist(),
            "A": affine.tolist(),
            "center_m": center.tolist(),
            "convergence": asdict(displacement_evidence),
            "reference": _observables(displacement_value),
        },
    }


def _environment() -> dict:
    wavelength = 34.3
    bounds = np.array(
        (
            (-3 * wavelength, 3 * wavelength),
            (-3 * wavelength, 3 * wavelength),
            (0.0, 2 * wavelength),
        )
    )
    targets = np.array(
        (
            (0.0, 0.0, -0.25 * wavelength),
            (0.125 * wavelength, 0.0, -0.5 * wavelength),
            (0.0, 0.25 * wavelength, -wavelength),
        )
    )
    k = 2 * np.pi / wavelength

    def density(points: np.ndarray) -> np.ndarray:
        return np.exp(1j * k * points[:, 0]) / 343.0**2

    # Numerical partitions resolve the oscillatory phase; they are not
    # material interfaces and therefore do not duplicate source volume.
    x_edges = np.linspace(bounds[0, 0], bounds[0, 1], 13)
    y_edges = np.linspace(bounds[1, 0], bounds[1, 1], 7)
    z_edges = np.linspace(bounds[2, 0], bounds[2, 1], 3)
    regions = []
    for x0, x1 in zip(x_edges[:-1], x_edges[1:], strict=True):
        for y0, y1 in zip(y_edges[:-1], y_edges[1:], strict=True):
            for z0, z1 in zip(z_edges[:-1], z_edges[1:], strict=True):
                regions.append((np.array(((x0, x1), (y0, y1), (z0, z1))), density))
    finite, evidence = adaptive_box_observables_g_free(
        regions,
        targets,
        rtol=GATE,
        max_evaluations=2048,
    )
    infinite = acoustic_infinite_halfspace_observables_g_free(targets)
    rayleigh_targets = np.array(
        ((0.0, 0.0, 10.0), (40.0, 0.0, 25.0), (-20.0, 15.0, 50.0))
    )
    rayleigh = rayleigh_halfspace_observables_g_free(rayleigh_targets)
    return {
        "classification": "Observed analytic and numerical references",
        "acoustic": {
            "bounds_m": bounds.tolist(),
            "targets_m": targets.tolist(),
            "finite_convergence": asdict(evidence),
            "finite_reference": _observables(finite),
            "infinite_horizontal_halfspace_reference": _observables(infinite),
        },
        "rayleigh": {
            "parameters": {
                "rho_kg_m3": 2300.0,
                "vp_m_s": 2800.0,
                "vs_m_s": 1500.0,
                "frequency_hz": 2.5,
                "vertical_surface_amplitude_m": 1e-9,
                "convention": "z-up exp(i k x-i omega t)",
            },
            "targets_m": rayleigh_targets.tolist(),
            "reference": _observables(rayleigh),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "classification": {
            "computed_values": "Observed",
            "closed_form_interpretations": "Inferred, CAS-verified",
            "imputed_values": [],
        },
        "metadata": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "gragra_source": str(Path(gragra.__file__).resolve()),
            "rtol": GATE,
            "potential_metric": "absolute value",
            "acceleration_metric": "vector 2-norm",
            "gradient_metric": "Frobenius norm",
            "complex_encoding": "final axis [real,imaginary]",
        },
        "rotor": _rotor(),
        "passing_cuboid": _passing(),
        "finite_bodies": _finite_bodies(),
        "environment": _environment(),
    }
    output = args.output_dir / "reference-values.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite prior oracle attempt: {output}")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
