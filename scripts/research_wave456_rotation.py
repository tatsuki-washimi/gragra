#!/usr/bin/env python3
"""Fixed Wave 4--6 rotating-cylinder and passing-cuboid study.

This is deliberately a bounded research script, not a source-model API.  It
uses the established framewise trajectory/rigid-transform observables, so it
never materialises a time-by-source tensor.  All inputs are float64 and the
existing observable layer applies ``G_SI`` exactly once.
"""

# ruff: noqa: E402, E501

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.meta_path[:] = [
    finder
    for finder in sys.meta_path
    if type(finder).__name__ != "ScikitBuildRedirectingFinder"
]
sys.path.insert(0, str(_ROOT / "src"))

from gragra.gravity_sources import PointMass, PointMassCloud
from gragra.gravity_sources.trajectory import (
    PrescribedPointTrajectory,
    PrescribedRigidTransform,
)
from gragra.observables import (
    point_acceleration,
    point_gravity_gradient,
    point_potential,
)
from gragra.observables.trajectory import (
    rigid_transform_acceleration,
    rigid_transform_gravity_gradient,
    rigid_transform_potential,
    trajectory_acceleration,
    trajectory_gravity_gradient,
    trajectory_potential,
)

CYLINDER_RADIUS_M = 0.1
CYLINDER_LENGTH_M = 0.2
CYLINDER_DENSITY_KG_M3 = 7800.0
ORBIT_RADIUS_M = 0.5
ROTATION_FREQUENCY_HZ = 10.0
ROTATION_QUADRATURES = ((4, 16, 4), (8, 32, 8), (16, 64, 16))
ROTATION_TARGETS_M = np.array(
    ((2.0, 0.3, 0.2), (4.0, 1.0, 0.5), (8.0, -1.5, 1.0)), dtype=np.float64
)
ROTATION_SAMPLES = (256, 512, 1024)
PASSING_TARGETS_M = np.array(((0.0, 0.0, 0.0), (0.0, 0.0, 2.0), (3.0, -2.0, 1.0)))
PASSING_QUADRATURE = (8, 8, 8)
RESEARCH_BACKEND = os.environ.get("GRAGRA_RESEARCH_BACKEND", "numpy")


def rotation_z(theta_rad: np.ndarray) -> np.ndarray:
    """Return active float64 Rz matrices for every angle in ``theta_rad``."""
    theta = np.asarray(theta_rad, dtype=np.float64)
    if theta.ndim != 1 or not np.isfinite(theta).all():
        raise ValueError("theta_rad must be a finite one-dimensional float64 array")
    rotations = np.zeros((theta.size, 3, 3), dtype=np.float64)
    rotations[:, 0, 0] = np.cos(theta)
    rotations[:, 0, 1] = -np.sin(theta)
    rotations[:, 1, 0] = np.sin(theta)
    rotations[:, 1, 1] = np.cos(theta)
    rotations[:, 2, 2] = 1.0
    return rotations


def _legendre_interval(
    count: int, lower: float, upper: float
) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(count)
    return (
        np.asarray((upper - lower) * (nodes + 1.0) / 2.0 + lower, dtype=np.float64),
        np.asarray((upper - lower) * weights / 2.0, dtype=np.float64),
    )


def cylinder_cloud(
    quadrature: tuple[int, int, int],
    *,
    radius_m: float,
    length_m: float,
    density_kg_m3: float,
    orbit_radius_m: float = ORBIT_RADIUS_M,
) -> PointMassCloud:
    """Discretise two opposed solid cylinders with total mass held explicit.

    Integrating in ``u=r²`` makes the cylindrical volume measure proportional
    to ``du dtheta dz``.  The duplicated body has exactly two matching masses
    and centres at ``±orbit_radius_m`` before the active transform.
    """
    radial, azimuthal, axial = quadrature
    if min(radial, azimuthal, axial) < 1:
        raise ValueError("all cylinder quadrature counts must be positive")
    if min(radius_m, length_m, density_kg_m3, orbit_radius_m) <= 0.0:
        raise ValueError(
            "cylinder dimensions, density, and orbit radius must be positive"
        )
    u, wu = _legendre_interval(radial, 0.0, radius_m**2)
    z, wz = _legendre_interval(axial, -length_m / 2.0, length_m / 2.0)
    theta = 2.0 * np.pi * np.arange(azimuthal, dtype=np.float64) / azimuthal
    # dV = 1/2 du dtheta dz, with a periodic trapezoid in theta.
    rr, tt, zz = np.meshgrid(np.sqrt(u), theta, z, indexing="ij")
    local = np.column_stack(
        (rr.ravel() * np.cos(tt.ravel()), rr.ravel() * np.sin(tt.ravel()), zz.ravel())
    )
    local_mass = (
        density_kg_m3
        * np.broadcast_to(
            0.5 * wu[:, None, None] * (2.0 * np.pi / azimuthal) * wz[None, None, :],
            (radial, azimuthal, axial),
        ).ravel()
    )
    left = local + np.array((-orbit_radius_m, 0.0, 0.0))
    right = local + np.array((orbit_radius_m, 0.0, 0.0))
    return PointMassCloud(
        np.concatenate((local_mass, local_mass)).astype(np.float64),
        np.vstack((left, right)).astype(np.float64),
    )


def cuboid_cloud(
    quadrature: tuple[int, int, int] = PASSING_QUADRATURE,
    *,
    side_lengths_m: tuple[float, float, float] = (2.0, 1.0, 1.0),
    density_kg_m3: float = 2700.0,
) -> PointMassCloud:
    """Return a body-frame uniform cuboid quadrature cloud."""
    if min(*quadrature, *side_lengths_m, density_kg_m3) <= 0.0:
        raise ValueError("cuboid quadrature, dimensions, and density must be positive")
    axes = [
        _legendre_interval(n, -side / 2.0, side / 2.0)
        for n, side in zip(quadrature, side_lengths_m, strict=True)
    ]
    xx, yy, zz = np.meshgrid(*(axis[0] for axis in axes), indexing="ij")
    mass = (
        density_kg_m3
        * (
            axes[0][1][:, None, None]
            * axes[1][1][None, :, None]
            * axes[2][1][None, None, :]
        ).ravel()
    )
    return PointMassCloud(
        mass.astype(np.float64),
        np.column_stack((xx.ravel(), yy.ravel(), zz.ravel())).astype(np.float64),
    )


def _rotation_transform(
    samples: int, *, axis_rotation_rad: float = 0.0, phase_rad: float = 0.0
) -> PrescribedRigidTransform:
    if samples not in ROTATION_SAMPLES:
        raise ValueError(f"samples must be one of {ROTATION_SAMPLES}")
    theta = 2.0 * np.pi * np.arange(samples, dtype=np.float64) / samples + phase_rad
    time_s = np.arange(samples, dtype=np.float64) / (samples * ROTATION_FREQUENCY_HZ)
    axis_tilt = np.array(
        (
            (1.0, 0.0, 0.0),
            (0.0, np.cos(axis_rotation_rad), -np.sin(axis_rotation_rad)),
            (0.0, np.sin(axis_rotation_rad), np.cos(axis_rotation_rad)),
        ),
        dtype=np.float64,
    )
    rotations = np.einsum("ij,tjk->tik", axis_tilt, rotation_z(theta))
    return PrescribedRigidTransform(time_s, np.zeros((samples, 3)), rotations)


def _observable_series(
    source: PointMassCloud, transform: PrescribedRigidTransform, targets: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        "potential": rigid_transform_potential(
            source, transform, targets, chunk_size=8192, backend=RESEARCH_BACKEND
        ),
        "acceleration": rigid_transform_acceleration(
            source, transform, targets, chunk_size=8192, backend=RESEARCH_BACKEND
        ),
        "gradient": rigid_transform_gravity_gradient(
            source, transform, targets, chunk_size=8192, backend=RESEARCH_BACKEND
        ),
    }


def _point_arrangement_series(
    total_cylinder_mass: float,
    transform: PrescribedRigidTransform,
    targets: np.ndarray,
    orbit_radius_m: float,
) -> dict[str, np.ndarray]:
    cloud = PointMassCloud(
        np.array((total_cylinder_mass, total_cylinder_mass), dtype=np.float64),
        np.array(
            ((-orbit_radius_m, 0.0, 0.0), (orbit_radius_m, 0.0, 0.0)), dtype=np.float64
        ),
    )
    return _observable_series(cloud, transform, targets)


def _monopole_series(
    total_mass_kg: float, frames: int, targets: np.ndarray
) -> dict[str, np.ndarray]:
    source = PointMass(total_mass_kg, (0.0, 0.0, 0.0))
    static = {
        "potential": point_potential(
            source, targets, chunk_size=8192, backend=RESEARCH_BACKEND
        ),
        "acceleration": point_acceleration(
            source, targets, chunk_size=8192, backend=RESEARCH_BACKEND
        ),
        "gradient": point_gravity_gradient(
            source, targets, chunk_size=8192, backend=RESEARCH_BACKEND
        ),
    }
    return {
        name: np.broadcast_to(values, (frames, *values.shape)).copy()
        for name, values in static.items()
    }


def _harmonic_arrays(values: np.ndarray, theta: np.ndarray) -> dict[str, np.ndarray]:
    flat = np.asarray(values, dtype=np.float64).reshape(values.shape[0], -1)
    result: dict[str, np.ndarray] = {
        "dc": np.mean(flat, axis=0).reshape(values.shape[1:]).astype(np.complex128)
    }
    for harmonic in range(1, 13):
        result[str(harmonic)] = 2.0 * np.mean(
            flat * np.exp(1j * harmonic * theta)[:, None], axis=0
        ).reshape(values.shape[1:])
    return result


def _phasors(values: np.ndarray, theta: np.ndarray) -> dict[str, Any]:
    return {
        name: _complex_pairs(value)
        for name, value in _harmonic_arrays(values, theta).items()
    }


def _complex_pairs(values: np.ndarray) -> Any:
    array = np.asarray(values, dtype=np.complex128)
    if array.ndim == 0:
        return [float(array.real), float(array.imag)]
    return [_complex_pairs(item) for item in array]


def _odd_harmonic_controls(
    values: np.ndarray, theta: np.ndarray, observable: str
) -> dict[str, list[float]]:
    """Compare odd amplitudes with the nonzero second-harmonic target scale."""
    harmonics = _harmonic_arrays(values, theta)
    even_scale = _target_norm(harmonics["2"][None, ...], observable)[0]
    odd = np.maximum.reduce(
        [
            _target_norm(harmonics[str(k)][None, ...], observable)[0]
            for k in (1, 3, 5, 7, 9, 11)
        ]
    )
    ratio = np.divide(
        odd, even_scale, out=np.full_like(odd, np.nan), where=even_scale != 0.0
    )
    return {
        "max_odd_abs_per_target": odd.tolist(),
        "second_harmonic_scale_per_target": even_scale.tolist(),
        "max_odd_over_second_per_target": ratio.tolist(),
    }


def _target_norm(values: np.ndarray, observable: str) -> np.ndarray:
    """Return the prescribed physical norm at every frame and target."""
    array = np.asarray(values)
    if observable == "potential":
        return np.abs(array)
    if observable == "acceleration":
        return np.linalg.norm(array, axis=-1)
    if observable == "gradient":
        return np.sqrt(np.sum(np.abs(array) ** 2, axis=(-2, -1)))
    raise ValueError(f"unsupported observable {observable!r}")


def _comparison_by_target(
    candidate: np.ndarray, reference: np.ndarray, observable: str
) -> dict[str, list[float]]:
    """Summarise an observable without mixing targets or component norms."""
    difference_norm = _target_norm(candidate - reference, observable)
    reference_norm = _target_norm(reference, observable)
    if difference_norm.ndim == 1:
        difference_norm = difference_norm[None, :]
        reference_norm = reference_norm[None, :]
    numerator = np.linalg.norm(difference_norm, axis=0)
    denominator = np.linalg.norm(reference_norm, axis=0)
    relative = np.divide(
        numerator,
        denominator,
        out=numerator.copy(),
        where=denominator != 0.0,
    )
    return {
        "max_abs_per_target": np.max(difference_norm, axis=0).tolist(),
        "rms_abs_per_target": np.sqrt(np.mean(difference_norm**2, axis=0)).tolist(),
        "relative_l2_per_target": relative.tolist(),
    }


def finite_difference_target_shift(
    source: PointMassCloud, targets: np.ndarray, *, axis: int, step_m: float
) -> dict[str, np.ndarray | float]:
    """Central target-position derivative of acceleration versus Gamma dr."""
    if axis not in (0, 1, 2) or step_m <= 0.0:
        raise ValueError("axis must be 0, 1, or 2 and step_m must be positive")
    plus = np.array(targets, dtype=np.float64, copy=True)
    minus = np.array(targets, dtype=np.float64, copy=True)
    plus[:, axis] += step_m
    minus[:, axis] -= step_m
    base = point_acceleration(source, targets, chunk_size=8192)
    # point_gravity_gradient is d(a)/d(target), by the established kernel contract.
    gradient = point_gravity_gradient(source, targets, chunk_size=8192)
    central = (
        point_acceleration(source, plus, chunk_size=8192)
        - point_acceleration(source, minus, chunk_size=8192)
    ) / (2.0 * step_m)
    prediction = gradient[:, :, axis]
    return {
        "base": base,
        "central_difference": central,
        "gradient_prediction": prediction,
        "relative_error": max(
            _comparison_by_target(central, prediction, "acceleration")[
                "relative_l2_per_target"
            ]
        ),
    }


def rotating_target_shift(
    source: PointMassCloud,
    transform: PrescribedRigidTransform,
    targets: np.ndarray,
    *,
    axis: int,
    step_m: float,
) -> dict[str, np.ndarray]:
    """Framewise target derivative of acceleration versus ``Gamma(t) dr``."""
    if axis not in (0, 1, 2) or step_m <= 0.0:
        raise ValueError("axis must be 0, 1, or 2 and step_m must be positive")
    plus = np.array(targets, dtype=np.float64, copy=True)
    minus = np.array(targets, dtype=np.float64, copy=True)
    plus[:, axis] += step_m
    minus[:, axis] -= step_m
    central = (
        rigid_transform_acceleration(
            source, transform, plus, chunk_size=8192, backend=RESEARCH_BACKEND
        )
        - rigid_transform_acceleration(
            source, transform, minus, chunk_size=8192, backend=RESEARCH_BACKEND
        )
    ) / (2.0 * step_m)
    gradient = rigid_transform_gravity_gradient(
        source, transform, targets, chunk_size=8192, backend=RESEARCH_BACKEND
    )
    return {
        "central_difference": central,
        "gradient_prediction": gradient[:, :, :, axis],
    }


def _symmetry_metrics(values: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        name: float(
            np.max(np.abs(series[: len(series) // 2] - series[len(series) // 2 :]))
        )
        for name, series in values.items()
    }


def _run_rotation_case(
    quadrature: tuple[int, int, int], samples: int
) -> dict[str, Any]:
    cloud = cylinder_cloud(
        quadrature,
        radius_m=CYLINDER_RADIUS_M,
        length_m=CYLINDER_LENGTH_M,
        density_kg_m3=CYLINDER_DENSITY_KG_M3,
    )
    transform = _rotation_transform(samples)
    theta = 2.0 * np.pi * np.arange(samples, dtype=np.float64) / samples
    finite = _observable_series(cloud, transform, ROTATION_TARGETS_M)
    cylinder_mass = float(cloud.masses_kg.sum() / 2.0)
    points = _point_arrangement_series(
        cylinder_mass, transform, ROTATION_TARGETS_M, ORBIT_RADIUS_M
    )
    far = _monopole_series(float(cloud.masses_kg.sum()), samples, ROTATION_TARGETS_M)
    return {
        "quadrature": list(quadrature),
        "samples_per_rotation": samples,
        "mass_per_cylinder_kg": cylinder_mass,
        "finite_phasors": {
            name: _phasors(series, theta) for name, series in finite.items()
        },
        "point_arrangement_phasors": {
            name: _phasors(series, theta) for name, series in points.items()
        },
        "far_monopole_phasors": {
            name: _phasors(series, theta) for name, series in far.items()
        },
        "pi_periodicity_max_abs": _symmetry_metrics(finite),
        "odd_harmonic_controls": {
            name: _odd_harmonic_controls(series, theta, name)
            for name, series in finite.items()
        },
        "finite_minus_point_by_target": {
            name: _comparison_by_target(finite[name], points[name], name)
            for name in finite
        },
        "finite_minus_far_monopole_by_target": {
            name: _comparison_by_target(finite[name], far[name], name)
            for name in finite
        },
    }


def _rotation_sensitivities() -> dict[str, Any]:
    cloud = cylinder_cloud(
        (16, 64, 16),
        radius_m=CYLINDER_RADIUS_M,
        length_m=CYLINDER_LENGTH_M,
        density_kg_m3=CYLINDER_DENSITY_KG_M3,
    )
    transform = _rotation_transform(512)
    theta = 2.0 * np.pi * np.arange(512, dtype=np.float64) / 512
    position: dict[str, dict[str, Any]] = {}
    for axis in range(3):
        for step in (0.01, 0.005, 0.001):
            shifted = rotating_target_shift(
                cloud, transform, ROTATION_TARGETS_M, axis=axis, step_m=step
            )
            position[f"axis_{axis}_step_{step:g}"] = {
                "step_m": step,
                "coefficient_quantity": "d acceleration vector / d target position",
                "coefficient_units": "s^-2",
                "central_difference_per_m": shifted["central_difference"].tolist(),
                "gamma_prediction_per_m": shifted["gradient_prediction"].tolist(),
                "comparison_by_target": _comparison_by_target(
                    shifted["central_difference"],
                    shifted["gradient_prediction"],
                    "acceleration",
                ),
                "central_harmonics": _phasors(shifted["central_difference"], theta),
                "gamma_harmonics": _phasors(shifted["gradient_prediction"], theta),
            }

    def response(
        builder: Callable[[float], PointMassCloud] | None,
        *,
        transform_builder: Callable[[float], PrescribedRigidTransform] | None,
        delta: float,
        units: str,
        status: str,
    ) -> dict[str, Any]:
        plus_cloud = builder(delta) if builder else cloud
        minus_cloud = builder(-delta) if builder else cloud
        plus_transform = transform_builder(delta) if transform_builder else transform
        minus_transform = transform_builder(-delta) if transform_builder else transform
        plus = _observable_series(plus_cloud, plus_transform, ROTATION_TARGETS_M)[
            "acceleration"
        ]
        minus = _observable_series(minus_cloud, minus_transform, ROTATION_TARGETS_M)[
            "acceleration"
        ]
        central = (plus - minus) / (2.0 * delta)
        return {
            "finite_difference_status": status,
            "perturbation": delta,
            "coefficient_units": units,
            "central_difference_per_unit": central.tolist(),
            "central_harmonics": _phasors(central, theta),
            "coefficient_vector2_peak_per_target": np.max(
                _target_norm(central, "acceleration"), axis=0
            ).tolist(),
        }

    results: dict[str, Any] = {"position": position}
    for name, builder in {
        "mass_fraction": lambda d: cylinder_cloud(
            (16, 64, 16),
            radius_m=CYLINDER_RADIUS_M,
            length_m=CYLINDER_LENGTH_M,
            density_kg_m3=CYLINDER_DENSITY_KG_M3 * (1.0 + d),
        ),
        "radius_fraction_fixed_total_mass": lambda d: cylinder_cloud(
            (16, 64, 16),
            radius_m=CYLINDER_RADIUS_M * (1.0 + d),
            length_m=CYLINDER_LENGTH_M,
            density_kg_m3=CYLINDER_DENSITY_KG_M3 / (1.0 + d) ** 2,
        ),
        "length_fraction_fixed_total_mass": lambda d: cylinder_cloud(
            (16, 64, 16),
            radius_m=CYLINDER_RADIUS_M,
            length_m=CYLINDER_LENGTH_M * (1.0 + d),
            density_kg_m3=CYLINDER_DENSITY_KG_M3 / (1.0 + d),
        ),
    }.items():
        results[name] = {
            str(delta): response(
                builder,
                transform_builder=None,
                delta=delta,
                units="m s^-2 per fractional change",
                status="Proposed finite-difference step choice",
            )
            for delta in (1.0e-3, 5.0e-4)
        }
    results["axis_rotation_rad"] = {
        str(delta): response(
            None,
            transform_builder=lambda d: _rotation_transform(512, axis_rotation_rad=d),
            delta=delta,
            units="m s^-2 rad^-1",
            status="Proposed finite-difference step choice",
        )
        for delta in (1.0e-3, 5.0e-4)
    }
    results["phase_rad"] = {
        str(delta): response(
            None,
            transform_builder=lambda d: _rotation_transform(512, phase_rad=d),
            delta=delta,
            units="m s^-2 rad^-1",
            status="Proposed finite-difference step choice",
        )
        for delta in (1.0e-3, 5.0e-4)
    }
    results["orbit_radius_m"] = {
        str(delta): response(
            lambda d: cylinder_cloud(
                (16, 64, 16),
                radius_m=CYLINDER_RADIUS_M,
                length_m=CYLINDER_LENGTH_M,
                density_kg_m3=CYLINDER_DENSITY_KG_M3,
                orbit_radius_m=ORBIT_RADIUS_M + d,
            ),
            transform_builder=None,
            delta=delta,
            units="s^-2",
            status="Proposed finite-difference step choice",
        )
        for delta in (0.01, 0.005)
    }
    return results


def _passing_case() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    body = cuboid_cloud()
    times = np.arange(-2.0, 2.0 + 0.005, 0.01, dtype=np.float64)
    translations = np.column_stack(
        (10.0 * times, np.full(times.size, 5.0), np.zeros(times.size))
    )
    transform = PrescribedRigidTransform(
        times, translations, np.repeat(np.eye(3)[None], times.size, axis=0)
    )
    finite = _observable_series(body, transform, PASSING_TARGETS_M)
    total_mass = float(body.masses_kg.sum())
    trajectory = PrescribedPointTrajectory(total_mass, times, translations)
    com = {
        "potential": trajectory_potential(
            trajectory, PASSING_TARGETS_M, chunk_size=8192, backend=RESEARCH_BACKEND
        ),
        "acceleration": trajectory_acceleration(
            trajectory, PASSING_TARGETS_M, chunk_size=8192, backend=RESEARCH_BACKEND
        ),
        "gradient": trajectory_gravity_gradient(
            trajectory, PASSING_TARGETS_M, chunk_size=8192, backend=RESEARCH_BACKEND
        ),
    }
    # The requested far model is the monopole/COM limit; it is therefore
    # intentionally identical to the COM point model for a uniform cuboid.
    summary = {
        "samples": int(times.size),
        "time_start_s": float(times[0]),
        "time_stop_s": float(times[-1]),
        "total_mass_kg": total_mass,
        "finite_vs_com_by_target": {
            name: _comparison_by_target(finite[name], com[name], name)
            for name in finite
        },
        "far_model": "moving monopole; identical to COM point model",
        "finite_physical_peak": {
            name: float(np.max(np.abs(series))) for name, series in finite.items()
        },
    }
    raw = {"time_s": times}
    raw.update({f"finite_{name}": values for name, values in finite.items()})
    raw.update({f"com_{name}": values for name, values in com.items()})
    return summary, raw


def _rss_bytes() -> int:
    # Linux ru_maxrss is KiB; record an explicit byte value in the manifest.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_identity() -> dict[str, Any]:
    """Identify this uncommitted research script, not only its base commit."""
    script = Path(__file__).resolve()
    status = subprocess.run(
        ("git", "status", "--short"),
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    script_sha256 = _sha256(script)
    return {
        "script_path": str(script.relative_to(_ROOT)),
        "script_sha256": script_sha256,
        "dirty_or_untracked_code": status,
        "dirty_or_untracked_code_identity": hashlib.sha256(
            json.dumps(
                {"script_sha256": script_sha256, "git_status_short": status},
                sort_keys=True,
            ).encode()
        ).hexdigest(),
    }


def _representative_timing_components() -> dict[str, Any]:
    """Measure named work scopes without treating them as I/O benchmarks."""
    started = time.perf_counter()
    cloud = cylinder_cloud(
        (16, 64, 16),
        radius_m=CYLINDER_RADIUS_M,
        length_m=CYLINDER_LENGTH_M,
        density_kg_m3=CYLINDER_DENSITY_KG_M3,
    )
    transform = _rotation_transform(1024)
    setup_s = time.perf_counter() - started
    started = time.perf_counter()
    _observable_series(cloud, transform, ROTATION_TARGETS_M)
    first_backend_call_s = time.perf_counter() - started
    started = time.perf_counter()
    _observable_series(cloud, transform, ROTATION_TARGETS_M)
    integration_s = time.perf_counter() - started
    return {
        "representative_case": {"quadrature": [16, 64, 16], "samples": 1024},
        "setup_s": setup_s,
        "conversion": {
            "status": "not applicable: rigid cloud and transform are used directly",
            "elapsed_s": 0.0,
        },
        "first_backend_call_s": first_backend_call_s,
        "first_backend_call_scope": "three observables, excludes harmonic conversion",
        "jit_or_cache_status": "unobserved; first_backend_call_s includes any compile or cache-load cost",
        "integration_s": integration_s,
        "integration_scope": "three observables after first call, excludes harmonic conversion and output",
    }


def run(output_dir: Path, *, warm_runs: int = 5) -> Path:
    """Run the prescribed matrix and write one external JSON result manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    representative_timing = _representative_timing_components()
    timing: list[float] = []
    cases: list[dict[str, Any]] = []
    for quadrature in ROTATION_QUADRATURES:
        for samples in ROTATION_SAMPLES:
            started = time.perf_counter()
            case = _run_rotation_case(quadrature, samples)
            elapsed = time.perf_counter() - started
            case["whole_case_including_harmonics_s"] = elapsed
            cases.append(case)
            timing.append(elapsed)
    # Warm repetitions measure the production evaluation without retaining
    # duplicate result tensors.  They use the largest fixed case.
    warm: list[float] = []
    for _ in range(warm_runs):
        started = time.perf_counter()
        _run_rotation_case((16, 64, 16), 1024)
        warm.append(time.perf_counter() - started)
    passing, passing_raw = _passing_case()
    raw_path = output_dir / "passing_raw_series.npz"
    raw_started = time.perf_counter()
    np.savez_compressed(raw_path, **passing_raw)
    raw_output_s = time.perf_counter() - raw_started
    result = {
        "study": "wave4-6 rotation and passing body",
        "classification": "computed; independent A oracle outstanding",
        "source_commit": os.popen("git rev-parse HEAD").read().strip(),
        "source_identity": _source_identity(),
        "numpy_version": np.__version__,
        "existing_observable_backend": RESEARCH_BACKEND,
        "python": platform.python_version(),
        "thread_limits": {
            key: os.environ.get(key)
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS")
        },
        "rotation": {
            "frequency_hz": ROTATION_FREQUENCY_HZ,
            "targets_m": ROTATION_TARGETS_M.tolist(),
            "phase_convention": "Re[q_hat exp(-i harmonic theta)]; q_hat=2 mean(q exp(+i harmonic theta))",
            "cases": cases,
            "sensitivities": _rotation_sensitivities(),
        },
        "passing": {
            **passing,
            "raw_series_path": str(raw_path),
            "raw_series_arrays": sorted(passing_raw),
        },
        "timings_s": {
            "matrix_whole_case_including_harmonics_mean": float(np.mean(timing)),
            "matrix_whole_case_including_harmonics_median": float(np.median(timing)),
            "matrix_whole_case_including_harmonics_std": float(np.std(timing)),
            "largest_case_warm_mean": float(np.mean(warm)),
            "largest_case_warm_median": float(np.median(warm)),
            "largest_case_warm_std": float(np.std(warm)),
            "warm_runs": warm,
            "representative_components": representative_timing,
            "raw_array_output_s": raw_output_s,
        },
        "peak_rss_bytes": _rss_bytes(),
        "independent_oracle": {
            "status": "OUTSTANDING",
            "interface": "/home/washimi/work/gragra-research/2026-09-12-wave4-6/A/interface.md",
            "reason": "A interface is available, but its implementation has not passed parent acceptance/copy; all current finite, point, and far calculations use production observable kernels.",
            "requested_follow_up": "A leading nonzero rotating multipole phasor is required for a meaningful non-DC far comparison.",
        },
    }
    path = output_dir / "rotation_passing_results.json"
    output_started = time.perf_counter()
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    # JSON timing is recorded in a sidecar because self-inserting the elapsed
    # time would change the exact JSON bytes after it was measured.
    (output_dir / "output_timing.json").write_text(
        json.dumps({"result_json_output_s": time.perf_counter() - output_started})
        + "\n"
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warm-runs", type=int, default=5)
    args = parser.parse_args()
    if args.warm_runs < 1:
        raise SystemExit("--warm-runs must be positive")
    result = run(args.output_dir, warm_runs=args.warm_runs)
    print(result)


if __name__ == "__main__":
    main()
