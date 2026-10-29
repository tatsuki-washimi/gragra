"""Run the bounded OpenFOAM cosine-bell density-anomaly study.

This script prescribes an analytic field at three times and runs OpenFOAM
utilities only.  It does not invoke a CFD solver.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import resource
import shlex
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from gragra.adapters.openfoam_density import load_openfoam_density
from gragra.constants import G_SI
from gragra.observables import (
    point_acceleration,
    point_gravity_gradient,
    point_potential,
)

DOMAIN_MIN_M = -2.0
DOMAIN_MAX_M = 2.0
LEVELS = (16, 32, 48)
TIMES_S = (0.0, 1.0, 2.0)
BACKGROUND_DENSITY_KG_M3 = 1.2
ANOMALY_AMPLITUDE_KG_M3 = -0.2
RADIUS_M = 0.5
TRANSLATION_VELOCITY_M_S = np.array([0.5, 0.0, 0.0])
TARGETS_M = np.array(
    [[3.0, 0.7, -0.4], [-2.8, 1.3, 1.7], [0.25, -3.0, 2.6]],
    dtype=np.float64,
)
MAX_RETAINED_BYTES = 1 << 30
MIN_FREE_BYTES = 10 * (1 << 30)
UTILITY_TIMEOUT_S = 120
ANALYSIS_TIMEOUT_S = 180
LEGACY_ASCII_PRECISION = 6

PROVENANCE_TEMPLATE = {
    "distribution": "OpenFOAM Foundation",
    "version": "14",
    "patch": "20260724",
    "solver": "manufactured-no-solver",
    "compressibility": "not-applicable",
    "region": "region0",
    "reconstructed": True,
    "exporter": "foamToVTK",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _all_retained_run_bytes(parent: Path) -> int:
    return sum(
        _tree_bytes(path)
        for path in parent.glob("openfoam-cosine-bell*")
        if path.is_dir()
    )


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_ascii(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="ascii")


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def centre_m(time_s: float) -> np.ndarray:
    """Return the prescribed anomaly centre in metres."""
    return np.array([-0.5 + 0.5 * float(time_s), 0.0, 0.0])


def cosine_bell_density_kg_m3(points_m: np.ndarray, time_s: float) -> np.ndarray:
    """Evaluate the compact density anomaly at nominal float64 coordinates."""
    points = np.asarray(points_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points_m must be a finite array with shape (N, 3)")
    radii = np.linalg.norm(points - centre_m(time_s), axis=1)
    values = np.zeros(len(points), dtype=np.float64)
    inside = radii < RADIUS_M
    values[inside] = (
        ANOMALY_AMPLITUDE_KG_M3 / 2.0 * (1.0 + np.cos(np.pi * radii[inside] / RADIUS_M))
    )
    return values


def nominal_cell_centres_m(level: int) -> np.ndarray:
    """Return uniform cell centres in OpenFOAM's x-fastest block order."""
    if isinstance(level, bool) or int(level) != level or level <= 0:
        raise ValueError("level must be a positive integer")
    width = (DOMAIN_MAX_M - DOMAIN_MIN_M) / int(level)
    axis = DOMAIN_MIN_M + (np.arange(int(level), dtype=np.float64) + 0.5) * width
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.column_stack((x.ravel(order="F"), y.ravel(order="F"), z.ravel(order="F")))


def analytic_total_mass_kg() -> float:
    """Return the exact continuum anomaly mass from the radial integral."""
    return float(
        2.0
        * np.pi
        * ANOMALY_AMPLITUDE_KG_M3
        * RADIUS_M**3
        * (1.0 / 3.0 - 2.0 / np.pi**2)
    )


def legacy_ascii_float32(values: np.ndarray) -> np.ndarray:
    """Model Foundation 14's float cast and default-precision ASCII output."""
    cast = np.asarray(values, dtype=np.float32)
    flattened = [
        format(float(value), f".{LEGACY_ASCII_PRECISION}g") for value in cast.flat
    ]
    return np.asarray(flattened, dtype=np.float32).reshape(cast.shape)


def affine_hex_geometry_diagnostic(
    points: np.ndarray,
    connectivity: np.ndarray,
    centres: np.ndarray,
    volumes: np.ndarray,
) -> dict[str, object]:
    """Reproduce the W6 affine-hex C/Vc residual tolerance for evidence."""
    points_array = np.asarray(points)
    connectivity_array = np.asarray(connectivity, dtype=np.int64)
    centres_array = np.asarray(centres)
    volumes_array = np.asarray(volumes)
    if connectivity_array.ndim != 2 or connectivity_array.shape[1] != 8:
        raise ValueError("connectivity must have shape (N, 8)")
    if centres_array.shape != (len(connectivity_array), 3):
        raise ValueError("centres must have shape (N, 3)")
    if volumes_array.shape != (len(connectivity_array),):
        raise ValueError("volumes must have shape (N,)")
    eps = max(
        np.finfo(points_array.dtype).eps,
        np.finfo(centres_array.dtype).eps,
        np.finfo(volumes_array.dtype).eps,
    )
    first_failure = None
    max_centre_ratio = 0.0
    max_volume_ratio = 0.0
    max_centre_cell = 0
    max_centre_component = 0
    max_volume_cell = 0
    for cell_index, row in enumerate(connectivity_array):
        vertices = np.asarray(points_array[row], dtype=np.float64)
        h = float(
            np.max(
                np.linalg.norm(
                    vertices[:, None, :] - vertices[None, :, :], axis=-1
                )
            )
        )
        p0, p1, _p2, p3, p4, _p5, _p6, _p7 = vertices
        geometric_centre = vertices.mean(axis=0)
        geometric_volume = abs(
            float(np.linalg.det(np.stack((p1 - p0, p3 - p0, p4 - p0), axis=1)))
        )
        native_centre = np.asarray(centres_array[cell_index], dtype=np.float64)
        native_volume = float(volumes_array[cell_index])
        centre_tolerance = (
            8 * eps * np.maximum(abs(native_centre), abs(geometric_centre))
            + 32 * eps * h
        )
        volume_tolerance = 32 * eps * max(
            abs(native_volume), abs(geometric_volume)
        ) + 64 * eps * h**3
        centre_residual = abs(native_centre - geometric_centre)
        volume_residual = abs(native_volume - geometric_volume)
        centre_ratios = centre_residual / centre_tolerance
        component = int(np.argmax(centre_ratios))
        if centre_ratios[component] > max_centre_ratio:
            max_centre_ratio = float(centre_ratios[component])
            max_centre_cell = cell_index
            max_centre_component = component
        volume_ratio = volume_residual / volume_tolerance
        if volume_ratio > max_volume_ratio:
            max_volume_ratio = float(volume_ratio)
            max_volume_cell = cell_index
        failed = bool(
            np.any(centre_residual > centre_tolerance)
            or volume_residual > volume_tolerance
        )
        if failed and first_failure is None:
            first_failure = {
                "cell_index": cell_index,
                "native_centre_m": native_centre.tolist(),
                "geometric_centre_from_exported_vertices_m": (
                    geometric_centre.tolist()
                ),
                "centre_residual_m": centre_residual.tolist(),
                "centre_tolerance_m": centre_tolerance.tolist(),
                "native_volume_m3": native_volume,
                "geometric_volume_from_exported_vertices_m3": geometric_volume,
                "volume_residual_m3": volume_residual,
                "volume_tolerance_m3": float(volume_tolerance),
                "cell_diameter_m": h,
                "float_epsilon": float(eps),
            }
    return {
        "passed": first_failure is None,
        "first_failure": first_failure,
        "max_centre_residual_to_tolerance_ratio": max_centre_ratio,
        "max_centre_ratio_cell": max_centre_cell,
        "max_centre_ratio_component": max_centre_component,
        "max_volume_residual_to_tolerance_ratio": max_volume_ratio,
        "max_volume_ratio_cell": max_volume_cell,
    }


def _literal_observables(
    positions_m: np.ndarray, weights_kg: np.ndarray, targets_m: np.ndarray
) -> dict[str, np.ndarray]:
    positions = np.asarray(positions_m, dtype=np.float64)
    weights = np.asarray(weights_kg, dtype=np.float64)
    targets = np.asarray(targets_m, dtype=np.float64)
    displacement = positions[None, :, :] - targets[:, None, :]
    radius2 = np.einsum("mni,mni->mn", displacement, displacement)
    if np.any(radius2 == 0.0):
        raise ValueError("literal oracle requires non-colliding targets")
    radius = np.sqrt(radius2)
    potential = -G_SI * np.sum(weights[None, :] / radius, axis=1)
    acceleration = G_SI * np.sum(
        weights[None, :, None] * displacement / radius[..., None] ** 3, axis=1
    )
    outer = displacement[..., :, None] * displacement[..., None, :]
    kernel = 3.0 * outer / radius[..., None, None] ** 5
    kernel -= np.eye(3)[None, None, :, :] / radius[..., None, None] ** 3
    gradient = G_SI * np.sum(weights[None, :, None, None] * kernel, axis=1)
    return {
        "potential": potential,
        "acceleration": acceleration,
        "gradient": gradient,
    }


def literal_discrete_observables(
    positions_m: np.ndarray, weights_kg: np.ndarray, targets_m: np.ndarray
) -> dict[str, np.ndarray]:
    """Evaluate literal point formulas without calling a gragra kernel."""
    return _literal_observables(positions_m, weights_kg, targets_m)


def point_anomaly_observables(
    anomaly_centre_m: np.ndarray, targets_m: np.ndarray
) -> dict[str, np.ndarray]:
    """Return the spherical-shell-theorem continuum reference."""
    centre = np.asarray(anomaly_centre_m, dtype=np.float64)
    targets = np.asarray(targets_m, dtype=np.float64)
    if centre.shape != (3,) or targets.ndim != 2 or targets.shape[1] != 3:
        raise ValueError("centre and targets must have shapes (3,) and (M, 3)")
    radii = np.linalg.norm(targets - centre, axis=1)
    if np.any(radii <= RADIUS_M):
        raise ValueError("continuum targets must lie outside the anomaly")
    return _literal_observables(centre[None, :], [analytic_total_mass_kg()], targets)


def _scalar_field(name: str, values: np.ndarray) -> str:
    body = "\n".join(format(float(value), ".17g") for value in values)
    return f"""FoamFile
{{
    version 2.0;
    format ascii;
    class volScalarField;
    object {name};
}}
dimensions [1 -3 0 0 0 0 0];
internalField nonuniform List<scalar> {len(values)}
(
{body}
)
;
boundaryField
{{
    walls {{ type zeroGradient; }}
}}
"""


def generate_case(case: Path, level: int) -> dict[str, object]:
    """Write one solver-free OpenFOAM case and its three analytic fields."""
    case = Path(case)
    if case.exists():
        raise FileExistsError(case)
    centres = nominal_cell_centres_m(level)
    _write_ascii(
        case / "system/blockMeshDict",
        f"""FoamFile {{ version 2.0; format ascii; class dictionary;
object blockMeshDict; }}
scale 1;
vertices
(
    (-2 -2 -2) (2 -2 -2) (2 2 -2) (-2 2 -2)
    (-2 -2 2) (2 -2 2) (2 2 2) (-2 2 2)
);
blocks (hex (0 1 2 3 4 5 6 7) ({level} {level} {level}) simpleGrading (1 1 1));
edges ();
boundary ();
defaultPatch {{ name walls; type wall; }}
""",
    )
    _write_ascii(
        case / "system/controlDict",
        """FoamFile { version 2.0; format ascii; class dictionary;
object controlDict; }
application none;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime 2;
deltaT 1;
writeControl timeStep;
writeInterval 1;
""",
    )
    snapshots: dict[str, object] = {}
    for time_s in TIMES_S:
        time_name = format(time_s, "g")
        density = cosine_bell_density_kg_m3(centres, time_s)
        cast = density.astype(np.float32)
        native = legacy_ascii_float32(density)
        _write_ascii(
            case / time_name / "delta_rho", _scalar_field("delta_rho", density)
        )
        snapshots[time_name] = {
            "time_s": time_s,
            "centre_m": centre_m(time_s).tolist(),
            "nonzero_cells": int(np.count_nonzero(density)),
            "nominal_density_sha256": hashlib.sha256(density.tobytes()).hexdigest(),
            "expected_native_density_sha256": hashlib.sha256(
                native.tobytes()
            ).hexdigest(),
            "max_abs_float32_conversion_kg_m3": float(
                np.max(np.abs(cast.astype(np.float64) - density), initial=0.0)
            ),
            "max_abs_legacy_ascii_roundtrip_kg_m3": float(
                np.max(np.abs(native.astype(np.float64) - cast), initial=0.0)
            ),
            "total_density_range_kg_m3": [
                float(np.min(BACKGROUND_DENSITY_KG_M3 + density)),
                float(np.max(BACKGROUND_DENSITY_KG_M3 + density)),
            ],
        }
    return {
        "status": "complete",
        "level": level,
        "cell_count": int(level**3),
        "cell_width_m": 4.0 / level,
        "density_evaluation_dtype": "float64",
        "native_export_dtype": "float32",
        "legacy_ascii_precision_significant_digits": LEGACY_ASCII_PRECISION,
        "cell_order": "x-fastest, then y, then z",
        "snapshots": snapshots,
    }


def _activation_environment(activation: Path) -> dict[str, str]:
    command = [
        "bash",
        "-lc",
        'source "$1" && env -0',
        "bash",
        str(activation),
    ]
    result = subprocess.run(command, check=True, capture_output=True)
    return {
        key.decode(): value.decode()
        for item in result.stdout.split(b"\0")
        if item
        for key, value in [item.split(b"=", 1)]
    }


def _parse_max_rss_kib(path: Path) -> int | None:
    for line in path.read_text(encoding="utf-8").splitlines():
        if "Maximum resident set size (kbytes):" in line:
            return int(line.rsplit(":", 1)[1])
    return None


def _run_utility(
    argv: list[str], *, environment: dict[str, str], log_dir: Path, label: str
) -> dict[str, object]:
    stdout_path = log_dir / f"{label}.stdout.log"
    stderr_path = log_dir / f"{label}.stderr.log"
    resource_path = log_dir / f"{label}.resource.log"
    executed_argv = ["/usr/bin/time", "-v", "-o", str(resource_path), *argv]
    started_at = _iso_now()
    start = time.monotonic()
    timed_out = False
    try:
        result = subprocess.run(
            executed_argv,
            env=environment,
            capture_output=True,
            text=True,
            timeout=UTILITY_TIMEOUT_S,
            check=False,
        )
        exit_code = result.returncode
        stdout, stderr = result.stdout, result.stderr
    except subprocess.TimeoutExpired as error:
        timed_out = True
        exit_code = 124
        stdout = error.stdout or ""
        stderr = error.stderr or ""
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    record = {
        "utility_argv": argv,
        "executed_argv": executed_argv,
        "started_at": started_at,
        "ended_at": _iso_now(),
        "duration_s": time.monotonic() - start,
        "timeout_s": UTILITY_TIMEOUT_S,
        "timed_out": timed_out,
        "exit": exit_code,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "stdout_sha256": _sha256(stdout_path),
        "stderr_sha256": _sha256(stderr_path),
        "resource_log": str(resource_path),
        "peak_rss_kib": _parse_max_rss_kib(resource_path)
        if resource_path.is_file()
        else None,
    }
    if exit_code:
        raise RuntimeError(json.dumps(record, indent=2))
    return record


@contextmanager
def _time_limit(seconds: int):
    def handle_timeout(_signum, _frame):
        raise TimeoutError(f"analysis exceeded {seconds} seconds")

    previous = signal.signal(signal.SIGALRM, handle_timeout)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _norm_per_target(value: np.ndarray) -> np.ndarray:
    return (
        np.abs(value)
        if value.ndim == 1
        else np.linalg.norm(value, axis=tuple(range(1, value.ndim)))
    )


def _relative_errors(got: np.ndarray, reference: np.ndarray) -> list[float]:
    return (_norm_per_target(got - reference) / _norm_per_target(reference)).tolist()


def _discrete_scales(
    positions: np.ndarray, weights: np.ndarray, targets: np.ndarray
) -> dict[str, np.ndarray]:
    radii = np.linalg.norm(positions[None, :, :] - targets[:, None, :], axis=-1)
    return {
        "potential": G_SI * np.sum(np.abs(weights)[None, :] / radii, axis=1),
        "acceleration": G_SI * np.sum(np.abs(weights)[None, :] / radii**2, axis=1),
        "gradient": G_SI * np.sum(np.abs(weights)[None, :] / radii**3, axis=1),
    }


def _analyse_level(
    case: Path, level: int, generated: dict[str, object]
) -> dict[str, object]:
    snapshots = []
    observations: dict[str, object] = {}
    nominal_centres = nominal_cell_centres_m(level)
    native_centres = legacy_ascii_float32(nominal_centres).astype(np.float64)
    expected_volume = legacy_ascii_float32(
        np.array((4.0 / level) ** 3)
    ).astype(np.float64)
    process_rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    provenance = provenance_for_case(case)
    started = time.monotonic()
    with _time_limit(ANALYSIS_TIMEOUT_S):
        for time_s in TIMES_S:
            time_name = format(time_s, "g")
            vtk_path = case / "VTK" / f"{case.name}_{time_name}.vtk"
            snapshot = load_openfoam_density(
                vtk_path,
                field_name="delta_rho",
                time_s=time_s,
                time_name=time_name,
                coordinate_frame="openfoam-case",
                coordinate_origin_m=np.zeros(3),
                density_unit="kg/m^3",
                coordinate_unit="m",
                volume_unit="m^3",
                time_unit="s",
                provenance=provenance,
            )
            expected_density = legacy_ascii_float32(
                cosine_bell_density_kg_m3(nominal_centres, time_s)
            ).astype(np.float64)
            if not np.array_equal(snapshot.positions_m, native_centres):
                raise AssertionError("loaded C differs from nominal float32 centres")
            if not np.array_equal(snapshot.density_kg_m3, expected_density):
                raise AssertionError(
                    "loaded density differs from generated float32 profile"
                )
            if not np.array_equal(snapshot.cell_ids, np.arange(level**3)):
                raise AssertionError("cellID order differs from nominal source order")
            if not np.array_equal(
                snapshot.volumes_m3, np.full(level**3, expected_volume)
            ):
                raise AssertionError("loaded Vc differs from the uniform native volume")
            snapshots.append(snapshot)

            weighted = snapshot.as_volume_element_source().as_weighted_source()
            production = {
                "potential": point_potential(weighted, TARGETS_M),
                "acceleration": point_acceleration(weighted, TARGETS_M),
                "gradient": point_gravity_gradient(weighted, TARGETS_M),
            }
            literal = literal_discrete_observables(
                snapshot.positions_m, weighted.weights_kg, TARGETS_M
            )
            continuum = point_anomaly_observables(centre_m(time_s), TARGETS_M)
            scales = _discrete_scales(
                snapshot.positions_m, weighted.weights_kg, TARGETS_M
            )
            literal_checks = {}
            continuum_errors = {}
            for name in production:
                error_norm = _norm_per_target(production[name] - literal[name])
                ref_norm = _norm_per_target(literal[name])
                tolerance = 1e-12 * ref_norm + 1e-14 * scales[name]
                literal_checks[name] = {
                    "error_norm": error_norm.tolist(),
                    "reference_norm": ref_norm.tolist(),
                    "scale": scales[name].tolist(),
                    "tolerance": tolerance.tolist(),
                    "passed": bool(np.all(error_norm <= tolerance)),
                }
                continuum_errors[name] = _relative_errors(
                    production[name], continuum[name]
                )
            mass = float(np.sum(weighted.weights_kg))
            anomaly_centre = centre_m(time_s)
            radial = TARGETS_M - anomaly_centre
            translation_sign = bool(
                np.all(production["potential"] > 0.0)
                and np.all(
                    np.einsum("mi,mi->m", production["acceleration"], radial) > 0.0
                )
            )
            observations[time_name] = {
                "vtk_path": str(vtk_path),
                "vtk_sha256": snapshot.provenance["input_sha256"],
                "native_dtypes": {
                    key: snapshot.provenance[key]
                    for key in (
                        "points_dtype",
                        "centre_dtype",
                        "volume_dtype",
                        "density_dtype",
                        "cell_id_dtype",
                    )
                },
                "cell_count": len(snapshot.cell_ids),
                "time_s": snapshot.time_s,
                "time_name": snapshot.time_name,
                "coordinate_frame": snapshot.coordinate_frame,
                "coordinate_origin_m": snapshot.coordinate_origin_m.tolist(),
                "export_command": snapshot.provenance["export_command"],
                "cell_ids_sha256": hashlib.sha256(
                    snapshot.cell_ids.tobytes()
                ).hexdigest(),
                "native_centres_sha256": hashlib.sha256(
                    snapshot.positions_m.tobytes()
                ).hexdigest(),
                "native_volumes_sha256": hashlib.sha256(
                    snapshot.volumes_m3.tobytes()
                ).hexdigest(),
                "native_density_sha256": hashlib.sha256(
                    snapshot.density_kg_m3.tobytes()
                ).hexdigest(),
                "grid_mass_kg": mass,
                "analytic_mass_kg": analytic_total_mass_kg(),
                "mass_relative_error": abs(mass - analytic_total_mass_kg())
                / abs(analytic_total_mass_kg()),
                "literal_checks": literal_checks,
                "continuum_relative_errors": continuum_errors,
                "translation_sign_passed": translation_sign,
                "production": {
                    key: value.tolist() for key, value in production.items()
                },
                "literal_discrete": {
                    key: value.tolist() for key, value in literal.items()
                },
                "continuum": {key: value.tolist() for key, value in continuum.items()},
                "generator": generated["snapshots"][time_name],
            }

        snapshots[0].assert_compatible_successor(snapshots[1])
        snapshots[1].assert_compatible_successor(snapshots[2])
        native_fields = [snapshot.density_kg_m3 for snapshot in snapshots]
        shift = int(round(0.5 / (4.0 / level)))
        translated = all(
            np.array_equal(
                np.roll(
                    native_fields[index].reshape((level,) * 3, order="F"), shift, axis=0
                ),
                native_fields[index + 1].reshape((level,) * 3, order="F"),
            )
            for index in (0, 1)
        )
    masses = np.array(
        [observations[format(time_s, "g")]["grid_mass_kg"] for time_s in TIMES_S]
    )
    absolute_mass = np.array(
        [
            np.sum(
                np.abs(
                    snapshot.as_volume_element_source().as_weighted_source().weights_kg
                )
            )
            for snapshot in snapshots
        ]
    )
    mass_time_tolerance = float(128 * np.finfo(np.float32).eps * np.max(absolute_mass))
    max_mass_time_difference = float(np.max(np.abs(masses - masses[0])))
    return {
        "status": "complete",
        "level": level,
        "analysis_duration_s": time.monotonic() - started,
        "analysis_timeout_s": ANALYSIS_TIMEOUT_S,
        "process_peak_rss_kib_before": process_rss_before,
        "process_peak_rss_kib_after": resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss,
        "successor_compatibility_passed": True,
        "native_translation_cells": shift,
        "native_translation_passed": translated,
        "max_mass_time_difference_kg": max_mass_time_difference,
        "mass_time_tolerance_kg": mass_time_tolerance,
        "mass_time_consistency_passed": max_mass_time_difference <= mass_time_tolerance,
        "times": observations,
    }


def _vtk_geometry_diagnostic(vtk_path: Path) -> dict[str, object]:
    import meshio

    mesh = meshio.read(vtk_path)
    hexahedra = [block.data for block in mesh.cells if block.type == "hexahedron"]
    if len(hexahedra) != 1:
        return {
            "passed": False,
            "diagnostic_error": "expected exactly one hexahedron block",
        }
    centres = np.concatenate(mesh.cell_data["C"])
    volumes = np.concatenate(mesh.cell_data["Vc"])
    if volumes.ndim == 2 and volumes.shape[1:] == (1,):
        volumes = volumes[:, 0]
    return affine_hex_geometry_diagnostic(
        mesh.points, hexahedra[0], centres, volumes
    )


def _scientific_derivation() -> dict[str, object]:
    import mpmath
    import sympy

    radius = sympy.symbols("R", positive=True, real=True)
    amplitude = sympy.symbols("Delta", real=True)
    support = sympy.symbols("a", positive=True, real=True)
    wave_number = sympy.pi / support
    profile = amplitude / 2 * (1 + sympy.cos(wave_number * radius))
    integrated = sympy.integrate(
        4 * sympy.pi * radius**2 * profile, (radius, 0, support)
    )
    expected = (
        2 * sympy.pi * amplitude * support**3 * (sympy.Rational(1, 3) - 2 / sympy.pi**2)
    )
    mass_difference = sympy.simplify(integrated - expected)
    boundary_value = sympy.simplify(profile.subs(radius, support))
    boundary_derivative = sympy.simplify(
        sympy.diff(profile, radius).subs(radius, support)
    )

    x, y, z, t, x0, velocity = sympy.symbols("x y z t x0 velocity", real=True)
    arbitrary_profile = sympy.Function("f")((x - x0 - velocity * t) ** 2 + y**2 + z**2)
    advection_residual = sympy.simplify(
        sympy.diff(arbitrary_profile, t) + velocity * sympy.diff(arbitrary_profile, x)
    )

    mpmath.mp.dps = 80
    amp_mp = mpmath.mpf(str(ANOMALY_AMPLITUDE_KG_M3))
    radius_mp = mpmath.mpf(str(RADIUS_M))
    numerical = mpmath.quad(
        lambda r: (
            4
            * mpmath.pi
            * r**2
            * amp_mp
            / 2
            * (1 + mpmath.cos(mpmath.pi * r / radius_mp))
        ),
        [0, radius_mp],
    )
    analytic_mp = (
        2 * mpmath.pi * amp_mp * radius_mp**3 * (mpmath.mpf(1) / 3 - 2 / mpmath.pi**2)
    )
    return {
        "classification": (
            "inferred_from_exact_CAS_and_independent_high_precision_quadrature"
        ),
        "assumptions": {
            "R": "real, positive",
            "a": "real, positive",
            "Delta": "real",
            "interior_advection": (
                "constant x velocity; radial profile represented as f(r^2)"
            ),
        },
        "sympy_version": sympy.__version__,
        "mass_integral_expression": str(integrated),
        "mass_expected_expression": str(expected),
        "mass_canonical_difference": str(mass_difference),
        "mass_exact_passed": mass_difference == 0,
        "boundary_density_expression": str(boundary_value),
        "boundary_radial_derivative_expression": str(boundary_derivative),
        "boundary_continuity_passed": boundary_value == 0 and boundary_derivative == 0,
        "advection_residual_expression": str(advection_residual),
        "advection_exact_passed": advection_residual == 0,
        "mpmath_version": mpmath.__version__,
        "mpmath_dps": mpmath.mp.dps,
        "quadrature_mass_kg": str(numerical),
        "high_precision_formula_mass_kg": str(analytic_mp),
        "quadrature_absolute_difference_kg": str(abs(numerical - analytic_mp)),
        "quadrature_passed": bool(abs(numerical - analytic_mp) <= mpmath.mpf("1e-75")),
        "units": {
            "density": "kg/m^3",
            "radial_volume_element": "m^3",
            "mass": "kg",
            "potential": "m^2/s^2",
            "acceleration": "m/s^2",
            "gravity_gradient": "1/s^2",
        },
    }


def _utility_commands(case: Path) -> list[list[str]]:
    time_selection = "0,1,2"
    return [
        ["blockMesh", "-case", str(case)],
        ["checkMesh", "-case", str(case), "-time", time_selection],
        [
            "foamPostProcess",
            "-case",
            str(case),
            "-time",
            time_selection,
            "-func",
            "writeCellCentres",
        ],
        [
            "foamPostProcess",
            "-case",
            str(case),
            "-time",
            time_selection,
            "-func",
            "writeCellVolumes",
        ],
        [
            "foamToVTK",
            "-case",
            str(case),
            "-time",
            time_selection,
            "-useTimeName",
            "-ascii",
            "-noPointValues",
            "-polyhedra",
            "none",
            "-fields",
            "(delta_rho C Vc)",
        ],
    ]


def provenance_for_case(case: Path) -> dict[str, object]:
    """Return loader provenance with the exact export argv for ``case``."""
    export_argv = _utility_commands(case)[-1]
    return {
        **PROVENANCE_TEMPLATE,
        "export_command": shlex.join(export_argv),
    }


def _hash_files(root: Path, paths: list[Path]) -> dict[str, dict[str, object]]:
    return {
        str(path.relative_to(root)): {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(paths)
        if path.is_file()
    }


def _capture_repository_delivery(output: Path, worktree: Path) -> dict[str, object]:
    relative_files = [
        Path("scripts/research_openfoam_cosine_bell.py"),
        Path("tests/test_openfoam_cosine_bell.py"),
    ]
    delivery = output / "delivery"
    copied = delivery / "new-files"
    copied.mkdir(parents=True)
    file_records = {}
    for relative in relative_files:
        source = worktree / relative
        target = copied / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        file_records[str(relative)] = {
            "before": "absent",
            "bytes": source.stat().st_size,
            "sha256": _sha256(source),
            "raw_copy": str(target),
            "raw_copy_sha256": _sha256(target),
        }
    diff = subprocess.run(
        ["git", "diff", "--no-ext-diff", "--", *map(str, relative_files)],
        cwd=worktree,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    for relative in relative_files:
        if not subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(relative)],
            cwd=worktree,
            check=False,
            capture_output=True,
        ).returncode:
            raise AssertionError(f"expected new untracked file: {relative}")
        addition = subprocess.run(
            ["git", "diff", "--no-index", "--", "/dev/null", str(relative)],
            cwd=worktree,
            check=False,
            capture_output=True,
            text=True,
        )
        diff += addition.stdout
    patch_path = delivery / "repository.patch"
    patch_path.write_text(diff, encoding="utf-8")
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=worktree,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "base_sha": base,
        "files": file_records,
        "raw_patch": {
            "path": str(patch_path),
            "bytes": patch_path.stat().st_size,
            "sha256": _sha256(patch_path),
        },
    }


def run_study(output: Path, activation: Path, worktree: Path) -> None:
    output = output.resolve()
    activation = activation.resolve()
    worktree = worktree.resolve()
    if output.exists():
        raise FileExistsError(output)
    if not activation.is_file():
        raise FileNotFoundError(activation)
    free_initial = shutil.disk_usage(output.parent).free
    if free_initial < MIN_FREE_BYTES:
        raise RuntimeError("less than 10 GiB free before OpenFOAM study")
    output.mkdir(parents=True)
    environment = _activation_environment(activation)
    logs = output / "logs"
    logs.mkdir()
    utility_names = ("blockMesh", "checkMesh", "foamPostProcess", "foamToVTK")
    binaries = {}
    for name in utility_names:
        path_text = shutil.which(name, path=environment.get("PATH"))
        if path_text is None:
            raise RuntimeError(f"OpenFOAM utility absent: {name}")
        path = Path(path_text).resolve()
        binaries[name] = {
            "path": str(path),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }

    raw_levels = {}
    science_levels = {}
    resource_levels = {}
    for level in LEVELS:
        free_before = shutil.disk_usage(output).free
        if free_before < MIN_FREE_BYTES:
            raise RuntimeError(f"less than 10 GiB free before level {level}")
        case = output / f"case-n{level}"
        generated = generate_case(case, level)
        commands = []
        for index, argv in enumerate(_utility_commands(case)):
            label = f"n{level}-{index + 1}-{argv[0]}"
            commands.append(
                _run_utility(argv, environment=environment, log_dir=logs, label=label)
            )
        retained = _all_retained_run_bytes(output.parent)
        if retained >= MAX_RETAINED_BYTES:
            raise RuntimeError("cumulative retained OpenFOAM evidence reached 1 GiB")
        input_paths = [
            path
            for path in case.rglob("*")
            if path.is_file()
            and "VTK" not in path.parts
            and "polyMesh" not in path.parts
        ]
        vtk_paths = list((case / "VTK").rglob("*.vtk"))
        raw_levels[str(level)] = {
            "generated": generated,
            "commands": commands,
            "input_files": _hash_files(case, input_paths),
            "vtk_files": _hash_files(case, vtk_paths),
        }
        _write_json(
            output / f"raw-level-n{level}.json",
            {
                "classification": "observed",
                "level": level,
                "openfoam": {
                    "distribution": "OpenFOAM Foundation",
                    "version": "14",
                    "patch": "20260724",
                    "binaries": binaries,
                },
                **raw_levels[str(level)],
            },
        )
        analysis_started_at = _iso_now()
        analysis_started = time.monotonic()
        try:
            science_level = _analyse_level(case, level, generated)
        except Exception as error:
            first_vtk = case / "VTK" / f"{case.name}_0.vtk"
            science_level = {
                "status": "partial",
                "level": level,
                "analysis_started_at": analysis_started_at,
                "analysis_ended_at": _iso_now(),
                "analysis_duration_s": time.monotonic() - analysis_started,
                "analysis_timeout_s": ANALYSIS_TIMEOUT_S,
                "failure": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "stage": "existing OpenFOAM loader before gravity evaluation",
                },
                "geometry_diagnostic": _vtk_geometry_diagnostic(first_vtk),
            }
        else:
            science_level["analysis_started_at"] = analysis_started_at
            science_level["analysis_ended_at"] = _iso_now()
        science_levels[str(level)] = science_level
        _write_json(output / f"scientific-level-n{level}.json", science_level)
        resource_levels[str(level)] = {
            "free_bytes_before": free_before,
            "free_bytes_after": shutil.disk_usage(output).free,
            "retained_bytes_after": _tree_bytes(output),
            "cumulative_retained_run_bytes_after": _all_retained_run_bytes(
                output.parent
            ),
            "cell_count_per_snapshot": level**3,
            "snapshot_count": 3,
            "max_utility_peak_rss_kib": max(
                record["peak_rss_kib"] or 0 for record in commands
            ),
            "analysis_duration_s": science_levels[str(level)]["analysis_duration_s"],
            "analysis_peak_rss_kib_after": science_levels[str(level)].get(
                "process_peak_rss_kib_after"
            ),
        }
        _write_json(
            output / f"resource-level-n{level}.json", resource_levels[str(level)]
        )

    derivation = _scientific_derivation()
    maxima = {
        "mass": {},
        "potential": {},
        "acceleration": {},
        "gradient": {},
    }
    loaded_levels = [
        level for level in LEVELS if science_levels[str(level)]["status"] == "complete"
    ]
    for level in loaded_levels:
        level_data = science_levels[str(level)]["times"]
        maxima["mass"][str(level)] = max(
            level_data[format(t, "g")]["mass_relative_error"] for t in TIMES_S
        )
        for observable in ("potential", "acceleration", "gradient"):
            maxima[observable][str(level)] = max(
                max(level_data[format(t, "g")]["continuum_relative_errors"][observable])
                for t in TIMES_S
            )
    endpoints_loaded = {16, 48} <= set(loaded_levels)
    refinement_gates = {
        "mass_finest_le_1_percent": endpoints_loaded
        and maxima["mass"]["48"] <= 0.01,
        "mass_E48_lt_E16": endpoints_loaded
        and maxima["mass"]["48"] < maxima["mass"]["16"],
    }
    for observable in ("potential", "acceleration", "gradient"):
        refinement_gates[f"{observable}_finest_le_2_percent"] = (
            endpoints_loaded and maxima[observable]["48"] <= 0.02
        )
        refinement_gates[f"{observable}_E48_lt_E16"] = (
            endpoints_loaded
            and maxima[observable]["48"] < maxima[observable]["16"]
        )
    literal_passed = all(
        time_data["literal_checks"][observable]["passed"]
        for level_data in science_levels.values()
        if level_data["status"] == "complete"
        for time_data in level_data["times"].values()
        for observable in ("potential", "acceleration", "gradient")
    )
    profile_passed = all(
        level_data["successor_compatibility_passed"]
        and level_data["native_translation_passed"]
        and level_data["mass_time_consistency_passed"]
        for level_data in science_levels.values()
        if level_data["status"] == "complete"
    )
    translation_passed = all(
        time_data["translation_sign_passed"]
        for level_data in science_levels.values()
        if level_data["status"] == "complete"
        for time_data in level_data["times"].values()
    )
    scientific_gates = {
        **refinement_gates,
        "all_levels_loaded": len(loaded_levels) == len(LEVELS),
        "literal_discrete_oracle_loaded_levels_passed": literal_passed,
        "profile_successor_mass_translation_loaded_levels_passed": profile_passed,
        "analytic_translation_sign_loaded_levels_passed": translation_passed,
        "mass_CAS_passed": derivation["mass_exact_passed"],
        "mass_high_precision_quadrature_passed": derivation["quadrature_passed"],
        "advection_and_boundary_passed": derivation["advection_exact_passed"]
        and derivation["boundary_continuity_passed"],
    }

    vtk_writer_source = (
        activation.parent
        / "rootfs/opt/openfoam14/src/fileFormats/vtk/vtkWriteOps.C"
    )
    vtk_ascii_evidence = {
        "source_path": str(vtk_writer_source),
        "source_sha256": _sha256(vtk_writer_source),
        "observed_rule": (
            "ASCII branch inserts floatScalar values into std::ostream without "
            "setting precision; the C++ stream default is six significant digits"
        ),
        "modeled_conversion": (
            "nominal float64 -> OpenFOAM floatScalar(float32) -> six-significant-"
            "digit legacy ASCII -> loader float32"
        ),
    }
    loader_module = sys.modules["gragra.adapters.openfoam_density"]
    loader_geometry = loader_module._geometry
    loader_source = Path(inspect.getsourcefile(loader_geometry)).resolve()
    loader_lines, loader_start_line = inspect.getsourcelines(loader_geometry)
    loader_tolerance_evidence = {
        "source_path": str(loader_source),
        "source_sha256": _sha256(loader_source),
        "function": "_geometry",
        "start_line": loader_start_line,
        "end_line": loader_start_line + len(loader_lines) - 1,
        "fixed_formula": (
            "ctol=8*eps*max(abs(C),abs(cg))+32*eps*h; "
            "vtol=32*eps*max(abs(Vc),abs(vg))+64*eps*h^3"
        ),
    }

    raw_manifest = {
        "classification": "observed",
        "study": "prescribed translating cosine-bell density deficit",
        "cfd_solver_executed": False,
        "activation_script": {
            "path": str(activation),
            "sha256": _sha256(activation),
        },
        "openfoam": {
            "distribution": "OpenFOAM Foundation",
            "version": "14",
            "patch": "20260724",
            "binaries": binaries,
            "legacy_ascii_conversion_evidence": vtk_ascii_evidence,
            "loader_geometry_tolerance_evidence": loader_tolerance_evidence,
        },
        "levels": raw_levels,
    }
    scientific_manifest = {
        "classifications": {
            "utility_outputs_and_loaded_values": "observed",
            "continuum_formula": "inferred and CAS/high-precision verified",
            "fixed_acceptance_thresholds": (
                "imputed by the approved plan before execution"
            ),
        },
        "model": {
            "domain_m": [[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0]],
            "levels": list(LEVELS),
            "times_s": list(TIMES_S),
            "background_density_kg_m3": BACKGROUND_DENSITY_KG_M3,
            "anomaly_amplitude_kg_m3": ANOMALY_AMPLITUDE_KG_M3,
            "radius_m": RADIUS_M,
            "centre": "(-0.5 + 0.5*t, 0, 0) m",
            "targets_m": TARGETS_M.tolist(),
            "field": "delta_rho only; the uniform background gravity is excluded",
        },
        "derivation": derivation,
        "levels": science_levels,
        "max_errors": maxima,
        "gates": scientific_gates,
        "all_scientific_gates_passed": all(scientific_gates.values()),
    }
    resource_manifest = {
        "classification": "observed",
        "initial_free_bytes": free_initial,
        "required_free_bytes": MIN_FREE_BYTES,
        "retained_limit_bytes": MAX_RETAINED_BYTES,
        "utility_timeout_s": UTILITY_TIMEOUT_S,
        "analysis_timeout_s": ANALYSIS_TIMEOUT_S,
        "maximum_snapshots": 9,
        "maximum_cells_per_snapshot": 48**3,
        "levels": resource_levels,
        "final_retained_bytes": _tree_bytes(output),
        "cumulative_retained_run_bytes": _all_retained_run_bytes(output.parent),
        "final_free_bytes": shutil.disk_usage(output).free,
    }
    (output / "raw-manifest.json").write_text(
        json.dumps(raw_manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output / "scientific-manifest.json").write_text(
        json.dumps(scientific_manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output / "resource-manifest.json").write_text(
        json.dumps(resource_manifest, indent=2) + "\n", encoding="utf-8"
    )
    delivery = _capture_repository_delivery(output, worktree)
    evidence_manifest = {
        "classification": "observed unless a nested classification says otherwise",
        "started_from_base_sha": delivery["base_sha"],
        "source_origin": {
            "interpreter": sys.executable,
            "gragra": __import__("gragra").__file__,
            "openfoam_adapter": sys.modules[
                "gragra.adapters.openfoam_density"
            ].__file__,
            "study_script": str(Path(__file__).resolve()),
        },
        "repository_delivery": delivery,
        "manifests": {
            name: {"path": str(output / name), "sha256": _sha256(output / name)}
            for name in (
                "raw-manifest.json",
                "scientific-manifest.json",
                "resource-manifest.json",
            )
        },
        "utility_execution_passed": True,
        "scientific_gates_passed": scientific_manifest["all_scientific_gates_passed"],
        "gwexpy_bridge_status": "deferred_to_lane_I_after_G_and_O_acceptance",
        "overall_O_preintegration_status": (
            "complete"
            if scientific_manifest["all_scientific_gates_passed"]
            else "partial"
        ),
    }
    (output / "evidence-manifest.json").write_text(
        json.dumps(evidence_manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation-script", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--worktree", required=True, type=Path)
    args = parser.parse_args()
    run_study(args.output_dir, args.activation_script, args.worktree)


if __name__ == "__main__":
    main()
