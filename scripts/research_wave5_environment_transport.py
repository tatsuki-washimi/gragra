"""Run the approved real OpenFOAM 14 scalar-transport benchmark.

The density anomaly evolves only through the ``scalarTransport`` function
object while ``foamRun -solver incompressibleFluid`` advances a uniform,
laminar, periodic flow.  No later-time analytic density fields are written.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import resource
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
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
from scripts.research_openfoam_cosine_bell import literal_discrete_observables
from scripts.research_openfoam_cosine_bell_precision import (
    _assert_native_roundtrip,
    _parse_native_field,
    _parse_native_points,
    _read_ascii,
)

DOMAIN_MIN_M = -0.5
DOMAIN_MAX_M = 0.5
VELOCITY_M_S = np.array([0.2, 0.0, 0.0], dtype=np.float64)
INITIAL_CENTER_M = np.array([-0.25, 0.0, 0.0], dtype=np.float64)
RADIUS_M = 0.15
AMPLITUDE_KG_M3 = -0.01
NU_M2_S = 1e-5
TIMES_S = (0.0, 0.5, 1.0)
TARGETS_M = np.array(
    [[3.0, 0.7, -0.4], [-2.8, 1.3, 1.7], [0.25, -3.0, 2.6]],
    dtype=np.float64,
)


@dataclass(frozen=True)
class CaseSpec:
    level: int
    delta_t_s: float

    @property
    def name(self) -> str:
        return f"n{self.level}-dt{format(self.delta_t_s, 'g')}"


CASE_SPECS = (CaseSpec(24, 0.01), CaseSpec(48, 0.01), CaseSpec(48, 0.005))


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_ascii(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="ascii")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def nominal_cell_centres(level: int) -> np.ndarray:
    if (
        isinstance(level, bool)
        or not isinstance(level, (int, np.integer))
        or level <= 0
    ):
        raise ValueError("level must be a positive integer")
    width = (DOMAIN_MAX_M - DOMAIN_MIN_M) / int(level)
    axis = DOMAIN_MIN_M + (np.arange(level, dtype=np.float64) + 0.5) * width
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    result = np.column_stack(
        (x.ravel(order="F"), y.ravel(order="F"), z.ravel(order="F"))
    )
    result.setflags(write=False)
    return result


def analytic_total_mass_kg() -> float:
    """CAS-verified continuous mass of the configured spherical cosine bell."""
    return float(2.0 * AMPLITUDE_KG_M3 * RADIUS_M**3 * (np.pi**2 - 6.0) / (3.0 * np.pi))


def cosine_bell_density(points_m: np.ndarray, time_s: float) -> np.ndarray:
    points = np.asarray(points_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points_m must be finite with shape (N,3)")
    time_value = float(time_s)
    if not np.isfinite(time_value):
        raise ValueError("time_s must be finite")
    center = INITIAL_CENTER_M + time_value * VELOCITY_M_S
    delta = points - center
    length = DOMAIN_MAX_M - DOMAIN_MIN_M
    delta -= length * np.round(delta / length)
    radius = np.linalg.norm(delta, axis=1)
    result = np.zeros(len(points), dtype=np.float64)
    inside = radius < RADIUS_M
    result[inside] = (
        AMPLITUDE_KG_M3 / 2.0 * (1.0 + np.cos(np.pi * radius[inside] / RADIUS_M))
    )
    result.setflags(write=False)
    return result


def _boundary_entries() -> str:
    return "\n".join(
        f"    {name} {{ type cyclic; }}"
        for name in ("xMin", "xMax", "yMin", "yMax", "zMin", "zMax")
    )


def _field(name: str, dimensions: str, internal: str, field_class: str) -> str:
    return f"""FoamFile
{{
    version 2.0;
    format ascii;
    class {field_class};
    object {name};
}}
dimensions {dimensions};
internalField {internal};
boundaryField
{{
{_boundary_entries()}
}}
"""


def _density_field(values: np.ndarray) -> str:
    body = "\n".join(format(float(value), ".17g") for value in values)
    return _field(
        "delta_rho",
        "[1 -3 0 0 0 0 0]",
        f"nonuniform List<scalar> {len(values)}\n(\n{body}\n)",
        "volScalarField",
    )


def generate_case(case: Path, *, level: int, delta_t_s: float) -> dict[str, object]:
    """Create one solver case containing only the initial density field."""
    case = Path(case)
    if case.exists():
        raise FileExistsError(case)
    if level <= 0 or int(level) != level:
        raise ValueError("level must be a positive integer")
    delta_t = float(delta_t_s)
    if not np.isfinite(delta_t) or delta_t <= 0.0:
        raise ValueError("delta_t_s must be positive and finite")
    write_interval = 0.5 / delta_t
    if not math.isclose(write_interval, round(write_interval), abs_tol=1e-12):
        raise ValueError("delta_t_s must divide the 0.5 s output interval")
    centres = nominal_cell_centres(int(level))
    initial = cosine_bell_density(centres, 0.0)
    _write_ascii(
        case / "system/blockMeshDict",
        f"""FoamFile
{{ version 2.0; format ascii; class dictionary; object blockMeshDict; }}
scale 1;
vertices
(
    (-0.5 -0.5 -0.5) (0.5 -0.5 -0.5) (0.5 0.5 -0.5) (-0.5 0.5 -0.5)
    (-0.5 -0.5 0.5) (0.5 -0.5 0.5) (0.5 0.5 0.5) (-0.5 0.5 0.5)
);
blocks (hex (0 1 2 3 4 5 6 7) ({level} {level} {level}) simpleGrading (1 1 1));
edges ();
boundary
(
    xMin {{ type cyclic; neighbourPatch xMax; faces ((0 4 7 3)); }}
    xMax {{ type cyclic; neighbourPatch xMin; faces ((1 2 6 5)); }}
    yMin {{ type cyclic; neighbourPatch yMax; faces ((0 1 5 4)); }}
    yMax {{ type cyclic; neighbourPatch yMin; faces ((3 7 6 2)); }}
    zMin {{ type cyclic; neighbourPatch zMax; faces ((0 3 2 1)); }}
    zMax {{ type cyclic; neighbourPatch zMin; faces ((4 5 6 7)); }}
);
""",
    )
    _write_ascii(
        case / "system/controlDict",
        f"""FoamFile
{{ version 2.0; format ascii; class dictionary; object controlDict; }}
solver          incompressibleFluid;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1;
deltaT          {format(delta_t, ".17g")};
writeControl    timeStep;
writeInterval   {int(round(write_interval))};
purgeWrite      0;
writeFormat     ascii;
writePrecision  17;
writeCompression off;
timeFormat      general;
timePrecision   12;
runTimeModifiable false;
functions
{{
    #include "functions"
}}
""",
    )
    _write_ascii(
        case / "system/functions",
        """transportDensity
{
    type            scalarTransport;
    libs            ("libsolverFunctionObjects.so");
    field           delta_rho;
    phi             phi;
    schemesField    delta_rho;
    solverField     delta_rho;
    diffusivity     none;
    executeControl  timeStep;
    executeInterval 1;
    writeControl    writeTime;
}
""",
    )
    _write_ascii(
        case / "system/fvSchemes",
        """FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default         Euler; }
gradSchemes { default         Gauss linear; }
divSchemes
{
    default         none;
    div(phi,U)      Gauss linear;
    div(phi,delta_rho) Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
""",
    )
    _write_ascii(
        case / "system/fvSolution",
        """FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }
solvers
{
    p { solver GAMG; tolerance 1e-10; relTol 0; smoother GaussSeidel; }
    pFinal { $p; tolerance 1e-10; relTol 0; }
    U { solver smoothSolver; smoother GaussSeidel; tolerance 1e-10; relTol 0; }
    UFinal { $U; tolerance 1e-10; relTol 0; }
    delta_rho { solver PBiCGStab; preconditioner DILU; tolerance 1e-12; relTol 0; }
}
PIMPLE
{
    nCorrectors     1;
    nNonOrthogonalCorrectors 0;
    pRefCell        0;
    pRefValue       0;
}
""",
    )
    _write_ascii(
        case / "constant/physicalProperties",
        f"""FoamFile
{{ version 2.0; format ascii; class dictionary; object physicalProperties; }}
viscosityModel constant;
nu {NU_M2_S:.17g} [m^2/s];
""",
    )
    _write_ascii(
        case / "constant/momentumTransport",
        """FoamFile
{ version 2.0; format ascii; class dictionary; object momentumTransport; }
simulationType  laminar;
""",
    )
    _write_ascii(
        case / "0/U",
        _field("U", "[0 1 -1 0 0 0 0]", "uniform (0.2 0 0)", "volVectorField"),
    )
    _write_ascii(
        case / "0/p",
        _field("p", "[0 2 -2 0 0 0 0]", "uniform 0", "volScalarField"),
    )
    _write_ascii(case / "0/delta_rho", _density_field(initial))
    return {
        "level": int(level),
        "cell_count": int(level**3),
        "delta_t_s": delta_t,
        "outputs_s": list(TIMES_S),
        "velocity_m_s": VELOCITY_M_S.tolist(),
        "nu_m2_s": NU_M2_S,
        "initial_density": {
            "center_m": INITIAL_CENTER_M.tolist(),
            "radius_m": RADIUS_M,
            "amplitude_kg_m3": AMPLITUDE_KG_M3,
        },
        "future_density_fields_written": False,
    }


def summarize_density_difference(
    observed: np.ndarray, reference: np.ndarray, volumes_m3: np.ndarray
) -> dict[str, float]:
    observed_array = np.asarray(observed, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    volumes = np.asarray(volumes_m3, dtype=np.float64)
    if (
        observed_array.shape != reference_array.shape
        or observed_array.shape != volumes.shape
    ):
        raise ValueError("observed, reference, and volumes_m3 must have matched shape")
    if not all(
        np.isfinite(value).all() for value in (observed_array, reference_array, volumes)
    ):
        raise ValueError("density comparison inputs must be finite")
    if np.any(volumes <= 0.0):
        raise ValueError("volumes_m3 must be positive")
    difference = observed_array - reference_array
    return {
        "l1_kg": float(np.sum(np.abs(difference) * volumes)),
        "l2_kg_m_neg_3_2": float(np.sqrt(np.sum(difference**2 * volumes))),
        "linf_kg_m3": float(np.max(np.abs(difference), initial=0.0)),
    }


def periodic_center_of_mass(
    positions_m: np.ndarray, signed_mass_kg: np.ndarray
) -> np.ndarray:
    positions = np.asarray(positions_m, dtype=np.float64)
    mass = np.asarray(signed_mass_kg, dtype=np.float64)
    if (
        positions.ndim != 2
        or positions.shape[1] != 3
        or mass.shape != (len(positions),)
    ):
        raise ValueError(
            "positions_m and signed_mass_kg must have shapes (N,3) and (N,)"
        )
    if not np.isfinite(positions).all() or not np.isfinite(mass).all():
        raise ValueError("positions_m and signed_mass_kg must be finite")
    total_mass = float(np.sum(mass))
    if total_mass == 0.0:
        raise ValueError("signed total mass must be nonzero")
    length = DOMAIN_MAX_M - DOMAIN_MIN_M
    phase = 2.0 * np.pi * (positions - DOMAIN_MIN_M) / length
    sine = np.sum(mass[:, None] * np.sin(phase), axis=0) / total_mass
    cosine = np.sum(mass[:, None] * np.cos(phase), axis=0) / total_mass
    angle = np.mod(np.arctan2(sine, cosine), 2.0 * np.pi)
    center = DOMAIN_MIN_M + length * angle / (2.0 * np.pi)
    center.setflags(write=False)
    return center


def _case_commands(case: Path) -> list[list[str]]:
    case_text = str(Path(case))
    commands = [
        ["blockMesh", "-case", case_text],
        ["checkMesh", "-case", case_text],
        ["foamRun", "-solver", "incompressibleFluid", "-case", case_text],
        [
            "foamPostProcess",
            "-case",
            case_text,
            "-time",
            "0,0.5,1",
            "-func",
            "writeCellCentres",
        ],
        [
            "foamPostProcess",
            "-case",
            case_text,
            "-time",
            "0,0.5,1",
            "-func",
            "writeCellVolumes",
        ],
    ]
    for time_name in ("0", "0.5", "1"):
        commands.append(
            [
                "foamToVTK",
                "-case",
                case_text,
                "-time",
                time_name,
                "-useTimeName",
                "-ascii",
                "-noPointValues",
                "-polyhedra",
                "none",
                "-fields",
                "(delta_rho C Vc)",
            ]
        )
    return commands


def _activated_environment(activation_script: Path) -> dict[str, str]:
    result = subprocess.run(
        [
            "bash",
            "-lc",
            'source "$1" && env -0',
            "bash",
            str(activation_script),
        ],
        check=True,
        capture_output=True,
    )
    environment = {
        key.decode(): value.decode()
        for item in result.stdout.split(b"\0")
        if item
        for key, value in [item.split(b"=", 1)]
    }
    environment.update(
        {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
        }
    )
    return environment


def _peak_rss_kib(path: Path) -> int | None:
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if "Maximum resident set size (kbytes):" in line:
            return int(line.rsplit(":", 1)[1])
    return None


def _run_command(
    argv: list[str],
    *,
    environment: dict[str, str],
    logs: Path,
    label: str,
    private_file_formats: Path | None = None,
) -> dict[str, object]:
    command_environment = environment.copy()
    if private_file_formats is not None:
        existing = command_environment.get("LD_LIBRARY_PATH", "")
        private_dir = str(private_file_formats.parent)
        command_environment["LD_LIBRARY_PATH"] = (
            private_dir if not existing else f"{private_dir}:{existing}"
        )
    stdout_path = logs / f"{label}.stdout.log"
    stderr_path = logs / f"{label}.stderr.log"
    resource_path = logs / f"{label}.time-v.log"
    executed = ["/usr/bin/time", "-v", "-o", str(resource_path), *argv]
    started_at = _iso_now()
    started = time.monotonic()
    result = subprocess.run(
        executed,
        env=command_environment,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    record = {
        "argv": argv,
        "executed_argv": executed,
        "argv_shell": shlex.join(argv),
        "started_at": started_at,
        "ended_at": _iso_now(),
        "elapsed_s": time.monotonic() - started,
        "exit": result.returncode,
        "stdout": {
            "path": str(stdout_path),
            "sha256": _sha256(stdout_path),
            "bytes": stdout_path.stat().st_size,
        },
        "stderr": {
            "path": str(stderr_path),
            "sha256": _sha256(stderr_path),
            "bytes": stderr_path.stat().st_size,
        },
        "resource": {
            "path": str(resource_path),
            "sha256": _sha256(resource_path),
            "bytes": resource_path.stat().st_size,
            "peak_rss_kib": _peak_rss_kib(resource_path),
            "limit_bytes": 2 * 1024**3,
        },
        "private_file_formats": (
            {
                "path": str(private_file_formats),
                "sha256": _sha256(private_file_formats),
            }
            if private_file_formats is not None
            else None
        ),
    }
    _write_json(logs / f"{label}.argv.json", record)
    if result.returncode != 0:
        raise RuntimeError(json.dumps(record, indent=2))
    peak = record["resource"]["peak_rss_kib"]
    if peak is not None and peak * 1024 > 2 * 1024**3:
        raise RuntimeError(f"{label} exceeded the 2 GiB RSS bound")
    return record


def _hash_tree(root: Path, paths: list[Path]) -> dict[str, dict[str, object]]:
    return {
        str(path.relative_to(root)): {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(paths)
        if path.is_file()
    }


def _parse_native_scalar_field(
    path: Path,
    *,
    object_name: str,
    expected_count: int,
    expected_dimensions: tuple[int, ...],
) -> np.ndarray:
    """Reuse the O48 parser and additionally accept OpenFOAM uniform scalars."""
    try:
        return _parse_native_field(
            path,
            object_name=object_name,
            components=1,
            expected_count=expected_count,
            expected_dimensions=expected_dimensions,
        ).values
    except ValueError as error:
        if "missing bounded internalField list" not in str(error):
            raise
    text = _read_ascii(Path(path))
    match = re.search(r"\binternalField\s+uniform\s+([^;]+);", text)
    if match is None:
        raise ValueError(f"{path}: missing scalar internalField")
    try:
        value = float(match.group(1).strip())
    except ValueError as error:
        raise ValueError(f"{path}: invalid uniform scalar") from error
    if not np.isfinite(value):
        raise ValueError(f"{path}: uniform scalar must be finite")
    return np.full(expected_count, value, dtype=np.float64)


def _norm_per_target(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 1:
        return np.abs(array)
    return np.linalg.norm(array, axis=tuple(range(1, array.ndim)))


def _observable_errors(
    observed: dict[str, np.ndarray], reference: dict[str, np.ndarray]
) -> dict[str, object]:
    result = {}
    for name in ("potential", "acceleration", "gradient"):
        difference = _norm_per_target(observed[name] - reference[name])
        scale = _norm_per_target(reference[name])
        result[name] = {
            "absolute_norm": difference.tolist(),
            "reference_norm": scale.tolist(),
            "normalized": (
                difference / np.maximum(scale, np.finfo(float).tiny)
            ).tolist(),
        }
    return result


def _stage_observables(
    positions: np.ndarray, density: np.ndarray, volumes: np.ndarray
) -> dict[str, np.ndarray]:
    return literal_discrete_observables(positions, density * volumes, TARGETS_M)


def _load_a_reference(path: Path, expected_sha256: str):
    observed_sha256 = _sha256(path)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"A reference source hash mismatch: {observed_sha256} != {expected_sha256}"
        )
    specification = importlib.util.spec_from_file_location(
        "gragra_lane_a_transport_reference", path
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load accepted A reference from {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _production_observables(snapshot) -> dict[str, np.ndarray]:
    source = snapshot.as_volume_element_source().as_weighted_source()
    return {
        "potential": point_potential(source, TARGETS_M),
        "acceleration": point_acceleration(source, TARGETS_M),
        "gradient": point_gravity_gradient(source, TARGETS_M),
    }


def _analyze_case(
    case: Path,
    spec: CaseSpec,
    commands: list[dict[str, object]],
    a_reference,
) -> dict[str, object]:
    count = spec.level**3
    points_native = _parse_native_points(
        case / "constant/polyMesh/points", expected_count=(spec.level + 1) ** 3
    ).values
    time_results = {}
    native_density_by_time = {}
    for time_s in TIMES_S:
        time_name = format(time_s, "g")
        native_centres = _parse_native_field(
            case / time_name / "C",
            object_name="C",
            components=3,
            expected_count=count,
            expected_dimensions=(0, 1, 0, 0, 0, 0, 0),
        ).values
        native_volumes = _parse_native_scalar_field(
            case / time_name / "Vc",
            object_name="Vc",
            expected_count=count,
            expected_dimensions=(0, 3, 0, 0, 0, 0, 0),
        )
        native_density = _parse_native_scalar_field(
            case / time_name / "delta_rho",
            object_name="delta_rho",
            expected_count=count,
            expected_dimensions=(1, -3, 0, 0, 0, 0, 0),
        )
        native_density_by_time[time_name] = native_density
        analytic_density = cosine_bell_density(native_centres, time_s)
        export_record = commands[5 + int(time_s * 2)]
        export_argv = export_record["argv"]
        vtk_path = case / "VTK" / f"{case.name}_{time_name}.vtk"
        provenance = {
            "distribution": "OpenFOAM Foundation",
            "version": "14",
            "patch": "20260724",
            "solver": "incompressibleFluid+scalarTransport-function-object",
            "compressibility": "incompressible",
            "region": "region0",
            "reconstructed": True,
            "exporter": "foamToVTK",
            "export_command": shlex.join(export_argv),
        }
        snapshot = load_openfoam_density(
            vtk_path,
            field_name="delta_rho",
            time_s=time_s,
            time_name=time_name,
            coordinate_frame="openfoam-periodic-cube",
            coordinate_origin_m=np.zeros(3),
            density_unit="kg/m^3",
            coordinate_unit="m",
            volume_unit="m^3",
            time_unit="s",
            provenance=provenance,
        )
        roundtrip = _assert_native_roundtrip(
            snapshot,
            native_points=points_native,
            native_centres=native_centres,
            native_volumes=native_volumes,
            native_density=native_density,
            expected_count=count,
        )
        native_mass = native_density * native_volumes
        analytic_mass = analytic_density * native_volumes
        exported_mass = snapshot.density_kg_m3 * snapshot.volumes_m3
        native_observables = _stage_observables(
            native_centres, native_density, native_volumes
        )
        analytic_observables = _stage_observables(
            native_centres, analytic_density, native_volumes
        )
        continuous_g_free = a_reference.point_mass_observables_g_free(
            (INITIAL_CENTER_M + time_s * VELOCITY_M_S)[None, :],
            np.array((analytic_total_mass_kg(),)),
            TARGETS_M,
        )
        continuous_observables = {
            name: G_SI * getattr(continuous_g_free, name)
            for name in ("potential", "acceleration", "gradient")
        }
        exported_literal = _stage_observables(
            snapshot.positions_m, snapshot.density_kg_m3, snapshot.volumes_m3
        )
        production = _production_observables(snapshot)
        time_results[time_name] = {
            "classification": (
                "observed solver/export values; analytic profile inferred"
            ),
            "vtk": {
                "path": str(vtk_path),
                "bytes": vtk_path.stat().st_size,
                "sha256": _sha256(vtk_path),
                "payload": "float32 legacy VTK ASCII with 9 significant digits",
            },
            "native": {
                "density_path": str(case / time_name / "delta_rho"),
                "density_sha256": _sha256(case / time_name / "delta_rho"),
                "mass_kg": float(np.sum(native_mass)),
                "center_of_mass_m": periodic_center_of_mass(
                    native_centres, native_mass
                ).tolist(),
                "observables": {
                    name: value.tolist() for name, value in native_observables.items()
                },
            },
            "analytic_discrete": {
                "mass_kg": float(np.sum(analytic_mass)),
                "center_of_mass_m": periodic_center_of_mass(
                    native_centres, analytic_mass
                ).tolist(),
                "observables": {
                    name: value.tolist() for name, value in analytic_observables.items()
                },
            },
            "continuous_exterior_point_reference": {
                "scope": (
                    "exact continuous translated cosine-bell profile at exterior "
                    "targets by spherical shell symmetry; not the evolved solver shape"
                ),
                "center_m": (INITIAL_CENTER_M + time_s * VELOCITY_M_S).tolist(),
                "radius_m": RADIUS_M,
                "signed_total_mass_kg": analytic_total_mass_kg(),
                "targets_are_exterior": bool(
                    np.all(
                        np.linalg.norm(
                            TARGETS_M
                            - (INITIAL_CENTER_M + time_s * VELOCITY_M_S)[None, :],
                            axis=1,
                        )
                        > RADIUS_M
                    )
                ),
                "observables": {
                    name: value.tolist()
                    for name, value in continuous_observables.items()
                },
            },
            "exported": {
                "mass_kg": float(np.sum(exported_mass)),
                "center_of_mass_m": periodic_center_of_mass(
                    snapshot.positions_m, exported_mass
                ).tolist(),
                "observables_literal": {
                    name: value.tolist() for name, value in exported_literal.items()
                },
                "observables_production": {
                    name: value.tolist() for name, value in production.items()
                },
            },
            "transport_error": {
                "density": summarize_density_difference(
                    native_density, analytic_density, native_volumes
                ),
                "mass_absolute_kg": float(
                    abs(np.sum(native_mass) - np.sum(analytic_mass))
                ),
                "observables": _observable_errors(
                    native_observables, analytic_observables
                ),
                "mass_vs_continuous_absolute_kg": float(
                    abs(np.sum(native_mass) - analytic_total_mass_kg())
                ),
                "observables_vs_continuous_exterior": _observable_errors(
                    native_observables, continuous_observables
                ),
            },
            "export_error": {
                "density": summarize_density_difference(
                    snapshot.density_kg_m3, native_density, native_volumes
                ),
                "mass_absolute_kg": float(
                    abs(np.sum(exported_mass) - np.sum(native_mass))
                ),
                "observables": _observable_errors(exported_literal, native_observables),
                "native_float32_roundtrip": roundtrip,
            },
            "integration_error": {
                "observables": _observable_errors(production, exported_literal),
                "comparison": (
                    "production observable versus literal discrete Newton sum"
                ),
            },
        }
    initial = native_density_by_time["0"]
    evolved = {
        time_name: {
            "different_from_initial": not np.array_equal(values, initial),
            "different_from_analytic_reinjection": not np.array_equal(
                values,
                cosine_bell_density(nominal_cell_centres(spec.level), float(time_name)),
            ),
        }
        for time_name, values in native_density_by_time.items()
        if time_name != "0"
    }
    return {
        "status": (
            "complete"
            if all(
                value["different_from_initial"]
                and value["different_from_analytic_reinjection"]
                for value in evolved.values()
            )
            else "partial"
        ),
        "case": spec.name,
        "level": spec.level,
        "delta_t_s": spec.delta_t_s,
        "actual_evolution": evolved,
        "times": time_results,
    }


def run_study(
    output: Path,
    activation_script: Path,
    private_file_formats: Path,
    worktree: Path,
    a_reference_module: Path,
    a_reference_sha256: str,
    a_acceptance: Path,
    cas_assumptions: Path,
    cas_check: Path,
    cas_result: Path,
) -> None:
    output = Path(output).resolve()
    activation_script = Path(activation_script).resolve()
    private_file_formats = Path(private_file_formats).resolve()
    worktree = Path(worktree).resolve()
    a_reference_module = Path(a_reference_module).resolve()
    a_acceptance = Path(a_acceptance).resolve()
    cas_assumptions = Path(cas_assumptions).resolve()
    cas_check = Path(cas_check).resolve()
    cas_result = Path(cas_result).resolve()
    if output.exists():
        raise FileExistsError(output)
    if not activation_script.is_file() or not private_file_formats.is_file():
        raise FileNotFoundError(
            "OpenFOAM activation or private fileFormats library absent"
        )
    for dependency in (
        a_reference_module,
        a_acceptance,
        cas_assumptions,
        cas_check,
        cas_result,
    ):
        if not dependency.is_file():
            raise FileNotFoundError(dependency)
    a_acceptance_data = json.loads(a_acceptance.read_text(encoding="utf-8"))
    if "ACCEPT" not in str(a_acceptance_data.get("status", "")).upper():
        raise ValueError("A parent acceptance is not accepted")
    cas_result_data = json.loads(cas_result.read_text(encoding="utf-8"))
    if cas_result_data.get("exit_code") != 0:
        raise ValueError("continuous cosine-bell CAS verification did not exit zero")
    if cas_result_data.get("check_sha256") != _sha256(cas_check):
        raise ValueError("continuous cosine-bell CAS check hash mismatch")
    cas_assumption_data = json.loads(cas_assumptions.read_text(encoding="utf-8"))
    expected_profile = "A/2*(1+cos(pi*r/R)) inside sphere, zero outside"
    if cas_assumption_data.get("profile") != expected_profile:
        raise ValueError(
            "continuous cosine-bell CAS profile does not match the solver input"
        )
    if not np.isclose(
        cas_result_data.get("mass_kg_at_proposed_transport_conditions"),
        analytic_total_mass_kg(),
        rtol=0.0,
        atol=1e-20,
    ):
        raise ValueError(
            "continuous cosine-bell CAS mass differs from configured profile"
        )
    a_reference = _load_a_reference(a_reference_module, a_reference_sha256)
    output.mkdir(parents=True)
    logs = output / "logs"
    logs.mkdir()
    environment = _activated_environment(activation_script)
    started_at = _iso_now()
    run_started = time.monotonic()
    executed_script = Path(__file__).resolve()
    executed_script_sha256 = _sha256(executed_script)
    binary_names = (
        "blockMesh",
        "checkMesh",
        "foamRun",
        "foamPostProcess",
        "foamToVTK",
    )
    binaries = {}
    for name in binary_names:
        resolved = shutil.which(name, path=environment.get("PATH"))
        if resolved is None:
            raise RuntimeError(f"OpenFOAM executable unavailable: {name}")
        path = Path(resolved).resolve()
        binaries[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    cases = {}
    for spec in CASE_SPECS:
        case = output / spec.name
        generated = generate_case(case, level=spec.level, delta_t_s=spec.delta_t_s)
        initial_inputs = [
            path
            for path in case.rglob("*")
            if path.is_file() and "polyMesh" not in path.parts
        ]
        input_hashes = _hash_tree(case, initial_inputs)
        command_records = []
        for index, argv in enumerate(_case_commands(case)):
            label = f"{spec.name}-{index + 1}-{argv[0]}"
            command_records.append(
                _run_command(
                    argv,
                    environment=environment,
                    logs=logs,
                    label=label,
                    private_file_formats=(
                        private_file_formats if argv[0] == "foamToVTK" else None
                    ),
                )
            )
        science = _analyze_case(case, spec, command_records, a_reference)
        raw_outputs = [path for path in case.rglob("*") if path.is_file()]
        case_record = {
            "generated": generated,
            "input_hashes_before_execution": input_hashes,
            "commands": command_records,
            "science": science,
            "raw_output_hashes": _hash_tree(case, raw_outputs),
        }
        cases[spec.name] = case_record
        _write_json(output / f"{spec.name}-result.json", case_record)
    comparisons = {
        "spatial_n24_to_n48_dt0.01": {
            "coarse": "n24-dt0.01",
            "fine": "n48-dt0.01",
        },
        "temporal_n48_dt0.01_to_dt0.005": {
            "coarse": "n48-dt0.01",
            "fine": "n48-dt0.005",
        },
    }
    manifest = {
        "status": (
            "complete"
            if all(
                record["science"]["status"] == "complete" for record in cases.values()
            )
            else "partial"
        ),
        "classification": "observed external solver and export execution",
        "started_at": started_at,
        "ended_at": _iso_now(),
        "worktree": str(worktree),
        "source": {
            "gragra": __import__("gragra").__file__,
            "script": str(executed_script),
            "script_sha256_at_start": executed_script_sha256,
            "script_sha256_at_end": _sha256(executed_script),
            "script_unchanged_during_run": (
                _sha256(executed_script) == executed_script_sha256
            ),
        },
        "software": {
            "python": os.sys.version,
            "numpy": np.__version__,
            "binaries": binaries,
            "private_file_formats": {
                "path": str(private_file_formats),
                "bytes": private_file_formats.stat().st_size,
                "sha256": _sha256(private_file_formats),
            },
        },
        "dependencies": {
            "A_parent_acceptance": {
                "path": str(a_acceptance),
                "sha256": _sha256(a_acceptance),
            },
            "A_literal_reference_module": {
                "path": str(a_reference_module),
                "sha256": _sha256(a_reference_module),
            },
            "continuous_cosine_bell_CAS": {
                "assumptions": {
                    "path": str(cas_assumptions),
                    "sha256": _sha256(cas_assumptions),
                },
                "check": {"path": str(cas_check), "sha256": _sha256(cas_check)},
                "result": {
                    "path": str(cas_result),
                    "sha256": _sha256(cas_result),
                    "verification_exit": cas_result_data["exit_code"],
                },
            },
        },
        "resource": {
            "single_process": True,
            "rss_limit_bytes": 2 * 1024**3,
            "analysis_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "thread_limits": {
                name: environment[name]
                for name in (
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "NUMBA_NUM_THREADS",
                )
            },
        },
        "execution": {
            "argv": list(sys.argv),
            "argv_shell": shlex.join(sys.argv),
            "elapsed_s": time.monotonic() - run_started,
        },
        "method": {
            "solver": "foamRun -solver incompressibleFluid",
            "transport": "scalarTransport function object",
            "diffusivity": "none",
            "density_reinjection": False,
            "export": "stock foamToVTK with accepted private libfileFormats.so",
            "export_payload": "float32 ASCII, 9 significant digits",
            "claim_boundary": "scalar transport only; not acoustic or compressible CFD",
        },
        "comparisons": comparisons,
        "cases": {
            name: {
                "result_path": str(output / f"{name}-result.json"),
                "result_sha256": _sha256(output / f"{name}-result.json"),
                "status": record["science"]["status"],
            }
            for name, record in cases.items()
        },
    }
    _write_json(output / "transport-manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--activation-script", type=Path, required=True)
    parser.add_argument("--private-file-formats", type=Path, required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--a-reference-module", type=Path, required=True)
    parser.add_argument("--a-reference-sha256", required=True)
    parser.add_argument("--a-acceptance", type=Path, required=True)
    parser.add_argument("--cas-assumptions", type=Path, required=True)
    parser.add_argument("--cas-check", type=Path, required=True)
    parser.add_argument("--cas-result", type=Path, required=True)
    args = parser.parse_args()
    run_study(
        args.output_dir,
        args.activation_script,
        args.private_file_formats,
        args.worktree,
        args.a_reference_module,
        args.a_reference_sha256,
        args.a_acceptance,
        args.cas_assumptions,
        args.cas_check,
        args.cas_result,
    )


if __name__ == "__main__":
    main()
