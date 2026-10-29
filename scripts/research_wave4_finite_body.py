#!/usr/bin/env python3
"""Wave 4 finite-body mesh, quadrature, and representation study.

All dimensions in this script are Proposed benchmark values.  Kernel calls are
G-free; :func:`to_physical_observables` is the only G multiplication boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import resource
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

import gragra
from gragra.constants import G_SI
from gragra.fields.finite_body_p1 import (
    _TETRA_BARYCENTRIC,
    compare_tetra_p1_density_representations,
    piecewise_tetra_p1_density_sources,
)
from gragra.fields.unstructured import UnstructuredDisplacementField
from gragra.gravity_sources import WeightedPointSource
from gragra.integrators import direct_point_sum

HOST_BOUNDS_M = np.array(((-20.0, 20.0), (-16.0, 16.0), (-24.0, 0.0)))
INNER_BOUNDS_M = np.array(((-4.0, 8.0), (-4.0, 4.0), (-16.0, -8.0)))
HOST_DENSITY_KG_M3 = 2300.0
INCLUSION_DENSITY_KG_M3 = 3200.0
OUTSIDE_TARGETS_M = np.array(
    ((20.5, 3.0, -14.0), (21.0, 3.0, -14.0), (22.0, 3.0, -14.0), (24.0, 3.0, -14.0))
)
CAVITY_TARGETS_M = np.array(
    (
        (2.31, -0.27, -11.81),
        (7.5, 0.37, -12.23),
        (7.0, -0.41, -11.71),
        (6.0, 0.33, -11.79),
    )
)
AFFINE_CENTER_M = np.array((2.0, 0.0, -12.0))
AFFINE_B_M = np.array((0.40, -0.25, 0.15))
AFFINE_A = np.array(
    ((0.010, 0.004, -0.003), (-0.002, -0.006, 0.001), (0.003, -0.002, 0.008))
)
_HEX_TETRA = np.array(
    (
        (0, 1, 2, 6),
        (0, 2, 3, 6),
        (0, 3, 7, 6),
        (0, 7, 4, 6),
        (0, 4, 5, 6),
        (0, 5, 1, 6),
    ),
    dtype=np.int64,
)
OBSERVABLES = ("potential", "acceleration", "gradient")
OBSERVABLE_UNITS = {
    "potential": "m^2 s^-2",
    "acceleration": "m s^-2",
    "gradient": "s^-2",
}


def _readonly(value, *, dtype=None) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ObservableTriple:
    potential: np.ndarray
    acceleration: np.ndarray
    gradient: np.ndarray

    def __post_init__(self) -> None:
        raw = tuple(np.asarray(value) for value in self)
        if (
            raw[0].ndim != 1
            or raw[1].shape != (len(raw[0]), 3)
            or raw[2].shape != (len(raw[0]), 3, 3)
        ):
            raise ValueError(
                "observables must use target-leading trailing-component shapes"
            )
        dtype = (
            np.complex128
            if any(np.iscomplexobj(value) for value in raw)
            else np.float64
        )
        for name, value in zip(OBSERVABLES, raw, strict=True):
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or Inf")
            object.__setattr__(self, name, _readonly(value, dtype=dtype))

    def __iter__(self):
        return iter((self.potential, self.acceleration, self.gradient))


@dataclass(frozen=True)
class CartesianTetraMesh:
    vertices_m: np.ndarray
    tetra: np.ndarray
    cell_density_kg_m3: np.ndarray
    tetra_volume_m3: np.ndarray
    nominal_h_m: float
    axis_cell_counts: tuple[int, int, int]
    axis_max_cell_width_m: np.ndarray
    host_volume_m3: float
    inner_volume_m3: float
    material_volume_m3: float
    material_mass_kg: float

    def __post_init__(self) -> None:
        vertices = _readonly(self.vertices_m, dtype=np.float64)
        tetra = _readonly(self.tetra, dtype=np.int64)
        density = _readonly(self.cell_density_kg_m3, dtype=np.float64)
        volumes = _readonly(self.tetra_volume_m3, dtype=np.float64)
        max_width = _readonly(self.axis_max_cell_width_m, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1:] != (3,):
            raise ValueError("vertices_m must have shape (N,3)")
        if tetra.ndim != 2 or tetra.shape[1:] != (4,):
            raise ValueError("tetra must have shape (K,4)")
        if density.shape != (len(tetra),) or volumes.shape != (len(tetra),):
            raise ValueError("tetra density and volume arrays must have shape (K,)")
        if len(self.axis_cell_counts) != 3 or max_width.shape != (3,):
            raise ValueError("axis mesh metadata must have three entries")
        if np.any(density < 0.0) or np.any(volumes <= 0.0):
            raise ValueError("density must be non-negative and volume positive")
        for name, value in (
            ("vertices_m", vertices),
            ("tetra", tetra),
            ("cell_density_kg_m3", density),
            ("tetra_volume_m3", volumes),
            ("axis_max_cell_width_m", max_width),
        ):
            object.__setattr__(self, name, value)


def _axis_edges(bounds: np.ndarray, h_m: float, inner: np.ndarray | None) -> np.ndarray:
    lo, hi = map(float, bounds)
    count = int(np.ceil((hi - lo) / h_m))
    regular = lo + np.arange(count + 1, dtype=np.float64) * h_m
    regular[-1] = hi
    values = (
        regular
        if inner is None
        else np.concatenate((regular, np.asarray(inner, dtype=np.float64)))
    )
    values = np.unique(values[(values >= lo) & (values <= hi)])
    if values[0] != lo or values[-1] != hi or np.any(np.diff(values) <= 0.0):
        raise ValueError("invalid Cartesian axis partition")
    return values


def build_cartesian_six_tetra_mesh(
    bounds_m,
    h_m: float,
    *,
    inner_bounds_m=None,
    host_density_kg_m3: float = HOST_DENSITY_KG_M3,
    inner_density_kg_m3: float | None = None,
) -> CartesianTetraMesh:
    """Build a conforming Cartesian mesh with six tetrahedra per cell.

    Material boundaries are inserted as axis breakpoints.  Thus extent sweeps
    keep the cavity exact even when an outer bound is not aligned to nominal h.
    """
    bounds = np.asarray(bounds_m, dtype=np.float64)
    if (
        bounds.shape != (3, 2)
        or not np.isfinite(bounds).all()
        or np.any(bounds[:, 1] <= bounds[:, 0])
    ):
        raise ValueError("bounds_m must be finite shape (3,2) with lower < upper")
    h = float(h_m)
    host_rho = float(host_density_kg_m3)
    if not np.isfinite(h) or h <= 0.0 or not np.isfinite(host_rho) or host_rho < 0.0:
        raise ValueError("h_m must be positive and host density non-negative")
    inner = (
        None if inner_bounds_m is None else np.asarray(inner_bounds_m, dtype=np.float64)
    )
    if inner is not None:
        if (
            inner.shape != (3, 2)
            or np.any(inner[:, 0] < bounds[:, 0])
            or np.any(inner[:, 1] > bounds[:, 1])
        ):
            raise ValueError("inner_bounds_m must lie inside bounds_m")
        if inner_density_kg_m3 is None:
            raise ValueError("inner_density_kg_m3 is required with inner_bounds_m")
        inner_rho = float(inner_density_kg_m3)
        if not np.isfinite(inner_rho) or inner_rho < 0.0:
            raise ValueError("inner density must be finite and non-negative")
    else:
        inner_rho = host_rho

    axes = [
        _axis_edges(bounds[axis], h, None if inner is None else inner[axis])
        for axis in range(3)
    ]
    vertices = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    shape = tuple(len(axis) - 1 for axis in axes)
    i, j, k = (item.reshape(-1) for item in np.indices(shape))
    nyv, nzv = len(axes[1]), len(axes[2])

    def index(ii, jj, kk):
        return (ii * nyv + jj) * nzv + kk

    corners = np.stack(
        (
            index(i, j, k),
            index(i + 1, j, k),
            index(i + 1, j + 1, k),
            index(i, j + 1, k),
            index(i, j, k + 1),
            index(i + 1, j, k + 1),
            index(i + 1, j + 1, k + 1),
            index(i, j + 1, k + 1),
        ),
        axis=1,
    )
    tetra = corners[:, _HEX_TETRA].reshape(-1, 4)
    dx = np.diff(axes[0])[i]
    dy = np.diff(axes[1])[j]
    dz = np.diff(axes[2])[k]
    cube_volume = dx * dy * dz
    centers = np.stack(
        (
            (axes[0][i] + axes[0][i + 1]) / 2,
            (axes[1][j] + axes[1][j + 1]) / 2,
            (axes[2][k] + axes[2][k + 1]) / 2,
        ),
        axis=1,
    )
    inside = np.zeros(len(centers), dtype=bool)
    if inner is not None:
        inside = np.all((centers >= inner[:, 0]) & (centers <= inner[:, 1]), axis=1)
    cube_density = np.where(inside, inner_rho, host_rho)
    tetra_density = np.repeat(cube_density, 6)
    tetra_volume = np.repeat(cube_volume / 6.0, 6)
    host_volume = float(np.prod(bounds[:, 1] - bounds[:, 0]))
    inner_volume = 0.0 if inner is None else float(np.prod(inner[:, 1] - inner[:, 0]))
    material_volume = float(np.sum(tetra_volume[tetra_density != 0.0]))
    material_mass = float(np.sum(tetra_density * tetra_volume))
    return CartesianTetraMesh(
        vertices,
        tetra,
        tetra_density,
        tetra_volume,
        h,
        shape,
        np.array([np.max(np.diff(axis)) for axis in axes]),
        host_volume,
        inner_volume,
        material_volume,
        material_mass,
    )


def degree2_mass_quadrature(mesh: CartesianTetraMesh) -> tuple[np.ndarray, np.ndarray]:
    """Use the existing immutable production degree-2 tetra rule."""
    nodes = mesh.vertices_m[mesh.tetra]
    positions = np.einsum("ql,nli->nqi", _TETRA_BARYCENTRIC, nodes).reshape(-1, 3)
    weights = np.repeat(mesh.cell_density_kg_m3 * mesh.tetra_volume_m3 / 4.0, 4)
    return _readonly(positions, dtype=np.float64), _readonly(weights, dtype=np.float64)


def evaluate_static_observables_g_free(
    positions_m, weights_kg, targets_m, *, chunk_size: int = 8192
) -> ObservableTriple:
    source = WeightedPointSource(positions_m, weights_kg)
    values = [
        direct_point_sum(
            source, targets_m, observable=name, chunk_size=chunk_size, backend="numpy"
        )
        for name in OBSERVABLES
    ]
    return ObservableTriple(*values)


def to_physical_observables(value: ObservableTriple) -> ObservableTriple:
    """Apply G exactly once at the observable boundary."""
    return ObservableTriple(*(G_SI * component for component in value))


def response_norm(value: np.ndarray, observable: str) -> np.ndarray:
    if observable == "potential":
        return np.abs(value)
    axes = (-1,) if observable == "acceleration" else (-2, -1)
    return np.linalg.norm(value, axis=axes)


def normalized_error(
    value: np.ndarray, reference: np.ndarray, observable: str
) -> np.ndarray:
    numerator = response_norm(np.asarray(value) - np.asarray(reference), observable)
    denominator = response_norm(np.asarray(reference), observable)
    scale = max(float(np.max(denominator)), np.finfo(np.float64).tiny)
    return numerator / np.maximum(denominator, np.finfo(np.float64).eps * scale)


def a_oracle_status(convergence: dict, *, parent_accepted: bool) -> str:
    """Classify an A oracle without weakening its frozen numerical contract."""
    if (
        convergence["status"] != "CONVERGED"
        or float(convergence["normalized_difference"]) > 1e-11
    ):
        return "ORACLE_UNRESOLVED"
    return "CONVERGED_ACCEPTED" if parent_accepted else "CONVERGED_PROVISIONAL"


def affine_displacement(points_m: np.ndarray) -> np.ndarray:
    return AFFINE_B_M + (np.asarray(points_m) - AFFINE_CENTER_M) @ AFFINE_A.T


def _array_record(value: np.ndarray) -> dict:
    array = np.asarray(value)
    if np.iscomplexobj(array):
        return {
            "encoding": "real_imag_last_axis",
            "dtype": "complex128",
            "shape": list(array.shape),
            "values": np.stack((array.real, array.imag), axis=-1).tolist(),
        }
    return {
        "encoding": "plain",
        "dtype": "float64",
        "shape": list(array.shape),
        "values": array.tolist(),
    }


def _observable_record(value: ObservableTriple) -> dict:
    return {
        name: {"unit": OBSERVABLE_UNITS[name], **_array_record(getattr(value, name))}
        for name in OBSERVABLES
    }


def _reference_array(record: dict, name: str) -> np.ndarray:
    return np.asarray(record["reference"][name], dtype=np.float64)


def _versions() -> dict:
    result = {"python": sys.version.split()[0], "numpy": np.__version__}
    for name in ("numba", "h5py", "scipy", "mpmath"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def run_wave4(
    output_dir: Path,
    oracle_json: Path,
    *,
    h_values=(4.0, 2.0, 1.0),
    a_parent_accepted_sha256: str | None = None,
) -> Path:
    root = Path(__file__).resolve().parents[1]
    if not Path(gragra.__file__).resolve().is_relative_to((root / "src").resolve()):
        raise RuntimeError(f"gragra imported from wrong tree: {gragra.__file__}")
    oracle_bytes = oracle_json.read_bytes()
    oracle = json.loads(oracle_bytes)
    finite = oracle["finite_bodies"]
    started = datetime.now(UTC)
    oracle_sha256 = hashlib.sha256(oracle_bytes).hexdigest()
    parent_accepted = a_parent_accepted_sha256 == oracle_sha256
    result = {
        "schema_version": 1,
        "classification": {
            "dimensions": "Proposed",
            "computed_curves": "Observed",
            "interpretations": "Inferred",
            "imputed_values": [],
        },
        "scope": (
            "finite prescribed Newton fields; no elastic scattering or "
            "infinite-ground claim"
        ),
        "g_boundary": (
            "G-free kernels; G_SI multiplied exactly once in recorded "
            "physical observables"
        ),
        "oracle_source": {
            "path": str(oracle_json),
            "sha256": oracle_sha256,
            "parent_acceptance": "accepted_exact_bytes"
            if parent_accepted
            else "PROVISIONAL_PARENT_UNACCEPTED",
        },
        "versions": _versions(),
        "cases": [],
    }
    for case_name, inner_density, targets in (
        ("cavity", 0.0, CAVITY_TARGETS_M),
        ("inclusion", INCLUSION_DENSITY_KG_M3, OUTSIDE_TARGETS_M),
    ):
        reference_record = finite[case_name]
        reference = {
            name: _reference_array(reference_record, name) for name in OBSERVABLES
        }
        for h in h_values:
            case_start = time.perf_counter()
            mesh = build_cartesian_six_tetra_mesh(
                HOST_BOUNDS_M,
                h,
                inner_bounds_m=INNER_BOUNDS_M,
                host_density_kg_m3=HOST_DENSITY_KG_M3,
                inner_density_kg_m3=inner_density,
            )
            points, mass = degree2_mass_quadrature(mesh)
            static_gfree = evaluate_static_observables_g_free(points, mass, targets)
            static_physical = to_physical_observables(static_gfree)
            field = UnstructuredDisplacementField(
                mesh.vertices_m, mesh.tetra, affine_displacement(mesh.vertices_m)
            )
            p1 = piecewise_tetra_p1_density_sources(
                field, np.repeat(mesh.cell_density_kg_m3[:, None], 4, axis=1)
            )
            representations = {}
            for observable in OBSERVABLES:
                report = compare_tetra_p1_density_representations(
                    p1, targets, observable=observable, chunk_size=8192, backend="numpy"
                )
                representations[observable] = {
                    "mass_displacement": _array_record(
                        G_SI * report.mass_displacement_response
                    ),
                    "bulk": _array_record(G_SI * report.bulk_response),
                    "interface": _array_record(G_SI * report.interface_response),
                    "bulk_plus_interface": _array_record(
                        G_SI * report.bulk_interface_response
                    ),
                    "comparison_policy": "compare_only_do_not_add_representations",
                    "normalized_representation_residual": report.rel_residual.tolist(),
                }
                if case_name == "cavity":
                    affine_reference = _reference_array(
                        finite["affine_displacement"], observable
                    )
                    representations[observable]["mass_vs_A_normalized_error"] = (
                        normalized_error(
                            report.mass_displacement_response,
                            affine_reference,
                            observable,
                        ).tolist()
                    )
                    representations[observable][
                        "bulk_interface_vs_A_normalized_error"
                    ] = normalized_error(
                        report.bulk_interface_response, affine_reference, observable
                    ).tolist()
                    representations[observable]["A_oracle_status"] = finite[
                        "affine_displacement"
                    ]["convergence"]["status"]
            result["cases"].append(
                {
                    "case": case_name,
                    "h_m": h,
                    "targets_m": targets.tolist(),
                    "mesh": {
                        "vertices": len(mesh.vertices_m),
                        "tetra": len(mesh.tetra),
                        "nominal_h_m": mesh.nominal_h_m,
                        "axis_cell_counts": list(mesh.axis_cell_counts),
                        "axis_max_cell_width_m": mesh.axis_max_cell_width_m.tolist(),
                        "host_volume_m3": mesh.host_volume_m3,
                        "inner_volume_m3": mesh.inner_volume_m3,
                        "material_volume_m3": mesh.material_volume_m3,
                        "material_mass_kg": mesh.material_mass_kg,
                        "geometry_volume_abs_error_m3": abs(
                            mesh.host_volume_m3 - float(np.sum(mesh.tetra_volume_m3))
                        ),
                        "estimated_persistent_array_bytes": int(
                            mesh.vertices_m.nbytes
                            + mesh.tetra.nbytes
                            + mesh.cell_density_kg_m3.nbytes
                            + mesh.tetra_volume_m3.nbytes
                            + points.nbytes
                            + mass.nbytes
                        ),
                    },
                    "static_physical": _observable_record(static_physical),
                    "static_vs_A_normalized_error": {
                        name: normalized_error(
                            getattr(static_gfree, name), reference[name], name
                        ).tolist()
                        for name in OBSERVABLES
                    },
                    "A_static_convergence": reference_record["convergence"],
                    "affine_linearized_physical": representations,
                    "elapsed_s": time.perf_counter() - case_start,
                    "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    / 1024.0,
                }
            )

    representative = next(
        item
        for item in result["cases"]
        if item["case"] == "cavity" and item["h_m"] == 4.0
    )
    mesh = build_cartesian_six_tetra_mesh(
        HOST_BOUNDS_M,
        4.0,
        inner_bounds_m=INNER_BOUNDS_M,
        host_density_kg_m3=HOST_DENSITY_KG_M3,
        inner_density_kg_m3=0.0,
    )
    points, mass = degree2_mass_quadrature(mesh)
    evaluate_static_observables_g_free(points, mass, CAVITY_TARGETS_M)
    samples = []
    for _ in range(5):
        tick = time.perf_counter()
        evaluate_static_observables_g_free(points, mass, CAVITY_TARGETS_M)
        samples.append(time.perf_counter() - tick)
    representative["performance_warm_five"] = {
        "seconds": samples,
        "mean_s": float(np.mean(samples)),
        "median_s": float(np.median(samples)),
        "std_s": float(np.std(samples)),
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    result["started_utc"] = started.isoformat()
    result["finished_utc"] = datetime.now(UTC).isoformat()
    result["elapsed_s"] = (datetime.now(UTC) - started).total_seconds()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "wave4_finite_body_results.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--oracle-json", type=Path, required=True)
    parser.add_argument("--h", type=float, nargs="*", default=(4.0, 2.0, 1.0))
    parser.add_argument("--a-parent-accepted-sha256")
    args = parser.parse_args()
    print(
        run_wave4(
            args.output_dir,
            args.oracle_json,
            h_values=tuple(args.h),
            a_parent_accepted_sha256=args.a_parent_accepted_sha256,
        )
    )


if __name__ == "__main__":
    main()
