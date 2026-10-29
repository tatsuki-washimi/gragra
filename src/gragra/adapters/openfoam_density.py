"""Bounded loader for Foundation OpenFOAM 14 legacy-ASCII density exports.

The adapter accepts only reconstructed single-region ``UNSTRUCTURED_GRID`` files
produced by the documented ``foamToVTK`` profile.  It deliberately preserves
the exporter-native cell centres and volumes after precision-aware geometry
validation; it is not a general OpenFOAM or VTK reader.
"""

from __future__ import annotations

import hashlib
import importlib.util
import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from gragra._arrays import _check_no_bool
from gragra.adapters.meshio_adapter import VolumeMesh
from gragra.gravity_sources.volume import VolumeElementSource

_UNITS = ("kg/m^3", "m", "m^3", "s")
_CALLER_KEYS = {
    "distribution",
    "version",
    "patch",
    "solver",
    "compressibility",
    "region",
    "reconstructed",
    "exporter",
    "export_command",
}
_PROFILE = {
    "distribution": "OpenFOAM Foundation",
    "version": "14",
    "patch": "20260724",
    "region": "region0",
    "reconstructed": True,
    "exporter": "foamToVTK",
}
_RESERVED = {
    "input_sha256",
    "points_dtype",
    "centre_dtype",
    "volume_dtype",
    "density_dtype",
    "cell_id_dtype",
}


def _readonly(a, dtype=None):
    out = np.array(a, dtype=dtype, copy=True)
    out.setflags(write=False)
    return out


def _real_array(name, value, shape=None):
    _check_no_bool(value, name)
    raw = np.asarray(value)
    if raw.dtype.kind in "bOc" or not np.issubdtype(raw.dtype, np.number):
        raise TypeError(f"{name} must be real numeric")
    out = np.asarray(raw, dtype=np.float64)
    if shape is not None and out.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite")
    return out


def _validate_provenance(value):
    if not isinstance(value, Mapping) or set(value) != _CALLER_KEYS:
        raise ValueError("provenance must contain exactly the nine caller keys")
    if set(value) & _RESERVED:
        raise ValueError("caller may not override reserved provenance keys")
    p = dict(value)
    if any(p[k] != v for k, v in _PROFILE.items()):
        raise ValueError("provenance does not match the Foundation 14 profile")
    if not isinstance(p["reconstructed"], bool) or p["reconstructed"] is not True:
        raise ValueError("reconstructed must be the boolean True")
    if (
        not isinstance(p["solver"], str)
        or not p["solver"].strip()
        or not isinstance(p["export_command"], str)
        or not p["export_command"].strip()
    ):
        raise ValueError("solver and export_command must be non-empty strings")
    if p["compressibility"] not in {"compressible", "incompressible", "not-applicable"}:
        raise ValueError("invalid compressibility")
    if (
        p["solver"] == "manufactured-no-solver"
        and p["compressibility"] != "not-applicable"
    ):
        raise ValueError("manufactured-no-solver requires not-applicable")
    return p


def _geometry(points, cells, centres, volumes, eps):
    for kind, _conn in cells:
        if kind not in {"tetra", "hexahedron"}:
            raise ValueError(f"unsupported cell block {kind!r}")
    index = 0
    for kind, conn in cells:
        for row in conn:
            q = points[row]
            h = float(np.max(np.linalg.norm(q[:, None] - q[None, :], axis=-1)))
            if kind == "tetra":
                vg = (
                    abs(
                        float(
                            np.linalg.det(
                                np.stack(
                                    (q[1] - q[0], q[2] - q[0], q[3] - q[0]), axis=1
                                )
                            )
                        )
                    )
                    / 6
                )
                cg = q.mean(axis=0)
                if 6 * vg <= 64 * eps * h**3:
                    raise ValueError("precision-degenerate tetra")
            else:
                p0, p1, p2, p3, p4, p5, p6, p7 = q

                def close(lhs, rhs, cell_h=h):
                    return np.all(
                        np.abs(lhs - rhs)
                        <= 8 * eps * np.maximum(abs(lhs), abs(rhs)) + 32 * eps * cell_h
                    )

                affine = (
                    close(p2, p1 + p3 - p0)
                    and close(p5, p1 + p4 - p0)
                    and close(p7, p3 + p4 - p0)
                    and close(p6, p1 + p3 + p4 - 2 * p0)
                )
                if not affine:
                    raise ValueError("nonaffine hexahedron")
                vg = abs(
                    float(np.linalg.det(np.stack((p1 - p0, p3 - p0, p4 - p0), axis=1)))
                )
                cg = q.mean(axis=0)
                if vg <= 64 * eps * h**3:
                    raise ValueError("precision-degenerate hexahedron")
            ctol = 8 * eps * np.maximum(abs(centres[index]), abs(cg)) + 32 * eps * h
            vtol = 32 * eps * max(abs(volumes[index]), abs(vg)) + 64 * eps * h**3
            if (
                not np.all(np.abs(centres[index] - cg) <= ctol)
                or abs(volumes[index] - vg) > vtol
            ):
                raise ValueError("native C/Vc fails fixed geometry tolerance")
            index += 1


@dataclass(frozen=True, init=False)
class OpenFOAMDensitySnapshot:
    """Immutable native-centre density snapshot in declared world coordinates."""

    input_path: str
    density_unit: str
    coordinate_unit: str
    volume_unit: str
    time_unit: str
    mesh: VolumeMesh
    positions_m: np.ndarray
    volumes_m3: np.ndarray
    density_kg_m3: np.ndarray
    cell_ids: np.ndarray
    time_s: float
    time_name: str
    field_name: str
    density_kind: str
    coordinate_frame: str
    coordinate_origin_m: np.ndarray
    provenance: Mapping[str, object]

    def __init__(self, *_args, **_kwargs):
        raise TypeError(
            "OpenFOAMDensitySnapshot is loader-only; use load_openfoam_density()"
        )

    @classmethod
    def _from_validated_loader(
        cls,
        *,
        input_path,
        density_unit,
        coordinate_unit,
        volume_unit,
        time_unit,
        mesh,
        positions_m,
        volumes_m3,
        density_kg_m3,
        cell_ids,
        time_s,
        time_name,
        field_name,
        coordinate_frame,
        coordinate_origin_m,
        provenance,
    ):
        """Build an immutable result after ``load_openfoam_density`` validation.

        This intentionally private factory is the sole construction path.  The
        loader has already validated the envelope, native payload, geometry,
        profile, and input hash before reaching this point.
        """
        result = object.__new__(cls)
        for name, value in (
            ("input_path", str(input_path)),
            ("density_unit", density_unit),
            ("coordinate_unit", coordinate_unit),
            ("volume_unit", volume_unit),
            ("time_unit", time_unit),
            ("mesh", mesh),
            ("positions_m", _readonly(positions_m, np.float64)),
            ("volumes_m3", _readonly(volumes_m3, np.float64)),
            ("density_kg_m3", _readonly(density_kg_m3, np.float64)),
            ("cell_ids", _readonly(cell_ids, np.int64)),
            ("time_s", float(time_s)),
            ("time_name", time_name),
            ("field_name", field_name),
            ("density_kind", field_name),
            ("coordinate_frame", coordinate_frame),
            ("coordinate_origin_m", _readonly(coordinate_origin_m, np.float64)),
            ("provenance", MappingProxyType(dict(provenance))),
        ):
            object.__setattr__(result, name, value)
        return result

    def __post_init__(self):
        """Validate and defensively freeze every public-constructor value."""
        path = Path(self.input_path)
        if not path.is_absolute():
            raise ValueError("input_path must be an absolute path string")
        if (
            self.density_unit,
            self.coordinate_unit,
            self.volume_unit,
            self.time_unit,
        ) != _UNITS:
            raise ValueError("snapshot units must be canonical")
        if (
            self.field_name not in {"rho", "delta_rho"}
            or self.density_kind != self.field_name
        ):
            raise ValueError("field_name and density_kind must agree")
        if not isinstance(self.mesh, VolumeMesh):
            raise TypeError("mesh must be a VolumeMesh")
        n = sum(len(block[1]) for block in self.mesh.cell_blocks)
        positions = _real_array("positions_m", self.positions_m, (n, 3))
        volumes = _real_array("volumes_m3", self.volumes_m3, (n,))
        density = _real_array("density_kg_m3", self.density_kg_m3, (n,))
        ids_raw = np.asarray(self.cell_ids)
        if ids_raw.dtype.kind not in "iu":
            raise TypeError("cell_ids must be integer")
        ids = np.asarray(ids_raw, dtype=np.int64)
        if (
            ids.shape != (n,)
            or (ids < 0).any()
            or not np.array_equal(np.sort(ids), np.arange(n))
        ):
            raise ValueError("cell_ids must be a permutation of 0..N-1")
        if (volumes <= 0).any() or (self.field_name == "rho" and (density < 0).any()):
            raise ValueError("invalid snapshot density or volume")
        origin = _real_array("coordinate_origin_m", self.coordinate_origin_m, (3,))
        if (
            not isinstance(self.coordinate_frame, str)
            or not self.coordinate_frame.strip()
        ):
            raise ValueError("coordinate_frame must be non-empty")
        if (
            not isinstance(self.time_name, str)
            or not isinstance(self.time_s, numbers.Real)
            or isinstance(self.time_s, bool)
            or not np.isfinite(self.time_s)
        ):
            raise ValueError("snapshot time is invalid")
        try:
            if (
                not np.isfinite(float(self.time_name))
                or abs(float(self.time_name) - self.time_s) > 1e-12
            ):
                raise ValueError("snapshot time_name and time_s must agree")
        except ValueError:
            raise ValueError("snapshot time_name must be finite numeric") from None
        if (
            not isinstance(self.provenance, Mapping)
            or set(self.provenance) != _CALLER_KEYS | _RESERVED
        ):
            raise ValueError("snapshot provenance keys are invalid")
        caller = _validate_provenance({k: self.provenance[k] for k in _CALLER_KEYS})
        reserved = {k: self.provenance[k] for k in _RESERVED}
        if (
            not isinstance(reserved["input_sha256"], str)
            or len(reserved["input_sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in reserved["input_sha256"])
            or any(
                not isinstance(reserved[k], str) for k in _RESERVED - {"input_sha256"}
            )
        ):
            raise ValueError("snapshot reserved provenance is invalid")
        expected_dtypes = {
            "points_dtype": "float32",
            "centre_dtype": "float32",
            "volume_dtype": "float32",
            "density_dtype": "float32",
            "cell_id_dtype": "int32",
        }
        if any(reserved[key] != value for key, value in expected_dtypes.items()):
            raise ValueError("snapshot payload dtype provenance is invalid")
        _geometry(
            self.mesh.vertices_m - origin,
            self.mesh.cell_blocks,
            positions - origin,
            volumes,
            np.finfo(np.float32).eps,
        )
        frozen = MappingProxyType({**caller, **reserved})
        for name, value in (
            ("positions_m", positions),
            ("volumes_m3", volumes),
            ("density_kg_m3", density),
            ("cell_ids", ids),
            ("coordinate_origin_m", origin),
        ):
            object.__setattr__(self, name, _readonly(value))
        object.__setattr__(self, "input_path", str(path))
        object.__setattr__(self, "time_s", float(self.time_s))
        object.__setattr__(self, "provenance", frozen)

    def as_volume_element_source(self):
        return VolumeElementSource(
            self.positions_m, self.volumes_m3, self.density_kg_m3
        )

    def assert_compatible_successor(self, next_snapshot):
        if not isinstance(next_snapshot, OpenFOAMDensitySnapshot):
            raise TypeError("next_snapshot must be OpenFOAMDensitySnapshot")
        if not next_snapshot.time_s > self.time_s:
            raise ValueError("successor time must be strictly increasing")
        attrs = (
            "field_name",
            "density_unit",
            "coordinate_unit",
            "volume_unit",
            "time_unit",
            "coordinate_frame",
        )
        arrays = ("positions_m", "volumes_m3", "cell_ids", "coordinate_origin_m")
        if any(getattr(self, x) != getattr(next_snapshot, x) for x in attrs) or any(
            not np.array_equal(getattr(self, x), getattr(next_snapshot, x))
            for x in arrays
        ):
            raise ValueError("successor geometry or profile differs")
        same_blocks = len(self.mesh.cell_blocks) == len(
            next_snapshot.mesh.cell_blocks
        ) and all(
            a[0] == b[0] and np.array_equal(a[1], b[1])
            for a, b in zip(
                self.mesh.cell_blocks, next_snapshot.mesh.cell_blocks, strict=True
            )
        )
        if not same_blocks or not np.array_equal(
            self.mesh.vertices_m, next_snapshot.mesh.vertices_m
        ):
            raise ValueError("successor mesh differs")
        keys = _PROFILE.keys() | {"solver", "compressibility"}
        if any(self.provenance[k] != next_snapshot.provenance[k] for k in keys):
            raise ValueError("successor producer profile differs")


def load_openfoam_density(
    path,
    *,
    field_name,
    time_s,
    time_name,
    coordinate_frame,
    coordinate_origin_m,
    density_unit,
    coordinate_unit,
    volume_unit,
    time_unit,
    provenance,
):
    """Load ``rho`` or ``delta_rho`` from the bounded legacy-ASCII profile."""
    if field_name not in {"rho", "delta_rho"}:
        raise ValueError("field_name must be 'rho' or 'delta_rho'")
    if (density_unit, coordinate_unit, volume_unit, time_unit) != _UNITS:
        raise ValueError("units must be canonical kg/m^3, m, m^3, s")
    if not isinstance(time_name, str):
        raise TypeError("time_name must be a string")
    try:
        parsed = float(time_name)
    except ValueError as e:
        raise ValueError("time_name must be finite numeric") from e
    if (
        not np.isfinite(parsed)
        or isinstance(time_s, bool)
        or not isinstance(time_s, numbers.Real)
        or not np.isfinite(time_s)
        or abs(float(time_s) - parsed) > 1e-12
    ):
        raise ValueError("time_s and time_name must agree")
    if not isinstance(coordinate_frame, str) or not coordinate_frame.strip():
        raise ValueError("coordinate_frame must be non-empty")
    origin = _real_array("coordinate_origin_m", coordinate_origin_m, (3,))
    p = Path(path).resolve()
    if p.suffix != ".vtk" or not p.name.endswith(f"_{time_name}.vtk"):
        raise ValueError("filename must end with _{time_name}.vtk")
    raw = p.read_bytes()
    try:
        header = raw.decode("ascii").splitlines()[:4]
    except UnicodeDecodeError as exc:
        raise ValueError("requires legacy ASCII UNSTRUCTURED_GRID") from exc
    if (
        len(header) != 4
        or not header[0].startswith("# vtk DataFile Version")
        or header[2] != "ASCII"
        or header[3] != "DATASET UNSTRUCTURED_GRID"
    ):
        raise ValueError("requires legacy ASCII UNSTRUCTURED_GRID")
    if importlib.util.find_spec("meshio") is None:
        raise ImportError("meshio is required; install gragra[mesh]")
    import meshio

    m = meshio.read(str(p))
    if m.point_data:
        raise ValueError("point_data is not allowed")
    required = {field_name, "C", "Vc", "cellID"}
    if not required <= set(m.cell_data):
        raise ValueError("missing required cell_data")
    cells = []
    values = {k: [] for k in required}
    for i, block in enumerate(m.cells):
        if block.type not in {"tetra", "hexahedron"}:
            raise ValueError(f"unsupported cell block {block.type!r}")
        cells.append((block.type, block.data))
        for k in required:
            if i >= len(m.cell_data[k]) or len(m.cell_data[k][i]) != len(block.data):
                raise ValueError(f"cell_data {k} is not aligned")
            values[k].append(m.cell_data[k][i])
    arr = {k: np.concatenate(v) for k, v in values.items()}
    points = np.asarray(m.points)
    density_raw, centre_raw, volume_raw, ids_raw = (
        arr[field_name],
        arr["C"],
        arr["Vc"],
        arr["cellID"],
    )
    if (
        any(x.dtype.kind in "bOc" for x in (density_raw, centre_raw, volume_raw))
        or ids_raw.dtype.kind not in "iu"
    ):
        raise TypeError("payload dtypes are invalid")
    # meshio represents legacy VTK scalar cell arrays as (N, 1); that is the
    # scalar shape, while vectors must remain exactly (N, 3).
    if density_raw.ndim == 2 and density_raw.shape[1:] == (1,):
        density_raw = density_raw[:, 0]
    if volume_raw.ndim == 2 and volume_raw.shape[1:] == (1,):
        volume_raw = volume_raw[:, 0]
    if ids_raw.ndim == 2 and ids_raw.shape[1:] == (1,):
        ids_raw = ids_raw[:, 0]
    total_cell_count = sum(len(block.data) for block in m.cells)
    if (
        density_raw.shape != (total_cell_count,)
        or centre_raw.shape != (total_cell_count, 3)
        or volume_raw.shape != (total_cell_count,)
        or ids_raw.shape != (total_cell_count,)
    ):
        raise ValueError("C, Vc, cellID, and density payloads have invalid shape")
    if any(
        x.dtype != np.dtype("float32")
        for x in (points, centre_raw, volume_raw, density_raw)
    ) or ids_raw.dtype != np.dtype("int32"):
        raise ValueError(
            "Foundation 14 profile requires float32 C/Vc/density and int32 "
            "cellID payloads"
        )
    density = _real_array(field_name, density_raw, (total_cell_count,))
    centres = _real_array("C", centre_raw, (total_cell_count, 3))
    volumes = _real_array("Vc", volume_raw, (total_cell_count,))
    if (volumes <= 0).any() or (field_name == "rho" and (density < 0).any()):
        raise ValueError("invalid density or Vc")
    ids = np.asarray(ids_raw, dtype=np.int64)
    if (ids < 0).any() or not np.array_equal(np.sort(ids), np.arange(total_cell_count)):
        raise ValueError("cellID must be a permutation of 0..N-1")
    eps = max(
        np.finfo(points.dtype).eps,
        np.finfo(centre_raw.dtype).eps,
        np.finfo(volume_raw.dtype).eps,
        np.finfo(density_raw.dtype).eps,
    )
    _geometry(np.asarray(points, dtype=np.float64), cells, centres, volumes, eps)
    world_points = _real_array(
        "world points", np.asarray(points, dtype=np.float64) + origin
    )
    world_centres = _real_array("world C", centres + origin, (len(density), 3))
    prov = _validate_provenance(provenance)
    prov.update(
        input_sha256=hashlib.sha256(raw).hexdigest(),
        points_dtype=points.dtype.name,
        centre_dtype=centre_raw.dtype.name,
        volume_dtype=volume_raw.dtype.name,
        density_dtype=density_raw.dtype.name,
        cell_id_dtype=ids_raw.dtype.name,
    )
    return OpenFOAMDensitySnapshot._from_validated_loader(
        input_path=str(p),
        density_unit=density_unit,
        coordinate_unit=coordinate_unit,
        volume_unit=volume_unit,
        time_unit=time_unit,
        mesh=VolumeMesh(world_points, tuple(cells)),
        positions_m=world_centres,
        volumes_m3=volumes,
        density_kg_m3=density,
        cell_ids=ids,
        time_s=time_s,
        time_name=time_name,
        field_name=field_name,
        coordinate_frame=coordinate_frame,
        coordinate_origin_m=origin,
        provenance=prov,
    )
