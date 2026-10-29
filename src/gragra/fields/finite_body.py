"""Linearized sources for finite bodies on conforming P1 tetra meshes.

The mass-displacement and bulk-plus-interface terms returned here are two
representations of the same first-order mass motion.  They must be compared,
not added.  Their cell/face centroid gravity quadratures are generally not
identical on a finite mesh.
"""

from dataclasses import dataclass

import numpy as np

from gragra._arrays import _check_no_bool, as_weights, as_xyz_table
from gragra.fields.surface import SurfaceDisplacementField
from gragra.fields.unstructured import UnstructuredDisplacementField, _tetra_geometry
from gragra.gravity_sources import VolumeElementSource, WeightedPointSource
from gragra.integrators import direct_point_sum
from gragra.integrators.point_direct import direct_point_displacement_sum


def _readonly_integer_vector(name: str, value, length: int) -> np.ndarray:
    _check_no_bool(value, name)
    raw = np.asarray(value)
    if raw.dtype.kind not in "iu" or raw.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},) and integer dtype")
    result = np.asarray(raw, dtype=np.int64).copy()
    result.setflags(write=False)
    return result


def _readonly_displacement(value, length: int) -> np.ndarray:
    _check_no_bool(value, "mass_displacement_m")
    raw = np.asarray(value)
    dtype = np.complex128 if np.iscomplexobj(raw) else np.float64
    try:
        result = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as error:
        raise TypeError("mass_displacement_m must be numeric") from error
    if result.shape != (length, 3):
        raise ValueError(
            f"mass_displacement_m must have shape ({length}, 3), got {result.shape}"
        )
    if not np.isfinite(result).all():
        raise ValueError("mass_displacement_m must contain only finite values")
    result = result.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class TetraLinearizedSources:
    """Auditable source terms for a piecewise-constant-density tetra body."""

    mass_positions_m: np.ndarray
    mass_kg: np.ndarray
    mass_displacement_m: np.ndarray
    bulk_source: VolumeElementSource
    interface_source: WeightedPointSource
    interface_field: SurfaceDisplacementField
    interface_density_jump_kg_m3: np.ndarray
    negative_side_cell: np.ndarray
    positive_side_cell: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.bulk_source, VolumeElementSource):
            raise TypeError("bulk_source must be a VolumeElementSource")
        if not isinstance(self.interface_source, WeightedPointSource):
            raise TypeError("interface_source must be a WeightedPointSource")
        if not isinstance(self.interface_field, SurfaceDisplacementField):
            raise TypeError("interface_field must be a SurfaceDisplacementField")

        positions = as_xyz_table("mass_positions_m", self.mass_positions_m)
        mass = as_weights("mass_kg", self.mass_kg)
        if np.iscomplexobj(mass):
            raise TypeError("mass_kg must be real-valued")
        if np.any(mass < 0.0):
            raise ValueError("mass_kg must be non-negative")
        if len(mass) != len(positions):
            raise ValueError("mass_kg length must match mass_positions_m")
        displacement = _readonly_displacement(self.mass_displacement_m, len(positions))

        jumps = as_weights(
            "interface_density_jump_kg_m3", self.interface_density_jump_kg_m3
        )
        if np.iscomplexobj(jumps):
            raise TypeError("interface_density_jump_kg_m3 must be real-valued")
        nf = len(jumps)
        negative = _readonly_integer_vector(
            "negative_side_cell", self.negative_side_cell, nf
        )
        positive = _readonly_integer_vector(
            "positive_side_cell", self.positive_side_cell, nf
        )
        nc = len(positions)
        if (
            np.any(negative < 0)
            or np.any(negative >= nc)
            or np.any(positive < -1)
            or np.any(positive >= nc)
        ):
            raise ValueError("interface cell indices are outside their valid range")
        if len(self.interface_source.positions_m) != nf:
            raise ValueError("interface_source length must match interface metadata")
        if len(self.interface_field.triangles) != nf:
            raise ValueError("interface_field length must match interface metadata")
        if len(self.bulk_source.positions_m) != len(positions):
            raise ValueError("bulk_source length must match mass_positions_m")

        object.__setattr__(self, "mass_positions_m", positions)
        object.__setattr__(self, "mass_kg", mass)
        object.__setattr__(self, "mass_displacement_m", displacement)
        object.__setattr__(self, "interface_density_jump_kg_m3", jumps)
        object.__setattr__(self, "negative_side_cell", negative)
        object.__setattr__(self, "positive_side_cell", positive)


@dataclass(frozen=True)
class TetraRepresentationReport:
    """Read-only G-free comparison of two linearized representations."""

    observable: str
    bulk_response: np.ndarray
    interface_response: np.ndarray
    bulk_interface_response: np.ndarray
    mass_displacement_response: np.ndarray
    abs_residual: np.ndarray
    rel_residual: np.ndarray

    def __post_init__(self) -> None:
        component_shape = {
            "potential": (),
            "acceleration": (3,),
            "gradient": (3, 3),
        }.get(self.observable)
        if component_shape is None:
            raise ValueError(
                "observable must be 'potential', 'acceleration', or 'gradient'"
            )
        expected = None
        for name in (
            "bulk_response",
            "interface_response",
            "bulk_interface_response",
            "mass_displacement_response",
        ):
            array = np.asarray(getattr(self, name))
            if (
                array.ndim != 1 + len(component_shape)
                or array.shape[1:] != component_shape
            ):
                raise ValueError(
                    f"{name} has component shape {array.shape[1:]}; "
                    f"expected {component_shape}"
                )
            if not np.isfinite(array).all():
                raise ValueError(f"{name} must contain only finite values")
            if expected is None:
                expected = array.shape[0]
            elif array.shape[0] != expected:
                raise ValueError("all response arrays must have the same target length")
            locked = array.copy()
            locked.setflags(write=False)
            object.__setattr__(self, name, locked)

        assert expected is not None
        for name in ("abs_residual", "rel_residual"):
            array = np.asarray(getattr(self, name), dtype=np.float64)
            if array.shape != (expected,):
                raise ValueError(f"{name} must have shape ({expected},)")
            if np.isnan(array).any() or (
                name == "abs_residual" and not np.isfinite(array).all()
            ):
                raise ValueError(f"{name} contains an invalid value")
            locked = array.copy()
            locked.setflags(write=False)
            object.__setattr__(self, name, locked)


def _normalize_density(value, nc: int) -> np.ndarray:
    _check_no_bool(value, "density_kg_m3")
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise TypeError("density_kg_m3 must be real-valued")
    if raw.ndim == 0:
        try:
            scalar = float(raw)
        except (TypeError, ValueError) as error:
            raise TypeError("density_kg_m3 must be numeric") from error
        if not np.isfinite(scalar):
            raise ValueError("density_kg_m3 must contain only finite values")
        if scalar < 0.0:
            raise ValueError("density_kg_m3 must be non-negative")
        density = np.full(nc, scalar, dtype=np.float64)
    elif raw.ndim == 1:
        if raw.shape != (nc,):
            raise ValueError(f"density_kg_m3 must have shape ({nc},)")
        if raw.dtype.kind not in "iuf":
            raise TypeError("density_kg_m3 must be numeric")
        density = np.asarray(raw, dtype=np.float64).copy()
    else:
        raise ValueError("density_kg_m3 must be a scalar or one-dimensional array")
    if not np.isfinite(density).all():
        raise ValueError("density_kg_m3 must contain only finite values")
    if np.any(density < 0.0):
        raise ValueError("density_kg_m3 must be non-negative")
    density.setflags(write=False)
    return density


def _oriented_face_topology(
    field: UnstructuredDisplacementField,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return every unique oriented face and its negative/positive owners."""
    tetra = np.asarray(field.tetra, dtype=np.int64)
    nc = len(tetra)
    if nc == 0:
        empty_faces = np.empty((0, 3), dtype=np.int64)
        empty = np.empty(0, dtype=np.int64)
        return empty_faces, empty, empty.copy()

    if len(np.unique(np.sort(tetra, axis=1), axis=0)) != nc:
        raise ValueError("duplicate tetrahedra are not allowed")

    vertices = field.vertices_m
    cell_centroids = np.mean(vertices[tetra], axis=1)
    owners: dict[tuple[int, int, int], list[tuple[int, np.ndarray]]] = {}
    local_faces = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))
    for cell_index, nodes in enumerate(tetra):
        for local in local_faces:
            face = nodes[np.asarray(local)].copy()
            points = vertices[face]
            normal = np.cross(points[1] - points[0], points[2] - points[0])
            face_centroid = np.mean(points, axis=0)
            if np.dot(normal, face_centroid - cell_centroids[cell_index]) < 0.0:
                face[[1, 2]] = face[[2, 1]]
            key = tuple(sorted(int(index) for index in face))
            entries = owners.setdefault(key, [])
            entries.append((cell_index, face))
            if len(entries) > 2:
                raise ValueError(
                    f"non-manifold tetra face {key} has more than two cells"
                )

    faces: list[np.ndarray] = []
    negative_cells: list[int] = []
    positive_cells: list[int] = []
    boundary_faces: list[np.ndarray] = []
    for key in sorted(owners):
        entries = owners[key]
        if len(entries) == 1:
            negative, face = entries[0]
            positive = -1
            boundary_faces.append(face)
        else:
            entries.sort(key=lambda item: item[0])
            negative, face = entries[0]
            positive = entries[1][0]
            points = vertices[face]
            normal = np.cross(points[1] - points[0], points[2] - points[0])
            face_centroid = np.mean(points, axis=0)
            edge_scale = max(
                np.linalg.norm(points[1] - points[0]),
                np.linalg.norm(points[2] - points[0]),
                np.linalg.norm(points[2] - points[1]),
            )
            side = float(np.dot(normal, cell_centroids[positive] - face_centroid))
            tolerance = 64.0 * np.finfo(np.float64).eps * edge_scale**3
            if side <= tolerance:
                raise ValueError(
                    f"cells sharing face {key} lie on the same side or are ambiguous"
                )
        faces.append(face)
        negative_cells.append(negative)
        positive_cells.append(positive)

    face_array = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    boundary_array = np.asarray(boundary_faces, dtype=np.int64).reshape(-1, 3)
    if len(boundary_array):
        boundary_field = SurfaceDisplacementField(
            vertices, boundary_array, field.displacement_m
        )
        orientation = boundary_field.check_orientation()
        if not (
            orientation.is_closed
            and orientation.is_consistent
            and orientation.boundary_edge_count == 0
            and orientation.nonmanifold_edge_count == 0
        ):
            raise ValueError("outer tetra boundary must be a closed manifold")

    return (
        face_array,
        np.asarray(negative_cells, dtype=np.int64),
        np.asarray(positive_cells, dtype=np.int64),
    )


def _oriented_faces(
    field: UnstructuredDisplacementField,
    density: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the legacy boundary and nonzero-density-jump face selection."""
    faces, negative, positive = _oriented_face_topology(field)
    positive_density = np.zeros(len(faces), dtype=np.float64)
    internal = positive >= 0
    positive_density[internal] = density[positive[internal]]
    jumps = density[negative] - positive_density
    keep = (~internal) | (jumps != 0.0)
    return faces[keep], negative[keep], positive[keep], jumps[keep]


def piecewise_tetra_linear_sources(
    field: UnstructuredDisplacementField,
    density_kg_m3: float | np.ndarray,
) -> TetraLinearizedSources:
    """Build mass-displacement and bulk/interface sources for one finite body.

    Faces are matched by shared vertex IDs only.  Geometric vertex merging and
    global self-intersection detection are outside this bounded contract.
    Exterior density is fixed at vacuum (zero).
    """
    if not isinstance(field, UnstructuredDisplacementField):
        raise TypeError("field must be an UnstructuredDisplacementField")
    density = _normalize_density(density_kg_m3, len(field.tetra))
    positions, volumes = _tetra_geometry(field.vertices_m, field.tetra)
    if len(field.tetra):
        nodes = field.vertices_m[field.tetra]
        jac = np.stack(
            [
                nodes[:, 1] - nodes[:, 0],
                nodes[:, 2] - nodes[:, 0],
                nodes[:, 3] - nodes[:, 0],
            ],
            axis=-1,
        )
        inverse = np.linalg.inv(jac)
        displacement_nodes = field.displacement_m[field.tetra]
        # Subtracting the first nodal value makes a rigid translation produce
        # exact zero instead of leaving cancellation-scale divergence noise.
        displacement_edges = displacement_nodes[:, 1:] - displacement_nodes[:, :1]
        divergence = np.einsum("nki,nki->n", displacement_edges, inverse)
        centroid_displacement = np.mean(displacement_nodes, axis=1)
    else:
        dtype = np.complex128 if np.iscomplexobj(field.displacement_m) else np.float64
        divergence = np.empty(0, dtype=dtype)
        centroid_displacement = np.empty((0, 3), dtype=dtype)

    bulk_source = VolumeElementSource(positions, volumes, -density * divergence)
    faces, negative, positive, jumps = _oriented_faces(field, density)
    interface_field = SurfaceDisplacementField(
        field.vertices_m, faces, field.displacement_m
    )
    unit_jump_source = interface_field.as_surface_mass_source(1.0).as_weighted_source()
    interface_source = WeightedPointSource(
        unit_jump_source.positions_m, unit_jump_source.weights_kg * jumps
    )
    return TetraLinearizedSources(
        mass_positions_m=positions,
        mass_kg=density * volumes,
        mass_displacement_m=centroid_displacement,
        bulk_source=bulk_source,
        interface_source=interface_source,
        interface_field=interface_field,
        interface_density_jump_kg_m3=jumps,
        negative_side_cell=negative,
        positive_side_cell=positive,
    )


def _response_norm(value: np.ndarray, observable: str) -> np.ndarray:
    if observable == "potential":
        return np.abs(value)
    if observable == "acceleration":
        return np.linalg.norm(value, axis=-1)
    return np.linalg.norm(value, axis=(-2, -1))


def compare_tetra_linearized_representations(
    sources: TetraLinearizedSources,
    targets,
    *,
    observable: str = "acceleration",
    softening_m: float = 0.0,
    chunk_size: int | None = None,
    backend: str = "numpy",
) -> TetraRepresentationReport:
    """Compare G-free bulk/interface and mass-displacement quadratures."""
    if not isinstance(sources, TetraLinearizedSources):
        raise TypeError("sources must be a TetraLinearizedSources")
    mass_displacement = direct_point_displacement_sum(
        sources.mass_kg,
        sources.mass_displacement_m,
        sources.mass_positions_m,
        targets,
        observable=observable,
        softening_m=softening_m,
        chunk_size=chunk_size,
        backend=backend,
    )
    bulk = direct_point_sum(
        sources.bulk_source.as_weighted_source(),
        targets,
        observable=observable,
        softening_m=softening_m,
        chunk_size=chunk_size,
        backend=backend,
    )
    interface = direct_point_sum(
        sources.interface_source,
        targets,
        observable=observable,
        softening_m=softening_m,
        chunk_size=chunk_size,
        backend=backend,
    )
    combined = bulk + interface
    absolute = _response_norm(combined - mass_displacement, observable)
    denominator = _response_norm(mass_displacement, observable)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(
            denominator > 0.0,
            absolute / denominator,
            np.where(absolute == 0.0, 0.0, np.inf),
        )
    return TetraRepresentationReport(
        observable,
        bulk,
        interface,
        combined,
        mass_displacement,
        absolute,
        relative,
    )


__all__ = [
    "TetraLinearizedSources",
    "TetraRepresentationReport",
    "piecewise_tetra_linear_sources",
    "compare_tetra_linearized_representations",
]
