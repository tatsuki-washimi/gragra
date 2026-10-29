"""Quadrature geometry and field sampling for generic Newton integration."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from gragra._arrays import _check_no_bool, as_xyz_table
from gragra.fields.finite_body_p1 import piecewise_tetra_p1_density_sources
from gragra.fields.surface import SurfaceDisplacementField
from gragra.fields.unstructured import UnstructuredDisplacementField
from gragra.geometry.tetra_quadrature import (
    tetra_gauss_quadrature,
    triangle_gauss_quadrature,
)
from gragra.integrators.generic import DisplacementSourceBlock, ScalarSourceBlock

_SURFACE_BARYCENTRIC_TOLERANCE = 2e-12


def _integer_connectivity(
    name: str, value, width: int, vertex_count: int
) -> np.ndarray:
    _check_no_bool(value, name)
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.shape[1] != width:
        raise ValueError(f"{name} must have shape (N,{width}), got {raw.shape}")
    if raw.dtype.kind not in "iu":
        raise TypeError(f"{name} must have integer dtype")
    result = np.asarray(raw, dtype=np.int64).copy()
    if result.size and (np.any(result < 0) or np.any(result >= vertex_count)):
        raise ValueError(f"{name} contains an invalid vertex index")
    return result


def _readonly(values, dtype=np.float64) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _array_digest(values: np.ndarray) -> bytes:
    """Return a compact digest binding stored quadrature data to its source."""
    contiguous = np.ascontiguousarray(values, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    if contiguous.size:
        digest.update(memoryview(contiguous).cast("B"))
    return digest.digest()


@dataclass(frozen=True, slots=True)
class QuadraturePlan:
    """Immutable quadrature points, physical measures, and element ownership."""

    points_m: np.ndarray
    weights: np.ndarray
    normals: np.ndarray | None
    measure: Literal["volume", "surface"]
    element_ids: np.ndarray
    quadrature_order: int
    _surface_reference_barycentric: np.ndarray | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _surface_geometry_digest: bytes | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _surface_points_digest: bytes | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.measure not in {"volume", "surface"}:
            raise ValueError("measure must be 'volume' or 'surface'")
        if isinstance(self.quadrature_order, (bool, np.bool_)) or not isinstance(
            self.quadrature_order, (int, np.integer)
        ):
            raise TypeError("quadrature_order must be an integer")
        if self.quadrature_order < 2:
            raise ValueError("quadrature_order must be >= 2")

        _check_no_bool(self.points_m, "points_m")
        points = as_xyz_table("points_m", self.points_m)
        _check_no_bool(self.weights, "weights")
        raw_weights = np.asarray(self.weights)
        if raw_weights.ndim != 1 or raw_weights.shape != (len(points),):
            raise ValueError("weights must have shape (Nq,)")
        if np.iscomplexobj(raw_weights) or raw_weights.dtype.kind not in "iuf":
            raise TypeError("weights must be real-valued")
        weights = np.asarray(raw_weights, dtype=np.float64).copy()
        if not np.isfinite(weights).all() or np.any(weights < 0.0):
            raise ValueError("weights must be finite and non-negative")

        _check_no_bool(self.element_ids, "element_ids")
        raw_ids = np.asarray(self.element_ids)
        if raw_ids.ndim != 1 or raw_ids.shape != (len(points),):
            raise ValueError("element_ids must have shape (Nq,)")
        if raw_ids.dtype.kind not in "iu":
            raise TypeError("element_ids must have integer dtype")
        element_ids = np.asarray(raw_ids, dtype=np.int64).copy()
        if np.any(element_ids < 0):
            raise ValueError("element_ids must be non-negative")

        if self.measure == "volume":
            if self.normals is not None:
                raise ValueError("volume plans must have normals=None")
            normals = None
        else:
            if self.normals is None:
                raise ValueError("surface plans require unit normals")
            _check_no_bool(self.normals, "normals")
            raw_normals = np.asarray(self.normals)
            if raw_normals.shape != points.shape or raw_normals.dtype.kind not in "iuf":
                raise ValueError("surface normals must be real with shape (Nq,3)")
            normals = np.asarray(raw_normals, dtype=np.float64).copy()
            lengths = np.linalg.norm(normals, axis=1)
            if not np.isfinite(normals).all() or not np.allclose(
                lengths, 1.0, rtol=0.0, atol=2e-13
            ):
                raise ValueError("surface normals must be finite unit vectors")
            normals.setflags(write=False)

        points = _readonly(points)
        weights.setflags(write=False)
        element_ids.setflags(write=False)
        object.__setattr__(self, "points_m", points)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "normals", normals)
        object.__setattr__(self, "element_ids", element_ids)
        object.__setattr__(self, "quadrature_order", int(self.quadrature_order))
        object.__setattr__(self, "_surface_reference_barycentric", None)
        object.__setattr__(self, "_surface_geometry_digest", None)
        object.__setattr__(self, "_surface_points_digest", None)

    @classmethod
    def from_surface_mesh(cls, vertices_m, triangles, *, order: int) -> QuadraturePlan:
        """Build a triangle plan, preserving the input winding for normals."""
        vertices = as_xyz_table("vertices_m", vertices_m)
        cells = _integer_connectivity("triangles", triangles, 3, len(vertices))
        barycentric, normalized_weights = triangle_gauss_quadrature(order)
        node_coordinates = vertices[cells]
        cross = np.cross(
            node_coordinates[:, 1] - node_coordinates[:, 0],
            node_coordinates[:, 2] - node_coordinates[:, 0],
        )
        lengths = np.linalg.norm(cross, axis=1)
        if np.any(lengths <= 1e-15):
            raise ValueError("degenerate zero-area triangle detected")
        areas = lengths / 2.0
        points = np.einsum("ql,eli->eqi", barycentric, node_coordinates).reshape(-1, 3)
        weights = (areas[:, None] * normalized_weights[None, :]).reshape(-1)
        normals = np.repeat(cross / lengths[:, None], len(normalized_weights), axis=0)
        element_ids = np.repeat(
            np.arange(len(cells), dtype=np.int64), len(normalized_weights)
        )
        plan = cls(points, weights, normals, "surface", element_ids, order)
        object.__setattr__(
            plan,
            "_surface_reference_barycentric",
            _readonly(barycentric),
        )
        object.__setattr__(
            plan, "_surface_geometry_digest", _array_digest(node_coordinates)
        )
        object.__setattr__(plan, "_surface_points_digest", _array_digest(plan.points_m))
        return plan

    @classmethod
    def from_tetra_mesh(cls, vertices_m, tetra, *, order: int) -> QuadraturePlan:
        """Build a tetrahedron plan with physical volume weights."""
        vertices = as_xyz_table("vertices_m", vertices_m)
        cells = _integer_connectivity("tetra", tetra, 4, len(vertices))
        barycentric, normalized_weights = tetra_gauss_quadrature(order)
        node_coordinates = vertices[cells]
        jacobian = np.stack(
            (
                node_coordinates[:, 1] - node_coordinates[:, 0],
                node_coordinates[:, 2] - node_coordinates[:, 0],
                node_coordinates[:, 3] - node_coordinates[:, 0],
            ),
            axis=-1,
        )
        volumes = np.abs(np.linalg.det(jacobian)) / 6.0
        if np.any(volumes <= 1e-15):
            raise ValueError("degenerate zero-volume tetrahedron detected")
        points = np.einsum("ql,eli->eqi", barycentric, node_coordinates).reshape(-1, 3)
        weights = (volumes[:, None] * normalized_weights[None, :]).reshape(-1)
        element_ids = np.repeat(
            np.arange(len(cells), dtype=np.int64), len(normalized_weights)
        )
        return cls(points, weights, None, "volume", element_ids, order)


def _triangle_barycentric(points: np.ndarray, triangles_m: np.ndarray) -> np.ndarray:
    """Return linear-triangle coordinates for points paired with each face."""
    first = triangles_m[:, 0]
    edge_a = triangles_m[:, 1] - first
    edge_b = triangles_m[:, 2] - first
    offset = points - first

    # Project each triangle onto the coordinate plane with its largest normal
    # component. The projected 2x2 determinant avoids subtracting nearly equal
    # Gram products for skinny but nondegenerate triangles.
    normal = np.cross(edge_a, edge_b)
    dominant_axis = np.argmax(np.abs(normal), axis=1)
    first_axis = (dominant_axis + 1) % 3
    second_axis = (dominant_axis + 2) % 3
    rows = np.arange(len(points))
    a_first = edge_a[rows, first_axis]
    a_second = edge_a[rows, second_axis]
    b_first = edge_b[rows, first_axis]
    b_second = edge_b[rows, second_axis]
    offset_first = offset[rows, first_axis]
    offset_second = offset[rows, second_axis]

    denominator = a_first * b_second - a_second * b_first
    u = (offset_first * b_second - offset_second * b_first) / denominator
    v = (a_first * offset_second - a_second * offset_first) / denominator
    return np.stack((1.0 - u - v, u, v), axis=1)


def _triangle_barycentric_uncertainty(
    points: np.ndarray, triangles_m: np.ndarray, barycentric: np.ndarray
) -> np.ndarray:
    """Bound float64 point/solve roundoff in reconstructed triangle coordinates."""
    first = triangles_m[:, 0]
    edge_a = triangles_m[:, 1] - first
    edge_b = triangles_m[:, 2] - first
    normal = np.cross(edge_a, edge_b)
    dominant_axis = np.argmax(np.abs(normal), axis=1)
    first_axis = (dominant_axis + 1) % 3
    second_axis = (dominant_axis + 2) % 3
    rows = np.arange(len(points))

    a_first = edge_a[rows, first_axis]
    a_second = edge_a[rows, second_axis]
    b_first = edge_b[rows, first_axis]
    b_second = edge_b[rows, second_axis]
    offset = points - first
    offset_first = offset[rows, first_axis]
    offset_second = offset[rows, second_axis]
    determinant = a_first * b_second - a_second * b_first
    determinant_terms = np.abs(a_first * b_second) + np.abs(a_second * b_first)
    eps = np.finfo(np.float64).eps
    gamma = 8.0 * eps
    determinant_error = gamma * determinant_terms
    determinant_lower_bound = np.abs(determinant) - determinant_error

    point_ulp = np.spacing(np.abs(points))
    point_evaluation_roundoff = (3.0 * eps) * np.einsum(
        "qi,qij->qj", np.abs(barycentric), np.abs(triangles_m)
    )
    point_error = point_ulp + point_evaluation_roundoff
    offset_error = point_error + eps * np.abs(offset)

    numerator_u_terms = np.abs(offset_first * b_second) + np.abs(
        offset_second * b_first
    )
    numerator_v_terms = np.abs(a_first * offset_second) + np.abs(
        a_second * offset_first
    )
    numerator_u = offset_first * b_second - offset_second * b_first
    numerator_v = a_first * offset_second - a_second * offset_first
    numerator_u_error = (
        np.abs(b_second) * offset_error[rows, first_axis]
        + np.abs(b_first) * offset_error[rows, second_axis]
        + gamma * numerator_u_terms
    )
    numerator_v_error = (
        np.abs(a_second) * offset_error[rows, first_axis]
        + np.abs(a_first) * offset_error[rows, second_axis]
        + gamma * numerator_v_terms
    )

    unresolved = determinant_lower_bound <= 0.0
    safe_denominator = np.where(unresolved, 1.0, determinant_lower_bound)
    u = numerator_u / determinant
    v = numerator_v / determinant
    u_error = (numerator_u_error + np.abs(u) * determinant_error) / safe_denominator
    v_error = (numerator_v_error + np.abs(v) * determinant_error) / safe_denominator
    uncertainty = np.stack((u_error + v_error, u_error, v_error), axis=1)
    uncertainty[unresolved] = np.inf
    return uncertainty


def _stored_surface_quadrature_rule(
    plan: QuadraturePlan, triangles_m: np.ndarray
) -> np.ndarray | None:
    """Return the provenance-bound reference rule for generated plans."""
    reference = plan._surface_reference_barycentric
    if reference is None:
        return None
    geometry_digest = plan._surface_geometry_digest
    points_digest = plan._surface_points_digest
    if geometry_digest is None or points_digest is None:
        raise ValueError("stored surface quadrature provenance is incomplete")

    rule, _ = triangle_gauss_quadrature(plan.quadrature_order)
    points_per_element = len(rule)
    if len(plan.points_m) % points_per_element:
        raise ValueError(
            "stored surface quadrature coordinates do not align by element"
        )
    if reference.shape != rule.shape or not np.array_equal(reference, rule):
        raise ValueError("stored surface quadrature rule does not match the plan")
    element_count = len(plan.points_m) // points_per_element
    if not np.all(
        plan.element_ids.reshape(element_count, points_per_element)
        == np.arange(element_count, dtype=np.int64)[:, None]
    ):
        raise ValueError("stored surface quadrature coordinates do not match the plan")

    element_nodes = triangles_m[::points_per_element]
    if _array_digest(element_nodes) != geometry_digest:
        raise ValueError("plan geometry does not match the source surface geometry")
    if _array_digest(plan.points_m) != points_digest:
        raise ValueError("plan geometry does not match its surface field")
    return reference


class FieldSampler:
    """Sample the existing linear surface and tetrahedral P1 field models."""

    def sample_surface_displacement(
        self,
        field: SurfaceDisplacementField,
        plan: QuadraturePlan,
        *,
        displacement_m=None,
    ) -> np.ndarray:
        """Sample triangle P1 displacements, with parameters before components."""
        if not isinstance(field, SurfaceDisplacementField):
            raise TypeError("field must be a SurfaceDisplacementField")
        if not isinstance(plan, QuadraturePlan) or plan.measure != "surface":
            raise ValueError("plan must be a surface QuadraturePlan")
        if np.any(plan.element_ids >= len(field.triangles)):
            raise ValueError("plan element_ids do not match the surface field")
        triangles_m = field.vertices_m[field.triangles[plan.element_ids]]
        reference_rule = _stored_surface_quadrature_rule(plan, triangles_m)
        barycentric_uncertainty = None
        if reference_rule is None:
            barycentric = _triangle_barycentric(plan.points_m, triangles_m)
            barycentric_uncertainty = _triangle_barycentric_uncertainty(
                plan.points_m, triangles_m, barycentric
            )
        else:
            barycentric = reference_rule
        edge_a = triangles_m[:, 1] - triangles_m[:, 0]
        edge_b = triangles_m[:, 2] - triangles_m[:, 0]
        face_normals = np.cross(edge_a, edge_b)
        # |a| + |b| is an upper bound on every local edge length.
        local_scale = np.linalg.norm(edge_a, axis=1) + np.linalg.norm(edge_b, axis=1)
        del edge_a, edge_b
        face_normals /= np.linalg.norm(face_normals, axis=1)[:, None]
        if not np.allclose(plan.normals, face_normals, rtol=0.0, atol=2e-12):
            raise ValueError("plan normals do not follow the surface field winding")
        plane_distance = np.abs(
            np.einsum("qi,qi->q", plan.points_m - triangles_m[:, 0], face_normals)
        )
        coordinate_ulp = np.maximum(
            np.spacing(np.abs(plan.points_m)),
            np.max(np.spacing(np.abs(triangles_m)), axis=1),
        )
        coordinate_roundoff = 4.0 * np.einsum(
            "qi,qi->q", np.abs(face_normals), coordinate_ulp
        )
        local_roundoff = 64.0 * np.finfo(np.float64).eps * local_scale
        if np.any(plane_distance > local_roundoff + coordinate_roundoff):
            raise ValueError("plan points must lie on their indexed surface elements")
        if barycentric_uncertainty is not None and np.any(
            ~np.isfinite(barycentric_uncertainty)
            | (barycentric_uncertainty > _SURFACE_BARYCENTRIC_TOLERANCE)
        ):
            raise ValueError(
                "plan barycentric coordinates cannot be resolved to "
                f"{_SURFACE_BARYCENTRIC_TOLERANCE:g} in float64"
            )
        if not np.isfinite(barycentric).all() or np.any(
            (barycentric < -_SURFACE_BARYCENTRIC_TOLERANCE)
            | (barycentric > 1.0 + _SURFACE_BARYCENTRIC_TOLERANCE)
        ):
            raise ValueError("plan points do not lie in their indexed surface elements")
        if displacement_m is None:
            nodal_values = field.displacement_m[:, None, :]
        else:
            nodal_values = _parameterized_displacements(
                "displacement_m", displacement_m, len(field.vertices_m)
            )
        if reference_rule is None:
            nodal_displacement = nodal_values[field.triangles[plan.element_ids]]
            sampled = np.einsum("ql,qlpi->qpi", barycentric, nodal_displacement)
        else:
            element_ids = plan.element_ids[:: len(reference_rule)]
            nodal_displacement = nodal_values[field.triangles[element_ids]]
            sampled = np.einsum(
                "ql,elpi->eqpi", reference_rule, nodal_displacement
            ).reshape(len(plan.points_m), nodal_values.shape[1], 3)
        return _readonly(sampled, sampled.dtype)

    def sample_tetra_p1_density(
        self,
        field: UnstructuredDisplacementField,
        cell_density_nodes_kg_m3,
        plan: QuadraturePlan,
    ) -> np.ndarray:
        if not isinstance(field, UnstructuredDisplacementField):
            raise TypeError("field must be an UnstructuredDisplacementField")
        if not isinstance(plan, QuadraturePlan) or plan.measure != "volume":
            raise ValueError("plan must be a volume QuadraturePlan")
        if np.any(plan.element_ids >= len(field.tetra)):
            raise ValueError("plan element_ids do not match the tetrahedral field")
        sources = piecewise_tetra_p1_density_sources(
            field,
            cell_density_nodes_kg_m3,
            quadrature_order=plan.quadrature_order,
        )
        if (
            not np.allclose(
                plan.points_m, sources.mass_positions_m, rtol=0.0, atol=2e-14
            )
            or not np.allclose(
                plan.weights,
                sources.bulk_source.volumes_m3,
                rtol=2e-13,
                atol=0.0,
            )
            or not np.array_equal(plan.element_ids, sources.mass_cell_index)
        ):
            raise ValueError("plan geometry does not match the tetrahedral field")
        sampled = sources.mass_kg / plan.weights
        return _readonly(sampled[:, None], sampled.dtype)


def _parameterized_values(name: str, value, rows: int) -> np.ndarray:
    _check_no_bool(value, name)
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.shape[0] != rows:
        raise ValueError(f"{name} must have shape (Nq,Bp)")
    if raw.dtype.kind not in "iufc":
        raise TypeError(f"{name} must be numeric")
    dtype = np.complex128 if np.iscomplexobj(raw) else np.float64
    result = np.asarray(raw, dtype=dtype)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _parameterized_displacements(name: str, value, rows: int) -> np.ndarray:
    _check_no_bool(value, name)
    raw = np.asarray(value)
    if raw.ndim != 3 or raw.shape[0] != rows or raw.shape[2] != 3:
        raise ValueError(f"{name} must have shape (Nq,Bp,3)")
    if raw.dtype.kind not in "iufc":
        raise TypeError(f"{name} must be numeric")
    dtype = np.complex128 if np.iscomplexobj(raw) else np.float64
    result = np.asarray(raw, dtype=dtype)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _parameter_slice(value: slice | None, parameter_count: int) -> slice:
    if value is None:
        return slice(0, parameter_count)
    if not isinstance(value, slice):
        raise TypeError("parameter_slice must be a slice")
    if (
        isinstance(value.start, bool)
        or isinstance(value.stop, bool)
        or not isinstance(value.start, (int, np.integer))
        or not isinstance(value.stop, (int, np.integer))
        or value.start < 0
        or value.stop <= value.start
        or value.step is not None
        or value.stop - value.start != parameter_count
    ):
        raise ValueError(
            "parameter_slice must span exactly the supplied parameter axis"
        )
    return slice(int(value.start), int(value.stop))


def _real_density(name: str, value, rows: int, *, nonnegative: bool) -> np.ndarray:
    _check_no_bool(value, name)
    raw = np.asarray(value)
    if np.iscomplexobj(raw) or raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be real-valued")
    if raw.ndim == 0:
        result = np.full(rows, float(raw), dtype=np.float64)
    elif raw.shape == (rows,):
        result = np.asarray(raw, dtype=np.float64).copy()
    else:
        raise ValueError(f"{name} must be a scalar or have shape (Nq,)")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    if nonnegative and np.any(result < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return result


class MassCoupling:
    """Convert sampled density and displacement fields to G-free source blocks."""

    @staticmethod
    def volume_density(
        plan: QuadraturePlan,
        density_perturbation_kg_m3,
        *,
        parameter_slice: slice | None = None,
    ) -> ScalarSourceBlock:
        """Multiply sampled volume-density perturbations by their m³ weights."""
        if not isinstance(plan, QuadraturePlan) or plan.measure != "volume":
            raise ValueError("plan must be a volume QuadraturePlan")
        density = _parameterized_values(
            "density_perturbation_kg_m3", density_perturbation_kg_m3, len(plan.weights)
        )
        parameters = _parameter_slice(parameter_slice, density.shape[1])
        weights = plan.weights[:, None] * density
        return ScalarSourceBlock(plan.points_m, weights, 0, parameters)

    @staticmethod
    def surface_density(
        plan: QuadraturePlan,
        surface_density_perturbation_kg_m2,
        *,
        parameter_slice: slice | None = None,
    ) -> ScalarSourceBlock:
        """Integrate an already computed δσ without applying a second jump."""
        if not isinstance(plan, QuadraturePlan) or plan.measure != "surface":
            raise ValueError("plan must be a surface QuadraturePlan")
        density = _parameterized_values(
            "surface_density_perturbation_kg_m2",
            surface_density_perturbation_kg_m2,
            len(plan.weights),
        )
        parameters = _parameter_slice(parameter_slice, density.shape[1])
        weights = plan.weights[:, None] * density
        return ScalarSourceBlock(plan.points_m, weights, 0, parameters)

    @staticmethod
    def interface_density_jump(
        plan: QuadraturePlan,
        displacement_m,
        *,
        density_jump_kg_m3,
        parameter_slice: slice | None = None,
    ) -> ScalarSourceBlock:
        """Convert Δρ(ξ·n̂) to signed/complex surface mass weights."""
        if (
            not isinstance(plan, QuadraturePlan)
            or plan.measure != "surface"
            or plan.normals is None
        ):
            raise ValueError("plan must be a surface QuadraturePlan with normals")
        displacement = _parameterized_displacements(
            "displacement_m", displacement_m, len(plan.weights)
        )
        parameters = _parameter_slice(parameter_slice, displacement.shape[1])
        jump = _real_density(
            "density_jump_kg_m3",
            density_jump_kg_m3,
            len(plan.weights),
            nonnegative=False,
        )
        normal_displacement = np.einsum("qpi,qi->qp", displacement, plan.normals)
        weights = plan.weights[:, None] * jump[:, None] * normal_displacement
        return ScalarSourceBlock(plan.points_m, weights, 0, parameters)

    @staticmethod
    def linearized_displacement(
        plan: QuadraturePlan,
        *,
        background_density_kg_m3,
        displacement_m,
        parameter_slice: slice | None = None,
    ) -> DisplacementSourceBlock:
        """Pair non-negative background mass with sampled displacement vectors."""
        if not isinstance(plan, QuadraturePlan) or plan.measure != "volume":
            raise ValueError("plan must be a volume QuadraturePlan")
        displacement = _parameterized_displacements(
            "displacement_m", displacement_m, len(plan.weights)
        )
        parameters = _parameter_slice(parameter_slice, displacement.shape[1])
        density = _real_density(
            "background_density_kg_m3",
            background_density_kg_m3,
            len(plan.weights),
            nonnegative=True,
        )
        mass = density * plan.weights
        return DisplacementSourceBlock(plan.points_m, mass, displacement, 0, parameters)


__all__ = ["QuadraturePlan", "FieldSampler", "MassCoupling"]
