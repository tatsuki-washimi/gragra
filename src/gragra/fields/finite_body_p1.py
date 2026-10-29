"""Linearized finite-body sources with cell-local affine density.

The exterior is vacuum. Geometry and displacement are conforming P1, while
density values use each tetrahedron's local node order and may jump across a
shared face. The two returned representations are compared, never added.
"""

from dataclasses import dataclass

import numpy as np

from gragra._arrays import _check_no_bool, as_weights, as_xyz_table
from gragra.fields.finite_body import (
    TetraRepresentationReport,
    _oriented_face_topology,
    _readonly_displacement,
    _readonly_integer_vector,
    _response_norm,
)
from gragra.fields.unstructured import UnstructuredDisplacementField, _tetra_geometry
from gragra.geometry.tetra_quadrature import (
    tetra_gauss_quadrature,
    triangle_gauss_quadrature,
)
from gragra.gravity_sources import VolumeElementSource, WeightedPointSource
from gragra.integrators import direct_point_sum
from gragra.integrators.point_direct import direct_point_displacement_sum

_TETRA_A = (5.0 + 3.0 * np.sqrt(5.0)) / 20.0
_TETRA_B = (5.0 - np.sqrt(5.0)) / 20.0
_TETRA_BARYCENTRIC = np.full((4, 4), _TETRA_B, dtype=np.float64)
np.fill_diagonal(_TETRA_BARYCENTRIC, _TETRA_A)
_TRIANGLE_BARYCENTRIC = np.full((3, 3), 1.0 / 6.0, dtype=np.float64)
np.fill_diagonal(_TRIANGLE_BARYCENTRIC, 2.0 / 3.0)


def _density_nodes(value, nc: int) -> np.ndarray:
    _check_no_bool(value, "cell_density_nodes_kg_m3")
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise TypeError("cell_density_nodes_kg_m3 must be real-valued")
    if raw.shape != (nc, 4):
        raise ValueError(
            f"cell_density_nodes_kg_m3 must have shape ({nc}, 4), got {raw.shape}"
        )
    if raw.dtype.kind not in "iuf":
        raise TypeError("cell_density_nodes_kg_m3 must be numeric")
    result = np.asarray(raw, dtype=np.float64).copy()
    if not np.isfinite(result).all():
        raise ValueError("cell_density_nodes_kg_m3 must contain only finite values")
    if np.any(result < 0.0):
        raise ValueError("cell_density_nodes_kg_m3 must be non-negative")
    result.setflags(write=False)
    return result


def _integer_triangles(name: str, value, length: int, nv: int) -> np.ndarray:
    _check_no_bool(value, name)
    raw = np.asarray(value)
    if raw.dtype.kind not in "iu":
        raise TypeError(f"{name} must have integer dtype")
    if raw.shape != (length, 3):
        raise ValueError(f"{name} must have shape ({length}, 3)")
    result = np.asarray(raw, dtype=np.int64).copy()
    if result.size and (np.any(result < 0) or np.any(result >= nv)):
        raise ValueError(f"{name} contains an invalid vertex index")
    result.setflags(write=False)
    return result


def _real_table(name: str, value, shape: tuple[int, int]) -> np.ndarray:
    _check_no_bool(value, name)
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise TypeError(f"{name} must be real-valued")
    if raw.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {raw.shape}")
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be numeric")
    result = np.asarray(raw, dtype=np.float64).copy()
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    result.setflags(write=False)
    return result


def _resolve_cell_orders(quadrature_order, nc: int) -> np.ndarray | None:
    """Validate/broadcast ``quadrature_order`` to a per-cell ``(nc,)`` array.

    ``None`` (the default) is a sentinel meaning "keep the fixed degree-2
    four-point/three-point rule" -- callers must special-case it rather than
    treat it as an order value, since the fixed rule is not a member of the
    Gauss-Legendre order family (see ``gragra.geometry.tetra_quadrature``).
    """
    if quadrature_order is None:
        return None
    _check_no_bool(quadrature_order, "quadrature_order")
    raw = np.asarray(quadrature_order)
    if np.iscomplexobj(raw):
        raise TypeError("quadrature_order must be real-valued")
    if raw.dtype.kind not in "iu":
        raise TypeError("quadrature_order must be integer-valued")
    if raw.ndim == 0:
        orders = np.full(nc, int(raw), dtype=np.int64)
    elif raw.shape == (nc,):
        orders = raw.astype(np.int64)
    else:
        raise ValueError(f"quadrature_order must be a scalar or shape ({nc},)")
    if orders.size and np.any(orders < 2):
        raise ValueError("quadrature_order must be >= 2 for every cell")
    return orders


def _cell_quadrature(field, density, orders: np.ndarray | None = None):
    nc = len(field.tetra)
    dtype = np.complex128 if np.iscomplexobj(field.displacement_m) else np.float64
    if nc == 0:
        return (
            np.empty((0, 3)),
            np.empty(0),
            np.empty(0),
            np.empty((0, 3), dtype=dtype),
            np.empty(0, dtype=dtype),
            np.empty(0, dtype=np.int64),
        )
    nodes = field.vertices_m[field.tetra]
    displacement_nodes = field.displacement_m[field.tetra]
    _, volumes = _tetra_geometry(field.vertices_m, field.tetra)

    jac = np.stack(
        [
            nodes[:, 1] - nodes[:, 0],
            nodes[:, 2] - nodes[:, 0],
            nodes[:, 3] - nodes[:, 0],
        ],
        axis=-1,
    )
    inverse = np.linalg.inv(jac)
    density_gradient = np.einsum("nk,nkj->nj", density[:, 1:] - density[:, :1], inverse)
    displacement_edges = displacement_nodes[:, 1:] - displacement_nodes[:, :1]
    divergence = np.einsum("nki,nki->n", displacement_edges, inverse)

    if orders is None:
        positions = np.einsum("ql,nli->nqi", _TETRA_BARYCENTRIC, nodes).reshape(-1, 3)
        density_q = np.einsum("ql,nl->nq", _TETRA_BARYCENTRIC, density)
        displacement_q = np.einsum(
            "ql,nli->nqi", _TETRA_BARYCENTRIC, displacement_nodes
        ).reshape(-1, 3)
        quadrature_volumes = np.repeat(volumes / 4.0, 4)
        mass = density_q.reshape(-1) * quadrature_volumes
        advection = np.einsum(
            "nqi,ni->nq", displacement_q.reshape(nc, 4, 3), density_gradient
        )
        bulk_density = (
            (-density_q * divergence[:, None] - advection).reshape(-1).astype(dtype)
        )
        points_per_cell = np.full(nc, 4, dtype=np.int64)
        return (
            positions,
            quadrature_volumes,
            mass,
            displacement_q,
            bulk_density,
            points_per_cell,
        )

    unique_orders = np.unique(orders)
    points_per_cell = np.zeros(nc, dtype=np.int64)
    position_blocks, volume_blocks, mass_blocks = [], [], []
    displacement_blocks, bulk_blocks, cell_index_blocks = [], [], []
    for order in unique_orders:
        idx = np.flatnonzero(orders == order)
        bary, weights = tetra_gauss_quadrature(int(order))
        q = len(weights)
        points_per_cell[idx] = q
        group_nodes = nodes[idx]
        group_displacement_nodes = displacement_nodes[idx]
        pos_q = np.einsum("ql,nli->nqi", bary, group_nodes).reshape(-1, 3)
        density_q = np.einsum("ql,nl->nq", bary, density[idx])
        disp_q = np.einsum("ql,nli->nqi", bary, group_displacement_nodes)
        qvol = (weights[None, :] * volumes[idx, None]).reshape(-1)
        m = density_q.reshape(-1) * qvol
        advection = np.einsum("nqi,ni->nq", disp_q, density_gradient[idx])
        bulk = (
            (-density_q * divergence[idx, None] - advection).reshape(-1).astype(dtype)
        )
        position_blocks.append(pos_q)
        volume_blocks.append(qvol)
        mass_blocks.append(m)
        displacement_blocks.append(disp_q.reshape(-1, 3))
        bulk_blocks.append(bulk)
        cell_index_blocks.append(np.repeat(idx, q))

    order_cell_index = np.concatenate(cell_index_blocks)
    sort_perm = np.argsort(order_cell_index, kind="stable")
    positions = np.concatenate(position_blocks, axis=0)[sort_perm]
    quadrature_volumes = np.concatenate(volume_blocks)[sort_perm]
    mass = np.concatenate(mass_blocks)[sort_perm]
    displacement_q = np.concatenate(displacement_blocks, axis=0)[sort_perm]
    bulk_density = np.concatenate(bulk_blocks)[sort_perm]
    return (
        positions,
        quadrature_volumes,
        mass,
        displacement_q,
        bulk_density,
        points_per_cell,
    )


def _face_quadrature(
    field, density, faces, negative, positive, orders: np.ndarray | None = None
):
    nf = len(faces)
    dtype = np.complex128 if np.iscomplexobj(field.displacement_m) else np.float64
    if nf == 0:
        return (
            np.empty((0, 3)),
            np.empty((0, 3)),
            np.empty(0, dtype=dtype),
            np.empty(0, dtype=np.int64),
        )
    face_nodes = field.vertices_m[faces]
    cross = np.cross(
        face_nodes[:, 1] - face_nodes[:, 0], face_nodes[:, 2] - face_nodes[:, 0]
    )
    lengths = np.linalg.norm(cross, axis=1)
    normals = cross / lengths[:, None]
    areas = lengths / 2.0

    neg_locals: list[list[int]] = []
    pos_locals: list[list[int] | None] = []
    jumps = np.empty((nf, 3), dtype=np.float64)
    for face_index, face in enumerate(faces):
        neg = int(negative[face_index])
        neg_local = [int(np.flatnonzero(field.tetra[neg] == node)[0]) for node in face]
        negative_trace = _TRIANGLE_BARYCENTRIC @ density[neg, neg_local]
        pos = int(positive[face_index])
        if pos < 0:
            positive_trace = 0.0
            pos_local = None
        else:
            pos_local = [
                int(np.flatnonzero(field.tetra[pos] == node)[0]) for node in face
            ]
            positive_trace = _TRIANGLE_BARYCENTRIC @ density[pos, pos_local]
        jumps[face_index] = negative_trace - positive_trace
        neg_locals.append(neg_local)
        pos_locals.append(pos_local)

    if orders is None:
        positions = np.einsum("ql,fli->fqi", _TRIANGLE_BARYCENTRIC, face_nodes)
        displacement_q = np.einsum(
            "ql,fli->fqi", _TRIANGLE_BARYCENTRIC, field.displacement_m[faces]
        )
        normal_displacement = np.einsum("fqi,fi->fq", displacement_q, normals)
        weights = (
            (jumps * normal_displacement * (areas / 3.0)[:, None])
            .reshape(-1)
            .astype(dtype)
        )
        points_per_face = np.full(nf, 3, dtype=np.int64)
        return positions.reshape(-1, 3), jumps, weights, points_per_face

    face_orders = orders[negative].copy()
    internal = positive >= 0
    if np.any(internal):
        face_orders[internal] = np.maximum(
            face_orders[internal], orders[positive[internal]]
        )

    unique_orders = np.unique(face_orders)
    points_per_face = np.zeros(nf, dtype=np.int64)
    position_blocks, weight_blocks, face_index_blocks = [], [], []
    for order in unique_orders:
        idx = np.flatnonzero(face_orders == order)
        bary, weights_rule = triangle_gauss_quadrature(int(order))
        q = len(weights_rule)
        points_per_face[idx] = q
        group_nodes = face_nodes[idx]
        pos_q = np.einsum("ql,fli->fqi", bary, group_nodes)
        disp_q = np.einsum("ql,fli->fqi", bary, field.displacement_m[faces[idx]])
        normal_u = np.einsum("fqi,fi->fq", disp_q, normals[idx])
        jump_q = np.empty((len(idx), q), dtype=np.float64)
        for row, face_index in enumerate(idx):
            neg = int(negative[face_index])
            negative_trace_q = bary @ density[neg, neg_locals[face_index]]
            pos_local = pos_locals[face_index]
            if pos_local is None:
                positive_trace_q = 0.0
            else:
                pos = int(positive[face_index])
                positive_trace_q = bary @ density[pos, pos_local]
            jump_q[row] = negative_trace_q - positive_trace_q
        quadrature_area = weights_rule[None, :] * areas[idx, None]
        group_weight = (jump_q * normal_u * quadrature_area).reshape(-1).astype(dtype)
        position_blocks.append(pos_q.reshape(-1, 3))
        weight_blocks.append(group_weight)
        face_index_blocks.append(np.repeat(idx, q))

    order_face_index = np.concatenate(face_index_blocks)
    sort_perm = np.argsort(order_face_index, kind="stable")
    positions = np.concatenate(position_blocks, axis=0)[sort_perm]
    weights_out = np.concatenate(weight_blocks)[sort_perm]
    return positions, jumps, weights_out, points_per_face


@dataclass(frozen=True, init=False)
class TetraP1DensityLinearizedSources:
    """Builder-validated source terms for cell-local affine tetra density."""

    cell_density_nodes_kg_m3: np.ndarray
    mass_positions_m: np.ndarray
    mass_kg: np.ndarray
    mass_displacement_m: np.ndarray
    mass_cell_index: np.ndarray
    bulk_source: VolumeElementSource
    interface_source: WeightedPointSource
    interface_triangles: np.ndarray
    interface_density_jump_kg_m3: np.ndarray
    interface_quadrature_face_index: np.ndarray
    negative_side_cell: np.ndarray
    positive_side_cell: np.ndarray

    def __init__(self):
        raise TypeError("TetraP1DensityLinearizedSources is builder-only")

    @classmethod
    def _from_builder(
        cls,
        *,
        field: UnstructuredDisplacementField,
        quadrature_order: int | np.ndarray | None = None,
        **values,
    ):
        """Validate every companion against the inspected mesh and build a bundle."""
        if not isinstance(field, UnstructuredDisplacementField):
            raise TypeError("field must be an UnstructuredDisplacementField")
        nc = len(field.tetra)
        density = _density_nodes(values["cell_density_nodes_kg_m3"], nc)
        faces_expected, negative_expected, positive_expected = _oriented_face_topology(
            field
        )
        nf = len(faces_expected)
        orders = _resolve_cell_orders(quadrature_order, nc)

        (
            expected_position,
            expected_volume,
            expected_mass,
            expected_u,
            expected_bulk,
            points_per_cell,
        ) = _cell_quadrature(field, density, orders)
        (
            expected_face_position,
            expected_jumps,
            expected_weights,
            points_per_face,
        ) = _face_quadrature(
            field, density, faces_expected, negative_expected, positive_expected, orders
        )
        total_mass_points = int(np.sum(points_per_cell))
        total_face_points = int(np.sum(points_per_face))

        _check_no_bool(values["mass_positions_m"], "mass_positions_m")
        if np.iscomplexobj(np.asarray(values["mass_positions_m"])):
            raise TypeError("mass_positions_m must be real-valued")
        positions = as_xyz_table("mass_positions_m", values["mass_positions_m"])
        _check_no_bool(values["mass_kg"], "mass_kg")
        mass = as_weights("mass_kg", values["mass_kg"])
        if np.iscomplexobj(mass):
            raise TypeError("mass_kg must be real-valued")
        if (
            np.any(mass < 0.0)
            or len(mass) != total_mass_points
            or len(positions) != total_mass_points
        ):
            raise ValueError(
                "mass arrays must match the resolved quadrature point count and be "
                "non-negative"
            )
        displacement_q = _readonly_displacement(
            values["mass_displacement_m"], total_mass_points
        )
        mass_cell = _readonly_integer_vector(
            "mass_cell_index", values["mass_cell_index"], total_mass_points
        )
        if not np.array_equal(mass_cell, np.repeat(np.arange(nc), points_per_cell)):
            raise ValueError(
                "mass_cell_index must match the per-cell quadrature point count"
            )
        bulk = values["bulk_source"]
        interface = values["interface_source"]
        if not isinstance(bulk, VolumeElementSource):
            raise TypeError("bulk_source must be a VolumeElementSource")
        if not isinstance(interface, WeightedPointSource):
            raise TypeError("interface_source must be a WeightedPointSource")
        triangles = _integer_triangles(
            "interface_triangles",
            values["interface_triangles"],
            nf,
            len(field.vertices_m),
        )
        jumps = _real_table(
            "interface_density_jump_kg_m3",
            values["interface_density_jump_kg_m3"],
            (nf, 3),
        )
        face_index = _readonly_integer_vector(
            "interface_quadrature_face_index",
            values["interface_quadrature_face_index"],
            total_face_points,
        )
        negative = _readonly_integer_vector(
            "negative_side_cell", values["negative_side_cell"], nf
        )
        positive = _readonly_integer_vector(
            "positive_side_cell", values["positive_side_cell"], nf
        )
        if not np.array_equal(face_index, np.repeat(np.arange(nf), points_per_face)):
            raise ValueError(
                "interface_quadrature_face_index must match the per-face quadrature "
                "point count"
            )
        if not (
            np.array_equal(triangles, faces_expected)
            and np.array_equal(negative, negative_expected)
            and np.array_equal(positive, positive_expected)
        ):
            raise ValueError("interface topology or owners do not match field")
        if (
            len(bulk.positions_m) != total_mass_points
            or len(interface.positions_m) != total_face_points
        ):
            raise ValueError("source lengths do not match quadrature metadata")

        checks = (
            (positions, expected_position, "mass_positions_m"),
            (mass, expected_mass, "mass_kg"),
            (displacement_q, expected_u, "mass_displacement_m"),
            (bulk.positions_m, expected_position, "bulk_source.positions_m"),
            (bulk.volumes_m3, expected_volume, "bulk_source.volumes_m3"),
            (
                np.asarray(bulk.density_kg_m3),
                expected_bulk,
                "bulk_source.density_kg_m3",
            ),
            (
                interface.positions_m,
                expected_face_position,
                "interface_source.positions_m",
            ),
            (interface.weights_kg, expected_weights, "interface_source.weights_kg"),
            (jumps, expected_jumps, "interface_density_jump_kg_m3"),
        )
        for actual, expected, name in checks:
            if not np.array_equal(actual, expected):
                raise ValueError(f"{name} does not match the field quadrature")

        result = object.__new__(cls)
        normalized = dict(values)
        normalized.update(
            cell_density_nodes_kg_m3=density,
            mass_positions_m=positions,
            mass_kg=mass,
            mass_displacement_m=displacement_q,
            mass_cell_index=mass_cell,
            interface_triangles=triangles,
            interface_density_jump_kg_m3=jumps,
            interface_quadrature_face_index=face_index,
            negative_side_cell=negative,
            positive_side_cell=positive,
        )
        for name in cls.__dataclass_fields__:
            object.__setattr__(result, name, normalized[name])
        return result


def piecewise_tetra_p1_density_sources(
    field: UnstructuredDisplacementField,
    cell_density_nodes_kg_m3: np.ndarray,
    *,
    quadrature_order: int | np.ndarray | None = None,
) -> TetraP1DensityLinearizedSources:
    """Build P1-density mass-displacement and bulk/interface quadratures.

    ``quadrature_order`` selects the tetra/face quadrature rule:

    - ``None`` (default): the fixed classical degree-2 rule (4-point tetra,
      3-point face) -- bit-identical to every call from before this option
      existed.
    - a scalar ``int >= 2``: every cell (and every face) uses a
      Gauss-Legendre Duffy rule (``gragra.geometry.tetra_quadrature``) with
      that many Gauss points per axis, uniformly.
    - an ``(Nc,) int`` array (each entry ``>= 2``): a per-cell Gauss order,
      i.e. local higher-order refinement without changing subcell count or
      geometry. A shared interface face takes the higher (``max``) of its
      two owning cells' orders; a boundary face takes its single owner's
      order.

    Complex outputs represent linear complex amplitudes or
    transfer-function-like responses, not directly real-valued time-domain
    gravitational fields; ``mass_kg`` stays real (mass-element weight only,
    see ``source-types.md``) regardless of ``quadrature_order``.
    """
    if not isinstance(field, UnstructuredDisplacementField):
        raise TypeError("field must be an UnstructuredDisplacementField")
    nc = len(field.tetra)
    density = _density_nodes(cell_density_nodes_kg_m3, nc)
    orders = _resolve_cell_orders(quadrature_order, nc)
    positions, volumes, mass, displacement_q, bulk_density, points_per_cell = (
        _cell_quadrature(field, density, orders)
    )
    faces, negative, positive = _oriented_face_topology(field)
    face_positions, jumps, face_weights, points_per_face = _face_quadrature(
        field, density, faces, negative, positive, orders
    )
    return TetraP1DensityLinearizedSources._from_builder(
        field=field,
        quadrature_order=quadrature_order,
        cell_density_nodes_kg_m3=density,
        mass_positions_m=positions,
        mass_kg=mass,
        mass_displacement_m=displacement_q,
        mass_cell_index=np.repeat(np.arange(nc, dtype=np.int64), points_per_cell),
        bulk_source=VolumeElementSource(positions, volumes, bulk_density),
        interface_source=WeightedPointSource(face_positions, face_weights),
        interface_triangles=faces,
        interface_density_jump_kg_m3=jumps,
        interface_quadrature_face_index=np.repeat(
            np.arange(len(faces), dtype=np.int64), points_per_face
        ),
        negative_side_cell=negative,
        positive_side_cell=positive,
    )


def compare_tetra_p1_density_representations(
    sources: TetraP1DensityLinearizedSources,
    targets,
    *,
    observable: str = "acceleration",
    softening_m: float = 0.0,
    chunk_size: int | None = None,
    backend: str = "numpy",
) -> TetraRepresentationReport:
    """Compare G-free bulk/interface and mass-displacement quadratures."""
    if not isinstance(sources, TetraP1DensityLinearizedSources):
        raise TypeError("sources must be a TetraP1DensityLinearizedSources")
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
        observable, bulk, interface, combined, mass_displacement, absolute, relative
    )


__all__ = [
    "TetraP1DensityLinearizedSources",
    "piecewise_tetra_p1_density_sources",
    "compare_tetra_p1_density_representations",
]
