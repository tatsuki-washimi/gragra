"""M2 numerical checks for Harms factors and the ENIGMA tunnel geometry.

The SCA tensor checks use the G-free interface mass path at the cavity center.
The sphere discretization and cube face quadrature budgets below are fixed
before comparing with the Harms small-cavern coefficient 4*pi/3.
"""

from __future__ import annotations

import numpy as np
import pytest

from gragra.analytic.cavity_coupling import sca_wall_coupling_factor
from gragra.analytic.cavity_sphere import spherical_cavity_wall
from gragra.fields.generic_integration import (
    FieldSampler,
    MassCoupling,
    QuadraturePlan,
)
from gragra.fields.surface import SurfaceDisplacementField
from gragra.integrators.generic import (
    ExecutionPlanner,
    GenericExecutor,
    ScalarSourceBlock,
)
from gragra.kernels.generic_backend import NumpyContractionBackend

_SPHERE_RELATIVE_BUDGET = 5.0e-3
_CUBE_RELATIVE_BUDGET = 5.0e-4
_TUNNEL_DISK_RELATIVE_BUDGET = 1.0e-2


class _ScalarBlockProvider:
    kind = "scalar"
    replayable = True
    random_access = True

    def __init__(self, source: ScalarSourceBlock) -> None:
        self._source = source
        self.source_count = len(source.positions_m)
        self.parameter_count = source.weights_kg.shape[1]
        self.resident_bytes = source.positions_m.nbytes + source.weights_kg.nbytes

    def iter_blocks(self, parameter_slice: slice, source_block_size: int):
        for start in range(0, self.source_count, source_block_size):
            stop = min(start + source_block_size, self.source_count)
            yield ScalarSourceBlock(
                self._source.positions_m[start:stop],
                self._source.weights_kg[start:stop, parameter_slice],
                start,
                parameter_slice,
            )


def _integrate_interface(
    vertices: np.ndarray,
    triangles: np.ndarray,
    *,
    quadrature_order: int,
    displacement_at_vertices: np.ndarray,
    density_jump_kg_m3: float = -1.0,
    target_m: np.ndarray | None = None,
) -> np.ndarray:
    field = SurfaceDisplacementField(vertices, triangles, displacement_at_vertices)
    quadrature = QuadraturePlan.from_surface_mesh(
        vertices, triangles, order=quadrature_order
    )
    sampled = FieldSampler().sample_surface_displacement(field, quadrature)
    source = MassCoupling.interface_density_jump(
        quadrature,
        sampled,
        density_jump_kg_m3=density_jump_kg_m3,
    )
    provider = _ScalarBlockProvider(source)
    targets = np.zeros((1, 3)) if target_m is None else np.asarray(target_m)[None, :]
    plan = ExecutionPlanner.plan(
        provider,
        target_count=1,
        observable="acceleration",
        result_dtype=source.weights_kg.dtype,
        backend_workspace_bytes=0,
        target_block_size=1,
        parameter_block_size=1,
        source_block_size=2048,
        kernel_chunk_size=1024,
        memory_budget_bytes=64 * 1024**2,
    )
    result = np.empty(plan.output_shape, dtype=plan.result_dtype)
    for target_slice, parameter_slice, values in GenericExecutor.iter_blocks(
        provider, targets, plan, NumpyContractionBackend()
    ):
        result[target_slice, parameter_slice] = values
    return result[0, 0]


def _sphere_mesh(latitude_steps: int = 16, azimuth_steps: int = 32):
    vertices = [np.array([0.0, 0.0, 1.0])]
    rings: list[list[int]] = []
    for j in range(1, latitude_steps):
        theta = np.pi * j / latitude_steps
        ring = []
        for k in range(azimuth_steps):
            phi = 2.0 * np.pi * k / azimuth_steps
            ring.append(len(vertices))
            vertices.append(
                np.array(
                    [
                        np.sin(theta) * np.cos(phi),
                        np.sin(theta) * np.sin(phi),
                        np.cos(theta),
                    ]
                )
            )
        rings.append(ring)
    bottom = len(vertices)
    vertices.append(np.array([0.0, 0.0, -1.0]))

    triangles: list[list[int]] = []
    for k in range(azimuth_steps):
        following = (k + 1) % azimuth_steps
        triangles.append([0, rings[0][k], rings[0][following]])
        for j in range(len(rings) - 1):
            upper, lower = rings[j], rings[j + 1]
            triangles.append([upper[k], lower[k], lower[following]])
            triangles.append([upper[k], lower[following], upper[following]])
        triangles.append([bottom, rings[-1][following], rings[-1][k]])

    points = np.asarray(vertices)
    cells = np.asarray(triangles, dtype=np.int64)
    coords = points[cells]
    outward = np.einsum(
        "ij,ij->i",
        np.cross(coords[:, 1] - coords[:, 0], coords[:, 2] - coords[:, 0]),
        np.sum(coords, axis=1),
    )
    inward_cells = np.flatnonzero(outward < 0.0)
    cells[inward_cells, 1], cells[inward_cells, 2] = (
        cells[inward_cells, 2].copy(),
        cells[inward_cells, 1].copy(),
    )
    return points, cells


def _cube_mesh(subdivisions: int = 8):
    vertex_ids: dict[tuple[int, int, int], int] = {}
    vertices: list[list[float]] = []
    triangles: list[list[int]] = []

    def vertex_id(index: tuple[int, int, int]) -> int:
        if index not in vertex_ids:
            vertex_ids[index] = len(vertices)
            vertices.append([coordinate / subdivisions for coordinate in index])
        return vertex_ids[index]

    coordinate_values = np.linspace(-subdivisions, subdivisions, subdivisions + 1)
    integer_values = np.rint(coordinate_values).astype(int)
    for axis in range(3):
        tangential = [candidate for candidate in range(3) if candidate != axis]
        for sign in (-1, 1):
            for i in range(subdivisions):
                for j in range(subdivisions):
                    q = []
                    for du, dv in (
                        (0, 0),
                        (1, 0),
                        (1, 1),
                        (0, 1),
                    ):
                        index = [0, 0, 0]
                        index[axis] = sign * subdivisions
                        index[tangential[0]] = integer_values[i + du]
                        index[tangential[1]] = integer_values[j + dv]
                        q.append(vertex_id(tuple(index)))
                    points = np.asarray(vertices)[q]
                    normal = np.cross(points[1] - points[0], points[2] - points[0])
                    if normal[axis] * sign < 0.0:
                        q = [q[0], q[3], q[2], q[1]]
                    triangles.extend(([q[0], q[1], q[2]], [q[0], q[2], q[3]]))
    return np.asarray(vertices), np.asarray(triangles, dtype=np.int64)


def _open_tunnel_mesh(
    *,
    radius_m: float = 5.0,
    x_min_m: float = -5.0,
    x_max_m: float = 50.0,
    azimuth_steps: int = 24,
    axial_steps: int = 32,
    radial_steps: int = 16,
):
    """Rebuild the finite tunnel with only its x=-5 m end face closed."""
    vertices: list[list[float]] = []
    vertex_ids: dict[tuple[float, float, float], int] = {}
    triangles: list[list[int]] = []

    def vertex_id(point: np.ndarray) -> int:
        key = tuple(float(value) for value in np.round(point, 12))
        if key not in vertex_ids:
            vertex_ids[key] = len(vertices)
            vertices.append(list(point))
        return vertex_ids[key]

    x_values = np.linspace(x_min_m, x_max_m, axial_steps + 1)
    angles = 2.0 * np.pi * np.arange(azimuth_steps) / azimuth_steps
    side_ids = np.empty((axial_steps + 1, azimuth_steps), dtype=np.int64)
    for ix, x_value in enumerate(x_values):
        for ia, angle in enumerate(angles):
            point = np.array(
                [x_value, radius_m * np.cos(angle), radius_m * np.sin(angle)]
            )
            side_ids[ix, ia] = vertex_id(point)
    for ix in range(axial_steps):
        for ia in range(azimuth_steps):
            following = (ia + 1) % azimuth_steps
            quad = [
                side_ids[ix, ia],
                side_ids[ix + 1, ia],
                side_ids[ix + 1, following],
                side_ids[ix, following],
            ]
            for cell in ([quad[0], quad[1], quad[2]], [quad[0], quad[2], quad[3]]):
                points = np.asarray(vertices)[cell]
                radial = np.array([0.0, points[:, 1].mean(), points[:, 2].mean()])
                normal = np.cross(points[1] - points[0], points[2] - points[0])
                if np.dot(normal, radial) < 0.0:
                    cell[1], cell[2] = cell[2], cell[1]
                triangles.append(cell)

    center_id = vertex_id(np.array([x_min_m, 0.0, 0.0]))
    rings: list[list[int]] = []
    for ir in range(1, radial_steps + 1):
        ring_radius = radius_m * ir / radial_steps
        ring = []
        for angle in angles:
            ring.append(
                vertex_id(
                    np.array(
                        [
                            x_min_m,
                            ring_radius * np.cos(angle),
                            ring_radius * np.sin(angle),
                        ]
                    )
                )
            )
        rings.append(ring)
    for ia in range(azimuth_steps):
        following = (ia + 1) % azimuth_steps
        triangles.append([center_id, rings[0][following], rings[0][ia]])
        for ir in range(len(rings) - 1):
            inner, outer = rings[ir], rings[ir + 1]
            for cell in (
                [inner[ia], outer[following], outer[ia]],
                [inner[ia], inner[following], outer[following]],
            ):
                points = np.asarray(vertices)[cell]
                normal = np.cross(points[1] - points[0], points[2] - points[0])
                if normal[0] > 0.0:
                    cell[1], cell[2] = cell[2], cell[1]
                triangles.append(cell)

    points = np.asarray(vertices, dtype=np.float64)
    cells = np.asarray(triangles, dtype=np.int64)
    return points, cells


def test_harms_sphere_arbitrary_center_matches_surface_integral() -> None:
    vertices, triangles = _sphere_mesh()
    displacement = np.tile([0.0, 0.0, 1.0], (len(vertices), 1))

    acceleration = _integrate_interface(
        vertices,
        triangles,
        quadrature_order=6,
        displacement_at_vertices=displacement,
    )
    expected = sca_wall_coupling_factor("sphere", "arbitrary") * np.array(
        [0.0, 0.0, -1.0]
    )

    np.testing.assert_allclose(
        acceleration,
        expected,
        rtol=_SPHERE_RELATIVE_BUDGET,
        atol=1e-12,
    )


def test_harms_cube_center_3d_matches_surface_integral() -> None:
    vertices, triangles = _cube_mesh()
    displacement = np.tile([1.0, 0.0, 0.0], (len(vertices), 1))

    acceleration = _integrate_interface(
        vertices,
        triangles,
        quadrature_order=6,
        displacement_at_vertices=displacement,
    )
    expected = sca_wall_coupling_factor("cube", "center_3d") * np.array(
        [-1.0, 0.0, 0.0]
    )

    np.testing.assert_allclose(
        acceleration,
        expected,
        rtol=_CUBE_RELATIVE_BUDGET,
        atol=1e-12,
    )


def test_enigma_open_tunnel_axis_p_mode_wall_is_below_sphere() -> None:
    """Check the slide's inequality for one explicit low-frequency P mode.

    The open end is intentional. The result is a new generic-surface
    calculation for this declared mode, not a reproduction of legacy output
    or a comparison with the closed-cylinder SCA tensor. The separate legacy
    numerical comparison is recorded in the M2 evidence ledger.
    """
    frequency_hz = 1.0
    p_velocity_m_s = 5500.0
    wavenumber = 2.0 * np.pi * frequency_hz / p_velocity_m_s
    quadrature_order = 4

    tunnel_vertices, tunnel_triangles = _open_tunnel_mesh()
    tunnel_displacement = np.zeros((len(tunnel_vertices), 3), dtype=np.complex128)
    tunnel_displacement[:, 0] = np.exp(1j * wavenumber * tunnel_vertices[:, 0])
    tunnel_acceleration = _integrate_interface(
        tunnel_vertices,
        tunnel_triangles,
        quadrature_order=quadrature_order,
        displacement_at_vertices=tunnel_displacement,
    )

    sphere_vertices, sphere_triangles = _sphere_mesh()
    sphere_vertices = 5.0 * sphere_vertices
    sphere_displacement = np.zeros((len(sphere_vertices), 3), dtype=np.complex128)
    sphere_displacement[:, 0] = np.exp(1j * wavenumber * sphere_vertices[:, 0])
    sphere_acceleration = _integrate_interface(
        sphere_vertices,
        sphere_triangles,
        quadrature_order=quadrature_order,
        displacement_at_vertices=sphere_displacement,
    )
    sphere_reference = float(
        spherical_cavity_wall(5.0, wavenumber, 1.0, 1.0, wave_type="P")
    )
    # For this axial P mode the tunnel's cylindrical side has xi·n=0. The
    # one closed disk at x=-5 m is therefore the full wall source. Integrating
    # its uniform sheet analytically gives this independent finite-geometry
    # reference; it is not the closed-cylinder SCA tensor.
    tunnel_reference = (
        -2.0 * np.pi * (1.0 - 1.0 / np.sqrt(2.0)) * np.exp(-1j * wavenumber * 5.0)
    )

    assert abs(sphere_acceleration[0]) == pytest.approx(
        abs(sphere_reference), rel=_SPHERE_RELATIVE_BUDGET
    )
    np.testing.assert_allclose(
        tunnel_acceleration[0],
        tunnel_reference,
        rtol=_TUNNEL_DISK_RELATIVE_BUDGET,
        atol=1e-12,
    )
    assert abs(tunnel_acceleration[0]) < 0.5 * abs(sphere_acceleration[0])
    assert abs(tunnel_acceleration[1:]).max() < 1e-11


@pytest.mark.parametrize(
    ("shape", "location", "expected"),
    [
        ("cube", "floor_center_2d", 2.8),
        ("half_sphere", "floor_center_2d", 2.1),
        ("elongated_cuboid_2to1to1", "floor_center_2d", 1.3),
    ],
)
def test_harms_floor_rows_remain_literal_lookup_regressions(shape, location, expected):
    """The three floor rows are literal regressions, not tensor predictions."""
    assert sca_wall_coupling_factor(shape, location) == pytest.approx(expected)
