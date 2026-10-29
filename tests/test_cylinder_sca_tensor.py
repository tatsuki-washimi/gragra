"""Independent outward surface integration for the cylindrical SCA tensor."""

import numpy as np
import pytest

from gragra.analytic import (
    cylinder_sca_wall_coupling_tensor_g_free,
    sca_wall_acceleration_from_tensor_g_free,
)
from gragra.observables import sca_wall_acceleration_from_tensor

TARGETS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.2, 0.3, 0.1],
        [1e-5, -2e-5, 0.7],
        [0.0, 0.0, 0.7],
        [1.6, 0.0, 0.4],
        [0.4, 0.2, 1.2],
    ]
)


def surface_tensor(
    targets, *, radius=2.0, height=3.0, order=64, center=(0.0, 0.0, 0.0), rotation=None
):
    """Direct vector surface integrand, no static-field or production nodes.

    Integrate caps in area coordinate r² and the side in z. Transform
    source coordinates and outward normals to world before evaluating.
    """
    x, w = np.polynomial.legendre.leggauss(order)
    angle = (np.arange(6 * order) + 0.5) * (2 * np.pi / (6 * order))
    ca, sa = np.cos(angle), np.sin(angle)
    nodes, normals, areas = [], [], []
    for z, weight in zip(height * x / 2, height * w / 2, strict=True):
        nodes.append(np.column_stack((radius * ca, radius * sa, np.full_like(ca, z))))
        normals.append(np.column_stack((ca, sa, np.zeros_like(ca))))
        areas.append(np.full_like(ca, radius * weight * 2 * np.pi / len(angle)))
    # dS=r dr dphi=(R²/4) dx dphi, with r=R sqrt((x+1)/2).
    for sign in (-1.0, 1.0):
        for r, weight in zip(
            radius * np.sqrt((x + 1) / 2), radius**2 * w / 4, strict=True
        ):
            nodes.append(
                np.column_stack((r * ca, r * sa, np.full_like(ca, sign * height / 2)))
            )
            normal = np.zeros((len(angle), 3))
            normal[:, 2] = sign
            normals.append(normal)
            areas.append(np.full_like(ca, weight * 2 * np.pi / len(angle)))
    q = np.eye(3) if rotation is None else rotation
    nodes = np.concatenate(nodes) @ q.T + center
    normals = np.concatenate(normals) @ q.T
    areas = np.concatenate(areas)
    result = []
    for point in np.asarray(targets):
        delta = nodes - point
        result.append(
            np.einsum(
                "ni,nj,n->ij",
                delta,
                normals,
                areas / np.linalg.norm(delta, axis=1) ** 3,
            )
        )
    return np.asarray(result)


def test_tensor_against_independent_surface_targetwise():
    actual = cylinder_sca_wall_coupling_tensor_g_free(
        [0.0, 0.0, 0.0], 2.0, 3.0, TARGETS
    )
    low = surface_tensor(TARGETS, order=48)
    high = surface_tensor(TARGETS, order=80)
    width = 1e-12 + 1e-9 * np.linalg.norm(high, axis=(1, 2))
    assert np.all(np.linalg.norm(high - low, axis=(1, 2)) <= width / 10)
    assert np.all(np.linalg.norm(actual - high, axis=(1, 2)) <= width)
    np.testing.assert_allclose(
        np.trace(actual, axis1=-2, axis2=-1), 4 * np.pi, rtol=2e-14
    )
    np.testing.assert_allclose(actual, np.swapaxes(actual, -1, -2), atol=2e-14)
    assert abs(actual[1, 0, 2]) > 1e-3
    assert not actual.flags.writeable
    assert actual.dtype == np.float64


def test_axis_projection_and_center_anisotropy():
    from gragra.analytic import cylinder_filled_axis_field

    z = np.array([0.0, 0.4, -0.7])
    points = np.column_stack((np.zeros_like(z), np.zeros_like(z), z))
    actual = cylinder_sca_wall_coupling_tensor_g_free([0, 0, 0], 2.0, 3.0, points)
    # Independent elementary on-axis cap solid angles, not Table 1.
    h = 1.5
    zz = (
        2
        * np.pi
        * (
            2
            - (h - z) / np.sqrt(4 + (h - z) ** 2)
            - (h + z) / np.sqrt(4 + (h + z) ** 2)
        )
    )
    transverse = (4 * np.pi - zz) / 2
    np.testing.assert_allclose(actual[:, 2, 2], zz, rtol=2e-14)
    np.testing.assert_allclose(actual[:, 0, 0], transverse, rtol=2e-14)
    np.testing.assert_allclose(actual[:, 1, 1], transverse, rtol=2e-14)
    # Center is axially symmetric but not isotropic for R=2,H=3.
    assert abs(actual[0, 2, 2] - actual[0, 0, 0]) > 1.0
    # Existing axis scalar filled field supplies the limiting derivative.
    eps = 1e-4
    plus = cylinder_filled_axis_field(2.0, 3.0, z + eps, 1.0)
    minus = cylinder_filled_axis_field(2.0, 3.0, z - eps, 1.0)
    np.testing.assert_allclose(-(plus - minus) / (2 * eps), actual[:, 2, 2], rtol=2e-9)


def test_rotation_translation_and_wall_contraction():
    from gragra import G_SI

    theta = 0.7
    q = np.array(
        [
            [np.cos(theta), 0.0, np.sin(theta)],
            [0.0, 1.0, 0.0],
            [-np.sin(theta), 0.0, np.cos(theta)],
        ]
    )
    center = np.array([2.0, -3.0, 4.0])
    world = TARGETS @ q.T + center
    snapshots = [value.copy() for value in (q, center, world)]
    actual = cylinder_sca_wall_coupling_tensor_g_free(
        center, 2.0, 3.0, world, rotation_matrix=q
    )
    local = cylinder_sca_wall_coupling_tensor_g_free([0, 0, 0], 2.0, 3.0, TARGETS)
    np.testing.assert_allclose(actual, q @ local @ q.T, rtol=1e-12, atol=1e-13)
    ref = surface_tensor(world, center=center, rotation=q, order=80)
    width = 1e-12 + 1e-9 * np.linalg.norm(ref, axis=(1, 2))
    assert np.all(np.linalg.norm(actual - ref, axis=(1, 2)) <= width)
    u = np.array([1e-6 + 2e-6j, -2e-6j, 3e-6])
    rho = 2700.0
    g_free = sca_wall_acceleration_from_tensor_g_free(actual, rho, u)
    wall = sca_wall_acceleration_from_tensor(actual, rho, u)
    np.testing.assert_allclose(g_free, -rho * np.einsum("mij,j->mi", actual, u))
    np.testing.assert_allclose(wall, G_SI * g_free)
    reference_wall = -G_SI * rho * np.einsum("mij,j->mi", ref, u)
    np.testing.assert_allclose(wall, reference_wall, rtol=1e-9, atol=1e-23)
    for original, saved in zip((q, center, world), snapshots, strict=True):
        np.testing.assert_array_equal(original, saved)


@pytest.mark.parametrize("point", [[2, 0, 0], [0, 0, 1.5], [2, 0, 1.5]])
def test_boundary_rejected(point):
    with pytest.raises(ValueError, match="physical cylinder boundary"):
        cylinder_sca_wall_coupling_tensor_g_free([0, 0, 0], 2.0, 3.0, [point])


@pytest.mark.parametrize(
    "rotation",
    [
        np.diag([1.0, 1.0, -1.0]),
        np.diag([1.0, 1.0, 2.0]),
        np.full((3, 3), np.nan),
        np.eye(2),
    ],
)
def test_rotation_rejected(rotation):
    with pytest.raises(ValueError):
        cylinder_sca_wall_coupling_tensor_g_free(
            [0, 0, 0], 2.0, 3.0, TARGETS, rotation_matrix=rotation
        )


def test_shape_finiteness_and_source_validation():
    for radius, height in [(-1, 3), (2, 0), (np.inf, 3), (2, np.nan)]:
        with pytest.raises(ValueError):
            cylinder_sca_wall_coupling_tensor_g_free([0, 0, 0], radius, height, TARGETS)
    with pytest.raises(TypeError):
        cylinder_sca_wall_coupling_tensor_g_free([0, 0, 0], True, 3.0, TARGETS)
    with pytest.raises(ValueError):
        cylinder_sca_wall_coupling_tensor_g_free([0, 0, 0], 2.0, 3.0, [[0, 0, np.nan]])
    with pytest.raises(TypeError):
        cylinder_sca_wall_coupling_tensor_g_free([0, 0, 0], 2.0, 3.0, [[0, 0, 1j]])


def test_inherits_exterior_and_no_quadrature_production(monkeypatch):
    import gragra.analytic.cavity_cylinder as cylinder

    def forbidden(*_args, **_kwargs):
        raise AssertionError("production used spatial quadrature")

    monkeypatch.setattr(cylinder, "_cylinder_boundary_quadrature", forbidden)
    actual = cylinder_sca_wall_coupling_tensor_g_free(
        [0, 0, 0], 2.0, 3.0, [[2.8, 0.2, 2.0]]
    )
    assert abs(np.trace(actual[0])) < 1e-13
    ref = surface_tensor([[2.8, 0.2, 2.0]], order=80)
    np.testing.assert_allclose(actual, ref, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("scale", [0.01, 100.0])
def test_dimensionless_uniform_scale_invariance(scale):
    baseline = cylinder_sca_wall_coupling_tensor_g_free([0, 0, 0], 2.0, 3.0, TARGETS)
    scaled = cylinder_sca_wall_coupling_tensor_g_free(
        [0, 0, 0], 2.0 * scale, 3.0 * scale, TARGETS * scale
    )
    np.testing.assert_allclose(scaled, baseline, rtol=2e-12, atol=1e-13)
