"""Phase 12-b Slice 2: filled-cuboid internal field tests.

Design: docs/design/phase12b_sca_cavity_coupling.md; derivation/CAS
verification record: docs/references/02_NN_Modeling_and_Simulation/
02-1_Analytical/Nagy_1966_rectangular_prism_gravity.md.
"""

import numpy as np
import pytest

from gragra.analytic import cuboid_filled_internal_field, sca_wall_coupling_factor

RHO = 1234.5  # kg/m^3, arbitrary positive density (matches CAS script)
EPS = 1e-6


def _numeric_gradient(lengths, point, rho, eps=EPS):
    """Central-difference gradient d(a_i)/d(x_j) at a single point."""
    grad = np.zeros((3, 3))
    for j in range(3):
        dp = np.zeros(3)
        dp[j] = eps
        gp = cuboid_filled_internal_field(lengths, np.array([point + dp]), rho)[0]
        gm = cuboid_filled_internal_field(lengths, np.array([point - dp]), rho)[0]
        grad[:, j] = (gp - gm) / (2 * eps)
    return grad


def test_golden_matches_direct_integration_exterior():
    pytest.importorskip("scipy")
    from scipy import integrate

    lengths = np.array([3.0, 2.0, 1.5])
    targets = np.array([[2.4, -1.7, 1.9], [5.0, 4.0, -6.0]])

    got = cuboid_filled_internal_field(lengths, targets, RHO)

    hx, hy, hz = lengths / 2.0
    for i, t in enumerate(targets):
        tx, ty, tz = t

        def integrand(z, y, x, comp, tx=tx, ty=ty, tz=tz):
            d = np.array([x - tx, y - ty, z - tz])
            return d[comp] / np.linalg.norm(d) ** 3

        want = []
        for comp in range(3):
            val, _ = integrate.tplquad(
                integrand,
                -hx,
                hx,
                -hy,
                hy,
                -hz,
                hz,
                args=(comp,),
                epsabs=1e-12,
                epsrel=1e-11,
            )
            want.append(RHO * val)
        np.testing.assert_allclose(got[i], want, rtol=1e-7)


def test_cube_center_gradient_matches_table1_inversion():
    """Inversion principle: -grad(a_filled) at the cube center == Table 1
    cube/center_3d coupling factor (4*pi/3) times rho, on the diagonal."""
    a = 2.0
    cube = np.array([a, a, a])
    grad = _numeric_gradient(cube, np.zeros(3), RHO)

    factor = sca_wall_coupling_factor("cube", "center_3d")
    want = -factor * RHO * np.eye(3)
    np.testing.assert_allclose(grad, want, rtol=1e-6, atol=1e-4 * RHO)


def test_poisson_divergence_interior():
    lengths = np.array([3.0, 2.0, 1.5])
    point = np.array([0.7, -0.4, 0.3])
    grad = _numeric_gradient(lengths, point, RHO)
    np.testing.assert_allclose(np.trace(grad), -4.0 * np.pi * RHO, rtol=1e-6)


def test_far_field_point_mass_asymptote():
    lengths = np.array([3.0, 2.0, 1.5])
    volume = np.prod(lengths)
    far = np.array([[120.0, -80.0, 95.0]])
    r = np.linalg.norm(far[0])

    got = cuboid_filled_internal_field(lengths, far, RHO)[0]
    want = RHO * volume * (-far[0]) / r**3
    np.testing.assert_allclose(got, want, rtol=2e-3)


def test_symmetry_center_field_is_zero():
    lengths = np.array([3.0, 2.0, 1.5])
    got = cuboid_filled_internal_field(lengths, np.zeros((1, 3)), RHO)
    np.testing.assert_allclose(got, np.zeros((1, 3)), atol=1e-10)


def test_symmetry_opposite_points_are_antisymmetric():
    lengths = np.array([3.0, 2.0, 1.5])
    p = np.array([0.9, -0.5, 0.2])
    got_pos = cuboid_filled_internal_field(lengths, np.array([p]), RHO)[0]
    got_neg = cuboid_filled_internal_field(lengths, np.array([-p]), RHO)[0]
    np.testing.assert_allclose(got_neg, -got_pos, rtol=1e-12)


@pytest.mark.parametrize(
    "point",
    [
        [1.5, 0.0, 0.0],  # exactly on a face
        [1.5, 1.0, 0.0],  # exactly on an edge
        [1.5, 1.0, 0.75],  # exactly at a vertex
        [1.5, 1.0, -0.75],  # another vertex
        [0.0, 0.0, 0.0],  # center
    ],
)
def test_face_edge_vertex_finite(point):
    lengths = np.array([3.0, 2.0, 1.5])
    got = cuboid_filled_internal_field(lengths, np.array([point]), RHO)
    assert np.all(np.isfinite(got))


def test_rejects_non_positive_lengths():
    targets = np.array([[1.0, 1.0, 1.0]])
    with pytest.raises(ValueError):
        cuboid_filled_internal_field([1.0, -1.0, 1.0], targets, RHO)
    with pytest.raises(ValueError):
        cuboid_filled_internal_field([1.0, 0.0, 1.0], targets, RHO)


def test_rejects_wrong_lengths_shape():
    targets = np.array([[1.0, 1.0, 1.0]])
    with pytest.raises(ValueError):
        cuboid_filled_internal_field([1.0, 1.0], targets, RHO)


def test_rejects_non_finite_points():
    lengths = np.array([1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        cuboid_filled_internal_field(lengths, np.array([[np.nan, 0.0, 0.0]]), RHO)
    with pytest.raises(ValueError):
        cuboid_filled_internal_field(lengths, np.array([[np.inf, 0.0, 0.0]]), RHO)


def test_rejects_single_point_shape_3():
    """points_m must be a strict (M, 3) table, not a (3,) vector."""
    lengths = np.array([1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        cuboid_filled_internal_field(lengths, [1.0, 1.0, 1.0], RHO)


@pytest.mark.parametrize("density_kg_m3", [0.0, -1.0, float("nan"), float("inf")])
def test_rejects_invalid_density(density_kg_m3):
    lengths = np.array([1.0, 1.0, 1.0])
    targets = np.array([[1.0, 1.0, 1.0]])
    with pytest.raises(ValueError):
        cuboid_filled_internal_field(lengths, targets, density_kg_m3)


def test_output_read_only():
    lengths = np.array([1.0, 1.0, 1.0])
    targets = np.array([[1.5, 0.3, -0.2]])
    out = cuboid_filled_internal_field(lengths, targets, RHO)
    assert out.flags.writeable is False
    assert out.dtype == np.float64
    assert out.shape == (1, 3)
