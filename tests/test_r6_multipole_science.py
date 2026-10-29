"""Scientific acceptance test for R6 finite-body multipole decomposition.

Compares direct numerical gravitational perturbation of a homogeneous cube vs.
a 5-layer multi-density body with identical outer shape, identical total mass,
and identical center of mass.
Validates that the difference in observed fields matches the quadrupole
(second moment) perturbation within 1% relative error independently for:
1. Potential
2. Acceleration
3. Gravity Gradient
"""

import numpy as np

from gragra.adapters.finite_body_mesh import cell_point_mass_cloud_from_group_tags
from gragra.adapters.meshio_adapter import VolumeMesh
from gragra.constants import G_SI
from gragra.observables import (
    point_acceleration,
    point_gravity_gradient,
    point_potential,
)


def _build_5layer_hex_mesh(n_xy: int = 4, n_z_per_layer: int = 2):
    """Build a 2x2x2 m cube centered at origin with 5 z-layers."""
    z_interfaces = np.array([-1.0, -0.6, -0.2, 0.2, 0.6, 1.0])
    x_grid = np.linspace(-1.0, 1.0, n_xy + 1)
    y_grid = np.linspace(-1.0, 1.0, n_xy + 1)

    # Build unique nodes
    all_z = []
    for layer_idx in range(5):
        z_start = z_interfaces[layer_idx]
        z_end = z_interfaces[layer_idx + 1]
        z_sub = np.linspace(z_start, z_end, n_z_per_layer + 1)
        if layer_idx == 0:
            all_z.extend(z_sub)
        else:
            all_z.extend(z_sub[1:])

    z_vals = np.array(all_z)
    _, ny, nz = len(x_grid), len(y_grid), len(z_vals)

    # 3D grid of vertices
    gx, gy, gz = np.meshgrid(x_grid, y_grid, z_vals, indexing="ij")
    vertices = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)

    def node_id(i, j, k):
        return (i * ny + j) * nz + k

    hex_cells = []
    layer_tags = []

    k_offset = 0
    for layer_idx in range(5):
        tag = layer_idx + 1  # 1..5
        for _ in range(n_z_per_layer):
            k = k_offset
            for i in range(n_xy):
                for j in range(n_xy):
                    # VTK hex8 ordering
                    n0 = node_id(i, j, k)
                    n1 = node_id(i + 1, j, k)
                    n2 = node_id(i + 1, j + 1, k)
                    n3 = node_id(i, j + 1, k)
                    n4 = node_id(i, j, k + 1)
                    n5 = node_id(i + 1, j, k + 1)
                    n6 = node_id(i + 1, j + 1, k + 1)
                    n7 = node_id(i, j + 1, k + 1)
                    hex_cells.append([n0, n1, n2, n3, n4, n5, n6, n7])
                    layer_tags.append(tag)
            k_offset += 1

    cells_arr = np.array(hex_cells, dtype=np.int64)
    tags_arr = np.array(layer_tags, dtype=np.int64)

    return VolumeMesh(
        vertices_m=vertices,
        cell_blocks=(("hexahedron", cells_arr),),
        cell_groups=(tags_arr,),
    )


def test_r6_multipole_quadrupole_science():
    """Verify quadrupole perturbation matches direct numerical difference to < 1%."""
    mesh = _build_5layer_hex_mesh(n_xy=6, n_z_per_layer=3)

    # 1. Homogeneous case (rho = 2700 kg/m3 across all 5 layers)
    homo_densities = {1: 2700.0, 2: 2700.0, 3: 2700.0, 4: 2700.0, 5: 2700.0}
    cloud_homo = cell_point_mass_cloud_from_group_tags(mesh, homo_densities)

    # 2. Multi-density case (rho = [1350, 2700, 5400, 2700, 1350])
    multi_densities = {1: 1350.0, 2: 2700.0, 3: 5400.0, 4: 2700.0, 5: 1350.0}
    cloud_multi = cell_point_mass_cloud_from_group_tags(mesh, multi_densities)

    # Verify identical total mass
    m_homo = np.sum(cloud_homo.masses_kg)
    m_multi = np.sum(cloud_multi.masses_kg)
    assert np.isclose(m_homo, 21600.0, rtol=1e-12)
    assert np.isclose(m_multi, 21600.0, rtol=1e-12)

    # Verify identical center of mass
    com_homo = (
        np.sum(cloud_homo.positions_m * cloud_homo.masses_kg[:, None], axis=0) / m_homo
    )
    com_multi = (
        np.sum(cloud_multi.positions_m * cloud_multi.masses_kg[:, None], axis=0)
        / m_multi
    )
    assert np.allclose(com_homo, [0.0, 0.0, 0.0], atol=1e-14)
    assert np.allclose(com_multi, [0.0, 0.0, 0.0], atol=1e-14)

    # Compute Delta I_zz from numerical clouds
    i_zz_homo = np.sum(cloud_homo.masses_kg * (cloud_homo.positions_m[:, 2] ** 2))
    i_zz_multi = np.sum(cloud_multi.masses_kg * (cloud_multi.positions_m[:, 2] ** 2))
    delta_i_zz = i_zz_multi - i_zz_homo

    # Theoretical Delta I_zz = 4435.2 - 7200.0 = -2764.8 kg m^2
    assert np.isclose(delta_i_zz, -2764.8, rtol=1e-3)

    # Traceless quadrupole tensor:
    # Delta Q_xx = Delta Q_yy = -Delta I_zz, Delta Q_zz = 2 Delta I_zz
    delta_q = np.diag([-delta_i_zz, -delta_i_zz, 2.0 * delta_i_zz])

    # Target points at r >= 18 m (outside near-field r >= 9 * L_half)
    targets = np.array(
        [
            [0.0, 0.0, 20.0],
            [20.0, 0.0, 0.0],
            [12.0, 12.0, 18.0],
            [10.0, -12.0, 18.0],
        ],
        dtype=np.float64,
    )

    r_norms = np.linalg.norm(targets, axis=1)

    # Scale reference for zero-near floors: G * |Delta I_zz| / r^3
    scale_potential = G_SI * abs(delta_i_zz) / (r_norms**3)
    scale_accel = G_SI * abs(delta_i_zz) / (r_norms**4)
    scale_grad = G_SI * abs(delta_i_zz) / (r_norms**5)

    # 1. Potential
    v_homo = point_potential(cloud_homo, targets)
    v_multi = point_potential(cloud_multi, targets)
    delta_v_num = v_multi - v_homo

    # Quadrupole potential: V_quad = - 1/2 G sum_jk Q_jk r_j r_k / r^5
    quad_v = np.zeros(len(targets))
    for idx, (t, r) in enumerate(zip(targets, r_norms, strict=True)):
        quad_v[idx] = -0.5 * G_SI * float(t @ delta_q @ t) / (r**5)

    # Individual Potential acceptance check:
    err_v = np.abs(delta_v_num - quad_v)
    tol_v = 0.01 * np.maximum(np.abs(quad_v), 1e-3 * scale_potential)
    assert np.all(err_v <= tol_v), f"Potential error exceeded 1%: {err_v} > {tol_v}"

    # 2. Acceleration: a_quad = -grad(V_quad) = +G ( (Q r) / r^5 - 5/2 r (r Q r) / r^7 )
    a_homo = point_acceleration(cloud_homo, targets)
    a_multi = point_acceleration(cloud_multi, targets)
    delta_a_num = a_multi - a_homo

    quad_a = np.zeros_like(targets)
    for idx, (t, r) in enumerate(zip(targets, r_norms, strict=True)):
        term1 = (delta_q @ t) / (r**5)
        term2 = -2.5 * t * float(t @ delta_q @ t) / (r**7)
        quad_a[idx] = G_SI * (term1 + term2)

    # Individual Acceleration acceptance check:
    err_a = np.linalg.norm(delta_a_num - quad_a, axis=1)
    norm_quad_a = np.linalg.norm(quad_a, axis=1)
    tol_a = 0.01 * np.maximum(norm_quad_a, 1e-3 * scale_accel)
    assert np.all(err_a <= tol_a), f"Acceleration error exceeded 1%: {err_a} > {tol_a}"

    # 3. Gradient: Gamma_quad = grad(a_quad)
    g_homo = point_gravity_gradient(cloud_homo, targets)
    g_multi = point_gravity_gradient(cloud_multi, targets)
    delta_g_num = g_multi - g_homo

    quad_g = np.zeros_like(g_homo)
    for idx, (t, r) in enumerate(zip(targets, r_norms, strict=True)):
        q_r = delta_q @ t
        r_q_r = float(t @ q_r)
        term1 = delta_q / (r**5)
        term2 = (
            -5.0
            * (np.outer(t, q_r) + np.outer(q_r, t) + np.eye(3) * 0.5 * r_q_r)
            / (r**7)
        )
        term3 = 17.5 * np.outer(t, t) * r_q_r / (r**9)
        quad_g[idx] = G_SI * (term1 + term2 + term3)

    # Individual Gradient acceptance check:
    err_g = np.linalg.norm(delta_g_num - quad_g, axis=(-2, -1))
    norm_quad_g = np.linalg.norm(quad_g, axis=(-2, -1))
    tol_g = 0.01 * np.maximum(norm_quad_g, 1e-3 * scale_grad)
    assert np.all(err_g <= tol_g), f"Gradient error exceeded 1%: {err_g} > {tol_g}"
