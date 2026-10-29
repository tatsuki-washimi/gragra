"""Tests for SurfaceDisplacementField (Phase 10-c-c).

Surface/interface displacement -> effective surface mass perturbation
``delta_sigma = delta_rho * (u . n_hat)``. Analytic-first per testing.md.

The key sign-pinning test is a bulk+surface mass-conservation cross-check on a
cube: a uniform isotropic expansion ``u = alpha * r`` produces a uniform bulk
perturbation ``delta_rho = -3*alpha*rho0`` (via the 10-c-b tetra helper) and a
surface perturbation ``delta_sigma = rho0 * (u . n_hat)`` on the boundary (this
helper). By the divergence theorem the two integrated masses must cancel
exactly, which fixes both the +/- of ``delta_rho`` and the relative sign of the
bulk and surface terms.
"""

import subprocess
import sys

import numpy as np
import pytest

from gragra.fields.surface import SurfaceDisplacementField
from gragra.fields.unstructured import UnstructuredDisplacementField

# Cube surface/volume mesh fixtures (cube_vertices / cube_triangles /
# cube_tetra / cube_half) live in tests/conftest.py as a single source of truth.


# ---------------------------------------------------------------------------
# Single-triangle exact value
# ---------------------------------------------------------------------------


def test_single_triangle_exact_delta_sigma_and_mass():
    # Triangle in the z=0 plane: normal = +z, area = 0.5.
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
    )
    triangles = np.array([[0, 1, 2]], dtype=np.int64)
    # Nodal displacements purely along z: mean z = (1+2+3)/3 = 2.0 -> u_n = 2.0.
    displacement = np.array(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]], dtype=np.float64
    )
    delta_rho = 5.0

    field = SurfaceDisplacementField(
        vertices_m=vertices, triangles=triangles, displacement_m=displacement
    )
    source = field.as_surface_mass_source(delta_rho)

    # u_n = mean(u) . n_hat = 2.0 ; delta_sigma = delta_rho * u_n = 10.0
    np.testing.assert_allclose(source.normal_displacement_m, 2.0)
    np.testing.assert_allclose(source.areas_m2, 0.5)
    assert source.density_kg_m3 == 5.0
    # delta_m = delta_sigma * area = 10.0 * 0.5 = 5.0
    weights = source.as_weighted_source().weights_kg
    np.testing.assert_allclose(weights, 5.0)


# ---------------------------------------------------------------------------
# Bulk + surface mass conservation (sign-pinning, cube, exact)
# ---------------------------------------------------------------------------


def test_bulk_plus_surface_mass_conservation_cube(
    cube_vertices, cube_triangles, cube_tetra, cube_half
):
    half = cube_half
    alpha = 1.0e-3
    rho0 = 2700.0
    vertices = cube_vertices
    # Uniform isotropic expansion u = alpha * r at every vertex.
    displacement = alpha * vertices

    surf = SurfaceDisplacementField(
        vertices_m=vertices,
        triangles=cube_triangles,
        displacement_m=displacement,
    )
    bulk = UnstructuredDisplacementField(
        vertices_m=vertices, tetra=cube_tetra, displacement_m=displacement
    )

    m_surf = surf.as_surface_mass_source(rho0).as_weighted_source().weights_kg.sum()
    bulk_source = bulk.as_density_perturbation_source(rho0)
    m_bulk = bulk_source.as_weighted_source().weights_kg.sum()

    cube_volume = (2.0 * half) ** 3
    np.testing.assert_allclose(m_bulk, -3.0 * alpha * rho0 * cube_volume, rtol=1e-12)
    np.testing.assert_allclose(m_surf, +3.0 * alpha * rho0 * cube_volume, rtol=1e-12)
    # The integrated perturbed mass must cancel (divergence theorem).
    np.testing.assert_allclose(m_bulk + m_surf, 0.0, atol=1e-9)


def test_closed_surface_uniform_translation_zero_net_mass(
    cube_vertices, cube_triangles
):
    # Uniform translation has sum(A * n_hat) = 0 over a closed surface, so the
    # net surface mass is zero regardless of the (sign of the) jump.
    vertices = cube_vertices
    displacement = np.tile(np.array([1.0, 2.0, 3.0]), (len(vertices), 1))
    surf = SurfaceDisplacementField(
        vertices_m=vertices, triangles=cube_triangles, displacement_m=displacement
    )
    m = surf.as_surface_mass_source(2700.0).as_weighted_source().weights_kg.sum()
    np.testing.assert_allclose(m, 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# Sign / orientation
# ---------------------------------------------------------------------------


def test_sign_outward_displacement_positive_jump_gives_positive_sigma():
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
    )
    triangles = np.array([[0, 1, 2]], dtype=np.int64)  # normal +z
    displacement = np.tile(np.array([0.0, 0.0, 1.0]), (3, 1))  # u . n_hat = +1
    pos = SurfaceDisplacementField(vertices, triangles, displacement)
    sigma_pos = pos.as_surface_mass_source(3.0).normal_displacement_m
    assert np.real(sigma_pos).item() > 0.0
    # Flip displacement -> negative u_n.
    neg = SurfaceDisplacementField(vertices, triangles, -displacement)
    sigma_neg = neg.as_surface_mass_source(3.0).normal_displacement_m
    assert np.real(sigma_neg).item() < 0.0


def test_winding_reversal_flips_normal_displacement_sign():
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
    )
    displacement = np.tile(np.array([0.0, 0.0, 1.0]), (3, 1))
    fwd = SurfaceDisplacementField(vertices, np.array([[0, 1, 2]]), displacement)
    rev = SurfaceDisplacementField(vertices, np.array([[0, 2, 1]]), displacement)
    u_fwd = fwd.as_surface_mass_source(1.0).normal_displacement_m
    u_rev = rev.as_surface_mass_source(1.0).normal_displacement_m
    np.testing.assert_allclose(u_fwd, -u_rev)
    np.testing.assert_allclose(np.abs(u_fwd), np.abs(u_rev))


# ---------------------------------------------------------------------------
# Complex / dtype / read-only
# ---------------------------------------------------------------------------


def test_complex_displacement_gives_complex_output():
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
    )
    triangles = np.array([[0, 1, 2]], dtype=np.int64)
    displacement = np.tile(np.array([0.0, 0.0, 1.0 + 2.0j]), (3, 1))
    source = SurfaceDisplacementField(
        vertices, triangles, displacement
    ).as_surface_mass_source(1.0)
    assert np.iscomplexobj(np.asarray(source.normal_displacement_m))
    np.testing.assert_allclose(source.normal_displacement_m, 1.0 + 2.0j)


def test_returned_source_arrays_are_read_only(cube_vertices, cube_triangles):
    vertices = cube_vertices
    displacement = 1e-3 * vertices
    source = SurfaceDisplacementField(
        vertices, cube_triangles, displacement
    ).as_surface_mass_source(2700.0)
    assert source.positions_m.flags.writeable is False
    assert source.normals.flags.writeable is False
    assert source.areas_m2.flags.writeable is False
    assert np.asarray(source.normal_displacement_m).flags.writeable is False


# ---------------------------------------------------------------------------
# Empty mesh
# ---------------------------------------------------------------------------


def test_empty_mesh_returns_empty_source(cube_vertices):
    vertices = cube_vertices
    displacement = 1e-3 * vertices
    triangles = np.zeros((0, 3), dtype=np.int64)
    source = SurfaceDisplacementField(
        vertices, triangles, displacement
    ).as_surface_mass_source(2700.0)
    assert source.positions_m.shape == (0, 3)
    assert source.areas_m2.shape == (0,)
    assert source.as_weighted_source().weights_kg.shape == (0,)


def test_empty_mesh_complex_displacement_keeps_complex_dtype(cube_vertices):
    vertices = cube_vertices
    displacement = (1e-3 * vertices).astype(np.complex128)
    triangles = np.zeros((0, 3), dtype=np.int64)
    source = SurfaceDisplacementField(
        vertices, triangles, displacement
    ).as_surface_mass_source(2700.0)
    assert np.asarray(source.normal_displacement_m).dtype == np.complex128


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _good_tri_vertices():
    return np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
    )


def test_vertices_non_finite_raises():
    v = _good_tri_vertices()
    v[0, 0] = np.inf
    with pytest.raises(ValueError):
        SurfaceDisplacementField(v, np.array([[0, 1, 2]]), np.zeros((3, 3)))


def test_vertices_bad_shape_raises():
    with pytest.raises(ValueError):
        SurfaceDisplacementField(
            np.zeros((3, 2)), np.array([[0, 1, 2]]), np.zeros((3, 2))
        )


def test_vertices_bool_raises():
    with pytest.raises((TypeError, ValueError)):
        SurfaceDisplacementField(
            np.ones((3, 3), dtype=bool), np.array([[0, 1, 2]]), np.zeros((3, 3))
        )


def test_triangles_bad_shape_raises():
    with pytest.raises(ValueError):
        SurfaceDisplacementField(
            _good_tri_vertices(), np.array([[0, 1, 2, 0]]), np.zeros((3, 3))
        )


def test_triangles_non_integer_raises():
    with pytest.raises((TypeError, ValueError)):
        SurfaceDisplacementField(
            _good_tri_vertices(),
            np.array([[0.0, 1.0, 2.0]]),
            np.zeros((3, 3)),
        )


def test_triangles_bool_raises():
    with pytest.raises((TypeError, ValueError)):
        SurfaceDisplacementField(
            _good_tri_vertices(),
            np.array([[True, False, True]]),
            np.zeros((3, 3)),
        )


def test_triangles_index_out_of_bounds_raises():
    with pytest.raises(ValueError):
        SurfaceDisplacementField(
            _good_tri_vertices(), np.array([[0, 1, 5]]), np.zeros((3, 3))
        )


def test_degenerate_triangle_raises():
    # Repeated index -> zero area.
    with pytest.raises(ValueError):
        SurfaceDisplacementField(
            _good_tri_vertices(), np.array([[0, 1, 1]]), np.zeros((3, 3))
        )


def test_displacement_length_mismatch_raises():
    with pytest.raises(ValueError):
        SurfaceDisplacementField(
            _good_tri_vertices(), np.array([[0, 1, 2]]), np.zeros((2, 3))
        )


def test_displacement_non_finite_raises():
    disp = np.zeros((3, 3))
    disp[1, 2] = np.nan
    with pytest.raises(ValueError):
        SurfaceDisplacementField(_good_tri_vertices(), np.array([[0, 1, 2]]), disp)


def test_displacement_bool_raises():
    with pytest.raises((TypeError, ValueError)):
        SurfaceDisplacementField(
            _good_tri_vertices(),
            np.array([[0, 1, 2]]),
            np.ones((3, 3), dtype=bool),
        )


def test_delta_rho_non_finite_raises():
    field = SurfaceDisplacementField(
        _good_tri_vertices(), np.array([[0, 1, 2]]), np.zeros((3, 3))
    )
    with pytest.raises(ValueError):
        field.as_surface_mass_source(np.inf)


def test_delta_rho_complex_raises():
    field = SurfaceDisplacementField(
        _good_tri_vertices(), np.array([[0, 1, 2]]), np.zeros((3, 3))
    )
    with pytest.raises(TypeError):
        field.as_surface_mass_source(1.0 + 1.0j)


def test_delta_rho_accepts_zero_dim_real_array():
    # 0-d real numpy scalars are accepted, matching the source-side validator.
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
    )
    displacement = np.tile(np.array([0.0, 0.0, 1.0]), (3, 1))
    field = SurfaceDisplacementField(vertices, np.array([[0, 1, 2]]), displacement)
    source = field.as_surface_mass_source(np.array(3.0))
    assert source.density_kg_m3 == 3.0


def test_delta_rho_non_scalar_array_raises():
    field = SurfaceDisplacementField(
        _good_tri_vertices(), np.array([[0, 1, 2]]), np.zeros((3, 3))
    )
    with pytest.raises(TypeError):
        field.as_surface_mass_source(np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# Core purity: importing the helper must not pull in the adapters layer
# ---------------------------------------------------------------------------


def test_import_does_not_load_adapters_or_optional_backends():
    code = (
        "import sys\n"
        "import gragra.fields.surface\n"
        "leaked = [m for m in ('gragra.adapters', 'meshio', 'gwexpy', "
        "'scipy', 'h5py') if m in sys.modules]\n"
        "assert not leaked, leaked\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
