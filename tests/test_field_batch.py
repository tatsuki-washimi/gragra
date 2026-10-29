"""Tests for the batched field-import helpers (Phase 10-d).

spec: docs/design/phase10d_batch_import.md.

Golden values in this file are pinned per the design spec (recorded before
the implementation): do not re-derive or change
them without updating the spec first.
"""

import numpy as np
import pytest

from gragra.constants import G_SI
from gragra.fields.batch import (
    batched_displacement_grid_weights,
    batched_surface_weights,
    batched_unstructured_weights,
)
from gragra.fields.regular_grid import DisplacementGrid
from gragra.fields.surface import SurfaceDisplacementField
from gragra.fields.unstructured import UnstructuredDisplacementField
from gragra.observables.coherent import coherent_point_acceleration

# --- Shared fixtures (unit tetra / unit triangle / 3x3x3 grid) -------------

_TETRA_VERTS = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)
_TETRA = np.array([[0, 1, 2, 3]])

_TRI_VERTS = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
_TRI = np.array([[0, 1, 2]])

_GRID_AXIS = np.array([0.0, 1.0, 2.0])


def _single_tetra_weight(beta, rho0=2.0):
    u = np.zeros((4, 3))
    u[:, 0] = beta * _TETRA_VERTS[:, 0]
    field = UnstructuredDisplacementField(_TETRA_VERTS, _TETRA, u)
    src = field.as_density_perturbation_source(rho0)
    return src.as_weighted_source()


def _single_surface_weight(gamma, delta_rho=3.0):
    u = np.zeros((3, 3))
    u[:, 2] = gamma
    field = SurfaceDisplacementField(_TRI_VERTS, _TRI, u)
    return field.as_surface_mass_source(delta_rho).as_weighted_source()


def _single_grid_weight(alpha, rho0=1.0):
    x, y, z = np.meshgrid(_GRID_AXIS, _GRID_AXIS, _GRID_AXIS, indexing="ij")
    disp = np.stack([alpha * x, alpha * y, alpha * z], axis=-1)
    grid = DisplacementGrid(_GRID_AXIS, _GRID_AXIS, _GRID_AXIS, disp)
    perturb = grid.as_density_perturbation_grid(rho0)
    return perturb.as_volume_element_source().as_weighted_source()


# --- T1: P=1 bit-identity with the single-snapshot path --------------------


def test_t1_tetra_p1_bit_identity():
    beta = 0.5
    u = np.zeros((1, 4, 3))
    u[0, :, 0] = beta * _TETRA_VERTS[:, 0]
    positions, weights = batched_unstructured_weights(_TETRA_VERTS, _TETRA, u, 2.0)
    single = _single_tetra_weight(beta)
    assert weights.shape == (1, 1)
    np.testing.assert_array_equal(weights[:, 0], single.weights_kg)
    np.testing.assert_array_equal(positions, single.positions_m)


def test_t1_surface_p1_bit_identity():
    gamma = 0.2
    u = np.zeros((1, 3, 3))
    u[0, :, 2] = gamma
    positions, weights = batched_surface_weights(_TRI_VERTS, _TRI, u, 3.0)
    single = _single_surface_weight(gamma)
    assert weights.shape == (1, 1)
    np.testing.assert_array_equal(weights[:, 0], single.weights_kg)
    np.testing.assert_array_equal(positions, single.positions_m)


def test_t1_grid_p1_bit_identity():
    alpha = 0.01
    x, y, z = np.meshgrid(_GRID_AXIS, _GRID_AXIS, _GRID_AXIS, indexing="ij")
    disp = np.stack([alpha * x, alpha * y, alpha * z], axis=-1)[np.newaxis]
    positions, weights = batched_displacement_grid_weights(
        _GRID_AXIS, _GRID_AXIS, _GRID_AXIS, disp, 1.0
    )
    single = _single_grid_weight(alpha)
    assert weights.shape == (27, 1)
    np.testing.assert_array_equal(weights[:, 0], single.weights_kg)
    np.testing.assert_array_equal(positions, single.positions_m)


# --- T2: non-uniform 2-snapshot stack consistency (grid path only) --------


def test_t2_grid_stack_consistency():
    alpha = 0.01
    x, y, z = np.meshgrid(_GRID_AXIS, _GRID_AXIS, _GRID_AXIS, indexing="ij")
    u0 = np.stack([alpha * x, alpha * y, alpha * z], axis=-1)
    rng = np.random.default_rng(42)
    u1 = rng.normal(size=(3, 3, 3, 3))
    batched_disp = np.stack([u0, u1], axis=0)

    positions, weights = batched_displacement_grid_weights(
        _GRID_AXIS, _GRID_AXIS, _GRID_AXIS, batched_disp, 1.0
    )

    grid0 = DisplacementGrid(_GRID_AXIS, _GRID_AXIS, _GRID_AXIS, u0)
    w0 = (
        grid0.as_density_perturbation_grid(1.0)
        .as_volume_element_source()
        .as_weighted_source()
    )
    grid1 = DisplacementGrid(_GRID_AXIS, _GRID_AXIS, _GRID_AXIS, u1)
    w1 = (
        grid1.as_density_perturbation_grid(1.0)
        .as_volume_element_source()
        .as_weighted_source()
    )

    expected = np.stack([w0.weights_kg, w1.weights_kg], axis=-1)
    np.testing.assert_array_equal(weights, expected)
    np.testing.assert_array_equal(positions, w0.positions_m)


# --- T3: single-tetra literal golden ---------------------------------------


@pytest.mark.parametrize("beta,expected", [(0.5, -1.0 / 6.0), (-1.0, 1.0 / 3.0)])
def test_t3_tetra_golden(beta, expected):
    u = np.zeros((1, 4, 3))
    u[0, :, 0] = beta * _TETRA_VERTS[:, 0]
    positions, weights = batched_unstructured_weights(_TETRA_VERTS, _TETRA, u, 2.0)
    np.testing.assert_allclose(weights[0, 0], expected, rtol=1e-12)
    np.testing.assert_allclose(positions[0], [0.25, 0.25, 0.25])


def test_t3_tetra_golden_both_snapshots():
    u = np.zeros((2, 4, 3))
    u[0, :, 0] = 0.5 * _TETRA_VERTS[:, 0]
    u[1, :, 0] = -1.0 * _TETRA_VERTS[:, 0]
    positions, weights = batched_unstructured_weights(_TETRA_VERTS, _TETRA, u, 2.0)
    assert weights.shape == (1, 2)
    np.testing.assert_allclose(weights[0], [-1.0 / 6.0, 1.0 / 3.0], rtol=1e-12)


# --- T4: single-triangle literal golden ------------------------------------


@pytest.mark.parametrize("gamma,expected", [(0.2, 0.3), (-0.4, -0.6)])
def test_t4_surface_golden(gamma, expected):
    u = np.zeros((1, 3, 3))
    u[0, :, 2] = gamma
    positions, weights = batched_surface_weights(_TRI_VERTS, _TRI, u, 3.0)
    np.testing.assert_allclose(weights[0, 0], expected, rtol=1e-12)
    np.testing.assert_allclose(positions[0], [1.0 / 3.0, 1.0 / 3.0, 0.0])


# --- T5: grid literal golden ------------------------------------------------


@pytest.mark.parametrize("alpha,expected", [(0.01, -0.03), (0.02, -0.06)])
def test_t5_grid_golden(alpha, expected):
    x, y, z = np.meshgrid(_GRID_AXIS, _GRID_AXIS, _GRID_AXIS, indexing="ij")
    disp = np.stack([alpha * x, alpha * y, alpha * z], axis=-1)[np.newaxis]
    positions, weights = batched_displacement_grid_weights(
        _GRID_AXIS, _GRID_AXIS, _GRID_AXIS, disp, 1.0
    )
    np.testing.assert_allclose(weights[:, 0], np.full(27, expected), rtol=1e-12)


# --- T6: bulk-only uniform-sphere e2e (analytic oracle) --------------------


def test_t6_bulk_only_uniform_sphere_e2e():
    """Bulk-only formulation: compares against the uniform-sphere closed form
    for density = -3*alpha*rho0, NOT the physical expanding-sphere field
    (which would vanish outside by the shell theorem -- out of scope here).

    Mirrors the staircase-after-divergence construction of
    tests/test_field_endtoend.py: the displacement field u=alpha*r is kept
    smooth over the *entire* grid domain (div(u) is computed with no
    discontinuity, so it is exactly 3*alpha everywhere), and only afterward
    are the resulting per-cell weights truncated to the cells whose centers
    fall inside the ball radius. Zeroing u outside a mask *before* computing
    the divergence (as an earlier draft of this test did) introduces a
    spurious discontinuity that np.gradient turns into large fictitious
    density spikes at the boundary shell -- unrelated to implementation
    correctness of the batch helper itself.
    """
    from gragra.analytic.uniform_sphere import uniform_sphere_acceleration

    radius_m = 1.0
    half_extent = 1.2
    n = 61
    axis = np.linspace(-half_extent, half_extent, n)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")

    rho0 = 2700.0
    alphas = [1.0e-3, 2.0e-3]
    batched_disp = np.stack(
        [np.stack([alpha * x, alpha * y, alpha * z], axis=-1) for alpha in alphas],
        axis=0,
    )

    positions, weights = batched_displacement_grid_weights(
        axis, axis, axis, batched_disp, rho0
    )

    radii = np.linalg.norm(positions, axis=1)
    inside = radii < radius_m
    positions = positions[inside]
    weights = weights[inside]

    targets = np.array([[5.0, 0.0, 0.0], [0.0, 4.0, 3.0]])
    acc = coherent_point_acceleration(positions, weights, targets)  # (M, P, 3)

    center = np.array([0.0, 0.0, 0.0])
    for p, alpha in enumerate(alphas):
        delta_rho = -3.0 * alpha * rho0
        expected = G_SI * uniform_sphere_acceleration(
            center, radius_m, delta_rho, targets
        )
        atol = 1.0e-3 * np.abs(expected).max()
        np.testing.assert_allclose(acc[:, p, :], expected, rtol=2.0e-2, atol=atol)


# --- T7: complex dtype follows input -----------------------------------


def test_t7_complex_dtype_follows_input():
    u_real = np.zeros((1, 4, 3))
    u_real[0, :, 0] = 0.5 * _TETRA_VERTS[:, 0]
    _, weights_real = batched_unstructured_weights(_TETRA_VERTS, _TETRA, u_real, 2.0)
    assert weights_real.dtype == np.float64

    u_complex = np.zeros((1, 4, 3), dtype=np.complex128)
    u_complex[0, :, 0] = (0.5 + 0.1j) * _TETRA_VERTS[:, 0]
    _, weights_complex = batched_unstructured_weights(
        _TETRA_VERTS, _TETRA, u_complex, 2.0
    )
    assert weights_complex.dtype == np.complex128


# --- T8: ValueError contract -------------------------------------------


def test_t8_p_zero_raises():
    with pytest.raises(ValueError):
        batched_unstructured_weights(_TETRA_VERTS, _TETRA, np.zeros((0, 4, 3)), 2.0)


def test_t8_single_snapshot_shape_raises():
    with pytest.raises(ValueError):
        batched_unstructured_weights(_TETRA_VERTS, _TETRA, np.zeros((4, 3)), 2.0)


def test_t8_grid_single_snapshot_shape_raises():
    x, y, z = np.meshgrid(_GRID_AXIS, _GRID_AXIS, _GRID_AXIS, indexing="ij")
    disp = np.stack([x, y, z], axis=-1)  # (3,3,3,3) -- no leading batch axis
    with pytest.raises(ValueError):
        batched_displacement_grid_weights(_GRID_AXIS, _GRID_AXIS, _GRID_AXIS, disp, 1.0)


def test_t8_snapshot_index_in_error_message():
    u = np.zeros((2, 4, 3))
    u[0, :, 0] = 0.5 * _TETRA_VERTS[:, 0]
    u[1, :, 0] = 0.5 * _TETRA_VERTS[:, 0]
    # Corrupt snapshot 1's displacement shape indirectly is hard without an
    # invalid mesh; instead exercise an invalid rho0 that the delegated
    # single-snapshot class rejects, and check the index annotation.
    with pytest.raises(ValueError, match=r"snapshot p=\d+"):
        batched_unstructured_weights(_TETRA_VERTS, _TETRA, u, -1.0)


# --- T9: read-only outputs ----------------------------------------------


def test_t9_read_only():
    u = np.zeros((1, 4, 3))
    u[0, :, 0] = 0.5 * _TETRA_VERTS[:, 0]
    positions, weights = batched_unstructured_weights(_TETRA_VERTS, _TETRA, u, 2.0)
    assert positions.flags.writeable is False
    assert weights.flags.writeable is False


# --- T10: chunk invariance + kernels smoke ------------------------------


def test_t10_chunk_invariance_and_kernel_shape():
    u = np.zeros((2, 4, 3))
    u[0, :, 0] = 0.5 * _TETRA_VERTS[:, 0]
    u[1, :, 0] = -1.0 * _TETRA_VERTS[:, 0]
    positions, weights = batched_unstructured_weights(_TETRA_VERTS, _TETRA, u, 2.0)
    targets = np.array([[1.0, 0.0, 0.0]])

    acc_full = coherent_point_acceleration(positions, weights, targets, chunk_size=None)
    acc_chunked = coherent_point_acceleration(positions, weights, targets, chunk_size=1)

    assert acc_full.shape == (1, 2, 3)
    np.testing.assert_allclose(acc_full, acc_chunked, rtol=1e-12)
