"""End-to-end physics validation for the external field schema (Phase 10-c).

These tests exercise the *full chain* from an external displacement field down
to a public observable and compare against gragra's own Phase-3 analytic
benchmark (``uniform_sphere_*``), rather than only checking the internal
divergence math in isolation.

Reference construction (no external solver / no extra dependency required):
A uniform isotropic expansion ``u = alpha * r`` has constant divergence
``div(u) = 3 * alpha``, so the linearized bulk density perturbation is the
uniform value ``delta_rho = -rho0 * div(u) = -3 * alpha * rho0``. Restricting
the perturbed cells to a ball of radius ``radius`` therefore reproduces a
uniform-density sphere of density ``-3 * alpha * rho0``, whose external field is
known in closed form. This validates:

    DisplacementGrid -> divergence -> delta_rho -> VolumeElementSource
        -> WeightedPointSource -> point_* observable

against ``uniform_sphere_*`` (times ``G_SI``, since the analytic helpers are
G-free while observables include ``G_SI``).
"""

import numpy as np

from gragra.analytic.uniform_sphere import (
    uniform_sphere_acceleration,
    uniform_sphere_gradient,
    uniform_sphere_potential,
)
from gragra.constants import G_SI
from gragra.fields.regular_grid import DisplacementGrid
from gragra.gravity_sources.volume import VolumeElementSource
from gragra.observables.acceleration import point_acceleration
from gragra.observables.gradient import point_gravity_gradient
from gragra.observables.potential import point_potential

# Common reference-ball parameters shared by all end-to-end checks.
_RADIUS = 1.0
_ALPHA = 1.0e-3
_RHO0 = 2700.0
_N = 61
_HALF_EXTENT = 1.2
_TARGETS = np.array([[5.0, 0.0, 0.0], [0.0, 4.0, 3.0]])


def _uniform_expansion_ball_source(radius, alpha, rho0, n, half_extent):
    """Build a uniform-density ball via the displacement-field chain.

    Returns ``(source, delta_rho_expected)`` where ``source`` is the
    ``VolumeElementSource`` for the cells whose centers fall inside ``radius``.
    """
    ax = np.linspace(-half_extent, half_extent, n)
    grid_x, grid_y, grid_z = np.meshgrid(ax, ax, ax, indexing="ij")
    # Displacement u = alpha * r (linear field -> exact gradient at all nodes).
    displacement = np.stack([alpha * grid_x, alpha * grid_y, alpha * grid_z], axis=-1)

    grid = DisplacementGrid(x_m=ax, y_m=ax, z_m=ax, displacement_m=displacement)
    drho_grid = grid.as_density_perturbation_grid(rho0)

    # div(u) = 3 alpha exactly for a linear field, even at boundary nodes with
    # edge_order=2, so the perturbation must be the uniform expected value.
    delta_rho_expected = -3.0 * alpha * rho0
    np.testing.assert_allclose(
        drho_grid.delta_density_kg_m3, delta_rho_expected, rtol=1e-9, atol=0.0
    )

    cell_source = drho_grid.as_volume_element_source()
    centers = cell_source.positions_m
    radii = np.linalg.norm(centers, axis=1)
    inside = radii < radius

    ball = VolumeElementSource(
        positions_m=centers[inside],
        volumes_m3=cell_source.volumes_m3[inside],
        density_kg_m3=np.asarray(cell_source.density_kg_m3)[inside],
    )
    return ball, delta_rho_expected


def test_endtoend_uniform_expansion_acceleration_matches_uniform_sphere():
    ball, delta_rho = _uniform_expansion_ball_source(
        _RADIUS, _ALPHA, _RHO0, _N, _HALF_EXTENT
    )
    source = ball.as_weighted_source()

    a_num = point_acceleration(source, _TARGETS)
    a_ana = G_SI * uniform_sphere_acceleration(
        [0.0, 0.0, 0.0], _RADIUS, delta_rho, _TARGETS
    )

    # atol absorbs the structurally-zero components (e.g. transverse a at an
    # on-axis target) where the analytic value is exactly 0; the active
    # components are still constrained by rtol.
    atol = 1.0e-3 * np.abs(a_ana).max()
    np.testing.assert_allclose(a_num, a_ana, rtol=2.0e-2, atol=atol)


def test_endtoend_uniform_expansion_potential_matches_uniform_sphere():
    ball, delta_rho = _uniform_expansion_ball_source(
        _RADIUS, _ALPHA, _RHO0, _N, _HALF_EXTENT
    )
    source = ball.as_weighted_source()

    phi_num = point_potential(source, _TARGETS)
    phi_ana = G_SI * uniform_sphere_potential(
        [0.0, 0.0, 0.0], _RADIUS, delta_rho, _TARGETS
    )

    np.testing.assert_allclose(phi_num, phi_ana, rtol=2.0e-2, atol=0.0)


def test_endtoend_uniform_expansion_gradient_matches_uniform_sphere():
    ball, delta_rho = _uniform_expansion_ball_source(
        _RADIUS, _ALPHA, _RHO0, _N, _HALF_EXTENT
    )
    source = ball.as_weighted_source()

    t_num = point_gravity_gradient(source, _TARGETS)
    t_ana = G_SI * uniform_sphere_gradient(
        [0.0, 0.0, 0.0], _RADIUS, delta_rho, _TARGETS
    )

    # Gradient (~1/r^3) is more sensitive to the staircased boundary than the
    # acceleration/potential, so allow a slightly looser tolerance. atol absorbs
    # the structurally-zero off-diagonal components (exactly 0 analytically).
    atol = 1.0e-3 * np.abs(t_ana).max()
    np.testing.assert_allclose(t_num, t_ana, rtol=3.0e-2, atol=atol)


def test_endtoend_total_mass_is_transported_through_the_chain():
    """The integrated perturbed mass must match delta_rho * ball volume."""
    ball, delta_rho = _uniform_expansion_ball_source(
        _RADIUS, _ALPHA, _RHO0, _N, _HALF_EXTENT
    )

    total_mass = ball.as_weighted_source().weights_kg.sum()
    # Staircased ball volume approximates the analytic ball volume.
    analytic_mass = delta_rho * (4.0 / 3.0) * np.pi * _RADIUS**3
    np.testing.assert_allclose(total_mass, analytic_mass, rtol=2.0e-2, atol=0.0)
