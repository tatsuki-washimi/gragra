"""Phase 12-b Slice 1: SCA cavern-wall coupling factor tests.

Design: docs/design/phase12b_sca_cavity_coupling.md Sec.0b/Sec.2. Golden
values are Harms et al. 2022, Table 1 (p. 5 of 16;
docs/references/02_NN_Modeling_and_Simulation/02-1_Analytical/
A_lower_limit_for_Newtonian-noise_models_of_the_Einstein_Telescope.md).
"""

import numpy as np
import pytest

from gragra.analytic import sca_wall_acceleration, sca_wall_coupling_factor
from gragra.analytic._bessel import cavity_wall_geometric
from gragra.analytic.cavity_sphere import spherical_cavity_wall

RHO = 1000.0  # kg/m^3
XI = 1e-6  # m


def test_table1_sphere_arbitrary():
    """Literal Table 1 sphere/arbitrary lookup, not a center alias."""
    assert sca_wall_coupling_factor("sphere", "arbitrary") == pytest.approx(
        4.0 * np.pi / 3.0, rel=1e-15
    )


def test_table1_cube_center_3d():
    """Literal Table 1 cube/3D-center lookup."""
    assert sca_wall_coupling_factor("cube", "center_3d") == pytest.approx(
        4.0 * np.pi / 3.0, rel=1e-15
    )


def test_table1_cube_floor_center():
    """Literal source value; it is not a tensor or free-surface assertion."""
    assert sca_wall_coupling_factor("cube", "floor_center_2d") == pytest.approx(
        2.8, rel=1e-15
    )


def test_table1_half_sphere_floor_center():
    """Literal source value; it is not a tensor or free-surface assertion."""
    assert sca_wall_coupling_factor("half_sphere", "floor_center_2d") == pytest.approx(
        2.1, rel=1e-15
    )


def test_table1_elongated_cuboid_floor_center():
    """Literal Table 1 value for the finite 2:1:1 cuboid row."""
    assert sca_wall_coupling_factor(
        "elongated_cuboid_2to1to1", "floor_center_2d"
    ) == pytest.approx(1.3, rel=1e-15)


@pytest.mark.parametrize(
    ("cavern_shape", "test_mass_location"),
    [
        ("sphere", "floor_center_2d"),
        ("sphere", "center_3d"),
        ("cube", "arbitrary"),
        ("half_sphere", "arbitrary"),
        ("half_sphere", "center_3d"),
        ("elongated_cuboid_2to1to1", "arbitrary"),
        ("elongated_cuboid_2to1to1", "center_3d"),
    ],
)
def test_invalid_combination_raises(cavern_shape, test_mass_location):
    """Only the five (shape, location) pairs in Table 1 are valid -- not a
    full product space. In particular, sphere/arbitrary is a Table 1 label,
    not a substitute for cube/3D-center or any floor condition."""
    with pytest.raises(ValueError, match="No Harms"):
        sca_wall_coupling_factor(cavern_shape, test_mass_location)


def test_sca_wall_acceleration_matches_coupling_factor():
    factor = sca_wall_coupling_factor("half_sphere", "floor_center_2d")
    a = sca_wall_acceleration("half_sphere", "floor_center_2d", RHO, XI)
    assert a == pytest.approx(-factor * RHO * XI, rel=1e-15)


def test_sca_wall_acceleration_sign_is_negative_for_positive_inputs():
    a = sca_wall_acceleration("sphere", "arbitrary", RHO, XI)
    assert a < 0.0


@pytest.mark.parametrize("density_kg_m3", [0.0, -1.0, float("nan"), float("inf")])
def test_sca_wall_acceleration_rejects_invalid_density(density_kg_m3):
    with pytest.raises(ValueError):
        sca_wall_acceleration("sphere", "arbitrary", density_kg_m3, XI)


@pytest.mark.parametrize("displacement_m", [0.0, -1e-6, float("nan"), float("inf")])
def test_sca_wall_acceleration_rejects_invalid_displacement(displacement_m):
    with pytest.raises(ValueError):
        sca_wall_acceleration("sphere", "arbitrary", RHO, displacement_m)


def test_sphere_matches_existing_cavity_sphere_small_x_limit():
    """Cross-check against the already-reviewed Phase 12-a wall term.

    In the small-cavern limit (x = k*R -> 0), spherical_cavity_wall's P-wave
    geometric factor tends to 1/3 (cavity_wall_geometric(0) series term),
    reproducing the Table 1 sphere/arbitrary scalar value of 4*pi/3. This is
    a same-approximation-tier comparison (both sides are small-cavern
    limits), unlike the Slice 4 guard about comparing an SCA constant
    against an arbitrary-frequency/arbitrary-size exact closed form -- see
    docs/design/phase12b_sca_cavity_coupling.md Sec.3.
    """
    tiny_x = 1e-6
    expected_geom = cavity_wall_geometric(tiny_x)
    assert expected_geom == pytest.approx(1.0 / 3.0, rel=1e-9)

    r = 1.0
    k = [tiny_x / r]
    wall_p = spherical_cavity_wall(r, k, RHO, XI, wave_type="P")
    sca = sca_wall_acceleration("sphere", "arbitrary", RHO, XI)
    np.testing.assert_allclose(wall_p[0], sca, rtol=1e-9)
