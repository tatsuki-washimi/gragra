"""Tests for the Phase 14-b diagnostic helper `dipole_consistency_check`.

Cube surface/volume mesh fixtures (cube_vertices / cube_triangles /
cube_tetra / cube_half) live in tests/conftest.py as a single source of
truth, shared with test_surface_field.py's bulk/surface mass-conservation
sign-pinning tests.

Design note on the displacement field used below: a naive uniform
isotropic expansion `u = alpha * r` about the cube's centroid (origin) is
NOT usable for the equivalence case. That field is odd-symmetric about
the origin, so the net dipole moment `sum(mass_n * d_n)` is analytically
exactly zero (`sum(vol_n * centroid_n) = V * centroid_of_cube = 0`). The
dipole/total path represents only the dipole-moment content of the field,
so with a truly zero dipole moment both paths are left comparing
quadrupole-order noise, and the residual ratio does not shrink with
distance (a genuine multipole-order mismatch, not a discretization
error). Adding a uniform translation component `_D0` gives the field a
nonzero net dipole moment while leaving mass conservation (net mass
change via the divergence theorem) exactly unaffected (`div` of a
constant vector field is zero), which lets the equivalence converge with
distance as expected.
"""

import warnings

import numpy as np
import pytest

from gragra.fields.dipole_diagnostics import (
    DipoleConsistencyReport,
    dipole_consistency_check,
)
from gragra.fields.surface import SurfaceDisplacementField
from gragra.fields.unstructured import UnstructuredDisplacementField
from gragra.integrators import direct_point_dipole_sum, direct_point_sum

_RHO0 = 2700.0
_ALPHA = 1e-3
_D0 = np.array([2.0e-3, -1.0e-3, 0.5e-3])


def _cube_fields(cube_vertices, cube_triangles, cube_tetra, alpha=_ALPHA, d0=_D0):
    displacement = alpha * cube_vertices + d0
    volume_field = UnstructuredDisplacementField(
        vertices_m=cube_vertices, tetra=cube_tetra, displacement_m=displacement
    )
    surface_field = SurfaceDisplacementField(
        vertices_m=cube_vertices, triangles=cube_triangles, displacement_m=displacement
    )
    return volume_field, surface_field


# ---------------------------------------------------------------------------
# 1. Equivalence (closed, consistent, outward cube)
# ---------------------------------------------------------------------------


def test_equivalence_cube_closed_surface_converges_with_distance(
    cube_vertices, cube_triangles, cube_tetra
):
    volume_field, surface_field = _cube_fields(
        cube_vertices, cube_triangles, cube_tetra
    )

    orientation = surface_field.check_orientation()
    assert orientation.is_consistent
    assert orientation.is_closed
    assert orientation.is_outward

    direction = np.array([2.0, 1.0, -1.0])
    direction = direction / np.linalg.norm(direction)
    distances = [10.0, 30.0, 100.0]
    targets = np.array([d * direction for d in distances])

    report = dipole_consistency_check(
        volume_field,
        surface_field,
        rho0_kg_m3=_RHO0,
        delta_rho_kg_m3=_RHO0,
        targets_xyz=targets,
    )

    # Overall decreasing trend (avoid a strict pairwise-monotonic assertion,
    # which is brittle for a coarse 6-tetra mesh under roundoff).
    assert report.rel_residual[-1] < report.rel_residual[0]
    # Generous O(L/r) bound (L = cube half-width = 1; the residual is
    # dominated by the quadrupole/dipole ratio, which is one power of
    # distance down from the leading dipole term -- not O((L/r)^2)).
    # Empirically ~0.43-0.47 at these distances; the factor of ~2 margin
    # below absorbs mesh/roundoff variation without losing regression power.
    for distance, rel in zip(distances, report.rel_residual, strict=True):
        assert rel < 1.0 / distance


# ---------------------------------------------------------------------------
# 2. Non-equivalence (intentional, open sub-volume)
# ---------------------------------------------------------------------------


def test_nonequivalence_open_subvolume_is_expected(cube_vertices, cube_tetra):
    displacement = _ALPHA * cube_vertices + _D0
    partial_tetra = cube_tetra[:1]
    volume_field = UnstructuredDisplacementField(
        vertices_m=cube_vertices, tetra=partial_tetra, displacement_m=displacement
    )

    targets = 100.0 * np.eye(3)
    report = dipole_consistency_check(
        volume_field,
        None,
        rho0_kg_m3=_RHO0,
        delta_rho_kg_m3=None,
        targets_xyz=targets,
    )
    # Do NOT assert equivalence -- the mismatch itself is the expected,
    # correct diagnostic outcome for an open sub-volume (missing boundary
    # cancellation).
    assert np.all(report.rel_residual > 1.0)


# ---------------------------------------------------------------------------
# 3. as_dipole_source / _tetra_geometry sharing (bit-identical regression)
# ---------------------------------------------------------------------------


def test_as_dipole_source_matches_shared_geometry(cube_vertices, cube_tetra):
    displacement = _ALPHA * cube_vertices + _D0
    field = UnstructuredDisplacementField(
        vertices_m=cube_vertices, tetra=cube_tetra, displacement_m=displacement
    )

    positions, mass, disp = field.as_dipole_source(_RHO0)
    bulk_source = field.as_density_perturbation_source(_RHO0)

    np.testing.assert_array_equal(positions, bulk_source.positions_m)
    np.testing.assert_array_equal(mass, _RHO0 * bulk_source.volumes_m3)

    # as_density_perturbation_source does not externally expose its
    # centroid-averaged displacement (u_centroid is internal-only, used
    # only when include_rho0_gradient_term=True) -- recompute the
    # identical formula independently from the raw fixture data.
    u0 = displacement[cube_tetra[:, 0]]
    u1 = displacement[cube_tetra[:, 1]]
    u2 = displacement[cube_tetra[:, 2]]
    u3 = displacement[cube_tetra[:, 3]]
    expected_disp = (u0 + u1 + u2 + u3) / 4.0
    np.testing.assert_array_equal(disp, expected_disp)


# ---------------------------------------------------------------------------
# 4/5. Empty mesh and non-empty read-only regression
# ---------------------------------------------------------------------------


def test_empty_mesh_as_dipole_source_and_density_perturbation():
    vertices = np.zeros((0, 3))
    tetra = np.zeros((0, 4), dtype=np.int64)
    displacement = np.zeros((0, 3))
    field = UnstructuredDisplacementField(
        vertices_m=vertices, tetra=tetra, displacement_m=displacement
    )

    positions, mass, disp = field.as_dipole_source(1000.0)
    assert positions.shape == (0, 3)
    assert mass.shape == (0,)
    assert disp.shape == (0, 3)
    assert positions.flags.writeable is False
    assert mass.flags.writeable is False
    assert disp.flags.writeable is False

    # Regression: as_density_perturbation_source's pre-existing nc==0
    # behavior must be byte-for-byte unchanged after the _tetra_geometry
    # extraction.
    source = field.as_density_perturbation_source(1000.0)
    assert source.positions_m.shape == (0, 3)
    assert source.volumes_m3.shape == (0,)
    assert source.density_kg_m3.shape == (0,)


def test_as_dipole_source_returns_read_only_arrays_nonempty(cube_vertices, cube_tetra):
    displacement = _ALPHA * cube_vertices + _D0
    field = UnstructuredDisplacementField(
        vertices_m=cube_vertices, tetra=cube_tetra, displacement_m=displacement
    )
    positions, mass, disp = field.as_dipole_source(_RHO0)
    assert positions.flags.writeable is False
    assert mass.flags.writeable is False
    assert disp.flags.writeable is False


# ---------------------------------------------------------------------------
# 6. G-free output pinning
# ---------------------------------------------------------------------------


def test_dipole_consistency_check_is_g_free(cube_vertices, cube_triangles, cube_tetra):
    volume_field, surface_field = _cube_fields(
        cube_vertices, cube_triangles, cube_tetra
    )
    targets = np.array([[50.0, 0.0, 0.0]])

    report = dipole_consistency_check(
        volume_field,
        surface_field,
        rho0_kg_m3=_RHO0,
        delta_rho_kg_m3=_RHO0,
        targets_xyz=targets,
    )

    bulk_source = volume_field.as_density_perturbation_source(
        _RHO0
    ).as_weighted_source()
    surf_source = surface_field.as_surface_mass_source(_RHO0).as_weighted_source()
    direct_bulk_surface = direct_point_sum(
        bulk_source, targets, observable="acceleration"
    ) + direct_point_sum(surf_source, targets, observable="acceleration")

    positions, mass, disp = volume_field.as_dipole_source(_RHO0)
    direct_dipole = direct_point_dipole_sum(mass, disp, positions, targets)

    np.testing.assert_array_equal(report.acceleration_bulk_surface, direct_bulk_surface)
    np.testing.assert_array_equal(report.acceleration_dipole, direct_dipole)


# 7. Existing regression: no dedicated test here -- covered by running
#    tests/test_unstructured_field.py and tests/test_surface_field.py
#    unmodified (see the verification command list in the PR description).

# ---------------------------------------------------------------------------
# 8. Error paths (bidirectional surface/delta_rho validation, rho0 validation)
# ---------------------------------------------------------------------------


def test_surface_field_without_delta_rho_raises(
    cube_vertices, cube_triangles, cube_tetra
):
    volume_field, surface_field = _cube_fields(
        cube_vertices, cube_triangles, cube_tetra
    )
    targets = np.array([[10.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="delta_rho_kg_m3"):
        dipole_consistency_check(
            volume_field,
            surface_field,
            rho0_kg_m3=_RHO0,
            delta_rho_kg_m3=None,
            targets_xyz=targets,
        )


def test_delta_rho_without_surface_field_raises(cube_vertices, cube_tetra):
    displacement = _ALPHA * cube_vertices + _D0
    volume_field = UnstructuredDisplacementField(
        vertices_m=cube_vertices, tetra=cube_tetra, displacement_m=displacement
    )
    targets = np.array([[10.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="delta_rho_kg_m3"):
        dipole_consistency_check(
            volume_field,
            None,
            rho0_kg_m3=_RHO0,
            delta_rho_kg_m3=_RHO0,
            targets_xyz=targets,
        )


def test_as_dipole_source_rejects_bool_and_negative_and_complex_rho0(
    cube_vertices, cube_tetra
):
    displacement = _ALPHA * cube_vertices + _D0
    field = UnstructuredDisplacementField(
        vertices_m=cube_vertices, tetra=cube_tetra, displacement_m=displacement
    )
    with pytest.raises(TypeError):
        field.as_dipole_source(True)
    with pytest.raises(ValueError):
        field.as_dipole_source(-1.0)
    with pytest.raises(TypeError):
        field.as_dipole_source(1.0 + 2.0j)


# ---------------------------------------------------------------------------
# 9. Read-only report contract
# ---------------------------------------------------------------------------


def test_dipole_consistency_report_is_read_only(
    cube_vertices, cube_triangles, cube_tetra
):
    volume_field, surface_field = _cube_fields(
        cube_vertices, cube_triangles, cube_tetra
    )
    targets = np.array([[10.0, 0.0, 0.0]])
    report = dipole_consistency_check(
        volume_field,
        surface_field,
        rho0_kg_m3=_RHO0,
        delta_rho_kg_m3=_RHO0,
        targets_xyz=targets,
    )
    assert isinstance(report, DipoleConsistencyReport)
    for arr in (
        report.acceleration_bulk_surface,
        report.acceleration_dipole,
        report.abs_residual,
        report.rel_residual,
    ):
        assert arr.flags.writeable is False


def test_dipole_consistency_report_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        DipoleConsistencyReport(
            acceleration_bulk_surface=np.zeros((2, 3)),
            acceleration_dipole=np.zeros((3, 3)),
            abs_residual=np.zeros((2,)),
            rel_residual=np.zeros((2,)),
        )


# ---------------------------------------------------------------------------
# 10. Empty targets regression
# ---------------------------------------------------------------------------


def test_empty_targets_returns_empty_report(cube_vertices, cube_triangles, cube_tetra):
    volume_field, surface_field = _cube_fields(
        cube_vertices, cube_triangles, cube_tetra
    )
    targets = np.zeros((0, 3))
    report = dipole_consistency_check(
        volume_field,
        surface_field,
        rho0_kg_m3=_RHO0,
        delta_rho_kg_m3=_RHO0,
        targets_xyz=targets,
    )
    assert report.acceleration_bulk_surface.shape == (0, 3)
    assert report.acceleration_dipole.shape == (0, 3)
    assert report.abs_residual.shape == (0,)
    assert report.rel_residual.shape == (0,)


# ---------------------------------------------------------------------------
# 11. Complex displacement end-to-end
# ---------------------------------------------------------------------------


def test_complex_displacement_end_to_end(cube_vertices, cube_triangles, cube_tetra):
    alpha = 1e-3 + 2e-3j
    displacement = (alpha * cube_vertices).astype(np.complex128) + _D0
    volume_field = UnstructuredDisplacementField(
        vertices_m=cube_vertices, tetra=cube_tetra, displacement_m=displacement
    )
    surface_field = SurfaceDisplacementField(
        vertices_m=cube_vertices, triangles=cube_triangles, displacement_m=displacement
    )

    targets = np.array([[50.0, 20.0, -10.0]])
    report = dipole_consistency_check(
        volume_field,
        surface_field,
        rho0_kg_m3=_RHO0,
        delta_rho_kg_m3=_RHO0,
        targets_xyz=targets,
    )
    assert report.acceleration_bulk_surface.dtype == np.complex128
    assert report.acceleration_dipole.dtype == np.complex128
    assert report.rel_residual[0] < 0.05


# ---------------------------------------------------------------------------
# 12. Zero-response / degenerate denominator case
# ---------------------------------------------------------------------------


def test_zero_rho0_gives_exact_zero_rel_residual(
    cube_vertices, cube_triangles, cube_tetra
):
    # rho0=0 -> both paths' mass elements vanish -> acceleration_dipole is
    # exactly zero and abs_residual is exactly zero -> the degenerate-
    # denominator rule defines rel_residual as exactly 0.0 (perfect
    # agreement), not merely "some finite value".
    volume_field, surface_field = _cube_fields(
        cube_vertices, cube_triangles, cube_tetra
    )
    targets = np.array([[10.0, 0.0, 0.0]])
    report = dipole_consistency_check(
        volume_field,
        surface_field,
        rho0_kg_m3=0.0,
        delta_rho_kg_m3=0.0,
        targets_xyz=targets,
    )
    np.testing.assert_array_equal(report.abs_residual, 0.0)
    np.testing.assert_array_equal(report.rel_residual, 0.0)


def test_zero_dipole_norm_with_nonzero_residual_gives_inf_not_overflow(
    cube_vertices, cube_triangles, cube_tetra
):
    # Regression for a code-review finding: acceleration_dipole exactly
    # zero (zero volume-field displacement) while acceleration_bulk_surface
    # is nonzero (nonzero surface displacement) previously divided by a
    # fixed epsilon floor and overflowed to +inf with a RuntimeWarning
    # instead of being an intentional, documented +inf.
    zero_displacement = np.zeros_like(cube_vertices)
    volume_field = UnstructuredDisplacementField(
        vertices_m=cube_vertices, tetra=cube_tetra, displacement_m=zero_displacement
    )
    surface_field = SurfaceDisplacementField(
        vertices_m=cube_vertices,
        triangles=cube_triangles,
        displacement_m=0.1 * cube_vertices,
    )
    targets = np.array([[10.0, 0.0, 0.0]])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report = dipole_consistency_check(
            volume_field,
            surface_field,
            rho0_kg_m3=_RHO0,
            delta_rho_kg_m3=_RHO0,
            targets_xyz=targets,
        )

    assert report.abs_residual[0] > 0.0
    assert report.acceleration_dipole[0].sum() == 0.0
    assert report.rel_residual[0] == np.inf
