"""Phase 16-a convergence suite: Cylinder h-refinement vs analytic closed form.

Design: docs/design/phase16a_convergence_suite.md Sec.6. Reference is
``analytic.cylinder_filled_axis_field`` (G-free, on-axis z-component only;
valid on interior and exterior axis points alike -- see its docstring).
Compared to G-included observables via
``(0, 0, G_SI * cylinder_filled_axis_field(...))``. Marked ``slow`` so it is
deselected by the default ``pytest`` run (``addopts = "-m 'not slow'"`` in
pyproject.toml):

    pytest -m slow tests/test_convergence_cylinder.py -v

Full acceleration *vectors* are compared (not just a_z), so a single
``_relative_error_vector`` call verifies the z-component magnitude/sign *and*
the x/y cancellation together.

Naming convention (fixed to avoid an ``h`` ambiguity flagged in review):
``_HEIGHT`` (H, the cylinder's fixed physical height) and ``spacing_m``
(the h-refinement grid spacing) are always spelled out separately -- neither
is ever abbreviated to a bare ``h``.
"""

import numpy as np
import pytest
from _convergence_utils import (
    _min_distance,
    _relative_error_scalar,
    _relative_error_vector,
    _slope,
)

from gragra import G_SI, point_acceleration
from gragra.analytic import cylinder_filled_axis_field
from gragra.geometry.primitives import Cylinder

pytestmark = pytest.mark.slow

_RADIUS = 1.0
_HEIGHT = 1.5
_AXIS = "z"
_CENTER = np.array([0.0, 0.0, 0.0])
_DENSITY = 1000.0

# spacing_m = _RADIUS / n, n in {5, 10, 20, 40}. With axis="z" the x/y bbox is
# exactly 2*_RADIUS, so lx/spacing_m = ly/spacing_m = 2n exactly (an integer,
# unaffected by ceil()) -- the x/y split is always into an EVEN number of
# cells, so no source cell center ever lands exactly on the z-axis (x=y=0).
_NS = [5, 10, 20, 40]

# On-axis targets, expressed as fractions of the fixed cylinder height _HEIGHT
# (never of spacing_m) so the target position does not move as spacing_m is
# refined. 0.137/1.7 are non-round fractions chosen to avoid landing exactly
# on a z-cell boundary at any swept resolution.
_Z_INTERNAL = 0.137 * _HEIGHT  # inside [-_HEIGHT/2, _HEIGHT/2]
_Z_EXTERNAL = 1.7 * _HEIGHT  # well outside the cylinder

_TARGET_INTERNAL = np.array([[0.0, 0.0, _Z_INTERNAL]])
_TARGET_EXTERNAL = np.array([[0.0, 0.0, _Z_EXTERNAL]])

# Same geometric sanity bound as the Cuboid suite (see its rationale comment
# in test_convergence_cuboid.py: 0.05 proved too loose to rule out
# near-field point-mass domination there, so 0.2 is used uniformly). On-axis
# Cylinder targets measured 0.71-0.81 (internal) and >9 (external) at every
# swept n, comfortably clearing 0.2 with no target re-picking needed.
_MIN_D_OVER_SPACING = 0.2


def _discretized_cylinder(spacing_m):
    cylinder = Cylinder(
        center_m=_CENTER,
        radius_m=_RADIUS,
        height_m=_HEIGHT,
        axis=_AXIS,
        density_kg_m3=_DENSITY,
        spacing_m=spacing_m,
        max_points=None,
    )
    return cylinder.to_weighted_points()


def _exact_vector(z_m):
    az = G_SI * cylinder_filled_axis_field(_RADIUS, _HEIGHT, z_m, _DENSITY)
    return np.array([0.0, 0.0, float(az)])


@pytest.fixture(scope="module")
def cylinder_sweep():
    """Build each resolution's discretized Cylinder once; keep summaries only.

    Lazily evaluated (pytest fixtures only execute when a dependent test
    actually runs) -- see test_convergence_cuboid.py's ``cuboid_sweep`` for
    the identical rationale.
    """
    exact_internal = _exact_vector(_Z_INTERNAL)
    exact_external = _exact_vector(_Z_EXTERNAL)
    expected_mass = _DENSITY * np.pi * _RADIUS**2 * _HEIGHT

    summaries = []
    for n in _NS:
        spacing_m = _RADIUS / n
        src = _discretized_cylinder(spacing_m)

        d_min_internal = _min_distance(src.positions_m, _TARGET_INTERNAL[0])
        d_min_external = _min_distance(src.positions_m, _TARGET_EXTERNAL[0])

        acc_internal = point_acceleration(
            src, _TARGET_INTERNAL, backend="numpy", chunk_size=None
        )
        acc_external = point_acceleration(
            src, _TARGET_EXTERNAL, backend="numpy", chunk_size=None
        )

        mass = float(np.sum(src.weights_kg))
        summaries.append(
            {
                "n": n,
                "spacing_m": spacing_m,
                "point_count": int(src.positions_m.shape[0]),
                "mass": mass,
                "mass_error": _relative_error_scalar(mass, expected_mass),
                "d_min_over_spacing_internal": d_min_internal / spacing_m,
                "d_min_over_spacing_external": d_min_external / spacing_m,
                "internal_error": _relative_error_vector(
                    acc_internal[0], exact_internal
                ),
                "external_error": _relative_error_vector(
                    acc_external[0], exact_external
                ),
            }
        )
    return summaries


def test_cylinder_targets_are_geometrically_well_separated(cylinder_sweep):
    """d_min/spacing sanity bound, fixed independently of any error result."""
    for s in cylinder_sweep:
        assert s["d_min_over_spacing_internal"] > _MIN_D_OVER_SPACING, s
        assert s["d_min_over_spacing_external"] > _MIN_D_OVER_SPACING, s


def test_cylinder_mass_converges_overall(cylinder_sweep):
    """Overall trend only -- no sequential monotonicity is required.

    Cylinder discretization uses cell-center inclusion with no partial-cell
    volume (_grid.py), so the circular cross-section's staircase
    approximation changes non-continuously between resolutions and the mass
    error is not guaranteed to decrease at every intermediate step (same
    staircase-non-monotonicity noted for Sphere in
    docs/design/phase16a_convergence_suite.md Sec.5/measured log). Only the
    coarsest-vs-finest trend and an absolute finest-resolution bound are
    asserted; slope is recorded (design doc), not gated.
    """
    mass_errors = [s["mass_error"] for s in cylinder_sweep]
    hs = [s["spacing_m"] for s in cylinder_sweep]

    assert all(np.isfinite(mass_errors))
    _slope(hs, mass_errors)

    assert mass_errors[-1] < mass_errors[0]
    # Finalized against the measured log in
    # docs/design/phase16a_convergence_suite.md: finest (n=40) measured
    # 5.07e-4; note n=10 and n=20 measured an *identical* 5.86e-3 (the
    # doubled resolution reproduces the same relative circular-staircase
    # pattern -- not a bug, see measured log). 0.01 keeps a ~20x margin over
    # the finest measurement while staying tighter than the provisional 0.05.
    assert mass_errors[-1] <= 0.01


def test_cylinder_internal_axis_field_converges(cylinder_sweep):
    errors = [s["internal_error"] for s in cylinder_sweep]
    hs = [s["spacing_m"] for s in cylinder_sweep]

    assert all(np.isfinite(errors))
    _slope(hs, errors)

    # Finalized against the measured log in
    # docs/design/phase16a_convergence_suite.md: finest (n=40) measured
    # 1.63e-2; 0.03 keeps a ~1.8x margin (measured slope ~0.53 is far from
    # the O(h^2) design target -- consistent with an on-axis interior point
    # close to the staircase-approximated circular boundary, not a bug).
    assert errors[-1] <= 0.03
    assert errors[-1] < errors[0]


def test_cylinder_external_axis_field_converges(cylinder_sweep):
    errors = [s["external_error"] for s in cylinder_sweep]
    hs = [s["spacing_m"] for s in cylinder_sweep]

    assert all(np.isfinite(errors))
    _slope(hs, errors)

    # Finalized against the measured log in
    # docs/design/phase16a_convergence_suite.md: finest (n=40) measured
    # 4.66e-4; 0.002 keeps a ~4x margin over a series with a small
    # non-monotonic bump at n=20 (4.87e-3 -> 5.08e-3), well below the
    # provisional/policy default (0.05).
    assert errors[-1] <= 0.002
    assert errors[-1] < errors[0]
