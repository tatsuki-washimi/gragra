"""Phase 16-a convergence suite: Half primitives vs mirror-image identity.

Design: docs/design/phase16a_convergence_suite.md Sec.8. ``HalfSphere`` /
``HalfCylinder`` are floor-mounted half-bodies (mass occupies z >= center_z
only). Mirroring a half-body across its z=0 floor plane reconstructs the
full body (``Sphere`` / ``Cylinder``); at a target ON that floor plane, the
source point and its mirror image produce equal in-plane (x, y)
acceleration contributions (additive) and equal-magnitude, opposite-sign
normal (z) contributions (cancelling). Hence, on the floor plane:

    a_full(P) = [2 * a_half_x(P), 2 * a_half_y(P), 0]

This lets the existing closed forms (``uniform_sphere_acceleration`` /
``cylinder_filled_axis_field``) verify the half-primitive discretization
without deriving any new analytic reference. ``a_half``'s own z-component
(the dome's non-zero normal-direction field) is NOT the identity's claim --
only the in-plane components are (see design doc Sec.8.1 for the DOF
count: HalfSphere verifies 2 in-plane components, HalfCylinder verifies 1
on-axis component).

Targets are restricted to the floor plane AND outside the body's
footprint (exterior): interior floor-plane points sit immediately above
the half-body's densest source layer (near-field domination), which the
identity's derivation does not require and which would confound a clean
h-refinement signal. See design doc Sec.8.3.

Marked ``slow`` so it is deselected by the default ``pytest`` run
(``addopts = "-m 'not slow'"`` in pyproject.toml):

    pytest -m slow tests/test_convergence_half_primitives.py -v
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
from gragra.analytic import cylinder_filled_axis_field, uniform_sphere_acceleration
from gragra.geometry import HalfCylinder, HalfSphere

pytestmark = pytest.mark.slow

_R = 1.0
_RHO = 1000.0
_CENTER = np.array([0.0, 0.0, 0.0])
_NS = [5, 10, 20, 40]  # spacing_m = _R / n, same range as the Cylinder suite
_HC_LENGTH = 3.0  # HalfCylinder axial extent (maps to analytic height_m)

# HalfSphere target: on the floor plane (z=0), outside the footprint
# (hypot(x,y) > _R), grid-non-aligned (non-round) coordinates.
_HS_TARGET = np.array([[1.83, 0.57, 0.0]])
assert _HS_TARGET[0, 2] == _CENTER[2], "HalfSphere target must be on the floor plane"
assert np.hypot(_HS_TARGET[0, 0], _HS_TARGET[0, 1]) > _R, (
    "HalfSphere target must be outside the footprint (exterior)"
)

# HalfCylinder target: on the axis (y=z=0, matching the analytic reference's
# on-axis restriction), outside the footprint (|x_off| > _HC_LENGTH/2),
# non-round.
_HC_TARGET = np.array([[2.37, 0.0, 0.0]])
assert _HC_TARGET[0, 1] == _CENTER[1] and _HC_TARGET[0, 2] == _CENTER[2], (
    "HalfCylinder target must be on-axis"
)
assert abs(_HC_TARGET[0, 0] - _CENTER[0]) > _HC_LENGTH / 2.0, (
    "HalfCylinder target must be outside the footprint (exterior)"
)

# Same geometric sanity bound as the Cuboid/Cylinder suites (see
# test_convergence_cuboid.py's rationale comment for the 0.05 -> 0.2 history).
_MIN_D_OVER_SPACING = 0.2

# Fixed round-off floor for HalfCylinder's a_y residual (see
# test_half_cylinder_axial_residual_bounded below for the mechanism).
_HC_Y_RESIDUAL_CEILING = 1e-10


def _discretized_half_sphere(spacing_m):
    # max_points=None: not required at this sweep's n<=40 (HalfSphere n=40
    # needs ~2*R/spacing_m cells per axis, well under the 1,000,000 default),
    # but set explicitly following the Cuboid/Cylinder suite precedent
    # (test_convergence_cuboid.py:85) and in case a future higher-resolution
    # point is added (n=80 was probed and exceeds the default).
    return HalfSphere(
        center_m=_CENTER,
        radius_m=_R,
        density_kg_m3=_RHO,
        spacing_m=spacing_m,
        max_points=None,
    )


def _discretized_half_cylinder(spacing_m):
    return HalfCylinder(
        center_m=_CENTER,
        radius_m=_R,
        length_m=_HC_LENGTH,
        density_kg_m3=_RHO,
        spacing_m=spacing_m,
        max_points=None,
    )


@pytest.fixture(scope="module")
def half_sphere_sweep():
    """Build each resolution's discretized HalfSphere once; keep summaries only.

    Lazily evaluated (pytest fixtures only execute when a dependent test
    actually runs) -- see test_convergence_cylinder.py's ``cylinder_sweep``
    for the identical rationale.
    """
    exact_full = G_SI * uniform_sphere_acceleration(_CENTER, _R, _RHO, _HS_TARGET)[0]
    # Bit-exact (not approximate) because rel_pos_z = target_z - center_z
    # subtracts two IDENTICAL float64 values (both literal 0.0 here), which
    # IEEE 754 subtraction always returns as exact zero. The essential
    # condition is that target_z and center_z are the same stored float
    # value, not that the body is centered at the origin per se -- an
    # off-origin fixture that reuses the identical stored coordinate for
    # both would retain this property, but two independently computed
    # coordinates that are mathematically equal are not guaranteed to be
    # bit-identical, so this must not be assumed without re-deriving the
    # floating-point argument for a new fixture.
    assert exact_full[2] == 0.0, (
        "full-sphere z-component at a floor-plane target must be exact zero "
        "-- the mirror identity has no analytic z to compare against"
    )

    summaries = []
    for n in _NS:
        spacing_m = _R / n
        hs = _discretized_half_sphere(spacing_m)
        src = hs.to_weighted_points()

        half_acc = point_acceleration(
            src, _HS_TARGET, backend="numpy", chunk_size=None
        )[0]
        # z is fixed to 0.0 here (not measured from half_acc[2]): the mirror
        # identity only claims the in-plane (x, y) components are doubled;
        # a_half_z is the dome's own legitimate non-zero normal-direction
        # field and is not the identity's subject (design doc Sec.8.1).
        doubled_inplane = np.array([2.0 * half_acc[0], 2.0 * half_acc[1], 0.0])

        d_min = _min_distance(src.positions_m, _HS_TARGET[0])
        # Reuse the primitive's own analytic_volume_m3 (not a hand-rewritten
        # formula) to avoid duplicating the (2/3)*pi*R^3 expression.
        expected_mass = hs.analytic_volume_m3 * _RHO
        mass = float(np.sum(src.weights_kg))

        summaries.append(
            {
                "n": n,
                "spacing_m": spacing_m,
                "point_count": int(src.positions_m.shape[0]),
                "d_min_over_spacing": d_min / spacing_m,
                "mass": mass,
                "mass_error": _relative_error_scalar(mass, expected_mass),
                "vector_error": _relative_error_vector(doubled_inplane, exact_full),
                "contains_all": bool(np.all(hs.contains(src.positions_m))),
            }
        )
    return summaries


@pytest.fixture(scope="module")
def half_cylinder_sweep():
    """Build each resolution's discretized HalfCylinder once; keep summaries only."""
    x_off = float(_HC_TARGET[0, 0] - _CENTER[0])
    # cylinder_filled_axis_field is derived for axis=z; HalfCylinder's axis
    # is x. No sign flip is needed in this relabeling because the on-axis
    # field always points toward the mass distribution's center regardless
    # of which coordinate is called the axis -- the analytic closed form's
    # sign is tied to the *signed offset from center* (x_off here), not to
    # a fixed global axis label, so passing x_off in the z_m slot preserves
    # the correct sign. Confirmed numerically (design doc Sec.8.2).
    exact_axial = G_SI * float(cylinder_filled_axis_field(_R, _HC_LENGTH, x_off, _RHO))

    summaries = []
    for n in _NS:
        spacing_m = _R / n
        hc = _discretized_half_cylinder(spacing_m)
        src = hc.to_weighted_points()

        half_acc = point_acceleration(
            src, _HC_TARGET, backend="numpy", chunk_size=None
        )[0]

        d_min = _min_distance(src.positions_m, _HC_TARGET[0])
        expected_mass = hc.analytic_volume_m3 * _RHO
        mass = float(np.sum(src.weights_kg))

        summaries.append(
            {
                "n": n,
                "spacing_m": spacing_m,
                "point_count": int(src.positions_m.shape[0]),
                "d_min_over_spacing": d_min / spacing_m,
                "mass": mass,
                "mass_error": _relative_error_scalar(mass, expected_mass),
                # _relative_error_scalar(num, exact): 2nd arg is the
                # denominator -- exact_axial is the reference, not half_acc.
                "x_error": _relative_error_scalar(2.0 * half_acc[0], exact_axial),
                # Denominator is the analytic half-amplitude (|exact_axial|/2,
                # i.e. the expected magnitude of a_half_x), not the numeric
                # a_half_y itself -- normalizing a round-off residual against
                # its own value would be meaningless.
                "y_residual_ratio": abs(half_acc[1]) / (abs(exact_axial) / 2.0),
                "contains_all": bool(np.all(hc.contains(src.positions_m))),
            }
        )
    return summaries


def test_half_sphere_targets_are_geometrically_well_separated(half_sphere_sweep):
    """d_min/spacing sanity bound, fixed independently of any error result."""
    for s in half_sphere_sweep:
        assert s["d_min_over_spacing"] > _MIN_D_OVER_SPACING, s


def test_half_sphere_points_are_contained(half_sphere_sweep):
    """Sanity check: every point to_weighted_points() generates is accepted by contains.

    This is a one-directional (soundness-only) check, not a proof that the
    mask and contains() are fully equivalent -- a point that to_weighted_points()
    wrongly dropped, but that should have been inside, would not be caught
    here. Both use algebraically equivalent inequalities (norm(dp) <= r vs.
    sum(dp**2) <= r**2), but float evaluation could in principle flip a
    boundary cell by 1 ULP for a cell center that happens to land
    arbitrarily close to the curved boundary. Measured: all n, all points
    True.
    """
    for s in half_sphere_sweep:
        assert s["contains_all"] is True, s


def test_half_sphere_mass_converges(half_sphere_sweep):
    """Overall trend only -- curved-boundary staircase is not guaranteed monotonic.

    Same cell-center-inclusion staircase behavior as Sphere/Cylinder (see
    test_convergence_cylinder.py's ``test_cylinder_mass_converges_overall``
    docstring); only the coarsest-vs-finest trend and an absolute
    finest-resolution bound are asserted, slope is recorded (design doc),
    not gated.
    """
    mass_errors = [s["mass_error"] for s in half_sphere_sweep]
    hs = [s["spacing_m"] for s in half_sphere_sweep]

    assert all(np.isfinite(mass_errors))
    # Recorded (measured slope ~3.30), not gated -- see the mirror-identity
    # test below for the superconvergence rationale that applies equally here.
    _slope(hs, mass_errors)

    assert mass_errors[-1] < mass_errors[0]
    # Finalized against the measured log in
    # docs/design/phase16a_convergence_suite.md: finest (n=40) measured
    # 5.01e-5; 1e-4 keeps a ~2.0x margin (Cuboid/Cylinder precedent).
    assert mass_errors[-1] <= 1e-4


def test_half_sphere_mirror_identity_converges(half_sphere_sweep):
    """Primary gate: 2*a_half|_xy converges to the full-sphere closed form.

    z is excluded from the comparison by construction (fixed to 0.0 in the
    fixture) -- it is a 0=0 identity match that contributes no error signal,
    not a measured quantity.
    """
    errors = [s["vector_error"] for s in half_sphere_sweep]
    hs = [s["spacing_m"] for s in half_sphere_sweep]

    assert all(np.isfinite(errors))
    # Recorded (measured slope ~3.17), not gated (design doc Sec.8.5). This
    # is an empirical fit over this four-level sweep only -- it is not used
    # as a gate and is not a formal convergence-order proof (a curved
    # cell-center-inclusion boundary still has O(h) staircase error at the
    # source; a resolution-series-specific cancellation or a
    # subsequence-dependent apparent higher order cannot be ruled out from
    # 4 points). The target sits at |r|~1.92*_R, just outside the body, so
    # a smooth exterior field plausibly permits faster-than-O(h^2)
    # empirical convergence than an interior near-field target would (the
    # Cuboid exterior suite similarly measured slope ~3.6); a slope above
    # the O(h^2) design target here is not a bug.
    _slope(hs, errors)

    # Finalized against the measured log in
    # docs/design/phase16a_convergence_suite.md: finest (n=40) measured
    # 6.82e-5; 1.4e-4 keeps a ~2.1x margin.
    assert errors[-1] <= 1.4e-4
    assert errors[-1] < errors[0]


def test_half_cylinder_targets_are_geometrically_well_separated(half_cylinder_sweep):
    """d_min/spacing sanity bound, fixed independently of any error result."""
    for s in half_cylinder_sweep:
        assert s["d_min_over_spacing"] > _MIN_D_OVER_SPACING, s


def test_half_cylinder_points_are_contained(half_cylinder_sweep):
    """Sanity check: every point to_weighted_points() generates is accepted by contains.

    One-directional (soundness-only) check -- see
    test_half_sphere_points_are_contained's docstring for the full
    rationale (this is not a proof that the mask and contains() predicate
    are fully equivalent).
    """
    for s in half_cylinder_sweep:
        assert s["contains_all"] is True, s


def test_half_cylinder_mass_converges(half_cylinder_sweep):
    """Overall trend only; also documents a structural (not accidental) match.

    Same circular-cross-section staircase non-monotonicity as the full
    Cylinder suite (n=10/n=20 measured an identical relative error there);
    this half-cylinder's mass_error series is expected to match the full
    Cylinder suite's (docs/design/phase16a_convergence_suite.md measured
    log) at every n -- NOT a transcription accident, but a structural
    necessity. NOTE: this is a cross-sectional-error-rate equality, not a
    total-mass 1:2 claim -- the two bodies compared (this suite's
    length_m=3.0 vs the full-Cylinder suite's height_m=1.5) are NOT in a
    1:2 mass ratio and do not have half the point count. What actually
    transfers is that mass_error depends only on the 2-D circular
    cross-section's staircase approximation (radius R, spacing R/n): with
    per-axis grid spacing, the axial (length) direction is fully filled (no
    boundary staircase along x) and contributes exactly zero relative
    error, so mass_error reduces to the cross-section's relative error
    alone. A half-cylinder's cross-section cell centers are exactly the
    z>=0 half of the full cylinder's y-z-symmetric grid, so the half's
    cross-sectional point count and area are each exactly half the full
    cylinder's -- the 1/2 factor cancels in the ratio, and the match holds
    for ANY length_m (probed independently at length_m in {1.5, 2.2, 3.0}:
    mass_error was identical across all three at every n, and independent
    of the full-Cylinder suite's unrelated height_m=1.5).
    """
    mass_errors = [s["mass_error"] for s in half_cylinder_sweep]
    hs = [s["spacing_m"] for s in half_cylinder_sweep]

    assert all(np.isfinite(mass_errors))
    # Recorded (measured slope ~1.56), not gated.
    _slope(hs, mass_errors)

    assert mass_errors[-1] < mass_errors[0]
    # Finalized against the measured log in
    # docs/design/phase16a_convergence_suite.md: n=10 and n=20 measured an
    # *identical* 5.86e-3 (staircase bump, same as the full Cylinder suite);
    # finest (n=40) measured 5.07e-4. 1e-3 keeps a ~2.0x margin.
    assert mass_errors[-1] <= 1e-3


def test_half_cylinder_axial_residual_bounded(half_cylinder_sweep):
    """a_half_y stays at a fixed round-off floor, not an O(h) discretization error.

    HalfCylinder's bounding box is symmetric in y (y in [-R, R] about the
    axis) and its mask (radial_dist <= radius & z >= 0) is symmetric under
    y -> -y, so source cells always appear in +/-y pairs whose
    y-direction contributions cancel to a round-off floor (NOT bit-exact 0
    -- the cancellation depends on floating-point summation order), while
    their x-direction contributions add (this is what the mirror-identity
    test above compares against the analytic axial field). The
    ``y_residual_ratio`` recorded above is a NORMALIZED ratio
    (|a_half_y| / (|exact_axial| / 2), i.e. relative to the expected
    a_half_x magnitude), not the absolute a_y value -- measured ~2.2e-17 to
    3.8e-17 for this ratio across the swept n (the absolute a_y itself is
    far smaller, ~1e-24). A conservative estimate for the same ratio is
    eps*sqrt(N) ~ 2.2e-16*sqrt(3e5) ~ 1.2e-13 for the largest N in this
    sweep. The fixed ceiling below (1e-10)
    keeps ~800x headroom over that conservative estimate (and ~2.5e6x over
    the measurement) to avoid flakiness, following the same fixed-floor
    treatment as PR-5's bulk/P-wall residuals. This differs in mechanism
    from PR-5's S-wall residual (UV-mesh diagonal splitting breaks the
    symmetry there, giving an O(h^2) discretization error instead of a
    round-off floor) -- the two must not be conflated.
    """
    for s in half_cylinder_sweep:
        assert s["y_residual_ratio"] <= _HC_Y_RESIDUAL_CEILING, s


def test_half_cylinder_mirror_identity_converges(half_cylinder_sweep):
    """Primary gate: 2*a_half_x converges to the full-cylinder on-axis closed form.

    Only the x (axial) component is compared -- the analytic reference is
    on-axis-only (see cylinder_filled_axis_field's docstring), so this is
    a 1-component check by construction, not a simplification. a_half_y is
    covered separately (test_half_cylinder_axial_residual_bounded);
    a_half_z is not claimed by the identity at all.
    """
    errors = [s["x_error"] for s in half_cylinder_sweep]
    hs = [s["spacing_m"] for s in half_cylinder_sweep]

    assert all(np.isfinite(errors))
    # Recorded (measured slope ~1.46), not gated. A non-monotonic bump
    # between intermediate n is expected (circular cross-section staircase
    # O(h) boundary error, compounded by a moderately close exterior
    # target) and a slope well below the O(h^2) design target is not a bug.
    _slope(hs, errors)

    # Finalized against the measured log in
    # docs/design/phase16a_convergence_suite.md: n=10->20 measured a small
    # non-monotonic bump (4.03e-3 -> 4.37e-3); finest (n=40) measured
    # 4.18e-4. 8.4e-4 keeps a ~2.0x margin.
    assert errors[-1] <= 8.4e-4
    assert errors[-1] < errors[0]
