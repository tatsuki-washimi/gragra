"""Tests for the SPECFEM3D GLL numerical core (``gragra.adapters.specfem.gll``).

The tests cover the GLL numerical design and the A-series scientific checks:

- The GLL rule is pinned by *analytic* values (n=5 abscissae/weights in closed
  form) plus a **two-sided** polynomial-exactness test. One-sided exactness is
  not enough: a Gauss-Legendre rule of the same point count is exact to a
  *higher* degree, so "exact at degree 2n-3" alone does not identify the rule.
  Asserting that it is exact at 2n-3 **and inexact at 2n-2** does.
- The quadrature volume is pinned against the two independent oracles in
  ``tests/_hex_volume_oracle.py`` (24-tetra closed form and 2x2x2 Gauss),
  which by design do not import the module under test.
- The orientation guard (``detJ <= 0`` -> ``ValueError``) and the recentring
  guard (translation invariance at a 1e6 m offset) both get tests that would
  fail if the corresponding safeguard were removed -- the offset test is not
  cosmetic: without recentring the same element differs by ~5e-11 relative,
  i.e. ~50x the Tier A2 rtol < 1e-12 check.
"""

import numpy as np
import pytest
from _hex_volume_oracle import hex_volume_24tetra, hex_volume_gauss2

from gragra.adapters.specfem import gll, schema

# ---------------------------------------------------------------------------
# Shared element fixtures
# ---------------------------------------------------------------------------

#: A non-trivially warped hexahedron (no two edges parallel, all 6 faces
#: non-planar). Every coordinate is a multiple of 1/16, hence exactly
#: representable in float64 *and* exactly shiftable by _LARGE_OFFSET below --
#: so the translation test measures the implementation, not the quantization
#: of the shifted input coordinates.
WARPED_CORNERS = np.array(
    [
        [0.0000, 0.0000, 0.0000],
        [1.3125, -0.1250, 0.0625],
        [1.4375, 1.1250, -0.1875],
        [0.1250, 0.9375, 0.1875],
        [-0.0625, 0.1875, 1.1875],
        [1.1875, -0.1875, 0.9375],
        [1.5625, 1.2500, 1.3750],
        [0.1875, 1.0625, 1.1250],
    ]
)

#: Absolute-coordinate offset used by the recentring tests (metres); a
#: realistic magnitude for a UTM-like SPECFEM mesh.
LARGE_OFFSET = 1.0e6


def axis_aligned_box(a: float, b: float, c: float, origin=(0.0, 0.0, 0.0)):
    """Corners of an ``a x b x c`` axis-aligned box in ``usual_hex_nodes``
    order (see ``gll`` module docstring / ``tests/_hex_volume_oracle.py``)."""
    x0, y0, z0 = origin
    return np.array(
        [
            [x0, y0, z0],
            [x0 + a, y0, z0],
            [x0 + a, y0 + b, z0],
            [x0, y0 + b, z0],
            [x0, y0, z0 + c],
            [x0 + a, y0, z0 + c],
            [x0 + a, y0 + b, z0 + c],
            [x0, y0 + b, z0 + c],
        ],
        dtype=np.float64,
    )


# ===========================================================================
# 1. GLL points and weights
# ===========================================================================


def test_gll_n5_matches_closed_form_points_and_weights():
    """n=5 (the documented profile's NGLL) against the textbook closed form."""
    points, weights = gll.gll_points_weights(5)
    root = np.sqrt(3.0 / 7.0)
    np.testing.assert_allclose(
        points, [-1.0, -root, 0.0, root, 1.0], rtol=0.0, atol=1e-15
    )
    np.testing.assert_allclose(
        weights, [1 / 10, 49 / 90, 32 / 45, 49 / 90, 1 / 10], rtol=0.0, atol=1e-15
    )


def test_gll_n2_and_n3_are_the_trivial_rules():
    points2, weights2 = gll.gll_points_weights(2)
    np.testing.assert_array_equal(points2, [-1.0, 1.0])
    np.testing.assert_allclose(weights2, [1.0, 1.0], rtol=0.0, atol=1e-15)

    points3, weights3 = gll.gll_points_weights(3)
    np.testing.assert_array_equal(points3, [-1.0, 0.0, 1.0])
    np.testing.assert_allclose(weights3, [1 / 3, 4 / 3, 1 / 3], rtol=0.0, atol=1e-15)


@pytest.mark.parametrize("n", range(2, 13))
def test_gll_weights_sum_to_two(n):
    """The rule integrates f == 1 over [-1, 1] exactly."""
    _points, weights = gll.gll_points_weights(n)
    assert float(np.sum(weights)) == pytest.approx(2.0, rel=0.0, abs=1e-15)


@pytest.mark.parametrize("n", range(2, 13))
def test_gll_points_are_ascending_symmetric_and_endpoint_anchored(n):
    """Endpoints exact, strictly ascending, and exactly antisymmetric.

    Exactness (rather than mere closeness) of the endpoints and of the
    symmetry is what upstream ``zwgljd`` also guarantees (it assigns
    ``z(1) = -1``, ``z(np) = 1`` and, for Legendre with odd ``np``, an exact
    zero midpoint), so a flatten/ordering bug can never be masked by a
    1e-16 asymmetry.
    """
    points, weights = gll.gll_points_weights(n)
    assert points[0] == -1.0
    assert points[-1] == 1.0
    assert np.all(np.diff(points) > 0.0)
    np.testing.assert_array_equal(points, -points[::-1])
    np.testing.assert_array_equal(weights, weights[::-1])
    if n % 2 == 1:
        assert points[(n - 1) // 2] == 0.0


def test_gll_points_weights_returns_read_only_arrays():
    points, weights = gll.gll_points_weights(5)
    assert points.flags.writeable is False
    assert weights.flags.writeable is False
    assert points.dtype == np.float64
    assert weights.dtype == np.float64


@pytest.mark.parametrize("n", [1, 0, -3])
def test_gll_rejects_fewer_than_two_points(n):
    with pytest.raises(ValueError, match="at least 2 points"):
        gll.gll_points_weights(n)


def test_gll_rejects_non_integer_point_count():
    with pytest.raises(TypeError, match="must be an int"):
        gll.gll_points_weights(5.0)


def _quadrature(points, weights, degree):
    """Quadrature estimate of int_{-1}^{1} (1 + x)^degree dx."""
    return float(np.sum(weights * (1.0 + points) ** degree))


def _exact_shifted_monomial_integral(degree):
    """int_{-1}^{1} (1 + x)^degree dx = 2^(degree+1) / (degree + 1)."""
    return 2.0 ** (degree + 1) / (degree + 1)


def test_gll_n5_is_exact_at_degree_7_and_inexact_at_degree_8():
    """The headline two-sided exactness case for the documented NGLL=5 rule.

    ``(1 + x)^d`` is used rather than ``x^d`` because an odd monomial
    integrates to zero under *any* symmetric rule, exact or not; the shifted
    monomial has every power present, so nothing can cancel accidentally.

    The inexact side matters as much as the exact side: a 5-point
    Gauss-Legendre rule would pass the degree-7 assertion (it is exact to
    degree 9) and fail only this one.
    """
    points, weights = gll.gll_points_weights(5)

    exact7 = _exact_shifted_monomial_integral(7)
    assert _quadrature(points, weights, 7) == pytest.approx(exact7, rel=1e-14)

    exact8 = _exact_shifted_monomial_integral(8)
    residual8 = abs(_quadrature(points, weights, 8) - exact8) / exact8
    assert residual8 > 1e-5  # measured 2.55e-04


@pytest.mark.parametrize("n", range(3, 8))
def test_gll_exactness_degree_is_exactly_2n_minus_3(n):
    """Same two-sided property across point counts: exact at 2n-3, not at 2n-2."""
    points, weights = gll.gll_points_weights(n)

    exact_degree = 2 * n - 3
    reference = _exact_shifted_monomial_integral(exact_degree)
    assert _quadrature(points, weights, exact_degree) == pytest.approx(
        reference, rel=1e-14
    )

    beyond = 2 * n - 2
    reference_beyond = _exact_shifted_monomial_integral(beyond)
    residual = (
        abs(_quadrature(points, weights, beyond) - reference_beyond) / reference_beyond
    )
    assert residual > 1e-7  # measured 4.2e-02 (n=3) ... 1.4e-06 (n=7)


def test_rule_identity_matches_the_profiles_jacobi_weights():
    """``gll_points_weights`` implements the alpha=beta=0 (Legendre) rule.

    If the documented profile ever moved to a non-zero Gauss-Jacobi weight,
    this module's rule would no longer be the solver's rule -- so pin the
    assumption against the schema's recorded upstream expectation rather
    than leaving it in prose only.
    """
    assert schema.EXPECTED_GAUSSALPHA == 0.0
    assert schema.EXPECTED_GAUSSBETA == 0.0


# ===========================================================================
# 2. Quadrature volume -- analytic and oracle cross-checks
# ===========================================================================


def test_affine_box_volume_matches_analytic_product():
    a, b, c = 1.5, 2.5, 0.5
    quadrature = gll.element_quadrature_volume(
        axis_aligned_box(a, b, c), schema.EXPECTED_NGLL
    )
    assert quadrature.shape == (schema.ngll_from_shape(schema.EXPECTED_NGLL),)
    assert float(quadrature.sum()) == pytest.approx(a * b * c, rel=1e-14)
    assert np.all(quadrature > 0.0)


def test_affine_box_per_point_values_follow_the_schema_flatten_order():
    """Anisotropic ``ngll_shape`` pins the (k, j, i) C-order flatten.

    For an affine box ``detJ`` is the constant ``a*b*c/8``, so each entry must
    be that constant times ``w_x[i] w_y[j] w_z[k]``. With three *different*
    point counts per axis the three weight vectors differ in both length and
    value, so any transposed/permuted flatten produces a mismatch here.
    """
    ngll_shape = (5, 4, 3)
    a, b, c = 1.5, 2.5, 0.5
    quadrature = gll.element_quadrature_volume(axis_aligned_box(a, b, c), ngll_shape)
    assert quadrature.shape == (schema.ngll_from_shape(ngll_shape),)

    _px, weights_x = gll.gll_points_weights(ngll_shape[0])
    _py, weights_y = gll.gll_points_weights(ngll_shape[1])
    _pz, weights_z = gll.gll_points_weights(ngll_shape[2])
    det = a * b * c / 8.0

    expected = np.empty_like(quadrature)
    for i in range(ngll_shape[0]):
        for j in range(ngll_shape[1]):
            for k in range(ngll_shape[2]):
                flat = schema.flatten_index(i, j, k, ngll_shape)
                expected[flat] = det * weights_x[i] * weights_y[j] * weights_z[k]
    np.testing.assert_allclose(quadrature, expected, rtol=1e-15, atol=0.0)


@pytest.mark.parametrize("ngll_shape", [(5, 5, 5), (5, 4, 3), (3, 3, 3), (4, 6, 5)])
def test_warped_element_matches_both_independent_volume_oracles(ngll_shape):
    """The A2 check at element scope, against both oracles.

    Exactness (rather than convergence) is expected for every ``NGLL* >= 3``:
    the trilinear ``detJ`` is degree <= 2 per reference axis, and a GLL rule
    with n points is exact to degree 2n-3 >= 3. This simultaneously validates
    the GLL node positions and the individual weights -- an error in either
    would perturb the sum, since ``detJ`` is genuinely non-constant here.
    """
    quadrature = gll.element_quadrature_volume(WARPED_CORNERS, ngll_shape)
    total = float(quadrature.sum())
    assert total == pytest.approx(hex_volume_24tetra(WARPED_CORNERS), rel=1e-12)
    assert total == pytest.approx(hex_volume_gauss2(WARPED_CORNERS), rel=1e-12)


def test_warped_element_quadrature_is_positive_everywhere():
    quadrature = gll.element_quadrature_volume(WARPED_CORNERS, schema.EXPECTED_NGLL)
    assert np.all(quadrature > 0.0)


# ===========================================================================
# 3. Recentring / translation invariance
# ===========================================================================


def test_large_translation_leaves_the_quadrature_volume_invariant():
    """Recentring guard, measured as a pure translation.

    Each coordinate of ``WARPED_CORNERS`` is a multiple of 1/16, so adding
    1e6 m is *exact* in float64 (verified below) and the shifted element is
    geometrically identical to the original. Without the corner-centroid
    recentring inside the Jacobian the two sums differ by ~5e-11 relative --
    ~50x the Tier A2 rtol < 1e-12 check -- because the Jacobian columns are
    sums whose shape-function coefficients cancel to zero.
    """
    shifted = WARPED_CORNERS + LARGE_OFFSET
    np.testing.assert_array_equal(shifted - LARGE_OFFSET, WARPED_CORNERS)

    base = float(gll.element_quadrature_volume(WARPED_CORNERS, (5, 5, 5)).sum())
    moved = float(gll.element_quadrature_volume(shifted, (5, 5, 5)).sum())
    assert moved == pytest.approx(base, rel=1e-12)


def test_translated_element_still_matches_the_volume_oracle():
    """Same guard expressed the way Tier A2 applies it on real data:
    exporter sum vs. independent oracle, both on large absolute coordinates."""
    shifted = WARPED_CORNERS + np.array([4.0e6, -3.0e6, 2.0e6])
    total = float(gll.element_quadrature_volume(shifted, (5, 5, 5)).sum())
    assert total == pytest.approx(hex_volume_24tetra(shifted), rel=1e-12)
    assert total == pytest.approx(hex_volume_gauss2(shifted), rel=1e-12)


# ===========================================================================
# 4. Orientation / degeneracy guard
# ===========================================================================


def test_orientation_flipped_element_raises_and_names_the_gll_index():
    """A mirrored corner ordering (top and bottom faces swapped) makes detJ
    negative everywhere; this is the only defence against a reflected
    ordering, since the A1 index checks pin rotations only."""
    flipped = WARPED_CORNERS[[4, 5, 6, 7, 0, 1, 2, 3]]
    with pytest.raises(ValueError) as excinfo:
        gll.element_quadrature_volume(flipped, schema.EXPECTED_NGLL)
    message = str(excinfo.value)
    assert "GLL flat index 0 (i=0, j=0, k=0)" in message
    assert "usual_hex_nodes" in message
    # Single-element entry point: no meaningless "element index 0" prefix.
    assert "element index" not in message


def test_degenerate_element_reports_the_actual_offending_gll_index():
    """Collapsing the top face to a point leaves detJ positive on the lower
    GLL planes and exactly zero on the top plane, so the reported index must
    be the first point of the k = NGLLZ-1 plane -- not a hardcoded 0."""
    pyramid = WARPED_CORNERS.copy()
    pyramid[4:] = np.array([0.75, 0.5, 1.375])
    with pytest.raises(ValueError) as excinfo:
        gll.element_quadrature_volume(pyramid, schema.EXPECTED_NGLL)
    message = str(excinfo.value)
    expected_flat = schema.flatten_index(0, 0, 4, schema.EXPECTED_NGLL)
    assert expected_flat == 100
    assert f"GLL flat index {expected_flat} (i=0, j=0, k=4)" in message


def test_batch_reports_the_offending_element_index():
    batch = np.stack(
        [
            WARPED_CORNERS,
            WARPED_CORNERS[[4, 5, 6, 7, 0, 1, 2, 3]],
            axis_aligned_box(1.0, 1.0, 1.0),
        ]
    )
    with pytest.raises(ValueError, match=r"element index 1, GLL flat index 0"):
        gll.elements_quadrature_volume(batch, schema.EXPECTED_NGLL)


def test_batch_element_index_is_absolute_across_blocks(monkeypatch):
    """The reported element index must be the global one even when the
    offending element falls in a later processing block."""
    monkeypatch.setattr(gll, "_ELEMENT_BLOCK_SIZE", 2)
    batch = np.stack([WARPED_CORNERS] * 5 + [WARPED_CORNERS[[4, 5, 6, 7, 0, 1, 2, 3]]])
    with pytest.raises(ValueError, match=r"element index 5, GLL flat index 0"):
        gll.elements_quadrature_volume(batch, schema.EXPECTED_NGLL)


# ===========================================================================
# 5. Batched API
# ===========================================================================


def test_batch_rows_equal_the_single_element_results():
    batch = np.stack(
        [
            axis_aligned_box(1.5, 2.5, 0.5),
            WARPED_CORNERS,
            WARPED_CORNERS + LARGE_OFFSET,
            WARPED_CORNERS * 2.0,
        ]
    )
    quadrature = gll.elements_quadrature_volume(batch, schema.EXPECTED_NGLL)
    assert quadrature.shape == (4, schema.ngll_from_shape(schema.EXPECTED_NGLL))
    for index in range(batch.shape[0]):
        single = gll.element_quadrature_volume(batch[index], schema.EXPECTED_NGLL)
        np.testing.assert_allclose(quadrature[index], single, rtol=1e-15, atol=0.0)


def test_batch_blocking_does_not_change_the_result(monkeypatch):
    """Block size is a memory knob, not a numerical parameter: unlike the
    kernel contract's ``chunk_size`` it does not chunk a reduction, so the
    output must be bit-identical."""
    batch = np.stack([WARPED_CORNERS + shift for shift in np.linspace(0.0, 3.0, 7)])
    reference = gll.elements_quadrature_volume(batch, schema.EXPECTED_NGLL)
    for block in (1, 2, 3, 1000):
        monkeypatch.setattr(gll, "_ELEMENT_BLOCK_SIZE", block)
        np.testing.assert_array_equal(
            gll.elements_quadrature_volume(batch, schema.EXPECTED_NGLL), reference
        )


def test_empty_batch_returns_an_empty_table():
    quadrature = gll.elements_quadrature_volume(
        np.zeros((0, 8, 3)), schema.EXPECTED_NGLL
    )
    assert quadrature.shape == (0, schema.ngll_from_shape(schema.EXPECTED_NGLL))
    assert quadrature.dtype == np.float64


def test_quadrature_volume_arrays_are_read_only():
    single = gll.element_quadrature_volume(WARPED_CORNERS, schema.EXPECTED_NGLL)
    batched = gll.elements_quadrature_volume(
        WARPED_CORNERS[None, ...], schema.EXPECTED_NGLL
    )
    assert single.flags.writeable is False
    assert batched.flags.writeable is False
    assert single.dtype == np.float64
    assert batched.dtype == np.float64


# ===========================================================================
# 6. Input validation / numerical defence
# ===========================================================================


@pytest.mark.parametrize(
    "corners",
    [
        np.zeros((7, 3)),
        np.zeros((8, 2)),
        np.zeros(3),
        np.zeros((1, 8, 3)),
    ],
)
def test_single_element_rejects_wrong_shape(corners):
    with pytest.raises(ValueError, match=r"must have shape \(8, 3\)"):
        gll.element_quadrature_volume(corners, schema.EXPECTED_NGLL)


@pytest.mark.parametrize(
    "corners",
    [
        np.zeros((8, 3)),
        np.zeros((2, 8, 2)),
        np.zeros((2, 7, 3)),
    ],
)
def test_batch_rejects_wrong_shape(corners):
    with pytest.raises(ValueError, match=r"must have shape \(nspec, 8, 3\)"):
        gll.elements_quadrature_volume(corners, schema.EXPECTED_NGLL)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_non_finite_corners_are_rejected(bad_value):
    corners = WARPED_CORNERS.copy()
    corners[3, 1] = bad_value
    with pytest.raises(ValueError, match="finite"):
        gll.element_quadrature_volume(corners, schema.EXPECTED_NGLL)
    with pytest.raises(ValueError, match="finite"):
        gll.elements_quadrature_volume(corners[None, ...], schema.EXPECTED_NGLL)


@pytest.mark.parametrize("ngll_shape", [(5, 5), (5, 5, 5, 5), (5, 0, 5), (5, -1, 5)])
def test_invalid_ngll_shape_is_rejected(ngll_shape):
    with pytest.raises(ValueError, match="ngll_shape"):
        gll.element_quadrature_volume(WARPED_CORNERS, ngll_shape)


def test_documented_profile_shape_produces_125_points():
    """The documented profile's NGLL is (5, 5, 5); keep the module and the
    schema's recorded upstream expectation tied together."""
    assert schema.EXPECTED_NGLL == (5, 5, 5)
    quadrature = gll.element_quadrature_volume(WARPED_CORNERS, schema.EXPECTED_NGLL)
    assert quadrature.size == 125
