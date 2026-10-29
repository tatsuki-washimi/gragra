"""Tests for directional power average and directional cross average."""

import numpy as np
import pytest

from gragra.coherent.average import directional_cross_average, directional_power_average


def test_directional_power_average_basics():
    # response of shape (2, 3, 4)
    # direction_axis = 1, so len(weights) must be 3
    response = np.arange(24, dtype=np.complex128).reshape(2, 3, 4)
    direction_weights = np.array([0.2, 0.3, 0.5])

    res = directional_power_average(response, direction_weights, direction_axis=1)

    # Shape should be (2, 4)
    assert res.shape == (2, 4)
    assert res.dtype == np.float64
    assert not res.flags.writeable

    # Verify manual value
    # response[:, 0, :] = [[0,1,2,3], [12,13,14,15]]
    # response[:, 1, :] = [[4,5,6,7], [16,17,18,19]]
    # response[:, 2, :] = [[8,9,10,11], [20,21,22,23]]
    # For res[0, 0]:
    # 0.2 * 0^2 + 0.3 * 4^2 + 0.5 * 8^2 = 0.2*0 + 0.3*16 + 0.5*64 = 4.8 + 32 = 36.8
    np.testing.assert_allclose(res[0, 0], 36.8, rtol=1e-12)


def test_directional_power_average_invalid():
    response = np.arange(24).reshape(2, 3, 4)
    direction_weights = np.array([0.2, 0.3, 0.5])

    # axis out of bounds
    with pytest.raises(ValueError, match="out of bounds"):
        directional_power_average(response, direction_weights, direction_axis=3)

    # weight length mismatch
    bad_weights = np.array([0.1, 0.2])
    with pytest.raises(ValueError, match="Length of direction_weights must match"):
        directional_power_average(response, bad_weights, direction_axis=1)

    # negative weight
    neg_weights = np.array([0.2, -0.3, 0.5])
    with pytest.raises(ValueError, match="must be non-negative"):
        directional_power_average(response, neg_weights, direction_axis=1)

    # non-finite weight
    inf_weights = np.array([0.2, np.inf, 0.5])
    with pytest.raises(ValueError, match="finite numbers"):
        directional_power_average(response, inf_weights, direction_axis=1)

    # bool weight
    bool_weights = np.array([True, False, True])
    with pytest.raises(TypeError, match="must not be bool"):
        directional_power_average(response, bool_weights, direction_axis=1)


# ---------------------------------------------------------------------------
# directional_cross_average (Phase 15-a-1, spec:
# docs/design/phase15a_cross_statistics.md)
# ---------------------------------------------------------------------------


def test_c1_cross_literal_golden():
    """C1: hand-derived literal golden with a negative weight."""
    a = np.array([1 + 1j, 2 + 0j])
    b = np.array([1 - 1j, 1j])
    w = np.array([0.5, -0.25])
    result = directional_cross_average(a, b, w, direction_axis=0)
    np.testing.assert_allclose(result, 0 + 1.5j, rtol=1e-15)


def test_c2_cross_equals_power_identity_w_nonneg():
    """C2: cross(A,A,w) == power(A,w) for w >= 0 (real/imag compared separately)."""
    a = np.array([1 + 1j, 2 + 0j])
    w = np.array([0.5, 0.25])
    cross = directional_cross_average(a, a, w, direction_axis=0)
    power = directional_power_average(a, w, direction_axis=0)
    np.testing.assert_allclose(cross.real, power)
    np.testing.assert_allclose(cross.imag, 0.0)


def test_c3_auto_cross_imag_exactly_zero():
    """C3: Im(cross(A,A,w)) is machine-epsilon scale (NOT exactly 0 -- numpy's
    complex multiply uses a 3-multiply Karatsuba-style formula, so even a
    single A_i*conj(A_i) carries a ~1e-17..1e-20 residual; spec corrected
    2026-07-05). Must hold generically, not only via an object-identity
    special case (e.g. broadcast views of equal-valued arrays)."""
    rng = np.random.default_rng(7)
    a = rng.normal(size=5) + 1j * rng.normal(size=5)
    w = rng.normal(size=5)  # may be negative
    result = directional_cross_average(a, a, w, direction_axis=0)
    np.testing.assert_allclose(result.imag, 0.0, atol=1e-10)

    # Must hold for a value-equal-but-not-identical array too (this is the
    # regression the object-identity hack in an earlier draft missed).
    b = a.copy()
    result2 = directional_cross_average(a, b, w, direction_axis=0)
    np.testing.assert_allclose(result2.imag, 0.0, atol=1e-10)


def test_c4_negative_weight_power_rejects_cross_accepts():
    """C4: asymmetric weight contract -- power requires w>=0, cross allows negative."""
    a = np.array([1 + 1j, 2 + 0j])
    neg_w = np.array([0.5, -0.25])
    with pytest.raises(ValueError, match="non-negative"):
        directional_power_average(a, neg_w, direction_axis=0)
    # Must not raise:
    directional_cross_average(a, a, neg_w, direction_axis=0)


def test_c5_complex_weight_raises_typeerror():
    a = np.array([1 + 1j, 2 + 0j])
    complex_w = np.array([1 + 1j, 2 + 0j])
    with pytest.raises(TypeError):
        directional_cross_average(a, a, complex_w, direction_axis=0)
    with pytest.raises(TypeError):
        directional_power_average(a, complex_w, direction_axis=0)


def test_c6_lebedev_negative_weight_order_passes():
    pytest.importorskip("scipy")
    try:
        from scipy.integrate import lebedev_rule  # noqa: F401
    except (ImportError, AttributeError):
        pytest.skip("scipy.integrate.lebedev_rule is not available")
    from gragra.adapters.lebedev import lebedev_grid

    k_hat, wnorm = lebedev_grid(13)
    assert (wnorm < 0).any()  # order 13 is known to carry negative weights
    a = np.exp(1j * k_hat[:, 0])  # arbitrary complex response per direction
    result = directional_cross_average(a, a, wnorm, direction_axis=0)
    assert np.isfinite(result)


def test_c7_shape_and_axis_valueerror_contract():
    a = np.array([1 + 1j, 2 + 0j, 3 + 0j])
    b = np.array([1 - 1j, 1j])  # mismatched shape
    w = np.array([0.5, 0.25, 0.25])
    with pytest.raises(ValueError):
        directional_cross_average(a, b, w, direction_axis=0)

    a2 = np.array([1 + 1j, 2 + 0j])
    w_bad_len = np.array([0.5, 0.25, 0.1])  # mismatched weight length
    with pytest.raises(ValueError):
        directional_cross_average(a2, a2, w_bad_len, direction_axis=0)


def test_c8_output_contract():
    response = np.arange(24, dtype=np.complex128).reshape(2, 3, 4)
    weights = np.array([0.2, 0.3, 0.5])
    result = directional_cross_average(response, response, weights, direction_axis=1)
    assert result.shape == (2, 4)
    assert result.dtype == np.complex128
    assert not result.flags.writeable


def test_c9_csd_phase_sign_fixed_point():
    """C9: arg(cross(a(r1), a(r2))) == k*(x1 - x2) for a plane wave a=exp(ikx)."""
    k = 2.0
    x1, x2 = 0.3, 0.1
    a1 = np.array([np.exp(1j * k * x1)])
    a2 = np.array([np.exp(1j * k * x2)])
    w = np.array([1.0])
    cross_12 = directional_cross_average(a1, a2, w, direction_axis=0)
    np.testing.assert_allclose(np.angle(cross_12), k * (x1 - x2), rtol=1e-12)
    cross_21 = directional_cross_average(a2, a1, w, direction_axis=0)
    np.testing.assert_allclose(np.angle(cross_21), -k * (x1 - x2), rtol=1e-12)


def test_c10_integer_response_does_not_overflow():
    """Regression (Codex review #41): int64 response values must be
    promoted to a floating dtype before the cross/power multiplication, or
    large values (~1e10) silently wrap instead of producing the correct
    (much larger) product."""
    a = np.array([10**10], dtype=np.int64)
    b = np.array([10**10], dtype=np.int64)
    w = np.array([1.0])

    cross = directional_cross_average(a, b, w, direction_axis=0)
    np.testing.assert_allclose(cross, 1e20 + 0j, rtol=1e-12)

    power = directional_power_average(a, w, direction_axis=0)
    np.testing.assert_allclose(power, 1e20, rtol=1e-12)
