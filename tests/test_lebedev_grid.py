import numpy as np
import pytest

# Module-level skip if scipy is not available or doesn't have lebedev_rule
pytest.importorskip("scipy")
try:
    from scipy.integrate import lebedev_rule  # noqa: F401
except (ImportError, AttributeError):
    pytest.skip(
        "scipy.integrate.lebedev_rule is not available", allow_module_level=True
    )

from gragra.adapters.lebedev import lebedev_grid


def test_lebedev_grid_valid_orders():
    # Test valid representative orders and point counts
    # Supported orders: 9 -> 38 points, 13 -> 74 points, 15 -> 86 points
    test_cases = [
        (9, 38),
        (13, 74),
        (15, 86),
    ]

    for order, expected_points in test_cases:
        khat, wnorm = lebedev_grid(order)

        # Check shapes
        assert khat.shape == (expected_points, 3)
        assert wnorm.shape == (expected_points,)

        # Check float64
        assert khat.dtype == np.float64
        assert wnorm.dtype == np.float64

        # Check directions normalization
        norms = np.linalg.norm(khat, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-12, atol=1e-12)

        # Check weights sum to 1.0
        np.testing.assert_allclose(np.sum(wnorm), 1.0, rtol=1e-12, atol=1e-12)

        # Check finite
        assert np.all(np.isfinite(khat))
        assert np.all(np.isfinite(wnorm))

        # Check read-only
        assert khat.flags.writeable is False
        assert wnorm.flags.writeable is False

        if order == 13:
            assert np.any(wnorm < 0), "order=13 must contain negative weights"


def test_lebedev_grid_invalid_types_and_bounds():
    # order must be int, not bool
    with pytest.raises(ValueError, match="order must be a positive integer, got bool"):
        lebedev_grid(True)

    # order must be int, not float
    with pytest.raises(
        ValueError, match="order must be a positive integer, got type float"
    ):
        lebedev_grid(9.5)

    # order must be positive
    with pytest.raises(ValueError, match="order must be positive, got 0"):
        lebedev_grid(0)
    with pytest.raises(ValueError, match="order must be positive, got -5"):
        lebedev_grid(-5)


def test_lebedev_grid_unsupported_order():
    # Unsupported positive integers (e.g. 4, 33)
    # scipy.integrate.lebedev_rule will raise ValueError or NotImplementedError,
    # which we wrap in a ValueError providing a list of valid orders.
    for order in [4, 33]:
        with pytest.raises(ValueError) as excinfo:
            lebedev_grid(order)
        assert "Valid orders include" in str(excinfo.value)
        assert str(order) in str(excinfo.value)
