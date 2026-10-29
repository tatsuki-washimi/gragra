"""Tests for coherent gravitational observables."""

import numpy as np
import pytest

from gragra.constants import G_SI
from gragra.kernels.parametric import (
    acceleration_contract_parametric,
    gradient_contract_parametric,
)
from gragra.observables.coherent import (
    coherent_point_acceleration,
    coherent_point_gravity_gradient,
    coherent_projected_acceleration,
)


def test_coherent_point_acceleration_basics():
    sources = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    # parametric weights: shape (N, *P) = (2, 3)
    weights = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    targets = np.array([[0.5, 0.5, 0.0], [1.0, 1.0, 1.0]])

    acc = coherent_point_acceleration(sources, weights, targets, softening_m=0.1)

    # Expected shape: (M, *P, 3) = (2, 3, 3)
    assert acc.shape == (2, 3, 3)
    assert acc.dtype == np.float64
    assert not acc.flags.writeable

    # Verify that it matches G_SI * acceleration_contract_parametric
    expected = G_SI * acceleration_contract_parametric(
        sources, weights, targets, softening_m=0.1
    )
    np.testing.assert_allclose(acc, expected, rtol=1e-12)


def test_coherent_projected_acceleration():
    sources = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    weights = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    targets = np.array([[0.5, 0.5, 0.0], [1.0, 1.0, 1.0]])
    direction = np.array([3.0, 4.0, 0.0])  # norm = 5.0 -> unit: [0.6, 0.8, 0.0]

    proj_acc = coherent_projected_acceleration(
        sources, weights, targets, direction, softening_m=0.1
    )

    # Expected shape: (M, *P) = (2, 3)
    assert proj_acc.shape == (2, 3)
    assert proj_acc.dtype == np.float64
    assert not proj_acc.flags.writeable

    # Verify manual calculation against full 3D acceleration
    acc = coherent_point_acceleration(sources, weights, targets, softening_m=0.1)
    unit_dir = np.array([0.6, 0.8, 0.0])
    expected = np.sum(acc * unit_dir, axis=-1)
    np.testing.assert_allclose(proj_acc, expected, rtol=1e-12)


def test_coherent_observables_invalid():
    sources = np.array([[0.0, 0.0, 0.0]])
    weights = np.ones((1, 2))
    targets = np.array([[1.0, 1.0, 1.0]])

    # Zero direction vector
    with pytest.raises(ValueError, match="direction vector cannot be a zero vector"):
        coherent_projected_acceleration(sources, weights, targets, [0.0, 0.0, 0.0])


def _available_backends():
    backends = ["numpy"]
    try:
        import numba  # noqa: F401

        backends.append("numba")
    except ImportError:
        pass
    try:
        import gragra._cpp_backend  # noqa: F401

        backends.append("cpp")
    except ImportError:
        pass
    return backends


@pytest.mark.parametrize("backend", _available_backends())
@pytest.mark.parametrize("complex_weights", [False, True])
def test_coherent_backend_matches_numpy(backend, complex_weights):
    """All backends agree with the numpy reference for both point and
    projected coherent acceleration (the G-free factor is backend-identical).
    """
    rng = np.random.default_rng(0)
    sources = rng.random((6, 3)) * 10.0
    targets = rng.random((4, 3)) * 10.0 + 50.0
    weights = rng.random((6, 3, 2))
    if complex_weights:
        weights = weights + 1j * rng.random((6, 3, 2))
    direction = [3.0, 4.0, 0.0]

    expected = coherent_point_acceleration(
        sources, weights, targets, softening_m=0.1, backend="numpy"
    )
    actual = coherent_point_acceleration(
        sources, weights, targets, softening_m=0.1, backend=backend
    )
    assert actual.dtype == expected.dtype
    assert not actual.flags.writeable
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-15)

    pe = coherent_projected_acceleration(
        sources, weights, targets, direction, softening_m=0.1, backend="numpy"
    )
    pa = coherent_projected_acceleration(
        sources, weights, targets, direction, softening_m=0.1, backend=backend
    )
    np.testing.assert_allclose(pa, pe, rtol=1e-12, atol=1e-15)


def test_coherent_unknown_backend_raises():
    sources = np.array([[0.0, 0.0, 0.0]])
    weights = np.ones((1, 2))
    targets = np.array([[1.0, 1.0, 1.0]])
    with pytest.raises(ValueError, match="Unsupported backend"):
        coherent_point_acceleration(sources, weights, targets, backend="bogus")


# ---------------------------------------------------------------------------
# coherent_point_gravity_gradient（公開配線 — roadmap「gradient 配線」単独小 PR）
# ---------------------------------------------------------------------------


def test_coherent_point_gravity_gradient_basics():
    sources = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    weights = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])  # (N, *P)
    targets = np.array([[0.5, 0.5, 0.0], [1.0, 1.0, 1.0]])

    grad = coherent_point_gravity_gradient(sources, weights, targets, softening_m=0.1)

    # 軸順規約: component/tensor 軸は末尾 — (M, *P, 3, 3)
    assert grad.shape == (2, 3, 3, 3)
    assert grad.dtype == np.float64
    assert not grad.flags.writeable

    expected = G_SI * gradient_contract_parametric(
        sources, weights, targets, softening_m=0.1
    )
    np.testing.assert_allclose(grad, expected, rtol=1e-12)


def test_coherent_gradient_tensor_sign_fixed():
    # 源@原点・標的 (1,0,0)・単位重み（P 軸 1 点）・softening=0:
    # T == G_SI · [[2,0,0],[0,-1,0],[0,0,-1]]（kernel 符号規約の固定テスト）
    sources = np.array([[0.0, 0.0, 0.0]])
    weights = np.array([[1.0]])  # (N=1, P=1)
    targets = np.array([[1.0, 0.0, 0.0]])

    grad = coherent_point_gravity_gradient(sources, weights, targets)

    expected = G_SI * np.array([[2.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    np.testing.assert_allclose(grad[0, 0], expected, rtol=1e-12, atol=0.0)
    # tr(T) = 0 は softening=0 のみ成立
    np.testing.assert_allclose(np.trace(grad[0, 0]), 0.0, atol=1e-25)


def test_coherent_gradient_complex_weights_dtype():
    sources = np.array([[0.0, 0.0, 0.0]])
    weights = np.array([[1.0 + 2.0j, 3.0 - 1.0j]])  # (1, 2) complex
    targets = np.array([[1.0, 0.0, 0.0]])

    grad = coherent_point_gravity_gradient(sources, weights, targets)

    assert grad.shape == (1, 2, 3, 3)
    assert grad.dtype == np.complex128
    # 線形性: 複素重みは実部・虚部の重ね合わせ
    grad_re = coherent_point_gravity_gradient(sources, weights.real, targets)
    grad_im = coherent_point_gravity_gradient(sources, weights.imag, targets)
    np.testing.assert_allclose(grad, grad_re + 1j * grad_im, rtol=1e-12)


def test_coherent_gradient_chunk_invariance():
    rng = np.random.default_rng(42)
    sources = rng.uniform(-1.0, 1.0, (7, 3))
    weights = rng.uniform(0.5, 2.0, (7, 2))
    targets = np.array([[3.0, 0.5, -0.2]])

    full = coherent_point_gravity_gradient(sources, weights, targets)
    chunked = coherent_point_gravity_gradient(sources, weights, targets, chunk_size=3)
    np.testing.assert_allclose(chunked, full, rtol=1e-12)

    with pytest.raises(ValueError):
        coherent_point_gravity_gradient(sources, weights, targets, chunk_size=0)


def test_coherent_gradient_cupy_no_device_raises_import_error(monkeypatch):
    """coherent_point_gravity_gradient backend='cupy' without cupy raises ImportError.

    Replaces the old test_coherent_gradient_cupy_rejected which expected a
    ValueError('point-direct only').  That guard was removed in Phase C-2b
    when gradient_contract_parametric was added to _cupy_parametric.
    """
    import sys

    monkeypatch.setitem(sys.modules, "cupy", None)
    monkeypatch.delitem(
        sys.modules, "gragra.kernels._cupy_inverse_square", raising=False
    )
    monkeypatch.delitem(sys.modules, "gragra.kernels._cupy_parametric", raising=False)
    sources = np.array([[0.0, 0.0, 0.0]])
    weights = np.array([[1.0]])
    targets = np.array([[1.0, 0.0, 0.0]])
    with pytest.raises(ImportError, match="cupy is not installed"):
        coherent_point_gravity_gradient(sources, weights, targets, backend="cupy")
