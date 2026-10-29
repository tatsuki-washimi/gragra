"""M3-B Step 10: L1-b fixed orthogonal Q rotation–DFT exchange (scale-normalised)."""

from __future__ import annotations

import numpy as np
import pytest

# P2 proposal (06-tolerance-budget L1-b): |Δ|∞ <= 16·eps·max|S_ref|.  Rounding
# error scales with the spectrum maximum, so a fixed absolute tolerance is
# neither scale-invariant nor justified (not G-最終).
_EPS = np.finfo(np.float64).eps
_C = 16.0


def assert_l1b_exchange(actual: np.ndarray, reference: np.ndarray) -> None:
    """``|Δ|∞ ≤ 16·ε·max|S_ref|`` (06 L1-b). Zero reference ⇒ exact match."""
    delta_max = float(np.max(np.abs(actual - reference)))
    ref_scale = float(np.max(np.abs(reference)))
    if ref_scale == 0.0:
        assert delta_max == 0.0
        return
    assert delta_max <= _C * _EPS * ref_scale


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _flatten_gamma(g: np.ndarray) -> np.ndarray:
    return g.reshape(9)


def _unflatten_gamma(v: np.ndarray) -> np.ndarray:
    return v.reshape(3, 3)


def _dft_rfft(y: np.ndarray, dt: float, t0: float, tref: float) -> np.ndarray:
    """C3 Fourier mode (Q·s), aligned with step3-num rev6 L1-b script."""
    m = y.shape[0]
    x = np.fft.rfft(y)
    df = 1.0 / (m * dt)
    out = []
    for k, xk in enumerate(x):
        fk = k * df
        out.append(dt * np.exp(-2j * np.pi * fk * (t0 - tref)) * xk)
    return np.array(out)


def _rot_dft_max_abs(g_t: np.ndarray, r: np.ndarray, *, dt: float) -> float:
    g_p = np.einsum("ij,tjk,kl->til", r, g_t, r.T)
    t0 = 0.0
    tref = 0.0
    spec_a = np.stack(
        [_dft_rfft(g_p[:, i, j], dt, t0, tref) for i in range(3) for j in range(3)],
        axis=0,
    )
    spec_b_flat = np.stack(
        [_dft_rfft(g_t[:, i, j], dt, t0, tref) for i in range(3) for j in range(3)],
        axis=0,
    )
    spec_b_rot = np.zeros_like(spec_b_flat)
    for b in range(spec_b_flat.shape[1]):
        spec_b_rot[:, b] = _flatten_gamma(
            r @ _unflatten_gamma(spec_b_flat[:, b]) @ r.T
        )
    max_delta = float(np.max(np.abs(spec_a - spec_b_rot)))
    max_ref = float(np.max(np.abs(spec_b_rot)))
    if max_ref == 0.0:
        return 0.0 if max_delta == 0.0 else float("inf")
    return max_delta / max_ref


@pytest.mark.parametrize("trial", range(5))
def test_rot_dft_time_varying_symmetric_gradient(trial):
    rng = np.random.default_rng(42 + trial)
    r = _random_rotation(rng)
    t_len, dt = 128, 1.0 / 256.0
    t = np.arange(t_len) * dt
    g_t = np.zeros((t_len, 3, 3))
    for ti in range(t_len):
        base = np.outer(
            [
                np.sin(2 * np.pi * 10 * t[ti]),
                np.cos(2 * np.pi * 7 * t[ti]),
                np.sin(2 * np.pi * 3 * t[ti]),
            ],
            [1.0, 0.5, -0.3],
        )
        g_t[ti] = 0.5 * (base + base.T)
    max_abs = _rot_dft_max_abs(g_t, r, dt=dt)
    assert max_abs <= _C * _EPS


def test_rot_dft_constant_gradient_dc_only():
    rng = np.random.default_rng(7)
    r = _random_rotation(rng)
    t_len = 64
    dt = 1.0
    g0 = np.array([[1.2, 0.3, -0.1], [0.3, -0.7, 0.2], [-0.1, 0.2, 0.5]])
    g0 = 0.5 * (g0 + g0.T)
    g_t = np.broadcast_to(g0, (t_len, 3, 3)).copy()
    max_abs = _rot_dft_max_abs(g_t, r, dt=dt)
    assert max_abs <= _C * _EPS


def test_rot_dft_vector_three_components():
    rng = np.random.default_rng(9)
    r = _random_rotation(rng)
    t_len, dt = 64, 1.0 / 256.0
    v0 = np.array([0.4, -0.2, 0.9])
    v_raw = np.stack([v0 * (1.0 + 0.01 * ti) for ti in range(t_len)], axis=0)
    v_rot = (r @ v_raw.T).T
    t0 = tref = 0.0
    spec_a = np.stack([_dft_rfft(v_rot[:, i], dt, t0, tref) for i in range(3)], axis=0)
    spec_b = np.stack([_dft_rfft(v_raw[:, i], dt, t0, tref) for i in range(3)], axis=0)
    spec_b_rot = r @ spec_b
    assert_l1b_exchange(spec_a, spec_b_rot)


def test_rot_dft_zero_reference_requires_exact_match():
    """All-zero spectrum: 06 L1-b requires strict equality, not 0/0 normalisation."""
    rng = np.random.default_rng(3)
    r = _random_rotation(rng)
    t_len, dt = 32, 1.0 / 64.0
    g_t = np.zeros((t_len, 3, 3))
    max_norm = _rot_dft_max_abs(g_t, r, dt=dt)
    assert max_norm == 0.0
