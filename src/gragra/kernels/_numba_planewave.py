"""Numba implementation of plane-wave weight factor."""

import numba
import numpy as np


@numba.njit(parallel=True, cache=True)
def _plane_wave_weight_factor_numba(
    w: np.ndarray,
    pos: np.ndarray,
    k_hat: np.ndarray,
    k: np.ndarray,
    phase0: float,
    sign: int,
) -> np.ndarray:
    n_sources = pos.shape[0]
    ndir = k_hat.shape[0]
    nf = k.shape[0]

    out = np.zeros((n_sources, ndir, nf), dtype=np.complex128)

    for n in numba.prange(n_sources):
        w_n = w[n]
        pos_x = pos[n, 0]
        pos_y = pos[n, 1]
        pos_z = pos[n, 2]

        for j in range(ndir):
            kh_x = k_hat[j, 0]
            kh_y = k_hat[j, 1]
            kh_z = k_hat[j, 2]

            dot_val = pos_x * kh_x + pos_y * kh_y + pos_z * kh_z

            for f in range(nf):
                phase = sign * (dot_val * k[f] + phase0)
                out[n, j, f] = w_n * (np.cos(phase) + 1j * np.sin(phase))

    return out


def plane_wave_weight_factor(
    w: np.ndarray,
    pos: np.ndarray,
    k_hat: np.ndarray,
    k: np.ndarray,
    phase0: float,
    sign: int,
) -> np.ndarray:
    """Numba-accelerated plane-wave weights contraction.

    Note: Inputs are assumed to be validated by the public interface.
    """
    res = _plane_wave_weight_factor_numba(w, pos, k_hat, k, phase0, sign)
    res = np.ascontiguousarray(res)
    res.setflags(write=False)
    return res
