"""Numba-accelerated parametric inverse-square gravity kernels.

Computes potential, acceleration, and gravity gradient contractions with
multi-dimensional parametric weights of shape ``(N, *P)``. All kernel
contractions are G-free (do not multiply by ``G_SI``).

The weight tensor ``(N, *P)`` is flattened to ``(N, P_flat)`` in the Python
wrappers and the compiled kernels operate on that fixed 2-D interface; the
geometric factor for each (target, source) pair is computed once and reused
across the ``P_flat`` axis, so adding parametric dimensions does not increase
the geometry work. Outputs are reshaped back to ``(M, *P, ...)``.

Like the point-direct compiled backends, these kernels compute over all
sources at once and ignore ``chunk_size`` (still validated by the shared
``_validate_inputs_parametric``); results are chunk-size independent.
"""

import numba
import numpy as np

from gragra.kernels._collision import _check_collisions
from gragra.kernels.parametric import _flatten_weights, _validate_inputs_parametric


@numba.njit(parallel=True, cache=True)
def _potential_parametric_real_jit(
    sources_xyz: np.ndarray,
    weights: np.ndarray,
    targets_xyz: np.ndarray,
    eps2: float,
    collision_source_index: np.ndarray,
) -> np.ndarray:
    m = len(targets_xyz)
    n = len(sources_xyz)
    p = weights.shape[1]
    out = np.zeros((m, p), dtype=np.float64)
    for i in numba.prange(m):
        tx = targets_xyz[i, 0]
        ty = targets_xyz[i, 1]
        tz = targets_xyz[i, 2]
        for j in range(n):
            dx = sources_xyz[j, 0] - tx
            dy = sources_xyz[j, 1] - ty
            dz = sources_xyz[j, 2] - tz
            r2 = dx * dx + dy * dy + dz * dz
            if eps2 == 0.0 and r2 == 0.0:
                if collision_source_index[i] == -1:
                    collision_source_index[i] = j
                continue
            r_eff = np.sqrt(r2 + eps2)
            k = -1.0 / r_eff
            for q in range(p):
                out[i, q] += k * weights[j, q]
    return out


@numba.njit(parallel=True, cache=True)
def _potential_parametric_complex_jit(
    sources_xyz: np.ndarray,
    weights: np.ndarray,
    targets_xyz: np.ndarray,
    eps2: float,
    collision_source_index: np.ndarray,
) -> np.ndarray:
    m = len(targets_xyz)
    n = len(sources_xyz)
    p = weights.shape[1]
    out = np.zeros((m, p), dtype=np.complex128)
    for i in numba.prange(m):
        tx = targets_xyz[i, 0]
        ty = targets_xyz[i, 1]
        tz = targets_xyz[i, 2]
        for j in range(n):
            dx = sources_xyz[j, 0] - tx
            dy = sources_xyz[j, 1] - ty
            dz = sources_xyz[j, 2] - tz
            r2 = dx * dx + dy * dy + dz * dz
            if eps2 == 0.0 and r2 == 0.0:
                if collision_source_index[i] == -1:
                    collision_source_index[i] = j
                continue
            r_eff = np.sqrt(r2 + eps2)
            k = -1.0 / r_eff
            for q in range(p):
                out[i, q] += k * weights[j, q]
    return out


@numba.njit(parallel=True, cache=True)
def _acceleration_parametric_real_jit(
    sources_xyz: np.ndarray,
    weights: np.ndarray,
    targets_xyz: np.ndarray,
    eps2: float,
    collision_source_index: np.ndarray,
) -> np.ndarray:
    m = len(targets_xyz)
    n = len(sources_xyz)
    p = weights.shape[1]
    out = np.zeros((m, p, 3), dtype=np.float64)
    for i in numba.prange(m):
        tx = targets_xyz[i, 0]
        ty = targets_xyz[i, 1]
        tz = targets_xyz[i, 2]
        for j in range(n):
            dx = sources_xyz[j, 0] - tx
            dy = sources_xyz[j, 1] - ty
            dz = sources_xyz[j, 2] - tz
            r2 = dx * dx + dy * dy + dz * dz
            if eps2 == 0.0 and r2 == 0.0:
                if collision_source_index[i] == -1:
                    collision_source_index[i] = j
                continue
            r_eff = np.sqrt(r2 + eps2)
            r_eff3 = (r2 + eps2) * r_eff
            kx = dx / r_eff3
            ky = dy / r_eff3
            kz = dz / r_eff3
            for q in range(p):
                wq = weights[j, q]
                out[i, q, 0] += kx * wq
                out[i, q, 1] += ky * wq
                out[i, q, 2] += kz * wq
    return out


@numba.njit(parallel=True, cache=True)
def _acceleration_parametric_complex_jit(
    sources_xyz: np.ndarray,
    weights: np.ndarray,
    targets_xyz: np.ndarray,
    eps2: float,
    collision_source_index: np.ndarray,
) -> np.ndarray:
    m = len(targets_xyz)
    n = len(sources_xyz)
    p = weights.shape[1]
    out = np.zeros((m, p, 3), dtype=np.complex128)
    for i in numba.prange(m):
        tx = targets_xyz[i, 0]
        ty = targets_xyz[i, 1]
        tz = targets_xyz[i, 2]
        for j in range(n):
            dx = sources_xyz[j, 0] - tx
            dy = sources_xyz[j, 1] - ty
            dz = sources_xyz[j, 2] - tz
            r2 = dx * dx + dy * dy + dz * dz
            if eps2 == 0.0 and r2 == 0.0:
                if collision_source_index[i] == -1:
                    collision_source_index[i] = j
                continue
            r_eff = np.sqrt(r2 + eps2)
            r_eff3 = (r2 + eps2) * r_eff
            kx = dx / r_eff3
            ky = dy / r_eff3
            kz = dz / r_eff3
            for q in range(p):
                wq = weights[j, q]
                out[i, q, 0] += kx * wq
                out[i, q, 1] += ky * wq
                out[i, q, 2] += kz * wq
    return out


@numba.njit(parallel=True, cache=True)
def _gradient_parametric_real_jit(
    sources_xyz: np.ndarray,
    weights: np.ndarray,
    targets_xyz: np.ndarray,
    eps2: float,
    collision_source_index: np.ndarray,
) -> np.ndarray:
    m = len(targets_xyz)
    n = len(sources_xyz)
    p = weights.shape[1]
    out = np.zeros((m, p, 3, 3), dtype=np.float64)
    for i in numba.prange(m):
        tx = targets_xyz[i, 0]
        ty = targets_xyz[i, 1]
        tz = targets_xyz[i, 2]
        for j in range(n):
            dx = sources_xyz[j, 0] - tx
            dy = sources_xyz[j, 1] - ty
            dz = sources_xyz[j, 2] - tz
            r2 = dx * dx + dy * dy + dz * dz
            if eps2 == 0.0 and r2 == 0.0:
                if collision_source_index[i] == -1:
                    collision_source_index[i] = j
                continue
            r_eff = np.sqrt(r2 + eps2)
            r_eff3 = (r2 + eps2) * r_eff
            r_eff5 = r_eff3 * (r2 + eps2)
            inv_r3 = 1.0 / r_eff3
            inv_r5 = 3.0 / r_eff5
            t00 = dx * dx * inv_r5 - inv_r3
            t01 = dx * dy * inv_r5
            t02 = dx * dz * inv_r5
            t11 = dy * dy * inv_r5 - inv_r3
            t12 = dy * dz * inv_r5
            t22 = dz * dz * inv_r5 - inv_r3
            for q in range(p):
                wq = weights[j, q]
                out[i, q, 0, 0] += t00 * wq
                out[i, q, 0, 1] += t01 * wq
                out[i, q, 0, 2] += t02 * wq
                out[i, q, 1, 0] += t01 * wq
                out[i, q, 1, 1] += t11 * wq
                out[i, q, 1, 2] += t12 * wq
                out[i, q, 2, 0] += t02 * wq
                out[i, q, 2, 1] += t12 * wq
                out[i, q, 2, 2] += t22 * wq
    return out


@numba.njit(parallel=True, cache=True)
def _gradient_parametric_complex_jit(
    sources_xyz: np.ndarray,
    weights: np.ndarray,
    targets_xyz: np.ndarray,
    eps2: float,
    collision_source_index: np.ndarray,
) -> np.ndarray:
    m = len(targets_xyz)
    n = len(sources_xyz)
    p = weights.shape[1]
    out = np.zeros((m, p, 3, 3), dtype=np.complex128)
    for i in numba.prange(m):
        tx = targets_xyz[i, 0]
        ty = targets_xyz[i, 1]
        tz = targets_xyz[i, 2]
        for j in range(n):
            dx = sources_xyz[j, 0] - tx
            dy = sources_xyz[j, 1] - ty
            dz = sources_xyz[j, 2] - tz
            r2 = dx * dx + dy * dy + dz * dz
            if eps2 == 0.0 and r2 == 0.0:
                if collision_source_index[i] == -1:
                    collision_source_index[i] = j
                continue
            r_eff = np.sqrt(r2 + eps2)
            r_eff3 = (r2 + eps2) * r_eff
            r_eff5 = r_eff3 * (r2 + eps2)
            inv_r3 = 1.0 / r_eff3
            inv_r5 = 3.0 / r_eff5
            t00 = dx * dx * inv_r5 - inv_r3
            t01 = dx * dy * inv_r5
            t02 = dx * dz * inv_r5
            t11 = dy * dy * inv_r5 - inv_r3
            t12 = dy * dz * inv_r5
            t22 = dz * dz * inv_r5 - inv_r3
            for q in range(p):
                wq = weights[j, q]
                out[i, q, 0, 0] += t00 * wq
                out[i, q, 0, 1] += t01 * wq
                out[i, q, 0, 2] += t02 * wq
                out[i, q, 1, 0] += t01 * wq
                out[i, q, 1, 1] += t11 * wq
                out[i, q, 1, 2] += t12 * wq
                out[i, q, 2, 0] += t02 * wq
                out[i, q, 2, 1] += t12 * wq
                out[i, q, 2, 2] += t22 * wq
    return out


def potential_contract_parametric(
    sources_xyz: list | tuple | np.ndarray,
    weights: list | tuple | np.ndarray,
    targets_xyz: list | tuple | np.ndarray,
    *,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
) -> np.ndarray:
    """Numba parametric potential contraction. K_Φ = -1 / sqrt(r^2 + ε^2)."""
    sources, w, targets, eps, _ = _validate_inputs_parametric(
        sources_xyz, weights, targets_xyz, softening_m, chunk_size
    )
    n = len(sources)
    m = len(targets)
    w2d, p_shape = _flatten_weights(w, n)
    collision_source_index = np.full(m, -1, dtype=np.int64)
    eps2 = eps**2

    if np.iscomplexobj(w):
        out2d = _potential_parametric_complex_jit(
            sources, w2d, targets, eps2, collision_source_index
        )
    else:
        out2d = _potential_parametric_real_jit(
            sources, w2d, targets, eps2, collision_source_index
        )

    _check_collisions(collision_source_index, eps)
    out = out2d.reshape((m,) + p_shape)
    out.setflags(write=False)
    return out


def acceleration_contract_parametric(
    sources_xyz: list | tuple | np.ndarray,
    weights: list | tuple | np.ndarray,
    targets_xyz: list | tuple | np.ndarray,
    *,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
) -> np.ndarray:
    """Numba parametric acceleration contraction. K_a = Δ / (r^2 + ε^2)^(3/2)."""
    sources, w, targets, eps, _ = _validate_inputs_parametric(
        sources_xyz, weights, targets_xyz, softening_m, chunk_size
    )
    n = len(sources)
    m = len(targets)
    w2d, p_shape = _flatten_weights(w, n)
    collision_source_index = np.full(m, -1, dtype=np.int64)
    eps2 = eps**2

    if np.iscomplexobj(w):
        out3d = _acceleration_parametric_complex_jit(
            sources, w2d, targets, eps2, collision_source_index
        )
    else:
        out3d = _acceleration_parametric_real_jit(
            sources, w2d, targets, eps2, collision_source_index
        )

    _check_collisions(collision_source_index, eps)
    out = out3d.reshape((m,) + p_shape + (3,))
    out.setflags(write=False)
    return out


def gradient_contract_parametric(
    sources_xyz: list | tuple | np.ndarray,
    weights: list | tuple | np.ndarray,
    targets_xyz: list | tuple | np.ndarray,
    *,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
) -> np.ndarray:
    """Numba parametric gravity-gradient contraction.

    K_T = 3 ΔΔ / (r^2 + ε^2)^(5/2) - I / (r^2 + ε^2)^(3/2).

    Note that for softening_m > 0, the softened tensor does not satisfy tr(T) = 0.
    """
    sources, w, targets, eps, _ = _validate_inputs_parametric(
        sources_xyz, weights, targets_xyz, softening_m, chunk_size
    )
    n = len(sources)
    m = len(targets)
    w2d, p_shape = _flatten_weights(w, n)
    collision_source_index = np.full(m, -1, dtype=np.int64)
    eps2 = eps**2

    if np.iscomplexobj(w):
        out4d = _gradient_parametric_complex_jit(
            sources, w2d, targets, eps2, collision_source_index
        )
    else:
        out4d = _gradient_parametric_real_jit(
            sources, w2d, targets, eps2, collision_source_index
        )

    _check_collisions(collision_source_index, eps)
    out = out4d.reshape((m,) + p_shape + (3, 3))
    out.setflags(write=False)
    return out
