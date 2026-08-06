"""Numba-accelerated gravity kernels.

Computes potential, acceleration, and gravity gradient contractions.
All kernel contractions are G-free (do not multiply by G_SI).
"""

import numba
import numpy as np

from gragra.array_types import ComplexArray, FloatArray
from gragra.kernels._collision import _check_collisions
from gragra.kernels.inverse_square import _validate_dipole_inputs, _validate_inputs


@numba.njit(parallel=True, cache=True)
def potential_contract_real_jit(
    sources_xyz: np.ndarray,
    weights: np.ndarray,
    targets_xyz: np.ndarray,
    eps2: float,
    collision_source_index: np.ndarray,
) -> np.ndarray:
    m = len(targets_xyz)
    n = len(sources_xyz)
    out = np.zeros(m, dtype=np.float64)
    for i in numba.prange(m):
        tx = targets_xyz[i, 0]
        ty = targets_xyz[i, 1]
        tz = targets_xyz[i, 2]
        s = 0.0
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
            s += (-1.0 / r_eff) * weights[j]
        out[i] = s
    return out


@numba.njit(parallel=True, cache=True)
def potential_contract_complex_jit(
    sources_xyz: np.ndarray,
    weights: np.ndarray,
    targets_xyz: np.ndarray,
    eps2: float,
    collision_source_index: np.ndarray,
) -> np.ndarray:
    m = len(targets_xyz)
    n = len(sources_xyz)
    out = np.zeros(m, dtype=np.complex128)
    for i in numba.prange(m):
        tx = targets_xyz[i, 0]
        ty = targets_xyz[i, 1]
        tz = targets_xyz[i, 2]
        s = 0.0 + 0.0j
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
            s += (-1.0 / r_eff) * weights[j]
        out[i] = s
    return out


@numba.njit(parallel=True, cache=True)
def acceleration_contract_real_jit(
    sources_xyz: np.ndarray,
    weights: np.ndarray,
    targets_xyz: np.ndarray,
    eps2: float,
    collision_source_index: np.ndarray,
) -> np.ndarray:
    m = len(targets_xyz)
    n = len(sources_xyz)
    out = np.zeros((m, 3), dtype=np.float64)
    for i in numba.prange(m):
        tx = targets_xyz[i, 0]
        ty = targets_xyz[i, 1]
        tz = targets_xyz[i, 2]
        ax = 0.0
        ay = 0.0
        az = 0.0
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
            factor = weights[j] / r_eff3
            ax += dx * factor
            ay += dy * factor
            az += dz * factor
        out[i, 0] = ax
        out[i, 1] = ay
        out[i, 2] = az
    return out


@numba.njit(parallel=True, cache=True)
def acceleration_contract_complex_jit(
    sources_xyz: np.ndarray,
    weights: np.ndarray,
    targets_xyz: np.ndarray,
    eps2: float,
    collision_source_index: np.ndarray,
) -> np.ndarray:
    m = len(targets_xyz)
    n = len(sources_xyz)
    out = np.zeros((m, 3), dtype=np.complex128)
    for i in numba.prange(m):
        tx = targets_xyz[i, 0]
        ty = targets_xyz[i, 1]
        tz = targets_xyz[i, 2]
        ax = 0.0 + 0.0j
        ay = 0.0 + 0.0j
        az = 0.0 + 0.0j
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
            factor = weights[j] / r_eff3
            ax += dx * factor
            ay += dy * factor
            az += dz * factor
        out[i, 0] = ax
        out[i, 1] = ay
        out[i, 2] = az
    return out


@numba.njit(parallel=True, cache=True)
def gradient_contract_real_jit(
    sources_xyz: np.ndarray,
    weights: np.ndarray,
    targets_xyz: np.ndarray,
    eps2: float,
    collision_source_index: np.ndarray,
) -> np.ndarray:
    m = len(targets_xyz)
    n = len(sources_xyz)
    out = np.zeros((m, 3, 3), dtype=np.float64)
    for i in numba.prange(m):
        tx = targets_xyz[i, 0]
        ty = targets_xyz[i, 1]
        tz = targets_xyz[i, 2]

        g00 = 0.0
        g01 = 0.0
        g02 = 0.0
        g10 = 0.0
        g11 = 0.0
        g12 = 0.0
        g20 = 0.0
        g21 = 0.0
        g22 = 0.0

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

            w = weights[j]
            inv_r3_w = (1.0 / r_eff3) * w
            inv_r5_w = (3.0 / r_eff5) * w

            g00 += dx * dx * inv_r5_w - inv_r3_w
            g01 += dx * dy * inv_r5_w
            g02 += dx * dz * inv_r5_w

            g10 += dy * dx * inv_r5_w
            g11 += dy * dy * inv_r5_w - inv_r3_w
            g12 += dy * dz * inv_r5_w

            g20 += dz * dx * inv_r5_w
            g21 += dz * dy * inv_r5_w
            g22 += dz * dz * inv_r5_w - inv_r3_w

        out[i, 0, 0] = g00
        out[i, 0, 1] = g01
        out[i, 0, 2] = g02
        out[i, 1, 0] = g10
        out[i, 1, 1] = g11
        out[i, 1, 2] = g12
        out[i, 2, 0] = g20
        out[i, 2, 1] = g21
        out[i, 2, 2] = g22
    return out


@numba.njit(parallel=True, cache=True)
def gradient_contract_complex_jit(
    sources_xyz: np.ndarray,
    weights: np.ndarray,
    targets_xyz: np.ndarray,
    eps2: float,
    collision_source_index: np.ndarray,
) -> np.ndarray:
    m = len(targets_xyz)
    n = len(sources_xyz)
    out = np.zeros((m, 3, 3), dtype=np.complex128)
    for i in numba.prange(m):
        tx = targets_xyz[i, 0]
        ty = targets_xyz[i, 1]
        tz = targets_xyz[i, 2]

        g00 = 0.0 + 0.0j
        g01 = 0.0 + 0.0j
        g02 = 0.0 + 0.0j
        g10 = 0.0 + 0.0j
        g11 = 0.0 + 0.0j
        g12 = 0.0 + 0.0j
        g20 = 0.0 + 0.0j
        g21 = 0.0 + 0.0j
        g22 = 0.0 + 0.0j

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

            w = weights[j]
            inv_r3_w = (1.0 / r_eff3) * w
            inv_r5_w = (3.0 / r_eff5) * w

            g00 += dx * dx * inv_r5_w - inv_r3_w
            g01 += dx * dy * inv_r5_w
            g02 += dx * dz * inv_r5_w

            g10 += dy * dx * inv_r5_w
            g11 += dy * dy * inv_r5_w - inv_r3_w
            g12 += dy * dz * inv_r5_w

            g20 += dz * dx * inv_r5_w
            g21 += dz * dy * inv_r5_w
            g22 += dz * dz * inv_r5_w - inv_r3_w

        out[i, 0, 0] = g00
        out[i, 0, 1] = g01
        out[i, 0, 2] = g02
        out[i, 1, 0] = g10
        out[i, 1, 1] = g11
        out[i, 1, 2] = g12
        out[i, 2, 0] = g20
        out[i, 2, 1] = g21
        out[i, 2, 2] = g22
    return out


@numba.njit(parallel=True, cache=True)
def dipole_contract_real_jit(
    sources_xyz: np.ndarray,
    mass: np.ndarray,
    disp: np.ndarray,
    targets_xyz: np.ndarray,
    eps2: float,
    collision_source_index: np.ndarray,
) -> np.ndarray:
    # K_T = 3*dd/r_eff^5 - I/r_eff^3 is symmetric, so only 6 independent
    # components are computed per source (unlike gradient_contract_*_jit's
    # 9), and each is contracted against disp[j] and discarded immediately
    # -- the full 3x3 tensor is never needed, only the resulting 3-vector.
    m = len(targets_xyz)
    n = len(sources_xyz)
    out = np.zeros((m, 3), dtype=np.float64)
    for i in numba.prange(m):
        tx = targets_xyz[i, 0]
        ty = targets_xyz[i, 1]
        tz = targets_xyz[i, 2]
        ax = 0.0
        ay = 0.0
        az = 0.0
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

            txx = dx * dx * inv_r5 - inv_r3
            txy = dx * dy * inv_r5
            txz = dx * dz * inv_r5
            tyy = dy * dy * inv_r5 - inv_r3
            tyz = dy * dz * inv_r5
            tzz = dz * dz * inv_r5 - inv_r3

            mj = mass[j]
            d0 = disp[j, 0]
            d1 = disp[j, 1]
            d2 = disp[j, 2]

            ax += -mj * (txx * d0 + txy * d1 + txz * d2)
            ay += -mj * (txy * d0 + tyy * d1 + tyz * d2)
            az += -mj * (txz * d0 + tyz * d1 + tzz * d2)
        out[i, 0] = ax
        out[i, 1] = ay
        out[i, 2] = az
    return out


@numba.njit(parallel=True, cache=True)
def dipole_contract_complex_jit(
    sources_xyz: np.ndarray,
    mass: np.ndarray,
    disp: np.ndarray,
    targets_xyz: np.ndarray,
    eps2: float,
    collision_source_index: np.ndarray,
) -> np.ndarray:
    m = len(targets_xyz)
    n = len(sources_xyz)
    out = np.zeros((m, 3), dtype=np.complex128)
    for i in numba.prange(m):
        tx = targets_xyz[i, 0]
        ty = targets_xyz[i, 1]
        tz = targets_xyz[i, 2]
        ax = 0.0 + 0.0j
        ay = 0.0 + 0.0j
        az = 0.0 + 0.0j
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

            txx = dx * dx * inv_r5 - inv_r3
            txy = dx * dy * inv_r5
            txz = dx * dz * inv_r5
            tyy = dy * dy * inv_r5 - inv_r3
            tyz = dy * dz * inv_r5
            tzz = dz * dz * inv_r5 - inv_r3

            mj = mass[j]
            d0 = disp[j, 0]
            d1 = disp[j, 1]
            d2 = disp[j, 2]

            ax += -mj * (txx * d0 + txy * d1 + txz * d2)
            ay += -mj * (txy * d0 + tyy * d1 + tyz * d2)
            az += -mj * (txz * d0 + tyz * d1 + tzz * d2)
        out[i, 0] = ax
        out[i, 1] = ay
        out[i, 2] = az
    return out


def potential_contract(
    sources_xyz: list | tuple | np.ndarray,
    weights: list | tuple | np.ndarray,
    targets_xyz: list | tuple | np.ndarray,
    *,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
) -> FloatArray | ComplexArray:
    sources, w, targets, eps, _ = _validate_inputs(
        sources_xyz, weights, targets_xyz, softening_m, chunk_size
    )
    m = len(targets)
    collision_source_index = np.full(m, -1, dtype=np.int64)
    eps2 = eps**2

    if np.iscomplexobj(w):
        out = potential_contract_complex_jit(
            sources, w, targets, eps2, collision_source_index
        )
    else:
        out = potential_contract_real_jit(
            sources, w, targets, eps2, collision_source_index
        )

    _check_collisions(collision_source_index, eps)
    out.setflags(write=False)
    return out


def acceleration_contract(
    sources_xyz: list | tuple | np.ndarray,
    weights: list | tuple | np.ndarray,
    targets_xyz: list | tuple | np.ndarray,
    *,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
) -> FloatArray | ComplexArray:
    sources, w, targets, eps, _ = _validate_inputs(
        sources_xyz, weights, targets_xyz, softening_m, chunk_size
    )
    m = len(targets)
    collision_source_index = np.full(m, -1, dtype=np.int64)
    eps2 = eps**2

    if np.iscomplexobj(w):
        out = acceleration_contract_complex_jit(
            sources, w, targets, eps2, collision_source_index
        )
    else:
        out = acceleration_contract_real_jit(
            sources, w, targets, eps2, collision_source_index
        )

    _check_collisions(collision_source_index, eps)
    out.setflags(write=False)
    return out


def gradient_contract(
    sources_xyz: list | tuple | np.ndarray,
    weights: list | tuple | np.ndarray,
    targets_xyz: list | tuple | np.ndarray,
    *,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
) -> FloatArray | ComplexArray:
    """Numba accelerated gravity gradient contraction.

    Note that for softening_m > 0, the softened tensor does not satisfy tr(T) = 0.
    """
    sources, w, targets, eps, _ = _validate_inputs(
        sources_xyz, weights, targets_xyz, softening_m, chunk_size
    )
    m = len(targets)
    collision_source_index = np.full(m, -1, dtype=np.int64)
    eps2 = eps**2

    if np.iscomplexobj(w):
        out = gradient_contract_complex_jit(
            sources, w, targets, eps2, collision_source_index
        )
    else:
        out = gradient_contract_real_jit(
            sources, w, targets, eps2, collision_source_index
        )

    _check_collisions(collision_source_index, eps)
    out.setflags(write=False)
    return out


def dipole_contract(
    sources_xyz: list | tuple | np.ndarray,
    mass_kg: list | tuple | np.ndarray,
    displacement_m: list | tuple | np.ndarray,
    targets_xyz: list | tuple | np.ndarray,
    *,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
) -> FloatArray | ComplexArray:
    """Numba accelerated vector-weight (dipole/total) contraction.

    Note: chunk_size is accepted/validated but ignored (compiled CPU
    backends do not chunk the source axis; see backend_strategy.md).
    """
    sources, mass, disp, targets, eps, _ = _validate_dipole_inputs(
        sources_xyz, mass_kg, displacement_m, targets_xyz, softening_m, chunk_size
    )
    m = len(targets)
    collision_source_index = np.full(m, -1, dtype=np.int64)
    eps2 = eps**2

    if np.iscomplexobj(disp):
        out = dipole_contract_complex_jit(
            sources, mass, disp, targets, eps2, collision_source_index
        )
    else:
        out = dipole_contract_real_jit(
            sources, mass, disp, targets, eps2, collision_source_index
        )

    _check_collisions(collision_source_index, eps)
    out.setflags(write=False)
    return out
