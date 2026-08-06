"""Wave polarization utilities."""

from typing import Literal

import numpy as np

from gragra.waves.directions import (
    normalize_directions,
    orthonormal_basis_from_direction,
)


def p_wave_polarization(k_hat: list | tuple | np.ndarray) -> np.ndarray:
    """P-wave polarization direction (same as normalized k_hat)."""
    return normalize_directions(k_hat)


def _vertical_anchored_s_wave_basis(
    k_norm: np.ndarray,
    v_norm: np.ndarray,
    degeneracy_tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Internal helper to compute vertical-anchored SH/SV basis."""
    n_dir = len(k_norm)
    v_unit = v_norm[0]  # shape (3,)

    e_sh_list = []
    e_sv_list = []

    for i in range(n_dir):
        k = k_norm[i]  # shape (3,)

        cross_v_k = np.cross(v_unit, k)
        cross_norm = np.linalg.norm(cross_v_k)

        if cross_norm < degeneracy_tol:
            # Degenerate case: use x-axis [1.0, 0.0, 0.0]
            ref_unit = np.array([1.0, 0.0, 0.0])
            # If v_unit itself is parallel to x-axis,
            # use y-axis [0.0, 1.0, 0.0] as fallback
            if np.abs(np.abs(v_unit[0]) - 1.0) < degeneracy_tol:
                ref_unit = np.array([0.0, 1.0, 0.0])

            cross_ref_k = np.cross(ref_unit, k)
            cross_ref_norm = np.linalg.norm(cross_ref_k)

            if cross_ref_norm < degeneracy_tol:
                # Fallback if somehow still degenerate
                ref_unit = np.array([0.0, 1.0, 0.0])
                cross_ref_k = np.cross(ref_unit, k)
                cross_ref_norm = np.linalg.norm(cross_ref_k)

            e_sh = cross_ref_k / cross_ref_norm
        else:
            e_sh = cross_v_k / cross_norm

        e_sv = np.cross(k, e_sh)

        e_sh_list.append(e_sh)
        e_sv_list.append(e_sv)

    e_sh_arr = np.array(e_sh_list, dtype=np.float64)
    e_sv_arr = np.array(e_sv_list, dtype=np.float64)

    e_sh_arr.setflags(write=False)
    e_sv_arr.setflags(write=False)

    return e_sh_arr, e_sv_arr


def s_wave_basis(
    k_hat: list | tuple | np.ndarray,
    *,
    vertical_unit: list | tuple | np.ndarray | None = None,
    degeneracy_tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """S-wave polarization basis (e1, e2) orthogonal to k_hat.

    If vertical_unit is None, a generic orthonormal basis is returned.
    If vertical_unit is provided, the physical vertical-anchored (e_SH, e_SV) basis
    matching the seismic convention is returned.
    """
    if vertical_unit is not None:
        v_arr = np.asarray(vertical_unit)
        if v_arr.ndim == 1:
            if v_arr.shape != (3,):
                raise ValueError(
                    f"vertical_unit must have shape (3,) or (1,3), got {v_arr.shape}"
                )
        elif v_arr.ndim == 2:
            if v_arr.shape != (1, 3):
                raise ValueError(
                    f"vertical_unit must have shape (3,) or (1,3), got {v_arr.shape}"
                )
        else:
            raise ValueError(
                f"vertical_unit must have shape (3,) or (1,3), got {v_arr.shape}"
            )

        if not isinstance(
            degeneracy_tol, (int, float, np.integer, np.floating)
        ) or isinstance(degeneracy_tol, bool):
            raise TypeError("degeneracy_tol must be a finite float or int")
        if not np.isfinite(degeneracy_tol):
            raise ValueError("degeneracy_tol must be finite")
        if not (0.0 < degeneracy_tol <= 1.0):
            raise ValueError(
                f"degeneracy_tol must be in interval (0, 1.0], got {degeneracy_tol}"
            )

        k_norm = normalize_directions(k_hat)
        v_norm = normalize_directions(vertical_unit)
        return _vertical_anchored_s_wave_basis(k_norm, v_norm, float(degeneracy_tol))

    k_norm = normalize_directions(k_hat)
    n_dir = len(k_norm)

    e1_list = []
    e2_list = []
    for i in range(n_dir):
        e1, e2 = orthonormal_basis_from_direction(k_norm[i])
        e1_list.append(e1)
        e2_list.append(e2)

    e1_arr = np.array(e1_list, dtype=np.float64)
    e2_arr = np.array(e2_list, dtype=np.float64)

    e1_arr.setflags(write=False)
    e2_arr.setflags(write=False)

    return e1_arr, e2_arr


def _custom_polarization(
    k_hat: list | tuple | np.ndarray,
    polarization: list | tuple | np.ndarray | None,
    orthogonality_tol: float,
) -> np.ndarray:
    """Internal helper to validate and compute custom polarization directions."""
    if polarization is None:
        raise ValueError("polarization must be provided for wave_type='custom'")

    if not isinstance(
        orthogonality_tol, (int, float, np.integer, np.floating)
    ) or isinstance(orthogonality_tol, bool):
        raise TypeError("orthogonality_tol must be a finite float or int")
    if not np.isfinite(orthogonality_tol):
        raise ValueError("orthogonality_tol must be finite")
    if not (0.0 < orthogonality_tol <= 1.0):
        raise ValueError(
            f"orthogonality_tol must be in interval (0, 1.0], got {orthogonality_tol}"
        )

    k_norm = normalize_directions(k_hat)
    p_norm = normalize_directions(polarization)

    if k_norm.shape != p_norm.shape:
        raise ValueError(
            f"polarization shape {p_norm.shape} must match k_hat shape {k_norm.shape}"
        )

    dots = np.sum(k_norm * p_norm, axis=1)
    if np.any(np.abs(dots) > orthogonality_tol):
        raise ValueError("Custom polarization must be orthogonal to k_hat")

    p_norm_readonly = p_norm.copy()
    p_norm_readonly.setflags(write=False)
    return p_norm_readonly


def displacement_direction(
    wave_type: Literal["P", "S1", "S2", "SH", "SV", "custom"],
    k_hat: list | tuple | np.ndarray,
    *,
    vertical_unit: list | tuple | np.ndarray = (0.0, 0.0, 1.0),
    polarization: list | tuple | np.ndarray | None = None,
    orthogonality_tol: float = 1e-5,
) -> np.ndarray:
    """Displacement direction units according to wave type."""
    if wave_type != "custom" and polarization is not None:
        raise ValueError(f"polarization must be None for wave_type='{wave_type}'")

    if wave_type == "P":
        return p_wave_polarization(k_hat)
    if wave_type == "S1":
        return s_wave_basis(k_hat)[0]
    if wave_type == "S2":
        return s_wave_basis(k_hat)[1]
    if wave_type == "SH":
        return s_wave_basis(k_hat, vertical_unit=vertical_unit)[0]
    if wave_type == "SV":
        return s_wave_basis(k_hat, vertical_unit=vertical_unit)[1]
    if wave_type == "custom":
        return _custom_polarization(k_hat, polarization, orthogonality_tol)
    raise ValueError(
        f"Unknown wave_type: {wave_type}. "
        "Must be 'P', 'S1', 'S2', 'SH', 'SV', or 'custom'"
    )
