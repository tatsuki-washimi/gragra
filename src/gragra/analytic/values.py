"""Value types for analytical and G-free gravitational representations."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

ArrayLike = list | tuple | np.ndarray


def _validate_readonly_array(
    name: str,
    value: ArrayLike,
    *,
    complex_allowed: bool = True,
) -> np.ndarray:
    """Validate numeric array, ensure float64 or complex128, copy, and mark read-only."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must not be boolean")
    raw = np.asarray(value)
    if raw.dtype.kind == "b":
        raise TypeError(f"{name} must not contain boolean values")
    if raw.dtype.kind == "O":
        raise TypeError(f"{name} must not contain object dtype")

    is_complex = np.iscomplexobj(raw)
    if is_complex and not complex_allowed:
        raise TypeError(f"{name} must contain real values")

    dtype = np.complex128 if is_complex else np.float64
    try:
        arr = np.asarray(value, dtype=dtype).copy()
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be numeric") from error

    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")

    arr.setflags(write=False)
    return arr


@dataclass(frozen=True, init=False)
class GFreeGravityValues:
    """Potential, acceleration, and gradient in G-free units.

    Physical units:
    - potential: kg / m
    - acceleration: kg / m^2
    - gradient: kg / m^3

    Shapes:
    - Static targets: ``potential: (M,)``, ``acceleration: (M, 3)``, ``gradient: (M, 3, 3)``
    - Series targets: ``potential: (S, M)``, ``acceleration: (S, M, 3)``, ``gradient: (S, M, 3, 3)``
    """

    potential: np.ndarray
    acceleration: np.ndarray
    gradient: np.ndarray

    def __init__(
        self,
        potential: ArrayLike,
        acceleration: ArrayLike,
        gradient: ArrayLike,
    ) -> None:
        raw_p = np.asarray(potential)
        raw_a = np.asarray(acceleration)
        raw_g = np.asarray(gradient)

        if raw_p.ndim not in (1, 2):
            raise ValueError(
                f"potential must have 1 or 2 dimensions (M,) or (S, M); got ndim={raw_p.ndim}"
            )
        prefix = raw_p.shape
        if raw_a.shape != (*prefix, 3):
            raise ValueError(
                f"acceleration shape {raw_a.shape} must match (*potential.shape, 3) = {(*prefix, 3)}"
            )
        if raw_g.shape != (*prefix, 3, 3):
            raise ValueError(
                f"gradient shape {raw_g.shape} must match (*potential.shape, 3, 3) = {(*prefix, 3, 3)}"
            )

        is_complex = any(np.iscomplexobj(x) for x in (raw_p, raw_a, raw_g))
        target_dtype = np.complex128 if is_complex else np.float64

        p_arr = np.asarray(raw_p, dtype=target_dtype).copy()
        a_arr = np.asarray(raw_a, dtype=target_dtype).copy()
        g_arr = np.asarray(raw_g, dtype=target_dtype).copy()

        for name, arr in [("potential", p_arr), ("acceleration", a_arr), ("gradient", g_arr)]:
            if not np.isfinite(arr).all():
                raise ValueError(f"{name} must contain only finite values")
            arr.setflags(write=False)

        object.__setattr__(self, "potential", p_arr)
        object.__setattr__(self, "acceleration", a_arr)
        object.__setattr__(self, "gradient", g_arr)

    @property
    def is_series(self) -> bool:
        return self.potential.ndim == 2

    def __iter__(self) -> Iterator[np.ndarray]:
        return iter((self.potential, self.acceleration, self.gradient))


@dataclass(frozen=True, init=False)
class GFreeSphericalShellPComponents:
    """Component decomposition of spherical shell P-wave acceleration in G-free units (kg / m^2).

    Fields:
    - ``bulk``: Volume compression contribution of the shell.
    - ``inner_wall``: Surface mass sheet contribution of the inner cavity wall.
    - ``outer_boundary``: Surface mass sheet contribution of the outer boundary.
    - ``total``: Total acceleration, equal to ``bulk + inner_wall + outer_boundary``.
    - ``unit_system``: Fixed identifier string ``"g_free"``.

    Shapes:
    - Scalar wavenumber: each field is ``(1, 3)`` (1 center point, 3 spatial components).
    - 1D wavenumber array of length K: each field is ``(K, 1, 3)``.
    """

    bulk: np.ndarray
    inner_wall: np.ndarray
    outer_boundary: np.ndarray
    total: np.ndarray
    unit_system: str = "g_free"

    def __init__(
        self,
        bulk: ArrayLike,
        inner_wall: ArrayLike,
        outer_boundary: ArrayLike,
        total: ArrayLike,
    ) -> None:
        raw_b = np.asarray(bulk)
        raw_iw = np.asarray(inner_wall)
        raw_ob = np.asarray(outer_boundary)
        raw_tot = np.asarray(total)

        if raw_tot.ndim not in (2, 3) or raw_tot.shape[-2:] != (1, 3):
            raise ValueError(
                f"components must have shape (1, 3) or (K, 1, 3); got shape={raw_tot.shape}"
            )
        expected_shape = raw_tot.shape
        for name, raw in [
            ("bulk", raw_b),
            ("inner_wall", raw_iw),
            ("outer_boundary", raw_ob),
        ]:
            if raw.shape != expected_shape:
                raise ValueError(
                    f"{name} shape {raw.shape} must match total shape {expected_shape}"
                )

        is_complex = any(
            np.iscomplexobj(x) for x in (raw_b, raw_iw, raw_ob, raw_tot)
        )
        target_dtype = np.complex128 if is_complex else np.float64

        b_arr = np.asarray(raw_b, dtype=target_dtype).copy()
        iw_arr = np.asarray(raw_iw, dtype=target_dtype).copy()
        ob_arr = np.asarray(raw_ob, dtype=target_dtype).copy()
        tot_arr = np.asarray(raw_tot, dtype=target_dtype).copy()

        for name, arr in [
            ("bulk", b_arr),
            ("inner_wall", iw_arr),
            ("outer_boundary", ob_arr),
            ("total", tot_arr),
        ]:
            if not np.isfinite(arr).all():
                raise ValueError(f"{name} must contain only finite values")
            arr.setflags(write=False)

        object.__setattr__(self, "bulk", b_arr)
        object.__setattr__(self, "inner_wall", iw_arr)
        object.__setattr__(self, "outer_boundary", ob_arr)
        object.__setattr__(self, "total", tot_arr)
        object.__setattr__(self, "unit_system", "g_free")

    def __iter__(self) -> Iterator[np.ndarray]:
        return iter((self.bulk, self.inner_wall, self.outer_boundary, self.total))


@dataclass(frozen=True, init=False)
class GFreeSphericalShellSComponents:
    """Component decomposition of spherical shell S-wave acceleration in G-free units (kg / m^2).

    Fields:
    - ``bulk``: Always identically zero. An S wave (shear wave) has zero
      divergence (div(u) = 0), so it produces no volumetric density
      perturbation and therefore no bulk contribution. This field is kept
      only so the S-wave container has the same 4-field structure as
      ``GFreeSphericalShellPComponents``, letting the same
      ``total == bulk + inner_wall + outer_boundary`` identity apply
      uniformly to both wave types.
    - ``inner_wall``: Surface mass sheet contribution of the inner cavity wall.
    - ``outer_boundary``: Surface mass sheet contribution of the outer boundary.
    - ``total``: Total acceleration, equal to ``inner_wall + outer_boundary``
      (since ``bulk`` is always zero).
    - ``unit_system``: Fixed identifier string ``"g_free"``.

    Shapes:
    - Scalar wavenumber: each field is ``(1, 3)`` (1 center point, 3 spatial components).
    - 1D wavenumber array of length K: each field is ``(K, 1, 3)``.
    """

    bulk: np.ndarray
    inner_wall: np.ndarray
    outer_boundary: np.ndarray
    total: np.ndarray
    unit_system: str = "g_free"

    def __init__(
        self,
        bulk: ArrayLike,
        inner_wall: ArrayLike,
        outer_boundary: ArrayLike,
        total: ArrayLike,
    ) -> None:
        raw_b = np.asarray(bulk)
        raw_iw = np.asarray(inner_wall)
        raw_ob = np.asarray(outer_boundary)
        raw_tot = np.asarray(total)

        if raw_tot.ndim not in (2, 3) or raw_tot.shape[-2:] != (1, 3):
            raise ValueError(
                f"components must have shape (1, 3) or (K, 1, 3); got shape={raw_tot.shape}"
            )
        expected_shape = raw_tot.shape
        for name, raw in [
            ("bulk", raw_b),
            ("inner_wall", raw_iw),
            ("outer_boundary", raw_ob),
        ]:
            if raw.shape != expected_shape:
                raise ValueError(
                    f"{name} shape {raw.shape} must match total shape {expected_shape}"
                )

        is_complex = any(
            np.iscomplexobj(x) for x in (raw_b, raw_iw, raw_ob, raw_tot)
        )
        target_dtype = np.complex128 if is_complex else np.float64

        b_arr = np.asarray(raw_b, dtype=target_dtype).copy()
        iw_arr = np.asarray(raw_iw, dtype=target_dtype).copy()
        ob_arr = np.asarray(raw_ob, dtype=target_dtype).copy()
        tot_arr = np.asarray(raw_tot, dtype=target_dtype).copy()

        for name, arr in [
            ("bulk", b_arr),
            ("inner_wall", iw_arr),
            ("outer_boundary", ob_arr),
            ("total", tot_arr),
        ]:
            if not np.isfinite(arr).all():
                raise ValueError(f"{name} must contain only finite values")
            arr.setflags(write=False)

        object.__setattr__(self, "bulk", b_arr)
        object.__setattr__(self, "inner_wall", iw_arr)
        object.__setattr__(self, "outer_boundary", ob_arr)
        object.__setattr__(self, "total", tot_arr)
        object.__setattr__(self, "unit_system", "g_free")

    def __iter__(self) -> Iterator[np.ndarray]:
        return iter((self.bulk, self.inner_wall, self.outer_boundary, self.total))


# Backward compatibility alias
SphericalShellPComponents = GFreeSphericalShellPComponents

