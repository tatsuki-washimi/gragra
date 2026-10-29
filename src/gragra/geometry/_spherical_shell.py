"""Pure, unit-aware spherical shell geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy import units as u

from gragra.geometry._appearance import Appearance
from gragra.geometry._frame import Frame, _lengths_m


@dataclass(frozen=True, eq=False, init=False)
class SphericalShell:
    """Spherical shell centered in a Frame, with ordered positive radii."""

    frame: Frame
    _center_tuple_m: tuple[float, float, float]
    inner_radius_m: float
    outer_radius_m: float
    appearance: Appearance | None

    def __init__(
        self,
        center: u.Quantity,
        inner_radius: u.Quantity,
        outer_radius: u.Quantity,
        frame: Frame,
        *,
        appearance: Appearance | None = None,
    ) -> None:
        if not isinstance(frame, Frame):
            raise TypeError("frame must be a Frame")
        if appearance is not None and not isinstance(appearance, Appearance):
            raise TypeError("appearance must be an Appearance or None")
        center_m = _lengths_m("center", center, (3,))
        inner_radius_m = float(_lengths_m("inner_radius", inner_radius, ()))
        outer_radius_m = float(_lengths_m("outer_radius", outer_radius, ()))
        if inner_radius_m <= 0:
            raise ValueError("inner_radius must be positive")
        if outer_radius_m <= inner_radius_m:
            raise ValueError("outer_radius must be greater than inner_radius")
        object.__setattr__(self, "frame", frame)
        object.__setattr__(self, "_center_tuple_m", tuple(float(x) for x in center_m))
        object.__setattr__(self, "inner_radius_m", inner_radius_m)
        object.__setattr__(self, "outer_radius_m", outer_radius_m)
        object.__setattr__(self, "appearance", appearance)

    @property
    def center_m(self) -> np.ndarray:
        return np.asarray(self._center_tuple_m, dtype=np.float64)

    @property
    def center(self) -> u.Quantity:
        return self.center_m * u.m

    @property
    def inner_radius(self) -> u.Quantity:
        return self.inner_radius_m * u.m

    @property
    def outer_radius(self) -> u.Quantity:
        return self.outer_radius_m * u.m
