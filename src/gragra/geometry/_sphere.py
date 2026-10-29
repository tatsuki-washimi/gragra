"""Pure high-level sphere geometry; density and discretization live elsewhere."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy import units as u

from gragra.geometry._appearance import Appearance
from gragra.geometry._frame import Frame, _lengths_m


@dataclass(frozen=True, eq=False, init=False)
class Sphere:
    """Static sphere with an owned center and positive radius in a Frame."""

    frame: Frame
    _center_tuple_m: tuple[float, float, float]
    radius_m: float
    appearance: Appearance | None

    def __init__(
        self,
        center: u.Quantity,
        radius: u.Quantity,
        frame: Frame,
        *,
        appearance: Appearance | None = None,
    ) -> None:
        if not isinstance(frame, Frame):
            raise TypeError("frame must be a Frame")
        if appearance is not None and not isinstance(appearance, Appearance):
            raise TypeError("appearance must be an Appearance or None")
        center_m = _lengths_m("center", center, (3,))
        radius_m = _lengths_m("radius", radius, ())
        radius_value = float(radius_m)
        if radius_value <= 0:
            raise ValueError("radius must be positive")
        object.__setattr__(self, "frame", frame)
        object.__setattr__(self, "_center_tuple_m", tuple(float(x) for x in center_m))
        object.__setattr__(self, "radius_m", radius_value)
        object.__setattr__(self, "appearance", appearance)

    @property
    def center_m(self) -> np.ndarray:
        return np.asarray(self._center_tuple_m, dtype=np.float64)

    @property
    def center(self) -> u.Quantity:
        return self.center_m * u.m

    @property
    def radius(self) -> u.Quantity:
        return self.radius_m * u.m

    @property
    def volume(self) -> u.Quantity:
        return (4.0 * np.pi / 3.0) * self.radius**3
