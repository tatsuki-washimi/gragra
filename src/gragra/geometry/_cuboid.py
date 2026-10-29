"""Pure, unit-aware cuboid geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy import units as u

from gragra.geometry._appearance import Appearance
from gragra.geometry._frame import Frame, _lengths_m


@dataclass(frozen=True, eq=False, init=False)
class Cuboid:
    """Cuboid centered in a Frame, with edges parallel to its local axes."""

    frame: Frame
    _center_tuple_m: tuple[float, float, float]
    _lengths_tuple_m: tuple[float, float, float]
    appearance: Appearance | None

    def __init__(
        self,
        center: u.Quantity,
        lengths: u.Quantity,
        frame: Frame,
        *,
        appearance: Appearance | None = None,
    ) -> None:
        if not isinstance(frame, Frame):
            raise TypeError("frame must be a Frame")
        if appearance is not None and not isinstance(appearance, Appearance):
            raise TypeError("appearance must be an Appearance or None")
        center_m = _lengths_m("center", center, (3,))
        lengths_m = _lengths_m("lengths", lengths, (3,))
        if np.any(lengths_m <= 0):
            raise ValueError("lengths must be positive")
        object.__setattr__(self, "frame", frame)
        object.__setattr__(self, "_center_tuple_m", tuple(float(x) for x in center_m))
        object.__setattr__(self, "_lengths_tuple_m", tuple(float(x) for x in lengths_m))
        object.__setattr__(self, "appearance", appearance)

    @property
    def center_m(self) -> np.ndarray:
        return np.asarray(self._center_tuple_m, dtype=np.float64)

    @property
    def center(self) -> u.Quantity:
        return self.center_m * u.m

    @property
    def lengths_m(self) -> np.ndarray:
        return np.asarray(self._lengths_tuple_m, dtype=np.float64)

    @property
    def lengths(self) -> u.Quantity:
        return self.lengths_m * u.m
