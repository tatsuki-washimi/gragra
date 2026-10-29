"""Geometry of a flat Rayleigh half-space."""

from dataclasses import dataclass

from gragra.geometry._appearance import Appearance
from gragra.geometry._frame import Frame


@dataclass(frozen=True, eq=False)
class HalfSpace:
    """The local z=0 free surface; the elastic medium occupies local z<0."""

    frame: Frame
    appearance: Appearance | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.frame, Frame):
            raise TypeError("frame must be a Frame")
        if self.appearance is not None and not isinstance(self.appearance, Appearance):
            raise TypeError("appearance must be an Appearance or None")
