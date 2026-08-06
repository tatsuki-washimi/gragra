"""Representation of target points where gravitational
field observables are calculated.
"""

from dataclasses import dataclass

import numpy as np

from gragra._arrays import as_xyz_table, as_xyz_vector
from gragra.array_types import FloatArray


@dataclass(frozen=True)
class TargetPoints:
    """A collection of M target points in space.

    Parameters
    ----------
    positions_m : array-like
        Coordinates table of shape (M, 3) in meters.
    """

    positions_m: FloatArray

    def __post_init__(self):
        positions = as_xyz_table("positions_m", self.positions_m)
        object.__setattr__(self, "positions_m", positions)

    @classmethod
    def from_single(cls, point: list | tuple | np.ndarray) -> "TargetPoints":
        """Create a TargetPoints instance from a single 3D point.

        Parameters
        ----------
        point : array-like
            A 3D coordinate vector of shape (3,).

        Returns
        -------
        TargetPoints
            A TargetPoints collection containing only the single point.
        """
        # Validate vector and convert to (1, 3) table
        vec = as_xyz_vector("point", point)
        return cls(positions_m=vec[np.newaxis, :])
