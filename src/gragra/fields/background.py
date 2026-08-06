"""Background density protocols for external fields."""

from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


@runtime_checkable
class BackgroundDensityModel(Protocol):
    """Protocol for background density models.

    Evaluates the background density rho0 at the given coordinates.
    """

    def __call__(self, points_m: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Evaluate background density.

        Parameters
        ----------
        points_m : (Nc, 3) float64 array
            Coordinates where the background density is evaluated.

        Returns
        -------
        (Nc,) float64 array
            The background density value at each point.
        """
        ...


@runtime_checkable
class DifferentiableBackgroundDensityModel(BackgroundDensityModel, Protocol):
    """Protocol for differentiable background density models.

    Extends BackgroundDensityModel to provide the spatial gradient (nabla rho0).
    """

    def gradient(self, points_m: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Evaluate the spatial gradient of background density.

        Parameters
        ----------
        points_m : (Nc, 3) float64 array
            Coordinates where the gradient is evaluated.

        Returns
        -------
        (Nc, 3) float64 array
            The spatial gradient (d_rho0/dx, d_rho0/dy, d_rho0/dz) at each point.
        """
        ...
