"""Cylinder geometry for the established dimensionless SCA wall convention."""

from __future__ import annotations

import numpy as np

from gragra.analytic.cavity_cylinder import cylinder_acceleration_gradient_g_free


def cylinder_sca_wall_coupling_tensor_g_free(
    center_m: list | tuple | np.ndarray,
    radius_m: float,
    height_m: float,
    targets_m: list | tuple | np.ndarray,
    *,
    rotation_matrix: list | tuple | np.ndarray | None = None,
) -> np.ndarray:
    """Return the dimensionless cylinder SCA wall tensor ``C=-Gamma(rho=1)``.

    The outward normal points from the cavity into the surrounding material:
    ``C_ij = integral_wall((s_i-x_i)*n_j / |s-x|**3 dS)``. Uniform wall
    displacement gives ``delta_a_wall/G = -rho * C @ u``. This is the wall
    term only, with no bulk, wavelength or elastic-scattering contribution.

    Parameters follow the existing analytic cylinder field: center (3,),
    positive radius and full height in meters, targets (M,3), and optional
    proper local-to-world rotation (3,3). Internal and external targets are
    supported; physical side/cap/rim targets are rejected. Interior trace
    is 4*pi; exterior trace is zero. Rotation validation and analytic field
    evaluation delegate to the existing cylinder implementation without
    projection or any spatial quadrature.

    Returns a read-only float64 array of shape (M,3,3). Density and complex
    displacement do not enter this geometry-only API. Apply the established
    SCA tensor contraction after this step; do not multiply the tensor by G.
    """
    _, gradient = cylinder_acceleration_gradient_g_free(
        center_m,
        radius_m,
        height_m,
        1.0,
        targets_m,
        rotation_matrix=rotation_matrix,
    )
    result = -gradient
    result.setflags(write=False)
    return result
