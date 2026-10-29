"""LIT-C: general-point finite-cylinder gravity smoke example.

Analytic acceleration/gradient support proper rotation. The older bundled
API supplies numerical potential as a separate z-aligned diagnostic.
"""

from __future__ import annotations

import numpy as np

from gragra.analytic import (
    cylinder_acceleration_gradient_g_free,
    cylinder_observables_g_free,
)
from gragra.observables import cylinder_acceleration_gradient


def main() -> None:
    center = np.array([0.5, -0.25, 0.3])
    targets = np.array(
        [
            [3.1, -1.3, 2.4],
            [0.5, -0.25, 5.0],
        ]
    )
    kwargs = {
        "center_m": center,
        "radius_m": 2.0,
        "height_m": 3.0,
        "density_kg_m3": 1800.0,
        "targets_m": targets,
        "quadrature_order": 32,
    }

    g_free = cylinder_observables_g_free(**kwargs)
    analytic_kwargs = {
        key: value for key, value in kwargs.items() if key != "quadrature_order"
    }
    angle = 0.4
    rotation = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    analytic_kwargs["targets_m"] = center + (targets - center) @ rotation.T
    analytic_acc, analytic_grad = cylinder_acceleration_gradient_g_free(
        **analytic_kwargs, rotation_matrix=rotation
    )
    si_acc, _ = cylinder_acceleration_gradient(
        **analytic_kwargs, rotation_matrix=rotation
    )

    print("LIT-C general-point finite cylinder")
    print(f"targets: {len(targets)} external points")
    print(f"Numerical diagnostic potential, z-aligned [kg/m]: {g_free.potential}")
    print(f"Analytic rotated acceleration [kg/m^2]:\n{analytic_acc}")
    print(f"SI rotated acceleration [m/s^2]:\n{si_acc}")
    print(f"Analytic rotated gradient[0] [kg/m^3]:\n{analytic_grad[0]}")


if __name__ == "__main__":
    main()
