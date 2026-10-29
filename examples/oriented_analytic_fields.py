"""Evaluate a translated and rotated uniform cuboid."""

from __future__ import annotations

import numpy as np

from gragra.analytic import oriented_cuboid_observables_g_free
from gragra.observables import oriented_cuboid_observables


def main() -> None:
    angle = 0.37
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    kwargs = {
        "center_m": np.array([1.0, -2.0, 0.5]),
        "lengths_m": np.array([2.0, 3.0, 4.0]),
        "density_kg_m3": 1800.0,
        "targets_m": np.array([[4.0, 1.0, 3.0], [8.0, -4.0, 6.0]]),
        "rotation_matrix": rotation,
    }

    g_free = oriented_cuboid_observables_g_free(**kwargs)
    si = oriented_cuboid_observables(**kwargs)

    print("Oriented analytic cuboid")
    print(f"G-free potential: {g_free.potential}")
    print(f"G-free acceleration:\n{g_free.acceleration}")
    print(f"SI acceleration [m/s^2]:\n{si.acceleration}")
    print(f"G-free gradient[0]:\n{g_free.gradient[0]}")


if __name__ == "__main__":
    main()
