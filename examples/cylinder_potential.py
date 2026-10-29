"""V: analytic rotated cylinder triple and the separate numerical diagnostic."""

import numpy as np

from gragra.analytic import (
    cylinder_analytic_observables_g_free,
    cylinder_observables_g_free,
)
from gragra.constants import G_SI
from gragra.observables import cylinder_analytic_observables


def main():
    center = np.array([0.5, -0.25, 0.3])
    local_targets = np.array([[0.2, 0.3, 0.1], [2.6, -1.05, 2.1]])
    rotation = np.array([[0.6, 0, 0.8], [0, 1, 0], [-0.8, 0, 0.6]])
    world_targets = center + local_targets @ rotation.T
    values = cylinder_analytic_observables_g_free(
        center, 2, 3, 1800, world_targets, rotation_matrix=rotation
    )
    si = cylinder_analytic_observables(
        center, 2, 3, 1800, world_targets, rotation_matrix=rotation
    )
    numerical = cylinder_observables_g_free(
        [0, 0, 0], 2, 3, 1800, local_targets, quadrature_order=96
    )
    np.testing.assert_allclose(values.potential, numerical.potential, rtol=1e-10)
    for name in ("potential", "acceleration", "gradient"):
        np.testing.assert_array_equal(getattr(si, name), G_SI * getattr(values, name))
        assert not getattr(si, name).flags.writeable
    print("Analytic infinity-zero potential [kg/m]:", values.potential)
    print("SI potential [m²/s²]:", si.potential)
    print("SI acceleration [m/s²]:", si.acceleration)
    print("SI gradient [1/s²]:", si.gradient)


if __name__ == "__main__":
    main()
