"""SI wrappers for U9 uniform cavity-wall observables; G is applied once."""

from gragra.analytic.u9_sca_wall_observables import (
    cuboid_sca_wall_observables_g_free,
    cylinder_sca_wall_observables_g_free,
    sphere_sca_wall_observables_g_free,
)
from gragra.observables.analytic import to_gravity_observables


def sphere_sca_wall_observables(targets_m, **kwargs):
    """SI wall potential (m²/s²), acceleration (m/s²), gradient (s⁻²)."""
    return to_gravity_observables(
        sphere_sca_wall_observables_g_free(targets_m, **kwargs)
    )


def cuboid_sca_wall_observables(targets_m, **kwargs):
    """SI cuboid wall values; half lengths, sample axes and rotation as G-free API."""
    return to_gravity_observables(
        cuboid_sca_wall_observables_g_free(targets_m, **kwargs)
    )


def cylinder_sca_wall_observables(targets_m, **kwargs):
    """SI cylinder wall values; full height and sample axes as G-free API."""
    return to_gravity_observables(
        cylinder_sca_wall_observables_g_free(targets_m, **kwargs)
    )
