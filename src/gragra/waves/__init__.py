"""Wave kinematics package.

Provides helper functions for direction vectors, polarizations, and phases.
"""

from gragra.waves.directions import (
    normalize_directions,
    orthonormal_basis_from_direction,
)
from gragra.waves.displacement import plane_wave_displacement
from gragra.waves.halfspace import (
    free_surface_reflection_vertical,
    surface_displacement_factor_vertical,
)
from gragra.waves.phase import (
    evanescent_complex_factor,
    plane_wave_complex_factor,
    plane_wave_phase,
)
from gragra.waves.polarization import (
    displacement_direction,
    p_wave_polarization,
    s_wave_basis,
)
from gragra.waves.rayleigh import (
    rayleigh_displacement,
    rayleigh_ellipticity,
    rayleigh_secular,
    rayleigh_speed,
)
from gragra.waves.stochastic import StochasticPSParameters, stochastic_ps_realization

__all__ = [
    "normalize_directions",
    "orthonormal_basis_from_direction",
    "p_wave_polarization",
    "s_wave_basis",
    "displacement_direction",
    "plane_wave_displacement",
    "plane_wave_phase",
    "plane_wave_complex_factor",
    "evanescent_complex_factor",
    "rayleigh_secular",
    "rayleigh_speed",
    "rayleigh_displacement",
    "rayleigh_ellipticity",
    "free_surface_reflection_vertical",
    "surface_displacement_factor_vertical",
    "StochasticPSParameters",
    "stochastic_ps_realization",
]
