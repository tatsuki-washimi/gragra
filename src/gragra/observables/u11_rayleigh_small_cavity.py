"""SI acceleration wrapper for the underground Rayleigh leading SCA model."""

from gragra.analytic.u11_rayleigh_small_cavity import (
    rayleigh_small_cavity_acceleration_g_free,
)
from gragra.constants import G_SI


def rayleigh_small_cavity_acceleration(targets_m, **kwargs):
    """Same arguments as the G-free API; three acceleration arrays in m/s².

    The SI conversion does not change diagnostic displacement u_c (meters).
    """
    result = rayleigh_small_cavity_acceleration_g_free(targets_m, **kwargs)
    parts, diagnostics = result if isinstance(result, tuple) else (result, None)
    converted = {}
    for name, value in parts.items():
        array = value * G_SI
        array.setflags(write=False)
        converted[name] = array
    return (converted, diagnostics) if diagnostics is not None else converted
