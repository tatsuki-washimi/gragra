"""SI wrapper for the finite prescribed-wave spherical shell."""

from gragra.analytic.u10_spherical_shell_general import (
    _tolerance,
    spherical_shell_observables_g_free,
)
from gragra.constants import G_SI
from gragra.observables.analytic import to_gravity_observables


def spherical_shell_observables(targets_m, *, atol_si=None, **kwargs):
    """SI potential, acceleration and gradient for all four shell parts.

    Keywords match spherical_shell_observables_g_free except that its `atol`
    becomes `atol_si`, in (m²/s², m/s², s⁻²). It is converted to G-free once;
    the output conversion applies G_SI once. Diagnostics explicitly retain
    G-free units. return_diagnostics also returns the per-call diagnostics.
    """
    if "atol" in kwargs:
        raise TypeError("SI tolerances must use atol_si")
    atol = None if atol_si is None else _tolerance(atol_si, "atol_si") / G_SI
    result = spherical_shell_observables_g_free(targets_m, atol=atol, **kwargs)
    if kwargs.get("return_diagnostics", False):
        parts, diagnostics = result
        return {
            part: to_gravity_observables(value) for part, value in parts.items()
        }, diagnostics
    return {part: to_gravity_observables(value) for part, value in result.items()}
