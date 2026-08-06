"""Half-space free-surface reflection coefficients for vertical incidence.

Ported from nn-sus src/nnsus/newtonian/halfspace.py (commit 9a5e426, 2026-06-23)
as specified in docs/architecture/halfspace_conventions.md.

G-free (no ``G_SI`` import or multiplication).
General (non-vertical) incidence-angle P<->SV conversion is out of scope.
"""

from typing import Literal


def free_surface_reflection_vertical(
    wave_type: Literal["P", "SH", "SV"],
) -> dict[str, float]:
    """Calculate free-surface displacement reflection coefficients.

    Calculation is for vertical incidence.

    Parameters
    ----------
    wave_type : {"P", "SH", "SV"}
        The type of the incident wave.

    Returns
    -------
    dict[str, float]
        Real displacement reflection coefficients.
        - For "P" and "SV": {"R_PP": 1.0, "R_PS": 0.0, "R_SP": 0.0, "R_SS": 1.0}
        - For "SH": {"R_SH": 1.0}

    Raises
    ------
    ValueError
        If wave_type is not one of "P", "SH", "SV".
    """
    if wave_type not in ("P", "SH", "SV"):
        raise ValueError(f"Unknown wave_type: {wave_type}. Must be 'P', 'SH', or 'SV'.")

    if wave_type == "SH":
        return {"R_SH": 1.0}
    return {"R_PP": 1.0, "R_PS": 0.0, "R_SP": 0.0, "R_SS": 1.0}


def surface_displacement_factor_vertical() -> float:
    """Free-surface displacement-doubling factor at vertical incidence.

    Returns
    -------
    float
        The displacement doubling factor, which is 2.0.
    """
    return 2.0
