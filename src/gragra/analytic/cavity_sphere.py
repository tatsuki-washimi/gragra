"""Analytical benchmark solutions for a spherical cavity in an infinite medium.

Physics conventions and formulation notes:
- G-free convention: this package computes in a G-free (G_SI = 1) semantic
  space. To obtain an actual acceleration in m/s^2 the caller must multiply
  by G_SI (6.6743e-11). The physical unit of the output is kg/m^2
  (acceleration divided by G).
- Complex-output disclaimer: outputs of this module represent linear
  complex amplitudes or transfer-function-like responses, not directly
  real-valued time-domain gravitational fields.
- Formulation distinction: Newtonian gravity for a spherical cavity admits
  a "pure bulk" formulation and a "dipole/total" formulation.
  1. "Pure bulk": the bulk term (from the density perturbation) and the
     wall term (from the cavity-wall displacement) are treated as separate
     physical quantities. gragra's numerical path (bulk weights +
     SurfaceMassSheetSource) corresponds to this formulation.
  2. "Dipole/total": the total acceleration is obtained directly from a
     dipole expansion of the displaced mass elements. This holds only for
     the sum total = bulk + wall; it cannot be compared against either
     term individually.
  Every function here states which formulation and which term it computes,
  in its name and in its arguments.
"""

from typing import Literal

import numpy as np

from gragra.analytic._bessel import (
    _j1_over_x,
    cavity_wall_geometric,
    spherical_j0,
)
from gragra.analytic._validation import validate_array_input, validate_scalar_input


def spherical_cavity_bulk_p(
    radius_m: float,
    wavenumber_rad_m: float | list | tuple | np.ndarray,
    density_kg_m3: float,
    displacement_m: float,
) -> np.ndarray:
    """Calculate the G-free Newtonian acceleration amplitude of the P-wave bulk term.

    G-free convention:
        The output is G-free (G_SI = 1); its physical unit is kg/m^2
        (acceleration divided by G).
    Complex-output disclaimer:
        The output is the real-valued magnitude scale of a linear complex
        amplitude, not a real-valued time-domain gravitational field.
    Formulation distinction:
        Computes the bulk component of the pure-bulk formulation only.
        Formula: bulk(P) = 4*pi * rho0 * xi * j0(k_P * R)

    Parameters
    ----------
    radius_m : float
        Radius of the spherical cavity in meters (positive).
    wavenumber_rad_m : float or array-like
        Wavenumber of the P-wave in rad/m (non-negative).
    density_kg_m3 : float
        Medium background mass density in kg/m^3 (positive).
    displacement_m : float
        Displacement amplitude of the P-wave in meters (positive).

    Returns
    -------
    np.ndarray
        P-wave bulk acceleration amplitude array of shape matching wavenumber_rad_m.
    """
    r = validate_scalar_input(radius_m, "radius_m")
    rho = validate_scalar_input(density_kg_m3, "density_kg_m3")
    xi = validate_scalar_input(displacement_m, "displacement_m")
    k = validate_array_input(wavenumber_rad_m, "wavenumber_rad_m")

    x = k * r
    j0_val = spherical_j0(x)
    out = 4.0 * np.pi * rho * xi * j0_val

    out.setflags(write=False)
    return out


def spherical_cavity_wall(
    radius_m: float,
    wavenumber_rad_m: float | list | tuple | np.ndarray,
    density_kg_m3: float,
    displacement_m: float,
    *,
    wave_type: Literal["P", "S"] = "P",
) -> np.ndarray:
    """Calculate the G-free Newtonian acceleration amplitude of the cavity wall term.

    G-free convention:
        The output is G-free (G_SI = 1); its physical unit is kg/m^2
        (acceleration divided by G).
    Complex-output disclaimer:
        The output is the real-valued magnitude scale of a linear complex
        amplitude, not a real-valued time-domain gravitational field.
    Formulation distinction:
        Computes the wall (surface mass sheet) component of the pure-bulk
        formulation.
        P wave: wall(P) = -4*pi * rho0 * xi * (j0(x) - 2*j1(x)/x)
        S wave: wall(S) = -4*pi * rho0 * xi * j1(x)/x
        (Note: for an S wave bulk(S) = 0, so wall(S) is itself the total.)

    Parameters
    ----------
    radius_m : float
        Radius of the spherical cavity in meters (positive).
    wavenumber_rad_m : float or array-like
        Wavenumber of the wave in rad/m (non-negative).
    density_kg_m3 : float
        Medium background mass density in kg/m^3 (positive).
    displacement_m : float
        Displacement amplitude of the wave in meters (positive).
    wave_type : str, optional
        Wave type, either "P" or "S" (defaults to "P").

    Returns
    -------
    np.ndarray
        Cavity wall acceleration amplitude array of shape matching wavenumber_rad_m.
    """
    r = validate_scalar_input(radius_m, "radius_m")
    rho = validate_scalar_input(density_kg_m3, "density_kg_m3")
    xi = validate_scalar_input(displacement_m, "displacement_m")
    k = validate_array_input(wavenumber_rad_m, "wavenumber_rad_m")

    x = k * r

    if wave_type == "P":
        geom = cavity_wall_geometric(x)
    elif wave_type == "S":
        geom = _j1_over_x(x)
    else:
        raise ValueError(f"Unsupported wave_type: {wave_type}. Must be 'P' or 'S'")

    out = -4.0 * np.pi * rho * xi * geom
    out.setflags(write=False)
    return out


def spherical_cavity_total_p(
    radius_m: float,
    wavenumber_rad_m: float | list | tuple | np.ndarray,
    density_kg_m3: float,
    displacement_m: float,
) -> np.ndarray:
    """Calculate the G-free Newtonian acceleration amplitude of the P-wave total term.

    G-free convention:
        The output is G-free (G_SI = 1); its physical unit is kg/m^2
        (acceleration divided by G).
    Complex-output disclaimer:
        The output is the real-valued magnitude scale of a linear complex
        amplitude, not a real-valued time-domain gravitational field.
    Formulation distinction:
        Corresponds to the dipole/total formulation, and equals the sum of
        the bulk and wall terms.
        Formula: total(P) = bulk(P) + wall(P) = 8*pi * rho0 * xi * j1(x)/x

    Parameters
    ----------
    radius_m : float
        Radius of the spherical cavity in meters (positive).
    wavenumber_rad_m : float or array-like
        Wavenumber of the P-wave in rad/m (non-negative).
    density_kg_m3 : float
        Medium background mass density in kg/m^3 (positive).
    displacement_m : float
        Displacement amplitude of the P-wave in meters (positive).

    Returns
    -------
    np.ndarray
        P-wave total acceleration amplitude array of shape matching wavenumber_rad_m.
    """
    r = validate_scalar_input(radius_m, "radius_m")
    rho = validate_scalar_input(density_kg_m3, "density_kg_m3")
    xi = validate_scalar_input(displacement_m, "displacement_m")
    k = validate_array_input(wavenumber_rad_m, "wavenumber_rad_m")

    x = k * r
    geom = _j1_over_x(x)
    out = 8.0 * np.pi * rho * xi * geom

    out.setflags(write=False)
    return out


def spherical_shell_dipole_factor(
    k_r: float | list | tuple | np.ndarray,
) -> np.ndarray:
    """Calculate the dimensionless spherical shell dipole factor, -j1(x)/x.

    G-free convention:
        The output is dimensionless.
    Complex-output disclaimer:
        The output is a real-valued dimensionless coefficient.
    Formulation distinction:
        Geometric factor of the spherical-shell dipole/total formulation.
        Relation: total(P) = -8*pi * rho0 * xi * shell_dipole_factor(k * R)

    Parameters
    ----------
    k_r : float or array-like
        Dimensionless wavenumber * radius product (non-negative).

    Returns
    -------
    np.ndarray
        The dipole factor values of shape matching k_r.
    """
    x = validate_array_input(k_r, "k_r")
    geom = _j1_over_x(x)
    out = -geom

    out.setflags(write=False)
    return out
