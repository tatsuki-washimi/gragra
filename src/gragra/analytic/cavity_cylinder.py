"""Filled (mass-solid) right-circular-cylinder on-axis gravitational field.

Published basis for the closed form: Blakely, R. J. (1995), *Potential
Theory in Gravity and Magnetic Applications*, Cambridge University Press --
the standard textbook closed form for the on-axis attraction of a uniform
cylinder.

The implementation itself is gragra's own derivation, verified with a
computer algebra system and numerically against the far-field point-mass
asymptote and the interior/exterior symmetry properties (see
``tests/test_analytic_cavity_cylinder.py``). Nothing from the textbook is
reproduced here; it is cited only as the standard reference for the result.

Physics conventions and formulation notes:
- G-free convention: the same convention as `cavity_sphere.py`. The output
  is G-free (G_SI = 1) and its physical unit is kg/m^2 (acceleration
  divided by G).
- The output is real (a static field): as in `cavity_cuboid.py`, there is
  no wavenumber or displacement amplitude here, so the complex-amplitude
  disclaimer does not apply.
- Formulation distinction (the ``filled`` term): the same
  "internal field of the notionally filled body" formulation as
  `cavity_cuboid.cuboid_filled_internal_field` -- neither `cavity_sphere.py`'s
  ``bulk`` nor its ``wall``, but the input to the inversion principle. See
  `cavity_cuboid.py`'s module docstring for the details.
- Scope is **on-axis only**: generalizing to an arbitrary (off-axis) point
  requires elliptic integrals and is out of scope for this module. The
  caller is responsible for transforming coordinates and passing on-axis
  coordinates only.
- Sign convention: follows gragra's ``Delta = r'_source - r_target``
  (``docs/architecture/kernel_contract.md`` Sec. 3). The cylinder's axis is
  fixed to z and its center to the origin (there is no ``axis`` orientation
  argument as in `geometry.Cylinder` -- reorienting is the caller's
  coordinate-transform responsibility), and the result satisfies the
  far-field point-mass limit ``a_z -> -rho*V/z_t^2`` on the ``z_t > 0``
  side (verified symbolically and numerically in this module's tests).
"""

import numpy as np

from gragra.analytic._validation import validate_coordinate_input, validate_scalar_input
from gragra.array_types import FloatArray


def cylinder_filled_axis_field(
    radius_m: float,
    height_m: float,
    z_m: float | list | tuple | np.ndarray,
    density_kg_m3: float,
) -> FloatArray:
    """Calculate the G-free static on-axis gravitational field of a filled cylinder.

    G-free convention:
        The output is G-free (G_SI = 1); its physical unit is kg/m^2
        (acceleration divided by G).
    Real output (a static field):
        There is no wavenumber or displacement amplitude here, so the
        complex-amplitude disclaimer does not apply.
    Formulation distinction:
        The ``filled`` term -- the same formulation as
        `cavity_cuboid.cuboid_filled_internal_field` (see the module
        docstring).
    Numerical defence:
        Unlike `cavity_cuboid.py`'s ``_corner_term``, this function carries
        no guards for removable singularities on faces, edges or vertices:
        ``sqrt(r^2+u^2)`` is identically positive as long as ``r > 0`` (a
        valid radius), so neither a division by zero nor a 0/0
        indeterminate form of a log/arctan can arise in principle
        (restricting to the axis removes the multi-variable cancellation
        singularities that the cuboid's 8-corner sum has). In the far field
        (``|u| >> r``), however, evaluating ``f(u) = |u| - sqrt(r^2+u^2)``
        naively loses precision by subtracting two nearly equal large
        values, so the algebraically equivalent rationalized form
        ``f(u) = -r^2 / (|u| + sqrt(r^2+u^2))`` is used instead (its
        denominator ``|u| + sqrt(r^2+u^2)`` is identically positive, so
        there is no division by zero).

    Parameters
    ----------
    radius_m : float
        Cylinder radius in meters (positive).
    height_m : float
        Cylinder height in meters (positive). The cylinder is centered at
        the origin, spanning z in [-height_m/2, height_m/2].
    z_m : float or array-like, shape (M,)
        On-axis observation point z-coordinates in meters, relative to the
        cylinder center. May be negative, zero, or exceed the cylinder
        extent (interior/boundary/exterior all valid).
    density_kg_m3 : float
        Mass density of the filled cylinder in kg/m^3 (positive).

    Returns
    -------
    FloatArray
        G-free on-axis acceleration z-component, shape matching `z_m`
        (scalar input -> shape (), array input -> matching shape), float64,
        read-only.
    """
    r = validate_scalar_input(radius_m, "radius_m")
    h = validate_scalar_input(height_m, "height_m")
    rho = validate_scalar_input(density_kg_m3, "density_kg_m3")
    z_t = validate_coordinate_input(z_m, "z_m")

    def f(u: np.ndarray) -> np.ndarray:
        # Rationalized form of |u| - sqrt(r^2+u^2): avoids catastrophic
        # cancellation of two nearly-equal large values when |u| >> r
        # (far field). Algebraically identical (see module docstring).
        return -r * r / (np.abs(u) + np.sqrt(r * r + u * u))

    u1 = -h / 2.0 - z_t  # lower-end - target
    u2 = h / 2.0 - z_t  # upper-end - target
    out = 2.0 * np.pi * rho * (f(u2) - f(u1))

    out.setflags(write=False)
    return out
