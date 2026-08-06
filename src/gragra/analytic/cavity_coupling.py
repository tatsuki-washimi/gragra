"""SCA (small-cavern approximation) cavern-wall NN coupling factors.

Primary source: J. Harms, L. Naticchioni, E. Calloni, R. De Rosa, F. Ricci
and D. D'Urso, "A lower limit for Newtonian-noise models of the Einstein
Telescope", Eur. Phys. J. Plus 137, 687 (2022), Table 1 (p. 5 of 16).
https://doi.org/10.1140/epjp/s13360-022-02851-z

Attribution (required by CC BY 4.0 Sec. 3(a)(1)): (C) The Author(s) 2022 --
Jan Harms, Luca Naticchioni, Enrico Calloni, Rosario De Rosa, Fulvio Ricci,
Domenico D'Urso. That article is licensed under the Creative Commons
Attribution 4.0 International License (CC BY 4.0),
https://creativecommons.org/licenses/by/4.0/. This module reuses three
numeric coupling constants (2.8, 2.1, 1.3) and two closed forms from its
Table 1 / Eq. (7). **Changes were made**: the values are re-expressed as a
Python lookup table keyed by ``(cavern_shape, test_mass_location)``, the two
whole-space entries are computed as the exact closed form ``4*pi/3`` instead
of being quoted as decimals, and gragra's own G-free sign and unit
conventions are applied to the acceleration helper.

Physics conventions and formulation notes:
- G-free convention: the same convention as `cavity_sphere.py`. The output
  is G-free (G_SI = 1) and its physical unit is kg/m^2 (acceleration
  divided by G).
- Complex-output disclaimer: the output is real. The small-cavern
  approximation carries no wavenumber dependence (see below), so the notion
  of a complex amplitude does not arise in this module at all.
- Return type (a deliberate exception to the array conventions): the two
  functions in this module return a plain Python ``float``, not an
  ``np.ndarray``. Other modules under `analytic/` (e.g. `cavity_sphere.py`)
  accept array arguments such as a wavenumber array and therefore return an
  ndarray made read-only via ``out.setflags(write=False)``; the
  small-cavern closed forms here have no wavenumber or component axis at
  all (scalar inputs only), so there is no array to make read-only.
  Immutability itself is preserved regardless, because a Python ``float``
  is an immutable primitive.
- Formulation distinction: the Table 1 coupling factors correspond to the
  **cavern-wall term only** of Harms et al. 2022 Eq. (7) ("purely from the
  cavern-wall contribution"). That Eq. (7) deliberately sets the bulk term
  to zero in order to keep the Newtonian-noise lower limit conservative, so
  the values here correspond to the **wall** term of `cavity_sphere.py`'s
  ``bulk``/``wall``/``total`` (= bulk + wall) breakdown -- **not**
  ``total``.
- Small-cavern approximation (common to all four constants; the Table 1
  caption states they are "valid in the small-cavern approximation"): the
  cavity dimensions are assumed small compared with the seismic wavelength
  (equivalent, in `cavity_sphere.py`'s notation, to the limit of the
  dimensionless wavenumber ``x = k*R -> 0``), so that the displacement
  vector ``xi`` may be treated as uniform over the entire cavity wall. In
  that limit the acceleration's dependence on the relative geometry
  ``Delta = r'_source - r_target`` disappears and the acceleration
  direction is fixed by the displacement direction ``xi_hat`` alone -- a
  purely local relation (Harms et al. 2022 Sec. 2.4). The functions in this
  module are therefore **not** kernels that follow the ``Delta`` sign
  convention of `kernel_contract.md` Sec. 3 directly; the sign follows
  Eq. (7)'s ``delta_a_low = -4*pi*G*rho/3 * xi``, the same negative sign as
  the existing wall term `spherical_cavity_wall`.
- Reference point (Harms et al. 2022 Table 1, p. 5):
  - "Arbitrary" (sphere) / "3D center" (cube): the **volumetric center** of
    the cavity (the whole-space isolated-closed-surface case, to which the
    six existing conditions of `kernel_contract.md` Sec. 2.2 apply).
  - "2D center on floor" (cube floor-center, half sphere, elongated cuboid
    2:1:1): the horizontal (2D) center point of the cavity **floor** --
    note that this is *not* the cavity's volumetric center. It is a
    half-space geometry touching the free surface, and is covered by the
    floor/free-surface extension conditions (image-method equivalence)
    rather than by the whole-space conditions above.
"""

from typing import Literal

import numpy as np

from gragra.analytic._validation import validate_scalar_input

CavernShape = Literal["sphere", "cube", "half_sphere", "elongated_cuboid_2to1to1"]
TestMassLocation = Literal["arbitrary", "center_3d", "floor_center_2d"]

# Harms et al. 2022, Table 1 (p. 5 of 16). Keys are (cavern_shape,
# test_mass_location) pairs; only the four combinations tabulated in the
# source are valid (this is a lookup table, not a general product space).
#
# "sphere"/"arbitrary" and "cube"/"center_3d" are the whole-space closed-
# manifold case (kernel_contract.md Sec.2.2's existing six conditions);
# 4*pi/3 is an exact closed form (Eq.7), not a numerically-fitted value, so
# it is computed rather than hardcoded as a decimal.
#
# The remaining three are floor/free-surface cases; the source states these
# were "obtained analytically or by numerical integration" (Sec.2.4) and
# quotes them to 2 significant figures only -- no higher-precision value is
# available from the source, so these remain literal constants.
_SCA_WALL_COUPLING_TABLE: dict[tuple[CavernShape, TestMassLocation], float] = {
    ("sphere", "arbitrary"): 4.0 * np.pi / 3.0,
    ("cube", "center_3d"): 4.0 * np.pi / 3.0,
    ("cube", "floor_center_2d"): 2.8,
    ("half_sphere", "floor_center_2d"): 2.1,
    ("elongated_cuboid_2to1to1", "floor_center_2d"): 1.3,
}


def sca_wall_coupling_factor(
    cavern_shape: CavernShape, test_mass_location: TestMassLocation
) -> float:
    """Look up the dimensionless SCA cavern-wall coupling factor (Harms Table 1).

    G-free convention:
        The output is dimensionless.
    Complex-output disclaimer:
        The output is real. The return value is a plain Python ``float``,
        not an ``np.ndarray`` -- see the module docstring's "Return type"
        note.
    Formulation distinction:
        A coefficient of the wall term, matching `spherical_cavity_wall`;
        it does **not** correspond to ``total`` (bulk + wall). See the
        module docstring.

    Parameters
    ----------
    cavern_shape : {"sphere", "cube", "half_sphere", "elongated_cuboid_2to1to1"}
        Cavern shape per Harms et al. 2022 Table 1. ``"elongated_cuboid_2to1to1"``
        is a **finite** cuboid with side-length ratio 2:1:1 (elongated along
        the horizontal) -- not a non-finite/tunnel-like geometry.
    test_mass_location : {"arbitrary", "center_3d", "floor_center_2d"}
        Where the test mass sits relative to the cavern. ``"arbitrary"``
        (sphere only, by point symmetry) and ``"center_3d"`` (cube) are the
        whole-space volumetric-center case; ``"floor_center_2d"`` is the
        horizontal (2D) center of the cavern floor, a floor/free-surface
        geometry (see module docstring).

    Returns
    -------
    float
        Dimensionless coupling factor from Table 1.

    Raises
    ------
    ValueError
        If ``(cavern_shape, test_mass_location)`` is not one of the four
        combinations tabulated in Harms et al. 2022 Table 1.
    """
    key = (cavern_shape, test_mass_location)
    if key not in _SCA_WALL_COUPLING_TABLE:
        valid = sorted(
            f"{shape!r} + {loc!r}" for shape, loc in _SCA_WALL_COUPLING_TABLE
        )
        raise ValueError(
            f"No Harms et al. 2022 Table 1 entry for cavern_shape={cavern_shape!r}, "
            f"test_mass_location={test_mass_location!r}. Valid combinations: "
            f"{', '.join(valid)}"
        )
    return _SCA_WALL_COUPLING_TABLE[key]


def sca_wall_acceleration(
    cavern_shape: CavernShape,
    test_mass_location: TestMassLocation,
    density_kg_m3: float,
    displacement_m: float,
) -> float:
    """Calculate the G-free small-cavern-limit cavern-wall acceleration amplitude.

    G-free convention:
        The output is G-free (G_SI = 1); its physical unit is kg/m^2
        (acceleration divided by G).
    Complex-output disclaimer:
        The output is real (the small-cavern approximation has no
        wavenumber dependence). The return value is a plain Python
        ``float``, not an ``np.ndarray`` -- see the module docstring's
        "Return type" note.
    Formulation distinction:
        The cavern-wall term of Harms et al. 2022 Eq. (7) (wall only; the
        bulk term is deliberately excluded -- see the module docstring):
        ``a = -coupling_factor * density_kg_m3 * displacement_m``
        The spherical case ("sphere"/"arbitrary") agrees exactly with the
        ``x = k*R -> 0`` limit of `spherical_cavity_wall`
        (``-4*pi/3 * rho * xi == cavity_wall_geometric(0) * -4*pi * rho *
        xi``), as verified in
        ``tests/test_analytic_cavity_coupling.py``.

    Parameters
    ----------
    cavern_shape : {"sphere", "cube", "half_sphere", "elongated_cuboid_2to1to1"}
        Cavern shape per Harms et al. 2022 Table 1 (see
        `sca_wall_coupling_factor`).
    test_mass_location : {"arbitrary", "center_3d", "floor_center_2d"}
        Test-mass location per Harms et al. 2022 Table 1 (see
        `sca_wall_coupling_factor`).
    density_kg_m3 : float
        Medium background mass density in kg/m^3 (positive).
    displacement_m : float
        Ground displacement amplitude in meters (positive; direction is the
        caller's responsibility -- the small-cavern-limit acceleration is
        parallel to the displacement direction, Harms et al. 2022 Sec.2.4).

    Returns
    -------
    float
        G-free cavern-wall acceleration amplitude.
    """
    rho = validate_scalar_input(density_kg_m3, "density_kg_m3")
    xi = validate_scalar_input(displacement_m, "displacement_m")
    factor = sca_wall_coupling_factor(cavern_shape, test_mass_location)
    return -factor * rho * xi
