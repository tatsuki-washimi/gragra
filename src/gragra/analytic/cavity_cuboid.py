"""Filled (mass-solid) rectangular-prism internal gravitational field.

Published basis for the closed form: Nagy, D. (1966), "The gravitational
attraction of a right rectangular prism", *Geophysics* 31(2), 362-371; and
Nagy, D., Papp, G. & Benedek, J. (2000), "The gravitational potential and
its derivatives for the prism", *J. Geodesy* 74(7-8), 552-560, together with
its Erratum (*J. Geodesy* 76(8), 475, 2002), which corrects the ``atan``
branch selection at interior points.

The implementation itself is gragra's own derivation: the antiderivative
below was derived and verified independently with a computer algebra system
(the identity ``d^2 F2/dx dy == 1/R``) and checked numerically against the
Poisson divergence, the far-field point-mass asymptote, and the symmetry
properties -- see ``tests/test_analytic_cavity_cuboid.py``. The published
works above are cited as the standard reference for this closed form and
for the ``atan``-branch correction; no text, figure, or equation from them
is reproduced here.

Physics conventions and formulation notes:
- G-free convention: the same convention as `cavity_sphere.py`. The output
  is G-free (G_SI = 1) and its physical unit is kg/m^2 (acceleration
  divided by G).
- The output is real (a static field): ``points_m`` here holds coordinates,
  not wavenumbers or displacement amplitudes, so `cavity_sphere.py`'s
  complex-amplitude disclaimer does not apply -- the output is a literally
  real-valued static gravitational acceleration (G-free).
- Formulation distinction (the ``filled`` term):
  `cuboid_filled_internal_field` returns the internal gravitational field of
  the uniform-density rectangular prism obtained by notionally **filling**
  the cavity with the surrounding medium. That is neither `cavity_sphere.py`'s
  ``bulk`` (the wavenumber-dependent density-perturbation term from
  compression of the medium) nor its ``wall`` (the surface mass-sheet term
  from displacement of the cavity wall): it is a **third** formulation, and
  it is the input to the *inversion principle*.

  The inversion principle used here is gragra's own derivation, not a
  transcription of any external result: the cavern-wall coupling is
  obtained by letting the **gradient** of the filled-body internal field
  act directly on the displacement,

      delta_a_wall,i = (d a_filled,i / d x_j) * xi_j

  with **no** extra sign inversion -- the gradient is already negative, so
  inserting a further sign flip would apply the negative sign twice and
  invert the result. For a sphere this gives
  ``d a_filled/d x = -(4*pi/3)*rho*I``, hence
  ``delta_a = -(4*pi/3)*rho*xi``, agreeing with
  ``cavity_coupling.sca_wall_acceleration("sphere", "arbitrary", ...)``.
  That agreement is verified numerically at the cube center by
  ``tests/test_analytic_cavity_cuboid.py::
  test_cube_center_gradient_matches_table1_inversion``. Applying the same
  principle to shapes outside that table is demonstrated in
  ``examples/13_sca_lower_limit_gate.py``.
- Sign convention: follows gragra's ``Delta = r'_source - r_target``
  (``docs/architecture/kernel_contract.md`` Sec. 3) directly. Corner
  coordinates are computed as ``(source_corner - target)``, and the result
  satisfies the far-field point-mass limit ``a -> rho*V*Delta/r^3``
  (verified symbolically and numerically in this module's tests).
"""

import numpy as np

from gragra._arrays import as_xyz_table, as_xyz_vector
from gragra.analytic._validation import validate_scalar_input
from gragra.array_types import FloatArray


def _corner_term(x: float, y: float, z: float) -> float:
    """One (x, y, z) corner evaluation of the Nagy-type antiderivative F2.

    F2(x,y,z) = x*ln(y+R) + y*ln(x+R) - z*atan(x*y/(z*R)), R = sqrt(x^2+y^2+z^2)
    (verified with a computer algebra system: d^2 F2/dx dy == 1/R
    identically).

    Guards the removable singularities at faces/edges/vertices (x==0 and/or
    y==0 and/or z==0, or the exact corner r==0): each guarded sub-term's
    mathematical limit is 0 (finite 0*log/0*atan removed rather than
    evaluated as a naive IEEE 0*inf -> nan). Uses the naive `atan` branch
    (not `atan2`), the branch confirmed correct for interior points by the
    antiderivative identity above -- `atan2` has a +-pi jump at
    z<0, x*y~=0 that breaks it (the Nagy/Papp/Benedek 2000 Erratum 2002
    issue).
    """
    r = np.sqrt(x * x + y * y + z * z)
    if r == 0.0:
        return 0.0
    total = 0.0
    if x != 0.0:
        total += x * np.log(y + r)
    if y != 0.0:
        total += y * np.log(x + r)
    if z != 0.0:
        total -= z * np.arctan(x * y / (z * r))
    return total


def _corner_sum(
    xs: tuple[float, float], ys: tuple[float, float], zs: tuple[float, float]
) -> float:
    """Sum (-1)^(i+j+k) * F2(x_i, y_j, z_k) over the 8 box corners."""
    total = 0.0
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            for k, z in enumerate(zs):
                total += (-1.0) ** (i + j + k) * _corner_term(x, y, z)
    return total


def _single_point_field(half_lengths: np.ndarray, point: np.ndarray) -> np.ndarray:
    """G-free/density-free (a/rho) field at one target point, box centered at origin."""
    rel_lo = -half_lengths - point  # lower-corner - target, per axis
    rel_hi = half_lengths - point  # upper-corner - target, per axis
    xs = (rel_lo[0], rel_hi[0])
    ys = (rel_lo[1], rel_hi[1])
    zs = (rel_lo[2], rel_hi[2])
    # a_z uses (x,y,z) as-is; a_x/a_y are cyclic permutations (F2 treats its
    # 3rd argument specially via the atan term, so each output component
    # must place its own axis last).
    gz = _corner_sum(xs, ys, zs)
    gx = _corner_sum(ys, zs, xs)
    gy = _corner_sum(zs, xs, ys)
    return np.array([gx, gy, gz])


def cuboid_filled_internal_field(
    lengths_m: list | tuple | np.ndarray,
    points_m: list | tuple | np.ndarray,
    density_kg_m3: float,
) -> FloatArray:
    """Calculate the G-free static gravitational field of a filled uniform prism.

    G-free convention:
        The output is G-free (G_SI = 1); its physical unit is kg/m^2
        (acceleration divided by G).
    Real output (a static field):
        There is no wavenumber or displacement-amplitude here, so the
        complex-amplitude disclaimer does not apply.
    Formulation distinction:
        The ``filled`` term -- the internal field of the notionally filled
        body, which is the input to the inversion principle (see the module
        docstring). It differs from both `cavity_sphere.py`'s ``bulk`` and
        its ``wall``.

    Parameters
    ----------
    lengths_m : array-like, shape (3,)
        Full side lengths (Lx, Ly, Lz) of the prism in meters, all positive.
        The prism is centered at the origin of the coordinate system used by
        `points_m` (caller's responsibility to translate observation points
        relative to the prism center).
    points_m : array-like, shape (M, 3)
        Observation point coordinates in meters, relative to the prism
        center. May be negative or zero (interior/boundary/exterior points
        are all valid -- the closed form is finite everywhere, including on
        faces, edges, and at vertices; see the corner-term guards in
        `_corner_term`). Points strictly near (but not exactly on) a face,
        edge, or vertex are not specially guarded -- the guards only cover
        the exact-zero degenerate case -- so, as with any log/atan-based
        closed form, expect gradual precision loss (not a discontinuity)
        within a few floating-point ULPs of the boundary, well below the
        rtol used by this module's tests.
    density_kg_m3 : float
        Mass density of the filled prism in kg/m^3 (positive).

    Returns
    -------
    FloatArray
        G-free gravitational acceleration field of shape (M, 3), float64,
        read-only.
    """
    lengths = as_xyz_vector("lengths_m", lengths_m)
    if np.any(lengths <= 0.0):
        raise ValueError("lengths_m must be positive")
    points = as_xyz_table("points_m", points_m)
    rho = validate_scalar_input(density_kg_m3, "density_kg_m3")

    half_lengths = lengths / 2.0
    out = np.empty((points.shape[0], 3), dtype=np.float64)
    for m in range(points.shape[0]):
        out[m] = _single_point_field(half_lengths, points[m])
    out *= rho

    out.setflags(write=False)
    return out
