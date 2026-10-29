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
from __future__ import annotations

import numpy as np

from gragra._arrays import as_xyz_table, as_xyz_vector
from gragra.analytic._validation import (
    validate_nonnegative_scalar_input,
    validate_real_xyz_table,
    validate_real_xyz_vector,
    validate_scalar_input,
)
from gragra.analytic.values import GFreeGravityValues
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


def _safe_log_arg(w: float, u: float, v: float) -> float:
    """Evaluate w + sqrt(u^2 + v^2 + w^2) safely without catastrophic cancellation."""
    r = np.sqrt(u * u + v * v + w * w)
    if w >= 0.0:
        return float(w + r)
    num = u * u + v * v
    if num == 0.0:
        return 0.0
    return float(num / (r - w))


def _corner_term_phi(x: float, y: float, z: float) -> float:
    """One corner evaluation of the potential antiderivative F1 (Nagy et al. 2000)."""
    r = np.sqrt(x * x + y * y + z * z)
    if r == 0.0:
        return 0.0
    total = 0.0
    if x != 0.0 and y != 0.0:
        arg_z = _safe_log_arg(z, x, y)
        if arg_z > 0.0:
            total += x * y * np.log(arg_z)
    if y != 0.0 and z != 0.0:
        arg_x = _safe_log_arg(x, y, z)
        if arg_x > 0.0:
            total += y * z * np.log(arg_x)
    if z != 0.0 and x != 0.0:
        arg_y = _safe_log_arg(y, z, x)
        if arg_y > 0.0:
            total += z * x * np.log(arg_y)
    if x != 0.0:
        total -= 0.5 * x * x * np.arctan(y * z / (x * r))
    if y != 0.0:
        total -= 0.5 * y * y * np.arctan(z * x / (y * r))
    if z != 0.0:
        total -= 0.5 * z * z * np.arctan(x * y / (z * r))
    return total


def _corner_term_diag(u: float, v: float, w: float) -> float:
    """One corner evaluation of diagonal gradient term atan(v*w / (u*R))."""
    r = np.sqrt(u * u + v * v + w * w)
    if r == 0.0 or u == 0.0:
        return 0.0
    return float(np.arctan(v * w / (u * r)))


def _single_point_potential(half_lengths: np.ndarray, point: np.ndarray) -> float:
    rel_lo = -half_lengths - point
    rel_hi = half_lengths - point
    xs = (rel_lo[0], rel_hi[0])
    ys = (rel_lo[1], rel_hi[1])
    zs = (rel_lo[2], rel_hi[2])
    total = 0.0
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            for k, z in enumerate(zs):
                s = (-1.0) ** (i + j + k)
                total += s * _corner_term_phi(x, y, z)
    return total


def _diff_log_pair(u: float, v: float, w0: float, w1: float) -> float:
    """Evaluate ln(w0 + r0) - ln(w1 + r1) stably, including when u=v=0.

    Algebraic formulation:
    - If w1 <= 0 (both w0 < 0 and w1 <= 0):
      (w0 + r0) / (w1 + r1) = [rho_perp^2 / (r0 - w0)] / [rho_perp^2 / (r1 - w1)]
                            = (r1 - w1) / (r0 - w0)
      which is strictly positive and finite even when u = v = 0.
    - If w0 >= 0 (both w0 >= 0 and w1 > 0):
      (w0 + r0) / (w1 + r1) is strictly positive and finite.
    - If w0 < 0 < w1:
      w0 + r0 = rho_perp^2 / (r0 - w0), so
      (w0 + r0) / (w1 + r1) = rho_perp^2 / ((r0 - w0) * (w1 + r1)).
      For valid observation points (not on the material edge/vertex of the cuboid),
      rho_perp^2 > 0 strictly.
    """
    rho2 = u * u + v * v
    r0 = np.sqrt(rho2 + w0 * w0)
    r1 = np.sqrt(rho2 + w1 * w1)
    if w1 <= 0.0:
        return float(np.log((r1 - w1) / (r0 - w0)))
    elif w0 >= 0.0:
        return float(np.log((w0 + r0) / (w1 + r1)))
    else:
        denom = (r0 - w0) * (w1 + r1)
        if rho2 == 0.0 or denom == 0.0:
            return 0.0
        return float(np.log(rho2 / denom))


def _single_point_gradient(half_lengths: np.ndarray, point: np.ndarray) -> np.ndarray:
    rel_lo = -half_lengths - point
    rel_hi = half_lengths - point
    xs = (rel_lo[0], rel_hi[0])
    ys = (rel_lo[1], rel_hi[1])
    zs = (rel_lo[2], rel_hi[2])
    g = np.zeros((3, 3), dtype=np.float64)

    # Diagonals: 8-corner sum of arctan terms
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            for k, z in enumerate(zs):
                s = (-1.0) ** (i + j + k)
                g[0, 0] += s * _corner_term_diag(x, y, z)
                g[1, 1] += s * _corner_term_diag(y, z, x)
                g[2, 2] += s * _corner_term_diag(z, x, y)

    # Off-diagonals: evaluate pairwise log difference along paired axis
    # Gamma_xy: paired along z (k in {0, 1})
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            s = (-1.0) ** (i + j)
            g[0, 1] -= s * _diff_log_pair(x, y, zs[0], zs[1])

    # Gamma_xz: paired along y (j in {0, 1})
    for i, x in enumerate(xs):
        for k, z in enumerate(zs):
            s = (-1.0) ** (i + k)
            g[0, 2] -= s * _diff_log_pair(x, z, ys[0], ys[1])

    # Gamma_yz: paired along x (i in {0, 1})
    for j, y in enumerate(ys):
        for k, z in enumerate(zs):
            s = (-1.0) ** (j + k)
            g[1, 2] -= s * _diff_log_pair(y, z, xs[0], xs[1])

    g[1, 0] = g[0, 1]
    g[2, 0] = g[0, 2]
    g[2, 1] = g[1, 2]
    return g


def _is_on_cuboid_boundary(
    points_rel: np.ndarray,
    half_lengths: np.ndarray,
    atol: float | None = None,
) -> np.ndarray:
    """Detect whether target points lie on the cuboid material boundary.

    A point is on the material boundary if it lies within `atol` of any of the
    six bounding faces of the cuboid [-hx, hx] x [-hy, hy] x [-hz, hz].
    If `atol` is None, it defaults to 1e-12 * max(half_lengths), scaling
    with the cuboid dimensions.
    """
    if atol is None:
        atol = 1e-12 * float(np.max(half_lengths))
    abs_pts = np.abs(points_rel)
    within_box = np.all(abs_pts <= half_lengths + atol, axis=-1)
    on_any_face = np.any(np.isclose(abs_pts, half_lengths, atol=atol, rtol=0.0), axis=-1)
    return within_box & on_any_face


def _validate_cuboid_inputs(
    center_m: list | tuple | np.ndarray,
    lengths_m: list | tuple | np.ndarray,
    density_kg_m3: float,
    targets_m: list | tuple | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    center = validate_real_xyz_vector(center_m, "center_m")
    lengths = validate_real_xyz_vector(lengths_m, "lengths_m")
    if np.any(lengths <= 0.0):
        raise ValueError("lengths_m must be positive")
    rho = validate_nonnegative_scalar_input(density_kg_m3, "density_kg_m3")
    targets = validate_real_xyz_table(targets_m, "targets_m")
    return center, lengths, rho, targets


def cuboid_potential_g_free(
    center_m: list | tuple | np.ndarray,
    lengths_m: list | tuple | np.ndarray,
    density_kg_m3: float,
    targets_m: list | tuple | np.ndarray,
) -> FloatArray:
    """Calculate G-free gravitational potential of a uniform cuboid (rectangular prism).

    Physical unit: kg / m.
    """
    center, lengths, rho, targets = _validate_cuboid_inputs(
        center_m, lengths_m, density_kg_m3, targets_m
    )
    half_lengths = lengths / 2.0
    rel_pts = targets - center
    out = np.empty(targets.shape[0], dtype=np.float64)
    for m in range(targets.shape[0]):
        out[m] = rho * _single_point_potential(half_lengths, rel_pts[m])
    out.setflags(write=False)
    return out


def cuboid_acceleration_g_free(
    center_m: list | tuple | np.ndarray,
    lengths_m: list | tuple | np.ndarray,
    density_kg_m3: float,
    targets_m: list | tuple | np.ndarray,
) -> FloatArray:
    """Calculate G-free gravitational acceleration vector of a uniform cuboid.

    Physical unit: kg / m^2.
    """
    center, lengths, rho, targets = _validate_cuboid_inputs(
        center_m, lengths_m, density_kg_m3, targets_m
    )
    half_lengths = lengths / 2.0
    rel_pts = targets - center
    out = np.empty((targets.shape[0], 3), dtype=np.float64)
    for m in range(targets.shape[0]):
        out[m] = _single_point_field(half_lengths, rel_pts[m])
    out *= rho
    out.setflags(write=False)
    return out


def cuboid_gradient_g_free(
    center_m: list | tuple | np.ndarray,
    lengths_m: list | tuple | np.ndarray,
    density_kg_m3: float,
    targets_m: list | tuple | np.ndarray,
) -> FloatArray:
    """Calculate G-free gravity gradient tensor of a uniform cuboid.

    Physical unit: kg / m^3.
    Rejects points on the material boundary of the cuboid.
    """
    center, lengths, rho, targets = _validate_cuboid_inputs(
        center_m, lengths_m, density_kg_m3, targets_m
    )
    half_lengths = lengths / 2.0
    rel_pts = targets - center
    boundary_mask = _is_on_cuboid_boundary(rel_pts, half_lengths)
    if np.any(boundary_mask):
        first_idx = int(np.where(boundary_mask)[0][0])
        raise ValueError(
            f"target point at index {first_idx} ({targets[first_idx]}) lies on the cuboid boundary "
            "where gravity gradient is discontinuous or singular"
        )
    out = np.empty((targets.shape[0], 3, 3), dtype=np.float64)
    for m in range(targets.shape[0]):
        out[m] = rho * _single_point_gradient(half_lengths, rel_pts[m])
    out.setflags(write=False)
    return out


def cuboid_observables_g_free(
    center_m: list | tuple | np.ndarray,
    lengths_m: list | tuple | np.ndarray,
    density_kg_m3: float,
    targets_m: list | tuple | np.ndarray,
) -> GFreeGravityValues:
    """Calculate G-free potential, acceleration, and gradient of a uniform cuboid.

    Parameters
    ----------
    center_m : array-like, shape (3,)
        Center of the cuboid in meters.
    lengths_m : array-like, shape (3,)
        Full side lengths (Lx, Ly, Lz) in meters, all positive.
    density_kg_m3 : float
        Uniform mass density in kg/m^3 (positive).
    targets_m : array-like, shape (M, 3)
        Observation target points in meters. Points lying on the cuboid
        material boundary are rejected.

    Returns
    -------
    GFreeGravityValues
        Immutable container holding potential (M,), acceleration (M, 3),
        and gradient (M, 3, 3) in G-free units.
    """
    center, lengths, rho, targets = _validate_cuboid_inputs(
        center_m, lengths_m, density_kg_m3, targets_m
    )
    half_lengths = lengths / 2.0
    rel_pts = targets - center
    boundary_mask = _is_on_cuboid_boundary(rel_pts, half_lengths)
    if np.any(boundary_mask):
        first_idx = int(np.where(boundary_mask)[0][0])
        raise ValueError(
            f"target point at index {first_idx} ({targets[first_idx]}) lies on the cuboid boundary "
            "where gravity gradient is discontinuous or singular"
        )

    n_targets = targets.shape[0]
    pot = np.empty(n_targets, dtype=np.float64)
    acc = np.empty((n_targets, 3), dtype=np.float64)
    grad = np.empty((n_targets, 3, 3), dtype=np.float64)

    for m in range(n_targets):
        pt = rel_pts[m]
        pot[m] = rho * _single_point_potential(half_lengths, pt)
        acc[m] = rho * _single_point_field(half_lengths, pt)
        grad[m] = rho * _single_point_gradient(half_lengths, pt)

    return GFreeGravityValues(potential=pot, acceleration=acc, gradient=grad)

