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
- The existing closed-form field functions are **on-axis only**: generalizing
  their antiderivatives to an arbitrary point requires elliptic integrals.
  `cylinder_observables_g_free` provides a separate opt-in numerical
  boundary-integral path for general points; it is not a closed-form elliptic
  integral implementation and still uses a z-aligned cylinder.
  The separate ``cylinder_acceleration_gradient_g_free`` pair API uses real
  elliptic integrals and analytic axis/far series, including proper rotation.
- Sign convention: follows gragra's ``Delta = r'_source - r_target``
  (``docs/architecture/kernel_contract.md`` Sec. 3). The cylinder's axis is
  fixed to z and its center to the origin (there is no ``axis`` orientation
  argument as in `geometry.primitives.Cylinder` -- reorienting is the caller's
  coordinate-transform responsibility), and the result satisfies the
  far-field point-mass limit ``a_z -> -rho*V/z_t^2`` on the ``z_t > 0``
  side (verified symbolically and numerically in this module's tests).
"""

from __future__ import annotations

from math import factorial

import numpy as np

from gragra._arrays import _check_no_bool
from gragra.analytic._validation import (
    validate_coordinate_input,
    validate_nonnegative_scalar_input,
    validate_real_xyz_table,
    validate_real_xyz_vector,
    validate_scalar_input,
)
from gragra.analytic.values import GFreeGravityValues
from gragra.array_types import FloatArray


def _cel(kc: float, p: float, a: float, b: float) -> np.longdouble:
    """Real Bulirsch cel, for kc > 0, p > 0 (no principal-value branch).

    Bartky duplication: Bulirsch (1969), as given by Derby & Olbert
    arXiv:0909.3880, Appendix A. This evaluates a special function, not
    a spatial quadrature. See docs/design/literature-c.md for its definition.
    """
    k, p, a, b = map(np.longdouble, (kc, p, a, b))
    if not (k > 0 and p > 0):
        raise ValueError("cel requires positive complementary modulus and p")
    p = np.sqrt(p)
    b /= p
    mean = np.longdouble(1)
    product = k
    for _ in range(64):
        previous_a = a
        a += b / p
        ratio = product / p
        b = 2 * (b + previous_a * ratio)
        p += ratio
        previous_mean = mean
        mean += k
        if abs(previous_mean - k) <= previous_mean * 1e-10:
            pi = np.arccos(np.longdouble(-1))
            return pi / 2 * (b + a * mean) / (mean * (mean + p))
        k = 2 * np.sqrt(product)
        product = k * mean
    raise ArithmeticError("cel duplication did not converge")


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
        there is no division by zero). Beyond 64 times the enclosing sphere
        radius, a degree-8 solid-harmonic expansion avoids endpoint
        subtraction and is evaluated with scaled products.

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

    flat_z = z_t.reshape(-1)
    far = _cylinder_axis_far_field_mask(flat_z, r, h)
    out = np.empty_like(flat_z)

    def f(u: np.ndarray) -> np.ndarray:
        # Rationalized form of |u| - sqrt(r^2+u^2): avoids catastrophic
        # cancellation of two nearly-equal large values when |u| >> r
        # (far field). Algebraically identical (see module docstring).
        return -r * r / (np.abs(u) + np.sqrt(r * r + u * u))

    ordinary = ~far
    if np.any(ordinary):
        u1 = -h / 2.0 - flat_z[ordinary]  # lower-end - target
        u2 = h / 2.0 - flat_z[ordinary]  # upper-end - target
        out[ordinary] = 2.0 * np.pi * rho * (f(u2) - f(u1))
    if np.any(far):
        out[far] = _cylinder_axis_far_field_values(flat_z[far], r, h, rho)[1]

    out = out.reshape(z_t.shape)
    out.setflags(write=False)
    return out


def _cylinder_potential_antiderivative(u: np.ndarray, r: float) -> np.ndarray:
    """Odd, zero-at-origin primitive of sqrt(r²+u²)-|u|."""
    r_hyp = np.hypot(r, u)
    abs_u = np.abs(u)
    term1 = 0.5 * u * (r * r / (abs_u + r_hyp))
    term2 = 0.5 * r * r * np.arcsinh(u / r)
    return term1 + term2


def _cylinder_potential_interval(z: np.ndarray, r: float, h: float) -> np.ndarray:
    """Integral of sqrt(r²+u²)-|u| across the source height.

    An asinh primitive avoids log(1+tiny) at the center.  Outside the source
    height, even that primitive loses digits when its endpoints are close.
    Evaluate its difference directly using log1p/expm1, retaining h as the
    interval width even when z +/- h/2 rounds to the same float64 number.
    """
    distance = np.abs(z)
    half = h / 2.0
    inside = distance < half
    integral = np.empty_like(z)
    integral[inside] = _cylinder_potential_antiderivative(
        half - distance[inside], r
    ) + _cylinder_potential_antiderivative(half + distance[inside], r)

    a = distance[~inside] - half
    b = distance[~inside] + half
    sa, sb = np.hypot(r, a), np.hypot(r, b)
    # D = asinh(b/r)-asinh(a/r), with b-a=h evaluated without
    # subtracting endpoints.  b/(sb+b)-a/(sa+a) equals
    # (r/(sa+a))² * (1-exp(-2*D))/2.
    difference = np.log1p(h * (1.0 + (a + b) / (sa + sb)) / (a + sa))
    integral[~inside] = (
        0.5
        * r
        * r
        * (difference - 0.5 * (r / (sa + a)) ** 2 * np.expm1(-2.0 * difference))
    )
    return integral


def _cylinder_df_du(u: np.ndarray, r: float) -> np.ndarray:
    """Derivative df/du of f(u) = |u| - sqrt(r^2 + u^2) = sgn(u) - u/sqrt(r^2+u^2)."""
    r_hyp = np.sqrt(r * r + u * u)
    abs_u = np.abs(u)
    return np.sign(u) * (r * r / (r_hyp * (abs_u + r_hyp)))


def _validate_cylinder_inputs(
    center_m: list | tuple | np.ndarray,
    radius_m: float,
    height_m: float,
    density_kg_m3: float,
    targets_m: list | tuple | np.ndarray,
) -> tuple[np.ndarray, float, float, float, np.ndarray]:
    center = validate_real_xyz_vector(center_m, "center_m")
    r = validate_scalar_input(radius_m, "radius_m")
    h = validate_scalar_input(height_m, "height_m")
    rho = validate_nonnegative_scalar_input(density_kg_m3, "density_kg_m3")
    targets = validate_real_xyz_table(targets_m, "targets_m")
    return center, r, h, rho, targets


def _cylinder_local_targets(
    center: np.ndarray, targets: np.ndarray, rotation: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Transform targets, retaining scaled coordinates when float64 overflows.

    Finite world coordinates can have a non-finite difference or rotated local
    component even when the far-field observables remain representable. Such
    rows use ``coordinate_scale * scaled_point`` without materializing that
    product. Ordinary finite rows retain the historical direct calculation.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        relative = targets - center
        points = relative @ rotation
    scaled_mask = ~np.all(np.isfinite(points), axis=1)
    coordinate_scales = np.zeros(len(targets), dtype=np.float64)
    scaled_points = np.zeros_like(points)
    for index in np.flatnonzero(scaled_mask):
        coordinate_scale = float(
            max(np.max(np.abs(targets[index])), np.max(np.abs(center)))
        )
        if coordinate_scale == 0.0:
            raise ArithmeticError("non-finite local target from zero coordinates")
        # Keep directly representable coordinate differences. Scaling each
        # endpoint before subtracting can erase a small but finite component
        # when another component overflows during subtraction.
        with np.errstate(over="ignore", invalid="ignore"):
            scaled_relative = relative[index] / coordinate_scale
        overflowed = ~np.isfinite(relative[index])
        if np.any(overflowed):
            scaled_relative[overflowed] = (
                targets[index, overflowed] / coordinate_scale
                - center[overflowed] / coordinate_scale
            )
        scaled_points[index] = scaled_relative @ rotation
        coordinate_scales[index] = coordinate_scale
        # This sentinel is never used for a scaled row; it prevents accidental
        # propagation of inf/nan into later vectorized operations.
        points[index] = 0.0
    return points, scaled_mask, coordinate_scales, scaled_points


def _verify_on_axis_targets(
    targets_rel: np.ndarray,
    radius: float,
    atol: float = 1e-12,
    *,
    scaled_mask: np.ndarray | None = None,
    coordinate_scales: np.ndarray | None = None,
    scaled_points: np.ndarray | None = None,
) -> None:
    scaled_mask = (
        np.zeros(len(targets_rel), dtype=bool) if scaled_mask is None else scaled_mask
    )
    off_axis = np.zeros(len(targets_rel), dtype=bool)
    ordinary = ~scaled_mask
    if np.any(ordinary):
        points = targets_rel[ordinary]
        radial_scale = np.maximum.reduce(
            (np.abs(points[:, 0]), np.abs(points[:, 1]), np.full(len(points), radius))
        )
        radial = np.hypot(points[:, 0] / radial_scale, points[:, 1] / radial_scale)
        limit = atol * (radius / radial_scale)
        off_axis[ordinary] = radial > limit
    if np.any(scaled_mask):
        assert coordinate_scales is not None and scaled_points is not None
        points = scaled_points[scaled_mask]
        scales = coordinate_scales[scaled_mask]
        radial = np.hypot(points[:, 0], points[:, 1])
        limit = atol * (radius / scales)
        off_axis[scaled_mask] = radial > limit
    if np.any(off_axis):
        first_idx = int(np.where(off_axis)[0][0])
        raise ValueError(
            f"target point at index {first_idx} is off-axis; "
            "cylinder axis observables require targets on the z-axis"
        )


def _cylinder_target_axis_mask(
    targets_rel: np.ndarray,
    radius: float,
    *,
    scaled_mask: np.ndarray | None = None,
    coordinate_scales: np.ndarray | None = None,
    scaled_points: np.ndarray | None = None,
) -> np.ndarray:
    """Classify axis points with a scaled radial comparison."""
    scaled_mask = (
        np.zeros(len(targets_rel), dtype=bool) if scaled_mask is None else scaled_mask
    )
    axis = np.zeros(len(targets_rel), dtype=bool)
    ordinary = ~scaled_mask
    if np.any(ordinary):
        points = targets_rel[ordinary]
        radial_scale = np.maximum.reduce(
            (np.abs(points[:, 0]), np.abs(points[:, 1]), np.full(len(points), radius))
        )
        radial = np.hypot(points[:, 0] / radial_scale, points[:, 1] / radial_scale)
        axis[ordinary] = radial <= 1e-12 * (radius / radial_scale)
    if np.any(scaled_mask):
        assert coordinate_scales is not None and scaled_points is not None
        points = scaled_points[scaled_mask]
        scales = coordinate_scales[scaled_mask]
        radial = np.hypot(points[:, 0], points[:, 1])
        axis[scaled_mask] = radial <= 1e-12 * (radius / scales)
    return axis


def _verify_not_on_end_faces(
    z_rel: np.ndarray,
    height: float,
    atol: float = 1e-12,
    *,
    scaled_mask: np.ndarray | None = None,
    coordinate_scales: np.ndarray | None = None,
    scaled_points: np.ndarray | None = None,
) -> None:
    half_h = height / 2.0
    scaled_mask = (
        np.zeros(len(z_rel), dtype=bool) if scaled_mask is None else scaled_mask
    )
    on_face = np.zeros(len(z_rel), dtype=bool)
    ordinary = ~scaled_mask
    on_face[ordinary] = np.isclose(
        np.abs(z_rel[ordinary]), half_h, atol=atol * height, rtol=0.0
    )
    if np.any(scaled_mask):
        assert coordinate_scales is not None and scaled_points is not None
        scales = coordinate_scales[scaled_mask]
        z_scaled = np.abs(scaled_points[scaled_mask, 2])
        half_scaled = half_h / scales
        atol_scaled = atol * (height / scales)
        on_face[scaled_mask] = np.abs(z_scaled - half_scaled) <= atol_scaled
    if np.any(on_face):
        first_idx = int(np.where(on_face)[0][0])
        raise ValueError(
            f"target point at index {first_idx} (z={z_rel[first_idx]}) lies "
            "on the cylinder end face "
            "where gravity gradient is undefined or discontinuous"
        )


def cylinder_potential_axis_g_free(
    center_m: list | tuple | np.ndarray,
    radius_m: float,
    height_m: float,
    density_kg_m3: float,
    targets_m: list | tuple | np.ndarray,
) -> FloatArray:
    """Calculate G-free on-axis gravitational potential of a uniform cylinder.

    Physical unit: kg / m.
    """
    center, r, h, rho, targets = _validate_cylinder_inputs(
        center_m, radius_m, height_m, density_kg_m3, targets_m
    )
    rel_pts, scaled_mask, coordinate_scales, scaled_points = _cylinder_local_targets(
        center, targets, np.eye(3)
    )
    _verify_on_axis_targets(
        rel_pts,
        r,
        scaled_mask=scaled_mask,
        coordinate_scales=coordinate_scales,
        scaled_points=scaled_points,
    )

    z_t = rel_pts[:, 2]
    far = _cylinder_axis_far_field_mask(z_t, r, h)
    for index in np.flatnonzero(scaled_mask):
        far[index] = _cylinder_is_far_field_scaled(
            coordinate_scales[index], scaled_points[index], r, h
        )
    if np.any(scaled_mask & ~far):
        raise OverflowError(
            "non-far axial target exceeds float64 local-coordinate range"
        )
    out = np.empty_like(z_t)
    ordinary = ~far
    if np.any(ordinary):
        out[ordinary] = (
            -2.0 * np.pi * rho * _cylinder_potential_interval(z_t[ordinary], r, h)
        )
    if np.any(far):
        ordinary_far = far & ~scaled_mask
        if np.any(ordinary_far):
            out[ordinary_far] = _cylinder_axis_far_field_values(
                z_t[ordinary_far], r, h, rho
            )[0]
        for index in np.flatnonzero(scaled_mask):
            out[index] = _cylinder_far_field_values_scaled(
                r, h, rho, coordinate_scales[index], scaled_points[index]
            )[0]

    out.setflags(write=False)
    return out


def cylinder_gradient_axis_g_free(
    center_m: list | tuple | np.ndarray,
    radius_m: float,
    height_m: float,
    density_kg_m3: float,
    targets_m: list | tuple | np.ndarray,
) -> FloatArray:
    """Calculate G-free on-axis 3x3 gravity gradient tensor of a uniform cylinder.

    Physical unit: kg / m^3.
    Rejects end face points z = ±h/2.
    """
    center, r, h, rho, targets = _validate_cylinder_inputs(
        center_m, radius_m, height_m, density_kg_m3, targets_m
    )
    rel_pts, scaled_mask, coordinate_scales, scaled_points = _cylinder_local_targets(
        center, targets, np.eye(3)
    )
    _verify_on_axis_targets(
        rel_pts,
        r,
        scaled_mask=scaled_mask,
        coordinate_scales=coordinate_scales,
        scaled_points=scaled_points,
    )

    z_t = rel_pts[:, 2]
    _verify_not_on_end_faces(
        z_t,
        h,
        scaled_mask=scaled_mask,
        coordinate_scales=coordinate_scales,
        scaled_points=scaled_points,
    )

    far = _cylinder_axis_far_field_mask(z_t, r, h)
    for index in np.flatnonzero(scaled_mask):
        far[index] = _cylinder_is_far_field_scaled(
            coordinate_scales[index], scaled_points[index], r, h
        )
    if np.any(scaled_mask & ~far):
        raise OverflowError(
            "non-far axial target exceeds float64 local-coordinate range"
        )
    gzz = np.empty_like(z_t)
    ordinary = ~far
    if np.any(ordinary):
        u1 = -h / 2.0 - z_t[ordinary]
        u2 = h / 2.0 - z_t[ordinary]
        df1 = _cylinder_df_du(u1, r)
        df2 = _cylinder_df_du(u2, r)
        gzz[ordinary] = -2.0 * np.pi * rho * (df2 - df1)
    if np.any(far):
        ordinary_far = far & ~scaled_mask
        if np.any(ordinary_far):
            gzz[ordinary_far] = _cylinder_axis_far_field_values(
                z_t[ordinary_far], r, h, rho
            )[2]
        for index in np.flatnonzero(scaled_mask):
            gzz[index] = _cylinder_far_field_values_scaled(
                r, h, rho, coordinate_scales[index], scaled_points[index]
            )[2][2, 2]

    half_h = h / 2.0
    rho_local = np.where((np.abs(z_t) < half_h) & ~scaled_mask, rho, 0.0)
    gxx = (-4.0 * np.pi * rho_local - gzz) / 2.0

    n_targets = targets.shape[0]
    out = np.zeros((n_targets, 3, 3), dtype=np.float64)
    out[:, 0, 0] = gxx
    out[:, 1, 1] = gxx
    out[:, 2, 2] = gzz

    out.setflags(write=False)
    return out


def cylinder_axis_observables_g_free(
    center_m: list | tuple | np.ndarray,
    radius_m: float,
    height_m: float,
    density_kg_m3: float,
    targets_m: list | tuple | np.ndarray,
) -> GFreeGravityValues:
    """Calculate G-free on-axis potential, acceleration, and gradient.

    The source is a uniform cylinder.

    Parameters
    ----------
    center_m : array-like, shape (3,)
        Center of the cylinder in meters.
    radius_m : float
        Radius in meters (positive).
    height_m : float
        Height in meters (positive).
    density_kg_m3 : float
        Uniform mass density in kg/m^3 (positive).
    targets_m : array-like, shape (M, 3)
        On-axis observation target points. Off-axis points and points on
        the end faces z = ±h/2 are rejected.

    Returns
    -------
    GFreeGravityValues
        Immutable container holding potential (M,), acceleration (M, 3),
        and gradient (M, 3, 3) in G-free units.
    """
    center, r, h, rho, targets = _validate_cylinder_inputs(
        center_m, radius_m, height_m, density_kg_m3, targets_m
    )
    rel_pts, scaled_mask, coordinate_scales, scaled_points = _cylinder_local_targets(
        center, targets, np.eye(3)
    )
    _verify_on_axis_targets(
        rel_pts,
        r,
        scaled_mask=scaled_mask,
        coordinate_scales=coordinate_scales,
        scaled_points=scaled_points,
    )

    z_t = rel_pts[:, 2]
    _verify_not_on_end_faces(
        z_t,
        h,
        scaled_mask=scaled_mask,
        coordinate_scales=coordinate_scales,
        scaled_points=scaled_points,
    )

    far = _cylinder_axis_far_field_mask(z_t, r, h)
    for index in np.flatnonzero(scaled_mask):
        far[index] = _cylinder_is_far_field_scaled(
            coordinate_scales[index], scaled_points[index], r, h
        )
    if np.any(scaled_mask & ~far):
        raise OverflowError(
            "non-far axial target exceeds float64 local-coordinate range"
        )
    ordinary = ~far
    pot = np.empty_like(z_t)
    az = np.empty_like(z_t)
    gzz = np.empty_like(z_t)
    if np.any(ordinary):
        u1 = -h / 2.0 - z_t[ordinary]
        u2 = h / 2.0 - z_t[ordinary]
        pot[ordinary] = (
            -2.0 * np.pi * rho * _cylinder_potential_interval(z_t[ordinary], r, h)
        )

        def f_acc(u: np.ndarray) -> np.ndarray:
            return -r * r / (np.abs(u) + np.sqrt(r * r + u * u))

        az[ordinary] = 2.0 * np.pi * rho * (f_acc(u2) - f_acc(u1))
        df1 = _cylinder_df_du(u1, r)
        df2 = _cylinder_df_du(u2, r)
        gzz[ordinary] = -2.0 * np.pi * rho * (df2 - df1)
    if np.any(far):
        ordinary_far = far & ~scaled_mask
        if np.any(ordinary_far):
            far_potential, far_acceleration, far_gradient = (
                _cylinder_axis_far_field_values(z_t[ordinary_far], r, h, rho)
            )
            pot[ordinary_far] = far_potential
            az[ordinary_far] = far_acceleration
            gzz[ordinary_far] = far_gradient
        for index in np.flatnonzero(scaled_mask):
            pot[index], far_acceleration, far_gradient = (
                _cylinder_far_field_values_scaled(
                    r, h, rho, coordinate_scales[index], scaled_points[index]
                )
            )
            az[index] = far_acceleration[2]
            gzz[index] = far_gradient[2, 2]

    n_targets = targets.shape[0]
    acc = np.zeros((n_targets, 3), dtype=np.float64)
    acc[:, 2] = az

    # Gradient: diagonal tensor
    half_h = h / 2.0
    rho_local = np.where((np.abs(z_t) < half_h) & ~scaled_mask, rho, 0.0)
    gxx = (-4.0 * np.pi * rho_local - gzz) / 2.0

    grad = np.zeros((n_targets, 3, 3), dtype=np.float64)
    grad[:, 0, 0] = gxx
    grad[:, 1, 1] = gxx
    grad[:, 2, 2] = gzz

    return GFreeGravityValues(potential=pot, acceleration=acc, gradient=grad)


def _validate_cylinder_quadrature_order(order: int) -> int:
    if isinstance(order, (bool, np.bool_)):
        raise TypeError("quadrature_order must be an integer, not boolean")
    if isinstance(order, (complex, np.complexfloating)):
        raise TypeError("quadrature_order must be real integer")
    if not isinstance(order, (int, np.integer)):
        raise TypeError("quadrature_order must be an integer")
    value = int(order)
    if value < 4:
        raise ValueError("quadrature_order must be at least 4")
    if value > 128:
        raise ValueError("quadrature_order must be at most 128")
    return value


def _cylinder_boundary_quadrature(
    radius: float, height: float, order: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return points, outward normals, and area weights on a cylinder boundary.

    The side uses a tensor-product Gauss--Legendre/periodic rule in ``(z,phi)``.
    Each cap uses a polar-area rule in ``(r,phi)``.  The periodic midpoint rule
    avoids duplicate seam endpoints.  This is a numerical evaluation of the
    divergence-theorem boundary representation, not the Conway elliptic
    integral formula.
    """
    nodes, weights = np.polynomial.legendre.leggauss(order)
    azimuthal_order = max(64, 4 * order)
    azimuth = 2.0 * np.pi * np.arange(azimuthal_order) / azimuthal_order
    azimuth_weight = 2.0 * np.pi / azimuthal_order

    z, phi = np.meshgrid(0.5 * height * nodes, azimuth, indexing="ij")
    side = np.stack((radius * np.cos(phi), radius * np.sin(phi), z), axis=-1).reshape(
        -1, 3
    )
    side_normals = np.stack(
        (np.cos(phi), np.sin(phi), np.zeros_like(phi)), axis=-1
    ).reshape(-1, 3)
    side_weights = np.broadcast_to(
        (0.5 * height * weights)[:, None] * radius * azimuth_weight,
        z.shape,
    ).reshape(-1)

    radii = 0.5 * radius * (nodes + 1.0)
    radial, phi = np.meshgrid(radii, azimuth, indexing="ij")
    top = np.stack(
        (
            radial * np.cos(phi),
            radial * np.sin(phi),
            np.full_like(radial, height / 2.0),
        ),
        axis=-1,
    ).reshape(-1, 3)
    bottom = top.copy()
    bottom[:, 2] = -height / 2.0
    top_normals = np.zeros_like(top)
    top_normals[:, 2] = 1.0
    bottom_normals = -top_normals
    cap_weights = np.broadcast_to(
        (0.5 * radius * weights * radii)[:, None] * azimuth_weight,
        radial.shape,
    ).reshape(-1)

    return (
        np.concatenate((side, top, bottom)),
        np.concatenate((side_normals, top_normals, bottom_normals)),
        np.concatenate((side_weights, cap_weights, cap_weights)),
    )


def _reject_cylinder_boundary_targets(
    targets_rel: np.ndarray,
    radius: float,
    height: float,
    *,
    scaled_mask: np.ndarray | None = None,
    coordinate_scales: np.ndarray | None = None,
    scaled_points: np.ndarray | None = None,
) -> None:
    scale = max(radius, height, 1.0)
    atol = 1e-12 * scale
    scaled_mask = (
        np.zeros(len(targets_rel), dtype=bool) if scaled_mask is None else scaled_mask
    )
    radial_scaled = np.zeros(len(targets_rel), dtype=np.float64)
    radius_scaled = np.zeros(len(targets_rel), dtype=np.float64)
    atol_scaled = np.zeros(len(targets_rel), dtype=np.float64)
    ordinary = ~scaled_mask
    if np.any(ordinary):
        points = targets_rel[ordinary]
        radial_scale = np.maximum.reduce(
            (
                np.abs(points[:, 0]),
                np.abs(points[:, 1]),
                np.full(len(points), radius),
            )
        )
        radial_scaled[ordinary] = np.hypot(
            points[:, 0] / radial_scale, points[:, 1] / radial_scale
        )
        radius_scaled[ordinary] = radius / radial_scale
        atol_scaled[ordinary] = atol / radial_scale
    if np.any(scaled_mask):
        assert coordinate_scales is not None and scaled_points is not None
        points = scaled_points[scaled_mask]
        scales = coordinate_scales[scaled_mask]
        radial_scaled[scaled_mask] = np.hypot(points[:, 0], points[:, 1])
        radius_scaled[scaled_mask] = radius / scales
        atol_scaled[scaled_mask] = atol / scales

    inside_height = np.abs(targets_rel[:, 2]) <= height / 2.0 + atol
    inside_radius = radial_scaled <= radius_scaled + atol_scaled
    on_side = np.abs(radial_scaled - radius_scaled) <= atol_scaled
    on_side &= inside_height
    on_end = (
        np.isclose(np.abs(targets_rel[:, 2]), height / 2.0, atol=atol, rtol=0.0)
        & inside_radius
    )
    if np.any(scaled_mask):
        assert coordinate_scales is not None and scaled_points is not None
        scales = coordinate_scales[scaled_mask]
        z_scaled = np.abs(scaled_points[scaled_mask, 2])
        half_scaled = (height / 2.0) / scales
        vertical_atol_scaled = atol / scales
        inside_height[scaled_mask] = z_scaled <= half_scaled + vertical_atol_scaled
        on_side[scaled_mask] = (
            np.abs(radial_scaled[scaled_mask] - radius_scaled[scaled_mask])
            <= atol_scaled[scaled_mask]
        ) & inside_height[scaled_mask]
        on_end[scaled_mask] = (
            np.abs(z_scaled - half_scaled) <= vertical_atol_scaled
        ) & inside_radius[scaled_mask]
    boundary = on_side | on_end
    if np.any(boundary):
        first_index = int(np.flatnonzero(boundary)[0])
        raise ValueError(
            f"target point at index {first_index} lies on the physical "
            "cylinder boundary"
        )


def cylinder_observables_g_free(
    center_m: list | tuple | np.ndarray,
    radius_m: float,
    height_m: float,
    density_kg_m3: float,
    targets_m: list | tuple | np.ndarray,
    *,
    quadrature_order: int = 32,
) -> GFreeGravityValues:
    """Calculate G-free observables of a finite z-aligned solid cylinder.

    This opt-in general-point path numerically evaluates the
    divergence-theorem boundary representation, except beyond 64 times the
    enclosing sphere radius where it uses a degree-8 solid-harmonic expansion
    to avoid far-field cancellation. It is a numerical semi-analytic
    implementation, not Conway's closed elliptic-integral formula:
    ``quadrature_order`` controls the Gauss--Legendre order and the periodic
    azimuthal order is four times larger. The existing axis entry point is
    reused for axis targets.

    Targets on the physical cylindrical boundary are rejected because the
    boundary integrands are singular there and the gradient has distinct
    one-sided limits. Acceleration is continuous. With ``Delta = s - x``, the
    evaluated formulas are ``Phi = -rho/2 * integral(Delta dot n / |Delta|)``
    and ``a_i = -rho * integral(n_i / |Delta|)``.  The gradient is
    ``Gamma_ij = -rho * integral(Delta_i*n_j / |Delta|**3)``.  The trace
    therefore follows from the enclosed-target version of the divergence
    theorem; no local term is added to this boundary expression.

    This path remains a numerical diagnostic/reference implementation until
    Conway's elliptic formula is transcribed and independently verified. Its
    far-field branch uses the documented degree-10 tail bound.

    Parameters
    ----------
    center_m : array-like, shape (3,)
        Cylinder center in meters; its axis is parallel to world z.
    radius_m, height_m : float
        Positive radius and height in meters. The cylinder spans
        ``center_z +/- height_m/2``.
    density_kg_m3 : float
        Non-negative uniform density in kg/m^3.
    targets_m : array-like, shape (M, 3)
        World-coordinate observation points, excluding the material boundary.
    quadrature_order : int, keyword-only
        Gauss--Legendre order in ``[4, 128]`` for the side height and cap
        radius coordinates.

    Returns
    -------
    GFreeGravityValues
        Potential, acceleration, and gravity gradient in G-free units with
        shapes ``(M,)``, ``(M,3)``, and ``(M,3,3)``.
    """
    order = _validate_cylinder_quadrature_order(quadrature_order)
    center, radius, height, density, targets = _validate_cylinder_inputs(
        center_m, radius_m, height_m, density_kg_m3, targets_m
    )
    relative_targets, scaled_mask, coordinate_scales, scaled_points = (
        _cylinder_local_targets(center, targets, np.eye(3))
    )
    _reject_cylinder_boundary_targets(
        relative_targets,
        radius,
        height,
        scaled_mask=scaled_mask,
        coordinate_scales=coordinate_scales,
        scaled_points=scaled_points,
    )

    potential = np.zeros(len(targets), dtype=np.float64)
    acceleration = np.zeros((len(targets), 3), dtype=np.float64)
    gradient = np.zeros((len(targets), 3, 3), dtype=np.float64)

    axis = _cylinder_target_axis_mask(
        relative_targets,
        radius,
        scaled_mask=scaled_mask,
        coordinate_scales=coordinate_scales,
        scaled_points=scaled_points,
    )
    if np.any(axis):
        axis_values = cylinder_axis_observables_g_free(
            center, radius, height, density, targets[axis]
        )
        potential[axis] = axis_values.potential
        acceleration[axis] = axis_values.acceleration
        gradient[axis] = axis_values.gradient

    off_axis = ~axis
    boundary_data = None
    if np.any(off_axis) and density != 0.0:
        for index in np.flatnonzero(off_axis):
            if scaled_mask[index]:
                if not _cylinder_is_far_field_scaled(
                    coordinate_scales[index], scaled_points[index], radius, height
                ):
                    raise OverflowError(
                        "non-far target exceeds float64 local-coordinate range"
                    )
                (
                    potential[index],
                    acceleration[index],
                    gradient[index],
                ) = _cylinder_far_field_values_scaled(
                    radius,
                    height,
                    density,
                    coordinate_scales[index],
                    scaled_points[index],
                )
                continue
            if _cylinder_is_far_field_point(relative_targets[index], radius, height):
                (
                    potential[index],
                    acceleration[index],
                    gradient[index],
                ) = _cylinder_far_field_values(
                    radius, height, density, relative_targets[index]
                )
                continue

            if boundary_data is None:
                boundary_data = _cylinder_boundary_quadrature(radius, height, order)
            boundary_points, boundary_normals, boundary_weights = boundary_data
            delta = boundary_points - relative_targets[index]
            inverse_distance = 1.0 / np.linalg.norm(delta, axis=1)
            potential[index] = (
                -0.5
                * density
                * np.dot(
                    boundary_weights,
                    np.einsum("ni,ni->n", delta, boundary_normals) * inverse_distance,
                )
            )
            acceleration[index] = -density * np.einsum(
                "n,ni,n->i",
                boundary_weights,
                boundary_normals,
                inverse_distance,
            )
            gradient[index] = -density * np.einsum(
                "n,ni,nj,n->ij",
                boundary_weights,
                delta,
                boundary_normals,
                inverse_distance**3,
            )

    return GFreeGravityValues(potential, acceleration, gradient)


def _cylinder_end_fields(radius: float, r: float, u: float) -> np.ndarray:
    """Radial primitive, cap potential, mixed derivative, solid angle.

    ``u = z_end - z_target``. Radial primitive is Dubetsky (2019),
    Eq. A22, with this opposite axial-coordinate convention. The cap
    and derivative expressions are derived in literature-c.md.
    """
    radius, r, u = map(np.longdouble, (radius, r, u))
    d = np.hypot(radius + r, u)
    kc = np.hypot(radius - r, u) / d
    p = ((radius - r) / (radius + r)) ** 2
    k = _cel(kc, 1, 1, 1)
    e = _cel(kc, 1, 1, kc * kc)
    pi3 = _cel(kc, p, 1, 1) if p > 0 else np.longdouble(0)
    radial = (
        -u
        / (r * d)
        * (
            (u * u + 2 * r * r + 2 * radius * radius) * k
            - d * d * e
            - (radius - r) ** 2 * pi3
        )
    )
    pi = np.arccos(np.longdouble(-1))
    step = (1 + np.sign(radius - r)) / 2
    cap = (
        2
        * (
            d * e
            + (radius * radius - r * r) / d * k
            + (radius - r) / (radius + r) * u * u / d * pi3
        )
        - 2 * pi * abs(u) * step
    )
    mixed = 2 / r * ((radius * radius + r * r + u * u) / d * k - d * e)
    omega = 2 * pi * np.sign(u) * step - 2 * u / d * (
        k + (radius - r) / (radius + r) * pi3
    )
    return np.array([radial, cap, mixed, omega], dtype=np.longdouble)


def _cylinder_near_axis(
    radius: float, height: float, r: float, z: float, density: float
) -> tuple[float, float, float, float]:
    """Axis Taylor series through r^5; retains radial and mixed components.

    Poisson/Laplace recurrences and endpoint derivatives are CAS checked in
    tests/oracles/cylinder_elliptic_derivation.py. Used only for r/R <= 1e-3.
    """
    u = np.array([-height / 2 - z, height / 2 - z], dtype=np.longdouble)
    radius = np.longdouble(radius)
    d = np.hypot(radius, u)
    pi = np.arccos(np.longdouble(-1))
    derivatives = [
        -(radius**2) / (abs(u) + d),
        u / d - np.sign(u),
        -(radius**2) / d**3,
        -3 * radius**2 * u / d**5,
        3 * radius**2 * (radius**2 - 4 * u**2) / d**7,
        15 * radius**2 * u * (3 * radius**2 - 4 * u**2) / d**9,
    ]
    a0, a1, a2, a3, a4, a5 = [2 * pi * density * (v[1] - v[0]) for v in derivatives]
    local = density if abs(z) < height / 2 else 0.0
    ar = (-2 * pi * local - a1 / 2) * r + a3 * r**3 / 16 - a5 * r**5 / 384
    az = a0 - a2 * r * r / 4 + a4 * r**4 / 64
    rz = -a2 * r / 2 + a4 * r**3 / 16
    zz = a1 - a3 * r * r / 4 + a5 * r**4 / 64
    return ar, az, rz, zz


def _scaled_product_quotient(
    numerator_factors: tuple[float | np.ndarray, ...],
    denominator_factors: tuple[float | np.ndarray, ...],
) -> np.ndarray:
    """Evaluate a finite product/quotient without overflowing intermediates."""
    factors = tuple(
        np.asarray(value, dtype=np.float64)
        for value in (*numerator_factors, *denominator_factors)
    )
    shape = np.broadcast_shapes(*(value.shape for value in factors))
    mantissa = np.ones(shape, dtype=np.float64)
    exponent = np.zeros(shape, dtype=np.int32)

    for value in factors[: len(numerator_factors)]:
        part, power = np.frexp(value)
        mantissa *= part
        exponent += power
        mantissa, adjustment = np.frexp(mantissa)
        exponent += adjustment

    for value in factors[len(numerator_factors) :]:
        part, power = np.frexp(value)
        mantissa /= part
        exponent -= power
        mantissa, adjustment = np.frexp(mantissa)
        exponent += adjustment

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return np.ldexp(mantissa, exponent)


def _scaled_vector_norm(point: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Return a scaled Euclidean norm and unit vector without squaring coordinates."""
    vector = np.asarray(point, dtype=np.float64)
    scale = float(np.max(np.abs(vector)))
    if scale == 0.0:
        return 0.0, 0.0, np.zeros(3, dtype=np.float64)
    scaled = vector / scale
    norm_scaled = float(np.hypot(np.hypot(scaled[0], scaled[1]), scaled[2]))
    return scale, norm_scaled, scaled / norm_scaled


def _cylinder_is_far_field_point(
    point: np.ndarray, radius: float, height: float
) -> bool:
    """Apply the documented ``D >= 64*hypot(R, H/2)`` gate dimensionlessly."""
    point = np.asarray(point, dtype=np.float64)
    scale = float(np.max(np.abs(point)))
    if scale == 0.0:
        return False

    return _cylinder_is_far_field_scaled(scale, point / scale, radius, height)


def _cylinder_is_far_field_scaled(
    coordinate_scale: float,
    scaled_point: np.ndarray,
    radius: float,
    height: float,
) -> bool:
    """Apply the far-field gate to ``coordinate_scale * scaled_point``."""
    norm_scaled = float(
        np.hypot(np.hypot(scaled_point[0], scaled_point[1]), scaled_point[2])
    )
    if coordinate_scale == 0.0 or norm_scaled == 0.0:
        return False

    relative_limit = norm_scaled / 64.0
    dimensional_limit = coordinate_scale * relative_limit
    half_height = height / 2.0
    if radius > dimensional_limit or half_height > dimensional_limit:
        return False

    radius_ratio = (radius / coordinate_scale) / norm_scaled
    half_height_ratio = (half_height / coordinate_scale) / norm_scaled
    return np.hypot(radius_ratio, half_height_ratio) <= 1.0 / 64.0


def _cylinder_axis_far_field_mask(
    z: np.ndarray, radius: float, height: float
) -> np.ndarray:
    """Select axial targets using the shared dimensionless solid-harmonic gate."""
    values = np.asarray(z, dtype=np.float64)
    return np.fromiter(
        (
            _cylinder_is_far_field_point(np.array([0.0, 0.0, value]), radius, height)
            for value in values.reshape(-1)
        ),
        dtype=np.bool_,
        count=values.size,
    ).reshape(values.shape)


def _cylinder_far_field_values_scaled(
    radius: float,
    height: float,
    density: float,
    coordinate_scale: float,
    scaled_point: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate the degree-8 expansion without materializing local coordinates.

    ``coordinate_scale * scaled_point`` is the target vector. The
    dimensionless gate is ``hypot(R/D, H/(2D)) <= 1/64``. Exact volume moments
    avoid endpoint cancellation; scaled products avoid forming ``D`` or
    ``D**n`` when either is outside binary64 range. The degree-10-and-higher
    remainder is bounded in ``docs/design/literature-c.md``.
    """
    norm_scaled = float(
        np.hypot(np.hypot(scaled_point[0], scaled_point[1]), scaled_point[2])
    )
    if coordinate_scale == 0.0 or norm_scaled == 0.0:
        raise ValueError("far-field evaluation requires a nonzero target vector")

    direction = scaled_point / norm_scaled
    radius_ratio = (radius / coordinate_scale) / norm_scaled
    half_height_ratio = (height / 2.0 / coordinate_scale) / norm_scaled
    ez = np.array([0.0, 0.0, 1.0])
    potential = 0.0
    acc = np.zeros(3, dtype=np.float64)
    grad = np.zeros((3, 3), dtype=np.float64)
    distance_factors = (coordinate_scale, norm_scaled)
    pi = np.pi

    for degree in range(0, 10, 2):
        moment_scaled = sum(
            (-1) ** j
            * factorial(degree)
            / (factorial(degree - 2 * j) * factorial(j) ** 2 * 4**j)
            * radius_ratio ** (2 * j)
            * half_height_ratio ** (degree - 2 * j)
            / ((j + 1) * (degree - 2 * j + 1))
            for j in range(degree // 2 + 1)
        )
        poly = np.polynomial.legendre.Legendre.basis(degree)
        p, dp, ddp = (
            poly(direction[2]),
            poly.deriv()(direction[2]),
            poly.deriv(2)(direction[2]),
        )
        a = (degree + 1) * p + direction[2] * dp
        da = (degree + 2) * dp + direction[2] * ddp

        potential -= float(
            _scaled_product_quotient(
                (density, pi, radius, radius, height, moment_scaled, p),
                distance_factors,
            )
        )

        acceleration_shape = dp * ez - a * direction
        for row in range(3):
            acc[row] += float(
                _scaled_product_quotient(
                    (
                        density,
                        pi,
                        radius,
                        radius,
                        height,
                        moment_scaled,
                        acceleration_shape[row],
                    ),
                    (*distance_factors, *distance_factors),
                )
            )

        gradient_shape = (
            ((degree + 3) * a + direction[2] * da) * np.outer(direction, direction)
            - da * (np.outer(direction, ez) + np.outer(ez, direction))
            + ddp * np.outer(ez, ez)
            - a * np.eye(3)
        )
        for row in range(3):
            for column in range(3):
                grad[row, column] += float(
                    _scaled_product_quotient(
                        (
                            density,
                            pi,
                            radius,
                            radius,
                            height,
                            moment_scaled,
                            gradient_shape[row, column],
                        ),
                        (*distance_factors, *distance_factors, *distance_factors),
                    )
                )
    return potential, acc, grad


def _cylinder_far_field_values(
    radius: float, height: float, density: float, point: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Potential, acceleration, and gradient through solid-harmonic degree 8."""
    coordinate_scale = float(np.max(np.abs(point)))
    if coordinate_scale == 0.0:
        raise ValueError("far-field evaluation requires a nonzero target vector")
    return _cylinder_far_field_values_scaled(
        radius,
        height,
        density,
        coordinate_scale,
        np.asarray(point, dtype=np.float64) / coordinate_scale,
    )


def _cylinder_axis_far_field_values(
    z: np.ndarray, radius: float, height: float, density: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate far axial potential, acceleration, and gradient by solid harmonics."""
    values = np.asarray(z, dtype=np.float64)
    flat = values.reshape(-1)
    potential = np.empty_like(flat)
    acceleration = np.empty_like(flat)
    gradient_zz = np.empty_like(flat)
    for index, axial_coordinate in enumerate(flat):
        potential[index], acc, grad = _cylinder_far_field_values(
            radius,
            height,
            density,
            np.array([0.0, 0.0, axial_coordinate]),
        )
        acceleration[index] = acc[2]
        gradient_zz[index] = grad[2, 2]
    return (
        potential.reshape(values.shape),
        acceleration.reshape(values.shape),
        gradient_zz.reshape(values.shape),
    )


def _cylinder_far_field(
    radius: float, height: float, density: float, point: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Exterior solid-harmonic acceleration and gradient through degree 8."""
    _, acceleration, gradient = _cylinder_far_field_values(
        radius, height, density, point
    )
    return acceleration, gradient


def cylinder_acceleration_gradient_g_free(
    center_m: list | tuple | np.ndarray,
    radius_m: float,
    height_m: float,
    density_kg_m3: float,
    targets_m: list | tuple | np.ndarray,
    *,
    rotation_matrix: list | tuple | np.ndarray | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Analytic acceleration and full gradient of a uniform finite cylinder.

    Returns read-only float64 arrays ``(M,3)`` and ``(M,3,3)`` in kg/m²
    and kg/m³. Uses real complete elliptic integrals, no spatial quadrature
    or finite differences. Material-boundary targets are rejected.
    Numerical potential remains available from ``cylinder_observables_g_free``.
    """
    center, radius, height, density, targets = _validate_cylinder_inputs(
        center_m, radius_m, height_m, density_kg_m3, targets_m
    )
    rotation = (
        np.eye(3) if rotation_matrix is None else _cylinder_rotation(rotation_matrix)
    )
    points, scaled_mask, coordinate_scales, scaled_points = _cylinder_local_targets(
        center, targets, rotation
    )
    _reject_cylinder_boundary_targets(
        points,
        radius,
        height,
        scaled_mask=scaled_mask,
        coordinate_scales=coordinate_scales,
        scaled_points=scaled_points,
    )
    acc = np.zeros((len(points), 3))
    grad = np.zeros((len(points), 3, 3))
    for i, point in enumerate(points):
        if density == 0:
            continue
        if scaled_mask[i]:
            if not _cylinder_is_far_field_scaled(
                coordinate_scales[i], scaled_points[i], radius, height
            ):
                raise OverflowError(
                    "non-far target exceeds float64 local-coordinate range"
                )
            _, local_acc, local_grad = _cylinder_far_field_values_scaled(
                radius,
                height,
                density,
                coordinate_scales[i],
                scaled_points[i],
            )
            acc[i], grad[i] = local_acc, local_grad
            continue
        x, y, z = point
        if _cylinder_is_far_field_point(points[i], radius, height):
            acc[i], grad[i] = _cylinder_far_field(radius, height, density, points[i])
            continue
        r = np.hypot(x, y)
        if r == 0:
            axis = cylinder_axis_observables_g_free(
                [0.0, 0.0, 0.0], radius, height, density, [[0.0, 0.0, z]]
            )
            acc[i], grad[i] = axis.acceleration[0], axis.gradient[0]
            continue
        if r <= 1e-3 * radius:
            ar, az, rz, zz = _cylinder_near_axis(radius, height, r, z, density)
        else:
            lower = _cylinder_end_fields(radius, r, -height / 2 - z)
            upper = _cylinder_end_fields(radius, r, height / 2 - z)
            ar, az, rz, zz = density * (upper - lower)
            az = -az
            zz = -zz
        pp = ar / r
        trace = -4 * np.pi * density if r < radius and abs(z) < height / 2 else 0.0
        rr = trace - pp - zz
        radial = np.array([x / r, y / r, 0.0])
        tangent = np.array([-y / r, x / r, 0.0])
        axial = np.array([0.0, 0.0, 1.0])
        acc[i] = ar * radial + az * axial
        grad[i] = (
            rr * np.outer(radial, radial)
            + pp * np.outer(tangent, tangent)
            + zz * np.outer(axial, axial)
            + rz * (np.outer(radial, axial) + np.outer(axial, radial))
        )
    acc = acc @ rotation.T
    grad = np.einsum("ij,mjk,lk->mil", rotation, grad, rotation)
    acc.setflags(write=False)
    grad.setflags(write=False)
    return acc, grad


def _cylinder_rotation(rotation_matrix: list | tuple | np.ndarray) -> np.ndarray:
    """Match the sibling oriented-field contract without importing its branch.

    #114 is not a dependency of #112. No projection/normalization is done.
    """
    _check_no_bool(rotation_matrix, "rotation_matrix")
    raw = np.asarray(rotation_matrix)
    if raw.dtype.kind in "bcO" or not np.issubdtype(raw.dtype, np.number):
        raise TypeError("rotation_matrix must be a real numeric array")
    q = np.array(raw, dtype=np.float64, copy=True)
    if q.shape != (3, 3):
        raise ValueError("rotation_matrix must have shape (3, 3)")
    if not np.all(np.isfinite(q)):
        raise ValueError("rotation_matrix must be finite")
    if not np.allclose(q.T @ q, np.eye(3), rtol=0, atol=1e-10):
        raise ValueError("rotation_matrix must satisfy Q.T @ Q = I")
    if not np.isclose(np.linalg.det(q), 1.0, rtol=0, atol=1e-10):
        raise ValueError("rotation_matrix must have determinant +1")
    return q
