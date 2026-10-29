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
from __future__ import annotations

from typing import Literal

import numpy as np

from gragra.analytic._bessel import (
    _j1_over_x,
    _j1_over_x_diff,
    cavity_wall_geometric,
    spherical_j0,
)
from gragra.analytic._validation import (
    validate_array_input,
    validate_nonnegative_scalar_input,
    validate_real_xyz_vector,
    validate_scalar_input,
    validate_strict_nonnegative_array,
)
from gragra.analytic.values import (
    GFreeSphericalShellPComponents,
    GFreeSphericalShellSComponents,
)


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


def _validate_shell_p_inputs(
    inner_radius_m: float,
    outer_radius_m: float,
    wavenumber_rad_m: float | list | tuple | np.ndarray,
    density_kg_m3: float,
    displacement_m: float | complex,
    direction: list | tuple | np.ndarray,
) -> tuple[float, float, np.ndarray, bool, float, float | complex, np.ndarray]:
    a = validate_scalar_input(inner_radius_m, "inner_radius_m")
    R = validate_scalar_input(outer_radius_m, "outer_radius_m")
    if R <= a:
        raise ValueError(
            f"outer_radius_m ({R}) must be strictly greater than inner_radius_m ({a})"
        )
    rho = validate_nonnegative_scalar_input(density_kg_m3, "density_kg_m3")

    if isinstance(displacement_m, (bool, np.bool_)):
        raise TypeError("displacement_m must not be boolean")
    if not isinstance(displacement_m, (int, float, complex, np.number)):
        raise TypeError("displacement_m must be a numeric or complex scalar")
    if np.iscomplexobj(displacement_m):
        disp = complex(displacement_m)
    else:
        disp = float(displacement_m)
    if not np.isfinite(disp):
        raise ValueError("displacement_m must be finite")

    dir_arr = validate_real_xyz_vector(direction, "direction")
    dir_norm = np.linalg.norm(dir_arr)
    if dir_norm == 0.0:
        raise ValueError("direction must be a non-zero finite 3-vector")
    e_hat = dir_arr / dir_norm

    raw_k = np.asarray(wavenumber_rad_m)
    if raw_k.ndim > 1:
        raise ValueError(
            f"wavenumber_rad_m must be a scalar or 1D array; got ndim={raw_k.ndim}"
        )
    is_scalar_k = raw_k.ndim == 0
    k_arr = np.atleast_1d(validate_strict_nonnegative_array(wavenumber_rad_m, "wavenumber_rad_m"))
    if np.any(k_arr < 0.0):
        raise ValueError("wavenumber_rad_m must be non-negative")

    return a, R, k_arr, is_scalar_k, rho, disp, e_hat


def spherical_shell_p_acceleration_components_g_free(
    inner_radius_m: float,
    outer_radius_m: float,
    wavenumber_rad_m: float | list | tuple | np.ndarray,
    density_kg_m3: float,
    displacement_m: float | complex,
    direction: list | tuple | np.ndarray,
) -> GFreeSphericalShellPComponents:
    """Calculate G-free P-wave acceleration component decomposition in a finite spherical shell.

    The model considers an isotropic, homogeneous medium in a spherical shell
    a < r < R with background density rho. A plane harmonic P-wave propagates in
    the given direction with center displacement amplitude xi. The observation point
    is fixed at the cavity center.

    Formulation:
        ANNA v4 (arXiv:2603.15157v4, §4.1, Eqs. 47-53):
        bulk = 4*pi * rho * [j0(ka) - j0(kR)] * xi * e_hat
        inner_wall = 4*pi * rho * [2*f(ka) - j0(ka)] * xi * e_hat
        outer_boundary = 4*pi * rho * [j0(kR) - 2*f(kR)] * xi * e_hat
        total = 8*pi * rho * [f(ka) - f(kR)] * xi * e_hat
        where f(x) = j1(x) / x.

    Parameters
    ----------
    inner_radius_m : float
        Inner cavity radius a in meters (positive).
    outer_radius_m : float
        Outer boundary radius R in meters (strictly greater than a, finite).
    wavenumber_rad_m : float or array-like
        P-wave wavenumber k in rad/m (non-negative, scalar or 1D array).
    density_kg_m3 : float
        Background mass density in kg/m^3 (non-negative, >= 0).
    displacement_m : float or complex
        Center displacement amplitude in meters (real or complex phasor).
    direction : array-like, shape (3,)
        P-wave propagation and displacement direction (non-zero 3-vector).
        Normalized internally to a unit vector.

    Returns
    -------
    GFreeSphericalShellPComponents
        Container with bulk, inner_wall, outer_boundary, and total acceleration
        vectors in G-free units (kg / m^2).
        Shapes: (1, 3) for scalar k, or (K, 1, 3) for 1D k array of length K.
    """
    (
        a,
        R,
        k_arr,
        is_scalar_k,
        rho,
        disp,
        e_hat,
    ) = _validate_shell_p_inputs(
        inner_radius_m,
        outer_radius_m,
        wavenumber_rad_m,
        density_kg_m3,
        displacement_m,
        direction,
    )

    ka = k_arr * a
    kR = k_arr * R
    j0_ka = spherical_j0(ka)
    j0_kR = spherical_j0(kR)
    f_ka = _j1_over_x(ka)
    f_kR = _j1_over_x(kR)

    bulk_c = 4.0 * np.pi * rho * (j0_ka - j0_kR)
    inner_c = 4.0 * np.pi * rho * (2.0 * f_ka - j0_ka)
    outer_c = 4.0 * np.pi * rho * (j0_kR - 2.0 * f_kR)
    tot_c = 8.0 * np.pi * rho * (f_ka - f_kR)

    if is_scalar_k:
        bulk = (bulk_c[0] * disp * e_hat).reshape(1, 3)
        inner = (inner_c[0] * disp * e_hat).reshape(1, 3)
        outer = (outer_c[0] * disp * e_hat).reshape(1, 3)
        total = (tot_c[0] * disp * e_hat).reshape(1, 3)
    else:
        bulk = bulk_c[:, np.newaxis, np.newaxis] * disp * e_hat[np.newaxis, np.newaxis, :]
        inner = inner_c[:, np.newaxis, np.newaxis] * disp * e_hat[np.newaxis, np.newaxis, :]
        outer = outer_c[:, np.newaxis, np.newaxis] * disp * e_hat[np.newaxis, np.newaxis, :]
        total = tot_c[:, np.newaxis, np.newaxis] * disp * e_hat[np.newaxis, np.newaxis, :]

    return GFreeSphericalShellPComponents(
        bulk=bulk,
        inner_wall=inner,
        outer_boundary=outer,
        total=total,
    )


def spherical_shell_p_infinite_difference_g_free(
    inner_radius_m: float,
    outer_radius_m: float,
    wavenumber_rad_m: float | list | tuple | np.ndarray,
    density_kg_m3: float,
    displacement_m: float | complex,
    direction: list | tuple | np.ndarray,
) -> np.ndarray:
    """Calculate the G-free difference (infinite_total - finite_total) for k > 0.

    Formula:
        infinite_total - finite_total = 8*pi * rho * (j1(kR) / (kR)) * xi * e_hat

    Parameters
    ----------
    inner_radius_m : float
        Inner cavity radius a in meters (positive).
    outer_radius_m : float
        Outer boundary radius R in meters (strictly greater than a, finite).
    wavenumber_rad_m : float or array-like
        P-wave wavenumber k in rad/m (strictly positive k > 0, scalar or 1D array).
    density_kg_m3 : float
        Background mass density in kg/m^3 (positive).
    displacement_m : float or complex
        Center displacement amplitude in meters.
    direction : array-like, shape (3,)
        P-wave direction (non-zero 3-vector).

    Returns
    -------
    np.ndarray
        Difference vector in G-free units (kg / m^2).
        Shape: (1, 3) for scalar k, or (K, 1, 3) for 1D k array.
    """
    (
        a,
        R,
        k_arr,
        is_scalar_k,
        rho,
        disp,
        e_hat,
    ) = _validate_shell_p_inputs(
        inner_radius_m,
        outer_radius_m,
        wavenumber_rad_m,
        density_kg_m3,
        displacement_m,
        direction,
    )

    if np.any(k_arr <= 0.0):
        raise ValueError(
            "wavenumber_rad_m must be strictly positive (> 0) for infinite_difference"
        )

    kR = k_arr * R
    f_kR = _j1_over_x(kR)
    diff_c = 8.0 * np.pi * rho * f_kR

    if is_scalar_k:
        out = (diff_c[0] * disp * e_hat).reshape(1, 3)
    else:
        out = diff_c[:, np.newaxis, np.newaxis] * disp * e_hat[np.newaxis, np.newaxis, :]

    out.setflags(write=False)
    return out


# Hard cap for rejecting a (numerically) exactly parallel/antiparallel
# (propagation_direction, polarization_direction) pair, independent of
# orthogonality_tol. Two ULPs of float64 normalization noise (~2.2e-16,
# see e.g. normalizing [1, 1, 0]) is many orders of magnitude below this
# margin, so it never rejects a genuinely transverse pair.
_PARALLEL_REJECT_MARGIN = 1e-9


def _scale_safe_unit_vector(vec: np.ndarray, name: str) -> np.ndarray:
    """Normalize a real 3-vector to a unit vector without overflow/underflow.

    ``np.linalg.norm`` squares each component before summing, so a finite
    but very large component (e.g. ~1e200) makes the squared sum overflow
    to ``inf`` in float64, and a finite but very small component (e.g.
    ~1e-200) makes it underflow to exactly ``0.0``. Dividing by that ``inf``
    silently produces the zero vector for a perfectly valid, non-zero
    direction, and dividing by that erroneous ``0.0`` would raise a
    zero-vector error for a perfectly valid, non-zero (merely tiny)
    direction. Scaling by the largest component's magnitude first keeps the
    intermediate squared terms near order unity, so the result depends only
    on the input's *direction*, not its scale.
    """
    scale = np.max(np.abs(vec))
    if scale == 0.0 or not np.isfinite(scale):
        raise ValueError(f"{name} must be a non-zero finite 3-vector")
    scaled = vec / scale
    norm = np.linalg.norm(scaled)
    unit = scaled / norm
    if not np.all(np.isfinite(unit)):
        raise ValueError(f"{name} must normalize to a finite unit vector")
    return unit


def _validate_shell_s_inputs(
    inner_radius_m: float,
    outer_radius_m: float,
    wavenumber_rad_m: float | list | tuple | np.ndarray,
    density_kg_m3: float,
    displacement_m: float | complex,
    propagation_direction: list | tuple | np.ndarray,
    polarization_direction: list | tuple | np.ndarray,
    orthogonality_tol: float,
) -> tuple[float, float, np.ndarray, bool, float, float | complex, np.ndarray]:
    """Validate finite-shell S-wave inputs.

    Reuses the same geometry/amplitude/wavenumber validation as the P-wave
    shell path (kept independent from ``_validate_shell_p_inputs`` so the
    existing P-wave function's API, shape, and validation behavior are left
    unmodified), and additionally validates a separate propagation direction
    and checks it is orthogonal to the polarization direction (S-wave
    transversality). Only the returned unit polarization vector ``e_hat``
    feeds into the closed-form output; ``propagation_direction`` is used
    solely for this input-validation step (see the "Rotational-symmetry
    note" in ``spherical_shell_s_acceleration_components_g_free``'s
    docstring for why the closed form itself has no explicit dependence on
    propagation direction).
    """
    a = validate_scalar_input(inner_radius_m, "inner_radius_m")
    R = validate_scalar_input(outer_radius_m, "outer_radius_m")
    if R <= a:
        raise ValueError(
            f"outer_radius_m ({R}) must be strictly greater than inner_radius_m ({a})"
        )
    rho = validate_nonnegative_scalar_input(density_kg_m3, "density_kg_m3")

    if isinstance(displacement_m, (bool, np.bool_)):
        raise TypeError("displacement_m must not be boolean")
    if not isinstance(displacement_m, (int, float, complex, np.number)):
        raise TypeError("displacement_m must be a numeric or complex scalar")
    if np.iscomplexobj(displacement_m):
        disp = complex(displacement_m)
    else:
        disp = float(displacement_m)
    if not np.isfinite(disp):
        raise ValueError("displacement_m must be finite")

    pol_arr = validate_real_xyz_vector(polarization_direction, "polarization_direction")
    e_hat = _scale_safe_unit_vector(pol_arr, "polarization_direction")

    prop_arr = validate_real_xyz_vector(propagation_direction, "propagation_direction")
    n_hat = _scale_safe_unit_vector(prop_arr, "propagation_direction")

    if isinstance(orthogonality_tol, (bool, np.bool_)):
        raise TypeError("orthogonality_tol must not be boolean")
    if not isinstance(orthogonality_tol, (int, float, np.integer, np.floating)):
        raise TypeError("orthogonality_tol must be a finite float or int")
    if not np.isfinite(orthogonality_tol):
        raise ValueError("orthogonality_tol must be finite")
    if not (0.0 < orthogonality_tol < 1.0):
        raise ValueError(
            f"orthogonality_tol must be in interval (0, 1.0), got {orthogonality_tol}"
        )

    dot_np = np.abs(np.dot(n_hat, e_hat))
    # Hard, orthogonality_tol-independent guard: reject a (nearly) exactly
    # parallel or antiparallel pair outright, before the tol-based check.
    # dot_np for two *identical*, independently-normalized unit vectors is
    # not guaranteed to be exactly 1.0 in float64 -- normalizing [1, 1, 0]
    # yields dot_np == 0.9999999999999998, two ULPs below 1.0 -- so an
    # orthogonality_tol set arbitrarily close to (but strictly below) 1.0
    # could otherwise fail to catch it via `dot_np > orthogonality_tol`
    # alone. _PARALLEL_REJECT_MARGIN is many orders of magnitude above that
    # few-ULP normalization noise, so it only ever fires on genuinely (or
    # numerically indistinguishably) parallel/antiparallel input.
    if dot_np > 1.0 - _PARALLEL_REJECT_MARGIN:
        raise ValueError(
            "propagation_direction and polarization_direction must not be "
            "(anti)parallel -- S-wave transversality requires a genuinely "
            "transverse pair, independent of orthogonality_tol (which only "
            f"controls how close to orthogonal is accepted); got |dot| = "
            f"{dot_np}"
        )
    if dot_np > orthogonality_tol:
        raise ValueError(
            "propagation_direction must be orthogonal to polarization_direction "
            "(S-wave transversality); "
            f"got |dot| = {dot_np} > orthogonality_tol = {orthogonality_tol}"
        )

    raw_k = np.asarray(wavenumber_rad_m)
    if raw_k.ndim > 1:
        raise ValueError(
            f"wavenumber_rad_m must be a scalar or 1D array; got ndim={raw_k.ndim}"
        )
    is_scalar_k = raw_k.ndim == 0
    k_arr = np.atleast_1d(
        validate_strict_nonnegative_array(wavenumber_rad_m, "wavenumber_rad_m")
    )
    if np.any(k_arr < 0.0):
        raise ValueError("wavenumber_rad_m must be non-negative")

    return a, R, k_arr, is_scalar_k, rho, disp, e_hat


def spherical_shell_s_acceleration_components_g_free(
    inner_radius_m: float,
    outer_radius_m: float,
    wavenumber_rad_m: float | list | tuple | np.ndarray,
    density_kg_m3: float,
    displacement_m: float | complex,
    propagation_direction: list | tuple | np.ndarray,
    polarization_direction: list | tuple | np.ndarray,
    *,
    orthogonality_tol: float = 1e-5,
) -> GFreeSphericalShellSComponents:
    """Calculate G-free S-wave acceleration component decomposition in a finite spherical shell.

    The model considers an isotropic, homogeneous medium in a spherical shell
    a < r < R with background density rho. A plane harmonic S-wave (shear
    wave) propagates in ``propagation_direction`` with polarization
    (displacement direction) ``polarization_direction``, orthogonal to the
    propagation direction (S-wave transversality), and center displacement
    amplitude xi. The observation point is fixed at the cavity center.

    Formulation:
        ANNA v4 (arXiv:2603.15157v4):
        Eq. (71) gives the finite-shell total:
            total(S)/G = -4*pi * rho * (f(ka) - f(kR)) * xi * e_hat_S
        where f(x) = j1(x) / x. The inner/outer decomposition is derived
        from the general surface-integral form, Eq. (22), evaluated with the
        outward normal of the material region at each boundary
        (n_inner = -r_hat at r=a, n_outer = +r_hat at r=R):
            bulk = 0
            inner_wall = -4*pi * rho * f(ka) * xi * e_hat_S
            outer_boundary = +4*pi * rho * f(kR) * xi * e_hat_S
        so that total = inner_wall + outer_boundary, consistent with Eq. (71).
        S waves are divergence-free (div(u) = 0), so they produce no
        volumetric density perturbation and ``bulk`` is identically zero.

    Numerical stability of total:
        ``total`` is computed via a direct stable difference (see
        ``gragra.analytic._bessel._j1_over_x_diff``), not by independently
        rounding ``inner_wall``/``outer_boundary`` to float64 and then
        adding them. At small wavenumber and/or a thin shell (R close to
        a), ``f(ka)`` and ``f(kR)`` each round to ~1/3 in float64, and their
        naive difference loses essentially all significant digits (up to
        100% relative error at extreme k -> 0), while the fully-cancelled
        ``total`` those two individual roundings imply is not recoverable
        from the returned ``inner_wall``/``outer_boundary`` values
        themselves. Consequently, in that regime, ``total`` and
        ``bulk + inner_wall + outer_boundary`` (reconstructed from the
        returned fields) can differ at the level of float64's resolution
        around ``inner_wall``/``outer_boundary``'s own magnitude; use
        ``total`` directly rather than reconstructing it by addition when
        that level of precision matters.

    Direction normalization:
        ``propagation_direction`` and ``polarization_direction`` are
        normalized with a scale-safe unit-vector routine (divide by the
        largest-magnitude component first) so that direction vectors with
        very large or very small (but finite, non-zero) components are not
        mishandled by squared-norm overflow/underflow.

    Rotational-symmetry note:
        Because the observation point is fixed at the common center of
        concentric spherical boundaries, the closed-form output above
        depends only on the wavenumber magnitude k = |k_vec| and the
        polarization direction ``e_hat_S``; it has no explicit dependence on
        the propagation direction ``n_hat``. ``propagation_direction`` is
        required only to validate S-wave transversality
        (n_hat . e_hat_S == 0, within ``orthogonality_tol``); it does not
        otherwise enter the returned values. See
        ``test_shell_s_propagation_direction_invariance`` for a direct
        regression check of this invariance.

    Complex-output disclaimer:
        The output is a linear complex amplitude / transfer-function-like
        response, not directly a real-valued time-domain gravitational field.

    Parameters
    ----------
    inner_radius_m : float
        Inner cavity radius a in meters (positive).
    outer_radius_m : float
        Outer boundary radius R in meters (strictly greater than a, finite).
    wavenumber_rad_m : float or array-like
        S-wave wavenumber k in rad/m (non-negative, scalar or 1D array).
    density_kg_m3 : float
        Background mass density in kg/m^3 (non-negative, >= 0).
    displacement_m : float or complex
        Center displacement amplitude in meters (real or complex phasor).
    propagation_direction : array-like, shape (3,)
        S-wave propagation direction (non-zero 3-vector). Normalized
        internally; validated to be orthogonal to ``polarization_direction``.
    polarization_direction : array-like, shape (3,)
        S-wave polarization (displacement) direction (non-zero 3-vector).
        Normalized internally to a unit vector; this is the direction that
        appears in the returned acceleration vectors.
    orthogonality_tol : float, optional
        Tolerance on ``abs(dot(n_hat, e_hat))`` for the transversality check
        (default 1e-5, matching ``gragra.waves.polarization``'s convention
        for custom-polarization orthogonality checks). Must be in
        ``(0, 1.0)`` (the upper bound is strict, unlike
        ``gragra.waves.polarization``'s ``(0, 1.0]``); this closed form
        assumes exact S-wave transversality, and a nonzero
        ``orthogonality_tol`` only accepts a caller's direction vectors as
        an *approximately* transverse input, not a redefinition of what
        transversality means. Independently of this parameter, a
        propagation/polarization pair whose normalized dot product exceeds
        ``1.0 - 1e-9`` (i.e. an exactly, or numerically indistinguishably,
        parallel or antiparallel pair) is always rejected, regardless of
        how permissive a requested tolerance is -- this guards against the
        case where a very permissive ``orthogonality_tol`` close to 1.0
        happens to sit above the actual (float64-normalization-rounded)
        dot product of a truly parallel pair, which need not be exactly
        1.0 (e.g. normalizing ``[1, 1, 0]`` against itself gives
        ``0.9999999999999998``, not ``1.0``).

    Returns
    -------
    GFreeSphericalShellSComponents
        Container with bulk (always zero), inner_wall, outer_boundary, and
        total acceleration vectors in G-free units (kg / m^2).
        Shapes: (1, 3) for scalar k, or (K, 1, 3) for 1D k array of length K.
    """
    (
        a,
        R,
        k_arr,
        is_scalar_k,
        rho,
        disp,
        e_hat,
    ) = _validate_shell_s_inputs(
        inner_radius_m,
        outer_radius_m,
        wavenumber_rad_m,
        density_kg_m3,
        displacement_m,
        propagation_direction,
        polarization_direction,
        orthogonality_tol,
    )

    ka = k_arr * a
    kR = k_arr * R
    f_ka = _j1_over_x(ka)
    f_kR = _j1_over_x(kR)

    inner_c = -4.0 * np.pi * rho * f_ka
    outer_c = 4.0 * np.pi * rho * f_kR
    # total uses _j1_over_x_diff (not f_ka - f_kR) because at small k and/or
    # a thin shell (R ~ a), f_ka and f_kR each round to ~1/3 in float64 and
    # naively subtracting them loses the O((ka)^2, (kR)^2) difference that
    # is the actual physical answer. See the "Numerical stability of total"
    # note below. ka_minus_kR is computed as -(k_arr*(R-a)) (a single R-a
    # subtraction, done once, before multiplying by k_arr) rather than
    # derived from the separately-rounded ka/kR above, so a thin shell
    # (R close to a) does not lose relative precision in the difference
    # before it ever reaches _j1_over_x_diff.
    ka_minus_kR = -(k_arr * (R - a))
    tot_c = -4.0 * np.pi * rho * _j1_over_x_diff(ka, kR, x_minus_y=ka_minus_kR)

    out_dtype = np.result_type(disp, np.float64)

    if is_scalar_k:
        bulk = np.zeros((1, 3), dtype=out_dtype)
        inner = (inner_c[0] * disp * e_hat).reshape(1, 3)
        outer = (outer_c[0] * disp * e_hat).reshape(1, 3)
        total = (tot_c[0] * disp * e_hat).reshape(1, 3)
    else:
        shape = (k_arr.shape[0], 1, 3)
        bulk = np.zeros(shape, dtype=out_dtype)
        inner = inner_c[:, np.newaxis, np.newaxis] * disp * e_hat[np.newaxis, np.newaxis, :]
        outer = outer_c[:, np.newaxis, np.newaxis] * disp * e_hat[np.newaxis, np.newaxis, :]
        total = tot_c[:, np.newaxis, np.newaxis] * disp * e_hat[np.newaxis, np.newaxis, :]

    return GFreeSphericalShellSComponents(
        bulk=bulk,
        inner_wall=inner,
        outer_boundary=outer,
        total=total,
    )


def spherical_shell_s_infinite_difference_g_free(
    inner_radius_m: float,
    outer_radius_m: float,
    wavenumber_rad_m: float | list | tuple | np.ndarray,
    density_kg_m3: float,
    displacement_m: float | complex,
    propagation_direction: list | tuple | np.ndarray,
    polarization_direction: list | tuple | np.ndarray,
    *,
    orthogonality_tol: float = 1e-5,
) -> np.ndarray:
    """Calculate the G-free difference (infinite_total - finite_total) for k > 0.

    ANNA v4 Eq. (71) (finite-shell total) and Eq. (72) (R -> infinity limit,
    fixed k > 0) imply the algebraic difference (this is not Eq. (72) itself,
    but a consequence of subtracting Eq. (71) from Eq. (72)):

        infinite_total (Eq. 72) - finite_total (Eq. 71)
            = -4*pi * rho * f(kR) * xi * e_hat_S

    See ``spherical_shell_s_acceleration_components_g_free`` for the
    rotational-symmetry note on why ``propagation_direction`` does not
    appear in the output despite being a required, validated input.

    Parameters
    ----------
    inner_radius_m : float
        Inner cavity radius a in meters (positive).
    outer_radius_m : float
        Outer boundary radius R in meters (strictly greater than a, finite).
    wavenumber_rad_m : float or array-like
        S-wave wavenumber k in rad/m (strictly positive k > 0, scalar or 1D array).
    density_kg_m3 : float
        Background mass density in kg/m^3 (non-negative, >= 0).
    displacement_m : float or complex
        Center displacement amplitude in meters (real or complex phasor).
    propagation_direction : array-like, shape (3,)
        S-wave propagation direction (non-zero 3-vector), validated to be
        orthogonal to ``polarization_direction``.
    polarization_direction : array-like, shape (3,)
        S-wave polarization (displacement) direction (non-zero 3-vector).
    orthogonality_tol : float, optional
        Tolerance for the transversality check (default 1e-5). Must be in
        ``(0, 1.0)``; see
        ``spherical_shell_s_acceleration_components_g_free``'s docstring
        for why the upper bound is strict and for the additional hard,
        tolerance-independent parallel/antiparallel rejection.

    Returns
    -------
    np.ndarray
        Difference vector in G-free units (kg / m^2).
        Shape: (1, 3) for scalar k, or (K, 1, 3) for 1D k array.
    """
    (
        a,
        R,
        k_arr,
        is_scalar_k,
        rho,
        disp,
        e_hat,
    ) = _validate_shell_s_inputs(
        inner_radius_m,
        outer_radius_m,
        wavenumber_rad_m,
        density_kg_m3,
        displacement_m,
        propagation_direction,
        polarization_direction,
        orthogonality_tol,
    )

    if np.any(k_arr <= 0.0):
        raise ValueError(
            "wavenumber_rad_m must be strictly positive (> 0) for infinite_difference"
        )

    kR = k_arr * R
    f_kR = _j1_over_x(kR)
    diff_c = -4.0 * np.pi * rho * f_kR

    if is_scalar_k:
        out = (diff_c[0] * disp * e_hat).reshape(1, 3)
    else:
        out = diff_c[:, np.newaxis, np.newaxis] * disp * e_hat[np.newaxis, np.newaxis, :]

    out.setflags(write=False)
    return out

