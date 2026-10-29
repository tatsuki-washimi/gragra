"""SI physical observable entrypoints for analytical gravity solutions."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from gragra.analytic.cavity_cuboid import cuboid_observables_g_free
from gragra.analytic.cavity_cylinder import (
    cylinder_acceleration_gradient_g_free,
    cylinder_axis_observables_g_free,
    cylinder_observables_g_free,
)
from gragra.analytic.cavity_sphere import (
    spherical_shell_p_acceleration_components_g_free,
    spherical_shell_p_infinite_difference_g_free,
    spherical_shell_s_acceleration_components_g_free,
    spherical_shell_s_infinite_difference_g_free,
)
from gragra.analytic.cylinder_potential import cylinder_analytic_observables_g_free
from gragra.analytic.oriented import oriented_cuboid_observables_g_free
from gragra.analytic.sca_tensor import sca_wall_acceleration_from_tensor_g_free
from gragra.analytic.uniform_sphere import uniform_sphere_observables_g_free
from gragra.analytic.values import (
    GFreeGravityValues,
    GFreeSphericalShellPComponents,
    GFreeSphericalShellSComponents,
)
from gragra.constants import G_SI

ArrayLike = list | tuple | np.ndarray


def spherical_cavity_observables(targets_m, **kwargs) -> dict[str, GravityObservables]:
    """SI bulk/wall/total analytic series; tolerances are specified in G-free units.

    Parameters follow ``spherical_cavity_observables_g_free`` exactly.
    """
    from gragra.analytic.spherical_cavity_series import (
        spherical_cavity_observables_g_free,
    )

    fields = spherical_cavity_observables_g_free(targets_m, **kwargs)
    return {name: to_gravity_observables(field) for name, field in fields.items()}


@dataclass(frozen=True, init=False)
class GravityObservables:
    """Potential, acceleration, and gradient in SI units.

    Physical units:
    - potential: m^2 / s^2
    - acceleration: m / s^2
    - gradient: s^-2

    Shapes:
    - Static targets: potential ``(M,)``, acceleration ``(M, 3)``,
      gradient ``(M, 3, 3)``.
    - Series targets: potential ``(S, M)``, acceleration ``(S, M, 3)``,
      gradient ``(S, M, 3, 3)``.
    """

    potential: np.ndarray
    acceleration: np.ndarray
    gradient: np.ndarray

    def __init__(
        self,
        potential: ArrayLike,
        acceleration: ArrayLike,
        gradient: ArrayLike,
    ) -> None:
        raw_p = np.asarray(potential)
        raw_a = np.asarray(acceleration)
        raw_g = np.asarray(gradient)

        if raw_p.ndim not in (1, 2):
            raise ValueError(
                "potential must have 1 or 2 dimensions (M,) or (S, M); "
                f"got ndim={raw_p.ndim}"
            )
        prefix = raw_p.shape
        if raw_a.shape != (*prefix, 3):
            raise ValueError(
                f"acceleration shape {raw_a.shape} must match "
                f"(*potential.shape, 3) = {(*prefix, 3)}"
            )
        if raw_g.shape != (*prefix, 3, 3):
            raise ValueError(
                f"gradient shape {raw_g.shape} must match "
                f"(*potential.shape, 3, 3) = {(*prefix, 3, 3)}"
            )

        is_complex = any(np.iscomplexobj(x) for x in (raw_p, raw_a, raw_g))
        target_dtype = np.complex128 if is_complex else np.float64

        p_arr = np.asarray(raw_p, dtype=target_dtype).copy()
        a_arr = np.asarray(raw_a, dtype=target_dtype).copy()
        g_arr = np.asarray(raw_g, dtype=target_dtype).copy()

        for name, arr in [
            ("potential", p_arr),
            ("acceleration", a_arr),
            ("gradient", g_arr),
        ]:
            if not np.isfinite(arr).all():
                raise ValueError(f"{name} must contain only finite values")
            arr.setflags(write=False)

        object.__setattr__(self, "potential", p_arr)
        object.__setattr__(self, "acceleration", a_arr)
        object.__setattr__(self, "gradient", g_arr)

    @property
    def is_series(self) -> bool:
        return self.potential.ndim == 2

    def __iter__(self) -> Iterator[np.ndarray]:
        return iter((self.potential, self.acceleration, self.gradient))


@dataclass(frozen=True, init=False)
class SphericalShellPComponents:
    """Component decomposition of spherical-shell P-wave acceleration in SI units.

    Fields:
    - ``bulk``: Volume compression contribution of the shell (m / s^2).
    - ``inner_wall``: Surface mass-sheet contribution of the inner cavity wall.
    - ``outer_boundary``: Surface mass-sheet contribution of the outer boundary.
    - ``total``: ``bulk + inner_wall + outer_boundary``.
    - ``unit_system``: Fixed identifier string ``"si"``.

    Shapes:
    - Scalar wavenumber: each field is ``(1, 3)`` (one center, three components).
    - 1D wavenumber array of length K: each field is ``(K, 1, 3)``.
    """

    bulk: np.ndarray
    inner_wall: np.ndarray
    outer_boundary: np.ndarray
    total: np.ndarray
    unit_system: str = "si"

    def __init__(
        self,
        bulk: ArrayLike,
        inner_wall: ArrayLike,
        outer_boundary: ArrayLike,
        total: ArrayLike,
    ) -> None:
        raw_b = np.asarray(bulk)
        raw_iw = np.asarray(inner_wall)
        raw_ob = np.asarray(outer_boundary)
        raw_tot = np.asarray(total)

        if raw_tot.ndim not in (2, 3) or raw_tot.shape[-2:] != (1, 3):
            raise ValueError(
                "components must have shape (1, 3) or (K, 1, 3); "
                f"got shape={raw_tot.shape}"
            )
        expected_shape = raw_tot.shape
        for name, raw in [
            ("bulk", raw_b),
            ("inner_wall", raw_iw),
            ("outer_boundary", raw_ob),
        ]:
            if raw.shape != expected_shape:
                raise ValueError(
                    f"{name} shape {raw.shape} must match total shape {expected_shape}"
                )

        is_complex = any(np.iscomplexobj(x) for x in (raw_b, raw_iw, raw_ob, raw_tot))
        target_dtype = np.complex128 if is_complex else np.float64

        b_arr = np.asarray(raw_b, dtype=target_dtype).copy()
        iw_arr = np.asarray(raw_iw, dtype=target_dtype).copy()
        ob_arr = np.asarray(raw_ob, dtype=target_dtype).copy()
        tot_arr = np.asarray(raw_tot, dtype=target_dtype).copy()

        for name, arr in [
            ("bulk", b_arr),
            ("inner_wall", iw_arr),
            ("outer_boundary", ob_arr),
            ("total", tot_arr),
        ]:
            if not np.isfinite(arr).all():
                raise ValueError(f"{name} must contain only finite values")
            arr.setflags(write=False)

        object.__setattr__(self, "bulk", b_arr)
        object.__setattr__(self, "inner_wall", iw_arr)
        object.__setattr__(self, "outer_boundary", ob_arr)
        object.__setattr__(self, "total", tot_arr)
        object.__setattr__(self, "unit_system", "si")

    def __iter__(self) -> Iterator[np.ndarray]:
        return iter((self.bulk, self.inner_wall, self.outer_boundary, self.total))


@dataclass(frozen=True, init=False)
class SphericalShellSComponents:
    """Component decomposition of spherical-shell S-wave acceleration in SI units.

    Fields:
    - ``bulk``: Always identically zero (S waves are divergence-free).
    - ``inner_wall``: Surface mass-sheet contribution of the inner cavity wall.
    - ``outer_boundary``: Surface mass-sheet contribution of the outer boundary.
    - ``total``: ``inner_wall + outer_boundary``.
    - ``unit_system``: Fixed identifier string ``"si"``.

    Shapes:
    - Scalar wavenumber: each field is ``(1, 3)`` (one center, three components).
    - 1D wavenumber array of length K: each field is ``(K, 1, 3)``.
    """

    bulk: np.ndarray
    inner_wall: np.ndarray
    outer_boundary: np.ndarray
    total: np.ndarray
    unit_system: str = "si"

    def __init__(
        self,
        bulk: ArrayLike,
        inner_wall: ArrayLike,
        outer_boundary: ArrayLike,
        total: ArrayLike,
    ) -> None:
        raw_b = np.asarray(bulk)
        raw_iw = np.asarray(inner_wall)
        raw_ob = np.asarray(outer_boundary)
        raw_tot = np.asarray(total)

        if raw_tot.ndim not in (2, 3) or raw_tot.shape[-2:] != (1, 3):
            raise ValueError(
                "components must have shape (1, 3) or (K, 1, 3); "
                f"got shape={raw_tot.shape}"
            )
        expected_shape = raw_tot.shape
        for name, raw in [
            ("bulk", raw_b),
            ("inner_wall", raw_iw),
            ("outer_boundary", raw_ob),
        ]:
            if raw.shape != expected_shape:
                raise ValueError(
                    f"{name} shape {raw.shape} must match total shape {expected_shape}"
                )

        is_complex = any(np.iscomplexobj(x) for x in (raw_b, raw_iw, raw_ob, raw_tot))
        target_dtype = np.complex128 if is_complex else np.float64

        b_arr = np.asarray(raw_b, dtype=target_dtype).copy()
        iw_arr = np.asarray(raw_iw, dtype=target_dtype).copy()
        ob_arr = np.asarray(raw_ob, dtype=target_dtype).copy()
        tot_arr = np.asarray(raw_tot, dtype=target_dtype).copy()

        for name, arr in [
            ("bulk", b_arr),
            ("inner_wall", iw_arr),
            ("outer_boundary", ob_arr),
            ("total", tot_arr),
        ]:
            if not np.isfinite(arr).all():
                raise ValueError(f"{name} must contain only finite values")
            arr.setflags(write=False)

        object.__setattr__(self, "bulk", b_arr)
        object.__setattr__(self, "inner_wall", iw_arr)
        object.__setattr__(self, "outer_boundary", ob_arr)
        object.__setattr__(self, "total", tot_arr)
        object.__setattr__(self, "unit_system", "si")

    def __iter__(self) -> Iterator[np.ndarray]:
        return iter((self.bulk, self.inner_wall, self.outer_boundary, self.total))


def to_gravity_observables(values: GFreeGravityValues) -> GravityObservables:
    """Convert G-free gravitational values to SI gravitational observables.

    Multiplies potential, acceleration, and gradient by G_SI exactly once.
    Rejects values already in SI units or legacy structures without clear units.

    Parameters
    ----------
    values : GFreeGravityValues
        G-free potential, acceleration, and gradient.

    Returns
    -------
    GravityObservables
        SI observables (m^2/s^2, m/s^2, s^-2).
    """
    if isinstance(values, GravityObservables):
        raise TypeError(
            "Values are already GravityObservables (SI units); cannot re-convert to SI"
        )
    if isinstance(values, (GFreeSphericalShellPComponents, SphericalShellPComponents)):
        raise TypeError(
            f"Expected GFreeGravityValues, got {type(values).__name__}. "
            "Use to_spherical_shell_p_components(...) to convert spherical "
            "shell components to SI."
        )
    if isinstance(values, (GFreeSphericalShellSComponents, SphericalShellSComponents)):
        raise TypeError(
            f"Expected GFreeGravityValues, got {type(values).__name__}. "
            "Use to_spherical_shell_s_components(...) to convert spherical "
            "shell components to SI."
        )
    if not isinstance(values, GFreeGravityValues):
        raise TypeError(
            f"Expected GFreeGravityValues instance, got {type(values).__name__}. "
            "Legacy GFreeObservables or unknown structures cannot be safely converted."
        )

    return GravityObservables(
        potential=G_SI * values.potential,
        acceleration=G_SI * values.acceleration,
        gradient=G_SI * values.gradient,
    )


def to_spherical_shell_p_components(
    values: GFreeSphericalShellPComponents,
) -> SphericalShellPComponents:
    """Convert G-free spherical-shell P-wave acceleration components to SI units.

    Multiplies bulk, inner_wall, outer_boundary, and total by G_SI exactly once.
    Rejects values already in SI units or incompatible structures.
    """
    if isinstance(values, SphericalShellPComponents):
        raise TypeError(
            "Values are already SphericalShellPComponents (SI units); "
            "cannot re-convert to SI"
        )
    if not isinstance(values, GFreeSphericalShellPComponents):
        raise TypeError(
            f"Expected GFreeSphericalShellPComponents, got {type(values).__name__}"
        )

    return SphericalShellPComponents(
        bulk=G_SI * values.bulk,
        inner_wall=G_SI * values.inner_wall,
        outer_boundary=G_SI * values.outer_boundary,
        total=G_SI * values.total,
    )


def to_spherical_shell_s_components(
    values: GFreeSphericalShellSComponents,
) -> SphericalShellSComponents:
    """Convert G-free spherical-shell S-wave acceleration components to SI units.

    Multiplies bulk, inner_wall, outer_boundary, and total by G_SI exactly once.
    Rejects values already in SI units or incompatible structures.
    """
    if isinstance(values, SphericalShellSComponents):
        raise TypeError(
            "Values are already SphericalShellSComponents (SI units); "
            "cannot re-convert to SI"
        )
    if not isinstance(values, GFreeSphericalShellSComponents):
        raise TypeError(
            f"Expected GFreeSphericalShellSComponents, got {type(values).__name__}"
        )

    return SphericalShellSComponents(
        bulk=G_SI * values.bulk,
        inner_wall=G_SI * values.inner_wall,
        outer_boundary=G_SI * values.outer_boundary,
        total=G_SI * values.total,
    )


def uniform_sphere_observables(
    center_m: list | tuple | np.ndarray,
    radius_m: float,
    density_kg_m3: float | complex,
    targets_m: list | tuple | np.ndarray,
) -> GravityObservables:
    """Calculate SI potential, acceleration, and gradient of a uniform sphere.

    Parameters
    ----------
    center_m : array-like, shape (3,)
        Coordinates of the sphere center in meters.
    radius_m : float
        Radius of the sphere in meters (must be positive).
    density_kg_m3 : float or complex
        Mass density in kg/m^3 (real or complex).
    targets_m : array-like, shape (M, 3)
        Coordinates of the target points in meters.

    Returns
    -------
    GravityObservables
        SI observables bundle (potential, acceleration, gradient).
    """
    g_free = uniform_sphere_observables_g_free(
        center_m, radius_m, density_kg_m3, targets_m
    )
    return to_gravity_observables(g_free)


def cuboid_observables(
    center_m: list | tuple | np.ndarray,
    lengths_m: list | tuple | np.ndarray,
    density_kg_m3: float,
    targets_m: list | tuple | np.ndarray,
) -> GravityObservables:
    """Calculate SI potential, acceleration, and gradient of a uniform cuboid.

    Parameters
    ----------
    center_m : array-like, shape (3,)
        Center coordinates of the cuboid in meters.
    lengths_m : array-like, shape (3,)
        Side lengths (Lx, Ly, Lz) in meters (all positive).
    density_kg_m3 : float
        Uniform mass density in kg/m^3 (positive).
    targets_m : array-like, shape (M, 3)
        Observation target points in meters. Target points lying on the
        material boundary of the cuboid are rejected.

    Returns
    -------
    GravityObservables
        SI observables bundle (potential in m^2/s^2, acceleration in m/s^2,
        gradient in s^-2).
    """
    g_free = cuboid_observables_g_free(center_m, lengths_m, density_kg_m3, targets_m)
    return to_gravity_observables(g_free)


def oriented_cuboid_observables(
    center_m: list | tuple | np.ndarray,
    lengths_m: list | tuple | np.ndarray,
    density_kg_m3: float,
    targets_m: list | tuple | np.ndarray,
    *,
    rotation_matrix: list | tuple | np.ndarray,
) -> GravityObservables:
    """Evaluate a rotated uniform cuboid in physical SI units.

    ``rotation_matrix`` is the explicit local-to-world proper rotation. The
    G-free local cuboid closed form is evaluated first and multiplied by
    ``G_SI`` exactly once through :func:`to_gravity_observables`.
    """
    g_free = oriented_cuboid_observables_g_free(
        center_m,
        lengths_m,
        density_kg_m3,
        targets_m,
        rotation_matrix=rotation_matrix,
    )
    return to_gravity_observables(g_free)


def sca_wall_acceleration_from_tensor(
    coupling_tensor: list | tuple | np.ndarray,
    density_kg_m3: float,
    displacement_m: list | tuple | np.ndarray,
) -> np.ndarray:
    """Apply an SCA wall tensor and return the SI acceleration amplitude.

    The G-free operation is ``-rho * C @ displacement``; this wrapper applies
    ``G_SI`` exactly once.  The displacement may be a shared vector ``(3,)``
    or one vector per tensor row ``(M,3)``.
    """
    g_free = sca_wall_acceleration_from_tensor_g_free(
        coupling_tensor, density_kg_m3, displacement_m
    )
    result = G_SI * g_free
    result.setflags(write=False)
    return result


def cylinder_axis_observables(
    center_m: list | tuple | np.ndarray,
    radius_m: float,
    height_m: float,
    density_kg_m3: float,
    targets_m: list | tuple | np.ndarray,
) -> GravityObservables:
    """Calculate SI on-axis potential, acceleration, and gradient of a uniform cylinder.

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
        Observation target points in meters. Off-axis points and points on
        the end faces z = ±h/2 are rejected.

    Returns
    -------
    GravityObservables
        SI observables bundle on the cylinder axis.
    """
    g_free = cylinder_axis_observables_g_free(
        center_m, radius_m, height_m, density_kg_m3, targets_m
    )
    return to_gravity_observables(g_free)


def cylinder_acceleration_gradient(
    center_m: list | tuple | np.ndarray,
    radius_m: float,
    height_m: float,
    density_kg_m3: float,
    targets_m: list | tuple | np.ndarray,
    *,
    rotation_matrix: list | tuple | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Analytic finite-cylinder acceleration (m/s²) and gradient (1/s²).

    ``rotation_matrix`` is the proper local-to-world rotation. Returns
    read-only arrays (M,3), (M,3,3); rejects the physical material boundary.
    See ``cylinder_acceleration_gradient_g_free`` for the analytic method.
    """
    values = cylinder_acceleration_gradient_g_free(
        center_m,
        radius_m,
        height_m,
        density_kg_m3,
        targets_m,
        rotation_matrix=rotation_matrix,
    )
    acceleration, gradient = (G_SI * value for value in values)
    acceleration.setflags(write=False)
    gradient.setflags(write=False)
    return acceleration, gradient


def cylinder_analytic_observables(
    center_m: ArrayLike,
    radius_m: float,
    height_m: float,
    density_kg_m3: float,
    targets_m: ArrayLike,
    *,
    rotation_matrix: ArrayLike | None = None,
) -> GravityObservables:
    """Analytic finite-cylinder triple in SI units, with infinity-zero potential.

    See ``cylinder_analytic_observables_g_free`` for the real elliptic/series
    method and material-boundary exclusions. Rotation maps local to world.
    The existing ``cylinder_observables`` is the numerical diagnostic route.
    """
    return to_gravity_observables(
        cylinder_analytic_observables_g_free(
            center_m,
            radius_m,
            height_m,
            density_kg_m3,
            targets_m,
            rotation_matrix=rotation_matrix,
        )
    )


def cylinder_observables(
    center_m: list | tuple | np.ndarray,
    radius_m: float,
    height_m: float,
    density_kg_m3: float,
    targets_m: list | tuple | np.ndarray,
    *,
    quadrature_order: int = 32,
) -> GravityObservables:
    """Calculate SI general-point observables of a z-aligned solid cylinder.

    This is the SI wrapper for the opt-in cylindrical quadrature path in
    :func:`gragra.analytic.cylinder_observables_g_free`.
    """
    g_free = cylinder_observables_g_free(
        center_m,
        radius_m,
        height_m,
        density_kg_m3,
        targets_m,
        quadrature_order=quadrature_order,
    )
    return to_gravity_observables(g_free)


def spherical_shell_p_acceleration_components(
    inner_radius_m: float,
    outer_radius_m: float,
    wavenumber_rad_m: float | list | tuple | np.ndarray,
    density_kg_m3: float,
    displacement_m: float | complex,
    direction: list | tuple | np.ndarray,
) -> SphericalShellPComponents:
    """Calculate SI P-wave acceleration component decomposition in a spherical shell.

    Evaluates the bulk, inner cavity wall, outer boundary, and total acceleration
    components at the center of a spherical cavity in a finite shell a < r < R,
    multiplying by G_SI exactly once.

    Parameters
    ----------
    inner_radius_m : float
        Inner cavity radius a in meters (positive).
    outer_radius_m : float
        Outer boundary radius R in meters (strictly greater than a, finite).
    wavenumber_rad_m : float or array-like
        P-wave wavenumber k in rad/m (non-negative, scalar or 1D array).
    density_kg_m3 : float
        Background mass density in kg/m^3 (positive).
    displacement_m : float or complex
        Center displacement amplitude in meters (real or complex phasor).
    direction : array-like, shape (3,)
        P-wave propagation and displacement direction (non-zero 3-vector).

    Returns
    -------
    SphericalShellPComponents
        Container with bulk, inner_wall, outer_boundary, and total acceleration
        vectors in SI units (m / s^2).
        Shapes: (1, 3) for scalar k, or (K, 1, 3) for 1D k array of length K.
    """
    g_free = spherical_shell_p_acceleration_components_g_free(
        inner_radius_m=inner_radius_m,
        outer_radius_m=outer_radius_m,
        wavenumber_rad_m=wavenumber_rad_m,
        density_kg_m3=density_kg_m3,
        displacement_m=displacement_m,
        direction=direction,
    )
    return to_spherical_shell_p_components(g_free)


def spherical_shell_p_infinite_difference(
    inner_radius_m: float,
    outer_radius_m: float,
    wavenumber_rad_m: float | list | tuple | np.ndarray,
    density_kg_m3: float,
    displacement_m: float | complex,
    direction: list | tuple | np.ndarray,
) -> np.ndarray:
    """Calculate SI difference (infinite_total - finite_total) for k > 0.

    Formula:
        infinite_total - finite_total = 8*pi * G_SI * rho * (j1(kR) / (kR)) * xi * e_hat

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
        Difference vector in SI units (m / s^2).
        Shape: (1, 3) for scalar k, or (K, 1, 3) for 1D k array.
    """
    g_free = spherical_shell_p_infinite_difference_g_free(
        inner_radius_m=inner_radius_m,
        outer_radius_m=outer_radius_m,
        wavenumber_rad_m=wavenumber_rad_m,
        density_kg_m3=density_kg_m3,
        displacement_m=displacement_m,
        direction=direction,
    )
    out = (G_SI * g_free).copy()
    out.setflags(write=False)
    return out


def spherical_shell_s_acceleration_components(
    inner_radius_m: float,
    outer_radius_m: float,
    wavenumber_rad_m: float | list | tuple | np.ndarray,
    density_kg_m3: float,
    displacement_m: float | complex,
    propagation_direction: list | tuple | np.ndarray,
    polarization_direction: list | tuple | np.ndarray,
    *,
    orthogonality_tol: float = 1e-5,
) -> SphericalShellSComponents:
    """Calculate SI S-wave acceleration component decomposition in a spherical shell.

    Evaluates the (always-zero) bulk, inner cavity wall, outer boundary, and
    total acceleration components at the center of a spherical cavity in a
    finite shell a < r < R, multiplying by G_SI exactly once. See
    ``gragra.analytic.spherical_shell_s_acceleration_components_g_free`` for
    the closed-form derivation, primary-source equation references (ANNA v4
    Eq. 71/72/22), and the rotational-symmetry note explaining why
    ``propagation_direction`` does not appear in the output.

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
        S-wave propagation direction (non-zero 3-vector), validated to be
        orthogonal to ``polarization_direction``.
    polarization_direction : array-like, shape (3,)
        S-wave polarization (displacement) direction (non-zero 3-vector).
    orthogonality_tol : float, optional
        Tolerance for the transversality check (default 1e-5). Must be in
        ``(0, 1.0)``; see
        ``gragra.analytic.spherical_shell_s_acceleration_components_g_free``'s
        docstring for why the upper bound is strict and for the additional
        hard, tolerance-independent parallel/antiparallel rejection.

    Returns
    -------
    SphericalShellSComponents
        Container with bulk (always zero), inner_wall, outer_boundary, and
        total acceleration vectors in SI units (m / s^2).
        Shapes: (1, 3) for scalar k, or (K, 1, 3) for 1D k array of length K.
    """
    g_free = spherical_shell_s_acceleration_components_g_free(
        inner_radius_m=inner_radius_m,
        outer_radius_m=outer_radius_m,
        wavenumber_rad_m=wavenumber_rad_m,
        density_kg_m3=density_kg_m3,
        displacement_m=displacement_m,
        propagation_direction=propagation_direction,
        polarization_direction=polarization_direction,
        orthogonality_tol=orthogonality_tol,
    )
    return to_spherical_shell_s_components(g_free)


def spherical_shell_s_infinite_difference(
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
    """Calculate SI difference (infinite_total - finite_total) for k > 0.

    Formula (ANNA v4 Eq. 71 and Eq. 72 imply this algebraic difference):
        infinite_total - finite_total = -4*pi * G_SI * rho *
        (j1(kR) / (kR)) * xi * e_hat_S

    See ``gragra.analytic.spherical_shell_s_acceleration_components_g_free``'s
    "Rotational-symmetry note" for why ``propagation_direction`` is a
    required, validated input but does not itself appear in the formula
    above.

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
        Center displacement amplitude in meters.
    propagation_direction : array-like, shape (3,)
        S-wave propagation direction (non-zero 3-vector), validated to be
        orthogonal to ``polarization_direction``.
    polarization_direction : array-like, shape (3,)
        S-wave polarization (displacement) direction (non-zero 3-vector).
    orthogonality_tol : float, optional
        Tolerance for the transversality check (default 1e-5). Must be in
        ``(0, 1.0)``; see
        ``gragra.analytic.spherical_shell_s_acceleration_components_g_free``'s
        docstring for why the upper bound is strict and for the additional
        hard, tolerance-independent parallel/antiparallel rejection.

    Returns
    -------
    np.ndarray
        Difference vector in SI units (m / s^2).
        Shape: (1, 3) for scalar k, or (K, 1, 3) for 1D k array.
    """
    g_free = spherical_shell_s_infinite_difference_g_free(
        inner_radius_m=inner_radius_m,
        outer_radius_m=outer_radius_m,
        wavenumber_rad_m=wavenumber_rad_m,
        density_kg_m3=density_kg_m3,
        displacement_m=displacement_m,
        propagation_direction=propagation_direction,
        polarization_direction=polarization_direction,
        orthogonality_tol=orthogonality_tol,
    )
    out = (G_SI * g_free).copy()
    out.setflags(write=False)
    return out
