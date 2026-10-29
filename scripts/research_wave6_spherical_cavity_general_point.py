#!/usr/bin/env python3
"""Independent general-point reference for a fixed spherical cavity.

The reference is deliberately separate from the production kernels.  It uses
direct Newton integrals for the removed spherical bulk source and the cavity
wall, plus the plane-wave particular solution for the unbounded bulk source.
The latter fixes the otherwise arbitrary potential gauge; acceleration and
gradient are gauge independent.

The implementation is limited to a prescribed, non-scattering plane wave.
It is a numerical reference, not a public closed-form API.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

Array = np.ndarray


def _unit_vector(value: Array, name: str) -> Array:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real numeric vector") from exc
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must be finite")
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError(f"{name} must be non-zero")
    return vector.copy() / norm


def _finite_targets(value: Array) -> Array:
    try:
        targets = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("targets_m must be a real numeric table") from exc
    if targets.ndim != 2 or targets.shape[1:] != (3,):
        raise ValueError("targets_m must have shape (M, 3)")
    if not np.isfinite(targets).all():
        raise ValueError("targets_m must be finite")
    return targets


def _sphere_quadrature(order: int) -> tuple[Array, Array]:
    """Return tensor Gauss--Legendre sphere directions and solid-angle weights."""
    mu, mu_weights = np.polynomial.legendre.leggauss(order)
    phi = 2.0 * np.pi * np.arange(2 * order, dtype=np.float64) / (2 * order)
    mu_grid, phi_grid = np.meshgrid(mu, phi, indexing="ij")
    sin_theta = np.sqrt(1.0 - mu_grid**2)
    directions = np.stack(
        (
            sin_theta * np.cos(phi_grid),
            sin_theta * np.sin(phi_grid),
            mu_grid,
        ),
        axis=-1,
    ).reshape(-1, 3)
    weights = (
        (mu_weights[:, None] * (2.0 * np.pi / (2 * order)))
        .repeat(2 * order, axis=1)
        .ravel()
    )
    return directions, weights


def _readonly(value: Array) -> Array:
    result = np.asarray(value, dtype=np.complex128).copy()
    result.setflags(write=False)
    return result


def _zero_observables(target_count: int) -> dict[str, Array]:
    return {
        "potential": np.zeros(target_count, dtype=np.complex128),
        "acceleration": np.zeros((target_count, 3), dtype=np.complex128),
        "gradient": np.zeros((target_count, 3, 3), dtype=np.complex128),
    }


def _add(*values: dict[str, Array]) -> dict[str, Array]:
    return {
        name: sum(
            (value[name] for value in values), start=np.zeros_like(values[0][name])
        )
        for name in ("potential", "acceleration", "gradient")
    }


def _require_finite_observables(
    values: dict[str, dict[str, Array]],
) -> None:
    for term, observables in values.items():
        for name, value in observables.items():
            if not np.isfinite(value).all():
                raise ValueError(f"computed {term}.{name} contains non-finite values")


def _plane_wave_parameters(
    wave_type: str,
    wavenumber_rad_m: float,
    displacement_m: complex,
    propagation_direction: Array,
    polarization: Array,
) -> tuple[float, complex, Array, Array, float]:
    if wave_type not in ("P", "S"):
        raise ValueError("wave_type must be 'P' or 'S'")
    direction = _unit_vector(propagation_direction, "propagation_direction")
    pol = _unit_vector(polarization, "polarization")
    projection = float(np.dot(direction, pol))
    if wave_type == "P":
        if not np.isclose(abs(projection), 1.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("P polarization must be parallel to propagation_direction")
    elif not np.isclose(projection, 0.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("S polarization must be transverse to propagation_direction")
    k = float(wavenumber_rad_m)
    if not np.isfinite(k) or k <= 0.0:
        raise ValueError("wavenumber_rad_m must be positive for this reference")
    try:
        displacement = complex(displacement_m)
    except (TypeError, ValueError) as exc:
        raise TypeError("displacement_m must be numeric") from exc
    if not np.isfinite(displacement):
        raise ValueError("displacement_m must be finite")
    if wave_type == "S":
        projection = 0.0
    return k, displacement, direction, pol, projection


def _density_perturbation(
    points_m: Array,
    *,
    density_kg_m3: float,
    k: float,
    displacement_m: complex,
    direction: Array,
    projection: float,
) -> Array:
    phase = np.exp(1j * k * (points_m @ direction))
    return -1j * density_kg_m3 * k * displacement_m * projection * phase


def _volume_sphere_observables(
    targets_m: Array,
    *,
    radius_m: float,
    density_kg_m3: float,
    k: float,
    displacement_m: complex,
    direction: Array,
    projection: float,
    order: int,
    boundary_fraction: float = 2.0e-7,
) -> dict[str, Array]:
    """Integrate the density perturbation over the removed sphere.

    For an interior target the integration is target-centred.  The gradient
    uses the exact local term ``-4*pi*f(target)*I/3`` for a spherical domain;
    subtracting ``f(target)`` from the principal-value integrand removes the
    remaining 1/r^3 singularity.  This is the material-side local treatment
    used by the reference.
    """
    targets = _finite_targets(targets_m)
    directions, solid_angle_weights = _sphere_quadrature(order)
    mu, radial_weights_unit = np.polynomial.legendre.leggauss(order)
    result = _zero_observables(len(targets))
    eye = np.eye(3)
    for index, target in enumerate(targets):
        target_radius = float(np.linalg.norm(target))
        if abs(target_radius - radius_m) <= boundary_fraction * radius_m:
            raise ValueError("target is too close to the spherical cavity wall")
        if target_radius < radius_m:
            dot = directions @ target
            ray_length = -dot + np.sqrt(dot**2 + radius_m**2 - target_radius**2)
            radii = 0.5 * ray_length[:, None] * (mu[None, :] + 1.0)
            radial_weights = 0.5 * ray_length[:, None] * radial_weights_unit[None, :]
            points = target + directions[:, None, :] * radii[:, :, None]
            points_flat = points.reshape(-1, 3)
            density = _density_perturbation(
                points_flat,
                density_kg_m3=density_kg_m3,
                k=k,
                displacement_m=displacement_m,
                direction=direction,
                projection=projection,
            ).reshape(len(directions), order)
            angular_weights = solid_angle_weights[:, None]
            result["potential"][index] = -np.sum(
                density * radii * radial_weights * angular_weights
            )
            result["acceleration"][index] = np.sum(
                density[:, :, None]
                * directions[:, None, :]
                * radial_weights[:, :, None]
                * angular_weights[:, :, None],
                axis=(0, 1),
            )
            center_density = _density_perturbation(
                target[None, :],
                density_kg_m3=density_kg_m3,
                k=k,
                displacement_m=displacement_m,
                direction=direction,
                projection=projection,
            )[0]
            angular_tensor = 3.0 * directions[:, :, None] * directions[:, None, :]
            angular_tensor -= eye[None, :, :]
            result["gradient"][index] = (
                np.sum(
                    (density - center_density)[:, :, None, None]
                    * angular_tensor[:, None, :, :]
                    / np.maximum(radii[:, :, None, None], np.finfo(float).tiny)
                    * radial_weights[:, :, None, None]
                    * angular_weights[:, :, None, None],
                    axis=(0, 1),
                )
                - (4.0 * np.pi / 3.0) * center_density * eye
            )
        else:
            radii = 0.5 * radius_m * (mu + 1.0)
            radial_weights = 0.5 * radius_m * radial_weights_unit
            points = radii[:, None, None] * directions[None, :, :]
            points_flat = points.reshape(-1, 3)
            density = _density_perturbation(
                points_flat,
                density_kg_m3=density_kg_m3,
                k=k,
                displacement_m=displacement_m,
                direction=direction,
                projection=projection,
            ).reshape(order, len(directions))
            delta = points - target
            distance = np.linalg.norm(delta, axis=-1)
            volume_weights = (
                radii[:, None] ** 2
                * radial_weights[:, None]
                * solid_angle_weights[None, :]
            )
            result["potential"][index] = -np.sum(density * volume_weights / distance)
            result["acceleration"][index] = np.sum(
                density[:, :, None]
                * delta
                / distance[:, :, None] ** 3
                * volume_weights[:, :, None],
                axis=(0, 1),
            )
            tensor = (
                3.0
                * delta[:, :, :, None]
                * delta[:, :, None, :]
                / distance[:, :, None, None] ** 5
                - eye[None, None, :, :] / distance[:, :, None, None] ** 3
            )
            result["gradient"][index] = np.sum(
                density[:, :, None, None] * tensor * volume_weights[:, :, None, None],
                axis=(0, 1),
            )
    return result


def _surface_wall_observables(
    targets_m: Array,
    *,
    radius_m: float,
    density_kg_m3: float,
    k: float,
    displacement_m: complex,
    direction: Array,
    polarization: Array,
    order: int,
    boundary_fraction: float = 2.0e-7,
) -> dict[str, Array]:
    """Directly integrate the displaced spherical wall on both sides."""
    targets = _finite_targets(targets_m)
    normals, solid_angle_weights = _sphere_quadrature(order)
    surface_weights = radius_m**2 * solid_angle_weights
    points = radius_m * normals
    sigma = (
        -density_kg_m3
        * displacement_m
        * (normals @ polarization)
        * np.exp(1j * k * (points @ direction))
    )
    result = _zero_observables(len(targets))
    eye = np.eye(3)
    for index, target in enumerate(targets):
        delta = points - target
        distance = np.linalg.norm(delta, axis=-1)
        if (
            abs(float(np.linalg.norm(target)) - radius_m)
            <= boundary_fraction * radius_m
        ):
            raise ValueError("target is too close to the spherical cavity wall")
        weights = sigma * surface_weights
        result["potential"][index] = -np.sum(weights / distance)
        result["acceleration"][index] = np.sum(
            weights[:, None] * delta / distance[:, None] ** 3, axis=0
        )
        tensor = (
            3.0 * delta[:, :, None] * delta[:, None, :] / distance[:, None, None] ** 5
            - eye[None, :, :] / distance[:, None, None] ** 3
        )
        result["gradient"][index] = np.sum(weights[:, None, None] * tensor, axis=0)
    return result


def _full_space_bulk_observables(
    targets_m: Array,
    *,
    density_kg_m3: float,
    k: float,
    displacement_m: complex,
    direction: Array,
    projection: float,
) -> dict[str, Array]:
    targets = _finite_targets(targets_m)
    result = _zero_observables(len(targets))
    if projection == 0.0:
        return result
    phase = np.exp(1j * k * (targets @ direction))
    amplitude = 4.0 * np.pi * density_kg_m3 * displacement_m * projection
    result["potential"] = 1j * amplitude / k * phase
    result["acceleration"] = amplitude * phase[:, None] * direction[None, :]
    result["gradient"] = (
        1j
        * k
        * amplitude
        * phase[:, None, None]
        * direction[None, :, None]
        * direction[None, None, :]
    )
    return result


def _cavity_observables_at_order(
    targets_m: Array,
    *,
    radius_m: float,
    density_kg_m3: float,
    wavenumber_rad_m: float,
    displacement_m: complex,
    propagation_direction: Array,
    polarization: Array,
    wave_type: str,
    order: int,
) -> dict[str, dict[str, Array]]:
    """Return bulk, wall, and total fields for general non-wall targets."""
    k, displacement, direction, pol, projection = _plane_wave_parameters(
        wave_type,
        wavenumber_rad_m,
        displacement_m,
        propagation_direction,
        polarization,
    )
    targets = _finite_targets(targets_m)
    removed = _volume_sphere_observables(
        targets,
        radius_m=radius_m,
        density_kg_m3=density_kg_m3,
        k=k,
        displacement_m=displacement,
        direction=direction,
        projection=projection,
        order=order,
    )
    wall = _surface_wall_observables(
        targets,
        radius_m=radius_m,
        density_kg_m3=density_kg_m3,
        k=k,
        displacement_m=displacement,
        direction=direction,
        polarization=pol,
        order=order,
    )
    full = _full_space_bulk_observables(
        targets,
        density_kg_m3=density_kg_m3,
        k=k,
        displacement_m=displacement,
        direction=direction,
        projection=projection,
    )
    bulk = {name: full[name] - removed[name] for name in full}
    total = _add(bulk, wall)
    result = {
        "bulk": {name: _readonly(value) for name, value in bulk.items()},
        "wall": {name: _readonly(value) for name, value in wall.items()},
        "total": {name: _readonly(value) for name, value in total.items()},
    }
    _require_finite_observables(result)
    return result


def _targetwise_difference(
    left: dict[str, Array], right: dict[str, Array]
) -> tuple[dict[str, Array], dict[str, Array]]:
    """Return relative and absolute order differences for each target."""
    relative: dict[str, Array] = {}
    absolute: dict[str, Array] = {}
    for name in ("potential", "acceleration", "gradient"):
        delta = np.asarray(left[name] - right[name])
        if delta.ndim == 1:
            absolute_difference = np.abs(delta)
            reference_norm = np.abs(right[name])
        else:
            component_axes = tuple(range(1, delta.ndim))
            absolute_difference = np.linalg.norm(delta, axis=component_axes)
            reference_norm = np.linalg.norm(right[name], axis=component_axes)
        absolute[name] = absolute_difference
        relative[name] = absolute_difference / np.maximum(
            reference_norm, np.finfo(float).tiny
        )
    return relative, absolute


def cavity_observables_g_free(
    targets_m: Array,
    *,
    radius_m: float,
    density_kg_m3: float,
    wavenumber_rad_m: float,
    displacement_m: complex,
    propagation_direction: Array,
    polarization: Array,
    wave_type: str,
    order: int = 32,
    convergence_rtol: dict[str, float] | None = None,
    convergence_atol: dict[str, float] | None = None,
) -> dict[str, dict[str, Array]]:
    """Return a checked general-point reference or raise when it is unresolved.

    The raw order-specific calculation remains private so an under-resolved
    near-wall diagnostic cannot be mistaken for a validated observable.
    """
    targets = _finite_targets(targets_m)
    try:
        radius = float(radius_m)
        density = float(density_kg_m3)
    except (TypeError, ValueError) as exc:
        raise TypeError("radius_m and density_kg_m3 must be real scalars") from exc
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius_m must be positive and finite")
    if not np.isfinite(density):
        raise ValueError("density_kg_m3 must be finite")
    if isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer)):
        raise TypeError("order must be an integer")
    if order < 2:
        raise ValueError("order must be at least 2")

    tolerances = {"potential": 1.0e-8, "acceleration": 1.0e-7, "gradient": 1.0e-6}
    absolute_tolerances = {
        "potential": 1.0e-12,
        "acceleration": 1.0e-12,
        "gradient": 1.0e-11,
    }
    if convergence_rtol is not None:
        tolerances.update(convergence_rtol)
    if convergence_atol is not None:
        absolute_tolerances.update(convergence_atol)
    unknown = (set(tolerances) | set(absolute_tolerances)) - {
        "potential",
        "acceleration",
        "gradient",
    }
    if unknown:
        raise ValueError(f"unknown convergence observable names: {sorted(unknown)}")
    if any(value < 0.0 or not np.isfinite(value) for value in tolerances.values()):
        raise ValueError("convergence_rtol values must be finite and non-negative")
    if any(
        value < 0.0 or not np.isfinite(value) for value in absolute_tolerances.values()
    ):
        raise ValueError("convergence_atol values must be finite and non-negative")
    low = _cavity_observables_at_order(
        targets,
        radius_m=radius,
        density_kg_m3=density,
        wavenumber_rad_m=wavenumber_rad_m,
        displacement_m=displacement_m,
        propagation_direction=propagation_direction,
        polarization=polarization,
        wave_type=wave_type,
        order=order,
    )
    high = _cavity_observables_at_order(
        targets,
        radius_m=radius,
        density_kg_m3=density,
        wavenumber_rad_m=wavenumber_rad_m,
        displacement_m=displacement_m,
        propagation_direction=propagation_direction,
        polarization=polarization,
        wave_type=wave_type,
        order=2 * order,
    )
    failures = {}
    for term in ("bulk", "wall", "total"):
        relative, absolute = _targetwise_difference(low[term], high[term])
        for name in ("potential", "acceleration", "gradient"):
            relative_value = relative[name]
            absolute_value = absolute[name]
            bad = (
                ~np.isfinite(relative_value)
                | ~np.isfinite(absolute_value)
                | (
                    (relative_value > tolerances[name])
                    & (absolute_value > absolute_tolerances[name])
                )
            )
            if np.any(bad):
                indices = np.flatnonzero(bad).tolist()
                failures[f"{term}.{name}"] = {
                    "target_indices": indices,
                    "max_relative": float(np.nanmax(relative_value[bad])),
                    "max_absolute": float(np.nanmax(absolute_value[bad])),
                }
    if failures:
        raise ValueError(
            "spherical-cavity quadrature did not converge; "
            f"increase order or move the target from the wall: {failures}"
        )
    return high


def _j0(x: float) -> float:
    return float(np.sin(x) / x) if x != 0.0 else 1.0


def _j1_over_x(x: float) -> float:
    return float(np.sin(x) / x**3 - np.cos(x) / x**2) if x != 0.0 else 1.0 / 3.0


def center_closed_forms(
    *, radius_m: float, density_kg_m3: float, k: float, displacement_m: complex
) -> dict[str, complex]:
    x = k * radius_m
    bulk = 4.0 * np.pi * density_kg_m3 * displacement_m * _j0(x)
    wall_p = (
        -4.0 * np.pi * density_kg_m3 * displacement_m * (_j0(x) - 2.0 * _j1_over_x(x))
    )
    wall_s = -4.0 * np.pi * density_kg_m3 * displacement_m * _j1_over_x(x)
    return {"bulk_p": bulk, "wall_p": wall_p, "wall_s": wall_s}


def _relative_difference(
    left: dict[str, Array], right: dict[str, Array]
) -> dict[str, float]:
    result = {}
    for name in ("potential", "acceleration", "gradient"):
        numerator = np.linalg.norm(left[name] - right[name])
        denominator = max(np.linalg.norm(right[name]), np.finfo(float).tiny)
        result[name] = float(numerator / denominator)
    return result


def _complex_json(value: Array) -> list:
    array = np.asarray(value, dtype=np.complex128)
    return np.stack((array.real, array.imag), axis=-1).tolist()


def run(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    radius = 1.0
    density = 2700.0
    k = 0.8
    displacement = 1.0e-6 + 0.4e-6j
    targets = np.array(
        (
            (0.0, 0.0, 0.0),
            (0.2, -0.1, 0.1),
            (0.0, 0.0, 0.99),
            (0.0, 0.0, 1.01),
            (1.4, 0.2, -0.1),
        ),
        dtype=np.float64,
    )
    cases = {}
    for wave_type, polarization in (
        ("P", np.array((0.0, 0.0, 1.0))),
        ("S", np.array((1.0, 0.0, 0.0))),
    ):
        low = _cavity_observables_at_order(
            targets,
            radius_m=radius,
            density_kg_m3=density,
            wavenumber_rad_m=k,
            displacement_m=displacement,
            propagation_direction=np.array((0.0, 0.0, 1.0)),
            polarization=polarization,
            wave_type=wave_type,
            order=32,
        )
        high = _cavity_observables_at_order(
            targets,
            radius_m=radius,
            density_kg_m3=density,
            wavenumber_rad_m=k,
            displacement_m=displacement,
            propagation_direction=np.array((0.0, 0.0, 1.0)),
            polarization=polarization,
            wave_type=wave_type,
            order=64,
        )
        cases[wave_type] = {
            "order32_vs_order64": {
                term: _relative_difference(low[term], high[term])
                for term in ("bulk", "wall", "total")
            },
            "trace_order64": {
                term: _complex_json(np.trace(high[term]["gradient"], axis1=1, axis2=2))
                for term in ("bulk", "wall", "total")
            },
        }
        center = _cavity_observables_at_order(
            np.array(((0.0, 0.0, 0.0),)),
            radius_m=radius,
            density_kg_m3=density,
            wavenumber_rad_m=k,
            displacement_m=displacement,
            propagation_direction=np.array((0.0, 0.0, 1.0)),
            polarization=polarization,
            wave_type=wave_type,
            order=64,
        )
        cases[wave_type]["center_acceleration"] = _complex_json(
            center["total"]["acceleration"][0]
        )
    result = {
        "classification": "Observed independent direct Newton general-point reference",
        "radius_m": radius,
        "density_kg_m3": density,
        "wavenumber_rad_m": k,
        "displacement_m": [displacement.real, displacement.imag],
        "targets_m": targets.tolist(),
        "cases": cases,
        "center_closed_forms": {
            key: [value.real, value.imag]
            for key, value in center_closed_forms(
                radius_m=radius,
                density_kg_m3=density,
                k=k,
                displacement_m=displacement,
            ).items()
        },
        "note": (
            "The unbounded bulk particular solution fixes a potential gauge; the "
            "reported acceleration and gradient are gauge independent. Targets "
            "within 2e-7 radius of the wall are rejected."
        ),
    }
    path = output_dir / "wave6_spherical_cavity_general_point.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.output_dir))


if __name__ == "__main__":
    main()
