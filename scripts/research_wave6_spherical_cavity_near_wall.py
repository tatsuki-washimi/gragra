#!/usr/bin/env python3
"""Off-axis near-wall Newton reference pilot for the spherical-cavity series.

This is a research diagnostic, not a production API.  It evaluates the
removed volume and displaced spherical wall directly with the Newton kernel.
The angular domain is split into a local cap around the closest wall point;
the interior volume uses target-centred rays and the exterior volume splits the
radial interval near the wall.  No production series coefficient is used by
the reference calculation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from gragra.analytic import spherical_cavity_observables_g_free

Array = np.ndarray

_GENERAL_PATH = Path(__file__).with_name(
    "research_wave6_spherical_cavity_general_point.py"
)
_SPEC = importlib.util.spec_from_file_location("wave6_general_point", _GENERAL_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import contract
    raise ImportError(f"cannot load {_GENERAL_PATH}")
_GENERAL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_GENERAL)


def _frame(q: Array) -> tuple[Array, Array]:
    reference = (
        np.array([0.0, 0.0, 1.0]) if abs(q[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    )
    first = np.cross(reference, q)
    first /= np.linalg.norm(first)
    return first, np.cross(q, first)


def _cap_breaks(width: float, alpha: float = 0.5) -> list[float]:
    values = [0.0, alpha]
    value = min(alpha, max(2.0 * width, 1.0e-6))
    while value < alpha:
        values.append(value)
        value *= 2.0
    return sorted(set(values))


def _angular_parts(target: Array, radius: float, order: int):
    target_radius = float(np.linalg.norm(target))
    q = target / target_radius if target_radius else np.array([0.0, 0.0, 1.0])
    first, second = _frame(q)
    alpha = 0.5
    n_phi = 2 * order
    phi = 2.0 * np.pi * np.arange(n_phi, dtype=np.float64) / n_phi
    tangent = (
        np.cos(phi)[:, None] * first[None, :] + np.sin(phi)[:, None] * second[None, :]
    )
    legendre_x, legendre_w = np.polynomial.legendre.leggauss(order)

    for left, right in zip(
        _cap_breaks(abs(target_radius - radius) / radius)[:-1],
        _cap_breaks(abs(target_radius - radius) / radius)[1:],
        strict=True,
    ):
        theta = 0.5 * (right - left) * legendre_x + 0.5 * (right + left)
        omega_theta = 0.5 * (right - left) * legendre_w * np.sin(theta)
        normals = (
            np.cos(theta)[:, None, None] * q[None, None, :]
            + np.sin(theta)[:, None, None] * tangent[None, :, :]
        ).reshape(-1, 3)
        omega = (
            omega_theta[:, None] * np.full((1, n_phi), 2.0 * np.pi / n_phi)
        ).ravel()
        yield normals, omega

    cosine = np.cos(alpha)
    mu = 0.5 * (cosine + 1.0) * legendre_x + 0.5 * (cosine - 1.0)
    mu_weight = 0.5 * (cosine + 1.0) * legendre_w
    normals = (
        mu[:, None, None] * q[None, None, :]
        + np.sqrt(1.0 - mu * mu)[:, None, None] * tangent[None, :, :]
    ).reshape(-1, 3)
    omega = (mu_weight[:, None] * np.full((1, n_phi), 2.0 * np.pi / n_phi)).ravel()
    yield normals, omega


def _wall_at_order(
    target: Array,
    *,
    radius: float,
    density: float,
    wavenumber: float,
    displacement: complex,
    direction: Array,
    polarization: Array,
    order: int,
) -> dict[str, Array]:
    result = _GENERAL._zero_observables(1)
    eye = np.eye(3)
    for normals, omega in _angular_parts(target, radius, order):
        points = radius * normals
        delta = points - target
        distance = np.linalg.norm(delta, axis=1)
        sigma = (
            -density
            * displacement
            * (normals @ polarization)
            * np.exp(1j * wavenumber * (points @ direction))
        )
        weights = sigma * radius**2 * omega
        result["potential"][0] += -np.sum(weights / distance)
        result["acceleration"][0] += np.sum(
            weights[:, None] * delta / distance[:, None] ** 3, axis=0
        )
        tensor = (
            3.0 * delta[:, :, None] * delta[:, None, :] / distance[:, None, None] ** 5
            - eye[None, :, :] / distance[:, None, None] ** 3
        )
        result["gradient"][0] += np.sum(weights[:, None, None] * tensor, axis=0)
    return result


def _volume_at_order(
    target: Array,
    *,
    radius: float,
    density: float,
    wavenumber: float,
    displacement: complex,
    direction: Array,
    polarization: Array,
    wave_type: str,
    order: int,
) -> dict[str, Array]:
    target_radius = float(np.linalg.norm(target))
    _, _, direction, polarization, projection = _GENERAL._plane_wave_parameters(
        wave_type, wavenumber, displacement, direction, polarization
    )
    result = _GENERAL._zero_observables(1)
    eye = np.eye(3)
    radial_x, radial_w = np.polynomial.legendre.leggauss(order)
    center_density = _GENERAL._density_perturbation(
        target[None, :],
        density_kg_m3=density,
        k=wavenumber,
        displacement_m=displacement,
        direction=direction,
        projection=projection,
    )[0]

    for normals, omega in _angular_parts(target, radius, order):
        if target_radius < radius:
            dot = normals @ target
            ray_length = -dot + np.sqrt(dot * dot + radius**2 - target_radius**2)
            radii = 0.5 * ray_length[:, None] * (radial_x[None, :] + 1.0)
            radial_weight = 0.5 * ray_length[:, None] * radial_w[None, :]
            points = target[None, None, :] + normals[:, None, :] * radii[:, :, None]
            density_values = _GENERAL._density_perturbation(
                points.reshape(-1, 3),
                density_kg_m3=density,
                k=wavenumber,
                displacement_m=displacement,
                direction=direction,
                projection=projection,
            ).reshape(len(normals), order)
            angular_weight = omega[:, None]
            result["potential"][0] += -np.sum(
                density_values * radii * radial_weight * angular_weight
            )
            result["acceleration"][0] += np.sum(
                density_values[:, :, None]
                * normals[:, None, :]
                * radial_weight[:, :, None]
                * angular_weight[:, :, None],
                axis=(0, 1),
            )
            tensor = (
                3.0 * normals[:, None, :, None] * normals[:, None, None, :]
                - eye[None, None, :, :]
            )
            result["gradient"][0] += np.sum(
                (density_values - center_density)[:, :, None, None]
                * tensor
                / radii[:, :, None, None]
                * radial_weight[:, :, None, None]
                * angular_weight[:, :, None, None],
                axis=(0, 1),
            )
        else:
            gap = abs(target_radius - radius)
            radial_breaks = sorted(
                {0.0, max(0.0, radius - 2.0 * gap), max(0.0, radius - gap), radius}
            )
            for left, right in zip(radial_breaks[:-1], radial_breaks[1:], strict=True):
                if right <= left:
                    continue
                radii = 0.5 * (right - left) * radial_x + 0.5 * (right + left)
                radial_weight = 0.5 * (right - left) * radial_w
                points = radii[:, None, None] * normals[None, :, :]
                delta = points - target[None, None, :]
                distance = np.linalg.norm(delta, axis=-1)
                density_values = _GENERAL._density_perturbation(
                    points.reshape(-1, 3),
                    density_kg_m3=density,
                    k=wavenumber,
                    displacement_m=displacement,
                    direction=direction,
                    projection=projection,
                ).reshape(order, len(normals))
                volume_weight = (
                    radii[:, None] ** 2 * radial_weight[:, None] * omega[None, :]
                )
                result["potential"][0] += -np.sum(
                    density_values * volume_weight / distance
                )
                result["acceleration"][0] += np.sum(
                    density_values[:, :, None]
                    * delta
                    / distance[:, :, None] ** 3
                    * volume_weight[:, :, None],
                    axis=(0, 1),
                )
                tensor = (
                    3.0
                    * delta[:, :, :, None]
                    * delta[:, :, None, :]
                    / distance[:, :, None, None] ** 5
                    - eye[None, None, :, :] / distance[:, :, None, None] ** 3
                )
                result["gradient"][0] += np.sum(
                    density_values[:, :, None, None]
                    * tensor
                    * volume_weight[:, :, None, None],
                    axis=(0, 1),
                )
    if target_radius < radius:
        result["gradient"][0] -= (4.0 * np.pi / 3.0) * center_density * eye
    return result


def _reference(
    target: Array,
    *,
    radius: float,
    density: float,
    wavenumber: float,
    displacement: complex,
    direction: Array,
    polarization: Array,
    wave_type: str,
    order: int,
) -> dict[str, dict[str, Array]]:
    _, _, direction, polarization, projection = _GENERAL._plane_wave_parameters(
        wave_type, wavenumber, displacement, direction, polarization
    )
    volume = _volume_at_order(
        target,
        radius=radius,
        density=density,
        wavenumber=wavenumber,
        displacement=displacement,
        direction=direction,
        polarization=polarization,
        wave_type=wave_type,
        order=order,
    )
    wall = _wall_at_order(
        target,
        radius=radius,
        density=density,
        wavenumber=wavenumber,
        displacement=displacement,
        direction=direction,
        polarization=polarization,
        order=order,
    )
    full = _GENERAL._full_space_bulk_observables(
        np.asarray([target]),
        density_kg_m3=density,
        k=wavenumber,
        displacement_m=displacement,
        direction=direction,
        projection=projection,
    )
    bulk = {name: full[name] - volume[name] for name in full}
    total = {name: bulk[name] + wall[name] for name in full}
    return {"bulk": bulk, "wall": wall, "total": total}


def compare_case(
    target: Array,
    *,
    radius: float,
    density: float,
    wavenumber: float,
    displacement: complex,
    direction: Array,
    polarization: Array,
    wave_type: str,
    low_order: int = 64,
    high_order: int = 128,
) -> dict:
    low = _reference(
        target,
        radius=radius,
        density=density,
        wavenumber=wavenumber,
        displacement=displacement,
        direction=direction,
        polarization=polarization,
        wave_type=wave_type,
        order=low_order,
    )
    high = _reference(
        target,
        radius=radius,
        density=density,
        wavenumber=wavenumber,
        displacement=displacement,
        direction=direction,
        polarization=polarization,
        wave_type=wave_type,
        order=high_order,
    )
    production = spherical_cavity_observables_g_free(
        np.asarray([target]),
        radius_m=radius,
        density_kg_m3=density,
        wavenumber_rad_m=wavenumber,
        displacement_m=displacement,
        propagation_direction=direction,
        polarization=polarization,
        wave_type=wave_type,
        max_degree=128,
    )
    qscale = (
        4.0
        * np.pi
        * density
        * abs(displacement)
        * np.array((radius, 1.0, 1.0 / radius))
    )
    names = ("potential", "acceleration", "gradient")
    stability = {term: {} for term in ("bulk", "wall", "total")}
    error = {term: {} for term in ("bulk", "wall", "total")}
    for term in stability:
        for index, name in enumerate(names):
            stability[term][name] = float(
                np.linalg.norm(high[term][name][0] - low[term][name][0]) / qscale[index]
            )
            error[term][name] = float(
                np.linalg.norm(getattr(production[term], name)[0] - high[term][name][0])
                / qscale[index]
            )
    return {
        "wave_type": wave_type,
        "target_m": np.asarray(target).tolist(),
        "order_pair": [low_order, high_order],
        "qscale_normalized_reference_change": stability,
        "qscale_normalized_production_error": error,
    }


def run_pilot(
    *,
    d_over_a: float = 1.0e-2,
    wavenumber: float = 0.8,
    low_order: int = 64,
    high_order: int = 128,
) -> dict:
    if not np.isfinite(d_over_a) or d_over_a <= 0.0 or d_over_a >= 1.0:
        raise ValueError("d_over_a must be finite and in (0, 1)")
    if not np.isfinite(wavenumber) or wavenumber <= 0.0:
        raise ValueError("wavenumber must be finite and positive")
    if low_order <= 0 or high_order <= low_order:
        raise ValueError("reference orders must be positive and increasing")
    radius = 1.0
    density = 2700.0
    displacement = (1.0 + 0.4j) * 1.0e-6
    direction = np.array([0.0, 0.0, 1.0])
    directions = (
        np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0),
        np.array([2.0, -1.0, 1.0]) / np.sqrt(6.0),
    )
    cases = []
    for wave_type, polarization in (
        ("P", np.array([0.0, 0.0, 1.0])),
        ("S", np.array([1.0, 0.0, 0.0])),
    ):
        for direction_index, radial_direction in enumerate(directions, 1):
            for side, factor in (
                ("inside", 1.0 - d_over_a),
                ("outside", 1.0 + d_over_a),
            ):
                case = compare_case(
                    factor * radial_direction,
                    radius=radius,
                    density=density,
                    wavenumber=wavenumber,
                    displacement=displacement,
                    direction=direction,
                    polarization=polarization,
                    wave_type=wave_type,
                    low_order=low_order,
                    high_order=high_order,
                )
                case["direction_index"] = direction_index
                case["side"] = side
                cases.append(case)
    return {
        "method": (
            "direct Newton volume and wall integrals with local cap and "
            "near-wall radial splitting"
        ),
        "radius_m": radius,
        "density_kg_m3": density,
        "displacement_m": [displacement.real, displacement.imag],
        "wavenumber_rad_m": wavenumber,
        "d_over_a": d_over_a,
        "propagation_direction": direction.tolist(),
        "reference_orders": [low_order, high_order],
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--d-over-a", type=float, default=1.0e-2)
    parser.add_argument("--ka", type=float, default=0.8)
    parser.add_argument("--low-order", type=int, default=64)
    parser.add_argument("--high-order", type=int, default=128)
    args = parser.parse_args()
    result = run_pilot(
        d_over_a=args.d_over_a,
        wavenumber=args.ka,
        low_order=args.low_order,
        high_order=args.high_order,
    )
    text = json.dumps(result, indent=2) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.write_text(text)
        print(args.output)


if __name__ == "__main__":
    main()
