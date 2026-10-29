"""Independent prescribed-field Newton integrals for U11 (validation only).

No gragra imports. Shared data: material, geometry, phase and sign conventions.
Root, eigenfunctions, Newton kernels and surface meshes are independent of
production. Quadrature refinement is empirical, not a rigorous error bound.
"""

from functools import lru_cache

import mpmath as mp
import numpy as np
from scipy.integrate import quad


@lru_cache(maxsize=16)
def elastic_root(vp, vs):
    with mp.workdps(60):
        beta = (mp.mpf(str(vs)) / mp.mpf(str(vp))) ** 2
        # The cubic is the nonzero secular polynomial in (c/vs)^2.
        root = mp.findroot(
            lambda y: y**3 - 8 * y * y + (24 - 16 * beta) * y - 16 * (1 - beta),
            (0.7, 0.95),
        )
        cr = mp.mpf(str(vs)) * mp.sqrt(root)
        q = mp.sqrt(1 - (cr / mp.mpf(str(vp))) ** 2)
        s = mp.sqrt(1 - root)
        assert abs((1 + s * s) ** 2 - 4 * q * s) < mp.mpf("1e-50")
        return float(cr), float(q), float(s)


def wave_parameters(args):
    cr, q, s = elastic_root(args["v_p_m_s"], args["v_s_m_s"])
    k = 2 * np.pi * args["frequency_hz"] / cr
    d = np.asarray(args["direction"], float)
    d = d / np.linalg.norm(d)
    zeta = 2 * q / (1 + s * s)
    amp = args["vertical_surface_amplitude_m"] / (k * (q - zeta))
    return k, q, s, zeta, amp, d


def displacement(points, args):
    k, q, s, zeta, amp, d = wave_parameters(args)
    points = np.asarray(points)
    phase = np.exp(1j * k * ((points - np.asarray(args["phase_origin_m"])) @ d))
    ep = np.exp(k * q * points[..., 2])
    es = np.exp(k * s * points[..., 2])
    out = (1j * k * amp * (ep - s * zeta * es))[..., None] * d
    out[..., 2] = k * amp * (q * ep - zeta * es)
    return out * phase[..., None]


def density_perturbation(points, args):
    k, q, _, _, amp, d = wave_parameters(args)
    return (
        -args["density_kg_m3"]
        * amp
        * k
        * k
        * (q * q - 1)
        * np.exp(k * q * points[..., 2])
        * np.exp(1j * k * ((points - np.asarray(args["phase_origin_m"])) @ d))
    )


def background_integrals(targets, args, tolerance=1e-13):
    k, q, _, _, amp, d = wave_parameters(args)
    rho = args["density_kg_m3"]
    bulk = []
    surface = []
    quad_errors = []
    for x in targets:
        z = k * x[2]
        phase = np.exp(1j * k * np.dot(x - np.asarray(args["phase_origin_m"]), d))
        # Source depth t=k*z_source. Fourier-reduced acceleration kernel:
        # 2*pi exp(-|t-z|) * (i*d_x,i*d_y,sign(t-z)).
        lo, elo = quad(
            lambda t, depth=z: np.exp(q * t - abs(t - depth)),
            -np.inf,
            z,
            epsabs=tolerance,
            epsrel=tolerance,
        )
        hi, ehi = quad(
            lambda t, depth=z: np.exp(q * t - abs(t - depth)),
            z,
            0,
            epsabs=tolerance,
            epsrel=tolerance,
        )
        prefactor = -rho * amp * k * (q * q - 1) * 2 * np.pi * phase
        bulk.append(
            prefactor
            * np.array([1j * d[0] * (lo + hi), 1j * d[1] * (lo + hi), hi - lo])
        )
        surface.append(
            2
            * np.pi
            * rho
            * args["vertical_surface_amplitude_m"]
            * phase
            * np.exp(z)
            * np.array([1j * d[0], 1j * d[1], 1])
        )
        quad_errors.append([elo, ehi])
    return np.asarray(bulk), np.asarray(surface), np.asarray(quad_errors)


def wall_nodes(args, order):
    """Independently parameterize a closed surface with outward body normals."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    phi = 2 * np.pi * np.arange(2 * order) / (2 * order)
    shape = args["cavity_shape"]
    points = []
    normals = []
    areas = []
    if shape == "sphere":
        mu, az = np.meshgrid(nodes, phi, indexing="ij")
        n = np.stack(
            [np.sqrt(1 - mu * mu) * np.cos(az), np.sqrt(1 - mu * mu) * np.sin(az), mu],
            axis=-1,
        )
        points.append(args["radius_m"] * n)
        normals.append(n)
        areas.append(
            np.broadcast_to(
                weights[:, None] * 2 * np.pi / (2 * order) * args["radius_m"] ** 2,
                mu.shape,
            )
        )
    elif shape == "cuboid":
        h = np.asarray(args["half_lengths_m"])
        a, b = np.meshgrid(nodes, nodes, indexing="ij")
        for axis in range(3):
            j, other = [v for v in range(3) if v != axis]
            for sign in (-1, 1):
                p = np.zeros(a.shape + (3,))
                n = np.zeros_like(p)
                p[..., axis] = sign * h[axis]
                p[..., j] = a * h[j]
                p[..., other] = b * h[other]
                n[..., axis] = sign
                points.append(p)
                normals.append(n)
                areas.append(np.outer(weights, weights) * h[j] * h[other])
    else:
        radius = args["radius_m"]
        half = args["height_m"] / 2
        zz, az = np.meshgrid(nodes, phi, indexing="ij")
        n = np.stack([np.cos(az), np.sin(az), np.zeros_like(az)], axis=-1)
        p = radius * n.copy()
        p[..., 2] = half * zz
        points.append(p)
        normals.append(n)
        areas.append(
            np.broadcast_to(
                weights[:, None] * half * radius * 2 * np.pi / (2 * order), zz.shape
            )
        )
        radial = (nodes + 1) * radius / 2
        rr, az = np.meshgrid(radial, phi, indexing="ij")
        for sign in (-1, 1):
            p = np.stack(
                [rr * np.cos(az), rr * np.sin(az), np.full_like(rr, sign * half)],
                axis=-1,
            )
            n = np.zeros_like(p)
            n[..., 2] = sign
            points.append(p)
            normals.append(n)
            areas.append(weights[:, None] * radius / 2 * rr * 2 * np.pi / (2 * order))
    rot = np.asarray(args.get("rotation", np.eye(3)))
    points = (
        np.concatenate([p.reshape(-1, 3) for p in points]) @ rot.T + args["center_m"]
    )
    normals = np.concatenate([n.reshape(-1, 3) for n in normals]) @ rot.T
    return points, normals, np.concatenate([a.ravel() for a in areas])


def removed_volume(target, points, normals, weights, density_function, radial_order):
    """Target-centered pyramid transform with analytic kernel cancellation."""
    delta = points - target
    distance = np.linalg.norm(delta, axis=1)
    projection = np.einsum("ni,ni->n", normals, delta)
    assert np.all(projection > 0), "target must lie in the strict convex interior"
    t, w = np.polynomial.legendre.leggauss(radial_order)
    radial = np.zeros(len(points), complex)
    for node, weight in zip((t + 1) / 2, w / 2, strict=True):
        radial += weight * density_function(target + node * delta)
    return np.einsum(
        "n,ni,n->i", weights, delta / distance[:, None] ** 3, projection * radial
    )


def evaluate(targets, args, order=64, radial_order=20, tolerance=1e-13):
    bulk, surface, quad_errors = background_integrals(targets, args, tolerance)
    points, normals, weights = wall_nodes(args, order)
    uc = displacement(np.asarray(args["center_m"])[None, :], args)[0]
    full_displacement = displacement(points, args)
    sigma_constant = -args["density_kg_m3"] * (normals @ uc)
    sigma_full = -args["density_kg_m3"] * np.einsum(
        "ni,ni->n", normals, full_displacement
    )
    constant = []
    full = []
    removed = []
    for target in targets:
        delta = points - target
        kernel = delta / np.linalg.norm(delta, axis=1)[:, None] ** 3
        constant.append(np.einsum("n,ni,n->i", weights, kernel, sigma_constant))
        full.append(np.einsum("n,ni,n->i", weights, kernel, sigma_full))
        removed.append(
            removed_volume(
                target,
                points,
                normals,
                weights,
                lambda p: density_perturbation(p, args),
                radial_order,
            )
        )
    constant = np.asarray(constant)
    full = np.asarray(full)
    removed = np.asarray(removed)
    background = bulk + surface
    leading = background + constant
    finite = background - removed + full
    variation = full - constant
    model = leading - finite
    return {
        "rock_bulk": bulk,
        "ground_surface": surface,
        "background_halfspace": background,
        "cavity_wall_sca": constant,
        "wall_full": full,
        "removed_bulk": removed,
        "total_leading_sca": leading,
        "finite": finite,
        "wall_variation": variation,
        "model_error": model,
        "closure": model - (removed - variation),
        "u_c": uc,
        "fourier_quad_error": quad_errors,
    }
