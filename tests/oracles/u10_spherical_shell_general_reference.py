"""Independent finite-volume and two-surface Newton integration for U10.

No production fields, jets, Bessel approximations or series are imported.
Fixed-order Gaussian angular/radial integration is verification only.
Refinement is checked by the validation runner, independently per observable.
Material gradients include the isotropic distributional local term.
"""

import numpy as np

_PARTS = ("bulk", "inner_wall", "outer_boundary", "total")


def _frame(axis):
    seed = np.eye(3)[np.argmin(abs(axis))]
    first = np.cross(axis, seed)
    first /= np.linalg.norm(first)
    return first, np.cross(axis, first)


def _angles(axis, nphi):
    first, second = _frame(axis)
    phi = 2 * np.pi * np.arange(nphi) / nphi
    return np.cos(phi)[:, None] * first + np.sin(phi)[:, None] * second


def _kernel(vector):
    distance = np.sqrt(np.sum(vector * vector, axis=-1))
    direction = vector / distance[..., None]
    tensor = 3 * direction[..., :, None] * direction[..., None, :] - np.eye(3)
    return np.concatenate(
        (
            -1 / distance[..., None],
            direction / distance[..., None] ** 2,
            (tensor / distance[..., None, None] ** 3).reshape(*distance.shape, 9),
        ),
        axis=-1,
    )


def _integrate_mu(function, points, order):
    total = np.zeros(13, dtype=np.clongdouble)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes = nodes.astype(np.longdouble)
    weights = weights.astype(np.longdouble)
    for lo, hi in zip(points[:-1], points[1:], strict=True):
        if hi <= lo:
            continue
        for node, weight in zip(nodes, weights, strict=True):
            angle = np.longdouble(np.pi) * (node + 1) / 4
            mu = lo + (hi - lo) * np.sin(angle) ** 2
            jac = (hi - lo) * np.longdouble(np.pi) / 4 * np.sin(2 * angle)
            total += weight * function(mu) * jac
    return total, {"angular_order": order, "intervals": len(points) - 1}


def _volume(point, a, q, d, nphi, nradial, nmu):
    r = np.sqrt(np.sum(point * point))
    axis = point / r if r else np.array([0.0, 0.0, 1.0], dtype=np.longdouble)
    circle = _angles(axis, nphi)
    nodes, weights = np.polynomial.legendre.leggauss(nradial)
    nodes = np.asarray(nodes, dtype=np.longdouble)
    weights = np.asarray(weights, dtype=np.longdouble)
    material = a < r < 1
    epsilon = min(r - a, 1 - r) / 2 if material else None
    points = [np.longdouble(-1), np.longdouble(0), np.longdouble(1)]
    if r > a:
        points.append(-np.sqrt(1 - (a / r) ** 2))
    if r > 1:
        points.append(-np.sqrt(1 - 1 / r**2))
    points = sorted(set(points))

    def ray(mu):
        directions = mu * axis + np.sqrt(max(np.longdouble(0), 1 - mu * mu)) * circle
        alpha = q * (directions @ d)
        outer_disc = 1 - r * r * (1 - mu * mu)
        if outer_disc <= 0:
            return np.zeros(13, dtype=np.clongdouble)
        root = np.sqrt(outer_disc)
        upper = -r * mu + root
        lower = max(np.longdouble(0), -r * mu - root)
        if upper <= lower:
            return np.zeros(13, dtype=np.clongdouble)
        intervals = [(lower, upper)]
        inner_disc = a * a - r * r * (1 - mu * mu)
        if inner_disc > 0:
            root_a = np.sqrt(inner_disc)
            left, right = -r * mu - root_a, -r * mu + root_a
            if right > lower and left < upper:
                intervals = []
                if left > lower:
                    intervals.append((lower, min(left, upper)))
                if right < upper:
                    intervals.append((max(right, lower), upper))
        radial = np.zeros((nphi, 3), dtype=np.clongdouble)
        for low, high in intervals:
            if high <= low:
                continue
            distance = low + (high - low) * (nodes + 1) / 2
            w = weights * (high - low) / 2
            phase = np.expm1(1j * alpha[:, None] * distance)
            radial[:, 0] += np.sum(w * distance * phase, axis=1)
            radial[:, 1] += np.sum(w * phase, axis=1)
            radial[:, 2] += np.sum(w * phase / distance, axis=1)
        tensor = 3 * directions[:, :, None] * directions[:, None, :] - np.eye(3)
        values = np.concatenate(
            (
                -radial[:, 0, None],
                radial[:, 1, None] * directions,
                (radial[:, 2, None, None] * tensor).reshape(nphi, 9),
            ),
            axis=1,
        )
        return np.sum(values, axis=0) * (2 * np.longdouble(np.pi) / nphi)

    result, errors = _integrate_mu(ray, points, nmu)
    # Integrate delta_rho(s)-delta_rho(x) numerically, add the exact finite
    # constant-density shell. Its material gradient contains the local
    # -(4*pi/3) I term plus the trace-free field of the removed inner ball.
    # This is global constant subtraction, equivalent to the local-ball split
    # but without log(radius/epsilon) cancellation in the numerical integral.
    pi = np.longdouble(np.pi)
    uniform = np.zeros(13, dtype=np.clongdouble)
    if r < a:
        uniform[0] = -2 * pi * (1 - a) * (1 + a)
    elif material:
        uniform[0] = (
            -4
            * pi
            * ((r - a) * (r * r + r * a + a * a) / (3 * r) + (1 - r) * (1 + r) / 2)
        )
        uniform[1:4] = -4 * pi / 3 * point * (r - a) * (r * r + r * a + a * a) / r**3
        uniform[4:] = (
            -4 * pi / 3 * np.eye(3)
            - 4 * pi / 3 * (a / r) ** 3 * (3 * np.outer(axis, axis) - np.eye(3))
        ).ravel()
    else:
        mass = 4 * pi / 3 * (1 - a) * (1 + a + a * a)
        uniform[0] = -mass / r
        uniform[1:4] = -mass * axis / r**2
        uniform[4:] = (mass / r**3 * (3 * np.outer(axis, axis) - np.eye(3))).ravel()
    result += uniform
    return result, {
        "angular_quadrature": errors,
        "epsilon": None if epsilon is None else float(epsilon),
        "local_gradient_coefficient": -4 * np.pi / 3 if material else 0.0,
        "constant_shell_subtracted": True,
    }


def _low_surface(point, h, q, d, p):
    """Exact dipole and quadratic-density control variates, independently derived."""
    pi = np.longdouble(np.pi)
    r = np.sqrt(point @ point)
    identity = np.eye(3, dtype=np.longdouble)
    projection = d @ p
    tensor = (np.outer(d, p) + np.outer(p, d)) / 2 - projection * identity / 3
    quad = point @ tensor @ point
    vec = tensor @ point
    linear = p @ point
    dipole = np.zeros(13, dtype=np.clongdouble)
    quadratic = dipole.copy()
    if r < h:
        dipole[0] = -4 * pi / 3 * linear
        dipole[1:4] = 4 * pi / 3 * p
        # Include i*q*h now to cancel h in the quadrupole gradient exactly.
        quadratic[0] = -4 * pi * 1j * q * (h * h * projection / 3 + quad / 5)
        quadratic[1:4] = 8 * pi * 1j * q / 5 * vec
        quadratic[4:] = (8 * pi * 1j * q / 5 * tensor).ravel()
    else:
        coefficient = 4 * pi / 3 * h**3
        dipole[0] = -coefficient * linear / r**3
        dipole[1:4] = coefficient * (p / r**3 - 3 * linear * point / r**5)
        dipole[4:] = (
            coefficient
            * (
                -3
                * (np.outer(p, point) + np.outer(point, p) + linear * identity)
                / r**5
                + 15 * linear * np.outer(point, point) / r**7
            )
        ).ravel()
        mass = 4 * pi * h * h * projection / 3
        coefficient = 4 * pi * h**4 / 5
        quadratic[0] = -mass / r - coefficient * quad / r**5
        quadratic[1:4] = -mass * point / r**3 + coefficient * (
            2 * vec / r**5 - 5 * quad * point / r**7
        )
        quadratic[4:] = (
            mass * (3 * np.outer(point, point) / r**5 - identity / r**3)
            + coefficient
            * (
                2 * tensor / r**5
                - 10 * (np.outer(vec, point) + np.outer(point, vec)) / r**7
                - 5 * quad * identity / r**7
                + 35 * quad * np.outer(point, point) / r**9
            )
        ).ravel()
        quadratic *= 1j * q * h
    return dipole + quadratic


def _surface(point, h, q, d, p, nphi):
    r = np.sqrt(np.sum(point * point))
    axis = point / r if r else np.array([0.0, 0.0, 1.0], dtype=np.longdouble)
    circle = _angles(axis, nphi)
    nearest = h * axis
    phase0 = np.exp(1j * q * (d @ nearest))
    sigma0 = (p @ axis) * phase0
    nodes, weights = np.polynomial.legendre.leggauss(nphi)
    nodes = nodes.astype(np.longdouble)
    weights = weights.astype(np.longdouble)
    if r <= h / 2 or r >= 1.8 * h:
        # Common source-sphere nodes preserve correlated inner/outer errors.
        # Longdouble accumulation retains their final finite-shell difference.
        value = np.zeros(13, dtype=np.clongdouble)
        for mu, weight in zip(nodes, weights, strict=True):
            normals = mu * axis + np.sqrt(1 - mu * mu) * circle
            phase = 1j * q * h * (normals @ d)
            density = (normals @ p) * (np.expm1(phase) - phase)
            value += (
                weight
                * np.sum(_kernel(h * normals - point) * density[:, None], axis=0)
                * h
                * h
                * (2 * np.longdouble(np.pi) / nphi)
            )
        value += _low_surface(point, h, q, d, p)
        return value, {
            "angular_order": nphi,
            "constant_sheet_subtracted": False,
            "method": "common_source_sphere_nodes",
        }
    gap = abs(h - r)
    upper = np.log((h + r) / gap)

    def logarithmic(v):
        v = np.longdouble(v)
        distance = gap * np.exp(v)
        one_minus_mu = gap * gap * np.expm1(2 * v) / (2 * h * r)
        sine = np.sqrt(max(np.longdouble(0), one_minus_mu * (2 - one_minus_mu)))
        delta = -one_minus_mu * axis + sine * circle
        vectors = h * delta + (h - r) * axis
        # Subtraction is evaluated before rounding; avoids huge uniform-sheet
        # gradient cancellation close to either material boundary.
        phase_delta = 1j * q * h * (delta @ d)
        density = phase0 * (
            (delta @ p) * np.exp(phase_delta) + (p @ axis) * np.expm1(phase_delta)
        )
        values = np.sum(_kernel(vectors) * density[:, None], axis=0)
        return values * (2 * np.longdouble(np.pi) / nphi) * h / r * distance**2

    value = np.zeros(13, dtype=np.clongdouble)
    edges = np.linspace(0, upper, int(np.ceil(upper)) + 1, dtype=np.longdouble)
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        for node, weight in zip(nodes, weights, strict=True):
            value += (
                weight
                * (high - low)
                / 2
                * logarithmic(low + (high - low) * (node + 1) / 2)
            )
    uniform = np.zeros(13, dtype=np.clongdouble)
    if r < h:
        uniform[0] = -4 * np.longdouble(np.pi) * h
    else:
        mass = 4 * np.longdouble(np.pi) * h * h
        uniform[0] = -mass / r
        uniform[1:4] = -mass * axis / r**2
        uniform[4:] = (mass / r**3 * (3 * np.outer(axis, axis) - np.eye(3))).ravel()
    return np.asarray(value, dtype=np.clongdouble) + sigma0 * uniform, {
        "angular_order": nphi,
        "log_intervals": len(edges) - 1,
        "constant_sheet_subtracted": True,
        "method": "composite_log_distance",
    }


def reference(
    targets_m,
    *,
    inner_radius_m,
    outer_radius_m,
    density_kg_m3,
    wavenumber_rad_m,
    displacement_m,
    propagation_direction,
    polarization,
    wave_type,
    center_m=(0.0, 0.0, 0.0),
    phase_origin_m=(0.0, 0.0, 0.0),
    nphi=48,
    nradial=24,
    nmu=96,
):
    """Scalar-sample reference; outputs raw arrays and independent diagnostics."""
    d = np.asarray(propagation_direction, dtype=np.longdouble)
    d = d / max(abs(d))
    d /= np.sqrt(d @ d)
    p = np.asarray(polarization, dtype=np.longdouble)
    p = p / max(abs(p))
    p /= np.sqrt(p @ p)
    a = np.longdouble(inner_radius_m) / outer_radius_m
    b = np.longdouble(outer_radius_m)
    q = np.longdouble(wavenumber_rad_m) * b
    center = np.asarray(center_m, dtype=np.longdouble)
    origin = np.asarray(phase_origin_m, dtype=np.longdouble)
    amp = (
        np.clongdouble(density_kg_m3)
        * displacement_m
        * np.exp(1j * wavenumber_rad_m * (d @ (center - origin)))
    )
    arrays = {
        part: np.zeros((len(targets_m), 13), dtype=np.clongdouble) for part in _PARTS
    }
    diagnostics = []
    for target, coordinates in enumerate(targets_m):
        point = (np.asarray(coordinates, dtype=np.longdouble) - center) / b
        bulk, volume_diagnostics = (
            _volume(point, a, q, d, nphi, nradial, nmu)
            if wave_type == "P"
            else (np.zeros(13), {"status": "EXACT_ZERO_SOURCE"})
        )
        delta = (
            -1j
            * density_kg_m3
            * wavenumber_rad_m
            * displacement_m
            * (d @ p)
            * np.exp(1j * wavenumber_rad_m * (d @ (np.asarray(coordinates) - origin)))
            if wave_type == "P"
            else 0j
        )
        arrays["bulk"][target] = bulk * delta * np.r_[b * b, np.full(3, b), np.ones(9)]
        for part, h, sign in (
            ("inner_wall", a, -1),
            ("outer_boundary", np.longdouble(1), 1),
        ):
            sheet, surface_diagnostics = _surface(point, h, q, d, p, nphi)
            arrays[part][target] = (
                sign * amp * sheet * np.r_[b, np.ones(3), np.full(9, 1 / b)]
            )
            diagnostics.append(dict(target=target, part=part, **surface_diagnostics))
        arrays["total"][target] = sum(arrays[part][target] for part in _PARTS[:3])
        diagnostics.append(dict(target=target, part="bulk", **volume_diagnostics))
    return arrays, diagnostics
