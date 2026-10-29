"""Independent oracles for the P6 scientific validation (V1-V8).

Nothing in this module imports ``gragra``. The gravitational constant is
written out locally (CODATA 2018 value), the point-mass field and the rigid
pose are re-implemented in ``mpmath``, and quadrature is composite
Gauss-Legendre whose nodes are computed here by Newton iteration on ``P_n``.

Conventions (identical to the production contract, re-derived here):

* Relative time ``t`` in seconds, ``Delta = epoch - tref`` in seconds.
* Periodic coefficient ``C_n = (1/T) * int_0^T q(t) exp(-2 pi i n (Delta+t)/T) dt``.
* Window transform ``X(f) = int_{t0}^{t1} q(t) exp(-2 pi i f (Delta+t)) dt``.
* Point field at target ``x`` from source ``s``: ``Delta_s = s - x``,
  ``r = |Delta_s|``; potential ``-G m / r``, acceleration ``G m Delta_s / r^3``,
  gradient ``d a_i / d x_j = G m (3 Delta_i Delta_j / r^5 - delta_ij / r^3)``.
"""

from __future__ import annotations

import mpmath as mp
import numpy as np

G_CODATA_STR = "6.67430e-11"
"""CODATA 2018 value, written locally so the oracle does not depend on gragra."""

_GL_CACHE: dict[tuple[int, int], tuple[list, list]] = {}


def gravitational_constant() -> mp.mpf:
    """Return G as an mpmath number at the current precision."""
    return mp.mpf(G_CODATA_STR)


def gauss_legendre(order: int) -> tuple[list, list]:
    """Nodes and weights of the ``order``-point Gauss-Legendre rule on [-1, 1].

    Computed at the current ``mp.dps`` by Newton iteration on ``P_order``.
    """
    key = (order, int(mp.mp.dps))
    if key in _GL_CACHE:
        return _GL_CACHE[key]
    nodes: list = []
    weights: list = []
    tol = mp.mpf(10) ** (-(mp.mp.dps + 5))
    for i in range(1, order + 1):
        x = mp.cos(mp.pi * (i - mp.mpf(1) / 4) / (order + mp.mpf(1) / 2))
        for _ in range(200):
            p_prev, p_curr = mp.mpf(1), x
            for k in range(2, order + 1):
                p_prev, p_curr = (
                    p_curr,
                    ((2 * k - 1) * x * p_curr - (k - 1) * p_prev) / k,
                )
            dp = order * (x * p_curr - p_prev) / (x * x - 1)
            dx = p_curr / dp
            x -= dx
            if abs(dx) < tol:
                break
        p_prev, p_curr = mp.mpf(1), x
        for k in range(2, order + 1):
            p_prev, p_curr = p_curr, ((2 * k - 1) * x * p_curr - (k - 1) * p_prev) / k
        dp = order * (x * p_curr - p_prev) / (x * x - 1)
        nodes.append(x)
        weights.append(2 / ((1 - x * x) * dp * dp))
    _GL_CACHE[key] = (nodes, weights)
    return nodes, weights


def composite_nodes(a, b, panels: int, order: int = 20) -> tuple[list, list]:
    """Return composite Gauss-Legendre sample points and weights on [a, b]."""
    xs, ws = gauss_legendre(order)
    a = mp.mpf(a)
    b = mp.mpf(b)
    h = (b - a) / panels
    points: list = []
    weights: list = []
    for p in range(panels):
        mid = a + (p + mp.mpf(1) / 2) * h
        for x, w in zip(xs, ws, strict=True):
            points.append(mid + x * h / 2)
            weights.append(w * h / 2)
    return points, weights


# ---------------------------------------------------------------------------
# Rigid pose and point-mass field (mpmath)
# ---------------------------------------------------------------------------


def rot_z(theta) -> list[list]:
    """Rotation matrix about the z axis by ``theta`` (mpmath)."""
    c, s = mp.cos(theta), mp.sin(theta)
    return [[c, -s, mp.mpf(0)], [s, c, mp.mpf(0)], [mp.mpf(0), mp.mpf(0), mp.mpf(1)]]


def mat_vec(m, v) -> list:
    return [sum(m[i][j] * v[j] for j in range(3)) for i in range(3)]


def field_components(sources, target, observable: str) -> list:
    """Field of a set of point masses at a static target, in mpmath.

    ``sources`` is a list of ``(mass_kg, position_m)`` with mpmath positions.
    Returns the observable components flattened in the production order:
    ``potential`` -> [Phi]; ``acceleration`` -> [a_x, a_y, a_z];
    ``gradient`` -> row-major 3x3.
    """
    g = gravitational_constant()
    if observable == "potential":
        total = mp.mpf(0)
        for mass, pos in sources:
            d = [pos[i] - target[i] for i in range(3)]
            total += -g * mass / mp.sqrt(sum(c * c for c in d))
        return [total]
    if observable == "acceleration":
        acc = [mp.mpf(0)] * 3
        for mass, pos in sources:
            d = [pos[i] - target[i] for i in range(3)]
            r = mp.sqrt(sum(c * c for c in d))
            acc = [acc[i] + g * mass * d[i] / r**3 for i in range(3)]
        return acc
    if observable == "gradient":
        grad = [[mp.mpf(0)] * 3 for _ in range(3)]
        for mass, pos in sources:
            d = [pos[i] - target[i] for i in range(3)]
            r2 = sum(c * c for c in d)
            r = mp.sqrt(r2)
            for i in range(3):
                for j in range(3):
                    delta = mp.mpf(1) if i == j else mp.mpf(0)
                    grad[i][j] += g * mass * (3 * d[i] * d[j] / r**5 - delta / r**3)
        return [grad[i][j] for i in range(3) for j in range(3)]
    raise ValueError(f"unknown observable {observable!r}")


def body_sources(t, body: list, rot_fn, trans_fn) -> list:
    """Place body-frame points at time ``t`` via ``x = R(t) x_b + c(t)``."""
    rot = rot_fn(t)
    shift = trans_fn(t)
    rotated = [mat_vec(rot, p) for p in body]
    return [[r[i] + shift[i] for i in range(3)] for r in rotated]


def periodic_coefficients(
    series,
    period,
    delta,
    orders,
    *,
    panels: int = 64,
    order: int = 20,
) -> list:
    """Return ``C_n`` for each ``n`` in ``orders`` (mpmath, composite GL).

    ``series(t)`` returns the scalar component at relative time ``t`` (mp).
    """
    period = mp.mpf(period)
    delta = mp.mpf(delta)
    points, weights = composite_nodes(0, period, panels, order)
    values = [series(t) for t in points]
    out = []
    for n in orders:
        total = mp.mpc(0)
        for t, w, q in zip(points, weights, values, strict=True):
            total += w * q * mp.exp(-2j * mp.pi * n * (delta + t) / period)
        out.append(total / period)
    return out


def window_transform(
    series,
    t0,
    t1,
    freqs,
    delta,
    *,
    panels: int = 400,
    order: int = 20,
) -> list:
    """Return ``X(f)`` for each frequency in ``freqs`` on ``[t0, t1]`` (mpmath)."""
    delta = mp.mpf(delta)
    points, weights = composite_nodes(t0, t1, panels, order)
    values = [series(t) for t in points]
    out = []
    for f in freqs:
        total = mp.mpc(0)
        for t, w, q in zip(points, weights, values, strict=True):
            total += w * q * mp.exp(-2j * mp.pi * mp.mpf(f) * (delta + t))
        out.append(total)
    return out


def circular_centre_closed_form(delta, period_s: float = 1.0) -> list[complex]:
    """Closed form for the circular fixture (radius 1 m, period 1 s).

    One 1 kg source at ``c(t) = (cos 2 pi t, sin 2 pi t, 0)``, target at the
    origin, identity rotation. Then ``a(t) = G c(t)`` and the only nonzero
    coefficient is ``A_1 = G exp(-2 pi i delta) (1, -i, 0)`` with ``A_n = 2 C_n``.
    Returns ``[A_0, A_1, A_2, A_3]`` for the acceleration components as
    ``(3,)`` complex lists, in SI units (m/s^2).
    """
    if period_s != 1.0:
        raise ValueError("closed form is written for period 1 s only")
    g = float(mp.mpf(G_CODATA_STR))
    phase = complex(mp.exp(-2j * mp.pi * mp.mpf(delta)))
    a1 = [g * phase, -1j * g * phase, 0j]
    zero = [0j, 0j, 0j]
    return [zero, a1, zero, zero]


def scale_si(mass_kg: float, length_m: float, power: int) -> float:
    """Normalisation scale ``S = G M / L^p`` as a float (SI)."""
    return float(gravitational_constant() * mp.mpf(mass_kg) / mp.mpf(length_m) ** power)


# ---------------------------------------------------------------------------
# Rigid body helpers (M1): Rodrigues rotation written out here, not via scipy
# ---------------------------------------------------------------------------


def rodrigues_numpy(rotvec) -> np.ndarray:
    """Rotation matrix (float64) for a rotation vector whose norm is the angle."""
    v = np.asarray(rotvec, dtype=np.float64)
    theta = float(np.linalg.norm(v))
    eye = np.eye(3)
    if theta == 0.0:
        return eye
    k = v / theta
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return eye + np.sin(theta) * kx + (1.0 - np.cos(theta)) * (kx @ kx)


def rodrigues_mp(rotvec) -> list[list]:
    """Rotation matrix (mpmath) for a rotation vector of mpmath numbers."""
    theta = mp.sqrt(sum(c * c for c in rotvec))
    if theta == 0:
        return [[mp.mpf(int(i == j)) for j in range(3)] for i in range(3)]
    k = [c / theta for c in rotvec]
    kx = [[mp.mpf(0), -k[2], k[1]], [k[2], mp.mpf(0), -k[0]], [-k[1], k[0], mp.mpf(0)]]
    s, c = mp.sin(theta), mp.cos(theta)
    out = []
    for i in range(3):
        row = []
        for j in range(3):
            kk = sum(kx[i][m] * kx[m][j] for m in range(3))
            delta = mp.mpf(1) if i == j else mp.mpf(0)
            row.append(delta + s * kx[i][j] + (1 - c) * kk)
        out.append(row)
    return out


def rigid_field_numpy(
    masses, body, rotation, centre, targets, observable
) -> np.ndarray:
    """Field of a rigid point-mass body at one instant (float64, independent).

    Body points are placed by ``x = R x_b + c``. Returns ``(n_target, n_comp)``
    with components in the production order: potential ``[Phi]``, acceleration
    ``[a_x, a_y, a_z]``, gradient row-major ``d a_i / d x_j``.
    """
    g = float(G_CODATA_STR)
    m = np.asarray(masses, dtype=np.float64)
    pos = np.asarray(body, dtype=np.float64) @ np.asarray(rotation).T + centre
    out = []
    for x in np.asarray(targets, dtype=np.float64):
        d = pos - x
        r = np.linalg.norm(d, axis=1)
        if observable == "potential":
            out.append([float(np.sum(-g * m / r))])
        elif observable == "acceleration":
            out.append(np.sum((g * m / r**3)[:, None] * d, axis=0))
        elif observable == "gradient":
            w = g * m
            tensor = np.einsum("k,ki,kj->ij", 3.0 * w / r**5, d, d)
            tensor -= np.eye(3) * np.sum(w / r**3)
            out.append(tensor.reshape(9))
        else:
            raise ValueError(f"unknown observable {observable!r}")
    return np.asarray(out, dtype=np.float64)


def periodic_coefficients_all(
    field, period, orders, *, delta=0, panels: int = 64, order: int = 20
) -> list[list]:
    """``C_n`` for every ``n`` in ``orders`` of a vector-valued periodic signal.

    ``field(t)`` returns a list of mpmath values (all components at time ``t``).
    The quadrature samples are evaluated once and reused for every order.
    """
    period = mp.mpf(period)
    delta = mp.mpf(delta)
    points, weights = composite_nodes(0, period, panels, order)
    values = [field(t) for t in points]
    ncomp = len(values[0])
    out = []
    for n in orders:
        acc = [mp.mpc(0)] * ncomp
        for t, w, q in zip(points, weights, values, strict=True):
            ph = w * mp.exp(-2j * mp.pi * n * (delta + t) / period)
            acc = [acc[k] + ph * q[k] for k in range(ncomp)]
        out.append([x / period for x in acc])
    return out
