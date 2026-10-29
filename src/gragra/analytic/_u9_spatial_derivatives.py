"""Lane-local analytic target derivatives, with no spatial quadrature.

H[i,j,k] = partial_k Gamma[i,j]. See the U9 design for derivations and
switches. NumPy longdouble reduces cancellation but public storage is float64.
"""

from itertools import permutations, product
from math import factorial

import numpy as np

from gragra.analytic.cavity_cylinder import _cel


def cuboid_static_acceleration(half, point):
    """Nagy corner acceleration in extended precision with rationalized logs.

    The existing float64 corner formula can form log(0) on an exterior edge
    extension even though the physical field is finite there.
    """
    ends = np.array([-half - point, half - point], dtype=np.longdouble)

    def safe_log(w, u, v):
        distance = np.sqrt(w * w + u * u + v * v)
        if w < 0:
            return np.log((u * u + v * v) / (distance - w))
        return np.log(w + distance)

    def corner(x, y, z):
        distance = np.sqrt(x * x + y * y + z * z)
        if distance == 0:
            return np.longdouble(0)
        value = np.longdouble(0)
        if x != 0:
            value += x * safe_log(y, x, z)
        if y != 0:
            value += y * safe_log(x, y, z)
        if z != 0:
            value -= z * np.arctan(x * y / (z * distance))
        return value

    result = np.zeros(3, dtype=np.longdouble)
    for indices in product(range(2), repeat=3):
        x, y, z = (ends[index, axis] for axis, index in enumerate(indices))
        sign = (-1) ** sum(indices)
        result += sign * np.array([corner(y, z, x), corner(z, x, y), corner(x, y, z)])
    return np.asarray(result, dtype=np.float64)


def cuboid_third(half, point):
    """Differentiate paired logarithms, including exterior edge extensions."""
    ends = np.array([-half - point, half - point], dtype=np.longdouble)
    result = np.zeros((3, 3, 3), dtype=np.longdouble)
    # Every mixed third derivative is a derivative of an off-diagonal log
    # pair. Pair before differentiating to remove divergent individual logs.
    for i, j in ((0, 1), (0, 2), (1, 2)):
        k = 3 - i - j
        for derivative in (i, j, k):
            total = np.longdouble(0)
            for low_i, low_j in product(range(2), repeat=2):
                x, y = ends[low_i, i], ends[low_j, j]
                w0, w1 = ends[:, k]
                t = x * x + y * y
                r0, r1 = np.sqrt(t + w0 * w0), np.sqrt(t + w1 * w1)
                if derivative == k:
                    value = 1 / r0 - 1 / r1
                else:
                    coordinate = x if derivative == i else y
                    if w1 <= 0:
                        value = coordinate * (
                            1 / (r1 * (r1 - w1)) - 1 / (r0 * (r0 - w0))
                        )
                    elif w0 >= 0:
                        value = coordinate * (
                            1 / (r0 * (r0 + w0)) - 1 / (r1 * (r1 + w1))
                        )
                    else:
                        value = coordinate * (
                            2 / t - 1 / (r0 * (r0 - w0)) - 1 / (r1 * (r1 + w1))
                        )
                total += (-1) ** (low_i + low_j) * value
            for indices in set(permutations((i, j, derivative))):
                result[indices] = total
    # Differentiate the analytic constant-density Poisson identity, not a
    # projection of a computed tensor. It supplies the three pure derivatives.
    for i in range(3):
        result[i, i, i] = -sum(result[j, j, i] for j in range(3) if j != i)
    return np.asarray(result, dtype=np.float64)


def _axis_derivatives(radius, height, z):
    """a_j = d^j a_z(0,z)/dz^j, j=0..13, from endpoint square roots."""
    radius, height, z = map(np.longdouble, (radius, height, z))
    v = np.array([-height / 2 - z, height / 2 - z])
    d = np.sqrt(radius * radius + v * v)
    pi = np.arccos(np.longdouble(-1))
    values = [-radius * radius / (abs(v) + d), v / d - np.sign(v)]
    # d^n sqrt(a²+v²)/dv^n = P_n(v)*(a²+v²)^(1/2-n).
    polynomial = np.array([0.0, 1.0], dtype=np.longdouble)
    for n in range(1, 13):
        differentiated = np.polynomial.polynomial.polyder(polynomial)
        left = np.polynomial.polynomial.polymul(differentiated, [radius**2, 0, 1])
        right = np.polynomial.polynomial.polymul(polynomial, [0, 1 - 2 * n])
        polynomial = np.polynomial.polynomial.polyadd(left, right)
        derivative = np.polynomial.polynomial.polyval(v, polynomial) * d ** (-2 * n - 1)
        values.append((-1) ** (n + 2) * derivative)
    return np.array([2 * pi * (value[1] - value[0]) for value in values])


def cylinder_axis_third(radius, height, point):
    """Cartesian series; stable at and near the axis, including mixed terms."""
    x, y, z = point
    a = _axis_derivatives(radius, height, z)
    t = np.longdouble(x * x + y * y)
    # B(t)=a_r/r. B_z=F(t), with consistent analytic coefficient derivatives.
    coeff = np.array(
        [
            (-1) ** (n + 1) / (2 * 4**n * factorial(n) * factorial(n + 1))
            for n in range(6)
        ],
        dtype=np.longdouble,
    )
    b = coeff * a[1:12:2]
    f = coeff * a[2:13:2]
    fz = coeff * a[3:14:2]
    bt = np.polynomial.polynomial.polyval(t, np.polynomial.polynomial.polyder(b))
    btt = np.polynomial.polynomial.polyval(t, np.polynomial.polynomial.polyder(b, 2))
    fv = np.polynomial.polynomial.polyval(t, f)
    ft = np.polynomial.polynomial.polyval(t, np.polynomial.polynomial.polyder(f))
    fzv = np.polynomial.polynomial.polyval(t, fz)
    result = np.zeros((3, 3, 3), dtype=np.longdouble)
    xy = (x, y)
    for i, j, k in product(range(2), repeat=3):
        result[i, j, k] = (
            2 * bt * (xy[k] * (i == j) + xy[j] * (i == k) + xy[i] * (j == k))
            + 4 * btt * xy[i] * xy[j] * xy[k]
        )
    for i, j in product(range(2), repeat=2):
        value = fv * (i == j) + 2 * ft * xy[i] * xy[j]
        for indices in set(permutations((i, j, 2))):
            result[indices] = value
    for i in range(2):
        for indices in set(permutations((i, 2, 2))):
            result[indices] = xy[i] * fzv
    result[2, 2, 2] = sum(
        (-1) ** n * a[2 * n + 2] * t**n / (4**n * factorial(n) ** 2) for n in range(6)
    )
    return np.asarray(result, dtype=np.float64)


def cylinder_axis_static(radius, height, point):
    """Static a and Gamma from the same axis Taylor coefficients as H."""
    x, y, z = point
    t = np.longdouble(x * x + y * y)
    a = _axis_derivatives(radius, height, z)
    coeff = np.array(
        [
            (-1) ** (n + 1) / (2 * 4**n * factorial(n) * factorial(n + 1))
            for n in range(6)
        ],
        dtype=np.longdouble,
    )
    b = coeff * a[1:12:2]
    if abs(z) < height / 2:
        b[0] -= 2 * np.arccos(np.longdouble(-1))
    bz = coeff * a[2:13:2]
    az_coeff = np.array(
        [(-1) ** n * a[2 * n] / (4**n * factorial(n) ** 2) for n in range(7)]
    )
    zz_coeff = np.array(
        [(-1) ** n * a[2 * n + 1] / (4**n * factorial(n) ** 2) for n in range(6)]
    )
    bv = np.polynomial.polynomial.polyval(t, b)
    bt = np.polynomial.polynomial.polyval(t, np.polynomial.polynomial.polyder(b))
    bzv = np.polynomial.polynomial.polyval(t, bz)
    az = np.polynomial.polynomial.polyval(t, az_coeff)
    zz = np.polynomial.polynomial.polyval(t, zz_coeff)
    acc = np.array([x * bv, y * bv, az])
    gamma = np.array(
        [
            [bv + 2 * x * x * bt, 2 * x * y * bt, x * bzv],
            [2 * x * y * bt, bv + 2 * y * y * bt, y * bzv],
            [x * bzv, y * bzv, zz],
        ]
    )
    return np.asarray(acc, dtype=np.float64), np.asarray(gamma, dtype=np.float64)


def _ring_derivatives(radius, r, v):
    """Analytic partial_r and partial_v of mixed endpoint; Omega_v.

    K_m=E/(2*m*(1-m))-K/(2*m), E_m=(E-K)/(2*m).
    Only ordinary off-axis points reach here, so m>0; kc² is formed directly.
    No derivative of the singular third-kind integral is needed at r=radius.
    """
    radius, r, v = map(np.longdouble, (radius, r, v))
    d2 = (radius + r) ** 2 + v * v
    l2 = (radius - r) ** 2 + v * v
    d = np.sqrt(d2)
    kc2 = l2 / d2
    m = 4 * radius * r / d2
    k = _cel(np.sqrt(kc2), 1, 1, 1)
    e = _cel(np.sqrt(kc2), 1, 1, kc2)
    km, em = (e / kc2 - k) / (2 * m), (e - k) / (2 * m)
    w = radius * radius + r * r + v * v
    core = w / d * k - d * e
    derivatives = []
    for _variable, d2q, wq, mq in (
        (r, 2 * (radius + r), 2 * r, m * (1 / r - 2 * (radius + r) / d2)),
        (v, 2 * v, 2 * v, -2 * m * v / d2),
    ):
        dq = d2q / (2 * d)
        coreq = (wq / d - w * dq / d2) * k + w / d * km * mq - dq * e - d * em * mq
        derivatives.append(2 * coreq / r)
    derivatives[0] -= 2 * core / r**2
    omega_v = 2 * (r * r + v * v - radius * radius) * e / (d * l2) - 2 * k / d
    return derivatives[0], derivatives[1], omega_v


def cylinder_far_third(radius, height, point):
    """Analytic third derivative of degree-8 solid-harmonic potential."""
    distance = np.linalg.norm(point)
    n = point / distance
    ez, identity = np.array([0.0, 0.0, 1.0]), np.eye(3)
    mu = n[2]
    nn, ne, ee = np.outer(n, n), np.outer(n, ez) + np.outer(ez, n), np.outer(ez, ez)
    out = np.zeros((3, 3, 3))
    mass = np.pi * radius**2 * height
    for degree in range(0, 10, 2):
        moment = sum(
            (-1) ** j
            * factorial(degree)
            / (factorial(degree - 2 * j) * factorial(j) ** 2 * 4**j)
            * (radius / distance) ** (2 * j)
            * (height / (2 * distance)) ** (degree - 2 * j)
            / ((j + 1) * (degree - 2 * j + 1))
            for j in range(degree // 2 + 1)
        )
        poly = np.polynomial.legendre.Legendre.basis(degree)
        p, dp, ddp, dddp = [poly.deriv(j)(mu) for j in range(4)]
        aa = (degree + 1) * p + mu * dp
        da = (degree + 2) * dp + mu * ddp
        dda = (degree + 3) * ddp + mu * dddp
        ff, df = (degree + 3) * aa + mu * da, (degree + 4) * da + mu * dda
        tensor = ff * nn - da * ne + ddp * ee - aa * identity
        for k in range(3):
            dn = (identity[:, k] - n * n[k]) / distance
            dmu = dn[2]
            deriv = (
                df * dmu * nn
                + ff * (np.outer(dn, n) + np.outer(n, dn))
                - dda * dmu * ne
                - da * (np.outer(dn, ez) + np.outer(ez, dn))
                + dddp * dmu * ee
                - da * dmu * identity
            )
            out[:, :, k] += (
                mass
                * moment
                / distance**3
                * (deriv - (degree + 3) * n[k] / distance * tensor)
            )
    return out


def cylinder_third(radius, height, point, gamma):
    """Use far, axis-series, or ordinary elliptic analytic derivatives."""
    x, y, z = point
    r = np.hypot(x, y)
    if np.linalg.norm(point) >= 16 * np.hypot(radius, height / 2):
        return cylinder_far_third(radius, height, point)
    if r <= 0.1 * radius:
        return cylinder_axis_third(radius, height, point)
    er = np.array([x / r, y / r, 0.0])
    ep = np.array([-y / r, x / r, 0.0])
    ez = np.array([0.0, 0.0, 1.0])
    basis = np.column_stack((er, ep, ez))
    lower = np.array(_ring_derivatives(radius, r, -height / 2 - z))
    upper = np.array(_ring_derivatives(radius, r, height / 2 - z))
    dr, dv, cz = upper - lower
    dz = -dv
    b_r = (er @ gamma @ er - ep @ gamma @ ep) / r
    b_z = (er @ gamma @ ez) / r
    local = np.zeros((3, 3, 3))
    for indices, value in (
        ((0, 0, 0), -b_r - dz),
        ((0, 0, 2), dr),
        ((0, 1, 1), b_r),
        ((1, 1, 2), b_z),
        ((0, 2, 2), dz),
        ((2, 2, 2), cz),
    ):
        for perm in set(permutations(indices)):
            local[perm] = value
    return np.einsum("ia,jb,kc,abc->ijk", basis, basis, basis, local)
