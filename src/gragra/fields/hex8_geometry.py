"""Hex8 geometry foundation: shape functions, Gauss quadrature, and Jacobian validation.

Provides reusable geometric building blocks for 8-node hexahedron (Q1) elements,
used by VolumeMesh discretization (R6) and MaterialRegions Q1 field quadrature (R4).
"""

import numpy as np

# VTK / meshio canonical hex8 signs in reference domain [-1, 1]^3
# Node ordering:
# Bottom face (z = -1): 0: (-1,-1,-1), 1: (1,-1,-1), 2: (1,1,-1), 3: (-1,1,-1)
# Top face    (z = +1): 4: (-1,-1,+1), 5: (1,-1,+1), 6: (1,1,+1), 7: (-1,1,+1)
HEX8_SIGNS = np.array(
    [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ],
    dtype=np.float64,
)
HEX8_SIGNS.setflags(write=False)


def hex8_shape_and_derivative(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute trilinear shape functions and their reference derivatives.

    Parameters
    ----------
    points : np.ndarray
        Reference coordinates of shape (Nq, 3) in [-1, 1]^3.

    Returns
    -------
    shape : np.ndarray
        Array of shape (Nq, 8).
    derivative : np.ndarray
        Array of shape (Nq, 8, 3) containing [dN/dxi, dN/deta, dN/dzeta].
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (Nq, 3), got {pts.shape}")

    factors = 1.0 + pts[:, None, :] * HEX8_SIGNS[None, :, :]  # (Nq, 8, 3)
    shape = np.prod(factors, axis=-1) / 8.0  # (Nq, 8)

    derivative = np.empty((len(pts), 8, 3), dtype=np.float64)
    for axis in range(3):
        other = [i for i in range(3) if i != axis]
        derivative[:, :, axis] = (
            HEX8_SIGNS[None, :, axis] * np.prod(factors[:, :, other], axis=-1) / 8.0
        )

    return shape, derivative


def hex8_gauss_rule(order: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Return tensor-product Gauss-Legendre quadrature points and weights in [-1, 1]^3.

    Parameters
    ----------
    order : int
        Number of 1D quadrature points (order >= 2).

    Returns
    -------
    points : np.ndarray
        Quadrature points of shape (order^3, 3).
    weights : np.ndarray
        Quadrature weights of shape (order^3,).
    """
    if isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer)):
        raise TypeError("quadrature order must be integer")
    if order < 2:
        raise ValueError("quadrature order must be >= 2")

    x, w = np.polynomial.legendre.leggauss(int(order))
    gx, gy, gz = np.meshgrid(x, x, x, indexing="ij")
    gw = np.meshgrid(w, w, w, indexing="ij")

    points = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
    weights = (gw[0] * gw[1] * gw[2]).reshape(-1)
    return points, weights


def hex8_jacobians(
    vertices: np.ndarray,
    hex8_conn: np.ndarray,
    derivative: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Jacobians and determinants for cells at quadrature/sample points.

    Parameters
    ----------
    vertices : np.ndarray
        Array of vertex coordinates of shape (Nv, 3).
    hex8_conn : np.ndarray
        Hex8 connectivity array of shape (Nc, 8).
    derivative : np.ndarray
        Shape function derivatives of shape (Nq, 8, 3).

    Returns
    -------
    jac : np.ndarray
        Jacobian matrices of shape (Nc, Nq, 3, 3), J_ij = dx_i / dxi_j.
    det : np.ndarray
        Jacobian determinants of shape (Nc, Nq).
    """
    nodes = vertices[hex8_conn]  # (Nc, 8, 3)
    # J_ij = sum_a x_{a, i} * (dN_a / dxi_j)
    # Use relative node positions to avoid precision loss for distant bodies:
    rel_nodes = nodes - nodes[:, :1, :]  # (Nc, 8, 3)
    jac = np.einsum("cni,qnj->cqij", rel_nodes, derivative)
    det = np.linalg.det(jac)  # (Nc, Nq)

    # Relative conditioning check
    scale = np.linalg.norm(jac, axis=(-2, -1)) ** 3
    eps = np.finfo(np.float64).eps
    if not np.isfinite(det).all() or np.any(det <= 64.0 * eps * scale):
        raise ValueError("hex8 Jacobian is non-positive, degenerate, or inverted")

    return jac, det


def check_hex8_jacobians(vertices: np.ndarray, hex8_conn: np.ndarray) -> None:
    """Verify that all hex8 cells have positive Jacobians at 27 lattice points."""
    v = np.asarray(vertices, dtype=np.float64)
    c = np.asarray(hex8_conn, dtype=np.int64)
    if c.size == 0:
        return

    # Check for repeated nodes in a cell
    for row in c:
        if len(set(row)) != 8:
            raise ValueError("hex8 cell contains duplicate vertex indices (degenerate)")

    # 27 reference points: -1, 0, +1 in each axis
    coords = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    grid = np.stack(
        np.meshgrid(coords, coords, coords, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    _, deriv = hex8_shape_and_derivative(grid)
    hex8_jacobians(v, c, deriv)


def hex8_centroids_and_volumes_gauss(
    vertices: np.ndarray,
    hex8_conn: np.ndarray,
    order: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute hex8 cell centroids and volumes via Gauss-Legendre quadrature.

    Parameters
    ----------
    vertices : np.ndarray
        Vertices of shape (Nv, 3).
    hex8_conn : np.ndarray
        Connectivity of shape (Nc, 8).
    order : int, optional
        Gauss quadrature order in each reference direction (default 2, 8 points total).

    Returns
    -------
    centroids : np.ndarray
        Cell centroids of shape (Nc, 3).
    volumes : np.ndarray
        Cell volumes of shape (Nc,).
    """
    v = np.asarray(vertices, dtype=np.float64)
    c = np.asarray(hex8_conn, dtype=np.int64)
    if len(c) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.float64)

    pts, w = hex8_gauss_rule(order)
    shape, deriv = hex8_shape_and_derivative(pts)
    _, det = hex8_jacobians(v, c, deriv)  # (Nc, Nq)

    # Physical coordinates at Gauss points: (Nc, Nq, 3)
    nodes = v[c]  # (Nc, 8, 3)
    x_q = np.einsum("cni,qn->cqi", nodes, shape)  # (Nc, Nq, 3)

    # dV_q = det * w
    dv = det * w[None, :]  # (Nc, Nq)
    volumes = np.sum(dv, axis=-1)  # (Nc,)

    if (volumes <= 1e-15).any():
        raise ValueError("hex8 cell has degenerate non-positive volume")

    # Centroid: integral(x dV) / volume
    centroids = np.sum(x_q * dv[:, :, None], axis=1) / volumes[:, None]

    centroids.setflags(write=False)
    volumes.setflags(write=False)
    return centroids, volumes
