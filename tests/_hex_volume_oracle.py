"""Independent volume oracles for an 8-node (trilinear) hexahedron.

This module provides the **A2 (element quadrature volume) reference oracle**
described in ``docs/design/specfem_import.md`` Tier A / A2 and the
A-series scientific tests.

Two *independent* implementations are provided (``hex_volume_24tetra`` and
``hex_volume_gauss2``); ``tests/test_specfem_tier_a.py`` cross-checks them
against each other and against known analytic volumes before either is ever
trusted as a reference for A2.

IMPORTANT -- independence discipline (do not weaken this file):

- This module **must never import** ``gragra.adapters.specfem.gll`` (the
    production GLL/Jacobian implementation -- an oracle that shares
  code with the implementation it is meant to validate is not independent.
- The two functions below **must not share helper code** with each other
  either: they use genuinely different numerical methods (a closed-form
  24-tetrahedron decomposition vs. a 2x2x2 Gauss-Legendre quadrature of the
  trilinear Jacobian), so a bug in one algorithm's derivation is very
  unlikely to also appear in the other.
- Existing gragra volume helpers (e.g.
  ``gragra.adapters.meshio_adapter.from_meshio_volume``'s 6-tetra corner
  decomposition) must **not** be reused here either: that decomposition is
  not exact for a warped trilinear hexahedron (a mild warp already produces
  ~1e-2 relative error), so it cannot serve as an oracle for the rtol<1e-12
  A2 gate.

Corner node ordering: both functions expect ``corners8`` in the SPECFEM3D
``usual_hex_nodes`` order (``hex_nodes.f90``, ``NGNOD=8``):
``(1,1,1), (N,1,1), (N,N,1), (1,N,1), (1,1,N), (N,1,N), (N,N,N), (1,N,N)``
i.e. (using x,y,z reference axes, low=-1/high=+1):

=====  ======  ======  ======
index  xi      eta     zeta
=====  ======  ======  ======
0      -1      -1      -1
1      +1      -1      -1
2      +1      +1      -1
3      -1      +1      -1
4      -1      -1      +1
5      +1      -1      +1
6      +1      +1      +1
7      -1      +1      +1
=====  ======  ======  ======

This is also exactly the standard VTK_HEXAHEDRON corner order.
"""

import numpy as np

# Reference (xi, eta, zeta) corner coordinates for the 8 corners, in the
# usual_hex_nodes / VTK_HEXAHEDRON order documented above.
_REF_CORNERS = np.array(
    [
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
    ],
    dtype=np.float64,
)

# The 6 faces of the hexahedron, each as a 4-tuple of local corner indices
# (0-7) ordered so that walking a->b->c->d is counter-clockwise as seen from
# OUTSIDE the hexahedron (outward-normal winding). This is the standard
# VTK_HEXAHEDRON face table, verified against _REF_CORNERS by hand (see the
# Builder session notes / plan discussion for the cross-product derivation):
#   F0 bottom (-z): (0,3,2,1)   F1 top   (+z): (4,5,6,7)
#   F2 front  (-y): (0,1,5,4)   F3 right (+x): (1,2,6,5)
#   F4 back   (+y): (2,3,7,6)   F5 left  (-x): (3,0,4,7)
_HEX_FACES: tuple[tuple[int, int, int, int], ...] = (
    (0, 3, 2, 1),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
)

#: 2-point Gauss-Legendre abscissae on [-1, 1] (weight 1 each).
_GAUSS2_NODES: tuple[float, float] = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))


def _validate_corners8(corners8) -> np.ndarray:
    corners = np.asarray(corners8, dtype=np.float64)
    if corners.shape != (8, 3):
        raise ValueError(f"corners8 must have shape (8, 3), got {corners.shape}")
    if not np.isfinite(corners).all():
        raise ValueError("corners8 must be finite")
    return corners


def hex_volume_24tetra(corners8) -> float:
    """Face-centroid 24-tetrahedron (Grandy-type) closed-form hexahedron
    volume -- the A2 primary oracle.

    Decomposition: each of the 6 faces is split into 4 triangles by fanning
    from the **face centroid** (the arithmetic mean of that face's 4 corner
    nodes -- *not* the true bilinear-surface centroid; using the true
    centroid of a possibly non-planar face breaks the exactness of this
    decomposition for a warped hexahedron). Each of the resulting 24
    triangles forms a tetrahedron with the **apex** (the arithmetic mean of
    all 8 corner nodes -- Grandy-type decomposition). The signed volume of
    each such tetrahedron is summed; because the face winding is
    outward-consistent, this signed sum equals the exact trilinear-mapped
    hexahedron volume for *any* choice of the shared reference point
    (a standard closed-polyhedron volume identity via the divergence
    theorem) -- the centroid is simply a numerically convenient choice.

    Numerical stability: all 8 corners are recentred (translated so their
    own mean sits at the origin) before any products are formed, guarding
    against cancellation for meshes with large coordinate offsets (per
    Tier A2's recentring requirement).
    """
    corners = _validate_corners8(corners8)
    apex = corners.mean(axis=0)
    local = corners - apex  # apex is now exactly the origin

    total = 0.0
    for a_idx, b_idx, c_idx, d_idx in _HEX_FACES:
        face_centroid = (
            local[a_idx] + local[b_idx] + local[c_idx] + local[d_idx]
        ) / 4.0
        for p_idx, q_idx in (
            (a_idx, b_idx),
            (b_idx, c_idx),
            (c_idx, d_idx),
            (d_idx, a_idx),
        ):
            p = local[p_idx]
            q = local[q_idx]
            # Signed tetrahedron volume with the apex translated to the
            # origin: (1/6) * p . (q x face_centroid).
            total += np.dot(p, np.cross(q, face_centroid)) / 6.0
    return float(total)


def _trilinear_shape_derivatives(
    xi: float, eta: float, zeta: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """dN_i/dxi, dN_i/deta, dN_i/dzeta for the 8 trilinear shape functions
    N_i(xi,eta,zeta) = (1/8)(1+xi*xi_i)(1+eta*eta_i)(1+zeta*zeta_i),
    evaluated at one (xi, eta, zeta) point, for all 8 nodes at once.
    """
    xi_i = _REF_CORNERS[:, 0]
    eta_i = _REF_CORNERS[:, 1]
    zeta_i = _REF_CORNERS[:, 2]
    dn_dxi = 0.125 * xi_i * (1.0 + eta * eta_i) * (1.0 + zeta * zeta_i)
    dn_deta = 0.125 * (1.0 + xi * xi_i) * eta_i * (1.0 + zeta * zeta_i)
    dn_dzeta = 0.125 * (1.0 + xi * xi_i) * (1.0 + eta * eta_i) * zeta_i
    return dn_dxi, dn_deta, dn_dzeta


def hex_volume_gauss2(corners8) -> float:
    """2x2x2 Gauss-Legendre quadrature of the trilinear Jacobian determinant
    -- the A2 second, independent oracle.

    The trilinear map's Jacobian ``J = d(x,y,z)/d(xi,eta,zeta)`` has columns
    that are each bilinear in the *other* two reference coordinates (e.g.
    the d/dxi column depends only on eta and zeta, not xi -- because it was
    already differentiated with respect to xi). The determinant of a 3x3
    matrix built from three such columns therefore has degree <= 2 in each
    of xi, eta, zeta individually, so a 2-point Gauss-Legendre rule per axis
    (exact for polynomials up to degree 3 per axis) integrates ``detJ``
    over ``[-1,1]^3`` *exactly* -- this is not an approximation.

    As with ``hex_volume_24tetra``, corners are recentred to their own mean
    before use (numerical cancellation guard); recentring does not change
    the Jacobian (a rigid translation) so it cannot change the result.
    """
    corners = _validate_corners8(corners8)
    apex = corners.mean(axis=0)
    local = corners - apex

    total = 0.0
    for xi in _GAUSS2_NODES:
        for eta in _GAUSS2_NODES:
            for zeta in _GAUSS2_NODES:
                dn_dxi, dn_deta, dn_dzeta = _trilinear_shape_derivatives(xi, eta, zeta)
                jacobian = np.empty((3, 3), dtype=np.float64)
                jacobian[:, 0] = local.T @ dn_dxi
                jacobian[:, 1] = local.T @ dn_deta
                jacobian[:, 2] = local.T @ dn_dzeta
                # Gauss-Legendre weight is 1 at each of the two nodes, so
                # the quadrature sum is a plain sum of detJ values.
                total += float(np.linalg.det(jacobian))
    return total
