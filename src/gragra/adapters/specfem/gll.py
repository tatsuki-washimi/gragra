"""GLL quadrature rule and trilinear-hexahedron Jacobian for the SPECFEM3D
Cartesian certified GLL export (numerical core).

This module provides the two numerical ingredients the certified exporter
needs in order to build ``/mesh/quadrature_volume``, which the certified
path fixes as ``w_mass = rho0 * J * w_x w_y w_z``:

1. :func:`gll_points_weights` -- the Gauss-Lobatto-Legendre (GLL) abscissae
   and weights on ``[-1, 1]`` for a general point count ``n >= 2``.
2. :func:`element_quadrature_volume` / :func:`elements_quadrature_volume` --
   ``detJ * w_i w_j w_k`` at every GLL point of an 8-node (trilinear)
   hexahedral element, given that element's 8 corner (anchor) nodes.

Upstream correspondence (SPECFEM3D Cartesian, pilot commit ``cc2e9ffa``)
-----------------------------------------------------------------------

This module is an **independent implementation**: the quadrature rule is
built here from published numerical methods (Newton iteration on the
Legendre derivative, closed-form GLL weights) and the Jacobian from
standard trilinear shape functions. No SPECFEM3D source code is copied or
translated; the upstream file and routine names cited below identify only
*which interface conventions* (node ordering, quadrature weighting) the two
implementations must agree on, so that Tier C (solver-vs-exporter)
comparison compares like with like:

- Corner (anchor) node order is SPECFEM's ``usual_hex_nodes`` with
  ``NGNOD = 8`` (``src/shared/hex_nodes.f90``), i.e. reference coordinates
  ``(xi, eta, zeta)`` in the order ``(-,-,-), (+,-,-), (+,+,-), (-,+,-),
  (-,-,+), (+,-,+), (+,+,+), (-,+,+)``. This is also the order the
  ``shape3D(1..8)`` expressions of ``recompute_jacobian_gravity``
  (``src/specfem3D/gravity_perturbation.f90``) encode, and the standard
  ``VTK_HEXAHEDRON`` corner order.
- The Jacobian is *recomputed from the 8 control nodes* (not read back from
  a solver-saved array) with the same trilinear shape-function gradients and
  the same cofactor-expansion determinant as
  ``recompute_jacobian_gravity``, and the quadrature volume is formed as
  ``Jac3D * wxgll(i) * wygll(j) * wzgll(k)`` exactly as in that routine's
  caller (``gravity_init``).
- The GLL rule is the ``alpha = beta = 0`` (standard Legendre) special case
  of upstream's ``zwgljd`` (``src/shared/gll_library.f90``), matching
  ``schema.EXPECTED_GAUSSALPHA`` / ``schema.EXPECTED_GAUSSBETA``.

Conventions and invariants
--------------------------

- **float64 throughout.** Corner coordinates are converted to float64 and
  all reference-grid quantities are float64. SPECFEM
  stores mesh coordinates as float32 on disk; the exporter is responsible
  for widening them before calling into this module, so that the quadrature
  itself never runs at reduced precision.
- **Flatten order is delegated to** :mod:`gragra.adapters.specfem.schema`.
  The returned per-element array is in the canonical ``(k, j, i) = (z, y, x)``
  C-order (x innermost) that ``schema.flatten_index`` defines; this module
  derives the grid shape from ``schema.reshape_shape_zyx`` rather than
  re-encoding the convention.
- **Returned arrays are read-only** (``setflags(write=False)``), following
  the package-wide immutability convention.
- **The returned shapes are the schema's on-disk dataset shapes**, not
  in-memory field shapes: ``(ngll,)`` per element and ``(nspec, ngll)`` for a
  batch, matching ``/mesh/quadrature_volume``'s ``("element", "gll")`` row in
  ``schema.DATASET_SPECS``. These are scalars per quadrature point, so there
  is no component axis and the trailing-component-axis convention (which
  governs vector/tensor *field* arrays) does not apply here -- the leading
  ``nspec`` axis is an element index, not a batch axis smuggled in front of a
  component axis.
- **This module returns no coordinates.** Recentring (below) is an internal
  numerical device only; ``/mesh/coordinates`` is written in absolute
  coordinates by the writer, which does not obtain coordinates from here.

Core purity: numpy-only. This module must not import ``h5py`` or any other
optional dependency.

Independence note: ``tests/_hex_volume_oracle.py`` contains its own copy of
the reference-corner table and its own Jacobian construction. That
duplication is **intentional and must not be refactored away** -- the oracle
exists to validate this module and may not import it (see that file's
"independence discipline" header).
"""

from dataclasses import dataclass
from functools import cache
from typing import Final

import numpy as np

from gragra.adapters.specfem import schema
from gragra.array_types import FloatArray

# ===========================================================================
# 1. Reference element (8-node trilinear hexahedron)
# ===========================================================================

#: Reference ``(xi, eta, zeta)`` coordinates of the 8 corner (anchor) nodes,
#: in SPECFEM's ``usual_hex_nodes`` order for ``NGNOD = 8`` (see the module
#: docstring for the upstream reference). Row ``a`` holds
#: ``(xi_a, eta_a, zeta_a)``, each ``+-1``.
_REFERENCE_CORNERS: Final[FloatArray] = np.array(
    [
        (-1.0, -1.0, -1.0),
        (+1.0, -1.0, -1.0),
        (+1.0, +1.0, -1.0),
        (-1.0, +1.0, -1.0),
        (-1.0, -1.0, +1.0),
        (+1.0, -1.0, +1.0),
        (+1.0, +1.0, +1.0),
        (-1.0, +1.0, +1.0),
    ],
    dtype=np.float64,
)
_REFERENCE_CORNERS.setflags(write=False)

#: Number of corner (anchor) nodes of the supported element type
#: (``NGNOD = 8``; ``NGNOD = 27`` is out of scope for the certified profile
#: -- see ``docs/design/specfem_import.md`` §certified profile).
N_CORNER_NODES: Final[int] = 8

#: How many elements are processed per vectorized block in
#: :func:`elements_quadrature_volume`. Purely a memory-strategy knob: every
#: element's quadrature volume depends only on that element's own corners,
#: so blocking changes neither the values nor their floating-point rounding
#: (unlike the source-axis ``chunk_size`` of the kernel contract, which
#: chunks a *reduction*). A block holds three ``(block, ngll, 3)`` float64
#: intermediates.
_ELEMENT_BLOCK_SIZE: Final[int] = 512

# ===========================================================================
# 2. Gauss-Lobatto-Legendre points and weights
# ===========================================================================

#: Maximum Newton iterations allowed per interior GLL node.
_NEWTON_MAX_ITER: Final[int] = 100

#: Newton convergence threshold on the absolute step size. The iteration is
#: quadratically convergent from the Chebyshev-Gauss-Lobatto seed, so the
#: node error after the final accepted step is ~step^2, i.e. already at the
#: float64 round-off floor when the step first drops below this value.
_NEWTON_TOL: Final[float] = 1e-15


def _legendre_p_and_pm1(x: float, degree: int) -> tuple[float, float]:
    """Return ``(P_degree(x), P_{degree-1}(x))`` via the three-term Legendre
    recurrence ``k P_k = (2k-1) x P_{k-1} - (k-1) P_{k-2}``.

    ``degree`` must be ``>= 1``; ``P_{-1}`` is never needed because the
    callers only use this for ``degree = n - 1 >= 1``.
    """
    p_prev = 1.0  # P_0
    p_curr = x  # P_1
    for k in range(2, degree + 1):
        p_prev, p_curr = p_curr, ((2 * k - 1) * x * p_curr - (k - 1) * p_prev) / k
    return p_curr, p_prev


def _interior_gll_node(seed: float, degree: int) -> float:
    """Newton-refine one interior GLL node from an initial guess.

    The ``n`` GLL abscissae are the roots of ``(1 - x^2) P'_{n-1}(x)``: the
    two endpoints ``+-1`` plus the ``n - 2`` roots of ``P'_{n-1}``. This
    function solves ``P'_N(x) = 0`` (``N = degree = n - 1``) by Newton's
    method, which needs ``P''_N`` as well. Both follow from the Legendre
    recurrence and the Legendre differential equation:

    - ``P'_N(x) = N (x P_N - P_{N-1}) / (x^2 - 1)``
    - ``(1 - x^2) P''_N - 2 x P'_N + N (N + 1) P_N = 0``

    Initial guess: the caller passes a Chebyshev-Gauss-Lobatto point
    ``-cos(pi k / N)``, the standard seed for this root-finding problem --
    the CGL and GLL point families interlace closely, so Newton converges in
    a handful of iterations without bracketing.
    """
    x = seed
    for _ in range(_NEWTON_MAX_ITER):
        p_n, p_nm1 = _legendre_p_and_pm1(x, degree)
        one_minus_x2 = 1.0 - x * x
        # d/dx P_N and d2/dx2 P_N (both finite here: interior nodes satisfy
        # |x| < 1 strictly, and the Newton step never leaves (-1, 1) from a
        # CGL seed -- the guard below turns any such excursion into an
        # explicit error instead of a silent NaN).
        d_p = degree * (x * p_n - p_nm1) / (-one_minus_x2)
        d2_p = (2.0 * x * d_p - degree * (degree + 1) * p_n) / one_minus_x2
        step = d_p / d2_p
        x -= step
        if not (-1.0 < x < 1.0):
            raise RuntimeError(
                f"GLL Newton iteration left the open interval (-1, 1) at "
                f"x={x!r} (degree={degree}, seed={seed!r})"
            )
        if abs(step) <= _NEWTON_TOL:
            return x
    raise RuntimeError(
        f"GLL Newton iteration did not converge within {_NEWTON_MAX_ITER} "
        f"iterations (degree={degree}, seed={seed!r}, last x={x!r})"
    )


@cache
def gll_points_weights(n: int) -> tuple[FloatArray, FloatArray]:
    """Gauss-Lobatto-Legendre abscissae and weights on ``[-1, 1]``.

    Parameters
    ----------
    n : int
        Number of quadrature points, ``n >= 2`` (the two endpoints ``+-1``
        are always included). ``n < 2`` raises ``ValueError``.

    Returns
    -------
    (points, weights) : tuple of FloatArray
        Two read-only float64 arrays of shape ``(n,)``. ``points`` is
        ascending, starts at exactly ``-1.0`` and ends at exactly ``+1.0``,
        and is exactly antisymmetric (``points[i] == -points[n-1-i]``); for
        odd ``n`` the middle abscissa is exactly ``0.0``. Both arrays are
        cached and shared between calls -- copy before mutating.

    Notes
    -----
    **Rule identity.** These are the standard *Legendre* Gauss-Lobatto
    points, i.e. the ``alpha = beta = 0`` special case of the
    Gauss-Lobatto-Jacobi rule that SPECFEM3D's ``zwgljd``
    (``src/shared/gll_library.f90``) implements. The certified profile fixes
    ``GAUSSALPHA = GAUSSBETA = 0`` (recorded as
    ``schema.EXPECTED_GAUSSALPHA`` / ``schema.EXPECTED_GAUSSBETA``, checked
    against the real source tree by ``tests/test_specfem_schema.py``), so
    this module's rule and the solver's rule are the same rule. If a future
    profile ever adopted a non-zero Jacobi weight, this function would no
    longer describe it.

    **Construction.** Endpoints are set exactly; the ``n - 2`` interior nodes
    are the roots of ``P'_{n-1}``, found by Newton's method from a
    Chebyshev-Gauss-Lobatto seed (see :func:`_interior_gll_node`). Only the
    lower half is solved and mirrored, and the middle node of an odd-``n``
    rule is set to exactly ``0.0`` -- both choices make the symmetry of the
    rule exact in floating point rather than merely accurate, which is also
    what upstream ``zwgljd`` does for the odd-``n`` midpoint. Weights use
    the closed form ``w_i = 2 / (n (n-1) [P_{n-1}(x_i)]^2)``.

    **Precision / exactness.** float64 throughout. The rule integrates
    polynomials of degree ``<= 2n - 3`` exactly and is *not* exact at degree
    ``2n - 2`` (this two-sided property is what distinguishes it from a
    Gauss-Legendre rule of the same point count; both are tested). For the
    trilinear-hexahedron Jacobian used below, ``detJ`` has degree ``<= 2``
    per reference axis, so any ``n >= 3`` per axis integrates it exactly.

    Note that upstream's ``zwgljd`` additionally refuses ``n == 2`` with a
    "minimum number of Gauss-Lobatto points for the SEM is 3" stop; that is
    an SEM-specific restriction on the *solver's* element order, not a
    property of the rule, so ``n == 2`` is accepted here (it is only ever
    used by tests, never by the certified export, which uses
    ``schema.EXPECTED_NGLL``).
    """
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool):
        raise TypeError(f"n must be an int, got {type(n).__name__}")
    n = int(n)
    if n < 2:
        raise ValueError(f"GLL requires at least 2 points (the endpoints), got n={n}")

    degree = n - 1  # N in P_N; the rule's nodes are the roots of (1-x^2) P'_N
    points = np.empty(n, dtype=np.float64)
    points[0] = -1.0
    points[n - 1] = +1.0
    if n % 2 == 1:
        points[(n - 1) // 2] = 0.0
    for idx in range(1, (n - 2) // 2 + 1):
        node = _interior_gll_node(-np.cos(np.pi * idx / degree), degree)
        points[idx] = node
        points[n - 1 - idx] = -node

    weights = np.empty(n, dtype=np.float64)
    for i in range(n):
        p_n, _p_nm1 = _legendre_p_and_pm1(float(points[i]), degree)
        weights[i] = 2.0 / (n * degree * p_n * p_n)

    points.setflags(write=False)
    weights.setflags(write=False)
    return points, weights


# ===========================================================================
# 3. Reference grid (tensor-product GLL points + shape-function gradients)
# ===========================================================================


@dataclass(frozen=True)
class _ReferenceGrid:
    """Precomputed, element-independent reference-element quantities.

    All arrays are read-only float64 and laid out in the canonical GLL
    flatten order (``schema.flatten_index``): ``(k, j, i) = (z, y, x)`` in
    C-order, x innermost.

    Attributes
    ----------
    dn_dxi, dn_deta, dn_dzeta : FloatArray
        Trilinear shape-function gradients, shape ``(ngll, 8)``:
        ``dn_dxi[p, a] = dN_a/dxi`` evaluated at GLL point ``p``.
    weights : FloatArray
        Tensor-product quadrature weights ``w_x[i] w_y[j] w_z[k]``, shape
        ``(ngll,)``.
    """

    dn_dxi: FloatArray
    dn_deta: FloatArray
    dn_dzeta: FloatArray
    weights: FloatArray


@cache
def _reference_grid_for_zyx(grid_shape: tuple[int, int, int]) -> _ReferenceGrid:
    """Build (and cache) the reference grid for a ``(NGLLZ, NGLLY, NGLLX)``
    C-order grid shape (the output of ``schema.reshape_shape_zyx``)."""
    n_z, n_y, n_x = grid_shape
    points_x, weights_x = gll_points_weights(n_x)
    points_y, weights_y = gll_points_weights(n_y)
    points_z, weights_z = gll_points_weights(n_z)

    # Build the tensor-product grid with the z axis outermost and the x axis
    # innermost, then ravel in C-order: this reproduces
    # schema.flatten_index(i, j, k) = k*(ny*nx) + j*nx + i by construction
    # (cross-checked against the schema helper in test_specfem_gll.py).
    xi = np.broadcast_to(points_x[None, None, :], grid_shape).ravel()
    eta = np.broadcast_to(points_y[None, :, None], grid_shape).ravel()
    zeta = np.broadcast_to(points_z[:, None, None], grid_shape).ravel()
    weights = (
        weights_z[:, None, None] * weights_y[None, :, None] * weights_x[None, None, :]
    ).ravel()

    corner_xi = _REFERENCE_CORNERS[:, 0]
    corner_eta = _REFERENCE_CORNERS[:, 1]
    corner_zeta = _REFERENCE_CORNERS[:, 2]

    # N_a(xi, eta, zeta) = (1/8)(1 + xi*xi_a)(1 + eta*eta_a)(1 + zeta*zeta_a)
    # -- the 8-node trilinear shape functions of
    # recompute_jacobian_gravity's shape3D(1..8), differentiated once.
    one_plus_xi = 1.0 + xi[:, None] * corner_xi[None, :]
    one_plus_eta = 1.0 + eta[:, None] * corner_eta[None, :]
    one_plus_zeta = 1.0 + zeta[:, None] * corner_zeta[None, :]

    dn_dxi = 0.125 * corner_xi[None, :] * one_plus_eta * one_plus_zeta
    dn_deta = 0.125 * one_plus_xi * corner_eta[None, :] * one_plus_zeta
    dn_dzeta = 0.125 * one_plus_xi * one_plus_eta * corner_zeta[None, :]

    for arr in (dn_dxi, dn_deta, dn_dzeta, weights):
        arr.setflags(write=False)
    return _ReferenceGrid(
        dn_dxi=dn_dxi, dn_deta=dn_deta, dn_dzeta=dn_dzeta, weights=weights
    )


def _reference_grid(ngll_shape: tuple[int, int, int]) -> _ReferenceGrid:
    """Validate ``ngll_shape`` (via the schema helper) and return its cached
    reference grid."""
    return _reference_grid_for_zyx(schema.reshape_shape_zyx(ngll_shape))


# ===========================================================================
# 4. Element quadrature volume
# ===========================================================================


def _as_corners(name: str, value, *, batched: bool) -> FloatArray:
    """Normalize a corner-node table to a finite float64 ``(nspec, 8, 3)``
    array. With ``batched=False`` the input must be a single ``(8, 3)``
    element, which is returned with a leading axis of length 1 so that the
    single- and multi-element paths share one implementation.
    """
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        msg = f"{name} must be numeric convertible to float64: {exc}"
        raise TypeError(msg) from exc
    if batched:
        valid = arr.ndim == 3 and arr.shape[1:] == (N_CORNER_NODES, 3)
        expected = f"(nspec, {N_CORNER_NODES}, 3)"
    else:
        valid = arr.shape == (N_CORNER_NODES, 3)
        expected = f"({N_CORNER_NODES}, 3)"
    if not valid:
        raise ValueError(f"{name} must have shape {expected}, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite numbers")
    return arr if batched else arr[None, ...]


def _raise_bad_jacobian(
    det_value: float,
    ngll_shape: tuple[int, int, int],
    flat_index: int,
    element_index: int | None,
) -> None:
    i, j, k = schema.unflatten_index(flat_index, ngll_shape)
    where = f"GLL flat index {flat_index} (i={i}, j={j}, k={k})"
    if element_index is not None:
        where = f"element index {element_index}, {where}"
    raise ValueError(
        f"Non-positive or non-finite Jacobian determinant detJ={det_value!r} "
        f"at {where}. Corner nodes must be given in SPECFEM's usual_hex_nodes "
        f"order; detJ <= 0 means the element is degenerate (zero/inverted "
        f"volume) or its node ordering is orientation-flipped. This guard is "
        f"deliberately sign-sensitive and must not be relaxed to abs(detJ): "
        f"it is the primary defence against an orientation-flipped corner "
        f"ordering, which the A1 index-ordering checks cannot detect on their "
        f"own (they pin rotations, not reflections)."
    )


def _block_quadrature_volume(
    corners: FloatArray,
    grid: _ReferenceGrid,
    ngll_shape: tuple[int, int, int],
    element_index_offset: int | None,
) -> FloatArray:
    """Quadrature volumes for a block of elements, ``corners`` of shape
    ``(block, 8, 3)`` (already validated float64).

    ``element_index_offset`` is the global index of ``corners[0]``, used only
    to build the error message; pass ``None`` for the single-element entry
    point so that the message does not mention a meaningless index 0.
    """
    # Recentre each element on the arithmetic mean of its own 8 corner nodes
    # before forming any product. The Jacobian columns are sums
    # sum_a (dN_a/dxi) X_a whose shape-function coefficients sum to zero
    # (because sum_a N_a == 1 identically), so on absolute coordinates the
    # terms cancel and the relative accuracy of detJ degrades roughly like
    # eps * |X| / L (|X| = distance from the origin, L = element size).
    # Measured on a 1 m element at |X| = 1e6 m: ~5e-11 relative error without
    # recentring vs ~1e-16 with it -- i.e. the difference between failing and
    # passing the Tier A2 rtol < 1e-12 gate. Subtracting the mean is exact in
    # IEEE arithmetic here (Sterbenz: the mean and each corner are within a
    # factor of two of each other), so the recentred element is an exact
    # translate of the original and the volume is unchanged.
    #
    # This is an internal numerical device only: no coordinate leaves this
    # module, and /mesh/coordinates is written in absolute coordinates.
    local = corners - corners.mean(axis=1, keepdims=True)

    # Jacobian columns d(x, y, z)/d(xi | eta | zeta), each of shape
    # (block, ngll, 3). np.matmul broadcasts the (ngll, 8) shape-gradient
    # matrix over the element (batch) axis of local.
    d_xi = grid.dn_dxi @ local
    d_eta = grid.dn_deta @ local
    d_zeta = grid.dn_dzeta @ local

    # Cofactor expansion, written in the same grouping as upstream's
    # recompute_jacobian_gravity:
    #   jacobian = xxi*(yeta*zgamma - ygamma*zeta)
    #            - xeta*(yxi*zgamma - ygamma*zxi)
    #            + xgamma*(yxi*zeta - yeta*zxi)
    x_xi, y_xi, z_xi = d_xi[..., 0], d_xi[..., 1], d_xi[..., 2]
    x_eta, y_eta, z_eta = d_eta[..., 0], d_eta[..., 1], d_eta[..., 2]
    x_zeta, y_zeta, z_zeta = d_zeta[..., 0], d_zeta[..., 1], d_zeta[..., 2]
    det = (
        x_xi * (y_eta * z_zeta - y_zeta * z_eta)
        - x_eta * (y_xi * z_zeta - y_zeta * z_xi)
        + x_zeta * (y_xi * z_eta - y_eta * z_xi)
    )

    # Sign/finiteness guard (see _raise_bad_jacobian). The ~isfinite term
    # also catches a NaN detJ, which a bare "det <= 0" would silently pass.
    invalid = ~np.isfinite(det) | (det <= 0.0)
    if invalid.any():
        block_index, flat_index = (int(v) for v in np.argwhere(invalid)[0])
        element_index = (
            None if element_index_offset is None else element_index_offset + block_index
        )
        _raise_bad_jacobian(
            float(det[block_index, flat_index]),
            ngll_shape,
            flat_index,
            element_index,
        )

    return det * grid.weights[None, :]


def element_quadrature_volume(corners8, ngll_shape: tuple[int, int, int]) -> FloatArray:
    """Per-GLL-point quadrature volume ``detJ * w_i w_j w_k`` for one
    8-node hexahedral element.

    Parameters
    ----------
    corners8 : array-like
        The element's 8 corner (anchor) node coordinates in metres, shape
        ``(8, 3)``, in SPECFEM's ``usual_hex_nodes`` order (module
        docstring). Converted to float64; non-finite values raise
        ``ValueError``.
    ngll_shape : tuple[int, int, int]
        ``(NGLLX, NGLLY, NGLLZ)`` (e.g. ``schema.EXPECTED_NGLL``). Validated
        by ``schema.reshape_shape_zyx``.

    Returns
    -------
    FloatArray
        Read-only float64 array of shape ``(ngll,)`` with
        ``ngll = schema.ngll_from_shape(ngll_shape)``, in the canonical
        ``(k, j, i)`` C-order flatten (x innermost) that
        ``schema.flatten_index`` defines. Summing it gives the element's
        volume, exactly (up to round-off) for any ``NGLL* >= 3``, since the
        trilinear ``detJ`` is degree ``<= 2`` per reference axis.

    Raises
    ------
    ValueError
        If ``corners8`` has the wrong shape or is non-finite, if
        ``ngll_shape`` is invalid, or if ``detJ <= 0`` (or non-finite) at any
        GLL point. The last case names the offending GLL index and is the
        primary detector of an orientation-flipped corner ordering -- it must
        not be relaxed to ``abs(detJ)``.

    Notes
    -----
    Coordinates are recentred on the element's corner-node centroid before
    the Jacobian is formed (a numerical-cancellation guard for meshes with
    large absolute coordinates; volume is translation invariant). No
    coordinate is returned, so callers keep their absolute coordinates --
    ``/mesh/coordinates`` is written unrecentred.
    """
    corners = _as_corners("corners8", corners8, batched=False)
    grid = _reference_grid(ngll_shape)
    out = _block_quadrature_volume(corners, grid, ngll_shape, None)[0]
    out.setflags(write=False)
    return out


def elements_quadrature_volume(corners, ngll_shape: tuple[int, int, int]) -> FloatArray:
    """Per-GLL-point quadrature volume for a batch of 8-node hexahedra.

    The batched form the exporter uses to build ``/mesh/quadrature_volume``
    in one call.

    Parameters
    ----------
    corners : array-like
        Corner (anchor) node coordinates in metres, shape ``(nspec, 8, 3)``,
        each element's 8 nodes in SPECFEM's ``usual_hex_nodes`` order.
        ``nspec == 0`` is allowed and yields an empty result.
    ngll_shape : tuple[int, int, int]
        ``(NGLLX, NGLLY, NGLLZ)``.

    Returns
    -------
    FloatArray
        Read-only float64 array of shape ``(nspec, ngll)``. Row ``e`` equals
        ``element_quadrature_volume(corners[e], ngll_shape)``.

    Raises
    ------
    ValueError
        Same conditions as :func:`element_quadrature_volume`; the ``detJ <= 0``
        message additionally names the offending element index.

    Notes
    -----
    Elements are processed in blocks of ``_ELEMENT_BLOCK_SIZE`` to bound peak
    memory. Blocking is not a numerical parameter: each element's result
    depends only on its own corners, so the output is bit-identical for any
    block size (contrast the kernel contract's ``chunk_size``, which chunks a
    reduction).
    """
    corner_table = _as_corners("corners", corners, batched=True)
    grid = _reference_grid(ngll_shape)
    n_elements = corner_table.shape[0]
    out = np.empty((n_elements, grid.weights.size), dtype=np.float64)
    for start in range(0, n_elements, _ELEMENT_BLOCK_SIZE):
        stop = min(start + _ELEMENT_BLOCK_SIZE, n_elements)
        out[start:stop] = _block_quadrature_volume(
            corner_table[start:stop], grid, ngll_shape, start
        )
    out.setflags(write=False)
    return out
