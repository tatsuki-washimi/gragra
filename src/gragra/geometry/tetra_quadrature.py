"""Tetrahedron and triangle numerical quadrature rules for P1 finite-body sources.

Two independent rule families are provided.  Both return
``(barycentric, weights)`` where ``barycentric`` has shape ``(Q, 4)``
(tetrahedron) or ``(Q, 3)`` (triangle) with rows summing to 1 -- the same
barycentric-matrix convention as
``gragra.fields.finite_body_p1._TETRA_BARYCENTRIC`` /
``_TRIANGLE_BARYCENTRIC`` -- and ``weights`` has shape ``(Q,)`` summing to 1
(the quadrature point's share of the *actual* cell's volume/area). A caller
multiplies ``weights`` by the real tetrahedron volume or triangle area; it
never multiplies by a reference-cell volume.

``tetra_fixed_degree2`` / ``triangle_fixed_degree2`` package the existing
classical 4-point / 3-point degree-2 rules -- numerically identical to
``gragra.fields.finite_body_p1._TETRA_BARYCENTRIC`` /
``_TRIANGLE_BARYCENTRIC`` -- for parity testing only.
``piecewise_tetra_p1_density_sources`` keeps using its own module-level
constants on its default (no ``quadrature_order``) call path, so that path
stays bit-identical independent of this module.

``tetra_gauss_quadrature`` / ``triangle_gauss_quadrature`` implement a
Gauss-Legendre-per-axis Duffy (collapsed-coordinate) rule -- the same rule
family used for the read-only research sweep in
``scripts/research_wave6_finite_body_cavity.py``
(``_duffy_tetra_rule``/``_duffy_triangle_rule``) and
``scripts/research_wave6_quadrature_order_study.py`` -- packaged here as a
production, validated, closed-form-normalized selectable rule. Subcell
count is fixed at 1 (no subdivision); ``order`` (Gauss points per Duffy
axis) is the sole degree of freedom, matching the finding recorded in
``docs/research/wave4-6-F.md`` (stage 4/5) that raising the Gauss order,
not the subcell count, is what drove the fixed-field gradient residual down
(order 2->4->6->8 shrank the residual roughly 15-30x per step, confirmed
against an order-10 self-reference).

Weight normalization: the raw Duffy-transform weight for a full-order rule
sums to the reference cell's own barycentric volume (1/6 for the unit
tetrahedron, 1/2 for the unit triangle) -- this is an exact property of the
coordinate transform's Jacobian, not an approximation the quadrature order
affects. Dividing by that constant is therefore an exact rescaling (not an
additional numerical approximation) that makes ``weights`` sum to 1
regardless of order, matching the fixed degree-2 rule's convention.
"""

from __future__ import annotations

import numpy as np

from gragra.array_types import FloatArray

_TETRA_A = (5.0 + 3.0 * np.sqrt(5.0)) / 20.0
_TETRA_B = (5.0 - np.sqrt(5.0)) / 20.0
_TETRA_REFERENCE_VOLUME = 1.0 / 6.0
_TRIANGLE_REFERENCE_AREA = 1.0 / 2.0


def _readonly_pair(
    barycentric: np.ndarray, weights: np.ndarray
) -> tuple[FloatArray, FloatArray]:
    barycentric = np.ascontiguousarray(barycentric, dtype=np.float64)
    weights = np.ascontiguousarray(weights, dtype=np.float64)
    barycentric.setflags(write=False)
    weights.setflags(write=False)
    return barycentric, weights


def tetra_fixed_degree2() -> tuple[FloatArray, FloatArray]:
    """Return the classical 4-point degree-2 tetrahedron rule.

    Equal weights of 0.25 at the barycentric points built from
    ``_TETRA_A``/``_TETRA_B`` -- numerically identical to
    ``gragra.fields.finite_body_p1._TETRA_BARYCENTRIC`` with its implicit
    uniform 0.25 weights. Exposed for parity testing only; production's
    default (no ``quadrature_order``) path does not call this function.
    """
    barycentric = np.full((4, 4), _TETRA_B, dtype=np.float64)
    np.fill_diagonal(barycentric, _TETRA_A)
    weights = np.full(4, 0.25, dtype=np.float64)
    return _readonly_pair(barycentric, weights)


def triangle_fixed_degree2() -> tuple[FloatArray, FloatArray]:
    """Return the classical 3-point degree-2 triangle rule.

    Equal weights of 1/3 -- numerically identical to
    ``gragra.fields.finite_body_p1._TRIANGLE_BARYCENTRIC`` with its
    implicit uniform 1/3 weights. Exposed for parity testing only.
    """
    barycentric = np.full((3, 3), 1.0 / 6.0, dtype=np.float64)
    np.fill_diagonal(barycentric, 2.0 / 3.0)
    weights = np.full(3, 1.0 / 3.0, dtype=np.float64)
    return _readonly_pair(barycentric, weights)


def _gauss_unit_interval(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes/weights on [0, 1] (weights sum to 1)."""
    if order < 2:
        raise ValueError("order must be >= 2 for the Gauss-Legendre quadrature family")
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    return 0.5 * nodes + 0.5, 0.5 * weights


def tetra_gauss_quadrature(order: int) -> tuple[FloatArray, FloatArray]:
    """Gauss-Legendre Duffy (collapsed-coordinate) tetrahedron rule.

    ``order`` Gauss points per Duffy axis (``order**3`` points total, one
    fixed subcell, no subdivision -- see module docstring); ``order >= 2``
    is required. ``weights`` sum to 1 (fraction of the tetrahedron's actual
    volume): the raw Duffy weight is divided by the reference tetrahedron's
    exact barycentric volume (1/6), not re-fit to the specific rule.
    """
    a, wa = _gauss_unit_interval(order)
    b, wb = _gauss_unit_interval(order)
    c, wc = _gauss_unit_interval(order)
    aa, bb, cc = np.meshgrid(a, b, c, indexing="ij")
    waa, wbb, wcc = np.meshgrid(wa, wb, wc, indexing="ij")
    barycentric = np.stack(
        (
            (1.0 - aa) * (1.0 - bb) * (1.0 - cc),
            aa,
            (1.0 - aa) * bb,
            (1.0 - aa) * (1.0 - bb) * cc,
        ),
        axis=-1,
    ).reshape(-1, 4)
    raw_weights = (waa * wbb * wcc * (1.0 - aa) ** 2 * (1.0 - bb)).reshape(-1)
    weights = raw_weights / _TETRA_REFERENCE_VOLUME
    return _readonly_pair(barycentric, weights)


def triangle_gauss_quadrature(order: int) -> tuple[FloatArray, FloatArray]:
    """Gauss-Legendre Duffy (collapsed-coordinate) triangle rule.

    ``order`` Gauss points per Duffy axis (``order**2`` points total, one
    fixed subcell); ``order >= 2`` required. ``weights`` sum to 1 (fraction
    of the triangle's actual area), matching :func:`tetra_gauss_quadrature`'s
    normalization convention.
    """
    a, wa = _gauss_unit_interval(order)
    b, wb = _gauss_unit_interval(order)
    aa, bb = np.meshgrid(a, b, indexing="ij")
    waa, wbb = np.meshgrid(wa, wb, indexing="ij")
    barycentric = np.stack(
        ((1.0 - aa) * (1.0 - bb), aa, (1.0 - aa) * bb), axis=-1
    ).reshape(-1, 3)
    raw_weights = (waa * wbb * (1.0 - aa)).reshape(-1)
    weights = raw_weights / _TRIANGLE_REFERENCE_AREA
    return _readonly_pair(barycentric, weights)


__all__ = [
    "tetra_fixed_degree2",
    "triangle_fixed_degree2",
    "tetra_gauss_quadrature",
    "triangle_gauss_quadrature",
]
