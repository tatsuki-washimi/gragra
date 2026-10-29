"""Shared helpers for Phase 16-a h-refinement convergence suites.

Used by ``test_convergence_suite.py`` (Sphere), ``test_convergence_cuboid.py``,
and ``test_convergence_cylinder.py``. Extracted verbatim from the original
Sphere-only suite (no formula/threshold changes) so the three files share one
error-norm and slope-fit definition. Design: docs/design/phase16a_convergence_suite.md.
"""

import numpy as np

_EPS = 1e-300


def _relative_error_scalar(num, exact):
    return float(np.abs(num - exact) / max(np.abs(exact), _EPS))


def _relative_error_vector(num, exact):
    return float(
        np.linalg.norm(num - exact, ord=2) / max(np.linalg.norm(exact, ord=2), _EPS)
    )


def _relative_error_tensor(num, exact):
    return float(
        np.linalg.norm(num - exact, ord="fro")
        / max(np.linalg.norm(exact, ord="fro"), _EPS)
    )


def _slope(hs, errors):
    """Log-log linear fit slope (expected ~2 for O(h^2) discretization error)."""
    log_h = np.log(hs)
    log_err = np.log(errors)
    slope, _intercept = np.polyfit(log_h, log_err, 1)
    return float(slope)


def _min_distance(positions_m, target):
    """Nearest discretized source point to a single target, in meters."""
    return float(np.min(np.linalg.norm(positions_m - target, axis=1)))
