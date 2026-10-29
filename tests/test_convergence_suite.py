"""Phase 16-a convergence suite: discretization convergence vs analytic (Sphere).

Design: docs/design/phase16a_convergence_suite.md. This file covers Sphere
only; Cuboid/Cylinder h-refinement convergence lives in the sibling files
``test_convergence_cuboid.py`` / ``test_convergence_cylinder.py`` (added in
PR-4, once ``analytic/cavity_cuboid.py`` / ``cavity_cylinder.py`` provided
closed-form references — those did not exist when this Sphere-only suite was
first written). Shared error-norm/slope helpers live in
``_convergence_utils.py`` so all three files use one definition. Marked
``slow`` so it is deselected by the default ``pytest`` run
(``addopts = "-m 'not slow'"`` in pyproject.toml) and must be run explicitly:

    pytest -m slow tests/test_convergence_suite.py -v
"""

import numpy as np
import pytest
from _convergence_utils import (
    _relative_error_scalar,
    _relative_error_tensor,
    _relative_error_vector,
    _slope,
)

from gragra import G_SI, point_acceleration, point_gravity_gradient, point_potential
from gragra.analytic import (
    uniform_sphere_acceleration,
    uniform_sphere_gradient,
    uniform_sphere_potential,
)
from gragra.geometry.primitives import Sphere

pytestmark = pytest.mark.slow

# h = R/10, R/20, R/40, R/80 (analytical_benchmarks.md Sec.3 uses h=R/20 as the
# midpoint of this sweep). max_points=None avoids the default 1_000_000 cap:
# h=R/80 needs ceil(2R/(R/80))**3 = 160**3 = 4_096_000 bbox cells.
_DENOMINATORS = [10, 20, 40, 80]
_CENTER = np.array([0.0, 0.0, 0.0])
_RADIUS = 1.0
_DENSITY = 1000.0

# Outside target, off-axis and off any grid cell boundary/center
# (analytical_benchmarks.md: "Target coordinates should avoid cell
# boundaries/centers to prevent singular behavior").
_TARGET_OUTSIDE = np.array([[2.7, 0.3, -0.4]])


def _discretized_sphere(spacing_m):
    sphere = Sphere(
        center_m=_CENTER,
        radius_m=_RADIUS,
        density_kg_m3=_DENSITY,
        spacing_m=spacing_m,
        max_points=None,
    )
    return sphere.to_weighted_points()


def test_potential_convergence_sweep():
    hs = [_RADIUS / n for n in _DENOMINATORS]
    exact = G_SI * uniform_sphere_potential(_CENTER, _RADIUS, _DENSITY, _TARGET_OUTSIDE)

    errors = []
    for h in hs:
        src = _discretized_sphere(h)
        num = point_potential(src, _TARGET_OUTSIDE)
        errors.append(_relative_error_scalar(num[0], exact[0]))

    # Design target (not gated yet): slope ~2 (O(h^2)). See design doc Sec.5.
    _slope(hs, errors)

    # Primary gate: finest resolution meets the existing outer-target policy
    # (analytical_benchmarks.md Sec.3: rtol<=0.05 @ h=R/20; h=R/80 is finer).
    assert errors[-1] <= 0.05


def test_acceleration_convergence_sweep():
    hs = [_RADIUS / n for n in _DENOMINATORS]
    exact = G_SI * uniform_sphere_acceleration(
        _CENTER, _RADIUS, _DENSITY, _TARGET_OUTSIDE
    )

    errors = []
    for h in hs:
        src = _discretized_sphere(h)
        num = point_acceleration(src, _TARGET_OUTSIDE)
        errors.append(_relative_error_vector(num[0], exact[0]))

    _slope(hs, errors)

    assert errors[-1] <= 0.05


def test_gradient_convergence_sweep():
    hs = [_RADIUS / n for n in _DENOMINATORS]
    exact = G_SI * uniform_sphere_gradient(_CENTER, _RADIUS, _DENSITY, _TARGET_OUTSIDE)

    errors = []
    for h in hs:
        src = _discretized_sphere(h)
        num = point_gravity_gradient(src, _TARGET_OUTSIDE)
        errors.append(_relative_error_tensor(num[0], exact[0]))

    _slope(hs, errors)

    assert errors[-1] <= 0.05


def test_convergence_error_decreases_overall():
    """Sanity check: finest resolution should beat coarsest (any observable).

    Cell-center inclusion (staircase discretization, no partial-cell volume)
    means the error is not guaranteed to decrease strictly at every
    intermediate step (see design doc Sec.5) -- only the overall trend from
    coarsest to finest is asserted here.
    """
    hs = [_RADIUS / n for n in _DENOMINATORS]
    exact = G_SI * uniform_sphere_potential(_CENTER, _RADIUS, _DENSITY, _TARGET_OUTSIDE)

    errors = []
    for h in hs:
        src = _discretized_sphere(h)
        num = point_potential(src, _TARGET_OUTSIDE)
        errors.append(_relative_error_scalar(num[0], exact[0]))

    assert errors[-1] < errors[0]
