"""PR-8b: gragra's production Rayleigh-wave code path (``rayleigh_displacement``
-> finite point cloud -> ``displacement_transfer_acceleration``) vs. the
published closed form (ANNA eqs.75-76).

Convention bridge to the same physical wave
--------------------------------------------
ANNA uses a z-down coordinate system while gragra uses z-up, with position
mapping ``x_g = J @ x_A`` for ``J = diag(1, 1, -1)``. Matching ANNA's time
phasor requires complex conjugation, and matching its positive surface
amplitude introduces an overall minus sign. Therefore the Rayleigh vector in
gragra's convention is ``v_g = -conj(J @ v_A)``.

Unlike ``tests/test_rayleigh_gravity_oracle.py`` (an *independent* Newton
integral that shares no code with gragra), this module exercises gragra's
own observable-layer code: it discretizes a finite chunk of the half-space
into point masses, evaluates gragra's actual kinematic Rayleigh field at
each point (``gragra.waves.rayleigh_displacement``), and computes their
Newtonian gravity via gragra's actual dipole/displacement-transfer
observable (``gragra.observables.displacement_transfer_acceleration``,
which uses the K_T gravity-gradient tensor -- the vector-weight-dipole
route documented in ``kernel_contract.md`` Sec.2.2).

Truncation geometry (why a sphere, not the oracle's x'_cut line-reduction)
-----------------------------------------------------------------------------
``generate_rayleigh_oracle.py``'s independent oracle integrates the
translation-invariant y' direction analytically (in closed form), leaving
only a 1D x'_cut truncation. gragra's point-cloud + dipole-sum path has no
such reduction available (it discretizes all three spatial directions), so
its natural truncation is a genuinely 3D region -- and by the divergence-
theorem identity linking a dipole sum to a bulk+surface integral (see the
oracle module docstring's y-reduction discussion), a truncation region's
*artificial* boundary faces contribute their own (unwanted) dipole terms
that must vanish as the region grows. A box grown along one axis at a time
was found (during design, not re-run by this test) to converge extremely
slowly -- an 8x growth in the x' extent alone barely reduced the residual,
because the side/bottom face *areas* did not shrink relative to the
kernel's ~1/distance^2 falloff. A **spherical** truncation centered at the
observer -- matching the finite-element domain the ANNA paper itself uses
for its own numerical validation (arXiv:2603.15157v4 Sec.4.3, Fig.8: "a
sphere with radius R=2000m and center at x0") -- converges cleanly instead,
since all boundary directions recede together (verified during design:
R=10 to R=80 monotonically tightens from ~1.3% to <0.5% relative error at
a fixed grid density, before flattening at the discretization floor).

Because of this, this test compares the production path directly against
the **closed-form target** (eq.75-76), not against a matching finite-domain
value from the independent oracle -- the two truncation geometries (an
infinite-y' line reduction vs. a literal finite sphere) are not simple to
match value-for-value without adding a third, non-reduced 3D quadrature
scheme solely for this comparison. Both routes independently converge to
the same physical answer, which is what is actually being checked. This is
a deliberate, documented scope reduction (Limitation Over Perfection) from
matching finite domains exactly.

Tolerance: at the modest sphere radius/grid density used here (chosen to
keep this a "slow" test rather than a many-minute one), the discretization
floor is ~0.04-0.5% relative error across the three (v_P/v_S, kH) points
tested; RTOL below carries a >10x margin over the worst observed value.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("scipy")
# generate_rayleigh_oracle unconditionally imports mpmath at module level
# (needed for its own cross-check), even though this test never calls into
# that code path -- guard it too so a validation-extra-less CI job doesn't
# hit a collection-time ModuleNotFoundError (see
# test_halfspace_gravity_oracle.py for the same pattern and its rationale).
pytest.importorskip("mpmath")

from oracles.generate_rayleigh_oracle import (  # noqa: E402
    RHO_0,
    G,
    rayleigh_gamma,
    solve_rayleigh,
)

from gragra import G_SI  # noqa: E402
from gragra.observables.dipole import displacement_transfer_acceleration  # noqa: E402
from gragra.waves.rayleigh import rayleigh_displacement  # noqa: E402

SPHERE_RADIUS = 15.0
POINTS_PER_UNIT_LENGTH = 3.0
RTOL = 5e-2
ATOL = 5e-2


def _closed_form(gamma: float, k_h: float) -> tuple[float, float]:
    decay = math.exp(-k_h)
    return (
        -2.0 * math.pi * G * RHO_0 * gamma * decay,
        2.0 * math.pi * G * RHO_0 * gamma * decay,
    )


def _production_acceleration(vp_vs_ratio: float, k_h: float) -> np.ndarray:
    """Discretize a sphere of radius SPHERE_RADIUS centered at the observer
    (0,0,H) into point masses, evaluate gragra's own Rayleigh displacement
    field, and return gragra's own dipole-observable acceleration (G-free,
    G_SI divided out for comparison with this module's G=1 convention)."""
    vp, vs = vp_vs_ratio, 1.0
    h = k_h  # k=1 fixed, matching the oracle's units
    r = SPHERE_RADIUS
    n = max(int(2.0 * r * POINTS_PER_UNIT_LENGTH), 4)
    nz = max(int(r * POINTS_PER_UNIT_LENGTH), 4)

    xs = np.linspace(-r, r, n, endpoint=False) + (2.0 * r / n) / 2.0
    ys = np.linspace(-r, r, n, endpoint=False) + (2.0 * r / n) / 2.0
    zs = np.linspace(-r, 0.0, nz, endpoint=False) + (r / nz) / 2.0
    dx, dy, dz = 2.0 * r / n, 2.0 * r / n, r / nz

    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
    dist2 = points[:, 0] ** 2 + points[:, 1] ** 2 + (points[:, 2] - h) ** 2
    points = points[dist2 <= r * r]

    mass_kg = RHO_0 * dx * dy * dz * np.ones(points.shape[0])
    displacement_m = rayleigh_displacement(
        points, [1.0, 0.0, 0.0], 1.0, vp, vs, amplitude_m=1.0
    )
    targets = [[0.0, 0.0, h]]
    a = displacement_transfer_acceleration(
        points, mass_kg, displacement_m, targets, softening_m=0.0
    )[0]
    return np.asarray(a) / G_SI


@pytest.mark.slow
@pytest.mark.parametrize(
    "vp_vs_ratio,k_h",
    [
        (math.sqrt(3.0), 0.5),
        (math.sqrt(3.0), 1.0),
        (2.0, 1.0),
    ],
)
def test_production_path_matches_closed_form(vp_vs_ratio, k_h):
    _, q, s = solve_rayleigh(vp_vs_ratio)
    gamma = rayleigh_gamma(q, s)
    target_az, target_ax = _closed_form(gamma, k_h)

    a = _production_acceleration(vp_vs_ratio, k_h)

    # y-component must vanish (translation invariance along propagation
    # direction; no y-dependence in the source).
    assert a[1] == pytest.approx(0.0, abs=1e-8)
    # x-component purely imaginary, z-component purely real (quadrature
    # phase convention, matching the oracle and the closed form).
    assert a[0].real == pytest.approx(0.0, abs=1e-6)
    assert a[2].imag == pytest.approx(0.0, abs=1e-6)

    assert a[2].real == pytest.approx(target_az, abs=ATOL, rel=RTOL), (
        f"vp_vs={vp_vs_ratio} kH={k_h}: production a_z={a[2].real}, "
        f"closed-form={target_az}"
    )
    assert a[0].imag == pytest.approx(target_ax, abs=ATOL, rel=RTOL), (
        f"vp_vs={vp_vs_ratio} kH={k_h}: production a_x={a[0].imag}, "
        f"closed-form={target_ax}"
    )
