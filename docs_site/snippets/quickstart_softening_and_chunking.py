"""What softening replaces the point mass with, and what chunking changes.

The softened trace is compared against the closed-form Plummer density rather
than merely being shown to be non-zero: that is what distinguishes "a
different mass distribution" from "the same field with the singularity
patched out".
"""

import numpy as np

from gragra import (
    G_SI,
    PointMass,
    WeightedPointSource,
    point_acceleration,
    point_gravity_gradient,
)

mass_kg = 1.0
softening_m = 0.5
source = PointMass(mass_kg=mass_kg, position_m=[0.0, 0.0, 0.0])
target = [[1.0, 0.0, 0.0]]

# With no softening the gravity-gradient tensor is exactly trace-free: outside
# the source the field satisfies Laplace's equation.
sharp = point_gravity_gradient(source, target, softening_m=0.0)
assert np.trace(sharp[0]) == 0.0

# With softening it is not, and the residual is not numerical noise. It is
# -4 pi G rho evaluated at the observation point, for the Plummer density
#     rho(r) = 3 M eps^2 / (4 pi r_eff^5),   r_eff^2 = r^2 + eps^2,
# which is the mass distribution that softening_m substitutes for the point
# mass. The observation point is inside that distribution, so Poisson's
# equation rather than Laplace's applies there.
softened = point_gravity_gradient(source, target, softening_m=softening_m)
r_eff_sq = 1.0 + softening_m**2
plummer_density_kg_m3 = 3.0 * mass_kg * softening_m**2 / (4.0 * np.pi * r_eff_sq**2.5)
assert np.isclose(
    np.trace(softened[0]),
    -4.0 * np.pi * G_SI * plummer_density_kg_m3,
    rtol=1e-13,
    atol=0.0,
)

# chunk_size bounds the size of the intermediate arrays and nothing else. On
# the NumPy backend it does change the order in which the source sum is
# accumulated, so two chunk sizes agree to rounding rather than bit for bit;
# the residual is normalised by the largest component instead of being
# compared against zero.
generator = np.random.default_rng(20260805)
cloud = WeightedPointSource(
    positions_m=generator.normal(scale=10.0, size=(2000, 3)),
    weights_kg=generator.normal(scale=5.0, size=2000),
)
observation_points = [[40.0, 0.0, 0.0], [0.0, 35.0, 5.0]]
whole = point_acceleration(cloud, observation_points, chunk_size=None)
tiled = point_acceleration(cloud, observation_points, chunk_size=64)
relative_deviation = np.max(np.abs(whole - tiled)) / np.max(np.abs(whole))
assert relative_deviation < 1e-12

# A zero separation with no softening is an error naming the offending pair,
# never a silently softened or infinite value ...
try:
    point_acceleration(source, [[0.0, 0.0, 0.0]], softening_m=0.0)
except ValueError as error:
    assert "target point 0 and source point 0" in str(error)
else:
    raise AssertionError("a zero separation must not be accepted")

# ... and a non-positive chunk_size is rejected rather than reinterpreted.
try:
    point_acceleration(source, target, chunk_size=0)
except ValueError as error:
    assert "chunk_size" in str(error)
else:
    raise AssertionError("chunk_size <= 0 must be rejected")
