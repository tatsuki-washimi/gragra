"""Smallest end-to-end check that an installation actually works.

Imports the package, confirms one public name and the value of the
gravitational constant, then reproduces the closed-form potential of a point
mass. If this runs, the install is sound.
"""

import gragra
from gragra import G_SI, PointMass, point_potential

# The public surface is importable and carries the names the docs use.
assert "point_potential" in gragra.__all__
assert "point_acceleration" in gragra.__all__

# The gravitational constant is a declared CODATA value, not a fitted one.
assert G_SI == 6.67430e-11

# One mass, one observation point, one closed-form answer:
# Phi = -G M / r, which at M = 1 kg and r = 1 m is exactly -G_SI.
potential_j_kg = point_potential(PointMass(1.0, [0.0, 0.0, 0.0]), [[1.0, 0.0, 0.0]])
assert potential_j_kg.shape == (1,)
assert potential_j_kg[0] == -G_SI

print("gragra reproduces Phi = -G M / r at r = 1 m:", potential_j_kg[0], "J/kg")
