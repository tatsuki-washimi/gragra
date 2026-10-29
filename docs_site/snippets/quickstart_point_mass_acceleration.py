"""Newtonian acceleration from a single point mass -- a complete first run.

A 1 kg point mass sits at the origin and the observation point is 1 m away
along x. The closed-form answer is a = -G_SI * x_hat, and the checks below
are exact equalities rather than tolerance comparisons: at r = 1 m the
geometric kernel evaluates to exactly -1 in IEEE-754 double precision, so the
only arithmetic left is a single multiplication by G_SI.
"""

import numpy as np

from gragra import G_SI, PointMass, point_acceleration

source = PointMass(mass_kg=1.0, position_m=[0.0, 0.0, 0.0])

# Low-level observables take source-frame SI coordinates in meters.
target_xyz_m = [[1.0, 0.0, 0.0]]
acceleration_m_s2 = point_acceleration(source, target_xyz_m)
print("acceleration at (1, 0, 0) m:", acceleration_m_s2, "m/s^2")

# One observation point, three vector components. The component axis is last,
# so the shape is (number of observation points, 3).
assert acceleration_m_s2.shape == (1, 3)
assert acceleration_m_s2.dtype == np.float64

# The acceleration points from the observation point towards the mass -- here
# along -x -- and its magnitude is G M / r^2 = G_SI.
assert acceleration_m_s2[0, 0] == -G_SI
assert acceleration_m_s2[0, 1] == 0.0
assert acceleration_m_s2[0, 2] == 0.0

# Arrays returned by an observable are read-only. Copy before modifying.
assert acceleration_m_s2.flags.writeable is False

# Doubling the distance divides the magnitude by four. Both magnitudes are
# exact multiples of G_SI by a power of two, so the ratio is exactly 4.
farther = point_acceleration(source, [[2.0, 0.0, 0.0]])
assert acceleration_m_s2[0, 0] / farther[0, 0] == 4.0
