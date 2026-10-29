"""The three kernel sign conventions, pinned at one exact fixed point.

Source: 1 kg at the origin. Observation point: (1, 0, 0) m. Softening: zero.
At r = 1 m every geometric factor is exactly 0, +-1 or +-2 in IEEE-754 double
precision, so each check below is an exact equality and fails on a sign error
of any magnitude -- which a default-tolerance comparison would not, because
NumPy's default atol of 1e-8 is larger than the entire quantity being tested.
"""

import numpy as np

from gragra import (
    G_SI,
    PointMass,
    point_acceleration,
    point_gravity_gradient,
    point_potential,
)

source = PointMass(mass_kg=1.0, position_m=[0.0, 0.0, 0.0])
target = [[1.0, 0.0, 0.0]]

# K_Phi = -1 / r_eff, so the potential of a positive mass is negative.
potential_j_kg = point_potential(source, target)
assert potential_j_kg.shape == (1,)
assert potential_j_kg[0] == -G_SI

# K_a = Delta / r_eff**3 with Delta = r_source - r_target, so the acceleration
# points from the observation point towards the source: along -x here.
acceleration_m_s2 = point_acceleration(source, target)
assert acceleration_m_s2[0, 0] == -G_SI
assert (acceleration_m_s2[0, 1:] == 0.0).all()

# K_T = 3 Delta Delta / r_eff**5 - I / r_eff**3, differentiated with respect
# to the *observation* coordinate. On the separation axis it is diag(2, -1, -1)
# and the tensor axes are the two trailing ones.
gradient_s2 = point_gravity_gradient(source, target, softening_m=0.0)
assert gradient_s2.shape == (1, 3, 3)
assert (gradient_s2[0] == G_SI * np.diag([2.0, -1.0, -1.0])).all()

# With zero softening the tensor is trace-free exactly, not approximately.
assert np.trace(gradient_s2[0]) == 0.0
