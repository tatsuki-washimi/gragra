"""dtype rules, axis order, shape validation and the read-only contract.

Every assertion here restates a convention that the rest of the site relies
on, so a change to any of them breaks this file rather than only the prose.
"""

import numpy as np

from gragra import (
    PointMass,
    WeightedPointSource,
    point_acceleration,
    point_gravity_gradient,
)

# Coordinates are float64 tables of shape (N, 3): one row per source point,
# the component axis last.
positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
real_source = WeightedPointSource(positions_m=positions, weights_kg=[1.0, -1.0])
assert real_source.positions_m.dtype == np.float64
assert real_source.positions_m.shape == (2, 3)

# Weights keep the kind of number they were given: real stays float64, complex
# is promoted to complex128. Coordinates are float64 either way.
assert real_source.weights_kg.dtype == np.float64
complex_source = WeightedPointSource(
    positions_m=positions, weights_kg=[1.0 + 0.5j, -1.0]
)
assert complex_source.weights_kg.dtype == np.complex128
assert complex_source.positions_m.dtype == np.float64

# Low-level numeric observables consume bare source-frame SI target arrays.
# The output dtype follows the weights, never the coordinates.
targets_xyz_m = [[0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
assert point_acceleration(real_source, targets_xyz_m).dtype == np.float64
assert point_acceleration(complex_source, targets_xyz_m).dtype == np.complex128

# A vector observable ends in one component axis of length 3, a tensor
# observable in two. The observation-point axis stays in front of them.
assert point_acceleration(real_source, targets_xyz_m).shape == (2, 3)
assert point_gravity_gradient(real_source, targets_xyz_m).shape == (2, 3, 3)

# Stored arrays are defensive copies, so a later edit of the caller's array
# cannot reach inside the frozen source ...
mutable_input = np.array([[4.0, 0.0, 0.0]])
frozen_source = WeightedPointSource(positions_m=mutable_input, weights_kg=[2.0])
mutable_input[0, 0] = 99.0
assert frozen_source.positions_m[0, 0] == 4.0
assert frozen_source.positions_m.flags.writeable is False
assert real_source.weights_kg.flags.writeable is False

# ... and the arrays an observable hands back are read-only too.
assert point_acceleration(real_source, targets_xyz_m).flags.writeable is False

# A table of points has shape (N, 3). A bare (3,) array means "one point", and
# low-level observable functions accept that shorthand. Source dataclasses
# remain strict so a vector cannot be mistaken for a one-column table.
single = point_acceleration(PointMass(1.0, [0.0, 0.0, 0.0]), [5.0, 0.0, 0.0])
assert single.shape == (1, 3)
try:
    WeightedPointSource(positions_m=[5.0, 0.0, 0.0], weights_kg=[1.0])
except ValueError as error:
    assert "(N, 3)" in str(error)
else:
    raise AssertionError("a (3,) array must not be accepted as source table")
