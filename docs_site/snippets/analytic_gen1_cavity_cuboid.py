"""Evaluate one real cuboid Neumann mode at an exterior target."""

import numpy as np
from astropy import units as u

from gragra.analytic.atmospheric import (
    AcousticCavityNumerics,
    CuboidAcousticMode,
    acoustic_cavity_mode,
)
from gragra.geometry import Cuboid, Frame
from gragra.targets import TargetPoints

frame = Frame("cuboid cavity")
box = Cuboid(
    np.array([0.0, 0.0, 0.0]) * u.m,
    np.array([2.0, 3.0, 4.0]) * u.m,
    frame,
)
model = acoustic_cavity_mode(
    box,
    CuboidAcousticMode(1, 0, 0),
    (2.0e-3 + 1.0e-3j) * u.Pa,
    sound_speed=340.0 * u.m / u.s,
    reference_time=(1_400_000_000, 0),
    numerics=AcousticCavityNumerics(max_evaluations=200_000, max_refinements=12),
)
targets = TargetPoints([[1.5, 0.2, -0.1]] * u.m, ("outside-face",), frame)
result = model.at(targets, observables=("potential",)).run()

value = np.asarray(result.potential)[0, 0, 0]
assert result.potential.shape == (1, 1, 1)
assert np.isfinite(value.real) and np.isfinite(value.imag)
assert abs(value) > 0.0
assert result.record.g_applications == 1
assert result.context.frequencies_hz[0] > 0.0
assert result.context.reference_time_gps == (1_400_000_000, 0)
