"""Evaluate a nonzero real spherical Neumann mode outside a rigid cavity."""

import numpy as np
from astropy import units as u

from gragra.analytic.atmospheric import (
    AcousticCavityNumerics,
    SphereAcousticMode,
    acoustic_cavity_mode,
)
from gragra.geometry import Frame, Sphere
from gragra.targets import TargetPoints

frame = Frame("spherical cavity")
sphere = Sphere([0.0, 0.0, 0.0] * u.m, 1.0 * u.m, frame)
mode = SphereAcousticMode(l=1, m=1, n=1)
model = acoustic_cavity_mode(
    sphere,
    mode,
    (2.0e-3 + 1.0e-3j) * u.Pa,
    sound_speed=340.0 * u.m / u.s,
    reference_time=(1_400_000_000, 0),
    # These are integration resource ceilings, not a user-set accuracy target.
    numerics=AcousticCavityNumerics(max_evaluations=200_000, max_refinements=12),
)
targets = TargetPoints(
    [[2.0, 0.3, -0.2], [1.6, 0.5, 0.4]] * u.m,
    ("outside-a", "outside-b"),
    frame,
)
result = model.at(targets).run()

assert result.potential.shape == (2, 1, 1)
assert result.acceleration.shape == (2, 3, 1)
assert result.gradient.shape == (2, 9, 1)
assert np.isfinite(np.asarray(result.potential)).all()
assert result.record.g_applications == 1
assert len(result.context.frequencies_hz) == 1
assert result.context.frequencies_hz[0] > 0.0
assert result.context.reference_time_gps == (1_400_000_000, 0)
