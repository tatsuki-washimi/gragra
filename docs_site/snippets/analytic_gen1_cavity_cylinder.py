"""Evaluate all three observables for one real cylindrical Neumann mode."""

import numpy as np
from astropy import units as u

from gragra.analytic.atmospheric import (
    AcousticCavityNumerics,
    CylinderAcousticMode,
    acoustic_cavity_mode,
)
from gragra.geometry import Cylinder, Frame
from gragra.targets import TargetPoints

frame = Frame("cylindrical cavity")
cylinder = Cylinder(
    np.array([0.2, -0.4, 0.1]) * u.m,
    1.7 * u.m,
    2.3 * u.m,
    frame,
)
model = acoustic_cavity_mode(
    cylinder,
    CylinderAcousticMode(m=0, n=0, q=1),
    (2.0e-3 + 1.0e-3j) * u.Pa,
    sound_speed=340.0 * u.m / u.s,
    reference_time=(1_400_000_000, 0),
    numerics=AcousticCavityNumerics(max_evaluations=200_000, max_refinements=12),
)
targets = TargetPoints(
    [[0.43, -0.57, -0.21]] * u.m,
    ("inside-cylinder",),
    frame,
)
result = model.at(targets).run()

assert result.potential.shape == (1, 1, 1)
assert result.acceleration.shape == (1, 3, 1)
assert result.gradient.shape == (1, 9, 1)
assert np.isfinite(np.asarray(result.potential)).all()
assert np.isfinite(np.asarray(result.acceleration)).all()
assert np.isfinite(np.asarray(result.gradient)).all()
assert result.record.g_applications == 1
assert len(result.record.numerical_diagnostics) == 3
assert all(
    item.termination == "converged" for item in result.record.numerical_diagnostics
)
