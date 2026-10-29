"""Evaluate exact moving point masses and a linearized oscillation."""

import numpy as np
from astropy import units as u

from gragra.analytic.objects import (
    constant_velocity,
    fixed_axis_rotation,
    translational_oscillation,
)
from gragra.geometry import Frame
from gragra.gravity_sources import PointMass, PointMassCloud
from gragra.results import TimeSampling
from gragra.targets import TargetPoints

root = Frame("motion root")
output = Frame(
    "rotated output",
    basis=[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    parent=root,
)
targets = TargetPoints([[4.0, 2.0, 1.0]] * u.m, ("far-field",), root)
sampling = TimeSampling(
    epoch=(1_400_000_000, 0),
    dt=0.25 * u.s,
    count=3,
)

# PointMass positions are SI coordinates in the supplied frame. motion_epoch
# sets the time at which this position is valid; sampling.epoch is independent.
mass = PointMass(2.0e8, [0.0, 0.0, 0.0])
translation = constant_velocity(
    mass,
    [0.1, 0.0, 0.0] * u.m / u.s,
    frame=root,
    motion_epoch=(1_400_000_000, 0),
    sampling=sampling,
)
translated_result = translation.at(targets, output_frame=output).run()
assert translated_result.potential.shape == (1, 1, 3)
assert translated_result.acceleration.shape == (1, 3, 3)
assert translated_result.context.frame is output
assert np.isfinite(np.asarray(translated_result.acceleration)).all()
assert translated_result.potential.epoch == sampling.epoch

# A cloud's input coordinates define zero angle. angle_epoch is not the
# TimeSampling epoch, even when both happen to have the same value here.
cloud = PointMassCloud([1.0e8, 1.5e8], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
rotating = fixed_axis_rotation(
    cloud,
    [0.0, 0.0, 1.0],
    0.2 * u.rad / u.s,
    frame=root,
    origin=[0.0, 0.0, 0.0] * u.m,
    angle0=0.0 * u.rad,
    angle_epoch=(1_400_000_000, 0),
    sampling=sampling,
)
rotated_result = rotating.at(targets, output_frame=output).run()
assert rotated_result.gradient.shape == (1, 9, 3)
assert np.isfinite(np.asarray(rotated_result.gradient)).all()

# Oscillation is a small-displacement, peak-phasor response about fixed source
# positions. Reconstruct at a quarter period with the package's negative-time
# exponential convention; do not divide the phasor by two.
oscillating = translational_oscillation(
    mass,
    [1e-3 + 2e-4j, 0.0, 0.0] * u.m,
    frame=root,
    frequency=2.0 * u.Hz,
    reference_time=(1_400_000_000, 0),
)
phasor_result = oscillating.at(targets, output_frame=output).run()
potential_phasor = np.asarray(phasor_result.potential)[0, 0, 0]
at_reference = np.real(potential_phasor)
at_quarter_period = np.real(potential_phasor * np.exp(-0.5j * np.pi))
assert np.isfinite(potential_phasor)
assert not np.isclose(at_reference, at_quarter_period, rtol=0.0, atol=1e-30)
