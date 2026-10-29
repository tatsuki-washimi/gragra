"""Evaluate the rigid-reflection acoustic halfspace on both sides."""

import numpy as np
from astropy import units as u

from gragra.analytic.atmospheric import acoustic_halfspace
from gragra.geometry import Frame, HalfSpace
from gragra.results import PotentialReference
from gragra.targets import TargetPoints

frame = Frame("acoustic halfspace")
model = acoustic_halfspace(
    HalfSpace(frame),
    2.0 * u.Pa,
    sound_speed=340.0 * u.m / u.s,
    frequency=3.0 * u.Hz,
    incidence_angle=(np.pi / 4.0) * u.rad,
    azimuth=0.3 * u.rad,
    phase_origin=np.zeros(3) * u.m,
    reference_time=(1_400_000_000, 0),
)
targets = TargetPoints(
    [[0.4, -0.2, 1e-8], [0.4, -0.2, -1e-8]] * u.m,
    ("air-side", "ground-side"),
    frame,
)
result = model.at(targets).run()

potential = np.asarray(result.potential)
assert potential.shape == (2, 1, 1)
assert np.isfinite(potential).all()
assert result.record.g_applications == 1
assert (
    result.context.model_spec.field_definition.potential_reference
    is PotentialReference.ACOUSTIC_HALFSPACE_FOURIER_POSITIVE_Q
)
assert np.allclose(potential[0], potential[1], rtol=1e-6, atol=1e-25)
