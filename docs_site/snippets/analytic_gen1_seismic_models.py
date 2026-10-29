"""Run a phasor SCA wall model and a sampled causal point force."""

import numpy as np
from astropy import units as u

from gragra.analytic.seismic import point_force, small_cavity_wall
from gragra.geometry import Frame, Sphere
from gragra.gravity_sources import ForceHistory
from gragra.results import PotentialReference, TimeSampling
from gragra.targets import TargetPoints

frame = Frame("elastic medium")
sphere = Sphere([0.0, 0.0, 0.0] * u.m, 1.0 * u.m, frame)
targets = TargetPoints([[3.0, 0.4, -0.2]] * u.m, ("outside-wall",), frame)
wall = small_cavity_wall(
    sphere,
    [1e-4, 2e-4, -1e-4] * u.m,
    density=2500.0 * u.kg / u.m**3,
    frequency=2.0 * u.Hz,
    reference_time=(1_400_000_000, 0),
)
wall_result = wall.at(targets).run()
assert wall_result.record.g_applications == 1
assert wall_result.potential.shape == (1, 1, 1)
assert (
    wall_result.context.model_spec.field_definition.potential_reference
    is PotentialReference.ZERO_AT_INFINITY
)

# Each interval is 16 u^2 (1-u)^2. It starts and ends with zero force and
# zero time derivative, so the joined history is C1; after="zero" sets force
# to zero after the final knot.
onset = (1_400_000_000, 0)
history = ForceHistory(
    knots=[0.0, 0.5, 1.0] * u.s,
    coefficients=[
        [0.0, 0.0, 16.0, -32.0, 16.0],
        [0.0, 0.0, 16.0, -32.0, 16.0],
    ]
    * u.N,
    onset=onset,
    after="zero",
)
sampling = TimeSampling(epoch=onset, dt=0.1 * u.s, count=3)
force = point_force(
    [0.0, 0.0, 0.0] * u.m,
    [1.0, 0.0, 0.0],
    history,
    frame=frame,
    vp=3000.0 * u.m / u.s,
    sampling=sampling,
)
force_result = force.at(targets).run()
potential_samples = np.asarray(force_result.potential)
assert potential_samples.shape == (1, 1, 3)
assert np.isfinite(potential_samples).all()
assert np.abs(potential_samples[0, 0, 1]) > 0.0
assert force_result.potential.epoch == sampling.epoch
