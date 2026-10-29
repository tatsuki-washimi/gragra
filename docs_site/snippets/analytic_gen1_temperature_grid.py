"""Convert a signed Kelvin grid to a frame-aware point-sum model."""

import numpy as np
from astropy import units as u

from gragra.analytic.atmospheric import isobaric_temperature
from gragra.fields import TemperatureGaussian, TemperaturePerturbationGrid
from gragra.geometry import Frame
from gragra.gravity_sources import point_sum
from gragra.targets import TargetPoints

root = Frame("world")
grid_frame = Frame("temperature grid", origin=[0.4, -0.2, 0.1] * u.m, parent=root)
output_frame = Frame(
    "rotated output",
    basis=[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    parent=root,
)
grid = TemperaturePerturbationGrid(
    x_edges=[-1.0, -0.1, 1.2] * u.m,
    y_edges=[-0.7, 0.5, 1.5] * u.m,
    z_edges=[-0.3, 0.8] * u.m,
    delta_temperature=[[[1.0], [-0.4]], [[-0.7], [1.3]]] * u.K,
    frame=grid_frame,
)
rho0 = 1.4 * u.kg / u.m**3
T0 = 285.0 * u.K
source = grid.as_volume_element_source(rho0=rho0, T0=T0)
model = point_sum(
    source,
    frame=grid.frame,
    field_definition=grid.field_definition,
    provenance=grid.point_sum_provenance(rho0=rho0, T0=T0),
)
targets = TargetPoints(
    [[3.2, -2.1, 1.7], [-2.4, 1.3, 4.0]] * u.m,
    ("target-a", "target-b"),
    root,
)
result = model.at(targets, output_frame=output_frame).run()

analytic_definition = isobaric_temperature(
    TemperatureGaussian(center=[0, 0, 0] * u.m, sigma=1 * u.m, frame=grid.frame),
    1 * u.K,
    rho0=rho0,
    T0=T0,
).spec.field_definition
assert result.context.model_spec.field_definition == analytic_definition
assert result.context.frame is output_frame
assert model.binding.frame is grid.frame
assert result.record.g_applications == 1
assert np.isfinite(np.asarray(result.potential)).all()
assert np.isfinite(np.asarray(result.acceleration)).all()
assert np.isfinite(np.asarray(result.gradient)).all()
