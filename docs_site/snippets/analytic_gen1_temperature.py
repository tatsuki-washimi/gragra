"""Evaluate static and phasor isobaric plane-wave temperature models."""

import numpy as np
from astropy import units as u

from gragra.analytic.atmospheric import isobaric_temperature
from gragra.fields import TemperatureFourierMode, TemperatureGaussian
from gragra.geometry import Frame
from gragra.results import PotentialReference
from gragra.targets import TargetPoints

frame = Frame("temperature field")
wave = TemperatureFourierMode(
    wavevector=[2.0, 0.0, 0.0] / u.m,
    phase_origin=[0.0, 0.0, 0.0] * u.m,
    frame=frame,
)
targets = TargetPoints([[0.0, 0.0, 0.0]] * u.m, ("origin",), frame)

# A complex delta_temperature is a peak phasor. Time reconstruction uses the
# package convention Re[P exp(-2 pi i f (t - t_ref))], without a half factor.
frequency_hz = 5.0
reference_seconds = 1_400_000_000
phasor_model = isobaric_temperature(
    wave,
    (4.0 + 2.0j) * u.K,
    rho0=2.0 * u.kg / u.m**3,
    T0=400.0 * u.K,
    frequency=frequency_hz * u.Hz,
    reference_time=(reference_seconds, 0),
)
phasor_result = phasor_model.at(targets).run()
potential_phasor = np.asarray(phasor_result.potential)[0, 0, 0]
assert phasor_result.record.g_applications == 1
assert (
    phasor_result.context.model_spec.field_definition.potential_reference
    is PotentialReference.PERIODIC_ZERO_MEAN_NONZERO_K
)
at_reference = np.real(potential_phasor)
at_quarter_period = np.real(
    potential_phasor * np.exp(-2j * np.pi * frequency_hz / (4.0 * frequency_hz))
)
assert np.isclose(at_reference, 0.02 * np.pi * 6.67430e-11, rtol=1e-12, atol=1e-25)
assert np.isclose(at_quarter_period, 0.5 * at_reference, rtol=1e-12, atol=1e-25)

# The Gaussian uses a zero-at-infinity gauge and has a finite center limit.
gaussian = TemperatureGaussian(
    center=[0.3, -0.2, 0.1] * u.m,
    sigma=0.4 * u.m,
    frame=frame,
)
gaussian_model = isobaric_temperature(
    gaussian,
    1.0 * u.K,
    rho0=1.2 * u.kg / u.m**3,
    T0=280.0 * u.K,
)
center_result = gaussian_model.at(
    TargetPoints(gaussian.center[None, :], ("center",), frame)
).run()
assert np.isfinite(np.asarray(center_result.potential)).all()
assert center_result.record.g_applications == 1
