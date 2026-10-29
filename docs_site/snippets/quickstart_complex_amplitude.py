"""A first-order complex amplitude, and how to turn it back into a real field.

The complex number an observable returns here is the amplitude of a linear
response, not a field value. Recovering something measurable takes an explicit
time factor and a real part, and the sign convention of that time factor is
the caller's -- gragra never applies one.
"""

import numpy as np

from gragra import G_SI, WeightedPointSource, point_acceleration

# One oscillating mass element with amplitude 1 kg and a phase of +pi/2
# relative to whatever reference the caller has chosen: delta_m = 1j kg.
source = WeightedPointSource(positions_m=[[0.0, 0.0, 0.0]], weights_kg=[1.0j])
amplitude_m_s2 = point_acceleration(source, [[1.0, 0.0, 0.0]])

# The output dtype follows the weights: a complex source gives complex128.
assert amplitude_m_s2.dtype == np.complex128

# The x component is -G_SI * 1j. Modulus G_SI, argument -pi/2 -- the response
# lags the source by half a turn relative to the source's own +pi/2, because
# the acceleration points back towards the mass. Both parts are exact here.
assert amplitude_m_s2[0, 0] == complex(0.0, -G_SI)
assert abs(amplitude_m_s2[0, 0]) == G_SI

# Attaching a time factor and taking the real part is what produces a physical
# acceleration. This snippet uses exp(-i omega t); a caller whose amplitudes
# came out of a forward FFT with the opposite sign convention must use
# exp(+i omega t) instead, and gragra cannot tell the difference.
angular_frequency_rad_s = 2.0 * np.pi  # one cycle per second
times_s = np.array([0.0, 0.25, 0.5, 0.75])
real_signal_m_s2 = np.real(
    amplitude_m_s2[0, 0] * np.exp(-1j * angular_frequency_rad_s * times_s)
)

# At t = 0 the real part starts at zero. That target value is zero, so the
# residual is bounded relative to the amplitude rather than compared to zero
# with a bare tolerance.
assert abs(real_signal_m_s2[0]) < 1e-15 * G_SI
# A quarter cycle later it reaches its negative extreme.
assert np.isclose(real_signal_m_s2[1], -G_SI, rtol=1e-12, atol=0.0)
# Half a cycle after that, the positive one.
assert np.isclose(real_signal_m_s2[3], +G_SI, rtol=1e-12, atol=0.0)

# A real-valued source needs no such reconstruction: the result is already the
# field, and its dtype says so.
static = WeightedPointSource(positions_m=[[0.0, 0.0, 0.0]], weights_kg=[1.0])
assert point_acceleration(static, [[1.0, 0.0, 0.0]]).dtype == np.float64
