from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
import importlib.util

_MODULE_PATH = Path(__file__).parents[1] / "src/gragra/research/rayleigh_timefit.py"
_SPEC = importlib.util.spec_from_file_location("rayleigh_timefit", _MODULE_PATH)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
complex_relative_error = _MODULE.complex_relative_error
fit_complex_amplitude = _MODULE.fit_complex_amplitude
snapshot_physical_times = _MODULE.snapshot_physical_times


def test_snapshot_physical_times_uses_post_predictor_mapping():
    times = snapshot_physical_times(np.array([0, 1, 239]), t_init_s=10.0, dt_s=0.005)
    np.testing.assert_allclose(times, [10.005, 10.010, 11.200])
    assert not times.flags.writeable


def test_fit_recovers_non_bin_frequency_and_common_time_origin():
    times = np.arange(37, dtype=float) * 0.037
    omega = 2.0 * np.pi * 2.5
    t_ref = 0.123
    amplitude = np.array([1.2 - 0.4j, -0.3 + 0.9j])
    values = np.real(amplitude[None, :] * np.exp(-1j * omega * (times[:, None] - t_ref)))
    fitted = fit_complex_amplitude(values, times, omega, t_ref_s=t_ref)
    np.testing.assert_allclose(fitted, amplitude, rtol=1e-12, atol=1e-12)


def test_fit_rejects_underdetermined_or_invalid_inputs():
    with pytest.raises(ValueError, match="at least 3"):
        fit_complex_amplitude(np.ones((2, 1)), [0.0, 1.0], 1.0)
    with pytest.raises(ValueError, match="positive"):
        fit_complex_amplitude(np.ones((3, 1)), [0.0, 1.0, 2.0], 0.0)


def test_complex_relative_error_uses_one_vector_scale():
    reference = np.array([3.0 + 4.0j, 0.0j])
    numerical = reference + np.array([0.5, 0.0j])
    assert complex_relative_error(numerical, reference) == pytest.approx(0.5 / 5.0)
