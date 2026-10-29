"""Examples of grouping existing SI and GWexpy objects with GravityResult.

The sphere example wraps results from the existing SI entry point. The U11
example uses the lower-level SI implementation
``gragra.observables.u11_rayleigh_small_cavity.rayleigh_small_cavity_acceleration``
to demonstrate wrapping; that function is not in ``gragra.observables.__all__``
and this example does not recommend it as a new public API.

The GWexpy examples follow A20: matrix rows identify targets, columns identify
components, and the internal data shape is (target, component, sample). The
frequency-indexed example demonstrates storage by frequency only; it does not
assign spectral semantics to that data.
"""

from __future__ import annotations

import numpy as np
from astropy import units as u
from gwexpy.frequencyseries import FrequencySeriesMatrix
from gwexpy.timeseries import TimeSeriesMatrix

from gragra.observables.analytic import uniform_sphere_observables
from gragra.observables.u11_rayleigh_small_cavity import (
    rayleigh_small_cavity_acceleration,
)
from gragra.results import GravityResult


def uniform_sphere_case() -> dict[str, object]:
    """Return the fixed SI inputs used by the uniform-sphere example."""
    return {
        "center_m": [0.0, 0.0, 0.0],
        "radius_m": 1.0,
        "density_kg_m3": 2000.0,
        "targets_m": np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 0.5]]),
    }


def uniform_sphere_result_example() -> GravityResult:
    """Wrap potential, acceleration, and gradient from the existing SI API."""
    observables = uniform_sphere_observables(**uniform_sphere_case())
    return GravityResult(
        name="uniform sphere SI observables",
        potential=u.Quantity(observables.potential, unit=u.m**2 / u.s**2, copy=True),
        acceleration=u.Quantity(observables.acceleration, unit=u.m / u.s**2, copy=True),
        gradient=u.Quantity(observables.gradient, unit=u.s**-2, copy=True),
    )


def u11_case() -> tuple[np.ndarray, dict[str, object]]:
    """Return the fixed sphere case for the lower-level U11 SI example."""
    targets = np.array([[13.0, -7.0, -90.0], [13.6, -7.3, -89.8]], dtype=float)
    kwargs: dict[str, object] = {
        "cavity_shape": "sphere",
        "center_m": [13.0, -7.0, -90.0],
        "radius_m": 2.0,
        "density_kg_m3": 2300.0,
        "v_p_m_s": 5800.0,
        "v_s_m_s": 3200.0,
        "frequency_hz": 5.0,
        "vertical_surface_amplitude_m": (1.0 + 0.3j) * 1e-9,
        "direction": [1.0, 2.0, 0.0],
        "phase_origin_m": [-4.0, 9.0, 0.0],
    }
    return targets, kwargs


def u11_result_examples() -> dict[str, GravityResult]:
    """Wrap the three named acceleration arrays returned by the U11 SI path.

    This is a wrapping demonstration for the lower-level SI implementation,
    not a recommendation to add that path to ``gragra.observables.__all__``.
    The returned arrays are already in m/s²; this example does not multiply by
    the gravitational constant again.
    """
    targets, kwargs = u11_case()
    parts = rayleigh_small_cavity_acceleration(targets, **kwargs)
    return {
        name: GravityResult(name=name, acceleration=value)
        for name, value in parts.items()
    }


def _matrix_metadata(prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    units = np.full((2, 3), "m/s^2", dtype=object)
    names = np.array(
        [
            [f"{prefix} target_a {component} acceleration" for component in "xyz"],
            [f"{prefix} target_b {component} acceleration" for component in "xyz"],
        ],
        dtype=object,
    )
    channels = np.array(
        [
            [f"GRAGRA:{prefix}:target_a:{component}" for component in "xyz"],
            [f"GRAGRA:{prefix}:target_b:{component}" for component in "xyz"],
        ],
        dtype=object,
    )
    return units, names, channels


def time_series_matrix_payload_example() -> TimeSeriesMatrix:
    """Create native time data with target rows and component columns."""
    data = np.arange(18, dtype=float).reshape(2, 3, 3) + 0.25
    units, names, channels = _matrix_metadata("time")
    return TimeSeriesMatrix(
        data,
        dt=0.25,
        t0=0.0,
        xunit="s",
        units=units,
        names=names,
        channels=channels,
        rows=["target_a", "target_b"],
        cols=["x", "y", "z"],
        name="time-indexed acceleration matrix",
    )


def frequency_series_matrix_payload_example() -> FrequencySeriesMatrix:
    """Create frequency-indexed storage without assigning spectral meaning."""
    data = np.arange(18, dtype=float).reshape(2, 3, 3) + 10.25
    units, names, channels = _matrix_metadata("frequency")
    return FrequencySeriesMatrix(
        data,
        frequencies=np.array([1.0, 2.0, 3.0]) * u.Hz,
        units=units,
        names=names,
        channels=channels,
        rows=["target_a", "target_b"],
        cols=["x", "y", "z"],
        name="frequency-indexed acceleration matrix",
    )


def wrap_native_matrix_result(payload: object, *, name: str) -> GravityResult:
    """Place a native matrix in an acceleration-only result envelope."""
    return GravityResult(name=name, acceleration=payload)


def main() -> None:
    """Run each result-wrapping example and print its compact summary."""
    sphere = uniform_sphere_result_example()
    u11 = u11_result_examples()

    time_payload = time_series_matrix_payload_example()
    time_result = wrap_native_matrix_result(
        time_payload, name="time-indexed acceleration result"
    )
    frequency_payload = frequency_series_matrix_payload_example()
    frequency_result = wrap_native_matrix_result(
        frequency_payload, name="frequency-indexed acceleration result"
    )

    print(f"Sphere result: {sphere.name}; potential unit={sphere.potential.unit}")
    print(f"U11 acceleration results: {', '.join(u11)}")
    print(
        "Time matrix result:",
        time_result.name,
        f"shape={time_result.acceleration.shape}",
    )
    print(
        "Frequency-indexed result: "
        f"{frequency_result.name}; shape={frequency_result.acceleration.shape}"
    )


if __name__ == "__main__":
    main()
