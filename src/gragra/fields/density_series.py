"""Fixed-volume density time and frequency series."""

from dataclasses import dataclass

import numpy as np

from gragra.fields._series import _array, _common, _finite_real, _Series


@dataclass(frozen=True, init=False)
class DensityTimeSeries(_Series):
    volumes_m3: np.ndarray
    density_kg_m3: np.ndarray
    density_kind: str
    time_s: np.ndarray

    def __init__(
        self,
        positions_m,
        volumes_m3,
        density_kg_m3,
        time_s,
        density_kind,
        coordinate_frame,
        coordinate_origin_m,
        source_ids,
        source_metadata=None,
    ):
        n = len(positions_m)
        s = len(time_s)
        _common(
            self,
            positions_m,
            time_s,
            "time",
            coordinate_frame,
            coordinate_origin_m,
            source_ids,
            source_metadata,
        )
        v = _array("volumes_m3", volumes_m3, (n,))
        d = _array("density_kg_m3", density_kg_m3, (s, n))
        if (v <= 0).any():
            raise ValueError("volumes_m3 must be positive")
        if density_kind not in ("rho", "delta_rho"):
            raise ValueError("density_kind must be 'rho' or 'delta_rho'")
        if density_kind == "rho" and (d < 0).any():
            raise ValueError("rho density must be non-negative")
        object.__setattr__(self, "volumes_m3", v)
        object.__setattr__(self, "density_kg_m3", d)
        object.__setattr__(self, "density_kind", density_kind)
        object.__setattr__(self, "time_s", self.axis_values)


@dataclass(frozen=True, init=False)
class DensityPhasors(_Series):
    volumes_m3: np.ndarray
    density_kg_m3: np.ndarray
    frequency_hz: np.ndarray
    t_ref_s: float
    phase_origin_m: np.ndarray
    density_kind: str = "delta_rho"
    fourier_convention: str = "Re[q_hat exp(-2pi i f(t-t_ref_s))]"

    def __init__(
        self,
        positions_m,
        volumes_m3,
        density_kg_m3,
        frequency_hz,
        t_ref_s,
        phase_origin_m,
        coordinate_frame,
        coordinate_origin_m,
        source_ids,
        source_metadata=None,
    ):
        n = len(positions_m)
        s = len(frequency_hz)
        _common(
            self,
            positions_m,
            frequency_hz,
            "frequency",
            coordinate_frame,
            coordinate_origin_m,
            source_ids,
            source_metadata,
        )
        v = _array("volumes_m3", volumes_m3, (n,))
        d = _array("density_kg_m3", density_kg_m3, (s, n), complex_ok=True)
        if (v <= 0).any():
            raise ValueError("volumes_m3 must be positive")
        object.__setattr__(self, "volumes_m3", v)
        object.__setattr__(self, "density_kg_m3", d)
        object.__setattr__(self, "frequency_hz", self.axis_values)
        object.__setattr__(self, "t_ref_s", _finite_real("t_ref_s", t_ref_s))
        object.__setattr__(
            self, "phase_origin_m", _array("phase_origin_m", phase_origin_m, (3,))
        )
