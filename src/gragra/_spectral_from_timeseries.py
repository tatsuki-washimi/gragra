"""C3 discrete Fourier and amplitude from uniform TimeSeriesMatrix payloads."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from astropy import units as u
from gwexpy.frequencyseries import FrequencySeriesMatrix
from gwexpy.timeseries import TimeSeriesMatrix

from gragra.results._coords import (
    gps_offset_seconds,
    normalize_gps,
    one_sided_frequencies_hz,
)
from gragra.results._spectral import FiniteRecordInterpretation
from gragra.results._time import TimeSampling

_COMPONENTS = {
    "potential": ("potential",),
    "acceleration": ("x", "y", "z"),
    "gradient": ("xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"),
}
_SI_UNITS = {
    "potential": u.m**2 / u.s**2,
    "acceleration": u.m / u.s**2,
    "gradient": 1 / u.s**2,
}
_FOURIER_SI_UNITS = {
    "potential": u.m**2 / u.s,
    "acceleration": u.m / u.s,
    "gradient": 1 / u.s,
}


def parse_spectrum_kind(
    kind: str | FiniteRecordInterpretation,
) -> FiniteRecordInterpretation:
    if isinstance(kind, FiniteRecordInterpretation):
        return kind
    if isinstance(kind, str):
        normalized = kind.strip().lower()
        for member in FiniteRecordInterpretation:
            if member.value == normalized or member.name.lower() == normalized:
                return member
    raise ValueError(
        'spectrum kind must be "periodic" or "transient" '
        "(FiniteRecordInterpretation member)"
    )


def _c_k(m: int, k: int) -> float:
    if k == 0:
        return 1.0
    if m % 2 == 0 and k == m // 2:
        return 1.0
    return 2.0


def transform_time_payloads(
    payloads: Mapping[str, TimeSeriesMatrix],
    *,
    sampling: TimeSampling,
    observables: tuple[str, ...],
    tref: object,
    mode: str,
) -> dict[str, FrequencySeriesMatrix]:
    """Build frequency-domain matrices from time-series payloads (C3)."""
    tref_gps = normalize_gps(tref)
    m = sampling.count
    dt = sampling.dt_seconds
    tau = gps_offset_seconds(sampling.epoch_gps, tref_gps)
    freqs = tuple(float(value) for value in one_sided_frequencies_hz(m, dt))
    df = 1.0 / (m * dt)
    out: dict[str, FrequencySeriesMatrix] = {}
    first = payloads[observables[0]]
    target_ids = tuple(first.rows.keys())
    for observable in observables:
        matrix = payloads[observable]
        if not isinstance(matrix, TimeSeriesMatrix):
            raise TypeError(f"{observable} payload must be a TimeSeriesMatrix")
        components = _COMPONENTS[observable]
        n_target = len(target_ids)
        n_comp = len(components)
        n_bin = len(freqs)
        coeff = np.empty((n_target, n_comp, n_bin), dtype=np.complex128)
        si_unit = _SI_UNITS[observable]
        out_unit = _FOURIER_SI_UNITS[observable] if mode == "fourier" else si_unit
        # np.full(..., Unit, dtype=object) raises UnitConversionError for the
        # gradient units 1/s and 1/s^2; assign each cell explicitly instead.
        out_units = np.empty((n_target, n_comp), dtype=object)
        for row in range(n_target):
            for col in range(n_comp):
                out_units[row, col] = out_unit
        for row in range(n_target):
            for col in range(n_comp):
                cell_unit = matrix.meta[row, col].unit
                y_raw = np.asarray(matrix[row, col], dtype=np.float64)
                if y_raw.shape != (m,):
                    raise ValueError(f"{observable} cell shape must be ({m},)")
                try:
                    y = np.asarray(
                        u.Quantity(y_raw, cell_unit).to_value(si_unit),
                        dtype=np.float64,
                    )
                except (TypeError, ValueError, u.UnitConversionError) as exc:
                    raise ValueError(
                        f"{observable} cell ({row}, {col}) must convert to {si_unit}"
                    ) from exc
                if not np.isfinite(y).all():
                    raise ValueError(
                        f"{observable} cell ({row}, {col}) SI samples must be finite"
                    )
                x = np.fft.rfft(y)
                for k, xk in enumerate(x):
                    fk = k * df
                    phase = np.exp(-2j * np.pi * fk * tau)
                    if mode == "fourier":
                        coeff[row, col, k] = dt * phase * xk
                    else:
                        ck = _c_k(m, k)
                        coeff[row, col, k] = (ck / m) * phase * xk
                out_units[row, col] = out_unit
        out[observable] = FrequencySeriesMatrix(
            coeff,
            frequencies=np.asarray(freqs) * u.Hz,
            units=out_units,
            rows=list(target_ids),
            cols=list(components),
            name=f"m3b_{mode}_{observable}",
        )
    return out


__all__ = ["parse_spectrum_kind", "transform_time_payloads"]
