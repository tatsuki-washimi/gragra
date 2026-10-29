"""Package labelled phasor values in native frequency matrices."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
from astropy import units as u
from gwexpy.frequencyseries import FrequencySeriesMatrix

from gragra._evaluation_setup import OBSERVABLES

_COMPONENTS = {
    "potential": ("potential",),
    "acceleration": ("x", "y", "z"),
    "gradient": ("xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"),
}


def _phasor_matrices(
    arrays: Mapping[str, np.ndarray],
    target_ids: tuple[str, ...],
    frequencies_hz: tuple[float, ...],
    *,
    units: Mapping[str, u.UnitBase],
    cell_name: Callable[[str, str, str, int, int], str],
    cell_channel: Callable[[str, str, str, int, int], str],
    matrix_name: Callable[[str], str],
) -> dict[str, FrequencySeriesMatrix]:
    """Pack ``(target, component, frequency)`` arrays with supplied metadata."""
    count, bins = len(target_ids), len(frequencies_hz)
    result = {}
    for observable in OBSERVABLES:
        components = _COMPONENTS[observable]
        raw = np.asarray(arrays[observable], dtype=np.complex128)
        expected = (count, len(components), bins)
        if raw.shape != expected or not np.isfinite(raw).all():
            raise ValueError(
                f"{observable} must have finite phasor shape {expected}, "
                f"got {raw.shape}"
            )
        names = np.asarray(
            [
                [
                    cell_name(observable, target, component, row, column)
                    for column, component in enumerate(components)
                ]
                for row, target in enumerate(target_ids)
            ],
            dtype=object,
        )
        channels = np.asarray(
            [
                [
                    cell_channel(observable, target, component, row, column)
                    for column, component in enumerate(components)
                ]
                for row, target in enumerate(target_ids)
            ],
            dtype=object,
        )
        result[observable] = FrequencySeriesMatrix(
            raw.copy(),
            frequencies=np.asarray(frequencies_hz) * u.Hz,
            units=np.full(
                (count, len(components)), u.Unit(units[observable]), dtype=object
            ),
            names=names,
            channels=channels,
            rows=list(target_ids),
            cols=list(components),
            name=matrix_name(observable),
        )
    return result
