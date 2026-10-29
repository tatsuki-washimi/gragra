"""Thin per-sample observable dispatch for gragra.fields series types.

Each function evaluates one of the four fixed-geometry series types
(:class:`~gragra.fields.density_series.DensityTimeSeries`,
:class:`~gragra.fields.density_series.DensityPhasors`,
:class:`~gragra.fields.displacement_series.FixedMeshDisplacementSeries`,
:class:`~gragra.fields.surface_density_series.FixedMeshSurfaceDensitySeries`)
sample-by-sample through the existing ``WeightedPointSource``-based point
observables (:mod:`gragra.observables.potential`/``acceleration``/``gradient``)
or the existing displacement-transfer observables
(:mod:`gragra.observables.dipole`). No new source type or I/O format is
introduced here; this module only folds each series sample into the input
shape those existing functions already accept.

Output has shape ``(S, M, ...)`` with the sample axis leading, matching the
sample-leading convention used by :mod:`gragra.observables.streaming`.
"""

from __future__ import annotations

import numpy as np

from gragra.array_types import ComplexArray, FloatArray
from gragra.fields.density_series import DensityPhasors, DensityTimeSeries
from gragra.fields.displacement_series import FixedMeshDisplacementSeries
from gragra.fields.surface_density_series import FixedMeshSurfaceDensitySeries
from gragra.gravity_sources import WeightedPointSource
from gragra.observables.acceleration import point_acceleration
from gragra.observables.dipole import (
    displacement_transfer_acceleration,
    displacement_transfer_gravity_gradient,
    displacement_transfer_potential,
)
from gragra.observables.gradient import point_gravity_gradient
from gragra.observables.potential import point_potential

_POINT_OBSERVABLES = {
    "potential": point_potential,
    "acceleration": point_acceleration,
    "gradient": point_gravity_gradient,
}
_DISPLACEMENT_OBSERVABLES = {
    "potential": displacement_transfer_potential,
    "acceleration": displacement_transfer_acceleration,
    "gradient": displacement_transfer_gravity_gradient,
}


def _stack_samples(rows: list[np.ndarray]) -> FloatArray | ComplexArray:
    stacked = np.stack(rows, axis=0)
    stacked.setflags(write=False)
    return stacked


def _point_observable_fn(observable: str):
    try:
        return _POINT_OBSERVABLES[observable]
    except KeyError as error:
        raise ValueError(
            f"observable must be one of {sorted(_POINT_OBSERVABLES)}, "
            f"got {observable!r}"
        ) from error


def density_time_series_observable(
    series: DensityTimeSeries,
    observable: str,
    targets_m,
    *,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
    backend: str = "numpy",
) -> FloatArray:
    """Evaluate potential/acceleration/gradient over a density time series.

    Each sample is folded into ``weights_kg = density_kg_m3[s] * volumes_m3``
    (mass element, matching ``VolumeElementSource.as_weighted_source()``) and
    passed through the existing ``point_{observable}`` function unchanged.
    """
    fn = _point_observable_fn(observable)
    rows = [
        fn(
            WeightedPointSource(
                series.positions_m, series.density_kg_m3[s] * series.volumes_m3
            ),
            targets_m,
            softening_m=softening_m,
            chunk_size=chunk_size,
            backend=backend,
        )
        for s in range(series.sample_count)
    ]
    return _stack_samples(rows)


def density_phasors_observable(
    series: DensityPhasors,
    observable: str,
    targets_m,
    *,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
    backend: str = "numpy",
) -> ComplexArray:
    """Frequency-axis counterpart of :func:`density_time_series_observable`.

    ``density_kg_m3`` is complex128 (delta_rho phasors); the resulting
    ``weights_kg`` is complex, so the output is a complex amplitude per
    :mod:`gragra.observables.potential`/``acceleration``/``gradient``'s
    existing complex-output convention.
    """
    fn = _point_observable_fn(observable)
    rows = [
        fn(
            WeightedPointSource(
                series.positions_m, series.density_kg_m3[s] * series.volumes_m3
            ),
            targets_m,
            softening_m=softening_m,
            chunk_size=chunk_size,
            backend=backend,
        )
        for s in range(series.sample_count)
    ]
    return _stack_samples(rows)


def surface_density_series_observable(
    series: FixedMeshSurfaceDensitySeries,
    observable: str,
    targets_m,
    *,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
    backend: str = "numpy",
) -> FloatArray | ComplexArray:
    """Evaluate potential/acceleration/gradient over a surface-density series.

    Each sample is folded into
    ``weights_kg = areas_m2 * surface_density_kg_m2[s]`` and passed through
    the existing ``point_{observable}`` function unchanged.
    """
    fn = _point_observable_fn(observable)
    rows = [
        fn(
            WeightedPointSource(
                series.positions_m, series.areas_m2 * series.surface_density_kg_m2[s]
            ),
            targets_m,
            softening_m=softening_m,
            chunk_size=chunk_size,
            backend=backend,
        )
        for s in range(series.sample_count)
    ]
    return _stack_samples(rows)


def fixed_mesh_displacement_series_observable(
    series: FixedMeshDisplacementSeries,
    observable: str,
    targets_m,
    *,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
    backend: str = "numpy",
) -> FloatArray | ComplexArray:
    """Evaluate potential/acceleration/gradient over a displacement series.

    Each sample is passed through the existing
    ``displacement_transfer_{observable}`` function
    (:mod:`gragra.observables.dipole`) unchanged; background mass and
    displacement are a two-argument composition, not a ``weights_kg``-only
    quantity, so this path does not go through ``WeightedPointSource``.
    """
    try:
        fn = _DISPLACEMENT_OBSERVABLES[observable]
    except KeyError as error:
        raise ValueError(
            "observable must be one of "
            f"{sorted(_DISPLACEMENT_OBSERVABLES)}, got {observable!r}"
        ) from error
    rows = [
        fn(
            series.positions_m,
            series.mass_kg,
            series.displacement_m[s],
            targets_m,
            softening_m=softening_m,
            chunk_size=chunk_size,
            backend=backend,
        )
        for s in range(series.sample_count)
    ]
    return _stack_samples(rows)
