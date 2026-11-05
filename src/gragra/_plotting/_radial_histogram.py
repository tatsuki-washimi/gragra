"""Radial histograms of additive source quantities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
from astropy import units as u

from gragra._plotting._diagnostics import _resolve_source, _resolved_from
from gragra._plotting._hist import _resolve_edges
from gragra._plotting._source_quantities import (
    resolve_quantity,
    source_quantity_data,
    source_radial_membership,
)

if TYPE_CHECKING:
    from gwexpy.histogram import Histogram

    from gragra._evaluation_setup import ResolvedEvaluation
    from gragra._plotting._source_contracts import SourceContributionResult
    from gragra.evaluation import EvaluationSetup
    from gragra.geometry import Frame

_NORMALIZATIONS = (
    "none",
    "per_length",
    "fraction",
    "fraction_per_length",
)
_FRACTION_NORMALIZATIONS = frozenset({"fraction", "fraction_per_length"})


@dataclass(frozen=True, eq=False)
class RadialHistogramResult:
    """A raw GWexpy histogram and its selected finite-bin values.

    ``histogram`` always contains unnormalized integrated contributions in
    the source quantity's unit, including raw underflow and overflow. ``values``
    contains only finite-bin values after ``normalization`` is applied.
    """

    histogram: Histogram
    values: u.Quantity
    quantity: str
    normalization: str
    normalization_total: u.Quantity
    target_id: str
    target_xyz_m: np.ndarray
    frame: Frame
    domain: tuple[float, float] | None
    source_index: np.ndarray

    def __post_init__(self) -> None:
        values = u.Quantity(self.values, copy=True)
        total = u.Quantity(self.normalization_total, copy=True)
        if values.ndim != 1 or len(values) != len(self.histogram.values):
            raise ValueError("values must align with the finite histogram bins")
        target = np.array(self.target_xyz_m, dtype=np.float64, copy=True)
        indices = np.array(self.source_index, dtype=np.int64, copy=True)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("target_xyz_m must be a finite 3-vector")
        if indices.ndim != 1 or not np.array_equal(indices, np.arange(len(indices))):
            raise ValueError("source_index must be the stable ordered index 0..N-1")
        if not np.isfinite(values.value).all() or not np.isfinite(total.value).all():
            raise ValueError("radial histogram result values must be finite")
        target.setflags(write=False)
        indices.setflags(write=False)
        values.setflags(write=False)
        total.setflags(write=False)
        if self.domain is not None:
            low, high = self.domain
            if not (np.isfinite(low) and np.isfinite(high) and low <= high):
                raise ValueError("domain must be an ordered finite radius range")
            object.__setattr__(self, "domain", (float(low), float(high)))
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "normalization_total", total)
        object.__setattr__(self, "target_xyz_m", target)
        object.__setattr__(self, "source_index", indices)

    @property
    def unit(self) -> u.UnitBase:
        """Unit of the selected finite-bin ``values``."""
        return self.values.unit

    def cumulative(self, fraction: bool = False) -> u.Quantity:
        """Return raw cumulative integrated values through each finite bin.

        The sequence starts with underflow and ends at the last finite bin's
        right edge. Overflow is excluded. ``fraction=True`` divides by the
        signed all-source total and therefore requires a finite nonzero total.
        """
        if not isinstance(fraction, (bool, np.bool_)):
            raise TypeError("fraction must be a boolean")
        raw = np.asarray(self.histogram.values.value, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            cumulative = np.cumsum(raw, dtype=np.float64) + float(
                self.histogram.underflow.value
            )
        if not np.isfinite(cumulative).all():
            raise ValueError("cumulative radial contribution is not finite")
        unit = self.histogram.unit
        if fraction:
            denominator = float(self.normalization_total.value)
            if not np.isfinite(denominator) or denominator == 0.0:
                raise ValueError("normalization total must be finite and nonzero")
            cumulative = cumulative / denominator
            unit = u.dimensionless_unscaled
        result = u.Quantity(cumulative, unit=unit, copy=True)
        result.setflags(write=False)
        return result


@dataclass(frozen=True, eq=False)
class DynamicRadialHistogramResult:
    """One selected dynamic diagnostic as a radial GWexpy histogram.

    ``values`` and ``histogram.values`` contain finite radial bins. The
    histogram's underflow and overflow hold the corresponding radial-flow
    values. ``valid`` aligns with finite bins; flow validity is kept separately
    because phase and interference can be undefined in a flow group.
    """

    histogram: Histogram
    values: u.Quantity
    valid: np.ndarray
    quantity: str
    target_id: str
    target_xyz_m: np.ndarray
    component: str
    frequency_hz: float | None
    radial_edges_m: np.ndarray
    underflow_valid: bool
    overflow_valid: bool
    frame: str
    source_index: np.ndarray
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        values = u.Quantity(self.values, copy=True)
        valid = np.asarray(self.valid)
        if values.ndim != 1 or len(values) != len(self.histogram.values):
            raise ValueError("values must align with finite histogram bins")
        if valid.dtype.kind != "b" or valid.shape != values.shape:
            raise ValueError("valid must be a boolean vector aligned with finite bins")
        if not np.isfinite(values.value).all():
            raise ValueError("dynamic radial histogram values must be finite")
        target = np.array(self.target_xyz_m, dtype=np.float64, copy=True)
        edges = np.array(self.radial_edges_m, dtype=np.float64, copy=True)
        indices = np.array(self.source_index, dtype=np.int64, copy=True)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("target_xyz_m must be a finite 3-vector")
        if (
            edges.ndim != 1
            or len(edges) < 2
            or not np.isfinite(edges).all()
            or np.any(np.diff(edges) <= 0.0)
        ):
            raise ValueError("radial_edges_m must be finite and strictly increasing")
        if indices.ndim != 1 or not np.array_equal(indices, np.arange(len(indices))):
            raise ValueError("source_index must be the stable ordered index 0..N-1")
        if not isinstance(self.underflow_valid, (bool, np.bool_)) or not isinstance(
            self.overflow_valid, (bool, np.bool_)
        ):
            raise TypeError("flow validity values must be booleans")
        if self.frequency_hz is not None and not np.isfinite(self.frequency_hz):
            raise ValueError("frequency_hz must be finite or None")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        frozen_valid = np.frombuffer(
            np.array(valid, dtype=bool, copy=True).tobytes(), dtype=bool
        ).reshape(valid.shape)
        values.setflags(write=False)
        target.setflags(write=False)
        edges.setflags(write=False)
        indices.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "valid", frozen_valid)
        object.__setattr__(self, "target_xyz_m", target)
        object.__setattr__(self, "radial_edges_m", edges)
        object.__setattr__(self, "source_index", indices)
        object.__setattr__(self, "underflow_valid", bool(self.underflow_valid))
        object.__setattr__(self, "overflow_valid", bool(self.overflow_valid))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def unit(self) -> u.UnitBase:
        """Unit of the selected finite-bin values."""
        return self.values.unit


def radial_histogram(
    setup: EvaluationSetup | ResolvedEvaluation,
    quantity: str,
    *,
    target_id: str | None = None,
    bins: int | str | np.ndarray | u.Quantity = "auto",
    normalization: str = "none",
    frame: Frame | None = None,
) -> RadialHistogramResult:
    """Aggregate a source contribution quantity in distance bins.

    Every registered scalar source quantity can be integrated by distance.
    Extensive quantities such as ``mass_over_r2`` are summed directly.
    Intensive quantities such as ``density_over_r2`` are multiplied by their
    verified source-cell volumes before binning. Fixed aliases from the
    source-quantity registry resolve to canonical IDs. Distances and ratios
    use the unsoftened Euclidean radius.

    The raw ``histogram`` stores integrated finite-bin and flow contributions
    in the integrated quantity's unit. ``values`` uses one of four normalizations:
    ``none`` retains the integrated unit, ``per_length`` divides each bin by
    its width, ``fraction`` divides by the signed all-source total, and
    ``fraction_per_length`` performs both operations. The fraction denominator
    includes sources outside the displayed bin range.
    """
    spec = resolve_quantity(quantity)
    if normalization not in _NORMALIZATIONS:
        raise ValueError(
            f"normalization must be one of {_NORMALIZATIONS}, got {normalization!r}"
        )
    # Target-independent values are resolved without target_id. The radial
    # coordinate is selected separately, so users can still bin mass or
    # density around a chosen target.
    source_kwargs = {"frame": frame}
    if spec.target_dependent:
        source_kwargs["target_id"] = target_id
    source_data = source_quantity_data(setup, quantity, **source_kwargs)
    distance_data = source_quantity_data(
        setup, "distance", target_id=target_id, frame=frame
    )
    values = np.asarray(source_data.values, dtype=np.float64)
    radii = np.asarray(distance_data.values, dtype=np.float64)
    if values.shape != radii.shape or not np.array_equal(
        source_data.source_index, distance_data.source_index
    ):
        raise ValueError("source contribution and radial distance arrays do not align")
    if not (np.isfinite(values).all() and np.isfinite(radii).all()):
        raise ValueError("source contributions and radii must be finite")

    integrated_unit = source_data.unit
    if source_data.extensive:
        contributions = values
    else:
        volumes = np.asarray(
            _resolve_source(_resolved_from(setup)).volumes(), dtype=np.float64
        )
        if volumes.shape != values.shape or not np.isfinite(volumes).all():
            raise ValueError(
                "verified source volumes must be finite and source-aligned"
            )
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            contributions = values * volumes
        integrated_unit = source_data.unit * u.m**3
    if not np.isfinite(contributions).all():
        raise ValueError("integrated source contributions must be finite")

    edges = _resolve_edges(bins, radii, u.m)
    membership, _ = source_radial_membership(radii.reshape(1, -1), edges)
    # Avoid np.histogram's cumulative weighted-bin reduction: differencing
    # adjacent cumulative totals can erase a small bin after a large one.
    grouped = np.bincount(
        membership.reshape(-1),
        weights=contributions,
        minlength=len(edges) + 1,
    )
    raw_values = grouped[1:-1]
    underflow = float(grouped[0])
    overflow = float(grouped[-1])
    with np.errstate(over="ignore", invalid="ignore"):
        normalization_total = float(np.sum(contributions, dtype=np.float64))
    if not np.isfinite(raw_values).all() or not np.isfinite(underflow + overflow):
        raise ValueError("aggregated radial contributions are not finite")
    if not np.isfinite(normalization_total):
        raise ValueError("normalization total is not finite")
    if normalization in _FRACTION_NORMALIZATIONS and normalization_total == 0.0:
        raise ValueError("normalization total must be finite and nonzero")

    histogram = _raw_histogram(
        raw_values,
        edges,
        unit=integrated_unit,
        name=f"{spec.canonical} target={distance_data.target_id}",
        underflow=underflow,
        overflow=overflow,
    )
    values = raw_values.copy()
    value_unit = integrated_unit
    if normalization in _FRACTION_NORMALIZATIONS:
        values = values / normalization_total
        value_unit = u.dimensionless_unscaled
    if normalization in {"per_length", "fraction_per_length"}:
        with np.errstate(over="ignore", invalid="ignore"):
            widths = np.diff(edges)
        if not np.isfinite(widths).all():
            raise ValueError("radial bin widths must be finite")
        values = values / widths
        value_unit = value_unit / u.m
    if not np.isfinite(values).all():
        raise ValueError("normalized radial histogram values are not finite")

    domain = None if radii.size == 0 else (float(np.min(radii)), float(np.max(radii)))
    return RadialHistogramResult(
        histogram=histogram,
        values=u.Quantity(values, unit=value_unit, copy=True),
        quantity=spec.canonical,
        normalization=normalization,
        normalization_total=u.Quantity(
            np.array(normalization_total), unit=integrated_unit, copy=True
        ),
        target_id=distance_data.target_id,
        target_xyz_m=distance_data.target_xyz_m,
        frame=distance_data.frame,
        domain=domain,
        source_index=source_data.source_index,
    )


_DYNAMIC_QUANTITIES = {
    "complex_real",
    "complex_imag",
    "amplitude",
    "phase",
    "amplitude_fraction",
    "rms_amplitude_fraction",
    "interference_share",
    "cumulative_real",
    "cumulative_imag",
}
_DYNAMIC_STATISTICS = {
    "phase": "phase_rad",
    "amplitude_fraction": "amplitude_share",
    "interference_share": "interference_share",
    "cumulative_real": "cumulative_complex",
    "cumulative_imag": "cumulative_complex",
}


def dynamic_radial_histogram(
    result: SourceContributionResult | Sequence[SourceContributionResult],
    quantity: str,
    *,
    target_id: str | None = None,
    frequency_hz: object | None = None,
    component: str | None = None,
    frequency_weights: object | None = None,
    direction_weights: object | None = None,
) -> DynamicRadialHistogramResult:
    """Build a radial ``Histogram`` from stored dynamic source groups.

    ``result`` must come from ``source_contributions(...,
    radial_edges_m=...)`` so its immutable group data already represent
    underflow, finite distance bins, and overflow. Frequency-resolved
    quantities require one exact ``frequency_hz``. RMS amplitude fraction
    instead requires ``frequency_weights`` and may combine compatible results
    using ``direction_weights``. A Histogram stores real values, so cumulative
    complex responses are selected as real and imaginary parts separately.
    """
    from gragra._plotting._source_contracts import SourceContributionResult

    if not isinstance(quantity, str):
        raise TypeError("quantity must be a string")
    if quantity not in _DYNAMIC_QUANTITIES:
        raise ValueError(
            f"quantity must be one of {tuple(sorted(_DYNAMIC_QUANTITIES))}"
        )
    if isinstance(result, SourceContributionResult):
        results = (result,)
    else:
        if isinstance(result, (str, bytes)):
            raise TypeError("result must be a SourceContributionResult or sequence")
        try:
            results = tuple(result)
        except TypeError as error:
            raise TypeError(
                "result must be a SourceContributionResult or sequence"
            ) from error
    if not results or any(
        not isinstance(item, SourceContributionResult) for item in results
    ):
        raise TypeError("result must contain SourceContributionResult values")

    reference = results[0]
    data = reference._data
    if data.radial_edges_m is None:
        raise ValueError(
            "source contributions must have radial_edges_m to build a radial histogram"
        )
    edges = np.asarray(data.radial_edges_m, dtype=np.float64)
    expected_group_ids = (
        "underflow",
        *(f"bin_{index}" for index in range(len(edges) - 1)),
        "overflow",
    )
    if tuple(data.group_ids) != expected_group_ids:
        raise ValueError("dynamic radial groups do not match their stored bin edges")

    for candidate in results[1:]:
        candidate_data = candidate._data
        if (
            candidate_data.group_ids != data.group_ids
            or candidate_data.radial_edges_m is None
            or not np.array_equal(candidate_data.radial_edges_m, edges)
        ):
            raise ValueError("direction results must have identical radial groups")

    if quantity == "rms_amplitude_fraction":
        if frequency_hz is not None:
            raise ValueError(
                "frequency_hz cannot be combined with rms_amplitude_fraction; "
                "use frequency_weights"
            )
        if frequency_weights is None:
            raise ValueError("rms_amplitude_fraction requires frequency_weights")
        if len(results) > 1 and direction_weights is None:
            raise ValueError("multiple direction results require direction_weights")
        if len(results) == 1 and direction_weights is None:
            from gragra._plotting._dynamic_contribution import contribution_statistics

            statistics = contribution_statistics(
                reference, frequency_weights=frequency_weights
            )
        else:
            from gragra._plotting._dynamic_contribution import (
                directional_contribution_statistics,
            )

            statistics = directional_contribution_statistics(
                results,
                direction_weights=direction_weights,
                frequency_weights=frequency_weights,
            )
        frequency_index = None
        selected_frequency = None
    else:
        if len(results) != 1:
            raise ValueError("only rms_amplitude_fraction accepts multiple results")
        if frequency_hz is None:
            raise ValueError(f"{quantity} requires frequency_hz")
        if frequency_weights is not None or direction_weights is not None:
            raise ValueError(
                "frequency_weights and direction_weights apply only to "
                "rms_amplitude_fraction"
            )
        frequency = _frequency_hz(frequency_hz)
        matches = np.flatnonzero(data.frequencies_hz == frequency)
        if len(matches) != 1:
            raise ValueError(
                "frequency_hz must exactly match one stored frequency; "
                f"got {frequency:g} Hz"
            )
        frequency_index = int(matches[0])
        selected_frequency = frequency
        if quantity in _DYNAMIC_STATISTICS:
            from gragra._plotting._dynamic_contribution import (
                contribution_statistics,
            )

            statistics = contribution_statistics(reference)

    if target_id is None:
        if len(data.target_ids) != 1:
            raise ValueError("target_id is required when multiple targets are present")
        target_id = data.target_ids[0]
    if not isinstance(target_id, str) or target_id not in data.target_ids:
        raise ValueError(f"unknown target_id {target_id!r}")
    target_index = data.target_ids.index(target_id)

    component_labels = tuple(reference.component_labels)
    if component is None:
        if len(component_labels) != 1:
            raise ValueError("component is required for vector or tensor responses")
        component = component_labels[0]
    if not isinstance(component, str) or component not in component_labels:
        raise ValueError(
            f"unknown component {component!r}; choose from {component_labels}"
        )
    component_index = component_labels.index(component)

    selected_groups = None
    if quantity != "rms_amplitude_fraction":
        group_values = reference.to_array(kind="group", layout="sample_first")
        group_values = group_values.reshape(
            data.frequency_count,
            data.target_count,
            len(data.group_ids),
            len(component_labels),
        )
        selected_groups = group_values[:, target_index, :, component_index]
    statistic_name = _DYNAMIC_STATISTICS.get(quantity)
    if quantity == "rms_amplitude_fraction":
        metric_unit = u.dimensionless_unscaled
        statistic = statistics.rms
        if statistic is None:
            raise ValueError("rms_amplitude_fraction requires frequency_weights")
        group_metrics = np.zeros(len(data.group_ids), dtype=np.float64)
        group_valid = np.zeros(len(data.group_ids), dtype=bool)
        for group_index, group_id in enumerate(data.group_ids):
            group_metrics[group_index] = float(
                statistic.rms_share[group_id].to_value(u.dimensionless_unscaled)[
                    target_index, component_index
                ]
            )
            group_valid[group_index] = bool(
                statistic.rms_share_valid[group_id].to_array()[
                    target_index, component_index
                ]
            )
    elif quantity in _DYNAMIC_STATISTICS:
        statistic = getattr(statistics, statistic_name)
        group_metrics = np.zeros(len(data.group_ids), dtype=np.float64)
        group_valid = np.ones(len(data.group_ids), dtype=bool)
        for group_index, group_id in enumerate(data.group_ids):
            values = statistic.to_array(group_id=group_id, layout="sample_first")
            selected = values[frequency_index, target_index, component_index]
            component_value = (
                selected.imag if quantity == "cumulative_imag" else selected.real
            )
            group_metrics[group_index] = float(component_value)
            if statistic_name != "cumulative_complex":
                group_valid[group_index] = bool(
                    statistic.validity(group_id).to_array(layout="sample_first")[
                        frequency_index, target_index, component_index
                    ]
                )
        metric_unit = (
            u.rad
            if quantity == "phase"
            else u.dimensionless_unscaled
            if quantity in {"amplitude_fraction", "interference_share"}
            else u.Unit(data.unit)
        )
    else:
        assert selected_groups is not None
        selected = selected_groups[frequency_index]
        if quantity == "complex_real":
            group_metrics = selected.real.astype(np.float64, copy=True)
            metric_unit = u.Unit(data.unit)
        elif quantity == "complex_imag":
            group_metrics = selected.imag.astype(np.float64, copy=True)
            metric_unit = u.Unit(data.unit)
        else:
            group_metrics = np.abs(selected).astype(np.float64, copy=False)
            metric_unit = u.Unit(data.unit)
        group_valid = np.ones(len(data.group_ids), dtype=bool)

    if not np.isfinite(group_metrics).all():
        raise ValueError("dynamic radial histogram values must be finite")
    finite_values = group_metrics[1:-1]
    name = f"{quantity} target={target_id} component={component}"
    if selected_frequency is not None:
        name += f" frequency={selected_frequency:g} Hz"
    histogram = _raw_histogram(
        finite_values,
        edges,
        unit=metric_unit,
        name=name,
        underflow=float(group_metrics[0]),
        overflow=float(group_metrics[-1]),
    )
    weight_tuple = _weight_tuple(frequency_weights)
    direction_tuple = _weight_tuple(direction_weights)
    metadata = {
        "quantity": quantity,
        "metric_definition": _dynamic_metric_definition(quantity),
        "target_id": target_id,
        "target_xyz_m": tuple(float(value) for value in data.targets_m[target_index]),
        "component": component,
        "frequency_hz": selected_frequency,
        "frequencies_hz": tuple(float(value) for value in data.frequencies_hz),
        "frequency_weights": weight_tuple,
        "direction_weights": direction_tuple,
        "radial_edges_m": tuple(float(value) for value in edges),
        "group_ids": tuple(data.group_ids),
        "frame": data.frame,
        "unit": metric_unit.to_string(),
    }
    return DynamicRadialHistogramResult(
        histogram=histogram,
        values=u.Quantity(finite_values, unit=metric_unit, copy=True),
        valid=group_valid[1:-1],
        quantity=quantity,
        target_id=target_id,
        target_xyz_m=data.targets_m[target_index],
        component=component,
        frequency_hz=selected_frequency,
        radial_edges_m=edges,
        underflow_valid=bool(group_valid[0]),
        overflow_valid=bool(group_valid[-1]),
        frame=data.frame,
        source_index=data.source_index,
        metadata=metadata,
    )


def _frequency_hz(value: object) -> float:
    if isinstance(value, u.Quantity):
        raw = np.asarray(value.to_value(u.Hz))
    else:
        raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
        raise ValueError("frequency_hz must be one finite scalar in Hz")
    frequency = float(raw)
    if not np.isfinite(frequency):
        raise ValueError("frequency_hz must be one finite scalar in Hz")
    return frequency


def _weight_tuple(value: object | None) -> tuple[float, ...] | None:
    if value is None:
        return None
    if isinstance(value, u.Quantity):
        raw = np.asarray(value.to_value(u.dimensionless_unscaled))
    else:
        raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
        raise ValueError("weights must be a real vector")
    return tuple(float(item) for item in raw)


def _dynamic_metric_definition(quantity: str) -> str:
    return {
        "complex_real": (
            "real part of the complex response summed within each radial group"
        ),
        "complex_imag": (
            "imaginary part of the complex response summed within each radial group"
        ),
        "amplitude": (
            "absolute value of the complex response summed within each radial group"
        ),
        "phase": "phase of the complex response summed within each radial group",
        "amplitude_fraction": (
            "group amplitude divided by the sum of all group amplitudes"
        ),
        "rms_amplitude_fraction": (
            "group weighted RMS amplitude divided by the sum of group weighted RMS "
            "amplitudes"
        ),
        "interference_share": (
            "real group-total cross contribution divided by total response power"
        ),
        "cumulative_real": "real part of the near-to-far cumulative complex response",
        "cumulative_imag": (
            "imaginary part of the near-to-far cumulative complex response"
        ),
    }[quantity]


def _raw_histogram(
    values: np.ndarray,
    edges: np.ndarray,
    *,
    unit: u.UnitBase,
    name: str,
    underflow: float,
    overflow: float,
) -> Histogram:
    from gwexpy.histogram import Histogram

    return Histogram(
        values=values,
        edges=edges,
        unit=unit,
        xunit=u.m,
        underflow=underflow,
        overflow=overflow,
        name=name,
    )
