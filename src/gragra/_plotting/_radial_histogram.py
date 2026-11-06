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
    from gragra._plotting._dynamic_contribution import (
        ContributionStatistics,
        DirectionalContributionStatistics,
    )
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
    norm: str | None = None
    softening_m: float = 0.0

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
        if self.norm is not None and (
            not isinstance(self.norm, str) or self.norm not in {"frobenius", "spectral"}
        ):
            raise ValueError("norm must be 'frobenius', 'spectral', or None")
        softening = float(self.softening_m)
        if not np.isfinite(softening) or softening < 0.0:
            raise ValueError("softening_m must be finite and non-negative")
        if self.norm is None and softening != 0.0:
            raise ValueError("softening_m requires a gravity-gradient kernel norm")
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
        object.__setattr__(self, "softening_m", softening)

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


_GG_COMPONENT_INDICES = {
    "xx": (0, 0),
    "xy": (0, 1),
    "xz": (0, 2),
    "yx": (1, 0),
    "yy": (1, 1),
    "yz": (1, 2),
    "zx": (2, 0),
    "zy": (2, 1),
    "zz": (2, 2),
}


@dataclass(frozen=True, eq=False)
class GravityGradientRadialHistogramResult:
    """Signed static GG tensor sums grouped by source-target distance."""

    tensor_values: u.Quantity
    underflow_tensor: u.Quantity
    overflow_tensor: u.Quantity
    total_tensor: u.Quantity
    radial_edges_m: np.ndarray
    target_id: str
    target_xyz_m: np.ndarray
    frame: Frame
    source_index: np.ndarray
    softening_m: float
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        tensor = u.Quantity(self.tensor_values, copy=True)
        underflow = u.Quantity(self.underflow_tensor, copy=True).to(tensor.unit)
        overflow = u.Quantity(self.overflow_tensor, copy=True).to(tensor.unit)
        total = u.Quantity(self.total_tensor, copy=True).to(tensor.unit)
        if tensor.ndim != 3 or tensor.shape[1:] != (3, 3):
            raise ValueError("tensor_values must have shape (bins, 3, 3)")
        if underflow.shape != (3, 3) or overflow.shape != (3, 3):
            raise ValueError("flow tensors must have shape (3, 3)")
        if total.shape != (3, 3):
            raise ValueError("total_tensor must have shape (3, 3)")
        for value in (tensor, underflow, overflow, total):
            raw_value = np.asarray(value.value)
            if raw_value.dtype.kind == "b" or np.iscomplexobj(raw_value):
                raise TypeError("gravity-gradient tensors must contain real values")
        if not (
            np.isfinite(tensor.value).all()
            and np.isfinite(underflow.value).all()
            and np.isfinite(overflow.value).all()
            and np.isfinite(total.value).all()
        ):
            raise ValueError("gravity-gradient radial values must be finite")
        if not (
            tensor.unit.is_equivalent(u.s**-2)
            and underflow.unit.is_equivalent(tensor.unit)
            and overflow.unit.is_equivalent(tensor.unit)
            and total.unit.is_equivalent(tensor.unit)
        ):
            raise ValueError(
                "all GG tensors must use a common acceleration-gradient unit"
            )
        edges = np.array(self.radial_edges_m, dtype=np.float64, copy=True)
        target = np.array(self.target_xyz_m, dtype=np.float64, copy=True)
        raw_indices = np.asarray(self.source_index)
        if raw_indices.dtype.kind not in "iu":
            raise TypeError("source_index must contain integers")
        if (
            raw_indices.dtype.kind == "u"
            and raw_indices.size
            and raw_indices.max() > (2**63 - 1)
        ):
            raise ValueError("source_index values must fit in int64")
        indices = np.array(raw_indices, dtype=np.int64, copy=True)
        if (
            edges.ndim != 1
            or len(edges) < 2
            or not np.isfinite(edges).all()
            or np.any(np.diff(edges) <= 0.0)
        ):
            raise ValueError("radial_edges_m must be finite and strictly increasing")
        if tensor.shape[0] != len(edges) - 1:
            raise ValueError("tensor_values must align with radial_edges_m")
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("target_xyz_m must be a finite 3-vector")
        if indices.ndim != 1 or not np.array_equal(indices, np.arange(len(indices))):
            raise ValueError("source_index must be the stable ordered index 0..N-1")
        if isinstance(self.softening_m, (bool, np.bool_)):
            raise ValueError("softening_m must be finite and non-negative")
        softening = float(self.softening_m)
        if not np.isfinite(softening) or softening < 0.0:
            raise ValueError("softening_m must be finite and non-negative")
        if not isinstance(self.target_id, str) or not self.target_id:
            raise ValueError("target_id must be a nonempty string")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        for value in (tensor, underflow, overflow, total):
            value.setflags(write=False)
        for array in (edges, target, indices):
            array.setflags(write=False)
        object.__setattr__(self, "tensor_values", tensor)
        object.__setattr__(self, "underflow_tensor", underflow)
        object.__setattr__(self, "overflow_tensor", overflow)
        object.__setattr__(self, "total_tensor", total)
        object.__setattr__(self, "radial_edges_m", edges)
        object.__setattr__(self, "target_xyz_m", target)
        object.__setattr__(self, "source_index", indices)
        object.__setattr__(self, "softening_m", softening)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def unit(self) -> u.UnitBase:
        """Unit of the tensor contributions, normally s^-2."""
        return self.tensor_values.unit

    def histogram(self, component: str) -> Histogram:
        """Return one signed tensor component as a GWexpy radial histogram."""
        if not isinstance(component, str) or component not in _GG_COMPONENT_INDICES:
            raise ValueError(f"component must be one of {tuple(_GG_COMPONENT_INDICES)}")
        row, column = _GG_COMPONENT_INDICES[component]
        return _raw_histogram(
            np.asarray(self.tensor_values[:, row, column].value),
            self.radial_edges_m,
            unit=self.unit,
            name=f"gravity_gradient target={self.target_id} component={component}",
            underflow=float(self.underflow_tensor[row, column].value),
            overflow=float(self.overflow_tensor[row, column].value),
        )


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
    norm: str | None = None,
    softening_m: float = 0.0,
) -> RadialHistogramResult:
    """Aggregate a source contribution quantity in distance bins.

    Every registered scalar source quantity can be integrated by distance.
    Extensive quantities such as ``mass_over_r2`` are summed directly.
    Intensive quantities such as ``density_over_r2`` are multiplied by their
    verified source-cell volumes before binning. Fixed aliases from the
    source-quantity registry resolve to canonical IDs. Distances and ratios
    use the unsoftened Euclidean radius. ``gravity_gradient_kernel_norm``
    requires ``norm="frobenius"`` or ``norm="spectral"``; each source kernel
    is normed before radial grouping. Its kernel uses ``softening_m`` while
    radial bin membership continues to use the unsoftened radius.

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
    source_data = source_quantity_data(
        setup,
        quantity,
        norm=norm,
        softening_m=softening_m,
        **source_kwargs,
    )
    values = np.asarray(source_data.values, dtype=np.float64)
    if spec.requires_kernel_norm:
        assert source_data.radial_distance_m is not None
        radii = np.asarray(source_data.radial_distance_m, dtype=np.float64)
        radial_target_id = source_data.target_id
        radial_target_xyz_m = source_data.target_xyz_m
        radial_frame = source_data.frame
        radial_source_index = source_data.source_index
    else:
        distance_data = source_quantity_data(
            setup, "distance", target_id=target_id, frame=frame
        )
        radii = np.asarray(distance_data.values, dtype=np.float64)
        radial_target_id = distance_data.target_id
        radial_target_xyz_m = distance_data.target_xyz_m
        radial_frame = distance_data.frame
        radial_source_index = distance_data.source_index
    if values.shape != radii.shape or not np.array_equal(
        source_data.source_index, radial_source_index
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
        name=(
            f"{spec.canonical} norm={source_data.kernel_norm} target={radial_target_id}"
            if spec.requires_kernel_norm
            else f"{spec.canonical} target={radial_target_id}"
        ),
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
        target_id=radial_target_id,
        target_xyz_m=radial_target_xyz_m,
        frame=radial_frame,
        domain=domain,
        source_index=source_data.source_index,
        norm=source_data.kernel_norm,
        softening_m=source_data.softening_m,
    )


def gravity_gradient_radial_histogram(
    setup: EvaluationSetup | ResolvedEvaluation,
    *,
    target_id: str | None = None,
    bins: int | str | np.ndarray | u.Quantity = "auto",
    softening_m: float | None = None,
) -> GravityGradientRadialHistogramResult:
    """Sum the signed static gravity-gradient tensor in distance bins.

    Each source term is evaluated by the established point gravity-gradient
    kernel with its Plummer softening and physical mass weight. Bin membership
    uses the unsoftened Euclidean source-target distance. The result retains
    all nine signed tensor components, including underflow and overflow.
    """
    from gragra._plotting._source_field import (
        _BLOCK_SIZE,
        _check_static_physical_precision,
        _static_source_values,
    )
    from gragra._plotting._source_quantities import source_quantity_data
    from gragra.observables._source_terms import iter_point_observable_source_terms

    source_data = _static_source_values(
        setup,
        "gravity_gradient_contribution",
        target_id=target_id,
        aggregation="sum",
        softening_m=softening_m,
        norm=None,
    )
    _check_static_physical_precision(source_data, "gravity_gradient")
    if (
        source_data.target_id is None
        or source_data.target_xyz_m is None
        or source_data.physical_weights is None
    ):
        raise RuntimeError("static gravity-gradient source data lack target context")
    target_id_value = source_data.target_id
    difference = source_data.positions_m - source_data.target_xyz_m
    with np.errstate(over="ignore", invalid="ignore"):
        squared_radii = np.sum(difference * difference, axis=1)
    stable_radii = np.hypot.reduce(difference, axis=1)
    distance_requires_stable_norm = ~np.isfinite(squared_radii) | (
        squared_radii < np.finfo(np.float64).tiny
    )
    if source_data.softening_m > 0.0 and np.any(distance_requires_stable_norm):
        radii = stable_radii
    else:
        distance_data = source_quantity_data(
            setup,
            "distance",
            target_id=target_id_value,
            frame=source_data.frame,
        )
        if not np.array_equal(distance_data.source_index, source_data.source_index):
            raise ValueError("source positions and radial distances do not align")
        radii = np.asarray(distance_data.values, dtype=np.float64)
    if not np.isfinite(radii).all():
        raise ValueError("source-target distances must be finite")
    edges = _resolve_edges(bins, radii, u.m)
    membership, group_ids = source_radial_membership(radii.reshape(1, -1), edges)
    expected_group_count = len(edges) + 1
    if len(group_ids) != expected_group_count:
        raise RuntimeError("radial membership returned an unexpected group layout")

    grouped = np.zeros((expected_group_count, 3, 3), dtype=np.float64)
    total = np.zeros((3, 3), dtype=np.float64)
    grouped_flat = grouped.reshape(expected_group_count, 9)
    total_flat = total.reshape(9)
    for source_slice, terms in iter_point_observable_source_terms(
        source_data.positions_m,
        source_data.physical_weights,
        source_data.target_xyz_m.reshape(1, 3),
        observable="gravity_gradient",
        softening_m=source_data.softening_m,
        chunk_size=_BLOCK_SIZE,
    ):
        block = np.asarray(terms[0], dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError("gravity-gradient source terms must be finite")
        block_flat = block.reshape(len(block), 9)
        block_membership = membership[0, source_slice]
        for component_index in range(9):
            np.add.at(
                grouped_flat[:, component_index],
                block_membership,
                block_flat[:, component_index],
            )
            np.add.at(
                total_flat[component_index : component_index + 1],
                np.zeros(len(block), dtype=np.intp),
                block_flat[:, component_index],
            )

    if not np.isfinite(grouped).all() or not np.isfinite(total).all():
        raise ValueError("aggregated gravity-gradient tensors must be finite")
    unit = source_data.unit
    underflow = grouped[0]
    overflow = grouped[-1]
    finite = grouped[1:-1]
    target_xyz = np.asarray(source_data.target_xyz_m, dtype=np.float64)
    metadata = {
        "quantity": "gravity_gradient_contribution",
        "metric_definition": (
            "G times the source-mass-weighted signed sum of the point "
            "gravity-gradient tensor within each geometric distance group"
        ),
        "component_order": tuple(_GG_COMPONENT_INDICES),
        "unit": unit.to_string(),
        "target_id": source_data.target_id,
        "target_xyz_m": tuple(float(value) for value in target_xyz),
        "frame": source_data.frame.name,
        "coordinate_origin_m": tuple(
            float(value) for value in source_data.coordinate_origin_m
        ),
        "radial_edges_m": tuple(float(value) for value in edges),
        "group_ids": tuple(group_ids),
        "radial_distance_definition": "unsoftened Euclidean source-target distance",
        "softening_m": source_data.softening_m,
        "softening_kernel": "existing Plummer-softened point gravity-gradient kernel",
        "source_index": tuple(int(value) for value in source_data.source_index),
    }
    return GravityGradientRadialHistogramResult(
        tensor_values=u.Quantity(finite, unit=unit, copy=True),
        underflow_tensor=u.Quantity(underflow, unit=unit, copy=True),
        overflow_tensor=u.Quantity(overflow, unit=unit, copy=True),
        total_tensor=u.Quantity(total, unit=unit, copy=True),
        radial_edges_m=edges,
        target_id=source_data.target_id,
        target_xyz_m=target_xyz,
        frame=source_data.frame,
        source_index=source_data.source_index,
        softening_m=source_data.softening_m,
        metadata=metadata,
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
    using ``direction_weights``. Interference share may also combine compatible
    directional results; its cross term and total power are averaged before
    the ratio is formed. A Histogram stores real values, so cumulative complex
    responses are selected as real and imaginary parts separately.
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

    statistics: ContributionStatistics | DirectionalContributionStatistics
    directional_statistics: DirectionalContributionStatistics | None = None
    normalized_direction_weights: tuple[float, ...] | None = None
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
            directional_statistics = statistics
        frequency_index = None
        selected_frequency = None
    elif quantity == "interference_share":
        if frequency_weights is not None:
            raise ValueError("frequency_weights apply only to rms_amplitude_fraction")
        if frequency_hz is None:
            raise ValueError(f"{quantity} requires frequency_hz")
        if len(results) > 1 and direction_weights is None:
            raise ValueError("multiple direction results require direction_weights")
        frequency = _frequency_hz(frequency_hz)
        matches = np.flatnonzero(data.frequencies_hz == frequency)
        if len(matches) != 1:
            raise ValueError(
                "frequency_hz must exactly match one stored frequency; "
                f"got {frequency:g} Hz"
            )
        frequency_index = int(matches[0])
        selected_frequency = frequency
        if direction_weights is None:
            from gragra._plotting._dynamic_contribution import (
                contribution_statistics,
            )

            statistics = contribution_statistics(reference)
        else:
            from gragra._plotting._dynamic_contribution import (
                directional_contribution_statistics,
            )

            statistics = directional_contribution_statistics(
                results, direction_weights=direction_weights
            )
            directional_statistics = statistics
    else:
        if len(results) != 1:
            raise ValueError(
                "multiple results are supported only for rms_amplitude_fraction "
                "and interference_share"
            )
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

    if directional_statistics is not None:
        assert direction_weights is not None
        from gragra._plotting._dynamic_contribution import _direction_weights

        normalized_direction_weights = tuple(
            float(value)
            for value in _direction_weights(direction_weights, len(results))
        )

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
        "metric_definition": (
            "ratio of the direction-weighted mean group-total cross term to the "
            "direction-weighted mean total response power"
            if quantity == "interference_share" and directional_statistics is not None
            else _dynamic_metric_definition(quantity)
        ),
        "target_id": target_id,
        "target_xyz_m": tuple(float(value) for value in data.targets_m[target_index]),
        "component": component,
        "frequency_hz": selected_frequency,
        "frequencies_hz": tuple(float(value) for value in data.frequencies_hz),
        "frequency_weights": weight_tuple,
        "direction_weights": direction_tuple,
        "normalized_direction_weights": normalized_direction_weights,
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
