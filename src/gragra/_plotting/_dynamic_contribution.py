"""Dynamic source-resolved gravity contributions and statistics.

The numerical calculation uses the bounded point-observable source-term path.
Per-source values remain behind :class:`SourceContributionData`'s provider;
only coherent group sums and totals are materialized during construction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

from gragra.fields.density_series import DensityPhasors

if TYPE_CHECKING:
    from gragra._plotting._source_contracts import SourceContributionResult

_COMPONENT_LABELS = {
    "potential": ("potential",),
    "acceleration": ("x", "y", "z"),
    "gravity_gradient": (
        "xx",
        "xy",
        "xz",
        "yx",
        "yy",
        "yz",
        "zx",
        "zy",
        "zz",
    ),
}
_UNITS = {
    "potential": "m**2 / s**2",
    "acceleration": "m / s**2",
    "gravity_gradient": "1 / s**2",
}
_LAYOUTS = ("sample_first", "target_first")


def _readonly_array(value: Any, *, dtype: Any) -> np.ndarray:
    copied = np.array(value, dtype=dtype, copy=True, order="C")
    return np.frombuffer(copied.tobytes(order="C"), dtype=copied.dtype).reshape(
        copied.shape
    )


def _canonical_observable(observable: str) -> str:
    if not isinstance(observable, str):
        raise TypeError("observable must be a string")
    aliases = {
        "potential": "potential",
        "acceleration": "acceleration",
        "gravity_gradient": "gravity_gradient",
        "GG": "gravity_gradient",
        "gg": "gravity_gradient",
        "gradient": "gravity_gradient",
    }
    try:
        return aliases[observable]
    except KeyError as error:
        raise ValueError(
            "observable must be 'potential', 'acceleration', or "
            "'gravity_gradient' (aliases: 'GG', 'gg', 'gradient')"
        ) from error


def _validate_softening(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("softening_m must be finite and non-negative")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("softening_m must be finite and non-negative") from error
    if not np.isfinite(result) or result < 0.0:
        raise ValueError("softening_m must be finite and non-negative")
    return result


def _validate_chunk_size(value: Any, source_count: int) -> int:
    if value is None:
        return min(source_count, 256)
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value <= 0
    ):
        raise ValueError("chunk_size must be a positive integer or None")
    return int(value)


def _normalize_group_labels(
    labels: Any, source_count: int, target_count: int
) -> tuple[np.ndarray, tuple[int, ...], tuple[tuple[int, int], ...]]:
    try:
        raw = np.asarray(labels)
    except (TypeError, ValueError) as error:
        raise TypeError("group_labels must be an integer array") from error
    if raw.dtype.kind not in "iu":
        raise TypeError("group_labels must be an integer array")
    if raw.shape != (source_count,):
        raise ValueError(f"group_labels must have shape ({source_count},)")
    if raw.dtype.kind == "u" and raw.size and raw.max() > np.iinfo(np.int64).max:
        raise ValueError("group_labels must fit in int64")
    values = np.asarray(raw, dtype=np.int64)
    if (values < 0).any():
        raise ValueError("group_labels must be non-negative integers")
    labels_sorted = tuple(int(item) for item in np.unique(values))
    mapping = {label: index for index, label in enumerate(labels_sorted)}
    dense = np.fromiter((mapping[int(label)] for label in values), dtype=np.int64)
    membership = np.broadcast_to(dense, (target_count, source_count)).copy()
    label_mapping = tuple((label, index) for index, label in enumerate(labels_sorted))
    return membership, labels_sorted, label_mapping


def _accumulate_group_block(
    grouped_values: np.ndarray, group_indices: np.ndarray, source_terms: np.ndarray
) -> None:
    """Scatter-add one target's source block into its groups in source order."""
    np.add.at(grouped_values, group_indices, source_terms)


def _frequency_weights(value: Any, count: int) -> np.ndarray:
    from astropy import units as u

    if isinstance(value, u.Quantity):
        if not value.unit.is_equivalent(u.dimensionless_unscaled):
            raise ValueError("frequency_weights must be dimensionless")
        raw = np.asarray(value.to_value(u.dimensionless_unscaled))
    else:
        try:
            raw = np.asarray(value)
        except (TypeError, ValueError) as error:
            raise TypeError(
                "frequency_weights must be a real numeric vector"
            ) from error
    if raw.dtype.kind == "b" or raw.dtype.kind not in "iuf":
        raise TypeError("frequency_weights must be a real numeric vector")
    weights = np.asarray(raw, dtype=np.float64)
    if weights.shape != (count,):
        raise ValueError(f"frequency_weights must have shape ({count},)")
    if not np.isfinite(weights).all() or (weights < 0.0).any():
        raise ValueError("frequency_weights must be finite and non-negative")
    if not np.any(weights > 0.0):
        raise ValueError("frequency_weights must contain at least one positive value")
    return _readonly_array(weights, dtype=np.float64)


def _direction_weights(value: Any, count: int) -> np.ndarray:
    from astropy import units as u

    if isinstance(value, u.Quantity):
        if not value.unit.is_equivalent(u.dimensionless_unscaled):
            raise ValueError("direction_weights must be dimensionless")
        raw = np.asarray(value.to_value(u.dimensionless_unscaled))
    else:
        try:
            raw = np.asarray(value)
        except (TypeError, ValueError) as error:
            raise TypeError(
                "direction_weights must be a real numeric vector"
            ) from error
    if raw.dtype.kind == "b" or raw.dtype.kind not in "iuf":
        raise TypeError("direction_weights must be a real numeric vector")
    weights = np.asarray(raw, dtype=np.float64)
    if weights.shape != (count,):
        raise ValueError(f"direction_weights must have shape ({count},)")
    if not np.isfinite(weights).all() or (weights < 0.0).any():
        raise ValueError("direction_weights must be finite and non-negative")
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("direction_weights must have a finite positive sum")
    return _readonly_array(weights / total, dtype=np.float64)


def _flat_groups(result) -> np.ndarray:
    data = result._data
    components = int(np.prod(data.component_shape, dtype=np.int64)) or 1
    return np.asarray(data.group_values).reshape(
        data.frequency_count, data.target_count, data.group_count, components
    )


def _flat_total(result) -> np.ndarray:
    data = result._data
    components = int(np.prod(data.component_shape, dtype=np.int64)) or 1
    return np.asarray(data.total_values).reshape(
        data.frequency_count, data.target_count, components
    )


def _frequency_matrix(result, values: np.ndarray, unit: Any, name: str, selection: str):
    from astropy import units as u
    from gwexpy.frequencyseries import FrequencySeriesMatrix

    data = result._data
    components = result.component_labels
    matrix_values = np.asarray(values, dtype=np.complex128).transpose(1, 2, 0).copy()
    units = np.full((data.target_count, len(components)), u.Unit(unit), dtype=object)
    names = np.asarray(
        [
            [f"{target}:{component}" for component in components]
            for target in data.target_ids
        ],
        dtype=object,
    )
    channels = np.full(
        (data.target_count, len(components)), f"{name}[{selection}]", dtype=object
    )
    return FrequencySeriesMatrix(
        matrix_values,
        frequencies=np.array(data.frequencies_hz, copy=True) * u.Hz,
        units=units,
        names=names,
        channels=channels,
        rows=list(data.target_ids),
        cols=list(components),
        name=f"{name} {selection}",
    )


class ContributionValidityMask:
    """Labelled frequency-domain validity mask with explicit array layout."""

    __slots__ = ("_values", "target_ids", "component_labels", "frequency_hz")

    def __init__(self, values, target_ids, component_labels, frequency_hz):
        values = _readonly_array(values, dtype=np.bool_)
        expected = (len(frequency_hz), len(target_ids), len(component_labels))
        if values.shape != expected:
            raise ValueError(f"validity mask must have shape {expected}")
        self._values = values
        self.target_ids = tuple(target_ids)
        self.component_labels = tuple(component_labels)
        self.frequency_hz = _readonly_array(frequency_hz, dtype=np.float64)

    def to_array(self, *, layout: str) -> np.ndarray:
        if layout not in _LAYOUTS:
            raise ValueError(f"layout must be one of {_LAYOUTS}")
        values = (
            self._values
            if layout == "sample_first"
            else np.moveaxis(self._values, 0, 1)
        )
        return _readonly_array(values, dtype=np.bool_)


class _AggregateValidityMask:
    """Target/component validity labels for frequency-integrated quantities."""

    __slots__ = ("_values", "target_ids", "component_labels")

    def __init__(self, values, target_ids, component_labels):
        values = _readonly_array(values, dtype=np.bool_)
        expected = (len(target_ids), len(component_labels))
        if values.shape != expected:
            raise ValueError(f"aggregate validity mask must have shape {expected}")
        self._values = values
        self.target_ids = tuple(target_ids)
        self.component_labels = tuple(component_labels)

    def to_array(self) -> np.ndarray:
        return _readonly_array(self._values, dtype=np.bool_)


class ContributionFrequencyStatistic:
    """Native frequency matrices and validity masks, one matrix per group."""

    __slots__ = (
        "group_ids",
        "component_labels",
        "target_ids",
        "frequency_hz",
        "unit",
        "name",
        "_series",
        "_validity",
        "_values",
    )

    def __init__(self, result, values, valid, *, unit: Any, name: str):
        from astropy import units as u

        data = result._data
        self.group_ids = tuple(data.group_ids)
        self.component_labels = tuple(result.component_labels)
        self.target_ids = tuple(data.target_ids)
        self.frequency_hz = _readonly_array(data.frequencies_hz, dtype=np.float64)
        self.unit = u.Unit(unit)
        self.name = name
        series = {}
        validity = {}
        saved_values = {}
        for group_index, group_id in enumerate(self.group_ids):
            block = _readonly_array(values[:, :, group_index, :], dtype=np.complex128)
            mask = _readonly_array(valid[:, :, group_index, :], dtype=np.bool_)
            series[group_id] = _frequency_matrix(
                result, block, self.unit, name, repr(group_id)
            )
            validity[group_id] = ContributionValidityMask(
                mask, self.target_ids, self.component_labels, self.frequency_hz
            )
            saved_values[group_id] = block
        self._series = MappingProxyType(series)
        self._validity = MappingProxyType(validity)
        self._values = MappingProxyType(saved_values)

    def series(self, group_id: Any):
        try:
            return self._series[group_id]
        except (KeyError, TypeError) as error:
            raise KeyError(f"unknown group ID {group_id!r}") from error

    def validity(self, group_id: Any) -> ContributionValidityMask:
        try:
            return self._validity[group_id]
        except (KeyError, TypeError) as error:
            raise KeyError(f"unknown group ID {group_id!r}") from error

    def to_array(self, *, group_id: Any, layout: str) -> np.ndarray:
        if layout not in _LAYOUTS:
            raise ValueError(f"layout must be one of {_LAYOUTS}")
        try:
            values = self._values[group_id]
        except (KeyError, TypeError) as error:
            raise KeyError(f"unknown group ID {group_id!r}") from error
        if layout == "target_first":
            values = np.moveaxis(values, 0, 1)
        return _readonly_array(values, dtype=np.complex128)


@dataclass(frozen=True)
class ContributionRMSStatistics:
    """Labelled weighted spectral norms and same-frequency cross quantities.

    ``group_rms`` stores ``sqrt(0.5 * sum_f(w_f * |a_j(f)|**2))`` and
    ``total_rms`` applies the same norm after coherent group summation. The
    weights are used as supplied, with no frequency-spacing factor or
    normalization. This is a peak-phasor spectral quadratic norm; the
    long-time RMS interpretation assumes distinct positive-frequency tones
    are orthogonal. It is not a general finite-window time RMS or PSD integral.
    ``rms_share`` divides by the sum of group RMS values, while ``total_ratio``
    divides by the RMS of the coherent total. ``group_total_cross`` is
    ``0.5 * sum_f(w_f * Re(a_j * conj(a_total)))``.
    """

    group_ids: tuple[Any, ...]
    target_ids: tuple[str, ...]
    component_labels: tuple[str, ...]
    unit: Any
    group_rms: Mapping[Any, Any]
    total_rms: Any
    rms_share: Mapping[Any, Any]
    total_ratio: Mapping[Any, Any]
    group_total_cross: Mapping[Any, Any]
    total_power: Any
    rms_share_valid: Mapping[Any, _AggregateValidityMask]
    total_ratio_valid: Mapping[Any, _AggregateValidityMask]


@dataclass(frozen=True)
class ContributionStatistics:
    """Frequency-resolved component statistics and optional weighted RMS.

    Amplitude shares use ``abs(a_j) / sum_k(abs(a_k))``. Phase uses
    ``arg(a_j)`` under the input phasor convention. Interference shares use
    ``Re(a_j * conj(a_total)) / abs(a_total)**2`` and may be negative or
    greater than one. Undefined phase and ratios use zero placeholders with
    independent validity masks.
    """

    amplitude_share: ContributionFrequencyStatistic
    phase_rad: ContributionFrequencyStatistic
    cumulative_complex: ContributionFrequencyStatistic
    interference_share: ContributionFrequencyStatistic
    rms: ContributionRMSStatistics | None


@dataclass(frozen=True)
class DirectionalContributionStatistics:
    """Direction-ensemble cross contributions and optional weighted RMS.

    ``directional_cross`` is the direction-weighted group/total cross term,
    ``directional_power`` is the corresponding total power, and
    ``interference_share`` divides those ensemble means. Ratios are therefore
    formed after the direction average, not averaged from per-direction
    ratios. Directional RMS quantities use averaged power, not a complex sum
    of responses from distinct directions.
    """

    directional_cross: ContributionFrequencyStatistic
    directional_power: Any
    interference_share: ContributionFrequencyStatistic
    rms: ContributionRMSStatistics | None
    direction_metadata: tuple[Mapping[str, Any] | None, ...]
    direction_source_metadata: tuple[tuple[Mapping[str, Any] | None, ...] | None, ...]

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Return immutable provenance in the same order as the directions."""
        return MappingProxyType(
            {
                "direction_metadata": self.direction_metadata,
                "direction_source_metadata": self.direction_source_metadata,
            }
        )


def _quantity(values: np.ndarray, unit: Any):
    from astropy import units as u

    immutable = _readonly_array(values, dtype=np.float64)
    return u.Quantity(immutable, unit=u.Unit(unit), copy=False)


def _positive_product(
    *factors: np.ndarray, divide_sqrt_two: bool = False
) -> np.ndarray:
    """Multiply non-negative float factors without avoidable range loss."""
    broadcast = np.broadcast_arrays(
        *(np.asarray(value, dtype=np.float64) for value in factors)
    )
    mantissas = []
    exponents = []
    for value in broadcast:
        mantissa, exponent = np.frexp(value)
        mantissas.append(mantissa)
        exponents.append(exponent)
    product = np.ones_like(mantissas[0])
    for mantissa in mantissas:
        product *= mantissa
    if divide_sqrt_two:
        product /= np.sqrt(2.0)
    normalized_mantissa, extra_exponent = np.frexp(product)
    exponent_sum = np.add.reduce(np.stack(exponents, axis=0), axis=0) + extra_exponent
    with np.errstate(over="ignore", under="ignore"):
        return np.ldexp(normalized_mantissa, exponent_sum)


def _signed_product_quotient(
    numerator_factors: tuple[np.ndarray, ...],
    denominator_factors: tuple[np.ndarray, ...],
) -> np.ndarray:
    """Evaluate signed products with an extended intermediate exponent range.

    The Linux numerical environment uses an extended-range ``longdouble``.
    Keeping values extended until cross terms are summed allows large opposite
    terms to cancel before the public float64 result is formed.
    """
    broadcast = np.broadcast_arrays(
        *(
            np.asarray(value, dtype=np.float64)
            for value in (*numerator_factors, *denominator_factors)
        )
    )
    numerator_values = broadcast[: len(numerator_factors)]
    denominator_values = broadcast[len(numerator_factors) :]
    numerator_mantissas = []
    numerator_exponents = []
    sign = np.ones_like(broadcast[0])
    for value in numerator_values:
        sign *= np.sign(value)
        mantissa, exponent = np.frexp(np.abs(value))
        numerator_mantissas.append(mantissa)
        numerator_exponents.append(exponent)
    denominator_mantissas = []
    denominator_exponents = []
    for value in denominator_values:
        mantissa, exponent = np.frexp(value)
        denominator_mantissas.append(mantissa)
        denominator_exponents.append(exponent)
    quotient_mantissa = np.ones_like(sign)
    for mantissa in numerator_mantissas:
        quotient_mantissa *= mantissa
    for mantissa in denominator_mantissas:
        quotient_mantissa /= mantissa
    exponent = np.add.reduce(np.stack(numerator_exponents, axis=0), axis=0)
    exponent -= np.add.reduce(np.stack(denominator_exponents, axis=0), axis=0)
    normalized_mantissa, extra_exponent = np.frexp(quotient_mantissa)
    exponent += extra_exponent
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        magnitude = np.ldexp(normalized_mantissa.astype(np.longdouble), exponent)
    return np.copysign(magnitude, sign)


def _weighted_response_norm(
    values: np.ndarray,
    frequency_weights: np.ndarray,
    direction_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Compute ``sqrt(0.5 * sum(w * |value|**2))`` without squaring values."""
    magnitudes = np.abs(values)
    frequency_shape = (frequency_weights.size,) + (1,) * (magnitudes.ndim - 1)
    frequency_roots = np.sqrt(frequency_weights).reshape(frequency_shape)
    if direction_weights is None:
        terms = _positive_product(magnitudes, frequency_roots, divide_sqrt_two=True)
        return np.hypot.reduce(terms, axis=0)

    if magnitudes.ndim < 3:
        raise ValueError(
            "directional responses must include frequency and direction axes"
        )
    direction_shape = (1, 1, direction_weights.size) + (1,) * (magnitudes.ndim - 3)
    direction_roots = np.sqrt(direction_weights).reshape(direction_shape)
    terms = _positive_product(
        magnitudes, frequency_roots, direction_roots, divide_sqrt_two=True
    )
    direction_first = np.moveaxis(terms, 2, 1)
    flattened = direction_first.reshape(
        (frequency_weights.size * direction_weights.size,) + direction_first.shape[2:]
    )
    return np.hypot.reduce(flattened, axis=0)


def _weighted_response_mask(
    values: np.ndarray,
    frequency_weights: np.ndarray,
    direction_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Mark response entries supported by strictly positive supplied weights."""
    if direction_weights is None:
        return (frequency_weights > 0.0).reshape(
            (frequency_weights.size,) + (1,) * (values.ndim - 1)
        ) & (values != 0.0)
    frequency_support = (frequency_weights > 0.0).reshape(
        (frequency_weights.size,) + (1,) * (values.ndim - 1)
    )
    direction_shape = (1, 1, direction_weights.size) + (1,) * (values.ndim - 3)
    direction_support = (direction_weights > 0.0).reshape(direction_shape)
    return frequency_support & direction_support & (values != 0.0)


def _weighted_response_support(
    values: np.ndarray,
    frequency_weights: np.ndarray,
    direction_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Return exact nonzero support over strictly positive supplied weights."""
    supported = _weighted_response_mask(values, frequency_weights, direction_weights)
    return np.any(supported, axis=(0, 2) if direction_weights is not None else 0)


def _divide_complex_by_real(values: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Divide complex values componentwise to support subnormal real scales."""
    scale = np.asarray(scale, dtype=np.float64)
    real = np.divide(
        values.real,
        scale,
        out=np.zeros_like(values.real),
        where=scale > 0.0,
    )
    imag = np.divide(
        values.imag,
        scale,
        out=np.zeros_like(values.imag),
        where=scale > 0.0,
    )
    return real + 1j * imag


def _supported_component_scale(
    values: np.ndarray,
    frequency_weights: np.ndarray,
    direction_weights: np.ndarray | None,
    *,
    group_axis: bool,
) -> np.ndarray:
    supported = _weighted_response_mask(values, frequency_weights, direction_weights)
    component_magnitudes = np.maximum(np.abs(values.real), np.abs(values.imag))
    supported_magnitudes = np.where(supported, component_magnitudes, 0.0)
    if direction_weights is None:
        axes = (0, 2) if group_axis else (0,)
    else:
        axes = (0, 2, 3) if group_axis else (0, 2)
    return np.max(supported_magnitudes, axis=axes)


def _response_scale_shape(
    scale: np.ndarray, *, direction_weights: np.ndarray | None, group_axis: bool
) -> tuple[int, ...]:
    if direction_weights is None:
        return (
            (1, scale.shape[0], 1, scale.shape[1]) if group_axis else (1, *scale.shape)
        )
    return (
        (1, scale.shape[0], 1, 1, scale.shape[1])
        if group_axis
        else (1, scale.shape[0], 1, scale.shape[1])
    )


def _rms_statistics(
    result,
    groups: np.ndarray,
    total: np.ndarray,
    weights: np.ndarray,
    *,
    direction_power: np.ndarray | None = None,
    direction_responses: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> ContributionRMSStatistics:
    from astropy import units as u

    if direction_responses is None:
        group_rms_values = _weighted_response_norm(groups, weights)
        total_rms_values = _weighted_response_norm(total, weights)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            total_mean_power = 0.5 * np.sum(
                weights[:, None, None] * np.abs(total) ** 2, axis=0
            )
        group_cross = 0.5 * np.sum(
            weights[:, None, None, None]
            * np.real(groups * np.conj(total[:, :, None, :])),
            axis=0,
        )
        group_support = _weighted_response_support(groups, weights)
        total_support = _weighted_response_support(total, weights)
        ratio_groups = groups
        ratio_total = total
        ratio_direction_weights = None
    else:
        if direction_power is None:
            raise ValueError("directional RMS requires direction power values")
        group_responses, total_responses, direction_weights = direction_responses
        group_rms_values = _weighted_response_norm(
            group_responses, weights, direction_weights
        )
        total_rms_values = _weighted_response_norm(
            total_responses, weights, direction_weights
        )
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            total_mean_power = 0.5 * np.sum(
                weights[:, None, None] * direction_power, axis=0
            )
        group_cross = 0.5 * np.sum(
            weights[:, None, None, None] * np.real(groups),
            axis=0,
        )

        group_support = _weighted_response_support(
            group_responses, weights, direction_weights
        )
        total_support = _weighted_response_support(
            total_responses, weights, direction_weights
        )
        ratio_groups = group_responses
        ratio_total = total_responses
        ratio_direction_weights = direction_weights

    group_share_scale = _supported_component_scale(
        ratio_groups,
        weights,
        ratio_direction_weights,
        group_axis=True,
    )
    group_total_scale = _supported_component_scale(
        ratio_total,
        weights,
        ratio_direction_weights,
        group_axis=False,
    )
    common_scale = np.maximum(group_share_scale, group_total_scale)
    group_share_scale = group_share_scale.reshape(
        _response_scale_shape(
            group_share_scale,
            direction_weights=ratio_direction_weights,
            group_axis=True,
        )
    )
    common_group_scale = common_scale.reshape(
        _response_scale_shape(
            common_scale,
            direction_weights=ratio_direction_weights,
            group_axis=True,
        )
    )
    common_total_scale = common_scale.reshape(
        _response_scale_shape(
            common_scale,
            direction_weights=ratio_direction_weights,
            group_axis=False,
        )
    )
    scaled_group_share = _divide_complex_by_real(
        np.where(
            _weighted_response_mask(ratio_groups, weights, ratio_direction_weights),
            ratio_groups,
            0.0,
        ),
        group_share_scale,
    )
    scaled_group_total_ratio = _divide_complex_by_real(
        np.where(
            _weighted_response_mask(ratio_groups, weights, ratio_direction_weights),
            ratio_groups,
            0.0,
        ),
        common_group_scale,
    )
    scaled_total_ratio = _divide_complex_by_real(
        np.where(
            _weighted_response_mask(ratio_total, weights, ratio_direction_weights),
            ratio_total,
            0.0,
        ),
        common_total_scale,
    )
    normalized_share_norms = _weighted_response_norm(
        scaled_group_share, weights, ratio_direction_weights
    )
    normalized_group_norms = _weighted_response_norm(
        scaled_group_total_ratio, weights, ratio_direction_weights
    )
    normalized_total_norm = _weighted_response_norm(
        scaled_total_ratio, weights, ratio_direction_weights
    )
    group_denominator = np.sum(normalized_share_norms, axis=1)
    share_valid_base = np.any(group_support, axis=1)
    total_valid_base = total_support
    share_values = np.divide(
        normalized_share_norms,
        group_denominator[:, np.newaxis, :],
        out=np.zeros_like(normalized_share_norms),
        where=group_denominator[:, np.newaxis, :] > 0.0,
    )
    total_ratio_values = np.divide(
        normalized_group_norms,
        normalized_total_norm[:, np.newaxis, :],
        out=np.zeros_like(normalized_group_norms),
        where=(normalized_total_norm[:, np.newaxis, :] > 0.0),
    )
    if not all(
        np.isfinite(values).all()
        for values in (
            group_rms_values,
            total_rms_values,
            group_cross,
            share_values,
            total_ratio_values,
            total_mean_power,
        )
    ):
        raise ValueError("weighted contribution statistics are not finite")

    data = result._data
    unit = u.Unit(data.unit)
    group_rms = {}
    shares = {}
    total_ratios = {}
    crosses = {}
    share_masks = {}
    ratio_masks = {}
    for index, group_id in enumerate(data.group_ids):
        group_rms[group_id] = _quantity(group_rms_values[:, index, :], unit)
        shares[group_id] = _quantity(
            share_values[:, index, :], u.dimensionless_unscaled
        )
        total_ratios[group_id] = _quantity(
            total_ratio_values[:, index, :], u.dimensionless_unscaled
        )
        crosses[group_id] = _quantity(group_cross[:, index, :], unit**2)
        share_masks[group_id] = _AggregateValidityMask(
            share_valid_base, data.target_ids, result.component_labels
        )
        ratio_masks[group_id] = _AggregateValidityMask(
            total_valid_base, data.target_ids, result.component_labels
        )
    return ContributionRMSStatistics(
        group_ids=tuple(data.group_ids),
        target_ids=tuple(data.target_ids),
        component_labels=tuple(result.component_labels),
        unit=unit,
        group_rms=MappingProxyType(group_rms),
        total_rms=_quantity(total_rms_values, unit),
        rms_share=MappingProxyType(shares),
        total_ratio=MappingProxyType(total_ratios),
        group_total_cross=MappingProxyType(crosses),
        total_power=_quantity(total_mean_power, unit**2),
        rms_share_valid=MappingProxyType(share_masks),
        total_ratio_valid=MappingProxyType(ratio_masks),
    )


def source_contributions(
    phasors: DensityPhasors,
    observable: str,
    targets_m: object,
    *,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
    group_labels: object | None = None,
    radial_edges_m: object | None = None,
) -> SourceContributionResult:
    """Compute complex per-source gravity contributions from density phasors.

    Targets are interpreted as SI coordinates in the phasors' own frame and
    origin. The returned source values are computed on demand from immutable
    snapshots. Physical source weights are ``density_kg_m3 * volumes_m3``;
    group sums and totals are accumulated source-block by source-block. No
    automatic frame transform or phase adjustment is applied.
    """
    if not isinstance(phasors, DensityPhasors):
        raise TypeError("phasors must be a DensityPhasors instance")
    canonical = _canonical_observable(observable)
    eps = _validate_softening(softening_m)
    n_sources = len(phasors.positions_m)
    effective_chunk = _validate_chunk_size(chunk_size, n_sources)

    # Reuse the common numeric target and radial-bin contracts at this boundary.
    from gragra._plotting._source_quantities import (
        normalize_targets_m,
        source_radial_membership,
    )

    targets = normalize_targets_m(targets_m)
    n_targets = len(targets)
    if group_labels is not None and radial_edges_m is not None:
        raise ValueError("group_labels and radial_edges_m are mutually exclusive")
    if group_labels is not None:
        membership, group_ids, label_mapping = _normalize_group_labels(
            group_labels, n_sources, n_targets
        )
        saved_edges = None
    elif radial_edges_m is not None:
        distances = np.linalg.norm(
            phasors.positions_m[np.newaxis, :, :] - targets[:, np.newaxis, :],
            axis=-1,
        )
        membership, group_ids = source_radial_membership(distances, radial_edges_m)
        label_mapping = None
        saved_edges = _readonly_array(radial_edges_m, dtype=np.float64)
    else:
        membership = np.zeros((n_targets, n_sources), dtype=np.int64)
        group_ids = ("all",)
        label_mapping = None
        saved_edges = None
    membership = _readonly_array(membership, dtype=np.int64)

    positions = _readonly_array(phasors.positions_m, dtype=np.float64)
    target_snapshot = _readonly_array(targets, dtype=np.float64)
    frequency_snapshot = _readonly_array(phasors.frequency_hz, dtype=np.float64)
    mass_phasors = _readonly_array(
        phasors.density_kg_m3 * phasors.volumes_m3[np.newaxis, :],
        dtype=np.complex128,
    )
    component_shape = {
        "potential": (),
        "acceleration": (3,),
        "gravity_gradient": (3, 3),
    }[canonical]
    group_values = np.zeros(
        (len(frequency_snapshot), n_targets, len(group_ids), *component_shape),
        dtype=np.complex128,
    )
    total_values = np.zeros(
        (len(frequency_snapshot), n_targets, *component_shape), dtype=np.complex128
    )

    from gragra.observables._source_terms import iter_point_observable_source_terms

    # This pass uses the existing iterator for global collision indices and only
    # holds one frequency/source block of pairwise values at any time.
    for frequency_index, weights in enumerate(mass_phasors):
        for source_slice, terms in iter_point_observable_source_terms(
            positions,
            weights,
            target_snapshot,
            observable=canonical,
            softening_m=eps,
            chunk_size=effective_chunk,
        ):
            source_start, source_stop = source_slice.start, source_slice.stop
            total_values[frequency_index] += np.sum(terms, axis=1)
            block_membership = membership[:, source_start:source_stop]
            for target_index in range(n_targets):
                _accumulate_group_block(
                    group_values[frequency_index, target_index],
                    block_membership[target_index],
                    terms[target_index],
                )

    if not np.isfinite(group_values).all() or not np.isfinite(total_values).all():
        raise ValueError("source contributions must be finite")

    def source_block_provider(start: int, stop: int) -> np.ndarray:
        from gragra.observables._source_terms import point_observable_source_terms

        if start < 0 or stop > n_sources or start >= stop:
            raise IndexError("source block bounds are out of range")
        values = np.empty(
            (len(frequency_snapshot), n_targets, stop - start, *component_shape),
            dtype=np.complex128,
        )
        for frequency_index, weights in enumerate(mass_phasors):
            values[frequency_index] = point_observable_source_terms(
                positions[start:stop],
                weights[start:stop],
                target_snapshot,
                observable=canonical,
                softening_m=eps,
                chunk_size=min(effective_chunk, stop - start),
            )
        return values

    from gragra._plotting._source_contracts import SourceContributionResult
    from gragra._plotting._source_data import SourceContributionData

    data = SourceContributionData(
        source_values=None,
        group_values=group_values,
        total_values=total_values,
        positions_m=positions,
        targets_m=target_snapshot,
        target_ids=tuple(f"target-{index}" for index in range(n_targets)),
        frequencies_hz=frequency_snapshot,
        unit=_UNITS[canonical],
        observable=canonical,
        frame=phasors.coordinate_frame,
        membership=membership,
        group_ids=group_ids,
        source_block_provider=source_block_provider,
        source_ids=phasors.source_ids,
        source_metadata=None,
        source_index=np.arange(n_sources, dtype=np.int64),
        group_label_mapping=label_mapping,
        radial_edges_m=saved_edges,
        coordinate_origin_m=phasors.coordinate_origin_m,
        softening_m=eps,
        t_ref_s=phasors.t_ref_s,
        phase_origin_m=phasors.phase_origin_m,
        fourier_convention=phasors.fourier_convention,
        peak_phasor=True,
        global_metadata=phasors.source_metadata,
    )
    return SourceContributionResult(data)


def contribution_statistics(
    result: SourceContributionResult,
    *,
    frequency_weights: object | None = None,
) -> ContributionStatistics:
    """Return componentwise phasor shares, phase, interference, and optional RMS.

    With no ``frequency_weights``, all frequency-integrated quantities are
    ``None``. Supplied weights are finite, non-negative, dimensionless, and
    used without normalization or an automatic frequency-spacing factor.
    """
    from astropy import units as u

    from gragra._plotting._source_contracts import SourceContributionResult

    if not isinstance(result, SourceContributionResult):
        raise TypeError("result must be a SourceContributionResult")
    groups = _flat_groups(result)
    total = _flat_total(result)
    if not np.isfinite(groups).all() or not np.isfinite(total).all():
        raise ValueError("contribution statistics require finite inputs")

    magnitudes = np.abs(groups)
    denominator = np.sum(magnitudes, axis=2)
    if not np.isfinite(denominator).all():
        raise ValueError("amplitude-share denominator is not finite")
    share_valid_base = denominator > 0.0
    share_values = np.divide(
        magnitudes,
        denominator[:, :, np.newaxis, :],
        out=np.zeros_like(magnitudes),
        where=share_valid_base[:, :, np.newaxis, :],
    )
    phase_valid = magnitudes > 0.0
    phase_values = np.where(phase_valid, np.angle(groups), 0.0)
    cumulative = np.cumsum(groups, axis=2)
    cumulative_valid = np.ones(cumulative.shape, dtype=np.bool_)

    interference_valid_base = total != 0.0
    total_scale = np.maximum(np.abs(total.real), np.abs(total.imag))
    safe_total_scale = np.where(interference_valid_base, total_scale, 1.0)
    total_scaled = _divide_complex_by_real(total, safe_total_scale)
    denominator_scaled = total_scaled.real**2 + total_scaled.imag**2
    group_total_scale = safe_total_scale[:, :, np.newaxis, :]
    numerator_scaled = _signed_product_quotient(
        (groups.real, total.real[:, :, np.newaxis, :]),
        (group_total_scale, group_total_scale),
    ) + _signed_product_quotient(
        (groups.imag, total.imag[:, :, np.newaxis, :]),
        (group_total_scale, group_total_scale),
    )
    interference_values = np.divide(
        numerator_scaled,
        denominator_scaled[:, :, np.newaxis, :],
        out=np.zeros_like(numerator_scaled),
        where=interference_valid_base[:, :, np.newaxis, :],
    )
    with np.errstate(over="ignore", invalid="ignore"):
        interference_values = np.asarray(interference_values, dtype=np.float64)
    share_valid = np.broadcast_to(share_valid_base[:, :, np.newaxis, :], groups.shape)
    interference_valid = np.broadcast_to(
        interference_valid_base[:, :, np.newaxis, :], groups.shape
    )
    if not all(
        np.isfinite(values).all()
        for values in (share_values, phase_values, cumulative, interference_values)
    ):
        raise ValueError("contribution statistics are not finite")

    data = result._data
    amplitude_share = ContributionFrequencyStatistic(
        result,
        share_values,
        share_valid,
        unit=u.dimensionless_unscaled,
        name="amplitude share",
    )
    phase = ContributionFrequencyStatistic(
        result,
        phase_values,
        phase_valid,
        unit=u.rad,
        name="phase",
    )
    cumulative_complex = ContributionFrequencyStatistic(
        result,
        cumulative,
        cumulative_valid,
        unit=u.Unit(data.unit),
        name="cumulative complex contribution",
    )
    interference_share = ContributionFrequencyStatistic(
        result,
        interference_values,
        interference_valid,
        unit=u.dimensionless_unscaled,
        name="interference share",
    )
    rms = None
    if frequency_weights is not None:
        weights = _frequency_weights(frequency_weights, data.frequency_count)
        rms = _rms_statistics(result, groups, total, weights)
    return ContributionStatistics(
        amplitude_share=amplitude_share,
        phase_rad=phase,
        cumulative_complex=cumulative_complex,
        interference_share=interference_share,
        rms=rms,
    )


def _metadata_equal(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return np.array_equal(np.asarray(left), np.asarray(right))
        except (TypeError, ValueError):
            return False
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(
            _metadata_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(
            _metadata_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def _validate_compatible_results(results: tuple[Any, ...]) -> None:
    from gragra._plotting._source_contracts import SourceContributionResult

    if not results:
        raise ValueError("results must contain at least one SourceContributionResult")
    if any(not isinstance(item, SourceContributionResult) for item in results):
        raise TypeError("every result must be a SourceContributionResult")
    reference = results[0]._data
    comparable_fields = (
        "observable",
        "unit",
        "frame",
        "softening_m",
        "t_ref_s",
        "fourier_convention",
        "peak_phasor",
        "group_ids",
        "group_label_mapping",
        "source_ids",
        "target_ids",
    )
    comparable_arrays = (
        "positions_m",
        "source_index",
        "targets_m",
        "frequencies_hz",
        "coordinate_origin_m",
        "phase_origin_m",
        "membership",
        "radial_edges_m",
    )
    for candidate_result in results[1:]:
        candidate = candidate_result._data
        if any(
            not _metadata_equal(getattr(reference, field), getattr(candidate, field))
            for field in comparable_fields
        ):
            raise ValueError("results are not compatible for directional statistics")
        if any(
            not _metadata_equal(getattr(reference, field), getattr(candidate, field))
            for field in comparable_arrays
        ):
            raise ValueError("results are not compatible for directional statistics")


def directional_contribution_statistics(
    results: Sequence[SourceContributionResult],
    *,
    direction_weights: object,
    frequency_weights: object | None = None,
) -> DirectionalContributionStatistics:
    """Compute second-order contribution statistics for uncorrelated directions.

    Direction weights are normalized to sum to one. Complex source responses
    are crossed before division, so this reports a direction ensemble rather
    than the coherent field obtained by summing different directions.
    """
    from astropy import units as u

    from gragra.coherent.average import directional_cross_average

    if isinstance(results, (str, bytes)):
        raise TypeError("results must be a nonempty sequence of contribution results")
    try:
        result_tuple = tuple(results)
    except TypeError as error:
        raise TypeError(
            "results must be a nonempty sequence of contribution results"
        ) from error
    _validate_compatible_results(result_tuple)
    weights_d = _direction_weights(direction_weights, len(result_tuple))
    reference = result_tuple[0]
    data = reference._data

    groups_by_direction = np.stack(
        [_flat_groups(item) for item in result_tuple], axis=2
    )
    totals_by_direction = np.stack([_flat_total(item) for item in result_tuple], axis=2)
    group_cross_complex = directional_cross_average(
        groups_by_direction,
        np.broadcast_to(
            totals_by_direction[:, :, :, np.newaxis, :],
            groups_by_direction.shape,
        ),
        weights_d,
        direction_axis=2,
    )
    power_by_frequency = np.real(
        directional_cross_average(
            totals_by_direction,
            totals_by_direction,
            weights_d,
            direction_axis=2,
        )
    )
    group_cross_by_frequency = np.real(group_cross_complex)
    if (
        not np.isfinite(group_cross_by_frequency).all()
        or not np.isfinite(power_by_frequency).all()
    ):
        raise ValueError("directional contribution statistics are not finite")
    unit_frequency_weights = np.ones(data.frequency_count, dtype=np.float64)
    total_support_by_direction = _weighted_response_mask(
        totals_by_direction, unit_frequency_weights, weights_d
    )
    power_valid = np.any(total_support_by_direction, axis=2)

    # Normalize supported response components before applying direction
    # weights. The common scale cancels from the ratio and avoids both complex
    # division by subnormal scales and loss from weighting a subnormal response.
    total_component_scale = np.maximum(
        np.abs(totals_by_direction.real), np.abs(totals_by_direction.imag)
    )
    total_scale = np.max(
        np.where(total_support_by_direction, total_component_scale, 0.0), axis=2
    )
    safe_total_scale = np.where(total_scale > 0.0, total_scale, 1.0)
    scaled_totals = _divide_complex_by_real(
        np.where(total_support_by_direction, totals_by_direction, 0.0),
        safe_total_scale[:, :, np.newaxis, :],
    )
    scaled_power = np.real(
        directional_cross_average(
            scaled_totals,
            scaled_totals,
            weights_d,
            direction_axis=2,
        )
    )
    direction_weight_shape = (1, 1, weights_d.size, 1, 1)
    total_scale_for_groups = safe_total_scale[:, :, np.newaxis, np.newaxis, :]
    numerator_scaled = np.sum(
        _signed_product_quotient(
            (
                groups_by_direction.real,
                totals_by_direction.real[:, :, :, np.newaxis, :],
                weights_d.reshape(direction_weight_shape),
            ),
            (total_scale_for_groups, total_scale_for_groups),
        )
        + _signed_product_quotient(
            (
                groups_by_direction.imag,
                totals_by_direction.imag[:, :, :, np.newaxis, :],
                weights_d.reshape(direction_weight_shape),
            ),
            (total_scale_for_groups, total_scale_for_groups),
        ),
        axis=2,
    )
    interference_values = np.divide(
        numerator_scaled,
        scaled_power[:, :, np.newaxis, :],
        out=np.zeros_like(numerator_scaled),
        where=scaled_power[:, :, np.newaxis, :] > 0.0,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        interference_values = np.asarray(interference_values, dtype=np.float64)
    if not np.isfinite(interference_values).all():
        raise ValueError("directional contribution statistics are not finite")
    interference_valid = np.broadcast_to(
        power_valid[:, :, np.newaxis, :], interference_values.shape
    )
    cross_metric = ContributionFrequencyStatistic(
        reference,
        group_cross_by_frequency,
        np.ones(group_cross_by_frequency.shape, dtype=np.bool_),
        unit=u.Unit(data.unit) ** 2,
        name="directional group-total cross",
    )
    interference_metric = ContributionFrequencyStatistic(
        reference,
        interference_values,
        interference_valid,
        unit=u.dimensionless_unscaled,
        name="directional interference share",
    )
    # The total directional power has no group axis. It is a labelled native
    # FrequencySeriesMatrix, constructed with the same target/component labels.
    directional_power = _frequency_matrix(
        reference,
        power_by_frequency.astype(np.complex128),
        u.Unit(data.unit) ** 2,
        "directional total power",
        "ensemble",
    )

    rms = None
    if frequency_weights is not None:
        weights_f = _frequency_weights(frequency_weights, data.frequency_count)
        # _rms_statistics expects cross values in its first argument only for
        # the directional path. These are averaged group-total cross spectra.
        rms = _rms_statistics(
            reference,
            group_cross_by_frequency,
            power_by_frequency,
            weights_f,
            direction_power=power_by_frequency,
            direction_responses=(groups_by_direction, totals_by_direction, weights_d),
        )
    return DirectionalContributionStatistics(
        directional_cross=cross_metric,
        directional_power=directional_power,
        interference_share=interference_metric,
        rms=rms,
        direction_metadata=tuple(item._data.global_metadata for item in result_tuple),
        direction_source_metadata=tuple(
            item._data.source_metadata for item in result_tuple
        ),
    )
