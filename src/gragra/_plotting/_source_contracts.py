"""Public accessors for source-contribution snapshots.

The numeric record lives in :mod:`gragra._plotting._source_data` and is safe to
import without plotting dependencies. Native GWexpy matrices are constructed
only when a caller requests a source, group, or total matrix.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Any

import numpy as np

from gragra._plotting._source_data import SourceContributionData

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
_ARRAY_KINDS = ("source", "group", "total")
_ARRAY_LAYOUTS = ("sample_first", "target_first")


class SourceContributionResult:
    """Metadata and native-matrix access to source-resolved phasors.

    Parameters
    ----------
    data
        A validated, immutable :class:`SourceContributionData` record.

    Notes
    -----
    Matrix rows identify targets, columns identify observable components, and
    the native frequency axis is in Hz. Array extraction is explicit because
    both frequency-first and target-first layouts are useful to downstream
    code and easy to confuse.
    """

    __slots__ = ("_data", "_unit")

    def __init__(self, data: SourceContributionData) -> None:
        if not isinstance(data, SourceContributionData):
            raise TypeError("data must be a SourceContributionData record")
        # Validate physical unit syntax at this high-level boundary without
        # importing GWexpy until a Matrix is requested.
        from astropy import units as u

        try:
            unit = u.Unit(data.unit)
        except (TypeError, ValueError) as error:
            raise ValueError(f"unit must be valid unit text: {error}") from error
        self._data = data
        self._unit = unit

    def __deepcopy__(self, memo: dict[int, Any]) -> SourceContributionResult:
        """Copy the wrapper while safely sharing its immutable numeric record."""
        copied = type(self).__new__(type(self))
        memo[id(self)] = copied
        copied._data = self._data
        copied._unit = self._unit
        return copied

    @property
    def observable(self) -> str:
        return self._data.observable

    @property
    def unit(self) -> str:
        return self._data.unit

    @property
    def source_ids(self) -> tuple[str, ...] | None:
        return self._data.source_ids

    @property
    def source_index(self) -> np.ndarray:
        return self._data.source_index

    @property
    def target_ids(self) -> tuple[str, ...]:
        return self._data.target_ids

    @property
    def group_ids(self) -> tuple[Any, ...]:
        return self._data.group_ids

    @property
    def global_metadata(self) -> Mapping[str, Any] | None:
        """Return shared input metadata as its immutable, single snapshot."""
        return self._data.global_metadata

    @property
    def component_labels(self) -> tuple[str, ...]:
        return _COMPONENT_LABELS[self.observable]

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Deeply immutable source, target, phase, and coordinate metadata."""
        data = self._data
        values = {
            "observable": data.observable,
            "unit": data.unit,
            "frame": data.frame,
            "coordinate_origin_m": tuple(
                float(item) for item in data.coordinate_origin_m
            ),
            "positions_m": tuple(
                tuple(float(x) for x in row) for row in data.positions_m
            ),
            "source_index": tuple(int(item) for item in data.source_index),
            "source_ids": data.source_ids,
            "source_metadata": data.source_metadata,
            "global_metadata": data.global_metadata,
            "targets_m": tuple(tuple(float(x) for x in row) for row in data.targets_m),
            "target_ids": data.target_ids,
            "frequencies_hz": tuple(float(item) for item in data.frequencies_hz),
            "group_ids": data.group_ids,
            "group_label_mapping": data.group_label_mapping,
            "radial_edges_m": (
                None
                if data.radial_edges_m is None
                else tuple(float(item) for item in data.radial_edges_m)
            ),
            "membership": tuple(tuple(int(x) for x in row) for row in data.membership),
            "softening_m": data.softening_m,
            "t_ref_s": data.t_ref_s,
            "phase_origin_m": (
                None
                if data.phase_origin_m is None
                else tuple(float(item) for item in data.phase_origin_m)
            ),
            "fourier_convention": data.fourier_convention,
            "peak_phasor": data.peak_phasor,
        }
        # Every member is immutable already: arrays and mappings come from the
        # data record's one-time deep snapshot, and sequences are tuples.
        return MappingProxyType(values)

    def iter_source_blocks(
        self, block_size: int = 256, *, frequency_index: int | None = None
    ) -> Iterator[tuple[int, int, np.ndarray]]:
        """Delegate lazy source iteration without materializing all sources."""
        return self._data.iter_source_blocks(
            block_size, frequency_index=frequency_index
        )

    def source(self, source_id: str | int | np.integer):
        """Return one source's native ``FrequencySeriesMatrix``.

        String selectors require source IDs that were present in the input.
        Integer selectors always mean the stable source index and are accepted
        whether or not source IDs were supplied.
        """
        index = self._source_index(source_id)
        values = self._data.source_block(index, index + 1)[:, :, 0, ...]
        label = (
            self._data.source_ids[index]
            if self._data.source_ids is not None
            else f"source_index={index}"
        )
        return self._matrix(values, selection=f"source[{label}]")

    def group(self, group_id: Any):
        """Return one grouped contribution's native frequency matrix."""
        if isinstance(group_id, (bool, np.bool_)):
            raise TypeError("group ID must be a string, integer, or immutable tuple")
        try:
            index = self._data.group_ids.index(group_id)
        except (ValueError, TypeError) as error:
            raise KeyError(f"unknown group ID {group_id!r}") from error
        values = self._data.group_values[:, :, index, ...]
        return self._matrix(values, selection=f"group[{group_id!r}]")

    @property
    def total(self):
        """Return the total contribution as a fresh native frequency matrix."""
        return self._matrix(self._data.total_values, selection="total")

    def to_array(self, *, kind: str, layout: str) -> np.ndarray:
        """Return a read-only array in a caller-selected axis layout.

        ``kind`` is one of ``source``, ``group``, or ``total``. The
        ``sample_first`` layout places frequency first; ``target_first`` moves
        target first and frequency after the optional item axis. Components,
        when present, remain the final axes. Totals have no source/group item
        axis.
        """
        if kind not in _ARRAY_KINDS:
            raise ValueError(f"kind must be one of {_ARRAY_KINDS}, got {kind!r}")
        if layout not in _ARRAY_LAYOUTS:
            raise ValueError(f"layout must be one of {_ARRAY_LAYOUTS}, got {layout!r}")
        data = self._data
        if kind == "source":
            shape = (
                data.frequency_count,
                data.target_count,
                data.source_count,
                *data.component_shape,
            )
            sample_first = np.empty(shape, dtype=np.complex128)
            for start, stop, block in data.iter_source_blocks():
                sample_first[:, :, start:stop, ...] = block
        elif kind == "group":
            sample_first = np.array(data.group_values, dtype=np.complex128, copy=True)
        else:
            sample_first = np.array(data.total_values, dtype=np.complex128, copy=True)

        if layout == "sample_first":
            selected = sample_first
        elif kind == "total":
            selected = np.moveaxis(sample_first, 0, 1)
        else:
            selected = np.moveaxis(sample_first, 0, 2)
        return _readonly_array(selected)

    def _source_index(self, selector: Any) -> int:
        if isinstance(selector, (bool, np.bool_)):
            raise TypeError("source selector must be an ID or integer source index")
        if isinstance(selector, (int, np.integer)):
            index = int(selector)
            if index < 0 or index >= self._data.source_count:
                raise IndexError("source index is out of range")
            return index
        if isinstance(selector, str):
            if self._data.source_ids is None:
                raise TypeError(
                    "source IDs were not supplied; select with an integer source index"
                )
            try:
                return self._data.source_ids.index(selector)
            except ValueError as error:
                raise KeyError(f"unknown source ID {selector!r}") from error
        raise TypeError("source selector must be an ID or integer source index")

    def _matrix(self, values: np.ndarray, *, selection: str):
        # High-level dependencies are deliberately loaded only on this path.
        from astropy import units as u
        from gwexpy.frequencyseries import FrequencySeriesMatrix

        data = self._data
        components = self.component_labels
        matrix_values = np.asarray(values, dtype=np.complex128).reshape(
            data.frequency_count, data.target_count, len(components)
        )
        matrix_values = np.array(matrix_values.transpose(1, 2, 0), copy=True)
        names = np.asarray(
            [
                [f"{target}:{component}" for component in components]
                for target in data.target_ids
            ],
            dtype=object,
        )
        channels = np.full(
            (data.target_count, len(components)), selection, dtype=object
        )
        units = np.full((data.target_count, len(components)), self._unit, dtype=object)
        return FrequencySeriesMatrix(
            matrix_values,
            frequencies=np.array(data.frequencies_hz, copy=True) * u.Hz,
            units=units,
            names=names,
            channels=channels,
            rows=list(data.target_ids),
            cols=list(components),
            name=f"{data.observable} contribution {selection}",
        )


def _readonly_array(value: Any) -> np.ndarray:
    copied = np.array(value, dtype=np.complex128, copy=True, order="C")
    return np.frombuffer(copied.tobytes(order="C"), dtype=copied.dtype).reshape(
        copied.shape
    )
