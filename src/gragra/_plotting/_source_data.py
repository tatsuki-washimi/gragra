"""NumPy-only records for source-resolved contribution results.

This module deliberately avoids Astropy, GWexpy, and plotting imports. It is
the neutral data contract shared by dynamic contribution and source-field
layers.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

_OBSERVABLES = {
    "potential": (),
    "acceleration": (3,),
    "gravity_gradient": (3, 3),
}
_OBSERVABLE_ALIASES = {
    "potential": "potential",
    "acceleration": "acceleration",
    "gravity_gradient": "gravity_gradient",
    "GG": "gravity_gradient",
    "gg": "gravity_gradient",
    "gradient": "gravity_gradient",
}


def _copy_readonly(value: Any, name: str, *, dtype: Any = None) -> np.ndarray:
    """Return an immutable, finite, bytes-backed copy of a numeric array."""
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a numeric array") from error
    if raw.dtype.kind == "b" or raw.dtype.kind not in "iufc":
        raise TypeError(f"{name} must be a numeric non-boolean array")
    if dtype is None:
        dtype = np.complex128 if np.iscomplexobj(raw) else np.float64
    if np.dtype(dtype).kind == "f" and raw.dtype.kind == "c":
        raise TypeError(f"{name} must contain real values")
    try:
        copied = np.array(raw, dtype=dtype, copy=True, order="C")
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{name} cannot be represented by {np.dtype(dtype)}") from error
    if not np.isfinite(copied).all():
        raise ValueError(f"{name} must contain only finite values")
    # A read-only flag on an owning ndarray can be turned back on by callers.
    # Use immutable bytes as the backing store to make the snapshot irreversible.
    return np.frombuffer(copied.tobytes(order="C"), dtype=copied.dtype).reshape(
        copied.shape
    )


def _copy_real(value: Any, name: str) -> np.ndarray:
    return _copy_readonly(value, name, dtype=np.float64)


def _copy_integer(value: Any, name: str) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be an integer array") from error
    if raw.dtype.kind not in "iu":
        raise TypeError(f"{name} must be an integer array")
    if raw.dtype.kind == "u" and raw.size and raw.max() > np.iinfo(np.int64).max:
        raise ValueError(f"{name} values must fit in int64")
    return _copy_readonly(raw, name, dtype=np.int64)


def _copy_metadata_array(value: np.ndarray) -> np.ndarray:
    """Snapshot metadata arrays without changing their dtype or values."""
    raw = np.asarray(value)
    if raw.dtype.hasobject:
        raise TypeError("metadata arrays must not contain Python objects")
    copied = np.array(raw, copy=True, order="C")
    return np.frombuffer(copied.tobytes(order="C"), dtype=copied.dtype).reshape(
        copied.shape
    )


def _freeze_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        frozen = {}
        for key, item in value.items():
            frozen_key = _freeze_metadata(key)
            try:
                hash(frozen_key)
            except TypeError as error:
                raise TypeError("metadata mapping keys must be immutable") from error
            frozen[frozen_key] = _freeze_metadata(item)
        return MappingProxyType(frozen)
    if isinstance(value, np.ndarray):
        return _copy_metadata_array(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata(item) for item in value)
    if isinstance(value, np.generic):
        return _freeze_metadata(value.item())
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError("metadata float values must be finite")
        return value
    raise TypeError(f"metadata value of type {type(value).__name__} is not supported")


def _freeze_identifier(value: Any, name: str) -> Any:
    frozen = _freeze_metadata(value)
    if frozen is None or isinstance(frozen, (bool, float)):
        raise TypeError(f"{name} must be a string, integer, or immutable tuple")
    if not isinstance(frozen, (str, int, tuple)):
        raise TypeError(f"{name} must be a string, integer, or immutable tuple")
    try:
        hash(frozen)
    except TypeError as error:
        raise TypeError(f"{name} must be immutable and hashable") from error
    return frozen


def canonical_observable(value: str) -> str:
    """Return a canonical source-contribution observable identifier."""
    if not isinstance(value, str):
        raise TypeError("observable must be a string")
    try:
        return _OBSERVABLE_ALIASES[value]
    except KeyError as error:
        raise ValueError(
            "observable must be 'potential', 'acceleration', or "
            "'gravity_gradient' (aliases: 'GG', 'gg', 'gradient')"
        ) from error


@dataclass(frozen=True, eq=False)
class SourceContributionData:
    """Immutable source-resolved numeric values and replayable source blocks.

    The logical array layouts are ``(F, M, N, components...)`` for source
    values, ``(F, M, B, components...)`` for grouped values, and
    ``(F, M, components...)`` for totals. Exactly one of ``source_values`` or
    ``source_block_provider`` is required. A deferred provider is an internal,
    trusted factory and must close over immutable snapshots of its inputs; its
    returned blocks are validated and copied on every request.

    ``unit`` is stored as unit text so this numeric layer does not import
    Astropy. ``SourceContributionResult`` resolves it when constructing native
    frequency-series matrices.
    """

    source_values: np.ndarray | None
    group_values: np.ndarray
    total_values: np.ndarray
    positions_m: np.ndarray
    targets_m: np.ndarray
    target_ids: tuple[str, ...]
    frequencies_hz: np.ndarray
    unit: str
    observable: str
    frame: str
    membership: np.ndarray
    group_ids: tuple[Any, ...]
    source_block_provider: Callable[[int, int], np.ndarray] | None = None
    source_ids: tuple[str, ...] | None = None
    source_metadata: tuple[Mapping[str, Any] | None, ...] | None = None
    source_index: np.ndarray | None = None
    group_label_mapping: tuple[tuple[int, int], ...] | None = None
    radial_edges_m: np.ndarray | None = None
    coordinate_origin_m: np.ndarray | None = None
    softening_m: float = 0.0
    t_ref_s: float | None = None
    phase_origin_m: np.ndarray | None = None
    fourier_convention: str | None = None
    peak_phasor: bool = True
    global_metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        positions = _copy_real(self.positions_m, "positions_m")
        targets = _copy_real(self.targets_m, "targets_m")
        if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) == 0:
            raise ValueError("positions_m must have nonempty shape (N, 3)")
        if targets.ndim != 2 or targets.shape[1] != 3 or len(targets) == 0:
            raise ValueError("targets_m must have nonempty shape (M, 3)")
        n_sources, n_targets = len(positions), len(targets)

        target_ids = tuple(self.target_ids)
        if len(target_ids) != n_targets or any(
            not isinstance(item, str) or not item for item in target_ids
        ):
            raise ValueError("target_ids must be nonempty strings matching targets_m")
        if len(set(target_ids)) != n_targets:
            raise ValueError("target_ids must be unique and ordered")

        frequencies = _copy_real(self.frequencies_hz, "frequencies_hz")
        if frequencies.ndim != 1 or len(frequencies) == 0:
            raise ValueError("frequencies_hz must be a nonempty vector")
        if (frequencies <= 0.0).any() or (np.diff(frequencies) <= 0.0).any():
            raise ValueError("frequencies_hz must be positive and strictly increasing")

        observable = canonical_observable(self.observable)
        component_shape = _OBSERVABLES[observable]
        n_frequencies = len(frequencies)
        source = (
            None
            if self.source_values is None
            else _copy_readonly(
                self.source_values, "source_values", dtype=np.complex128
            )
        )
        groups = _copy_readonly(self.group_values, "group_values", dtype=np.complex128)
        total = _copy_readonly(self.total_values, "total_values", dtype=np.complex128)
        expected_source_shape = (n_frequencies, n_targets, n_sources, *component_shape)
        expected_total_shape = (n_frequencies, n_targets, *component_shape)
        expected_group_ndim = 3 + len(component_shape)
        if source is not None and source.shape != expected_source_shape:
            raise ValueError(
                f"source_values must have shape {expected_source_shape}, "
                f"got {source.shape}"
            )
        if (
            groups.ndim != expected_group_ndim
            or groups.shape[:2] != (n_frequencies, n_targets)
            or groups.shape[3:] != component_shape
        ):
            raise ValueError(
                "group_values must have shape (F,M,B,components...) matching observable"
            )
        if groups.shape[2] == 0:
            raise ValueError("group_values must contain at least one group")
        if total.shape != expected_total_shape:
            raise ValueError(
                f"total_values must have shape {expected_total_shape}, "
                f"got {total.shape}"
            )
        provider = self.source_block_provider
        if source is None and not callable(provider):
            raise ValueError(
                "source_values=None requires a source_block_provider callable"
            )
        if source is not None and provider is not None:
            raise ValueError("provide source_values or source_block_provider, not both")

        membership = _copy_integer(self.membership, "membership")
        n_groups = groups.shape[2]
        if membership.shape != (n_targets, n_sources):
            raise ValueError(f"membership must have shape {(n_targets, n_sources)}")
        if (membership < 0).any() or (membership >= n_groups).any():
            raise ValueError(
                "membership entries must identify a group for every source"
            )
        group_ids = tuple(
            _freeze_identifier(item, "group ID") for item in self.group_ids
        )
        if len(group_ids) != n_groups or len(set(group_ids)) != n_groups:
            raise ValueError("group_ids must be unique and match group_values")

        if not isinstance(self.frame, str) or not self.frame.strip():
            raise ValueError("frame must be a nonempty string")
        if not isinstance(self.unit, str):
            raise TypeError("unit must be nonempty unit text")
        unit = self.unit
        if not unit.strip():
            raise ValueError("unit must be nonempty unit text")
        if isinstance(self.softening_m, (bool, np.bool_)):
            raise ValueError("softening_m must be a finite non-negative scalar")
        try:
            softening = float(self.softening_m)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "softening_m must be a finite non-negative scalar"
            ) from error
        if not np.isfinite(softening) or softening < 0.0:
            raise ValueError("softening_m must be a finite non-negative scalar")

        source_ids = None
        if self.source_ids is not None:
            source_ids = tuple(self.source_ids)
            if len(source_ids) != n_sources or any(
                not isinstance(item, str) or not item for item in source_ids
            ):
                raise ValueError(
                    "source_ids must match positions_m with nonempty strings"
                )
            if len(set(source_ids)) != n_sources:
                raise ValueError("source_ids must be unique and ordered")

        metadata = None
        if self.source_metadata is not None:
            if len(self.source_metadata) != n_sources:
                raise ValueError("source_metadata must match positions_m")
            for item in self.source_metadata:
                if item is not None and not isinstance(item, Mapping):
                    raise TypeError(
                        "each source_metadata entry must be a mapping or None"
                    )
            metadata = tuple(
                None if item is None else _freeze_metadata(item)
                for item in self.source_metadata
            )

        global_metadata = self.global_metadata
        if global_metadata is not None:
            if not isinstance(global_metadata, Mapping):
                raise TypeError("global_metadata must be a mapping or None")
            global_metadata = _freeze_metadata(global_metadata)

        source_index = _copy_integer(
            np.arange(n_sources) if self.source_index is None else self.source_index,
            "source_index",
        )
        if source_index.shape != (n_sources,) or not np.array_equal(
            source_index, np.arange(n_sources)
        ):
            raise ValueError("source_index must be the stable ordered indices 0..N-1")

        label_mapping = self.group_label_mapping
        if label_mapping is not None:
            normalized = []
            for original, index in label_mapping:
                if isinstance(original, (bool, np.bool_)) or not isinstance(
                    original, (int, np.integer)
                ):
                    raise TypeError("group_label_mapping labels must be integers")
                if original < 0:
                    raise ValueError("group_label_mapping labels must be non-negative")
                if isinstance(index, (bool, np.bool_)) or not isinstance(
                    index, (int, np.integer)
                ):
                    raise TypeError("group_label_mapping indices must be integers")
                normalized.append((int(original), int(index)))
            label_mapping = tuple(normalized)
            if len({original for original, _ in label_mapping}) != len(label_mapping):
                raise ValueError("group_label_mapping labels must be unique")
            if tuple(index for _, index in label_mapping) != tuple(range(n_groups)):
                raise ValueError(
                    "group_label_mapping indices must be dense and ordered"
                )

        edges = None
        if self.radial_edges_m is not None:
            edges = _copy_real(self.radial_edges_m, "radial_edges_m")
            if edges.ndim != 1 or len(edges) < 2 or (np.diff(edges) <= 0.0).any():
                raise ValueError(
                    "radial_edges_m must be finite and strictly increasing"
                )
        origin = (
            _copy_real(np.zeros(3), "coordinate_origin_m")
            if self.coordinate_origin_m is None
            else _copy_real(self.coordinate_origin_m, "coordinate_origin_m")
        )
        if origin.shape != (3,):
            raise ValueError("coordinate_origin_m must have shape (3,)")
        phase_origin = None
        if self.phase_origin_m is not None:
            phase_origin = _copy_real(self.phase_origin_m, "phase_origin_m")
            if phase_origin.shape != (3,):
                raise ValueError("phase_origin_m must have shape (3,)")
        t_ref = self.t_ref_s
        if t_ref is not None:
            if isinstance(t_ref, (bool, np.bool_)):
                raise ValueError("t_ref_s must be finite")
            try:
                t_ref = float(t_ref)
            except (TypeError, ValueError) as error:
                raise ValueError("t_ref_s must be finite") from error
            if not np.isfinite(t_ref):
                raise ValueError("t_ref_s must be finite")
        if self.fourier_convention is not None and not isinstance(
            self.fourier_convention, str
        ):
            raise TypeError("fourier_convention must be a string or None")
        if not isinstance(self.peak_phasor, (bool, np.bool_)):
            raise TypeError("peak_phasor must be bool")

        object.__setattr__(self, "source_values", source)
        object.__setattr__(self, "group_values", groups)
        object.__setattr__(self, "total_values", total)
        object.__setattr__(self, "positions_m", positions)
        object.__setattr__(self, "targets_m", targets)
        object.__setattr__(self, "target_ids", target_ids)
        object.__setattr__(self, "frequencies_hz", frequencies)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "observable", observable)
        object.__setattr__(self, "membership", membership)
        object.__setattr__(self, "group_ids", group_ids)
        object.__setattr__(self, "source_ids", source_ids)
        object.__setattr__(self, "source_metadata", metadata)
        object.__setattr__(self, "global_metadata", global_metadata)
        object.__setattr__(self, "source_index", source_index)
        object.__setattr__(self, "group_label_mapping", label_mapping)
        object.__setattr__(self, "radial_edges_m", edges)
        object.__setattr__(self, "coordinate_origin_m", origin)
        object.__setattr__(self, "softening_m", softening)
        object.__setattr__(self, "t_ref_s", t_ref)
        object.__setattr__(self, "phase_origin_m", phase_origin)
        object.__setattr__(self, "peak_phasor", bool(self.peak_phasor))

    def __deepcopy__(self, memo: dict[int, Any]) -> SourceContributionData:
        """Reuse this immutable snapshot instead of copying its arrays/provider."""
        memo[id(self)] = self
        return self

    @property
    def source_count(self) -> int:
        return len(self.positions_m)

    @property
    def target_count(self) -> int:
        return len(self.targets_m)

    @property
    def frequency_count(self) -> int:
        return len(self.frequencies_hz)

    @property
    def group_count(self) -> int:
        return len(self.group_ids)

    @property
    def component_shape(self) -> tuple[int, ...]:
        return _OBSERVABLES[self.observable]

    def source_block(self, start: int, stop: int) -> np.ndarray:
        """Return one validated immutable source slab ``(F,M,B,...)``."""
        for value, label in ((start, "start"), (stop, "stop")):
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise TypeError(f"source block {label} must be an integer")
        start, stop = int(start), int(stop)
        if start < 0 or stop > self.source_count or start >= stop:
            raise IndexError("source block bounds must satisfy 0 <= start < stop <= N")
        if self.source_values is not None:
            raw = self.source_values[:, :, start:stop, ...]
        else:
            assert self.source_block_provider is not None
            raw = self.source_block_provider(start, stop)
        values = _copy_readonly(raw, "source block", dtype=np.complex128)
        expected = (
            self.frequency_count,
            self.target_count,
            stop - start,
            *self.component_shape,
        )
        if values.shape != expected:
            raise ValueError(
                f"source block provider returned shape {values.shape}, "
                f"expected {expected}"
            )
        return values

    def iter_source_blocks(
        self, block_size: int = 256, *, frequency_index: int | None = None
    ) -> Iterator[tuple[int, int, np.ndarray]]:
        """Yield validated read-only source blocks in input order.

        Unselected blocks have shape ``(F,M,B,components...)``. Selecting a
        frequency removes only the frequency axis. A lazy provider receives
        exactly the ``(start, stop)`` bounds shown in each yielded tuple.
        """
        if (
            isinstance(block_size, (bool, np.bool_))
            or not isinstance(block_size, (int, np.integer))
            or block_size <= 0
        ):
            raise ValueError("block_size must be a positive integer")
        if frequency_index is not None:
            if isinstance(frequency_index, (bool, np.bool_)) or not isinstance(
                frequency_index, (int, np.integer)
            ):
                raise TypeError("frequency_index must be an integer")
            if frequency_index < 0 or frequency_index >= self.frequency_count:
                raise IndexError("frequency_index is out of range")
        for start in range(0, self.source_count, int(block_size)):
            stop = min(start + int(block_size), self.source_count)
            values = self.source_block(start, stop)
            if frequency_index is not None:
                values = _copy_readonly(values[int(frequency_index)], "source block")
            yield start, stop, values
