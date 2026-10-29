"""Context-managed bounded SPECFEM GLL displacement provider."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from gragra.adapters.specfem import schema
from gragra.adapters.specfem._validation import (
    validate_numeric_slab,
    validate_specfem_h5,
)
from gragra.integrators.streaming import (
    DEFAULT_MEMORY_BUDGET_BYTES,
    DisplacementSourceSeriesBlock,
    SeriesMetadata,
    _bounded_slice,
    _positive_count,
    as_series_provider,
)


def _import_h5py():
    try:
        import h5py
    except ImportError as error:
        raise ImportError("h5py is required for SPECFEM GLL HDF5 streaming") from error
    return h5py


class SpecfemGLLStreamReader:
    """Validate and expose a canonical SPECFEM file without loading its field."""

    def __init__(
        self,
        path: str | Path,
        *,
        profile_density_range: tuple[float, float] | None,
        sample_slice: slice | None = None,
        element_slice: slice | None = None,
        slab_elements: int = 512,
        memory_budget_bytes: int = DEFAULT_MEMORY_BUDGET_BYTES,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.profile_density_range = profile_density_range
        self.requested_sample_slice = sample_slice
        self.requested_element_slice = element_slice
        self.slab_elements = _positive_count(slab_elements, "slab_elements")
        self.memory_budget_bytes = _positive_count(
            memory_budget_bytes, "memory_budget_bytes"
        )
        self._h5 = None
        self._provider = None
        self._layout = None
        self.dataset_chunk_cache_bytes = 0
        self.chunk_cache_bytes = 0
        self.decoded_chunk_workspace_bytes = 0
        self.validation_field_buffer_bytes = 0
        self.estimated_resident_bytes = 0
        self.last_provider_block_estimated_bytes = 0
        self._provider_fixed_bytes = 0

    @property
    def closed(self) -> bool:
        return self._h5 is None

    def _chunk_cache_plan(self, h5: Any) -> tuple[int, int, int]:
        chunk_sizes = []
        for path in schema.required_dataset_paths("time_sampled"):
            dataset = h5[path]
            if dataset.chunks is not None:
                chunk_sizes.append(
                    int(
                        np.prod(dataset.chunks, dtype=np.int64) * dataset.dtype.itemsize
                    )
                )
        if not chunk_sizes:
            return 0, 0, 0
        # A time-major request that touches many full-width field chunks can
        # make HDF5 retain one decoded chunk per time sample despite the byte
        # target.  Disable the raw chunk cache and reuse explicit NumPy slabs.
        per_dataset = 1
        return per_dataset, per_dataset * len(chunk_sizes), max(chunk_sizes)

    def __enter__(self):
        if self._h5 is not None:
            raise RuntimeError("SPECFEM stream reader is already open")
        h5py = _import_h5py()
        try:
            with h5py.File(self.path, "r") as probe:
                cache_limit, cache_resident, decoded_chunk_bytes = (
                    self._chunk_cache_plan(probe)
                )
                coordinate_shape = probe[schema.PATH_MESH_COORDINATES].shape
                displacement_dataset = probe[schema.PATH_FIELD_DISPLACEMENT]
                n_time = displacement_dataset.shape[0]
                displacement_chunks = displacement_dataset.chunks
                displacement_itemsize = displacement_dataset.dtype.itemsize
                if len(coordinate_shape) != 3:
                    raise ValueError(
                        "/mesh/coordinates must have shape (element, gll, 3)"
                    )
                n_elements, ngll, _ = coordinate_shape
            selected_elements = (
                slice(0, n_elements)
                if self.requested_element_slice is None
                else _bounded_slice(
                    self.requested_element_slice, n_elements, "element_slice"
                )
            )
            selected_element_count = selected_elements.stop - selected_elements.start
            source_count = selected_element_count * ngll
            selected_samples = (
                slice(0, n_time)
                if self.requested_sample_slice is None
                else _bounded_slice(self.requested_sample_slice, n_time, "sample_slice")
            )
            if displacement_chunks is None:
                validation_time_slab = 1
                validation_chunk_elements = 1
            else:
                validation_time_slab = min(
                    displacement_chunks[0],
                    selected_samples.stop - selected_samples.start,
                )
                validation_chunk_elements = min(displacement_chunks[1], n_elements)

            def validation_field_buffer_bytes(slab_elements: int) -> int:
                field_elements = min(
                    max(slab_elements, validation_chunk_elements), n_elements
                )
                return int(
                    validation_time_slab
                    * field_elements
                    * ngll
                    * 3
                    * displacement_itemsize
                )

            layout_bytes = n_elements * 8 + n_time * 3 * 8 + 4096
            fixed_bytes = cache_resident + source_count * (3 * 8 + 72) + layout_bytes
            # During a provider yield, HDF5 read buffers and the immutable
            # block's defensive copies coexist.  Include both, plus mass and
            # source-ID slice temporaries, rather than counting output alone.
            per_element_slab_bytes = ngll * 160

            def estimated_peak_bytes(slab_elements: int) -> int:
                provider_elements = min(selected_element_count, slab_elements)
                return int(
                    fixed_bytes
                    + decoded_chunk_bytes
                    + validation_field_buffer_bytes(slab_elements)
                    + provider_elements * per_element_slab_bytes
                )

            if self.memory_budget_bytes < estimated_peak_bytes(1):
                raise MemoryError(
                    "memory budget cannot hold decoded HDF5 chunk workspace, "
                    "validation field buffer, and minimum SPECFEM slab"
                )
            low = 1
            high = min(self.slab_elements, n_elements)
            while low < high:
                candidate = (low + high + 1) // 2
                if estimated_peak_bytes(candidate) <= self.memory_budget_bytes:
                    low = candidate
                else:
                    high = candidate - 1
            self.slab_elements = low
            self.dataset_chunk_cache_bytes = cache_limit
            self.chunk_cache_bytes = cache_resident
            self.decoded_chunk_workspace_bytes = decoded_chunk_bytes
            self.validation_field_buffer_bytes = validation_field_buffer_bytes(
                self.slab_elements
            )
            self._h5 = h5py.File(
                self.path,
                "r",
                rdcc_nbytes=max(cache_limit, 1),
                rdcc_nslots=521,
                rdcc_w0=1.0,
            )
            layout = validate_specfem_h5(
                self._h5,
                h5py,
                profile_density_range=self.profile_density_range,
                slab_elements=self.slab_elements,
                sample_slice=self.requested_sample_slice,
            )
            if layout.ngll != 125:
                raise ValueError(
                    "SPECFEM streaming requires schema-v1 5x5x5 "
                    "(125-point) GLL elements"
                )
            self._layout = layout
            positions_bytes = source_count * 3 * np.dtype(np.float64).itemsize
            ids_bytes = source_count * 72
            layout_bytes = layout.element_id.nbytes + sum(
                value.nbytes for value in (layout.time_s, layout.time_step)
            )
            axis_bytes = layout.time_s.nbytes
            slab_sources = min(selected_element_count, self.slab_elements) * layout.ngll
            slab_bytes = slab_sources * 160
            self.estimated_resident_bytes = int(
                self.chunk_cache_bytes
                + self.decoded_chunk_workspace_bytes
                + self.validation_field_buffer_bytes
                + positions_bytes
                + ids_bytes
                + layout_bytes
                + axis_bytes
                + slab_bytes
                + 4096
            )
            if self.estimated_resident_bytes > self.memory_budget_bytes:
                raise MemoryError(
                    "estimated SPECFEM resident memory exceeds memory budget"
                )
            self._provider_fixed_bytes = int(
                self.chunk_cache_bytes
                + self.decoded_chunk_workspace_bytes
                + positions_bytes
                + ids_bytes
                + layout_bytes
                + axis_bytes
                + 4096
            )
            positions = np.empty((source_count, 3), dtype=np.float64)
            for element_start in range(
                selected_elements.start, selected_elements.stop, self.slab_elements
            ):
                element_stop = min(
                    element_start + self.slab_elements, selected_elements.stop
                )
                slab = np.asarray(
                    self._h5[schema.PATH_MESH_COORDINATES][
                        slice(element_start, element_stop), :, :
                    ]
                )
                validate_numeric_slab(slab)
                start = (element_start - selected_elements.start) * layout.ngll
                stop = (element_stop - selected_elements.start) * layout.ngll
                positions[start:stop] = slab.reshape(-1, 3)
            source_ids = tuple(
                f"{int(element_id)}:{local_index}"
                for element_id in layout.element_id[selected_elements]
                for local_index in range(layout.ngll)
            )
            metadata = SeriesMetadata(
                positions_m=positions,
                source_ids=source_ids,
                coordinate_frame="ENU",
                coordinate_origin_m=np.zeros(3, dtype=np.float64),
                axis_kind="time",
                axis_values=layout.time_s,
                sample_count=len(layout.time_s),
                source_count=source_count,
                quantity="displacement",
            )
            sample_offset = layout.sample_slice.start

            def factory(samples: slice, source_block_size: int):
                if self._h5 is None:
                    raise RuntimeError("SPECFEM stream reader is closed")
                global_samples = slice(
                    sample_offset + samples.start, sample_offset + samples.stop
                )
                sample_count = samples.stop - samples.start

                def provider_peak_bytes(source_start: int, width: int) -> int:
                    local_element_start = source_start // layout.ngll
                    local_element_stop = math.ceil((source_start + width) / layout.ngll)
                    read_sources = (
                        local_element_stop - local_element_start
                    ) * layout.ngll
                    raw_geometry = read_sources * (3 * 8 + 8 + 8)
                    raw_displacement = sample_count * read_sources * 3 * 8
                    raw_mass = width * 8
                    defensive_block = width * (3 * 8 + 8 + 8)
                    defensive_displacement = sample_count * width * 3 * 8
                    finite_workspace = sample_count * width * 3
                    return int(
                        self._provider_fixed_bytes
                        + raw_geometry
                        + raw_displacement
                        + raw_mass
                        + defensive_block
                        + defensive_displacement
                        + finite_workspace
                    )

                source_start = 0
                while source_start < source_count:
                    if self._h5 is None:
                        raise RuntimeError("SPECFEM stream reader is closed")
                    maximum_width = min(source_block_size, source_count - source_start)
                    if provider_peak_bytes(source_start, 1) > self.memory_budget_bytes:
                        raise MemoryError(
                            "memory budget cannot hold the requested sample count "
                            "and minimum SPECFEM provider block"
                        )
                    low = 1
                    high = maximum_width
                    while low < high:
                        candidate = (low + high + 1) // 2
                        if (
                            provider_peak_bytes(source_start, candidate)
                            <= self.memory_budget_bytes
                        ):
                            low = candidate
                        else:
                            high = candidate - 1
                    source_stop = source_start + low
                    self.last_provider_block_estimated_bytes = provider_peak_bytes(
                        source_start, low
                    )
                    local_element_start = source_start // layout.ngll
                    local_element_stop = math.ceil(source_stop / layout.ngll)
                    element_start = selected_elements.start + local_element_start
                    element_stop = selected_elements.start + local_element_stop
                    local_start = source_start - local_element_start * layout.ngll
                    local_stop = local_start + (source_stop - source_start)
                    element_key = slice(element_start, element_stop)
                    positions_slab = np.asarray(
                        self._h5[schema.PATH_MESH_COORDINATES][element_key, :, :]
                    ).reshape(-1, 3)[local_start:local_stop]
                    density_slab = np.asarray(
                        self._h5[schema.PATH_MATERIAL_DENSITY][element_key, :]
                    ).reshape(-1)[local_start:local_stop]
                    volume_slab = np.asarray(
                        self._h5[schema.PATH_MESH_QUADRATURE_VOLUME][element_key, :]
                    ).reshape(-1)[local_start:local_stop]
                    element_count = element_stop - element_start
                    displacement_elements = np.empty(
                        (sample_count, element_count, layout.ngll, 3),
                        dtype=np.float64,
                    )
                    displacement_dataset = self._h5[schema.PATH_FIELD_DISPLACEMENT]
                    for local_sample, global_sample in enumerate(
                        range(global_samples.start, global_samples.stop)
                    ):
                        displacement_dataset.read_direct(
                            displacement_elements,
                            source_sel=(
                                slice(global_sample, global_sample + 1),
                                element_key,
                                slice(0, layout.ngll),
                                slice(0, 3),
                            ),
                            dest_sel=(
                                slice(local_sample, local_sample + 1),
                                slice(0, element_count),
                                slice(0, layout.ngll),
                                slice(0, 3),
                            ),
                        )
                    displacement_slab = displacement_elements.reshape(
                        sample_count, -1, 3
                    )[:, local_start:local_stop, :]
                    validate_numeric_slab(
                        positions_slab,
                        density_slab,
                        volume_slab,
                        displacement_slab,
                    )
                    mass = density_slab * volume_slab
                    block = DisplacementSourceSeriesBlock(
                        positions_slab,
                        samples,
                        source_start,
                        mass,
                        displacement_slab,
                        source_ids[source_start:source_stop],
                    )
                    del (
                        density_slab,
                        displacement_elements,
                        displacement_slab,
                        mass,
                        positions_slab,
                        volume_slab,
                    )
                    yield block
                    del block
                    source_start = source_stop

            self._provider = as_series_provider(metadata, factory)
            self._provider._set_resident_bytes(self.estimated_resident_bytes)
            return self
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
        self._h5 = None
        self._provider = None
        self._layout = None

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.close()
        return False

    @property
    def provider(self):
        if self._h5 is None or self._provider is None:
            raise RuntimeError("SPECFEM stream reader is closed")
        return self._provider
