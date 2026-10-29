"""HDF5 profile ``gragra.fixed-mesh-series`` version 1.

Numeric field values are written and read only through explicit sample/source
slabs.  Successful writes are published from a sibling ``.partial`` path only
after complete readback validation; failed partial files remain inspectable.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from gragra.integrators.streaming import (
    DisplacementSourceSeriesBlock,
    ScalarSourceSeriesBlock,
    SeriesMetadata,
    _finite_scalar,
    _numeric_copy,
    _positive_count,
    as_series_provider,
)

PROFILE = "gragra.fixed-mesh-series"
VERSION = 1
FOURIER_CONVENTION = "Re[q_hat exp(-2pi i f(t-t_ref_s))]"

_VALUE_UNITS = {
    "density": "kg/m3",
    "displacement": "m",
    "surface_density": "kg/m2",
}

_OBSERVABLE_UNITS = {
    "potential": "m2/s2",
    "acceleration": "m/s2",
    "gradient": "1/s2",
}


def _h5py():
    try:
        import h5py
    except ImportError as error:
        raise ImportError(
            "h5py is required for fixed-mesh series HDF5 input"
        ) from error
    return h5py


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _is_literal_integer(value: Any, expected: int) -> bool:
    return (
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
        and int(value) == expected
    )


def _is_literal_boolean(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_))


def _strings(values: Any) -> tuple[str, ...]:
    return tuple(_decode(value) for value in values)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"metadata value {type(value).__name__} is not JSON serializable")


def _quantity(series: Any) -> tuple[str, str, str]:
    if hasattr(series, "displacement_m") and hasattr(series, "mass_kg"):
        return "displacement", "displacement_m", "mass_kg"
    if hasattr(series, "surface_density_kg_m2") and hasattr(series, "areas_m2"):
        return "surface_density", "surface_density_kg_m2", "areas_m2"
    if hasattr(series, "density_kg_m3") and hasattr(series, "volumes_m3"):
        return "density", "density_kg_m3", "volumes_m3"
    raise TypeError("unsupported fixed-mesh series object")


def _series_metadata(series: Any, quantity: str) -> SeriesMetadata:
    return SeriesMetadata(
        positions_m=series.positions_m,
        source_ids=series.source_ids,
        coordinate_frame=series.coordinate_frame,
        coordinate_origin_m=series.coordinate_origin_m,
        axis_kind=series.axis_kind,
        axis_values=series.axis_values,
        sample_count=series.sample_count,
        source_count=series.source_count,
        t_ref_s=getattr(series, "t_ref_s", None),
        phase_origin_m=getattr(series, "phase_origin_m", None),
        quantity=quantity,
        density_kind=getattr(series, "density_kind", None),
    )


def _set_unit(dataset: Any, unit: str) -> None:
    dataset.attrs["unit"] = unit


def _hash_slab(digest: Any, slab: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(slab)
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(repr(contiguous.shape).encode("ascii"))
    digest.update(contiguous.view(np.uint8))


def _read_numeric_dataset_slabs(
    dataset: Any,
    *,
    shape: tuple[int, ...],
    name: str,
    block_size: int,
) -> np.ndarray:
    if dataset.shape != shape or np.dtype(dataset.dtype) != np.dtype(np.float64):
        raise ValueError(f"{name} has incorrect shape or dtype")
    result = np.empty(shape, dtype=np.float64)
    for start in range(0, shape[0], block_size):
        stop = min(start + block_size, shape[0])
        key = (slice(start, stop),) + tuple(slice(0, width) for width in shape[1:])
        result[key] = dataset[key]
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _validate_profile(
    h5: Any,
    *,
    allow_incomplete: bool = False,
    sample_block_size: int = 1,
    source_block_size: int = 8192,
) -> tuple[str, SeriesMetadata]:
    samples_per_block = _positive_count(sample_block_size, "sample_block_size")
    sources_per_block = _positive_count(source_block_size, "source_block_size")
    if _decode(h5.attrs.get("profile", "")) != PROFILE:
        raise ValueError("unsupported fixed-mesh series profile")
    if not _is_literal_integer(h5.attrs.get("version"), VERSION):
        raise ValueError("unsupported fixed-mesh series version")
    complete = h5.attrs.get("complete")
    if not _is_literal_boolean(complete):
        raise ValueError("fixed-mesh series complete flag must be Boolean")
    if not allow_incomplete and not bool(complete):
        raise ValueError("fixed-mesh series file is incomplete")
    quantity = _decode(h5.attrs.get("quantity", ""))
    if quantity not in _VALUE_UNITS:
        raise ValueError("unsupported fixed-mesh quantity")
    if "metadata" not in h5 or "values" not in h5:
        raise ValueError("fixed-mesh series is missing required datasets")
    metadata_group = h5["metadata"]
    required = ("positions_m", "source_ids", "coordinate_origin_m", "axis_values")
    if any(name not in metadata_group for name in required):
        raise ValueError("fixed-mesh series is missing metadata datasets")
    axis_kind = _decode(metadata_group.attrs.get("axis_kind", ""))
    coordinate_frame = _decode(metadata_group.attrs.get("coordinate_frame", ""))
    positions_dataset = metadata_group["positions_m"]
    if positions_dataset.ndim != 2 or positions_dataset.shape[1:] != (3,):
        raise ValueError("metadata/positions_m has incorrect shape")
    source_count = positions_dataset.shape[0]
    positions = _read_numeric_dataset_slabs(
        positions_dataset,
        shape=(source_count, 3),
        name="metadata/positions_m",
        block_size=sources_per_block,
    )
    ids_dataset = metadata_group["source_ids"]
    if (
        ids_dataset.shape != (source_count,)
        or _h5py().check_string_dtype(ids_dataset.dtype) is None
    ):
        raise ValueError("metadata/source_ids has incorrect shape or dtype")
    source_id_values: list[str] = []
    for start in range(0, source_count, sources_per_block):
        stop = min(start + sources_per_block, source_count)
        source_id_values.extend(_strings(ids_dataset[slice(start, stop)]))
    source_ids = tuple(source_id_values)
    axis_dataset = metadata_group["axis_values"]
    if axis_dataset.ndim != 1:
        raise ValueError("metadata/axis_values has incorrect shape")
    sample_count = axis_dataset.shape[0]
    axis = _read_numeric_dataset_slabs(
        axis_dataset,
        shape=(sample_count,),
        name="metadata/axis_values",
        block_size=samples_per_block,
    )
    origin = _read_numeric_dataset_slabs(
        metadata_group["coordinate_origin_m"],
        shape=(3,),
        name="metadata/coordinate_origin_m",
        block_size=3,
    )
    phase = (
        _read_numeric_dataset_slabs(
            metadata_group["phase_origin_m"],
            shape=(3,),
            name="metadata/phase_origin_m",
            block_size=3,
        )
        if "phase_origin_m" in metadata_group
        else None
    )
    t_ref = metadata_group.attrs.get("t_ref_s")
    density_kind = metadata_group.attrs.get("density_kind")
    if density_kind is not None:
        density_kind = _decode(density_kind)
    fourier_convention = metadata_group.attrs.get("fourier_convention")
    if axis_kind == "frequency":
        if _decode(fourier_convention) != FOURIER_CONVENTION:
            raise ValueError("frequency input has incorrect Fourier convention")
    elif fourier_convention is not None:
        raise ValueError("time input must not declare a Fourier convention")
    metadata = SeriesMetadata(
        positions_m=positions,
        source_ids=source_ids,
        coordinate_frame=coordinate_frame,
        coordinate_origin_m=origin,
        axis_kind=axis_kind,
        axis_values=axis,
        sample_count=sample_count,
        source_count=source_count,
        t_ref_s=t_ref,
        phase_origin_m=phase,
        quantity=quantity,
        density_kind=density_kind,
    )
    expected_units = {
        "positions_m": "m",
        "coordinate_origin_m": "m",
        "axis_values": "Hz" if axis_kind == "frequency" else "s",
    }
    if phase is not None:
        expected_units["phase_origin_m"] = "m"
    for name, unit in expected_units.items():
        if _decode(metadata_group[name].attrs.get("unit", "")) != unit:
            raise ValueError(f"metadata/{name} has incorrect unit")
    if _decode(ids_dataset.attrs.get("unit", "")) != "1":
        raise ValueError("metadata/source_ids has incorrect unit")
    values = h5["values"]
    expected_value_shape = (
        (metadata.sample_count, metadata.source_count, 3)
        if quantity == "displacement"
        else (metadata.sample_count, metadata.source_count)
    )
    if values.shape != expected_value_shape:
        raise ValueError("values dataset has incorrect shape")
    expected_dtype = np.dtype(np.complex128 if axis_kind == "frequency" else np.float64)
    if np.dtype(values.dtype) != expected_dtype:
        raise ValueError("values dataset has incorrect dtype")
    if _decode(values.attrs.get("unit", "")) != _VALUE_UNITS[quantity]:
        raise ValueError("values dataset has incorrect unit")
    geometry_specs = {
        "density": ("volumes_m3", (metadata.source_count,), "m3"),
        "displacement": ("mass_kg", (metadata.source_count,), "kg"),
        "surface_density": ("areas_m2", (metadata.source_count,), "m2"),
    }
    geometry_name, geometry_shape, geometry_unit = geometry_specs[quantity]
    if geometry_name not in metadata_group:
        raise ValueError(f"missing metadata/{geometry_name}")
    geometry = metadata_group[geometry_name]
    if geometry.shape != geometry_shape or np.dtype(geometry.dtype) != np.dtype(
        np.float64
    ):
        raise ValueError(f"metadata/{geometry_name} has incorrect shape or dtype")
    if _decode(geometry.attrs.get("unit", "")) != geometry_unit:
        raise ValueError(f"metadata/{geometry_name} has incorrect unit")
    for start in range(0, metadata.source_count, sources_per_block):
        stop = min(start + sources_per_block, metadata.source_count)
        geometry_values = np.asarray(geometry[slice(start, stop)])
        if not np.isfinite(geometry_values).all():
            raise ValueError(
                f"metadata/{geometry_name} must contain only finite values"
            )
        if quantity == "displacement":
            if np.any(geometry_values < 0.0):
                raise ValueError("metadata/mass_kg must be non-negative")
        elif np.any(geometry_values <= 0.0):
            raise ValueError(f"metadata/{geometry_name} must be positive")
    if quantity == "surface_density":
        if "normals" not in metadata_group or metadata_group["normals"].shape != (
            metadata.source_count,
            3,
        ):
            raise ValueError("surface density requires normals with shape (N,3)")
        if _decode(metadata_group["normals"].attrs.get("unit", "")) != "1":
            raise ValueError("metadata/normals has incorrect unit")
        normals_dataset = metadata_group["normals"]
        if np.dtype(normals_dataset.dtype) != np.dtype(np.float64):
            raise ValueError("metadata/normals has incorrect dtype")
        for start in range(0, metadata.source_count, sources_per_block):
            stop = min(start + sources_per_block, metadata.source_count)
            normals = np.asarray(normals_dataset[(slice(start, stop), slice(0, 3))])
            if not np.isfinite(normals).all() or not np.allclose(
                np.linalg.norm(normals, axis=1), 1.0, rtol=0.0, atol=1e-12
            ):
                raise ValueError("metadata/normals must contain finite unit vectors")
    return quantity, metadata


def _validate_value_slabs(
    h5: Any, metadata: SeriesMetadata, *, sample_block_size: int, source_block_size: int
) -> str:
    digest = hashlib.sha256()
    values = h5["values"]
    for sample_start in range(0, metadata.sample_count, sample_block_size):
        sample_stop = min(sample_start + sample_block_size, metadata.sample_count)
        for source_start in range(0, metadata.source_count, source_block_size):
            source_stop = min(source_start + source_block_size, metadata.source_count)
            key = (slice(sample_start, sample_stop), slice(source_start, source_stop))
            if values.ndim == 3:
                key += (slice(0, 3),)
            slab = np.asarray(values[key])
            if not np.isfinite(slab).all():
                raise ValueError("values must contain only finite values")
            if (
                metadata.quantity == "density"
                and metadata.density_kind == "rho"
                and np.any(slab < 0.0)
            ):
                raise ValueError("rho density values must be non-negative")
            _hash_slab(digest, slab)
    return digest.hexdigest()


def _same_metadata(left: SeriesMetadata, right: SeriesMetadata) -> bool:
    scalar_names = (
        "source_ids",
        "coordinate_frame",
        "axis_kind",
        "sample_count",
        "source_count",
        "t_ref_s",
        "quantity",
        "density_kind",
    )
    if any(getattr(left, name) != getattr(right, name) for name in scalar_names):
        return False
    array_names = ("positions_m", "coordinate_origin_m", "axis_values")
    if any(
        not np.array_equal(getattr(left, name), getattr(right, name))
        for name in array_names
    ):
        return False
    if left.phase_origin_m is None or right.phase_origin_m is None:
        return left.phase_origin_m is None and right.phase_origin_m is None
    return np.array_equal(left.phase_origin_m, right.phase_origin_m)


def _dataset_equals_array_in_slabs(
    dataset: Any, expected: np.ndarray, *, block_size: int
) -> bool:
    """Compare a numeric dataset exactly without materializing it in full."""
    expected_array = np.asarray(expected)
    if dataset.shape != expected_array.shape or np.dtype(dataset.dtype) != np.dtype(
        expected_array.dtype
    ):
        return False
    for start in range(0, expected_array.shape[0], block_size):
        stop = min(start + block_size, expected_array.shape[0])
        key = (slice(start, stop),) + tuple(
            slice(0, width) for width in expected_array.shape[1:]
        )
        if not np.array_equal(np.asarray(dataset[key]), expected_array[key]):
            return False
    return True


def write_fixed_mesh_series_hdf5(
    path: str | Path,
    series: Any,
    *,
    sample_block_size: int = 1,
    source_block_size: int = 8192,
) -> Path:
    """Write, read back, and atomically publish one fixed-mesh series."""
    h5py = _h5py()
    final_path = Path(path).expanduser().resolve()
    partial_path = Path(f"{final_path}.partial")
    if final_path.exists() or partial_path.exists():
        raise FileExistsError(f"refusing to overwrite {final_path} or {partial_path}")
    samples_per_block = _positive_count(sample_block_size, "sample_block_size")
    sources_per_block = _positive_count(source_block_size, "source_block_size")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    quantity, value_name, geometry_name = _quantity(series)
    metadata = _series_metadata(series, quantity)
    expected_json_attributes = {}
    with h5py.File(partial_path, "x") as h5:
        h5.attrs["profile"] = PROFILE
        h5.attrs["version"] = VERSION
        h5.attrs["quantity"] = quantity
        h5.attrs["complete"] = False
        group = h5.create_group("metadata")
        group.attrs["axis_kind"] = metadata.axis_kind
        group.attrs["coordinate_frame"] = metadata.coordinate_frame
        if metadata.density_kind is not None:
            group.attrs["density_kind"] = metadata.density_kind
        if metadata.t_ref_s is not None:
            group.attrs["t_ref_s"] = metadata.t_ref_s
            group.attrs["fourier_convention"] = FOURIER_CONVENTION
        for key in ("source_metadata", "face_metadata"):
            if hasattr(series, key):
                value = json.dumps(
                    _jsonable(getattr(series, key)),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                group.attrs[f"{key}_json"] = value
                expected_json_attributes[f"{key}_json"] = value
        positions = group.create_dataset("positions_m", data=metadata.positions_m)
        _set_unit(positions, "m")
        ids = group.create_dataset(
            "source_ids",
            data=np.asarray(metadata.source_ids, dtype=h5py.string_dtype("utf-8")),
        )
        ids.attrs["unit"] = "1"
        origin = group.create_dataset(
            "coordinate_origin_m", data=metadata.coordinate_origin_m
        )
        _set_unit(origin, "m")
        axis = group.create_dataset("axis_values", data=metadata.axis_values)
        _set_unit(axis, "Hz" if metadata.axis_kind == "frequency" else "s")
        if metadata.phase_origin_m is not None:
            phase = group.create_dataset("phase_origin_m", data=metadata.phase_origin_m)
            _set_unit(phase, "m")
        geometry = _numeric_copy(
            getattr(series, geometry_name), geometry_name, complex_ok=False
        )
        if geometry.shape != (metadata.source_count,):
            raise ValueError(f"{geometry_name} must have shape (N,)")
        if quantity == "displacement":
            if np.any(geometry < 0.0):
                raise ValueError("mass_kg must be non-negative")
            geometry_unit = "kg"
        else:
            if np.any(geometry <= 0.0):
                raise ValueError(f"{geometry_name} must be positive")
            geometry_unit = "m3" if quantity == "density" else "m2"
        geometry_dataset = group.create_dataset(geometry_name, data=geometry)
        _set_unit(geometry_dataset, geometry_unit)
        if quantity == "surface_density":
            normals = _numeric_copy(series.normals, "normals", complex_ok=False)
            if normals.shape != (metadata.source_count, 3) or not np.allclose(
                np.linalg.norm(normals, axis=1), 1.0, rtol=0.0, atol=1e-12
            ):
                raise ValueError("normals must be unit vectors with shape (N,3)")
            normals_dataset = group.create_dataset("normals", data=normals)
            _set_unit(normals_dataset, "1")
        shape = (
            (metadata.sample_count, metadata.source_count, 3)
            if quantity == "displacement"
            else (metadata.sample_count, metadata.source_count)
        )
        dtype = np.complex128 if metadata.axis_kind == "frequency" else np.float64
        chunks = (
            min(samples_per_block, metadata.sample_count),
            min(sources_per_block, metadata.source_count),
        ) + ((3,) if quantity == "displacement" else ())
        values_dataset = h5.create_dataset(
            "values", shape=shape, dtype=dtype, chunks=chunks
        )
        _set_unit(values_dataset, _VALUE_UNITS[quantity])
        source_values = getattr(series, value_name)
        write_digest = hashlib.sha256()
        for sample_start in range(0, metadata.sample_count, samples_per_block):
            sample_stop = min(sample_start + samples_per_block, metadata.sample_count)
            for source_start in range(0, metadata.source_count, sources_per_block):
                source_stop = min(
                    source_start + sources_per_block, metadata.source_count
                )
                suffix = (slice(0, 3),) if quantity == "displacement" else ()
                key = (
                    slice(sample_start, sample_stop),
                    slice(source_start, source_stop),
                ) + suffix
                slab = _numeric_copy(source_values[key], value_name)
                if metadata.axis_kind == "time" and np.iscomplexobj(slab):
                    raise TypeError("time values must be real float64")
                if metadata.axis_kind == "frequency" and not np.iscomplexobj(slab):
                    raise TypeError("frequency values must be complex128")
                expected_slab_shape = (
                    sample_stop - sample_start,
                    source_stop - source_start,
                ) + ((3,) if quantity == "displacement" else ())
                if slab.shape != expected_slab_shape:
                    raise ValueError(f"{value_name} has incorrect shape")
                values_dataset[key] = slab
                _hash_slab(write_digest, slab)
        h5.flush()
    with h5py.File(partial_path, "r") as h5:
        try:
            checked_quantity, checked_metadata = _validate_profile(
                h5,
                allow_incomplete=True,
                sample_block_size=samples_per_block,
                source_block_size=sources_per_block,
            )
        except ValueError as error:
            raise OSError("fixed-mesh series metadata failed readback") from error
        group = h5["metadata"]
        actual_json_attributes = {
            name: _decode(group.attrs[name])
            for name in ("source_metadata_json", "face_metadata_json")
            if name in group.attrs
        }
        geometry_matches = _dataset_equals_array_in_slabs(
            group[geometry_name], geometry, block_size=sources_per_block
        )
        normals_matches = quantity != "surface_density" or (
            _dataset_equals_array_in_slabs(
                group["normals"], normals, block_size=sources_per_block
            )
        )
        if (
            checked_quantity != quantity
            or not _same_metadata(checked_metadata, metadata)
            or actual_json_attributes != expected_json_attributes
            or not geometry_matches
            or not normals_matches
        ):
            raise OSError("fixed-mesh series metadata differs from original input")
        read_digest = _validate_value_slabs(
            h5,
            checked_metadata,
            sample_block_size=samples_per_block,
            source_block_size=sources_per_block,
        )
    if read_digest != write_digest.hexdigest():
        raise OSError("fixed-mesh series readback differs from written values")
    with h5py.File(partial_path, "r+") as h5:
        h5.attrs.modify("complete", True)
        h5.flush()
    if final_path.exists():
        raise FileExistsError(f"refusing to overwrite {final_path}")
    os.link(partial_path, final_path)
    partial_path.unlink()
    return final_path


def read_fixed_mesh_series_hdf5(
    path: str | Path,
    *,
    metadata_sample_block_size: int = 1,
    metadata_source_block_size: int = 8192,
):
    """Return a replayable provider backed by explicit HDF5 hyperslabs."""
    h5py = _h5py()
    resolved = Path(path).expanduser().resolve()
    metadata_samples = _positive_count(
        metadata_sample_block_size, "metadata_sample_block_size"
    )
    metadata_sources = _positive_count(
        metadata_source_block_size, "metadata_source_block_size"
    )
    with h5py.File(resolved, "r") as h5:
        quantity, metadata = _validate_profile(
            h5,
            sample_block_size=metadata_samples,
            source_block_size=metadata_sources,
        )

    def factory(samples: slice, source_block_size: int):
        with h5py.File(resolved, "r") as h5:
            checked_quantity, checked_metadata = _validate_profile(
                h5,
                sample_block_size=metadata_samples,
                source_block_size=metadata_sources,
            )
            if checked_quantity != quantity or not _same_metadata(
                checked_metadata, metadata
            ):
                raise ValueError("fixed-mesh series metadata changed between replays")
            group = h5["metadata"]
            values = h5["values"]
            for start in range(0, metadata.source_count, source_block_size):
                stop = min(start + source_block_size, metadata.source_count)
                value_key = (samples, slice(start, stop))
                if quantity == "displacement":
                    value_key += (slice(0, 3),)
                slab = np.asarray(values[value_key])
                if not np.isfinite(slab).all():
                    raise ValueError("values must contain only finite values")
                if quantity == "displacement":
                    yield DisplacementSourceSeriesBlock(
                        metadata.positions_m[start:stop],
                        samples,
                        start,
                        np.asarray(group["mass_kg"][slice(start, stop)]),
                        slab,
                        metadata.source_ids[start:stop],
                    )
                else:
                    geometry_name = (
                        "volumes_m3" if quantity == "density" else "areas_m2"
                    )
                    scale = np.asarray(group[geometry_name][slice(start, stop)])
                    yield ScalarSourceSeriesBlock(
                        metadata.positions_m[start:stop],
                        samples,
                        start,
                        slab * scale,
                        metadata.source_ids[start:stop],
                    )

    provider = as_series_provider(metadata, factory)
    provider._set_resident_bytes(
        metadata.positions_m.nbytes
        + metadata.axis_values.nbytes
        + metadata.coordinate_origin_m.nbytes
        + sum(len(value.encode("utf-8")) + 57 for value in metadata.source_ids)
    )
    return provider


def write_observable_series_hdf5(
    path: str | Path,
    blocks,
    *,
    sample_count: int,
    target_count: int,
    axis_kind: str,
    axis_values: Any,
    targets_m: Any,
    observable: str,
    coordinate_frame: str,
    coordinate_origin_m: Any,
    t_ref_s: float | None = None,
    phase_origin_m: Any | None = None,
    fourier_convention: str | None = None,
    sample_block_size: int = 1,
    target_block_size: int = 128,
) -> Path:
    """Persist a complete observable block tiling through ``.partial``."""
    from gragra.integrators.streaming import _bounded_slice
    from gragra.observables.streaming import ObservableSeriesBlock

    h5py = _h5py()
    samples = _positive_count(sample_count, "sample_count")
    targets = _positive_count(target_count, "target_count")
    samples_per_block = _positive_count(sample_block_size, "sample_block_size")
    targets_per_block = _positive_count(target_block_size, "target_block_size")
    if axis_kind not in {"time", "frequency"}:
        raise ValueError("axis_kind must be 'time' or 'frequency'")
    if observable not in {"potential", "acceleration", "gradient"}:
        raise ValueError("observable must be potential, acceleration, or gradient")
    axis = _numeric_copy(axis_values, "axis_values", complex_ok=False)
    target_positions = _numeric_copy(targets_m, "targets_m", complex_ok=False)
    origin = _numeric_copy(coordinate_origin_m, "coordinate_origin_m", complex_ok=False)
    if axis.shape != (samples,) or np.any(np.diff(axis) <= 0.0):
        raise ValueError(
            "axis_values must be strictly increasing with sample_count entries"
        )
    if axis_kind == "frequency" and np.any(axis <= 0.0):
        raise ValueError("frequency axis values must be positive")
    if target_positions.shape != (targets, 3):
        raise ValueError("targets_m must have shape (target_count,3)")
    if origin.shape != (3,):
        raise ValueError("coordinate_origin_m must have shape (3,)")
    if not isinstance(coordinate_frame, str) or not coordinate_frame.strip():
        raise ValueError("coordinate_frame must be a non-empty string")
    phase = None
    t_ref = None
    if axis_kind == "frequency":
        if t_ref_s is None:
            raise ValueError("frequency output requires t_ref_s")
        t_ref = _finite_scalar(t_ref_s, "t_ref_s")
        if phase_origin_m is None:
            raise ValueError("frequency output requires phase_origin_m")
        phase = _numeric_copy(phase_origin_m, "phase_origin_m", complex_ok=False)
        if phase.shape != (3,):
            raise ValueError("phase_origin_m must have shape (3,)")
        if fourier_convention != FOURIER_CONVENTION:
            raise ValueError(
                "frequency output requires the supported Fourier convention"
            )
    elif any(
        value is not None for value in (t_ref_s, phase_origin_m, fourier_convention)
    ):
        raise ValueError("time output must not declare frequency phase metadata")
    iterator = iter(blocks)
    first = next(iterator, None)
    if first is None:
        raise ValueError("cannot write an empty observable block stream")
    if not isinstance(first, ObservableSeriesBlock):
        raise TypeError("blocks must contain ObservableSeriesBlock values")
    expected_tail = {
        "potential": (),
        "acceleration": (3,),
        "gradient": (3, 3),
    }[observable]
    if first.values.shape[2:] != expected_tail:
        raise ValueError("observable block trailing shape does not match observable")
    expected_dtype = np.dtype(np.complex128 if axis_kind == "frequency" else np.float64)
    if first.values.dtype != expected_dtype:
        raise TypeError("observable block dtype does not match axis_kind")
    final_path = Path(path).expanduser().resolve()
    partial_path = Path(f"{final_path}.partial")
    if final_path.exists() or partial_path.exists():
        raise FileExistsError(f"refusing to overwrite {final_path} or {partial_path}")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    public_shape = (
        (targets, samples) if axis_kind == "frequency" else (samples, targets)
    ) + expected_tail
    with h5py.File(partial_path, "x") as h5:
        h5.attrs["profile"] = "gragra.observable-series"
        h5.attrs["version"] = 1
        h5.attrs["complete"] = False
        h5.attrs["observable"] = observable
        h5.attrs["axis_kind"] = axis_kind
        h5.attrs["coordinate_frame"] = coordinate_frame
        if axis_kind == "frequency":
            h5.attrs["t_ref_s"] = t_ref
            h5.attrs["fourier_convention"] = fourier_convention
        axis_dataset = h5.create_dataset("axis_values", data=axis)
        axis_dataset.attrs["unit"] = "Hz" if axis_kind == "frequency" else "s"
        target_dataset = h5.create_dataset("targets_m", data=target_positions)
        target_dataset.attrs["unit"] = "m"
        origin_dataset = h5.create_dataset("coordinate_origin_m", data=origin)
        origin_dataset.attrs["unit"] = "m"
        if phase is not None:
            phase_dataset = h5.create_dataset("phase_origin_m", data=phase)
            phase_dataset.attrs["unit"] = "m"
        values = h5.create_dataset(
            "values", shape=public_shape, dtype=expected_dtype, chunks=True
        )
        values.attrs["unit"] = _OBSERVABLE_UNITS[observable]
        coverage = h5.create_dataset(
            "coverage",
            shape=(samples, targets),
            dtype=np.bool_,
            chunks=True,
            fillvalue=False,
        )

        def place(block: Any) -> None:
            if not isinstance(block, ObservableSeriesBlock):
                raise TypeError("blocks must contain ObservableSeriesBlock values")
            sample_slice = _bounded_slice(block.sample_slice, samples, "sample_slice")
            target_slice = _bounded_slice(block.target_slice, targets, "target_slice")
            if (
                block.values.shape[2:] != expected_tail
                or block.values.dtype != expected_dtype
            ):
                raise ValueError("observable blocks have inconsistent shape or dtype")
            for sample_start in range(
                sample_slice.start, sample_slice.stop, samples_per_block
            ):
                sample_stop = min(sample_start + samples_per_block, sample_slice.stop)
                sample_key = slice(sample_start, sample_stop)
                local_sample = slice(
                    sample_start - sample_slice.start,
                    sample_stop - sample_slice.start,
                )
                for target_start in range(
                    target_slice.start, target_slice.stop, targets_per_block
                ):
                    target_stop = min(
                        target_start + targets_per_block, target_slice.stop
                    )
                    target_key = slice(target_start, target_stop)
                    local_target = slice(
                        target_start - target_slice.start,
                        target_stop - target_slice.start,
                    )
                    if np.asarray(coverage[(sample_key, target_key)]).any():
                        raise ValueError("observable block coverage overlaps")
                    slab = block.values[local_sample, local_target]
                    if axis_kind == "frequency":
                        values[(target_key, sample_key)] = np.swapaxes(slab, 0, 1)
                    else:
                        values[(sample_key, target_key)] = slab
                    coverage[(sample_key, target_key)] = True

        place(first)
        for block in iterator:
            place(block)
        h5.flush()

        def scan_readback(open_h5: Any) -> str:
            read_values = open_h5["values"]
            read_coverage = open_h5["coverage"]
            digest = hashlib.sha256()
            trailing = tuple(slice(0, width) for width in public_shape[2:])
            for sample_start in range(0, samples, samples_per_block):
                sample_stop = min(sample_start + samples_per_block, samples)
                sample_key = slice(sample_start, sample_stop)
                for target_start in range(0, targets, targets_per_block):
                    target_stop = min(target_start + targets_per_block, targets)
                    target_key = slice(target_start, target_stop)
                    covered = np.asarray(read_coverage[(sample_key, target_key)])
                    if not covered.all():
                        missing = np.argwhere(~covered)[0]
                        raise ValueError(
                            "observable block coverage is missing "
                            f"sample {sample_start + int(missing[0])}, "
                            f"target {target_start + int(missing[1])}"
                        )
                    key = (
                        (target_key, sample_key) + trailing
                        if axis_kind == "frequency"
                        else (sample_key, target_key) + trailing
                    )
                    slab = np.asarray(read_values[key])
                    if not np.isfinite(slab).all():
                        raise ValueError(
                            "observable readback contains non-finite values"
                        )
                    _hash_slab(digest, slab)
            return digest.hexdigest()

        expected_readback = scan_readback(h5)
        h5.flush()
    with h5py.File(partial_path, "r") as h5:
        if (
            _decode(h5.attrs.get("profile", "")) != "gragra.observable-series"
            or not _is_literal_integer(h5.attrs.get("version"), 1)
            or not _is_literal_boolean(h5.attrs.get("complete"))
            or bool(h5.attrs["complete"])
            or _decode(h5.attrs.get("observable", "")) != observable
            or _decode(h5.attrs.get("axis_kind", "")) != axis_kind
        ):
            raise OSError("observable output metadata failed readback")
        axis_dataset = h5.get("axis_values")
        targets_dataset = h5.get("targets_m")
        origin_dataset = h5.get("coordinate_origin_m")
        phase_dataset = h5.get("phase_origin_m")
        metadata_matches = (
            _decode(h5.attrs.get("coordinate_frame", "")) == coordinate_frame
            and axis_dataset is not None
            and _decode(axis_dataset.attrs.get("unit", ""))
            == ("Hz" if axis_kind == "frequency" else "s")
            and _dataset_equals_array_in_slabs(
                axis_dataset, axis, block_size=samples_per_block
            )
            and targets_dataset is not None
            and _decode(targets_dataset.attrs.get("unit", "")) == "m"
            and _dataset_equals_array_in_slabs(
                targets_dataset, target_positions, block_size=targets_per_block
            )
            and origin_dataset is not None
            and _decode(origin_dataset.attrs.get("unit", "")) == "m"
            and _dataset_equals_array_in_slabs(origin_dataset, origin, block_size=3)
        )
        if axis_kind == "frequency":
            metadata_matches = metadata_matches and (
                h5.attrs.get("t_ref_s") == t_ref
                and _decode(h5.attrs.get("fourier_convention", ""))
                == FOURIER_CONVENTION
                and phase_dataset is not None
                and _decode(phase_dataset.attrs.get("unit", "")) == "m"
                and _dataset_equals_array_in_slabs(phase_dataset, phase, block_size=3)
            )
        else:
            metadata_matches = metadata_matches and (
                "t_ref_s" not in h5.attrs
                and "fourier_convention" not in h5.attrs
                and phase_dataset is None
            )
        if not metadata_matches:
            raise OSError("observable metadata differs from original input")
        values = h5["values"]
        coverage = h5["coverage"]
        if (
            values.shape != public_shape
            or np.dtype(values.dtype) != expected_dtype
            or _decode(values.attrs.get("unit", "")) != _OBSERVABLE_UNITS[observable]
            or coverage.shape != (samples, targets)
            or np.dtype(coverage.dtype) != np.dtype(np.bool_)
        ):
            raise OSError("observable output datasets failed readback")
        try:
            readback_hex = scan_readback(h5)
        except ValueError as error:
            raise OSError("observable output failed reopened readback") from error
        if readback_hex != expected_readback:
            raise OSError("observable output differs on reopened readback")
    with h5py.File(partial_path, "r+") as h5:
        h5.attrs["readback_sha256"] = readback_hex
        h5.attrs.modify("complete", True)
        h5.flush()
    if final_path.exists():
        raise FileExistsError(f"refusing to overwrite {final_path}")
    os.link(partial_path, final_path)
    partial_path.unlink()
    return final_path
