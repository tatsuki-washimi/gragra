#!/usr/bin/env python3
"""Bounded raw-to-canonical export for the approved SPECFEM domain pair."""

from __future__ import annotations

import argparse
import contextlib
import errno
import fcntl
import hashlib
import json
import os
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from gragra.adapters.specfem import export, gll, provenance, schema, specfem_bin, writer
from scripts.research_wave4_specfem_protocol import (
    CASE_NAMES,
    DEFAULT_RUNS_ROOT,
    NSTEP,
    SOLVER_COMMIT,
    SOLVER_ROOT,
    accept_solver_run,
    validate_matched_actual_dt,
    validate_pinned_solver,
)


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _readback_displacement_identity(
    partial: Path,
    *,
    local_path: Path,
    nproc: int,
    native_steps: tuple[int, ...],
    all_ranks: Any,
    ngll: int,
    source_block_elements: int,
) -> dict[str, Any]:
    """Compare every reopened HDF5 slab exactly with its bounded raw source."""
    import h5py

    expected_digest = hashlib.sha256()
    observed_digest = hashlib.sha256()
    readback_blocks = 0
    expected_frame = np.empty((len(all_ranks.element_id), ngll, 3), dtype=np.float64)
    with h5py.File(partial, "r") as h5:
        field = h5[schema.PATH_FIELD_DISPLACEMENT]
        for sample_index, native_step in enumerate(native_steps):
            for rank in range(nproc):
                element_ids = all_ranks.element_id_by_rank[rank]
                values_rank, observed_steps, _layout = (
                    specfem_bin.read_rank_displacement_snapshots(
                        local_path,
                        rank,
                        nspec_rank=len(element_ids),
                        ngll=ngll,
                        glob_pattern=(
                            f"proc{rank:06d}_displ_*_it{native_step:06d}.bin"
                        ),
                    )
                )
                if observed_steps != [native_step]:
                    raise ValueError(
                        f"rank {rank} native step {native_step} readback mismatch"
                    )
                values_rank = values_rank[0]
                if not np.isfinite(values_rank).all():
                    raise ValueError(
                        f"raw readback rank {rank} native step {native_step} "
                        "contains non-finite values"
                    )
                expected_frame[element_ids] = values_rank
                del values_rank
            for start in range(0, len(expected_frame), source_block_elements):
                stop = min(start + source_block_elements, len(expected_frame))
                expected = expected_frame[start:stop]
                observed = np.asarray(field[sample_index, start:stop, :, :])
                label = f"{sample_index}:{native_step}:{start}:{stop}"
                expected_hash = _array_sha256(expected)
                observed_hash = _array_sha256(observed)
                expected_digest.update(label.encode("ascii"))
                expected_digest.update(expected_hash.encode("ascii"))
                observed_digest.update(label.encode("ascii"))
                observed_digest.update(observed_hash.encode("ascii"))
                if not np.isfinite(observed).all():
                    raise ValueError(
                        "canonical readback is non-finite at "
                        f"sample {sample_index}, global block {start}"
                    )
                expected_bytes = np.ascontiguousarray(expected).view(np.uint8)
                observed_bytes = np.ascontiguousarray(observed).view(np.uint8)
                if not np.array_equal(observed_bytes, expected_bytes):
                    differing_byte = int(
                        np.flatnonzero(observed_bytes != expected_bytes)[0]
                    )
                    first = tuple(
                        int(value)
                        for value in np.unravel_index(
                            differing_byte // expected.dtype.itemsize, expected.shape
                        )
                    )
                    max_abs = float(np.max(np.abs(observed - expected)))
                    raise ValueError(
                        "canonical readback identity mismatch at "
                        f"sample {sample_index}, native step {native_step}, "
                        f"global block {start}, block index {first}; "
                        f"first_differing_byte={differing_byte}, "
                        f"max_abs_difference={max_abs!r}"
                    )
                readback_blocks += 1
    expected_hex = expected_digest.hexdigest()
    observed_hex = observed_digest.hexdigest()
    if expected_hex != observed_hex:
        raise ValueError("canonical readback aggregate identity digest mismatch")
    return {
        "readback_blocks": readback_blocks,
        "expected_identity_sha256": expected_hex,
        "observed_identity_sha256": observed_hex,
        "identity_comparison": "exact contiguous-byte identity against raw slabs",
        "readback_strategy": (
            "one_float64_raw_sample_frame_then_contiguous_global_element_slabs"
        ),
    }


def _publish_no_overwrite(partial: Path, output: Path) -> None:
    """Atomically publish by hard link while refusing a racing final path."""
    os.link(partial, output)
    partial.unlink()


def _clone_file_exclusive(source: Path, partial: Path) -> str:
    """Clone a file without overwriting, with a bounded-copy fallback."""
    source_fd = os.open(source, os.O_RDONLY)
    partial_fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        try:
            fcntl.ioctl(partial_fd, 0x40049409, source_fd)  # Linux FICLONE
            strategy = "linux_ficlone"
        except OSError as error:
            if error.errno not in {
                errno.EINVAL,
                errno.ENOTTY,
                errno.EOPNOTSUPP,
                errno.EXDEV,
            }:
                raise
            os.lseek(source_fd, 0, os.SEEK_SET)
            os.ftruncate(partial_fd, 0)
            while block := os.read(source_fd, 8 * 1024 * 1024):
                view = memoryview(block)
                while view:
                    written = os.write(partial_fd, view)
                    view = view[written:]
            strategy = "bounded_8mib_copy"
        os.fsync(partial_fd)
        return strategy
    finally:
        os.close(partial_fd)
        os.close(source_fd)


def _attribute_bitwise_equal(left: Any, right: Any) -> bool:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.dtype != right_array.dtype or left_array.shape != right_array.shape:
        return False
    if left_array.dtype.hasobject or left_array.dtype.kind in "SU":
        return bool(np.array_equal(left_array, right_array))
    return bool(
        np.array_equal(
            np.ascontiguousarray(left_array).view(np.uint8),
            np.ascontiguousarray(right_array).view(np.uint8),
        )
    )


def _unlink_owned_link(path: Path, identity: tuple[int, int]) -> None:
    """Remove a prematurely published link only when it is still ours."""
    try:
        observed = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return
    if (observed.st_dev, observed.st_ino) == identity:
        path.unlink()


def _publish_cadence_correction_no_overwrite(
    partial: Path,
    output: Path,
    report_partial: Path,
    report_path: Path,
) -> None:
    """Reserve the success report before final publication."""
    report_stat = report_partial.stat(follow_symlinks=False)
    report_identity = (report_stat.st_dev, report_stat.st_ino)
    os.link(report_partial, report_path)
    try:
        os.link(partial, output)
    except BaseException:
        _unlink_owned_link(report_path, report_identity)
        raise
    for staging_link in (partial, report_partial):
        with contextlib.suppress(OSError):
            staging_link.unlink()


def _compare_canonical_dataset_bytes(
    source: Path,
    corrected: Path,
    *,
    data_block_elements: int,
) -> dict[str, Any]:
    """Compare every logical dataset byte, using raw bytes for HDF5 chunks."""
    import h5py

    source_digest = hashlib.sha256()
    corrected_digest = hashlib.sha256()
    chunk_count = 0
    slab_count = 0
    with h5py.File(source, "r") as left_h5, h5py.File(corrected, "r") as right_h5:
        left_paths: list[str] = []
        right_paths: list[str] = []
        left_h5.visit(left_paths.append)
        right_h5.visit(right_paths.append)
        if left_paths != right_paths:
            raise ValueError("cadence correction changed the HDF5 object path set")
        for relative_path in left_paths:
            left_object = left_h5[relative_path]
            right_object = right_h5[relative_path]
            if type(left_object) is not type(right_object):
                raise ValueError(
                    f"cadence correction changed object type at /{relative_path}"
                )
            if set(left_object.attrs) != set(right_object.attrs):
                raise ValueError(
                    f"cadence correction changed attribute keys at /{relative_path}"
                )
            for key in left_object.attrs:
                if relative_path == schema.METADATA_GROUP.lstrip("/") and key == (
                    "f_min_analysis"
                ):
                    continue
                if not _attribute_bitwise_equal(
                    left_object.attrs[key], right_object.attrs[key]
                ):
                    raise ValueError(
                        f"cadence correction changed attribute /{relative_path}:{key}"
                    )
            if not isinstance(left_object, h5py.Dataset):
                continue
            left = left_object
            right = right_object
            properties = (
                "shape",
                "dtype",
                "chunks",
                "compression",
                "compression_opts",
                "shuffle",
                "fletcher32",
            )
            if any(getattr(left, key) != getattr(right, key) for key in properties):
                raise ValueError(
                    f"cadence correction changed dataset layout at /{relative_path}"
                )
            label = f"/{relative_path}".encode()
            source_digest.update(label)
            corrected_digest.update(label)
            if left.chunks is not None:
                if left.id.get_num_chunks() != right.id.get_num_chunks():
                    raise ValueError(
                        f"cadence correction changed chunk count at /{relative_path}"
                    )
                for index in range(left.id.get_num_chunks()):
                    offset = left.id.get_chunk_info(index).chunk_offset
                    left_mask, left_bytes = left.id.read_direct_chunk(offset)
                    right_mask, right_bytes = right.id.read_direct_chunk(offset)
                    if left_mask != right_mask or left_bytes != right_bytes:
                        raise ValueError(
                            "cadence correction changed raw chunk bytes at "
                            f"/{relative_path}, offset {offset}"
                        )
                    source_digest.update(left_bytes)
                    corrected_digest.update(right_bytes)
                    chunk_count += 1
                continue
            if left.ndim == 0:
                slices = ((),)
            else:
                slices = tuple(
                    (slice(start, min(start + data_block_elements, left.shape[0])),)
                    + (slice(None),) * (left.ndim - 1)
                    for start in range(0, left.shape[0], data_block_elements)
                )
            for key in slices:
                left_value = np.asarray(left[key])
                right_value = np.asarray(right[key])
                if not _attribute_bitwise_equal(left_value, right_value):
                    raise ValueError(
                        f"cadence correction changed data bytes at /{relative_path}"
                    )
                if left_value.dtype.hasobject or left_value.dtype.kind in "SU":
                    payload = repr(left_value.tolist()).encode("utf-8")
                    source_digest.update(payload)
                    corrected_digest.update(payload)
                else:
                    left_bytes = np.ascontiguousarray(left_value).tobytes()
                    right_bytes = np.ascontiguousarray(right_value).tobytes()
                    source_digest.update(left_bytes)
                    corrected_digest.update(right_bytes)
                slab_count += 1
    if source_digest.digest() != corrected_digest.digest():
        raise ValueError("cadence correction aggregate dataset digest mismatch")
    return {
        "all_datasets_bitwise_equal": True,
        "dataset_identity_sha256": source_digest.hexdigest(),
        "raw_chunks_compared": chunk_count,
        "contiguous_dataset_slabs_compared": slab_count,
    }


def correct_canonical_cadence_no_overwrite(
    source: str | Path,
    output: str | Path,
    *,
    expected_source_f_min: float = 2.5,
    corrected_f_min: float = 2.4,
    expected_source_sha256: str | None = None,
    data_block_elements: int = 512,
) -> dict[str, Any]:
    """Derive a cadence-valid canonical while preserving every dataset byte."""
    import h5py

    source = Path(source).resolve()
    output = Path(output).resolve()
    partial = output.with_suffix(output.suffix + ".partial")
    failure = output.with_suffix(output.suffix + ".partial.failure.json")
    report_path = output.with_suffix(output.suffix + ".cadence-correction.json")
    report_partial = report_path.with_suffix(report_path.suffix + ".partial")
    if data_block_elements <= 0:
        raise ValueError("data_block_elements must be positive")
    if source == output:
        raise ValueError("cadence correction source and output must differ")
    existing = [
        path
        for path in (output, partial, failure, report_path, report_partial)
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "refusing to overwrite cadence correction artifact(s): "
            + ", ".join(str(path) for path in existing)
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        source_sha256 = _file_sha256(source)
        if (
            expected_source_sha256 is not None
            and source_sha256 != expected_source_sha256
        ):
            raise ValueError("cadence correction source SHA256 does not match")
        clone_strategy = _clone_file_exclusive(source, partial)
        pre_correction_clone_sha256 = _file_sha256(partial)
        if pre_correction_clone_sha256 != source_sha256 or (
            expected_source_sha256 is not None
            and pre_correction_clone_sha256 != expected_source_sha256
        ):
            raise ValueError(
                "pre-correction clone SHA256 does not match the pre-read source"
            )
        with h5py.File(partial, "r+") as h5:
            metadata = h5[schema.METADATA_GROUP].attrs
            source_f_min = float(metadata["f_min_analysis"])
            f_max = float(metadata["f_max_analysis"])
            if source_f_min != expected_source_f_min:
                raise ValueError("canonical source f_min_analysis is unexpected")
            if not np.isfinite(corrected_f_min) or not 0.0 < corrected_f_min < f_max:
                raise ValueError(
                    "corrected f_min_analysis must be finite, positive, and below f_max"
                )
            time_step = np.asarray(h5[schema.PATH_TIME_STEP][()])
            duration = float(time_step[-1] - time_step[0]) * float(metadata["dt_s"])
            required_duration = float(metadata["minimum_cycles"]) / corrected_f_min
            if duration < required_duration:
                raise ValueError(
                    "corrected f_min_analysis does not satisfy saved duration"
                )
            metadata.modify("f_min_analysis", np.float64(corrected_f_min))
            h5.flush()
        identity = _compare_canonical_dataset_bytes(
            source, partial, data_block_elements=data_block_elements
        )
        with h5py.File(partial, "r") as h5:
            observed = float(h5[schema.METADATA_GROUP].attrs["f_min_analysis"])
            if observed != corrected_f_min:
                raise ValueError("corrected cadence metadata readback mismatch")
        output_sha256 = _file_sha256(partial)
        report = {
            "classification": "Observed canonical cadence metadata correction",
            "status": "CANONICAL_CADENCE_CORRECTION_COMPLETE",
            "source_path": str(source),
            "source_size_bytes": source.stat().st_size,
            "source_sha256": source_sha256,
            "pre_correction_clone_sha256": pre_correction_clone_sha256,
            "output_path": str(output),
            "output_size_bytes": partial.stat().st_size,
            "output_sha256": output_sha256,
            "source_f_min_analysis": source_f_min,
            "corrected_f_min_analysis": corrected_f_min,
            "unchanged_f_max_analysis": f_max,
            "saved_duration_s": duration,
            "minimum_required_duration_s": required_duration,
            "clone_strategy": clone_strategy,
            "correction_scope": "metadata f_min_analysis only",
            **identity,
            "source_retained": True,
            "wall_seconds": time.monotonic() - started,
        }
        payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        with report_partial.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if report_partial.read_text(encoding="utf-8") != payload:
            raise ValueError("cadence correction success report readback mismatch")
        _publish_cadence_correction_no_overwrite(
            partial, output, report_partial, report_path
        )
        return report
    except BaseException as error:
        if not failure.exists():
            with failure.open("x", encoding="utf-8") as stream:
                json.dump(
                    {
                        "classification": (
                            "Observed incomplete canonical cadence correction"
                        ),
                        "status": "CANONICAL_CADENCE_CORRECTION_INCOMPLETE_RETAINED",
                        "source_path": str(source),
                        "output_partial": str(partial),
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "retention": "source and incomplete correction retained",
                    },
                    stream,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                stream.write("\n")
        raise


def _database_element_corners(database: Any) -> np.ndarray:
    """Return rank-local Database corners without loading field snapshots."""
    node_xyz = np.empty((database.nnodes_loc, 3), dtype=np.float64)
    for index in range(database.nnodes_loc):
        node_xyz[int(database.node_iloc[index]) - 1] = database.node_xyz[index]
    return np.asarray(
        [
            node_xyz[[node_id - 1 for node_id in element.node_loc_ids]]
            for element in database.elements
        ],
        dtype=np.float64,
    )


def _structured_element_ids(
    corners: np.ndarray,
    *,
    length_m: int,
    spacing_m: int = 250,
) -> np.ndarray:
    """Map exact protocol-grid corners to x-fastest pre-partition IDs."""
    if length_m <= 0 or length_m % spacing_m:
        raise ValueError("length_m must be a positive multiple of spacing_m")
    expected_shape = (length_m // spacing_m) ** 2 * 12
    if corners.ndim != 3 or corners.shape[1:] != (8, 3):
        raise ValueError(
            f"expected Database corners shaped (n, 8, 3), got {corners.shape}"
        )
    minima = np.min(corners, axis=1)
    maxima = np.max(corners, axis=1)
    coordinate_tolerance_m = 1.0e-8
    if not np.allclose(
        maxima - minima,
        np.full_like(minima, spacing_m),
        rtol=0.0,
        atol=coordinate_tolerance_m,
    ):
        raise ValueError("Database contains a non-250 m protocol element")
    xmin = 3000 - length_m // 2
    grid_origin = np.asarray((xmin, xmin, -3000), dtype=np.float64)
    grid_float = (minima - grid_origin) / float(spacing_m)
    grid = np.rint(grid_float).astype(np.int64)
    if not np.allclose(
        grid_float,
        grid.astype(np.float64),
        rtol=0.0,
        atol=coordinate_tolerance_m / spacing_m,
    ):
        raise ValueError("Database element minimum is not on the exact protocol grid")
    nx = length_m // spacing_m
    if (
        np.any(grid[:, 0] < 0)
        or np.any(grid[:, 0] >= nx)
        or np.any(grid[:, 1] < 0)
        or np.any(grid[:, 1] >= nx)
        or np.any(grid[:, 2] < 0)
        or np.any(grid[:, 2] >= 12)
    ):
        raise ValueError("Database element lies outside the fixed protocol bounds")
    element_ids = grid[:, 0] + nx * grid[:, 1] + nx * nx * grid[:, 2]
    if np.any(element_ids >= expected_shape):
        raise AssertionError("structured element ID exceeded expected range")
    return element_ids


def prepare_structured_part_array(
    local_path: str | Path,
    *,
    length_m: int,
    nproc: int = 4,
    attempt_label: str | None = None,
) -> dict[str, Any]:
    """Create the internal-mesher partition map with validated ordering.

    ``xmeshfem3D`` does not emit the external-mesher ``part_array.vtk`` that
    the importer consumes. The approved meshes are exact Cartesian 250 m
    grids, so their pre-partition ordering is reconstructed as x-fastest,
    then y, then z. Every rank Database and the complete ownership partition
    are checked before an exclusively published VTK is accepted.
    """
    local_path = Path(local_path).resolve()
    output = local_path / "part_array.vtk"
    if attempt_label is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", attempt_label):
        raise ValueError("attempt_label must contain only letters, digits, _ or -")
    attempt_suffix = "" if attempt_label is None else f".{attempt_label}"
    incomplete = local_path / f"part_array.vtk{attempt_suffix}.incomplete"
    failure = local_path / f"part_array.vtk{attempt_suffix}.incomplete.failure.json"
    report_path = local_path / f"part_array.vtk.generation{attempt_suffix}.json"
    existing = [
        path for path in (output, incomplete, failure, report_path) if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing partition-map artifact(s): "
            + ", ".join(str(path) for path in existing)
        )
    nx = length_m // 250
    expected_elements = nx * nx * 12
    owner = np.full(expected_elements, -1, dtype=np.int64)
    local_counts: list[int] = []
    try:
        for rank in range(nproc):
            database = specfem_bin.parse_database(
                local_path / f"proc{rank:06d}_Database"
            )
            ids = _structured_element_ids(
                _database_element_corners(database), length_m=length_m
            )
            if ids.size == 0 or not np.all(ids[1:] > ids[:-1]):
                raise ValueError(
                    f"rank {rank} Database elements are not strictly increasing "
                    "in protocol global order"
                )
            if np.any(owner[ids] != -1):
                raise ValueError(f"rank {rank} overlaps previously assigned elements")
            owner[ids] = rank
            local_counts.append(int(ids.size))
        missing = np.flatnonzero(owner < 0)
        if missing.size:
            raise ValueError(
                f"partition map leaves {missing.size} elements unassigned; "
                f"first global ID {int(missing[0])}"
            )
        if local_counts != [expected_elements // nproc] * nproc:
            raise ValueError(
                f"rank element counts {local_counts} do not match "
                "equal protocol partition"
            )

        xmin = 3000 - length_m // 2
        ny = nx
        nz = 12
        npx = nx + 1
        npy = ny + 1

        def point_id(ix: int, iy: int, iz: int) -> int:
            return ix + npx * iy + npx * npy * iz

        with incomplete.open("x", encoding="ascii") as stream:
            stream.write(
                "# vtk DataFile Version 3.1\n"
                f"gragra protocol structured {length_m} m partition map\n"
                "ASCII\nDATASET UNSTRUCTURED_GRID\n"
                f"POINTS {npx * npy * (nz + 1)} float\n"
            )
            for iz in range(nz + 1):
                z = -3000 + 250 * iz
                for iy in range(npy):
                    y = xmin + 250 * iy
                    for ix in range(npx):
                        x = xmin + 250 * ix
                        stream.write(f"{x} {y} {z}\n")
            stream.write(f"CELLS {expected_elements} {9 * expected_elements}\n")
            for iz in range(nz):
                for iy in range(ny):
                    for ix in range(nx):
                        corners = (
                            point_id(ix, iy, iz),
                            point_id(ix + 1, iy, iz),
                            point_id(ix + 1, iy + 1, iz),
                            point_id(ix, iy + 1, iz),
                            point_id(ix, iy, iz + 1),
                            point_id(ix + 1, iy, iz + 1),
                            point_id(ix + 1, iy + 1, iz + 1),
                            point_id(ix, iy + 1, iz + 1),
                        )
                        stream.write("8 " + " ".join(map(str, corners)) + "\n")
            stream.write(f"CELL_TYPES {expected_elements}\n")
            for _ in range(expected_elements):
                stream.write("12\n")
            stream.write(
                f"CELL_DATA {expected_elements}\n"
                "SCALARS elem_flag int 1\nLOOKUP_TABLE default\n"
            )
            for rank in owner:
                stream.write(f"{int(rank)}\n")

        parsed = specfem_bin.read_part_array_vtk(incomplete)
        if (
            parsed.cells.shape != (expected_elements, 8)
            or parsed.points.shape != (npx * npy * (nz + 1), 3)
            or not np.array_equal(parsed.elem_flag, owner)
        ):
            raise ValueError("reopened partition map differs from generated structure")
        _publish_no_overwrite(incomplete, output)
        report = {
            "classification": (
                "Observed validated internal-mesher partition-map reconstruction"
            ),
            "status": "PART_ARRAY_COMPLETE",
            "output_path": str(output),
            "output_size_bytes": output.stat().st_size,
            "output_sha256": _file_sha256(output),
            "length_m": length_m,
            "spacing_m": 250,
            "shape_elements": [nx, ny, nz],
            "element_count": expected_elements,
            "point_count": int(parsed.points.shape[0]),
            "nproc": nproc,
            "rank_element_counts": local_counts,
            "element_id_order": "x_fastest_then_y_then_z_zero_based",
            "source": "fresh proc*_Database files from the same solver run",
            "coordinate_grid_validation_tolerance_m": 1.0e-8,
            "attempt_label": attempt_label,
        }
        with report_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
        return report
    except BaseException as error:
        if not failure.exists():
            with failure.open("x", encoding="utf-8") as stream:
                json.dump(
                    {
                        "classification": (
                            "Observed incomplete partition-map reconstruction"
                        ),
                        "status": "PART_ARRAY_INCOMPLETE_RETAINED",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "incomplete_path": str(incomplete),
                    },
                    stream,
                    indent=2,
                    sort_keys=True,
                )
                stream.write("\n")
        raise


def _snapshot_coverage(
    local_path: Path, nproc: int, expected_native_steps: tuple[int, ...]
) -> dict[int, dict[int, set[str]]]:
    pattern = re.compile(r"^proc(\d{6})_displ_([XYZ])_it(\d{6})\.bin$")
    observed: dict[int, dict[int, set[str]]] = {rank: {} for rank in range(nproc)}
    for path in local_path.glob("proc*_displ_*_it*.bin"):
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        rank, component, native_step = int(match[1]), match[2], int(match[3])
        if rank in observed:
            observed[rank].setdefault(native_step, set()).add(component)
    expected_steps = set(expected_native_steps)
    failures = []
    for rank in range(nproc):
        steps = set(observed[rank])
        if steps != expected_steps:
            failures.append(
                f"rank {rank} step set missing={sorted(expected_steps - steps)} "
                f"extra={sorted(steps - expected_steps)}"
            )
        for native_step in sorted(expected_steps & steps):
            components = observed[rank][native_step]
            if components != {"X", "Y", "Z"}:
                failures.append(
                    f"rank {rank} step {native_step} components "
                    f"missing={sorted({'X', 'Y', 'Z'} - components)}"
                )
    if failures:
        raise ValueError("snapshot coverage mismatch: " + "; ".join(failures[:8]))
    return observed


def _build_provenance(
    config: export.ExportConfig,
    values: dict[str, str],
    all_ranks: Any,
    is_pml: np.ndarray,
    *,
    t0: float,
    t0_source: str,
    actual_dt_s: float | None,
) -> dict[str, Any]:
    specfem_dir = config.specfem_dir
    if config.mesh_input_files:
        mesh_manifest, mesh_hash = provenance.compute_mesh_input_hash(
            specfem_dir,
            config.mesh_input_files,
            par_file_relpath=config.par_file_relpath,
        )
    else:
        mesh_manifest, mesh_hash = provenance.build_file_manifest(
            specfem_dir, [config.par_file_relpath]
        )
    parsed_par = provenance.parse_par_file(specfem_dir / config.par_file_relpath)
    build_identity = (
        dict(config.build_identity)
        if config.build_identity is not None
        else provenance.collect_build_identity(specfem_dir)
    )
    runtime_entries = [dict(value) for value in config.runtime_manifest_entries]
    runtime_entries.append(
        provenance.collect_exporter_identity(exporter_code_ref=config.exporter_code_ref)
    )
    runtime = provenance.build_runtime_manifest(runtime_entries)
    par_dt_s = float(
        provenance.par_file_value_as_number(export._require_par_value(values, "DT"))
    )
    dt_s = par_dt_s if actual_dt_s is None else float(actual_dt_s)
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("actual_dt_s must be finite and positive")
    nstep = int(
        provenance.par_file_value_as_number(export._require_par_value(values, "NSTEP"))
    )
    cadence = export._resolve_cadence(config, values)
    scalars: dict[str, Any] = {
        "solver_kind": config.solver_kind,
        "solver_version": config.solver_version,
        "nproc": len(all_ranks.element_id_by_rank),
        "float_precision": config.float_precision,
        "density_source": config.density_source,
        "element_id_method": schema.ADOPTED_ELEMENT_ID_METHOD,
        "f_resolved_estimate": export._resolve_f_resolved_estimate(config),
        "field_units": dict(config.field_units),
        "suppress_utm_projection": provenance.par_file_value_as_bool(
            export._require_par_value(values, "SUPPRESS_UTM_PROJECTION")
        ),
        "endianness": all_ranks.endianness,
        "record_marker_bytes": all_ranks.record_marker_bytes,
        "integer_kind": all_ranks.integer_kind,
        "material_id_source": config.material_id_source,
        "domain_id_source": config.domain_id_source,
        "is_pml_source": config.is_pml_source,
        "dt_s": dt_s,
        "par_file_dt_s": par_dt_s,
        "actual_dt_source": (
            "par_file" if actual_dt_s is None else "accepted_output_solver_log"
        ),
        "nstep": nstep,
        "displacement_layout": "per_rank",
        "cpml_enabled": False,
        "cpml_elements_excluded": 0,
        "scientific_cpml_exclusion_claim": False,
        "protocol_physical_time_rule": "10 + native_it * actual_dt_from_solver_log",
    }
    return provenance.assemble_provenance(
        output_sampling_kind="time_sampled",
        par_file=parsed_par,
        mesh_input_hash=mesh_hash,
        mesh_input_manifest=mesh_manifest,
        build_identity=build_identity,
        runtime_manifest=runtime,
        pml_mask_hash=provenance.pml_mask_hash(is_pml),
        # Accepted schema-v1 readers require this mechanical mask-application
        # flag.  The explicit fields above prevent it from being interpreted
        # as a scientific claim that CPML elements existed or were removed.
        pml_exclusion_applied=True,
        t0=t0,
        t0_source=t0_source,
        t0_crosscheck=config.t0_crosscheck,
        t0_crosscheck_max_deviation=config.t0_crosscheck_max_deviation,
        time_step_origin="zero_based_normalized",
        native_step_origin=1,
        cadence=cadence,
        cadence_provenance=config.cadence_provenance,
        **scalars,
    )


def export_bounded_specfem_gll_hdf5(
    config: export.ExportConfig,
    *,
    expected_native_steps: Sequence[int],
    source_block_elements: int = 512,
    actual_dt_s: float | None = None,
) -> dict[str, Any]:
    """Export using rank reads, one sample frame, and contiguous bounded writes."""
    if config.output_sampling_kind != "time_sampled":
        raise ValueError(
            "bounded protocol export requires output_sampling_kind='time_sampled'"
        )
    if source_block_elements <= 0:
        raise ValueError("source_block_elements must be positive")
    native_steps = tuple(int(value) for value in expected_native_steps)
    if not native_steps or tuple(sorted(set(native_steps))) != native_steps:
        raise ValueError(
            "expected_native_steps must be non-empty, unique, and increasing"
        )
    output = Path(config.output_path)
    partial = output.with_suffix(output.suffix + ".partial")
    failure_path = output.with_suffix(output.suffix + ".partial.failure.json")
    if output.exists() or partial.exists() or failure_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing export artifact for {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    partial.touch()
    started = time.monotonic()
    try:
        par_path = config.specfem_dir / config.par_file_relpath
        parsed_par = provenance.parse_par_file(par_path)
        values = parsed_par["values"]
        export._check_profile_inputs(values, "time_sampled")
        local_path = export._resolve_local_path(
            config.specfem_dir, values, config.local_path
        )
        nproc = export._detect_nproc(local_path)
        _snapshot_coverage(local_path, nproc, native_steps)
        ngll = schema.ngll_from_shape(config.ngll_shape)
        all_ranks = export._read_all_ranks(local_path, nproc, ngll)
        export._check_anchor_gll_bridge(
            all_ranks.corners8, all_ranks.coordinates, config.ngll_shape
        )
        corner_indices = export._usual_hex_nodes_flat_indices(config.ngll_shape)
        coordinates = all_ranks.coordinates.copy()
        coordinates[:, corner_indices, :] = all_ranks.corners8
        quadrature_volume = gll.elements_quadrature_volume(
            all_ranks.corners8, config.ngll_shape
        )
        is_pml = np.zeros(len(all_ranks.element_id), dtype=np.bool_)
        t0, t0_source = export._resolve_t0(config)
        prov = _build_provenance(
            config,
            values,
            all_ranks,
            is_pml,
            t0=t0,
            t0_source=t0_source,
            actual_dt_s=actual_dt_s,
        )

        # Reuse the canonical writer for complete metadata and mesh layout.
        # One temporary sample is bounded (<= 186.7 MiB for L18) and is
        # replaced before any raw snapshot is copied.
        seed = np.zeros((1, len(all_ranks.element_id), ngll, 3), dtype=np.float64)
        writer.write_gll_hdf5(
            partial,
            output_sampling_kind="time_sampled",
            ngll_shape=config.ngll_shape,
            coordinates=coordinates,
            quadrature_volume=quadrature_volume,
            element_id=all_ranks.element_id,
            partition_id=all_ranks.partition_id,
            domain_id=all_ranks.domain_id,
            material_id=all_ranks.material_id,
            is_pml=is_pml,
            density=all_ranks.density,
            provenance=prov,
            displacement=seed,
            time_step=np.asarray([native_steps[0] - 1], dtype=np.int64),
            chunk_element_size=source_block_elements,
            gzip_level=config.gzip_level,
        )
        del seed

        import h5py

        time_step = np.asarray(native_steps, dtype=np.int64) - 1
        time_values = time_step.astype(np.float64) * float(prov["dt_s"]) - t0
        ce = min(source_block_elements, len(all_ranks.element_id))
        with h5py.File(partial, "r+") as h5:
            for path in (
                schema.PATH_FIELD_DISPLACEMENT,
                schema.PATH_TIME,
                schema.PATH_TIME_STEP,
            ):
                del h5[path]
            field = h5.create_dataset(
                schema.PATH_FIELD_DISPLACEMENT,
                shape=(len(native_steps), len(all_ranks.element_id), ngll, 3),
                dtype=np.float64,
                chunks=(1, ce, ngll, 3),
                compression="gzip",
                compression_opts=config.gzip_level,
            )
            h5.create_dataset(schema.PATH_TIME, data=time_values)
            h5.create_dataset(schema.PATH_TIME_STEP, data=time_step)
            sample_frame = np.empty(
                (len(all_ranks.element_id), ngll, 3), dtype=np.float64
            )
            peak_rank_input_elements = 0
            writes = 0
            for sample_index, native_step in enumerate(native_steps):
                for rank in range(nproc):
                    element_ids = all_ranks.element_id_by_rank[rank]
                    values_rank, observed_steps, _layout = (
                        specfem_bin.read_rank_displacement_snapshots(
                            local_path,
                            rank,
                            nspec_rank=len(element_ids),
                            ngll=ngll,
                            glob_pattern=(
                                f"proc{rank:06d}_displ_*_it{native_step:06d}.bin"
                            ),
                        )
                    )
                    if observed_steps != [native_step]:
                        raise ValueError(
                            f"rank {rank} native step {native_step} read mismatch"
                        )
                    values_rank = values_rank[0]
                    if not np.isfinite(values_rank).all():
                        raise ValueError(
                            f"rank {rank} native step {native_step} "
                            "contains non-finite values"
                        )
                    peak_rank_input_elements = max(
                        peak_rank_input_elements, len(element_ids)
                    )
                    sample_frame[element_ids] = values_rank
                    del values_rank
                for start in range(0, len(sample_frame), source_block_elements):
                    stop = min(start + source_block_elements, len(sample_frame))
                    field[sample_index, start:stop, :, :] = sample_frame[start:stop]
                    writes += 1
            h5.flush()

        readback = _readback_displacement_identity(
            partial,
            local_path=local_path,
            nproc=nproc,
            native_steps=native_steps,
            all_ranks=all_ranks,
            ngll=ngll,
            source_block_elements=source_block_elements,
        )
        _publish_no_overwrite(partial, output)
        report = {
            "classification": "Observed bounded canonical export",
            "status": "CANONICAL_HDF5_COMPLETE",
            "source_case": str(config.specfem_dir.resolve()),
            "output_path": str(output.resolve()),
            "output_size_bytes": output.stat().st_size,
            "output_sha256": _file_sha256(output),
            "partition_map_sha256": _file_sha256(local_path / "part_array.vtk"),
            "samples_written": len(native_steps),
            "native_steps": [native_steps[0], native_steps[-1]],
            "source_block_elements": source_block_elements,
            "peak_input_elements": len(all_ranks.element_id),
            "sample_frame_elements": len(all_ranks.element_id),
            "sample_frame_bytes": int(sample_frame.nbytes),
            "peak_rank_input_elements": peak_rank_input_elements,
            "estimated_peak_displacement_input_bytes": int(
                sample_frame.nbytes
                + peak_rank_input_elements * ngll * 3 * np.dtype(np.float64).itemsize
            ),
            "write_strategy": (
                "one_float64_sample_frame_then_contiguous_global_element_slabs"
            ),
            "write_blocks": writes,
            **readback,
            "all_raw_retained": True,
            "cpml_enabled": False,
            "cpml_elements_excluded": 0,
            "scientific_cpml_exclusion_claim": False,
            "pml_exclusion_applied_schema_semantics": (
                "all-false schema mask mechanically applied; no CPML exclusion claim"
            ),
            "wall_seconds": time.monotonic() - started,
        }
        _json_write(output.with_suffix(output.suffix + ".export.json"), report)
        return report
    except BaseException as error:
        _json_write(
            failure_path,
            {
                "classification": "Observed incomplete export",
                "status": "CANONICAL_HDF5_INCOMPLETE_RETAINED",
                "output_partial": str(partial),
                "error_type": type(error).__name__,
                "error": str(error),
                "retention": "partial output and all raw input retained",
            },
        )
        raise


def _runtime_entries() -> tuple[dict[str, Any], ...]:
    return tuple(
        provenance.capture_executable(SOLVER_ROOT / "bin" / name, role)
        for name, role in (
            ("xmeshfem3D", "mesher"),
            ("xgenerate_databases", "database"),
            ("xspecfem3D", "solver"),
        )
    )


def _protocol_build_identity() -> dict[str, Any]:
    """Build source identity from the protocol's verified linked worktree."""
    validated = validate_pinned_solver()
    return {
        "build_kind": "source",
        "specfem_git_commit": validated["git_head"],
        "source_tree_clean": validated["tracked_files_clean"],
        "build_configuration": provenance._collect_build_configuration(SOLVER_ROOT),
        "compiler": validated["compiler"],
        "mpi": validated["mpi"],
    }


def protocol_export_config(case_root: Path, output_path: Path) -> export.ExportConfig:
    """Construct the fixed production export configuration after solver acceptance."""
    exporter_hash = _file_sha256(Path(__file__).resolve())
    return export.ExportConfig(
        specfem_dir=case_root,
        output_path=output_path,
        output_sampling_kind="time_sampled",
        solver_version=SOLVER_COMMIT,
        build_identity=_protocol_build_identity(),
        runtime_manifest_entries=_runtime_entries(),
        exporter_code_ref=f"sha256:{exporter_hash}",
        f_resolved_estimate=4.79999971,
        cadence={
            "f_max_analysis": 2.5,
            "f_min_analysis": 2.4,
            "minimum_cycles": 2.0,
            "sampling_oversampling_factor": 80.0,
            "antialias_filter_applied": False,
        },
        cadence_provenance="confirmed",
        chunk_element_size=512,
        gzip_level=4,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("length_m", type=int, choices=sorted(CASE_NAMES))
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument(
        "--output-name",
        default=None,
        help="fresh .h5 basename; defaults to specfem-L<L>.h5",
    )
    parser.add_argument(
        "--part-array-attempt",
        default=None,
        help="suffix for a fresh retained partition-map preparation attempt",
    )
    parser.add_argument(
        "--cadence-correction-source",
        type=Path,
        default=None,
        help="derive a cadence-v2 canonical from this retained canonical",
    )
    parser.add_argument(
        "--expected-source-sha256",
        default=None,
        help="required source identity for --cadence-correction-source",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    case_root = args.runs_root / CASE_NAMES[args.length_m]
    current_acceptance = accept_solver_run(case_root, args.length_m)
    other_length = 18_000 if args.length_m == 12_000 else 12_000
    other_path = args.runs_root / CASE_NAMES[other_length] / "solver-acceptance.json"
    other_acceptance = json.loads(other_path.read_text())
    matched = validate_matched_actual_dt(
        current_acceptance["log"], other_acceptance["log"]
    )
    output_name = args.output_name or f"specfem-L{args.length_m}.h5"
    if Path(output_name).name != output_name or not output_name.endswith(".h5"):
        raise ValueError("--output-name must be a plain .h5 basename")
    local_path = case_root / "OUTPUT_FILES" / "DATABASES_MPI_RUN"
    if not (local_path / "part_array.vtk").exists():
        prepare_structured_part_array(
            local_path,
            length_m=args.length_m,
            nproc=4,
            attempt_label=args.part_array_attempt,
        )
    output = case_root / "canonical" / output_name
    if args.cadence_correction_source is None:
        if args.expected_source_sha256 is not None:
            raise ValueError(
                "--expected-source-sha256 requires --cadence-correction-source"
            )
        report = export_bounded_specfem_gll_hdf5(
            protocol_export_config(case_root, output),
            expected_native_steps=range(1, NSTEP + 1),
            source_block_elements=512,
            actual_dt_s=matched["actual_dt_s"],
        )
    else:
        if args.expected_source_sha256 is None:
            raise ValueError(
                "--cadence-correction-source requires --expected-source-sha256"
            )
        report = correct_canonical_cadence_no_overwrite(
            args.cadence_correction_source,
            output,
            expected_source_f_min=2.5,
            corrected_f_min=2.4,
            expected_source_sha256=args.expected_source_sha256,
            data_block_elements=512,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    raise SystemExit(main())
