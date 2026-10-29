"""Eager reader for the canonical SPECFEM3D Cartesian GLL HDF5 schema.

All schema, provenance, cadence, mesh, material, PML, and numeric acceptance
checks live in :mod:`gragra.adapters.specfem._validation`. This module only
materializes an already accepted file into the historical immutable API.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gragra.adapters.specfem import schema
from gragra.adapters.specfem._validation import validate_specfem_h5


@dataclass(frozen=True)
class SpecfemGLLFileIdentity:
    """Stable identity facts captured when a GLL HDF5 file is read."""

    path: Path
    device: int
    inode: int
    size_bytes: int
    mtime_ns: int


@dataclass(frozen=True)
class SpecfemGLLBlock:
    """One contiguous element-id block from :class:`SpecfemGLLDataset`."""

    element_id: np.ndarray
    coordinates_m: np.ndarray
    mass_kg: np.ndarray
    displacement_m: np.ndarray | None


def _readonly(array: np.ndarray) -> np.ndarray:
    copied = np.asarray(array).copy()
    copied.setflags(write=False)
    return copied


def _import_h5py():
    try:
        import h5py
    except ImportError as error:
        raise ImportError(
            "h5py is required for SPECFEM GLL HDF5 input. "
            'Install it with: pip install "gragra[field-io]"'
        ) from error
    return h5py


def _materialize(dataset) -> np.ndarray:
    """Materialize one accepted dataset without maintaining validation here."""
    result = np.empty(dataset.shape, dtype=dataset.dtype)
    dataset.read_direct(result)
    return result


@dataclass(frozen=True)
class SpecfemGLLDataset:
    """Closed immutable canonical GLL data, ordered by element then GLL.

    ``mass_kg`` is reconstructed as density times quadrature volume. Unlike
    the context-managed streaming adapter, this historical API intentionally
    holds every accepted sample in memory after the HDF5 file is closed.
    """

    file_identity: SpecfemGLLFileIdentity
    time_s: np.ndarray
    time_step: np.ndarray
    element_id: np.ndarray
    coordinates_m: np.ndarray
    quadrature_volume_m3: np.ndarray
    density_kg_m3: np.ndarray
    displacement_m: np.ndarray | None

    def iter_blocks(self, *, element_block_size: int | None = None):
        """Yield contiguous element-id blocks without merging shared GLL points."""
        n_elements = self.element_id.size
        if element_block_size is None:
            block_size = max(1, n_elements)
        else:
            if isinstance(element_block_size, (bool, np.bool_)):
                raise ValueError("element_block_size must be a positive integer")
            try:
                block_size = operator.index(element_block_size)
            except TypeError as error:
                raise ValueError(
                    "element_block_size must be a positive integer"
                ) from error
            if block_size <= 0:
                raise ValueError("element_block_size must be a positive integer")

        for start in range(0, n_elements, block_size):
            stop = min(start + block_size, n_elements)
            mass = _readonly(
                self.density_kg_m3[start:stop] * self.quadrature_volume_m3[start:stop]
            )
            displacement = self.displacement_m
            if displacement is not None:
                displacement = displacement[:, start:stop]
            yield SpecfemGLLBlock(
                element_id=self.element_id[start:stop],
                coordinates_m=self.coordinates_m[start:stop],
                mass_kg=mass,
                displacement_m=displacement,
            )


def read_specfem_gll_hdf5(
    path: str | Path,
    *,
    profile_density_range: tuple[float, float] | None,
) -> SpecfemGLLDataset:
    """Read an accepted canonical file into the historical eager data model."""
    h5py = _import_h5py()
    resolved_path = Path(path).expanduser().resolve()

    with h5py.File(resolved_path, "r") as h5:
        layout = validate_specfem_h5(
            h5,
            h5py,
            profile_density_range=profile_density_range,
        )
        coordinates = _materialize(h5[schema.PATH_MESH_COORDINATES])
        quadrature_volume = _materialize(h5[schema.PATH_MESH_QUADRATURE_VOLUME])
        density = _materialize(h5[schema.PATH_MATERIAL_DENSITY])
        displacement = _materialize(h5[schema.PATH_FIELD_DISPLACEMENT])

    stat = resolved_path.stat()
    return SpecfemGLLDataset(
        file_identity=SpecfemGLLFileIdentity(
            path=resolved_path,
            device=stat.st_dev,
            inode=stat.st_ino,
            size_bytes=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
        ),
        time_s=_readonly(layout.time_s),
        time_step=_readonly(layout.time_step),
        element_id=_readonly(layout.element_id),
        coordinates_m=_readonly(coordinates),
        quadrature_volume_m3=_readonly(quadrature_volume),
        density_kg_m3=_readonly(density),
        displacement_m=_readonly(displacement),
    )
