"""SPECFEM3D Cartesian certified exporter subpackage (gragra.adapters.specfem).

This subpackage implements the SPECFEM3D Cartesian certified GLL export and
rc1 reader paths.  The exporter reads a SPECFEM3D run tree and writes
gragra's canonical HDF5 layout (``mesh_type="gll"``).  The reader loads the
same closed HDF5 layout into immutable GLL blocks for the acceleration path.

"Certified" here means **self-certified by the gragra maintainer against
gragra's own written contract for a named solver profile**. It is not a
third-party audit, not accreditation, and not an endorsement by the
SPECFEM3D project or any other body.

Module layout:

- ``schema.py`` -- canonical schema single source of truth (numpy-only,
  h5py-free): dataset paths/dtypes, provenance key tables, GLL flatten
  convention, canonical JSON encoding, hash formatting.
- ``specfem_bin.py`` -- Fortran unformatted binary reading, proc-file
  discovery, Database parsing.
- ``gll.py`` -- GLL point/weight generation, trilinear Jacobian,
  quadrature volume.
- ``provenance.py`` -- Par_file/hash/manifest/build-identity/runtime-manifest/
  cadence provenance collection.
- ``writer.py`` -- h5py-lazy-import dataset writing, metadata encoding,
  the 4 scalar invariants.
- ``export.py`` -- ``ExportConfig`` (frozen dataclass), profile gate,
  orchestration, self-check.
- ``cli.py`` / ``__main__.py`` -- export entry point + ``capture-run``
  subcommand.

Public re-exports: ``export_specfem_gll_hdf5``/``ExportConfig`` from
``export.py`` and ``read_specfem_gll_hdf5``/``SpecfemGLLDataset`` from
``reader.py``. ``writer.py``/``cli.py`` are not re-exported here -- they are
internal to the export pipeline (``writer.py``'s h5py-lazy-import functions
are not part of the public surface; ``cli.py``'s entry point is
reached via ``python -m gragra.adapters.specfem``, not an import).

Core purity: this subpackage (like the rest of ``gragra.adapters``) must not
import ``h5py`` at module import time. Only ``schema.py`` is guaranteed
numpy-only; other modules in this subpackage perform lazy/deferred imports of
optional dependencies (h5py) inside functions, following the
``adapters/hdf5_field.py`` convention. Importing this ``__init__`` (which
pulls in ``export.py``/``writer.py`` at module scope) must therefore also
never load h5py as a side effect -- see
``tests/test_specfem_exporter.py``'s core-purity subprocess check.
"""

from gragra.adapters.specfem.export import ExportConfig, export_specfem_gll_hdf5
from gragra.adapters.specfem.reader import SpecfemGLLDataset, read_specfem_gll_hdf5

__all__ = [
    "ExportConfig",
    "SpecfemGLLDataset",
    "export_specfem_gll_hdf5",
    "read_specfem_gll_hdf5",
]
