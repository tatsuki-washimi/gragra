# Verification and limitations

The public tests check the field calculation against an independent first-order
point-mass expression and a central-difference point-gravity calculation. They
also cover zero displacement, rigid translation, GLL ordering, duplicate
coordinates from separate elements, and invariance under supported chunk sizes.

The CI backend jobs compare NumPy with Numba and C++ only for a small temporary
HDF5 input, at `rtol=1e-13`. They do not establish performance characteristics
or parity for a solver output file.

CI creates temporary HDF5 files. This repository does not distribute persistent
fixtures, solver output, or a reproduction archive. Metadata and cadence
validation checks internal consistency; it is not a solver-validation or
sampling-certification procedure. The first-order perturbation requires the
caller to assess small displacement relative to each target distance; the reader
does not establish that physical condition.

The writer can store solver configuration text and resolved executable or
library paths in HDF5 provenance. Inspect provenance before sharing an HDF5 file
outside its original project.

Read [rc2 scope](rc2-scope.md) and
[SPECFEM3D Cartesian](specfem3d-cartesian.md) before applying the reader to an
external file.
