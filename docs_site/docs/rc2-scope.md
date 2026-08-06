# rc2 scope

This release candidate documents one HDF5 field path: a canonical schema-v1,
single-file, time-sampled SPECFEM3D Cartesian input with ENU coordinates, one
elastic domain, one material, and no PML elements. It also documents the small
point-source API used in the installation smoke check.

The reader is fail-closed for machine-checkable schema-v1 conditions. It rejects
an input unless `field_units.displacement` is `"m"`. External mesh,
`xdecompose_mesh`, `NGNOD=8`, standard NGLL, physical GLL flattening order, and
the SI meaning of coordinates, density, and quadrature volume remain caller or
exporter assumptions. It does not accept PML data by discarding it, merge
duplicate GLL coordinates, stream snapshots from HDF5, or determine whether a
target lies in rock.

PyPI distribution, GitHub Pages deployment, C++ extension wheel distribution,
CuPy support, performance guarantees, and real-solver-output backend parity are
not part of this rc2 documentation scope.
