# Reproducibility and citation

## Citing gragra

A [`CITATION.cff`](https://github.com/tatsuki-washimi/gragra/blob/main/CITATION.cff)
file is included in the repository (Citation File Format 1.2.0). GitHub
renders it as a "Cite this repository" button on the repository page, and
tools such as `cffconvert` can turn it into BibTeX or other formats
directly from that file.

## Citing a checkout

For a reproducible result, cite the exact commit you installed from rather
than the version alone, for example:

```
gragra, commit <full 40-character SHA>,
https://github.com/tatsuki-washimi/gragra/tree/<SHA>
```

`git rev-parse HEAD` in your working copy gives you that SHA.

## Recording what you actually ran

Because the development tree can change independently of a version number,
reproducing a result later depends on recording more than that number:

- **The commit SHA** you installed from (see above).
- **The backend** used for the calculation (`"numpy"`, `"numba"`, `"cpp"` or
  `"cupy"`) — see [Compute backends](../guide/backends.md). Results are
  numerically equivalent across backends only within the tolerances the
  test suite asserts; for `cupy` specifically that equivalence is verified
  on the maintainer's local GPU only and is never run in CI (no GPU
  runner exists there). The `cupy` backend's chunk-size reproducibility
  guarantee is also weaker than the CPU backends' (see
  [Chunking, softening and tolerances](../concepts/numerics.md)).
- **Any external input's provenance** — for a SPECFEM3D-derived field, the
  provenance metadata written by the exporter (see
  [SPECFEM3D Cartesian](../external/specfem3d.md)); for an
  HDF5 field imported some other way, however you tracked its origin
  yourself, since gragra's importers do not require or read a provenance
  field on your behalf.
- **`chunk_size`** if you used the `cupy` backend, since it is an effective
  VRAM-tiling parameter there rather than a pure performance knob.

## Reproducibility caveats

Do not describe a result as agreeing to a tolerance tighter than what
[Verification](../verification/index.md) and the relevant page's
Verification Block actually assert. In particular, do not cite a SPECFEM3D
import beyond the documented profile and checks stated on
[SPECFEM3D Cartesian](../external/specfem3d.md); those checks do not cover
other solver configurations or establish domain convergence. The new
12/18 km acceleration pair completed its saved-vector arithmetic, but its
nested correspondence gate rejected the comparison.

## Prior and related software

Parts of gragra's Newtonian-noise kernels were independently re-derived
from, and their differences documented against, the predecessor research
codebase **nn-sus**. Where a docstring in gragra's source states a specific
formula difference (for example, the softening/collision guard term in
`gragra.kernels.inverse_square`) or a provenance commit and re-derivation
date (for example, in `gragra.analytic.halfspace` /
`gragra.waves.halfspace`), that docstring is the authoritative record —
this page does not duplicate those details.
