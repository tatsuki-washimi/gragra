# Stability and how to read this reference

This section is the generated reference: signatures, parameters, return
shapes and dtypes, extracted from the installed source. It is deliberately
thin on narrative. The pages under **Concepts** and **Guide** explain *why*
a function exists and how the pieces fit together; the pages here answer
*what exactly does it take and return*.

Read the numerical API reference alongside two conventions that are assumed
everywhere in those signatures themselves: component and tensor axes are the
trailing axes of numerical arrays, and a name ending in `_m`, `_kg`, `_m_s` or
`_rad_m` states its SI unit. Those numerical conventions are set out in
[Arrays, dtypes and axes](../concepts/arrays.md) and
[Coordinates, signs and units](../concepts/conventions.md). Experimental
payload envelopes can retain other object types and their own axis metadata.

## What is public

A name is part of the documented research API when it is exported by a
subpackage's `__all__` and documented here. `__all__` makes that package import
discoverable; the module documentation explains the interface contract. A
subpackage API does not need to be re-exported from the `gragra` root package.
The top-level package currently re-exports the implemented point, coherent,
and displacement observables alongside the source and target containers.

| Namespace | Public names | Documented on |
|---|---|---|
| `gragra` | 14 | [Top-level API](top-level.md) |
| `gragra.gravity_sources` | 15 | [Sources, targets and geometry](core.md) |
| `gragra.targets` | 1 | [Sources, targets and geometry](core.md) |
| `gragra.geometry` | 11 | [Sources, targets and geometry](core.md) |
| `gragra.integrators` | 8 | [Sources, targets and geometry](core.md) |
| `gragra.kernels` | 7 | [Sources, targets and geometry](core.md) |
| `gragra.observables` | 72 | [Top-level API](top-level.md) and [Fields, waves and coherent](physics.md) |
| `gragra.results` | 31 | [Results and spectral representations](results.md) |
| `gragra.reference` | 6 | [Numerical reference API](reference.md) |
| `gragra.interactions` | 2 | [Interactions](interactions.md) |
| `gragra.fields` | 38 | [Fields, waves and coherent](physics.md) |
| `gragra.waves` | 21 | [Fields, waves and coherent](physics.md) |
| `gragra.coherent` | 6 | [Fields, waves and coherent](physics.md) |
| `gragra.analytic` | 52 | [Fields, waves and coherent](physics.md) |
| `gragra.analytic.atmospheric` | 7 | [Sources, targets and geometry](core.md) |
| `gragra.analytic.objects` | 6 | [Sources, targets and geometry](core.md) and [Fields, waves and coherent](physics.md) |
| `gragra.analytic.seismic` | 7 | [Sources, targets and geometry](core.md) and [Fields, waves and coherent](physics.md) |
| `gragra.adapters` | 54 | [Additional APIs and internal modules](experimental.md) |
| `gragra.adapters.specfem` | 4 | [Additional APIs and internal modules](experimental.md) |

The package also contains helpers under `gragra.plotting`. They are outside
the documented public API scope for this release and are not listed in this
reference. For new visualization examples, use Matplotlib with values returned
by the documented calculation API.

Each symbol is rendered on exactly one page. Where a name is re-exported —
`point_acceleration` is both `gragra.point_acceleration` and
`gragra.observables.point_acceleration`, and they are the same object — the
reference documents it once, at the import path you are expected to use, and
the other page links to it.

Names absent from every documented package `__all__` are outside this
documented API, along with names beginning with an underscore and private
members of documented classes. Such internal names can change or disappear
without notice, and are listed on the
[Additional APIs and internal modules](experimental.md) page so you can
recognise them if you meet one in a traceback.

## Interface levels

Research interfaces are not all equally settled. Four levels are used, and they are the same four
words that appear in the **Status** row of the Verification Block on every
Concepts and Guide page, so a symbol's stability and its verification
evidence are read in the same vocabulary.

| Level | What it means |
|---|---|
| **Stable** | The signature, the physical convention and the sign convention are settled. A change that alters results is a breaking change and will be called one. |
| **Experimental** | Usable and tested, but the interface is still being shaped by real use. Argument names, defaults and return containers may change. |
| **Diagnostic only** | Reports on the *quality of your input* rather than computing a physical quantity. A non-zero residual from one of these is information about your mesh or field, not a defect in gragra. |
| **Not implemented** | Named in the documentation, not present in the package. Nothing here is at this level — it exists so that the Verification Blocks can say so explicitly rather than by omission. |

The core computation path — sources, targets, the point observables,
the kernels and integrators beneath them, and the geometry primitives — is
Stable. The adapters, the field helpers and the batch-import helpers are
Experimental; that is stated per symbol on their pages and is not a blanket
claim about their numerical usefulness.

## How signatures are rendered

Signatures come from the source itself, via
[mkdocstrings](https://mkdocstrings.github.io/), so a signature shown here
cannot drift from the installed package. Three consequences are worth
knowing when reading a rendered entry:

- **Source is not shown.** The reference documents the contract, not the
  implementation. Read the source in the repository if you need it.
- **Private members are filtered out.** A class shows the attributes and
  methods that are part of its contract, not its internal helpers.
- **Docstrings are NumPy style.** Parameters, Returns and Raises are
  separate sections, and the Raises section is the authoritative list of the
  validation errors a function produces.

Numerical return dtypes follow the input rather than being fixed by the function: real
weights give `float64` results, complex weights give `complex128` results.
The second-order cross statistics are the documented exception and are always
complex. Arrays returned by public numerical observables are read-only; copy
one before modifying it. This does not describe arbitrary payloads retained by
`GravityResult`, whose mutability is unchanged.

## Performance parameters

Two keyword arguments appear across many signatures and are performance
controls, not physics:

- **`backend=`** selects the implementation. The NumPy path is the
  **reference implementation**, and it is the one every convention on this
  site describes. `"numba"`, `"cpp"` and `"cupy"` are optional, provisional
  accelerators — they may change or become unavailable, and a backend never
  silently falls back to NumPy when it is missing. See
  [Compute backends](../guide/backends.md).
- **`chunk_size=`** bounds peak memory by tiling the sum over source
  elements. On the NumPy, numba and C++ paths the result does not depend on
  it. See
  [Chunking, softening and tolerances](../concepts/numerics.md) for the one
  place where that independence weakens.

Neither argument changes the meaning of a result, and neither is part of the
stability promise the physics APIs carry.

## Where the verification status lives

This section deliberately does **not** say how well any of it is verified.
That belongs next to the physics, where the conditions can be stated, so
each symbol's evidence lives in a Verification Block on the Concepts or
Guide page that explains it — with what it was compared against, at what
agreement level, which tests are the evidence, and, importantly, what is
*not* covered.

[The verification matrix](../verification/matrix.md) collects those blocks
into one table, and
[Limitations and non-guarantees](../verification/limitations.md) records the
things that no test asserts at all. If you are evaluating whether gragra is
suitable for a piece of work, read those two pages before this section.
