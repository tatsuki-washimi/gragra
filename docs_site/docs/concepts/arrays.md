
# Arrays, dtypes and axes

```python
--8<-- "snippets/quickstart_array_conventions.py"
```

## dtypes

Coordinates and other geometric quantities are always `float64`. Amplitudes
and weights keep the *kind* of number they were given: a real-valued
`weights_kg` stays `float64`, a complex-valued one is promoted to
`complex128`. An observable's output dtype follows its weights, never the
coordinates — a real source and a complex source passed through the same
function produce `float64` and `complex128` results respectively.

## Axis order

### Component and tensor axes

The component axis (for a vector observable) or the two tensor axes (for
`point_gravity_gradient`) are always **trailing**: a vector observable
returns shape `(M, 3)`, a tensor observable `(M, 3, 3)`, where `M` is the
number of observation points.

### Frequency and direction axes

When a result carries a frequency or direction axis in addition to the
component axis, that axis is inserted **before** the component axis:
`(M, Nf, 3)` for a frequency axis, `(M, Ndir, 3)` for a direction axis. When
both are present, direction comes before frequency: `(M, Ndir, Nf, 3)`. See
[Coherent responses and cross statistics](../guide/coherent.md) and
[Waves](../guide/waves.md) for the functions that produce these shapes.

## Read-only arrays

Two independent contracts, both enforced with `setflags(write=False)`:

- **Stored arrays.** The `positions_m` and `weights_kg` arrays held by a
  numerical source or `TargetPoints.positions_m` held by the high-level target
  container are read-only after construction. Input arrays are copied, so
  later caller mutations do not reach the stored data.
- **Returned arrays.** Every public observable function
  (`point_potential`, `point_acceleration`, `point_gravity_gradient`, and their
  coherent counterparts) returns a
  read-only array. If you need to modify a result, copy it first.

## Shape validation

Numerical source dataclasses and high-level `TargetPoints` require a **strict**
`(N, 3)` position table. High-level targets also require a length Quantity,
unique IDs and a `Frame`; pass them to a model's `.at(targets)` method. The
low-level numerical observables accept bare source-frame SI coordinate arrays,
including a single `(3,)` point promoted to `(1, 3)` for convenience.

## On-disk layout is not the in-memory shape

The axis-order rules above describe **in-memory API shapes** — the arrays
stored in dataclasses and returned by public functions. On-disk storage
(HDF5 field files, for example) follows a separate convention chosen for
I/O efficiency: a **leading batch axis**, e.g. `displacement_m` shaped
`(P, Nv, 3)` for `P` snapshots. Batch-conversion helpers (see
[External field data and HDF5](../external/hdf5-fields.md)) accept this
leading-batch-axis form as an input convenience, but their *return values*
are always converted back to the in-memory convention above — component
axis trailing, parametric axes before it. Do not assume a leading batch
axis anywhere a return value from gragra's public API is documented.
