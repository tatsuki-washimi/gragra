
# Point sources

## The generic signed and complex source

`WeightedPointSource` is gragra's generic source type: a table of positions
`(N, 3)` and a matching array of weights `(N,)`, where weights are mass
elements in kilograms that may be **negative or complex**. Use it whenever
your source is not simply a collection of positive real masses — a density
perturbation δρ, a phase-carrying element of a plane-wave source, or any
signed/complex quantity.

```python
--8<-- "snippets/point_sources_weighted_point_source.py"
```

## Positive-real-mass convenience wrappers

`PointMass` (a single mass) and `PointMassCloud` (many masses) are
**convenience wrappers for positive real masses only** — not a separate
source abstraction. Both reject non-positive or non-finite mass at
construction, and both convert internally to a `WeightedPointSource` via
`as_weighted_source()` before any calculation happens. If your source can be
negative or complex, use `WeightedPointSource` directly — `PointMass`/`PointMassCloud`
will never accept it.

```python
--8<-- "snippets/quickstart_point_mass_acceleration.py"
```

## Observation points

High-level `TargetPoints` stores `(N, 3)` length Quantities, unique IDs and a
`Frame`; pass it to a model's `.at(targets)` method. Low-level numerical
observables accept bare source-frame SI coordinate arrays, including a single
`(3,)` point promoted to `(1, 3)` for convenience. They do not infer units,
IDs or frame transforms from those arrays.

## What the source weights mean

## Prescribed motion

Finite-body mesh comparison uses signed `WeightedPointSource` terms for the
bulk and interface mass perturbations, plus a separate positive mass cloud
with per-cell displacement. The bulk and interface terms are summed before
comparison; the displaced mass cloud is an alternative representation, not an
additional source to superpose.

`PrescribedPointTrajectory` and `PrescribedRigidTransform` describe sampled
external motion. Their observable functions evaluate only the supplied frames;
they do not interpolate a trajectory, generate motion, or solve dynamics.

`weights_kg` is always a **mass element** — a point mass, or the mass
element of a discretized density perturbation (`δm = δρ·dV`). It is never a
pre-integrated transfer coefficient or a geometric weighting factor such as
`(n̂·d)·dS` — those live in a separate abstraction used by the coherent and
surface-source machinery (see
[Volume and surface sources](continuous-sources.md) and
[Coherent responses and cross statistics](coherent.md)), not in
`weights_kg`.

## Validation and error behaviour

- A `WeightedPointSource` position array and high-level `TargetPoints` input
  must be a strict 2-D table; passing `(3,)` raises `ValueError`.
- `PointMass`/`PointMassCloud` reject non-positive or non-finite mass.
- A source point coincident with a target point (`r == 0`) with
  `softening_m == 0` raises `ValueError` naming the offending index pair;
  see [Chunking, softening and tolerances](../concepts/numerics.md).
- All stored arrays (`positions_m`, `weights_kg`) are read-only after
  construction — see [Arrays, dtypes and axes](../concepts/arrays.md).

```python
--8<-- "snippets/quickstart_array_conventions.py"
```

## Status and verification

### `WeightedPointSource` { #status-weightedpointsource }

!!! abstract "Status & verification"

    |  |  |
    |---|---|
    | **Status** | Stable |
    | **Verified against** | invariance property; closed-form analytic solution |
    | **Agreement** | Exact — sign flip and complex dtype promotion are checked as exact equalities, not tolerances |
    | **Test evidence** | `tests/test_point_sources.py::test_weighted_point_source_valid_real`, `tests/test_point_sources.py::test_weighted_point_source_valid_complex`, `tests/test_point_sources.py::test_stored_arrays_read_only_and_defensive_copy`, `tests/test_observables.py::test_weighted_source_sign_flip`, `tests/test_observables.py::test_weighted_source_complex_output_dtype` |
    | **Conditions** | `positions_m` strict `(N, 3)` float64; `weights_kg` `(N,)`, real or complex |
    | **Not verified** | Behaviour for weight magnitudes outside the ranges the test suite exercises |
    | **Known limits** | `weights_kg` is a mass element only — see [What the source weights mean](#what-the-source-weights-mean) |
    | **Scope** | none |

### `PointMass` { #status-pointmass }

!!! abstract "Status & verification"

    |  |  |
    |---|---|
    | **Status** | Stable |
    | **Verified against** | closed-form analytic solution; invariance property |
    | **Agreement** | Exact at the unit-mass, unit-distance fixed point |
    | **Test evidence** | `tests/test_point_sources.py::test_point_mass_valid`, `tests/test_point_sources.py::test_point_mass_invalid`, `tests/test_point_sources.py::test_point_mass_conversion`, `tests/test_observables.py::test_acceleration_analytic_value` |
    | **Conditions** | `mass_kg > 0` and finite; `position_m` shape `(3,)` |
    | **Not verified** | none beyond the conditions above |
    | **Known limits** | Positive real mass only — it is a convenience wrapper, not the generic source type |
    | **Scope** | none |

### `PointMassCloud` { #status-pointmasscloud }

!!! abstract "Status & verification"

    |  |  |
    |---|---|
    | **Status** | Stable |
    | **Verified against** | invariance property (equals the sum of individual `PointMass` contributions) |
    | **Agreement** | Exact |
    | **Test evidence** | `tests/test_point_sources.py::test_point_mass_cloud_valid`, `tests/test_point_sources.py::test_point_mass_cloud_invalid`, `tests/test_observables.py::test_point_mass_cloud_equals_sum` |
    | **Conditions** | `masses_kg` all positive and finite; `positions_m` strict `(N, 3)` |
    | **Not verified** | none beyond the conditions above |
    | **Known limits** | Positive real mass only |
    | **Scope** | none |

### `TargetPoints` { #status-targetpoints }

!!! abstract "Status & verification"

    |  |  |
    |---|---|
    | **Status** | Stable |
    | **Verified against** | invariance property |
    | **Agreement** | Exact — shape, units, IDs and owned read-only coordinates are checked as behavioural assertions |
    | **Test evidence** | `tests/high_level/test_inputs.py::test_target_ids_are_nonempty_unique_strings_matching_count`, `tests/high_level/test_inputs.py::test_quantity_units_and_finite_inputs_are_required`, `tests/high_level/test_inputs.py::test_input_arrays_are_owned_even_if_original_and_returned_views_are_mutated` |
    | **Conditions** | Length Quantity with shape `(N, 3)`, unique non-empty IDs, and an explicit `Frame` |
    | **Not verified** | none beyond the conditions above |
    | **Known limits** | Low-level numerical observables use source-frame SI coordinate arrays and do not consume this high-level container |
    | **Scope** | Target coordinates and identity only |
