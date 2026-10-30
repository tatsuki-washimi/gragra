# Numerical methods

The numerical route depends on how the source is represented. This page maps
those routes to the detailed guides; it does not prescribe one method or one
accuracy threshold for every physical problem.

## Choose a route from the input

| Input or task | Numerical route | Detailed guide |
|---|---|---|
| Discrete point masses or weighted points | Sum the selected point kernel over the supplied elements | [Point sources](point-sources.md); [Observables](observables.md) |
| A finite body described by a shape | Discretize its geometry into volume or surface elements, then evaluate their contributions | [Geometry and discretization](geometry.md); [Volume and surface sources](continuous-sources.md) |
| A mesh or external field | Convert the supported mesh or field layout into gragra's source representation, then use the corresponding observable | [Using external meshes](meshes.md); [External field data and HDF5](../external/hdf5-fields.md); [Displacement and density-perturbation fields](fields.md) |
| A body with an available closed-form result | Evaluate the analytic model and use it as a reference for the matching geometry and assumptions | [Analytic benchmarks](analytic-benchmarks.md) |
| Box integration | Use the documented uniform or adaptive box observable for its supported configurations | [Numerical references and directional averages](../tutorials/numerical-reference.md); [Wave 4--6 reference API](../api/wave456.md) |
| Centered-cavern scalar reference | Use the optional high-precision kernel only for its documented centered spherical-cavern case | [Numerical references and directional averages](../tutorials/numerical-reference.md); [Numerical reference API](../api/reference.md) |
| Directional or stochastic statistics | Form the documented directional average or spectrum using the stated input normalization | [Coherent responses and cross statistics](coherent.md); [Stochastic Newtonian noise](stochastic-nn.md) |

The mesh and external-data routes are adapters; they do not validate how an
external solver produced a field. Their supported file layouts and limits are
listed in the linked guides.

## Numerical controls

| Control | What it changes | Where its limits are documented |
|---|---|---|
| Geometry spacing or mesh resolution | The discrete approximation of a continuous source | [Geometry and discretization](geometry.md); [Volume and surface sources](continuous-sources.md) |
| `softening_m` | The source model near a point-mass singularity; positive softening represents a Plummer-smoothed mass distribution | [Chunking, softening and tolerances](../concepts/numerics.md); [Point sources](point-sources.md) |
| `chunk_size` | Peak intermediate memory for supported summation paths | [Chunking, softening and tolerances](../concepts/numerics.md) |
| `backend` | The numerical implementation used for a calculation | [Compute backends](backends.md) |
| Comparison tolerances | The acceptance of a named numerical comparison, not the physical validity of a model | [How gragra is verified](../verification/index.md); [Limitations and non-guarantees](../verification/limitations.md) |

In particular, softening changes the modeled mass distribution; it is not a
way to recover the unsoftened field at a singular point. A backend comparison
checks agreement between implementations on its tested inputs, not the shared
formula against an independent derivation. The linked pages state these
limits alongside the relevant evidence.

## Read the evidence with the method

The verification pages distinguish checks that answer different questions:

- A closed-form comparison checks a case with a known analytic result.
- An independent numerical oracle checks a separate computational route for
  the named quantity and conditions.
- A convergence study checks refinement behavior for its stated geometry and
  input; it does not define a tolerance for every geometry.
- Backend equivalence checks whether an accelerated path follows the
  reference implementation on the tested inputs.

None of these checks implies that every model, geometry, input range or
scientific condition has been validated. Start with [How gragra is verified](../verification/index.md),
then check the relevant row in the [Verification matrix](../verification/matrix.md)
and the [Limitations and non-guarantees](../verification/limitations.md).
