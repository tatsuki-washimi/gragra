
# Limitations and non-guarantees

This page collects what gragra does not guarantee. It is written for the reader who is deciding whether to trust a number, and it is deliberately the least flattering page on the site: everything here is a limit the maintainer knows about and has chosen to state rather than let you discover.

Nothing on this page is a promise of future work. Where an item exists as an internal design note but not as code, it is described as absent, because that is what it is from your side of the interface.

## Physical model limitations

**Newtonian, first order, no self-gravity.** gragra integrates the instantaneous Newtonian field of a mass distribution. There is no retardation, no relativistic correction, and no feedback of the computed gravity field on the source that produced it. Density perturbations are treated to first order in the displacement; a large displacement is outside the model even where the arithmetic still returns a value.

**No dynamical PDE is solved.** Elastodynamics, fluid flow and instrument response are all outside the package, as stated on the [home page](../index.md). The kinematic field constructors gragra does ship — plane waves, Rayleigh eigenfunctions, stochastic P/S realizations — are idealizations: a homogeneous isotropic unbounded medium or a homogeneous half space, one mode or one plane wave at a time, no layering, no attenuation, no scattering, no source near field. They exist to feed the integrator and to serve as benchmarks, not to model a real site.

**Softening changes the physics, it does not regularize it.** Setting `softening_m > 0` replaces a point mass with an extended Plummer sphere. The near field is then a genuinely different mass distribution rather than the same field with a singularity removed, and the trace-free property of the gravity-gradient tensor holds **only** at exactly zero softening. Do not treat a softened result near the source as an approximation to the unsoftened one; see [Chunking, softening and tolerances](../concepts/numerics.md).

**Incomplete input fails quietly in specific, physical ways.** A rigidly translated body has no bulk density perturbation at all, so a bulk-only calculation returns exactly zero for a moving rigid object — a correct computation of the wrong thing. A pure shear wave in an unbounded homogeneous medium likewise produces a structurally zero bulk perturbation. Strain is not a substitute for displacement: the bulk term needs only its trace, but the surface term needs the displacement itself, and gragra cannot reconstruct one from the other. These cases are set out where they arise, in [Displacement and density-perturbation fields](../guide/fields.md) and [Waves](../guide/waves.md).

**Only a minority of the standard catalogue of physical cases has a dedicated treatment.** Measured against the review used as a yardstick, roughly 30% of the catalogued cases in its core chapters are covered by a dedicated analytic result or benchmark, and one whole chapter — seismic point sources — is at zero. [Literature coverage](coverage.md) gives the breakdown and the method.

## Numerical limitations

**Discretized geometry is an approximation with a measurable error, not an exact representation.** A volume or a surface becomes a finite set of source elements, and the result carries the corresponding quadrature error. The Sphere convergence suite compares to its closed form at `R=1 m`, density `1000 kg/m³`, and one exterior target; it gates the finest relative error at `0.05` for each observable and overall coarsest-to-finest improvement for potential. It computes a fitted slope but does not assert one. One separately recorded Sphere run was not monotonic between its two finest spacings; its measured errors and slopes are in the internal repository note `docs/validation/nonanalytic_static_sources.md`. These Sphere rules do not define convergence guarantees for other geometries. Error at any spacing still depends on the target's distance to the nearest source element, so run a refinement sweep for your own geometry and targets.

**Element types and geometries are restricted.** Surface meshes are triangle meshes and volume meshes are tetrahedral or hexahedral cells; arbitrary polyhedra, higher-order elements and mixed-element meshes are not handled. The built-in primitives are a fixed catalogue of simple shapes, not a general CAD kernel. External-solver data is narrower still: the exporter targets hexahedral spectral elements at one fixed node layout (5×5×5), and rejects anything else. See [Geometry and discretization](../guide/geometry.md) and [Using external meshes](../guide/meshes.md).

**Reproducibility is not uniform across compute paths.** On the NumPy, `numba` and `cpp` paths the result does not depend on `chunk_size`. On the GPU path it does: parallel reduction is not associative, so `chunk_size` becomes an effective tiling parameter and reproducibility narrows to "same device, same library version, same `chunk_size`". A workflow that changes `chunk_size` to fit available memory changes its numerical output there, by a small amount that is nevertheless not zero.

**Precision assumptions are fixed and can be silently exceeded by your input.** Coordinates are float64 and amplitudes complex128. Data arriving from an external solver may have been produced at float32 and cannot become more accurate by being read in at higher precision; differences at the $10^{-7}$ relative level in such input are expected and are not evidence of a defect in gragra.

**Absolute tolerances at this scale need a physical scale.** Every gravitational quantity gragra returns is scaled by $G \approx 6.7 \times 10^{-11}$. A general-purpose default absolute tolerance can therefore hide a zero or sign error in a nonzero fixed point. In a targeted audit, nonzero comparisons in `tests/test_observables.py` use relative tolerance with `atol=0`; geometrically exact structural zeros use exact assertions, while the zero-softening gradient trace uses a roundoff bound of `32 ε max(abs(T))`. This rule is not a claim that every comparison in the repository has been audited.

## What the test suite does not cover

The general limits of the verification methods used here — what an oracle, a convergence sweep, a backend comparison and a mutation test each cannot see — are described in [How gragra is verified](index.md#what-each-verification-method-can-and-cannot-detect). This section records the specific gaps.

**G-scale assertion audit, limited to the observable checks.** An earlier version of this page recorded two fixed-point assertions whose default absolute tolerance accepted zero and sign-flipped values. The current `tests/test_observables.py` audit classified 16 numerical comparison assertions and added eight mutation controls: zero and sign-flip mutations are rejected for the potential, acceleration, gradient, and signed-source checks. The physical nonzero fixed points use relative tolerances with `atol=0`; structurally zero components are exact only where geometry guarantees them, and the zero-softening trace allows `32 ε` times the tensor scale. This is evidence for the named checks only, not a broad guarantee about every suite or observable.

**Educational and physical soundness of examples is not machine-checked.** The site's executable examples are executed, and their assertions are real; nothing checks that an example teaches the right lesson, or that the configuration it demonstrates is a sensible one for your problem.

**The documentation checks are enumerative, not exhaustive.** The checks that
guard this site catch a fixed list of consistency patterns. A clean run means
"none of the enumerated problems is present", not "the page is accurate".
Claims about capabilities for which no rule was written pass unexamined, and
no automated rule can distinguish "supports triangle surface meshes" from
"supports arbitrary meshes" by degree.

### Example and hosted-workflow limits

The MkDocs build validates pages, links, anchors, and snippet include paths; it does not execute code. `tests/test_docs_snippets.py` runs the self-contained snippets under `docs_site/snippets`, and `tests/test_examples.py` smoke-runs its explicit list of scripts under `examples/`. A skipped optional-dependency example is not scientific verification. Scripts outside those test lists, including the in-memory density-grid and volume-mesh workflow, need a separate local run; the in-memory example does not need the file-reader extra.

The repository has one guarded GitHub Actions workflow, `Manual smoke`, triggered with `workflow_dispatch`. When its private-repository opt-in condition is met, it installs `.[dev]`, checks a public import, and runs `tests/test_ci_coverage.py` plus `tests/test_import.py`. It is not a scientific, optional-extra, backend, Python-version, or operating-system matrix. Research-required observable, oracle, convergence, and backend checks run locally with the source revision and conditions recorded; the [contributing guide](../help/contributing.md) lists the local commands.

**The GPU path.** The manual smoke workflow has no GPU runner and does not run `cupy` backend tests. Numerical agreement with NumPy (`rtol=1e-12`, `atol=1e-15`) is a local, single-machine measurement by the maintainer; it covers one device with one driver and one library version. If you use the GPU path for quantitative work, reproduce that comparison on your own hardware first.

**Checks that need pilot data.** Some external-solver checks run only against a
real solver export that is local research input. They are selected with
environment variables (`GRAGRA_SPECFEM_PILOT_H5`,
`GRAGRA_SPECFEM_PILOT_EXPECT`, `GRAGRA_SPECFEM_SOURCE`) and skip by default.
The pilot results cannot be reproduced without that input, and the checks do
not re-run automatically when the code changes.

**Rendering.** The site build fails on a broken link or a dangling anchor, but a mathematical expression that typesets incorrectly still builds successfully. Visual correctness is checked by a human pass, not by an automated check.

## External-solver limitations

**External-solver support is narrow; there is no multi-solver abstraction.** SPECFEM3D Cartesian has a one-directional native binary exporter to gragra's HDF5 layout. Separately, the `mesh` extra provides a fixed Foundation OpenFOAM 14 patch 20260724 legacy-ASCII `.vtk` density reader. Neither path is a solver-neutral importer interface, a plugin mechanism, or support for a further solver configuration.

**The exported file has a narrow loader in gragra.** The
`gragra.adapters.specfem` subpackage can open the documented time-sampled
schema either eagerly or through a context-managed bounded stream reader;
broader solver configurations remain outside the current path.
[SPECFEM3D Cartesian](../external/specfem3d.md) states the constraints in full.

**SPECFEM3D support is intentionally narrow.** The exporter's structural
checks (internally "Tier A") run on static, mesh-only output for a real pilot
run. They do not establish agreement with an analytic gravity reference, a
second reader, or any other solver run. The profile check rejects inputs
outside the one configuration the implementation currently handles; this is a
scope boundary, not a statement about the underlying physics.

**The saved older 6 km SPECFEM case is fixed-artifact evidence.** It completed
all 6,912 elements, two time windows, and three observables, but it does not
establish solver stability, domain convergence, or a general accuracy
threshold. The front/full difference can contain transients or other time
variation; both fit matrices have condition number about 1, so the difference
is not evidence of least-squares ill-conditioning. The input records an unknown
exporter code reference, no exporter source hash, and placeholder cadence
provenance, so raw-to-export regeneration is unresolved. The new matched 12/18
km S comparison is separate. Both full acceleration analyses and pair-vector
arithmetic completed, but the required nested element correspondence gate
rejected 7,491 missing and 7,484 extra keys before static-field byte matching.
B overlap contributions were not evaluated. The observed cross-domain vector
differences therefore cannot be assigned to domain size or used to claim
stability or convergence; the S scientific verdict is partial.

**Time-sampled export requires artifact-specific provenance review.** The older
6 km file has placeholder cadence and incomplete exporter identity. Its saved
artifact analysis is repeatable, but that does not prove byte-identical
regeneration from raw solver output; the full statement is on the
[SPECFEM3D Cartesian](../external/specfem3d.md) page.

**OpenFOAM support is fixed-profile only.** With the `mesh` extra, `load_openfoam_density` reads Foundation OpenFOAM 14 patch 20260724 reconstructed single-region legacy-ASCII `.vtk` snapshots containing `C`, `Vc`, `cellID`, and a caller-selected `rho` or `delta_rho` field. It does not read native `polyMesh`, general OpenFOAM formats, or run a CFD solver. Other OpenFOAM output can still be converted into an HDF5 layout described in [External field data and HDF5](../external/hdf5-fields.md); gragra makes no statement about the correctness of that conversion.

**The cosine-bell OpenFOAM study is complete only for its utility-only scope.**
The maximum continuum relative errors at 16³/32³/48³ are 0.705172%/0.161700%/
0.027547% (mass), 0.705332%/0.161723%/0.027553% (potential),
0.705962%/0.161817%/0.027579% (acceleration), and 0.707518%/0.162053%/
0.027644% (gradient); all eight gates pass. The 16³ and 32³ fixtures use
six-significant-digit legacy ASCII float32 VTK and 48³ uses nine-significant-digit
legacy ASCII float32 VTK. This is not pure spatial-convergence or convergence-order
evidence, and it is not float64 VTK evidence. Three separate time-specific
`foamToVTK` runs used a private library. Native inputs were export-audit data only;
science used only unchanged reader `Snapshot` values. The prior rejection is kept
as history; no CFD solver or 48³ GWexpy bridge was executed.

**GWexpy bridges do not discover files or infer geometry.** The snapshot bridge
handles one real density field. Separate fixed-mesh bridges handle real density
time series, complex density phasors, displacement series, and surface-density
series using explicit immutable geometry and phase sidecars. The tested native
versions are GWexpy 0.1.7 and 0.1.14.

**The external-solver evidence is fixed-condition only.** A SPECFEM3D comparison exists for the Rayleigh §3.4.2 configuration recorded in the [literature coverage](coverage.md). It does not establish agreement for other solver configurations or a general solver/domain convergence result. Criterion 5 in that scoring remains unmet for sections without such a fixture, and criterion 6 is not established by this comparison.

## Naming and provenance notes

**Provenance of ported formulas.** Parts of gragra's analytic layer were re-derived independently against a predecessor package developed by the same author, and the source docstrings record which revision each formula was checked against, together with the differences in numerical guarding between the two. That attribution is kept deliberately: naming the software a result came from is scientific traceability, and removing it would buy nothing. What is not published anywhere is any filesystem path into that package or into any other local working tree.

**Experimental and provisional API surface.** Three groups of names are explicitly unstable and may change without a deprecation period: everything under `gragra.adapters` (the mesh, HDF5, external-solver and third-party integration layer), the surface-mesh resolution and container types in the geometry layer, and the performance-selecting `backend` keyword accepted by the point observables. The NumPy path is the reference implementation and the stable one; the accelerated paths are experimental performance selectors whose availability and behaviour may change. Per-symbol status is recorded in the Verification Block on each symbol's own page and indexed in the [verification matrix](matrix.md).

**Citation.** Cite the commit used for a calculation and record the input data,
backend, solver settings, and numerical tolerances that affect the result.
[Reproducibility and citation](../help/citation.md) explains what to record.

## Known open issues

**A displacement-continuity check has never run on the path it guards, and did not pass when it was run.** One arm of the exporter's structural checks compares displacement values at grid points that several elements share. On a static, mesh-only export no displacement dataset is written at all, so that arm returns without asserting anything — it is vacuous on exactly the artifact the checks are usually run against. When it was exercised against a smoke artifact built from real pilot solver output it did **not** pass: a small fraction of shared points disagreed, at a relative magnitude consistent with float32 assembly-order differences between solver domains rather than with a defect in gragra. The measured fractions and deviations, and the reasoning behind that attribution, are on the [SPECFEM3D Cartesian](../external/specfem3d.md) page. The point for a reader here is narrower and unambiguous: the check does not currently pass, so it cannot be cited as a guarantee that displacement data is continuous across element boundaries.

**Blind spots in that same check are documented by tests.** Corruption
confined to points that no other element shares, and permutation of
displacement values between such interior points, are both undetectable by it.
The suite records these limits in
`tests/test_gate_mutation.py::test_a4_cg_continuity_blind_to_interior_only_displacement_corruption`
and
`::test_a4_check_blind_to_displacement_index_permutation_between_interior_points`.

The G-scale audit described [above](#what-the-test-suite-does-not-cover) resolved the two previously identified weak sign checks and tested eight zero/sign-flip mutations across the named observables. The audit remains limited to the listed `tests/test_observables.py` comparisons; it is not a repository-wide assertion audit.
