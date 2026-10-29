
# Literature coverage

## Scope of this comparison

A gravity-perturbation code can be judged two ways: how well it computes the cases it implements, and how many of the cases a reader might reasonably expect it to implement it actually has. This page answers the second question, which is the one a new user needs first and the one a test suite cannot answer at all.

The catalogue used as the yardstick is the standard review of the field:

> Harms, J. *Terrestrial Gravity Fluctuations.* Living Reviews in Relativity **22**, 6 (2019). <https://doi.org/10.1007/s41114-019-0022-2> (CC BY 4.0)

Its chapter and section structure is used here as an index of physical cases. No text, equation or figure from it is reproduced on this site; where a specific result is meant, it is identified by the review's own section and equation numbers, and you are expected to read it there.

Two things this page deliberately does **not** measure.

**It does not measure whether an API can produce a number.** Almost any density perturbation can be pushed through gragra's generic integrators and will yield a value. The question asked here is stricter: is there a *dedicated analytic result or benchmark* for this physical case, independent of the production path, against which an external simulation could be checked?

**It does not measure implementation status.** A section is scored on how far it satisfies gragra's own six acceptance conditions for a benchmark:

1. **Conventions** — coordinate system, sign of the depth axis, Fourier and complex-amplitude conventions, source amplitude, where the gravitational constant is applied, units.
2. **Applicability** — homogeneity, isotropy, small-displacement and first-order assumptions, wavelength regime, near- versus far-field, whether self-gravity is neglected, boundary conditions.
3. **Observable completeness** — potential, acceleration and gravity gradient wherever possible; at minimum, an explicit statement of which observable is missing.
4. **Independent verification** — a quadrature, finite-difference or symbolic check that does not re-use the algebra of the closed form.
5. **External-solver fixture** — at least one end-to-end comparison against an external solver or a solver-neutral field fixture.
6. **Convergence report** — recorded error behaviour along whichever axes apply: mesh spacing, domain radius, absorbing-boundary distance, time step, quadrature order, interpolation.

Because these are stricter than "is it implemented", a section can have a complete, well-tested implementation and still be scored as partially covered — most often because conditions 5 and 6 are unmet. That is a difference of definition, not a contradiction. The scoring is the maintainer's own reading of gragra against gragra's own conditions: a self-assessment, published so that it can be argued with, not an external audit.

The figures below were computed against the state of `main` at the time of writing and are re-derived whenever a benchmark is added.

## Coverage by chapter

Only leaf sections are counted: parent headings, chapter summaries and sections that fall outside gragra's responsibility are excluded from the denominators. Chapter 1 (introduction), Chapter 7 (noise mitigation, filtering, site selection) and Appendix A (mathematical formalism) contribute no counted leaves — the first two because they are not implementation targets for a forward gravity calculator, the appendix because it is supporting mathematics rather than a physical case.

| Chapter | Leaf sections counted | At least partly reached | Reach rate | Weighted coverage | Reading |
|---|---:|---:|---:|---:|---|
| 2 — Gravity measurements | 3 | 2 | 67% | **33%** | The observable side exists; there is no dedicated gravitoelastic coupling benchmark |
| 3 — Gravity perturbations from seismic fields | 12 | 10 | 83% | **42%** | The strongest area, and the one gragra was built for |
| 4 — Gravity perturbations from seismic point sources | 4 | 0 | 0% | **0%** | Untouched |
| 5 — Atmospheric gravity perturbations | 4 | 2 | 50% | **13%** | Generic density-integration machinery only |
| 6 — Gravity perturbations from objects | 6 | 6 | 100% | **54%** | Generic prescribed trajectories and rigid rotations are available; object-specific formula catalogues are incomplete |
| **Total, chapters 2–6** | **29** | **20** | **69%** | **34%** | A well-verified core with an incomplete formula catalogue |

"Reach rate" counts a section as reached if generic infrastructure alone can address it. "Weighted coverage" penalises the absence of a dedicated analytic result, and is the more honest of the two.

**A weighted coverage of about 34% does not mean the engine is 34% correct.** For the cases it does cover, the quality of the implementation and the independence of its verification are high — that is what the rest of the [verification section](index.md) is about. What the 34% measures is *breadth*: about a third of the physical cases catalogued in chapters 2–6 have a dedicated treatment here, and the remainder do not.

## What is fully covered

Three leaf sections score full marks — a dedicated implementation with its principal formulas, a numerical path and verification:

- **§3.2.1, gravity perturbation from a seismic displacement field** (eqs. 42–49). The bulk and surface decomposition and the displacement-dipole form are all implemented as production code, with golden values and a finite-difference cross-check. This is gragra's core path; see [Displacement and density-perturbation fields](../guide/fields.md).
- **§3.3.1, a spherical cavity in an infinite homogeneous medium, without scattering** (eqs. 61–65). The bulk P contribution, the P and S wall contributions, their combination and the small-cavity limit each have a dedicated closed form and a discretization convergence suite. See [Analytic benchmarks](../guide/analytic-benchmarks.md).
- **§6.3, oscillating point masses.** Harmonic small-displacement dipole response, available directly as a public observable.

Even here, the equation-level score is *partial* rather than complete, because acceptance conditions 5 and 6 — an external-solver fixture and a recorded convergence report — are not satisfied for every one of them. A fixed-condition external SPECFEM3D comparison exists for Rayleigh §3.4.2, but it does not establish general solver or domain-convergence behaviour; see [Limitations and non-guarantees](limitations.md).

## What is partially covered

The bulk of chapter 3, plus scattered sections elsewhere. In each case something real exists and something specific is missing:

- **§3.1, seismic waves** (eq. 39 for the Rayleigh eigenfunctions). Plane P and S waves, SH/SV polarizations and the fundamental Rayleigh mode are implemented as kinematic field constructors. Layered media, general interface waves and source near fields are not. See [Waves](../guide/waves.md).
- **§3.2.2, perturbations expressed through seismic potentials.** Individual results appear inside other implementations, but there is no API that takes the scalar and vector potentials as input.
- **§3.2.3, the transform domain.** Plane-wave complex weights and a half-space transfer function exist; a general Fourier-domain gravity operator does not.
- **§3.4.1, body waves in a half space** (eqs. 86–90). Only vertical incidence is implemented, for the bulk and full transfer functions. General incidence angles, P↔SV conversion at the free surface and complex phase coefficients are absent. The implemented transfer is a medium-interior limit and does not apply to a detector standing on the surface.
- **§3.4.2, Rayleigh waves** (eqs. 91–99, with the above-surface closed form at eq. 96). The eigenfunctions, the above-surface acceleration and a kinematic depth-reduction factor are implemented, and both a production path and an independent oracle exist. The sub-surface acceleration formula is not implemented.
- **§2.1.1 and §2.1.4, static gravity and gradient coupling.** Both quantities are computable; there is no dedicated coupling API or benchmark for the response of a suspended test mass, and the instrument side is out of scope by design.
- **§3.5, numerical simulations.** External field ingest and gravity post-processing exist. A standard solver benchmark workflow and an end-to-end mesh/HDF5 comparison do not; see [External field data and HDF5](../external/hdf5-fields.md).
- **§3.6.1 and §3.6.4, seismic noise estimates.** Known complex responses and directional weighting can be handled, and multi-target gradient calculations exist; there is no spectrum-to-noise-estimate API and no dedicated long-wavelength differential-response benchmark.
- **§5.1 and §5.2, atmospheric sound and quasi-static temperature perturbations.** Adiabatic pressure-to-density and isobaric temperature-to-density constructors are implemented, together with independent finite-air-box and infinite-halfspace references. The finite and infinite domains are different physical problems, and the prescribed-density references do not establish general acoustic-propagation or CFD accuracy.
- **§6.1, §6.4 and §6.5, rules of thumb, interacting distributions, oscillating extended objects.** The building blocks are present; the systematic APIs the review's formulas suggest are not.
- **§6.2, objects moving at constant speed along a prescribed trajectory.** `PrescribedPointTrajectory` evaluates supplied positive point-mass samples, with a point-flyby closed-form check. It does not generate motion, interpolate samples, or provide a vehicle-specific geometry or formula catalogue.
- **§6.6, rotating objects.** `PrescribedRigidTransform` evaluates supplied translations and proper rotations for a `PointMassCloud`, with rigid-rotation covariance checks. It does not provide rotating-equipment, NCal, or other object-specific geometry and formulas.

## What is not covered at all

**Chapter 4, gravity perturbations from seismic point sources, scores zero across all four of its leaf sections** — a point force in an elastic medium, the density perturbation of a point shear dislocation, the gravity perturbation from a seismic moment, and sources in a half space. This is the largest gap in the catalogue, and it is the most consequential one for validating an external solver: near-field terms, retarded fields, source-time functions and moment-tensor conventions are precisely the things a plane-wave comparison cannot see. A displacement field produced by an external solver can be converted to gravity today, but without these results there is no independent way to decide whether the solver setup that produced it was right.

Also unimplemented:

- **§3.3.2 and §3.3.3**, scattering of an incident compressional or shear wave by a spherical cavity, including the mode conversion and the scattering correction to the gravity perturbation.
- **§2.1.3**, ground motion induced *by* a gravity field — the reverse coupling, which would be the natural way to check whether an elastodynamic solver handles gravitational back-reaction correctly.
- **§5.3 and §5.4**, shock waves from an explosive point source, and gravity from turbulent flow.
- **Appendix A**, beyond the specific Bessel functions the implemented formulas need: Legendre polynomials, scalar and vector surface spherical harmonics, solid harmonics, the spherical multipole expansion and Clebsch–Gordan coefficients have no dedicated implementation. The multipole expansion is a prerequisite for the object and rotating-source benchmarks above.

**Chapter 7 is not a gap.** Coherent noise cancellation, Wiener filtering, sensor-array optimization and site selection are outside gragra's responsibility by design: gragra produces the gravity response that such an analysis consumes, and does not perform the analysis. Some directional cross-statistics and cross-spectral machinery does exist here, but as an experimental convenience rather than as a stable, supported layer — see [Coherent responses and cross statistics](../guide/coherent.md).

## How the weighting was chosen

Each counted leaf section receives one of four marks, and the chapter figure is the mean of their weights:

| Mark | Meaning | Weight |
|---|---|---:|
| ◎ | Dedicated implementation with principal formulas, numerical path and verification | 1.00 |
| ○ | Principal part implemented, but important conditions or components missing | 0.50 |
| △ | Only generic infrastructure (kernels, field input); no dedicated analytic benchmark | 0.25 |
| × | Desirable as core functionality or as a verification reference, but absent | 0 |
| — | Not an independent implementation target: parent heading, summary, or outside gragra's responsibility | excluded |

Two rates follow from these marks. The **reach rate** is the fraction of counted leaves scoring △ or better; it answers "could I get a number out of gragra for this case at all?" and is therefore generous. The **weighted coverage** is the mean of the weights; it answers "is there a dedicated, independently checkable treatment of this case?" and is therefore severe. The gap between 62% and 30% is exactly the set of cases where generic machinery exists and a dedicated result does not — which is where a user is most likely to over-trust the output, because a plausible number comes back either way.

Recomputing the figures is mechanical: take the leaf sections down to the third heading level, drop parents, summaries and out-of-responsibility rows, then compute both rates per chapter and for chapters 2–6 together. The ledger is updated in the same change that adds a benchmark, which is why the numbers here move only when the underlying capability does.
