
# How gragra is verified

## Why a test count is not evidence

This site never tells you how many tests gragra has, because the number answers a question nobody should be asking. A count measures how much was checked. What you need to know is what would have been *caught* — and a test that passes on the correct answer and on a specific wrong answer alike raises the count without raising the detection power.

That is not a hypothetical failure mode in this project. gragra's outputs are scaled by the gravitational constant, $G \approx 6.7 \times 10^{-11}$, so a default absolute tolerance can accept a wrong sign or an array of zeros. A targeted audit of `tests/test_observables.py` has since classified its 16 numerical comparison assertions and added eight zero/sign-flip mutation controls for the listed potential, acceleration, gravity-gradient and signed-source checks. This is evidence about those checks, not an assurance about every assertion in the repository; the scope and remaining limits are recorded in [Limitations and non-guarantees](limitations.md#what-the-test-suite-does-not-cover).

So the unit of evidence on this site is not a number of tests but a **Verification Block**: a fixed eight-field table, attached to the symbol it describes, that names the method, the level of agreement, the test that carries it, the conditions under which it holds — and what it does not cover. Every block is indexed in the [verification matrix](matrix.md). How to read one is [at the bottom of this page](#how-to-read-a-verification-block).

## What each verification method can and cannot detect

Each method below is strong against one family of error and structurally blind to another. The blind spots are not accidents of implementation; they follow from what the method compares against what. This is why the same quantity is attacked from several directions, and why a block that names only one method is a weaker claim than one that names two.

| Method | What it is here | Errors it catches | Errors it is blind to |
|---|---|---|---|
| **Independent numerical oracle** | A generator that recomputes the same physical quantity by a route sharing no code with gragra (`tests/oracles/generate_*.py`), stored as a fixed golden artifact under `tests/oracles/data/` | An error in the closed form itself, or in the derivation behind it: the two routes only agree if both algebras are right | Discretization error common to both routes, and any convention error the oracle *shares* with the implementation — the generators re-derive the physics, but they take the coordinate, sign and amplitude conventions from the same written contract |
| **Convergence sweep** | The Sphere suite uses `R=1 m`, `ρ=1000 kg/m³`, target `(2.7, 0.3, -0.4) m`, and `h=R/[10,20,40,80]`; it gates each finest-spacing relative error at `0.05` and requires potential to improve coarsest-to-finest. It computes but does not assert slope (`tests/test_convergence_suite.py`) | A wrong constant factor when it pushes the independently referenced error beyond that case's budget; a failed refinement gate | A fitted slope or step-to-step change alone does not guarantee normalization. An error below the stated budget can remain, and one case's gate does not establish another geometry's accuracy |
| **Backend equivalence** | The NumPy path as reference implementation, compared element-wise against the `numba`, `cpp` and `cupy` paths (`tests/test_planewave_backends.py`, `tests/test_parametric_*.py`, `tests/test_cupy_*.py`) | Divergence between implementations: a transcription slip, a threading or reduction-order bug, a stale compiled binary | A shared formula error. If all four paths implement the same wrong expression, they agree perfectly and say nothing about correctness. The GPU comparison is additionally blind in time: there is no GPU runner in continuous integration, so it happens only when someone runs it by hand |
| **Mutation test of a check** | Deliberate corruption injected into a copy of real exported data, asserting that the check fails (`tests/test_gate_mutation.py`) | A check that cannot see the change it is intended to detect — including checks that pass because they never ran | The physics. A check can detect corrupted files perfectly and still be checking a wrong formula |
| **Closed-form analytic fixed point** | A hand-derived value at a chosen configuration, compared exactly or at a stated tolerance | Sign conventions, factors, axis ordering — at that point | Anything that vanishes at the chosen point, and any error smaller than the tolerance actually asserted. This is the method whose tolerance must be stated explicitly, for the reason given above |
| **Invariance property** | A quantity that must be unchanged under a transformation — `chunk_size` independence, trace-free gradient at zero softening, structural zeros | Coupling between parameters that must not be coupled; broken symmetries | Any error that respects the symmetry. A constant factor survives every invariance check in this list |

The four sections that follow describe how the first four methods are actually implemented here, with their real limits.

## Independent numerical oracles

An oracle is a second, independent computation of a quantity gragra also computes in closed form. The generators live in `tests/oracles/` and are run by hand; their output is committed as versioned golden data (`halfspace_v1.json`, `rayleigh_v1.json`, `specfem_pilot_s1_v1.json`) that the test suite reads back. Two properties make them worth more than a second opinion:

- **They share no code with the module they check.** The half-space and Rayleigh generators do not import `gragra.analytic` or `gragra.waves`; each re-solves the problem from elasticity and Newton's law — a truncated volume-and-surface integral of the wave-induced mass perturbation, extrapolated toward the infinite-domain limit — rather than re-evaluating the closed form the implementation uses. That independence is asserted, not asserted-in-prose: `tests/test_halfspace_gravity_oracle.py::test_oracle_metadata_independent_of_gragra` checks the recorded provenance, and `tests/test_validation_extra_available.py::test_halfspace_oracle_content_addressed_provenance_is_verified` fails closed on the content hashes of the generator, its inputs and its output.
- **Their agreement is quoted, not implied.** The half-space transfer functions match their oracle to roughly a percent (`rtol=1e-2`, `atol=3e-3`). That is not the precision of the closed form; it is the precision of the *oracle*, which is limited by its own truncation radius and extrapolation. A percent-level agreement between two routes that share no algebra is strong evidence about the formula and weak evidence about its last digits — and the block that cites it says so.

Their blind spot is conventions. A generator re-derives the physics but adopts the same coordinate orientation, phase convention and amplitude normalization as the implementation, because otherwise the two numbers would not be comparable at all. An error that lives in the convention rather than in the algebra passes through both sides untouched. The first worked example below is exactly that kind of error.

The deepest of these comparisons carry the `slow` marker and are deselected from the default run, so an ordinary `pytest` invocation does not execute them.

## Convergence sweeps

The Sphere suite checks one stated case: radius $R=1\,\mathrm{m}$, density $1000\,\mathrm{kg/m^3}$, an exterior target at $(2.7, 0.3, -0.4)\,\mathrm{m}$, and spacings $h=R/[10,20,40,80]$. Each observable must have relative error at most `0.05` at the finest spacing; potential must also improve from the coarsest to the finest spacing. The log–log slope is computed but not asserted. The observed errors and fitted slopes for a separate high-level reproduction are recorded in the internal repository note `docs/validation/nonanalytic_static_sources.md`. These conditions and gates do not define the acceptance rules for other shapes.

The independent closed-form error check can detect a constant bias when it exceeds the stated budget. A fitted slope or a sequence of adjacent differences alone cannot guarantee normalization: both can look regular while preserving a scale error. The Sphere gate also permits any error that remains under its case-specific budget, so read it with the fixed-point evidence and the limits recorded for the route.

## Backend equivalence

The NumPy implementation is the reference implementation. The optional `numba`, `cpp` and `cupy` paths are compared against it on the same inputs, and the comparison is the only thing that licenses using them: they are accelerators, not independent derivations.

This method answers "did the fast path drift from the reference?" and nothing else. Four implementations of one wrong expression agree exactly. Worse, agreement can be reported at tolerances that hide real divergence, so the block for each accelerated path states the tolerance it was compared at rather than the word "equivalent".

Two limits are worth stating outright:

- **The GPU path is verified on a local GPU only, and is not run in CI.** Hosted runners have no GPU, so there is no continuous check of the `cupy` results at all; the comparison against NumPy (`rtol=1e-12`, `atol=1e-15`) is a one-machine measurement by the maintainer. Treat GPU results as unverified until you have reproduced that comparison on your own hardware.
- **Reproducibility on the GPU is weaker than on the CPU paths.** Parallel reduction is not associative, so a result there is reproducible for a fixed device, library version and `chunk_size`, and not bit-identical across them — unlike the CPU paths, where the result does not depend on `chunk_size` at all. [Chunking, softening and tolerances](../concepts/numerics.md) states the contract; [Compute backends](../guide/backends.md) states the failure modes.

## Mutation testing of the checks

The checks that guard external data are themselves code, and a check that cannot fail is worse than no check, because it is reported as a pass. `tests/test_gate_mutation.py` corrupts copies of real exported data — permuting displacement between grid points, altering values at points that must agree, removing a required provenance key — and asserts that the corresponding check raises.

The suite is unusually candid about the results. Alongside the tests that demonstrate detection, it pins the *absence* of detection where that is the true state of affairs: `test_a4_cg_continuity_blind_to_interior_only_displacement_corruption` and `test_a4_check_blind_to_displacement_index_permutation_between_interior_points` exist to fail if the blind spot ever silently changes shape. A documented blind spot is a limitation you can read; an unrecorded one is a surprise waiting for the person who relies on the check.

Two boundaries apply. First, mutation testing measures a check, never the physics behind it. Second, only implemented checks can be mutated: today that is the exporter's own structural checks (internally "Tier A") and the explicit scientific input checks exercised by the suite. The fixed SPECFEM3D 12/18 km acceleration comparison is implemented as a
research script, but its mandatory nested correspondence gate rejected the
pair; no general solver or domain-convergence benchmark is established. The
scope of external-solver evidence is set out in
[SPECFEM3D Cartesian](../external/specfem3d.md).

## Two worked examples of a blind spot

Both examples are drawn from gragra's own convention records, where they were written down as the reason a particular test had to exist.

**A magnitude test that cannot see a broken ellipse.** In a Rayleigh wave the horizontal and vertical displacement components are in quadrature: they differ by 90°, a factor of $i$. Implement them as real and in phase and the particle motion collapses from an ellipse onto a line — a qualitatively different physical field. Every test written on $|u|$, on component magnitudes, or on energy still passes, because the moduli are unchanged. Only a test that inspects the *relative phase* fails. That is why the sign of $\mathrm{Im}(u_x \bar{u}_z)$ is pinned directly, negative at the surface (retrograde motion) and positive below the horizontal-displacement node (prograde), in `tests/test_wave_rayleigh.py::test_particle_motion_retrograde_then_prograde`, next to the surface-ellipticity golden in `::test_surface_ellipticity_golden`. Note what this defeats: an independent oracle that adopts the same phase convention reproduces the same broken ellipse, and a convergence sweep of it converges perfectly.

**A factor of $G^2$ that no text search can find.** Second-order coherent statistics carry different powers of the gravitational constant: the witness displacement is $G^0$, the gravitoelastic tensor $c_{ij} = \langle \delta a_i \xi_j^* \rangle$ is $G^1$, and a mirror-pair cross-spectral density $\langle \delta a_i \delta a_j^* \rangle$ is $G^2$. Multiply once too often, or once too seldom, and the answer is wrong by about ten orders of magnitude — yet searching the kernel and coherent layers for the constant finds nothing, because those layers are correctly $G$-free and the error lives in the observables layer, where multiplying by $G$ is exactly what the code is supposed to do. A relative comparison against a same-order reference is equally blind. What catches it is an *absolute-value* golden derived by hand: `tests/test_coherent_gravitoelastic.py::test_gc1_c_ij_absolute_golden` pins $c_{xx} = 2\times10^{-9}\,G\,(\sin 1 + i\cos 1)$ and `::test_gc2_csd_absolute_golden` pins $\mathrm{CSD}_{xx} = 10^{-6}\,G^2$, each at `rtol=1e-12`. Ten orders of magnitude cannot hide from either.

The general lesson is the one the table encodes: choose the check by the error you are trying to exclude, and write down the errors it cannot exclude.

## How to read a Verification Block

A Verification Block is the eight-field table that appears under a symbol's own anchored heading on the page that documents it — for example [`point_acceleration`](../guide/observables.md#status-point-acceleration). The schema is fixed, and so is the vocabulary of two of its fields.

| Field | What it tells you |
|---|---|
| **Status** | One of `Stable`, `Experimental`, `Diagnostic only`, `Not implemented`. Nothing else is permitted, so "mostly working" cannot be expressed |
| **Verified against** | The method, from the closed list below. Several may be combined |
| **Agreement** | The level at which the comparison holds: a tolerance, `exact` (bit-exact arithmetic), or `n/a` with a reason. A method without an agreement level is an anecdote |
| **Test evidence** | `tests/<file>.py::<name>` references. These are checked mechanically: the file must exist and must define that test, so a block cannot cite a test that was renamed or deleted |
| **Conditions** | The assumptions under which the agreement holds — zero softening, a distance regime, a single mode, a named solver profile |
| **Not verified** | What this evidence does *not* cover. The field is mandatory and may not be left empty |
| **Known limits** | Limits that hold regardless of testing: an approximation in the model, a range where the formula does not apply |
| **Scope** | Which parts of the API surface may still change, or `none` |

`Verified against` is restricted to: `closed-form analytic solution`, `independent numerical oracle`, `invariance property`, `convergence sweep`, `backend equivalence`, `published reference` (which **must** carry a DOI), and `none`. The DOI requirement is deliberate: it keeps an unpublished talk, poster or thesis from entering the site as verification evidence, since a reader cannot check what they cannot obtain. `none` is a legitimate value and appears where it is true.

A constant such as `G_SI` carries a reduced two-field block (`Status`, `Source`) instead, because there is no oracle a CODATA value could have been compared against, and eight rows of boilerplate would teach you to skim the blocks that matter.

Finally, read a block together with two other pages. [Literature coverage](coverage.md) tells you whether your physical case is one of the ones with a dedicated treatment at all — a block cannot exist for a formula that was never implemented. [Limitations and non-guarantees](limitations.md) collects what no block covers, including the code paths that no continuous check ever executes.
