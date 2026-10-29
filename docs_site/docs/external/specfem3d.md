# SPECFEM3D Cartesian

## Bottom line

**Bottom line.** gragra can read a SPECFEM3D Cartesian mesh and write it into
gragra's own HDF5 layout for one narrow solver profile. It can then read a
time-sampled, single-material, non-PML schema-v1 file eagerly with
`read_specfem_gll_hdf5` or in bounded slabs with `SpecfemGLLStreamReader`.
The saved older 6 km Rayleigh case has a three-observable comparison against an
independent halfspace reference. A separate, newly run 12/18 km pair completed
its acceleration calculations, but the required nested correspondence gate
rejected the pair. Neither result makes the adapter a general FEM/SEM importer
or establishes domain convergence.

## What the exporter does

`gragra.adapters.specfem` reads a SPECFEM3D Cartesian solver's own binary
output — its mesh topology (`ibool`), GLL node coordinates, and material
density (`rho.bin`) — and writes them into gragra's HDF5 field schema via
`export_specfem_gll_hdf5`. `ExportConfig` selects which solver run and which
output mode (static or time-sampled) to export.

**An exporter and a narrow reader are available for the documented profile.**

`read_specfem_gll_hdf5` loads the exported canonical layout into a closed,
immutable `SpecfemGLLDataset`. It preserves element then GLL order and does
not deduplicate shared coordinates, because each quadrature contribution is
its own source mass. `displacement_field_acceleration` consumes that dataset
and returns a read-only `(Nt, M, 3)` series with zero softening.
The eager reader closes the HDF5 file after loading every snapshot into memory.
`SpecfemGLLStreamReader` is the context-managed alternative: it validates every
requested sample, preserves element-local GLL IDs, and bounds decoded chunks,
validation buffers, provider destinations, and immutable copies.
Targets must be inside a cavity or above the free surface. Rock-interior
targets require a local term that rc1 does not implement.

The exporter's structural checks (A1–A4) have been run against one real pilot
run and report **5 passed, 1 skipped**, **only for static, mesh-only output**.
Those results describe that input and do not establish behavior for other
meshes, solver versions, or time-series data.

## New matched finite-domain study

Two new solver runs used 12 km and 18 km domains, the same measured
`actual_DT=0.00499999989 s`, native steps 1–240, and fixed acceleration targets.
Both canonical exports and full-case two-window A/B/C analyses completed:
3,456,000 element-local GLL sources in 54 blocks for 12 km and 7,776,000
sources in 122 blocks for 18 km. Their analysis peaks were 1,574,805,504 B
and 2,003,505,152 B, both below the 2 GiB limit. Independent parent checks
recomputed every saved target difference and confirmed the case and input
hash chains. The pair JSON is `S/L12000-L18000-pair.json`; its arithmetic
and identity audit is `verification/S-parent-pair-metrics.json`.

The pair reports `PAIR_COMPARISON_COMPLETE` for saved-vector arithmetic, but
`NESTED_OVERLAP_BITWISE_REJECTED` for the mandatory spatial correspondence.
The element-key match found 7,491 missing and 7,484 extra keys, with no
duplicates. Corresponding static fields could not be compared bytewise, and
zero B element contributions were evaluated (`NOT_EVALUATED`). Consequently
`domain_size_difference_interpretable=false`: the computed cross-domain
acceleration differences are not evidence of a finite-domain effect, stability,
or convergence. The key mismatch alone does not identify a floating-point or
physical mesh cause. The S research verdict is partial, with no relaxed
nested check or new 1% gate. The older 6 km study remains separate.

## What is not available

- **No general reader.** rc1 accepts only time-sampled schema-v1 GLL files
  from the documented elastic, single-material, non-PML profile.
- **No independent second implementation.** Nothing cross-checks the
  exporter's reading of the solver's binary format against a second,
  unrelated implementation.
- **No general per-run qualification.** The pilot checks and the two fixed
  research runs do not qualify other runs, meshes, or SPECFEM3D versions.

**Time-sampled output remains fixed-profile research input.** The older 6 km
artifact records `cadence_provenance="placeholder"`, an unknown exporter code
reference, and no exporter source hash. Its stored bytes and downstream
analysis are repeatable, but raw solver output cannot yet be shown to regenerate
the identical HDF5 file.

## Profile requirements

The exporter only accepts a single, narrow SPECFEM3D `Par_file` profile. It
refuses to run — raising before any data is read — if any of these
conditions is not met: `NGNOD != 8` (only 8-node hexahedral elements),
`ATTENUATION=.true.`, `PML_INSTEAD_OF_FREE_SURFACE=.true.`, or
`SUPPRESS_UTM_PROJECTION` set the wrong way for the coordinate convention the
exporter assumes.

`ATTENUATION=.true.` and `PML_INSTEAD_OF_FREE_SURFACE=.true.` are rejected
as **input-profile requirements, not physical limitations**. Gravity computed
from an attenuating or PML-terminated run is not wrong in principle; it is
simply outside the profile gragra is prepared to make any statement about.
Do not read these two as "gragra cannot handle attenuation".

## A known limit, stated plainly

**A known limit, stated plainly.** Check A4 (`check_a4_masks_scalars_provenance`)
covers masks, scalar invariants, provenance, `element_id` integrity, **and**
displacement continuity at shared element-boundary points. On a static,
mesh-only artifact **the continuity arm does not run at all**: no
`/field/displacement` dataset is written, so the check returns without
asserting anything. It is vacuous, not merely tautological — it has not been
exercised on a static export. On synthetic fixtures the same check
does have real detection power (it is the primary defence against
axis/component transposition). When it was run against a smoke artifact
built from real pilot solver output, it **did not pass**: 0.004%–0.009% of
shared-coordinate points disagreed, with a maximum relative deviation of
roughly 2×10⁻⁷ to 5×10⁻⁷. Timesteps 0–4 showed zero mismatches; divergence
appeared only once non-zero signal arrived. The cause is MPI
domain-decomposition force assembly in the solver — different ranks summing
neighbour-element contributions in different orders at float32 — not a
defect in gragra. But the check does not pass and must not be cited as a
guarantee.

## Status and verification

### `gragra.adapters.specfem` { #status-adapters-specfem }

!!! abstract "Status & verification"

    |  |  |
    |---|---|
    | **Status** | Experimental |
    | **Verified against** | invariance property; independent numerical oracle |
    | **Agreement** | Checks A1–A3 pass on one real pilot run (5 passed, 1 skipped). The older 6 km artifact completed three observables and two windows. The new 12/18 km pair completed full acceleration arithmetic but failed nested correspondence. Check A4's continuity arm does not pass — see [A known limit, stated plainly](#a-known-limit-stated-plainly) |
    | **Test evidence** | `tests/test_specfem_pilot_gate.py::test_pilot_a1_gll_ordering`, `tests/test_specfem_pilot_gate.py::test_pilot_a2_quadrature_volume`, `tests/test_specfem_pilot_gate.py::test_pilot_a3_material_mass`, `tests/test_specfem_pilot_gate.py::test_pilot_a4_masks_scalars_provenance` |
    | **Conditions** | `NGNOD=8`, 5×5×5 GLL, `ATTENUATION=.false.`, `PML_INSTEAD_OF_FREE_SURFACE=.false.`, one elastic material, explicit cadence and provenance |
    | **Not verified** | Agreement with a second independent SPECFEM reader; nested static-field byte identity and domain convergence for the new pair; behaviour on other solver profiles; raw-to-export regeneration of the older 6 km artifact |
    | **Known limits** | The rc1 reader accepts only the documented time-sampled, single-material, non-PML profile. Check A4's continuity arm is either vacuous (static mode) or fails to pass (when displacement data is present) — see above |
    | **Scope** | The subpackage currently supports the documented profile; broader solver configurations require separate research work |
