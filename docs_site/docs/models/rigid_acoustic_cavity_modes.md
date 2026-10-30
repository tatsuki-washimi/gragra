<!-- GENERATED MODEL REFERENCE: regenerate through gragra-dev catalogue workflow. -->
# Rigid acoustic cavity modes

**Validation level:** Unassigned (the scale and assignment criteria have not yet been defined).

**Physical domain:** atmospheric acoustic  
**Geometry:** cavity

Neumann modes in a sphere, cuboid or cylinder; geometry-dependent normalization and numerical limits

**Source-declared observable / representation pairs:**

- Φ a Γ — Phasor

These pairs are a union over the listed calculation paths. They do not imply every API exposes every combination.

[Back to model catalogue](../guide/model-and-method-catalogue.md)

## Geometry

- No arbitrary modal amplitude plot; selected static Cuboid/Cylinder/SphericalShell plot capability only.

## Equations

**Existing source formula notes:**

- Cuboid Ψ=N∏cos(n_jπ(x_j+L_j/2)/L_j); k²=Σ(n_jπ/L_j)²; N=2^(r/2).
- Cylinder Ψ∝J_m(α_mn r/R)A_m(φ)cos(qπ(z+H/2)/H), J′_m(α)=0.
- Sphere uses nth strictly positive root of spherical Bessel derivative j′_l.

## Parameters and units

- Sphere R m and l,m,n; cuboid Lx,Ly,Lz m and nx,ny,nz; cylinder R,H m and m,n,q,azimuth; pressure Pa spatial-RMS peak phasor; sound_speed m/s; f derived.

## APIs and calculation paths

### Sphere Neumann mode

- `gragra.analytic.atmospheric.acoustic_cavity_mode`

**Path-level source-declared outputs:** Φ a Γ — Phasor

### Cuboid Neumann mode

- `gragra.analytic.atmospheric.acoustic_cavity_mode`

**Path-level source-declared outputs:** Φ a Γ — Phasor

### Cylinder Neumann mode

- `gragra.analytic.atmospheric.acoustic_cavity_mode`

**Path-level source-declared outputs:** Φ a Γ — Phasor

Factory APIs construct model/evaluation objects; observables may be obtained after the documented .at(...).run() sequence rather than directly from the factory call. Follow the linked API examples.

## Derivation and provenance

**Inventory status:** partial.

Equations documented; test/oracle code exists but is not a proof or newly run result.

The page cites public sources below. A fully typeset public derivation is a separate editorial item wherever the existing source material has not been reproduced here.

## References

Existing public documentation and original formulation context:

- [Analytic models](../guide/gen1-analytic-models.md)

**Bibliographic inventory:** located.

- Sphere: (internal evidence record)#sphere-acoustic-cavity (approved eigenmode, RMS normalization, potential and Cartesian derivatives); G_SI applied once.
- Cuboid: (internal evidence record)#cuboid-six-faces (separable source-volume Gaussian transform, interval derivatives, integrated-by-parts tails); (internal evidence record) (gen1_cavity_v1 runtime estimates and componentwise rigorous tails).

## Verification evidence

**Evidence classes are distinct:** source declarations, catalogue output-contract probes, existing test code, documentary validation claims, and saved numerical results.

- Representative catalogue output-contract entries: 0.
- Catalogue probe state: Not assessed (no catalogue output probe registered).
- This documentation generation does not perform scientific re-execution or assign an accuracy Level.

**Existing verification inventory:**

Shape-specific high-level tests; not run here.

- Cylinder frequency f=c_s k/(2π) rel=2e-15; gradient/Hessian finite differences use rtol 2e-8/3e-7 in selected checks. Test-code-only; other shapes have separate test suites.

## Applicability and limitations

- sphere derivative-root indices; cylinder m=0 sine azimuth undefined; cuboid all-zero constant mode unsupported.

**Scope note:** zero mode excluded; matching mode descriptor required; boundary targets restricted.

See the public verification and limitations guidance for test-specific conditions and interpretation:

- [Verification matrix](../verification/matrix.md)
- [Verification limitations](../verification/limitations.md)

## Information still to locate or publish

- modal plotting, exact references by geometry, saved run records.

*Publication scope: narrative summaries and links, not private derivation records, internal test artifacts or inventory JSON.*
