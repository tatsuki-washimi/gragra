<!-- GENERATED MODEL REFERENCE: regenerate through the catalogue workflow. -->
# Rigid acoustic cavity modes

## Model overview

**Level:** — (definition and assignment pending).

**Physical domain:** atmospheric acoustic  
**Geometry:** cavity

Neumann modes in a sphere, cuboid or cylinder; geometry-dependent normalization and numerical limits

**Source-declared observable / representation pairs:**

- Φ a Γ — Phasor

These pairs are a union over the listed calculation paths. They do not imply every API exposes every combination.

[Back to model catalogue](../guide/model-and-method-catalogue.md)

## Geometry and equations

![Rigid acoustic cavity modes schematic](../assets/models/rigid_acoustic_cavity_modes.svg)

*Standing acoustic modes in a rectangular cavity of dimensions Lx × Ly × Lz with rigid boundary conditions.*

- No arbitrary modal amplitude plot; selected static Cuboid/Cylinder/SphericalShell plot capability only.

### Equations

**Analytical formulation and field equations:**

- Modes satisfy the Helmholtz equation \((\nabla^2 + k^2)\Psi = 0\) with rigid Neumann boundary condition \(\partial_n \Psi = 0\) on the cavity boundary; frequency is derived as \(f = c_s k / (2\pi)\). The static zero mode (\(k=0\)) is excluded.
- Cuboid modes: \(\Psi = N \prod_{j\in\{x,y,z\}} \cos\left(\frac{n_j \pi (x_j + L_j/2)}{L_j}\right)\) with \(k^2 = \sum_j (n_j \pi / L_j)^2\) and \(N = 2^{r/2}\) where \(r\) is the number of non-zero indices, giving unit spatial RMS.
- Cylinder modes: \(\Psi \propto J_m(\alpha_{mn} r/R) A_m(\phi) \cos(q\pi (z+H/2)/H)\) with \(J\'_m(\alpha_{mn}) = 0\) and \(k^2 = (\alpha_{mn}/R)^2 + (q\pi/H)^2\).

## Parameters and calculation paths

### Parameters and units

- geometry: Cavity geometry instance (Sphere with radius R; Cuboid with edge lengths Lx, Ly, Lz; Cylinder with radius R and height H); mode: Mode descriptor (SphereAcousticMode(l, m, n), CuboidAcousticMode(nx, ny, nz), CylinderAcousticMode(m, n, q, azimuth)); pressure_amplitude_pa: complex or float, spatial-RMS peak pressure phasor in Pa; sound_speed_m_s: float, positive speed of sound in m/s; targets_m: array-like (M, 3) in cavity local frame.

### Calculation paths

#### Sphere Neumann mode

- `gragra.analytic.atmospheric.acoustic_cavity_mode`

**Path-level source-declared output union:** Φ a Γ — Phasor

#### Cuboid Neumann mode

- `gragra.analytic.atmospheric.acoustic_cavity_mode`

**Path-level source-declared output union:** Φ a Γ — Phasor

#### Cylinder Neumann mode

- `gragra.analytic.atmospheric.acoustic_cavity_mode`

**Path-level source-declared output union:** Φ a Γ — Phasor

## Derivation and sources

### Derivation

Equations documented; test/oracle code exists but is not a proof or newly run result.

- Cylinder tests check eigenfrequency, normalization and derivative-related properties; saved oracle result paths not fully assembled.

### References

Existing public documentation and original formulation context:

- [Analytic models](../guide/gen1-analytic-models.md)

**Bibliographic inventory:** located.

- Sphere: an unpublished design note (approved eigenmode, RMS normalization, potential and Cartesian derivatives); G_SI applied once.
- Cuboid: an unpublished design note (separable source-volume Gaussian transform, interval derivatives, integrated-by-parts tails); an unpublished design note (gen1_cavity_v1 runtime estimates and componentwise rigorous tails).

## Verification and applicability

### Verification

**Evidence classes are distinct:** source declarations, catalogue output-contract probes, existing test code, documentary validation claims, and saved numerical results.

- Representative catalogue output-contract entries: 0.
- Catalogue probe state: Not assessed (no catalogue output probe registered).
- Registered output-contract probes run during catalogue validation check selected API outputs only; they do not measure scientific accuracy.
- Scientific accuracy tests are not part of documentation generation; no accuracy Level is assigned.

**Existing verification inventory:**

Shape-specific high-level tests; not run here.

- Cylinder frequency f=c_s k/(2π) rel=2e-15; gradient/Hessian finite differences use rtol 2e-8/3e-7 in selected checks. Test-code-only; other shapes have separate test suites.

### Applicability and limitations

- sphere derivative-root indices; cylinder m=0 sine azimuth undefined; cuboid all-zero constant mode unsupported.

**Scope note:** zero mode excluded; matching mode descriptor required; boundary targets restricted.

See the public verification and limitations guidance for test-specific conditions and interpretation:

- [Verification matrix](../verification/matrix.md)
- [Verification limitations](../verification/limitations.md)


## Information not yet confirmed

- modal plotting, exact references by geometry, saved run records.

*Publication scope: narrative summaries and links, not private derivation records, internal test artifacts or inventory JSON.*
