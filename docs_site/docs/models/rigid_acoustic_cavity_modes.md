<!-- GENERATED MODEL REFERENCE: regenerate through the catalogue workflow. -->
# Rigid acoustic cavity modes

## Model overview

**Level:** — (definition and assignment pending).

**Physical domain:** atmospheric acoustic  
**Geometry:** `cavity`

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

Acoustic eigenmodes in a rigid cavity filled with air of sound speed \(c_s\) satisfy the Helmholtz equation \((\nabla^2 + k^2)\Psi = 0\) with rigid Neumann boundary condition:

$$
\left. \frac{\partial \Psi}{\partial n} \right|_{\partial V} = 0.
$$

The modal eigenfrequency is \(f = c_s k / (2\pi)\) (the static zero mode \(k = 0\) is excluded).

**1. Rectangular cuboid cavity** (\([-L_j/2, L_j/2]\) along each axis \(j \in \{x, y, z\}\)) with mode indices \((n_x, n_y, n_z)\):

$$
\Psi(\mathbf x) = N \prod_{j \in \{x,y,z\}} \cos\left[\frac{n_j \pi (x_j + L_j/2)}{L_j}\right], \qquad
k = \pi \sqrt{\sum_{j=x,y,z} \left(\frac{n_j}{L_j}\right)^2}
$$

where the spatial RMS normalization factor is:

$$
N = 2^{r/2}
$$

and \(r\) is the number of non-zero mode indices (\(r = \sum_j \mathbf 1_{n_j > 0}\)).

**2. Circular cylinder cavity** (radius \(R\), centered height \(z \in [-H/2, H/2]\)) with mode indices \((m, n, q)\):

$$
\Psi_{mnq}(r, \phi, z) = N_{mnq} J_m\left(\frac{\alpha_{mn} r}{R}\right) A_m(\phi) \cos\left[\frac{q \pi (z + H/2)}{H}\right], \qquad
k = \sqrt{\left(\frac{\alpha_{mn}}{R}\right)^2 + \left(\frac{q\pi}{H}\right)^2}
$$

where \(\alpha_{mn}\) is the \(n\)-th zero of the derivative of the Bessel function (\(J_m'(\alpha_{mn}) = 0\)), \(N_{mnq}\) is the spatial RMS normalization factor, and the azimuthal dependence is selected by `azimuth`:

$$
A_m(\phi) = \begin{cases}
\cos(m\phi), & \text{for } \texttt{azimuth="cos"}, \\
\sin(m\phi), & \text{for } \texttt{azimuth="sin"} \quad (m \ge 1).
\end{cases}
$$

For \(m = 0\), the sine azimuthal branch vanishes and is disallowed by the API.

**3. Spherical cavity** (radius \(R\)) with radial index \(n\) and angular index \(l\):

$$
\Psi(r, \theta, \phi) = j_l(k_{ln} r) Y_{lm}(\theta, \phi)
$$

where \(k_{ln}\) is determined by the boundary condition on the spherical Bessel derivative \(j_l'(k_{ln} R) = 0\).

The acoustic density perturbation is \(\delta\rho = (P_0 / c_s^2) \Psi(\mathbf x)\), and the induced G-free gravitational potential solves the Poisson system:

$$
\nabla^2 \Phi^{(G\text{-free})} = 4\pi \frac{P_0}{c_s^2} \Psi(\mathbf x).
$$

Target points on physical cavity walls, caps, and rims are singular and raise a `ValueError`.

## Parameters and calculation paths

### Parameters and units

#### API parameters

| Function / Path | Parameter | Type / Format | Default | Description |
|---|---|---|---|---|
| `acoustic_cavity_mode` | `geometry` | Sphere \| Cuboid \| Cylinder | required | Cavity boundary geometry instance |
| `acoustic_cavity_mode` | `mode` | SphereAcousticMode \| CuboidAcousticMode \| CylinderAcousticMode | required | Acoustic eigenmode descriptor instance |
| `acoustic_cavity_mode` | `pressure_amplitude` | u.Quantity (pressure) | required | Spatial-RMS modal peak pressure phasor amplitude |
| `acoustic_cavity_mode` | `sound_speed` | u.Quantity (speed) | required (kw-only) | Speed of sound inside cavity air |
| `acoustic_cavity_mode` | `reference_time` | object | required (kw-only) | Time origin for phasor evaluation |
| `acoustic_cavity_mode` | `numerics` | AcousticCavityNumerics | default settings | Numerical convergence controls (max evaluations/refinements) |

#### API return values

| Function / Path | Return type | Representation | Quantity & Unit | Shape / Structure | Description |
|---|---|---|---|---|---|
| `acoustic_cavity_mode` | SphereAcousticCavityModel \| CuboidAcousticCavityModel \| CylinderAcousticCavityModel | Phasor | Φ: m²/s², a: m/s², Γ: s⁻² | EvaluationResult with complex Astropy Quantities | Standing acoustic normal mode in rigid cavity evaluated via .at(targets).run() |

#### Mathematical symbols and notation

| Symbol | Meaning | SI unit | Definition / Relation |
|---|---|---|---|
| \((l, m, n)\) | Mode numbers | dimensionless | Acoustic eigenmode indices |
| \(c_s\) | Sound speed | m/s | Speed of sound inside cavity |
| \(\rho_0\) | Air equilibrium density | kg/m³ | Ambient unperturbed cavity air density |
| \(\delta p_0\) | Pressure amplitude | Pa | Spatial-RMS peak pressure phasor amplitude |
| \(k\) | Modal wavenumber | rad/m | Eigenvalue of Helmholtz equation \((\nabla^2 + k^2)\Psi = 0\) |
| \(f\) | Modal frequency | Hz | Acoustic eigenfrequency \(f = c_s k / (2\pi)\) |
| \(\Psi(\mathbf x)\) | Spatial mode function | dimensionless | Normalized eigenmode satisfying Neumann boundary condition |

#### Usage notes

- Evaluates via `.at(targets).run()` for target points strictly outside the cavity.
- Targets on the physical cavity walls, caps, or rims raise `ValueError`.

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

### Calculation example and output

The following Python snippet evaluates this model with representative parameters and prints the resulting physical quantities:

```python
import astropy.units as u
from gragra.geometry import Frame, Cuboid
from gragra.targets import TargetPoints
from gragra.analytic.atmospheric import acoustic_cavity_mode, CuboidAcousticMode

frame = Frame('world')
geom = Cuboid([0.0, 0.0, 0.0]*u.m, [5.0, 5.0, 5.0]*u.m, frame)
mode = CuboidAcousticMode(1, 0, 0)
model = acoustic_cavity_mode(
    geom, mode, pressure_amplitude=10.0*u.Pa,
    sound_speed=340.0*u.m/u.s, reference_time=(100, 0)
)
res = model.at(TargetPoints([[1.0, 1.0, 1.0]]*u.m, ('t1',), frame)).run()
print('potential:', res.potential.value, res.potential.units)
print('acceleration:', res.acceleration.value, res.acceleration.units)
print('gradient:', res.gradient.value, res.gradient.units)
```

Representative output:

```text
potential: [[[7.01645514e-14+0.j]]] [[Unit("m2 / s2")]]
acceleration: [[[-5.60469039e-14+0.j]
  [ 7.45364444e-15+0.j]
  [ 7.45364444e-15+0.j]]] [[Unit("m / s2") Unit("m / s2") Unit("m / s2")]]
gradient: [[[ 4.15877681e-14+0.j]
  [ 5.74388019e-15+0.j]
  [ 5.74388019e-15+0.j]
  [ 5.74388019e-15+0.j]
  [ 9.36126236e-15+0.j]
  [-1.03724514e-15+0.j]
  [ 5.74388019e-15+0.j]
  [-1.03724514e-15+0.j]
  [ 9.36126236e-15+0.j]]] [[Unit("1 / s2") Unit("1 / s2") Unit("1 / s2") Unit("1 / s2")
  Unit("1 / s2") Unit("1 / s2") Unit("1 / s2") Unit("1 / s2")
  Unit("1 / s2")]]
```

## Derivation and sources

### Derivation

Equations documented; test/oracle code exists but is not a proof or newly run result.

- Cylinder tests check eigenfrequency, normalization and derivative-related properties; saved oracle result paths not fully assembled.

### References

Existing public documentation and original formulation context:

- [Analytic models](../guide/gen1-analytic-models.md)

**Bibliographic inventory:** located.

- Sphere: an unpublished design note (approved eigenmode, RMS normalization, potential and Cartesian derivatives); `G_SI` applied once.
- Cuboid: an unpublished design note (separable source-volume Gaussian transform, interval derivatives, integrated-by-parts tails); an unpublished design note (`gen1_cavity_v1` runtime estimates and componentwise rigorous tails).
- Cylinder: an unpublished design note (real-mode normalization, exact overlaps, gap-selected transforms); #cylinder-rigorous-tail-coefficients (outward omitted-integral bounds); `G_SI` applied once.
- No external bibliographic DOI is recorded by these entries.

## Verification and applicability

### Verification

**Evidence classes are distinct:** source declarations, catalogue output-contract probes, existing test code, documentary validation claims, and saved numerical results.

- Representative catalogue output-contract entries: 0.
- Catalogue probe state: Not assessed (no catalogue output probe registered).
- Registered output-contract probes run during catalogue validation check selected API outputs only; they do not measure scientific accuracy.
- Scientific accuracy tests are not part of documentation generation; no accuracy Level is assigned.

**Existing verification inventory:**

Shape-specific high-level tests; not run here.

- Cylinder frequency \(f = c_s k / (2\pi)\) rel=2e-15; gradient/Hessian finite differences use rtol 2e-8/3e-7 in selected checks. Test-code-only; other shapes have separate test suites.

### Applicability and limitations

- sphere derivative-root indices; cylinder m=0 sine azimuth undefined; cuboid all-zero constant mode unsupported.

**Scope note:** zero mode excluded; matching mode descriptor required; boundary targets restricted.

See the public verification and limitations guidance for test-specific conditions and interpretation:

- [Verification matrix](../verification/matrix.md)
- [Verification limitations](../verification/limitations.md)


## Information not yet confirmed

- modal plotting, exact references by geometry, saved run records.

*Publication scope: narrative summaries and links, not private derivation records, internal test artifacts or inventory JSON.*
