<!-- GENERATED MODEL REFERENCE: regenerate through the catalogue workflow. -->
# Small-cavity wall approximation

## Model overview

**Level:** — (definition and assignment pending).

**Physical domain:** seismic elastic  
**Geometry:** cavity

Sphere, cuboid or cylinder; selected linearized wall contribution only

**Source-declared observable / representation pairs:**

- ① Φ a Γ — Static
- ② Φ a Γ — Phasor

These pairs are a union over the listed calculation paths. They do not imply every API exposes every combination.

[Back to model catalogue](../guide/model-and-method-catalogue.md)

## Geometry and equations

- The model accepts Sphere, Cuboid, Cylinder, but plot_shape helper supports only Cuboid, Cylinder, SphericalShell and explicitly rejects Cavity types; high-level small_cavity_wall model is cavity-named and is not automatically plot-able. A standalone Sphere geometry is not among the supported classes.
- Harms scalar table supports five discrete shape/location rows; tensor paths have separate geometric coverage.

### Equations

**Existing source formula notes:**

- High-level small_cavity_wall evaluates selected wall contributions to potential \(\Phi\), acceleration \(\mathbf a\), and gradient \(\mathbf\Gamma\) (static or single-frequency phasor) for cavity wall displacement \(\mathbf u\): \(\Phi_{\mathrm{wall}} = -\mathbf u \cdot \mathbf a_{\mathrm{body}}\), \(\mathbf a_{\mathrm{wall}} = \mathbf\Gamma_{\mathrm{body}} \mathbf u\).
- Harms 2022 scalar wall coupling factor is dimensionless geometric factor \(C\). sca_wall_acceleration computes the acceleration vector from the lookup factor: \(\mathbf a = C G \rho \mathbf u\).
- Tensor contraction G-free: \(\Delta\mathbf a / G = -\rho \mathbf C_g \mathbf u\); SI: \(\Delta\mathbf a = -G \rho \mathbf C_g \mathbf u\). SCA approximation applies only when the cavity is much smaller than source-to-target distance; high-level model displacement must be first-order small.

## Parameters and calculation paths

### Parameters and units

- High-level small_cavity_wall: geometry (Sphere, Cuboid, or Cylinder), displacement (Quantity 3-vector in geometry frame), density (Quantity kg/m^3), optional frequency (Quantity Hz) and reference_time for peak phasor representation.
- Scalar coupling factor sca_wall_coupling_factor: cavern_shape in {'sphere', 'cube', 'half_sphere', 'elongated_cuboid_2to1to1'}, test_mass_location in {'arbitrary', 'center_3d', 'floor_center_2d'}.
- Scalar acceleration sca_wall_acceleration: adds density_kg_m3 (float) and displacement_m (array-like (3,)) to cavern_shape and test_mass_location.

### Calculation paths

#### geometry-aware small_cavity_wall model

- `gragra.analytic.seismic.small_cavity_wall`

**Path-level source-declared output union:** Φ a Γ — Static; Φ a Γ — Phasor

#### Harms 2022 scalar table lookup and acceleration

- `gragra.analytic.sca_wall_acceleration`

**Path-level source-declared output union:** a — Static

#### Harms 2022 dimensionless coupling factor lookup

- `gragra.analytic.sca_wall_coupling_factor`

This path provides a related coefficient, coupling quantity, or helper rather than a direct Φ/a/Γ output.

#### shape-specific tensor SCA numeric paths (related, not the assigned scalar factory)

- `gragra.analytic.sca_wall_coupling_factor`
- `gragra.analytic.seismic.small_cavity_wall`

This path provides a related coefficient, coupling quantity, or helper rather than a direct Φ/a/Γ output.

## Derivation and sources

### Derivation

Scalar Harms coefficients are literal table entries; general tensors are local derivation/numerical surface paths with recorded convention mapping.

- Do not generalize table scalar into all-geometry analytic tensor. Tensor source-map explicitly restricts interpretation and physical-boundary handling.

### References

Existing public documentation and original formulation context:

- [Analytic models](../guide/gen1-analytic-models.md)
- [physics API](../api/physics.md)

**Bibliographic inventory:** located.

- Harms et al., ‘A lower limit for Newtonian-noise models of the Einstein Telescope’ (2022), EPJ Plus 137:687, DOI 10.1140/epjp/s13360-022-02851-z; Eqs 1–4,7, Table 1. Tempel et al. 2025 pp. 9–12,42–46 recorded for tensor convention. This inventory records repository citations, not independent paper inspection.
- [DOI 10.1140/epjp/s13360-022-02851-z](https://doi.org/10.1140/epjp/s13360-022-02851-z)

## Verification and applicability

### Verification

**Evidence classes are distinct:** source declarations, catalogue output-contract probes, existing test code, documentary validation claims, and saved numerical results.

- Representative catalogue output-contract entries: 0.
- Catalogue probe state: Not assessed (no catalogue output probe registered).
- Registered output-contract probes run during catalogue validation check selected API outputs only; they do not measure scientific accuracy.
- Scientific accuracy tests are not part of documentation generation; no accuracy Level is assigned.

**Existing verification inventory:**

Test code, prose verification, and saved run metadata separated; no new run executed.

- [`tests/test_analytic_sca_tensor.py`](https://github.com/tatsuki-washimi/gragra/blob/main/tests/test_analytic_sca_tensor.py) implements independent surface quadrature, target convergence, boundary/rotation/complex displacement checks. Reported tolerance: norm(current-previous)&lt;=1e-12+1e-9*max(norm(current),1), order doubles up to 128. Historical focused runs are not rerun here. The validation inventory records the scalar-focused command as successful. No formal execution performed.

### Applicability and limitations

- Harms floor-center table row does not specify boundary condition/tensor direction/one-sided limit. Scalar lookup does not carry k. high-level model excludes physical wall. No arbitrary model-wide formula/plot capability.

**Scope note:** Wall-only first-order SCA; small cavity relative to source-target distance; no bulk or scattering term. Scalar lookup only five combos; tensor APIs own separate boundaries and geometry scopes.

See the public verification and limitations guidance for test-specific conditions and interpretation:

- [Verification matrix](../verification/matrix.md)
- [Verification limitations](../verification/limitations.md)


## Information not yet confirmed

- No output_contracts in catalog group; no runnable representative catalogue probe defined.
- Exact all five scalar coefficient values omitted from this record.
- No new science run or artifact inspection; prose historical claims remain un-reproduced.

*Publication scope: narrative summaries and links, not private derivation records, internal test artifacts or inventory JSON.*
