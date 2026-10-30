<!-- GENERATED MODEL REFERENCE: regenerate through gragra-dev catalogue workflow. -->
# Spherical cavity and finite spherical shell

## Model overview

**Level:** — (definition and assignment pending).

**Physical domain:** seismic elastic  
**Geometry:** cavity, shell

Elastic displacement response, cavity wall and finite-shell boundary contributions

**Source-declared observable / representation pairs:**

- Φ a Γ — Phasor

These pairs are a union over the listed calculation paths. They do not imply every API exposes every combination.

[Back to model catalogue](../guide/model-and-method-catalogue.md)

## Geometry and equations

- center-only helper targets cavity center. high-level models accept geometry frame/phase origin and target fields; physical boundaries have exclusions per model.

### Equations

**Existing source formula notes:**

- Representative terms: a_bulk=4πGρ[j0(ka)-j0(kR)]ξe; a_total=8πGρ[j1(ka)/(ka)-j1(kR)/(kR)]ξe. Legacy component helper only gives acceleration. General cavity series supports bulk/wall/total Φ,a,Γ at general targets; fixed shell model supports P/S prescribed displacement.

## Parameters and calculation paths

### Parameters and units

- cavity: Sphere, density Quantity, phase_speed Quantity or FrequencySeries, displacement FrequencySeries, representation, reference_time, wave_type, propagation_direction, polarization, phase_origin, part. shell analogous SphericalShell and contribution. legacy: inner_radius_m a, outer_radius_m R, wavenumber_rad_m k, density_kg_m3, displacement_m ξ, direction (3). SI units: m, rad/m, kg/m³, m; output m/s².

### Calculation paths

#### high-level spherical cavity phasor model

- `gragra.analytic.seismic.spherical_cavity`

**Path-level source-declared outputs:** Φ a Γ — Phasor

#### high-level finite spherical shell P/S phasor model

- `gragra.analytic.seismic.spherical_shell`

**Path-level source-declared outputs:** Φ a Γ — Phasor

#### center-only finite-shell P-wave four-component acceleration

- `gragra.observables.spherical_shell_p_acceleration_components`

**Path-level source-declared outputs:** a — Phasor

Factory APIs construct model/evaluation objects; observables may be obtained after the documented .at(...).run() sequence rather than directly from the factory call. Follow the linked API examples.

## Derivation and sources

### Derivation

Finite shell general-point harmonic coefficient derivations are documented; exact CAS residuals are claimed in saved validation record. Cavity-series local derivation/implementation record exists.

- U10 document says 35 exact CAS residuals and general-point finite-interval harmonic coefficients/Cartesian jets are additional derivations. Formula ledger documents adopted fixed spherical cavity series and exact CAS/axis oracle scripts.

### References

Existing public documentation and original formulation context:

- [Analytic benchmarks](../guide/analytic-benchmarks.md)
- [analytic models](../guide/gen1-analytic-models.md)

**Bibliographic inventory:** located.

- ANNA v4 Section 4.1 Eqs.47–53 for finite-shell center P formulas. U10 validation says Harms 2019 continuity/bulk/surface definitions and Reumers v4 P/S center formulas were checked against actual PDFs in that recorded work.

## Verification and applicability

### Verification

**Evidence classes are distinct:** source declarations, catalogue output-contract probes, existing test code, documentary validation claims, and saved numerical results.

- Representative catalogue output-contract entries: 0.
- Catalogue probe state: Not assessed (no catalogue output probe registered).
- This documentation generation does not perform scientific re-execution or assign an accuracy Level.

**Existing verification inventory:**

Existing (internal evidence record) are evidence records only; none rerun. Shell validation records a fixed 24-case matrix and tolerances, but those runs were before candidate and raw artifacts external to repo.

- U10 cases: b=10m, a/b=.2,.8,.99,.9999; kb=.01,.8,5; P/S; rho=2300 kg/m³; ξ=1e-9(1+.4i)m; center/offaxis/inner/material/external targets. Acceptance=1e-12 Q_scale+r_Q norm(reference), r_Q=(1e-10,1e-10,1e-9); ratios recorded, reference &lt;=0.1. No elastic scattering comparison. P center formula test implementation (internal evidence record) Separate legacy P helper documented ANNA formula. Saved artifact root in U10 doc is external `(internal evidence record)`, with BASE_SHA/oracle-matrix-2. This artifact is not reproduced/inspected now.

### Applicability and limitations

- U10 validation explicitly claims no comparison with an elastically scattered field. General spherical cavity series is no-scattering; physical wall excluded. Do not infer static response or PSD/time-series outputs.

**Scope note:** Prescribed plane-wave displacement and finite first-order perturbation; finite shell response does not satisfy elastic scattering conditions. Center-only helper is P-wave acceleration only; high-level shell supports distinct P/S and contribution path.

See the public verification and limitations guidance for test-specific conditions and interpretation:

- [Verification matrix](../verification/matrix.md)
- [Verification limitations](../verification/limitations.md)


## Information not yet confirmed

- catalogue output_contracts empty; no runnable representative probe specified in ledger.
- Saved artifact root external and historical; not re-run or inspected here.
- No ANNA execution or elastically scattered field comparison claimed.

*Publication scope: narrative summaries and links, not private derivation records, internal test artifacts or inventory JSON.*
