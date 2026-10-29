
# Coordinates, signs and units

## Coordinate conventions

All positions are Cartesian, in meters, `float64`. Source and target
dataclasses store coordinates as a strict `(N, 3)` table — a bare `(3,)`
array is rejected, not silently promoted to a one-row table. Only the public
observable functions accept an array-like `(3,)` single target and promote
it to `(1, 3)`; the dataclasses themselves never do this promotion. See
[Arrays, dtypes and axes](arrays.md) for the general shape and dtype rules.

## Separation vector and sign conventions

The separation vector used throughout the kernels is `Δ = r'_source − r_target`
(source minus target), and `r = |Δ|` with Plummer-softened
`r_eff² = r² + ε²` for softening `ε ≥ 0`. The three kernel sign conventions
that follow from this choice are:

- **Potential**: `K_Φ = −1/r_eff` (scalar) — negative for a positive mass,
  as expected of an attractive potential well.
- **Acceleration**: `K_a = Δ/r_eff³` (vector) — points from the observation
  point back towards the source.
- **Gravity gradient**: `K_T = 3ΔΔ/r_eff⁵ − I/r_eff³` (tensor), where the
  differentiation is with respect to the *observation* (target) coordinate,
  not the source coordinate.

```python
--8<-- "snippets/quickstart_sign_conventions.py"
```

## Units

SI throughout: meters for length, kilograms for mass, seconds for time
(where time appears at all — see [What gragra computes](model-and-scope.md)
for why it usually doesn't). Potential is in J/kg, acceleration in m/s²,
gravity gradient in s⁻². Angular quantities (wavenumber, phase) are in
radians and rad/m, never degrees or cycles.

## The gravitational constant

`G_SI` is the CODATA gravitational constant, `6.67430 × 10⁻¹¹ m³ kg⁻¹ s⁻²`.
It is a fixed literal, not a fitted or configurable value — there is no way
to run gragra with a different value of G.

### `G_SI` { #status-g-si }

!!! abstract "Status & verification"

    |  |  |
    |---|---|
    | **Status** | Stable |
    | **Source** | CODATA recommended value, `6.67430e-11`, hard-coded in `gragra.constants` and re-exported as `gragra.G_SI`. Not applicable — a physical constant has no oracle to verify it against beyond the value itself |
