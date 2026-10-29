
# Minimal example

## A single point mass

The smallest complete gragra calculation: one point mass, one observation
point, one call.

```python
--8<-- "snippets/quickstart_point_mass_acceleration.py"
```

## Reading the result

`point_acceleration` returns a `(1, 3)` array — one row per observation
point, the vector's three components trailing. The dtype is `float64`
because the source's mass is a real number; a complex-weighted source (see
[Complex outputs](../concepts/complex-outputs.md)) produces a `complex128`
result instead. The returned array is read-only: `acceleration_m_s2.flags.writeable`
is `False`, so a caller that needs to modify it must copy it first.

## What the numbers mean

The acceleration points from the observation point back towards the mass —
here, along −x — with magnitude `G_SI · M / r²`. At `r = 1 m` and `M = 1 kg`
that magnitude is exactly `G_SI`, so the snippet above checks equality
rather than a tolerance: no rounding is involved at this particular
fixed point. See [Coordinates, signs and units](../concepts/conventions.md)
for why the sign is what it is, and
[Chunking, softening and tolerances](../concepts/numerics.md) for how to
compare results that are *not* exact fixed points.

## Next steps

- Replace the single `PointMass` with a `WeightedPointSource` to use signed
  or complex weights — see [Point sources](../guide/point-sources.md).
- Replace the point mass with a discretized volume or surface — see
  [Geometry and discretization](../guide/geometry.md) and
  [Volume and surface sources](../guide/continuous-sources.md).
- Compute the other observables (potential, projected acceleration,
  gravity-gradient tensor) — see [Observables](../guide/observables.md).
- Pick an accelerated backend for a large source count — see
  [Compute backends](../guide/backends.md).
