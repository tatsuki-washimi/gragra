# Numerical contract

For each GLL point, `gragra` uses this first-order acceleration perturbation:

```text
delta a = G sum_n m_n [u_n / |r_n|^3 - 3 r_n (r_n dot u_n) / |r_n|^5]
r_n = x_n - x_target
m_n = rho_n V_n
```

Here `u_n` is the displacement vector at GLL point `n`.

This is a first-order perturbation about the reference configuration, not a
finite-displacement gravity difference. For each contributing point, use it only
when `|u_n| / |r_n| << 1`. Neglected terms grow with that ratio (for an
individual non-cancelling contribution, they are order `O((|u_n| / |r_n|)^2)`
relative to background gravity). The code does not test this condition.

This contract assumes coordinates and displacements in metres, density in
kg/m³, quadrature volume in m³, and returned acceleration in m/s². The reader
enforces only `field_units.displacement == "m"`. The physical SI meaning of
coordinates, density, and quadrature volume remains a caller or exporter
responsibility. `G_SI` is applied once at the Observable boundary.

`displacement_field_acceleration(dataset, targets, *, chunk_size=None,
backend="numpy")` returns a read-only `(Nt, M, 3)` array. The reader retains a
complete immutable snapshot after closing HDF5; it is not a streaming reader.
The observable calculates one time index at a time.

Rows are ordered by ascending element ID and then GLL index
`flat = k*(ny*nx) + j*nx + i`. Duplicate coordinates are not merged; each
element's quadrature mass contributes separately.

The rc2 field path fixes `softening_m=0`. A target coincident with a GLL point
raises `ValueError`. Targets must be on the vacuum side, such as above a free
surface or inside a cavity; the code does not determine whether a target is in
rock.
