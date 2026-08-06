"""Diagnostic helper: divergence (bulk+surface) vs dipole/total equivalence.

Cross-checks the existing bulk+surface divergence path (fields/ +
acceleration_contract on delta-rho mass elements, dispatched via
integrators.direct_point_sum) against the dipole/total
(vector-weight) path (kernels.inverse_square.dipole_contract, dispatched
via integrators.direct_point_dipole_sum). See kernel_contract.md Section
2.2 for the six mathematical preconditions under which the two paths are
expected to agree -- this module does NOT verify those preconditions (see
CALLER RESPONSIBILITY in dipole_consistency_check's docstring); a nonzero
residual for a mesh/config that violates them is EXPECTED, not a bug.

Both accelerations returned by this module are G-free (kernels/integrators
layer convention; G_SI is not applied) -- this is a kernel-layer
diagnostic, not a public observable.

The equivalence itself is an exact continuum-level identity (a divergence
theorem rewriting of the linearized bulk density perturbation
delta_rho = -rho0 * div(u)), not a far-field approximation. In the
discrete implementation, both paths approximate the continuum integrals by
point sources at tetra centroids, so a nonzero residual for a mesh/config
that DOES satisfy the six preconditions reflects mesh discretization error
(shrinking with mesh refinement and/or target distance), not a violation
of the underlying identity.

Complex outputs (when the volume field's displacement is complex)
represent linear complex amplitudes or transfer-function-like responses,
not directly real-valued time-domain gravitational fields.
"""

from dataclasses import dataclass

import numpy as np

from gragra.array_types import ComplexArray, FloatArray
from gragra.fields.surface import SurfaceDisplacementField
from gragra.fields.unstructured import UnstructuredDisplacementField
from gragra.integrators import direct_point_dipole_sum, direct_point_sum


@dataclass(frozen=True)
class DipoleConsistencyReport:
    """G-free comparison between the divergence (bulk+surface) and
    dipole/total (vector-weight) paths.

    See kernel_contract.md Section 2.2 for the six equivalence
    preconditions (closed volume, consistent winding, outward normal,
    delta_rho sign convention, geometric boundary match, vacuum
    truncation) -- a nonzero residual for a mesh/config that violates any
    precondition is EXPECTED, not a bug (this is the diagnostic's purpose,
    see :func:`dipole_consistency_check`).

    Complex outputs (when the volume field's displacement is complex)
    represent linear complex amplitudes or transfer-function-like
    responses, not directly real-valued time-domain gravitational fields.

    Attributes
    ----------
    acceleration_bulk_surface : FloatArray or ComplexArray, shape (M, 3)
        G-free acceleration from the divergence (bulk [+ surface]) path.
    acceleration_dipole : FloatArray or ComplexArray, shape (M, 3)
        G-free acceleration from the dipole/total path.
    abs_residual : FloatArray, shape (M,)
        ``norm(acceleration_bulk_surface - acceleration_dipole, axis=-1)``.
    rel_residual : FloatArray, shape (M,)
        ``abs_residual / norm(acceleration_dipole, axis=-1)``, with an
        explicit rule at targets where ``acceleration_dipole`` is exactly
        zero (a fixed-epsilon floor is deliberately NOT used there: gragra
        spans many magnitude scales with no canonical unit, so any fixed
        floor either fails to guard the zero or overflows to ``+inf`` for
        realistic ``abs_residual`` magnitudes -- see numerical.md's
        zero-division guard rule): ``0.0`` if ``abs_residual`` is also
        (numerically) zero
        (perfect agreement), otherwise ``+inf`` (a clean, well-defined
        signal of total disagreement). Always inspect ``abs_residual``
        directly to interpret this degenerate case.

    All fields are read-only (``flags.writeable is False``) and shape-checked
    against each other (matching leading (M,) length) in ``__post_init__``.
    """

    acceleration_bulk_surface: FloatArray | ComplexArray
    acceleration_dipole: FloatArray | ComplexArray
    abs_residual: FloatArray
    rel_residual: FloatArray

    def __post_init__(self):
        bulk_surface = np.asarray(self.acceleration_bulk_surface)
        dipole = np.asarray(self.acceleration_dipole)
        abs_res = np.asarray(self.abs_residual)
        rel_res = np.asarray(self.rel_residual)

        if bulk_surface.ndim != 2 or bulk_surface.shape[-1] != 3:
            raise ValueError(
                "acceleration_bulk_surface must have shape (M, 3), got "
                f"{bulk_surface.shape}"
            )
        if dipole.ndim != 2 or dipole.shape[-1] != 3:
            raise ValueError(
                f"acceleration_dipole must have shape (M, 3), got {dipole.shape}"
            )
        if abs_res.ndim != 1:
            raise ValueError(f"abs_residual must have shape (M,), got {abs_res.shape}")
        if rel_res.ndim != 1:
            raise ValueError(f"rel_residual must have shape (M,), got {rel_res.shape}")

        m = bulk_surface.shape[0]
        if not (dipole.shape[0] == abs_res.shape[0] == rel_res.shape[0] == m):
            raise ValueError(
                "acceleration_bulk_surface, acceleration_dipole, abs_residual, "
                "and rel_residual must share the same leading (M,) length: "
                f"{m}, {dipole.shape[0]}, {abs_res.shape[0]}, {rel_res.shape[0]}"
            )

        for name, arr in (
            ("acceleration_bulk_surface", bulk_surface),
            ("acceleration_dipole", dipole),
            ("abs_residual", abs_res),
            ("rel_residual", rel_res),
        ):
            locked = arr.copy()
            locked.setflags(write=False)
            object.__setattr__(self, name, locked)


def dipole_consistency_check(
    volume_field: UnstructuredDisplacementField,
    surface_field: SurfaceDisplacementField | None,
    rho0_kg_m3: float,
    delta_rho_kg_m3: float | None,
    targets_xyz: list | tuple | np.ndarray,
    *,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
) -> DipoleConsistencyReport:
    """Cross-check the divergence path against the dipole/total path.

    Path A (divergence, existing):
        ``volume_field.as_density_perturbation_source(rho0_kg_m3)``
        -> ``direct_point_sum(..., observable="acceleration")`` (scalar
        weight, mass elements) + (if ``surface_field is not None``)
        ``surface_field.as_surface_mass_source(delta_rho_kg_m3)``
        -> ``direct_point_sum(..., observable="acceleration")``, summed
        into the volume contribution.
    Path B (dipole):
        ``volume_field.as_dipole_source(rho0_kg_m3)`` (background mass
        elements ``mass_n = rho0 * vol_n`` at undisturbed tetra centroids)
        -> ``direct_point_dipole_sum``. No surface term -- the
        dipole/total formula is a pure volume integral (ANNA eq. 15/17);
        it is not summed with any surface contribution.

    ``surface_field=None`` (with ``delta_rho_kg_m3=None``) is a valid,
    intentional "bulk-only" diagnostic mode for a volume mesh with no
    modeled interface -- it is not merely a device for constructing a
    deliberate non-equivalence probe.

    CALLER RESPONSIBILITY (kernel_contract.md Section 2.2): equivalence
    between Path A and Path B requires a closed volume mesh, consistent
    triangle winding, outward-pointing surface normals,
    ``delta_rho_kg_m3`` following ``fields/surface.py``'s sign convention,
    geometric coincidence of the volume boundary and surface mesh, and
    vacuum truncation outside the boundary. This function does NOT verify
    these preconditions (no automatic winding/outward check is run) --
    callers must confirm them independently, e.g. via
    ``SurfaceDisplacementField.check_orientation()`` /
    ``OrientationReport`` before calling this function on data expected to
    satisfy equivalence. Note that ``OrientationReport.is_outward`` is
    only a reliable outward-orientation certificate for a single connected
    component -- for a multi-component closed mesh a single boolean cannot
    certify every component is outward (see
    ``fields/surface_orientation.py``'s ``OrientationReport`` docstring).
    Calling this function on an open sub-volume is valid and expected to
    show a large, non-vanishing residual -- that is the diagnostic's
    purpose, not a bug.

    Output is G-free (``G_SI`` is not applied) -- this is a kernel-layer
    diagnostic, not a public observable; the residual between the two
    G-free paths is unaffected by ``G_SI`` (a common multiplicative
    factor), so the comparison need not multiply by it.

    Parameters
    ----------
    volume_field : UnstructuredDisplacementField
        Bulk displacement field supplying both Path A's bulk term
        (``as_density_perturbation_source``) and Path B's dipole source
        (``as_dipole_source``).
    surface_field : SurfaceDisplacementField or None
        Optional interface displacement field supplying Path A's surface
        term. If ``None``, Path A is bulk-only (a valid diagnostic mode,
        not merely a non-equivalence probe); ``delta_rho_kg_m3`` must also
        be ``None`` in that case.
    rho0_kg_m3 : float
        Uniform background density in kg/m^3, forwarded to both
        ``as_density_perturbation_source`` and ``as_dipole_source`` (each
        validates it independently: real, finite, non-negative).
    delta_rho_kg_m3 : float or None
        Density jump across the interface in kg/m^3, forwarded to
        ``surface_field.as_surface_mass_source``. Must be provided (not
        ``None``) exactly when ``surface_field`` is not ``None`` --
        specifying only one of the pair raises ``ValueError`` (a silent
        ignore of the caller's intended surface term is avoided).
    targets_xyz : array-like, shape (M, 3)
        Target coordinates in meters. Validated independently by the
        underlying ``direct_point_sum`` / ``direct_point_dipole_sum``
        calls.
    softening_m : float, optional
        Plummer softening in meters, forwarded to all kernel calls.
        Defaults to 0.0.
    chunk_size : int, optional
        Chunk size forwarded to all kernel calls. Defaults to None.

    Returns
    -------
    DipoleConsistencyReport
    """
    if (surface_field is None) != (delta_rho_kg_m3 is None):
        raise ValueError(
            "surface_field and delta_rho_kg_m3 must both be provided or "
            "both be None (surface_field=None with delta_rho_kg_m3=None is "
            "the valid bulk-only diagnostic mode; providing only one of the "
            "pair is rejected to avoid silently dropping the caller's "
            "intended surface term)"
        )

    # Path A: divergence (bulk [+ surface])
    bulk_source = volume_field.as_density_perturbation_source(
        rho0_kg_m3
    ).as_weighted_source()
    acceleration_bulk_surface = direct_point_sum(
        bulk_source,
        targets_xyz,
        observable="acceleration",
        softening_m=softening_m,
        chunk_size=chunk_size,
    )
    if surface_field is not None:
        surface_source = surface_field.as_surface_mass_source(
            delta_rho_kg_m3
        ).as_weighted_source()
        # Non-in-place addition: direct_point_sum returns a read-only
        # array (acceleration_contract sets write=False), so "+=" would
        # raise ValueError: output array is read-only.
        acceleration_bulk_surface = acceleration_bulk_surface + direct_point_sum(
            surface_source,
            targets_xyz,
            observable="acceleration",
            softening_m=softening_m,
            chunk_size=chunk_size,
        )

    # Path B: dipole/total (pure volume integral, no surface term)
    dipole_positions, dipole_mass, dipole_displacement = volume_field.as_dipole_source(
        rho0_kg_m3
    )
    acceleration_dipole = direct_point_dipole_sum(
        dipole_mass,
        dipole_displacement,
        dipole_positions,
        targets_xyz,
        softening_m=softening_m,
        chunk_size=chunk_size,
    )

    abs_residual = np.linalg.norm(
        acceleration_bulk_surface - acceleration_dipole, axis=-1
    )
    dipole_norm = np.linalg.norm(acceleration_dipole, axis=-1)
    # Explicit zero-division handling instead of a fixed-epsilon floor
    # (numerical.md's zero-division guard rule): dividing by a tiny epsilon
    # overflows to +inf for any abs_residual not itself tiny (a realistic
    # case in a G-free, unit-agnostic kernel), which silently violated the
    # intended "finite, not inf/nan" contract. At a target where dipole_norm is
    # exactly zero: rel_residual is 0.0 if abs_residual is also zero
    # (perfect agreement), else +inf (a clean disagreement signal instead
    # of an arbitrary large finite number).
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_residual = np.where(
            dipole_norm > 0.0,
            abs_residual / dipole_norm,
            np.where(abs_residual == 0.0, 0.0, np.inf),
        )

    return DipoleConsistencyReport(
        acceleration_bulk_surface=acceleration_bulk_surface,
        acceleration_dipole=acceleration_dipole,
        abs_residual=abs_residual,
        rel_residual=rel_residual,
    )
