"""Batched field-import helpers for time/frequency-axis processing.

Memory Strategy:
----------------
To process batch displacement snapshots efficiently, we avoid copying the entire
multi-snapshot (P, ...) array. Instead, we use slice views of the input array
to delegate construction of single-snapshot fields (DisplacementGrid,
UnstructuredDisplacementField, or SurfaceDisplacementField) sequentially.
The output weights are accumulated and materialized as a single (N, P) array.

Disclaimer on Complex Outputs (Linear Harmonic Amplitudes):
-----------------------------------------------------------
If the input displacement fields contain complex values (e.g., np.complex128),
the resulting weights will also be complex-valued. These represent linear
harmonic amplitude responses (transfer function responses) rather than real-time
physical fields.
"""

import numpy as np

from gragra.fields.regular_grid import DisplacementGrid
from gragra.fields.surface import SurfaceDisplacementField
from gragra.fields.unstructured import UnstructuredDisplacementField


def batched_displacement_grid_weights(
    x_m,
    y_m,
    z_m,
    displacement_m,
    rho0_kg_m3,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute batched weights from structured grid displacements.

    Parameters
    ----------
    x_m, y_m, z_m : array-like
        1D grid coordinate axes.
    displacement_m : (P, Nx, Ny, Nz, 3) array-like or sequence of (Nx, Ny, Nz, 3)
        arrays.
        Batched displacement field.
    rho0_kg_m3 : float or (Nx, Ny, Nz) ndarray
        Background density.

    Returns
    -------
    positions_m : (N, 3) ndarray
        Read-only grid cell center positions.
    weights : (N, P) ndarray
        Read-only computed batched density perturbation weights.
    """
    if isinstance(displacement_m, np.ndarray):
        if displacement_m.ndim == 4:
            raise ValueError(
                "displacement_m has shape (Nx, Ny, Nz, 3) which represents a "
                "single snapshot. A leading batch axis (P) is required. "
                "For single snapshots, use DisplacementGrid directly."
            )
        if displacement_m.ndim != 5:
            raise ValueError(
                f"displacement_m must be a 5-D array (P, Nx, Ny, Nz, 3) "
                f"or a sequence of 4-D arrays, got ndim={displacement_m.ndim}"
            )
        num_snapshots = displacement_m.shape[0]
        snapshots = [displacement_m[p] for p in range(num_snapshots)]
    else:
        try:
            num_snapshots = len(displacement_m)
        except TypeError as e:
            raise TypeError("displacement_m must be a sequence or ndarray") from e

        if num_snapshots == 0:
            raise ValueError("P must be >= 1")

        snapshots = list(displacement_m)
        expected_shape = (len(x_m), len(y_m), len(z_m), 3)
        first_snapshot = np.asanyarray(snapshots[0])
        if first_snapshot.shape != expected_shape:
            raise ValueError(
                "displacement_m shape or snapshot shape is invalid. "
                f"Expected snapshot shape {expected_shape}, "
                f"got {first_snapshot.shape}."
            )

    all_weights = []
    positions = None

    for p in range(num_snapshots):
        disp_p = snapshots[p]
        try:
            grid_p = DisplacementGrid(x_m, y_m, z_m, disp_p)
            perturbed_p = grid_p.as_density_perturbation_grid(rho0_kg_m3)
            vol_p = perturbed_p.as_volume_element_source()
            weighted_p = vol_p.as_weighted_source()
        except ValueError as e:
            raise ValueError(f"snapshot p={p}: {e}") from e
        except TypeError as e:
            raise TypeError(f"snapshot p={p}: {e}") from e

        if p == 0:
            positions = weighted_p.positions_m

        all_weights.append(weighted_p.weights_kg)

    weights = np.column_stack(all_weights)
    positions = positions.copy()
    positions.setflags(write=False)
    weights.setflags(write=False)
    return positions, weights


def batched_unstructured_weights(
    vertices_m,
    tetra,
    displacement_m,
    rho0_kg_m3,
    *,
    include_rho0_gradient_term: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute batched weights from unstructured tetrahedral mesh displacements.

    Parameters
    ----------
    vertices_m : (Nv, 3) array-like
        Mesh node coordinates.
    tetra : (Nc, 4) array-like
        Tetrahedron connectivity indices.
    displacement_m : (P, Nv, 3) array-like or sequence of (Nv, 3) arrays.
        Batched displacement field.
    rho0_kg_m3 : float or BackgroundDensityModel
        Background density.
    include_rho0_gradient_term : bool, optional
        Whether to include background density gradient term.

    Returns
    -------
    positions_m : (Nc, 3) ndarray
        Read-only cell centroid positions.
    weights : (Nc, P) ndarray
        Read-only computed batched density perturbation weights.
    """
    if isinstance(displacement_m, np.ndarray):
        if displacement_m.ndim == 2:
            raise ValueError(
                "displacement_m has shape (Nv, 3) which represents a "
                "single snapshot. A leading batch axis (P) is required. "
                "For single snapshots, use UnstructuredDisplacementField directly."
            )
        if displacement_m.ndim != 3:
            raise ValueError(
                f"displacement_m must be a 3-D array (P, Nv, 3) or "
                f"a sequence of 2-D arrays, got ndim={displacement_m.ndim}"
            )
        num_snapshots = displacement_m.shape[0]
        snapshots = [displacement_m[p] for p in range(num_snapshots)]
    else:
        try:
            num_snapshots = len(displacement_m)
        except TypeError as e:
            raise TypeError("displacement_m must be a sequence or ndarray") from e

        if num_snapshots == 0:
            raise ValueError("P must be >= 1")

        snapshots = list(displacement_m)
        expected_shape = (len(vertices_m), 3)
        first_snapshot = np.asanyarray(snapshots[0])
        if first_snapshot.shape != expected_shape:
            raise ValueError(
                "displacement_m shape or snapshot shape is invalid. "
                f"Expected snapshot shape {expected_shape}, "
                f"got {first_snapshot.shape}."
            )

    all_weights = []
    positions = None

    for p in range(num_snapshots):
        disp_p = snapshots[p]
        try:
            field_p = UnstructuredDisplacementField(vertices_m, tetra, disp_p)
            perturbed_p = field_p.as_density_perturbation_source(
                rho0_kg_m3, include_rho0_gradient_term=include_rho0_gradient_term
            )
            weighted_p = perturbed_p.as_weighted_source()
        except ValueError as e:
            raise ValueError(f"snapshot p={p}: {e}") from e
        except TypeError as e:
            raise TypeError(f"snapshot p={p}: {e}") from e

        if p == 0:
            positions = weighted_p.positions_m

        all_weights.append(weighted_p.weights_kg)

    weights = np.column_stack(all_weights)
    positions = positions.copy()
    positions.setflags(write=False)
    weights.setflags(write=False)
    return positions, weights


def batched_surface_weights(
    vertices_m,
    triangles,
    displacement_m,
    delta_rho_kg_m3: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute batched weights from surface mesh displacements.

    Parameters
    ----------
    vertices_m : (Nv, 3) array-like
        Mesh node coordinates.
    triangles : (Nt, 3) array-like
        Triangle connectivity indices.
    displacement_m : (P, Nv, 3) array-like or sequence of (Nv, 3) arrays.
        Batched displacement field.
    delta_rho_kg_m3 : float
        Density jump across the interface.

    Returns
    -------
    positions_m : (Nt, 3) ndarray
        Read-only surface facet centroid positions.
    weights : (Nt, P) ndarray
        Read-only computed batched density perturbation weights.
    """
    if isinstance(displacement_m, np.ndarray):
        if displacement_m.ndim == 2:
            raise ValueError(
                "displacement_m has shape (Nv, 3) which represents a "
                "single snapshot. A leading batch axis (P) is required. "
                "For single snapshots, use SurfaceDisplacementField directly."
            )
        if displacement_m.ndim != 3:
            raise ValueError(
                f"displacement_m must be a 3-D array (P, Nv, 3) or "
                f"a sequence of 2-D arrays, got ndim={displacement_m.ndim}"
            )
        num_snapshots = displacement_m.shape[0]
        snapshots = [displacement_m[p] for p in range(num_snapshots)]
    else:
        try:
            num_snapshots = len(displacement_m)
        except TypeError as e:
            raise TypeError("displacement_m must be a sequence or ndarray") from e

        if num_snapshots == 0:
            raise ValueError("P must be >= 1")

        snapshots = list(displacement_m)
        expected_shape = (len(vertices_m), 3)
        first_snapshot = np.asanyarray(snapshots[0])
        if first_snapshot.shape != expected_shape:
            raise ValueError(
                "displacement_m shape or snapshot shape is invalid. "
                f"Expected snapshot shape {expected_shape}, "
                f"got {first_snapshot.shape}."
            )

    all_weights = []
    positions = None

    for p in range(num_snapshots):
        disp_p = snapshots[p]
        try:
            field_p = SurfaceDisplacementField(vertices_m, triangles, disp_p)
            perturbed_p = field_p.as_surface_mass_source(delta_rho_kg_m3)
            weighted_p = perturbed_p.as_weighted_source()
        except ValueError as e:
            raise ValueError(f"snapshot p={p}: {e}") from e
        except TypeError as e:
            raise TypeError(f"snapshot p={p}: {e}") from e

        if p == 0:
            positions = weighted_p.positions_m

        all_weights.append(weighted_p.weights_kg)

    weights = np.column_stack(all_weights)
    positions = positions.copy()
    positions.setflags(write=False)
    weights.setflags(write=False)
    return positions, weights
