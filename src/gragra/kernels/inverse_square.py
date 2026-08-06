"""Inverse square gravity kernels.

Computes potential, acceleration, and gravity gradient contractions.
All kernel contractions are G-free (do not multiply by G_SI).

The chunk-loop bodies are factored into xp-agnostic private helpers
(``_potential_impl`` / ``_acceleration_impl`` / ``_gradient_impl``) that
receive the array module ``xp`` (numpy, or cupy from the CuPy backend
wrapper) as an argument, so the inverse-square formulas have a single
source shared across the numpy and cupy backends. This module itself
never imports cupy (numpy-only core purity; see
``docs/architecture/gpu_backend_adr.md`` §3).

Softening uses Plummer ``r_eff^2 = r^2 + eps^2`` — note this differs from
nn-sus, which guards the division with an additive ``r^3 + eps`` term
instead.
"""

import numpy as np

from gragra._arrays import _check_no_bool, as_weights, as_xyz_table
from gragra.array_types import ComplexArray, FloatArray


def _validate_inputs(
    sources_xyz: list | tuple | np.ndarray,
    weights: list | tuple | np.ndarray,
    targets_xyz: list | tuple | np.ndarray,
    softening_m: float,
    chunk_size: int | None,
) -> tuple[FloatArray, FloatArray | ComplexArray, FloatArray, float, int]:
    """Validate and normalize inputs for kernel contractions."""
    if softening_m < 0:
        raise ValueError(f"softening_m must be non-negative, got {softening_m}")

    if chunk_size is not None and chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    sources = as_xyz_table("sources_xyz", sources_xyz)
    w = as_weights("weights", weights)
    targets = as_xyz_table("targets_xyz", targets_xyz)

    if len(sources) != len(w):
        raise ValueError(
            f"Length of sources_xyz and weights must match: {len(sources)} != {len(w)}"
        )

    # Determine effective chunk size. Use max(n_sources, 1) so that N=0
    # (empty source) yields a zero-iteration loop returning zeros, matching
    # the numba/cpp backends, rather than raising range() step-zero ValueError.
    n_sources = len(sources)
    actual_chunk_size = chunk_size if chunk_size is not None else max(n_sources, 1)

    return sources, w, targets, float(softening_m), actual_chunk_size


def _validate_dipole_inputs(
    sources_xyz: list | tuple | np.ndarray,
    mass_kg: list | tuple | np.ndarray,
    displacement_m: list | tuple | np.ndarray,
    targets_xyz: list | tuple | np.ndarray,
    softening_m: float,
    chunk_size: int | None,
) -> tuple[FloatArray, FloatArray, FloatArray | ComplexArray, FloatArray, float, int]:
    """Validate and normalize inputs for the vector-weight (dipole) contraction.

    Unlike the scalar ``weights`` argument of potential_contract /
    acceleration_contract / gradient_contract (validated by ``as_weights``,
    which allows signed/complex values), ``mass_kg`` here is validated as
    real, finite, and non-negative only (kernel_contract.md §2.1.1) — it
    represents a physical background mass element, not a signed/complex
    weight. ``displacement_m`` is real or complex, shape (N, 3); this is
    the "vector weight" of kernel_contract.md §2.1 and is validated
    independently of ``as_xyz_table`` (which is real-only) to allow complex
    harmonic amplitudes.
    """
    if softening_m < 0:
        raise ValueError(f"softening_m must be non-negative, got {softening_m}")

    if chunk_size is not None and chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    sources = as_xyz_table("sources_xyz", sources_xyz)
    targets = as_xyz_table("targets_xyz", targets_xyz)

    _check_no_bool(mass_kg, "mass_kg")
    raw_mass = np.asarray(mass_kg)
    if np.iscomplexobj(raw_mass):
        raise TypeError(
            "mass_kg must be real-valued (complex mass is rejected; see "
            "kernel_contract.md §2.1.1 — complex amplitude belongs in "
            "displacement_m)"
        )
    try:
        mass = np.asarray(mass_kg, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise TypeError(f"mass_kg must be numeric convertible to float64: {e}") from e
    if mass.ndim != 1:
        raise ValueError(
            f"mass_kg must be a 1D array of shape (N,), got shape {mass.shape}"
        )
    if not np.isfinite(mass).all():
        raise ValueError("mass_kg must contain only finite numbers")
    if (mass < 0).any():
        raise ValueError("mass_kg must be non-negative")
    mass = mass.copy()
    mass.setflags(write=False)

    _check_no_bool(displacement_m, "displacement_m")
    raw_disp = np.asarray(displacement_m)
    disp_dtype = np.complex128 if np.iscomplexobj(raw_disp) else np.float64
    try:
        disp = np.asarray(displacement_m, dtype=disp_dtype)
    except (ValueError, TypeError) as e:
        raise TypeError(
            f"displacement_m must be numeric convertible to float64 or complex128: {e}"
        ) from e
    if disp.ndim != 2 or disp.shape[1] != 3:
        raise ValueError(f"displacement_m must have shape (N, 3), got {disp.shape}")
    if not np.isfinite(disp).all():
        raise ValueError("displacement_m must contain only finite numbers")
    disp = disp.copy()
    disp.setflags(write=False)

    if not (len(sources) == len(mass) == len(disp)):
        raise ValueError(
            "Length of sources_xyz, mass_kg, and displacement_m must match: "
            f"{len(sources)}, {len(mass)}, {len(disp)}"
        )

    n_sources = len(sources)
    actual_chunk_size = chunk_size if chunk_size is not None else max(n_sources, 1)

    return sources, mass, disp, targets, float(softening_m), actual_chunk_size


def _check_chunk_collision(xp, r2, start: int) -> None:
    """Raise ValueError if any pairwise distance in this chunk is zero.

    The reported indices are the first collision in row-major order
    *within the first chunk that contains a collision* (smallest target
    index, then smallest source index within it) — deterministic for both
    numpy and cupy ``nonzero``, and identical across backends for the same
    ``chunk_size``, but which collision is reported can depend on the
    chunk split when several collisions exist. ``start`` is the chunk
    offset used to recover the global source index. For the cupy backend
    the ``any()`` / ``nonzero`` calls synchronize the device once per
    chunk (only on the softening == 0 path).
    """
    zero_mask = r2 == 0.0
    if bool(zero_mask.any()):
        zero_idx = xp.nonzero(zero_mask)
        target_idx = int(zero_idx[0][0])
        source_idx = start + int(zero_idx[1][0])
        raise ValueError(
            "Collision detected between target point "
            f"{target_idx} and source point {source_idx} "
            "with zero softening."
        )


def _potential_impl(xp, sources, w, targets, eps: float, c_size: int, out) -> None:
    """xp-agnostic chunk loop for the potential contraction.

    K_Φ = -1 / sqrt(r^2 + ε^2), accumulated into the pre-allocated ``out``
    of shape (M,) in-place. ``sources`` / ``w`` / ``targets`` / ``out``
    must already live on the array module ``xp`` (numpy or cupy); output
    dtype selection and allocation are the caller's responsibility.
    """
    n_sources = len(sources)
    eps2 = eps**2

    # Loop over source chunks
    for start in range(0, n_sources, c_size):
        end = min(start + c_size, n_sources)
        s_chunk = sources[start:end]
        w_chunk = w[start:end]

        # Shape (M, C, 3)
        diff = s_chunk[xp.newaxis, :, :] - targets[:, xp.newaxis, :]
        r2 = xp.sum(diff**2, axis=-1)  # Shape (M, C)

        if eps == 0.0:
            _check_chunk_collision(xp, r2, start)

        r_eff = xp.sqrt(r2 + eps2)
        term = -1.0 / r_eff  # Shape (M, C)

        out += xp.sum(term * w_chunk, axis=-1)


def _acceleration_impl(xp, sources, w, targets, eps: float, c_size: int, out) -> None:
    """xp-agnostic chunk loop for the acceleration contraction.

    K_a = Δ / (r^2 + ε^2)^(3/2), accumulated into the pre-allocated
    ``out`` of shape (M, 3) in-place. See ``_potential_impl`` for the
    xp / allocation contract.
    """
    n_sources = len(sources)
    eps2 = eps**2

    # Loop over source chunks
    for start in range(0, n_sources, c_size):
        end = min(start + c_size, n_sources)
        s_chunk = sources[start:end]
        w_chunk = w[start:end]

        # Shape (M, C, 3)
        diff = s_chunk[xp.newaxis, :, :] - targets[:, xp.newaxis, :]
        r2 = xp.sum(diff**2, axis=-1)  # Shape (M, C)

        if eps == 0.0:
            _check_chunk_collision(xp, r2, start)

        r_eff = xp.sqrt(r2 + eps2)
        r_eff3 = (r2 + eps2) * r_eff

        # diff: (M, C, 3), w_chunk: (C,)
        # sum over C
        term = diff / r_eff3[:, :, xp.newaxis]
        out += xp.sum(term * w_chunk[xp.newaxis, :, xp.newaxis], axis=1)


def _gradient_impl(xp, sources, w, targets, eps: float, c_size: int, out) -> None:
    """xp-agnostic chunk loop for the gravity gradient contraction.

    K_T = 3 ΔΔ / (r^2 + ε^2)^(5/2) - I / (r^2 + ε^2)^(3/2), accumulated
    into the pre-allocated ``out`` of shape (M, 3, 3) in-place. See
    ``_potential_impl`` for the xp / allocation contract.
    """
    n_sources = len(sources)
    eps2 = eps**2
    eye3 = xp.eye(3)

    # Loop over source chunks
    for start in range(0, n_sources, c_size):
        end = min(start + c_size, n_sources)
        s_chunk = sources[start:end]
        w_chunk = w[start:end]

        # Shape (M, C, 3)
        diff = s_chunk[xp.newaxis, :, :] - targets[:, xp.newaxis, :]
        r2 = xp.sum(diff**2, axis=-1)  # Shape (M, C)

        if eps == 0.0:
            _check_chunk_collision(xp, r2, start)

        r_eff = xp.sqrt(r2 + eps2)
        r_eff3 = (r2 + eps2) * r_eff
        r_eff5 = r_eff3 * (r2 + eps2)

        # Outer product of diff vectors along space component axes
        # diff is (M, C, 3) -> diff_outer (M, C, 3, 3)
        diff_outer = diff[:, :, :, xp.newaxis] * diff[:, :, xp.newaxis, :]

        # (M, C, 3, 3)
        term = (
            3.0 * diff_outer / r_eff5[:, :, xp.newaxis, xp.newaxis]
            - eye3[xp.newaxis, xp.newaxis, :, :] / r_eff3[:, :, xp.newaxis, xp.newaxis]
        )

        out += xp.sum(term * w_chunk[xp.newaxis, :, xp.newaxis, xp.newaxis], axis=1)


def _dipole_impl(
    xp, sources, mass, disp, targets, eps: float, c_size: int, out
) -> None:
    """xp-agnostic chunk loop for the vector-weight (dipole/total) contraction.

    out_i = -Σ_n mass_n * Σ_j K_T[m,n,i,j] * disp_n[j], accumulated into the
    pre-allocated ``out`` of shape (M, 3) in-place. Reuses the same
    K_T = 3ΔΔ/r_eff^5 - I/r_eff^3 tensor computed per chunk by
    ``_gradient_impl``, contracted against a vector weight instead of a
    scalar weight broadcast. Scoped to ``disp``: (N, 3) only — see
    kernel_contract.md §2.1 (parametric (N,*P,3) is Deferred; the einsum
    contraction pattern below is (N,3)-specific and would need
    generalizing to "mcij,c...j->m...i" for a parametric extension).
    """
    n_sources = len(sources)
    eps2 = eps**2
    eye3 = xp.eye(3)

    # Loop over source chunks
    for start in range(0, n_sources, c_size):
        end = min(start + c_size, n_sources)
        s_chunk = sources[start:end]
        m_chunk = mass[start:end]
        d_chunk = disp[start:end]  # (C, 3)

        # Shape (M, C, 3)
        diff = s_chunk[xp.newaxis, :, :] - targets[:, xp.newaxis, :]
        r2 = xp.sum(diff**2, axis=-1)  # Shape (M, C)

        if eps == 0.0:
            _check_chunk_collision(xp, r2, start)

        r_eff = xp.sqrt(r2 + eps2)
        r_eff3 = (r2 + eps2) * r_eff
        r_eff5 = r_eff3 * (r2 + eps2)

        # Outer product of diff vectors along space component axes
        diff_outer = diff[:, :, :, xp.newaxis] * diff[:, :, xp.newaxis, :]

        # k_t (K_T), shape (M, C, 3, 3)
        k_t = (
            3.0 * diff_outer / r_eff5[:, :, xp.newaxis, xp.newaxis]
            - eye3[xp.newaxis, xp.newaxis, :, :] / r_eff3[:, :, xp.newaxis, xp.newaxis]
        )

        # Contract the j-axis of k_t against d_chunk: (M, C, 3, 3) x (C, 3) -> (M, C, 3)
        t_d = xp.einsum("mcij,cj->mci", k_t, d_chunk)
        out += -xp.sum(t_d * m_chunk[xp.newaxis, :, xp.newaxis], axis=1)


def potential_contract(
    sources_xyz: list | tuple | np.ndarray,
    weights: list | tuple | np.ndarray,
    targets_xyz: list | tuple | np.ndarray,
    *,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
) -> FloatArray | ComplexArray:
    """Calculate the G-free gravitational potential contraction.

    K_Φ = -1 / sqrt(r^2 + ε^2)

    Parameters
    ----------
    sources_xyz : array-like
        Source coordinates table of shape (N, 3) in meters.
    weights : array-like
        Weights vector of shape (N,) in kilograms (real or complex).
    targets_xyz : array-like
        Target coordinates table of shape (M, 3) in meters.
    softening_m : float, optional
        Plummer softening parameter in meters. Defaults to 0.0.
    chunk_size : int, optional
        Chunk size for source-axis loop to manage memory. Defaults to None.

    Returns
    -------
    FloatArray or ComplexArray
        Calculated potential of shape (M,).
    """
    sources, w, targets, eps, c_size = _validate_inputs(
        sources_xyz, weights, targets_xyz, softening_m, chunk_size
    )

    # Output dtype follows weights dtype
    out_dtype = np.complex128 if np.iscomplexobj(w) else np.float64
    out = np.zeros(len(targets), dtype=out_dtype)

    _potential_impl(np, sources, w, targets, eps, c_size, out)

    # Set flags read-only
    out.setflags(write=False)
    return out


def acceleration_contract(
    sources_xyz: list | tuple | np.ndarray,
    weights: list | tuple | np.ndarray,
    targets_xyz: list | tuple | np.ndarray,
    *,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
) -> FloatArray | ComplexArray:
    """Calculate the G-free gravitational acceleration contraction.

    K_a = Δ / (r^2 + ε^2)^(3/2)
    where Δ = r'_j(source) - r_i(target)

    Parameters
    ----------
    sources_xyz : array-like
        Source coordinates table of shape (N, 3) in meters.
    weights : array-like
        Weights vector of shape (N,) in kilograms (real or complex).
    targets_xyz : array-like
        Target coordinates table of shape (M, 3) in meters.
    softening_m : float, optional
        Plummer softening parameter in meters. Defaults to 0.0.
    chunk_size : int, optional
        Chunk size for source-axis loop to manage memory. Defaults to None.

    Returns
    -------
    FloatArray or ComplexArray
        Calculated acceleration of shape (M, 3).
    """
    sources, w, targets, eps, c_size = _validate_inputs(
        sources_xyz, weights, targets_xyz, softening_m, chunk_size
    )

    out_dtype = np.complex128 if np.iscomplexobj(w) else np.float64
    out = np.zeros((len(targets), 3), dtype=out_dtype)

    _acceleration_impl(np, sources, w, targets, eps, c_size, out)

    out.setflags(write=False)
    return out


def gradient_contract(
    sources_xyz: list | tuple | np.ndarray,
    weights: list | tuple | np.ndarray,
    targets_xyz: list | tuple | np.ndarray,
    *,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
) -> FloatArray | ComplexArray:
    """Calculate the G-free gravity gradient tensor contraction.

    K_T = 3 * ΔΔ / (r^2 + ε^2)^(5/2) - I / (r^2 + ε^2)^(3/2)
    where Δ = r'_j(source) - r_i(target)
    and I is the 3x3 identity matrix.

    Note that for softening_m > 0, the softened tensor does not satisfy
    tr(T) = 0.

    Parameters
    ----------
    sources_xyz : array-like
        Source coordinates table of shape (N, 3) in meters.
    weights : array-like
        Weights vector of shape (N,) in kilograms (real or complex).
    targets_xyz : array-like
        Target coordinates table of shape (M, 3) in meters.
    softening_m : float, optional
        Plummer softening parameter in meters. Defaults to 0.0.
    chunk_size : int, optional
        Chunk size for source-axis loop to manage memory. Defaults to None.

    Returns
    -------
    FloatArray or ComplexArray
        Calculated gravity gradient tensor of shape (M, 3, 3).
    """
    sources, w, targets, eps, c_size = _validate_inputs(
        sources_xyz, weights, targets_xyz, softening_m, chunk_size
    )

    out_dtype = np.complex128 if np.iscomplexobj(w) else np.float64
    out = np.zeros((len(targets), 3, 3), dtype=out_dtype)

    _gradient_impl(np, sources, w, targets, eps, c_size, out)

    out.setflags(write=False)
    return out


def dipole_contract(
    sources_xyz: list | tuple | np.ndarray,
    mass_kg: list | tuple | np.ndarray,
    displacement_m: list | tuple | np.ndarray,
    targets_xyz: list | tuple | np.ndarray,
    *,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
) -> FloatArray | ComplexArray:
    """Calculate the G-free displacement-transfer (dipole/total) contraction.

    δa_i = -Σ_n mass_n * Σ_j K_T_ij(r_n) * displacement_n[j]

    K_T is the same gravity gradient kernel as ``gradient_contract``
    (K_T = 3ΔΔ/r_eff^5 - I/r_eff^3, Δ = source - target; see
    kernel_contract.md §3). Moving a mass element by a small displacement d
    produces a linear acceleration response at the target equal to -T·d:
    the sign follows from the source-derivative of the acceleration being
    the negation of the target-derivative K_T (kernel_contract.md §2.1/§2.2).
    This is verified independently by a finite-difference oracle in tests
    (source-shifted, not target-shifted): a source-shifted central
    difference of ``acceleration_contract`` converges directly to this
    function's output with no extra sign flip; a target-shifted difference
    would converge to the negation.

    Equivalence to the bulk+surface divergence path (``fields/`` +
    ``gradient_contract``/``acceleration_contract`` on delta-rho mass
    elements) holds ONLY under the six conditions of kernel_contract.md
    §2.2 (closed volume, consistent triangle winding, outward-pointing
    surface normal, ``fields/surface.py`` delta_rho sign convention,
    geometric coincidence of the volume/surface boundary, vacuum
    truncation outside the boundary) — a sub-volume embedded in
    surrounding rock is NOT equivalent.

    Parameters
    ----------
    sources_xyz : array-like
        Source (undisturbed / equilibrium) coordinates table of shape
        (N, 3) in meters.
    mass_kg : array-like
        Background mass element at each source point, shape (N,) in
        kilograms. Real, finite, non-negative only (kernel_contract.md
        §2.1.1) — unlike the scalar ``weights`` of ``acceleration_contract``
        / ``gradient_contract``, signed or complex mass is rejected;
        complex harmonic amplitude belongs in ``displacement_m`` instead.
    displacement_m : array-like
        Displacement vector at each source point, shape (N, 3) in meters
        (real or complex for harmonic amplitude). This is the "vector
        weight" of kernel_contract.md §2.1.
    targets_xyz : array-like
        Target coordinates table of shape (M, 3) in meters.
    softening_m : float, optional
        Plummer softening parameter in meters. Defaults to 0.0. Note that
        for softening_m > 0, the underlying K_T tensor does not satisfy
        tr(T) = 0 (kernel_contract.md §4), so the dipole response inherits
        the same non-trace-free caveat.
    chunk_size : int, optional
        Chunk size for source-axis loop to manage memory. Defaults to None.

    Returns
    -------
    FloatArray or ComplexArray
        Calculated displacement-transfer acceleration of shape (M, 3).
        dtype is complex128 if ``displacement_m`` is complex, else float64
        (``mass_kg`` is always real per §2.1.1, so only ``displacement_m``
        drives the complex promotion).
    """
    sources, mass, disp, targets, eps, c_size = _validate_dipole_inputs(
        sources_xyz, mass_kg, displacement_m, targets_xyz, softening_m, chunk_size
    )

    out_dtype = np.complex128 if np.iscomplexobj(disp) else np.float64
    out = np.zeros((len(targets), 3), dtype=out_dtype)

    _dipole_impl(np, sources, mass, disp, targets, eps, c_size, out)

    out.setflags(write=False)
    return out
