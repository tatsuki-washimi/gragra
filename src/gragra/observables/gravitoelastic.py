"""Gravitoelastic cross-statistics observables.

These observables couple gravitational acceleration responses with a
G-free witness displacement (or with each other) via
``coherent.cross.directional_outer_cross_average``. ``G_SI`` is
multiplied exactly once per acceleration response computed here (via
``coherent_point_acceleration``); the cross-average contraction itself is
G-agnostic. See
``docs/architecture/coherent_statistics_conventions.md`` §5 for the G-order
table: the witness displacement is G^0, ``coherent_gravitoelastic_tensor``
is G^1, and ``coherent_acceleration_csd`` is G^2.
"""

import numpy as np

from gragra.coherent.cross import directional_outer_cross_average
from gragra.observables.coherent import coherent_point_acceleration


def coherent_gravitoelastic_tensor(
    source_positions_m: list | tuple | np.ndarray,
    weights_kg: list | tuple | np.ndarray,
    targets_m: list | tuple | np.ndarray,
    witness_displacement_m: list | tuple | np.ndarray,
    direction_weights: list | tuple | np.ndarray,
    *,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
    backend: str = "numpy",
) -> np.ndarray:
    """Calculate the gravitoelastic cross tensor c_ij = <delta_a_i * conj(xi_j)>.

    Formula: c_ij(r_m, r_w) = Σ_d w_d · δa_i(r_m) · conj(ξ_j(r_w)), where δa is
    the coherent gravitational acceleration response at ``targets_m`` sourced
    from ``source_positions_m``/``weights_kg``, and ξ is a G-free witness
    displacement (e.g. from ``waves.plane_wave_displacement``).

    G order (see ``coherent_statistics_conventions.md`` §5): this is a
    **G^1** quantity -- ``G_SI`` is multiplied exactly once, internally, via
    ``coherent_point_acceleration``; ``witness_displacement_m`` must be
    G-free (its own kinematic definition never includes ``G_SI``). Do not
    pass a G-included response as ``witness_displacement_m``; there is no
    additional G_SI multiplication in this function and none should be
    applied by the caller either.

    The source mass weights and the witness displacement must be built
    from the **same** wave parameters (k_hat, wavenumber_rad_m, phase0) --
    e.g. via ``coherent.plane_wave_bulk_mass_weights`` and
    ``waves.plane_wave_displacement`` respectively -- otherwise the phase
    relationship this tensor encodes is not physically meaningful.

    Parameters
    ----------
    source_positions_m : array_like
        Source positions, shape ``(N, 3)``.
    weights_kg : array_like
        Complex bulk mass weights, shape ``(N, Ndir, Nf)`` (e.g. from
        ``coherent.plane_wave_bulk_mass_weights``).
    targets_m : array_like
        Acceleration target positions, shape ``(M, 3)``.
    witness_displacement_m : array_like
        G-free witness displacement response, shape ``(W, Ndir, Nf, 3)``
        (component axis trailing; e.g. from
        ``waves.plane_wave_displacement``).
    direction_weights : array_like
        Real-valued direction weights, shape ``(Ndir,)``. Negative values
        are allowed. This is a raw weighted sum (not divided by the sum of
        weights); normalization is the caller's responsibility.
    softening_m : float, optional
        Plummer softening length in meters. Default 0.0.
    chunk_size : int or None, optional
        Source-axis chunk size passed through to
        ``coherent_point_acceleration``. Default None (no chunking).
    backend : str, optional
        Acceleration contraction backend ("numpy", "numba", "cpp", or "cupy").
        Default "numpy".

    Returns
    -------
    numpy.ndarray
        Cross tensor of shape ``(M, W, Nf, 3, 3)`` and dtype
        ``complex128``. It is read-only. Axis order: ``i`` (acceleration
        component) is the second-to-last axis, ``j`` (witness displacement
        component, conjugated) is the last axis.

    Notes
    -----
    - Under a directional ensemble model where waves from different
      directions are mutually uncorrelated, this average corresponds to
      the ensemble expectation. It does not apply to a coherent
      superposition of directionally correlated waves.
    - Since negative direction weights are allowed, this tensor is not
      guaranteed to be Hermitian positive semi-definite (PSD).
    - Complex outputs are second-order cross statistics, not time-domain
      fields; the phase encodes relative propagation delay/coherence
      between the acceleration response and the witness, not an absolute
      field phase.
    """
    acc = coherent_point_acceleration(
        source_positions_m,
        weights_kg,
        targets_m,
        softening_m=softening_m,
        chunk_size=chunk_size,
        backend=backend,
    )
    return directional_outer_cross_average(
        acc, witness_displacement_m, direction_weights
    )


def coherent_acceleration_csd(
    source_positions_m: list | tuple | np.ndarray,
    weights_kg: list | tuple | np.ndarray,
    targets_a_m: list | tuple | np.ndarray,
    targets_b_m: list | tuple | np.ndarray,
    direction_weights: list | tuple | np.ndarray,
    *,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
    backend: str = "numpy",
) -> np.ndarray:
    """Calculate the mirror-pair acceleration cross-spectral density (CSD).

    Formula: CSD_ij(r_a, r_b) = Σ_d w_d · δa_i(r_a) · conj(δa_j(r_b)), where
    both δa responses are sourced from the same
    ``source_positions_m``/``weights_kg``.

    G order (see ``coherent_statistics_conventions.md`` §5): this is a
    **G^2** quantity -- each of the two acceleration responses independently
    carries one ``G_SI`` factor (from its own
    ``coherent_point_acceleration`` call), and no additional G_SI
    multiplication is applied here.

    Parameters
    ----------
    source_positions_m : array_like
        Source positions, shape ``(N, 3)``.
    weights_kg : array_like
        Complex bulk mass weights, shape ``(N, Ndir, Nf)``.
    targets_a_m : array_like
        First acceleration target positions, shape ``(Ma, 3)``.
    targets_b_m : array_like
        Second (conjugated) acceleration target positions, shape
        ``(Mb, 3)``.
    direction_weights : array_like
        Real-valued direction weights, shape ``(Ndir,)``. Negative values
        are allowed. Raw weighted sum; normalization is the caller's
        responsibility.
    softening_m : float, optional
        Plummer softening length in meters. Default 0.0.
    chunk_size : int or None, optional
        Source-axis chunk size passed through to
        ``coherent_point_acceleration`` (applied independently to each of
        the two acceleration evaluations). Default None.
    backend : str, optional
        Acceleration contraction backend ("numpy", "numba", "cpp", or "cupy").
        Default "numpy".

    Returns
    -------
    numpy.ndarray
        CSD tensor of shape ``(Ma, Mb, Nf, 3, 3)`` and dtype
        ``complex128``. It is read-only. Axis order: ``i`` (targets_a
        component) is the second-to-last axis, ``j`` (targets_b component,
        conjugated) is the last axis.

    Notes
    -----
    - Under a directional ensemble model where waves from different
      directions are mutually uncorrelated, this average corresponds to
      the ensemble expectation. It does not apply to a coherent
      superposition of directionally correlated waves.
    - Since negative direction weights are allowed, this tensor is not
      guaranteed to be Hermitian positive semi-definite (PSD).
    - Complex outputs are second-order cross statistics, not time-domain
      fields; the phase encodes relative propagation delay/coherence
      between the two target responses, not an absolute field phase.
    """
    acc_a = coherent_point_acceleration(
        source_positions_m,
        weights_kg,
        targets_a_m,
        softening_m=softening_m,
        chunk_size=chunk_size,
        backend=backend,
    )
    acc_b = coherent_point_acceleration(
        source_positions_m,
        weights_kg,
        targets_b_m,
        softening_m=softening_m,
        chunk_size=chunk_size,
        backend=backend,
    )
    return directional_outer_cross_average(acc_a, acc_b, direction_weights)
