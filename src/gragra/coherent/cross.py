"""G-free component-outer cross average helper (gravitoelastic building block)."""

import numpy as np

from gragra.coherent.average import directional_cross_average


def directional_outer_cross_average(
    response_a,
    response_b,
    direction_weights,
) -> np.ndarray:
    """Calculate the direction-averaged component-outer cross tensor of two responses.

    Formula: c_ij(r_a, r_b) = Σ_d w_d · response_a_i,d · conj(response_b_j,d)

    This is a thin wrapper around ``directional_cross_average``
    (``coherent/average.py``): it inserts singleton axes so the component
    axes of the two inputs are broadcast against each other (outer product
    over components) rather than aligned element-wise, then delegates the
    direction-axis contraction and conjugate-second-argument convention to
    that function without reimplementing it (see
    ``docs/architecture/coherent_statistics_conventions.md`` §1/§4).

    This helper is **G-agnostic**: it performs a pure statistical
    contraction and never multiplies by ``G_SI``. It is the (A)-type
    (G-free helper) API in the G-order table of
    ``coherent_statistics_conventions.md`` §5; callers (e.g.
    ``observables.gravitoelastic``) are responsible for computing any
    G-included response before passing it in.

    Parameters
    ----------
    response_a : array_like
        First response array, shape ``(Ma, Ndir, Nf, 3)`` (component axis
        trailing). This is the *unconjugated* argument (index ``i``).
    response_b : array_like
        Second response array, shape ``(Mb, Ndir, Nf, 3)``. This is the
        conjugated argument (index ``j``) -- conjugation is applied
        internally by ``directional_cross_average``; do not pre-conjugate.
    direction_weights : array_like
        Real-valued direction weights, shape ``(Ndir,)``. Negative values
        are allowed (see ``coherent_statistics_conventions.md`` §2).
        Normalization is the caller's responsibility: this is a raw
        weighted sum, not divided by the sum of weights.

    Returns
    -------
    numpy.ndarray
        Cross tensor of shape ``(Ma, Mb, Nf, 3, 3)`` and dtype
        ``complex128``. It is read-only.

    Notes
    -----
    - Under a directional ensemble model where waves from different
      directions are mutually uncorrelated, this average corresponds to
      the ensemble expectation ``<response_a_i * conj(response_b_j)>``.
      It does not apply to a coherent superposition of directionally
      correlated waves.
    - Since negative weights are allowed, the resulting 3x3 tensor is not
      guaranteed to be Hermitian positive semi-definite (PSD); callers
      must not assume PSD when treating it as a covariance-like quantity.
    - Complex outputs are second-order cross statistics (cross-spectral
      quantities such as <A*B*>), not time-domain fields; the phase
      encodes relative propagation delay/coherence, not an absolute field
      phase.
    """
    resp_a = np.asarray(response_a)
    resp_b = np.asarray(response_b)

    if resp_a.ndim != 4:
        raise ValueError(
            "response_a must have shape (Ma, Ndir, Nf, 3) "
            f"(ndim 4), got ndim {resp_a.ndim}"
        )
    if resp_b.ndim != 4:
        raise ValueError(
            "response_b must have shape (Mb, Ndir, Nf, 3) "
            f"(ndim 4), got ndim {resp_b.ndim}"
        )
    if resp_a.shape[-1] != 3 or resp_b.shape[-1] != 3:
        raise ValueError(
            "response_a and response_b must have a trailing component "
            f"axis of size 3, got {resp_a.shape[-1]} and {resp_b.shape[-1]}"
        )
    if resp_a.shape[1] != resp_b.shape[1]:
        raise ValueError(
            "response_a and response_b must share the same Ndir (axis 1): "
            f"{resp_a.shape[1]} != {resp_b.shape[1]}"
        )
    if resp_a.shape[2] != resp_b.shape[2]:
        raise ValueError(
            "response_a and response_b must share the same Nf (axis 2): "
            f"{resp_a.shape[2]} != {resp_b.shape[2]}"
        )

    # Singleton insertion + broadcast (conventions §1/§4): outer product over
    # the trailing component axes, direction axis kept aligned at position 2.
    a_expanded = resp_a[:, None, :, :, :, None]
    b_expanded = resp_b[None, :, :, :, None, :]
    a2, b2 = np.broadcast_arrays(a_expanded, b_expanded)

    return directional_cross_average(a2, b2, direction_weights, direction_axis=2)
